---
name: compute
description: Device abstraction, spatial primitives, kernel dispatch, device-aware buffers, and vendor optimization
applyTo: "crates/core/**"
---

# Compute

## Device (`compute.rs`)

`Device` is the **single, universal dispatch handle** — one concrete type for ALL hardware backends. No trait overhead for callers. Users work exclusively with `Device`, never with raw traits.

```rust
let dev = Device::cpu();          // CPU with best SIMD
let dev = Device::best();         // auto-select (lazy, cached)
let dev = Device::simulated(p);   // throttled for benchmarking
let dev = Device::from_kernel(k); // custom backend (power users)
```

**Design:**

- `Device(Arc<DeviceInner>)` — `Clone + Send + Sync`, cheap to share.
- Internally holds `Box<dyn Kernel>` + `DeviceInfo` + optional throttle.
- `BEST_DEVICE: LazyLock<Device>` — global singleton, auto-detected CPU SIMD.
- Re-exported from `lib.rs` as `any_compute_core::Device`.

**High-level ops** (parallel, arbitrary fn):

- `map`, `filter`, `sort`, `sum`, `prefix_sum`, `map_hinted`, `sum_hinted`

**Low-level ops** (SIMD-accelerated, predefined ops):

- `unary(data, UnaryOp)`, `binary(a, b, BinaryOp)`, `reduce(data, ReduceOp)`
- `scan`, `gemm`, `gather`, `scatter`, `benchmark_op`

**Rules:**

- All parallel work **must** go through `Device` — never call rayon directly.
- New backends are added via `Kernel` trait + `Device::from_kernel()`.
- `ComputeBackend` trait no longer exists — `Device` replaces it entirely.

## Spatial primitives (`layout.rs`)

All spatial types are built on two const-generic foundations:

- **`V<N>`** — N-dimensional vector, `#[repr(transparent)]` over `[f64; N]`. THE universal numeric/spatial type.
- **`Region<N>`** — N-dimensional axis-aligned bounding box (`origin: V<N>`, `size: V<N>`).

Named-field access via `Deref`: `V<2>` → `V2Fields { x, y }`, `V<3>` → `V3Fields { x, y, z }`, `V<4>` → `V4Fields { x, y, z, w }`. Implemented via `impl_deref_fields!` macro with unsafe pointer cast (layout-safe: repr(transparent) + repr(C) targets, compile-time size assertions).

Type aliases: `Point = Size = V2 = V<2>`, `V3 = V<3>`, `V4 = V<4>`, `Rect = Region<2>`.

All arithmetic (Add, Sub, Neg, Mul<f64>, Div<f64>), Lerp, From, Index/IndexMut defined ONCE on `V<N>` — works for any dimension. `V<2>` adds `.w()` / `.h()` methods for size semantics plus `.area()`.

- Use `V::new(x, y)` only for 2D. For 3D+ use `V([x, y, z])` directly.
- Never create separate Point/Size structs; they ARE `V<2>`.
- Extend to new dimensions by adding a `VNFields` + `impl_deref_fields!` entry — nothing else changes.

## Buffer (`buffer.rs`)

- `Buffer` wraps `Vec<f64>` + a `Device` for transparent hardware dispatch.
- `Buffer::new(data)` uses `Device::best()`. `Buffer::on(&device, data)` selects a specific device.
- `buffer.device()` returns the device; all ops dispatch through it.
- Reductions: `sum`, `min`, `max`, `mean`, `product`. Unary maps: `neg`, `abs`, `sqrt`, `exp`, `log`, `sin`, `cos`, `tanh`, `relu`, `sigmoid`, `scale`, `offset`. Binary ops via Add/Sub/Mul/Div operator overloads.
- `prefix_sum`, `sort`, `gemm(other, m, n, k)` for higher-level operations.
- From: `Vec<f64>`, `&[f64]`.

## Dispatch hierarchy

- `Device` is the **only** dispatch entry point — it uses `Kernel` internally.
- `BackendKind` maps one-to-one to physical vendor hardware: `Cpu`, `Wgpu`, `Cuda`, `Rocm`.
- `KernelBackend` is the fine-grained variant: `CpuScalar`, `CpuSimd`, `Cuda`, `Rocm`, `Mkl`, `Metal`, `Wgpu`.
- `From<KernelBackend> for BackendKind` bridges the two — use it instead of manual mapping.
- `DeviceInfo` (`kind`, `name`, `memory_bytes`, `max_parallelism`) is the canonical device descriptor.
- `Hints` tune per-call thresholds (parallelism, batch size, memory strategy) — defaults are sensible.

## Type flexibility (From impls)

Core types maximize conversion flexibility via `From` trait:

- **V\<N\>**: from `[f64; N]`, `f64` (uniform fill). V<2> additionally from `(f64,f64)`, `(i32,i32)`, `(u32,u32)`.
- **Region\<N\>**: from `(V<N>, V<N>)` (origin, size). Region<2> from `(f64,f64,f64,f64)`, `[f64;4]`, `V<2>` (origin=zero).
- **Color**: from `(u8,u8,u8)`, `(u8,u8,u8,u8)`, `[u8;3]`, `[u8;4]`, `u32` hex (`0xRRGGBB`/`0xRRGGBBAA`), and back.
- **CellValue**: from `bool`, `i32`, `i64`, `f32`, `f64`, `String`, `&str`. Accessors: `as_f64()`, `as_i64()`, `as_str()`.
- **Buffer**: from `Vec<f64>`, `&[f64]`.

Push new From impls for any new type; prefer `From` over custom constructors.

## Core macros

- **`display_enum!`** (lib.rs) — Generates `Display` impl for enums mapping variants to string literals. Used by `KernelBackend` and `ShaderStage`.
- **`impl_deref_fields!`** (layout.rs) — Generates Deref/DerefMut from `V<N>` to field struct with compile-time layout assertions.

## OpQueue — batched dispatch (`compute.rs`)

`OpQueue` accumulates operations and flushes them in a single batch for cache locality (CPU) / fewer roundtrips (GPU).

- `Device::queue()` creates an `OpQueue`.
- `push(QueuedOp)` records an op lazily (Unary, Binary, Reduce, Scan, Sort, Gemm).
- `flush()` dispatches all queued ops through the Device, returns `Vec<OpResult>`.
- `OpResult` is either `Vector(Vec<f64>)` or `Scalar(f64)`.
- Consumed on flush — build a new queue for the next batch.

## OpCache — content-addressed memoization (`compute.rs`)

`OpCache` stores previous op results keyed by `(data_ptr, data_len, op_tag)` via FNV-1a hash.

- `OpCache::new(capacity)` — max number of cached entries.
- `.unary(data, UnaryOp, device)`, `.binary(data, other, BinaryOp, device)`, `.reduce(data, ReduceOp, device)` — cache-through methods.
- Thread-safe via `Mutex<CacheInner>`. Simple eviction (clear-all when full, placeholder for LRU).

## Scene graph — 3D primitives (`scene.rs`)

Flat 3D scene graph built on `V<3>` and `Buffer`:

- **`Mesh`**: flat `Buffer` vertices (x,y,z per vertex) + optional indices, normals, uvs. `compute_normals()` for area-weighted smooth normals. `bounds()` returns `AABB`.
- **`Camera`**: eye/target/up + `Projection` (Perspective / Orthographic). `forward()` / `right()` helpers.
- **`Light`**: Directional / Point / Ambient with color and intensity.
- **`Material`**: PBR-style — albedo, metallic, roughness, emissive, ior, alpha.
- **`Transform`**: position + quaternion rotation + scale. Implements `Lerp`. `apply(V3)` / `rotate(V3)`.
- **`SceneObject`**: Mesh + Transform.
- **`Scene`**: flat container (objects, materials, lights, camera). `bounds()` computes world-space AABB.
- **`Ray`**: origin + direction. `intersect_aabb()` (slab method), `intersect_tri()` (Möller–Trumbore).

**V<3> ops** (layout.rs): `cross()`, `reflect()`, `face_normal()`, `new3()`, `d()`, `volume()`. Generic `V<N>::normalized()` (single normalize for all dims).

**AABB** (layout.rs): `type AABB = Region<3>`. `Region<3>::new3()`, `depth()`, `volume()`, `intersects()`.

## Kernel layer

- `Kernel` trait is the internal implementation point for element-wise operations (unary, binary, reduce).
- `Device` delegates to `Kernel` — users never need to import `Kernel` directly.
- `best_kernel()` returns a boxed kernel for power-user scenarios.
- WGSL shaders live in `crates/core/shaders/` and are the cross-vendor GPU kernel source.

### SIMD intrinsics (`simd_avx2` module in `kernel.rs`)

- `binary_f64_avx2(a, b, op)` — 4-wide AVX2 f64 binary ops (Add/Sub/Mul/Div/Min/Max).
- `reduce_f64_avx2(data, op)` — 4-wide horizontal reduction (Sum/Min/Max).
- Both are `#[target_feature(enable = "avx2")]` unsafe fns.
- `CpuSimdKernel` dispatches to AVX2 path when `self.width >= 4` (detected at runtime).
- Edition 2024: all intrinsic calls inside `unsafe fn` must still be wrapped in `unsafe {}`.

## Vendor optimization rule

- Vendor-specific paths (CUDA, ROCm, MKL, Metal) live behind feature flags only — default path is `wgpu` + `rayon`.
- New vendor backends: implement `Kernel` trait, create via `Device::from_kernel()`.
