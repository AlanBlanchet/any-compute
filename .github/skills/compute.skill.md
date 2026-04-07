---
name: compute
description: Device abstraction, spatial primitives, kernel dispatch, device-aware buffers, and vendor optimization
applyTo: "crates/core/**"
---

# Compute

## Device (`compute/`)

`Device` is the single universal dispatch handle — one concrete type for ALL backends.
`Device(Arc<DeviceInner>)` — `Clone + Send + Sync`. Holds `Box<dyn Kernel>` + `DeviceInfo`.
Constructors: `cpu()`, `best()` (lazy cached), `simulated(p)`, `from_kernel(k)`.
All parallel work must go through `Device` — never call rayon directly.

## Spatial Primitives (`layout/`)

- **`V<N>`** — N-dimensional vector, `#[repr(transparent)]` over `[f64; N]`. Named field access via Deref (`V<2>` → x/y, `V<3>` → x/y/z)
- **`Region<N>`** — N-dimensional AABB (`origin: V<N>`, `size: V<N>`)
- **`Matrix<R,C>`** — dense row-major matrix. Mat × Vec, Mat ± Mat, transpose, row/col extraction
- Type aliases: `Point = Size = V2 = V<2>`, `Rect = Region<2>`, `Mat4 = Matrix<4,4>`, `AABB = Region<3>`
- All arithmetic defined ONCE on `V<N>` — works for any dimension. Extend by adding VNFields + `impl_deref_fields!`
- Maximize `From` conversions: tuples, arrays, scalars for V/Region/Color/Buffer

## Buffer (`buffer.rs`)

`Buffer` wraps `Vec<f64>` + `Device`. All ops dispatch through the device.
Reductions, unary maps, binary ops via operator overloads, `prefix_sum`, `sort`, `gemm`.

## Dispatch Hierarchy

- `Device` is the only entry point — uses `Kernel` internally
- `BackendKind` = physical hardware (Cpu, Wgpu, Cuda, Rocm)
- `KernelBackend` = fine-grained variant (CpuScalar, CpuSimd, Cuda, Rocm, Mkl, Metal, Wgpu)
- SIMD: AVX2 4-wide paths in `kernel/`, dispatched when `width >= 4`

## OpQueue / OpCache (`compute/`)

- `OpQueue`: batched dispatch — `push(QueuedOp)` → `flush() → Vec<OpResult>`
- `OpCache`: content-addressed memoization via FNV-1a hash

## Scene Graph (`scene/`)

Flat 3D scene: `Mesh` (Buffer vertices + indices), `Camera` (perspective/ortho with orbit/fly/strafe), `Light`, `Material`, `Transform` (pos + quat + scale, implements `Lerp`), `Scene`, `Ray`.
Mesh constructors: `cube`, `sphere`, `cylinder`, `torus`, `from_obj`. Shared helpers: `ring_angle`, `grid_indices`.

## Render Primitives (`render/`)

- `Primitive`: Rect, Text, Line, Triangle, PushClip, PopClip
- `RenderList`: push_line, push_dot, push_projected_line (generic over `Viewport<P>`), composite
- `Viewport<P>` trait: unified projection (2D `Viewport2D`, 3D `CameraView`)
- `PixelBuffer`: CPU software rasterizer for testing/headless

## Composable Ops Traits (`ops/`)

Implement `AsF64s` (expose `&[f64]`) → auto-get Stats, Norm, Histogram, SortOps, SearchOps, ReduceOps, Diff.
Implement `Renderable<()>` → auto-get Capture, Recording.
Standalone: Approx, Summary, Export (`to_json`/`to_csv`), DeviceAware, PairwiseOps.

## Core Macros

- `for_each_unary!` / `for_each_binary!` / `for_each_reduce!` — SSOT for op variants; adding an op = one line
- `display_enum!` — Display impl for enums
- `impl_deref_fields!` — V<N> → named field Deref with layout assertions

## Lazy Evaluation & Graph (`graph/`)

- `Buffer::lazy(graph)` → `LazyMut` — records ops without executing. Same named methods as eager API
- `Graph`: DAG with `eval()` (auto-fuses consecutive unary chains), `to_dot()`, `to_onnx()`, ONNX hand-rolled protobuf
- Neural network layers: `Layer` trait, `Linear`, `Activation`, `BatchNorm`, `Residual`, `Sequential`, `Network`
- Model builders: `resnet()`, `vit()`, `gpt()`, `mixer()` — share `mlp_block`/`attn_block`/`transformer_blocks`/`output_head`
- ML taxonomy: `TaskType`, `ModelKind` (implements `Graphable`), `DatasetSource` — domain enums with `from_tag()`, `for_task()`, `build_network()`

## Data & Dataset Providers (`data.rs`)

- `CellValue`: `Empty | Bool | Int | Float | Text | Bytes` — per-cell value in a virtualized table
- `DataSource` trait: `row_count()`, `columns()`, `fetch(rows)` — virtualized data access
- `VecSource`: in-memory `DataSource` backed by `Vec<Vec<CellValue>>`
- `DatasetProvider` trait (behind `dataset` feature): `id()`, `label()`, `available_datasets()`, `fetch_samples(dataset, limit)` → `Result<(DatasetMeta, VecSource), String>`
- `HuggingFaceProvider`: fetches rows from HuggingFace Dataset Server API, decodes base64 PNG → raw pixels stored as `CellValue::Bytes`
- `ImageShape { width, height, channels }`, `DatasetMeta { name, classes, total_samples, image_shape }`
- `decode_png_to_raw()`: PNG bytes → flat pixel buffer (behind `dataset` feature)

## Visual Graph (`visual/`)

- `VisualGraph<D>`: dimension-generic container of `VNode<D>` + `VEdge`
- `GraphNode<D>` trait: label, tag, position, size, ports, children
- `Graphable` trait: `to_graph() → VisualGraph<2>` — blanket `render_graph()`/`capture_graph()`
- `GraphView`: interactive zoom/pan/breadcrumb viewport with hit testing
- `auto_layout()`: left-to-right layered DAG layout
- `tag_color(tag)`: consistent color mapping for operation types

## Vendor Optimization

Vendor paths (CUDA, ROCm, MKL, Metal) behind feature flags only. Default: `wgpu` + `rayon`.
New backends: implement `Kernel` trait → `Device::from_kernel()`.
