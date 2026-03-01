---
applyTo: "**"
---

# Any-Compute Instructions

High-performance, framework-agnostic compute and data visualization library in Rust.

## Architecture

```
crates/
  core/   — Data, layout, interaction, render primitives, compute, kernel, shader, animation. ZERO UI framework deps.
  rsx/    — Dioxus-based declarative components + hooks. RSX code ONLY lives here.
  ffi/    — C ABI bindings + codegen for Python, JS, Java, WASM, etc.
out/      — Benchmark reports, flamegraphs, profiling artifacts, generated bindings.
```

### Hard Rules

1. **Core never depends on rsx.** Dependency flows: `rsx → core`, `ffi → core`, `ffi --(feature)→ rsx`.
2. **No RSX markup outside `crates/rsx/`.** If you need declarative UI, add a component there.
3. **Virtualize everything.** Never load/render more data than the viewport requires.
4. **Workspace deps only.** All third-party crates are pinned in the root `[workspace.dependencies]` and inherited via `.workspace = true`.
5. **FFI safety boundary.** Every `anc_*_new` has a matching `anc_*_free`. Document ownership in every `# Safety` block.

## Feature Flags

All optional backends and hardware APIs are gated behind feature flags in `crates/core/Cargo.toml`:

| Flag           | Purpose                                    | Deps                  |
| -------------- | ------------------------------------------ | --------------------- |
| `cuda`         | NVIDIA CUDA kernel backend                 | (stub, vendor SDK)    |
| `rocm`         | AMD ROCm/HIP kernel backend                | (stub, vendor SDK)    |
| `mkl`          | Intel MKL kernel backend                   | (stub, vendor SDK)    |
| `metal`        | Apple Metal kernel backend                 | (stub)                |
| `wgpu-backend` | Cross-platform GPU via wgpu                | `wgpu`                |
| `shader`       | Shader compilation (WGSL/GLSL/SPIR-V)     | `naga`                |
| `hwinfo`       | Hardware detection for benchmarks           | `sysinfo`, `num_cpus` |

## Single Source of Truth (SSOT)

**Before creating anything, check if it already exists.** If similar logic lives elsewhere, upgrade that instead. Never duplicate — always reference or generalize.

- `Lerp` trait (in `lib.rs`) is the **ONLY** interpolation interface. `Point`, `Size`, `Rect`, `Color`, `f64`, `f32` all implement it.
- `layout::Point` is the ONLY position type. Never add `x: f64, y: f64` fields to other structs — reference `Point`.
- `layout::Rect` is composed from `Point` + `Size` — never flatten these into raw fields.
- `render::Color` is the ONLY color type. All backends, components, animation use this.
- `render::Primitive` references `Rect` and `Point` — never duplicates spatial fields.
- `interaction::InputEvent` references `Point` for all positions, `Modifiers` for all key state.
- `compute::ComputeBackend` is the ONLY way to dispatch parallel work — never call `rayon` directly outside the CPU backend impl.
- `kernel::Kernel` trait is the ONLY low-level compute interface. CPU SIMD, CUDA, ROCm, MKL all implement it.
- `kernel::best_kernel()` auto-selects the fastest available kernel backend at runtime.
- `shader::ShaderCompiler` is the ONLY way to compile shaders. Uses `naga` for cross-compilation.
- `ffi::codegen::FfiRegistry` is the ONLY source of truth for cross-language bindings. All generators read from it.
- `hints::Hints` is the ONLY way to express optimization intent — never hardcode parallelism thresholds or cache decisions.
- `bench::format_bytes` uses `humansize::SizeFormatter` with `BINARY` — never manual `1024` division for byte formatting.
- `bench::BenchCategory` is the ONLY source of category identifiers — never hardcode category strings in runners.

## Generics & Trait Binding

- **Maximize generics.** `Transition<T: Lerp + Clone>` works with any interpolatable type. Prefer generic constraints over concrete types.
- **New interpolatable types** must implement `Lerp`. The animation system then works with them automatically.
- **Default type parameters** (e.g. `Transition<T = f64>`) keep the common case ergonomic while supporting advanced use.
- **Trait objects** (`&dyn ComputeBackend`) for runtime polymorphism where needed (benchmarks, backend selection).

## Auto-Optimization (Hints)

The `hints::Hints` system bridges user intent to low-level optimization. Users declare _what_ — the engine decides _how_.

- `Hints::cached()` → skip diff, cache aggressively
- `Hints::animated()` → pre-allocate interpolation buffers, skip cache
- `Hints::massive(n)` → always parallelize, double-buffer, prefetch
- `Hints::streaming()` → double-buffer, never cache
- **Philosophy:** "animate this div" should auto-select the best code path. Users never _need_ to know about buffer strategies — but _can_ override via `with_compute()`, `with_budget()`.
- `ComputeBackend` trait provides `*_hinted` methods that fall back to sequential for small data based on `Hints::parallelism_threshold()`.

## Device Simulation & Benchmarking

- `SimulatedBackend` wraps `CpuBackend` with `DeviceProfile` throttling to test on virtual hardware.
- Predefined profiles: `HIGH_END_DESKTOP`, `MID_RANGE_LAPTOP`, `LOW_END_MOBILE`, `EMBEDDED`, `WASM_BROWSER`.
- Benchmarks run 7 scenario categories across all profiles: data virtualization, compute, hints, layout, animation, render list, lerp throughput.
- Reports output to `out/bench-<device>.json`. Run: `cargo run --release --bin anc-bench`
- Full benchmark with hardware detection requires `hwinfo` feature: `cargo run --release --features hwinfo --bin anc-bench`
- Benchmark CLI detects: CPU brand, cores, frequency, RAM, SIMD capabilities, enabled feature flags.
- Comparison tables show throughput vs rayon, std::iter, polars, OpenBLAS, Intel MKL, cuBLAS, numpy, React, Angular, vanilla JS DOM, Dioxus, CSS Transitions, Web Animations API.

## Code-in-Files (Never Embed Code in Strings)

All non-Rust code lives in files with native extensions, loaded via `include_str!` at compile time:

- **CSS**: `crates/rsx/assets/bench.css` — loaded by bench_window.rs.
- **WGSL shaders**: `crates/core/shaders/{map,reduce,gemm}.wgsl` — parameterized templates with `{{PLACEHOLDER}}` markers.
- **FFI templates**: `crates/ffi/templates/{wrapper.py,test.py,wrapper.js,test.js,types.d.ts,AnyCompute.java,AnyComputeTest.java}` — language-native files with `{{PLACEHOLDER}}` markers.
- **Template instantiation**: `instantiate(&[(key, value)])` replaces `{{KEY}}` markers at runtime. Used in `shader::templates` and `ffi::codegen::tpl`.
- **Rule**: Never embed code (CSS, WGSL, Python, JS, Java, SQL, etc.) as Rust string literals. Always use the language's native file extension + `include_str!`.

## Named Constants & External Crates

- All benchmark data sizes, thresholds, and magic numbers are named constants at the top of `bench.rs`.
- Use `humansize` crate (BINARY format) for byte formatting — never manual `/ 1024` chains.
- `BenchCategory` enum is self-describing: `id()` (machine-readable), `label()` (human display), `group()` (classification).

## GPU / Compute Architecture

The `compute::ComputeBackend` trait abstracts over high-level parallel work. The `kernel::Kernel` trait abstracts over low-level element-wise operations.

| Vendor | API stack              | Key optimization                         |
| ------ | ---------------------- | ---------------------------------------- |
| NVIDIA | Vulkan / CUDA / OptiX  | Warp=32, shared mem tiling, tensor cores |
| AMD    | Vulkan / ROCm / HIP    | Wavefront=64 (RDNA=32), LDS              |
| Intel  | Vulkan / oneAPI / SYCL | Subgroup 8/16/32, zero-copy on iGPU      |
| Apple  | Metal / MPS            | Via wgpu Metal backend                   |
| Web    | WebGPU                 | Via wgpu web backend                     |
| CPU    | rayon + SIMD           | Cache-line aware, avoid false sharing    |

**wgpu** is the recommended cross-platform GPU backend. Vendor-specific (CUDA, ROCm) go behind feature flags.

### Kernel System (`kernel.rs`)

Low-level compute kernels with runtime SIMD auto-detection:

- `CpuSimdKernel` detects AVX-512, AVX2, SSE4.2, NEON, SIMD128 at runtime via `detect_simd()`.
- `CudaKernel`, `RocmKernel`, `MklKernel` are feature-gated stubs — implement when vendor SDKs are available.
- `best_kernel()` auto-selects the fastest available kernel at runtime.
- Operations: `map_unary_f64`, `map_binary_f64`, `reduce_f64`, `scan_f64`, `gemm_f64`, `sort_f64`, `gather_f64`, `scatter_f64`.
- 15 unary ops (Neg, Abs, Sqrt, Exp, Log, Sin, Cos, Tanh, Relu, Sigmoid, etc.), 7 binary ops, 5 reduce ops.
- `benchmark_op()` measures throughput for any `KernelOp` with configurable data size.

### Shader System (`shader.rs`)

Cross-compilation of compute shaders via `naga` (behind `shader` feature flag):

- Parse WGSL, GLSL, SPIR-V source → `ShaderObject` with metadata (bindings, workgroup size).
- Cross-compile to any target format: `to_wgsl()`, `to_spirv()`, `to_glsl()`.
- Pre-built templates: `map_shader()`, `reduce_shader()`, `gemm_shader()` generate parameterized WGSL compute shaders.
- Without `shader` feature: `ShaderCompiler::compile()` returns `ShaderError::FeatureDisabled`.

## FFI & Cross-Language Codegen

The `ffi` crate provides both the C ABI surface and automated binding generators.

- `FfiRegistry` holds the complete type-annotated FFI surface — single source of truth for all generators.
- `FfiRegistry::default_any_compute()` returns the current functions: `anc_source_new`, `anc_source_add_column`, `anc_source_push_row_ints`, `anc_source_free`.
- `generate_python()` → ctypes wrapper + pytest tests.
- `generate_javascript()` → WASM wrapper + vitest tests + TypeScript `.d.ts` types.
- `generate_java()` → Panama FFM (Java 22+) wrapper + JUnit 5 tests.
- `generate_all()` writes all bindings to disk.
- Run: `cargo run --bin anc-codegen` (outputs to `out/bindings/` by default).
- When adding new FFI functions: register them in `FfiRegistry::default_any_compute()` and all language bindings update automatically.

## Animation / Transitions

Core owns the timing engine (`animation::Transition<T>`, `animation::Easing`). RSX wraps it in hooks.

- `Transition<T: Lerp + Clone>` is generic — works with `f64`, `Point`, `Color`, `Rect`, or any `Lerp` type.
- Easing functions match CSS spec (`ease-in`, `ease-out`, `ease-in-out`, `cubic-bezier`).
- `TransitionManager` orchestrates multiple named `Transition<f64>` instances; for other types use `Transition<T>` directly.
- The animation system only holds timing + easing and calls `Lerp::lerp` — single source of truth for blending.

## Event Model

Web-like three-phase propagation: Capture → Target → Bubble.

- `EventContext` wraps events with `stop_propagation()` / `prevent_default()`.
- `Modifiers` tracks shift/ctrl/alt/meta.
- `InputEvent` covers pointer, keyboard, focus/blur, scroll.

## Performance Principles

- Arena allocation (`bumpalo`) for per-frame temporaries
- Parallel iteration via `ComputeBackend` (not raw rayon)
- `SmallVec` for small-count inline storage
- Only fetch/render the _visible window_ of rows (`ScrollState::visible_range`)
- Incremental dirty tracking — repaint only what changed
- Run `cargo run --release --bin anc-bench` to generate reports in `out/`

## General Instructions

- Always try to support as many other libs as possible
- New render backends implement `core::render::RenderBackend` trait
- New compute backends implement `core::compute::ComputeBackend` trait
