---
applyTo: "**"
---

# Any-Compute

High-performance, framework-agnostic compute and data visualization library in Rust.
One API. Every device. Every platform.

## Mission

- Abstract hardware complexity behind clean traits — users get the best code path automatically.
- Ship bindings for every major platform and framework without duplicating logic.
- Stay generic enough that any future target can be added without touching core.

## Architecture

```
crates/core/    — zero-dep primitives: V<N>/Region<N> spatial, Buffer, compute, kernels, animation, events, layout, data, render
crates/dom/     — arena-based scene graph, flexbox layout, HTML/CSS parsers, GPU renderer (behind `gpu` feature)
crates/bench/   — DOM perf comparisons, GPU dashboard window
crates/ffi/     — C ABI + codegen → Python, JS (WASM), Java, Node.js
examples/dom/   — interactive DOM playground (make dom)
bindings/       — generated + hand-reviewed framework adapters (React, Vue, Svelte, Angular, Python, Node)
out/            — benchmark reports, flamegraphs, generated headers
```

## Platforms

| Target                             | Status                        |
| ---------------------------------- | ----------------------------- |
| Linux / macOS / Windows (x64, ARM) | Native                        |
| Web (WASM)                         | `wasm-pack` via `crates/ffi`  |
| Android / iOS                      | C ABI via NDK / Swift interop |

## UI Frameworks

Web (React, Vue, Svelte, Angular, vanilla JS) — Desktop (Dioxus, egui, iced, wgpu native)

## Feature Flags

### `crates/core/Cargo.toml`

| Flag           | Purpose                                             |
| -------------- | --------------------------------------------------- |
| `wgpu-backend` | Cross-platform GPU (Vulkan / Metal / DX12 / WebGPU) |
| `cuda`         | NVIDIA CUDA (stub — requires vendor SDK)            |
| `rocm`         | AMD ROCm / HIP (stub)                               |
| `mkl`          | Intel oneMKL (stub)                                 |
| `metal`        | Apple Metal (stub)                                  |
| `shader`       | Shader compilation via naga (WGSL / GLSL / SPIR-V)  |
| `hwinfo`       | Hardware detection for benchmarks                   |

### `crates/dom/Cargo.toml`

| Flag  | Purpose                                                         |
| ----- | --------------------------------------------------------------- |
| `gpu` | GPU renderer (wgpu/glyphon/winit) — opt-in, keeps headless lean |

## Testing

Per-crate test targets via Makefile:
```
make test          # all crates (257 tests)
make test-core     # compute, kernel, layout, buffer, data, render (124 tests)
make test-dom      # CSS, parsing, style, tree, scenario, conformance (115 tests)
make test-bench    # benchmark crate (no-default-features)
make test-ffi      # codegen, bindings (9 tests)
```

## DOM GPU Feature

`crates/dom` default features are empty (`default = []`). GPU is opt-in via `features = ["gpu"]`.
Consumers that need GPU (dom-example, bench/window) explicitly request it.
This prevents pulling wgpu/winit into headless-only consumers.

## Skills

Read the relevant skill file before touching that domain:

- `.github/skills/compute.skill.md` — `Device`, `V<N>`/`Region<N>`, `Buffer`, kernels, dispatch
- `.github/skills/animation.skill.md` — `Transition<T>`, `Lerp`, easing, RSX hooks
- `.github/skills/event.skill.md` — `InputEvent`, propagation, `EventContext`
- `.github/skills/performance.skill.md` — allocation, dirty tracking, visible-range rendering
- `.github/skills/ffi.skill.md` — `FfiRegistry`, codegen, binding targets
- `.github/skills/canvas.skill.md` — REDIRECT → merged into dom (GPU renderer, harness, scenario)
- `.github/skills/bench.skill.md` — DOM perf comparisons, GPU dashboard (`crates/bench/`)
- `.github/skills/dom.skill.md` — `Tree`, `Slot`, `Style`, fault-tolerant `parse()`, GPU renderer, arena layout
