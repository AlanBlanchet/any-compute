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
crates/core/    — zero-dep primitives: compute, kernels, animation, events, layout, data, render
crates/ffi/     — C ABI + codegen → Python, JS (WASM), Java, Node.js
crates/rsx/     — Dioxus desktop UI (benchmark dashboard + component wrappers)
bindings/       — generated + hand-reviewed framework adapters (React, Vue, Svelte, Angular, Python, Node)
out/            — benchmark reports, flamegraphs, generated headers
```

## Platforms

| Target | Status |
|--------|--------|
| Linux / macOS / Windows (x64, ARM) | Native |
| Web (WASM) | `wasm-pack` via `crates/ffi` |
| Android / iOS | C ABI via NDK / Swift interop |

## UI Frameworks

Web (React, Vue, Svelte, Angular, vanilla JS) — Desktop (Dioxus, egui, iced, wgpu native)

## Feature Flags (`crates/core/Cargo.toml`)

| Flag | Purpose |
|------|---------|
| `wgpu-backend` | Cross-platform GPU (Vulkan / Metal / DX12 / WebGPU) |
| `cuda` | NVIDIA CUDA (stub — requires vendor SDK) |
| `rocm` | AMD ROCm / HIP (stub) |
| `mkl` | Intel oneMKL (stub) |
| `metal` | Apple Metal (stub) |
| `shader` | Shader compilation via naga (WGSL / GLSL / SPIR-V) |
| `hwinfo` | Hardware detection for benchmarks |

## Skills

Read the relevant skill file before touching that domain:

- `.github/skills/compute.skill.md` — `ComputeBackend`, dispatch, vendor backends
- `.github/skills/animation.skill.md` — `Transition<T>`, `Lerp`, easing, RSX hooks
- `.github/skills/event.skill.md` — `InputEvent`, propagation, `EventContext`
- `.github/skills/performance.skill.md` — allocation, dirty tracking, visible-range rendering
- `.github/skills/ffi.skill.md` — `FfiRegistry`, codegen, binding targets
- `.github/skills/bench.skill.md` — `BenchCategory`, comparisons, device profiles, dashboard
