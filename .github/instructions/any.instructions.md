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
crates/core/    — zero-dep primitives: V<N>/Region<N>/Matrix<R,C>, Buffer, compute, kernels,
                  animation, events, layout, data, render, ops (composable trait cascades),
                  graph (DAG, lazy eval, ONNX), interaction (event/dispatch/state/text_input),
                  visual (VisualGraph<D>, Graphable, GraphView), tree (Arena, Dirty, flexbox)
crates/ai/      — neural network layers (Layer trait, Linear, Activation, BatchNorm, Residual,
                  Sequential), network builders (resnet, vit, gpt, mixer), ML taxonomy
                  (TaskType, ModelKind, DatasetSource), dataset providers (HuggingFace Hub)
crates/dom/     — arena scene graph, flexbox layout, HTML/CSS parsers, GPU renderer (gpu feature),
                  Page runtime (HTML+CSS+JS), theme, harness, scenario pixel assertions
crates/js/      — V8-like JS engine: lexer → parser → bytecode → register VM → runtime
crates/bench/   — DOM perf comparisons, GPU dashboard window
crates/ffi/     — C ABI + codegen → Python, JS (WASM), Java, Node.js
examples/dom/   — interactive DOM playground (make dom)
examples/showcase/ — unified showcase: Browser, 3D Scene, Compute, AI tabs
                     app.rs (shared AppData), graph mode toggle, split layouts
bindings/       — generated framework adapters (React, Vue, Svelte, Angular, Python, Node)
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
| `png`          | PNG encoding for Capture/Record/PixelBuffer export  |
| `wgpu-backend` | Cross-platform GPU (Vulkan / Metal / DX12 / WebGPU) |
| `cuda`         | NVIDIA CUDA (stub — requires vendor SDK)            |
| `rocm`         | AMD ROCm / HIP (stub)                               |
| `mkl`          | Intel oneMKL (stub)                                 |
| `metal`        | Apple Metal (stub)                                  |
| `shader`       | Shader compilation via naga (WGSL / GLSL / SPIR-V)  |
| `hwinfo`       | Hardware detection for benchmarks                   |

### `crates/ai/Cargo.toml`

| Flag      | Purpose                                               |
| --------- | ----------------------------------------------------- |
| `dataset` | Dataset providers (HuggingFace Hub) — ureq/base64/png |

### `crates/dom/Cargo.toml`

| Flag  | Purpose                                                         |
| ----- | --------------------------------------------------------------- |
| `gpu` | GPU renderer (wgpu/glyphon/winit) — opt-in, keeps headless lean |

## Testing

Per-crate test targets via Makefile:

```
make test          # all crates (535 tests)
make test-core     # compute, kernel, layout, buffer, data, render, tree, flex, propagation, interaction, ops, graph, visual (285 tests)
make test-dom      # CSS, parsing, style, tree, scenario, conformance, page, visual regression (169 tests)
make test-visual   # GPU visual regression tests (9 tests, requires gpu feature)
make test-bench    # benchmark crate (no-default-features)
make test-ffi      # codegen, bindings (9 tests)
make test-js       # lexer, parser, bytecode, vm, runtime (44 tests)
```

## DOM GPU Feature

`crates/dom` default features are empty (`default = []`). GPU is opt-in via `features = ["gpu"]`.
Consumers that need GPU (dom-example, bench/window) explicitly request it.
This prevents pulling wgpu/winit into headless-only consumers.

## Logging

Library crates use `log` facade — zero configuration cost.
Application binaries configure `tracing-subscriber` (showcase) or `env_logger` (playground).
Override via `RUST_LOG` env var.

## Skills

Read the relevant skill file before touching that domain:

- `.github/skills/compute.skill.md` — `Device`, `V<N>`/`Region<N>`, `Buffer`, kernels, dispatch, ops traits, computational graph
- `.github/skills/animation.skill.md` — `Transition<T>`, `Lerp`, easing, RSX hooks
- `.github/skills/event.skill.md` — `InputEvent`, propagation, `EventContext`, `PropagationStrategy`
- `.github/skills/performance.skill.md` — allocation, dirty tracking, visible-range rendering
- `.github/skills/ffi.skill.md` — `FfiRegistry`, codegen, binding targets
- `.github/skills/canvas.skill.md` — REDIRECT → merged into dom (GPU renderer, harness, scenario)
- `.github/skills/bench.skill.md` — DOM perf comparisons, GPU dashboard (`crates/bench/`)
- `.github/skills/dom.skill.md` — `Tree`, `Slot`, `Style`, fault-tolerant `parse()`, inline CSS, GPU renderer, arena layout, scenario pixel assertions
- `.github/skills/js.skill.md` — JS engine: `Lexer`, `Parser`, `Compiler`, `Vm`, `JsValue`, `Runtime`
- `.github/skills/visual-testing.skill.md` — `TestHarness`, `Capture`, `PixelBuffer`, pixel assertions, cursor/scroll/hover behavioral tests
- `.github/skills/showcase.skill.md` — Showcase app: tabs, sidebar, shared AppData (app.rs), AI tab (AiState, datasets, graph mode, training), CSS cascade pitfalls, snapshot binary
