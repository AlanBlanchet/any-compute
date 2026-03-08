---
name: canvas
description: GPU renderer, headless capture, scenario replay, visual comparison — crates/canvas/
applyTo: "crates/canvas/**,examples/dom/**"
---

# Canvas — `crates/canvas/`

Reusable GPU renderer, headless capture, and scenario replay.
Separated from bench so any consumer (examples, tests, CI) can render a `RenderList`.

## Crate Structure

| File/Dir              | Purpose                                                        |
| --------------------- | -------------------------------------------------------------- |
| `src/gpu.rs`          | wgpu renderer — windowed + headless, paint + capture           |
| `src/scenario.rs`     | Action/StepResult/Scenario, replay free functions, 5 tests     |
| `src/theme.rs`        | Catppuccin Mocha palette constants (single source of truth)    |
| `src/lib.rs`          | Module declarations, re-exports `winit`, `PALETTE_CSS`, `DEFAULT_VIEWPORT` |
| `shaders/rect.wgsl`   | SDF rounded-rect + border shader (loaded via `include_str!`)   |
| `fixtures/palette.css` | Catppuccin Mocha `:root` CSS variables — canonical color source |
| `fixtures/`           | External CSS/HTML for visual comparison + scenario binaries    |
| `src/bin/visual_cmp.rs` | Visual CSS comparison against Chrome screenshots             |
| `src/bin/scenario.rs` | Headless scenario runner — replay + capture PNGs               |

## GPU Renderer (`gpu.rs`)

- WGSL shader: SDF rounded rectangles, per-pixel anti-aliased corners + borders
- `InstanceData`: bounds, fill, params (corner_radius, border_width), border_color — 64 bytes
- Premultiplied alpha blending
- `Gpu::init(window)` — windowed mode with surface
- `Gpu::init_headless(w, h)` — no window, capture-only
- `prepare()` + `draw()` — shared internal helpers
- `paint(&RenderList)` — render to window surface + present
- `capture(&RenderList) → (w, h, rgba)` — offscreen render to CPU bytes
- `capture_png(&RenderList, path)` — capture + BGRA→RGBA + save PNG

## Theme (`theme.rs`)

Catppuccin Mocha Rust-side constants: `BG`, `SURFACE_BRIGHT`, `TEXT_DIM`,
`GREEN`, `BLUE`, `RED`, `YELLOW`, `MAUVE`, `SIDEBAR_BG`, `ACCENT`, `BAR_COLORS`.
Import as `use any_compute_canvas::theme;`.

## Scenario Replay (`scenario.rs`)

- `Action`: Click, Hover, Scroll, Dispatch, AssertTag, Capture
- `StepResult`: constructors `dispatched`, `silent`, `asserted`, `captured`
- `replay_step(tree, action, index)` — single action → StepResult
- `replay(tree, scenario)` — full sequence → Vec<StepResult>
- 5 unit tests covering all action types

## Fixtures

All external content loaded via `include_str!` — never inline markup in Rust.

- `fixtures/palette.css` — Catppuccin Mocha `:root` CSS variables; exported as `PALETTE_CSS` from `lib.rs`
- `fixtures/visual_test.css` — CSS for visual comparison binary (uses `var()` referencing palette)
- `fixtures/visual_test.html` — HTML body for visual comparison binary

## Shared Constants (`lib.rs`)

- `PALETTE_CSS: &str` — prepend before any app CSS so `var(--base)` etc. resolve
- `DEFAULT_VIEWPORT: Size` — 800×600 default for visual tools (visual-cmp, scenario, playground)

## Running

```sh
make visual-cmp  # visual CSS comparison against Chrome
make scenario    # headless scenario replay + PNGs
make dom         # interactive DOM playground (examples/dom)
```

## Dependencies

- `any-compute-core` — layout, render, interaction, animation types
- `any-compute-dom` — Tree, StyleSheet, parse
- `wgpu`, `glyphon`, `winit`, `pollster`, `bytemuck`, `png` — all behind `gpu` feature
