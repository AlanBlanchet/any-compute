---
name: canvas
description: MERGED INTO DOM — GPU renderer, harness, scenario now live in crates/dom/ behind `gpu` feature
applyTo: "crates/dom/**"
---

# Canvas → DOM merge

The `crates/canvas/` crate no longer exists. All GPU rendering, test harness, scenario replay, and theme code now lives in `crates/dom/` behind the `gpu` feature flag.

See **dom.skill.md** for full documentation of these modules:

- `gpu.rs` — wgpu renderer (windowed + headless)
- `theme.rs` — Catppuccin Mocha palette constants
- `harness.rs` — TestHarness headless test driver
- `scenario.rs` — Action/StepResult/Scenario replay

Import path: `use any_compute_dom::{gpu, theme, harness, scenario};` (with `features = ["gpu"]`).

```

## Dependencies

- `any-compute-core` — layout, render, interaction, animation types
- `any-compute-dom` — Tree, StyleSheet, parse
- `wgpu`, `glyphon`, `winit`, `pollster`, `bytemuck`, `png` — all behind `gpu` feature
```
