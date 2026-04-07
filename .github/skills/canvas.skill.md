---
name: canvas
description: MERGED INTO DOM — GPU renderer lives behind `gpu` feature; theme, harness, scenario are always available
applyTo: "crates/dom/**"
---

# Canvas → DOM merge

`crates/canvas/` no longer exists. All GPU, harness, scenario, and theme code lives in `crates/dom/`.
GPU: `use any_compute_dom::gpu::Gpu;` (requires `gpu` feature).
CPU-only: `use any_compute_dom::{theme, harness, scenario};` (no feature flags).
See **dom.skill.md** for details.
