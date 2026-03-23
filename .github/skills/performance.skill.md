---
name: performance
description: Memory, allocation, and rendering performance patterns
applyTo: "crates/core/src/**"
---

# Performance

## Allocation

- `bumpalo` arena for **per-frame temporaries** — reset each frame, not per-element.
- `SmallVec` for collections with a small, bounded inline count — never use `Vec` for ≤8-item collections.
- All parallel iteration goes through `Device`, never raw `rayon`.

## Rendering

- Only compute/render the visible row window — `ScrollState::visible_range` is the authority.
- Incremental dirty tracking: repaint only changed nodes, never the full tree.
- Hints drive automatic optimization: static → cache aggressively; streaming → double-buffer + prefetch.

## Measurement

- Benchmarks live in `crates/bench/` — compute runners in `runner.rs`, DOM comparisons in `lib.rs`.
- `core` has zero benchmark code — it only exposes compute primitives and `FEATURES` for feature detection.
- Run `cargo run -p any-compute-bench --release --features hwinfo --bin anc-bench` for CLI reports to `out/`.
- Run `cargo run -p any-compute-bench --bin anv-bench-window` (or `make dashboard`) for GPU dashboard.
