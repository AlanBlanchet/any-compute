---
name: performance
description: Memory, allocation, and rendering performance patterns
applyTo: "crates/core/src/**"
---

# Performance

## Allocation

- `bumpalo` arena for **per-frame temporaries** — reset each frame, not per-element.
- `SmallVec` for collections with a small, bounded inline count — never use `Vec` for ≤8-item collections.
- All parallel iteration goes through `ComputeBackend`, never raw `rayon`.

## Rendering

- Only compute/render the visible row window — `ScrollState::visible_range` is the authority.
- Incremental dirty tracking: repaint only changed nodes, never the full tree.
- Hints drive automatic optimization: static → cache aggressively; streaming → double-buffer + prefetch.

## Measurement

- Benchmarks live in `core::bench` only — no benchmark code outside that module.
- Run `cargo run --release --features hwinfo --bin anc-bench` for CLI reports to `out/`.
- Run `cargo run -p any-compute-rsx --features bench --bin anc-bench-window` for GUI dashboard.
