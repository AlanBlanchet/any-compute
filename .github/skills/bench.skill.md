---
name: bench
description: Benchmark crate structure, DOM perf comparisons, and GPU dashboard
applyTo: "crates/bench/**"
---

# Benchmarks — `crates/bench/`

Standalone benchmark crate. Depends on `any-compute-core` (compute,
layout, render) and `any-compute-dom` (CSS parsing, tree building, flexbox).

**No CSS/HTML tests here** — CSS parser correctness and fault-tolerance belong in
`crates/dom/`. This crate only benchmarks and provides the GPU dashboard.

## Running

```sh
make           # launches the GPU dashboard (default target)
make bench     # CLI benchmark (core crate)
cargo test -p any-compute-bench  # 1 integration test (dashboard build+layout)
```

## Crate Structure

| File         | Purpose                                                                    |
| ------------ | -------------------------------------------------------------------------- |
| `lib.rs`     | DOM perf benchmarks vs heap-per-node reference, shared constants + helpers |
| `window.rs`  | GPU dashboard binary (wgpu + glyphon + winit), feature-gated `window`      |
| `bench.css`  | Catppuccin Mocha theme — single source of truth for all styling            |
| `Cargo.toml` | `window` feature (default) gates GPU deps; lib has zero optional deps      |

## Shared Constants (exported from `lib.rs`)

| Const / fn      | Purpose                                             |
| --------------- | --------------------------------------------------- |
| `BENCH_CSS`     | Raw CSS text (`include_str!`)                       |
| `VIEWPORT`      | Default `Size(1400, 900)`                           |
| `VERSION`       | `"vX.Y.Z"` from `Cargo.toml`                        |
| `TAB_LABELS`    | `["Hardware", "Benchmarks", "Live Showdown"]`       |
| `sheet()`       | Parse and return the bench stylesheet               |
| `s(cls)`/`sm()` | Shorthand class resolution                          |
| `kv_row()`      | Key-value row helper (label 72px + value)           |
| `build_shell()` | Common sidebar + tab shell (returns content NodeId) |

`window.rs` imports these instead of duplicating. Theme color constants live in
`window.rs::theme` for wgpu clear color and dynamic bar graph coloring.

## DOM Performance Comparison

Compares our arena `Tree` against a naive `Box<RefNode>` heap-per-node reference tree
(mimicking browser DOM allocation patterns).

| Benchmark              | Node count | What it measures                    |
| ---------------------- | ---------- | ----------------------------------- |
| create flat 1K nodes   | 1001       | Allocation throughput               |
| create deep 500 chain  | 501        | Linked-list pattern                 |
| layout flat 1K         | 1001       | Flexbox solver vs heap creation     |
| paint 100 nodes        | 101        | Render list generation              |
| CSS parse (bench.css)  | —          | Parse throughput vs HashMap alloc   |
| CSS resolve 1K classes | —          | Lookup speed vs Style::default      |
| HTML parse (small doc) | 6          | Scanner throughput vs byte scanning |
| full frame (dashboard) | ~40        | Build + layout + paint end-to-end   |

## GPU Dashboard (`window` feature)

- `anv-bench-window` binary — launched via `make` (default target)
- Makefile target: `cargo run -p any-compute-bench --bin anv-bench-window`
- wgpu instanced draw + glyphon text
- Background threads via rayon: hardware detection, compute benchmarks, live throughput loop
- `build_tree()` constructs sidebar + tabs + tab-specific content builders per frame
- Three tabs: Hardware (system info), Benchmarks (results + comparisons), Live Showdown (sigmoid throughput)
- All styling via CSS classes from `bench.css`; `theme` module only for dynamic color logic

### GPU Renderer

- WGSL shader uses **SDF rounded rectangles** (`sdf_round_rect`) for per-pixel anti-aliased corners
- `InstanceData`: bounds, fill color, params (corner_radius, border_width), border_color — 64 bytes/instance
- Premultiplied alpha blending (`PREMULTIPLIED_ALPHA_BLENDING` blend state)
- Border rendering via inner SDF: distance to outer edge < border_width → border color, else fill

### Transitions & Smooth Scroll

- `TransitionManager` in `AppData` drives tab-switch animations
- Tab clicks start 180ms `EaseOut` fade transitions per tab (old fades out, new fades in)
- `build_tree()` reads transition values and blends between inactive/active colors via `Color::lerp`
- Scroll uses exponential smoothing: `scroll_y` lerps toward `scroll_target` each frame (0.18 speed)
- `build_tree` takes `&mut AppData` (via `MutexGuard`) since `TransitionManager::value` updates internal state

### Click Handling

- `tree.click(pos)` returns the tag of the clicked node (walks parents if needed)
- `handle_tag()` dispatches tags: `"tab-N"` → switch tab + start transitions, `"run-bench"`, `"toggle-sim"`
- **Critical**: tab buttons must stretch to fill the sidebar width (cross-axis stretch) — if they
  collapse to padding-only width, clicks miss them entirely (this was a layout solver bug, fixed by
  making `final_w` always stretch to `avail_w` when no explicit width set)
