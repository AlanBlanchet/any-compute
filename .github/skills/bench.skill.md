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
| `SHEET`         | `LazyLock<StyleSheet>` — parsed once, O(1) lookups  |
| `s(cls)`/`sm()` | Shorthand class resolution via `SHEET`               |
| `kv_row()`      | Key-value row helper (label 72px + value)            |
| `build_shell()` | Common sidebar + tab shell (returns content NodeId)  |

`kv_row` and `build_shell` use the global `SHEET` directly — no `&StyleSheet` parameter.
`window.rs` imports `SHEET` from `lib.rs` and defines local `s()`/`sm()` wrappers.

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
- Styling: Tailwind CSS utilities + bench.css component classes, merged via `combined_css()` at startup
- `theme` module provides const `Color` values for dynamic logic (bar graphs, clear color)
- `TAILWIND_CSS` exported from `any-compute-dom` — compiled Tailwind v3 subset, no runtime

### CSS Pipeline

- `combined_css()` in `lib.rs` concatenates `TAILWIND_CSS` + `BENCH_CSS` → one string
- `SHEET = LazyLock::new(|| StyleSheet::parse(&combined_css()))` — parsed once, O(1) lookups
- Tailwind utilities provide spacing, layout, colors; bench.css provides component classes
- Compound CSS classes (`.row-gap-8`, `.row-gap-12`, `.section-hdr`, `.small-dim`, `.heading-text`) reduce multi-class lookups to single `s()` calls
- Both parsed by the same CSS engine, same `StyleOp` compilation, zero duplication

### GPU Renderer

- WGSL shader uses **SDF rounded rectangles** (`sdf_round_rect`) for per-pixel anti-aliased corners
- `InstanceData`: bounds, fill color, params (corner_radius, border_width), border_color — 64 bytes/instance
- Premultiplied alpha blending (`PREMULTIPLIED_ALPHA_BLENDING` blend state)
- Border rendering via inner SDF: distance to outer edge < border_width → border color, else fill

### Event System (V8-like)

All winit events are converted to `InputEvent` and dispatched through `Tree::dispatch()`:

| winit Event          | InputEvent        | Action                                    |
| -------------------- | ----------------- | ----------------------------------------- |
| MouseInput Pressed   | PointerDown       | Set focus, track active tag               |
| MouseInput Released  | PointerUp         | Fire click if same tag as press (web model)|
| CursorMoved          | PointerMove       | Hover tracking → transition fade in/out   |
| CursorLeft           | —                 | Clear hover                               |
| MouseWheel           | Scroll            | Smooth scroll + dispatch                  |
| KeyboardInput        | KeyDown           | Tab/Arrow navigation, Enter/Space activate|
| Focused(false)       | —                 | Clear hover                               |
| ModifiersChanged     | —                 | Track modifier state                      |

- `HoverState` tracks hovered tag; emits `HoverDelta` → starts 120ms EaseOut fade transitions
- `FocusState` tracks focused tag for keyboard activation (Enter/Space)
- Pointer click only fires on release *if* released on the same tag as pressed (web behavior)
- `winit_key_to_string()` / `winit_button()` / `winit_modifiers()` convert winit types → our types

### Transitions & Animations

- `ease_transition(mgr, key, from, to, dur)` — centralized helper for all transitions
- `switch_tab(d, new)` — single source of truth for tab-switch animation (fade out old, fade in new, reset scroll)
- Tab switch: 180ms EaseOut via `TransitionManager`
- Hover: 120ms EaseOut fade, blended into background color at draw time via `Color::lerp`
- Buttons: hover brightens background by 15% toward white
- Scroll: exponential smoothing (0.18 speed, `scroll_y` lerps toward `scroll_target` each frame)

### Click / Keyboard Handling

- `handle_click(state, tag)` — dispatches tags: `"tab-N"` → `switch_tab`, `"run-bench"`, `"toggle-sim"`
- `handle_keyboard(state, key, mods)` — Tab/ArrowDown/ArrowUp cycle tabs, Enter/Space activate focused, Escape stops sim
- `handle_hover(state, tag)` — hover transition management
- **Critical**: tab buttons must stretch to fill the sidebar width (cross-axis stretch) — if they
  collapse to padding-only width, clicks miss them entirely
