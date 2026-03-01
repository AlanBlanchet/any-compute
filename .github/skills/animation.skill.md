---
name: animation
description: Timing engine, interpolation, and transition lifecycle patterns
applyTo: "crates/core/src/animation.rs,crates/rsx/src/hooks.rs"
---

# Animation

## Single source of truth

- `Transition<T: Lerp + Clone>` is the **only** timing driver — no other code reimplements interpolation.
- `Lerp` trait is the single source of truth for blending any type (`f64`, `Point`, `Color`, `Rect`, user types).
- Easing maps exactly to CSS spec variants; don't add easing that isn't named after a CSS function.

## Composition

- `TransitionManager` composes named `Transition<f64>` instances — use for multiple simultaneous values.
- For non-`f64` types use `Transition<T>` directly; `TransitionManager` is a `f64` convenience wrapper.

## RSX integration

- `crates/rsx/src/hooks.rs` owns dioxus lifecycle only — no timing math lives there.
- Hooks (`use_transition`, `use_presence`) start/tick the core `Transition<T>` inside `use_signal`.
- Triggering a transition means calling `.start()` on the core type, not re-creating it.
