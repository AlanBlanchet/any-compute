---
name: animation
description: Timing engine, interpolation, and transition lifecycle patterns
applyTo: "crates/core/**"
---

# Animation

## Single source of truth

- `Transition<T: Lerp + Clone>` is the **only** timing driver — no other code reimplements interpolation.
- `Lerp` trait is the single source of truth for blending any type (`f64`, `Point`, `Color`, `Rect`, user types).
- Easing maps exactly to CSS spec variants; don't add easing that isn't named after a CSS function.

## Composition

- `TransitionManager` composes named `Transition<f64>` instances — use for multiple simultaneous values.
- For non-`f64` types use `Transition<T>` directly; `TransitionManager` is a `f64` convenience wrapper.
- `TransitionManager::ease(key, from, to, dur, easing)` — shorthand to create and add in one call.
- **Do not call `gc()`** on long-lived managers where finished transitions should keep returning their final value (e.g. hover states). Finished transitions' `value()` returns their `to` field. `gc()` removes finished entries, causing `value()` to return `None` — which can snap visual state back to default.


