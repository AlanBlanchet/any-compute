//! Dioxus hooks that wrap core animation/transition primitives.
//!
//! These hooks provide reactive access to `any-compute-core::animation` —
//! the core owns the math, these hooks own the dioxus lifecycle integration.

use any_compute_core::animation::{Easing, Transition};
use dioxus::prelude::*;
use std::time::Duration;

/// Hook that drives a single animated `f64` value.
///
/// Returns `(current_value, start_fn)`.
///
/// ```ignore
/// let (opacity, start) = use_transition(cx, 0.0, 1.0, Duration::from_millis(300), Easing::EaseOut);
/// ```
pub fn use_transition(
    from: f64,
    to: f64,
    duration: Duration,
    easing: Easing,
) -> (f64, Signal<bool>) {
    let mut transition = use_signal(|| Transition::new(from, to, duration).with_easing(easing));
    let trigger = use_signal(|| false);

    // When trigger flips to true, start the transition.
    use_effect(move || {
        if *trigger.read() {
            transition.write().start();
        }
    });

    let val = transition.write().value();
    (val, trigger)
}

/// Hook for a simple boolean presence animation (mount/unmount fade).
///
/// Returns eased progress [0..1]. 0 = hidden, 1 = fully visible.
pub fn use_presence(visible: bool, duration: Duration) -> f64 {
    let mut transition = use_signal(|| {
        let mut t = Transition::new(0.0, 1.0, duration).with_easing(Easing::EaseInOut);
        if visible {
            t.start();
        }
        t
    });

    use_effect(move || {
        let mut t = transition.write();
        if visible {
            *t = Transition::new(t.value(), 1.0, duration).with_easing(Easing::EaseInOut);
        } else {
            *t = Transition::new(t.value(), 0.0, duration).with_easing(Easing::EaseInOut);
        }
        t.start();
    });

    transition.write().value()
}
