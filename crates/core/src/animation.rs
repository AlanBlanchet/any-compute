//! Animation / transition engine — time-based interpolation, easing, state machines.
//!
//! Uses the [`Lerp`] trait as the single source of truth for interpolation.
//! `Point::lerp`, `Rect::lerp`, `Color::lerp` all implement `Lerp` —
//! the animation system only holds timing + easing and calls `Lerp::lerp`.

use crate::Lerp;
use std::time::{Duration, Instant};

/// Easing functions — standard CSS/web-compatible set.
///
/// Each variant is a pure `fn(f64) -> f64` mapping `[0..1] → [0..1]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Easing {
    #[default]
    Linear,
    EaseIn,
    EaseOut,
    EaseInOut,
    /// Cubic bezier defined by two control points (CSS `cubic-bezier(x1, y1, x2, y2)`).
    /// Stored in the transition, not here, to keep the enum Copy-friendly.
    CubicBezier,
}

impl Easing {
    /// Parse a CSS easing function name.
    ///
    /// `"ease"` maps to `EaseInOut` (the CSS `ease` keyword is a cubic-bezier shortcut).
    /// Unknown values default to `EaseInOut` (the CSS default for `transition-timing-function`).
    pub fn from_css(val: &str) -> Self {
        match val.trim() {
            "linear" => Self::Linear,
            "ease-in" => Self::EaseIn,
            "ease-out" => Self::EaseOut,
            "ease-in-out" | "ease" => Self::EaseInOut,
            v if v.starts_with("cubic-bezier(") => Self::CubicBezier,
            _ => Self::EaseInOut,
        }
    }

    /// Evaluate the easing curve at time `t` ∈ [0, 1].
    pub fn apply(self, t: f64) -> f64 {
        let t = t.clamp(0.0, 1.0);
        match self {
            Self::Linear => t,
            Self::EaseIn => t * t * t,
            Self::EaseOut => 1.0 - (1.0 - t).powi(3),
            Self::EaseInOut => {
                if t < 0.5 {
                    4.0 * t * t * t
                } else {
                    1.0 - (-2.0 * t + 2.0).powi(3) / 2.0
                }
            }
            // CubicBezier is handled separately with control points.
            Self::CubicBezier => t,
        }
    }
}

/// State of a single transition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransitionState {
    /// Not yet started / idle.
    Idle,
    /// Actively interpolating.
    Running,
    /// Completed (holds at target value).
    Finished,
}

/// A generic transition — drives any [`Lerp`] type from `from` to `to` over time.
///
/// The timing engine produces `t ∈ [0,1]` via easing; the interpolation
/// is delegated to `T::lerp` — single source of truth for blending.
#[derive(Debug, Clone)]
pub struct Transition<T: Lerp + Clone = f64> {
    pub from: T,
    pub to: T,
    pub duration: Duration,
    pub easing: Easing,
    pub delay: Duration,
    started_at: Option<Instant>,
    state: TransitionState,
}

impl<T: Lerp + Clone> Transition<T> {
    pub fn new(from: T, to: T, duration: Duration) -> Self {
        Self {
            from,
            to,
            duration,
            easing: Easing::default(),
            delay: Duration::ZERO,
            started_at: None,
            state: TransitionState::Idle,
        }
    }

    pub fn with_easing(mut self, easing: Easing) -> Self {
        self.easing = easing;
        self
    }

    pub fn with_delay(mut self, delay: Duration) -> Self {
        self.delay = delay;
        self
    }

    /// Start (or restart) the transition now.
    pub fn start(&mut self) {
        self.started_at = Some(Instant::now());
        self.state = TransitionState::Running;
    }

    /// Current state.
    pub fn state(&self) -> TransitionState {
        self.state
    }

    /// Raw linear progress [0..1] accounting for delay.
    fn raw_progress(&self) -> f64 {
        let Some(start) = self.started_at else {
            return 0.0;
        };
        let elapsed = start.elapsed();
        if elapsed < self.delay {
            return 0.0;
        }
        let active = elapsed - self.delay;
        if self.duration.is_zero() {
            return 1.0;
        }
        (active.as_secs_f64() / self.duration.as_secs_f64()).clamp(0.0, 1.0)
    }

    /// Eased progress [0..1].
    pub fn progress(&mut self) -> f64 {
        let raw = self.raw_progress();
        if raw >= 1.0 {
            self.state = TransitionState::Finished;
        }
        self.easing.apply(raw)
    }

    /// Current interpolated value — uses `T::lerp`.
    pub fn value(&mut self) -> T {
        let t = self.progress();
        self.from.clone().lerp(self.to.clone(), t)
    }

    /// Is the transition still running?
    pub fn is_running(&self) -> bool {
        self.state == TransitionState::Running
    }
}

/// Manages multiple named transitions — the orchestration layer.
///
/// Components register transitions by key; the manager ticks them each frame.
/// Uses `Transition<f64>` for the common case; for other types, use `Transition<T>` directly.
#[derive(Debug, Default)]
pub struct TransitionManager {
    transitions: Vec<(String, Transition<f64>)>,
}

impl TransitionManager {
    pub fn add(&mut self, key: impl Into<String>, transition: Transition<f64>) {
        let key = key.into();
        // Replace if key already exists.
        if let Some(entry) = self.transitions.iter_mut().find(|(k, _)| *k == key) {
            entry.1 = transition;
        } else {
            self.transitions.push((key, transition));
        }
    }

    /// Start all idle transitions.
    pub fn start_all(&mut self) {
        for (_, t) in &mut self.transitions {
            if t.state() == TransitionState::Idle {
                t.start();
            }
        }
    }

    /// Get the current value of a transition by key.
    pub fn value(&mut self, key: &str) -> Option<f64> {
        self.transitions
            .iter_mut()
            .find(|(k, _)| k == key)
            .map(|(_, t)| t.value())
    }

    /// Are any transitions still running?
    pub fn any_running(&self) -> bool {
        self.transitions.iter().any(|(_, t)| t.is_running())
    }

    /// Remove all finished transitions.
    pub fn gc(&mut self) {
        self.transitions
            .retain(|(_, t)| t.state() != TransitionState::Finished);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::{Point, Rect};
    use crate::render::Color;

    #[test]
    fn transition_idle_returns_from() {
        let mut t = Transition::new(10.0_f64, 90.0, Duration::from_millis(300));
        assert_eq!(t.state(), TransitionState::Idle);
        assert!((t.value() - 10.0).abs() < 1e-10);
    }

    #[test]
    fn transition_zero_duration_jumps() {
        let mut t = Transition::new(0.0_f64, 100.0, Duration::ZERO);
        t.start();
        assert!((t.value() - 100.0).abs() < 1e-10);
        assert_eq!(t.state(), TransitionState::Finished);
    }

    #[test]
    fn easing_linear_identity() {
        for i in 0..=10 {
            let t = i as f64 / 10.0;
            assert!((Easing::Linear.apply(t) - t).abs() < 1e-10);
        }
    }

    #[test]
    fn easing_endpoints() {
        for e in [
            Easing::Linear,
            Easing::EaseIn,
            Easing::EaseOut,
            Easing::EaseInOut,
        ] {
            assert!((e.apply(0.0)).abs() < 1e-10, "{e:?} at 0");
            assert!((e.apply(1.0) - 1.0).abs() < 1e-10, "{e:?} at 1");
        }
    }

    #[test]
    fn easing_clamps() {
        assert_eq!(Easing::Linear.apply(-0.5), 0.0);
        assert_eq!(Easing::Linear.apply(1.5), 1.0);
    }

    #[test]
    fn transition_generic_point() {
        let mut t = Transition::new(
            Point::new(0.0, 0.0),
            Point::new(100.0, 200.0),
            Duration::ZERO,
        );
        t.start();
        let v = t.value();
        assert!((v.x - 100.0).abs() < 1e-10);
        assert!((v.y - 200.0).abs() < 1e-10);
    }

    #[test]
    fn transition_generic_color() {
        let mut t = Transition::new(Color::BLACK, Color::WHITE, Duration::ZERO);
        t.start();
        let v = t.value();
        assert_eq!(v, Color::WHITE);
    }

    #[test]
    fn transition_generic_rect() {
        let mut t = Transition::new(
            Rect::new(0.0, 0.0, 100.0, 50.0),
            Rect::new(50.0, 50.0, 200.0, 100.0),
            Duration::ZERO,
        );
        t.start();
        let v = t.value();
        assert!((v.x() - 50.0).abs() < 1e-10);
    }

    #[test]
    fn transition_with_easing() {
        let t = Transition::new(0.0, 1.0, Duration::from_secs(1)).with_easing(Easing::EaseInOut);
        assert_eq!(t.easing, Easing::EaseInOut);
    }

    #[test]
    fn transition_with_delay() {
        let t = Transition::new(0.0, 1.0, Duration::from_secs(1))
            .with_delay(Duration::from_millis(500));
        assert_eq!(t.delay, Duration::from_millis(500));
    }

    #[test]
    fn manager_add_and_get() {
        let mut mgr = TransitionManager::default();
        mgr.add("opacity", Transition::new(0.0, 1.0, Duration::ZERO));
        mgr.start_all();
        assert!((mgr.value("opacity").unwrap() - 1.0).abs() < 1e-10);
        assert!(mgr.value("nonexistent").is_none());
    }

    #[test]
    fn manager_gc_removes_finished() {
        let mut mgr = TransitionManager::default();
        mgr.add("a", Transition::new(0.0, 1.0, Duration::ZERO));
        mgr.start_all();
        let _ = mgr.value("a"); // finishes it
        mgr.gc();
        assert!(mgr.value("a").is_none());
    }

    #[test]
    fn manager_replaces_existing_key() {
        let mut mgr = TransitionManager::default();
        mgr.add("x", Transition::new(0.0, 50.0, Duration::ZERO));
        mgr.add("x", Transition::new(0.0, 99.0, Duration::ZERO));
        mgr.start_all();
        assert!((mgr.value("x").unwrap() - 99.0).abs() < 1e-10);
    }
}
