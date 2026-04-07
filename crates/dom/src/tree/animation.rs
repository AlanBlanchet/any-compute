use crate::css::{AnimationDirection, AnimationFillMode, AnimationIterCount, Keyframe};
use crate::style::{Style, apply_ops};

/// Shared timing data for both transitions and animations — the core
/// delay/duration/easing logic is defined exactly once.
#[derive(Debug, Clone)]
pub struct Timing {
    pub elapsed: f64,
    pub duration: f64,
    pub delay: f64,
    pub easing: any_compute_core::animation::Easing,
}

impl Timing {
    /// Seconds of active playback (after delay has passed).
    #[inline]
    pub fn active_time(&self) -> f64 {
        (self.elapsed - self.delay).max(0.0)
    }

    /// Whether enough time has elapsed to be past the delay.
    #[inline]
    pub fn started(&self) -> bool {
        self.elapsed >= self.delay
    }

    /// Linear progress in [0,1] — delay-aware, clamped, **no** easing.
    pub fn raw_progress(&self) -> f64 {
        if !self.started() {
            return 0.0;
        }
        if self.duration <= 0.0 {
            return 1.0;
        }
        (self.active_time() / self.duration).clamp(0.0, 1.0)
    }

    /// Progress with easing curve applied.
    pub fn eased_progress(&self) -> f64 {
        self.easing.apply(self.raw_progress())
    }

    /// True when the single iteration is complete.
    pub fn finished(&self) -> bool {
        self.elapsed >= self.delay + self.duration
    }
}

/// Result of a [`Tree::tick`] call.
#[derive(Debug, Clone, Copy, Default)]
pub struct TickResult {
    /// Any animations/transitions still running — caller should redraw.
    pub active: bool,
}

/// Active CSS property transition — interpolates one `StyleOp` from→to.
#[derive(Debug, Clone)]
pub struct ActiveTransition {
    pub property: String,
    pub from: Style,
    pub to: Style,
    pub timing: Timing,
}

impl ActiveTransition {
    /// Normalized progress \in [0,1] with easing applied.
    pub fn progress(&self) -> f64 {
        self.timing.eased_progress()
    }

    /// True if this transition has completed.
    pub fn finished(&self) -> bool {
        self.timing.finished()
    }

    /// Build from a `TransitionSpec` + before/after snapshots.
    pub(super) fn from_spec(spec: &crate::css::TransitionSpec, from: Style, to: Style) -> Self {
        Self {
            property: spec.property.clone(),
            from,
            to,
            timing: Timing {
                elapsed: 0.0,
                duration: spec.duration_secs,
                delay: spec.delay_secs,
                easing: spec.easing,
            },
        }
    }
}

/// Active CSS @keyframes animation on a node.
#[derive(Debug, Clone)]
pub struct ActiveAnimation {
    pub name: String,
    pub keyframes: Vec<Keyframe>,
    pub timing: Timing,
    pub iteration_count: AnimationIterCount,
    pub direction: AnimationDirection,
    pub fill_mode: AnimationFillMode,
    pub iterations_done: f64,
}

impl ActiveAnimation {
    /// Current normalized progress \in [0,1] within the current iteration.
    pub fn progress(&self) -> f64 {
        if !self.timing.started() {
            return match self.fill_mode {
                AnimationFillMode::Backwards | AnimationFillMode::Both => 0.0,
                _ => 0.0,
            };
        }
        let active = self.timing.active_time();
        if self.timing.duration <= 0.0 {
            return 1.0;
        }
        let raw_iter = active / self.timing.duration;
        let iter_frac = raw_iter.fract();
        let iter_num = raw_iter.floor();

        // Check if finished
        match self.iteration_count {
            AnimationIterCount::Count(n) if iter_num >= n => {
                return match self.fill_mode {
                    AnimationFillMode::Forwards | AnimationFillMode::Both => 1.0,
                    _ => 0.0,
                };
            }
            _ => {}
        }

        let t = match self.direction {
            AnimationDirection::Normal => iter_frac,
            AnimationDirection::Reverse => 1.0 - iter_frac,
            AnimationDirection::Alternate => {
                if (iter_num as u64) % 2 == 0 {
                    iter_frac
                } else {
                    1.0 - iter_frac
                }
            }
            AnimationDirection::AlternateReverse => {
                if (iter_num as u64) % 2 == 0 {
                    1.0 - iter_frac
                } else {
                    iter_frac
                }
            }
        };
        self.timing.easing.apply(t)
    }

    /// True if this animation has completed all iterations.
    pub fn finished(&self) -> bool {
        if !self.timing.started() {
            return false;
        }
        match self.iteration_count {
            AnimationIterCount::Infinite => false,
            AnimationIterCount::Count(n) => self.timing.active_time() >= self.timing.duration * n,
        }
    }

    /// Apply current keyframe interpolation to a style.
    pub fn apply_to(&self, style: &mut Style) {
        let t = self.progress();
        if self.keyframes.is_empty() {
            return;
        }

        // Single-pass keyframe bracket search: find largest stop <= t (prev)
        // and smallest stop >= t (next).
        let mut prev = &self.keyframes[0];
        let mut next = self.keyframes.last().unwrap();
        for kf in &self.keyframes {
            if kf.stop <= t {
                prev = kf;
            }
            if kf.stop >= t && kf.stop < next.stop {
                next = kf;
            }
        }

        if (next.stop - prev.stop).abs() < f64::EPSILON {
            // Exact match — apply directly
            apply_ops(style, &prev.ops);
        } else {
            // Interpolate between surrounding keyframes via full Style::lerp.
            // Clone the base style, apply each keyframe's ops independently,
            // then lerp.  Properties not touched by keyframes stay identical
            // in both copies, so the lerp is a no-op for them.
            let local_t = (t - prev.stop) / (next.stop - prev.stop);
            let mut prev_style = style.clone();
            apply_ops(&mut prev_style, &prev.ops);
            let mut next_style = style.clone();
            apply_ops(&mut next_style, &next.ops);
            *style = prev_style.lerp(&next_style, local_t);
        }
    }
}

// ── Node identity (re-exported from core) ───────────────────────────────────
