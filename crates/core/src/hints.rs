//! Runtime optimization hints — how the engine auto-selects the best code path.
//!
//! Every element can carry [`Hints`] that describe its behavior at runtime.
//! The engine reads these to decide:
//! - Whether to cache layout / render output
//! - Whether to pre-allocate interpolation buffers
//! - Whether to batch changes or process immediately
//! - Which compute backend to prefer for this workload
//!
//! Users **never need to set hints** — defaults are conservative and correct.
//! But they can override any field for fine-grained control.
//!
//! ## Philosophy
//!
//! The user says "animate this div" — they shouldn't need to know that internally
//! we're double-buffering, pre-computing easing tables, or batching GPU uploads.
//! Hints bridge the gap: high-level intent → low-level optimization selection.

use crate::compute::BackendKind;

/// How frequently does this element's visual output change?
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Mutability {
    /// Never changes after initial render — cache aggressively.
    Static,
    /// Changes in response to user interaction (hover, click, etc.).
    #[default]
    Interactive,
    /// Continuously animated — pre-allocate buffers, avoid cache.
    Animated,
    /// Data is streaming/live — double-buffer, prefetch.
    Streaming,
}

/// How complex is the content? Guides batch size and parallelism decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Complexity {
    /// Few elements, simple shapes.
    #[default]
    Low,
    /// Moderate (hundreds of elements, some text).
    Medium,
    /// Heavy (thousands of elements, complex paths, large textures).
    High,
    /// Extreme (millions of data points, full-viewport coverage).
    Massive,
}

/// Preferred compute strategy. `Auto` lets the engine decide.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ComputePreference {
    /// Engine picks the best backend based on workload + available hardware.
    #[default]
    Auto,
    /// Force a specific backend.
    Prefer(BackendKind),
}

/// Cache strategy for rendered output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CachePolicy {
    /// Engine decides based on mutability.
    #[default]
    Auto,
    /// Always cache (even if animated — useful for expensive base layers).
    Always,
    /// Never cache (force re-render every frame).
    Never,
}

/// Runtime optimization hints for an element or subtree.
///
/// All fields default to `Auto` / engine-decides modes.
/// The user can override any field to control behavior.
#[derive(Debug, Clone, Copy, Default)]
pub struct Hints {
    pub mutability: Mutability,
    pub complexity: Complexity,
    pub compute: ComputePreference,
    pub cache: CachePolicy,
    /// Expected number of items (rows, points, etc.) — helps pre-allocate.
    /// 0 = unknown/auto.
    pub expected_count: usize,
    /// Target frame budget in microseconds. 0 = no constraint (best-effort).
    pub frame_budget_us: u32,
}

impl Hints {
    /// Hints for a static, rarely-changing element.
    pub fn cached() -> Self {
        Self {
            mutability: Mutability::Static,
            cache: CachePolicy::Always,
            ..Default::default()
        }
    }

    /// Hints for a continuously animated element.
    pub fn animated() -> Self {
        Self {
            mutability: Mutability::Animated,
            cache: CachePolicy::Never,
            ..Default::default()
        }
    }

    /// Hints for a massive dataset (millions of rows).
    pub fn massive(expected_count: usize) -> Self {
        Self {
            mutability: Mutability::Streaming,
            complexity: Complexity::Massive,
            expected_count,
            ..Default::default()
        }
    }

    /// Hints for streaming / live data.
    pub fn streaming() -> Self {
        Self {
            mutability: Mutability::Streaming,
            cache: CachePolicy::Never,
            ..Default::default()
        }
    }

    /// Override the compute preference.
    pub fn with_compute(mut self, pref: ComputePreference) -> Self {
        self.compute = pref;
        self
    }

    /// Override the frame budget.
    pub fn with_budget(mut self, us: u32) -> Self {
        self.frame_budget_us = us;
        self
    }

    /// Should the engine cache this element's output?
    pub fn should_cache(&self) -> bool {
        match self.cache {
            CachePolicy::Always => true,
            CachePolicy::Never => false,
            CachePolicy::Auto => matches!(self.mutability, Mutability::Static),
        }
    }

    /// Should the engine pre-allocate interpolation buffers?
    pub fn needs_interpolation_buffers(&self) -> bool {
        matches!(self.mutability, Mutability::Animated)
    }

    /// Should the engine use double-buffering for data access?
    pub fn needs_double_buffer(&self) -> bool {
        matches!(self.mutability, Mutability::Streaming)
    }

    /// Suggested parallelism threshold — below this, sequential is faster.
    pub fn parallelism_threshold(&self) -> usize {
        match self.complexity {
            Complexity::Low => 10_000,
            Complexity::Medium => 1_000,
            Complexity::High => 100,
            Complexity::Massive => 0, // always parallelize
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_hints_interactive() {
        let h = Hints::default();
        assert_eq!(h.mutability, Mutability::Interactive);
        assert_eq!(h.complexity, Complexity::Low);
        assert_eq!(h.cache, CachePolicy::Auto);
        assert_eq!(h.compute, ComputePreference::Auto);
    }

    #[test]
    fn cached_hints() {
        let h = Hints::cached();
        assert!(h.should_cache());
        assert!(!h.needs_interpolation_buffers());
        assert!(!h.needs_double_buffer());
    }

    #[test]
    fn animated_hints() {
        let h = Hints::animated();
        assert!(!h.should_cache());
        assert!(h.needs_interpolation_buffers());
        assert!(!h.needs_double_buffer());
    }

    #[test]
    fn massive_hints() {
        let h = Hints::massive(5_000_000);
        assert_eq!(h.expected_count, 5_000_000);
        assert!(h.needs_double_buffer());
        assert_eq!(h.parallelism_threshold(), 0); // always parallelize
    }

    #[test]
    fn streaming_hints() {
        let h = Hints::streaming();
        assert!(h.needs_double_buffer());
        assert!(!h.should_cache());
    }

    #[test]
    fn with_budget() {
        let h = Hints::animated().with_budget(16_000);
        assert_eq!(h.frame_budget_us, 16_000);
    }

    #[test]
    fn with_compute_preference() {
        let h = Hints::default().with_compute(ComputePreference::Prefer(BackendKind::Wgpu));
        assert_eq!(h.compute, ComputePreference::Prefer(BackendKind::Wgpu));
    }

    #[test]
    fn parallelism_thresholds() {
        assert_eq!(
            Hints {
                complexity: Complexity::Low,
                ..Default::default()
            }
            .parallelism_threshold(),
            10_000
        );
        assert_eq!(
            Hints {
                complexity: Complexity::Medium,
                ..Default::default()
            }
            .parallelism_threshold(),
            1_000
        );
        assert_eq!(
            Hints {
                complexity: Complexity::High,
                ..Default::default()
            }
            .parallelism_threshold(),
            100
        );
        assert_eq!(
            Hints {
                complexity: Complexity::Massive,
                ..Default::default()
            }
            .parallelism_threshold(),
            0
        );
    }

    #[test]
    fn cache_policy_auto_follows_mutability() {
        assert!(
            Hints {
                mutability: Mutability::Static,
                ..Default::default()
            }
            .should_cache()
        );
        assert!(
            !Hints {
                mutability: Mutability::Interactive,
                ..Default::default()
            }
            .should_cache()
        );
        assert!(
            !Hints {
                mutability: Mutability::Animated,
                ..Default::default()
            }
            .should_cache()
        );
    }

    #[test]
    fn cache_policy_override() {
        let h = Hints {
            cache: CachePolicy::Always,
            mutability: Mutability::Animated,
            ..Default::default()
        };
        assert!(h.should_cache()); // forced on despite animated
        let h = Hints {
            cache: CachePolicy::Never,
            mutability: Mutability::Static,
            ..Default::default()
        };
        assert!(!h.should_cache()); // forced off despite static
    }
}
