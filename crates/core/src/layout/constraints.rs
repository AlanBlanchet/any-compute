use super::V;

// ═══════════════════════════════════════════════════════════════════════════
// ── Constraints ──────────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

/// Size constraints flowing down during layout.
#[derive(Debug, Clone, Copy)]
pub struct Constraints {
    pub min: V<2>,
    pub max: V<2>,
}

impl Constraints {
    pub fn tight(size: V<2>) -> Self {
        Self {
            min: size,
            max: size,
        }
    }

    pub fn unbounded() -> Self {
        Self {
            min: V::ZERO,
            max: V::splat(f64::INFINITY),
        }
    }

    pub fn clamp(&self, size: V<2>) -> V<2> {
        V([
            size.w().clamp(self.min.w(), self.max.w()),
            size.h().clamp(self.min.h(), self.max.h()),
        ])
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ── ScrollState ──────────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

/// Scroll state for virtualized containers.
#[derive(Debug, Clone, Copy, Default)]
pub struct ScrollState {
    pub offset: V<2>,
}

impl ScrollState {
    /// Which range of items (by index) is visible given a fixed item height?
    pub fn visible_range(
        &self,
        item_height: f64,
        viewport_height: f64,
        total_items: usize,
    ) -> std::ops::Range<usize> {
        if item_height <= 0.0 || total_items == 0 {
            return 0..0;
        }
        let first = (self.offset.y / item_height).floor().max(0.0) as usize;
        let visible_count = (viewport_height / item_height).ceil() as usize + 1;
        let last = (first + visible_count).min(total_items);
        first..last
    }
}

