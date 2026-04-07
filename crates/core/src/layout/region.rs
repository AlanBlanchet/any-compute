use super::{V, V3};
use crate::Lerp;

// ═══════════════════════════════════════════════════════════════════════════
// ── Region<N> — N-dimensional axis-aligned bounding box ─────────────────
// ═══════════════════════════════════════════════════════════════════════════

/// N-dimensional axis-aligned region — origin + size.
///
/// `Region<2>` = 2D rect, `Region<3>` = 3D AABB, and so on.
/// All spatial queries (contains, center, intersect) are dimension-generic.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Region<const N: usize> {
    pub origin: V<N>,
    pub size: V<N>,
}

impl<const N: usize> Region<N> {
    pub const ZERO: Self = Self {
        origin: V::ZERO,
        size: V::ZERO,
    };

    /// Construct from origin + size vectors.
    pub fn from_parts(origin: V<N>, size: V<N>) -> Self {
        Self { origin, size }
    }

    /// Center point.
    pub fn center(&self) -> V<N> {
        self.origin + self.size * 0.5
    }

    /// Does this region contain a point?
    pub fn contains(&self, p: V<N>) -> bool {
        let mut i = 0;
        while i < N {
            if p.0[i] < self.origin.0[i] || p.0[i] > self.origin.0[i] + self.size.0[i] {
                return false;
            }
            i += 1;
        }
        true
    }

    /// Endpoint (origin + size) along each axis.
    pub fn end(&self) -> V<N> {
        self.origin + self.size
    }
}

// ── Generic Lerp for Region ──────────────────────────────────────────────

impl<const N: usize> Lerp for Region<N> {
    fn lerp(self, other: Self, t: f64) -> Self {
        Self {
            origin: self.origin.lerp(other.origin, t),
            size: self.size.lerp(other.size, t),
        }
    }
}

// ── Region<2> — 2D Rect API ─────────────────────────────────────────────

impl Region<2> {
    /// 2D convenience constructor (x, y, w, h).
    pub fn new(x: f64, y: f64, w: f64, h: f64) -> Self {
        Self {
            origin: V([x, y]),
            size: V([w, h]),
        }
    }

    pub fn x(&self) -> f64 {
        self.origin.x
    }
    pub fn y(&self) -> f64 {
        self.origin.y
    }
    pub fn w(&self) -> f64 {
        self.size.w()
    }
    pub fn h(&self) -> f64 {
        self.size.h()
    }

    pub fn right(&self) -> f64 {
        self.origin.x + self.size.w()
    }
    pub fn bottom(&self) -> f64 {
        self.origin.y + self.size.h()
    }
}

// ── From impls for Region<2> ─────────────────────────────────────────────

impl From<(f64, f64, f64, f64)> for Region<2> {
    fn from((x, y, w, h): (f64, f64, f64, f64)) -> Self {
        Self::new(x, y, w, h)
    }
}

impl From<[f64; 4]> for Region<2> {
    fn from([x, y, w, h]: [f64; 4]) -> Self {
        Self::new(x, y, w, h)
    }
}

impl From<(V<2>, V<2>)> for Region<2> {
    fn from((origin, size): (V<2>, V<2>)) -> Self {
        Self { origin, size }
    }
}

impl From<V<2>> for Region<2> {
    /// Size-only: origin = zero.
    fn from(size: V<2>) -> Self {
        Self {
            origin: V::ZERO,
            size,
        }
    }
}

/// 2D rectangle — semantic alias for `Region<2>`.
pub type Rect = Region<2>;

/// 3D axis-aligned bounding box.
pub type AABB = Region<3>;

impl Region<3> {
    /// 3D convenience constructor.
    pub fn new3(x: f64, y: f64, z: f64, w: f64, h: f64, d: f64) -> Self {
        Self {
            origin: V([x, y, z]),
            size: V([w, h, d]),
        }
    }

    pub fn depth(&self) -> f64 {
        self.size.0[2]
    }

    pub fn volume(&self) -> f64 {
        self.size.product()
    }

    /// Does this AABB intersect another?
    pub fn intersects(&self, other: &Self) -> bool {
        let a_end = self.end();
        let b_end = other.end();
        (0..3).all(|i| self.origin.0[i] <= b_end.0[i] && a_end.0[i] >= other.origin.0[i])
    }

    /// 8 corner vertices of this AABB.
    pub fn corners(&self) -> [V3; 8] {
        let o = self.origin;
        let s = self.size;
        [
            o,
            o + V([s.0[0], 0.0, 0.0]),
            o + V([0.0, s.0[1], 0.0]),
            o + V([0.0, 0.0, s.0[2]]),
            o + V([s.0[0], s.0[1], 0.0]),
            o + V([s.0[0], 0.0, s.0[2]]),
            o + V([0.0, s.0[1], s.0[2]]),
            o + s,
        ]
    }
}
