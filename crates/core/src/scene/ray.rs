use crate::layout::{AABB, V, V3};

// ═══════════════════════════════════════════════════════════════════════════
// ── Ray ──────────────────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

/// Ray for intersection testing.
#[derive(Debug, Clone, Copy)]
pub struct Ray {
    pub origin: V3,
    pub direction: V3,
}

impl Ray {
    pub fn new(origin: V3, direction: V3) -> Self {
        Self {
            origin,
            direction: direction.normalized(),
        }
    }

    /// Point along the ray at parameter t.
    pub fn at(&self, t: f64) -> V3 {
        self.origin + self.direction * t
    }

    /// Ray–AABB intersection (slab method). Returns Some(t_near) or None.
    pub fn intersect_aabb(&self, aabb: &AABB) -> Option<f64> {
        let inv = V([
            1.0 / self.direction.0[0],
            1.0 / self.direction.0[1],
            1.0 / self.direction.0[2],
        ]);
        let t0 = (aabb.origin - self.origin).zip(inv, |a, b| a * b);
        let t1 = (aabb.end() - self.origin).zip(inv, |a, b| a * b);
        let tmin_v = t0.comp_min(t1);
        let tmax_v = t0.comp_max(t1);
        let tmin = tmin_v.0[0].max(tmin_v.0[1]).max(tmin_v.0[2]);
        let tmax = tmax_v.0[0].min(tmax_v.0[1]).min(tmax_v.0[2]);
        if tmax >= tmin.max(0.0) {
            Some(tmin.max(0.0))
        } else {
            None
        }
    }

    /// Ray–triangle intersection (Möller–Trumbore). Returns Some(t) or None.
    pub fn intersect_tri(&self, a: V3, b: V3, c: V3) -> Option<f64> {
        let edge1 = b - a;
        let edge2 = c - a;
        let h = self.direction.cross(edge2);
        let det = edge1.dot(h);
        if det.abs() < 1e-12 {
            return None;
        }
        let inv_det = 1.0 / det;
        let s = self.origin - a;
        let u = s.dot(h) * inv_det;
        if !(0.0..=1.0).contains(&u) {
            return None;
        }
        let q = s.cross(edge1);
        let v = self.direction.dot(q) * inv_det;
        if v < 0.0 || u + v > 1.0 {
            return None;
        }
        let t = edge2.dot(q) * inv_det;
        if t > 1e-8 { Some(t) } else { None }
    }
}

