//! Spatial primitives — N-dimensional, generic, renderer-agnostic.
//!
//! Everything is built on [`V<N>`] (N-dimensional vector) and [`Region<N>`]
//! (N-dimensional axis-aligned region). All arithmetic, interpolation, and
//! conversions are defined **once** on the generic type — adding V<3>, V<4>,
//! or V<512> requires zero additional code.
//!
//! ## Type aliases
//!
//! | Alias   | Expands to   | Use case                 |
//! |---------|-------------|--------------------------|
//! | `V2`    | `V<2>`      | 2D position / 2D extent  |
//! | `V3`    | `V<3>`      | 3D position / volume     |
//! | `V4`    | `V<4>`      | 4D / quaternion          |
//! | `Point` | `V<2>`      | Backward-compat alias    |
//! | `Size`  | `V<2>`      | Backward-compat alias    |
//! | `Rect`  | `Region<2>` | 2D bounding box          |

use crate::Lerp;
use std::fmt;

// ═══════════════════════════════════════════════════════════════════════════
// ── V<N> — N-dimensional vector ─────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

/// N-dimensional vector — the universal spatial/numeric primitive.
///
/// Replaces separate Point / Size / Vec3 types. All arithmetic, interpolation,
/// and conversions are defined **once** and work for any dimension.
///
/// For 2D, field access (`.x`, `.y`) works via `Deref` to [`V2Fields`].
#[derive(Clone, Copy, PartialEq)]
#[repr(transparent)]
pub struct V<const N: usize>(pub [f64; N]);

impl<const N: usize> Default for V<N> {
    fn default() -> Self {
        Self([0.0; N])
    }
}

// ── Debug ────────────────────────────────────────────────────────────────

impl<const N: usize> fmt::Debug for V<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "V{N}(")?;
        for (i, v) in self.0.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{v}")?;
        }
        write!(f, ")")
    }
}

// ── Named-field Deref targets ────────────────────────────────────────────

/// Named fields for 2D access — `v.x`, `v.y` without method-call syntax.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct V2Fields {
    pub x: f64,
    pub y: f64,
}

/// Named fields for 3D access — `v.x`, `v.y`, `v.z`.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct V3Fields {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

/// Named fields for 4D access — `v.x`, `v.y`, `v.z`, `v.w`.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct V4Fields {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub w: f64,
}

// Layout safety: V<N> is repr(transparent) over [f64; N], and VNFields is
// repr(C) with N contiguous f64 — identical memory layout.
const _: () = assert!(size_of::<V<2>>() == size_of::<V2Fields>());
const _: () = assert!(size_of::<V<3>>() == size_of::<V3Fields>());
const _: () = assert!(size_of::<V<4>>() == size_of::<V4Fields>());

macro_rules! impl_deref_fields {
    ($n:literal, $fields:ty) => {
        impl std::ops::Deref for V<$n> {
            type Target = $fields;
            fn deref(&self) -> &$fields {
                // SAFETY: V<$n> is repr(transparent) over [f64; $n],
                // $fields is repr(C) with $n contiguous f64 — identical layout.
                unsafe { &*(self as *const V<$n> as *const $fields) }
            }
        }
        impl std::ops::DerefMut for V<$n> {
            fn deref_mut(&mut self) -> &mut $fields {
                unsafe { &mut *(self as *mut V<$n> as *mut $fields) }
            }
        }
    };
}

impl_deref_fields!(2, V2Fields);
impl_deref_fields!(3, V3Fields);
impl_deref_fields!(4, V4Fields);

// ── Generic ops (defined ONCE, work for ALL dimensions) ──────────────────

impl<const N: usize> V<N> {
    pub const ZERO: Self = Self([0.0; N]);

    /// Broadcast a scalar to all components.
    pub const fn splat(s: f64) -> Self {
        Self([s; N])
    }

    /// Dot product.
    pub fn dot(self, other: Self) -> f64 {
        let mut s = 0.0;
        let mut i = 0;
        while i < N {
            s += self.0[i] * other.0[i];
            i += 1;
        }
        s
    }

    /// Squared euclidean length.
    pub fn len_sq(self) -> f64 {
        self.dot(self)
    }

    /// Euclidean length.
    pub fn magnitude(self) -> f64 {
        self.len_sq().sqrt()
    }

    /// Distance to another vector.
    pub fn distance_to(self, other: Self) -> f64 {
        let mut s = 0.0;
        let mut i = 0;
        while i < N {
            let d = self.0[i] - other.0[i];
            s += d * d;
            i += 1;
        }
        s.sqrt()
    }

    /// Component-wise map.
    pub fn map(self, f: impl Fn(f64) -> f64) -> Self {
        let mut out = [0.0; N];
        let mut i = 0;
        while i < N {
            out[i] = f(self.0[i]);
            i += 1;
        }
        Self(out)
    }

    /// Component-wise zip with another vector.
    pub fn zip(self, other: Self, f: impl Fn(f64, f64) -> f64) -> Self {
        let mut out = [0.0; N];
        let mut i = 0;
        while i < N {
            out[i] = f(self.0[i], other.0[i]);
            i += 1;
        }
        Self(out)
    }

    /// Sum all components.
    pub fn component_sum(self) -> f64 {
        let mut s = 0.0;
        let mut i = 0;
        while i < N {
            s += self.0[i];
            i += 1;
        }
        s
    }

    /// Product of all components (area for 2D, volume for 3D, etc.).
    pub fn product(self) -> f64 {
        let mut p = 1.0;
        let mut i = 0;
        while i < N {
            p *= self.0[i];
            i += 1;
        }
        p
    }

    /// Component-wise min.
    pub fn comp_min(self, other: Self) -> Self {
        self.zip(other, f64::min)
    }

    /// Component-wise max.
    pub fn comp_max(self, other: Self) -> Self {
        self.zip(other, f64::max)
    }

    /// Component-wise clamp.
    pub fn clamp(self, lo: Self, hi: Self) -> Self {
        let mut out = [0.0; N];
        let mut i = 0;
        while i < N {
            out[i] = self.0[i].clamp(lo.0[i], hi.0[i]);
            i += 1;
        }
        Self(out)
    }
}

// ── V<2> — named constructors & size aliases ─────────────────────────────

impl V<2> {
    /// 2D constructor (for both position and size).
    pub const fn new(a: f64, b: f64) -> Self {
        Self([a, b])
    }

    /// Width alias — reads as "width" when used as size.
    pub fn w(&self) -> f64 {
        self.0[0]
    }

    /// Height alias — reads as "height" when used as size.
    pub fn h(&self) -> f64 {
        self.0[1]
    }

    /// Area (product of 2 components).
    pub fn area(self) -> f64 {
        self.0[0] * self.0[1]
    }
}

// ── V<3> — 3D operations ─────────────────────────────────────────────────

impl V<3> {
    /// 3D constructor.
    pub const fn new3(x: f64, y: f64, z: f64) -> Self {
        Self([x, y, z])
    }

    /// Cross product: a × b — perpendicular to both vectors.
    pub fn cross(self, b: Self) -> Self {
        Self([
            self.0[1] * b.0[2] - self.0[2] * b.0[1],
            self.0[2] * b.0[0] - self.0[0] * b.0[2],
            self.0[0] * b.0[1] - self.0[1] * b.0[0],
        ])
    }

    /// Reflect across a surface normal (both should be unit vectors).
    pub fn reflect(self, normal: Self) -> Self {
        self - normal * (2.0 * self.dot(normal))
    }

    /// Face normal from three triangle vertices (CCW winding).
    pub fn face_normal(a: Self, b: Self, c: Self) -> Self {
        (b - a).cross(c - a).normalized()
    }

    /// Depth alias — reads as "depth" when used as size.
    pub fn d(&self) -> f64 {
        self.0[2]
    }

    /// Volume (product of 3 components).
    pub fn volume(self) -> f64 {
        self.product()
    }
}

// ── V<N> generic normalize ───────────────────────────────────────────────

impl<const N: usize> V<N> {
    /// Unit vector (length = 1). Returns ZERO if length is ~0.
    pub fn normalized(self) -> Self {
        let len = self.magnitude();
        if len < 1e-15 { Self::ZERO } else { self / len }
    }
}

// ── Generic From/Into (defined ONCE for all N) ──────────────────────────

impl<const N: usize> From<[f64; N]> for V<N> {
    fn from(a: [f64; N]) -> Self {
        Self(a)
    }
}

impl<const N: usize> From<V<N>> for [f64; N] {
    fn from(v: V<N>) -> Self {
        v.0
    }
}

impl<const N: usize> From<f64> for V<N> {
    /// Broadcast: every component = the scalar.
    fn from(s: f64) -> Self {
        Self([s; N])
    }
}

// ── Dimension-specific tuple conversions ─────────────────────────────────

impl From<(f64, f64)> for V<2> {
    fn from((a, b): (f64, f64)) -> Self {
        Self([a, b])
    }
}

impl From<(i32, i32)> for V<2> {
    fn from((a, b): (i32, i32)) -> Self {
        Self([a as f64, b as f64])
    }
}

impl From<(u32, u32)> for V<2> {
    fn from((a, b): (u32, u32)) -> Self {
        Self([a as f64, b as f64])
    }
}

impl From<V<2>> for (f64, f64) {
    fn from(v: V<2>) -> Self {
        (v.0[0], v.0[1])
    }
}

impl From<(f64, f64, f64)> for V<3> {
    fn from((x, y, z): (f64, f64, f64)) -> Self {
        Self([x, y, z])
    }
}

impl From<(f64, f64, f64, f64)> for V<4> {
    fn from((x, y, z, w): (f64, f64, f64, f64)) -> Self {
        Self([x, y, z, w])
    }
}

// ── Generic arithmetic (all N) ───────────────────────────────────────────

impl<const N: usize> std::ops::Add for V<N> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        let mut out = [0.0; N];
        let mut i = 0;
        while i < N {
            out[i] = self.0[i] + rhs.0[i];
            i += 1;
        }
        Self(out)
    }
}

impl<const N: usize> std::ops::Sub for V<N> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        let mut out = [0.0; N];
        let mut i = 0;
        while i < N {
            out[i] = self.0[i] - rhs.0[i];
            i += 1;
        }
        Self(out)
    }
}

impl<const N: usize> std::ops::Neg for V<N> {
    type Output = Self;
    fn neg(self) -> Self {
        let mut out = [0.0; N];
        let mut i = 0;
        while i < N {
            out[i] = -self.0[i];
            i += 1;
        }
        Self(out)
    }
}

impl<const N: usize> std::ops::Mul<f64> for V<N> {
    type Output = Self;
    fn mul(self, s: f64) -> Self {
        let mut out = [0.0; N];
        let mut i = 0;
        while i < N {
            out[i] = self.0[i] * s;
            i += 1;
        }
        Self(out)
    }
}

impl<const N: usize> std::ops::Mul<V<N>> for f64 {
    type Output = V<N>;
    fn mul(self, v: V<N>) -> V<N> {
        v * self
    }
}

impl<const N: usize> std::ops::Div<f64> for V<N> {
    type Output = Self;
    fn div(self, s: f64) -> Self {
        self * (1.0 / s)
    }
}

// ── Index ────────────────────────────────────────────────────────────────

impl<const N: usize> std::ops::Index<usize> for V<N> {
    type Output = f64;
    fn index(&self, i: usize) -> &f64 {
        &self.0[i]
    }
}

impl<const N: usize> std::ops::IndexMut<usize> for V<N> {
    fn index_mut(&mut self, i: usize) -> &mut f64 {
        &mut self.0[i]
    }
}

// ── Generic Lerp ─────────────────────────────────────────────────────────

impl<const N: usize> Lerp for V<N> {
    fn lerp(self, other: Self, t: f64) -> Self {
        let mut out = [0.0; N];
        let mut i = 0;
        while i < N {
            out[i] = self.0[i] + (other.0[i] - self.0[i]) * t;
            i += 1;
        }
        Self(out)
    }
}

// ── Type aliases ─────────────────────────────────────────────────────────

/// 2D vector.
pub type V2 = V<2>;
/// 3D vector.
pub type V3 = V<3>;
/// 4D vector.
pub type V4 = V<4>;
/// Backward-compatible alias — same as `V<2>`.
pub type Point = V<2>;
/// Backward-compatible alias — same as `V<2>`, use `.w()` / `.h()` for size semantics.
pub type Size = V<2>;

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

// ── Region<2> — backward-compatible Rect API ─────────────────────────────

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

/// Backward-compatible alias — `Rect` = `Region<2>`.
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
}

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

// ═══════════════════════════════════════════════════════════════════════════
// ── Tests ────────────────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Lerp;

    #[test]
    fn v2_field_access() {
        let v = V::new(3.0, 4.0);
        assert_eq!(v.x, 3.0);
        assert_eq!(v.y, 4.0);
        assert_eq!(v.w(), 3.0);
        assert_eq!(v.h(), 4.0);
        assert_eq!(v[0], 3.0);
        assert_eq!(v[1], 4.0);
    }

    #[test]
    fn v2_mutable_field_access() {
        let mut v = V::new(1.0, 2.0);
        v.x = 10.0;
        v.y = 20.0;
        assert_eq!(v, V::new(10.0, 20.0));
    }

    #[test]
    fn v2_lerp() {
        let a = V::new(0.0, 0.0);
        let b = V::new(100.0, 200.0);
        let mid = a.lerp(b, 0.5);
        assert!((mid.x - 50.0).abs() < 1e-10);
        assert!((mid.y - 100.0).abs() < 1e-10);
    }

    #[test]
    fn v2_distance() {
        let a = V::new(0.0, 0.0);
        let b = V::new(3.0, 4.0);
        assert!((a.distance_to(b) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn v2_area() {
        assert!((V::new(10.0, 20.0).area() - 200.0).abs() < 1e-10);
    }

    #[test]
    fn v2_arithmetic() {
        let a = V::new(1.0, 2.0);
        let b = V::new(3.0, 4.0);
        assert_eq!(a + b, V::new(4.0, 6.0));
        assert_eq!(b - a, V::new(2.0, 2.0));
        assert_eq!(-a, V::new(-1.0, -2.0));
        assert_eq!(a * 3.0, V::new(3.0, 6.0));
        assert_eq!(2.0 * a, V::new(2.0, 4.0));
        assert_eq!(a / 2.0, V::new(0.5, 1.0));
    }

    #[test]
    fn v3_operations() {
        let a = V([1.0, 2.0, 3.0]);
        let b = V([4.0, 5.0, 6.0]);
        assert_eq!(a + b, V([5.0, 7.0, 9.0]));
        assert_eq!(a.x, 1.0);
        assert_eq!(a.z, 3.0);
        assert!((a.dot(b) - 32.0).abs() < 1e-10);
    }

    #[test]
    fn v_generic_n() {
        let v: V<8> = V::splat(2.0);
        assert_eq!(v.component_sum(), 16.0);
        assert_eq!(v.product(), 256.0);
        let big: V<128> = V::ZERO;
        assert_eq!(big.component_sum(), 0.0);
    }

    #[test]
    fn v2_from_conversions() {
        assert_eq!(Point::from((3.0, 4.0)), Point::new(3.0, 4.0));
        assert_eq!(Point::from((3i32, 4i32)), Point::new(3.0, 4.0));
        assert_eq!(Point::from([3.0, 4.0]), Point::new(3.0, 4.0));
        assert_eq!(Point::from(5.0), Point::new(5.0, 5.0));
        let (x, y): (f64, f64) = Point::new(1.0, 2.0).into();
        assert_eq!((x, y), (1.0, 2.0));
        let arr: [f64; 2] = Point::new(1.0, 2.0).into();
        assert_eq!(arr, [1.0, 2.0]);
        assert_eq!(Size::from((10.0, 20.0)), Size::new(10.0, 20.0));
        assert_eq!(Size::from((100u32, 200u32)), Size::new(100.0, 200.0));
    }

    #[test]
    fn point_size_same_type() {
        let p = Point::new(3.0, 4.0);
        let s: Size = p;
        assert_eq!(s, Size::new(3.0, 4.0));
    }

    #[test]
    fn region_contains() {
        let r = Rect::new(10.0, 10.0, 100.0, 50.0);
        assert!(r.contains(V::new(50.0, 30.0)));
        assert!(!r.contains(V::new(5.0, 30.0)));
        assert!(!r.contains(V::new(50.0, 70.0)));
    }

    #[test]
    fn region_accessors() {
        let r = Rect::new(10.0, 20.0, 100.0, 50.0);
        assert_eq!(r.x(), 10.0);
        assert_eq!(r.y(), 20.0);
        assert_eq!(r.w(), 100.0);
        assert_eq!(r.h(), 50.0);
        assert_eq!(r.right(), 110.0);
        assert_eq!(r.bottom(), 70.0);
        let c = r.center();
        assert!((c.x - 60.0).abs() < 1e-10);
        assert!((c.y - 45.0).abs() < 1e-10);
    }

    #[test]
    fn region_lerp() {
        let a = Rect::new(0.0, 0.0, 100.0, 50.0);
        let b = Rect::new(100.0, 100.0, 200.0, 100.0);
        let mid = a.lerp(b, 0.5);
        assert!((mid.x() - 50.0).abs() < 1e-10);
        assert!((mid.w() - 150.0).abs() < 1e-10);
    }

    #[test]
    fn region_from_parts() {
        let r = Rect::from_parts(V::new(5.0, 10.0), V::new(20.0, 30.0));
        assert_eq!(r.x(), 5.0);
        assert_eq!(r.h(), 30.0);
    }

    #[test]
    fn region_from_conversions() {
        assert_eq!(
            Rect::from((1.0, 2.0, 3.0, 4.0)),
            Rect::new(1.0, 2.0, 3.0, 4.0)
        );
        assert_eq!(
            Rect::from([1.0, 2.0, 3.0, 4.0]),
            Rect::new(1.0, 2.0, 3.0, 4.0)
        );
        let r: Rect = (V::new(1.0, 2.0), V::new(3.0, 4.0)).into();
        assert_eq!(r, Rect::new(1.0, 2.0, 3.0, 4.0));
        let r: Rect = V::new(800.0, 600.0).into();
        assert_eq!(r, Rect::new(0.0, 0.0, 800.0, 600.0));
    }

    #[test]
    fn region_3d() {
        let r = Region::<3>::from_parts(V([0.0, 0.0, 0.0]), V([10.0, 20.0, 30.0]));
        assert!(r.contains(V([5.0, 10.0, 15.0])));
        assert!(!r.contains(V([15.0, 10.0, 15.0])));
        assert_eq!(r.center(), V([5.0, 10.0, 15.0]));
    }

    #[test]
    fn constraints_clamp() {
        let c = Constraints {
            min: V::new(10.0, 10.0),
            max: V::new(200.0, 200.0),
        };
        assert_eq!(c.clamp(V::new(5.0, 300.0)), V::new(10.0, 200.0));
    }

    #[test]
    fn scroll_visible_range() {
        let s = ScrollState {
            offset: V::new(0.0, 280.0),
        };
        let range = s.visible_range(28.0, 280.0, 1000);
        assert_eq!(range.start, 10);
        assert!(range.end <= 21);
    }

    #[test]
    fn scroll_visible_range_edge_cases() {
        let s = ScrollState {
            offset: V::ZERO,
        };
        assert_eq!(s.visible_range(0.0, 100.0, 100), 0..0);
        assert_eq!(s.visible_range(10.0, 100.0, 0), 0..0);
    }
}
