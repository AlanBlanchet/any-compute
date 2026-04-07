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

// ── Debug / Display ──────────────────────────────────────────────────────

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

/// Display uses 1 decimal place: `"1.0, 2.0, 3.0"`.
impl<const N: usize> fmt::Display for V<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, v) in self.0.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{v:.1}")?;
        }
        Ok(())
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
        self.product()
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

// ═══════════════════════════════════════════════════════════════════════════
// ── Ops trait impls ─────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

use crate::ops::{Approx, AsF64s, Export, Summary, json_array, json_f64};

impl<const N: usize> AsF64s for V<N> {
    fn as_f64s(&self) -> &[f64] {
        &self.0
    }
}

impl<const N: usize> Summary for V<N> {
    fn summary(&self) -> String {
        format!("V{N}(mag={:.4}, {})", self.magnitude(), self)
    }
}

impl<const N: usize> Export for V<N> {
    fn to_json(&self) -> String {
        json_array(self.as_f64s().iter().map(|v| json_f64(*v)))
    }
    fn to_csv(&self) -> String {
        self.as_f64s()
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(",")
    }
}

impl<const N: usize> Approx for V<N> {
    fn approx_eq(&self, other: &Self, epsilon: f64) -> bool {
        self.0
            .iter()
            .zip(other.0.iter())
            .all(|(a, b)| (a - b).abs() <= epsilon)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ── Type aliases ─────────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

/// 2D vector.
pub type V2 = V<2>;
/// 3D vector.
pub type V3 = V<3>;
/// 4D vector.
pub type V4 = V<4>;
/// 2D point — semantic alias for `V<2>`.
pub type Point = V<2>;
/// 2D size — semantic alias for `V<2>`, use `.w()` / `.h()` for size semantics.
pub type Size = V<2>;
