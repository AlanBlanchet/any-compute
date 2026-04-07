use super::V;
use std::fmt;

// ═══════════════════════════════════════════════════════════════════════════
// ── Matrix<R,C> — compile-time-dimensioned dense matrix ─────────────────
// ═══════════════════════════════════════════════════════════════════════════

/// Dense R×C matrix, row-major — the 2D companion to `V<N>`.
///
/// Supports matrix–vector multiplication (`Matrix × V`), matrix–matrix
/// multiplication, transpose, and identity — all generic over dimensions.
/// This bridges spatial transforms (4×4) with linear algebra / ML ops.
#[derive(Clone, Copy, PartialEq)]
pub struct Matrix<const R: usize, const C: usize> {
    /// Row-major storage: `data[r][c]`.
    pub data: [[f64; C]; R],
}

impl<const R: usize, const C: usize> Default for Matrix<R, C> {
    fn default() -> Self {
        Self {
            data: [[0.0; C]; R],
        }
    }
}

impl<const R: usize, const C: usize> fmt::Debug for Matrix<R, C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Mat{R}x{C}(")?;
        for (i, row) in self.data.iter().enumerate() {
            if i > 0 {
                write!(f, "; ")?;
            }
            for (j, v) in row.iter().enumerate() {
                if j > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{v}")?;
            }
        }
        write!(f, ")")
    }
}

impl<const N: usize> Matrix<N, N> {
    /// NxN identity matrix.
    pub fn identity() -> Self {
        let mut m = Self::default();
        let mut i = 0;
        while i < N {
            m.data[i][i] = 1.0;
            i += 1;
        }
        m
    }
}

impl<const R: usize, const C: usize> Matrix<R, C> {
    pub const ZERO: Self = Self {
        data: [[0.0; C]; R],
    };

    /// Construct from row-major 2D array.
    pub const fn new(data: [[f64; C]; R]) -> Self {
        Self { data }
    }

    /// Transpose: R×C → C×R.
    pub fn transpose(&self) -> Matrix<C, R> {
        let mut out = [[0.0; R]; C];
        let mut r = 0;
        while r < R {
            let mut c = 0;
            while c < C {
                out[c][r] = self.data[r][c];
                c += 1;
            }
            r += 1;
        }
        Matrix { data: out }
    }

    /// Extract a row as a vector.
    pub fn row(&self, r: usize) -> V<C> {
        V(self.data[r])
    }

    /// Extract a column as a vector.
    pub fn col(&self, c: usize) -> V<R> {
        let mut out = [0.0; R];
        let mut r = 0;
        while r < R {
            out[r] = self.data[r][c];
            r += 1;
        }
        V(out)
    }

    /// Component-wise map.
    pub fn map(&self, f: impl Fn(f64) -> f64) -> Self {
        let mut out = self.data;
        for row in &mut out {
            for v in row.iter_mut() {
                *v = f(*v);
            }
        }
        Self { data: out }
    }

    /// Scalar multiply.
    pub fn scale(&self, s: f64) -> Self {
        self.map(|v| v * s)
    }
}

/// Matrix × Vector: R×C matrix times C-vector → R-vector.
impl<const R: usize, const C: usize> std::ops::Mul<V<C>> for Matrix<R, C> {
    type Output = V<R>;
    fn mul(self, v: V<C>) -> V<R> {
        let mut out = [0.0; R];
        let mut r = 0;
        while r < R {
            let mut sum = 0.0;
            let mut c = 0;
            while c < C {
                sum += self.data[r][c] * v.0[c];
                c += 1;
            }
            out[r] = sum;
            r += 1;
        }
        V(out)
    }
}

/// Matrix + Matrix.
impl<const R: usize, const C: usize> std::ops::Add for Matrix<R, C> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        let mut out = self.data;
        let mut r = 0;
        while r < R {
            let mut c = 0;
            while c < C {
                out[r][c] += rhs.data[r][c];
                c += 1;
            }
            r += 1;
        }
        Self { data: out }
    }
}

/// Matrix - Matrix.
impl<const R: usize, const C: usize> std::ops::Sub for Matrix<R, C> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        let mut out = self.data;
        let mut r = 0;
        while r < R {
            let mut c = 0;
            while c < C {
                out[r][c] -= rhs.data[r][c];
                c += 1;
            }
            r += 1;
        }
        Self { data: out }
    }
}

/// Matrix × Scalar.
impl<const R: usize, const C: usize> std::ops::Mul<f64> for Matrix<R, C> {
    type Output = Self;
    fn mul(self, s: f64) -> Self {
        self.scale(s)
    }
}

/// Type aliases for common matrix sizes.
pub type Mat2 = Matrix<2, 2>;
pub type Mat3 = Matrix<3, 3>;
pub type Mat4 = Matrix<4, 4>;

/// 4×4 transform helpers — the standard 3D transform matrix.
impl Mat4 {
    /// Translation matrix.
    pub fn translation(v: V<3>) -> Self {
        let mut m = Self::identity();
        m.data[0][3] = v.0[0];
        m.data[1][3] = v.0[1];
        m.data[2][3] = v.0[2];
        m
    }

    /// Uniform scale matrix.
    pub fn scaling(s: f64) -> Self {
        let mut m = Self::identity();
        m.data[0][0] = s;
        m.data[1][1] = s;
        m.data[2][2] = s;
        m
    }

    /// Non-uniform scale matrix.
    pub fn scaling3(v: V<3>) -> Self {
        let mut m = Self::identity();
        m.data[0][0] = v.0[0];
        m.data[1][1] = v.0[1];
        m.data[2][2] = v.0[2];
        m
    }

    /// Apply this 4×4 transform to a 3D point (w=1, perspective divide).
    pub fn transform_point(&self, p: V<3>) -> V<3> {
        let v4 = V([p.0[0], p.0[1], p.0[2], 1.0]);
        let out = *self * v4;
        let w = if out.0[3].abs() > 1e-15 {
            out.0[3]
        } else {
            1.0
        };
        V([out.0[0] / w, out.0[1] / w, out.0[2] / w])
    }

    /// Apply this 4×4 transform to a 3D direction (w=0, no translate).
    pub fn transform_dir(&self, d: V<3>) -> V<3> {
        let v4 = V([d.0[0], d.0[1], d.0[2], 0.0]);
        let out = *self * v4;
        V([out.0[0], out.0[1], out.0[2]])
    }
}

/// Convert V<N> → Matrix<N,1> (column vector) and Matrix<1,N> (row vector).
impl<const N: usize> From<V<N>> for Matrix<N, 1> {
    fn from(v: V<N>) -> Self {
        let mut data = [[0.0; 1]; N];
        let mut i = 0;
        while i < N {
            data[i][0] = v.0[i];
            i += 1;
        }
        Self { data }
    }
}

impl<const N: usize> From<Matrix<N, 1>> for V<N> {
    fn from(m: Matrix<N, 1>) -> Self {
        let mut out = [0.0; N];
        let mut i = 0;
        while i < N {
            out[i] = m.data[i][0];
            i += 1;
        }
        Self(out)
    }
}

// ═══════════════════════════════════════════════════════════════════════════// ── Ops trait impls ─────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

use crate::ops::{Approx, AsF64s, Export, Norm, Summary, json_array, json_f64};

impl<const R: usize, const C: usize> AsF64s for Matrix<R, C> {
    fn as_f64s(&self) -> &[f64] {
        // SAFETY: Matrix is repr(C) style [[f64; C]; R] — contiguous f64s.
        unsafe { std::slice::from_raw_parts(self.data.as_ptr() as *const f64, R * C) }
    }
}

impl<const R: usize, const C: usize> Summary for Matrix<R, C> {
    fn summary(&self) -> String {
        let n: f64 = self.frobenius();
        format!("Mat{R}x{C}(frobenius={n:.4})")
    }
}

impl<const R: usize, const C: usize> Export for Matrix<R, C> {
    fn to_json(&self) -> String {
        json_array(
            self.data
                .iter()
                .map(|row| json_array(row.iter().map(|v| json_f64(*v)))),
        )
    }
    fn to_csv(&self) -> String {
        self.data
            .iter()
            .map(|row| {
                row.iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

impl<const R: usize, const C: usize> Approx for Matrix<R, C> {
    fn approx_eq(&self, other: &Self, epsilon: f64) -> bool {
        self.data
            .iter()
            .flatten()
            .zip(other.data.iter().flatten())
            .all(|(a, b)| (a - b).abs() <= epsilon)
    }
}

// ═══════════════════════════════════════════════════════════════════════════// ── Tests ────────────────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════
