use crate::layout::{Matrix, V};

// ═══════════════════════════════════════════════════════════════════════════
// ── Extra V<N> utilities ────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

impl<const N: usize> V<N> {
    /// Angle in radians between two vectors.
    pub fn angle_to(self, other: Self) -> f64 {
        let d = self.dot(other);
        let m = self.magnitude() * other.magnitude();
        if m < f64::EPSILON {
            return 0.0;
        }
        (d / m).clamp(-1.0, 1.0).acos()
    }

    /// Project self onto `onto` (scalar projection × direction).
    pub fn project_onto(self, onto: Self) -> Self {
        let d = self.dot(onto);
        let len_sq = onto.len_sq();
        if len_sq < f64::EPSILON {
            return Self::ZERO;
        }
        onto * (d / len_sq)
    }

    /// Reject self from `from` (component perpendicular to `from`).
    pub fn reject_from(self, from: Self) -> Self {
        self - self.project_onto(from)
    }

    /// Whether all components are finite (not NaN or Inf).
    pub fn is_finite(self) -> bool {
        self.0.iter().all(|v| v.is_finite())
    }

    /// Component-wise absolute value.
    pub fn abs(self) -> Self {
        self.map(f64::abs)
    }

    /// Component-wise floor.
    pub fn floor(self) -> Self {
        self.map(f64::floor)
    }

    /// Component-wise ceil.
    pub fn ceil(self) -> Self {
        self.map(f64::ceil)
    }

    /// Component-wise round.
    pub fn round(self) -> Self {
        self.map(f64::round)
    }

    /// Component-wise sign (-1, 0, or 1).
    pub fn signum(self) -> Self {
        self.map(f64::signum)
    }

    /// Mix (weighted average) between self and other.
    pub fn mix(self, other: Self, t: f64) -> Self {
        self.zip(other, |a, b| a + (b - a) * t)
    }

    /// Step function: 0.0 where self < edge, else 1.0.
    pub fn step(self, edge: f64) -> Self {
        self.map(|v| if v < edge { 0.0 } else { 1.0 })
    }

    /// Smoothstep (Hermite interpolation between edges).
    pub fn smoothstep(self, edge0: f64, edge1: f64) -> Self {
        self.map(|v| {
            let t = ((v - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
            t * t * (3.0 - 2.0 * t)
        })
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ── Extra Matrix utilities ──────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

impl<const R: usize, const C: usize> Matrix<R, C> {
    /// Flatten to a 1D vector of all elements (row-major).
    pub fn flatten(&self) -> Vec<f64> {
        self.data
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect()
    }

    /// Component-wise multiply (Hadamard product).
    pub fn hadamard(&self, other: &Self) -> Self {
        let mut out = self.data;
        for r in 0..R {
            for c in 0..C {
                out[r][c] *= other.data[r][c];
            }
        }
        Self { data: out }
    }

    /// Clamp all elements to [lo, hi].
    pub fn clamp(&self, lo: f64, hi: f64) -> Self {
        self.map(|v| v.clamp(lo, hi))
    }
}

impl<const N: usize> Matrix<N, N> {
    /// Trace (sum of diagonal elements).
    pub fn trace(&self) -> f64 {
        let mut s = 0.0;
        for i in 0..N {
            s += self.data[i][i];
        }
        s
    }

    /// Diagonal as a vector.
    pub fn diagonal(&self) -> V<N> {
        let mut out = [0.0; N];
        for i in 0..N {
            out[i] = self.data[i][i];
        }
        V(out)
    }

    /// Create a diagonal matrix from a vector.
    pub fn from_diagonal(diag: V<N>) -> Self {
        let mut m = Self::default();
        for i in 0..N {
            m.data[i][i] = diag.0[i];
        }
        m
    }

    /// Whether this is an identity matrix (within epsilon).
    pub fn is_identity(&self, epsilon: f64) -> bool {
        for r in 0..N {
            for c in 0..N {
                let expected = if r == c { 1.0 } else { 0.0 };
                if (self.data[r][c] - expected).abs() > epsilon {
                    return false;
                }
            }
        }
        true
    }

    /// Whether this matrix is symmetric (within epsilon).
    pub fn is_symmetric(&self, epsilon: f64) -> bool {
        for r in 0..N {
            for c in (r + 1)..N {
                if (self.data[r][c] - self.data[c][r]).abs() > epsilon {
                    return false;
                }
            }
        }
        true
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ── Extra Buffer utilities ──────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

impl crate::buffer::Buffer {
    /// Slice a sub-range (returns a new Buffer on the same device).
    pub fn slice(&self, range: std::ops::Range<usize>) -> Self {
        Self::on(self.device(), self.data()[range].to_vec())
    }

    /// Concatenate another buffer onto this one.
    pub fn concat(&self, other: &Self) -> Self {
        let mut data = self.data().to_vec();
        data.extend_from_slice(other.data());
        Self::on(self.device(), data)
    }

    /// Reshape as a view: interpret as rows × cols for matrix operations.
    /// Returns (rows, cols) and the flat data reference.
    pub fn shape(&self, rows: usize, cols: usize) -> (usize, usize, &[f64]) {
        assert_eq!(rows * cols, self.len(), "shape mismatch");
        (rows, cols, self.data())
    }

    /// Apply a function to each element, returning a new Buffer.
    pub fn map_fn(&self, f: impl Fn(f64) -> f64) -> Self {
        Self::on(self.device(), self.data().iter().map(|&v| f(v)).collect())
    }

    /// Zip with another buffer using a function.
    pub fn zip_with(&self, other: &Self, f: impl Fn(f64, f64) -> f64) -> Self {
        let data: Vec<f64> = self
            .data()
            .iter()
            .zip(other.data().iter())
            .map(|(&a, &b)| f(a, b))
            .collect();
        Self::on(self.device(), data)
    }

    /// Clamp all values to [lo, hi].
    pub fn clamp(&self, lo: f64, hi: f64) -> Self {
        self.map_fn(|v| v.clamp(lo, hi))
    }

    /// Element-wise power.
    pub fn pow(&self, exp: f64) -> Self {
        self.map_fn(|v| v.powf(exp))
    }

    /// Reverse the buffer.
    pub fn reversed(&self) -> Self {
        let mut data = self.data().to_vec();
        data.reverse();
        Self::on(self.device(), data)
    }

    /// Unique values (sorted).
    pub fn unique(&self) -> Self {
        let mut v = self.data().to_vec();
        v.sort_by(crate::f64_cmp);
        v.dedup();
        Self::on(self.device(), v)
    }

    /// Repeat the buffer `n` times.
    pub fn repeat(&self, n: usize) -> Self {
        let data: Vec<f64> = self
            .data()
            .iter()
            .copied()
            .cycle()
            .take(self.len() * n)
            .collect();
        Self::on(self.device(), data)
    }

    /// Linspace: generate n evenly spaced values in [start, end].
    pub fn linspace(start: f64, end: f64, n: usize) -> Self {
        if n <= 1 {
            return Self::new(vec![start]);
        }
        let step = (end - start) / (n - 1) as f64;
        let data: Vec<f64> = (0..n).map(|i| start + i as f64 * step).collect();
        Self::new(data)
    }

    /// Arange: generate values from start to end with step.
    pub fn arange(start: f64, end: f64, step: f64) -> Self {
        let mut data = Vec::new();
        let mut v = start;
        while v < end {
            data.push(v);
            v += step;
        }
        Self::new(data)
    }

    /// Zeros buffer of length n.
    pub fn zeros(n: usize) -> Self {
        Self::new(vec![0.0; n])
    }

    /// Ones buffer of length n.
    pub fn ones(n: usize) -> Self {
        Self::new(vec![1.0; n])
    }

    /// Fill with a constant value.
    pub fn full(n: usize, value: f64) -> Self {
        Self::new(vec![value; n])
    }

    /// 1D convolution with a kernel (valid mode — output shorter by kernel_len - 1).
    pub fn convolve(&self, kernel: &[f64]) -> Self {
        let d = self.data();
        if d.len() < kernel.len() {
            return Self::on(self.device(), vec![]);
        }
        let out: Vec<f64> = (0..=d.len() - kernel.len())
            .map(|i| kernel.iter().enumerate().map(|(j, &k)| d[i + j] * k).sum())
            .collect();
        Self::on(self.device(), out)
    }

    /// Cross-correlation with another buffer (Pearson's r).
    pub fn correlation(&self, other: &Self) -> f64 {
        let a = self.data();
        let b = other.data();
        let n = a.len().min(b.len()) as f64;
        if n < 2.0 {
            return 0.0;
        }
        let ma = a.iter().sum::<f64>() / n;
        let mb = b.iter().sum::<f64>() / n;
        let mut cov = 0.0;
        let mut va = 0.0;
        let mut vb = 0.0;
        for i in 0..n as usize {
            let da = a[i] - ma;
            let db = b[i] - mb;
            cov += da * db;
            va += da * da;
            vb += db * db;
        }
        let denom = (va * vb).sqrt();
        if denom < f64::EPSILON {
            0.0
        } else {
            cov / denom
        }
    }

    /// Covariance with another buffer.
    pub fn covariance(&self, other: &Self) -> f64 {
        let a = self.data();
        let b = other.data();
        let n = a.len().min(b.len()) as f64;
        if n < 1.0 {
            return 0.0;
        }
        let ma = a.iter().sum::<f64>() / n;
        let mb = b.iter().sum::<f64>() / n;
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x - ma) * (y - mb))
            .sum::<f64>()
            / n
    }

    /// Gradient (central differences). Endpoints use forward/backward difference.
    pub fn gradient(&self) -> Self {
        let d = self.data();
        let n = d.len();
        if n < 2 {
            return Self::on(self.device(), vec![0.0; n]);
        }
        let mut out = Vec::with_capacity(n);
        out.push(d[1] - d[0]);
        for i in 1..n - 1 {
            out.push((d[i + 1] - d[i - 1]) * 0.5);
        }
        out.push(d[n - 1] - d[n - 2]);
        Self::on(self.device(), out)
    }

    /// Linear interpolation at fractional index positions.
    pub fn interp(&self, indices: &[f64]) -> Self {
        let d = self.data();
        let last = (d.len() - 1) as f64;
        let out: Vec<f64> = indices
            .iter()
            .map(|&idx| {
                let idx = idx.clamp(0.0, last);
                let lo = idx.floor() as usize;
                let hi = (lo + 1).min(d.len() - 1);
                let t = idx - lo as f64;
                d[lo] + (d[hi] - d[lo]) * t
            })
            .collect();
        Self::on(self.device(), out)
    }

    /// Exponential moving average with smoothing factor alpha ∈ (0, 1].
    pub fn ema(&self, alpha: f64) -> Self {
        let d = self.data();
        if d.is_empty() {
            return self.clone();
        }
        let mut out = Vec::with_capacity(d.len());
        out.push(d[0]);
        for &v in &d[1..] {
            let prev = *out.last().unwrap();
            out.push(alpha * v + (1.0 - alpha) * prev);
        }
        Self::on(self.device(), out)
    }

    /// Softmax (numerically stable).
    pub fn softmax(&self) -> Self {
        let d = self.data();
        let max_val = d.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = d.iter().map(|&v| (v - max_val).exp()).collect();
        let sum: f64 = exps.iter().sum();
        Self::on(self.device(), exps.iter().map(|v| v / sum).collect())
    }

    /// Log-softmax (numerically stable).
    pub fn log_softmax(&self) -> Self {
        let d = self.data();
        let max_val = d.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let shifted: Vec<f64> = d.iter().map(|&v| v - max_val).collect();
        let log_sum_exp = shifted.iter().map(|v| v.exp()).sum::<f64>().ln();
        Self::on(
            self.device(),
            shifted.iter().map(|v| v - log_sum_exp).collect(),
        )
    }

    /// Pairwise distance matrix (L2) — returns flat n×n buffer.
    pub fn distance_matrix(&self, dim: usize) -> Self {
        let d = self.data();
        let n = d.len() / dim;
        let mut out = vec![0.0; n * n];
        for i in 0..n {
            for j in i + 1..n {
                let mut sq = 0.0;
                for k in 0..dim {
                    let diff = d[i * dim + k] - d[j * dim + k];
                    sq += diff * diff;
                }
                let dist = sq.sqrt();
                out[i * n + j] = dist;
                out[j * n + i] = dist;
            }
        }
        Self::on(self.device(), out)
    }
}

