use crate::render::Color;

// ═══════════════════════════════════════════════════════════════════════════
// ── Shared helpers ──────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

/// Find min and max of a non-empty f64 slice.
pub(crate) fn min_max(data: &[f64]) -> (f64, f64) {
    let (mut lo, mut hi) = (data[0], data[0]);
    for &v in &data[1..] {
        if v < lo {
            lo = v;
        }
        if v > hi {
            hi = v;
        }
    }
    (lo, hi)
}

/// Sort f64 slice (handles NaN by treating as equal).
pub(crate) fn sorted_copy(data: &[f64]) -> Vec<f64> {
    let mut v = data.to_vec();
    v.sort_by(crate::f64_cmp);
    v
}

/// Convert RGBA bytes to Color pixels.
pub(crate) fn colors_from_rgba(rgba: &[u8]) -> Vec<Color> {
    rgba.chunks_exact(4)
        .map(|c| Color::rgba(c[0], c[1], c[2], c[3]))
        .collect()
}

/// Format f64 for JSON output (always includes decimal point).
pub(crate) fn json_f64(v: f64) -> String {
    if v.fract() == 0.0 && v.is_finite() {
        format!("{v:.1}")
    } else {
        v.to_string()
    }
}

/// Wrap a list of JSON values in brackets: `[v1,v2,v3]`.
pub(crate) fn json_array(items: impl Iterator<Item = String>) -> String {
    let joined: Vec<String> = items.collect();
    format!("[{}]", joined.join(","))
}

// ═══════════════════════════════════════════════════════════════════════════
// ── AsF64s — numeric data foundation ────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

/// Foundation trait: any type that can yield a sequence of f64 values.
///
/// Implementing this **once** gives your type Stats, Norm, Normalize,
/// Histogram, SortOps, SearchOps, and ReduceOps — all for free via
/// blanket implementations.
pub trait AsF64s {
    /// Borrow the underlying f64 data as a slice.
    fn as_f64s(&self) -> &[f64];
}

// ── Built-in impls ──────────────────────────────────────────────────────

impl AsF64s for [f64] {
    fn as_f64s(&self) -> &[f64] {
        self
    }
}

impl AsF64s for Vec<f64> {
    fn as_f64s(&self) -> &[f64] {
        self
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ── Stats — statistical analysis ────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

/// Statistical analysis result.
#[derive(Debug, Clone, Copy)]
pub struct StatsResult {
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub variance: f64,
    pub std_dev: f64,
    pub median: f64,
    pub count: usize,
}

impl std::fmt::Display for StatsResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "n={} min={:.4} max={:.4} mean={:.4} std={:.4} med={:.4}",
            self.count, self.min, self.max, self.mean, self.std_dev, self.median
        )
    }
}

/// Auto-derived statistical methods for any numeric data.
pub trait Stats: AsF64s {
    /// Full statistical summary (min, max, mean, variance, std_dev, median).
    fn stats(&self) -> StatsResult {
        let data = self.as_f64s();
        let n = data.len();
        if n == 0 {
            return StatsResult {
                min: f64::NAN,
                max: f64::NAN,
                mean: f64::NAN,
                variance: f64::NAN,
                std_dev: f64::NAN,
                median: f64::NAN,
                count: 0,
            };
        }
        let (lo, hi) = min_max(data);
        let sum: f64 = data.iter().sum();
        let mean = sum / n as f64;
        let var = data.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
        let sorted = sorted_copy(data);
        let median = if n % 2 == 0 {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        } else {
            sorted[n / 2]
        };
        StatsResult {
            min: lo,
            max: hi,
            mean,
            variance: var,
            std_dev: var.sqrt(),
            median,
            count: n,
        }
    }

    /// Value range (max - min).
    fn range(&self) -> f64 {
        let data = self.as_f64s();
        if data.is_empty() {
            return 0.0;
        }
        let (lo, hi) = min_max(data);
        hi - lo
    }

    /// Percentile (0..=100). Uses linear interpolation.
    fn percentile(&self, p: f64) -> f64 {
        let data = self.as_f64s();
        if data.is_empty() {
            return f64::NAN;
        }
        let sorted = sorted_copy(data);
        let idx = (p.clamp(0.0, 100.0) / 100.0) * (sorted.len() - 1) as f64;
        let lo = idx.floor() as usize;
        let hi = idx.ceil() as usize;
        if lo == hi {
            sorted[lo]
        } else {
            sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo as f64)
        }
    }
}

// Blanket impl: anything AsF64s gets Stats for free.
impl<T: ?Sized + AsF64s> Stats for T {}

// ═══════════════════════════════════════════════════════════════════════════
// ── Norm — vector/matrix norms ──────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

/// Vector/matrix norms — L1, L2, L∞, Frobenius.
pub trait Norm: AsF64s {
    /// L1 norm (sum of absolute values).
    fn l1_norm(&self) -> f64 {
        self.as_f64s().iter().map(|v| v.abs()).sum()
    }

    /// L2 norm (Euclidean length).
    fn l2_norm(&self) -> f64 {
        self.as_f64s().iter().map(|v| v * v).sum::<f64>().sqrt()
    }

    /// L∞ norm (max absolute value).
    fn linf_norm(&self) -> f64 {
        self.as_f64s()
            .iter()
            .map(|v| v.abs())
            .fold(0.0f64, f64::max)
    }

    /// Frobenius norm (same as L2 for vectors, standard for matrices).
    fn frobenius(&self) -> f64 {
        self.l2_norm()
    }
}

impl<T: ?Sized + AsF64s> Norm for T {}

// ═══════════════════════════════════════════════════════════════════════════
// ── NormalizeOps — rescale to standard ranges ───────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

/// Normalization methods — rescale data to [0,1] or standard score (z-score).
pub trait NormalizeOps: AsF64s {
    /// Min-max normalization → all values in [0, 1].
    fn normalize(&self) -> Vec<f64> {
        let data = self.as_f64s();
        if data.is_empty() {
            return vec![];
        }
        let (lo, hi) = min_max(data);
        let range = hi - lo;
        if range.abs() < f64::EPSILON {
            return vec![0.0; data.len()];
        }
        data.iter().map(|v| (v - lo) / range).collect()
    }

    /// Z-score standardization (mean=0, std=1).
    fn standardize(&self) -> Vec<f64> {
        let data = self.as_f64s();
        let n = data.len() as f64;
        if n == 0.0 {
            return vec![];
        }
        let mean = data.iter().sum::<f64>() / n;
        let std = (data.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n).sqrt();
        if std.abs() < f64::EPSILON {
            return vec![0.0; data.len()];
        }
        data.iter().map(|v| (v - mean) / std).collect()
    }

    /// Clamp all values to [lo, hi], then shift to [0, 1].
    fn normalize_clamped(&self, lo: f64, hi: f64) -> Vec<f64> {
        let range = hi - lo;
        if range.abs() < f64::EPSILON {
            return vec![0.0; self.as_f64s().len()];
        }
        self.as_f64s()
            .iter()
            .map(|v| (v.clamp(lo, hi) - lo) / range)
            .collect()
    }
}

impl<T: ?Sized + AsF64s> NormalizeOps for T {}

// ═══════════════════════════════════════════════════════════════════════════
// ── Histogram — distribution analysis ───────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

/// Distribution analysis via histograms.
pub trait Histogram: AsF64s {
    /// Compute a histogram with `bins` equal-width buckets.
    /// Returns (counts, bin_edges) where bin_edges.len() == bins + 1.
    fn histogram(&self, bins: usize) -> (Vec<usize>, Vec<f64>) {
        let data = self.as_f64s();
        if data.is_empty() || bins == 0 {
            return (vec![], vec![]);
        }
        let (lo, hi) = min_max(data);
        if (hi - lo).abs() < f64::EPSILON {
            let mut counts = vec![0; bins];
            counts[0] = data.len();
            let edges: Vec<f64> = (0..=bins).map(|i| lo + i as f64).collect();
            return (counts, edges);
        }
        let width = (hi - lo) / bins as f64;
        let edges: Vec<f64> = (0..=bins).map(|i| lo + i as f64 * width).collect();
        let mut counts = vec![0usize; bins];
        for &v in data {
            let idx = ((v - lo) / width).floor() as usize;
            counts[idx.min(bins - 1)] += 1;
        }
        (counts, edges)
    }
}

impl<T: ?Sized + AsF64s> Histogram for T {}

// ═══════════════════════════════════════════════════════════════════════════
// ── SortOps — sorted views and index-based sorts ────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

/// Sorting utilities — sorted copies, argsort, top-k / bottom-k.
pub trait SortOps: AsF64s {
    /// Return a sorted copy (ascending).
    fn sorted(&self) -> Vec<f64> {
        sorted_copy(self.as_f64s())
    }

    /// Indices that would sort the data (ascending).
    fn argsort(&self) -> Vec<usize> {
        let data = self.as_f64s();
        let mut indices: Vec<usize> = (0..data.len()).collect();
        indices.sort_by(|&a, &b| {
            data[a]
                .partial_cmp(&data[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        indices
    }

    /// Top-k largest values (descending order).
    fn topk(&self, k: usize) -> Vec<f64> {
        let mut v = self.as_f64s().to_vec();
        v.sort_by(|a, b| crate::f64_cmp(b, a));
        v.truncate(k);
        v
    }

    /// Bottom-k smallest values (ascending order).
    fn bottomk(&self, k: usize) -> Vec<f64> {
        let mut v = self.sorted();
        v.truncate(k);
        v
    }
}

impl<T: ?Sized + AsF64s> SortOps for T {}

// ═══════════════════════════════════════════════════════════════════════════
// ── SearchOps — searching and filtering ─────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

/// Searching and filtering operations on numeric data.
pub trait SearchOps: AsF64s {
    /// Index of the nearest value to `target`.
    fn nearest(&self, target: f64) -> Option<usize> {
        let d = self.as_f64s();
        if d.is_empty() {
            return None;
        }
        let mut best = 0;
        let mut best_dist = (d[0] - target).abs();
        for (i, &v) in d.iter().enumerate().skip(1) {
            let dist = (v - target).abs();
            if dist < best_dist {
                best = i;
                best_dist = dist;
            }
        }
        Some(best)
    }

    /// Count values satisfying a predicate.
    fn count_where(&self, pred: impl Fn(f64) -> bool) -> usize {
        self.as_f64s().iter().filter(|&&v| pred(v)).count()
    }

    /// Indices of values satisfying a predicate.
    fn where_indices(&self, pred: impl Fn(f64) -> bool) -> Vec<usize> {
        self.as_f64s()
            .iter()
            .enumerate()
            .filter(|&(_, &v)| pred(v))
            .map(|(i, _)| i)
            .collect()
    }

    /// Whether any value satisfies the predicate.
    fn any(&self, pred: impl Fn(f64) -> bool) -> bool {
        self.as_f64s().iter().any(|&v| pred(v))
    }

    /// Whether all values satisfy the predicate.
    fn all(&self, pred: impl Fn(f64) -> bool) -> bool {
        self.as_f64s().iter().all(|&v| pred(v))
    }
}

impl<T: ?Sized + AsF64s> SearchOps for T {}

// ═══════════════════════════════════════════════════════════════════════════
// ── ReduceOps — rolling, cumulative, windowed operations ────────────────
// ═══════════════════════════════════════════════════════════════════════════

/// Cumulative and windowed aggregation.
pub trait ReduceOps: AsF64s {
    /// Cumulative sum.
    fn cumsum(&self) -> Vec<f64> {
        let mut acc = 0.0;
        self.as_f64s()
            .iter()
            .map(|&v| {
                acc += v;
                acc
            })
            .collect()
    }

    /// Cumulative product.
    fn cumprod(&self) -> Vec<f64> {
        let mut acc = 1.0;
        self.as_f64s()
            .iter()
            .map(|&v| {
                acc *= v;
                acc
            })
            .collect()
    }

    /// First differences: out[i] = data[i+1] - data[i].
    fn diff(&self) -> Vec<f64> {
        let d = self.as_f64s();
        d.windows(2).map(|w| w[1] - w[0]).collect()
    }

    /// Rolling mean with window size `w`.
    fn rolling_mean(&self, w: usize) -> Vec<f64> {
        let d = self.as_f64s();
        if w == 0 || d.len() < w {
            return vec![];
        }
        let mut sum: f64 = d[..w].iter().sum();
        let mut out = Vec::with_capacity(d.len() - w + 1);
        out.push(sum / w as f64);
        for i in w..d.len() {
            sum += d[i] - d[i - w];
            out.push(sum / w as f64);
        }
        out
    }

    /// Dot product with another slice of the same length.
    fn dot_product(&self, other: &[f64]) -> f64 {
        self.as_f64s()
            .iter()
            .zip(other.iter())
            .map(|(a, b)| a * b)
            .sum()
    }

    /// Sum of all values.
    fn total(&self) -> f64 {
        self.as_f64s().iter().sum()
    }

    /// Product of all values.
    fn prod(&self) -> f64 {
        self.as_f64s().iter().product()
    }
}

impl<T: ?Sized + AsF64s> ReduceOps for T {}

// ═══════════════════════════════════════════════════════════════════════════
// ── Approx — approximate equality ──────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

/// Approximate equality with configurable epsilon.
pub trait Approx {
    /// Whether `self` and `other` are equal within `epsilon`.
    fn approx_eq(&self, other: &Self, epsilon: f64) -> bool;
}

impl Approx for f64 {
    fn approx_eq(&self, other: &Self, epsilon: f64) -> bool {
        (self - other).abs() <= epsilon
    }
}

impl Approx for f32 {
    fn approx_eq(&self, other: &Self, epsilon: f64) -> bool {
        (*self as f64 - *other as f64).abs() <= epsilon
    }
}

impl Approx for Color {
    fn approx_eq(&self, other: &Self, epsilon: f64) -> bool {
        let e = epsilon as i16;
        (self.r as i16 - other.r as i16).abs() <= e
            && (self.g as i16 - other.g as i16).abs() <= e
            && (self.b as i16 - other.b as i16).abs() <= e
            && (self.a as i16 - other.a as i16).abs() <= e
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ── Diff — compare two instances ────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

/// Numeric difference between two instances.
#[derive(Debug, Clone, Copy)]
pub struct DiffResult {
    pub max_abs_diff: f64,
    pub mean_abs_diff: f64,
    pub rms_diff: f64,
    pub count: usize,
}

impl std::fmt::Display for DiffResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "max_abs={:.6} mean_abs={:.6} rms={:.6} n={}",
            self.max_abs_diff, self.mean_abs_diff, self.rms_diff, self.count
        )
    }
}

/// Compare two numeric instances element-wise.
pub trait Diff: AsF64s {
    fn diff_from(&self, other: &Self) -> DiffResult {
        let a = self.as_f64s();
        let b = other.as_f64s();
        let n = a.len().min(b.len());
        if n == 0 {
            return DiffResult {
                max_abs_diff: 0.0,
                mean_abs_diff: 0.0,
                rms_diff: 0.0,
                count: 0,
            };
        }
        let mut max_d = 0.0f64;
        let mut sum_d = 0.0;
        let mut sum_sq = 0.0;
        for i in 0..n {
            let d = (a[i] - b[i]).abs();
            if d > max_d {
                max_d = d;
            }
            sum_d += d;
            sum_sq += d * d;
        }
        DiffResult {
            max_abs_diff: max_d,
            mean_abs_diff: sum_d / n as f64,
            rms_diff: (sum_sq / n as f64).sqrt(),
            count: n,
        }
    }
}

impl<T: ?Sized + AsF64s> Diff for T {}

// ═══════════════════════════════════════════════════════════════════════════
// ── Correlation trait — auto-derived covariance/correlation ──────────────
// ═══════════════════════════════════════════════════════════════════════════

/// Auto-derived pairwise analysis for any numeric data.
pub trait PairwiseOps: AsF64s {
    /// Compute cross-correlation coefficient with another sequence.
    fn cross_correlate(&self, other: &impl AsF64s) -> f64 {
        let a = self.as_f64s();
        let b = other.as_f64s();
        let n = a.len().min(b.len()) as f64;
        if n < 2.0 {
            return 0.0;
        }
        let ma = a.iter().sum::<f64>() / n;
        let mb = b.iter().sum::<f64>() / n;
        let (mut cov, mut va, mut vb) = (0.0, 0.0, 0.0);
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

    /// Euclidean distance to another numeric sequence.
    fn euclidean_distance(&self, other: &impl AsF64s) -> f64 {
        self.as_f64s()
            .iter()
            .zip(other.as_f64s().iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Cosine similarity with another sequence.
    fn cosine_similarity(&self, other: &impl AsF64s) -> f64 {
        let a = self.as_f64s();
        let b = other.as_f64s();
        let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let ma = a.iter().map(|v| v * v).sum::<f64>().sqrt();
        let mb = b.iter().map(|v| v * v).sum::<f64>().sqrt();
        let denom = ma * mb;
        if denom < f64::EPSILON {
            0.0
        } else {
            dot / denom
        }
    }

    /// Manhattan distance (L1) to another sequence.
    fn manhattan_distance(&self, other: &impl AsF64s) -> f64 {
        self.as_f64s()
            .iter()
            .zip(other.as_f64s().iter())
            .map(|(a, b)| (a - b).abs())
            .sum()
    }
}

impl<T: ?Sized + AsF64s> PairwiseOps for T {}
