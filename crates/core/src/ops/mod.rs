//! Composable utility traits — implement one foundational trait, get many methods for free.
//!
//! ## Design principle
//!
//! Small foundation traits (`AsF64s`, `Renderable`) provide raw data access.
//! Extension traits (`Stats`, `Norm`, `Approx`, `Capture`, etc.) are auto-derived
//! via blanket implementations — any type that yields numeric data instantly gets
//! statistical analysis, normalization, norms, histograms, and more.
//!
//! ## Trait cascade
//!
//! ```text
//! AsF64s ──► Stats      (min, max, mean, std_dev, variance, median, percentile)
//!        ──► Norm       (l1, l2, linf)
//!        ──► NormalizeOps (normalize to [0,1], standardize z-score)
//!        ──► Histogram  (distribution analysis)
//!        ──► SortOps    (sorted, argsort, topk, bottomk)
//!        ──► SearchOps  (binary search, nearest, count, contains)
//!        ──► ReduceOps  (sum, product, cumsum, diff, windowed)
//!
//! Renderable ──► Capture   (snapshot → PixelBuffer → PNG bytes)
//!            ──► Record    (multi-frame capture → frame sequence)
//!
//! Approx     (approximate equality for numerics)
//! Summary    (human-readable one-liner for any type)
//! Export     (serialize to format-specific encodings)
//! ```

mod capture;
mod math_ops;
mod pixel_ops;
mod summary;
mod traits;

pub use capture::*;
pub use summary::*;
pub use traits::*;

// ═══════════════════════════════════════════════════════════════════════════
// ── Tests ────────────────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::Buffer;
    use crate::layout::{Mat3, Rect, V3};
    use crate::render::{Color, ColoredRect, PixelBuffer};

    // ── AsF64s + Stats ──────────────────────────────────────────────────

    #[test]
    fn buffer_stats() {
        let b = Buffer::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let s = b.stats();
        assert_eq!(s.count, 5);
        assert!((s.min - 1.0).abs() < 1e-10);
        assert!((s.max - 5.0).abs() < 1e-10);
        assert!((s.mean - 3.0).abs() < 1e-10);
        assert!((s.median - 3.0).abs() < 1e-10);
    }

    #[test]
    fn vec_stats() {
        let v = V3::new3(3.0, 4.0, 0.0);
        let s = v.stats();
        assert_eq!(s.count, 3);
        assert!((s.median - 3.0).abs() < 1e-10);
    }

    #[test]
    fn slice_stats() {
        let data: &[f64] = &[10.0, 20.0, 30.0];
        let s = data.stats();
        assert!((s.mean - 20.0).abs() < 1e-10);
    }

    #[test]
    fn percentile_basic() {
        let b = Buffer::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!((b.percentile(0.0) - 1.0).abs() < 1e-10);
        assert!((b.percentile(100.0) - 5.0).abs() < 1e-10);
        assert!((b.percentile(50.0) - 3.0).abs() < 1e-10);
    }

    // ── Norm ────────────────────────────────────────────────────────────

    #[test]
    fn vector_norms() {
        let v = V3::new3(3.0, 4.0, 0.0);
        assert!((v.l2_norm() - 5.0).abs() < 1e-10);
        assert!((v.l1_norm() - 7.0).abs() < 1e-10);
        assert!((v.linf_norm() - 4.0).abs() < 1e-10);
    }

    #[test]
    fn matrix_frobenius() {
        let m = Mat3::identity();
        assert!((m.frobenius() - 3.0f64.sqrt()).abs() < 1e-10);
    }

    // ── NormalizeOps ────────────────────────────────────────────────────

    #[test]
    fn normalize_range() {
        let data = vec![0.0, 50.0, 100.0];
        let n = data.normalize();
        assert!((n[0]).abs() < 1e-10);
        assert!((n[1] - 0.5).abs() < 1e-10);
        assert!((n[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn standardize_zscore() {
        let data = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let z = data.standardize();
        // Mean of z-scores should be ~0
        let mean: f64 = z.iter().sum::<f64>() / z.len() as f64;
        assert!(mean.abs() < 1e-10);
    }

    // ── Histogram ───────────────────────────────────────────────────────

    #[test]
    fn histogram_basic() {
        let data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let (counts, edges) = data.histogram(5);
        assert_eq!(counts.len(), 5);
        assert_eq!(edges.len(), 6);
        assert_eq!(counts.iter().sum::<usize>(), 10);
    }

    // ── SortOps ─────────────────────────────────────────────────────────

    #[test]
    fn argsort_basic() {
        let data = vec![30.0, 10.0, 20.0];
        let idx = data.argsort();
        assert_eq!(idx, vec![1, 2, 0]);
    }

    #[test]
    fn topk_basic() {
        let data = vec![5.0, 1.0, 9.0, 3.0, 7.0];
        assert_eq!(data.topk(3), vec![9.0, 7.0, 5.0]);
    }

    // ── SearchOps ───────────────────────────────────────────────────────

    #[test]
    fn nearest_value() {
        let data = vec![1.0, 5.0, 10.0, 20.0];
        assert_eq!(data.nearest(6.0), Some(1)); // 5.0 is nearest
        assert_eq!(data.nearest(15.0), Some(2)); // 10.0 is nearest
    }

    #[test]
    fn count_and_filter() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(data.count_where(|v| v > 3.0), 2);
        assert_eq!(data.where_indices(|v| v > 3.0), vec![3, 4]);
    }

    // ── ReduceOps ───────────────────────────────────────────────────────

    #[test]
    fn cumsum_diff() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(data.cumsum(), vec![1.0, 3.0, 6.0, 10.0]);
        assert_eq!(data.diff(), vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn rolling_mean_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let rm = data.rolling_mean(3);
        assert_eq!(rm.len(), 3);
        assert!((rm[0] - 2.0).abs() < 1e-10);
        assert!((rm[1] - 3.0).abs() < 1e-10);
        assert!((rm[2] - 4.0).abs() < 1e-10);
    }

    // ── Approx ──────────────────────────────────────────────────────────

    #[test]
    fn approx_f64() {
        assert!(1.0f64.approx_eq(&1.0000001, 1e-6));
        assert!(!1.0f64.approx_eq(&2.0, 0.5));
    }

    #[test]
    fn approx_vector() {
        let a = V3::new3(1.0, 2.0, 3.0);
        let b = V3::new3(1.0001, 2.0001, 3.0001);
        assert!(a.approx_eq(&b, 1e-3));
        assert!(!a.approx_eq(&b, 1e-5));
    }

    #[test]
    fn approx_matrix() {
        let a = Mat3::identity();
        let b = Mat3::identity();
        assert!(a.approx_eq(&b, 1e-15));
    }

    #[test]
    fn approx_color() {
        let a = Color::rgb(100, 200, 50);
        let b = Color::rgb(101, 199, 51);
        assert!(a.approx_eq(&b, 2.0));
        assert!(!a.approx_eq(&b, 0.5));
    }

    // ── Diff ────────────────────────────────────────────────────────────

    #[test]
    fn diff_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let d = a.diff_from(&a);
        assert!(d.max_abs_diff < 1e-15);
    }

    #[test]
    fn diff_offset() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 1.0, 1.0];
        let d = a.diff_from(&b);
        assert!((d.max_abs_diff - 1.0).abs() < 1e-10);
        assert!((d.mean_abs_diff - 1.0).abs() < 1e-10);
    }

    // ── Summary ─────────────────────────────────────────────────────────

    #[test]
    fn buffer_summary() {
        let b = Buffer::new(vec![1.0, 2.0, 3.0]);
        let s = b.summary();
        assert!(s.contains("Buffer"));
        assert!(s.contains("n=3"));
    }

    #[test]
    fn vector_summary() {
        let v = V3::new3(3.0, 4.0, 0.0);
        let s = v.summary();
        assert!(s.contains("V3"));
        assert!(s.contains("5.0")); // magnitude
    }

    // ── Export ───────────────────────────────────────────────────────────

    #[test]
    fn buffer_csv() {
        let b = Buffer::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(b.to_csv(), "1,2,3");
    }

    #[test]
    fn vector_json() {
        let v = V3::new3(1.0, 2.0, 3.0);
        let j = v.to_json();
        assert!(j.contains("1.0"));
        assert!(j.contains("3.0"));
    }

    #[test]
    fn matrix_csv() {
        let m = crate::layout::Mat2::new([[1.0, 2.0], [3.0, 4.0]]);
        let csv = m.to_csv();
        assert!(csv.contains("1,2"));
        assert!(csv.contains("3,4"));
    }

    // ── Capture ─────────────────────────────────────────────────────────

    #[test]
    fn capture_colored_rect() {
        let obj = ColoredRect {
            rect: Rect::new(10.0, 10.0, 80.0, 80.0),
            color: Color::WHITE,
            corner_radius: 0.0,
        };
        let buf = obj.capture(100, 100);
        // Center should be white (the filled rect).
        assert_eq!(buf.pixel(50, 50), Color::WHITE);
    }

    // ── PixelBuffer utilities ───────────────────────────────────────────

    #[test]
    fn pixelbuf_crop() {
        let mut buf = PixelBuffer::new(10, 10, Color::BLACK);
        buf.pixels[55] = Color::WHITE; // (5, 5)
        let crop = buf.crop(4, 4, 4, 4);
        assert_eq!(crop.pixel(1, 1), Color::WHITE);
        assert_eq!(crop.width, 4);
    }

    #[test]
    fn pixelbuf_dominant_color() {
        let buf = PixelBuffer::new(10, 10, Color::rgb(42, 42, 42));
        assert_eq!(buf.dominant_color(), Color::rgb(42, 42, 42));
    }

    #[test]
    fn pixelbuf_downscale() {
        let buf = PixelBuffer::new(4, 4, Color::rgb(100, 100, 100));
        let small = buf.downscale(2);
        assert_eq!(small.width, 2);
        assert_eq!(small.height, 2);
        assert_eq!(small.pixel(0, 0), Color::rgb(100, 100, 100));
    }

    // ── Buffer utilities ────────────────────────────────────────────────

    #[test]
    fn buffer_linspace() {
        let b = Buffer::linspace(0.0, 1.0, 5);
        assert_eq!(b.len(), 5);
        assert!((b.data()[0]).abs() < 1e-10);
        assert!((b.data()[4] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn buffer_arange() {
        let b = Buffer::arange(0.0, 5.0, 1.0);
        assert_eq!(b.len(), 5);
        assert_eq!(b.data(), &[0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn buffer_concat_slice() {
        let a = Buffer::new(vec![1.0, 2.0]);
        let b = Buffer::new(vec![3.0, 4.0]);
        let c = a.concat(&b);
        assert_eq!(c.data(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn buffer_unique() {
        let b = Buffer::new(vec![3.0, 1.0, 2.0, 1.0, 3.0]);
        let u = b.unique();
        assert_eq!(u.data(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn buffer_clamp_pow() {
        let b = Buffer::new(vec![-1.0, 0.5, 2.0]);
        let c = b.clamp(0.0, 1.0);
        assert_eq!(c.data(), &[0.0, 0.5, 1.0]);
        let p = Buffer::new(vec![4.0, 9.0]).pow(0.5);
        assert!((p.data()[0] - 2.0).abs() < 1e-10);
        assert!((p.data()[1] - 3.0).abs() < 1e-10);
    }

    // ── V<N> utilities ──────────────────────────────────────────────────

    #[test]
    fn vector_angle_project() {
        let a = V3::new3(1.0, 0.0, 0.0);
        let b = V3::new3(0.0, 1.0, 0.0);
        assert!((a.angle_to(b) - std::f64::consts::FRAC_PI_2).abs() < 1e-10);

        let v = V3::new3(3.0, 4.0, 0.0);
        let onto = V3::new3(1.0, 0.0, 0.0);
        let proj = v.project_onto(onto);
        assert!((proj.x - 3.0).abs() < 1e-10);
        assert!(proj.y.abs() < 1e-10);
    }

    #[test]
    fn vector_smoothstep() {
        let v = V3::new3(0.0, 0.5, 1.0);
        let ss = v.smoothstep(0.0, 1.0);
        assert!((ss.0[0]).abs() < 1e-10);
        assert!((ss.0[2] - 1.0).abs() < 1e-10);
    }

    // ── Matrix utilities ────────────────────────────────────────────────

    #[test]
    fn matrix_trace_diagonal() {
        let m = Mat3::identity();
        assert!((m.trace() - 3.0).abs() < 1e-10);
        assert_eq!(m.diagonal(), V3::new3(1.0, 1.0, 1.0));
    }

    #[test]
    fn matrix_from_diagonal() {
        let m = Mat3::from_diagonal(V3::new3(2.0, 3.0, 4.0));
        assert!((m.data[0][0] - 2.0).abs() < 1e-10);
        assert!((m.data[1][1] - 3.0).abs() < 1e-10);
        assert!((m.data[2][2] - 4.0).abs() < 1e-10);
        assert!(m.data[0][1].abs() < 1e-10);
    }

    #[test]
    fn matrix_symmetric_check() {
        let m = Mat3::identity();
        assert!(m.is_symmetric(1e-10));
        assert!(m.is_identity(1e-10));
    }

    #[test]
    fn matrix_hadamard() {
        let a = crate::layout::Mat2::new([[1.0, 2.0], [3.0, 4.0]]);
        let b = crate::layout::Mat2::new([[2.0, 3.0], [4.0, 5.0]]);
        let h = a.hadamard(&b);
        assert!((h.data[0][0] - 2.0).abs() < 1e-10);
        assert!((h.data[1][1] - 20.0).abs() < 1e-10);
    }

    // ── Mesh export ─────────────────────────────────────────────────────

    #[test]
    fn mesh_to_obj_roundtrip() {
        let cube = crate::scene::Mesh::cube(V3::ZERO, 1.0);
        let obj_str = cube.to_obj();
        assert!(obj_str.contains("v "));
        assert!(obj_str.contains("f "));
        let parsed = crate::scene::Mesh::from_obj(&obj_str);
        assert_eq!(parsed.vertex_count(), cube.vertex_count());
    }

    // ── Convolution ─────────────────────────────────────────────────────

    #[test]
    fn buffer_convolve() {
        let b = Buffer::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let kernel = [1.0, 0.0, -1.0]; // simple edge detector
        let c = b.convolve(&kernel);
        assert_eq!(c.len(), 3); // valid mode
        // c[0] = 1*1 + 2*0 + 3*(-1) = -2
        assert!((c.data()[0] - (-2.0)).abs() < 1e-10);
    }

    // ── Correlation / Covariance ────────────────────────────────────────

    #[test]
    fn buffer_correlation() {
        let a = Buffer::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let b = Buffer::new(vec![2.0, 4.0, 6.0, 8.0, 10.0]);
        assert!((a.correlation(&b) - 1.0).abs() < 1e-10); // perfect correlation
    }

    #[test]
    fn buffer_covariance() {
        let a = Buffer::new(vec![1.0, 2.0, 3.0]);
        let b = Buffer::new(vec![1.0, 2.0, 3.0]);
        let cov = a.covariance(&b);
        // cov = mean((x-mx)(y-my)) = ((−1)(−1)+(0)(0)+(1)(1))/3 = 2/3
        assert!((cov - 2.0 / 3.0).abs() < 1e-10);
    }

    // ── Gradient ────────────────────────────────────────────────────────

    #[test]
    fn buffer_gradient() {
        let b = Buffer::new(vec![0.0, 1.0, 4.0, 9.0]);
        let g = b.gradient();
        assert_eq!(g.len(), 4);
        assert!((g.data()[0] - 1.0).abs() < 1e-10); // forward diff
        assert!((g.data()[1] - 2.0).abs() < 1e-10); // central diff (4-0)/2
    }

    // ── Interpolation ───────────────────────────────────────────────────

    #[test]
    fn buffer_interp() {
        let b = Buffer::new(vec![0.0, 10.0, 20.0]);
        let interp = b.interp(&[0.5, 1.5]);
        assert!((interp.data()[0] - 5.0).abs() < 1e-10);
        assert!((interp.data()[1] - 15.0).abs() < 1e-10);
    }

    // ── EMA ─────────────────────────────────────────────────────────────

    #[test]
    fn buffer_ema() {
        let b = Buffer::new(vec![1.0, 2.0, 3.0, 4.0]);
        let e = b.ema(0.5);
        assert!((e.data()[0] - 1.0).abs() < 1e-10); // first stays same
        assert!((e.data()[1] - 1.5).abs() < 1e-10); // 0.5*2 + 0.5*1
    }

    // ── Softmax ─────────────────────────────────────────────────────────

    #[test]
    fn buffer_softmax() {
        let b = Buffer::new(vec![1.0, 2.0, 3.0]);
        let s = b.softmax();
        let sum: f64 = s.data().iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        // Values should be monotonically increasing
        assert!(s.data()[0] < s.data()[1]);
        assert!(s.data()[1] < s.data()[2]);
    }

    // ── Distance matrix ─────────────────────────────────────────────────

    #[test]
    fn buffer_distance_matrix() {
        // 3 2D points: (0,0), (3,0), (0,4)
        let b = Buffer::new(vec![0.0, 0.0, 3.0, 0.0, 0.0, 4.0]);
        let dm = b.distance_matrix(2);
        assert_eq!(dm.len(), 9); // 3×3
        assert!((dm.data()[0] - 0.0).abs() < 1e-10); // self-distance
        assert!((dm.data()[1] - 3.0).abs() < 1e-10); // (0,0)→(3,0)
        assert!((dm.data()[2] - 4.0).abs() < 1e-10); // (0,0)→(0,4)
        assert!((dm.data()[5] - 5.0).abs() < 1e-10); // (3,0)→(0,4) = 5, at [1][2]
    }

    // ── PairwiseOps trait ───────────────────────────────────────────────

    #[test]
    fn pairwise_cosine() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(a.cosine_similarity(&b).abs() < 1e-10); // perpendicular = 0
        assert!((a.cosine_similarity(&a) - 1.0).abs() < 1e-10); // parallel = 1
    }

    #[test]
    fn pairwise_euclidean() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        assert!((a.euclidean_distance(&b) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn pairwise_manhattan() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        assert!((a.manhattan_distance(&b) - 7.0).abs() < 1e-10);
    }

    // ── PixelBuffer color ops ───────────────────────────────────────────

    #[test]
    fn pixelbuf_grayscale() {
        let mut buf = PixelBuffer::new(2, 1, Color::BLACK);
        buf.pixels[0] = Color::rgb(255, 0, 0);
        let gray = buf.to_grayscale();
        // Red → lum ≈ 54
        assert!(gray.pixel(0, 0).r > 50 && gray.pixel(0, 0).r < 60);
        assert_eq!(gray.pixel(0, 0).r, gray.pixel(0, 0).g); // R == G == B
    }

    #[test]
    fn pixelbuf_invert() {
        let buf = PixelBuffer::new(1, 1, Color::rgb(100, 150, 200));
        let inv = buf.invert();
        assert_eq!(inv.pixel(0, 0), Color::rgb(155, 105, 55));
    }

    #[test]
    fn pixelbuf_diff_count() {
        let a = PixelBuffer::new(2, 2, Color::rgb(100, 100, 100));
        let mut b = a.clone();
        b.pixels[0] = Color::rgb(200, 100, 100); // differs by 100
        assert_eq!(a.diff_count(&b, 50), 1); // 1 pixel over threshold
        assert_eq!(a.diff_count(&b, 150), 0); // none over threshold
    }

    #[test]
    fn pixelbuf_region_uniform() {
        let buf = PixelBuffer::new(4, 4, Color::rgb(42, 42, 42));
        assert!(buf.region_uniform(0, 0, 4, 4, 0));
    }

    #[test]
    fn pixelbuf_composite() {
        let bg = PixelBuffer::new(4, 4, Color::BLACK);
        let fg = PixelBuffer::new(2, 2, Color::WHITE);
        let comp = bg.composite(&fg, 1, 1);
        assert_eq!(comp.pixel(0, 0), Color::BLACK); // outside overlay
        assert_eq!(comp.pixel(1, 1), Color::WHITE); // inside overlay
    }
}
