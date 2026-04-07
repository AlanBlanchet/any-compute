//! Device-aware numeric buffer — operations auto-dispatch to the best hardware.
//!
//! Instead of calling `device.reduce(data, ReduceOp::Sum)` manually,
//! write `buffer.sum()`. The dispatch routes through CPU SIMD, or a GPU/CUDA
//! device if configured.
//!
//! ## Design
//!
//! `Buffer` wraps a `Vec<f64>` + a [`Device`] handle. By default it uses the
//! global best device (auto-detected CPU SIMD). Swap the device to route all
//! ops through GPU transparently.
//!
//! ```text
//! ┌─────────┐     ┌───────────────────────────┐
//! │  Buffer  │────▶│  Device                   │
//! │  ───────  │     │  ├─ CPU/SIMD (default)    │
//! │  data     │     │  ├─ CUDA    (feature)     │
//! │  device   │     │  ├─ wgpu   (feature)      │
//! └─────────┘     └───────────────────────────┘
//! ```

use crate::compute::Device;
use crate::kernel::{BinaryOp, ReduceOp, UnaryOp};

/// Generates reduction methods on Buffer (each delegates to `device.reduce`).
macro_rules! buffer_reduce_ops {
    ($($method:ident => $op:expr),* $(,)?) => {
        $(pub fn $method(&self) -> f64 {
            self.device.reduce(&self.data, $op)
        })*
    }
}

/// Generates unary map methods on Buffer (each delegates to `device.unary`).
macro_rules! buffer_unary_ops {
    ($($method:ident => $op:expr),* $(,)?) => {
        $(pub fn $method(&self) -> Self {
            self.with_data(self.device.unary(&self.data, $op))
        })*
    }
}

/// A numeric buffer whose operations auto-dispatch to the best available device.
///
/// All heavy computation (reductions, maps, GEMM) goes through the [`Device`],
/// so swapping from CPU to GPU transparently accelerates every operation.
#[derive(Clone)]
pub struct Buffer {
    data: Vec<f64>,
    device: Device,
}

impl std::fmt::Debug for Buffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Buffer(len={}, device={})",
            self.data.len(),
            self.device.name()
        )
    }
}

impl Buffer {
    pub fn new(data: Vec<f64>) -> Self {
        Self {
            data,
            device: Device::best(),
        }
    }

    /// Create a buffer on a specific device.
    pub fn on(device: &Device, data: Vec<f64>) -> Self {
        Self {
            data,
            device: device.clone(),
        }
    }

    /// The device this buffer dispatches through.
    pub fn device(&self) -> &Device {
        &self.device
    }

    fn with_data(&self, data: Vec<f64>) -> Self {
        Self {
            data,
            device: self.device.clone(),
        }
    }

    // ── Accessors ────────────────────────────────────────────────────────

    pub fn data(&self) -> &[f64] {
        &self.data
    }
    pub fn into_vec(self) -> Vec<f64> {
        self.data
    }
    pub fn len(&self) -> usize {
        self.data.len()
    }
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    // ── Reductions (device-dispatched) ────────────────────────────────────

    for_each_reduce!(buffer_reduce_ops);

    // ── Unary maps (device-dispatched) ────────────────────────────────────

    for_each_unary!(buffer_unary_ops);

    pub fn scale(&self, s: f64) -> Self {
        self.with_data(self.device.unary(&self.data, UnaryOp::Scale(s.into())))
    }
    pub fn offset(&self, o: f64) -> Self {
        self.with_data(self.device.unary(&self.data, UnaryOp::Offset(o.into())))
    }

    // ── Binary ops (device-dispatched) ────────────────────────────────────

    fn binary(&self, rhs: &Self, op: BinaryOp) -> Self {
        self.with_data(self.device.binary(&self.data, &rhs.data, op))
    }

    // ── Scan + Sort ──────────────────────────────────────────────────────

    pub fn prefix_sum(&self) -> Self {
        self.with_data(self.device.scan(&self.data, ReduceOp::Sum))
    }

    pub fn sort(&mut self) {
        self.device.sort(&mut self.data);
    }

    // ── Matrix ops ───────────────────────────────────────────────────────

    pub fn gemm(&self, rhs: &Self, m: usize, n: usize, k: usize) -> Self {
        self.with_data(self.device.gemm(&self.data, &rhs.data, m, n, k))
    }

    // ── Lazy bridge ──────────────────────────────────────────────────────

    /// Enter lazy mode: record operations in a [`Graph`] instead of executing them.
    ///
    /// All the same named methods (`.sqrt()`, `.scale()`, `.sum()`, etc.)
    /// are available on the returned [`LazyMut`] — they just record nodes
    /// instead of computing immediately.  Call `.eval()` to materialize.
    ///
    /// ```
    /// use any_compute_core::{buffer::Buffer, graph::Graph, compute::Device};
    /// use std::collections::HashMap;
    ///
    /// let buf = Buffer::new(vec![1.0, 4.0, 9.0]);
    /// let mut g = Graph::new();
    /// let result = buf.lazy(&mut g).sqrt().scale(2.0)
    ///     .eval(&Device::cpu(), &HashMap::new());
    /// assert_eq!(result, vec![2.0, 4.0, 6.0]);
    /// ```
    pub fn lazy<'g>(&self, graph: &'g mut crate::graph::Graph) -> crate::graph::LazyMut<'g> {
        let id = graph.from_buffer(self);
        crate::graph::LazyMut::from_raw(graph, id)
    }
}

// ── Operator overloads ───────────────────────────────────────────────────

impl std::ops::Add for Buffer {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        self.binary(&rhs, BinaryOp::Add)
    }
}

impl std::ops::Sub for Buffer {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        self.binary(&rhs, BinaryOp::Sub)
    }
}

impl std::ops::Mul for Buffer {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        self.binary(&rhs, BinaryOp::Mul)
    }
}

impl std::ops::Div for Buffer {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        self.binary(&rhs, BinaryOp::Div)
    }
}

// ── From conversions ─────────────────────────────────────────────────────

impl From<Vec<f64>> for Buffer {
    fn from(v: Vec<f64>) -> Self {
        Self::new(v)
    }
}

impl From<&[f64]> for Buffer {
    fn from(s: &[f64]) -> Self {
        Self::new(s.to_vec())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ── Tests ────────────────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

use crate::ops::{AsF64s, DeviceAware, Export, Stats, Summary, json_array, json_f64};

// ═══════════════════════════════════════════════════════════════════════════
// ── Ops trait impls ─────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

impl AsF64s for Buffer {
    fn as_f64s(&self) -> &[f64] {
        self.data()
    }
}

impl Summary for Buffer {
    fn summary(&self) -> String {
        let d = self.data();
        if d.is_empty() {
            return format!("Buffer(empty, device={})", self.device().name());
        }
        let s = self.stats();
        format!(
            "Buffer(n={}, min={:.3}, max={:.3}, mean={:.3}, std={:.3}, device={})",
            s.count,
            s.min,
            s.max,
            s.mean,
            s.std_dev,
            self.device().name()
        )
    }
}

impl DeviceAware for Buffer {
    fn to_device(&self, device: &crate::compute::Device) -> Self {
        Buffer::on(device, self.data().to_vec())
    }
    fn device_name(&self) -> String {
        self.device().name().to_string()
    }
}

impl Export for Buffer {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Buffer {
        Buffer::new(vec![1.0, 2.0, 3.0, 4.0, 5.0])
    }

    #[test]
    fn reductions() {
        let b = sample();
        assert!((b.sum() - 15.0).abs() < 1e-10);
        assert!((b.min() - 1.0).abs() < 1e-10);
        assert!((b.max() - 5.0).abs() < 1e-10);
        assert!((b.mean() - 3.0).abs() < 1e-10);
        assert!((b.product() - 120.0).abs() < 1e-10);
    }

    #[test]
    fn unary_ops() {
        let b = Buffer::new(vec![-2.0, 0.0, 3.0]);
        let abs = b.abs();
        assert_eq!(abs.data(), &[2.0, 0.0, 3.0]);
        let relu = b.relu();
        assert_eq!(relu.data(), &[0.0, 0.0, 3.0]);
        let scaled = b.scale(2.0);
        assert_eq!(scaled.data(), &[-4.0, 0.0, 6.0]);
    }

    #[test]
    fn binary_operators() {
        let a = Buffer::new(vec![1.0, 2.0, 3.0]);
        let b = Buffer::new(vec![4.0, 5.0, 6.0]);
        let c = a + b;
        assert_eq!(c.data(), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn prefix_sum() {
        let b = Buffer::new(vec![1.0, 2.0, 3.0, 4.0]);
        let ps = b.prefix_sum();
        assert_eq!(ps.data(), &[1.0, 3.0, 6.0, 10.0]);
    }

    #[test]
    fn gemm_identity() {
        // 2×2 identity × [1,2;3,4]
        let eye = Buffer::new(vec![1.0, 0.0, 0.0, 1.0]);
        let mat = Buffer::new(vec![1.0, 2.0, 3.0, 4.0]);
        let out = eye.gemm(&mat, 2, 2, 2);
        assert_eq!(out.data(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn from_conversions() {
        let b: Buffer = vec![1.0, 2.0].into();
        assert_eq!(b.len(), 2);
        let b2: Buffer = [3.0, 4.0, 5.0].as_slice().into();
        assert_eq!(b2.len(), 3);
    }

    #[test]
    fn sort_ascending() {
        let mut b = Buffer::new(vec![5.0, 1.0, 3.0, 2.0, 4.0]);
        b.sort();
        assert_eq!(b.data(), &[1.0, 2.0, 3.0, 4.0, 5.0]);
    }
}
