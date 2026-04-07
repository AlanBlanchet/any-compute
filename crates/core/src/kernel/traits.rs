use serde::Serialize;
use std::fmt;

use super::{BinaryOp, ReduceOp, UnaryOp};

// ── Kernel trait ──────────────────────────────────────────────────────────

/// Execution statistics returned after a kernel dispatch.
#[derive(Debug, Clone, Serialize)]
pub struct KernelStats {
    /// Wall-clock time for the dispatch.
    pub duration_us: u128,
    /// FLOPS achieved (0 if not measurable).
    pub flops: f64,
    /// Memory bandwidth achieved in bytes/sec (0 if not measurable).
    pub bandwidth_bytes_sec: f64,
}

/// Hardware-agnostic kernel interface.
///
/// Implementations live behind feature flags — the user's code only uses this trait.
/// The [`crate::compute::Device`] delegates to `Kernel` methods internally.
pub trait Kernel: Send + Sync + fmt::Debug {
    /// Human-readable name (e.g. "CpuSimd/AVX2", "CUDA/cuBLAS").
    fn name(&self) -> &str;

    /// Which vendor backend is this?
    fn backend_tag(&self) -> KernelBackend;

    /// Available SIMD / warp / wavefront width (elements per lane).
    fn vector_width(&self) -> usize;

    /// Execute a unary map: out[i] = op(data[i])
    fn map_unary_f64(&self, data: &[f64], op: UnaryOp) -> Vec<f64>;

    /// Execute a binary map: out[i] = op(a[i], b[i])
    fn map_binary_f64(&self, a: &[f64], b: &[f64], op: BinaryOp) -> Vec<f64>;

    /// Reduction to scalar.
    fn reduce_f64(&self, data: &[f64], op: ReduceOp) -> f64;

    /// Inclusive prefix scan.
    fn scan_f64(&self, data: &[f64], op: ReduceOp) -> Vec<f64>;

    /// Matrix multiply: C[m×n] = A[m×k] × B[k×n] (row-major).
    fn gemm_f64(&self, a: &[f64], b: &[f64], m: usize, n: usize, k: usize) -> Vec<f64>;

    /// Sort (unstable, ascending).
    fn sort_f64(&self, data: &mut [f64]);

    /// Gather: out[i] = data[indices[i]].
    fn gather_f64(&self, data: &[f64], indices: &[usize]) -> Vec<f64> {
        indices.iter().map(|&i| data[i]).collect()
    }

    /// Scatter: out[indices[i]] = values[i]. Returns a vec of size `out_len`.
    fn scatter_f64(&self, values: &[f64], indices: &[usize], out_len: usize) -> Vec<f64> {
        let mut out = vec![0.0; out_len];
        for (&v, &i) in values.iter().zip(indices) {
            out[i] = v;
        }
        out
    }

    /// Reports self-benchmark stats for the given op on current hardware.
    fn benchmark_op(&self, op: &super::KernelOp) -> KernelStats;
}

/// Which vendor backend a kernel targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum KernelBackend {
    CpuScalar,
    CpuSimd,
    Cuda,
    Rocm,
    Mkl,
    Metal,
    Wgpu,
}

display_enum!(KernelBackend {
    CpuScalar => "CPU/Scalar",
    CpuSimd   => "CPU/SIMD",
    Cuda      => "NVIDIA/CUDA",
    Rocm      => "AMD/ROCm",
    Mkl       => "Intel/MKL",
    Metal     => "Apple/Metal",
    Wgpu      => "wgpu",
});

/// Maps a fine-grained [`KernelBackend`] to the coarser [`BackendKind`]
/// used by the compute dispatch layer.
impl From<KernelBackend> for crate::compute::BackendKind {
    fn from(kb: KernelBackend) -> Self {
        match kb {
            KernelBackend::CpuScalar | KernelBackend::CpuSimd | KernelBackend::Mkl => Self::Cpu,
            KernelBackend::Cuda => Self::Cuda,
            KernelBackend::Rocm => Self::Rocm,
            KernelBackend::Metal | KernelBackend::Wgpu => Self::Wgpu,
        }
    }
}
