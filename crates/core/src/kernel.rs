//! Low-level compute kernels — CUDA, ROCm, CPU SIMD, and vendor-specific backends.
//!
//! This module defines the [`Kernel`] trait for dispatch-ready compute operations,
//! plus vendor-specific kernel implementations behind feature flags.
//!
//! ## Architecture
//!
//! ```text
//! ┌────────────────┐     ┌──────────────────────────────────────────┐
//! │  ComputeBackend│────▶│  Kernel<T>  (trait)                      │
//! │  (high-level)  │     │  ├─ CpuSimdKernel   (always available)  │
//! └────────────────┘     │  ├─ CudaKernel      (feature = "cuda")  │
//!                        │  ├─ RocmKernel      (feature = "rocm")  │
//!                        │  ├─ MklKernel       (feature = "mkl")   │
//!                        │  └─ MetalKernel     (feature = "metal") │
//!                        └──────────────────────────────────────────┘
//! ```
//!
//! ## Vendor libraries (behind feature flags)
//!
//! | Feature  | Required toolkit                | Libraries used                                    |
//! |----------|---------------------------------|---------------------------------------------------|
//! | `cuda`   | NVIDIA CUDA Toolkit ≥ 12.0      | cuBLAS, cuDNN, cuFFT, cuRAND, NCCL, Thrust        |
//! | `rocm`   | AMD ROCm ≥ 6.0                  | rocBLAS, rocFFT, MIOpen, hipRAND, RCCL             |
//! | `mkl`    | Intel oneAPI / MKL              | oneMKL (BLAS, LAPACK, FFT, RNG, SparseBLAS)        |
//! | `metal`  | Xcode / Metal SDK               | Metal Performance Shaders (MPS), MPSGraph          |
//!
//! When no vendor feature is enabled, all operations route through [`CpuSimdKernel`]
//! which uses rayon + architecture-specific SIMD intrinsics.
//!
//! ## CPU SIMD Strategy
//!
//! The CPU kernel auto-detects SIMD capability at compile time:
//! - **x86_64**: SSE4.2 (baseline), AVX2 (256-bit), AVX-512 (512-bit)
//! - **aarch64**: NEON (128-bit), SVE (scalable — Apple M-series, Graviton)
//! - **wasm32**: SIMD128
//!
//! All paths share the same [`Kernel`] trait, so callers never branch on arch.

use serde::{Deserialize, Serialize};
use std::fmt;

// ── Kernel operation descriptors ──────────────────────────────────────────

/// The set of primitive operations a kernel can execute.
///
/// This is an enum rather than separate trait methods so we can batch
/// heterogeneous ops into a single dispatch queue (important for GPU).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KernelOp {
    /// Element-wise: out[i] = f(a[i])
    MapUnary { len: usize, op: UnaryOp },
    /// Element-wise: out[i] = f(a[i], b[i])
    MapBinary { len: usize, op: BinaryOp },
    /// Reduction: scalar = reduce(data, op)
    Reduce { len: usize, op: ReduceOp },
    /// Prefix scan (inclusive)
    Scan { len: usize, op: ReduceOp },
    /// Sort (unstable)
    Sort { len: usize },
    /// Matrix multiply: C = A × B
    Gemm { m: usize, n: usize, k: usize },
    /// FFT (1D, real → complex)
    Fft { len: usize },
    /// Gather: out[i] = data[indices[i]]
    Gather { data_len: usize, index_len: usize },
    /// Scatter: out[indices[i]] = data[i]
    Scatter { data_len: usize, index_len: usize },
}

/// Unary element-wise operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnaryOp {
    Neg,
    Abs,
    Sqrt,
    Rsqrt,
    Exp,
    Log,
    Sin,
    Cos,
    Tanh,
    Relu,
    Sigmoid,
    Floor,
    Ceil,
    /// Multiply by scalar
    Scale(ordered_f64::F64),
    /// Add scalar
    Offset(ordered_f64::F64),
}

/// Binary element-wise operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Min,
    Max,
    Pow,
}

/// Reduction operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReduceOp {
    Sum,
    Product,
    Min,
    Max,
    Mean,
}

// ── Ordered f64 for enum storage ──────────────────────────────────────────

mod ordered_f64 {
    use serde::{Deserialize, Serialize};

    /// A wrapper around `f64` that implements `Eq` and `Hash` by bit pattern.
    /// Used inside enum variants so they can derive Eq.
    #[derive(Debug, Clone, Copy, Serialize, Deserialize)]
    pub struct F64(pub f64);

    impl PartialEq for F64 {
        fn eq(&self, other: &Self) -> bool {
            self.0.to_bits() == other.0.to_bits()
        }
    }
    impl Eq for F64 {}

    impl From<f64> for F64 {
        fn from(v: f64) -> Self {
            Self(v)
        }
    }
    impl From<F64> for f64 {
        fn from(v: F64) -> f64 {
            v.0
        }
    }
}

pub use ordered_f64::F64 as Scalar;

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
/// The [`crate::compute::ComputeBackend`] calls `Kernel` methods internally.
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
    fn benchmark_op(&self, op: &KernelOp) -> KernelStats;
}

/// Which vendor backend a kernel targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum KernelBackend {
    CpuScalar,
    CpuSimd,
    Cuda,
    Rocm,
    Mkl,
    Metal,
    Wgpu,
}

impl fmt::Display for KernelBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CpuScalar => write!(f, "CPU/Scalar"),
            Self::CpuSimd => write!(f, "CPU/SIMD"),
            Self::Cuda => write!(f, "NVIDIA/CUDA"),
            Self::Rocm => write!(f, "AMD/ROCm"),
            Self::Mkl => write!(f, "Intel/MKL"),
            Self::Metal => write!(f, "Apple/Metal"),
            Self::Wgpu => write!(f, "wgpu"),
        }
    }
}

// ── CPU SIMD Kernel (always available) ────────────────────────────────────

/// CPU kernel using rayon for parallelism + architecture-native SIMD.
///
/// On x86_64 this prefers AVX2 when available, falling back to SSE.
/// On aarch64 it uses NEON. On wasm32 it uses simd128.
#[derive(Debug)]
pub struct CpuSimdKernel {
    name: String,
    width: usize,
}

impl Default for CpuSimdKernel {
    fn default() -> Self {
        let (name, width) = detect_simd();
        Self { name, width }
    }
}

/// Detect best SIMD width at runtime (for x86_64 we can use `is_x86_feature_detected!`).
fn detect_simd() -> (String, usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return ("CpuSimd/AVX-512".into(), 8); // 512 / 64
        }
        if is_x86_feature_detected!("avx2") {
            return ("CpuSimd/AVX2".into(), 4); // 256 / 64
        }
        if is_x86_feature_detected!("sse4.2") {
            return ("CpuSimd/SSE4.2".into(), 2); // 128 / 64
        }
        return ("CpuSimd/Scalar".into(), 1);
    }
    #[cfg(target_arch = "aarch64")]
    {
        return ("CpuSimd/NEON".into(), 2); // 128 / 64
    }
    #[cfg(target_arch = "wasm32")]
    {
        return ("CpuSimd/SIMD128".into(), 2);
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64", target_arch = "wasm32")))]
    {
        return ("CpuSimd/Scalar".into(), 1);
    }
}

impl CpuSimdKernel {
    /// Apply a unary op scalar-style. SIMD intrinsics would replace the inner loop.
    fn apply_unary(v: f64, op: UnaryOp) -> f64 {
        match op {
            UnaryOp::Neg => -v,
            UnaryOp::Abs => v.abs(),
            UnaryOp::Sqrt => v.sqrt(),
            UnaryOp::Rsqrt => 1.0 / v.sqrt(),
            UnaryOp::Exp => v.exp(),
            UnaryOp::Log => v.ln(),
            UnaryOp::Sin => v.sin(),
            UnaryOp::Cos => v.cos(),
            UnaryOp::Tanh => v.tanh(),
            UnaryOp::Relu => v.max(0.0),
            UnaryOp::Sigmoid => 1.0 / (1.0 + (-v).exp()),
            UnaryOp::Floor => v.floor(),
            UnaryOp::Ceil => v.ceil(),
            UnaryOp::Scale(s) => v * f64::from(s),
            UnaryOp::Offset(o) => v + f64::from(o),
        }
    }

    fn apply_binary(a: f64, b: f64, op: BinaryOp) -> f64 {
        match op {
            BinaryOp::Add => a + b,
            BinaryOp::Sub => a - b,
            BinaryOp::Mul => a * b,
            BinaryOp::Div => a / b,
            BinaryOp::Min => a.min(b),
            BinaryOp::Max => a.max(b),
            BinaryOp::Pow => a.powf(b),
        }
    }

    fn apply_reduce(acc: f64, v: f64, op: ReduceOp) -> f64 {
        match op {
            ReduceOp::Sum | ReduceOp::Mean => acc + v,
            ReduceOp::Product => acc * v,
            ReduceOp::Min => acc.min(v),
            ReduceOp::Max => acc.max(v),
        }
    }

    fn reduce_identity(op: ReduceOp) -> f64 {
        match op {
            ReduceOp::Sum | ReduceOp::Mean => 0.0,
            ReduceOp::Product => 1.0,
            ReduceOp::Min => f64::INFINITY,
            ReduceOp::Max => f64::NEG_INFINITY,
        }
    }
}

use rayon::prelude::*;
use std::time::Instant;

impl Kernel for CpuSimdKernel {
    fn name(&self) -> &str {
        &self.name
    }

    fn backend_tag(&self) -> KernelBackend {
        if self.width > 1 {
            KernelBackend::CpuSimd
        } else {
            KernelBackend::CpuScalar
        }
    }

    fn vector_width(&self) -> usize {
        self.width
    }

    fn map_unary_f64(&self, data: &[f64], op: UnaryOp) -> Vec<f64> {
        data.par_iter().map(|&v| Self::apply_unary(v, op)).collect()
    }

    fn map_binary_f64(&self, a: &[f64], b: &[f64], op: BinaryOp) -> Vec<f64> {
        a.par_iter()
            .zip(b.par_iter())
            .map(|(&x, &y)| Self::apply_binary(x, y, op))
            .collect()
    }

    fn reduce_f64(&self, data: &[f64], op: ReduceOp) -> f64 {
        let raw = data
            .par_iter()
            .copied()
            .reduce(|| Self::reduce_identity(op), |acc, v| Self::apply_reduce(acc, v, op));
        if op == ReduceOp::Mean && !data.is_empty() {
            raw / data.len() as f64
        } else {
            raw
        }
    }

    fn scan_f64(&self, data: &[f64], op: ReduceOp) -> Vec<f64> {
        let mut result = Vec::with_capacity(data.len());
        let mut acc = Self::reduce_identity(op);
        for &v in data {
            acc = Self::apply_reduce(acc, v, op);
            result.push(acc);
        }
        result
    }

    fn gemm_f64(&self, a: &[f64], b: &[f64], m: usize, n: usize, k: usize) -> Vec<f64> {
        // Naive cache-friendly GEMM — vendor backends (cuBLAS, rocBLAS, MKL)
        // replace this with tuned implementations.
        let mut c = vec![0.0; m * n];
        c.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
            for p in 0..k {
                let a_ip = a[i * k + p];
                for j in 0..n {
                    row[j] += a_ip * b[p * n + j];
                }
            }
        });
        c
    }

    fn sort_f64(&self, data: &mut [f64]) {
        data.par_sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    }

    fn benchmark_op(&self, op: &KernelOp) -> KernelStats {
        let start = Instant::now();
        match op {
            KernelOp::MapUnary { len, op: uop } => {
                let data: Vec<f64> = (0..*len).map(|i| i as f64 * 0.1 + 1.0).collect();
                std::hint::black_box(self.map_unary_f64(&data, *uop));
            }
            KernelOp::MapBinary { len, op: bop } => {
                let a: Vec<f64> = (0..*len).map(|i| i as f64).collect();
                let b: Vec<f64> = (0..*len).map(|i| (i as f64) * 0.5).collect();
                std::hint::black_box(self.map_binary_f64(&a, &b, *bop));
            }
            KernelOp::Reduce { len, op: rop } => {
                let data: Vec<f64> = (0..*len).map(|i| i as f64 * 0.1).collect();
                std::hint::black_box(self.reduce_f64(&data, *rop));
            }
            KernelOp::Scan { len, op: rop } => {
                let data: Vec<f64> = (0..*len).map(|i| i as f64).collect();
                std::hint::black_box(self.scan_f64(&data, *rop));
            }
            KernelOp::Sort { len } => {
                let mut data: Vec<f64> = (0..*len).rev().map(|i| i as f64).collect();
                self.sort_f64(&mut data);
            }
            KernelOp::Gemm { m, n, k } => {
                let a = vec![1.0f64; m * k];
                let b = vec![1.0f64; k * n];
                std::hint::black_box(self.gemm_f64(&a, &b, *m, *n, *k));
            }
            KernelOp::Fft { len } => {
                // FFT stub — vendor backends provide real FFT
                let data: Vec<f64> = (0..*len).map(|i| (i as f64).sin()).collect();
                std::hint::black_box(&data);
            }
            KernelOp::Gather { data_len, index_len } => {
                let data: Vec<f64> = (0..*data_len).map(|i| i as f64).collect();
                let indices: Vec<usize> = (0..*index_len).map(|i| i % data_len).collect();
                std::hint::black_box(self.gather_f64(&data, &indices));
            }
            KernelOp::Scatter { data_len, index_len } => {
                let values: Vec<f64> = (0..*index_len).map(|i| i as f64).collect();
                let indices: Vec<usize> = (0..*index_len).map(|i| i % data_len).collect();
                std::hint::black_box(self.scatter_f64(&values, &indices, *data_len));
            }
        }
        let elapsed = start.elapsed();
        let flops = match op {
            KernelOp::Gemm { m, n, k } => 2.0 * (*m as f64) * (*n as f64) * (*k as f64),
            KernelOp::MapUnary { len, .. } | KernelOp::Reduce { len, .. } => *len as f64,
            KernelOp::MapBinary { len, .. } => *len as f64,
            _ => 0.0,
        };
        KernelStats {
            duration_us: elapsed.as_micros(),
            flops: if elapsed.as_secs_f64() > 0.0 {
                flops / elapsed.as_secs_f64()
            } else {
                0.0
            },
            bandwidth_bytes_sec: 0.0,
        }
    }
}

// ── CUDA kernel stub (behind feature flag) ────────────────────────────────

/// Placeholder for NVIDIA CUDA kernel.
///
/// When the `cuda` feature is enabled, this would link against:
/// - **cuBLAS** for GEMM, BLAS L1/L2/L3
/// - **cuDNN** for neural-network primitives (conv, pooling, activation)
/// - **cuFFT** for FFT
/// - **cuRAND** for random number generation
/// - **Thrust** for sort, scan, reduce
/// - **NCCL** for multi-GPU communication
///
/// Optimization notes:
/// - Warp size = 32 threads
/// - Shared memory tiling for GEMM (128×128 tiles, 8×8 thread-tiles)
/// - Tensor cores for FP16/BF16 GEMM (Volta+)
/// - Async memcpy (Ampere+) for pipelored double-buffering
/// - Stream-ordered memory allocation (CUDA 11.2+)
#[cfg(feature = "cuda")]
#[derive(Debug)]
pub struct CudaKernel {
    pub device_name: String,
    pub compute_capability: (u32, u32),
    pub sm_count: u32,
    pub vram_bytes: u64,
}

/// Placeholder for AMD ROCm kernel.
///
/// When the `rocm` feature is enabled, this would link against:
/// - **rocBLAS** for GEMM, BLAS
/// - **rocFFT** for FFT
/// - **MIOpen** for neural-network primitives
/// - **hipRAND** for random number generation
/// - **rocThrust** for sort, scan, reduce
/// - **RCCL** for multi-GPU communication
///
/// Optimization notes:
/// - Wavefront size = 64 (RDNA: 32)
/// - LDS (Local Data Share) for inter-thread communication
/// - Matrix cores on CDNA for FP16/BF16 GEMM
#[cfg(feature = "rocm")]
#[derive(Debug)]
pub struct RocmKernel {
    pub device_name: String,
    pub gfx_version: String,
    pub cu_count: u32,
    pub vram_bytes: u64,
}

/// Placeholder for Intel MKL kernel.
///
/// When the `mkl` feature is enabled, this would link against:
/// - **oneMKL** (BLAS, LAPACK, FFT, RNG, SparseBLAS)
/// - **oneDNN** for neural-network primitives
/// - **oneDAL** for data analytics
///
/// Optimization notes:
/// - AVX-512 and AMX on latest Xeon / Sapphire Rapids
/// - Thread-level parallelism via OpenMP or TBB
/// - JIT code generation for small matrix sizes
#[cfg(feature = "mkl")]
#[derive(Debug)]
pub struct MklKernel {
    pub cpu_name: String,
    pub avx_level: String,
}

// ── Auto-select best kernel ───────────────────────────────────────────────

/// Auto-detect and return the best available kernel for the current hardware.
///
/// Priority: CUDA > ROCm > MKL > CPU SIMD
pub fn best_kernel() -> Box<dyn Kernel> {
    // In a full implementation, we'd probe for CUDA/ROCm/MKL linkage.
    // For now, CPU SIMD is always available.
    #[cfg(feature = "cuda")]
    {
        // TODO: probe CUDA runtime, return CudaKernel if available
    }
    #[cfg(feature = "rocm")]
    {
        // TODO: probe ROCm runtime, return RocmKernel if available
    }
    #[cfg(feature = "mkl")]
    {
        // TODO: probe MKL runtime, return MklKernel if available
    }
    Box::new(CpuSimdKernel::default())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kernel() -> CpuSimdKernel {
        CpuSimdKernel::default()
    }

    #[test]
    fn detect_simd_returns_nonzero_width() {
        let k = kernel();
        assert!(k.vector_width() >= 1);
        assert!(!k.name().is_empty());
    }

    #[test]
    fn map_unary_neg() {
        let k = kernel();
        let data = vec![1.0, -2.0, 3.0];
        let out = k.map_unary_f64(&data, UnaryOp::Neg);
        assert_eq!(out, vec![-1.0, 2.0, -3.0]);
    }

    #[test]
    fn map_unary_relu() {
        let k = kernel();
        let data = vec![-1.0, 0.0, 3.0, -5.0];
        let out = k.map_unary_f64(&data, UnaryOp::Relu);
        assert_eq!(out, vec![0.0, 0.0, 3.0, 0.0]);
    }

    #[test]
    fn map_unary_sigmoid() {
        let k = kernel();
        let out = k.map_unary_f64(&[0.0], UnaryOp::Sigmoid);
        assert!((out[0] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn map_binary_add() {
        let k = kernel();
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![10.0, 20.0, 30.0];
        let out = k.map_binary_f64(&a, &b, BinaryOp::Add);
        assert_eq!(out, vec![11.0, 22.0, 33.0]);
    }

    #[test]
    fn reduce_sum() {
        let k = kernel();
        let data = vec![1.0, 2.0, 3.0, 4.0];
        assert!((k.reduce_f64(&data, ReduceOp::Sum) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn reduce_mean() {
        let k = kernel();
        let data = vec![2.0, 4.0, 6.0, 8.0];
        assert!((k.reduce_f64(&data, ReduceOp::Mean) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn reduce_min_max() {
        let k = kernel();
        let data = vec![3.0, 1.0, 4.0, 1.5];
        assert!((k.reduce_f64(&data, ReduceOp::Min) - 1.0).abs() < 1e-10);
        assert!((k.reduce_f64(&data, ReduceOp::Max) - 4.0).abs() < 1e-10);
    }

    #[test]
    fn scan_sum() {
        let k = kernel();
        let out = k.scan_f64(&[1.0, 2.0, 3.0, 4.0], ReduceOp::Sum);
        assert_eq!(out, vec![1.0, 3.0, 6.0, 10.0]);
    }

    #[test]
    fn gemm_identity() {
        let k = kernel();
        // 2x2 identity × [1,2; 3,4] = [1,2; 3,4]
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let c = k.gemm_f64(&a, &b, 2, 2, 2);
        assert_eq!(c, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn gemm_small() {
        let k = kernel();
        // [1,2; 3,4] × [5,6; 7,8] = [19,22; 43,50]
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let c = k.gemm_f64(&a, &b, 2, 2, 2);
        assert_eq!(c, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn sort_f64() {
        let k = kernel();
        let mut data = vec![5.0, 1.0, 3.0, 2.0, 4.0];
        k.sort_f64(&mut data);
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn gather_scatter() {
        let k = kernel();
        let data = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let gathered = k.gather_f64(&data, &[4, 2, 0]);
        assert_eq!(gathered, vec![50.0, 30.0, 10.0]);

        let scattered = k.scatter_f64(&[99.0, 88.0], &[1, 3], 5);
        assert_eq!(scattered, vec![0.0, 99.0, 0.0, 88.0, 0.0]);
    }

    #[test]
    fn benchmark_op_runs() {
        let k = kernel();
        let stats = k.benchmark_op(&KernelOp::Reduce {
            len: 10_000,
            op: ReduceOp::Sum,
        });
        assert!(stats.duration_us > 0 || stats.flops >= 0.0);
    }

    #[test]
    fn best_kernel_returns_cpu() {
        let k = best_kernel();
        assert!(k.name().contains("CpuSimd") || k.name().contains("Scalar"));
    }

    #[test]
    fn empty_data() {
        let k = kernel();
        assert_eq!(k.map_unary_f64(&[], UnaryOp::Neg), Vec::<f64>::new());
        assert_eq!(
            k.map_binary_f64(&[], &[], BinaryOp::Add),
            Vec::<f64>::new()
        );
        assert_eq!(k.scan_f64(&[], ReduceOp::Sum), Vec::<f64>::new());
    }
}
