use rayon::prelude::*;
use std::time::Instant;

use super::{BinaryOp, Kernel, KernelBackend, KernelOp, KernelStats, ReduceOp, UnaryOp};

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
    #[cfg(not(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "wasm32"
    )))]
    {
        return ("CpuSimd/Scalar".into(), 1);
    }
}

impl CpuSimdKernel {
    /// Apply a unary op scalar-style. Delegates to `UnaryOp::apply`.
    fn apply_unary(v: f64, op: UnaryOp) -> f64 {
        op.apply(v)
    }

    pub(super) fn apply_binary(a: f64, b: f64, op: BinaryOp) -> f64 {
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

    pub(super) fn apply_reduce(acc: f64, v: f64, op: ReduceOp) -> f64 {
        match op {
            ReduceOp::Sum | ReduceOp::Mean => acc + v,
            ReduceOp::Product => acc * v,
            ReduceOp::Min => acc.min(v),
            ReduceOp::Max => acc.max(v),
        }
    }

    pub(super) fn reduce_identity(op: ReduceOp) -> f64 {
        match op {
            ReduceOp::Sum | ReduceOp::Mean => 0.0,
            ReduceOp::Product => 1.0,
            ReduceOp::Min => f64::INFINITY,
            ReduceOp::Max => f64::NEG_INFINITY,
        }
    }
}

// ── x86_64 AVX2 SIMD implementations ─────────────────────────────────────
//
// These process 4× f64 per instruction (256-bit lanes).
// Activated at runtime via `is_x86_feature_detected!`.
// Rayon splits work across cores; each core's chunk runs the SIMD inner loop.

#[cfg(target_arch = "x86_64")]
mod simd_avx2 {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    use super::{BinaryOp, ReduceOp};

    /// 4-wide f64 binary op — caller must ensure AVX2 is available.
    ///
    /// # Safety
    /// Requires AVX2 (checked by caller via `is_x86_feature_detected!`).
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn binary_f64_avx2(a: &[f64], b: &[f64], op: BinaryOp) -> Vec<f64> {
        unsafe {
            debug_assert_eq!(a.len(), b.len());
            let n = a.len();
            let mut out = vec![0.0f64; n];
            let chunks = n / 4;
            let ap = a.as_ptr();
            let bp = b.as_ptr();
            let op_ = out.as_mut_ptr();
            for i in 0..chunks {
                let off = i * 4;
                let va = _mm256_loadu_pd(ap.add(off));
                let vb = _mm256_loadu_pd(bp.add(off));
                let vr = match op {
                    BinaryOp::Add => _mm256_add_pd(va, vb),
                    BinaryOp::Sub => _mm256_sub_pd(va, vb),
                    BinaryOp::Mul => _mm256_mul_pd(va, vb),
                    BinaryOp::Div => _mm256_div_pd(va, vb),
                    BinaryOp::Min => _mm256_min_pd(va, vb),
                    BinaryOp::Max => _mm256_max_pd(va, vb),
                    BinaryOp::Pow => {
                        let mut tmp = [0.0f64; 4];
                        _mm256_storeu_pd(tmp.as_mut_ptr(), va);
                        let mut tb = [0.0f64; 4];
                        _mm256_storeu_pd(tb.as_mut_ptr(), vb);
                        for j in 0..4 {
                            tmp[j] = tmp[j].powf(tb[j]);
                        }
                        _mm256_loadu_pd(tmp.as_ptr())
                    }
                };
                _mm256_storeu_pd(op_.add(off), vr);
            }
            // Scalar tail for remaining elements
            for i in (chunks * 4)..n {
                out[i] = super::CpuSimdKernel::apply_binary(a[i], b[i], op);
            }
            out
        }
    }

    /// 4-wide f64 horizontal reduction — processes data in 256-bit chunks.
    ///
    /// # Safety
    /// Requires AVX2 (checked by caller).
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn reduce_f64_avx2(data: &[f64], op: ReduceOp) -> f64 {
        unsafe {
            let n = data.len();
            if n == 0 {
                return super::CpuSimdKernel::reduce_identity(op);
            }
            let chunks = n / 4;
            let dp = data.as_ptr();

            let identity = super::CpuSimdKernel::reduce_identity(op);
            let mut acc = _mm256_set1_pd(identity);

            for i in 0..chunks {
                let v = _mm256_loadu_pd(dp.add(i * 4));
                acc = match op {
                    ReduceOp::Sum | ReduceOp::Mean => _mm256_add_pd(acc, v),
                    ReduceOp::Product => _mm256_mul_pd(acc, v),
                    ReduceOp::Min => _mm256_min_pd(acc, v),
                    ReduceOp::Max => _mm256_max_pd(acc, v),
                };
            }

            // Horizontal collapse: 4 lanes → 1 scalar
            let mut lanes = [0.0f64; 4];
            _mm256_storeu_pd(lanes.as_mut_ptr(), acc);
            let mut scalar = lanes[0];
            for &lane in &lanes[1..] {
                scalar = super::CpuSimdKernel::apply_reduce(scalar, lane, op);
            }

            // Scalar tail
            for i in (chunks * 4)..n {
                scalar = super::CpuSimdKernel::apply_reduce(scalar, data[i], op);
            }
            scalar
        }
    }
}

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
        #[cfg(target_arch = "x86_64")]
        if self.width >= 4 {
            let chunk_size = (a.len() / rayon::current_num_threads()).max(1024);
            let mut out = vec![0.0f64; a.len()];
            out.par_chunks_mut(chunk_size)
                .enumerate()
                .for_each(|(ci, chunk)| {
                    let start = ci * chunk_size;
                    let end = (start + chunk.len()).min(a.len());
                    let a_slice = &a[start..end];
                    let b_slice = &b[start..end];
                    // SAFETY: we checked `self.width >= 4` which means
                    // AVX2 was detected at construction time.
                    let result = unsafe { simd_avx2::binary_f64_avx2(a_slice, b_slice, op) };
                    chunk.copy_from_slice(&result);
                });
            return out;
        }
        // Scalar fallback (SSE / NEON / wasm / no SIMD)
        a.par_iter()
            .zip(b.par_iter())
            .map(|(&x, &y)| Self::apply_binary(x, y, op))
            .collect()
    }

    fn reduce_f64(&self, data: &[f64], op: ReduceOp) -> f64 {
        #[cfg(target_arch = "x86_64")]
        if self.width >= 4 {
            let chunk_size = (data.len() / rayon::current_num_threads()).max(1024);
            let raw: f64 = data
                .par_chunks(chunk_size)
                .map(|chunk| {
                    // SAFETY: AVX2 detected at construction.
                    unsafe { simd_avx2::reduce_f64_avx2(chunk, op) }
                })
                .reduce(
                    || Self::reduce_identity(op),
                    |a, b| Self::apply_reduce(a, b, op),
                );
            return if op == ReduceOp::Mean && !data.is_empty() {
                raw / data.len() as f64
            } else {
                raw
            };
        }
        let raw = data.par_iter().copied().reduce(
            || Self::reduce_identity(op),
            |acc, v| Self::apply_reduce(acc, v, op),
        );
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
        data.par_sort_unstable_by(crate::f64_cmp);
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
                let data: Vec<f64> = (0..*len).map(|i| (i as f64).sin()).collect();
                std::hint::black_box(&data);
            }
            KernelOp::Gather {
                data_len,
                index_len,
            } => {
                let data: Vec<f64> = (0..*data_len).map(|i| i as f64).collect();
                let indices: Vec<usize> = (0..*index_len).map(|i| i % data_len).collect();
                std::hint::black_box(self.gather_f64(&data, &indices));
            }
            KernelOp::Scatter {
                data_len,
                index_len,
            } => {
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

// ── Vendor kernel stubs (behind feature flags) ────────────────────────────

/// Placeholder for NVIDIA CUDA kernel.
#[cfg(feature = "cuda")]
#[derive(Debug)]
pub struct CudaKernel {
    pub device_name: String,
    pub compute_capability: (u32, u32),
    pub sm_count: u32,
    pub vram_bytes: u64,
}

/// Placeholder for AMD ROCm kernel.
#[cfg(feature = "rocm")]
#[derive(Debug)]
pub struct RocmKernel {
    pub device_name: String,
    pub gfx_version: String,
    pub cu_count: u32,
    pub vram_bytes: u64,
}

/// Placeholder for Intel MKL kernel.
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
