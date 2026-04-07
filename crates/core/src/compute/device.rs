use crate::hints::Hints;
use crate::kernel::{
    BinaryOp, CpuSimdKernel, Kernel, KernelBackend, KernelOp, KernelStats, ReduceOp, UnaryOp,
};
use rayon::prelude::*;
use std::fmt;
use std::sync::{Arc, LazyLock};

use super::DeviceProfile;

/// Identifies which hardware a backend targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendKind {
    Cpu,
    /// Cross-platform GPU via wgpu (Vulkan / Metal / DX12 / WebGPU).
    Wgpu,
    /// NVIDIA-specific (CUDA). Behind feature flag.
    Cuda,
    /// AMD-specific (ROCm / HIP). Behind feature flag.
    Rocm,
}

/// Info about the physical device a backend runs on.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub kind: BackendKind,
    pub name: String,
    /// Available memory in bytes (0 if unknown / CPU).
    pub memory_bytes: u64,
    /// Max parallelism (CPU cores, GPU compute units, etc.).
    pub max_parallelism: u32,
}

// ── Device ───────────────────────────────────────────────────────────────

/// Global best device — initialized once at first use.
static BEST_DEVICE: LazyLock<Device> = LazyLock::new(Device::cpu);

/// The universal hardware dispatch handle.
///
/// One concrete type for all backends. No traits to call, no dynamic dispatch
/// overhead for callers. Internally holds a [`Kernel`] for SIMD/GPU ops plus
/// parallel dispatch logic.
///
/// `Device` is `Clone + Send + Sync` — sharing across threads is cheap (Arc).
#[derive(Clone)]
pub struct Device(Arc<DeviceInner>);

struct DeviceInner {
    info: DeviceInfo,
    kernel: Box<dyn Kernel>,
    throttle: Option<DeviceProfile>,
}

impl fmt::Debug for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Device({})", self.name())
    }
}

impl Device {
    /// CPU device with best available SIMD (AVX2, NEON, SIMD128, …).
    pub fn cpu() -> Self {
        let kernel = CpuSimdKernel::default();
        let name = format!("CPU ({})", kernel.name());
        let info = DeviceInfo {
            kind: BackendKind::Cpu,
            name,
            memory_bytes: 0,
            max_parallelism: rayon::current_num_threads() as u32,
        };
        Self(Arc::new(DeviceInner {
            info,
            kernel: Box::new(kernel),
            throttle: None,
        }))
    }

    /// Auto-select the best available device.
    ///
    /// Currently returns CPU; when GPU feature flags are enabled this will
    /// probe available hardware and pick the fastest.
    pub fn best() -> Self {
        BEST_DEVICE.clone()
    }

    /// Simulated device — throttles CPU to mimic constrained hardware.
    ///
    /// Useful for benchmarking optimization strategies without physical devices.
    pub fn simulated(profile: DeviceProfile) -> Self {
        let kernel = CpuSimdKernel::default();
        let info = DeviceInfo {
            kind: BackendKind::Cpu,
            name: format!("Simulated: {}", profile.name),
            memory_bytes: 0,
            max_parallelism: profile.cores,
        };
        Self(Arc::new(DeviceInner {
            info,
            kernel: Box::new(kernel),
            throttle: Some(profile),
        }))
    }

    /// Create a device from any [`Kernel`] implementation.
    ///
    /// Power-user escape hatch for custom backends (CUDA, ROCm, etc.).
    pub fn from_kernel(kernel: Box<dyn Kernel>) -> Self {
        let kind: BackendKind = kernel.backend_tag().into();
        let info = DeviceInfo {
            kind,
            name: kernel.name().to_string(),
            memory_bytes: 0,
            max_parallelism: 1,
        };
        Self(Arc::new(DeviceInner {
            info,
            kernel,
            throttle: None,
        }))
    }

    // ── Device info ──────────────────────────────────────────────────────

    pub fn kind(&self) -> BackendKind {
        self.0.info.kind
    }
    pub fn name(&self) -> &str {
        &self.0.info.name
    }
    pub fn info(&self) -> &DeviceInfo {
        &self.0.info
    }
    pub fn vector_width(&self) -> usize {
        self.0.kernel.vector_width()
    }

    // ── High-level parallel ops (was ComputeBackend) ─────────────────────

    /// Parallel map over a slice with an arbitrary function.
    pub fn map(&self, data: &[f64], f: fn(f64) -> f64) -> Vec<f64> {
        self.maybe_throttle(data.len());
        data.par_iter().map(|&v| f(v)).collect()
    }

    /// Parallel filter — returns indices of matching elements.
    pub fn filter(&self, data: &[f64], pred: fn(f64) -> bool) -> Vec<usize> {
        self.maybe_throttle(data.len());
        data.par_iter()
            .enumerate()
            .filter_map(|(i, &v)| pred(v).then_some(i))
            .collect()
    }

    /// Parallel sort (unstable, ascending).
    pub fn sort(&self, data: &mut [f64]) {
        self.maybe_throttle(data.len());
        self.0.kernel.sort_f64(data);
    }

    /// Parallel reduction (sum).
    pub fn sum(&self, data: &[f64]) -> f64 {
        self.maybe_throttle(data.len());
        data.par_iter().sum()
    }

    /// Parallel prefix sum (inclusive scan).
    pub fn prefix_sum(&self, data: &[f64]) -> Vec<f64> {
        self.maybe_throttle(data.len());
        let mut result = Vec::with_capacity(data.len());
        let mut acc = 0.0;
        for &v in data {
            acc += v;
            result.push(acc);
        }
        result
    }

    /// Hint-aware map — sequential for small data, parallel otherwise.
    pub fn map_hinted(&self, data: &[f64], f: fn(f64) -> f64, hints: &Hints) -> Vec<f64> {
        self.maybe_throttle(data.len());
        if data.len() < hints.parallelism_threshold() {
            data.iter().map(|&v| f(v)).collect()
        } else {
            data.par_iter().map(|&v| f(v)).collect()
        }
    }

    /// Hint-aware sum.
    pub fn sum_hinted(&self, data: &[f64], hints: &Hints) -> f64 {
        self.maybe_throttle(data.len());
        if data.len() < hints.parallelism_threshold() {
            data.iter().sum()
        } else {
            data.par_iter().sum()
        }
    }

    // ── Low-level kernel ops (was dyn Kernel) ────────────────────────────

    /// Element-wise unary: out[i] = op(data[i]) — SIMD-accelerated.
    pub fn unary(&self, data: &[f64], op: UnaryOp) -> Vec<f64> {
        self.maybe_throttle(data.len());
        self.0.kernel.map_unary_f64(data, op)
    }

    /// Fused unary chain: apply a sequence of ops in a single pass.
    pub fn fused_unary(&self, data: &[f64], ops: &[UnaryOp]) -> Vec<f64> {
        self.maybe_throttle(data.len());
        let mut buf = data.to_vec();
        for &op in ops {
            buf = self.0.kernel.map_unary_f64(&buf, op);
        }
        buf
    }

    /// Element-wise binary: out[i] = op(a[i], b[i]) — SIMD-accelerated.
    pub fn binary(&self, a: &[f64], b: &[f64], op: BinaryOp) -> Vec<f64> {
        self.maybe_throttle(a.len());
        self.0.kernel.map_binary_f64(a, b, op)
    }

    /// Reduction to scalar — SIMD-accelerated.
    pub fn reduce(&self, data: &[f64], op: ReduceOp) -> f64 {
        self.maybe_throttle(data.len());
        self.0.kernel.reduce_f64(data, op)
    }

    /// Inclusive prefix scan.
    pub fn scan(&self, data: &[f64], op: ReduceOp) -> Vec<f64> {
        self.maybe_throttle(data.len());
        self.0.kernel.scan_f64(data, op)
    }

    /// Matrix multiply: C[m×n] = A[m×k] × B[k×n] (row-major).
    pub fn gemm(&self, a: &[f64], b: &[f64], m: usize, n: usize, k: usize) -> Vec<f64> {
        self.maybe_throttle(m * n);
        self.0.kernel.gemm_f64(a, b, m, n, k)
    }

    /// Gather: out[i] = data[indices[i]].
    pub fn gather(&self, data: &[f64], indices: &[usize]) -> Vec<f64> {
        self.0.kernel.gather_f64(data, indices)
    }

    /// Scatter: out[indices[i]] = values[i].
    pub fn scatter(&self, values: &[f64], indices: &[usize], out_len: usize) -> Vec<f64> {
        self.0.kernel.scatter_f64(values, indices, out_len)
    }

    /// Benchmark a kernel operation on this device.
    pub fn benchmark_op(&self, op: &KernelOp) -> KernelStats {
        self.0.kernel.benchmark_op(op)
    }

    /// The underlying kernel backend tag.
    pub fn kernel_backend(&self) -> KernelBackend {
        self.0.kernel.backend_tag()
    }

    // ── Internal ─────────────────────────────────────────────────────────

    /// Simulate slower hardware by burning cycles proportional to the throttle factor.
    fn maybe_throttle(&self, n: usize) {
        if let Some(ref profile) = self.0.throttle {
            let extra = ((1.0 / profile.compute_factor - 1.0) * n as f64) as usize;
            let mut _sink = 0u64;
            for i in 0..extra.min(n * 10) {
                _sink = _sink.wrapping_add(i as u64);
            }
            std::hint::black_box(_sink);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ── Ops trait impls ─────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

impl crate::ops::Summary for Device {
    fn summary(&self) -> String {
        let info = self.info();
        format!(
            "Device({}, backend={:?}, parallelism={})",
            info.name, info.kind, info.max_parallelism
        )
    }
}
