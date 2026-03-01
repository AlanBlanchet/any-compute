//! Compute backend abstraction — CPU vs GPU dispatch for data transforms.
//!
//! The trait [`ComputeBackend`] defines operations that can run on any hardware.
//! Core ships a [`CpuBackend`] using rayon. GPU backends (wgpu, CUDA)
//! implement the same trait without touching core's dependencies.
//!
//! **All parallel work must go through `ComputeBackend`** — never call rayon
//! directly outside the CPU backend impl. This is the single source of truth
//! for dispatching parallel computation.
//!
//! ## GPU architecture notes
//!
//! | Vendor   | API stack              | Notes                             |
//! |----------|------------------------|-----------------------------------|
//! | NVIDIA   | Vulkan / CUDA / OptiX  | Best compute via CUDA, Vulkan for graphics |
//! | AMD      | Vulkan / ROCm / HIP    | ROCm mirrors CUDA API surface     |
//! | Intel    | Vulkan / oneAPI / SYCL  | Arc GPUs, integrated graphics     |
//! | Apple    | Metal / MPS            | Through wgpu's Metal backend      |
//! | Web      | WebGPU                 | Through wgpu's web backend        |
//!
//! **wgpu** is the recommended cross-platform backend: one implementation that covers
//! Vulkan, Metal, DX12, and WebGPU. Vendor-specific backends (CUDA, ROCm) can be added
//! behind feature flags for workloads where they outperform the generic path.
//!
//! ### Optimization strategies by vendor
//!
//! - **NVIDIA**: Prefer warp-level primitives (warp size = 32), shared memory tiling,
//!   tensor cores for f16 ops, async copy for double-buffering.
//! - **AMD**: Wavefront size = 64 (RDNA: 32), maximize occupancy via register pressure
//!   management, LDS (Local Data Share) for inter-thread communication.
//! - **Intel**: Subgroup size varies (8/16/32), use subgroup operations, EU threading
//!   model favors wider dispatch. Integrated GPUs share memory with CPU — exploit zero-copy.
//! - **CPU**: Rayon parallel iterators, SIMD via `std::simd` or `packed_simd`,
//!   cache-line-aware data layout, avoid false sharing.

use crate::hints::Hints;
use rayon::prelude::*;

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

/// Compute backend trait — any operation that can be hardware-accelerated.
///
/// All methods take `&self` so backends can be shared across threads.
/// Internal state (command buffers, etc.) uses interior mutability.
///
/// Methods accept optional [`Hints`] to auto-tune parallelism thresholds,
/// batch sizes, and memory strategies per-call.
pub trait ComputeBackend: Send + Sync {
    /// What hardware does this backend target?
    fn device_info(&self) -> &DeviceInfo;

    /// Parallel `map` over a slice, producing a new vec.
    fn map_f64(&self, data: &[f64], f: fn(f64) -> f64) -> Vec<f64>;

    /// Parallel `filter` — returns indices of elements matching the predicate.
    fn filter_indices(&self, data: &[f64], pred: fn(f64) -> bool) -> Vec<usize>;

    /// Parallel sort (unstable for performance).
    fn sort_f64(&self, data: &mut [f64]);

    /// Parallel reduction (sum).
    fn sum_f64(&self, data: &[f64]) -> f64;

    /// Parallel prefix sum (inclusive scan).
    fn prefix_sum_f64(&self, data: &[f64]) -> Vec<f64>;

    /// Hint-aware map — falls back to sequential if data size is below
    /// the parallelism threshold indicated by hints.
    fn map_f64_hinted(&self, data: &[f64], f: fn(f64) -> f64, hints: &Hints) -> Vec<f64> {
        if data.len() < hints.parallelism_threshold() {
            data.iter().map(|&v| f(v)).collect()
        } else {
            self.map_f64(data, f)
        }
    }

    /// Hint-aware sum.
    fn sum_f64_hinted(&self, data: &[f64], hints: &Hints) -> f64 {
        if data.len() < hints.parallelism_threshold() {
            data.iter().sum()
        } else {
            self.sum_f64(data)
        }
    }
}

/// CPU backend — uses rayon for parallelism, zero additional dependencies.
#[derive(Debug)]
pub struct CpuBackend {
    info: DeviceInfo,
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self {
            info: DeviceInfo {
                kind: BackendKind::Cpu,
                name: "CPU (rayon)".into(),
                memory_bytes: 0,
                max_parallelism: rayon::current_num_threads() as u32,
            },
        }
    }
}

impl ComputeBackend for CpuBackend {
    fn device_info(&self) -> &DeviceInfo {
        &self.info
    }

    fn map_f64(&self, data: &[f64], f: fn(f64) -> f64) -> Vec<f64> {
        data.par_iter().map(|&v| f(v)).collect()
    }

    fn filter_indices(&self, data: &[f64], pred: fn(f64) -> bool) -> Vec<usize> {
        data.par_iter()
            .enumerate()
            .filter_map(|(i, &v)| pred(v).then_some(i))
            .collect()
    }

    fn sort_f64(&self, data: &mut [f64]) {
        data.par_sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    }

    fn sum_f64(&self, data: &[f64]) -> f64 {
        data.par_iter().sum()
    }

    fn prefix_sum_f64(&self, data: &[f64]) -> Vec<f64> {
        // Sequential prefix sum — parallel version requires work-efficient scan
        let mut result = Vec::with_capacity(data.len());
        let mut acc = 0.0;
        for &v in data {
            acc += v;
            result.push(acc);
        }
        result
    }
}

/// Simulated device profile for benchmarking on different hardware characteristics.
///
/// Throttles the CpuBackend to mimic constrained devices (mobile, low-end GPU, etc.).
/// This lets us test optimization strategies without physical hardware.
#[derive(Debug)]
pub struct SimulatedBackend {
    inner: CpuBackend,
    profile: DeviceProfile,
    info: DeviceInfo,
}

/// Hardware profile for simulation.
#[derive(Debug, Clone)]
pub struct DeviceProfile {
    pub name: &'static str,
    /// Simulated core count — throttles rayon thread pool.
    pub cores: u32,
    /// Simulated memory bandwidth factor (1.0 = native, 0.1 = 10x slower).
    pub bandwidth_factor: f64,
    /// Simulated compute throughput factor.
    pub compute_factor: f64,
}

impl DeviceProfile {
    pub const HIGH_END_DESKTOP: Self = Self {
        name: "High-end Desktop (16 cores)",
        cores: 16,
        bandwidth_factor: 1.0,
        compute_factor: 1.0,
    };

    pub const MID_RANGE_LAPTOP: Self = Self {
        name: "Mid-range Laptop (4 cores)",
        cores: 4,
        bandwidth_factor: 0.6,
        compute_factor: 0.5,
    };

    pub const LOW_END_MOBILE: Self = Self {
        name: "Low-end Mobile (2 cores)",
        cores: 2,
        bandwidth_factor: 0.2,
        compute_factor: 0.15,
    };

    pub const EMBEDDED: Self = Self {
        name: "Embedded / IoT (1 core)",
        cores: 1,
        bandwidth_factor: 0.05,
        compute_factor: 0.03,
    };

    pub const WASM_BROWSER: Self = Self {
        name: "WASM in Browser (4 threads)",
        cores: 4,
        bandwidth_factor: 0.4,
        compute_factor: 0.3,
    };
}

impl SimulatedBackend {
    pub fn new(profile: DeviceProfile) -> Self {
        let info = DeviceInfo {
            kind: BackendKind::Cpu,
            name: format!("Simulated: {}", profile.name),
            memory_bytes: 0,
            max_parallelism: profile.cores,
        };
        Self {
            inner: CpuBackend::default(),
            profile,
            info,
        }
    }

    /// Simulate slower hardware by doing extra work proportional to the factor.
    fn throttle(&self, n: usize) {
        let extra_iters = ((1.0 / self.profile.compute_factor - 1.0) * n as f64) as usize;
        let mut _sink = 0u64;
        for i in 0..extra_iters.min(n * 10) {
            _sink = _sink.wrapping_add(i as u64);
        }
        std::hint::black_box(_sink);
    }
}

impl ComputeBackend for SimulatedBackend {
    fn device_info(&self) -> &DeviceInfo {
        &self.info
    }

    fn map_f64(&self, data: &[f64], f: fn(f64) -> f64) -> Vec<f64> {
        self.throttle(data.len());
        self.inner.map_f64(data, f)
    }

    fn filter_indices(&self, data: &[f64], pred: fn(f64) -> bool) -> Vec<usize> {
        self.throttle(data.len());
        self.inner.filter_indices(data, pred)
    }

    fn sort_f64(&self, data: &mut [f64]) {
        self.throttle(data.len());
        self.inner.sort_f64(data)
    }

    fn sum_f64(&self, data: &[f64]) -> f64 {
        self.throttle(data.len());
        self.inner.sum_f64(data)
    }

    fn prefix_sum_f64(&self, data: &[f64]) -> Vec<f64> {
        self.throttle(data.len());
        self.inner.prefix_sum_f64(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn backend() -> CpuBackend {
        CpuBackend::default()
    }

    #[test]
    fn cpu_device_info() {
        let b = backend();
        let info = b.device_info();
        assert_eq!(info.kind, BackendKind::Cpu);
        assert!(info.max_parallelism > 0);
    }

    #[test]
    fn map_f64() {
        let b = backend();
        let data = vec![1.0, 2.0, 3.0];
        let result = b.map_f64(&data, |v| v * 10.0);
        assert_eq!(result, vec![10.0, 20.0, 30.0]);
    }

    #[test]
    fn filter_indices() {
        let b = backend();
        let data = vec![1.0, 5.0, 2.0, 8.0, 3.0];
        let idx = b.filter_indices(&data, |v| v > 4.0);
        assert_eq!(idx, vec![1, 3]);
    }

    #[test]
    fn sum_f64() {
        let b = backend();
        let data = vec![1.0, 2.0, 3.0, 4.0];
        assert!((b.sum_f64(&data) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn sort_f64() {
        let b = backend();
        let mut data = vec![3.0, 1.0, 4.0, 1.0, 5.0];
        b.sort_f64(&mut data);
        assert_eq!(data, vec![1.0, 1.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn prefix_sum() {
        let b = backend();
        let result = b.prefix_sum_f64(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(result, vec![1.0, 3.0, 6.0, 10.0]);
    }

    #[test]
    fn hinted_map_sequential_for_small_data() {
        let b = backend();
        let hints = Hints::default(); // Low complexity → threshold 10_000
        let data = vec![1.0, 2.0, 3.0]; // below threshold
        let result = b.map_f64_hinted(&data, |v| v * 2.0, &hints);
        assert_eq!(result, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn hinted_sum_sequential_for_small_data() {
        let b = backend();
        let hints = Hints::default();
        let data = vec![1.0, 2.0, 3.0];
        assert!((b.sum_f64_hinted(&data, &hints) - 6.0).abs() < 1e-10);
    }

    #[test]
    fn hinted_massive_always_parallel() {
        let b = backend();
        let hints = Hints::massive(1_000_000);
        assert_eq!(hints.parallelism_threshold(), 0);
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        // Should use parallel path even for small data with massive hints
        let result = b.map_f64_hinted(&data, |v| v + 1.0, &hints);
        assert_eq!(result.len(), 100);
        assert!((result[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn simulated_backend_runs() {
        let b = SimulatedBackend::new(DeviceProfile::WASM_BROWSER);
        let info = b.device_info();
        assert_eq!(info.max_parallelism, 4);
        assert!(info.name.contains("Simulated"));

        let data = vec![1.0, 2.0, 3.0];
        assert_eq!(b.map_f64(&data, |v| v * 2.0), vec![2.0, 4.0, 6.0]);
        assert!((b.sum_f64(&data) - 6.0).abs() < 1e-10);
    }

    #[test]
    fn simulated_profiles_exist() {
        // Ensure all profile constants compile and have sane values
        assert_eq!(DeviceProfile::HIGH_END_DESKTOP.cores, 16);
        assert_eq!(DeviceProfile::MID_RANGE_LAPTOP.cores, 4);
        assert_eq!(DeviceProfile::LOW_END_MOBILE.cores, 2);
        assert_eq!(DeviceProfile::EMBEDDED.cores, 1);
        assert_eq!(DeviceProfile::WASM_BROWSER.cores, 4);
    }

    #[test]
    fn empty_data_operations() {
        let b = backend();
        assert_eq!(b.map_f64(&[], |v| v), Vec::<f64>::new());
        assert_eq!(b.filter_indices(&[], |_| true), Vec::<usize>::new());
        assert!((b.sum_f64(&[]) - 0.0).abs() < 1e-10);
        assert_eq!(b.prefix_sum_f64(&[]), Vec::<f64>::new());
    }
}
