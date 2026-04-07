//! Unified device abstraction — one type for every hardware backend.
//!
//! [`Device`] is the single entry point for all hardware-dispatched computation.
//! Instead of separate traits for compute vs render vs kernel, `Device` is a
//! concrete value you create, pass around, and use directly.
//!
//! **All parallel work must go through `Device`** — never call rayon directly
//! outside the Device implementation. This is the single source of truth for
//! dispatching parallel computation.
//!
//! ## Usage
//!
//! ```
//! use any_compute_core::compute::Device;
//!
//! let dev = Device::cpu();      // CPU with best SIMD
//! let data = vec![1.0, 2.0, 3.0];
//! let doubled = dev.map(&data, |v| v * 2.0);
//! let total  = dev.sum(&data);
//! ```
//!
//! ## Mixing devices
//!
//! Data lives on a specific device. `Buffer` carries its `Device`, so operations
//! always dispatch to the right hardware. When GPU/CUDA backends are added behind
//! feature flags, the same API surface works — just swap the device.
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

mod cache;
mod device;
mod profile;
mod queue;

pub use cache::OpCache;
pub use device::{BackendKind, Device, DeviceInfo};
pub use profile::DeviceProfile;
pub use queue::{OpQueue, OpResult, QueuedOp};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hints::Hints;
    use crate::kernel::{CpuSimdKernel, ReduceOp, UnaryOp};

    fn dev() -> Device {
        Device::cpu()
    }

    #[test]
    fn cpu_device_info() {
        let d = dev();
        assert_eq!(d.kind(), BackendKind::Cpu);
        assert!(d.info().max_parallelism > 0);
    }

    #[test]
    fn map_f64() {
        let d = dev();
        let data = vec![1.0, 2.0, 3.0];
        let result = d.map(&data, |v| v * 10.0);
        assert_eq!(result, vec![10.0, 20.0, 30.0]);
    }

    #[test]
    fn filter_indices() {
        let d = dev();
        let data = vec![1.0, 5.0, 2.0, 8.0, 3.0];
        let idx = d.filter(&data, |v| v > 4.0);
        assert_eq!(idx, vec![1, 3]);
    }

    #[test]
    fn sum_f64() {
        let d = dev();
        let data = vec![1.0, 2.0, 3.0, 4.0];
        assert!((d.sum(&data) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn sort_f64() {
        let d = dev();
        let mut data = vec![3.0, 1.0, 4.0, 1.0, 5.0];
        d.sort(&mut data);
        assert_eq!(data, vec![1.0, 1.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn prefix_sum() {
        let d = dev();
        let result = d.prefix_sum(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(result, vec![1.0, 3.0, 6.0, 10.0]);
    }

    #[test]
    fn hinted_map_sequential_for_small_data() {
        let d = dev();
        let hints = Hints::default();
        let data = vec![1.0, 2.0, 3.0];
        let result = d.map_hinted(&data, |v| v * 2.0, &hints);
        assert_eq!(result, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn hinted_sum_sequential_for_small_data() {
        let d = dev();
        let hints = Hints::default();
        let data = vec![1.0, 2.0, 3.0];
        assert!((d.sum_hinted(&data, &hints) - 6.0).abs() < 1e-10);
    }

    #[test]
    fn hinted_massive_always_parallel() {
        let d = dev();
        let hints = Hints::massive(1_000_000);
        assert_eq!(hints.parallelism_threshold(), 0);
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let result = d.map_hinted(&data, |v| v + 1.0, &hints);
        assert_eq!(result.len(), 100);
        assert!((result[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn simulated_device_runs() {
        let d = Device::simulated(DeviceProfile::WASM_BROWSER);
        assert_eq!(d.info().max_parallelism, 4);
        assert!(d.name().contains("Simulated"));

        let data = vec![1.0, 2.0, 3.0];
        assert_eq!(d.map(&data, |v| v * 2.0), vec![2.0, 4.0, 6.0]);
        assert!((d.sum(&data) - 6.0).abs() < 1e-10);
    }

    #[test]
    fn simulated_profiles_exist() {
        assert_eq!(DeviceProfile::HIGH_END_DESKTOP.cores, 16);
        assert_eq!(DeviceProfile::MID_RANGE_LAPTOP.cores, 4);
        assert_eq!(DeviceProfile::LOW_END_MOBILE.cores, 2);
        assert_eq!(DeviceProfile::EMBEDDED.cores, 1);
        assert_eq!(DeviceProfile::WASM_BROWSER.cores, 4);
    }

    #[test]
    fn empty_data_operations() {
        let d = dev();
        assert_eq!(d.map(&[], |v| v), Vec::<f64>::new());
        assert_eq!(d.filter(&[], |_| true), Vec::<usize>::new());
        assert!((d.sum(&[]) - 0.0).abs() < 1e-10);
        assert_eq!(d.prefix_sum(&[]), Vec::<f64>::new());
    }

    #[test]
    fn device_clone_is_cheap() {
        let d1 = Device::cpu();
        let d2 = d1.clone();
        assert_eq!(d1.kind(), d2.kind());
        assert_eq!(d1.name(), d2.name());
    }

    #[test]
    fn device_from_kernel() {
        let d = Device::from_kernel(Box::new(CpuSimdKernel::default()));
        assert_eq!(d.kind(), BackendKind::Cpu);
        let result = d.unary(&[1.0, 4.0, 9.0], UnaryOp::Sqrt);
        assert!((result[0] - 1.0).abs() < 1e-10);
        assert!((result[1] - 2.0).abs() < 1e-10);
        assert!((result[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn device_best_returns_cpu() {
        let d = Device::best();
        assert_eq!(d.kind(), BackendKind::Cpu);
    }

    // ── OpQueue tests ────────────────────────────────────────────────────

    #[test]
    fn op_queue_basic_flush() {
        let d = dev();
        let mut q = d.queue();
        q.push(QueuedOp::Unary {
            data: vec![1.0, 4.0, 9.0],
            op: UnaryOp::Sqrt,
        });
        q.push(QueuedOp::Reduce {
            data: vec![1.0, 2.0, 3.0],
            op: ReduceOp::Sum,
        });
        assert_eq!(q.len(), 2);
        let results = q.flush();
        assert_eq!(results.len(), 2);
        let sqrts = results[0].clone().into_vec();
        assert!((sqrts[0] - 1.0).abs() < 1e-10);
        assert!((sqrts[1] - 2.0).abs() < 1e-10);
        assert!((results[1].as_scalar() - 6.0).abs() < 1e-10);
        assert!(q.is_empty());
    }

    #[test]
    fn op_queue_empty_flush() {
        let d = dev();
        let mut q = d.queue();
        assert!(q.flush().is_empty());
    }

    #[test]
    fn op_queue_sort() {
        let d = dev();
        let mut q = d.queue();
        q.push(QueuedOp::Sort {
            data: vec![3.0, 1.0, 2.0],
        });
        let results = q.flush();
        assert_eq!(results[0].clone().into_vec(), vec![1.0, 2.0, 3.0]);
    }

    // ── OpCache tests ────────────────────────────────────────────────────

    #[test]
    fn op_cache_reduce_hit() {
        let d = dev();
        let cache = OpCache::new(64);
        let data = vec![1.0, 2.0, 3.0];
        let sum1 = cache.reduce(&d, &data, ReduceOp::Sum);
        let sum2 = cache.reduce(&d, &data, ReduceOp::Sum);
        assert!((sum1 - 6.0).abs() < 1e-10);
        assert_eq!(sum1, sum2);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn op_cache_clear() {
        let d = dev();
        let cache = OpCache::new(64);
        cache.reduce(&d, &[1.0, 2.0], ReduceOp::Sum);
        assert_eq!(cache.len(), 1);
        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn op_cache_evicts_when_full() {
        let d = dev();
        let cache = OpCache::new(2);
        let a = vec![1.0];
        let b = vec![2.0];
        let c = vec![3.0];
        cache.reduce(&d, &a, ReduceOp::Sum);
        cache.reduce(&d, &b, ReduceOp::Sum);
        assert_eq!(cache.len(), 2);
        // Third insert triggers eviction (clear-all strategy).
        cache.reduce(&d, &c, ReduceOp::Sum);
        assert_eq!(cache.len(), 1);
    }
}
