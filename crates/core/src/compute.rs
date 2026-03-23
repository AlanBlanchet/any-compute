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

use crate::hints::Hints;
use crate::kernel::{
    BinaryOp, CpuSimdKernel, Kernel, KernelBackend, KernelOp, KernelStats, ReduceOp, UnaryOp,
};
use rayon::prelude::*;
use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, LazyLock, Mutex};

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
// ── OpQueue — batched operation recording + flushing ────────────────────
// ═══════════════════════════════════════════════════════════════════════════

/// A recorded operation with its data pointers — ready for batched dispatch.
#[derive(Debug, Clone)]
pub enum QueuedOp {
    Unary { data: Vec<f64>, op: UnaryOp },
    Binary { a: Vec<f64>, b: Vec<f64>, op: BinaryOp },
    Reduce { data: Vec<f64>, op: ReduceOp },
    Scan { data: Vec<f64>, op: ReduceOp },
    Sort { data: Vec<f64> },
    Gemm { a: Vec<f64>, b: Vec<f64>, m: usize, n: usize, k: usize },
}

/// Result of a flushed operation — matches the shape of what was queued.
#[derive(Debug, Clone)]
pub enum OpResult {
    Vector(Vec<f64>),
    Scalar(f64),
}

impl OpResult {
    pub fn into_vec(self) -> Vec<f64> {
        match self { Self::Vector(v) => v, Self::Scalar(s) => vec![s] }
    }
    pub fn as_scalar(&self) -> f64 {
        match self { Self::Scalar(s) => *s, Self::Vector(v) => v[0] }
    }
}

/// Records operations lazily, flushes them as a batch to a [`Device`].
///
/// On CPU this provides cache-locality wins (data stays hot between ops).
/// On GPU this will collapse into fewer roundtrips / command buffer submissions.
///
/// ```
/// use any_compute_core::compute::{Device, QueuedOp};
/// use any_compute_core::kernel::UnaryOp;
///
/// let dev = Device::cpu();
/// let mut q = dev.queue();
/// q.push(QueuedOp::Unary { data: vec![1.0, 4.0, 9.0], op: UnaryOp::Sqrt });
/// q.push(QueuedOp::Unary { data: vec![0.0, 1.0, -1.0], op: UnaryOp::Abs });
/// let results = q.flush();
/// assert_eq!(results.len(), 2);
/// ```
pub struct OpQueue {
    device: Device,
    ops: Vec<QueuedOp>,
}

impl OpQueue {
    fn new(device: Device) -> Self {
        Self { device, ops: Vec::new() }
    }

    /// Enqueue an operation for batched dispatch.
    pub fn push(&mut self, op: QueuedOp) {
        self.ops.push(op);
    }

    /// Number of pending operations.
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// Execute all queued ops in order, return results. Clears the queue.
    pub fn flush(&mut self) -> Vec<OpResult> {
        let ops = std::mem::take(&mut self.ops);
        ops.into_iter().map(|op| self.dispatch(op)).collect()
    }

    fn dispatch(&self, op: QueuedOp) -> OpResult {
        let d = &self.device;
        match op {
            QueuedOp::Unary { data, op } => OpResult::Vector(d.unary(&data, op)),
            QueuedOp::Binary { a, b, op } => OpResult::Vector(d.binary(&a, &b, op)),
            QueuedOp::Reduce { data, op } => OpResult::Scalar(d.reduce(&data, op)),
            QueuedOp::Scan { data, op } => OpResult::Vector(d.scan(&data, op)),
            QueuedOp::Sort { mut data } => { d.sort(&mut data); OpResult::Vector(data) }
            QueuedOp::Gemm { a, b, m, n, k } => OpResult::Vector(d.gemm(&a, &b, m, n, k)),
        }
    }
}

impl Device {
    /// Create an operation queue for batched dispatch on this device.
    pub fn queue(&self) -> OpQueue {
        OpQueue::new(self.clone())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ── OpCache — content-addressed memoization of compute results ──────────
// ═══════════════════════════════════════════════════════════════════════════

/// Content-addressed cache for device operations.
///
/// Keys on `(op_tag, data_generation)` so identical computations on unchanged
/// data return instantly. Thread-safe via internal `Mutex`.
///
/// ```
/// use any_compute_core::compute::{Device, OpCache};
/// use any_compute_core::kernel::ReduceOp;
///
/// let dev = Device::cpu();
/// let cache = OpCache::new(1024);
/// let data = vec![1.0, 2.0, 3.0];
///
/// // First call computes and caches.
/// let sum = cache.reduce(&dev, &data, ReduceOp::Sum);
/// // Second call with same data hits cache.
/// let sum2 = cache.reduce(&dev, &data, ReduceOp::Sum);
/// assert_eq!(sum, sum2);
/// ```
pub struct OpCache {
    inner: Mutex<CacheInner>,
}

struct CacheInner {
    entries: HashMap<u64, CacheEntry>,
    capacity: usize,
}

#[derive(Clone)]
enum CacheEntry {
    Vector(Vec<f64>),
    Scalar(f64),
}

impl OpCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: Mutex::new(CacheInner {
                entries: HashMap::with_capacity(capacity.min(256)),
                capacity,
            }),
        }
    }

    /// Cache key from data pointer + length + op discriminant.
    fn key(data_ptr: usize, data_len: usize, op_tag: u64) -> u64 {
        // FNV-1a style mix — fast, good enough for cache keys.
        let mut h = 0xcbf29ce484222325u64;
        h ^= data_ptr as u64;
        h = h.wrapping_mul(0x100000001b3);
        h ^= data_len as u64;
        h = h.wrapping_mul(0x100000001b3);
        h ^= op_tag;
        h.wrapping_mul(0x100000001b3)
    }

    fn op_tag_unary(op: UnaryOp) -> u64 {
        // Discriminant as tag — UnaryOp variants are small integers.
        // Safety: enum discriminant read via mem::discriminant is stable.
        let disc = std::mem::discriminant(&op);
        let mut bytes = [0u8; 8];
        bytes[0] = 1; // namespace: unary
        // Use pointer to discriminant as hash input
        let d_ptr = &disc as *const _ as usize;
        bytes[1..5].copy_from_slice(&(d_ptr as u32).to_le_bytes());
        u64::from_le_bytes(bytes)
    }

    fn op_tag_reduce(op: ReduceOp) -> u64 {
        let mut bytes = [0u8; 8];
        bytes[0] = 2; // namespace: reduce
        bytes[1] = op as u8;
        u64::from_le_bytes(bytes)
    }

    fn op_tag_binary(op: BinaryOp) -> u64 {
        let mut bytes = [0u8; 8];
        bytes[0] = 3; // namespace: binary
        bytes[1] = op as u8;
        u64::from_le_bytes(bytes)
    }

    /// Cached unary: returns from cache if data pointer + len + op match.
    pub fn unary(&self, dev: &Device, data: &[f64], op: UnaryOp) -> Vec<f64> {
        let k = Self::key(data.as_ptr() as usize, data.len(), Self::op_tag_unary(op));
        if let Some(CacheEntry::Vector(v)) = self.get(k) {
            return v;
        }
        let result = dev.unary(data, op);
        self.put(k, CacheEntry::Vector(result.clone()));
        result
    }

    /// Cached reduce.
    pub fn reduce(&self, dev: &Device, data: &[f64], op: ReduceOp) -> f64 {
        let k = Self::key(data.as_ptr() as usize, data.len(), Self::op_tag_reduce(op));
        if let Some(CacheEntry::Scalar(s)) = self.get(k) {
            return s;
        }
        let result = dev.reduce(data, op);
        self.put(k, CacheEntry::Scalar(result));
        result
    }

    /// Cached binary.
    pub fn binary(&self, dev: &Device, a: &[f64], b: &[f64], op: BinaryOp) -> Vec<f64> {
        let k = Self::key(a.as_ptr() as usize, a.len(), Self::op_tag_binary(op));
        if let Some(CacheEntry::Vector(v)) = self.get(k) {
            return v;
        }
        let result = dev.binary(a, b, op);
        self.put(k, CacheEntry::Vector(result.clone()));
        result
    }

    /// Evict all entries.
    pub fn clear(&self) {
        self.inner.lock().unwrap().entries.clear();
    }

    /// Number of cached entries.
    pub fn len(&self) -> usize {
        self.inner.lock().unwrap().entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn get(&self, key: u64) -> Option<CacheEntry> {
        self.inner.lock().unwrap().entries.get(&key).cloned()
    }

    fn put(&self, key: u64, entry: CacheEntry) {
        let mut inner = self.inner.lock().unwrap();
        if inner.entries.len() >= inner.capacity {
            // Simple eviction: clear all when full. A real LRU can replace this.
            inner.entries.clear();
        }
        inner.entries.insert(key, entry);
    }
}

// ── DeviceProfile ────────────────────────────────────────────────────────

/// Hardware profile for simulation — describes constrained device characteristics.
#[derive(Debug, Clone)]
pub struct DeviceProfile {
    pub name: &'static str,
    /// Simulated core count.
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

#[cfg(test)]
mod tests {
    use super::*;

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
        q.push(QueuedOp::Unary { data: vec![1.0, 4.0, 9.0], op: UnaryOp::Sqrt });
        q.push(QueuedOp::Reduce { data: vec![1.0, 2.0, 3.0], op: ReduceOp::Sum });
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
        q.push(QueuedOp::Sort { data: vec![3.0, 1.0, 2.0] });
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
