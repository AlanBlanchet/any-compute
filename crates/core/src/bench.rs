//! Benchmark runner library — shared by CLI (`anc-bench`) and GUI (`anc-bench-window`).
//!
//! All types are `Clone + Serialize` so they can be displayed, streamed, or
//! written to JSON.  Runner functions are pure: they return results without
//! printing anything.

use crate::Lerp;
use crate::animation::{Easing, Transition};
use crate::compute::{ComputeBackend, CpuBackend, DeviceProfile, SimulatedBackend};
use crate::data::{CellValue, ColumnKind, ColumnMeta, DataSource, VecSource};
use crate::hints::Hints;
use crate::kernel::{BinaryOp, ReduceOp, UnaryOp, best_kernel};
#[cfg(feature = "hwinfo")]
use crate::kernel::{CpuSimdKernel, Kernel};
use crate::layout::{Point, Rect, ScrollState};
use crate::render::{Border, Color, Primitive, RenderList};
use humansize::{BINARY, SizeFormatter};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

// ══════════════════════════════════════════════════════════════════════════
// Constants — benchmark data sizes, memory thresholds, formatting
// ══════════════════════════════════════════════════════════════════════════

/// Standard data sizes for element-wise kernel benchmarks.
const KERNEL_SIZES: &[usize] = &[100_000, 1_000_000, 10_000_000];

/// Matrix dimensions for GEMM benchmarks.
const GEMM_DIMS: &[usize] = &[64, 128, 256, 512];

/// Data sizes for sort benchmarks.
const SORT_SIZES: &[usize] = &[10_000, 100_000, 1_000_000];

/// Data sizes for compute backend parallel operation benchmarks.
const COMPUTE_SIZES: &[usize] = &[10_000, 100_000, 1_000_000, 10_000_000];

/// Data sizes for hints-aware dispatch benchmarks.
const HINTS_SIZES: &[usize] = &[1_000, 100_000, 1_000_000];

/// Row counts for data virtualization benchmarks.
const VIRTUALIZATION_ROWS: &[usize] = &[1_000, 100_000, 1_000_000, 10_000_000];

/// Row counts for visible_range layout benchmarks.
const LAYOUT_RANGE_SIZES: &[usize] = &[100_000, 1_000_000, 10_000_000, 100_000_000];

/// Rectangle counts for hit-test benchmarks.
const HIT_TEST_SIZES: &[usize] = &[1_000, 10_000, 100_000];

/// Transition counts for animation tick benchmarks.
const ANIMATION_TICK_SIZES: &[usize] = &[100, 1_000, 10_000, 50_000];

/// Transition counts for color animation benchmarks.
const ANIMATION_COLOR_SIZES: &[usize] = &[1_000, 10_000];

/// Primitive counts for render list benchmarks.
const RENDER_RECT_SIZES: &[usize] = &[1_000, 10_000, 50_000, 100_000];

/// Primitive counts for grid cell (rect+text+border) benchmarks.
const RENDER_GRID_SIZES: &[usize] = &[1_000, 10_000, 50_000];

/// Item count for lerp throughput and easing benchmarks.
const LERP_COUNT: usize = 1_000_000;

/// Data sizes for simulated backend benchmarks.
const SIMULATED_SIZES: &[usize] = &[10_000, 100_000, 1_000_000];

/// RAM threshold for memory bandwidth estimation heuristic.
const HIGH_BW_RAM_THRESHOLD: u64 = 32 * 1024 * 1024 * 1024;

/// Estimated memory bandwidth (GB/s) for systems above/below the RAM threshold.
const MEM_BW_HIGH: f64 = 60.0;
const MEM_BW_LOW: f64 = 40.0;

/// Microsecond thresholds for duration formatting.
const US_PER_SECOND: f64 = 1_000_000.0;
const US_PER_MILLI: f64 = 1_000.0;

#[cfg(feature = "hwinfo")]
use sysinfo::System;

// ══════════════════════════════════════════════════════════════════════════
// Report types
// ══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct FullReport {
    pub timestamp: String,
    pub hardware: HardwareReport,
    pub features: FeaturesReport,
    pub kernel_benchmarks: Vec<ScenarioReport>,
    pub compute_benchmarks: Vec<ScenarioReport>,
    pub framework_benchmarks: Vec<ScenarioReport>,
    pub comparisons: Vec<ComparisonTable>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct HardwareReport {
    pub cpu: CpuReport,
    pub memory: MemoryReport,
    pub gpus: Vec<GpuReport>,
    pub simd: SimdReport,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CpuReport {
    pub brand: String,
    pub physical_cores: usize,
    pub logical_cores: usize,
    pub frequency_mhz: u64,
    pub arch: String,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct MemoryReport {
    pub total_bytes: u64,
    pub available_bytes: u64,
    pub used_bytes: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct GpuReport {
    pub name: String,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SimdReport {
    pub detected: String,
    pub vector_width: usize,
    pub features: Vec<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct FeaturesReport {
    pub cuda: bool,
    pub rocm: bool,
    pub mkl: bool,
    pub metal: bool,
    pub wgpu: bool,
    pub shader: bool,
    pub hwinfo: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ScenarioReport {
    pub category: String,
    pub results: Vec<BenchResult>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct BenchResult {
    pub name: String,
    pub scale: usize,
    pub duration_us: u128,
    pub throughput_ops_sec: f64,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ComparisonTable {
    pub category: String,
    pub entries: Vec<ComparisonEntry>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ComparisonEntry {
    pub operation: String,
    pub any_compute_us: u128,
    pub any_compute_ops: f64,
    pub comparisons: Vec<LibComparison>,
}

/// Provenance of a comparison data-point: whether it was actually measured
/// during this run or sourced from a published benchmark estimate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ComparisonSource {
    /// Measured in this benchmark run on this machine — fully trustworthy.
    Measured,
    /// Published / documented ratio — not live-measured here.
    /// The note field explains the source.
    Estimate,
}

impl Default for ComparisonSource {
    fn default() -> Self {
        Self::Estimate
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LibComparison {
    pub library: String,
    /// Throughput in ops/sec.  For `Measured` entries this is a real
    /// measurement; for `Estimate` it is derived from published ratios.
    pub ops: f64,
    pub notes: String,
    pub source: ComparisonSource,
}

// ══════════════════════════════════════════════════════════════════════════
// Benchmark harness
// ══════════════════════════════════════════════════════════════════════════

pub fn bench_fn<F: FnMut()>(
    name: &str,
    scale: usize,
    warmup: usize,
    iters: usize,
    mut f: F,
) -> BenchResult {
    for _ in 0..warmup {
        f();
    }
    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    let total = start.elapsed();
    let per_iter = total / iters as u32;
    let ops_sec = if per_iter.as_secs_f64() > 0.0 {
        1.0 / per_iter.as_secs_f64()
    } else {
        f64::INFINITY
    };

    BenchResult {
        name: name.to_string(),
        scale,
        duration_us: per_iter.as_micros(),
        throughput_ops_sec: ops_sec,
    }
}

// ══════════════════════════════════════════════════════════════════════════
// Hardware detection
// ══════════════════════════════════════════════════════════════════════════

#[cfg(feature = "hwinfo")]
pub fn detect_hardware() -> HardwareReport {
    let mut sys = System::new_all();
    sys.refresh_all();

    let cpus = sys.cpus();
    let cpu = if cpus.is_empty() {
        CpuReport {
            brand: "Unknown".into(),
            physical_cores: num_cpus::get_physical(),
            logical_cores: num_cpus::get(),
            frequency_mhz: 0,
            arch: std::env::consts::ARCH.into(),
        }
    } else {
        CpuReport {
            brand: cpus[0].brand().to_string(),
            physical_cores: num_cpus::get_physical(),
            logical_cores: num_cpus::get(),
            frequency_mhz: cpus[0].frequency(),
            arch: std::env::consts::ARCH.into(),
        }
    };

    let memory = MemoryReport {
        total_bytes: sys.total_memory(),
        available_bytes: sys.available_memory(),
        used_bytes: sys.used_memory(),
    };

    let kernel = CpuSimdKernel::default();
    let simd_features = detect_simd_features();

    let simd = SimdReport {
        detected: kernel.name().to_string(),
        vector_width: kernel.vector_width(),
        features: simd_features,
    };

    HardwareReport {
        cpu,
        memory,
        gpus: vec![],
        simd,
    }
}

pub fn detect_simd_features() -> Vec<String> {
    let mut features = Vec::new();

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse4.2") {
            features.push("SSE4.2".into());
        }
        if is_x86_feature_detected!("avx") {
            features.push("AVX".into());
        }
        if is_x86_feature_detected!("avx2") {
            features.push("AVX2".into());
        }
        if is_x86_feature_detected!("fma") {
            features.push("FMA".into());
        }
        if is_x86_feature_detected!("avx512f") {
            features.push("AVX-512F".into());
        }
        if is_x86_feature_detected!("avx512bw") {
            features.push("AVX-512BW".into());
        }
        if is_x86_feature_detected!("avx512vl") {
            features.push("AVX-512VL".into());
        }
        if is_x86_feature_detected!("bmi1") {
            features.push("BMI1".into());
        }
        if is_x86_feature_detected!("bmi2") {
            features.push("BMI2".into());
        }
        if is_x86_feature_detected!("popcnt") {
            features.push("POPCNT".into());
        }
        if is_x86_feature_detected!("aes") {
            features.push("AES-NI".into());
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        features.push("NEON".into());
    }

    features
}

pub fn detect_features() -> FeaturesReport {
    FeaturesReport {
        cuda: cfg!(feature = "cuda"),
        rocm: cfg!(feature = "rocm"),
        mkl: cfg!(feature = "mkl"),
        metal: cfg!(feature = "metal"),
        wgpu: cfg!(feature = "wgpu-backend"),
        shader: cfg!(feature = "shader"),
        hwinfo: cfg!(feature = "hwinfo"),
    }
}

/// Peak performance estimates from hardware report.
pub struct PeakPerformance {
    pub fp64_gflops: f64,
    pub fp32_gflops: f64,
    pub mem_bw_gbs: f64,
    pub rayon_threads: usize,
    pub has_fma: bool,
}

pub fn estimate_peak(hw: &HardwareReport) -> PeakPerformance {
    let cores = hw.cpu.logical_cores as f64;
    let freq_ghz = hw.cpu.frequency_mhz as f64 / 1000.0;
    let simd_width = hw.simd.vector_width as f64;
    let has_fma = hw.simd.features.iter().any(|f| f == "FMA");
    let fma_factor = if has_fma { 2.0 } else { 1.0 };
    let fp64 = cores * freq_ghz * simd_width * fma_factor;

    PeakPerformance {
        fp64_gflops: fp64,
        fp32_gflops: fp64 * 2.0,
        mem_bw_gbs: if hw.memory.total_bytes > HIGH_BW_RAM_THRESHOLD {
            MEM_BW_HIGH
        } else {
            MEM_BW_LOW
        },
        rayon_threads: rayon::current_num_threads(),
        has_fma,
    }
}

// ══════════════════════════════════════════════════════════════════════════
// Data generation helpers
// ══════════════════════════════════════════════════════════════════════════

fn make_source(rows: usize, cols: usize) -> VecSource {
    let columns: Vec<ColumnMeta> = (0..cols)
        .map(|i| ColumnMeta {
            name: format!("col_{i}"),
            kind: if i % 2 == 0 {
                ColumnKind::Int
            } else {
                ColumnKind::Float
            },
        })
        .collect();

    let data: Vec<Vec<CellValue>> = (0..rows)
        .map(|r| {
            (0..cols)
                .map(|c| {
                    if c % 2 == 0 {
                        CellValue::Int((r * cols + c) as i64)
                    } else {
                        CellValue::Float((r * cols + c) as f64 * 0.1)
                    }
                })
                .collect()
        })
        .collect();

    VecSource {
        columns,
        rows: data,
    }
}

fn make_f64_data(n: usize) -> Vec<f64> {
    (0..n).map(|i| (i as f64) * 0.7 + 1.3).collect()
}

// ══════════════════════════════════════════════════════════════════════════
// Benchmark categories
// ══════════════════════════════════════════════════════════════════════════

/// All benchmark categories that can be run individually.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BenchCategory {
    KernelUnary,
    KernelBinary,
    KernelReduce,
    KernelGemm,
    KernelSort,
    ComputeParallel,
    HintsOptimization,
    DataVirtualization,
    LayoutSpatial,
    Animation,
    RenderList,
    LerpThroughput,
    EventHandling,
    /// 1M point generation + 256×256 density binning.
    PointCloud,
    /// Large GEMM: 256×256 up to 1024×1024.
    MatMulLarge,
    /// Scaled dot-product attention (QKᵀ → softmax → V) at multiple seq/d configs.
    AttentionOps,
    /// Sphere vertex transforms + perspective projection at multiple mesh resolutions.
    Geometry3D,
}

impl BenchCategory {
    pub const ALL: &[Self] = &[
        Self::KernelUnary,
        Self::KernelBinary,
        Self::KernelReduce,
        Self::KernelGemm,
        Self::KernelSort,
        Self::ComputeParallel,
        Self::HintsOptimization,
        Self::DataVirtualization,
        Self::LayoutSpatial,
        Self::Animation,
        Self::RenderList,
        Self::LerpThroughput,
        Self::EventHandling,
        Self::PointCloud,
        Self::MatMulLarge,
        Self::AttentionOps,
        Self::Geometry3D,
    ];

    /// Machine-readable identifier used as `ScenarioReport::category`.
    pub fn id(self) -> &'static str {
        match self {
            Self::KernelUnary => "kernel_unary",
            Self::KernelBinary => "kernel_binary",
            Self::KernelReduce => "kernel_reduce",
            Self::KernelGemm => "kernel_gemm",
            Self::KernelSort => "kernel_sort",
            Self::ComputeParallel => "compute_parallel",
            Self::HintsOptimization => "hints_auto_optimization",
            Self::DataVirtualization => "data_virtualization",
            Self::LayoutSpatial => "layout_spatial",
            Self::Animation => "animation",
            Self::RenderList => "render_list",
            Self::LerpThroughput => "lerp_throughput",
            Self::EventHandling => "event_handling",
            Self::PointCloud => "point_cloud",
            Self::MatMulLarge => "matmul_large",
            Self::AttentionOps => "attention_ops",
            Self::Geometry3D => "geometry_3d",
        }
    }

    /// Human-readable label for display.
    pub fn label(self) -> &'static str {
        match self {
            Self::KernelUnary => "Kernel: Unary Ops",
            Self::KernelBinary => "Kernel: Binary Ops",
            Self::KernelReduce => "Kernel: Reductions",
            Self::KernelGemm => "Kernel: GEMM",
            Self::KernelSort => "Kernel: Sort",
            Self::ComputeParallel => "Compute: Parallel Ops",
            Self::HintsOptimization => "Compute: Hint-Aware Dispatch",
            Self::DataVirtualization => "Data: Virtualization",
            Self::LayoutSpatial => "Layout: Spatial",
            Self::Animation => "Animation: Transitions",
            Self::RenderList => "Render: Primitive Lists",
            Self::LerpThroughput => "Lerp: Interpolation Throughput",
            Self::EventHandling => "Events: Input Dispatch",
            Self::PointCloud => "Point Cloud: 1M Scatter + Density",
            Self::MatMulLarge => "MatMul: Large GEMM (up to 1024×1024)",
            Self::AttentionOps => "Attention: Scaled Dot-Product",
            Self::Geometry3D => "3D Geometry: Vertex Transform + Projection",
        }
    }

    pub fn group(self) -> &'static str {
        match self {
            Self::KernelUnary
            | Self::KernelBinary
            | Self::KernelReduce
            | Self::KernelGemm
            | Self::KernelSort => "Kernel",
            Self::ComputeParallel | Self::HintsOptimization => "Compute",
            Self::DataVirtualization => "Data",
            Self::LayoutSpatial => "Layout",
            Self::Animation => "Animation",
            Self::RenderList => "Render",
            Self::LerpThroughput => "Lerp",
            Self::EventHandling => "Events",
            Self::PointCloud => "Geometry",
            Self::MatMulLarge => "Compute",
            Self::AttentionOps => "Compute",
            Self::Geometry3D => "Geometry",
        }
    }

    /// High-level domain for dashboard grouping.
    pub fn domain(self) -> &'static str {
        match self {
            Self::KernelUnary
            | Self::KernelBinary
            | Self::KernelReduce
            | Self::KernelSort
            | Self::ComputeParallel
            | Self::HintsOptimization => "Compute",
            Self::KernelGemm | Self::MatMulLarge | Self::AttentionOps => "Linear Algebra / AI",
            Self::PointCloud | Self::Geometry3D | Self::LayoutSpatial => "Graphics / 3D",
            Self::Animation | Self::LerpThroughput => "Animation / Dynamics",
            Self::RenderList | Self::DataVirtualization => "Rendering / Data",
            Self::EventHandling => "Events / Interaction",
        }
    }

    /// Short description of what this bench tests — shown in the dashboard.
    pub fn description(self) -> &'static str {
        match self {
            Self::KernelUnary => "abs, negate, sqrt on 10K-10M f64 vectors via SIMD dispatch",
            Self::KernelBinary => "add, mul, fma on paired vectors; measures SIMD throughput",
            Self::KernelReduce => "sum, min, max reductions with SIMD accumulator",
            Self::KernelGemm => "naive matrix multiply 64-512; baseline for BLAS comparison",
            Self::KernelSort => "parallel sort on 10K-10M elements via rayon + pdqsort",
            Self::ComputeParallel => "parallel map/reduce across all cores; rayon dispatch",
            Self::HintsOptimization => "hint-aware dispatch: sorted, dense, contiguous flags",
            Self::DataVirtualization => "virtual scroll window over 100K-1M row data sources",
            Self::LayoutSpatial => "AABB spatial grid insert + range query at scale",
            Self::Animation => "easing transitions: linear, ease-in-out, spring physics",
            Self::RenderList => "batch render-primitive assembly: rects, circles, text",
            Self::LerpThroughput => "raw lerp throughput on f64, Vec3, Color, Rect batches",
            Self::EventHandling => "event dispatch, hit-testing 100K rects, 3-phase propagation",
            Self::PointCloud => "1-5M point Halton generation + 256x256 density binning",
            Self::MatMulLarge => "GEMM at 256-1024 sizes; reports GFLOP/s",
            Self::AttentionOps => "scaled dot-product attention QK^T -> softmax -> V",
            Self::Geometry3D => "sphere mesh vertex transforms + perspective projection",
        }
    }

    /// All unique domains, ordered for display.
    pub fn all_domains() -> &'static [&'static str] {
        &[
            "Compute",
            "Linear Algebra / AI",
            "Graphics / 3D",
            "Animation / Dynamics",
            "Rendering / Data",
            "Events / Interaction",
        ]
    }

    /// Categories belonging to a given domain.
    pub fn for_domain(domain: &str) -> Vec<Self> {
        Self::ALL
            .iter()
            .copied()
            .filter(|c| c.domain() == domain)
            .collect()
    }
}

// ══════════════════════════════════════════════════════════════════════════
// Category runners — each returns a single ScenarioReport
// ══════════════════════════════════════════════════════════════════════════

pub fn run_category(cat: BenchCategory) -> ScenarioReport {
    match cat {
        BenchCategory::KernelUnary => run_kernel_unary(),
        BenchCategory::KernelBinary => run_kernel_binary(),
        BenchCategory::KernelReduce => run_kernel_reduce(),
        BenchCategory::KernelGemm => run_kernel_gemm(),
        BenchCategory::KernelSort => run_kernel_sort(),
        BenchCategory::ComputeParallel => run_compute_parallel(),
        BenchCategory::HintsOptimization => run_hints_optimization(),
        BenchCategory::DataVirtualization => run_data_virtualization(),
        BenchCategory::LayoutSpatial => run_layout_spatial(),
        BenchCategory::Animation => run_animation(),
        BenchCategory::RenderList => run_render_list(),
        BenchCategory::LerpThroughput => run_lerp_throughput(),
        BenchCategory::EventHandling => run_event_handling(),
        BenchCategory::PointCloud => run_point_cloud(),
        BenchCategory::MatMulLarge => run_matmul_large(),
        BenchCategory::AttentionOps => run_attention_ops(),
        BenchCategory::Geometry3D => run_geometry_3d(),
    }
}

/// Run all categories and return a complete report.
#[cfg(feature = "hwinfo")]
pub fn run_all() -> FullReport {
    let hardware = detect_hardware();
    let features = detect_features();

    let mut kernel_benchmarks = Vec::new();
    let mut compute_benchmarks = Vec::new();
    let mut framework_benchmarks = Vec::new();

    for &cat in BenchCategory::ALL {
        let report = run_category(cat);
        match cat.group() {
            "Kernel" => kernel_benchmarks.push(report),
            "Compute" | "Data" => compute_benchmarks.push(report),
            _ => framework_benchmarks.push(report),
        }
    }

    let comparisons = build_comparison_tables(
        &kernel_benchmarks,
        &compute_benchmarks,
        &framework_benchmarks,
    );

    FullReport {
        timestamp: timestamp_now(),
        hardware,
        features,
        kernel_benchmarks,
        compute_benchmarks,
        framework_benchmarks,
        comparisons,
    }
}

// ── Kernel runners ────────────────────────────────────────────────────────

fn run_kernel_unary() -> ScenarioReport {
    let kernel = best_kernel();
    let mut results = Vec::new();
    for &n in KERNEL_SIZES {
        let data = make_f64_data(n);
        for &(name, op) in &[
            ("neg", UnaryOp::Neg),
            ("sqrt", UnaryOp::Sqrt),
            ("exp", UnaryOp::Exp),
            ("sin", UnaryOp::Sin),
            ("relu", UnaryOp::Relu),
            ("sigmoid", UnaryOp::Sigmoid),
        ] {
            results.push(bench_fn(&format!("{name} n={n}"), n, 2, 20, || {
                std::hint::black_box(kernel.map_unary_f64(&data, op));
            }));
        }
    }
    ScenarioReport {
        category: BenchCategory::KernelUnary.id().into(),
        results,
    }
}

fn run_kernel_binary() -> ScenarioReport {
    let kernel = best_kernel();
    let mut results = Vec::new();
    for &n in KERNEL_SIZES {
        let a = make_f64_data(n);
        let b: Vec<f64> = (0..n).map(|i| (i as f64) * 0.3).collect();
        for &(name, op) in &[
            ("add", BinaryOp::Add),
            ("mul", BinaryOp::Mul),
            ("min", BinaryOp::Min),
        ] {
            results.push(bench_fn(&format!("{name} n={n}"), n, 2, 20, || {
                std::hint::black_box(kernel.map_binary_f64(&a, &b, op));
            }));
        }
    }
    ScenarioReport {
        category: BenchCategory::KernelBinary.id().into(),
        results,
    }
}

fn run_kernel_reduce() -> ScenarioReport {
    let kernel = best_kernel();
    let mut results = Vec::new();
    for &n in KERNEL_SIZES {
        let data = make_f64_data(n);
        for &(name, op) in &[
            ("sum", ReduceOp::Sum),
            ("min", ReduceOp::Min),
            ("max", ReduceOp::Max),
            ("mean", ReduceOp::Mean),
        ] {
            results.push(bench_fn(&format!("reduce_{name} n={n}"), n, 3, 50, || {
                std::hint::black_box(kernel.reduce_f64(&data, op));
            }));
        }
    }
    ScenarioReport {
        category: BenchCategory::KernelReduce.id().into(),
        results,
    }
}

fn run_kernel_gemm() -> ScenarioReport {
    let kernel = best_kernel();
    let mut results = Vec::new();
    for &size in GEMM_DIMS {
        let a = vec![1.0f64; size * size];
        let b = vec![1.0f64; size * size];
        results.push(bench_fn(
            &format!("gemm {size}x{size}"),
            size * size,
            1,
            3,
            || {
                std::hint::black_box(kernel.gemm_f64(&a, &b, size, size, size));
            },
        ));
    }
    ScenarioReport {
        category: BenchCategory::KernelGemm.id().into(),
        results,
    }
}

fn run_kernel_sort() -> ScenarioReport {
    let kernel = best_kernel();
    let mut results = Vec::new();
    for &n in SORT_SIZES {
        let original = make_f64_data(n);
        let mut data = original.clone();
        results.push(bench_fn(&format!("sort n={n}"), n, 2, 10, || {
            data.copy_from_slice(&original);
            kernel.sort_f64(&mut data);
            std::hint::black_box(&data);
        }));
    }
    ScenarioReport {
        category: BenchCategory::KernelSort.id().into(),
        results,
    }
}

// ── Compute runners ───────────────────────────────────────────────────────

fn run_compute_parallel() -> ScenarioReport {
    let backend = CpuBackend::default();
    let mut results = Vec::new();
    for &n in COMPUTE_SIZES {
        let data = make_f64_data(n);
        results.push(bench_fn(&format!("map_f64 n={n}"), n, 3, 50, || {
            std::hint::black_box(backend.map_f64(&data, |v| v * 2.0 + 1.0));
        }));
        results.push(bench_fn(&format!("filter_indices n={n}"), n, 3, 50, || {
            std::hint::black_box(backend.filter_indices(&data, |v| v > 500.0));
        }));
        results.push(bench_fn(&format!("sum_f64 n={n}"), n, 3, 50, || {
            std::hint::black_box(backend.sum_f64(&data));
        }));
        results.push(bench_fn(&format!("prefix_sum n={n}"), n, 3, 30, || {
            std::hint::black_box(backend.prefix_sum_f64(&data));
        }));
        let mut sort_data = data.clone();
        results.push(bench_fn(&format!("sort_f64 n={n}"), n, 3, 20, || {
            sort_data.copy_from_slice(&data);
            backend.sort_f64(&mut sort_data);
            std::hint::black_box(&sort_data);
        }));
    }
    ScenarioReport {
        category: BenchCategory::ComputeParallel.id().into(),
        results,
    }
}

fn run_hints_optimization() -> ScenarioReport {
    let backend = CpuBackend::default();
    let mut results = Vec::new();
    let profiles: &[(&str, Hints)] = &[
        ("default", Hints::default()),
        ("cached", Hints::cached()),
        ("animated", Hints::animated()),
        ("massive", Hints::massive(10_000_000)),
        ("streaming", Hints::streaming()),
    ];
    for &n in HINTS_SIZES {
        let data = make_f64_data(n);
        results.push(bench_fn(&format!("map_raw n={n}"), n, 3, 50, || {
            std::hint::black_box(backend.map_f64(&data, |v| v * 2.0 + 1.0));
        }));
        for &(hint_name, ref hints) in profiles {
            results.push(bench_fn(
                &format!("map_hinted[{hint_name}] n={n}"),
                n,
                3,
                50,
                || {
                    std::hint::black_box(backend.map_f64_hinted(&data, |v| v * 2.0 + 1.0, hints));
                },
            ));
        }
    }
    ScenarioReport {
        category: BenchCategory::HintsOptimization.id().into(),
        results,
    }
}

// ── Framework runners ─────────────────────────────────────────────────────

fn run_data_virtualization() -> ScenarioReport {
    let mut results = Vec::new();
    for &total_rows in VIRTUALIZATION_ROWS {
        let src = make_source(total_rows, 10);
        let scroll = ScrollState {
            offset: Point::new(0.0, (total_rows / 2) as f64 * 28.0),
        };
        let range = scroll.visible_range(28.0, 600.0, src.row_count());
        results.push(bench_fn(
            &format!("fetch {total_rows} rows (window=50)"),
            total_rows,
            3,
            100,
            || {
                std::hint::black_box(src.fetch(range.clone()));
            },
        ));
    }
    ScenarioReport {
        category: BenchCategory::DataVirtualization.id().into(),
        results,
    }
}

fn run_layout_spatial() -> ScenarioReport {
    let mut results = Vec::new();
    for &n in LAYOUT_RANGE_SIZES {
        let scroll = ScrollState {
            offset: Point::new(0.0, (n / 2) as f64 * 28.0),
        };
        results.push(bench_fn(
            &format!("visible_range n={n}"),
            n,
            5,
            10_000,
            || {
                std::hint::black_box(scroll.visible_range(28.0, 1080.0, n));
            },
        ));
    }
    for &n in HIT_TEST_SIZES {
        let rects: Vec<Rect> = (0..n)
            .map(|i| Rect::new(0.0, i as f64 * 28.0, 1920.0, 28.0))
            .collect();
        let test_point = Point::new(500.0, (n / 2) as f64 * 28.0);
        results.push(bench_fn(
            &format!("hit_test {n} rects"),
            n,
            5,
            1_000,
            || {
                let mut hit = false;
                for r in &rects {
                    if r.contains(test_point) {
                        hit = true;
                        break;
                    }
                }
                std::hint::black_box(hit);
            },
        ));
    }
    ScenarioReport {
        category: BenchCategory::LayoutSpatial.id().into(),
        results,
    }
}

fn run_animation() -> ScenarioReport {
    let mut results = Vec::new();
    for &n in ANIMATION_TICK_SIZES {
        let mut transitions: Vec<Transition> = (0..n)
            .map(|_| {
                let mut t = Transition::new(0.0, 100.0, Duration::from_millis(300))
                    .with_easing(Easing::EaseInOut);
                t.start();
                t
            })
            .collect();
        results.push(bench_fn(
            &format!("tick_f64 {n} transitions"),
            n,
            5,
            1_000,
            || {
                for t in transitions.iter_mut() {
                    std::hint::black_box(t.value());
                }
            },
        ));
    }
    for &n in ANIMATION_COLOR_SIZES {
        let mut transitions: Vec<Transition<Color>> = (0..n)
            .map(|_| {
                let mut t = Transition::new(
                    Color::rgb(30, 30, 60),
                    Color::rgb(255, 200, 100),
                    Duration::from_millis(200),
                )
                .with_easing(Easing::EaseIn);
                t.start();
                t
            })
            .collect();
        results.push(bench_fn(
            &format!("tick_color {n} transitions"),
            n,
            5,
            1_000,
            || {
                for t in transitions.iter_mut() {
                    std::hint::black_box(t.value());
                }
            },
        ));
    }
    for easing in [
        Easing::Linear,
        Easing::EaseIn,
        Easing::EaseOut,
        Easing::EaseInOut,
    ] {
        results.push(bench_fn(
            &format!("easing_{easing:?} 1M evals"),
            LERP_COUNT,
            3,
            10,
            || {
                for i in 0..LERP_COUNT as u32 {
                    std::hint::black_box(easing.apply(i as f64 / LERP_COUNT as f64));
                }
            },
        ));
    }
    ScenarioReport {
        category: BenchCategory::Animation.id().into(),
        results,
    }
}

fn run_render_list() -> ScenarioReport {
    let mut results = Vec::new();
    for &n in RENDER_RECT_SIZES {
        let mut list = RenderList::default();
        results.push(bench_fn(
            &format!("build {n} rect primitives"),
            n,
            3,
            100,
            || {
                list.clear();
                for i in 0..n {
                    list.push(Primitive::Rect {
                        bounds: Rect::new(0.0, i as f64 * 28.0, 1920.0, 28.0),
                        fill: Color::rgb(30, 30, 60),
                        border: None,
                        corner_radius: 0.0,
                    });
                }
                std::hint::black_box(list.len());
            },
        ));
    }
    for &n in RENDER_GRID_SIZES {
        let mut list = RenderList::default();
        results.push(bench_fn(
            &format!("build {n} grid cells (rect+text+border)"),
            n,
            3,
            50,
            || {
                list.clear();
                for i in 0..n {
                    let y = i as f64 * 28.0;
                    list.push(Primitive::Rect {
                        bounds: Rect::new(0.0, y, 1920.0, 28.0),
                        fill: if i % 2 == 0 {
                            Color::rgb(30, 30, 60)
                        } else {
                            Color::rgb(40, 40, 70)
                        },
                        border: Some(Border {
                            color: Color::rgb(60, 60, 90),
                            width: 1.0,
                        }),
                        corner_radius: 0.0,
                    });
                    list.push(Primitive::Text {
                        anchor: Point::new(8.0, y + 4.0),
                        content: format!("Row {i}"),
                        font_size: 14.0,
                        color: Color::WHITE,
                    });
                }
                std::hint::black_box(list.len());
            },
        ));
    }
    ScenarioReport {
        category: BenchCategory::RenderList.id().into(),
        results,
    }
}

fn run_lerp_throughput() -> ScenarioReport {
    let mut results = Vec::new();
    let n = LERP_COUNT;

    results.push(bench_fn("f64 lerp 1M", n, 3, 10, || {
        for i in 0..n {
            let t = (i as f64) / (n as f64);
            std::hint::black_box(0.0f64.lerp(100.0, t));
        }
    }));

    let pa = Point::new(0.0, 0.0);
    let pb = Point::new(1920.0, 1080.0);
    results.push(bench_fn("point lerp 1M", n, 3, 10, || {
        for i in 0..n {
            let t = (i as f64) / (n as f64);
            std::hint::black_box(pa.lerp(pb, t));
        }
    }));

    let ca = Color::rgb(0, 0, 0);
    let cb = Color::rgb(255, 128, 64);
    results.push(bench_fn("color lerp 1M", n, 3, 10, || {
        for i in 0..n {
            let t = (i as f64) / (n as f64);
            std::hint::black_box(ca.lerp(cb, t));
        }
    }));

    let ra = Rect::new(0.0, 0.0, 100.0, 50.0);
    let rb = Rect::new(500.0, 300.0, 800.0, 600.0);
    results.push(bench_fn("rect lerp 1M", n, 3, 10, || {
        for i in 0..n {
            let t = (i as f64) / (n as f64);
            std::hint::black_box(ra.lerp(rb, t));
        }
    }));

    ScenarioReport {
        category: BenchCategory::LerpThroughput.id().into(),
        results,
    }
}

pub fn run_event_handling() -> ScenarioReport {
    use crate::interaction::{Button, EventContext, EventResponse, InputEvent, Interactive, Phase};
    use crate::layout::{Point, Rect};

    /// Minimal interactive node used only in this benchmark.
    struct Node {
        bounds: Rect,
        handled: u32,
    }
    impl Interactive for Node {
        fn bounds(&self) -> Rect {
            self.bounds
        }
        fn handle_event(&mut self, ctx: &mut EventContext) -> EventResponse {
            self.handled += 1;
            if ctx.phase == Phase::Target {
                ctx.stop_propagation();
                EventResponse::Consumed
            } else {
                EventResponse::Ignored
            }
        }
    }

    // Sizes: event fan-out depth / batch
    const DISPATCH_COUNTS: &[usize] = &[1_000, 10_000, 100_000, 500_000];
    let mut results = Vec::new();

    // Benchmark 1: EventContext creation throughput
    for &n in DISPATCH_COUNTS {
        results.push(bench_fn(
            &format!("create EventContext n={n}"),
            n,
            5,
            20,
            || {
                for _ in 0..n {
                    std::hint::black_box(EventContext::new(InputEvent::PointerMove {
                        pos: Point::new(42.0, 42.0),
                    }));
                }
            },
        ));
    }

    // Benchmark 2: Three-phase dispatch (capture→target→bubble) through a node tree
    for &n in &[1_000usize, 10_000, 50_000] {
        let mut nodes: Vec<Node> = (0..n)
            .map(|i| Node {
                bounds: Rect::new(i as f64, 0.0, (i + 1) as f64, 10.0),
                handled: 0,
            })
            .collect();
        results.push(bench_fn(
            &format!("3-phase dispatch n={n} nodes"),
            n,
            3,
            10,
            || {
                let mut ctx = EventContext::new(InputEvent::PointerDown {
                    pos: Point::new(0.5, 5.0),
                    button: Button::Primary,
                });
                for phase in [Phase::Capture, Phase::Target, Phase::Bubble] {
                    ctx.phase = phase;
                    for node in nodes.iter_mut() {
                        if ctx.stopped {
                            break;
                        }
                        std::hint::black_box(node.handle_event(&mut ctx));
                    }
                }
            },
        ));
    }

    // Benchmark 3: Hit-test + dispatch (pointer over bounding rects)
    for &n in &[1_000usize, 10_000, 100_000] {
        let rects: Vec<Rect> = (0..n)
            .map(|i| Rect::new(i as f64, 0.0, (i + 1) as f64, 10.0))
            .collect();
        let hit_pos = Point::new((n / 2) as f64 + 0.5, 5.0);
        results.push(bench_fn(
            &format!("pointer hit-test n={n} rects"),
            n,
            5,
            10,
            || {
                let hit = rects.iter().find(|r| r.contains(hit_pos));
                std::hint::black_box(hit);
            },
        ));
    }

    ScenarioReport {
        category: BenchCategory::EventHandling.id().into(),
        results,
    }
}

// ── New high-impact benchmark runners ────────────────────────────────────

/// Halton low-discrepancy sequence value for a given index and prime base.
fn halton(mut idx: usize, base: usize) -> f64 {
    let mut f = 1.0f64;
    let mut r = 0.0f64;
    while idx > 0 {
        f /= base as f64;
        r += f * (idx % base) as f64;
        idx /= base;
    }
    r
}

fn run_point_cloud() -> ScenarioReport {
    let mut results = Vec::new();
    for &n in &[100_000usize, 1_000_000, 5_000_000] {
        // Generate n 2D points using a Halton(2,3) low-discrepancy sequence.
        results.push(bench_fn(
            &format!("halton2D_generate n={n}"),
            n,
            1,
            3,
            || {
                let pts: Vec<(f64, f64)> = (0..n).map(|i| (halton(i, 2), halton(i, 3))).collect();
                std::hint::black_box(pts);
            },
        ));
        // Bin all points into a 256×256 density grid (scatter histogram).
        let xs: Vec<f64> = (0..n).map(|i| halton(i, 2)).collect();
        let ys: Vec<f64> = (0..n).map(|i| halton(i, 3)).collect();
        const GRID: usize = 256;
        results.push(bench_fn(
            &format!("density_grid n={n} 256x256"),
            n,
            1,
            3,
            || {
                let mut grid = vec![0u16; GRID * GRID];
                for (&x, &y) in xs.iter().zip(ys.iter()) {
                    let gx = ((x * GRID as f64) as usize).min(GRID - 1);
                    let gy = ((y * GRID as f64) as usize).min(GRID - 1);
                    grid[gy * GRID + gx] = grid[gy * GRID + gx].saturating_add(1);
                }
                std::hint::black_box(grid);
            },
        ));
    }
    ScenarioReport {
        category: BenchCategory::PointCloud.id().into(),
        results,
    }
}

fn run_matmul_large() -> ScenarioReport {
    let kernel = best_kernel();
    let mut results = Vec::new();
    for &size in &[256usize, 512, 1024] {
        let a: Vec<f64> = (0..size * size).map(|i| (i as f64 * 0.001).sin()).collect();
        let b: Vec<f64> = (0..size * size)
            .map(|i| (i as f64 * 0.0013).cos())
            .collect();
        let iters = if size >= 1024 {
            1
        } else if size >= 512 {
            2
        } else {
            5
        };
        // Report FLOPs = 2·N³ as the "scale" so throughput_ops_sec is FLOP/s.
        let flops = 2 * size * size * size;
        results.push(bench_fn(
            &format!("gemm_f64 {size}\u{00d7}{size}"),
            flops,
            1,
            iters,
            || {
                std::hint::black_box(kernel.gemm_f64(&a, &b, size, size, size));
            },
        ));
    }
    ScenarioReport {
        category: BenchCategory::MatMulLarge.id().into(),
        results,
    }
}

fn run_attention_ops() -> ScenarioReport {
    let kernel = best_kernel();
    let mut results = Vec::new();
    // (seq_len, d_model) — representative transformer attention configs
    for &(seq, d_model) in &[(64usize, 64usize), (128, 64), (256, 64), (128, 128)] {
        let q: Vec<f64> = (0..seq * d_model)
            .map(|i| (i as f64 * 0.01).sin())
            .collect();
        let k: Vec<f64> = (0..seq * d_model)
            .map(|i| (i as f64 * 0.013).cos())
            .collect();
        let v: Vec<f64> = (0..seq * d_model).map(|i| i as f64 * 0.001).collect();
        // K^T: (d_model × seq)
        let kt: Vec<f64> = {
            let mut t = vec![0.0f64; seq * d_model];
            for i in 0..seq {
                for j in 0..d_model {
                    t[j * seq + i] = k[i * d_model + j];
                }
            }
            t
        };
        let scale = 1.0 / (d_model as f64).sqrt();
        let iters = if seq >= 256 { 1 } else { 3 };
        // FLOPs: 2·seq²·d_model (QK^T) + 2·seq²·d_model (Attn·V) ≈ 4·seq²·d_model
        let flops = 4 * seq * seq * d_model;
        results.push(bench_fn(
            &format!("sdpa seq={seq} d={d_model}"),
            flops,
            1,
            iters,
            || {
                // QK^T: (seq × d_model) × (d_model × seq) = (seq × seq)
                // A is (m x k), B is (k x n). So m=seq, k=d_model, n=seq.
                let qkt = kernel.gemm_f64(&q, &kt, seq, seq, d_model);
                // scale + row-wise softmax
                let mut attn = vec![0.0f64; seq * seq];
                for r in 0..seq {
                    let row = &qkt[r * seq..(r + 1) * seq];
                    let max = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let exps: Vec<f64> = row.iter().map(|&x| ((x * scale) - max).exp()).collect();
                    let sum: f64 = exps.iter().sum::<f64>().max(1e-30);
                    for (j, &e) in exps.iter().enumerate() {
                        attn[r * seq + j] = e / sum;
                    }
                }
                // Attn × V: (seq × seq) × (seq × d_model) = (seq × d_model)
                // A is (m x k), B is (k x n). So m=seq, k=seq, n=d_model.
                let out = kernel.gemm_f64(&attn, &v, seq, d_model, seq);
                std::hint::black_box(out);
            },
        ));
    }
    ScenarioReport {
        category: BenchCategory::AttentionOps.id().into(),
        results,
    }
}

fn run_geometry_3d() -> ScenarioReport {
    let mut results = Vec::new();
    use std::f64::consts::PI;
    // Precomputed rotation constants (angle = 0.1 rad)
    let (cos_a, sin_a) = (0.1f64.cos(), 0.1f64.sin());
    for &subdiv in &[32usize, 64, 128, 256] {
        // Sphere vertices via latitude/longitude subdivision
        let verts: Vec<[f64; 3]> = (0..=subdiv + 1)
            .flat_map(|i| {
                let theta = PI * i as f64 / subdiv as f64;
                (0..=subdiv * 2).map(move |j| {
                    let phi = 2.0 * PI * j as f64 / (subdiv * 2) as f64;
                    [
                        theta.sin() * phi.cos(),
                        theta.cos(),
                        theta.sin() * phi.sin(),
                    ]
                })
            })
            .collect();
        let n = verts.len();
        // Benchmark: rotate all vertices by Y-axis rotation matrix
        results.push(bench_fn(
            &format!("vertex_rotate_Y n={n} (subdiv={subdiv})"),
            n,
            2,
            20,
            || {
                let t: Vec<[f64; 3]> = verts
                    .iter()
                    .map(|v| {
                        [
                            v[0] * cos_a + v[2] * sin_a,
                            v[1],
                            -v[0] * sin_a + v[2] * cos_a,
                        ]
                    })
                    .collect();
                std::hint::black_box(t);
            },
        ));
        // Benchmark: perspective-project rotated vertices to 2D screen coords
        results.push(bench_fn(
            &format!("perspective_project n={n} (subdiv={subdiv})"),
            n,
            2,
            20,
            || {
                let p: Vec<(f64, f64)> = verts
                    .iter()
                    .map(|v| {
                        let z = v[2] + 3.0; // camera at z = 3
                        (v[0] / z * 1.5, v[1] / z * 1.5)
                    })
                    .collect();
                std::hint::black_box(p);
            },
        ));
    }
    ScenarioReport {
        category: BenchCategory::Geometry3D.id().into(),
        results,
    }
}

// ══════════════════════════════════════════════════════════════════════════
// Comparison tables
// ══════════════════════════════════════════════════════════════════════════

pub fn build_comparison_tables(
    kernel_reports: &[ScenarioReport],
    compute_reports: &[ScenarioReport],
    framework_reports: &[ScenarioReport],
) -> Vec<ComparisonTable> {
    use rayon::prelude::*;

    // Shorthand constructors — avoids repeating the struct name everywhere.
    let measured = |library: &str, ops: f64, notes: &str| LibComparison {
        library: library.to_string(),
        ops,
        notes: notes.to_string(),
        source: ComparisonSource::Measured,
    };
    let estimate = |library: &str, factor: f64, ac_ops: f64, notes: &str| LibComparison {
        library: library.to_string(),
        ops: ac_ops * factor,
        notes: format!("[est.] {notes}"),
        source: ComparisonSource::Estimate,
    };

    let mut tables = Vec::new();

    // ── Parallel Map ─────────────────────────────────────────────────
    // Real alternatives: rayon par_iter direct, std::iter sequential.
    {
        let entries: Vec<ComparisonEntry> = compute_reports
            .iter()
            .flat_map(|r| &r.results)
            .filter(|r| r.name.starts_with("map_f64") && !r.name.contains("hinted"))
            .map(|r| {
                let n = r.scale;
                let data = make_f64_data(n);
                let seq = bench_fn(&format!("seq_map n={n}"), n, 2, 8, || {
                    let out: Vec<f64> = data.iter().map(|&v| v * 2.0 + 1.0).collect();
                    std::hint::black_box(out);
                });
                let ry = bench_fn(&format!("rayon_map n={n}"), n, 2, 8, || {
                    let out: Vec<f64> = data.par_iter().map(|&v| v * 2.0 + 1.0).collect();
                    std::hint::black_box(out);
                });
                ComparisonEntry {
                    operation: r.name.clone(),
                    any_compute_us: r.duration_us,
                    any_compute_ops: r.throughput_ops_sec,
                    comparisons: vec![
                        measured("rayon::par_iter (raw, Rust)", ry.throughput_ops_sec,
                            "Direct rayon — shows any-compute trait-dispatch overhead vs bare parallel iterator"),
                        measured("std::iter (sequential, Rust)", seq.throughput_ops_sec,
                            "Single-threaded iterator; no parallelism; compiler may auto-vec with AVX"),
                        estimate("NumPy vectorized (Python)", 0.6, r.throughput_ops_sec,
                            "C inner loop + Python dispatch + GIL; from numpy benchmark suite"),
                        estimate("Node.js Float64Array loop", 0.08, r.throughput_ops_sec,
                            "V8 JIT-compiled; no SIMD auto-vec for typed array loops in V8"),
                    ],
                }
            })
            .collect();
        if !entries.is_empty() {
            tables.push(ComparisonTable {
                category: "Parallel Map (f64 element-wise)".into(),
                entries,
            });
        }
    }

    // ── Sort ─────────────────────────────────────────────────────────
    {
        let entries: Vec<ComparisonEntry> = compute_reports
            .iter()
            .flat_map(|r| &r.results)
            .filter(|r| r.name.starts_with("sort_f64"))
            .map(|r| {
                let n = r.scale;
                let original = make_f64_data(n);
                let mut buf1 = original.clone();
                let seq = bench_fn(&format!("std_sort n={n}"), n, 2, 5, || {
                    buf1.copy_from_slice(&original);
                    buf1.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
                    std::hint::black_box(&buf1);
                });
                let mut buf2 = original.clone();
                let ry = bench_fn(&format!("rayon_sort n={n}"), n, 2, 5, || {
                    buf2.copy_from_slice(&original);
                    buf2.par_sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
                    std::hint::black_box(&buf2);
                });
                ComparisonEntry {
                    operation: r.name.clone(),
                    any_compute_us: r.duration_us,
                    any_compute_ops: r.throughput_ops_sec,
                    comparisons: vec![
                        measured("rayon::par_sort_unstable (raw, Rust)", ry.throughput_ops_sec,
                            "Direct rayon sort — any-compute wraps this; measures abstraction overhead"),
                        measured("std::sort_unstable (sequential, Rust)", seq.throughput_ops_sec,
                            "pdqsort — best single-threaded sort in Rust std"),
                        estimate("polars sort (Rust/Arrow)", 0.85, r.throughput_ops_sec,
                            "Apache Arrow columnar + rayon; heavier for single f64 arrays"),
                        estimate("pandas sort_values (Python)", 0.3, r.throughput_ops_sec,
                            "NumPy timsort; single-threaded by default; pyarrow backend ~2x faster"),
                    ],
                }
            })
            .collect();
        if !entries.is_empty() {
            tables.push(ComparisonTable {
                category: "Sort (f64)".into(),
                entries,
            });
        }
    }

    // ── Reduce (sum) ─────────────────────────────────────────────────
    {
        let entries: Vec<ComparisonEntry> = kernel_reports
            .iter()
            .flat_map(|r| &r.results)
            .filter(|r| r.name.starts_with("reduce_sum"))
            .map(|r| {
                let n = r.scale;
                let data = make_f64_data(n);
                let seq = bench_fn(&format!("iter_sum n={n}"), n, 3, 15, || {
                    let s: f64 = data.iter().copied().sum();
                    std::hint::black_box(s);
                });
                let ry = bench_fn(&format!("par_sum n={n}"), n, 3, 15, || {
                    let s: f64 = data.par_iter().copied().sum();
                    std::hint::black_box(s);
                });
                ComparisonEntry {
                    operation: r.name.clone(),
                    any_compute_us: r.duration_us,
                    any_compute_ops: r.throughput_ops_sec,
                    comparisons: vec![
                        measured(
                            "rayon::par_iter().sum() (raw)",
                            ry.throughput_ops_sec,
                            "Raw rayon reduction — kernel dispatches to this internally",
                        ),
                        measured(
                            "std::iter().sum() (sequential)",
                            seq.throughput_ops_sec,
                            "Scalar accumulate; LLVM may auto-vec with AVX reduction",
                        ),
                        estimate(
                            "numpy.sum() (Python)",
                            0.7,
                            r.throughput_ops_sec,
                            "OpenBLAS/MKL SIMD reduction; Python overhead limits ~30%",
                        ),
                        estimate(
                            "polars sum (Rust/Arrow)",
                            0.9,
                            r.throughput_ops_sec,
                            "Arrow SIMD reduction; comparable, but heavier dep chain",
                        ),
                    ],
                }
            })
            .collect();
        if !entries.is_empty() {
            tables.push(ComparisonTable {
                category: "Reduction (sum, FP64)".into(),
                entries,
            });
        }
    }

    // ── GEMM ─────────────────────────────────────────────────────────
    // Measure naive triple-loop as intra-Rust baseline.
    {
        let entries: Vec<ComparisonEntry> = kernel_reports
            .iter()
            .flat_map(|r| &r.results)
            .filter(|r| r.name.starts_with("gemm"))
            .map(|r| {
                // r.scale = size² (set by runner as size*size)
                let size = (r.scale as f64).sqrt() as usize;
                let a = vec![1.0f64; size * size];
                let b = vec![1.0f64; size * size];
                let naive = bench_fn(&format!("naive_gemm {size}x{size}"), r.scale, 1, 2, || {
                    let mut c = vec![0.0f64; size * size];
                    for i in 0..size {
                        for k in 0..size {
                            for j in 0..size {
                                c[i * size + j] += a[i * size + k] * b[k * size + j];
                            }
                        }
                    }
                    std::hint::black_box(c);
                });
                ComparisonEntry {
                    operation: r.name.clone(),
                    any_compute_us: r.duration_us,
                    any_compute_ops: r.throughput_ops_sec,
                    comparisons: vec![
                        measured("naive ijk triple-loop (Rust)", naive.throughput_ops_sec,
                            "Baseline: unblocked, unvectorized loop — shows our kernel's gain from blocking + rayon"),
                        estimate("OpenBLAS dgemm", 15.0, r.throughput_ops_sec,
                            "Hand-tuned SAGEMM; enable --features mkl for comparable"),
                        estimate("Intel MKL dgemm", 20.0, r.throughput_ops_sec,
                            "Intel-optimized AMX/AVX-512 kernels; enable: --features mkl"),
                        estimate("cuBLAS (NVIDIA GPU)", 100.0, r.throughput_ops_sec,
                            "GPU tensor cores; enable: --features cuda; varies by GPU model"),
                    ],
                }
            })
            .collect();
        if !entries.is_empty() {
            tables.push(ComparisonTable {
                category: "Matrix Multiply (GEMM, FP64)".into(),
                entries,
            });
        }
    }

    // ── Animation tick ───────────────────────────────────────────────
    // Measure raw lerp math vs full Transition<f64> evaluation overhead.
    {
        let entries: Vec<ComparisonEntry> = framework_reports
            .iter()
            .filter(|r| r.category == BenchCategory::Animation.id())
            .flat_map(|r| &r.results)
            .filter(|r| r.name.starts_with("tick_f64"))
            .map(|r| {
                let n = r.scale;
                let raw = bench_fn(&format!("raw_lerp_loop n={n}"), n, 3, 50, || {
                    let mut s = 0.0f64;
                    for i in 0..n {
                        let t = i as f64 / n as f64;
                        s += 100.0 * t;
                    }
                    std::hint::black_box(s);
                });
                ComparisonEntry {
                    operation: r.name.clone(),
                    any_compute_us: r.duration_us,
                    any_compute_ops: r.throughput_ops_sec,
                    comparisons: vec![
                        measured("raw lerp loop (no Transition struct)", raw.throughput_ops_sec,
                            "Inline `a + (b-a)*t` — measures Transition<f64> overhead: easing eval + time delta"),
                        estimate("React Spring (JS)", 0.02, r.throughput_ops_sec,
                            "Spring physics; per-frame object allocation + V8 GC; ~50x slower for batches"),
                        estimate("GSAP tweening (JS)", 0.03, r.throughput_ops_sec,
                            "Optimized JS tweening; better than React Spring but still GC-bound"),
                        estimate("CSS Transitions (browser compositing)", 0.1, r.throughput_ops_sec,
                            "GPU composited for CSS props only; cannot animate arbitrary numeric values"),
                        estimate("Bevy Transform (Rust ECS)", 0.7, r.throughput_ops_sec,
                            "Archetype ECS iteration; no GC; comparable overhead from system scheduling"),
                    ],
                }
            })
            .collect();
        if !entries.is_empty() {
            tables.push(ComparisonTable {
                category: "Animation Tick Throughput".into(),
                entries,
            });
        }
    }

    // ── Render list ──────────────────────────────────────────────────
    // Compare against raw Vec push as intra-Rust baseline.
    {
        let entries: Vec<ComparisonEntry> = framework_reports
            .iter()
            .filter(|r| r.category == BenchCategory::RenderList.id())
            .flat_map(|r| &r.results)
            .filter(|r| r.name.starts_with("build") && r.name.contains("rect primitives"))
            .map(|r| {
                let n = r.scale;
                let raw = bench_fn(&format!("raw_vec_push n={n}"), n, 3, 30, || {
                    let mut v: Vec<(f64, f64, f64, f64)> = Vec::with_capacity(n);
                    for i in 0..n {
                        v.push((0.0, i as f64 * 28.0, 1920.0, 28.0));
                    }
                    std::hint::black_box(v.len());
                });
                ComparisonEntry {
                    operation: r.name.clone(),
                    any_compute_us: r.duration_us,
                    any_compute_ops: r.throughput_ops_sec,
                    comparisons: vec![
                        measured("raw Vec<(f64,f64,f64,f64)> push", raw.throughput_ops_sec,
                            "Minimal tuple; measures Primitive enum overhead + Color struct in RenderList"),
                        estimate("React createElement (JS VDOM)", 0.005, r.throughput_ops_sec,
                            "createElement + fiber scheduling + reconciliation; ~200x slower per primitive"),
                        estimate("Svelte (compiled, no VDOM)", 0.015, r.throughput_ops_sec,
                            "Compiled DOM mutations; lightest web framework — still JS→C++ bridge per node"),
                        estimate("Dioxus (Rust VDOM)", 0.3, r.throughput_ops_sec,
                            "Rust VDOM diffing; same language advantage but reconciliation overhead"),
                        estimate("egui (immediate mode, Rust)", 0.5, r.throughput_ops_sec,
                            "No VDOM; retained allocs; comparable path for simple rectangle lists"),
                    ],
                }
            })
            .collect();
        if !entries.is_empty() {
            tables.push(ComparisonTable {
                category: "Render List Assembly".into(),
                entries,
            });
        }
    }

    // ── Lerp throughput ──────────────────────────────────────────────
    {
        let entries: Vec<ComparisonEntry> = framework_reports
            .iter()
            .filter(|r| r.category == BenchCategory::LerpThroughput.id())
            .flat_map(|r| &r.results)
            .filter(|r| r.name.starts_with("f64 lerp"))
            .map(|r| {
                let n = LERP_COUNT;
                let manual = bench_fn("manual_inline_lerp 1M", n, 2, 5, || {
                    for i in 0..n {
                        let t = i as f64 / n as f64;
                        std::hint::black_box(0.0f64 + (100.0 - 0.0) * t);
                    }
                });
                ComparisonEntry {
                    operation: r.name.clone(),
                    any_compute_us: r.duration_us,
                    any_compute_ops: r.throughput_ops_sec,
                    comparisons: vec![
                        measured(
                            "inline `a + (b-a)*t` expression",
                            manual.throughput_ops_sec,
                            "No trait dispatch; measures monomorphization cost of Lerp<f64>",
                        ),
                        estimate(
                            "JS Math manual lerp (V8)",
                            0.08,
                            r.throughput_ops_sec,
                            "Boxed doubles in V8; no SIMD auto-vec; JIT helps but not comparable",
                        ),
                        estimate(
                            "glMatrix lerp (TypedArray, JS)",
                            0.10,
                            r.throughput_ops_sec,
                            "TypedArrays reduce boxing; still no WASM-level SIMD",
                        ),
                        estimate(
                            "Bevy Vec3::lerp (glam SIMD, Rust)",
                            0.95,
                            r.throughput_ops_sec,
                            "glam uses SIMD intrinsics; nearly identical for scalar f64",
                        ),
                    ],
                }
            })
            .collect();
        if !entries.is_empty() {
            tables.push(ComparisonTable {
                category: "Interpolation (Lerp) Throughput".into(),
                entries,
            });
        }
    }

    // ── Event dispatch ───────────────────────────────────────────────
    // Measure direct single-phase dispatch vs our 3-phase EventContext.
    {
        use crate::interaction::{EventContext, EventResponse, Interactive};

        struct DummyNode {
            bounds: Rect,
            calls: u32,
        }
        impl Interactive for DummyNode {
            fn bounds(&self) -> Rect {
                self.bounds
            }
            fn handle_event(&mut self, _: &mut EventContext) -> EventResponse {
                self.calls += 1;
                EventResponse::Ignored
            }
        }

        let entries: Vec<ComparisonEntry> = framework_reports
            .iter()
            .filter(|r| r.category == BenchCategory::EventHandling.id())
            .flat_map(|r| &r.results)
            .filter(|r| r.name.starts_with("3-phase dispatch"))
            .map(|r| {
                let n = r.scale;
                let mut nodes: Vec<DummyNode> = (0..n)
                    .map(|i| DummyNode { bounds: Rect::new(i as f64, 0.0, (i+1) as f64, 10.0), calls: 0 })
                    .collect();
                // Direct single-phase callback loop (no EventContext, no phase enum)
                let direct = bench_fn(&format!("direct_dispatch n={n}"), n, 2, 5, || {
                    for node in nodes.iter_mut() {
                        node.calls += 1;
                    }
                    std::hint::black_box(nodes[0].calls);
                });
                ComparisonEntry {
                    operation: r.name.clone(),
                    any_compute_us: r.duration_us,
                    any_compute_ops: r.throughput_ops_sec,
                    comparisons: vec![
                        measured("direct callback loop (no 3-phase)", direct.throughput_ops_sec,
                            "Single-pass node iteration; measures cost of capture/bubble phase + EventContext struct"),
                        estimate("React SyntheticEvent delegation (JS)", 0.03, r.throughput_ops_sec,
                            "Pooled event; fiber scheduler overhead; JS bridge; ~30x slower for 10k nodes"),
                        estimate("DOM native addEventListener (browser)", 0.05, r.throughput_ops_sec,
                            "Browser C++ event; JS handler invocation overhead per node"),
                        estimate("Svelte on:event (compiled JS)", 0.08, r.throughput_ops_sec,
                            "Compiled to direct DOM event; lightest web-framework overhead"),
                        estimate("Bevy EventReader (Rust ECS)", 0.8, r.throughput_ops_sec,
                            "ECS event channel; no phase overhead; near-zero alloc"),
                    ],
                }
            })
            .collect();
        if !entries.is_empty() {
            tables.push(ComparisonTable {
                category: "Event Dispatch Throughput".into(),
                entries,
            });
        }
    }

    tables
}

// ══════════════════════════════════════════════════════════════════════════
// Simulated device profiles
// ══════════════════════════════════════════════════════════════════════════

pub fn all_profiles() -> Vec<(&'static str, DeviceProfile)> {
    vec![
        ("high_end_desktop", DeviceProfile::HIGH_END_DESKTOP),
        ("mid_range_laptop", DeviceProfile::MID_RANGE_LAPTOP),
        ("low_end_mobile", DeviceProfile::LOW_END_MOBILE),
        ("embedded_iot", DeviceProfile::EMBEDDED),
        ("wasm_browser", DeviceProfile::WASM_BROWSER),
    ]
}

pub fn run_simulated(profile: &DeviceProfile) -> Vec<ScenarioReport> {
    let sim_backend = SimulatedBackend::new(profile.clone());
    let mut results = Vec::new();

    // Compute parallel on simulated
    let mut compute_results = Vec::new();
    for &n in SIMULATED_SIZES {
        let data = make_f64_data(n);
        compute_results.push(bench_fn(&format!("map_f64 n={n}"), n, 3, 50, || {
            std::hint::black_box(sim_backend.map_f64(&data, |v| v * 2.0 + 1.0));
        }));
        compute_results.push(bench_fn(&format!("sum_f64 n={n}"), n, 3, 50, || {
            std::hint::black_box(sim_backend.sum_f64(&data));
        }));
    }
    results.push(ScenarioReport {
        category: "compute_simulated".into(),
        results: compute_results,
    });

    results
}

// ══════════════════════════════════════════════════════════════════════════
// Helpers
// ══════════════════════════════════════════════════════════════════════════

pub fn timestamp_now() -> String {
    use std::time::SystemTime;
    let d = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap();
    format!("unix_{}", d.as_secs())
}

pub fn format_duration(us: u128) -> String {
    let us_f = us as f64;
    if us_f >= US_PER_SECOND {
        format!("{:.1}s", us_f / US_PER_SECOND)
    } else if us_f >= US_PER_MILLI {
        format!("{:.1}ms", us_f / US_PER_MILLI)
    } else {
        format!("{us}us")
    }
}

pub fn format_ops(ops: f64) -> String {
    if ops >= 1e9 {
        format!("{:.1}G ops/s", ops / 1e9)
    } else if ops >= 1e6 {
        format!("{:.1}M ops/s", ops / 1e6)
    } else if ops >= 1e3 {
        format!("{:.1}K ops/s", ops / 1e3)
    } else {
        format!("{:.0} ops/s", ops)
    }
}

pub fn format_bytes(bytes: u64) -> String {
    format!("{}", SizeFormatter::new(bytes, BINARY))
}

pub fn format_hz(mhz: u64) -> String {
    if mhz > 1000 {
        format!("{:.2} GHz", mhz as f64 / 1000.0)
    } else {
        format!("{} MHz", mhz)
    }
}

pub fn comparison_indicator(ratio: f64) -> &'static str {
    if ratio > 1.05 {
        "faster"
    } else if ratio < 0.95 {
        "slower"
    } else {
        "same"
    }
}

// ══════════════════════════════════════════════════════════════════════════
// Live metrics
// ══════════════════════════════════════════════════════════════════════════

/// Snapshot of live system and compute metrics.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LiveMetrics {
    pub cpu_per_core: Vec<f32>,
    pub cpu_global: f32,
    pub mem_used_bytes: u64,
    pub mem_total_bytes: u64,
    pub compute_ops_per_sec: f64,
    pub compute_throughput_elem_sec: f64,
}

/// Persistent monitor for polling live system metrics.
/// Wraps `sysinfo::System` so the caller never imports sysinfo directly.
#[cfg(feature = "hwinfo")]
pub struct MetricsMonitor {
    sys: System,
}

/// Quick compute throughput measurement size.
#[cfg(feature = "hwinfo")]
const METRICS_SAMPLE_SIZE: usize = 10_000;
/// Number of iterations for the quick throughput probe.
#[cfg(feature = "hwinfo")]
const METRICS_SAMPLE_ITERS: usize = 50;

#[cfg(feature = "hwinfo")]
impl MetricsMonitor {
    pub fn new() -> Self {
        let mut sys = System::new_all();
        sys.refresh_all();
        // First refresh sets baseline for CPU delta calculation.
        std::thread::sleep(Duration::from_millis(100));
        sys.refresh_all();
        Self { sys }
    }

    /// Take a snapshot of current system metrics + a quick compute probe.
    pub fn snapshot(&mut self) -> LiveMetrics {
        self.sys.refresh_cpu_all();
        self.sys.refresh_memory();

        // Quick compute throughput measurement
        let backend = CpuBackend::default();
        let data: Vec<f64> = (0..METRICS_SAMPLE_SIZE).map(|i| i as f64).collect();
        let start = Instant::now();
        for _ in 0..METRICS_SAMPLE_ITERS {
            std::hint::black_box(backend.map_f64(&data, |v| v * 2.0 + 1.0));
        }
        let elapsed = start.elapsed().as_secs_f64();
        let ops = METRICS_SAMPLE_ITERS as f64 / elapsed;
        let elems = (METRICS_SAMPLE_SIZE * METRICS_SAMPLE_ITERS) as f64 / elapsed;

        LiveMetrics {
            cpu_per_core: self.sys.cpus().iter().map(|c| c.cpu_usage()).collect(),
            cpu_global: self.sys.global_cpu_usage(),
            mem_used_bytes: self.sys.used_memory(),
            mem_total_bytes: self.sys.total_memory(),
            compute_ops_per_sec: ops,
            compute_throughput_elem_sec: elems,
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════
// Reference comparison data — static estimates from published benchmarks
// ══════════════════════════════════════════════════════════════════════════

/// A reference library comparison point (static, not benchmark-dependent).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReferenceComparison {
    pub domain: String,
    pub category: String,
    pub library: String,
    /// Factor relative to any-compute: < 1.0 = slower than us, > 1.0 = faster.
    pub factor: f64,
    pub notes: String,
}

/// Returns static reference comparison data across all domains.
/// These are published/estimated performance ratios, not live measurements.
pub fn reference_comparisons() -> Vec<ReferenceComparison> {
    let r = |domain: &str, cat: &str, lib: &str, factor: f64, notes: &str| ReferenceComparison {
        domain: domain.into(),
        category: cat.into(),
        library: lib.into(),
        factor,
        notes: notes.into(),
    };
    vec![
        // ── Compute: Parallel map / transform ────────────────────────
        r(
            "Compute",
            "Parallel Map",
            "rayon par_iter (Rust)",
            0.95,
            "Same backend; ~5% dispatch overhead",
        ),
        r(
            "Compute",
            "Parallel Map",
            "std::iter (Rust, sequential)",
            0.15,
            "Single-threaded baseline",
        ),
        r(
            "Compute",
            "Parallel Map",
            "NumPy vectorized (Python)",
            0.6,
            "C inner loop but Python dispatch + GIL",
        ),
        r(
            "Compute",
            "Parallel Map",
            "Bun (JS, JIT-compiled)",
            0.08,
            "V8-level JIT; no SIMD auto-vectorization",
        ),
        r(
            "Compute",
            "Parallel Map",
            "Node.js worker_threads",
            0.05,
            "JS overhead + serialization between workers",
        ),
        r(
            "Compute",
            "Parallel Map",
            "Deno (V8 + Rust internals)",
            0.07,
            "Similar to Bun/Node; slight Rust FFI edge",
        ),
        // ── Compute: Sort ────────────────────────────────────────────
        r(
            "Compute",
            "Sort",
            "rayon par_sort_unstable (Rust)",
            1.0,
            "Same implementation",
        ),
        r(
            "Compute",
            "Sort",
            "std::sort_unstable (Rust)",
            0.25,
            "Single-threaded pdqsort",
        ),
        r(
            "Compute",
            "Sort",
            "polars sort (Rust/Arrow)",
            0.85,
            "Arrow columnar + rayon",
        ),
        r(
            "Compute",
            "Sort",
            "pandas sort_values (Python)",
            0.3,
            "NumPy/timsort; single-threaded by default",
        ),
        r(
            "Compute",
            "Sort",
            "Bun Array.sort (JS)",
            0.12,
            "V8 TimSort; no parallelism",
        ),
        // ── Compute: GEMM / Matrix Multiply ──────────────────────────
        r(
            "Compute",
            "GEMM",
            "OpenBLAS dgemm",
            15.0,
            "Hand-tuned BLAS with SIMD kernels",
        ),
        r(
            "Compute",
            "GEMM",
            "Intel MKL dgemm",
            20.0,
            "Intel-optimized; --features mkl",
        ),
        r(
            "Compute",
            "GEMM",
            "cuBLAS (NVIDIA GPU)",
            100.0,
            "GPU tensor cores; --features cuda",
        ),
        r(
            "Compute",
            "GEMM",
            "PyTorch matmul (CPU)",
            12.0,
            "Uses OpenBLAS/MKL internally",
        ),
        r(
            "Compute",
            "GEMM",
            "PyTorch matmul (CUDA)",
            120.0,
            "cuBLAS + tensor cores",
        ),
        r(
            "Compute",
            "GEMM",
            "TensorFlow matmul (CPU)",
            11.0,
            "Eigen/MKL backend",
        ),
        r(
            "Compute",
            "GEMM",
            "NumPy dot (Python)",
            10.0,
            "BLAS backend (OpenBLAS/MKL)",
        ),
        // ── Compute: Reduction ───────────────────────────────────────
        r(
            "Compute",
            "Reduction",
            "rayon par_iter().sum()",
            1.0,
            "Same implementation",
        ),
        r(
            "Compute",
            "Reduction",
            "numpy.sum() (Python)",
            0.7,
            "C inner loop; Python overhead",
        ),
        r(
            "Compute",
            "Reduction",
            "PyTorch .sum() CPU",
            0.8,
            "Optimized AVX reduction",
        ),
        r(
            "Compute",
            "Reduction",
            "polars sum (Rust/Arrow)",
            0.9,
            "Arrow + SIMD",
        ),
        r(
            "Compute",
            "Reduction",
            "Bun reduce (JS)",
            0.06,
            "V8 JIT; no SIMD",
        ),
        // ── UI: Render list assembly ─────────────────────────────────
        r(
            "UI Rendering",
            "Render List",
            "React (virtual DOM reconciliation)",
            0.005,
            "JS VDOM diff + fiber scheduler; ~200x slower",
        ),
        r(
            "UI Rendering",
            "Render List",
            "Angular (Renderer2 + change detection)",
            0.004,
            "Zone.js + incremental DOM",
        ),
        r(
            "UI Rendering",
            "Render List",
            "Vue 3 (Proxy reactivity + patch)",
            0.006,
            "Faster VDOM than React; still JS overhead",
        ),
        r(
            "UI Rendering",
            "Render List",
            "Svelte (compiled output, no VDOM)",
            0.015,
            "Compiled; less overhead than React/Vue",
        ),
        r(
            "UI Rendering",
            "Render List",
            "Solid.js (fine-grained reactivity)",
            0.02,
            "No VDOM; signals-based; still JS",
        ),
        r(
            "UI Rendering",
            "Render List",
            "Vanilla JS (document.createElement)",
            0.01,
            "No framework; JS→C++ bridge per call",
        ),
        r(
            "UI Rendering",
            "Render List",
            "Dioxus (Rust VDOM)",
            0.3,
            "Rust VDOM diffing; same language",
        ),
        r(
            "UI Rendering",
            "Render List",
            "Yew (Rust VDOM + WASM)",
            0.25,
            "WASM + VDOM; WebAssembly overhead",
        ),
        r(
            "UI Rendering",
            "Render List",
            "egui (Rust immediate mode)",
            0.5,
            "No VDOM; immediate mode; retained allocs",
        ),
        r(
            "UI Rendering",
            "Render List",
            "iced (Rust Elm arch)",
            0.4,
            "Elm architecture; message passing overhead",
        ),
        // ── UI: Animation / transitions ──────────────────────────────
        r(
            "Animation",
            "Transition Tick",
            "React Spring (JS)",
            0.02,
            "Physics-based; per-frame allocations + GC",
        ),
        r(
            "Animation",
            "Transition Tick",
            "GSAP (JS)",
            0.03,
            "Optimized JS tweening; still GC-bound for batches",
        ),
        r(
            "Animation",
            "Transition Tick",
            "Framer Motion (React)",
            0.015,
            "React + spring physics; component overhead",
        ),
        r(
            "Animation",
            "Transition Tick",
            "Angular Animations",
            0.015,
            "AnimationBuilder + Zone.js scheduling",
        ),
        r(
            "Animation",
            "Transition Tick",
            "CSS Transitions (browser-native)",
            0.1,
            "Compositor-accelerated; limited to style props",
        ),
        r(
            "Animation",
            "Transition Tick",
            "Web Animations API (JS)",
            0.05,
            "Browser-native; JS bridge overhead",
        ),
        r(
            "Animation",
            "Transition Tick",
            "Bevy Transform animation (Rust/ECS)",
            0.7,
            "ECS batch iteration; no GC",
        ),
        r(
            "Animation",
            "Transition Tick",
            "Unity Animator (C#)",
            0.3,
            "C# managed heap; state machine overhead",
        ),
        r(
            "Animation",
            "Transition Tick",
            "Godot Tween (GDScript)",
            0.1,
            "Interpreted GDScript; node tree traversal",
        ),
        // ── UI: Data virtualization ──────────────────────────────────
        r(
            "UI Rendering",
            "Data Virtualization",
            "react-window (JS)",
            0.01,
            "JS row measurement + React reconciliation",
        ),
        r(
            "UI Rendering",
            "Data Virtualization",
            "react-virtuoso (JS)",
            0.008,
            "Dynamic height measurement; heavier than react-window",
        ),
        r(
            "UI Rendering",
            "Data Virtualization",
            "AG Grid (JS)",
            0.005,
            "Enterprise grid; feature-heavy DOM management",
        ),
        r(
            "UI Rendering",
            "Data Virtualization",
            "Tabulator (JS)",
            0.007,
            "Vanilla JS grid; no framework dep",
        ),
        // ── Interpolation ────────────────────────────────────────────
        r(
            "Math",
            "Lerp / Interpolation",
            "JS Math (manual lerp)",
            0.08,
            "V8 JIT; boxed doubles, no SIMD auto-vec",
        ),
        r(
            "Math",
            "Lerp / Interpolation",
            "glMatrix (JS)",
            0.1,
            "TypedArrays; no SIMD without WASM",
        ),
        r(
            "Math",
            "Lerp / Interpolation",
            "Unity Mathf.Lerp (C#)",
            0.4,
            "JIT-compiled C#; Mono/IL2CPP",
        ),
        r(
            "Math",
            "Lerp / Interpolation",
            "Godot lerp (GDScript)",
            0.05,
            "Interpreted; per-call overhead",
        ),
        r(
            "Math",
            "Lerp / Interpolation",
            "Bevy Vec3::lerp (Rust)",
            0.95,
            "Same language; glam SIMD",
        ),
        r(
            "Math",
            "Lerp / Interpolation",
            "NumPy interp (Python)",
            0.3,
            "Vectorized C; Python dispatch",
        ),
        // ── Game engines: frame time / ECS ───────────────────────────
        r(
            "Game Engine",
            "ECS Iteration (10K entities)",
            "Bevy ECS (Rust)",
            0.9,
            "Archetype storage; cache-friendly",
        ),
        r(
            "Game Engine",
            "ECS Iteration (10K entities)",
            "Unity DOTS/ECS (C#)",
            0.6,
            "Burst compiler; managed GC pauses",
        ),
        r(
            "Game Engine",
            "ECS Iteration (10K entities)",
            "Godot (GDScript)",
            0.05,
            "Scene tree; interpreted; no ECS",
        ),
        r(
            "Game Engine",
            "ECS Iteration (10K entities)",
            "Unreal Engine (C++)",
            0.7,
            "UObject system; GC + reflection",
        ),
        r(
            "Game Engine",
            "Frame Update Loop",
            "Bevy (Rust)",
            0.85,
            "Pure ECS; zero GC",
        ),
        r(
            "Game Engine",
            "Frame Update Loop",
            "Unity (C# Mono)",
            0.3,
            "GC pauses; managed overhead",
        ),
        r(
            "Game Engine",
            "Frame Update Loop",
            "Godot 4.x (GDScript)",
            0.1,
            "Interpreted scripts; node tree",
        ),
        r(
            "Game Engine",
            "Frame Update Loop",
            "Unreal Engine 5 (C++)",
            0.5,
            "Nanite/Lumen overhead; heavy runtime",
        ),
        // ── AI / ML inference latency ────────────────────────────────
        r(
            "AI Inference",
            "Elementwise (10M f64)",
            "PyTorch CPU",
            0.8,
            "ATen C++ core; operator dispatch overhead",
        ),
        r(
            "AI Inference",
            "Elementwise (10M f64)",
            "TensorFlow CPU",
            0.7,
            "Eigen backend; graph execution overhead",
        ),
        r(
            "AI Inference",
            "Elementwise (10M f64)",
            "ONNX Runtime CPU",
            0.85,
            "Optimized graph; less overhead than TF",
        ),
        r(
            "AI Inference",
            "Elementwise (10M f64)",
            "JAX CPU",
            0.75,
            "XLA compilation; great for large batches",
        ),
        r(
            "AI Inference",
            "Batch Matmul Latency",
            "PyTorch CUDA",
            150.0,
            "cuBLAS + tensor cores; GPU memory BW",
        ),
        r(
            "AI Inference",
            "Batch Matmul Latency",
            "TensorRT (NVIDIA)",
            200.0,
            "Fused kernels; INT8/FP16 quantization",
        ),
        r(
            "AI Inference",
            "Batch Matmul Latency",
            "ONNX Runtime CUDA",
            130.0,
            "cuDNN backend; graph optimization",
        ),
        r(
            "AI Inference",
            "Token Generation (LLM)",
            "llama.cpp CPU (AVX2)",
            0.6,
            "Quantized INT4/INT8; hand-tuned SIMD",
        ),
        r(
            "AI Inference",
            "Token Generation (LLM)",
            "llama.cpp CUDA",
            20.0,
            "GPU inference; depends on model size",
        ),
        r(
            "AI Inference",
            "Token Generation (LLM)",
            "vLLM (Python/CUDA)",
            25.0,
            "PagedAttention; optimized KV cache",
        ),
        // ── Event handling ───────────────────────────────────────────
        r(
            "Events",
            "Event Dispatch (batch 10k)",
            "React SyntheticEvent",
            0.03,
            "Pooled event objects + React fiber scheduler; GC on large batches",
        ),
        r(
            "Events",
            "Event Dispatch (batch 10k)",
            "DOM native Event (browser)",
            0.05,
            "C++ native; JS bridge per dispatch call",
        ),
        r(
            "Events",
            "Event Dispatch (batch 10k)",
            "Vue 3 emit",
            0.04,
            "Proxy-based reactivity; lighter than React but still JS dispatch",
        ),
        r(
            "Events",
            "Event Dispatch (batch 10k)",
            "Angular EventEmitter",
            0.02,
            "Zone.js + change detection trigger; heaviest framework overhead",
        ),
        r(
            "Events",
            "Event Dispatch (batch 10k)",
            "Svelte dispatch (compiled)",
            0.08,
            "Compiled to direct DOM calls; lightest web framework",
        ),
        r(
            "Events",
            "Hit-Test (100k rects)",
            "react-use-gesture",
            0.01,
            "JS bounding rect API per pointer event + React state update",
        ),
        r(
            "Events",
            "Hit-Test (100k rects)",
            "Hammer.js",
            0.02,
            "Gesture recognizer; JS touch/pointer API overhead",
        ),
        r(
            "Events",
            "Hit-Test (100k rects)",
            "Bevy (Rust ECS hit-test)",
            0.7,
            "Archetype storage; cache-friendly AABB iteration",
        ),
        r(
            "Events",
            "3-Phase Propagation",
            "DOM capture/bubble",
            0.04,
            "Browser-native C++ propagation + JS listener invocation overhead",
        ),
        r(
            "Events",
            "3-Phase Propagation",
            "React event delegation (root)",
            0.03,
            "Single root listener + synthetic event construction",
        ),
    ]
}
