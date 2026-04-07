use serde::{Deserialize, Serialize};

// ══════════════════════════════════════════════════════════════════════════
// Constants — benchmark data sizes, memory thresholds, formatting
// ══════════════════════════════════════════════════════════════════════════

/// Standard data sizes for element-wise kernel benchmarks.
pub(super) const KERNEL_SIZES: &[usize] = &[100_000, 1_000_000, 10_000_000];

/// Matrix dimensions for GEMM benchmarks.
pub(super) const GEMM_DIMS: &[usize] = &[64, 128, 256, 512];

/// Data sizes for sort benchmarks.
pub(super) const SORT_SIZES: &[usize] = &[10_000, 100_000, 1_000_000];

/// Data sizes for compute backend parallel operation benchmarks.
pub(super) const COMPUTE_SIZES: &[usize] = &[10_000, 100_000, 1_000_000, 10_000_000];

/// Data sizes for hints-aware dispatch benchmarks.
pub(super) const HINTS_SIZES: &[usize] = &[1_000, 100_000, 1_000_000];

/// Row counts for data virtualization benchmarks.
pub(super) const VIRTUALIZATION_ROWS: &[usize] = &[1_000, 100_000, 1_000_000, 10_000_000];

/// Row counts for visible_range layout benchmarks.
pub(super) const LAYOUT_RANGE_SIZES: &[usize] = &[100_000, 1_000_000, 10_000_000, 100_000_000];

/// Rectangle counts for hit-test benchmarks.
pub(super) const HIT_TEST_SIZES: &[usize] = &[1_000, 10_000, 100_000];

/// Transition counts for animation tick benchmarks.
pub(super) const ANIMATION_TICK_SIZES: &[usize] = &[100, 1_000, 10_000, 50_000];

/// Transition counts for color animation benchmarks.
pub(super) const ANIMATION_COLOR_SIZES: &[usize] = &[1_000, 10_000];

/// Primitive counts for render list benchmarks.
pub(super) const RENDER_RECT_SIZES: &[usize] = &[1_000, 10_000, 50_000, 100_000];

/// Primitive counts for grid cell (rect+text+border) benchmarks.
pub(super) const RENDER_GRID_SIZES: &[usize] = &[1_000, 10_000, 50_000];

/// Item count for lerp throughput and easing benchmarks.
pub(super) const LERP_COUNT: usize = 1_000_000;

/// Data sizes for simulated backend benchmarks.
pub(super) const SIMULATED_SIZES: &[usize] = &[10_000, 100_000, 1_000_000];

/// RAM threshold for memory bandwidth estimation heuristic.
pub(super) const HIGH_BW_RAM_THRESHOLD: u64 = 32 * 1024 * 1024 * 1024;

/// Estimated memory bandwidth (GB/s) for systems above/below the RAM threshold.
pub(super) const MEM_BW_HIGH: f64 = 60.0;
pub(super) const MEM_BW_LOW: f64 = 40.0;

/// Microsecond thresholds for duration formatting.
pub(super) const US_PER_SECOND: f64 = 1_000_000.0;
pub(super) const US_PER_MILLI: f64 = 1_000.0;

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
