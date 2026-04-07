//! Benchmark runner library — shared by CLI (`anc-bench`) and GUI (`anc-bench-window`).
//!
//! All types are `Clone + Serialize` so they can be displayed, streamed, or
//! written to JSON.  Runner functions are pure: they return results without
//! printing anything.

use any_compute_core::Lerp;
use any_compute_core::animation::{Easing, Transition};
use any_compute_core::compute::{Device, DeviceProfile};
use any_compute_core::data::{CellValue, ColumnKind, ColumnMeta, DataSource, VecSource};
use any_compute_core::hints::Hints;
use any_compute_core::kernel::{BinaryOp, ReduceOp, UnaryOp, best_kernel};
#[cfg(feature = "hwinfo")]
use any_compute_core::kernel::{CpuSimdKernel, Kernel};
use any_compute_core::layout::{Point, Rect, ScrollState};
use any_compute_core::render::{Border, Color, Primitive, RenderList};
use humansize::{BINARY, SizeFormatter};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
#[cfg(feature = "hwinfo")]
use sysinfo::System;

mod comparison;
mod report;

pub use comparison::*;
pub use report::*;

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
    let (ops_sec, us_per) = crate::bench_throughput(warmup as u32, iters as u32, &mut f);
    BenchResult {
        name: name.to_string(),
        scale,
        duration_us: us_per as u128,
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
    let f = any_compute_core::FEATURES;
    FeaturesReport {
        cuda: f.cuda,
        rocm: f.rocm,
        mkl: f.mkl,
        metal: f.metal,
        wgpu: f.wgpu,
        shader: f.shader,
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
// ══════════════════════════════════════════════════════════════════════════
// Benchmark categories — macro-generated enum, metadata, and dispatch
// ══════════════════════════════════════════════════════════════════════════

/// Declares `BenchCategory` enum + `ALL`, five accessor methods, and `run_category`.
/// Adding a category = adding one block. No other match arms to update.
macro_rules! bench_categories {
    ($( $(#[doc = $doc:literal])* $variant:ident {
        id: $id:literal,
        label: $label:literal,
        group: $group:literal,
        domain: $domain:literal,
        desc: $desc:literal,
        runner: $runner:ident,
    }),+ $(,)?) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
        pub enum BenchCategory { $( $(#[doc = $doc])* $variant, )+ }

        impl BenchCategory {
            pub const ALL: &[Self] = &[ $( Self::$variant, )+ ];
            pub fn id(self) -> &'static str { match self { $( Self::$variant => $id, )+ } }
            pub fn label(self) -> &'static str { match self { $( Self::$variant => $label, )+ } }
            pub fn group(self) -> &'static str { match self { $( Self::$variant => $group, )+ } }
            pub fn domain(self) -> &'static str { match self { $( Self::$variant => $domain, )+ } }
            pub fn description(self) -> &'static str { match self { $( Self::$variant => $desc, )+ } }
        }

        pub fn run_category(cat: BenchCategory) -> ScenarioReport {
            match cat { $( BenchCategory::$variant => $runner(), )+ }
        }
    };
}

bench_categories! {
    KernelUnary {
        id: "kernel_unary", label: "Kernel: Unary Ops", group: "Kernel",
        domain: "Compute", desc: "abs, negate, sqrt on 10K-10M f64 vectors via SIMD dispatch",
        runner: run_kernel_unary,
    },
    KernelBinary {
        id: "kernel_binary", label: "Kernel: Binary Ops", group: "Kernel",
        domain: "Compute", desc: "add, mul, fma on paired vectors; measures SIMD throughput",
        runner: run_kernel_binary,
    },
    KernelReduce {
        id: "kernel_reduce", label: "Kernel: Reductions", group: "Kernel",
        domain: "Compute", desc: "sum, min, max reductions with SIMD accumulator",
        runner: run_kernel_reduce,
    },
    KernelGemm {
        id: "kernel_gemm", label: "Kernel: GEMM", group: "Kernel",
        domain: "Linear Algebra / AI", desc: "naive matrix multiply 64-512; baseline for BLAS comparison",
        runner: run_kernel_gemm,
    },
    KernelSort {
        id: "kernel_sort", label: "Kernel: Sort", group: "Kernel",
        domain: "Compute", desc: "parallel sort on 10K-10M elements via rayon + pdqsort",
        runner: run_kernel_sort,
    },
    ComputeParallel {
        id: "compute_parallel", label: "Compute: Parallel Ops", group: "Compute",
        domain: "Compute", desc: "parallel map/reduce across all cores; rayon dispatch",
        runner: run_compute_parallel,
    },
    HintsOptimization {
        id: "hints_auto_optimization", label: "Compute: Hint-Aware Dispatch", group: "Compute",
        domain: "Compute", desc: "hint-aware dispatch: sorted, dense, contiguous flags",
        runner: run_hints_optimization,
    },
    DataVirtualization {
        id: "data_virtualization", label: "Data: Virtualization", group: "Data",
        domain: "Rendering / Data", desc: "virtual scroll window over 100K-1M row data sources",
        runner: run_data_virtualization,
    },
    LayoutSpatial {
        id: "layout_spatial", label: "Layout: Spatial", group: "Layout",
        domain: "Graphics / 3D", desc: "AABB spatial grid insert + range query at scale",
        runner: run_layout_spatial,
    },
    Animation {
        id: "animation", label: "Animation: Transitions", group: "Animation",
        domain: "Animation / Dynamics", desc: "easing transitions: linear, ease-in-out, spring physics",
        runner: run_animation,
    },
    RenderList {
        id: "render_list", label: "Render: Primitive Lists", group: "Render",
        domain: "Rendering / Data", desc: "batch render-primitive assembly: rects, circles, text",
        runner: run_render_list,
    },
    LerpThroughput {
        id: "lerp_throughput", label: "Lerp: Interpolation Throughput", group: "Lerp",
        domain: "Animation / Dynamics", desc: "raw lerp throughput on f64, Vec3, Color, Rect batches",
        runner: run_lerp_throughput,
    },
    EventHandling {
        id: "event_handling", label: "Events: Input Dispatch", group: "Events",
        domain: "Events / Interaction", desc: "event dispatch, hit-testing 100K rects, 3-phase propagation",
        runner: run_event_handling,
    },
    /// 1M point generation + 256×256 density binning.
    PointCloud {
        id: "point_cloud", label: "Point Cloud: 1M Scatter + Density", group: "Geometry",
        domain: "Graphics / 3D", desc: "1-5M point Halton generation + 256x256 density binning",
        runner: run_point_cloud,
    },
    /// Large GEMM: 256×256 up to 1024×1024.
    MatMulLarge {
        id: "matmul_large", label: "MatMul: Large GEMM (up to 1024×1024)", group: "Compute",
        domain: "Linear Algebra / AI", desc: "GEMM at 256-1024 sizes; reports GFLOP/s",
        runner: run_matmul_large,
    },
    /// Scaled dot-product attention (QKᵀ → softmax → V) at multiple seq/d configs.
    AttentionOps {
        id: "attention_ops", label: "Attention: Scaled Dot-Product", group: "Compute",
        domain: "Linear Algebra / AI", desc: "scaled dot-product attention QK^T -> softmax -> V",
        runner: run_attention_ops,
    },
    /// Sphere vertex transforms + perspective projection at multiple mesh resolutions.
    Geometry3D {
        id: "geometry_3d", label: "3D Geometry: Vertex Transform + Projection", group: "Geometry",
        domain: "Graphics / 3D", desc: "sphere mesh vertex transforms + perspective projection",
        runner: run_geometry_3d,
    },
    /// HTML parse throughput: small doc, website fixture, with/without CSS.
    DomParse {
        id: "dom_parse", label: "DOM: HTML/CSS Parse", group: "DOM",
        domain: "DOM / Layout", desc: "HTML parse throughput: small doc, full website, with/without CSS resolve",
        runner: run_dom_parse,
    },
    /// Full website frame: parse + layout + paint pipeline.
    DomFullFrame {
        id: "dom_full_frame", label: "DOM: Full Frame Pipeline", group: "DOM",
        domain: "DOM / Layout", desc: "full frame pipeline: parse HTML+CSS → layout → paint on website fixture",
        runner: run_dom_full_frame,
    },
}

impl BenchCategory {
    /// All unique domains, ordered for display.
    pub fn all_domains() -> &'static [&'static str] {
        &[
            "Compute",
            "Linear Algebra / AI",
            "Graphics / 3D",
            "Animation / Dynamics",
            "Rendering / Data",
            "Events / Interaction",
            "DOM / Layout",
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
    let device = Device::cpu();
    let mut results = Vec::new();
    for &n in COMPUTE_SIZES {
        let data = make_f64_data(n);
        results.push(bench_fn(&format!("map n={n}"), n, 3, 50, || {
            std::hint::black_box(device.map(&data, |v| v * 2.0 + 1.0));
        }));
        results.push(bench_fn(&format!("filter n={n}"), n, 3, 50, || {
            std::hint::black_box(device.filter(&data, |v| v > 500.0));
        }));
        results.push(bench_fn(&format!("sum n={n}"), n, 3, 50, || {
            std::hint::black_box(device.sum(&data));
        }));
        results.push(bench_fn(&format!("prefix_sum n={n}"), n, 3, 30, || {
            std::hint::black_box(device.prefix_sum(&data));
        }));
        let mut sort_data = data.clone();
        results.push(bench_fn(&format!("sort n={n}"), n, 3, 20, || {
            sort_data.copy_from_slice(&data);
            device.sort(&mut sort_data);
            std::hint::black_box(&sort_data);
        }));
    }
    ScenarioReport {
        category: BenchCategory::ComputeParallel.id().into(),
        results,
    }
}

fn run_hints_optimization() -> ScenarioReport {
    let device = Device::cpu();
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
            std::hint::black_box(device.map(&data, |v| v * 2.0 + 1.0));
        }));
        for &(hint_name, ref hints) in profiles {
            results.push(bench_fn(
                &format!("map_hinted[{hint_name}] n={n}"),
                n,
                3,
                50,
                || {
                    std::hint::black_box(device.map_hinted(&data, |v| v * 2.0 + 1.0, hints));
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
                        border: Some(Border::uniform(Color::rgb(60, 60, 90), 1.0)),
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
    use any_compute_core::interaction::{
        Button, EventContext, EventResponse, InputEvent, Interactive, Phase,
    };
    use any_compute_core::layout::{Point, Rect};

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

// ── DOM runners ──────────────────────────────────────────────────────────

fn run_dom_parse() -> ScenarioReport {
    use any_compute_dom::css::StyleSheet;
    use any_compute_dom::parse;

    let small_html = r##"<div w="1400" h="900" direction="row"><div w="220" pad="12" gap="8"><span font="16">Sidebar</span></div><div grow="1" pad="24" gap="16"><span font="22">Main</span><progress value="0.6" color="#a6e3a1" h="8" /></div></div>"##;
    let website_html = crate::WEBSITE_HTML;
    let website_css = crate::WEBSITE_CSS;
    let sheet = StyleSheet::parse(website_css);

    let mut results = Vec::new();

    results.push(bench_fn("html_parse small (6 nodes)", 6, 3, 500, || {
        std::hint::black_box(parse::parse(small_html));
    }));

    let tree = parse::parse(website_html);
    let n = tree.arena.len();

    results.push(bench_fn(
        &format!("html_parse website ({n} nodes)"),
        n,
        3,
        500,
        || {
            std::hint::black_box(parse::parse(website_html));
        },
    ));

    results.push(bench_fn("css_parse website.css", 0, 3, 500, || {
        std::hint::black_box(StyleSheet::parse(website_css));
    }));

    results.push(bench_fn(
        &format!("html+css parse website ({n} nodes)"),
        n,
        3,
        500,
        || {
            std::hint::black_box(parse::parse_with_css(website_html, &sheet));
        },
    ));

    ScenarioReport {
        category: BenchCategory::DomParse.id().into(),
        results,
    }
}

fn run_dom_full_frame() -> ScenarioReport {
    use any_compute_core::render::RenderList;
    use any_compute_dom::css::StyleSheet;
    use any_compute_dom::parse;

    let website_html = crate::WEBSITE_HTML;
    let website_css = crate::WEBSITE_CSS;
    let sheet = StyleSheet::parse(website_css);
    let viewport = crate::VIEWPORT;

    let tree = parse::parse_with_css(website_html, &sheet);
    let n = tree.arena.len();

    let mut results = Vec::new();

    // Parse + layout (no paint)
    results.push(bench_fn(
        &format!("parse+layout website ({n} nodes)"),
        n,
        3,
        200,
        || {
            let mut t = parse::parse_with_css(website_html, &sheet);
            t.layout(viewport);
            std::hint::black_box(&t);
        },
    ));

    // Full frame: parse + layout + paint
    results.push(bench_fn(
        &format!("full_frame website ({n} nodes)"),
        n,
        3,
        200,
        || {
            let mut t = parse::parse_with_css(website_html, &sheet);
            t.layout(viewport);
            let mut list = RenderList::default();
            t.paint(&mut list);
            std::hint::black_box(&list);
        },
    ));

    // Layout-only (re-parse + layout, no paint)
    results.push(bench_fn(
        &format!("layout_only website ({n} nodes)"),
        n,
        3,
        500,
        || {
            let mut t = parse::parse_with_css(website_html, &sheet);
            t.layout(viewport);
            std::hint::black_box(&t);
        },
    ));

    // Paint-only (build + layout, then just paint)
    results.push(bench_fn(
        &format!("paint_only website ({n} nodes)"),
        n,
        3,
        500,
        || {
            let mut t = parse::parse_with_css(website_html, &sheet);
            t.layout(viewport);
            let mut list = RenderList::default();
            t.paint(&mut list);
            // Only the paint part matters; parse+layout is amortized baseline
            std::hint::black_box(&list);
        },
    ));

    ScenarioReport {
        category: BenchCategory::DomFullFrame.id().into(),
        results,
    }
}
