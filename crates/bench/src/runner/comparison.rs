use super::report::*;
use super::*;
use serde::{Deserialize, Serialize};

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
        use any_compute_core::interaction::{EventContext, EventResponse, Interactive};

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
    let sim_device = Device::simulated(profile.clone());
    let mut results = Vec::new();

    // Compute parallel on simulated
    let mut compute_results = Vec::new();
    for &n in SIMULATED_SIZES {
        let data = make_f64_data(n);
        compute_results.push(bench_fn(&format!("map n={n}"), n, 3, 50, || {
            std::hint::black_box(sim_device.map(&data, |v| v * 2.0 + 1.0));
        }));
        compute_results.push(bench_fn(&format!("sum n={n}"), n, 3, 50, || {
            std::hint::black_box(sim_device.sum(&data));
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
        let device = Device::cpu();
        let data: Vec<f64> = (0..METRICS_SAMPLE_SIZE).map(|i| i as f64).collect();
        let start = Instant::now();
        for _ in 0..METRICS_SAMPLE_ITERS {
            std::hint::black_box(device.map(&data, |v| v * 2.0 + 1.0));
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

// Reference data lives in runner_references.rs to keep this file focused on logic.
#[path = "../runner_references.rs"]
mod runner_references;
pub use runner_references::reference_comparisons;
