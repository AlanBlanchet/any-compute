code = r"""#![allow(non_snake_case)]
//! Dioxus desktop benchmark dashboard - MASSIVE ANALYTICAL REWRITE

use any_compute_core::bench::*;
use dioxus::prelude::*;
use std::time::Instant;

fn main() {
    dioxus::launch(App);
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Tab {
    Dashboard,
    Benchmarks,
    Simulations,
    Comparisons,
}

impl Tab {
    const ALL: &[Self] = &[Self::Dashboard, Self::Benchmarks, Self::Simulations, Self::Comparisons];
    fn label(&self) -> &'static str {
        match self {
            Self::Dashboard => "Dashboard & Hardware",
            Self::Benchmarks => "Interactive Benchmarks",
            Self::Simulations => "Realtime Simulations",
            Self::Comparisons => "Versus Ecosystem",
        }
    }
}

// ── App State Context ────────────────────────────────────────────────────────
#[derive(Clone)]
struct AppContext {
    hardware: Signal<HardwareReport>,
    active_tab: Signal<Tab>,
    bench_filter: Signal<String>,
    results: Signal<Vec<ScenarioReport>>,
    comparisons: Signal<Vec<ComparisonTable>>,
    bench_running: Signal<bool>,
    bench_progress: Signal<(usize, usize)>,
    current_b_cat: Signal<Option<String>>,
}

// ── Components ───────────────────────────────────────────────────────────────

#[derive(Props, Clone, PartialEq)]
struct HwStatProps {
    label: String,
    value: String,
    clickable: Option<bool>,
    on_click: Option<EventHandler<MouseEvent>>,
}

#[component]
fn HwStat(props: HwStatProps) -> Element {
    let cls = if props.clickable.unwrap_or(false) { "hw-stat clickable" } else { "hw-stat" };
    rsx! {
        div { 
            class: "{cls}", 
            onclick: move |e| { if let Some(handler) = &props.on_click { handler.call(e) } },
            span { class: "hw-lbl", "{props.label}" } 
            span { class: "hw-val", "{props.value}" } 
        }
    }
}

#[derive(Props, Clone, PartialEq)]
struct BenchResultCardProps {
    report: ScenarioReport,
}

#[component]
fn BenchResultCard(props: BenchResultCardProps) -> Element {
    let report = props.report;
    let max_ops = report.results.iter().map(|r| r.throughput_ops_sec).fold(0.0_f64, f64::max).max(1.0);
    rsx! {
        div { class: "bench-result-card",
            h3 { class: "bench-cat-title", "{report.category}" }
            div { class: "chart-area",
                for r in report.results.iter() {
                    {
                        let h = (r.throughput_ops_sec / max_ops * 100.0).max(2.0);
                        rsx! {
                            div { class: "chart-col",
                                div { class: "chart-bar", style: "height: {h:.1}%;" }
                                div { class: "chart-lbl", "{r.scale}" }
                            }
                        }
                    }
                }
            }
            table { class: "bench-table text-xs",
                tr { th { "Size" } th { "Duration" } th { "Throughput" } }
                for r in report.results.iter() {
                    tr {
                        td { "{r.scale}" } 
                        td { class: "text-right", "{r.duration_us} µs" }
                        td { class: "text-right highlight", "{format_ops(r.throughput_ops_sec)}" }
                    }
                }
            }
        }
    }
}

#[derive(Props, Clone, PartialEq)]
struct CmpEntryCardProps {
    entry: ComparisonEntry,
}

#[component]
fn CmpEntryCard(props: CmpEntryCardProps) -> Element {
    let entry = props.entry;
    let mut all_ops = vec![entry.any_compute_ops];
    all_ops.extend(entry.comparisons.iter().map(|c| c.ops));
    let peak = all_ops.into_iter().fold(0.0_f64, f64::max).max(1.0);
    
    rsx! {
        div { class: "cmp-entry",
            div { class: "cmp-entry-title", "{entry.operation}" }
            div { class: "cmp-bar-row",
                span { class: "cmp-lib-name highlight", "any-compute" }
                div { class: "cmp-track",
                    div { class: "cmp-fill green", style: "width: {entry.any_compute_ops / peak * 100.0:.1}%" }
                }
                span { class: "cmp-factor", "1.00x Base" }
            }
            for lib in entry.comparisons.iter() {
                {
                    let factor = if entry.any_compute_ops > 0.0 { lib.ops / entry.any_compute_ops } else { 1.0 };
                    let color = if factor > 1.05 { "red" } else if factor < 0.95 { "blue" } else { "orange" };
                    let verdict = if factor > 1.05 { format!("{:.1}x Faster", factor) } else if factor < 0.95 { format!("{:.1}x Slower", 1.0 / factor) } else { "Parity".to_string() };
                    rsx! {
                        div { class: "cmp-bar-row",
                            span { class: "cmp-lib-name", "{lib.library}" }
                            div { class: "cmp-track",
                                div { class: "cmp-fill {color}", style: "width: {lib.ops / peak * 100.0:.1}%" }
                            }
                            span { class: "cmp-factor text-{color}", "{verdict}" }
                        }
                        if !lib.notes.is_empty() {
                            div { class: "cmp-note text-2xs text-muted", "{lib.notes}" }
                        }
                    }
                }
            }
        }
    }
}

#[derive(Props, Clone, PartialEq)]
struct CmpTableCardProps {
    table: ComparisonTable,
}

#[component]
fn CmpTableCard(props: CmpTableCardProps) -> Element {
    let table = props.table;
    rsx! {
        div { class: "cmp-card",
            h3 { "{table.category}" }
            for entry in table.entries.iter() {
                CmpEntryCard { entry: entry.clone() }
            }
        }
    }
}

// ── App Root ─────────────────────────────────────────────────────────────────
#[component]
fn App() -> Element {
    let hw = detect_hardware();
    use_context_provider(|| AppContext {
        hardware: Signal::new(hw),
        active_tab: Signal::new(Tab::Dashboard),
        bench_filter: Signal::new(String::new()),
        results: Signal::new(Vec::new()),
        comparisons: Signal::new(Vec::new()),
        bench_running: Signal::new(false),
        bench_progress: Signal::new((0, 0)),
        current_b_cat: Signal::new(None),
    });

    let mut active_tab = use_context::<AppContext>().active_tab;

    rsx! {
        style { {include_str!("../../assets/bench.css")} }
        div { class: "app",
            div { class: "sidebar",
                div { class: "brand",
                    div { class: "logo" }
                    span { "any-compute" }
                    span { class: "version", "v0.5.0" }
                }
                nav { class: "nav-menu",
                    for tab in Tab::ALL.iter() {
                        {
                            let t = *tab;
                            let is_active = *active_tab.read() == t;
                            let cls = if is_active { "nav-item active" } else { "nav-item" };
                            rsx! {
                                button { class: "{cls}", onclick: move |_| active_tab.set(t), "{t.label()}" }
                            }
                        }
                    }
                }
            }
            div { class: "content",
                match *active_tab.read() {
                    Tab::Dashboard => rsx! { DashboardTab {} },
                    Tab::Benchmarks => rsx! { BenchmarksTab {} },
                    Tab::Simulations => rsx! { SimulationsTab {} },
                    Tab::Comparisons => rsx! { ComparisonsTab {} },
                }
            }
        }
    }
}

// ── Dashboard Tab ────────────────────────────────────────────────────────────
#[component]
fn DashboardTab() -> Element {
    let ctx = use_context::<AppContext>();
    let hw = ctx.hardware.read().clone();
    
    let jump_with_filter = move |f: String| {
        let mut c = use_context::<AppContext>();
        c.bench_filter.set(f);
        c.active_tab.set(Tab::Benchmarks);
    };

    rsx! {
        div { class: "panel dashboard-layout",
            div { class: "hero-section",
                h1 { "Hardware Analytical Profile" }
                p { class: "text-muted", "Click attributes below to jump directly to targetted benchmarks analyzing these specific instruction sets or sub-systems." }
            }
            
            div { class: "grid-3",
                div { class: "hw-card",
                    h2 { "Processor Core" }
                    HwStat { label: "Brand".to_string(), value: hw.cpu.brand.clone() }
                    HwStat { label: "Architecture".to_string(), value: hw.cpu.arch.clone() }
                    HwStat {
                        clickable: true,
                        on_click: {
                            let mut jump = jump_with_filter.clone();
                            move |_e| jump("Threads".to_string())
                        },
                        label: "Topology".to_string(),
                        value: format!("{} Cores / {} Threads", hw.cpu.physical_cores, hw.cpu.logical_cores)
                    }
                    HwStat { label: "Base Clock".to_string(), value: format_hz(hw.cpu.frequency_mhz) }
                }
                
                div { class: "hw-card",
                    h2 { "SIMD / Vector Pipeline" }
                    HwStat { label: "Detected Engine".to_string(), value: hw.simd.detected.clone() }
                    HwStat { label: "Vector Width".to_string(), value: format!("{} bits", hw.simd.vector_width) }
                    div { class: "tag-cloud mt-10",
                        for feat in hw.simd.features.iter() {
                            {
                                let f = feat.clone();
                                let mut jump = jump_with_filter.clone();
                                rsx! {
                                    button { class: "feature-tag", onclick: move |_| jump(f.clone()), "{feat}" }
                                }
                            }
                        }
                    }
                }
                
                div { class: "hw-card",
                    h2 { "Memory & Accelerators" }
                    HwStat { label: "System RAM".to_string(), value: format!("{} GB", hw.memory.total_bytes / 1024/1024/1024) }
                    HwStat { label: "Available".to_string(), value: format!("{} GB", hw.memory.available_bytes / 1024/1024/1024) }
                    div { class: "hr" }
                    for gpu in hw.gpus.iter() {
                        HwStat {
                            clickable: true,
                            on_click: {
                                let mut jump = jump_with_filter.clone();
                                move |_e| jump("GPU".to_string())
                            },
                            label: "GPU".to_string(),
                            value: gpu.name.clone()
                        }
                    }
                    if hw.gpus.is_empty() {
                        HwStat { label: "GPU".to_string(), value: "None Detected".to_string() }
                    }
                }
            }
        }
    }
}

// ── Benchmarks Tab ───────────────────────────────────────────────────────────
#[component]
fn BenchmarksTab() -> Element {
    let mut ctx = use_context::<AppContext>();
    let search = ctx.bench_filter.read().clone();
    
    let is_running = *ctx.bench_running.read();
    let res = ctx.results.read().clone();
    let (done, total) = *ctx.bench_progress.read();
    let cur = ctx.current_b_cat.read().clone();
    
    let launch_run = move |cats: Vec<BenchCategory>| {
        if *ctx.bench_running.read() { return; }
        ctx.bench_running.set(true);
        ctx.results.set(Vec::new());
        ctx.bench_progress.set((0, cats.len()));
        
        spawn(async move {
            let mut reports = Vec::new();
            let mut c_reports = Vec::new();
            for cat in cats.iter() {
                ctx.current_b_cat.set(Some(cat.label().to_string()));
                let cat_val = *cat;
                
                let report_res = tokio::task::spawn_blocking(move || std::panic::catch_unwind(|| run_category(cat_val))).await;
                if let Ok(Ok(report)) = report_res {
                    reports.push(report.clone());
                    c_reports.push(report);
                } else {
                    let mut err_report = ScenarioReport::default();
                    err_report.category = format!("{} (CRASHED)", cat.label());
                    reports.push(err_report);
                }
                
                let (d, t) = *ctx.bench_progress.read();
                ctx.bench_progress.set((d + 1, t));
                ctx.results.set(reports.clone());
            }
            
            let cmp = tokio::task::spawn_blocking(move || build_comparison_tables(&c_reports, &c_reports, &c_reports)).await.unwrap();
            ctx.comparisons.set(cmp);
            
            ctx.bench_running.set(false);
            ctx.current_b_cat.set(None);
        });
    };

    rsx! {
        div { class: "panel",
            div { class: "flex-row justify-between align-center mb-20",
                div {
                    h2 { "Interactive Analytical Benchmarks" }
                    if !search.is_empty() {
                        div { class: "active-filter-badge", 
                            "Filtered by cap: " span { class: "highlight", "{search}" }
                            button { class: "btn-clear", onclick: move |_| ctx.bench_filter.set(String::new()), "×" }
                        }
                    }
                }
                div {
                    button { 
                        class: if is_running { "btn btn-secondary disabled" } else { "btn btn-primary btn-run" },
                        onclick: {
                            let mut l = launch_run.clone();
                            move |_| l(BenchCategory::ALL.to_vec())
                        },
                        if is_running { "Running..." } else { "Run All Tests" }
                    }
                }
            }
            
            if is_running {
                div { class: "progress-container",
                    div { class: "flex-row justify-between text-xs",
                        span { "Computing real multi-scaled workloads..." }
                        span { "{done} / {total}" }
                    }
                    div { class: "progress-bar-bg",
                        div { class: "progress-bar-fill", style: "{((done as f64 / total.max(1) as f64) * 100.0):.1}%" }
                        if let Some(c) = cur {
                            div { class: "progress-label text-xs ml-10", "{c}" }
                        }
                    }
                }
            }

            if res.is_empty() && !is_running {
                div { class: "empty-state-notice", "No analytical data computed yet. Click 'Run All Tests'." }
            }
            
            div { class: "bench-results-grid",
                for report in res.iter() {
                    BenchResultCard { report: report.clone() }
                }
            }
        }
    }
}

// ── Comparisons Tab ──────────────────────────────────────────────────────────
#[component]
fn ComparisonsTab() -> Element {
    let ctx = use_context::<AppContext>();
    let cmps = ctx.comparisons.read().clone();
    
    rsx! {
        div { class: "panel",
            h2 { "Versus Ecosystem (Real-time computed relative factor)" }
            p { class: "text-muted text-sm", "If you haven't run benchmarks above, this uses extrapolated estimates. Run benchmarks to replace these bars with live-measured execution parity." }
            if cmps.is_empty() {
                div { class: "empty-state-notice", "No comparison traces accumulated. Please run benchmarks." }
            } else {
                div { class: "cmp-grid",
                    for table in cmps.iter() {
                        CmpTableCard { table: table.clone() }
                    }
                }
            }
        }
    }
}

// ── Simulations Tab (Real Compute Binding) ───────────────────────────────────
#[component]
fn SimulationsTab() -> Element {
    let mut sim_running = use_signal(|| false);
    
    // Live metrics
    let mut ac_ops = use_signal(|| 0_f64);
    let mut rayon_ops = use_signal(|| 0_f64);
    let mut std_ops = use_signal(|| 0_f64);

    let toggle = move |_| {
        let nxt = !*sim_running.read();
        sim_running.set(nxt);
        
        if nxt {
            // Any-Compute Worker
            spawn(async move {
                use any_compute_core::kernel::{best_kernel, UnaryOp};
                let mut last_tick = Instant::now();
                let mut raw_ops = 0;
                let data = vec![1.0_f64; 200_000];
                
                loop {
                    if !*sim_running.read() { break; }
                    let _ = tokio::task::spawn_blocking({
                        let c = data.clone();
                        move || {
                            let kern = best_kernel();
                            let r = kern.map_unary_f64(&c, UnaryOp::Sigmoid);
                            std::hint::black_box(r);
                        }
                    }).await;
                    
                    raw_ops += data.len();
                    let elapsed = last_tick.elapsed().as_secs_f64();
                    if elapsed > 0.5 {
                        ac_ops.set(raw_ops as f64 / elapsed);
                        raw_ops = 0;
                        last_tick = Instant::now();
                    }
                    tokio::task::yield_now().await;
                }
            });

            // Rayon Worker
            spawn(async move {
                use rayon::prelude::*;
                let mut last_tick = Instant::now();
                let mut raw_ops = 0;
                let data = vec![1.0_f64; 200_000];
                
                loop {
                    if !*sim_running.read() { break; }
                    let _ = tokio::task::spawn_blocking({
                        let c = data.clone();
                        move || {
                            let out: Vec<f64> = c.par_iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect();
                            std::hint::black_box(out);
                        }
                    }).await;
                    
                    raw_ops += data.len();
                    let elapsed = last_tick.elapsed().as_secs_f64();
                    if elapsed > 0.5 {
                        rayon_ops.set(raw_ops as f64 / elapsed);
                        raw_ops = 0;
                        last_tick = Instant::now();
                    }
                    tokio::task::yield_now().await;
                }
            });

            // Std Iterator Worker
            spawn(async move {
                let mut last_tick = Instant::now();
                let mut raw_ops = 0;
                let data = vec![1.0_f64; 200_000];
                
                loop {
                    if !*sim_running.read() { break; }
                    let _ = tokio::task::spawn_blocking({
                        let c = data.clone();
                        move || {
                            let out: Vec<f64> = c.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect();
                            std::hint::black_box(out);
                        }
                    }).await;
                    
                    raw_ops += data.len();
                    let elapsed = last_tick.elapsed().as_secs_f64();
                    if elapsed > 0.5 {
                        std_ops.set(raw_ops as f64 / elapsed);
                        raw_ops = 0;
                        last_tick = Instant::now();
                    }
                    tokio::task::yield_now().await;
                }
            });
        }
    };

    let ac = *ac_ops.read();
    let ry = *rayon_ops.read();
    let st = *std_ops.read();
    let peak = ac.max(ry).max(st).max(1.0);

    rsx! {
        div { class: "panel",
            h2 { "Realtime Showdown (Live Telemetry)" }
            p { class: "text-muted mb-20", "Direct real-time comparison firing parallel Sigmoid map operations over vectors. We execute equivalent underlying compute layers across Any-Compute, StdLib, and Rayon dynamically." }
            
            button { class: if *sim_running.read() { "btn btn-secondary" } else { "btn btn-primary" }, 
                onclick: toggle, 
                if *sim_running.read() { "Halt Execution Showdown" } else { "Ignite Head-To-Head Computing Matrix" }
            }
            
            if *sim_running.read() {
                div { class: "sim-racetrack mt-20",
                    div { class: "track-lane",
                        div { class: "flex-row justify-between mb-5",
                            span { class: "lib-name highlight", "Any-Compute (Kernel Vectorized)" }
                            span { class: "tel-val highlight", "{format_ops(ac)}" }
                        }
                        div { class: "cmp-track",
                            div { class: "cmp-fill green", style: "width: {ac / peak * 100.0:.1}%" }
                        }
                    }
                    
                    div { class: "track-lane mt-20",
                        div { class: "flex-row justify-between mb-5",
                            span { class: "lib-name", "Rayon (Thread Pool parallel iterator)" }
                            span { class: "tel-val text-blue", "{format_ops(ry)}" }
                        }
                        div { class: "cmp-track",
                            div { class: "cmp-fill blue", style: "width: {ry / peak * 100.0:.1}%" }
                        }
                    }

                    div { class: "track-lane mt-20",
                        div { class: "flex-row justify-between mb-5",
                            span { class: "lib-name text-muted", "Standard StdLib (Single Core Naive)" }
                            span { class: "tel-val text-orange", "{format_ops(st)}" }
                        }
                        div { class: "cmp-track",
                            div { class: "cmp-fill orange", style: "width: {st / peak * 100.0:.1}%" }
                        }
                    }
                }
            }
        }
    }
}

// ── Formatters ───────────────────────────────────────────────────────────────
fn format_hz(mhz: u64) -> String {
    if mhz > 1000 {
        format!("{:.2} GHz", mhz as f64 / 1000.0)
    } else {
        format!("{} MHz", mhz)
    }
}

fn format_ops(ops: f64) -> String {
    if ops > 1_000_000_000.0 {
        format!("{:.1} Gops/s", ops / 1_000_000_000.0)
    } else if ops > 1_000_000.0 {
        format!("{:.1} Mops/s", ops / 1_000_000.0)
    } else if ops > 1_000.0 {
        format!("{:.1} Kops/s", ops / 1_000.0)
    } else {
        format!("{:.0} ops/s", ops)
    }
}
"""

with open("crates/rsx/src/bin/bench_window.rs", "w") as f:
    f.write(code)
