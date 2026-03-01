//! Dioxus desktop benchmark dashboard.
//!
//! Five-tab dashboard: Overview, Benchmarks, Comparisons, Live Metrics, Device Profiles.
//!
//! Run: `cargo run -p any-compute-rsx --features bench --bin anc-bench-window`

use any_compute_core::bench::*;
use dioxus::prelude::*;
use std::collections::HashSet;
use std::sync::{Arc, Mutex};

fn main() {
    dioxus::launch(App);
}

// ══════════════════════════════════════════════════════════════════════════
// Tabs
// ══════════════════════════════════════════════════════════════════════════

#[derive(Clone, Copy, PartialEq, Eq)]
enum Tab {
    Overview,
    Benchmarks,
    Comparisons,
    LiveMetrics,
    DeviceProfiles,
}

impl Tab {
    const ALL: &[Self] = &[
        Self::Overview,
        Self::Benchmarks,
        Self::Comparisons,
        Self::LiveMetrics,
        Self::DeviceProfiles,
    ];

    fn label(self) -> &'static str {
        match self {
            Self::Overview => "Overview",
            Self::Benchmarks => "Benchmarks",
            Self::Comparisons => "Comparisons",
            Self::LiveMetrics => "Live Metrics",
            Self::DeviceProfiles => "Device Profiles",
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════
// App root
// ══════════════════════════════════════════════════════════════════════════

#[component]
fn App() -> Element {
    let hardware = use_signal(detect_hardware);
    let features = use_signal(detect_features);
    let mut active_tab = use_signal(|| Tab::Overview);

    // Benchmark state (shared across tabs)
    let mut running = use_signal(|| false);
    let mut current_cat = use_signal(|| None::<String>);
    let mut results = use_signal(Vec::<ScenarioReport>::new);
    let mut comparisons = use_signal(Vec::<ComparisonTable>::new);

    // Category selection for the Benchmarks tab
    let mut selected_cats = use_signal(|| {
        BenchCategory::ALL.iter().copied().collect::<HashSet<BenchCategory>>()
    });

    // Live metrics state
    let mut live_metrics = use_signal(|| LiveMetrics::default());
    let mut live_running = use_signal(|| false);

    // Device profile results
    let mut profile_results = use_signal(Vec::<(String, Vec<ScenarioReport>)>::new);
    let mut profiles_running = use_signal(|| false);

    // ── Run selected benchmarks ──────────────────────────────────────
    let run_benchmarks = move |_| {
        if *running.read() {
            return;
        }
        running.set(true);
        results.set(Vec::new());
        comparisons.set(Vec::new());

        let cats: Vec<BenchCategory> = selected_cats
            .read()
            .iter()
            .copied()
            .collect();

        spawn(async move {
            let mut kernel_reports = Vec::new();
            let mut compute_reports = Vec::new();
            let mut framework_reports = Vec::new();

            for cat in cats {
                current_cat.set(Some(cat.label().to_string()));
                let report = tokio::task::spawn_blocking(move || run_category(cat))
                    .await
                    .unwrap();
                match cat.group() {
                    "Kernel" => kernel_reports.push(report.clone()),
                    "Compute" | "Data" => compute_reports.push(report.clone()),
                    _ => framework_reports.push(report.clone()),
                }
                results.write().push(report);
            }

            let cmp = tokio::task::spawn_blocking(move || {
                build_comparison_tables(&kernel_reports, &compute_reports, &framework_reports)
            })
            .await
            .unwrap();
            comparisons.set(cmp);
            current_cat.set(None);
            running.set(false);
        });
    };

    // ── Toggle live metrics polling ──────────────────────────────────
    let toggle_live = move |_| {
        if *live_running.read() {
            live_running.set(false);
            return;
        }
        live_running.set(true);

        spawn(async move {
            let monitor = Arc::new(Mutex::new(MetricsMonitor::new()));
            loop {
                if !*live_running.read() {
                    break;
                }
                let m = monitor.clone();
                let snapshot = tokio::task::spawn_blocking(move || m.lock().unwrap().snapshot())
                    .await
                    .unwrap();
                live_metrics.set(snapshot);
                tokio::time::sleep(std::time::Duration::from_millis(750)).await;
            }
        });
    };

    // ── Run device profile simulations ───────────────────────────────
    let run_profiles = move |_| {
        if *profiles_running.read() {
            return;
        }
        profiles_running.set(true);
        profile_results.set(Vec::new());

        spawn(async move {
            for (name, profile) in all_profiles() {
                let profile_c = profile.clone();
                let reports = tokio::task::spawn_blocking(move || run_simulated(&profile_c))
                    .await
                    .unwrap();
                profile_results.write().push((name.to_string(), reports));
            }
            profiles_running.set(false);
        });
    };

    rsx! {
        style { {CSS} }
        div { class: "app",
            // ── Top bar ──────────────────────────────────────────────
            div { class: "top-bar",
                h1 { "any-compute" }
                span { class: "subtitle", "High-performance compute & visualization engine" }
            }

            // ── Tab navigation ───────────────────────────────────────
            div { class: "tab-nav",
                for tab in Tab::ALL {
                    {
                        let t = *tab;
                        let cls = if *active_tab.read() == t { "tab-btn active" } else { "tab-btn" };
                        rsx! {
                            button {
                                class: "{cls}",
                                onclick: move |_| active_tab.set(t),
                                "{t.label()}"
                            }
                        }
                    }
                }
            }

            // ── Tab content ──────────────────────────────────────────
            div { class: "tab-content",
                div { class: "tab-pane",
                    match *active_tab.read() {
                        Tab::Overview => rsx! {
                            OverviewTab {
                                hardware: hardware.read().clone(),
                                features: features.read().clone(),
                            }
                        },
                        Tab::Benchmarks => rsx! {
                            BenchmarksTab {
                                running: *running.read(),
                                current_cat: current_cat.read().clone(),
                                results: results.read().clone(),
                                comparisons: comparisons.read().clone(),
                                selected_cats: selected_cats.read().clone(),
                                on_toggle_cat: move |cat: BenchCategory| {
                                    let mut sel = selected_cats.write();
                                    if sel.contains(&cat) { sel.remove(&cat); } else { sel.insert(cat); }
                                },
                                on_run: run_benchmarks,
                                on_select_all: move |_| {
                                    selected_cats.set(BenchCategory::ALL.iter().copied().collect());
                                },
                                on_select_none: move |_| {
                                    selected_cats.set(HashSet::new());
                                },
                            }
                        },
                        Tab::Comparisons => rsx! {
                            ComparisonsTab {}
                        },
                        Tab::LiveMetrics => rsx! {
                            LiveMetricsTab {
                                metrics: live_metrics.read().clone(),
                                is_running: *live_running.read(),
                                on_toggle: toggle_live,
                            }
                        },
                        Tab::DeviceProfiles => rsx! {
                            DeviceProfilesTab {
                                results: profile_results.read().clone(),
                                is_running: *profiles_running.read(),
                                on_run: run_profiles,
                            }
                        },
                    }
                }
            }
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════
// Tab 1: Overview
// ══════════════════════════════════════════════════════════════════════════

#[component]
fn OverviewTab(hardware: HardwareReport, features: FeaturesReport) -> Element {
    let peak = estimate_peak(&hardware);

    rsx! {
        // ── Metric cards ─────────────────────────────────────────────
        div { class: "grid-4",
            MetricCard {
                value: format!("{:.1}", peak.fp64_gflops),
                label: "FP64 GFLOPS",
                sub: (if peak.has_fma { "FMA enabled" } else { "" }).to_string(),
                color: "blue".to_string(),
            }
            MetricCard {
                value: format!("{:.1}", peak.fp32_gflops),
                label: "FP32 GFLOPS",
                sub: format!("{} threads", peak.rayon_threads),
                color: "green".to_string(),
            }
            MetricCard {
                value: format!("{:.0} GB/s", peak.mem_bw_gbs),
                label: "Memory BW",
                sub: "estimated".to_string(),
                color: "purple".to_string(),
            }
            MetricCard {
                value: format_bytes(hardware.memory.total_bytes),
                label: "Total RAM",
                sub: format!("{} available", format_bytes(hardware.memory.available_bytes)),
                color: "orange".to_string(),
            }
        }

        // ── Hardware specs ───────────────────────────────────────────
        div { class: "panel",
            h2 { "Hardware" }
            div { class: "specs-grid",
                SpecRow { label: "CPU", value: hardware.cpu.brand.clone() }
                SpecRow { label: "Cores",
                    value: format!("{} physical / {} logical @ {} MHz",
                        hardware.cpu.physical_cores, hardware.cpu.logical_cores,
                        hardware.cpu.frequency_mhz)
                }
                SpecRow { label: "Architecture", value: hardware.cpu.arch.clone() }
                SpecRow { label: "SIMD",
                    value: format!("{} (width={})", hardware.simd.detected, hardware.simd.vector_width)
                }
                if !hardware.simd.features.is_empty() {
                    SpecRow { label: "ISA Extensions", value: hardware.simd.features.join(", ") }
                }
            }
        }

        // ── Features ─────────────────────────────────────────────────
        div { class: "panel",
            h2 { "Optimization Features" }
            div { class: "features-grid",
                FeatureFlag { name: "CUDA (cuBLAS/cuDNN)", enabled: features.cuda }
                FeatureFlag { name: "ROCm (rocBLAS/MIOpen)", enabled: features.rocm }
                FeatureFlag { name: "Intel MKL (oneMKL)", enabled: features.mkl }
                FeatureFlag { name: "Apple Metal (MPS)", enabled: features.metal }
                FeatureFlag { name: "wgpu (Vulkan/Metal/DX12)", enabled: features.wgpu }
                FeatureFlag { name: "Shader compiler (naga)", enabled: features.shader }
            }
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════
// Tab 2: Benchmarks
// ══════════════════════════════════════════════════════════════════════════

#[component]
fn BenchmarksTab(
    running: bool,
    current_cat: Option<String>,
    results: Vec<ScenarioReport>,
    comparisons: Vec<ComparisonTable>,
    selected_cats: HashSet<BenchCategory>,
    on_toggle_cat: EventHandler<BenchCategory>,
    on_run: EventHandler<MouseEvent>,
    on_select_all: EventHandler<MouseEvent>,
    on_select_none: EventHandler<MouseEvent>,
) -> Element {
    rsx! {
        // ── Category selector ────────────────────────────────────────
        div { class: "panel",
            h2 { "Select Categories" }
            div { class: "controls",
                button { class: "btn btn-secondary", onclick: on_select_all, "Select All" }
                button { class: "btn btn-secondary", onclick: on_select_none, "Clear" }
            }
            div { class: "category-grid",
                for &cat in BenchCategory::ALL.iter() {
                    {
                        let selected = selected_cats.contains(&cat);
                        let cls = if selected { "cat-item selected" } else { "cat-item" };
                        rsx! {
                            div {
                                class: "{cls}",
                                onclick: move |_| on_toggle_cat.call(cat),
                                div { class: "cat-check",
                                    if selected { "✓" }
                                }
                                span { "{cat.label()}" }
                            }
                        }
                    }
                }
            }
        }

        // ── Run controls ─────────────────────────────────────────────
        div { class: "controls",
            if running {
                div { class: "running",
                    div { class: "spinner" }
                    if let Some(cat) = current_cat.as_ref() {
                        span { "Running: {cat}" }
                    }
                }
            } else {
                button {
                    class: "btn btn-primary",
                    onclick: on_run,
                    "Run Selected ({selected_cats.len()})"
                }
            }
        }

        // ── Results ──────────────────────────────────────────────────
        if !results.is_empty() {
            div { class: "panel",
                h2 { "Results" }
                for report in results.iter() {
                    ResultsTable { report: report.clone() }
                }
            }
        }

        // ── Dynamic comparisons ──────────────────────────────────────
        if !comparisons.is_empty() {
            div { class: "panel",
                h2 { "Library Comparisons (from benchmarks)" }
                for table in comparisons.iter() {
                    ComparisonPanel { table: table.clone() }
                }
            }
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════
// Tab 3: Comparisons (static reference data)
// ══════════════════════════════════════════════════════════════════════════

#[component]
fn ComparisonsTab() -> Element {
    let refs = reference_comparisons();

    // Collect unique domains in order
    let mut domains = Vec::new();
    for r in &refs {
        if !domains.contains(&r.domain) {
            domains.push(r.domain.clone());
        }
    }

    // Domain filter state
    let mut active_domain = use_signal(|| None::<String>);

    rsx! {
        div { class: "panel",
            h2 { "Reference Comparisons" }
            p { class: "text-muted text-xs",
                "Estimated performance ratios from published benchmarks. factor < 1.0 = slower than any-compute, > 1.0 = faster."
            }

            // Domain filter chips
            div { class: "domain-filter",
                {
                    let ad = active_domain.read().clone();
                    rsx! {
                        button {
                            class: if ad.is_none() { "domain-chip active" } else { "domain-chip" },
                            onclick: move |_| active_domain.set(None),
                            "All"
                        }
                    }
                }
                for domain in domains.iter() {
                    {
                        let d = domain.clone();
                        let d2 = domain.clone();
                        let ad = active_domain.read().clone();
                        let cls = if ad.as_ref() == Some(&d) { "domain-chip active" } else { "domain-chip" };
                        rsx! {
                            button {
                                class: "{cls}",
                                onclick: move |_| active_domain.set(Some(d2.clone())),
                                "{d}"
                            }
                        }
                    }
                }
            }
        }

        // Render each domain section
        for domain in domains.iter() {
            {
                let ad = active_domain.read().clone();
                let show = ad.is_none() || ad.as_ref() == Some(domain);
                if show {
                    let domain_refs: Vec<_> = refs.iter()
                        .filter(|r| r.domain == *domain)
                        .collect();

                    // Group by category within domain
                    let mut categories = Vec::new();
                    for r in &domain_refs {
                        if !categories.contains(&r.category) {
                            categories.push(r.category.clone());
                        }
                    }

                    rsx! {
                        div { class: "domain-section",
                            div { class: "domain-title", "{domain}" }
                            for cat in categories.iter() {
                                {
                                    let cat_refs: Vec<_> = domain_refs.iter()
                                        .filter(|r| r.category == *cat)
                                        .collect();
                                    let max_factor = cat_refs.iter()
                                        .map(|r| r.factor)
                                        .fold(1.0_f64, f64::max)
                                        .max(1.0);

                                    rsx! {
                                        div { class: "panel",
                                            h3 { "{cat}" }
                                            // Bar chart
                                            div { class: "bar-row",
                                                span { class: "bar-label", "any-compute" }
                                                div { class: "bar-track",
                                                    div {
                                                        class: "bar-fill green",
                                                        style: "width: {100.0 / max_factor:.1}%",
                                                    }
                                                }
                                                span { class: "bar-value", "1.00x" }
                                            }
                                            for r in cat_refs.iter() {
                                                {
                                                    let pct = (r.factor / max_factor * 100.0).min(100.0);
                                                    let color_cls = if r.factor > 1.05 { "bar-fill red" }
                                                        else if r.factor > 0.5 { "bar-fill orange" }
                                                        else if r.factor > 0.1 { "bar-fill blue" }
                                                        else { "bar-fill purple" };
                                                    let badge_cls = comparison_indicator(r.factor);
                                                    rsx! {
                                                        div { class: "bar-row",
                                                            span { class: "bar-label", "{r.library}" }
                                                            div { class: "bar-track",
                                                                div {
                                                                    class: "{color_cls}",
                                                                    style: "width: {pct:.1}%",
                                                                }
                                                            }
                                                            span { class: "bar-value",
                                                                span { class: "badge {badge_cls}",
                                                                    if r.factor > 1.05 {
                                                                        {format!("{:.1}x faster", r.factor)}
                                                                    } else if r.factor < 0.95 {
                                                                        {format!("{:.1}x slower", 1.0 / r.factor)}
                                                                    } else {
                                                                        {"~same".to_string()}
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                            // Notes table
                                            table { class: "ref-table",
                                                thead { tr {
                                                    th { "Library" }
                                                    th { "Factor" }
                                                    th { "Notes" }
                                                }}
                                                tbody {
                                                    for r in cat_refs.iter() {
                                                        tr {
                                                            td { "{r.library}" }
                                                            td { class: "mono",
                                                                {format!("{:.2}x", r.factor)}
                                                            }
                                                            td { class: "text-muted text-xs", "{r.notes}" }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                } else {
                    rsx! {}
                }
            }
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════
// Tab 4: Live Metrics
// ══════════════════════════════════════════════════════════════════════════

#[component]
fn LiveMetricsTab(
    metrics: LiveMetrics,
    is_running: bool,
    on_toggle: EventHandler<MouseEvent>,
) -> Element {
    let mem_pct = if metrics.mem_total_bytes > 0 {
        metrics.mem_used_bytes as f64 / metrics.mem_total_bytes as f64 * 100.0
    } else {
        0.0
    };
    let mem_color = if mem_pct > 85.0 { "high" } else if mem_pct > 60.0 { "mid" } else { "low" };

    rsx! {
        // ── Control ──────────────────────────────────────────────────
        div { class: "controls",
            button {
                class: if is_running { "btn btn-secondary" } else { "btn btn-primary" },
                onclick: on_toggle,
                if is_running { "Stop Monitoring" } else { "Start Monitoring" }
            }
            div { class: "live-status",
                div { class: if is_running { "live-dot" } else { "live-dot stopped" } }
                span { class: "text-muted text-xs",
                    if is_running { "Sampling every 750ms" } else { "Stopped" }
                }
            }
        }

        // ── Summary cards ────────────────────────────────────────────
        div { class: "grid-4",
            MetricCard {
                value: format!("{:.1}%", metrics.cpu_global),
                label: "CPU Usage",
                sub: format!("{} cores", metrics.cpu_per_core.len()),
                color: (if metrics.cpu_global > 80.0 { "red" } else if metrics.cpu_global > 50.0 { "orange" } else { "green" }).to_string(),
            }
            MetricCard {
                value: format!("{:.1}%", mem_pct),
                label: "Memory",
                sub: format!("{} / {}", format_bytes(metrics.mem_used_bytes), format_bytes(metrics.mem_total_bytes)),
                color: (if mem_pct > 85.0 { "red" } else if mem_pct > 60.0 { "orange" } else { "green" }).to_string(),
            }
            MetricCard {
                value: format!("{:.0}", metrics.compute_ops_per_sec),
                label: "Compute ops/s",
                sub: "map_f64 batches".to_string(),
                color: "blue".to_string(),
            }
            MetricCard {
                value: format_ops(metrics.compute_throughput_elem_sec),
                label: "Throughput",
                sub: "elements/sec".to_string(),
                color: "purple".to_string(),
            }
        }

        // ── Per-core CPU bars ────────────────────────────────────────
        if !metrics.cpu_per_core.is_empty() {
            div { class: "panel",
                h2 { "Per-Core CPU Utilization" }
                div { class: "metrics-row",
                    for (i, &usage) in metrics.cpu_per_core.iter().enumerate() {
                        {
                            let color = if usage > 80.0 { "high" } else if usage > 50.0 { "mid" } else { "low" };
                            rsx! {
                                div { class: "core-bar",
                                    div { class: "core-label", "Core {i}" }
                                    div { class: "core-track",
                                        div {
                                            class: "core-fill progress-fill {color}",
                                            style: "width: {usage:.0}%",
                                        }
                                        span { class: "core-pct", "{usage:.0}%" }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // ── Memory bar ───────────────────────────────────────────────
        div { class: "panel",
            h2 { "Memory Usage" }
            div { class: "progress-row",
                div { class: "progress-header",
                    span { class: "progress-label",
                        "{format_bytes(metrics.mem_used_bytes)} used"
                    }
                    span { class: "progress-pct", "{mem_pct:.1}%" }
                }
                div { class: "progress-track",
                    div {
                        class: "progress-fill {mem_color}",
                        style: "width: {mem_pct:.1}%",
                    }
                }
            }
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════
// Tab 5: Device Profiles
// ══════════════════════════════════════════════════════════════════════════

#[component]
fn DeviceProfilesTab(
    results: Vec<(String, Vec<ScenarioReport>)>,
    is_running: bool,
    on_run: EventHandler<MouseEvent>,
) -> Element {
    let profiles = all_profiles();

    rsx! {
        // ── Profile cards ────────────────────────────────────────────
        div { class: "panel",
            h2 { "Simulated Device Profiles" }
            p { class: "text-muted text-xs",
                "Performance simulation using throttled backends. These simulate how any-compute would perform on different hardware."
            }
            div { class: "profile-grid",
                for (name, profile) in profiles.iter() {
                    div { class: "profile-card",
                        div { class: "profile-name", "{profile.name}" }
                        div { class: "profile-specs",
                            span { "Cores: {profile.cores}" }
                            span { {format!("Bandwidth: {:.0}%", profile.bandwidth_factor * 100.0)} }
                            span { {format!("Compute: {:.0}%", profile.compute_factor * 100.0)} }
                            span { "ID: {name}" }
                        }
                    }
                }
            }
        }

        // ── Run simulation ───────────────────────────────────────────
        div { class: "controls",
            if is_running {
                div { class: "running",
                    div { class: "spinner" }
                    span { "Simulating..." }
                }
            } else {
                button {
                    class: "btn btn-primary",
                    onclick: on_run,
                    "Run All Profile Simulations"
                }
            }
        }

        // ── Simulation results ───────────────────────────────────────
        for (profile_name, reports) in results.iter() {
            div { class: "panel",
                h2 { "{profile_name}" }
                for report in reports.iter() {
                    ResultsTable { report: report.clone() }
                }
            }
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════
// Shared sub-components
// ══════════════════════════════════════════════════════════════════════════

#[component]
fn MetricCard(value: String, label: String, sub: String, color: String) -> Element {
    rsx! {
        div { class: "metric-card",
            div { class: "metric-value {color}", "{value}" }
            div { class: "metric-label", "{label}" }
            if !sub.is_empty() {
                div { class: "metric-sub", "{sub}" }
            }
        }
    }
}

#[component]
fn SpecRow(label: String, value: String) -> Element {
    rsx! {
        div { class: "spec-row",
            span { class: "spec-label", "{label}" }
            span { class: "spec-value", "{value}" }
        }
    }
}

#[component]
fn FeatureFlag(name: String, enabled: bool) -> Element {
    let class = if enabled { "feature enabled" } else { "feature disabled" };
    let icon = if enabled { "+" } else { "-" };
    rsx! {
        div { class: "{class}",
            span { class: "feature-icon", "[{icon}]" }
            span { "{name}" }
        }
    }
}

#[component]
fn ResultsTable(report: ScenarioReport) -> Element {
    let title = BenchCategory::ALL
        .iter()
        .find(|c| c.id() == report.category)
        .map(|c| c.label())
        .unwrap_or(&report.category);

    rsx! {
        div { class: "result-section",
            h3 { "{title}" }
            table { class: "bench-table",
                thead { tr {
                    th { "Operation" }
                    th { "Scale" }
                    th { "Time" }
                    th { "Throughput" }
                }}
                tbody {
                    for result in report.results.iter() {
                        tr {
                            td { class: "op-name", "{result.name}" }
                            td { class: "scale", "{result.scale}" }
                            td { class: "time", {format_duration(result.duration_us)} }
                            td { class: "throughput", {format_ops(result.throughput_ops_sec)} }
                        }
                    }
                }
            }
        }
    }
}

#[component]
fn ComparisonPanel(table: ComparisonTable) -> Element {
    rsx! {
        div { class: "comparison-section",
            h3 { "{table.category}" }
            for entry in table.entries.iter() {
                div { class: "comparison-entry",
                    div { class: "comparison-header",
                        span { class: "op-name", "{entry.operation}" }
                        span { class: "our-result",
                            {format_duration(entry.any_compute_us)}
                            " ({format_ops(entry.any_compute_ops)})"
                        }
                    }
                    div { class: "comparison-libs",
                        for cmp in entry.comparisons.iter() {
                            {
                                let ratio = if entry.any_compute_ops > 0.0 {
                                    cmp.estimated_ops / entry.any_compute_ops
                                } else {
                                    1.0
                                };
                                let badge_class = format!("badge {}", comparison_indicator(ratio));
                                let factor_text = if ratio > 1.05 {
                                    format!("{:.1}x faster", ratio)
                                } else if ratio < 0.95 {
                                    format!("{:.1}x slower", 1.0 / ratio)
                                } else {
                                    "~same".to_string()
                                };
                                rsx! {
                                    div { class: "lib-comparison",
                                        span { class: "lib-name", "vs {cmp.library}" }
                                        span { class: "{badge_class}", "{factor_text}" }
                                        span { class: "lib-notes", "{cmp.notes}" }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Loaded at compile time — edit `crates/rsx/assets/bench.css` directly.
const CSS: &str = include_str!("../../assets/bench.css");
