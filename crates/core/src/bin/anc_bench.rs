//! CLI benchmark runner.
//!
//! Run: `cargo run --release --features hwinfo --bin anc-bench`

use any_compute_core::bench::*;
use any_compute_core::compute::{ComputeBackend, SimulatedBackend};
use any_compute_core::kernel::best_kernel;

fn main() {
    let out_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("out");
    std::fs::create_dir_all(&out_dir).ok();

    // Header
    println!();
    println!("======================================================================");
    println!("                   any-compute benchmark suite                        ");
    println!("======================================================================");
    println!();

    // Hardware
    let hardware = detect_hardware();
    print_hardware(&hardware);

    let features = detect_features();
    print_features(&features);
    let peak = estimate_peak(&hardware);
    print_peak(&peak, &hardware);

    // Kernel
    let kernel = best_kernel();
    println!("======================================================================");
    println!(
        "  Kernel: {}  (backend: {}, width: {})",
        kernel.name(),
        kernel.backend_tag(),
        kernel.vector_width()
    );
    println!("======================================================================\n");

    // Run all categories, printing as we go
    let mut kernel_benchmarks = Vec::new();
    let mut compute_benchmarks = Vec::new();
    let mut framework_benchmarks = Vec::new();

    for &cat in BenchCategory::ALL {
        println!("  [{}] {}", cat.group().to_lowercase(), cat.label());
        let report = run_category(cat);
        for r in &report.results {
            println!(
                "    {:<55} scale={:<10} {:>10}us  ({})",
                r.name,
                r.scale,
                r.duration_us,
                format_ops(r.throughput_ops_sec),
            );
        }
        match cat.group() {
            "Kernel" => kernel_benchmarks.push(report),
            "Compute" | "Data" => compute_benchmarks.push(report),
            _ => framework_benchmarks.push(report),
        }
        println!();
    }

    // Comparison tables
    let comparisons = build_comparison_tables(
        &kernel_benchmarks,
        &compute_benchmarks,
        &framework_benchmarks,
    );
    println!("======================================================================");
    println!("  LIBRARY COMPARISONS (estimated)");
    println!("======================================================================");
    print_comparisons(&comparisons);

    // Write JSON
    let report = FullReport {
        timestamp: timestamp_now(),
        hardware,
        features,
        kernel_benchmarks,
        compute_benchmarks,
        framework_benchmarks,
        comparisons,
    };

    let report_path = out_dir.join("bench-full.json");
    let json = serde_json::to_string_pretty(&report).expect("serialize report");
    std::fs::write(&report_path, &json).expect("write report");
    println!("\n  Full report: {}", report_path.display());

    // Simulated devices
    for (tag, profile) in all_profiles() {
        let sim = SimulatedBackend::new(profile.clone());
        let info = sim.device_info();
        println!(
            "\n--- Simulated: {} (cores={}, compute={:.2}) ---\n",
            info.name, info.max_parallelism, profile.compute_factor
        );
        let scenarios = run_simulated(&profile);
        for sc in &scenarios {
            for r in &sc.results {
                println!(
                    "    {:<55} {:>10}us  ({})",
                    r.name,
                    r.duration_us,
                    format_ops(r.throughput_ops_sec),
                );
            }
        }

        let sim_path = out_dir.join(format!("bench-{tag}.json"));
        let json = serde_json::to_string_pretty(&scenarios).expect("serialize");
        std::fs::write(&sim_path, &json).expect("write");
    }

    println!("\n  All reports written to {}/", out_dir.display());
}

// ── CLI display helpers ──────────────────────────────────────────────────

fn print_hardware(hw: &HardwareReport) {
    println!("----------------------------------------------------------------------");
    println!("  HARDWARE SPECIFICATIONS");
    println!("----------------------------------------------------------------------");
    println!("  CPU:    {}", hw.cpu.brand);
    println!(
        "  Cores:  {} physical / {} logical @ {} MHz",
        hw.cpu.physical_cores, hw.cpu.logical_cores, hw.cpu.frequency_mhz
    );
    println!("  Arch:   {}", hw.cpu.arch);
    println!(
        "  RAM:    {} total / {} available",
        format_bytes(hw.memory.total_bytes),
        format_bytes(hw.memory.available_bytes)
    );
    println!(
        "  SIMD:   {} (width={})",
        hw.simd.detected, hw.simd.vector_width
    );
    if !hw.simd.features.is_empty() {
        println!("  ISA:    {}", hw.simd.features.join(", "));
    }
    for gpu in &hw.gpus {
        println!("  GPU:    {}", gpu.name);
    }
    println!("----------------------------------------------------------------------\n");
}

fn print_features(features: &FeaturesReport) {
    println!("----------------------------------------------------------------------");
    println!("  OPTIMIZATION FEATURES");
    println!("----------------------------------------------------------------------");
    let flags = [
        ("CUDA", features.cuda),
        ("ROCm", features.rocm),
        ("Intel MKL", features.mkl),
        ("Apple Metal", features.metal),
        ("wgpu", features.wgpu),
        ("Shader (naga)", features.shader),
    ];
    for (name, enabled) in &flags {
        let icon = if *enabled { "+" } else { "-" };
        println!("  [{icon}] {name}");
    }
    println!("----------------------------------------------------------------------\n");
}

fn print_peak(peak: &PeakPerformance, hw: &HardwareReport) {
    println!("----------------------------------------------------------------------");
    println!("  THEORETICAL PEAK PERFORMANCE");
    println!("----------------------------------------------------------------------");
    println!(
        "  FP64 peak:     {:.1} GFLOPS ({} cores x {:.1} GHz x {}w SIMD{})",
        peak.fp64_gflops,
        hw.cpu.logical_cores,
        hw.cpu.frequency_mhz as f64 / 1000.0,
        hw.simd.vector_width,
        if peak.has_fma { " x2 FMA" } else { "" }
    );
    println!("  FP32 peak:     {:.1} GFLOPS", peak.fp32_gflops);
    println!("  Mem bandwidth: ~{:.0} GB/s (estimated)", peak.mem_bw_gbs);
    println!("  Rayon threads:  {}", peak.rayon_threads);
    println!("----------------------------------------------------------------------\n");
}

fn print_comparisons(tables: &[ComparisonTable]) {
    for table in tables {
        println!("\n  {}", table.category);
        println!("  {}", "-".repeat(60));
        for entry in &table.entries {
            println!(
                "  {:<40} {:>10}us  ({})",
                entry.operation,
                entry.any_compute_us,
                format_ops(entry.any_compute_ops)
            );
            for cmp in &entry.comparisons {
                let ratio = if entry.any_compute_ops > 0.0 {
                    cmp.estimated_ops / entry.any_compute_ops
                } else {
                    0.0
                };
                let ind = comparison_indicator(ratio);
                let factor = if ratio > 1.0 {
                    format!("{:.1}x {ind}", ratio)
                } else if ratio < 1.0 && ratio > 0.0 {
                    format!("{:.1}x {ind}", 1.0 / ratio)
                } else {
                    "~same".into()
                };
                println!("    vs {:<30} ({}) {}", cmp.library, factor, cmp.notes);
            }
        }
    }
}
