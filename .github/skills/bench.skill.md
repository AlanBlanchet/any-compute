---
name: bench
description: Benchmark structure, categories, comparison methodology, and device simulation
applyTo: "crates/core/src/bench.rs,crates/rsx/src/bin/bench_window.rs,crates/core/src/bin/anc_bench.rs"
---

# Benchmarks

## Structure

- `BenchCategory::ALL` is the canonical run order — add new categories there only.
- `ScenarioReport` groups timed results under a category id — one report per category per run.
- `BenchResult` is a single timed measurement: `name`, `scale`, `duration_us`, `throughput_ops_sec`.
- Categories span lightweight ops (VecAdd, DotProduct, Reduce, MapTransform, PrefixSum, Sort, FilterSearch) **and** heavyweight workloads (PointCloud 1M+, MatMulLarge 256–1024, AttentionOps, Geometry3D sphere transforms).

## Comparisons

- `ComparisonTable` holds any-compute ops + a `Vec<LibComparison>` of competitor entries.
- Each `LibComparison` carries a `ComparisonSource` enum: `Measured` (timed in-process) or `Estimate` (published ratio).
- Factor < 1.0 = competitor slower; > 1.0 = competitor faster. Green bar = ours; color-coded bars for competitors.
- `build_comparison_tables()` uses rayon to run real measurements against ndarray / nalgebra / std.

## Device simulation

- `SimulatedBackend` wraps a `DeviceProfile` to throttle compute/bandwidth for realistic projections.
- `all_profiles()` returns the canonical profile set (desktop, laptop, mobile, IoT, WASM).

## Dashboard

- `anc-bench-window` is the live Dioxus GUI — tabs: Overview, Benchmarks, Comparisons, Live Metrics, Device Profiles, Platforms.
- **Overview** shows system info (CPU, ISA, memory, SIMD), performance estimate cards, and a compact feature-chip grid showing all `--features` flags.
- **Benchmarks** and **Comparisons** render results as aligned bar charts using CSS grid (`.bar-row` with `grid-template-columns: 160px 1fr 145px`) — never text tables.
- All tabs read from core types — no benchmark logic lives in the window binary.
- `FeatureDetail` struct holds only `enabled`, `label`, `enable_cmd` — keep it lean.

## CSS conventions

- `.bar-row` is CSS grid, not flex — bars always align regardless of label length.
- `.feature-chip` (`.enabled` / `.disabled`) for compact feature flag display with icon + label + cmd.
