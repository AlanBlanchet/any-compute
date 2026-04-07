---
name: bench
description: Benchmark crate structure, DOM perf comparisons, and GPU dashboard
applyTo: "crates/bench/**"
---

# Benchmarks — `crates/bench/`

Standalone benchmark crate. No CSS/HTML tests here (those belong in `crates/dom/`).
No benchmark code in core.

## Running

- `make dashboard` — GPU dashboard window
- `make bench` — CLI benchmark (writes to `out/`)
- `cargo test -p any-compute-bench` — integration test

## Key Patterns

- `bench_categories!` macro: declares `BenchCategory` enum + metadata + dispatch. Adding a benchmark = one block
- `SHEET` / `s()` / `sm()` pattern for O(1) CSS class resolution (parsed once via `LazyLock`)
- `combined_css()` merges `TAILWIND_CSS` + `BENCH_CSS`
- DOM perf: arena `Tree` vs naive `Box<RefNode>` heap-per-node reference
- Dashboard: GPU-gated (`window` feature), three tabs (Hardware/Benchmarks/Live Showdown)
- Background workers via `std::thread` for hw detection, benchmarks, live throughput
- Uses `any_compute_dom::gpu::Gpu` and `any_compute_dom::theme` for rendering
