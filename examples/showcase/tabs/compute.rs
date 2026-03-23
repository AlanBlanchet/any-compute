//! Compute tab — kernel benchmarks, OpQueue batch, OpCache hit rate.
//!
//! Shows results from background benchmark worker + static feature info.

use any_compute_dom::css::StyleSheet;
use any_compute_dom::tree::*;
use any_compute_dom::theme;
use any_compute_core::render::Color;

use super::helpers::{s, sm, format_ops};
use crate::AppData;

pub fn build(sheet: &StyleSheet, t: &mut Tree, parent: NodeId, data: &AppData) {
    t.add_text(parent, "Compute Benchmarks", s(sheet, "title"));
    t.add_text(
        parent,
        "Kernel, OpQueue batch, OpCache memoization — SIMD-accelerated",
        s(sheet, "subtitle"),
    );
    t.add_box(parent, s(sheet, "spacer-12"));

    // Run button
    let btn = t.add_box(parent, s(sheet, "demo-btn"));
    let label = if data.compute_running { "Running..." } else { "Run Benchmarks" };
    t.add_text(btn, label, sm(sheet, &["font-13"]).color(Color::from((30, 30, 46))));
    t.tag(btn, "run-bench");

    t.add_box(parent, s(sheet, "spacer-12"));

    // Results table
    if !data.compute_results.is_empty() {
        t.add_text(parent, "Results", sm(sheet, &["heading", "text"]));

        // Header
        let hdr = t.add_box(parent, sm(sheet, &["table-row", "table-header-row"]));
        t.add_text(hdr, "Benchmark", sm(sheet, &["table-header"]).w(200.0));
        t.add_text(hdr, "Ops/sec", sm(sheet, &["table-header"]).w(120.0));
        t.add_text(hdr, "Throughput", sm(sheet, &["table-header"]).grow(1.0));

        let max_ops = data
            .compute_results
            .iter()
            .map(|(_, ops)| *ops)
            .fold(0.0f64, f64::max);

        for (i, (name, ops)) in data.compute_results.iter().enumerate() {
            let alt = if i % 2 == 1 { "table-row-alt" } else { "" };
            let classes: Vec<&str> = if alt.is_empty() {
                vec!["table-row"]
            } else {
                vec!["table-row", alt]
            };
            let row = t.add_box(parent, sm(sheet, &classes));
            t.add_text(row, name, sm(sheet, &["table-cell"]).w(200.0));
            t.add_text(row, &format_ops(*ops), sm(sheet, &["table-cell", "green"]).w(120.0));

            // Bar
            let bar_w = if max_ops > 0.0 { (*ops / max_ops) * 300.0 } else { 0.0 };
            let _bar = t.add_box(row, s(sheet, "bar-thin").w(bar_w).bg(theme::ACCENT));
        }
    } else if data.compute_running {
        t.add_text(parent, "Running benchmarks...", sm(sheet, &["font-14", "yellow"]));
    }

    t.add_box(parent, s(sheet, "spacer-24"));

    // Feature overview
    t.add_text(parent, "Compute Features", sm(sheet, &["heading", "text"]));
    let features = [
        ("Device", "Unified dispatch — cpu(), best(), simulated(), from_kernel()"),
        ("Kernel", "SIMD unary/binary/reduce/scan/sort/gemm/gather/scatter"),
        ("OpQueue", "Lazy batch dispatch — record ops, flush all at once"),
        ("OpCache", "Content-addressed memoization — FNV-1a hash keys"),
        ("Buffer", "Device-aware Vec<f64> with macro-generated ops"),
        ("Hints", "Per-call tuning — parallelism threshold, batch size"),
    ];
    for (label, desc) in &features {
        let row = t.add_box(parent, sm(sheet, &["row-gap-8"]));
        t.add_text(row, *label, sm(sheet, &["font-13", "mauve"]).w(80.0));
        t.add_text(row, *desc, s(sheet, "body"));
    }
}


