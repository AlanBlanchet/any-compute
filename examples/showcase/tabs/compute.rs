//! Compute tab — benchmarks + live metrics, split into subtabs.
//!
//! Subtabs:
//!   0. Benchmarks — kernel benchmarks, OpQueue batch, OpCache hit rate
//!   1. Live       — real-time throughput comparison, FPS graph, hardware

use any_compute_core::render::Color;
use any_compute_dom::css::StyleSheet;
use any_compute_dom::style::*;
use any_compute_dom::theme;
use any_compute_dom::tree::*;

use super::helpers::{build_subtab_bar, format_ops, kv_card, s, sm};
use crate::AppData;

/// Subtab labels for the Compute tab.
pub const SUBTABS: &[&str] = &["Benchmarks", "Live"];

pub fn build(sheet: &StyleSheet, t: &mut Tree, parent: NodeId, data: &AppData, subtab: usize) {
    build_subtab_bar(sheet, t, parent, SUBTABS, subtab, "subtab-2-");

    match subtab {
        0 => build_benchmarks(sheet, t, parent, data),
        1 => build_live(sheet, t, parent, data),
        _ => {}
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Benchmarks subtab
// ═══════════════════════════════════════════════════════════════════════════

fn build_benchmarks(sheet: &StyleSheet, t: &mut Tree, parent: NodeId, data: &AppData) {
    t.add_text(parent, "Compute Pipeline", s(sheet, "title"));
    t.add_text(
        parent,
        "SIMD kernel dispatch — click Run to compare against Rayon and std::iter",
        s(sheet, "subtitle"),
    );
    t.add_box(parent, s(sheet, "spacer-12"));

    // ── Compact pipeline + run button on same row ───────────────────
    let top_row = t.add_box(parent, sm(sheet, &["row-gap-12"]).align(Align::Center));

    let pipe_row = t.add_box(top_row, sm(sheet, &["row-gap-8"]).align(Align::Center));
    for (i, (name, detail, cls)) in [
        ("Input", "100K f64", "badge-green"),
        ("Sqrt", "SIMD √x", "badge-blue"),
        ("Reduce", "Σ = sum", "badge-yellow"),
        ("Output", "scalar", "badge-red"),
    ]
    .into_iter()
    .enumerate()
    {
        if i > 0 {
            t.add_text(pipe_row, "→", sm(sheet, &["font-14", "text-dim"]));
        }
        let stage = t.add_box(pipe_row, sm(sheet, &["badge", cls]).w(80.0));
        t.add_text(stage, name, sm(sheet, &["font-11"]).color(Color::WHITE));
        t.add_text(
            stage,
            detail,
            s(sheet, "font-9").color(Color::rgba(200, 200, 200, 180)),
        );
    }

    let btn = t.add_box(top_row, s(sheet, "demo-btn"));
    let label = if data.compute_running {
        "Running..."
    } else {
        "Run Benchmarks"
    };
    t.add_text(btn, label, sm(sheet, &["font-13"]).color(theme::BG));
    t.tag(btn, "run-bench");

    t.add_box(parent, s(sheet, "spacer-12"));

    // ── Results table ───────────────────────────────────────────────
    if !data.compute_results.is_empty() {
        t.add_text(
            parent,
            "Throughput Comparison",
            sm(sheet, &["heading", "text"]),
        );

        let hdr = t.add_box(parent, sm(sheet, &["table-row"]));
        t.add_text(hdr, "Operation", sm(sheet, &["table-header"]).w(200.0));
        t.add_text(hdr, "Ops/sec", sm(sheet, &["table-header"]).w(120.0));
        t.add_text(hdr, "Throughput", sm(sheet, &["table-header"]).grow(1.0));

        let max_ops = data
            .compute_results
            .iter()
            .map(|(_, ops)| *ops)
            .fold(0.0f64, f64::max);

        for (i, (name, ops)) in data.compute_results.iter().enumerate() {
            let classes: Vec<&str> = if i % 2 == 1 {
                vec!["table-row", "table-row-alt"]
            } else {
                vec!["table-row"]
            };
            let row = t.add_box(parent, sm(sheet, &classes));
            t.add_text(row, name, sm(sheet, &["table-cell"]).w(200.0));
            t.add_text(
                row,
                &format_ops(*ops),
                sm(sheet, &["table-cell", "green"]).w(120.0),
            );
            let bar_w = if max_ops > 0.0 {
                (*ops / max_ops) * 300.0
            } else {
                0.0
            };
            t.add_box(row, s(sheet, "bar-thin").w(bar_w).bg(theme::ACCENT));
        }
    } else if data.compute_running {
        t.add_text(
            parent,
            "Running benchmarks...",
            sm(sheet, &["font-14", "yellow"]),
        );
    } else {
        t.add_text(
            parent,
            "Press Run to execute kernel benchmarks.",
            sm(sheet, &["font-14", "text-dim"]),
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Live metrics subtab
// ═══════════════════════════════════════════════════════════════════════════

fn build_live(sheet: &StyleSheet, t: &mut Tree, parent: NodeId, data: &AppData) {
    t.add_text(parent, "Live Metrics", s(sheet, "title"));
    t.add_text(
        parent,
        "Real-time throughput comparison — SIMD kernel vs Rayon vs std::iter",
        s(sheet, "subtitle"),
    );
    t.add_box(parent, s(sheet, "spacer-12"));

    // Toggle button
    let btn = t.add_box(parent, s(sheet, "demo-btn"));
    let label = if data.live_running {
        "Stop Simulation"
    } else {
        "Start Simulation"
    };
    t.add_text(btn, label, sm(sheet, &["font-13"]).color(theme::BG));
    t.tag(btn, "toggle-sim");

    t.add_box(parent, s(sheet, "spacer-12"));

    // Throughput bars
    t.add_text(
        parent,
        "Sqrt 1M elements — ops/sec",
        sm(sheet, &["heading", "text"]),
    );
    t.add_box(parent, s(sheet, "spacer-12"));

    let max = data
        .live_ac_ops
        .max(data.live_rayon_ops)
        .max(data.live_std_ops)
        .max(1.0);

    throughput_bar(
        t,
        parent,
        sheet,
        "SIMD Kernel",
        data.live_ac_ops,
        max,
        theme::ACCENT,
    );
    throughput_bar(
        t,
        parent,
        sheet,
        "Rayon",
        data.live_rayon_ops,
        max,
        theme::GREEN,
    );
    throughput_bar(
        t,
        parent,
        sheet,
        "std::iter",
        data.live_std_ops,
        max,
        theme::YELLOW,
    );

    t.add_box(parent, s(sheet, "spacer-24"));

    // Frame time graph
    t.add_text(parent, "Frame Times (ms)", sm(sheet, &["heading", "text"]));
    t.add_box(parent, s(sheet, "spacer-12"));
    let graph = t.add_box(parent, s(sheet, "card").h(120.0));
    if !data.frame_times.is_empty() {
        let max_ms = data.frame_times.iter().fold(1.0f64, |a, &b| a.max(b));
        let bar_w = 3.0;
        let row = t.add_box(graph, sm(sheet, &["row-gap-2"]).h(100.0));
        let visible = ((700.0 / (bar_w + 2.0)) as usize).min(data.frame_times.len());
        let start = data.frame_times.len().saturating_sub(visible);
        for &ms in &data.frame_times[start..] {
            let h = (ms / max_ms * 90.0).max(2.0);
            let color = if ms > 16.7 {
                theme::RED
            } else if ms > 8.0 {
                theme::YELLOW
            } else {
                theme::ACCENT
            };
            let mut bar_s = Style::default().w(bar_w).h(h).bg(color).radius(1.0);
            bar_s.align_self = Some(Align::End);
            t.add_box(row, bar_s);
        }
    } else {
        t.add_text(
            graph,
            "Waiting for frames...",
            sm(sheet, &["font-14", "text-dim"]),
        );
    }

    t.add_box(parent, s(sheet, "spacer-24"));

    // Hardware info
    if let Some(hw) = &data.hw {
        t.add_text(parent, "Hardware", sm(sheet, &["heading", "text"]));
        let info = t.add_box(parent, sm(sheet, &["row-gap-16"]));
        kv_card(t, info, sheet, "CPU", &hw.cpu.brand);
        kv_card(
            t,
            info,
            sheet,
            "Cores",
            &format!("{} physical", hw.cpu.physical_cores),
        );
        kv_card(
            t,
            info,
            sheet,
            "Memory",
            &format!("{} GB", hw.memory.total_bytes / (1024 * 1024 * 1024)),
        );
    }

    t.add_box(parent, s(sheet, "spacer-12"));

    // FPS info
    let fps_row = t.add_box(parent, sm(sheet, &["row-gap-16"]));
    let fps_card = t.add_box(fps_row, s(sheet, "card-sm").w(120.0));
    t.add_text(fps_card, "FPS", s(sheet, "label"));
    t.add_text(
        fps_card,
        &data.fps.to_string(),
        sm(sheet, &["font-24", "green"]),
    );

    if !data.frame_times.is_empty() {
        let avg = data.frame_times.iter().sum::<f64>() / data.frame_times.len() as f64;
        let avg_card = t.add_box(fps_row, s(sheet, "card-sm").w(120.0));
        t.add_text(avg_card, "Avg Frame", s(sheet, "label"));
        t.add_text(
            avg_card,
            &format!("{avg:.1}ms"),
            sm(sheet, &["font-24", "blue"]),
        );
    }
}

fn throughput_bar(
    t: &mut Tree,
    parent: NodeId,
    sheet: &StyleSheet,
    label: &str,
    ops: f64,
    max: f64,
    color: Color,
) {
    let row = t.add_box(parent, sm(sheet, &["row-gap-8"]));
    t.add_text(row, label, sm(sheet, &["font-13", "text"]).w(100.0));
    let bar_w = (ops / max * 400.0).max(4.0);
    t.add_box(row, s(sheet, "bar-medium").w(bar_w).bg(color));
    t.add_text(row, &format_ops(ops), sm(sheet, &["font-13", "text"]));
}
