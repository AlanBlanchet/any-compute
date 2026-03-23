//! Live Metrics tab — real-time throughput comparison, FPS graph, memory.
//!
//! Background thread continuously measures kernel throughput (our SIMD vs rayon vs std)
//! and streams results into the UI at ~10Hz.

use any_compute_dom::css::StyleSheet;
use any_compute_dom::style::*;
use any_compute_dom::tree::*;
use any_compute_dom::theme;
use any_compute_core::render::Color;

use super::helpers::{s, sm, kv_card, format_ops};
use crate::AppData;

pub fn build(sheet: &StyleSheet, t: &mut Tree, parent: NodeId, data: &AppData) {
    t.add_text(parent, "Live Metrics", s(sheet, "title"));
    t.add_text(
        parent,
        "Real-time throughput comparison — SIMD kernel vs Rayon vs std::iter",
        s(sheet, "subtitle"),
    );
    t.add_box(parent, s(sheet, "spacer-12"));

    // Toggle button
    let btn = t.add_box(parent, s(sheet, "demo-btn"));
    let label = if data.live_running { "Stop Simulation" } else { "Start Simulation" };
    t.add_text(btn, label, sm(sheet, &["font-13"]).color(Color::from((30, 30, 46))));
    t.tag(btn, "toggle-sim");

    t.add_box(parent, s(sheet, "spacer-12"));

    // Throughput bars
    t.add_text(parent, "Sqrt 1M elements — ops/sec", sm(sheet, &["heading", "text"]));
    t.add_box(parent, s(sheet, "spacer-12"));

    let max = data
        .live_ac_ops
        .max(data.live_rayon_ops)
        .max(data.live_std_ops)
        .max(1.0);

    throughput_bar(t, parent, sheet, "SIMD Kernel", data.live_ac_ops, max, theme::ACCENT);
    throughput_bar(t, parent, sheet, "Rayon", data.live_rayon_ops, max, Color::from((166, 227, 161)));
    throughput_bar(t, parent, sheet, "std::iter", data.live_std_ops, max, Color::from((249, 226, 175)));

    t.add_box(parent, s(sheet, "spacer-24"));

    // Frame time graph
    t.add_text(parent, "Frame Times (ms)", sm(sheet, &["heading", "text"]));
    t.add_box(parent, s(sheet, "spacer-12"));
    let graph = t.add_box(parent, s(sheet, "card").h(120.0));
    if !data.frame_times.is_empty() {
        let max_ms = data.frame_times.iter().fold(1.0f64, |a, &b| a.max(b));
        let bar_w = 3.0;
        let row = t.add_box(graph, sm(sheet, &["row-gap-2"]).h(100.0));
        // Show last N frames that fit
        let visible = ((700.0 / (bar_w + 2.0)) as usize).min(data.frame_times.len());
        let start = data.frame_times.len().saturating_sub(visible);
        for &ms in &data.frame_times[start..] {
            let h = (ms / max_ms * 90.0).max(2.0);
            let color = if ms > 16.7 {
                Color::from((243, 139, 168)) // red
            } else if ms > 8.0 {
                Color::from((249, 226, 175)) // yellow
            } else {
                theme::ACCENT // blue
            };
            let mut bar_s = Style::default().w(bar_w).h(h).bg(color).radius(1.0);
            bar_s.align_self = Some(Align::End);
            t.add_box(row, bar_s);
        }
    } else {
        t.add_text(graph, "Waiting for frames...", sm(sheet, &["font-14", "text-dim"]));
    }

    t.add_box(parent, s(sheet, "spacer-24"));

    // Hardware info
    if let Some(hw) = &data.hw {
        t.add_text(parent, "Hardware", sm(sheet, &["heading", "text"]));
        let info = t.add_box(parent, sm(sheet, &["row-gap-16"]));

        kv_card(t, info, sheet, "CPU", &hw.cpu.brand);
        kv_card(t, info, sheet, "Cores", &format!("{} physical", hw.cpu.physical_cores));
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
    t.add_text(fps_card, &data.fps.to_string(), sm(sheet, &["font-24", "green"]));

    if !data.frame_times.is_empty() {
        let avg = data.frame_times.iter().sum::<f64>() / data.frame_times.len() as f64;
        let avg_card = t.add_box(fps_row, s(sheet, "card-sm").w(120.0));
        t.add_text(avg_card, "Avg Frame", s(sheet, "label"));
        t.add_text(avg_card, &format!("{avg:.1}ms"), sm(sheet, &["font-24", "blue"]));
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
    let _bar = t.add_box(row, s(sheet, "bar-medium").w(bar_w).bg(color));

    t.add_text(row, &format_ops(ops), sm(sheet, &["font-13", "text"]));
}
