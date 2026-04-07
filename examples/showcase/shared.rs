use any_compute_core::animation::{Easing, Transition};
use any_compute_core::compute::Device;
use any_compute_core::kernel::{ReduceOp, UnaryOp, best_kernel};
use any_compute_core::layout::Size;
use any_compute_dom::css::StyleSheet;
use any_compute_dom::tree::*;
use super::app::{AppData, COMBINED_CSS, TAB_LABELS};
use rayon::prelude::*;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

// Constants (interactive-mode only)
// ═══════════════════════════════════════════════════════════════════════════

pub(super) const VIEWPORT: Size = Size::new(1400.0, 900.0);
pub(super) const MAX_FRAME_DT: f64 = 0.032;
const TAB_TRANSITION_DUR: Duration = Duration::from_millis(200);
const HOVER_TRANSITION_DUR: Duration = Duration::from_millis(150);
const TAB_EASING: Easing = Easing::EaseOut;

/// Shorthand for the browser viewport tag.
pub(super) const BVP: &str = super::tabs::browser::VIEWPORT_TAG;

/// Shorthand for the browser DevTools panel tag.
pub(super) const BDT: &str = super::tabs::browser::DEVTOOLS_TAG;

/// Re-export error page HTML from helpers.
pub(super) const ERROR_PAGE_HTML: &str = super::tabs::helpers::ERROR_PAGE_HTML;

/// Minimal HTML entity escaping for error messages.
pub(super) fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// Decode raw HTTP response bytes to String, respecting charset from Content-Type.
/// Handles ISO-8859-1 / Latin-1 (common for older servers like Google).
pub(super) fn decode_http_body(bytes: &[u8], content_type: Option<&str>) -> String {
    let is_latin1 = content_type.map_or(false, |ct| {
        let ct = ct.to_ascii_lowercase();
        ct.contains("iso-8859-1") || ct.contains("latin-1")
    });
    if is_latin1 {
        bytes.iter().map(|&b| b as char).collect()
    } else {
        String::from_utf8_lossy(bytes).into_owned()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Shared state
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Clone)]
pub(super) struct Shared {
    inner: Arc<Mutex<AppData>>,
}

impl Shared {
    pub(super) fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(AppData::new())),
        }
    }

    pub(super) fn read<R>(&self, f: impl FnOnce(&AppData) -> R) -> R {
        f(&self.inner.lock().unwrap_or_else(|e| e.into_inner()))
    }

    pub(super) fn write<R>(&self, f: impl FnOnce(&mut AppData) -> R) -> R {
        f(&mut self.inner.lock().unwrap_or_else(|e| e.into_inner()))
    }

    pub(super) fn on_browser_preview(&self) -> bool {
        self.read(|d| d.tab == 0)
    }

    pub(super) fn on_scene(&self) -> bool {
        self.read(|d| d.tab == 1)
    }

    pub(super) fn on_graph(&self) -> bool {
        self.read(|d| {
            d.graph_mode || (d.tab == 3 && d.ai.center == super::tabs::graph::CenterView::ModelGraph)
        })
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Background workers
// ═══════════════════════════════════════════════════════════════════════════

impl Shared {
    pub(super) fn spawn_hw_detect(&self) {
        let state = self.clone();
        std::thread::spawn(move || {
            let hw = any_compute_bench::runner::detect_hardware();
            state.write(|d| d.hw = Some(hw));
        });
    }

    pub(super) fn spawn_compute_bench(&self) {
        if self.read(|d| d.compute_running) {
            return;
        }
        self.write(|d| {
            d.compute_running = true;
            d.compute_results.clear();
        });
        let state = self.clone();
        std::thread::spawn(move || {
            let dev = Device::cpu();
            let sizes = [10_000usize, 100_000, 1_000_000];
            for &n in &sizes {
                let data: Vec<f64> = (0..n).map(|i| i as f64).collect();
                // Unary throughput
                let t0 = Instant::now();
                for _ in 0..10 {
                    std::hint::black_box(dev.unary(&data, UnaryOp::Sqrt));
                }
                let ops = 10.0 / t0.elapsed().as_secs_f64();
                state.write(|d| d.compute_results.push((format!("Sqrt {n}"), ops)));

                // Reduce throughput
                let t0 = Instant::now();
                for _ in 0..10 {
                    std::hint::black_box(dev.reduce(&data, ReduceOp::Sum));
                }
                let ops = 10.0 / t0.elapsed().as_secs_f64();
                state.write(|d| d.compute_results.push((format!("Sum {n}"), ops)));
            }

            // OpQueue batch benchmark
            {
                use any_compute_core::kernel::UnaryOp;
                let data: Vec<f64> = (0..100_000).map(|i| i as f64).collect();
                let t0 = Instant::now();
                for _ in 0..10 {
                    let mut q = dev.queue();
                    for _ in 0..10 {
                        q.push(any_compute_core::compute::QueuedOp::Unary {
                            data: data.clone(),
                            op: UnaryOp::Sqrt,
                        });
                    }
                    std::hint::black_box(q.flush());
                }
                let ops = 100.0 / t0.elapsed().as_secs_f64();
                state.write(|d| {
                    d.compute_results
                        .push(("OpQueue batch 100K×10".into(), ops))
                });
            }

            // OpCache hit rate
            {
                use any_compute_core::compute::OpCache;
                let cache = OpCache::new(64);
                let data: Vec<f64> = (0..10_000).map(|i| i as f64).collect();
                // Warm
                let _ = cache.reduce(&dev, &data, ReduceOp::Sum);
                let t0 = Instant::now();
                for _ in 0..1000 {
                    std::hint::black_box(cache.reduce(&dev, &data, ReduceOp::Sum));
                }
                let ops = 1000.0 / t0.elapsed().as_secs_f64();
                state.write(|d| d.compute_results.push(("OpCache hit ×1K".into(), ops)));
            }

            state.write(|d| d.compute_running = false);
        });
    }

    pub(super) fn spawn_live_sim(&self) {
        if self.read(|d| d.live_running) {
            return;
        }
        self.write(|d| d.live_running = true);
        let state = self.clone();
        std::thread::spawn(move || {
            let kern = best_kernel();
            let n = 100_000usize;
            let data: Vec<f64> = (0..n).map(|i| i as f64).collect();
            let iters = 5u32;

            loop {
                if !state.read(|d| d.live_running) {
                    break;
                }
                let fi = iters as f64;
                // Our kernel
                let t0 = Instant::now();
                for _ in 0..iters {
                    std::hint::black_box(kern.map_unary_f64(&data, UnaryOp::Sqrt));
                }
                let ac = fi / t0.elapsed().as_secs_f64();

                // Rayon
                let t0 = Instant::now();
                for _ in 0..iters {
                    let _: Vec<f64> = data.par_iter().map(|x| x.sqrt()).collect();
                }
                let ray = fi / t0.elapsed().as_secs_f64();

                // Std
                let t0 = Instant::now();
                for _ in 0..iters {
                    let _: Vec<f64> = data.iter().map(|x| x.sqrt()).collect();
                }
                let st = fi / t0.elapsed().as_secs_f64();

                state.write(|d| {
                    d.live_ac_ops = ac;
                    d.live_rayon_ops = ray;
                    d.live_std_ops = st;
                });
                std::thread::sleep(Duration::from_millis(250));
            }
        });
    }
}


// ═══════════════════════════════════════════════════════════════════════════
// Tree building & event handling
// ═══════════════════════════════════════════════════════════════════════════

impl Shared {
    pub(super) fn build_tree(&self, sheet: &StyleSheet, w: f64, h: f64) -> Tree {
        self.inner.lock().unwrap().build_tree(sheet, w, h)
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Event handling
    // ═══════════════════════════════════════════════════════════════════════════

    pub(super) fn handle_click(&self, tag: &str) {
        // Tab switching
        if let Some(idx_str) = tag.strip_prefix("tab-") {
            if let Ok(idx) = idx_str.parse::<usize>() {
                self.write(|d| {
                    if d.tab != idx {
                        let old = d.tab;
                        d.tab = idx;
                        d.scroll_y = 0.0;
                        d.scroll_target = 0.0;
                        // Transition old tab out, new tab in
                        d.transitions.add(
                            &AppData::tab_tag(old),
                            Transition::new(1.0, 0.0, TAB_TRANSITION_DUR).with_easing(TAB_EASING),
                        );
                        d.transitions.add(
                            &AppData::tab_tag(idx),
                            Transition::new(0.0, 1.0, TAB_TRANSITION_DUR).with_easing(TAB_EASING),
                        );
                    }
                });
            }
        }

        // Compute tab: run benchmarks
        if tag == "run-bench" {
            self.spawn_compute_bench();
        }

        // Graph mode toggle
        if tag == "toggle-graph-mode" {
            self.write(|d| {
                d.graph_mode = !d.graph_mode;
                if d.graph_mode {
                    d.ai.clear_graph();
                }
            });
        }

        // Live tab: toggle simulation
        if tag == "toggle-sim" {
            let running = self.read(|d| d.live_running);
            if running {
                self.write(|d| d.live_running = false);
            } else {
                self.spawn_live_sim();
            }
        }

        // Subtab switching: "subtab-{tab}-{sub}"
        if let Some(rest) = tag.strip_prefix("subtab-") {
            let parts: Vec<&str> = rest.split('-').collect();
            if parts.len() == 2 {
                if let (Ok(tab), Ok(sub)) = (parts[0].parse::<usize>(), parts[1].parse::<usize>()) {
                    self.write(|d| {
                        if tab < d.subtabs.len() {
                            d.subtabs[tab] = sub;
                        }
                    });
                }
            }
        }

        // Scene hierarchy selection: "scene-obj-{idx}"
        if let Some(idx_str) = tag.strip_prefix("scene-obj-") {
            if let Ok(idx) = idx_str.parse::<usize>() {
                self.write(|d| {
                    d.scene_info.selected = if d.scene_info.selected == Some(idx) {
                        None // toggle off
                    } else {
                        Some(idx)
                    };
                });
            }
        }

        // Browser tab: reload, back, forward, devtools toggle, or blur
        if tag == "browser-reload" {
            self.write(|d| d.browser.reload(&COMBINED_CSS));
        } else if tag == "browser-back" {
            self.write(|d| d.browser.go_back(&COMBINED_CSS));
        } else if tag == "browser-forward" {
            self.write(|d| d.browser.go_forward(&COMBINED_CSS));
        } else if tag == "browser-devtools-toggle" {
            self.write(|d| d.browser.toggle_devtools());
        } else if tag != "browser-input" {
            self.write(|d| d.browser.input.blur());
        }

        // AI tab: delegate all ai-* tags to AiState
        if tag.starts_with("ai-") {
            self.write(|d| {
                d.ai.handle_click(tag);
            });
        }
    }

    pub(super) fn handle_hover(&self, tag: Option<String>) {
        self.write(|d| {
            for i in 0..TAB_LABELS.len() {
                let key = AppData::hover_tag(i);
                let is_hovered = tag.as_deref() == Some(&AppData::tab_tag(i));
                let current = d.transitions.value(&key).unwrap_or(0.0);
                let target = if is_hovered { 1.0 } else { 0.0 };
                if (current - target).abs() > 0.01 {
                    d.transitions.add(
                        &key,
                        Transition::new(current, target, HOVER_TRANSITION_DUR)
                            .with_easing(TAB_EASING),
                    );
                }
            }
        });
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════════════
