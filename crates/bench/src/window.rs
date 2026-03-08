//! Native WGPU benchmark dashboard — zero frameworks, zero DOM, zero overhead.
//!
//! Renders directly to GPU via instanced draw calls.
//! Runs real `any_compute_core::bench` workloads on background threads and
//! streams results into the render loop at 60+ FPS with zero stutter.

use any_compute_canvas::gpu::Gpu;
use any_compute_canvas::theme;
use any_compute_canvas::winit;
use any_compute_core::Lerp;
use any_compute_core::animation::{Easing, Transition, TransitionManager};
use any_compute_core::bench::*;
use any_compute_core::interaction::{Button, FocusState, HoverState, InputEvent, Modifiers};
use any_compute_core::kernel::{UnaryOp, best_kernel};
use any_compute_core::layout::{Point, Size};
use any_compute_core::render::{Color, RenderList};
use any_compute_dom::style::*;
use any_compute_dom::tree::*;
use rayon::prelude::*;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use std::time::Instant;
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    keyboard::{Key, NamedKey},
    window::WindowBuilder,
};

// ── Shared async state ──────────────────────────────────────────────────────
#[derive(Clone)]
struct SharedState {
    inner: Arc<Mutex<AppData>>,
}

struct AppData {
    hw: Option<HardwareReport>,
    bench_results: Vec<ScenarioReport>,
    bench_running: bool,
    bench_progress: (usize, usize),
    current_cat: Option<String>,
    sim_running: bool,
    ac_ops: f64,
    rayon_ops: f64,
    std_ops: f64,
    tab: usize,
    scroll_y: f64,
    scroll_target: f64,
    transitions: TransitionManager,
    hover: HoverState,
    focus: FocusState,
    /// Whether any pointer button is currently pressed.
    pointer_down: bool,
    /// Pressed tag (for active/pressed visual state).
    active_tag: Option<String>,
}

impl SharedState {
    fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(AppData {
                hw: None,
                bench_results: Vec::new(),
                bench_running: false,
                bench_progress: (0, 0),
                current_cat: None,
                sim_running: false,
                ac_ops: 0.0,
                rayon_ops: 0.0,
                std_ops: 0.0,
                tab: 0,
                scroll_y: 0.0,
                scroll_target: 0.0,
                transitions: {
                    let mut mgr = TransitionManager::default();
                    // Tab-0 starts active.
                    let mut t = Transition::new(0.0, 1.0, Duration::ZERO);
                    t.start();
                    mgr.add("tab-0", t);
                    mgr
                },
                hover: HoverState::default(),
                focus: FocusState::default(),
                pointer_down: false,
                active_tag: None,
            })),
        }
    }

    fn read<R>(&self, f: impl FnOnce(&AppData) -> R) -> R {
        f(&self.inner.lock().unwrap())
    }

    fn write<R>(&self, f: impl FnOnce(&mut AppData) -> R) -> R {
        f(&mut self.inner.lock().unwrap())
    }
}

// ── Background workers ──────────────────────────────────────────────────────
fn spawn_hw_detect(state: SharedState) {
    std::thread::spawn(move || {
        let hw = detect_hardware();
        state.write(|d| d.hw = Some(hw));
    });
}

fn spawn_benchmarks(state: SharedState) {
    if state.read(|d| d.bench_running) {
        return;
    }
    state.write(|d| {
        d.bench_running = true;
        d.bench_results.clear();
        d.bench_progress = (0, BenchCategory::ALL.len());
    });

    std::thread::spawn(move || {
        for (i, &cat) in BenchCategory::ALL.iter().enumerate() {
            state.write(|d| d.current_cat = Some(cat.label().to_string()));
            match std::panic::catch_unwind(|| run_category(cat)) {
                Ok(report) => {
                    state.write(|d| {
                        d.bench_results.push(report);
                        d.bench_progress.0 = i + 1;
                    });
                }
                Err(_) => {
                    state.write(|d| {
                        let mut r = ScenarioReport::default();
                        r.category = format!("{} (CRASHED)", cat.label());
                        d.bench_results.push(r);
                        d.bench_progress.0 = i + 1;
                    });
                }
            }
        }
        state.write(|d| {
            d.bench_running = false;
            d.current_cat = None;
        });
    });
}

fn spawn_simulation(state: SharedState) {
    if state.read(|d| d.sim_running) {
        state.write(|d| d.sim_running = false);
        return;
    }
    state.write(|d| {
        d.sim_running = true;
        d.ac_ops = 0.0;
        d.rayon_ops = 0.0;
        d.std_ops = 0.0;
    });

    fn throughput_loop(
        s: SharedState,
        compute: impl Fn(&[f64]) -> Vec<f64> + Send + 'static,
        report: impl Fn(&mut AppData, f64) + Send + 'static,
    ) {
        std::thread::spawn(move || {
            let data = vec![1.0_f64; 200_000];
            let mut last = Instant::now();
            let mut ops = 0usize;
            while s.read(|d| d.sim_running) {
                std::hint::black_box(&compute(&data));
                ops += data.len();
                let el = last.elapsed().as_secs_f64();
                if el > 0.3 {
                    s.write(|d| report(d, ops as f64 / el));
                    ops = 0;
                    last = Instant::now();
                }
            }
        });
    }

    throughput_loop(
        state.clone(),
        |data| {
            let kern = best_kernel();
            kern.map_unary_f64(data, UnaryOp::Sigmoid)
        },
        |d, t| d.ac_ops = t,
    );

    throughput_loop(
        state.clone(),
        |data| data.par_iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect(),
        |d, t| d.rayon_ops = t,
    );

    throughput_loop(
        state.clone(),
        |data| data.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect(),
        |d, t| d.std_ops = t,
    );
}

// ── Layout / Paint ──────────────────────────────────────────────────────────
use any_compute_bench::{SHEET, TAB_LABELS, VERSION, kv_row};

fn s(class: &str) -> Style {
    SHEET.class(class)
}
fn sm(classes: &[&str]) -> Style {
    SHEET.classes(classes)
}

// ── DOM construction helpers ────────────────────────────────────────────────
fn section_hdr(t: &mut Tree, parent: NodeId, title: &str) -> NodeId {
    let hdr = t.add_box(parent, s("section-hdr"));
    t.add_text(hdr, title, s("title"));
    t.add_box(hdr, s("grow"));
    hdr
}

fn action_btn(
    t: &mut Tree,
    parent: NodeId,
    label: &str,
    bg: Color,
    fg: Color,
    tag: &str,
    hover_alpha: f64,
) -> NodeId {
    let mut btn_s = s("btn");
    // Hover: brighten background slightly.
    btn_s.background = bg.lerp(Color::WHITE, hover_alpha * 0.15);
    let btn = t.add_box(parent, btn_s);
    t.add_text(btn, label, s("body").color(fg));
    t.tag(btn, tag);
    btn
}

fn card(t: &mut Tree, parent: NodeId, title: &str, accent: Color) -> NodeId {
    let c = t.add_box(parent, s("card"));
    t.add_text(c, title, s("heading").color(accent));
    c
}

fn build_tree(state: &SharedState, w: f64, h: f64) -> Tree {
    let mut data = state.inner.lock().unwrap();

    // Smooth scroll: lerp scroll_y toward scroll_target each frame.
    let scroll_speed = 0.18;
    data.scroll_y += (data.scroll_target - data.scroll_y) * scroll_speed;
    if (data.scroll_target - data.scroll_y).abs() < 0.5 {
        data.scroll_y = data.scroll_target;
    }

    let mut t = Tree::new(sm(&["bg", "row"]).w(w).h(h));
    let root = t.root;

    // ── Sidebar ──
    let sb = t.add_box(root, s("sidebar"));
    let brand = t.add_box(sb, s("brand"));
    t.add_box(brand, s("brand-icon"));
    let btext = t.add_box(brand, s("brand-text"));
    t.add_text(btext, "any-compute", s("heading-text"));
    t.add_text(btext, VERSION, s("small-dim"));
    t.add_box(sb, s("spacer-12"));
    for (i, &label) in TAB_LABELS.iter().enumerate() {
        let tag_name = format!("tab-{i}");
        // Transition-driven tab appearance: blend between inactive and active.
        let alpha = data
            .transitions
            .value(&tag_name)
            .unwrap_or(if data.tab == i { 1.0 } else { 0.0 });
        // Hover glow: subtle highlight when hovered.
        let hover_alpha = data
            .transitions
            .value(&format!("hover-{tag_name}"))
            .unwrap_or(0.0);
        let base_bg = Color::TRANSPARENT.lerp(theme::ACCENT, alpha);
        let bg = base_bg.lerp(theme::SURFACE_BRIGHT, hover_alpha * (1.0 - alpha));
        let fg = theme::TEXT_DIM.lerp(theme::SIDEBAR_BG, alpha);
        let mut tab_s = s("tab-btn");
        tab_s.background = bg;
        tab_s.color = fg;
        let btn = t.add_box(sb, tab_s);
        t.add_text(btn, label, s("font-13").color(fg));
        t.tag(btn, &tag_name);
    }

    // ── Main area ──
    let main_col = t.add_box(root, s("grow"));
    let hdr = t.add_box(main_col, s("header"));
    t.add_text(hdr, TAB_LABELS[data.tab], sm(&["font-18", "text"]));
    let content = t.add_box(main_col, s("content"));
    t.slot_mut(content).scroll.y = data.scroll_y;

    match data.tab {
        0 => build_hw(&mut t, content, &data),
        1 => build_bench(&mut t, content, &mut data),
        2 => build_sim(&mut t, content, &mut data),
        _ => {}
    }

    t
}

fn build_hw(t: &mut Tree, p: NodeId, data: &AppData) {
    t.add_text(p, "Hardware Profile", s("title"));
    t.add_text(p, "Detected system capabilities", s("subtitle"));

    let Some(hw) = &data.hw else {
        t.add_text(p, "Detecting hardware\u{2026}", sm(&["font-14", "yellow"]));
        return;
    };

    let row = t.add_box(p, s("row-gap-12"));

    let c = card(t, row, "Processor", theme::ACCENT);
    let topo = format!(
        "{} cores / {} threads",
        hw.cpu.physical_cores, hw.cpu.logical_cores
    );
    let freq = format_hz(hw.cpu.frequency_mhz);
    for (lbl, val) in [
        ("Brand", hw.cpu.brand.as_str()),
        ("Arch", hw.cpu.arch.as_str()),
        ("Topology", topo.as_str()),
        ("Frequency", freq.as_str()),
    ] {
        kv_row(t, c, lbl, val);
    }

    let c = card(t, row, "SIMD / Vector", theme::GREEN);
    t.add_text(c, &hw.simd.detected, s("body"));
    t.add_text(
        c,
        &format!("{}-bit vectors", hw.simd.vector_width),
        s("label"),
    );
    let tags_str = hw.simd.features.join("  \u{00b7}  ");
    t.add_text(c, &tags_str, sm(&["small", "mauve"]));

    let c = card(t, row, "Memory & GPU", theme::YELLOW);
    let total_gb = hw.memory.total_bytes / 1024 / 1024 / 1024;
    let avail_gb = hw.memory.available_bytes / 1024 / 1024 / 1024;
    t.add_text(
        c,
        &format!("{total_gb} GB total / {avail_gb} GB available"),
        s("body"),
    );
    let pct = if hw.memory.total_bytes > 0 {
        hw.memory.used_bytes as f64 / hw.memory.total_bytes as f64
    } else {
        0.0
    };
    let bar_c = if pct > 0.8 { theme::RED } else { theme::GREEN };
    t.add_bar(c, pct, bar_c, s("bar-thin"));
    t.add_text(c, &format!("{:.0}% used", pct * 100.0), s("small-dim"));
    for gpu in &hw.gpus {
        let g = t.add_box(c, s("gpu-badge"));
        t.add_text(g, &gpu.name, s("label"));
    }
    if hw.gpus.is_empty() {
        t.add_text(c, "No GPU detected", s("label"));
    }
}

fn build_bench(t: &mut Tree, p: NodeId, data: &mut AppData) {
    let hdr = section_hdr(t, p, "Benchmark Results");
    let (btn_bg, btn_fg, btn_lbl) = if data.bench_running {
        (theme::SURFACE_BRIGHT, theme::TEXT_DIM, "Running\u{2026}")
    } else {
        (theme::GREEN, theme::SIDEBAR_BG, "Run All Tests")
    };
    if !data.bench_running {
        let ha = data.transitions.value("hover-run-bench").unwrap_or(0.0);
        action_btn(t, hdr, btn_lbl, btn_bg, btn_fg, "run-bench", ha);
    } else {
        let mut btn_s = s("btn");
        btn_s.background = btn_bg;
        let btn = t.add_box(hdr, btn_s);
        t.add_text(btn, btn_lbl, s("body").color(btn_fg));
    }

    if data.bench_running {
        let (done, total) = data.bench_progress;
        let pct = done as f64 / total.max(1) as f64;
        t.add_bar(p, pct, theme::ACCENT, s("bar-medium"));
        if let Some(cat) = &data.current_cat {
            t.add_text(p, &format!("Running: {cat} ({done}/{total})"), s("label"));
        }
    }

    if data.bench_results.is_empty() && !data.bench_running {
        t.add_text(
            p,
            "No results yet \u{2014} click \u{2018}Run All Tests\u{2019} to begin.",
            sm(&["font-14", "text-dim"]),
        );
        return;
    }

    let mut i = 0;
    while i < data.bench_results.len() {
        let row = t.add_box(p, s("row-gap-12"));
        for j in 0..2 {
            let idx = i + j;
            if idx >= data.bench_results.len() {
                break;
            }
            let report = &data.bench_results[idx];
            let c = t.add_box(row, s("result-card"));
            t.add_text(c, &report.category, sm(&["font-14", "blue"]));
            let max_ops = report
                .results
                .iter()
                .map(|r| r.throughput_ops_sec)
                .fold(0.0_f64, f64::max)
                .max(1.0);
            for (bi, r) in report.results.iter().take(6).enumerate() {
                let name = if r.name.len() > 20 {
                    format!("{}\u{2026}", &r.name[..18])
                } else {
                    r.name.clone()
                };
                let pct = r.throughput_ops_sec / max_ops;
                let bar_c = theme::BAR_COLORS[bi % 4];
                let entry = t.add_box(c, s("gap-2"));
                let lr = t.add_box(entry, s("row"));
                t.add_text(lr, &name, sm(&["font-9", "text-dim"]));
                t.add_box(lr, s("grow"));
                t.add_text(
                    lr,
                    &format_ops(r.throughput_ops_sec),
                    sm(&["font-9", "text"]),
                );
                t.add_bar(entry, pct, bar_c, s("bar-small"));
            }
        }
        i += 2;
    }
}

fn build_sim(t: &mut Tree, p: NodeId, data: &mut AppData) {
    let hdr = section_hdr(t, p, "Live Showdown");
    let (btn_bg, btn_lbl) = if data.sim_running {
        (theme::RED, "Stop Showdown")
    } else {
        (theme::GREEN, "Start Showdown")
    };
    let ha = data.transitions.value("hover-toggle-sim").unwrap_or(0.0);
    action_btn(t, hdr, btn_lbl, btn_bg, theme::SIDEBAR_BG, "toggle-sim", ha);

    t.add_text(
        p,
        "Real-time Sigmoid(200K): any-compute vs rayon vs stdlib",
        s("subtitle"),
    );

    let peak = data.ac_ops.max(data.rayon_ops).max(data.std_ops).max(1.0);
    for &(label, ops, color) in &[
        ("any-compute (vectorized kernel)", data.ac_ops, theme::GREEN),
        ("rayon (parallel iterator)", data.rayon_ops, theme::BLUE),
        (
            "stdlib (single-thread iterator)",
            data.std_ops,
            theme::YELLOW,
        ),
    ] {
        let lane = t.add_box(p, s("gap-4"));
        let top = t.add_box(lane, s("row"));
        t.add_text(top, label, s("font-13").color(color));
        t.add_box(top, s("grow"));
        t.add_text(top, &format_ops(ops), sm(&["font-13", "text"]));
        let frac = if peak > 0.0 { ops / peak } else { 0.0 };
        t.add_bar(lane, frac, color, s("bar-large"));
        if ops != data.std_ops && data.std_ops > 0.0 {
            t.add_text(
                lane,
                &format!("{:.1}x vs stdlib", ops / data.std_ops),
                s("small-dim"),
            );
        }
    }
}

// ── Event dispatch ──────────────────────────────────────────────────────────
const TAB_DUR: Duration = Duration::from_millis(180);
const HOVER_DUR: Duration = Duration::from_millis(120);

/// Start a named f64 transition (from → to) with EaseOut and register it.
fn ease_transition(mgr: &mut TransitionManager, key: String, from: f64, to: f64, dur: Duration) {
    let mut t = Transition::new(from, to, dur).with_easing(Easing::EaseOut);
    t.start();
    mgr.add(key, t);
}

/// Animate tab switch: fade old out, fade new in, reset scroll.
fn switch_tab(d: &mut AppData, new: usize) {
    let old = d.tab;
    if old == new {
        return;
    }
    d.tab = new;
    d.scroll_y = 0.0;
    d.scroll_target = 0.0;
    ease_transition(&mut d.transitions, format!("tab-{old}"), 1.0, 0.0, TAB_DUR);
    ease_transition(&mut d.transitions, format!("tab-{new}"), 0.0, 1.0, TAB_DUR);
}

fn handle_click(state: &SharedState, tag: &str) {
    match tag {
        t @ ("tab-0" | "tab-1" | "tab-2") => {
            let new: usize = t[4..].parse().unwrap();
            state.write(|d| switch_tab(d, new));
        }
        "run-bench" => spawn_benchmarks(state.clone()),
        "toggle-sim" => spawn_simulation(state.clone()),
        _ => {}
    }
}

fn handle_hover(state: &SharedState, tag: Option<String>) {
    state.write(|d| {
        if let Some(delta) = d.hover.update(tag) {
            if let Some(left) = &delta.left {
                let key = format!("hover-{left}");
                let cur = d.transitions.value(&key).unwrap_or(0.0);
                ease_transition(&mut d.transitions, key, cur, 0.0, HOVER_DUR);
            }
            if let Some(entered) = &delta.entered {
                let key = format!("hover-{entered}");
                let cur = d.transitions.value(&key).unwrap_or(0.0);
                ease_transition(&mut d.transitions, key, cur, 1.0, HOVER_DUR);
            }
        }
    });
}

fn handle_keyboard(state: &SharedState, key: &str, _modifiers: Modifiers) {
    match key {
        "Tab" | "ArrowDown" => {
            state.write(|d| {
                let next = (d.tab + 1) % TAB_LABELS.len();
                switch_tab(d, next);
            });
        }
        "ArrowUp" => {
            state.write(|d| {
                let next = if d.tab == 0 {
                    TAB_LABELS.len() - 1
                } else {
                    d.tab - 1
                };
                switch_tab(d, next);
            });
        }
        "Enter" | " " => {
            let focused = state.read(|d| d.focus.focused.clone());
            if let Some(tag) = focused {
                handle_click(state, &tag);
            }
        }
        "Escape" => {
            state.write(|d| {
                d.sim_running = false;
                d.focus.focus(None);
            });
        }
        _ => {}
    }
}

// ── winit → InputEvent conversion ────────────────────────────────────────────
fn winit_button(b: winit::event::MouseButton) -> Button {
    match b {
        winit::event::MouseButton::Left => Button::Primary,
        winit::event::MouseButton::Right => Button::Secondary,
        winit::event::MouseButton::Middle => Button::Middle,
        _ => Button::Primary,
    }
}

fn winit_modifiers(m: &winit::event::Modifiers) -> Modifiers {
    let s = m.state();
    Modifiers {
        shift: s.shift_key(),
        ctrl: s.control_key(),
        alt: s.alt_key(),
        meta: s.super_key(),
    }
}

fn winit_key_to_string(key: &Key) -> String {
    match key {
        Key::Named(NamedKey::Tab) => "Tab".into(),
        Key::Named(NamedKey::Enter) => "Enter".into(),
        Key::Named(NamedKey::Escape) => "Escape".into(),
        Key::Named(NamedKey::Space) => " ".into(),
        Key::Named(NamedKey::ArrowUp) => "ArrowUp".into(),
        Key::Named(NamedKey::ArrowDown) => "ArrowDown".into(),
        Key::Named(NamedKey::ArrowLeft) => "ArrowLeft".into(),
        Key::Named(NamedKey::ArrowRight) => "ArrowRight".into(),
        Key::Named(NamedKey::Home) => "Home".into(),
        Key::Named(NamedKey::End) => "End".into(),
        Key::Named(NamedKey::PageUp) => "PageUp".into(),
        Key::Named(NamedKey::PageDown) => "PageDown".into(),
        Key::Character(c) => c.to_string(),
        _ => format!("{key:?}"),
    }
}

// ── Entry point ─────────────────────────────────────────────────────────────
pub fn main() {
    env_logger::init();

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let window = Arc::new(
        WindowBuilder::new()
            .with_title("any-compute \u{2014} Benchmark Dashboard")
            .with_inner_size(winit::dpi::LogicalSize::new(1400.0, 900.0))
            .build(&event_loop)
            .unwrap(),
    );

    let mut gpu = Gpu::init(window.clone());
    let state = SharedState::new();
    spawn_hw_detect(state.clone());

    let mut fps_timer = Instant::now();
    let mut fps_count = 0u32;
    let mut cursor_pos = Point::ZERO;
    let mut last_tree: Option<Tree> = None;
    let mut modifiers = winit::event::Modifiers::default();

    let _ = event_loop.run(move |event, elwt| match event {
        Event::WindowEvent {
            event: wevent,
            window_id,
        } if window_id == window.id() => match wevent {
            // ── Window lifecycle ─────────────────────────────
            WindowEvent::CloseRequested => {
                state.write(|d| d.sim_running = false);
                elwt.exit();
            }
            WindowEvent::Resized(s) => gpu.resize(s.width, s.height),

            // ── Modifier tracking ────────────────────────────
            WindowEvent::ModifiersChanged(m) => {
                modifiers = m;
            }

            // ── Pointer events ───────────────────────────────
            WindowEvent::MouseInput {
                state: elem_state,
                button,
                ..
            } => {
                let btn = winit_button(button);
                let event = match elem_state {
                    winit::event::ElementState::Pressed => {
                        // Track active (pressed) state.
                        if let Some(tree) = &last_tree {
                            let tag = tree.tag_at(cursor_pos);
                            state.write(|d| {
                                d.pointer_down = true;
                                d.active_tag = tag.clone();
                            });
                            // Set focus to clicked element.
                            state.write(|d| {
                                d.focus.focus(tag);
                            });
                        }
                        InputEvent::PointerDown {
                            pos: cursor_pos,
                            button: btn,
                        }
                    }
                    winit::event::ElementState::Released => {
                        // Dispatch click on release (matches web behavior).
                        let was_active = state.read(|d| d.active_tag.clone());
                        if let Some(tree) = &last_tree {
                            let release_tag = tree.tag_at(cursor_pos);
                            // Only fire click if released on the same tag as pressed.
                            if let (Some(pressed), Some(released)) = (&was_active, &release_tag) {
                                if pressed == released {
                                    handle_click(&state, pressed);
                                }
                            }
                        }
                        state.write(|d| {
                            d.pointer_down = false;
                            d.active_tag = None;
                        });
                        InputEvent::PointerUp {
                            pos: cursor_pos,
                            button: btn,
                        }
                    }
                };
                // Full dispatch through tree.
                if let Some(tree) = &last_tree {
                    let _result = tree.dispatch(event);
                }
            }

            WindowEvent::CursorMoved { position, .. } => {
                cursor_pos = Point::new(position.x, position.y);
                // Hover tracking: find the tag under cursor.
                if let Some(tree) = &last_tree {
                    let tag = tree.tag_at(cursor_pos);
                    handle_hover(&state, tag);
                    // Full dispatch.
                    let _result = tree.dispatch(InputEvent::PointerMove { pos: cursor_pos });
                }
            }

            WindowEvent::CursorLeft { .. } => {
                // Cursor left window → clear hover.
                handle_hover(&state, None);
            }

            // ── Scroll ───────────────────────────────────────
            WindowEvent::MouseWheel { delta, .. } => {
                let dy = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => y as f64 * 40.0,
                    winit::event::MouseScrollDelta::PixelDelta(p) => p.y,
                };
                state.write(|d| d.scroll_target = (d.scroll_target - dy).max(0.0));
                // Dispatch scroll event through tree.
                if let Some(tree) = &last_tree {
                    let _result = tree.dispatch(InputEvent::Scroll {
                        delta: Point::new(0.0, dy),
                    });
                }
            }

            // ── Keyboard events ──────────────────────────────
            WindowEvent::KeyboardInput {
                event:
                    winit::event::KeyEvent {
                        logical_key,
                        state: winit::event::ElementState::Pressed,
                        ..
                    },
                ..
            } => {
                let key_str = winit_key_to_string(&logical_key);
                let mods = winit_modifiers(&modifiers);
                handle_keyboard(&state, &key_str, mods);
            }

            // ── Focus / Blur ─────────────────────────────────
            WindowEvent::Focused(focused) => {
                if !focused {
                    // Window lost focus → clear hover.
                    handle_hover(&state, None);
                }
            }

            // ── Redraw ──────────────────────────────────────
            WindowEvent::RedrawRequested => {
                let sz = window.inner_size();
                if sz.width > 0 && sz.height > 0 {
                    let w = sz.width as f64;
                    let h = sz.height as f64;
                    let mut tree = build_tree(&state, w, h);
                    tree.layout(Size::new(w, h));
                    let mut list = RenderList::default();
                    tree.paint(&mut list);
                    gpu.paint(&list);
                    fps_count += 1;
                    if fps_timer.elapsed().as_secs() >= 1 {
                        window.set_title(&format!(
                            "any-compute \u{2014} {} FPS | {} nodes | {} primitives",
                            fps_count,
                            tree.arena.len(),
                            list.len(),
                        ));
                        fps_count = 0;
                        fps_timer = Instant::now();
                    }
                    last_tree = Some(tree);
                }
                window.request_redraw();
            }
            _ => {}
        },
        _ => {}
    });
}
