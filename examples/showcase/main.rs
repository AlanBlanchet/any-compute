//! Unified showcase — every feature of any-compute in one tabbed window.
//!
//! Run: `cargo run -p showcase` or `make showcase`
//!
//! Tabs:
//!   0. DOM Website    — full HTML/CSS parsing, animations, hover, z-index, scroll
//!   1. 3D Scene       — mesh, camera, transform, ray intersection, normals
//!   2. Compute        — kernel benchmarks, OpQueue batch, OpCache hit/miss
//!   3. Live Metrics   — real-time throughput, FPS, memory, background workers

mod tabs;

use any_compute_core::Lerp;
use any_compute_core::animation::{Easing, Transition, TransitionManager};
use any_compute_core::compute::Device;
use any_compute_core::interaction::{Button, InputEvent};
use any_compute_core::kernel::{ReduceOp, UnaryOp, best_kernel};
use any_compute_core::layout::{Point, Size};
use any_compute_core::render::{Color, RenderList};
use any_compute_dom::PALETTE_CSS;
use any_compute_dom::css::StyleSheet;
use any_compute_dom::gpu::Gpu;
use any_compute_dom::parse::parse_with_css;
use any_compute_dom::style::*;
use any_compute_dom::theme;
use any_compute_dom::tree::*;
use any_compute_dom::winit;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use winit::event::{ElementState, Event, MouseButton, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{CursorIcon, WindowBuilder};

// ═══════════════════════════════════════════════════════════════════════════
// Constants
// ═══════════════════════════════════════════════════════════════════════════

const SHOWCASE_CSS: &str = include_str!("showcase.css");
const WEBSITE_HTML: &str = include_str!("website.html");
const WEBSITE_CSS: &str = include_str!("website.css");
const VIEWPORT: Size = Size::new(1400.0, 900.0);
const VERSION: &str = concat!("v", env!("CARGO_PKG_VERSION"));
const MAX_FRAME_DT: f64 = 0.032;

const TAB_LABELS: &[&str] = &["DOM Website", "3D Scene", "Compute", "Live Metrics"];

/// Parsed once at startup.
fn build_sheet() -> StyleSheet {
    let css = format!("{PALETTE_CSS}\n{SHOWCASE_CSS}\n{WEBSITE_CSS}");
    StyleSheet::parse(&css)
}

/// Shorthand: one class from sheet.
fn s(sheet: &StyleSheet, class: &str) -> Style {
    sheet.class(class)
}

/// Shorthand: merge multiple classes.
fn sm(sheet: &StyleSheet, classes: &[&str]) -> Style {
    sheet.classes(classes)
}

// ═══════════════════════════════════════════════════════════════════════════
// Shared state
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Clone)]
struct Shared {
    inner: Arc<Mutex<AppData>>,
}

struct AppData {
    tab: usize,
    scroll_y: f64,
    scroll_target: f64,
    transitions: TransitionManager,

    // FPS
    fps: u32,
    fps_count: u32,
    fps_timer: Instant,
    frame_times: Vec<f64>,

    // DOM tab: parsed website tree (built once, then interacted)
    dom_tree: Option<Tree>,
    dom_cursor: Point,
    dom_needs_measure: bool,

    // 3D tab
    scene_info: tabs::scene::SceneInfo,

    // Compute tab
    compute_results: Vec<(String, f64)>,
    compute_running: bool,

    // Live metrics
    live_ac_ops: f64,
    live_rayon_ops: f64,
    live_std_ops: f64,
    live_running: bool,

    // Hardware
    hw: Option<any_compute_bench::runner::HardwareReport>,
}

impl Shared {
    fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(AppData {
                tab: 0,
                scroll_y: 0.0,
                scroll_target: 0.0,
                transitions: {
                    let mut mgr = TransitionManager::default();
                    let mut t = Transition::new(0.0, 1.0, Duration::ZERO);
                    t.start();
                    mgr.add("tab-0", t);
                    mgr
                },
                fps: 0,
                fps_count: 0,
                fps_timer: Instant::now(),
                frame_times: Vec::with_capacity(120),
                dom_tree: None,
                dom_cursor: Point::ZERO,
                dom_needs_measure: true,
                scene_info: tabs::scene::SceneInfo::default(),
                compute_results: Vec::new(),
                compute_running: false,
                live_ac_ops: 0.0,
                live_rayon_ops: 0.0,
                live_std_ops: 0.0,
                live_running: false,
                hw: None,
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

// ═══════════════════════════════════════════════════════════════════════════
// Background workers
// ═══════════════════════════════════════════════════════════════════════════

fn spawn_hw_detect(state: Shared) {
    std::thread::spawn(move || {
        let hw = any_compute_bench::runner::detect_hardware();
        state.write(|d| d.hw = Some(hw));
    });
}

fn spawn_compute_bench(state: Shared) {
    if state.read(|d| d.compute_running) {
        return;
    }
    state.write(|d| {
        d.compute_running = true;
        d.compute_results.clear();
    });
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

fn spawn_live_sim(state: Shared) {
    if state.read(|d| d.live_running) {
        return;
    }
    state.write(|d| d.live_running = true);
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

use rayon::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Tree building
// ═══════════════════════════════════════════════════════════════════════════

fn build_tree(state: &Shared, sheet: &StyleSheet, w: f64, h: f64) -> Tree {
    let mut data = state.inner.lock().unwrap();

    // Smooth scroll
    let scroll_speed = 0.18;
    data.scroll_y += (data.scroll_target - data.scroll_y) * scroll_speed;
    if (data.scroll_target - data.scroll_y).abs() < 0.5 {
        data.scroll_y = data.scroll_target;
    }

    // FPS tracking
    data.fps_count += 1;
    if data.fps_timer.elapsed().as_secs() >= 1 {
        data.fps = data.fps_count;
        data.fps_count = 0;
        data.fps_timer = Instant::now();
    }

    let mut t = Tree::new(sm(sheet, &["bg", "row"]).w(w).h(h));
    let root = t.root;

    // ── Sidebar ──
    let sb = t.add_box(root, s(sheet, "sidebar"));
    let brand = t.add_box(sb, s(sheet, "brand"));
    t.add_box(brand, s(sheet, "brand-icon"));
    let bt = t.add_box(brand, s(sheet, "brand-text"));
    t.add_text(bt, "any-compute", s(sheet, "heading-text"));
    t.add_text(bt, VERSION, s(sheet, "small-dim"));
    t.add_box(sb, s(sheet, "spacer-12"));

    for (i, &label) in TAB_LABELS.iter().enumerate() {
        let tag_name = format!("tab-{i}");
        let alpha = data
            .transitions
            .value(&tag_name)
            .unwrap_or(if data.tab == i { 1.0 } else { 0.0 });
        let hover_alpha = data
            .transitions
            .value(&format!("hover-{tag_name}"))
            .unwrap_or(0.0);
        let base_bg = Color::TRANSPARENT.lerp(theme::ACCENT, alpha);
        let bg = base_bg.lerp(theme::SURFACE_BRIGHT, hover_alpha * (1.0 - alpha));
        let fg = theme::TEXT_DIM.lerp(theme::SIDEBAR_BG, alpha);
        let mut tab_s = s(sheet, "tab-btn");
        tab_s.background = bg;
        tab_s.color = fg;
        let btn = t.add_box(sb, tab_s);
        t.add_text(btn, label, s(sheet, "font-13").color(fg));
        t.tag(btn, &tag_name);
    }

    // Spacer + FPS overlay at bottom of sidebar
    let _spacer = t.add_box(sb, s(sheet, "grow"));
    let fps_box = t.add_box(sb, s(sheet, "fps-overlay"));
    t.add_text(
        fps_box,
        &format!("{}fps", data.fps),
        s(sheet, "font-9").color(theme::ACCENT),
    );
    let node_count = t.arena.len();
    t.add_text(
        fps_box,
        &format!("{node_count}n"),
        s(sheet, "font-9").color(Color::from((180, 180, 180))),
    );

    // ── Main area ──
    let main_col = t.add_box(root, s(sheet, "grow"));
    let hdr = t.add_box(main_col, s(sheet, "header"));
    t.add_text(hdr, TAB_LABELS[data.tab], sm(sheet, &["font-18", "text"]));

    // HW info in header
    if let Some(hw) = &data.hw {
        let _hw_box = t.add_box(hdr, s(sheet, "grow"));
        let right = t.add_box(hdr, sm(sheet, &["row-gap-8"]));
        t.add_text(
            right,
            &hw.cpu.brand,
            s(sheet, "font-9").color(theme::TEXT_DIM),
        );
        t.add_text(
            right,
            &format!("{}c", hw.cpu.physical_cores),
            sm(sheet, &["font-9", "blue"]),
        );
    }

    let content = t.add_box(main_col, s(sheet, "content"));
    t.slot_mut(content).scroll.y = data.scroll_y;

    match data.tab {
        0 => tabs::dom::build(sheet, &mut t, content, &data),
        1 => tabs::scene::build(sheet, &mut t, content, &data),
        2 => tabs::compute::build(sheet, &mut t, content, &data),
        3 => tabs::live::build(sheet, &mut t, content, &data),
        _ => {}
    }

    t
}

// ═══════════════════════════════════════════════════════════════════════════
// Event handling
// ═══════════════════════════════════════════════════════════════════════════

fn handle_click(state: &Shared, tag: &str) {
    // Tab switching
    if let Some(idx_str) = tag.strip_prefix("tab-") {
        if let Ok(idx) = idx_str.parse::<usize>() {
            state.write(|d| {
                if d.tab != idx {
                    let old = d.tab;
                    d.tab = idx;
                    d.scroll_y = 0.0;
                    d.scroll_target = 0.0;
                    // Transition old tab out, new tab in
                    d.transitions.add(
                        &format!("tab-{old}"),
                        Transition::new(1.0, 0.0, Duration::from_millis(200))
                            .with_easing(Easing::EaseOut),
                    );
                    d.transitions.add(
                        &format!("tab-{idx}"),
                        Transition::new(0.0, 1.0, Duration::from_millis(200))
                            .with_easing(Easing::EaseOut),
                    );
                }
            });
        }
    }

    // Compute tab: run benchmarks
    if tag == "run-bench" {
        spawn_compute_bench(state.clone());
    }

    // Live tab: toggle simulation
    if tag == "toggle-sim" {
        let running = state.read(|d| d.live_running);
        if running {
            state.write(|d| d.live_running = false);
        } else {
            spawn_live_sim(state.clone());
        }
    }
}

fn handle_hover(state: &Shared, tag: Option<String>) {
    state.write(|d| {
        // Clear old hover transitions
        for i in 0..TAB_LABELS.len() {
            let key = format!("hover-tab-{i}");
            let is_hovered = tag.as_deref() == Some(&format!("tab-{i}"));
            let current = d.transitions.value(&key).unwrap_or(0.0);
            let target = if is_hovered { 1.0 } else { 0.0 };
            if (current - target).abs() > 0.01 {
                d.transitions.add(
                    &key,
                    Transition::new(current, target, Duration::from_millis(150))
                        .with_easing(Easing::EaseOut),
                );
            }
        }
    });
}

// ═══════════════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════════════

fn main() {
    env_logger::init();

    let sheet = build_sheet();
    let state = Shared::new();

    // Build DOM tree for tab 0 (parsed website)
    {
        let full_css = format!("{PALETTE_CSS}\n{WEBSITE_CSS}\n{SHOWCASE_CSS}");
        let ws = StyleSheet::parse(&full_css);
        let tree = parse_with_css(WEBSITE_HTML, &ws);
        state.write(|d| d.dom_tree = Some(tree));
    }

    spawn_hw_detect(state.clone());
    spawn_compute_bench(state.clone());
    spawn_live_sim(state.clone());

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let window = Arc::new(
        WindowBuilder::new()
            .with_title("any-compute — Showcase")
            .with_inner_size(winit::dpi::LogicalSize::new(VIEWPORT.w(), VIEWPORT.h()))
            .build(&event_loop)
            .unwrap(),
    );

    let mut gpu = Gpu::init(window.clone());

    // Measure text in the DOM tree with real font metrics
    state.write(|d| {
        if let Some(tree) = &mut d.dom_tree {
            tree.measure_text_nodes(|text, font_size| gpu.measure_text(text, font_size));
            tree.layout(Size::new(VIEWPORT.w() - 220.0, VIEWPORT.h() - 56.0));
            tree.start_animations();
            d.dom_needs_measure = false;
        }
    });

    let mut cursor_pos = Point::ZERO;
    let mut last_frame = Instant::now();
    let mut last_tree: Option<Tree> = None;
    let mut list = RenderList::default();
    let mut current_cursor = CursorIcon::Default;
    let frame_budget = Duration::from_micros(16_667); // ~60 fps

    let _ = event_loop.run(move |event, elwt| match event {
        Event::AboutToWait => {
            if last_frame.elapsed() >= frame_budget {
                window.request_redraw();
            } else {
                // Yield CPU until next frame
                let remaining = frame_budget.saturating_sub(last_frame.elapsed());
                if !remaining.is_zero() {
                    std::thread::sleep(remaining);
                }
                window.request_redraw();
            }
        }
        Event::WindowEvent {
            event: wevent,
            window_id,
        } if window_id == window.id() => match wevent {
            WindowEvent::CloseRequested => {
                state.write(|d| d.live_running = false);
                elwt.exit();
            }
            WindowEvent::Resized(sz) => gpu.resize(sz.width, sz.height),

            WindowEvent::CursorMoved { position, .. } => {
                cursor_pos = Point::new(position.x, position.y);
                // Hover tracking on sidebar tags
                if let Some(tree) = &last_tree {
                    let tag = tree.tag_at(cursor_pos);
                    handle_hover(&state, tag);
                }
                // Dispatch to DOM tree if on DOM tab
                let on_dom = state.read(|d| d.tab == 0);
                if on_dom {
                    state.write(|d| {
                        d.dom_cursor = cursor_pos;
                        if let Some(tree) = &mut d.dom_tree {
                            let result = tree.dispatch(InputEvent::PointerMove { pos: cursor_pos });
                            let icon = match result.cursor.as_str() {
                                "pointer" => CursorIcon::Pointer,
                                "text" => CursorIcon::Text,
                                "not-allowed" => CursorIcon::NotAllowed,
                                _ => CursorIcon::Default,
                            };
                            if icon != current_cursor {
                                window.set_cursor_icon(icon);
                                current_cursor = icon;
                            }
                        }
                    });
                }
            }

            WindowEvent::MouseInput {
                state: elem_state,
                button,
                ..
            } => {
                let btn = match button {
                    MouseButton::Left => Button::Primary,
                    MouseButton::Right => Button::Secondary,
                    MouseButton::Middle => Button::Middle,
                    _ => Button::Primary,
                };

                if elem_state == ElementState::Released {
                    // Check sidebar tags first
                    if let Some(tree) = &last_tree {
                        if let Some(tag) = tree.tag_at(cursor_pos) {
                            handle_click(&state, &tag);
                        }
                    }
                    // DOM tree dispatch on tab 0
                    let on_dom = state.read(|d| d.tab == 0);
                    if on_dom {
                        state.write(|d| {
                            if let Some(tree) = &mut d.dom_tree {
                                let result = tree.dispatch(InputEvent::PointerUp {
                                    pos: cursor_pos,
                                    button: btn,
                                });
                                if let Some(tag) = result.target_tag() {
                                    println!("  click → {tag}");
                                }
                            }
                        });
                    }
                } else {
                    let on_dom = state.read(|d| d.tab == 0);
                    if on_dom {
                        state.write(|d| {
                            if let Some(tree) = &mut d.dom_tree {
                                tree.dispatch(InputEvent::PointerDown {
                                    pos: cursor_pos,
                                    button: btn,
                                });
                            }
                        });
                    }
                }
            }

            WindowEvent::MouseWheel { delta, .. } => {
                let dy = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => y as f64 * 40.0,
                    winit::event::MouseScrollDelta::PixelDelta(p) => p.y,
                };
                // DOM tab: dispatch scroll to DOM tree
                let on_dom = state.read(|d| d.tab == 0);
                if on_dom {
                    state.write(|d| {
                        if let Some(tree) = &mut d.dom_tree {
                            tree.dispatch(InputEvent::Scroll {
                                delta: Point::new(0.0, dy),
                            });
                        }
                    });
                }
                // Also scroll the shell content
                state.write(|d| d.scroll_target = (d.scroll_target - dy).max(0.0));
            }

            WindowEvent::KeyboardInput {
                event:
                    winit::event::KeyEvent {
                        logical_key,
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => {
                // Tab key or number keys to switch tabs
                match &logical_key {
                    Key::Character(c) => {
                        if let Ok(n) = c.as_str().parse::<usize>() {
                            if n > 0 && n <= TAB_LABELS.len() {
                                handle_click(&state, &format!("tab-{}", n - 1));
                            }
                        }
                    }
                    Key::Named(NamedKey::Tab) => {
                        let next = state.read(|d| (d.tab + 1) % TAB_LABELS.len());
                        handle_click(&state, &format!("tab-{next}"));
                    }
                    _ => {}
                }
            }

            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = (now - last_frame).as_secs_f64().min(MAX_FRAME_DT);
                last_frame = now;

                // Tick transitions
                state.write(|d| {
                    d.transitions.start_all();
                    d.transitions.gc();
                    // Record frame time
                    d.frame_times.push(dt * 1000.0);
                    if d.frame_times.len() > 120 {
                        d.frame_times.remove(0);
                    }
                    // Tick DOM tree animations
                    if let Some(tree) = &mut d.dom_tree {
                        tree.tick(dt);
                    }
                });

                let sz = window.inner_size();
                if sz.width > 0 && sz.height > 0 {
                    let w = sz.width as f64;
                    let h = sz.height as f64;
                    let mut tree = build_tree(&state, &sheet, w, h);
                    tree.layout(Size::new(w, h));
                    list.clear();
                    tree.paint(&mut list);
                    tree.post_paint();
                    gpu.paint(&list);
                    last_tree = Some(tree);

                    // Update window title with FPS
                    let fps = state.read(|d| d.fps);
                    let nodes = list.len();
                    window.set_title(&format!(
                        "any-compute — Showcase | {fps} FPS | {nodes} prims"
                    ));
                }
            }
            _ => {}
        },
        _ => {}
    });
}
