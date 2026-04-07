//! Shared application state and tree building — used by both the interactive
//! showcase and the headless snapshot renderer.

use any_compute_core::Lerp;
use any_compute_core::animation::{Transition, TransitionManager};
use any_compute_core::layout::{Point, Rect, Size};
use any_compute_core::render::{Color, RenderList};
use any_compute_dom::PALETTE_CSS;
use any_compute_dom::css::StyleSheet;
use any_compute_dom::gpu::Gpu;
use any_compute_dom::style::{Align, Cursor, Justify, Style};
use any_compute_dom::theme;
use any_compute_dom::tree::*;
use std::path::Path;
use std::time::{Duration, Instant};

use crate::tabs::helpers::{s, sm};

// ── Constants ───────────────────────────────────────────────────────────

pub const SHOWCASE_CSS: &str = include_str!("showcase.css");
pub const WEBSITE_HTML: &str = include_str!("website.html");
pub const WEBSITE_CSS: &str = include_str!("website.css");
pub const SIDEBAR_W: f64 = 44.0;
pub const HEADER_H: f64 = 36.0;
pub const TAB_LABELS: &[&str] = &["Browser", "3D Scene", "Compute", "AI"];
pub const TAB_ICONS: &[&str] = &["\u{1F310}", "\u{1F3AE}", "\u{26A1}", "\u{1F9E0}"];

pub static COMBINED_CSS: std::sync::LazyLock<String> =
    std::sync::LazyLock::new(|| format!("{PALETTE_CSS}\n{SHOWCASE_CSS}\n{WEBSITE_CSS}"));

pub fn build_sheet() -> StyleSheet {
    StyleSheet::parse(&COMBINED_CSS)
}

// ── AppData ─────────────────────────────────────────────────────────────

pub struct AppData {
    pub tab: usize,
    pub subtabs: [usize; 4],
    pub scroll_y: f64,
    pub scroll_target: f64,
    pub transitions: TransitionManager,
    pub graph_mode: bool,
    pub fps: u32,
    pub fps_count: u32,
    pub fps_timer: Instant,
    pub frame_times: Vec<f64>,
    pub browser: crate::tabs::browser::BrowserState,
    pub scene_info: crate::tabs::scene::SceneInfo,
    pub compute_results: Vec<(String, f64)>,
    pub compute_running: bool,
    pub live_ac_ops: f64,
    pub live_rayon_ops: f64,
    pub live_std_ops: f64,
    pub live_running: bool,
    pub time: f64,
    pub dt: f64,
    pub hw: Option<any_compute_bench::runner::HardwareReport>,
    pub ai: crate::tabs::graph::AiState,
    pub keys_held: std::collections::HashSet<String>,
}

impl AppData {
    pub fn new() -> Self {
        Self {
            tab: 0,
            subtabs: [0; 4],
            scroll_y: 0.0,
            scroll_target: 0.0,
            transitions: {
                let mut mgr = TransitionManager::default();
                let mut t = Transition::new(0.0, 1.0, Duration::ZERO);
                t.start();
                mgr.add("tab-0", t);
                mgr
            },
            graph_mode: false,
            fps: 0,
            fps_count: 0,
            fps_timer: Instant::now(),
            frame_times: Vec::with_capacity(120),
            browser: crate::tabs::browser::BrowserState::default(),
            scene_info: crate::tabs::scene::SceneInfo::default(),
            compute_results: Vec::new(),
            compute_running: false,
            live_ac_ops: 0.0,
            live_rayon_ops: 0.0,
            live_std_ops: 0.0,
            live_running: false,
            time: 0.0,
            dt: 1.0 / 60.0,
            hw: None,
            ai: crate::tabs::graph::AiState::default(),
            keys_held: std::collections::HashSet::new(),
        }
    }

    // ── Associated helpers (on the instance conceptually) ───────────────

    /// Tag for tab button N.
    #[inline]
    pub fn tab_tag(i: usize) -> String {
        format!("tab-{i}")
    }

    /// Tag for hover transition on tab N.
    #[inline]
    pub fn hover_tag(i: usize) -> String {
        format!("hover-tab-{i}")
    }

    /// Is the cursor in the content area (past sidebar + header)?
    #[inline]
    pub fn in_content_area(pos: Point) -> bool {
        pos.x > SIDEBAR_W && pos.y > HEADER_H
    }

    /// Screen-space cursor → DOM content coordinates, relative to viewport rect.
    #[inline]
    pub fn to_dom_pos(pos: Point, vp: Rect) -> Point {
        Point::new(pos.x - vp.x(), pos.y - vp.y())
    }

    // ── Tree building ───────────────────────────────────────────────────

    pub fn build_tree(&mut self, sheet: &StyleSheet, w: f64, h: f64) -> Tree {
        // Smooth scroll — dt-corrected exponential interpolation
        let k = 1.0 - 0.82_f64.powf(self.dt * 60.0);
        self.scroll_y += (self.scroll_target - self.scroll_y) * k;
        if (self.scroll_target - self.scroll_y).abs() < 0.5 {
            self.scroll_y = self.scroll_target;
        }

        // FPS tracking
        self.fps_count += 1;
        if self.fps_timer.elapsed().as_secs() >= 1 {
            self.fps = self.fps_count;
            self.fps_count = 0;
            self.fps_timer = Instant::now();
        }

        // Advance time
        self.time += self.dt;

        let mut t = Tree::new(sm(sheet, &["bg", "row"]).w(w).h(h));
        let root = t.root;

        // ── Sidebar ──
        let sb = t.add_box(root, s(sheet, "sidebar"));
        t.add_box(sb, s(sheet, "brand-icon"));
        let mut spacer_s = Style::default().h(4.0);
        spacer_s.flex_shrink = 0.0;
        t.add_box(sb, spacer_s);

        for (i, &icon) in TAB_ICONS.iter().enumerate() {
            let tag_name = Self::tab_tag(i);
            let alpha = self
                .transitions
                .value(&tag_name)
                .unwrap_or(if self.tab == i { 1.0 } else { 0.0 });
            let hover_alpha = self.transitions.value(&Self::hover_tag(i)).unwrap_or(0.0);
            let base_bg = Color::TRANSPARENT.lerp(theme::ACCENT, alpha);
            let bg = base_bg.lerp(theme::SURFACE_BRIGHT, hover_alpha * (1.0 - alpha));
            let fg = theme::TEXT_DIM.lerp(theme::SIDEBAR_BG, alpha);
            let mut tab_s = s(sheet, "tab-btn");
            tab_s.background = bg;
            tab_s.color = fg;
            let btn = t.add_box(sb, tab_s);
            t.add_text(btn, icon, s(sheet, "font-14").color(fg));
            t.tag(btn, &tag_name);
        }

        // ── Main area ──
        let main_col = t.add_box(root, s(sheet, "grow"));
        let hdr = t.add_box(main_col, s(sheet, "header"));

        // Tab label (left)
        t.add_text(hdr, TAB_LABELS[self.tab], s(sheet, "heading"));

        // Graph mode toggle
        let gm_bg = if self.graph_mode {
            theme::ACCENT
        } else {
            theme::SURFACE_BRIGHT
        };
        let gm_fg = if self.graph_mode {
            theme::SIDEBAR_BG
        } else {
            theme::TEXT_DIM
        };
        let gm_btn = t.add_box(
            hdr,
            Style::default()
                .w(28.0)
                .h(22.0)
                .radius(4.0)
                .bg(gm_bg)
                .align(Align::Center)
                .justify(Justify::Center)
                .cursor(Cursor::Pointer),
        );
        t.add_text(gm_btn, "\u{2B12}", s(sheet, "font-11").color(gm_fg));
        t.tag(gm_btn, "toggle-graph-mode");

        let _spacer_h = t.add_box(hdr, s(sheet, "grow"));

        // FPS + node count (right side of header)
        let fps_box = t.add_box(hdr, sm(sheet, &["row-gap-8"]));
        t.add_text(
            fps_box,
            &format!("{}fps", self.fps),
            s(sheet, "font-9").color(theme::ACCENT),
        );
        let node_count = t.arena.len();
        t.add_text(
            fps_box,
            &format!("{node_count}n"),
            s(sheet, "font-9").color(Color::from((120, 120, 130))),
        );

        // HW info
        if let Some(hw) = &self.hw {
            t.add_text(fps_box, &hw.cpu.brand, s(sheet, "detail"));
            t.add_text(
                fps_box,
                &format!("{}c", hw.cpu.physical_cores),
                sm(sheet, &["font-9", "blue"]),
            );
        }

        let content = t.add_box(main_col, s(sheet, "content"));
        t.slot_mut(content).scroll.y = self.scroll_y;

        // Graph mode: render any tab's content as a node graph
        if self.graph_mode {
            use any_compute_core::visual::Graphable;
            let source = self.tab + 100;
            if self.ai.needs_rebuild(source) {
                let visual = match self.tab {
                    0 => self.browser.page.as_ref().map(|p| p.tree().to_graph()),
                    1 => Some(self.scene_info.scene.to_graph()),
                    2 => {
                        use any_compute_core::layout::V;
                        use any_compute_core::visual::{VNode, VisualGraph};
                        let mut g = VisualGraph::new("Compute");
                        let i = g.add(VNode::new(V([0.0, 0.0]), "Input", "input"));
                        let s = g.add(VNode::new(V([0.0, 0.0]), "Sqrt", "activation"));
                        let r = g.add(VNode::new(V([0.0, 0.0]), "Reduce", "reduce"));
                        let o = g.add(VNode::new(V([0.0, 0.0]), "Output", "output"));
                        g.edge(i, s);
                        g.edge(s, r);
                        g.edge(r, o);
                        g.auto_layout();
                        Some(g)
                    }
                    3 => self.ai.model.map(|m| m.to_graph()),
                    _ => None,
                };
                if let Some(v) = visual {
                    self.ai.set_graph(v, source);
                }
            }
            let viewport = t.add_box(content, s(sheet, "scene-panel").grow(1.0));
            t.tag(viewport, crate::tabs::graph::VIEWPORT_TAG);
            self.ai.render_graph();
        } else {
            // Normal tab content
            match self.tab {
                0 => crate::tabs::browser::build(
                    sheet,
                    &mut t,
                    content,
                    &self.browser,
                    self.subtabs[0],
                ),
                1 => {
                    let time = self.time;
                    crate::tabs::scene::build(sheet, &mut t, content, self, time);
                }
                2 => crate::tabs::compute::build(sheet, &mut t, content, self, self.subtabs[2]),
                3 => {
                    self.ai.build(sheet, &mut t, content);
                }
                _ => {}
            }
        }

        t
    }

    // ── Headless rendering (used by snapshot binary) ──────────────────

    #[allow(dead_code)]
    pub fn render_frame(&mut self, sheet: &StyleSheet, gpu: &mut Gpu, w: f64, h: f64, path: &Path) {
        let bvp = crate::tabs::browser::VIEWPORT_TAG;
        let mut tree = self.build_tree(sheet, w, h);
        tree.measure_text_nodes(|text, font_size| gpu.measure_text(text, font_size));
        tree.layout(Size::new(w, h));

        let mut list = RenderList::default();
        tree.paint(&mut list);
        tree.post_paint();

        // Browser tab: composite the page overlay
        if self.tab == 0 {
            if let Some(vp) = tree.tagged_rect(bvp) {
                if self.browser.needs_measure {
                    if let Some(page) = &mut self.browser.page {
                        page.measure_text_nodes(|text, font_size| {
                            gpu.measure_text(text, font_size)
                        });
                        page.start_animations();
                    }
                    self.browser.needs_measure = false;
                }
                if let Some(page) = &mut self.browser.page {
                    page.layout(Size::new(vp.w(), vp.h()));
                    let mut dl = RenderList::default();
                    page.paint(&mut dl);
                    page.post_paint();
                    list.composite(vp, &dl);
                }
            }
        }

        gpu.capture_png(&list, path);
        eprintln!("  → saved {}", path.display());
    }
}
