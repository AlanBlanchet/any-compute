//! Headless test harness — parse CSS/HTML → Tree → layout → scenario replay → GPU capture.
//!
//! Combines all pieces needed for integration testing without a visible window:
//! programmatic events, automatic restyle, and pixel capture.
//!
//! ```ignore
//! let mut h = TestHarness::from_css_html(css, html, (800, 600));
//! h.hover((100.0, 50.0));
//! let capture = h.capture();
//! assert!(capture.pixel(100, 50) != Color::TRANSPARENT);
//! ```

use any_compute_core::interaction::{Button, InputEvent};
use any_compute_core::layout::{Point, Size};
use any_compute_core::render::{Color, RenderList};
use crate::css::StyleSheet;
use crate::parse::parse_with_css;
use crate::style::Style;
use crate::tree::Tree;

use crate::PALETTE_CSS;
use crate::scenario::{Scenario, StepResult, replay};

// ── Capture ─────────────────────────────────────────────────────────────────

/// RGBA pixel buffer captured from headless GPU rendering.
pub struct Capture {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

impl Capture {
    /// Read a pixel at (x, y). Returns transparent black if out-of-bounds.
    pub fn pixel(&self, x: u32, y: u32) -> Color {
        if x >= self.width || y >= self.height {
            return Color::TRANSPARENT;
        }
        let idx = ((y * self.width + x) * 4) as usize;
        Color::rgba(
            self.rgba[idx],
            self.rgba[idx + 1],
            self.rgba[idx + 2],
            self.rgba[idx + 3],
        )
    }

    /// Check if a rectangular region is entirely one color (within tolerance).
    pub fn region_uniform(&self, x: u32, y: u32, w: u32, h: u32, expected: Color, tol: u8) -> bool {
        for dy in 0..h {
            for dx in 0..w {
                let c = self.pixel(x + dx, y + dy);
                if !colors_close(c, expected, tol) {
                    return false;
                }
            }
        }
        true
    }

    /// Count pixels matching a color (within tolerance) in a rectangular region.
    pub fn count_color(&self, x: u32, y: u32, w: u32, h: u32, target: Color, tol: u8) -> u32 {
        let mut count = 0;
        for dy in 0..h {
            for dx in 0..w {
                if colors_close(self.pixel(x + dx, y + dy), target, tol) {
                    count += 1;
                }
            }
        }
        count
    }

    /// Save to PNG file for visual inspection.
    #[cfg(feature = "gpu")]
    pub fn save_png(&self, path: &std::path::Path) {
        let file = std::fs::File::create(path).expect("create PNG file");
        let mut encoder = png::Encoder::new(file, self.width, self.height);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().expect("PNG header");
        writer.write_image_data(&self.rgba).expect("PNG data");
    }

    /// Compute pixel-diff count against another capture of the same size.
    pub fn diff_count(&self, other: &Capture, tol: u8) -> u32 {
        assert_eq!(self.width, other.width);
        assert_eq!(self.height, other.height);
        self.rgba
            .chunks_exact(4)
            .zip(other.rgba.chunks_exact(4))
            .filter(|(a, b)| {
                let ca = Color::rgba(a[0], a[1], a[2], a[3]);
                let cb = Color::rgba(b[0], b[1], b[2], b[3]);
                !colors_close(ca, cb, tol)
            })
            .count() as u32
    }
}

fn colors_close(a: Color, b: Color, tol: u8) -> bool {
    let dr = (a.r as i16 - b.r as i16).unsigned_abs() as u8;
    let dg = (a.g as i16 - b.g as i16).unsigned_abs() as u8;
    let db = (a.b as i16 - b.b as i16).unsigned_abs() as u8;
    let da = (a.a as i16 - b.a as i16).unsigned_abs() as u8;
    dr <= tol && dg <= tol && db <= tol && da <= tol
}

// ── TestHarness ─────────────────────────────────────────────────────────────

/// Headless test harness: tree + layout + scenario replay + GPU capture.
pub struct TestHarness {
    pub tree: Tree,
    pub viewport: Size,
    #[cfg(feature = "gpu")]
    gpu: crate::gpu::Gpu,
}

impl TestHarness {
    /// Create from raw CSS + HTML strings. Palette CSS is prepended automatically.
    #[cfg(feature = "gpu")]
    pub fn from_css_html(css: &str, html: &str, size: (u32, u32)) -> Self {
        let full_css = format!("{}\n{}", PALETTE_CSS, css);
        let sheet = StyleSheet::parse(&full_css);
        let mut tree = parse_with_css(html, &sheet);
        let viewport = Size::new(size.0 as f64, size.1 as f64);
        tree.layout(viewport);

        let gpu = crate::gpu::Gpu::init_headless(size.0, size.1);

        Self {
            tree,
            viewport,
            gpu,
        }
    }

    /// Create from a pre-built tree (no CSS parsing needed).
    #[cfg(feature = "gpu")]
    pub fn from_tree(mut tree: Tree, size: (u32, u32)) -> Self {
        let viewport = Size::new(size.0 as f64, size.1 as f64);
        tree.layout(viewport);
        let gpu = crate::gpu::Gpu::init_headless(size.0, size.1);
        Self {
            tree,
            viewport,
            gpu,
        }
    }

    /// Re-layout the tree (e.g. after style changes).
    pub fn layout(&mut self) {
        self.tree.layout(self.viewport);
    }

    // ── Events ──────────────────────────────────────────────────────────

    /// Move cursor to position — triggers hover/unhover + restyle.
    pub fn hover(&mut self, pos: impl Into<Point>) {
        self.tree
            .dispatch(InputEvent::PointerMove { pos: pos.into() });
        self.layout();
    }

    /// Click at position — pointer-down + pointer-up + restyle.
    pub fn click(&mut self, pos: impl Into<Point>) {
        let p = pos.into();
        self.tree.dispatch(InputEvent::PointerDown {
            pos: p,
            button: Button::Primary,
        });
        self.tree.dispatch(InputEvent::PointerUp {
            pos: p,
            button: Button::Primary,
        });
        self.layout();
    }

    /// Pointer-down at position (hold — for :active testing).
    pub fn pointer_down(&mut self, pos: impl Into<Point>) {
        self.tree.dispatch(InputEvent::PointerDown {
            pos: pos.into(),
            button: Button::Primary,
        });
        self.layout();
    }

    /// Pointer-up at position.
    pub fn pointer_up(&mut self, pos: impl Into<Point>) {
        self.tree.dispatch(InputEvent::PointerUp {
            pos: pos.into(),
            button: Button::Primary,
        });
        self.layout();
    }

    /// Scroll at position.
    pub fn scroll(&mut self, pos: impl Into<Point>, delta: impl Into<Point>) {
        self.tree.scroll(pos.into(), delta.into());
        self.layout();
    }

    /// Replay a full scenario.
    pub fn replay(&mut self, scenario: &Scenario) -> Vec<StepResult> {
        let results = replay(&mut self.tree, scenario);
        self.layout();
        results
    }

    // ── Capture ─────────────────────────────────────────────────────────

    /// Paint the current tree to the render list.
    pub fn render_list(&mut self) -> RenderList {
        let mut list = RenderList::default();
        self.tree.paint(&mut list);
        self.tree.post_paint();
        list
    }

    /// Capture the current tree as an RGBA pixel buffer.
    #[cfg(feature = "gpu")]
    pub fn capture(&mut self) -> Capture {
        let mut list = RenderList::default();
        self.tree.paint(&mut list);
        self.tree.post_paint();
        let (w, h, rgba) = self.gpu.capture(&list);
        Capture {
            width: w,
            height: h,
            rgba,
        }
    }

    /// Capture and save to PNG.
    #[cfg(feature = "gpu")]
    pub fn capture_png(&mut self, path: &std::path::Path) {
        let mut list = RenderList::default();
        self.tree.paint(&mut list);
        self.tree.post_paint();
        self.gpu.capture_png(&list, path);
    }

    // ── Query ───────────────────────────────────────────────────────────

    /// Get the style of a node at a position (for asserting style changes).
    pub fn style_at(&self, pos: impl Into<Point>) -> Option<Style> {
        let nid = self.tree.hit_test(pos.into())?;
        Some(self.tree.slot(nid).style.clone())
    }

    /// Get the tag at a position.
    pub fn tag_at(&self, pos: impl Into<Point>) -> Option<String> {
        self.tree.tag_at(pos.into())
    }

    /// Check if a node at position is in hover state.
    pub fn is_hovered(&self, pos: impl Into<Point>) -> bool {
        self.tree
            .hit_test(pos.into())
            .map(|nid| self.tree.slot(nid).hovered)
            .unwrap_or(false)
    }
}
