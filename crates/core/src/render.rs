//! Render primitives — declarative shapes and styles that any backend can paint.
//!
//! These are **descriptors**, not actual drawing calls.
//! A renderer (GPU, canvas, terminal) walks a `RenderList` and paints.
//!
//! All spatial data references [`layout::Rect`] and [`layout::Point`] —
//! this module never duplicates coordinate fields.

use crate::Lerp;
use crate::layout::{Point, Rect};

/// RGBA color, 8-bit per channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl Color {
    pub const fn rgba(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    pub const fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self::rgba(r, g, b, 255)
    }

    pub const WHITE: Self = Self::rgb(255, 255, 255);
    pub const BLACK: Self = Self::rgb(0, 0, 0);
    pub const TRANSPARENT: Self = Self::rgba(0, 0, 0, 0);
}

impl Lerp for Color {
    fn lerp(self, other: Self, t: f64) -> Self {
        let t = t.clamp(0.0, 1.0) as f32;
        Self {
            r: (self.r as f32 + (other.r as f32 - self.r as f32) * t) as u8,
            g: (self.g as f32 + (other.g as f32 - self.g as f32) * t) as u8,
            b: (self.b as f32 + (other.b as f32 - self.b as f32) * t) as u8,
            a: (self.a as f32 + (other.a as f32 - self.a as f32) * t) as u8,
        }
    }
}

/// A single draw command — references layout types for all spatial data.
#[derive(Debug, Clone)]
pub enum Primitive {
    /// Filled/stroked rectangle — references [`Rect`] for bounds.
    Rect {
        bounds: Rect,
        fill: Color,
        border: Option<Border>,
        corner_radius: f64,
    },
    /// Text at a position — references [`Point`] for anchor.
    Text {
        anchor: Point,
        content: String,
        font_size: f64,
        color: Color,
    },
    /// Line between two points — references [`Point`].
    Line {
        from: Point,
        to: Point,
        stroke: Color,
        width: f64,
    },
    /// Clip region — references [`Rect`].
    PushClip {
        bounds: Rect,
    },
    PopClip,
}

#[derive(Debug, Clone, Copy)]
pub struct Border {
    pub color: Color,
    pub width: f64,
}

/// Ordered list of draw commands — the output of layout/paint phase.
#[derive(Debug, Clone, Default)]
pub struct RenderList {
    pub primitives: Vec<Primitive>,
}

impl RenderList {
    pub fn push(&mut self, p: Primitive) {
        self.primitives.push(p);
    }

    pub fn clear(&mut self) {
        self.primitives.clear();
    }

    pub fn len(&self) -> usize {
        self.primitives.len()
    }

    pub fn is_empty(&self) -> bool {
        self.primitives.is_empty()
    }
}

/// Trait for render backends (GPU, canvas, terminal, etc.).
///
/// Each backend implements this once. Core produces `RenderList`, backend paints it.
pub trait RenderBackend: Send + Sync {
    /// Paint the entire render list to the target surface.
    fn paint(&mut self, list: &RenderList);

    /// Hint to the backend about viewport size (for buffer allocation, etc.).
    fn resize(&mut self, width: u32, height: u32);
}

// ═══════════════════════════════════════════════════════════════════════════
// ── PixelBuffer — CPU software rasterizer for testing + headless ────────
// ═══════════════════════════════════════════════════════════════════════════

/// CPU-side pixel buffer with SDF rounded-rect rasterizer.
///
/// Implements the same visual semantics as the GPU shader: SDF-based rounded
/// rectangles with per-pixel alpha compositing and border support.
///
/// ## Usage
///
/// ```
/// use any_compute_core::render::*;
/// use any_compute_core::layout::Rect;
///
/// let mut buf = PixelBuffer::new(100, 100, Color::BLACK);
/// let mut list = RenderList::default();
/// list.push(Primitive::Rect {
///     bounds: Rect::new(10.0, 10.0, 80.0, 80.0),
///     fill: Color::WHITE,
///     border: None,
///     corner_radius: 12.0,
/// });
/// buf.paint(&list);
/// // Center pixel is fill color.
/// assert_eq!(buf.pixel(50, 50), Color::WHITE);
/// // Top-left corner outside the radius is still the clear color.
/// assert_eq!(buf.pixel(10, 10), Color::BLACK);
/// ```
pub struct PixelBuffer {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<Color>,
    pub clear: Color,
}

impl PixelBuffer {
    pub fn new(width: u32, height: u32, clear: Color) -> Self {
        let pixels = vec![clear; (width * height) as usize];
        Self {
            width,
            height,
            pixels,
            clear,
        }
    }

    /// Read a single pixel (clamped to bounds).
    pub fn pixel(&self, x: u32, y: u32) -> Color {
        if x >= self.width || y >= self.height {
            return Color::TRANSPARENT;
        }
        self.pixels[(y * self.width + x) as usize]
    }

    /// Rasterize an entire `RenderList` with alpha compositing.
    pub fn paint(&mut self, list: &RenderList) {
        for p in &list.primitives {
            match p {
                Primitive::Rect {
                    bounds,
                    fill,
                    border,
                    corner_radius,
                } => {
                    self.rasterize_rect(*bounds, *fill, *border, *corner_radius);
                }
                // Text, Line, Clip — not rasterized by software backend (test rects only).
                _ => {}
            }
        }
    }

    fn rasterize_rect(&mut self, bounds: Rect, fill: Color, border: Option<Border>, radius: f64) {
        let x0 = (bounds.origin.x.floor() as i32).max(0) as u32;
        let y0 = (bounds.origin.y.floor() as i32).max(0) as u32;
        let x1 = ((bounds.origin.x + bounds.size.w).ceil() as u32).min(self.width);
        let y1 = ((bounds.origin.y + bounds.size.h).ceil() as u32).min(self.height);

        let hw = bounds.size.w * 0.5;
        let hh = bounds.size.h * 0.5;
        let cx = bounds.origin.x + hw;
        let cy = bounds.origin.y + hh;
        let r = radius.min(hw).min(hh);

        let (bw, bc) = border
            .map(|b| (b.width, b.color))
            .unwrap_or((0.0, Color::TRANSPARENT));

        for py in y0..y1 {
            for px in x0..x1 {
                let fx = px as f64 + 0.5 - cx;
                let fy = py as f64 + 0.5 - cy;

                let d = sdf_rounded_rect(fx, fy, hw, hh, r);
                if d > 0.5 {
                    continue;
                }
                let aa = (0.5 - d).clamp(0.0, 1.0);

                let src = if bw > 0.0 {
                    let ir = (r - bw).max(0.0);
                    let inner_d = sdf_rounded_rect(fx, fy, hw - bw, hh - bw, ir);
                    if inner_d > 0.0 { bc } else { fill }
                } else {
                    fill
                };

                let sa = (src.a as f64 / 255.0) * aa;
                if sa <= 0.0 {
                    continue;
                }

                let idx = (py * self.width + px) as usize;
                let dst = self.pixels[idx];
                self.pixels[idx] = alpha_over(dst, src, sa);
            }
        }
    }
}

impl RenderBackend for PixelBuffer {
    fn paint(&mut self, list: &RenderList) {
        PixelBuffer::paint(self, list);
    }

    fn resize(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
        self.pixels.resize((width * height) as usize, self.clear);
    }
}

/// SDF for a rounded rectangle centered at origin with half-extents (hw, hh) and radius r.
fn sdf_rounded_rect(px: f64, py: f64, hw: f64, hh: f64, r: f64) -> f64 {
    let qx = px.abs() - (hw - r);
    let qy = py.abs() - (hh - r);
    let outside = (qx.max(0.0).powi(2) + qy.max(0.0).powi(2)).sqrt();
    let inside = qx.max(qy).min(0.0);
    outside + inside - r
}

/// Alpha-over compositing: src (with pre-computed alpha) over dst.
fn alpha_over(dst: Color, src: Color, src_alpha: f64) -> Color {
    let inv = 1.0 - src_alpha;
    let da = dst.a as f64 / 255.0;
    let out_a = src_alpha + da * inv;
    if out_a <= 0.0 {
        return Color::TRANSPARENT;
    }
    let blend = |s: u8, d: u8| -> u8 {
        ((s as f64 * src_alpha + d as f64 * da * inv) / out_a).round() as u8
    };
    Color::rgba(
        blend(src.r, dst.r),
        blend(src.g, dst.g),
        blend(src.b, dst.b),
        (out_a * 255.0).round() as u8,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Lerp;

    #[test]
    fn color_lerp_black_to_white() {
        let c = Color::BLACK.lerp(Color::WHITE, 0.5);
        // Midpoint should be ~127-128
        assert!(c.r >= 126 && c.r <= 128);
        assert_eq!(c.a, 255); // both have a=255
    }

    #[test]
    fn color_lerp_endpoints() {
        assert_eq!(Color::BLACK.lerp(Color::WHITE, 0.0), Color::BLACK);
        assert_eq!(Color::BLACK.lerp(Color::WHITE, 1.0), Color::WHITE);
    }

    #[test]
    fn color_constants() {
        assert_eq!(Color::TRANSPARENT, Color::rgba(0, 0, 0, 0));
        assert_eq!(Color::WHITE, Color::rgb(255, 255, 255));
    }

    #[test]
    fn render_list_push_clear() {
        let mut list = RenderList::default();
        assert!(list.is_empty());
        list.push(Primitive::Rect {
            bounds: Rect::new(0.0, 0.0, 100.0, 50.0),
            fill: Color::BLACK,
            border: None,
            corner_radius: 0.0,
        });
        assert_eq!(list.len(), 1);
        list.clear();
        assert!(list.is_empty());
    }

    #[test]
    fn render_list_mixed_primitives() {
        let mut list = RenderList::default();
        list.push(Primitive::PushClip { bounds: Rect::ZERO });
        list.push(Primitive::Text {
            anchor: Point::ZERO,
            content: "hello".into(),
            font_size: 14.0,
            color: Color::WHITE,
        });
        list.push(Primitive::Line {
            from: Point::ZERO,
            to: Point::new(100.0, 100.0),
            stroke: Color::WHITE,
            width: 1.0,
        });
        list.push(Primitive::PopClip);
        assert_eq!(list.len(), 4);
    }

    // ── PixelBuffer visual-correctness tests ────────────────────────────

    fn make_rect(x: f64, y: f64, w: f64, h: f64, fill: Color, radius: f64) -> Primitive {
        Primitive::Rect {
            bounds: Rect::new(x, y, w, h),
            fill,
            border: None,
            corner_radius: radius,
        }
    }

    fn make_bordered_rect(
        x: f64,
        y: f64,
        w: f64,
        h: f64,
        fill: Color,
        radius: f64,
        border_color: Color,
        border_width: f64,
    ) -> Primitive {
        Primitive::Rect {
            bounds: Rect::new(x, y, w, h),
            fill,
            border: Some(Border {
                color: border_color,
                width: border_width,
            }),
            corner_radius: radius,
        }
    }

    #[test]
    fn pixel_flat_rect_fills_center() {
        let mut buf = PixelBuffer::new(100, 100, Color::BLACK);
        let mut list = RenderList::default();
        list.push(make_rect(10.0, 10.0, 80.0, 80.0, Color::WHITE, 0.0));
        buf.paint(&list);
        // Center is filled.
        assert_eq!(buf.pixel(50, 50), Color::WHITE);
        // Outside is untouched.
        assert_eq!(buf.pixel(5, 5), Color::BLACK);
    }

    #[test]
    fn pixel_rounded_rect_clips_corners() {
        // border-radius should erase corners — the CSS visual guarantee.
        let mut buf = PixelBuffer::new(100, 100, Color::BLACK);
        let mut list = RenderList::default();
        list.push(make_rect(0.0, 0.0, 100.0, 100.0, Color::WHITE, 20.0));
        buf.paint(&list);
        // Center is filled.
        assert_eq!(buf.pixel(50, 50), Color::WHITE);
        // Top-left corner (0,0) is OUTSIDE the radius → background color.
        assert_eq!(buf.pixel(0, 0), Color::BLACK);
        // Top-right corner likewise.
        assert_eq!(buf.pixel(99, 0), Color::BLACK);
        // Bottom-left.
        assert_eq!(buf.pixel(0, 99), Color::BLACK);
        // Bottom-right.
        assert_eq!(buf.pixel(99, 99), Color::BLACK);
        // Just inside the radius curve at (20, 0) — should be filled.
        assert_eq!(buf.pixel(50, 0), Color::WHITE);
    }

    #[test]
    fn pixel_border_renders_edge_ring() {
        let mut buf = PixelBuffer::new(100, 100, Color::BLACK);
        let red = Color::rgb(255, 0, 0);
        let blue = Color::rgb(0, 0, 255);
        let mut list = RenderList::default();
        list.push(make_bordered_rect(
            0.0, 0.0, 100.0, 100.0, blue, 0.0, red, 4.0,
        ));
        buf.paint(&list);
        // Center is fill color.
        assert_eq!(buf.pixel(50, 50), blue);
        // Top edge, middle column — within 4px border → border color.
        assert_eq!(buf.pixel(50, 1), red);
        // Left edge, middle row.
        assert_eq!(buf.pixel(1, 50), red);
    }

    #[test]
    fn pixel_border_with_radius() {
        let mut buf = PixelBuffer::new(100, 100, Color::BLACK);
        let fill = Color::rgb(100, 200, 100);
        let edge = Color::rgb(255, 255, 0);
        let mut list = RenderList::default();
        list.push(make_bordered_rect(
            0.0, 0.0, 100.0, 100.0, fill, 16.0, edge, 3.0,
        ));
        buf.paint(&list);
        // (0,0) is outside the rounded corner → background.
        assert_eq!(buf.pixel(0, 0), Color::BLACK);
        // Center is fill.
        assert_eq!(buf.pixel(50, 50), fill);
        // Top-center within border band → border color.
        assert_eq!(buf.pixel(50, 1), edge);
    }

    #[test]
    fn pixel_alpha_compositing() {
        let mut buf = PixelBuffer::new(100, 100, Color::BLACK);
        let semi = Color::rgba(255, 0, 0, 128);
        let mut list = RenderList::default();
        list.push(make_rect(0.0, 0.0, 100.0, 100.0, semi, 0.0));
        buf.paint(&list);
        let p = buf.pixel(50, 50);
        // Red channel should be roughly half (128 composited over black).
        assert!(p.r > 100 && p.r < 140, "got r={}", p.r);
        assert!(p.g < 10, "got g={}", p.g);
    }

    #[test]
    fn pixel_overlapping_rects_back_to_front() {
        // Later primitives draw on top — normal painter's algorithm.
        let mut buf = PixelBuffer::new(100, 100, Color::BLACK);
        let red = Color::rgb(255, 0, 0);
        let green = Color::rgb(0, 255, 0);
        let mut list = RenderList::default();
        list.push(make_rect(0.0, 0.0, 100.0, 100.0, red, 0.0));
        list.push(make_rect(25.0, 25.0, 50.0, 50.0, green, 0.0));
        buf.paint(&list);
        // Overlap region is green (on top).
        assert_eq!(buf.pixel(50, 50), green);
        // Outside overlap is red.
        assert_eq!(buf.pixel(10, 10), red);
    }

    #[test]
    fn pixel_radius_clamped_to_half_size() {
        // radius > half-extent → clamped to circle/stadium. Should not panic or glitch.
        let mut buf = PixelBuffer::new(60, 30, Color::BLACK);
        let mut list = RenderList::default();
        list.push(make_rect(0.0, 0.0, 60.0, 30.0, Color::WHITE, 999.0));
        buf.paint(&list);
        // Center filled.
        assert_eq!(buf.pixel(30, 15), Color::WHITE);
        // Corner clipped (stadium shape).
        assert_eq!(buf.pixel(0, 0), Color::BLACK);
    }

    #[test]
    fn pixel_sdf_symmetry() {
        // All four corners should behave identically.
        let mut buf = PixelBuffer::new(80, 80, Color::BLACK);
        let mut list = RenderList::default();
        list.push(make_rect(0.0, 0.0, 80.0, 80.0, Color::WHITE, 10.0));
        buf.paint(&list);
        // Check symmetry: pixel (2,2) should equal pixel (77,2), (2,77), (77,77).
        let tl = buf.pixel(2, 2);
        assert_eq!(tl, buf.pixel(77, 2));
        assert_eq!(tl, buf.pixel(2, 77));
        assert_eq!(tl, buf.pixel(77, 77));
    }
}
