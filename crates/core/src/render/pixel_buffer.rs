use crate::layout::Rect;
use super::{Border, Color, Primitive, Renderable, RenderList};

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
#[derive(Clone)]
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
                _ => {}
            }
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
        self.pixels.resize((width * height) as usize, self.clear);
    }

    fn rasterize_rect(&mut self, bounds: Rect, fill: Color, border: Option<Border>, radius: f64) {
        let x0 = (bounds.origin.x.floor() as i32).max(0) as u32;
        let y0 = (bounds.origin.y.floor() as i32).max(0) as u32;
        let x1 = ((bounds.origin.x + bounds.size.w()).ceil() as u32).min(self.width);
        let y1 = ((bounds.origin.y + bounds.size.h()).ceil() as u32).min(self.height);

        let hw = bounds.size.w() * 0.5;
        let hh = bounds.size.h() * 0.5;
        let cx = bounds.origin.x + hw;
        let cy = bounds.origin.y + hh;
        let r = radius.min(hw).min(hh);

        let (bt, br, bb, bl, bc) = border
            .map(|b| (b.top, b.right, b.bottom, b.left, b.color))
            .unwrap_or((0.0, 0.0, 0.0, 0.0, Color::TRANSPARENT));
        let has_border = bt > 0.0 || br > 0.0 || bb > 0.0 || bl > 0.0;

        for py in y0..y1 {
            for px in x0..x1 {
                let fx = px as f64 + 0.5 - cx;
                let fy = py as f64 + 0.5 - cy;

                let d = sdf_rounded_rect(fx, fy, hw, hh, r);
                if d > 0.5 {
                    continue;
                }
                let aa = (0.5 - d).clamp(0.0, 1.0);

                let src = if has_border {
                    let from_top = fy + hh;
                    let from_bottom = hh - fy;
                    let from_left = fx + hw;
                    let from_right = hw - fx;
                    if (bt > 0.0 && from_top < bt)
                        || (bb > 0.0 && from_bottom < bb)
                        || (bl > 0.0 && from_left < bl)
                        || (br > 0.0 && from_right < br)
                    {
                        bc
                    } else {
                        fill
                    }
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

    /// Per-pixel diff against another buffer.
    ///
    /// Returns the count of pixels where any channel differs by more than
    /// `tolerance` (0 = exact match).  Panics if dimensions differ.
    pub fn diff(&self, other: &Self, tolerance: u8) -> u32 {
        assert_eq!(
            (self.width, self.height),
            (other.width, other.height),
            "PixelBuffer::diff requires same dimensions"
        );
        let tol = tolerance as i16;
        self.pixels
            .iter()
            .zip(other.pixels.iter())
            .filter(|(a, b)| {
                (a.r as i16 - b.r as i16).abs() > tol
                    || (a.g as i16 - b.g as i16).abs() > tol
                    || (a.b as i16 - b.b as i16).abs() > tol
                    || (a.a as i16 - b.a as i16).abs() > tol
            })
            .count() as u32
    }

    /// Fraction of differing pixels (0.0 = identical, 1.0 = every pixel different).
    pub fn diff_ratio(&self, other: &Self, tolerance: u8) -> f64 {
        let total = (self.width * self.height) as f64;
        if total == 0.0 {
            return 0.0;
        }
        self.diff(other, tolerance) as f64 / total
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

// ═══════════════════════════════════════════════════════════════════════════
// ── Built-in Renderable implementations ─────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

/// A colored Rect (for rendering as a simple filled rectangle).
pub struct ColoredRect {
    pub rect: Rect,
    pub color: Color,
    pub corner_radius: f64,
}

impl Renderable<()> for ColoredRect {
    fn render(&self, list: &mut RenderList, _: &()) {
        list.push_rect(
            self.rect.x(),
            self.rect.y(),
            self.rect.w(),
            self.rect.h(),
            self.color,
        );
    }
}

