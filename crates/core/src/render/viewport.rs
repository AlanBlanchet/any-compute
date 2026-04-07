use crate::layout::{Point, Rect};
use super::RenderList;

// ═══════════════════════════════════════════════════════════════════════════
// ── Viewport — unified coordinate-space projection ──────────────────────
// ═══════════════════════════════════════════════════════════════════════════

/// A view into any coordinate space — unifies 2D DOM layout, 3D camera
/// projection, and any future N-dimensional mapping to screen pixels.
///
/// Both `Camera` (3D→screen) and DOM layout (box model→pixels) share this
/// trait: each maps source-space points to `(screen_x, screen_y, depth)`.
///
/// `P` is the source-space point type (`V<3>` for 3D, `Point` for 2D).
pub trait Viewport<P> {
    /// Map a source-space point to `(screen_x, screen_y, depth)`.
    /// Returns `None` when the point is clipped (behind camera, outside bounds).
    fn project(&self, point: P) -> Option<(f64, f64, f64)>;

    /// Screen-space bounds of this viewport (origin + size in pixels).
    fn screen_bounds(&self) -> Rect;
}

/// A 2D viewport — maps point coordinates directly to screen pixels with
/// an offset and scale.  Used by DOM layout (the content area is a viewport).
#[derive(Debug, Clone, Copy)]
pub struct Viewport2D {
    pub origin: Point,
    pub size: crate::layout::Size,
    pub scale: f64,
}

impl Viewport<Point> for Viewport2D {
    fn project(&self, point: Point) -> Option<(f64, f64, f64)> {
        let sx = self.origin.x + point.x * self.scale;
        let sy = self.origin.y + point.y * self.scale;
        // Cull if outside bounds
        if sx < self.origin.x
            || sy < self.origin.y
            || sx > self.origin.x + self.size.w()
            || sy > self.origin.y + self.size.h()
        {
            return None;
        }
        Some((sx, sy, 0.0))
    }

    fn screen_bounds(&self) -> Rect {
        Rect::new(self.origin.x, self.origin.y, self.size.w(), self.size.h())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ── Renderable — any object that can emit its own draw primitives ────────
// ═══════════════════════════════════════════════════════════════════════════

/// Trait for any object that can describe how to render itself.
///
/// Implementors push their own `Primitive`s into a `RenderList`, which
/// decouples scene-graph objects, UI widgets, and data visualizations
/// from the actual rendering backend (GPU, CPU software rasterizer, etc.).
///
/// ## Generic over viewport
///
/// The viewport parameter `V` lets the same object render into 2D or 3D
/// by receiving the appropriate projection mapping.
pub trait Renderable<V = ()> {
    /// Emit draw primitives into `list`, using `viewport` for projection.
    fn render(&self, list: &mut RenderList, viewport: &V);
}

/// Simplified version for objects that don't need a viewport.
impl<T: Renderable<()>> Renderable<()> for &T {
    fn render(&self, list: &mut RenderList, viewport: &()) {
        (*self).render(list, viewport);
    }
}

/// Extension on RenderList: render any `Renderable` by delegation.
impl RenderList {
    /// Render any `Renderable` into this list with the given viewport.
    pub fn draw<V>(&mut self, object: &impl Renderable<V>, viewport: &V) {
        object.render(self, viewport);
    }
}

// ═══════════════════════════════════════════════════════════════════════════

