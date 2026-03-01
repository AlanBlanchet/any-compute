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
}
