//! Render primitives — declarative shapes and styles that any backend can paint.

mod color;
mod pixel_buffer;
mod primitive;
mod viewport;

pub use color::*;
pub use pixel_buffer::*;
pub use primitive::*;
pub use viewport::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Lerp;
    use crate::layout::{Point, Rect};

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
            border: Some(Border::uniform(border_color, border_width)),
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

    // ── From trait conversions ──────────────────────────────────────────

    #[test]
    fn color_from_tuple() {
        assert_eq!(Color::from((255u8, 0u8, 0u8)), Color::rgb(255, 0, 0));
        assert_eq!(
            Color::from((0u8, 255u8, 0u8, 128u8)),
            Color::rgba(0, 255, 0, 128)
        );
    }

    #[test]
    fn color_from_array() {
        assert_eq!(Color::from([255, 0, 0]), Color::rgb(255, 0, 0));
        assert_eq!(Color::from([0, 255, 0, 128]), Color::rgba(0, 255, 0, 128));
    }

    #[test]
    fn color_from_u32_hex() {
        assert_eq!(Color::from(0xFF0000u32), Color::rgb(255, 0, 0));
        assert_eq!(Color::from(0x89B4FA80u32), Color::rgba(137, 180, 250, 128));
    }

    #[test]
    fn color_roundtrip_u32() {
        let c = Color::rgba(137, 180, 250, 255);
        let hex: u32 = c.into();
        let back: Color = hex.into();
        assert_eq!(back, c);
    }

    #[test]
    fn color_roundtrip_array() {
        let c = Color::rgba(10, 20, 30, 40);
        let arr: [u8; 4] = c.into();
        assert_eq!(arr, [10, 20, 30, 40]);
        let back: Color = arr.into();
        assert_eq!(back, c);
    }

    #[test]
    fn renderable_colored_rect() {
        let obj = ColoredRect {
            rect: Rect::new(10.0, 20.0, 30.0, 40.0),
            color: Color::rgb(255, 0, 0),
            corner_radius: 0.0,
        };
        let mut list = RenderList::default();
        list.draw(&obj, &());
        assert_eq!(list.len(), 1);
        match &list.primitives[0] {
            Primitive::Rect { bounds, fill, .. } => {
                assert_eq!(bounds.x(), 10.0);
                assert_eq!(fill.r, 255);
            }
            _ => panic!("expected Rect primitive"),
        }
    }
}
