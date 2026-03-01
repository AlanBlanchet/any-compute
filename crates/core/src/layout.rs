//! Layout primitives — constraint-based positioning that is renderer-agnostic.
//!
//! These types are the **single source of truth** for all spatial concepts.
//! Render primitives, events, hit-testing, and animations all reference these —
//! never duplicate x/y/w/h fields elsewhere.

use crate::Lerp;

/// 2D point — the atomic spatial unit. Used by everything that needs a position.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

impl Point {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0 };

    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    pub fn distance_to(self, other: Self) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }
}

impl Lerp for Point {
    fn lerp(self, other: Self, t: f64) -> Self {
        Self {
            x: self.x.lerp(other.x, t),
            y: self.y.lerp(other.y, t),
        }
    }
}

impl std::ops::Add for Point {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}

impl std::ops::Sub for Point {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}

/// 2D size — width and height without position.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Size {
    pub w: f64,
    pub h: f64,
}

impl Size {
    pub const ZERO: Self = Self { w: 0.0, h: 0.0 };

    pub const fn new(w: f64, h: f64) -> Self {
        Self { w, h }
    }

    pub fn area(self) -> f64 {
        self.w * self.h
    }
}

impl Lerp for Size {
    fn lerp(self, other: Self, t: f64) -> Self {
        Self {
            w: self.w.lerp(other.w, t),
            h: self.h.lerp(other.h, t),
        }
    }
}

/// Axis-aligned rectangle — composed from [`Point`] + [`Size`], never duplicates their fields.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Rect {
    pub origin: Point,
    pub size: Size,
}

impl Rect {
    pub const ZERO: Self = Self {
        origin: Point::ZERO,
        size: Size::ZERO,
    };

    pub fn new(x: f64, y: f64, w: f64, h: f64) -> Self {
        Self {
            origin: Point::new(x, y),
            size: Size::new(w, h),
        }
    }

    pub fn from_parts(origin: Point, size: Size) -> Self {
        Self { origin, size }
    }

    // Convenience accessors that delegate — no field duplication.
    pub fn x(&self) -> f64 {
        self.origin.x
    }
    pub fn y(&self) -> f64 {
        self.origin.y
    }
    pub fn w(&self) -> f64 {
        self.size.w
    }
    pub fn h(&self) -> f64 {
        self.size.h
    }

    pub fn right(&self) -> f64 {
        self.origin.x + self.size.w
    }
    pub fn bottom(&self) -> f64 {
        self.origin.y + self.size.h
    }

    pub fn center(&self) -> Point {
        Point::new(
            self.origin.x + self.size.w / 2.0,
            self.origin.y + self.size.h / 2.0,
        )
    }

    pub fn contains(&self, p: Point) -> bool {
        p.x >= self.origin.x && p.x <= self.right() && p.y >= self.origin.y && p.y <= self.bottom()
    }
}

impl Lerp for Rect {
    fn lerp(self, other: Self, t: f64) -> Self {
        Self {
            origin: self.origin.lerp(other.origin, t),
            size: self.size.lerp(other.size, t),
        }
    }
}

/// Size constraints flowing down during layout.
#[derive(Debug, Clone, Copy)]
pub struct Constraints {
    pub min: Size,
    pub max: Size,
}

impl Constraints {
    pub fn tight(size: Size) -> Self {
        Self {
            min: size,
            max: size,
        }
    }

    pub fn unbounded() -> Self {
        Self {
            min: Size::ZERO,
            max: Size::new(f64::INFINITY, f64::INFINITY),
        }
    }

    pub fn clamp(&self, size: Size) -> Size {
        Size {
            w: size.w.clamp(self.min.w, self.max.w),
            h: size.h.clamp(self.min.h, self.max.h),
        }
    }
}

/// Scroll state for virtualized containers.
#[derive(Debug, Clone, Copy, Default)]
pub struct ScrollState {
    pub offset: Point,
}

impl ScrollState {
    /// Which range of items (by index) is visible given a fixed item height?
    pub fn visible_range(
        &self,
        item_height: f64,
        viewport_height: f64,
        total_items: usize,
    ) -> std::ops::Range<usize> {
        if item_height <= 0.0 || total_items == 0 {
            return 0..0;
        }
        let first = (self.offset.y / item_height).floor().max(0.0) as usize;
        let visible_count = (viewport_height / item_height).ceil() as usize + 1;
        let last = (first + visible_count).min(total_items);
        first..last
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Lerp;

    #[test]
    fn point_lerp() {
        let a = Point::new(0.0, 0.0);
        let b = Point::new(100.0, 200.0);
        let mid = a.lerp(b, 0.5);
        assert!((mid.x - 50.0).abs() < 1e-10);
        assert!((mid.y - 100.0).abs() < 1e-10);
    }

    #[test]
    fn point_distance() {
        let a = Point::new(0.0, 0.0);
        let b = Point::new(3.0, 4.0);
        assert!((a.distance_to(b) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn size_lerp() {
        let s = Size::new(100.0, 50.0).lerp(Size::new(200.0, 100.0), 0.5);
        assert!((s.w - 150.0).abs() < 1e-10);
        assert!((s.h - 75.0).abs() < 1e-10);
    }

    #[test]
    fn size_area() {
        assert!((Size::new(10.0, 20.0).area() - 200.0).abs() < 1e-10);
    }

    #[test]
    fn rect_contains() {
        let r = Rect::new(10.0, 10.0, 100.0, 50.0);
        assert!(r.contains(Point::new(50.0, 30.0)));
        assert!(!r.contains(Point::new(5.0, 30.0)));
        assert!(!r.contains(Point::new(50.0, 70.0)));
    }

    #[test]
    fn rect_accessors() {
        let r = Rect::new(10.0, 20.0, 100.0, 50.0);
        assert_eq!(r.x(), 10.0);
        assert_eq!(r.y(), 20.0);
        assert_eq!(r.w(), 100.0);
        assert_eq!(r.h(), 50.0);
        assert_eq!(r.right(), 110.0);
        assert_eq!(r.bottom(), 70.0);
        let c = r.center();
        assert!((c.x - 60.0).abs() < 1e-10);
        assert!((c.y - 45.0).abs() < 1e-10);
    }

    #[test]
    fn rect_lerp() {
        let a = Rect::new(0.0, 0.0, 100.0, 50.0);
        let b = Rect::new(100.0, 100.0, 200.0, 100.0);
        let mid = a.lerp(b, 0.5);
        assert!((mid.x() - 50.0).abs() < 1e-10);
        assert!((mid.w() - 150.0).abs() < 1e-10);
    }

    #[test]
    fn rect_from_parts() {
        let r = Rect::from_parts(Point::new(5.0, 10.0), Size::new(20.0, 30.0));
        assert_eq!(r.x(), 5.0);
        assert_eq!(r.h(), 30.0);
    }

    #[test]
    fn constraints_clamp() {
        let c = Constraints {
            min: Size::new(10.0, 10.0),
            max: Size::new(200.0, 200.0),
        };
        assert_eq!(c.clamp(Size::new(5.0, 300.0)), Size::new(10.0, 200.0));
    }

    #[test]
    fn scroll_visible_range() {
        let s = ScrollState {
            offset: Point::new(0.0, 280.0),
        };
        let range = s.visible_range(28.0, 280.0, 1000);
        assert_eq!(range.start, 10);
        assert!(range.end <= 21);
    }

    #[test]
    fn scroll_visible_range_edge_cases() {
        let s = ScrollState {
            offset: Point::ZERO,
        };
        assert_eq!(s.visible_range(0.0, 100.0, 100), 0..0);
        assert_eq!(s.visible_range(10.0, 100.0, 0), 0..0);
    }
}
