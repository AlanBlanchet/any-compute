//! Flexbox layout primitives — generic axis-based child distribution.
//!
//! Defines the **abstract layout vocabulary** used by any flexbox-like solver:
//! axis direction, alignment, justification, wrap, sizing constraints.
//!
//! HTML/CSS is one consumer (DOM's `Style` implements [`FlexStyle`]).
//! A JavaScript engine or native widget toolkit can also implement
//! [`FlexStyle`] using these same primitives.
//!
//! ## Enums vs CSS strings
//!
//! The enums here use **generic names** (`Axis::Row`, `CrossAlign::Center`)
//! that happen to match CSS because the flexbox spec IS the right vocabulary.
//! DOM adds `from_css()` / `to_css()` methods via its own macro; core stays
//! framework-agnostic.

use crate::Lerp;

// ═══════════════════════════════════════════════════════════════════════════
// ── Layout enum macro ───────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

/// Generate layout enums with Default + fallback + from_name/from_css/to_name/to_css.
///
/// Syntax: `EnumName [Default, Fallback] { "str" => Variant, ... }`
/// - `from_name()` → `Option<Self>` (generic, returns None for unknown)
/// - `from_css()`  → `Self` (CSS-compatible, returns fallback for unknown)
/// - `to_name()` / `to_css()` → `&'static str` (same output)
macro_rules! layout_enums {
    ($( $(#[doc = $doc:literal])* $name:ident [$def:ident, $fb:ident] {
        $( $($str:literal)|+ => $variant:ident ),+ $(,)?
    })*) => {$(
        $(#[doc = $doc])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub enum $name { $( $variant, )+ }

        impl Default for $name {
            fn default() -> Self { Self::$def }
        }

        impl $name {
            /// Parse from a canonical string name.
            pub fn from_name(val: &str) -> Option<Self> {
                match val { $( $($str)|+ => Some(Self::$variant), )+ _ => None, }
            }
            /// Parse a CSS value string. Unknown values use the type's fallback.
            pub fn from_css(val: &str) -> Self {
                match val { $( $($str)|+ => Self::$variant, )+ _ => Self::$fb, }
            }
            /// Canonical string name for this variant.
            pub fn to_name(self) -> &'static str {
                match self { $( Self::$variant => layout_enums!(@first $($str),+), )+ }
            }
            /// CSS string for this variant (alias for `to_name`).
            pub fn to_css(self) -> &'static str { self.to_name() }
        }
    )*};
    (@first $first:literal $(, $rest:literal)*) => { $first };
}

layout_enums! {
    /// Display mode (CSS `display`).
    Display [Flex, Flex] {
        "flex" => Flex, "block" => Block, "none" => None
    }

    /// Main axis direction for child layout (flexbox model).
    Direction [Row, Column] {
        "row" => Row, "column" => Column
    }

    /// Whether flex children can wrap to new lines.
    FlexWrap [NoWrap, NoWrap] {
        "nowrap" | "no-wrap" => NoWrap, "wrap" => Wrap, "wrap-reverse" => WrapReverse
    }

    /// Cross-axis alignment.
    Align [Stretch, Start] {
        "start" | "flex-start" => Start, "center" => Center,
        "end" | "flex-end" => End, "stretch" => Stretch, "baseline" => Baseline
    }

    /// Main-axis distribution of remaining space.
    Justify [Start, Start] {
        "start" | "flex-start" => Start, "center" => Center,
        "end" | "flex-end" => End, "space-between" => SpaceBetween,
        "space-around" => SpaceAround, "space-evenly" => SpaceEvenly
    }

    /// How this node participates in parent layout.
    Position [Relative, Relative] {
        "relative" => Relative, "absolute" => Absolute, "fixed" => Fixed
    }

    /// Overflow behavior.
    Overflow [Visible, Visible] {
        "visible" => Visible, "hidden" => Hidden,
        "scroll" => Scroll, "auto" => Auto
    }

    /// Box-sizing model (CSS `box-sizing`).
    BoxSizing [BorderBox, BorderBox] {
        "border-box" => BorderBox, "content-box" => ContentBox
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ── Dimension ───────────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

/// A single dimension that can be auto, fixed, percentage, or calc.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Dimension {
    Auto,
    Px(f64),
    Percent(f64),
    /// `calc(A% ± Bpx)` — resolved at layout time with parent size.
    Calc {
        percent: f64,
        px: f64,
    },
}

impl Default for Dimension {
    fn default() -> Self {
        Self::Auto
    }
}

impl Dimension {
    /// Resolve against a parent length. `Auto` returns `None`.
    pub fn resolve(self, parent: f64) -> Option<f64> {
        match self {
            Self::Auto => None,
            Self::Px(v) => Some(v),
            Self::Percent(p) => Some(parent * p / 100.0),
            Self::Calc { percent, px } => Some(parent * percent / 100.0 + px),
        }
    }

    /// Clamp a computed value between min/max dimensions resolved against parent.
    pub fn clamp(value: f64, min: Self, max: Self, parent: f64) -> f64 {
        let lo = min.resolve(parent).unwrap_or(0.0);
        let hi = max.resolve(parent).unwrap_or(f64::INFINITY);
        value.clamp(lo, hi)
    }
}

impl Lerp for Dimension {
    fn lerp(self, other: Self, t: f64) -> Self {
        match (self, other) {
            (Self::Px(a), Self::Px(b)) => Self::Px(a.lerp(b, t)),
            (Self::Percent(a), Self::Percent(b)) => Self::Percent(a.lerp(b, t)),
            (
                Self::Calc {
                    percent: p1,
                    px: x1,
                },
                Self::Calc {
                    percent: p2,
                    px: x2,
                },
            ) => Self::Calc {
                percent: p1.lerp(p2, t),
                px: x1.lerp(x2, t),
            },
            _ => {
                if t < 0.5 {
                    self
                } else {
                    other
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ── Edges ───────────────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

/// Edge insets (padding / margin / border-width) — four sides.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Edges {
    pub top: f64,
    pub right: f64,
    pub bottom: f64,
    pub left: f64,
}

impl Edges {
    pub const ZERO: Self = Self {
        top: 0.0,
        right: 0.0,
        bottom: 0.0,
        left: 0.0,
    };

    pub const fn all(v: f64) -> Self {
        Self {
            top: v,
            right: v,
            bottom: v,
            left: v,
        }
    }

    pub const fn xy(x: f64, y: f64) -> Self {
        Self {
            top: y,
            right: x,
            bottom: y,
            left: x,
        }
    }

    pub fn horizontal(&self) -> f64 {
        self.left + self.right
    }
    pub fn vertical(&self) -> f64 {
        self.top + self.bottom
    }

    /// True when any side is non-zero.
    pub fn any_nonzero(&self) -> bool {
        self.top > 0.0 || self.right > 0.0 || self.bottom > 0.0 || self.left > 0.0
    }
}

impl Lerp for Edges {
    fn lerp(self, other: Self, t: f64) -> Self {
        Self {
            top: self.top.lerp(other.top, t),
            right: self.right.lerp(other.right, t),
            bottom: self.bottom.lerp(other.bottom, t),
            left: self.left.lerp(other.left, t),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ── FlexStyle trait ─────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

/// Trait for anything that provides flexbox layout parameters.
///
/// The generic flex layout algorithm operates on this trait — it never
/// sees CSS strings, DOM nodes, or framework-specific types.
pub trait FlexStyle {
    fn display(&self) -> Display;
    fn box_sizing(&self) -> BoxSizing;
    fn direction(&self) -> Direction;
    fn flex_wrap(&self) -> FlexWrap;
    fn align(&self) -> Align;
    fn self_align(&self) -> Option<Align>;
    fn justify(&self) -> Justify;
    fn position(&self) -> Position;
    fn overflow(&self) -> Overflow;

    fn width(&self) -> Dimension;
    fn height(&self) -> Dimension;
    fn min_width(&self) -> Dimension;
    fn min_height(&self) -> Dimension;
    fn max_width(&self) -> Dimension;
    fn max_height(&self) -> Dimension;
    fn aspect_ratio(&self) -> Option<f64>;

    fn padding(&self) -> Edges;
    fn margin(&self) -> Edges;
    fn border_widths(&self) -> Edges;

    fn gap(&self) -> f64;
    fn row_gap(&self) -> Option<f64>;
    fn column_gap(&self) -> Option<f64>;

    fn flex_grow(&self) -> f64;
    fn flex_shrink(&self) -> f64;
    fn flex_basis(&self) -> Dimension;
    fn order(&self) -> i32;

    fn left(&self) -> Dimension;
    fn top(&self) -> Dimension;
    fn right(&self) -> Dimension;
    fn bottom(&self) -> Dimension;

    // ── Derived helpers (default impls) ─────────────────────────────

    /// True when this node is out of the normal flow.
    fn is_out_of_flow(&self) -> bool {
        matches!(self.position(), Position::Absolute | Position::Fixed)
    }

    /// True when display is none.
    fn is_hidden(&self) -> bool {
        self.display() == Display::None
    }

    /// True when direction is row.
    fn is_row(&self) -> bool {
        self.direction() == Direction::Row
    }

    /// Main-axis gap for this container.
    fn main_gap(&self) -> f64 {
        if self.is_row() {
            self.column_gap().unwrap_or(self.gap())
        } else {
            self.row_gap().unwrap_or(self.gap())
        }
    }

    /// Cross-axis gap for this container.
    fn cross_gap(&self) -> f64 {
        if self.is_row() {
            self.row_gap().unwrap_or(self.gap())
        } else {
            self.column_gap().unwrap_or(self.gap())
        }
    }

    /// Total inset on both sides: padding + border.
    fn inset_horizontal(&self) -> f64 {
        self.padding().horizontal() + self.border_widths().horizontal()
    }

    fn inset_vertical(&self) -> f64 {
        self.padding().vertical() + self.border_widths().vertical()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ── Tests ───────────────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_enums_roundtrip() {
        assert_eq!(Direction::from_name("row"), Some(Direction::Row));
        assert_eq!(Direction::from_name("column"), Some(Direction::Column));
        assert_eq!(Direction::Row.to_name(), "row");
        assert_eq!(Direction::default(), Direction::Row);

        assert_eq!(Align::from_name("center"), Some(Align::Center));
        assert_eq!(Align::from_name("flex-start"), Some(Align::Start));
        assert_eq!(Align::default(), Align::Stretch);

        assert_eq!(
            Justify::from_name("space-between"),
            Some(Justify::SpaceBetween)
        );

        // from_css with fallback
        assert_eq!(Direction::from_css("row"), Direction::Row);
        assert_eq!(Direction::from_css("unknown"), Direction::Column);
        assert_eq!(Align::from_css("nope"), Align::Start);
    }

    #[test]
    fn dimension_resolve() {
        assert_eq!(Dimension::Auto.resolve(100.0), None);
        assert_eq!(Dimension::Px(50.0).resolve(100.0), Some(50.0));
        assert_eq!(Dimension::Percent(50.0).resolve(200.0), Some(100.0));
        assert_eq!(
            Dimension::Calc {
                percent: 50.0,
                px: 10.0
            }
            .resolve(200.0),
            Some(110.0)
        );
    }

    #[test]
    fn dimension_clamp() {
        let val = Dimension::clamp(50.0, Dimension::Px(20.0), Dimension::Px(80.0), 100.0);
        assert_eq!(val, 50.0);

        let val = Dimension::clamp(10.0, Dimension::Px(20.0), Dimension::Px(80.0), 100.0);
        assert_eq!(val, 20.0);
    }

    #[test]
    fn edges_arithmetic() {
        let e = Edges::all(10.0);
        assert_eq!(e.horizontal(), 20.0);
        assert_eq!(e.vertical(), 20.0);
        assert!(e.any_nonzero());
        assert!(!Edges::ZERO.any_nonzero());
    }

    #[test]
    fn dimension_lerp() {
        let a = Dimension::Px(0.0);
        let b = Dimension::Px(100.0);
        assert_eq!(a.lerp(b, 0.5), Dimension::Px(50.0));

        // Incompatible types snap at 0.5
        let a = Dimension::Px(10.0);
        let b = Dimension::Percent(50.0);
        assert_eq!(a.lerp(b, 0.3), Dimension::Px(10.0));
        assert_eq!(a.lerp(b, 0.7), Dimension::Percent(50.0));
    }

    #[test]
    fn edges_lerp() {
        let a = Edges::all(0.0);
        let b = Edges::all(10.0);
        let mid = a.lerp(b, 0.5);
        assert!((mid.top - 5.0).abs() < f64::EPSILON);
    }
}
