//! Style — all visual + layout properties for a DOM node.
//!
//! Designed as a single flat struct so the layout solver, painter, and
//! transition system can read/write fields without indirection.
//! Every spatial field is `f64` matching our [`layout`] types exactly.

use crate::render::Color;

/// Main axis direction for child layout (flexbox model).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Direction {
    /// Children placed left-to-right.
    #[default]
    Row,
    /// Children placed top-to-bottom.
    Column,
}

/// Cross-axis alignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Align {
    #[default]
    Start,
    Center,
    End,
    Stretch,
}

/// Main-axis distribution of remaining space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Justify {
    #[default]
    Start,
    Center,
    End,
    SpaceBetween,
    SpaceAround,
    SpaceEvenly,
}

/// How this node participates in parent layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Position {
    /// Normal flow (default).
    #[default]
    Relative,
    /// Removed from flow — positioned relative to parent's content box.
    Absolute,
}

/// Overflow behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Overflow {
    /// Content overflows visibly.
    #[default]
    Visible,
    /// Content is clipped.
    Hidden,
    /// Content is scrollable.
    Scroll,
}

/// A single dimension that can be auto, fixed, or percentage.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Dimension {
    Auto,
    Px(f64),
    Percent(f64),
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
        }
    }

    /// Clamp a computed value between min/max dimensions resolved against parent.
    pub fn clamp(value: f64, min: Self, max: Self, parent: f64) -> f64 {
        let lo = min.resolve(parent).unwrap_or(0.0);
        let hi = max.resolve(parent).unwrap_or(f64::INFINITY);
        value.clamp(lo, hi)
    }
}

/// Edge insets (padding / margin / border-width).
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
}

/// Complete style for one node — layout + visual in one struct.
///
/// All fields have sane defaults matching CSS initial values.
/// The struct is `Clone + Copy`-free (uses `f64` / enums / `Color`).
#[derive(Debug, Clone, PartialEq)]
pub struct Style {
    // ── Layout ──────────────────────────────────────────
    pub width: Dimension,
    pub height: Dimension,
    pub min_width: Dimension,
    pub min_height: Dimension,
    pub max_width: Dimension,
    pub max_height: Dimension,

    pub direction: Direction,
    pub align: Align,
    pub justify: Justify,
    pub gap: f64,

    pub padding: Edges,
    pub margin: Edges,

    pub position: Position,
    /// Offsets for `Position::Absolute` nodes.
    pub left: Dimension,
    pub top: Dimension,

    pub overflow: Overflow,

    /// Flex grow factor (how much of remaining space to absorb).
    pub flex_grow: f64,
    /// Flex shrink factor.
    pub flex_shrink: f64,

    // ── Visual ──────────────────────────────────────────
    pub background: Color,
    pub border_color: Color,
    pub border_width: f64,
    pub corner_radius: f64,
    pub opacity: f64,

    // ── Text (only meaningful on Text nodes) ────────────
    pub font_size: f64,
    pub color: Color,
}

impl Default for Style {
    fn default() -> Self {
        Self {
            width: Dimension::Auto,
            height: Dimension::Auto,
            min_width: Dimension::Auto,
            min_height: Dimension::Auto,
            max_width: Dimension::Auto,
            max_height: Dimension::Auto,
            direction: Direction::Column,
            align: Align::Start,
            justify: Justify::Start,
            gap: 0.0,
            padding: Edges::ZERO,
            margin: Edges::ZERO,
            position: Position::Relative,
            left: Dimension::Auto,
            top: Dimension::Auto,
            overflow: Overflow::Visible,
            flex_grow: 0.0,
            flex_shrink: 1.0,
            background: Color::TRANSPARENT,
            border_color: Color::TRANSPARENT,
            border_width: 0.0,
            corner_radius: 0.0,
            opacity: 1.0,
            font_size: 14.0,
            color: Color::WHITE,
        }
    }
}

impl Style {
    /// Builder: set width in pixels.
    pub fn w(mut self, px: f64) -> Self {
        self.width = Dimension::Px(px);
        self
    }
    /// Builder: set height in pixels.
    pub fn h(mut self, px: f64) -> Self {
        self.height = Dimension::Px(px);
        self
    }
    /// Builder: set width as percentage.
    pub fn w_pct(mut self, p: f64) -> Self {
        self.width = Dimension::Percent(p);
        self
    }
    /// Builder: set height as percentage.
    pub fn h_pct(mut self, p: f64) -> Self {
        self.height = Dimension::Percent(p);
        self
    }

    /// Builder: row direction.
    pub fn row(mut self) -> Self {
        self.direction = Direction::Row;
        self
    }
    /// Builder: column direction (default).
    pub fn col(mut self) -> Self {
        self.direction = Direction::Column;
        self
    }

    /// Builder: set gap.
    pub fn gap(mut self, px: f64) -> Self {
        self.gap = px;
        self
    }

    /// Builder: set padding all sides.
    pub fn pad(mut self, px: f64) -> Self {
        self.padding = Edges::all(px);
        self
    }
    /// Builder: set padding x/y.
    pub fn pad_xy(mut self, x: f64, y: f64) -> Self {
        self.padding = Edges::xy(x, y);
        self
    }

    /// Builder: set margin all sides.
    pub fn margin(mut self, px: f64) -> Self {
        self.margin = Edges::all(px);
        self
    }

    /// Builder: set background color.
    pub fn bg(mut self, c: Color) -> Self {
        self.background = c;
        self
    }

    /// Builder: set border.
    pub fn border(mut self, width: f64, color: Color) -> Self {
        self.border_width = width;
        self.border_color = color;
        self
    }

    /// Builder: set corner radius.
    pub fn radius(mut self, r: f64) -> Self {
        self.corner_radius = r;
        self
    }

    /// Builder: set text color.
    pub fn color(mut self, c: Color) -> Self {
        self.color = c;
        self
    }

    /// Builder: set font size.
    pub fn font(mut self, size: f64) -> Self {
        self.font_size = size;
        self
    }

    /// Builder: set alignment.
    pub fn align(mut self, a: Align) -> Self {
        self.align = a;
        self
    }

    /// Builder: set justify.
    pub fn justify(mut self, j: Justify) -> Self {
        self.justify = j;
        self
    }

    /// Builder: set flex grow.
    pub fn grow(mut self, g: f64) -> Self {
        self.flex_grow = g;
        self
    }

    /// Builder: set overflow.
    pub fn overflow(mut self, o: Overflow) -> Self {
        self.overflow = o;
        self
    }

    /// Builder: set position absolute with offsets.
    pub fn abs(mut self, left: f64, top: f64) -> Self {
        self.position = Position::Absolute;
        self.left = Dimension::Px(left);
        self.top = Dimension::Px(top);
        self
    }

    /// Builder: set opacity.
    pub fn opacity(mut self, o: f64) -> Self {
        self.opacity = o;
        self
    }
}
