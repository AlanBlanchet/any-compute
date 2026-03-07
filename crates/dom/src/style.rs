//! Style — all visual + layout properties for a DOM node.
//!
//! Designed as a single flat struct so the layout solver, painter, and
//! transition system can read/write fields without indirection.
//! Every spatial field is `f64` matching our [`layout`] types exactly.

use any_compute_core::render::Color;

/// Baseline rem-to-px multiplier (browser default: 1rem = 16px).
pub const REM_PX: f64 = 16.0;

/// Display mode (CSS `display`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Display {
    /// Visible flex container (our default layout model).
    #[default]
    Flex,
    /// Visible block — lays out like column-direction flex with no grow.
    Block,
    /// Invisible — excluded from layout and paint.
    None,
}

impl Display {
    /// Parse a CSS `display` value. Unknown values default to `Flex`.
    pub fn from_css(val: &str) -> Self {
        match val {
            "flex" => Self::Flex,
            "block" => Self::Block,
            "none" => Self::None,
            _ => Self::Flex,
        }
    }
}

/// Main axis direction for child layout (flexbox model).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Direction {
    /// Children placed left-to-right.
    #[default]
    Row,
    /// Children placed top-to-bottom.
    Column,
}

impl Direction {
    pub fn from_css(val: &str) -> Self {
        match val {
            "row" => Self::Row,
            _ => Self::Column,
        }
    }
}

/// Whether flex children can wrap to new lines.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FlexWrap {
    /// Single line (default).
    #[default]
    NoWrap,
    /// Wrap to next line when main axis overflows.
    Wrap,
    /// Wrap in reverse direction.
    WrapReverse,
}

impl FlexWrap {
    pub fn from_css(val: &str) -> Self {
        match val {
            "wrap" => Self::Wrap,
            "wrap-reverse" => Self::WrapReverse,
            _ => Self::NoWrap,
        }
    }
}

/// Cross-axis alignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Align {
    Start,
    Center,
    End,
    #[default]
    Stretch,
    Baseline,
}

impl Align {
    pub fn from_css(val: &str) -> Self {
        match val {
            "center" => Self::Center,
            "end" | "flex-end" => Self::End,
            "stretch" => Self::Stretch,
            "baseline" => Self::Baseline,
            _ => Self::Start,
        }
    }
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

impl Justify {
    pub fn from_css(val: &str) -> Self {
        match val {
            "center" => Self::Center,
            "end" | "flex-end" => Self::End,
            "space-between" => Self::SpaceBetween,
            "space-around" => Self::SpaceAround,
            "space-evenly" => Self::SpaceEvenly,
            _ => Self::Start,
        }
    }
}

/// How this node participates in parent layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Position {
    /// Normal flow (default).
    #[default]
    Relative,
    /// Removed from flow — positioned relative to parent's content box.
    Absolute,
    /// Fixed to the viewport.
    Fixed,
}

impl Position {
    pub fn from_css(val: &str) -> Self {
        match val {
            "absolute" => Self::Absolute,
            "fixed" => Self::Fixed,
            _ => Self::Relative,
        }
    }
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
    /// Auto (scrollable only when content overflows).
    Auto,
}

impl Overflow {
    pub fn from_css(val: &str) -> Self {
        match val {
            "hidden" => Self::Hidden,
            "scroll" => Self::Scroll,
            "auto" => Self::Auto,
            _ => Self::Visible,
        }
    }
}

/// Text alignment within a text node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TextAlign {
    #[default]
    Left,
    Center,
    Right,
}

impl TextAlign {
    pub fn from_css(val: &str) -> Self {
        match val {
            "center" => Self::Center,
            "right" | "end" => Self::Right,
            _ => Self::Left,
        }
    }
}

/// Font weight (CSS `font-weight`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FontWeight(pub u16);

impl FontWeight {
    pub const NORMAL: Self = Self(400);
    pub const BOLD: Self = Self(700);
    pub const LIGHT: Self = Self(300);
    pub const THIN: Self = Self(100);
    pub const SEMIBOLD: Self = Self(600);
    pub const EXTRABOLD: Self = Self(800);
    pub const BLACK: Self = Self(900);

    /// Parse CSS `font-weight` value (keyword or numeric 100–900).
    pub fn from_css(val: &str) -> Option<Self> {
        match val.trim() {
            "normal" => Some(Self::NORMAL),
            "bold" => Some(Self::BOLD),
            "lighter" | "light" => Some(Self::LIGHT),
            v => v.parse::<u16>().ok().map(Self),
        }
    }
}

impl Default for FontWeight {
    fn default() -> Self {
        Self::NORMAL
    }
}

/// Visibility (CSS `visibility`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Visibility {
    #[default]
    Visible,
    Hidden,
}

impl Visibility {
    pub fn from_css(val: &str) -> Self {
        match val {
            "hidden" => Self::Hidden,
            _ => Self::Visible,
        }
    }
}

/// White-space handling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WhiteSpace {
    #[default]
    Normal,
    NoWrap,
    Pre,
}

impl WhiteSpace {
    pub fn from_css(val: &str) -> Self {
        match val {
            "nowrap" => Self::NoWrap,
            "pre" => Self::Pre,
            _ => Self::Normal,
        }
    }
}

/// Box-sizing model (CSS `box-sizing`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BoxSizing {
    /// Width/height include only content (CSS default).
    ContentBox,
    /// Width/height include padding + border (the pragmatic default).
    #[default]
    BorderBox,
}

impl BoxSizing {
    pub fn from_css(val: &str) -> Self {
        match val {
            "content-box" => Self::ContentBox,
            _ => Self::BorderBox,
        }
    }
}

/// Text decoration line (CSS `text-decoration`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TextDecoration {
    #[default]
    None,
    Underline,
    Overline,
    LineThrough,
}

impl TextDecoration {
    pub fn from_css(val: &str) -> Self {
        match val {
            "underline" => Self::Underline,
            "overline" => Self::Overline,
            "line-through" => Self::LineThrough,
            "none" => Self::None,
            _ => Self::None,
        }
    }
}

/// Text transform (CSS `text-transform`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TextTransform {
    #[default]
    None,
    Uppercase,
    Lowercase,
    Capitalize,
}

impl TextTransform {
    pub fn from_css(val: &str) -> Self {
        match val {
            "uppercase" => Self::Uppercase,
            "lowercase" => Self::Lowercase,
            "capitalize" => Self::Capitalize,
            _ => Self::None,
        }
    }
}

/// Cursor style (CSS `cursor`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Cursor {
    #[default]
    Default,
    Pointer,
    Text,
    Move,
    NotAllowed,
    Grab,
    Grabbing,
    Crosshair,
    Help,
    Wait,
    None,
}

impl Cursor {
    pub fn from_css(val: &str) -> Self {
        match val {
            "pointer" => Self::Pointer,
            "text" => Self::Text,
            "move" => Self::Move,
            "not-allowed" => Self::NotAllowed,
            "grab" => Self::Grab,
            "grabbing" => Self::Grabbing,
            "crosshair" => Self::Crosshair,
            "help" => Self::Help,
            "wait" => Self::Wait,
            "none" => Self::None,
            _ => Self::Default,
        }
    }
}

/// Pointer events (CSS `pointer-events`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PointerEvents {
    #[default]
    Auto,
    None,
}

impl PointerEvents {
    pub fn from_css(val: &str) -> Self {
        match val {
            "none" => Self::None,
            _ => Self::Auto,
        }
    }
}

/// User select (CSS `user-select`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum UserSelect {
    #[default]
    Auto,
    None,
    Text,
    All,
}

impl UserSelect {
    pub fn from_css(val: &str) -> Self {
        match val {
            "none" => Self::None,
            "text" => Self::Text,
            "all" => Self::All,
            _ => Self::Auto,
        }
    }
}

/// Text overflow (CSS `text-overflow`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TextOverflow {
    #[default]
    Clip,
    Ellipsis,
}

impl TextOverflow {
    pub fn from_css(val: &str) -> Self {
        match val {
            "ellipsis" => Self::Ellipsis,
            _ => Self::Clip,
        }
    }
}

/// Word break (CSS `word-break` / `overflow-wrap`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WordBreak {
    #[default]
    Normal,
    BreakAll,
    KeepAll,
    BreakWord,
}

impl WordBreak {
    pub fn from_css(val: &str) -> Self {
        match val {
            "break-all" => Self::BreakAll,
            "keep-all" => Self::KeepAll,
            "break-word" => Self::BreakWord,
            _ => Self::Normal,
        }
    }
}

/// Border style (CSS `border-style`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BorderStyle {
    #[default]
    None,
    Solid,
    Dashed,
    Dotted,
    Double,
    Groove,
    Ridge,
    Inset,
    Outset,
}

impl BorderStyle {
    pub fn from_css(val: &str) -> Self {
        match val {
            "solid" => Self::Solid,
            "dashed" => Self::Dashed,
            "dotted" => Self::Dotted,
            "double" => Self::Double,
            "groove" => Self::Groove,
            "ridge" => Self::Ridge,
            "inset" => Self::Inset,
            "outset" => Self::Outset,
            "none" => Self::None,
            _ => Self::None,
        }
    }
}

/// Object fit (CSS `object-fit`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ObjectFit {
    #[default]
    Fill,
    Contain,
    Cover,
    ScaleDown,
    None,
}

impl ObjectFit {
    pub fn from_css(val: &str) -> Self {
        match val {
            "contain" => Self::Contain,
            "cover" => Self::Cover,
            "scale-down" => Self::ScaleDown,
            "none" => Self::None,
            _ => Self::Fill,
        }
    }
}

/// A single dimension that can be auto, fixed, percentage, or calc.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Dimension {
    Auto,
    Px(f64),
    Percent(f64),
    /// `calc(A% ± Bpx)` — resolved at layout time with parent size.
    /// Covers the vast majority of real-world `calc()` usage while staying `Copy`.
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

/// Box shadow (CSS `box-shadow`).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Shadow {
    pub x: f64,
    pub y: f64,
    pub blur: f64,
    pub spread: f64,
    pub color: Color,
    pub inset: bool,
}

/// Bitmask tracking which [`Style`] fields were explicitly set by CSS.
///
/// Used to distinguish "default because unset" from "explicitly set to the default value",
/// enabling correct CSS property inheritance. Each bit corresponds to one field group.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct StyleWritten(pub u64);

impl StyleWritten {
    pub const fn has(self, bit: u64) -> bool {
        self.0 & bit != 0
    }
    pub fn set(&mut self, bit: u64) {
        self.0 |= bit;
    }
}

// ── Inheritance bit constants ─────────────────────
// Properties that inherit from parent by default in CSS:
pub const INHERIT_COLOR: u64 = 1 << 0;
pub const INHERIT_FONT_SIZE: u64 = 1 << 1;
pub const INHERIT_FONT_WEIGHT: u64 = 1 << 2;
pub const INHERIT_LINE_HEIGHT: u64 = 1 << 3;
pub const INHERIT_TEXT_ALIGN: u64 = 1 << 4;
pub const INHERIT_WHITE_SPACE: u64 = 1 << 5;
pub const INHERIT_VISIBILITY: u64 = 1 << 6;
pub const INHERIT_CURSOR: u64 = 1 << 7;
pub const INHERIT_LETTER_SPACING: u64 = 1 << 8;
pub const INHERIT_WORD_SPACING: u64 = 1 << 9;
pub const INHERIT_TEXT_TRANSFORM: u64 = 1 << 10;
pub const INHERIT_TEXT_INDENT: u64 = 1 << 11;
pub const INHERIT_WORD_BREAK: u64 = 1 << 12;
pub const INHERIT_DIRECTION: u64 = 1 << 13;

/// Mask of all inheritable properties.
pub const INHERIT_ALL: u64 = INHERIT_COLOR
    | INHERIT_FONT_SIZE
    | INHERIT_FONT_WEIGHT
    | INHERIT_LINE_HEIGHT
    | INHERIT_TEXT_ALIGN
    | INHERIT_WHITE_SPACE
    | INHERIT_VISIBILITY
    | INHERIT_CURSOR
    | INHERIT_LETTER_SPACING
    | INHERIT_WORD_SPACING
    | INHERIT_TEXT_TRANSFORM
    | INHERIT_TEXT_INDENT
    | INHERIT_WORD_BREAK
    | INHERIT_DIRECTION;

/// Complete style for one node — layout + visual in one struct.
///
/// All fields have sane defaults matching CSS initial values.
/// The struct is `Clone + Copy`-free (uses `f64` / enums / `Color`).
#[derive(Debug, Clone, PartialEq)]
pub struct Style {
    // ── Display / Box Model ─────────────────────────────
    pub display: Display,
    pub box_sizing: BoxSizing,
    pub visibility: Visibility,

    // ── Layout ──────────────────────────────────────────
    pub width: Dimension,
    pub height: Dimension,
    pub min_width: Dimension,
    pub min_height: Dimension,
    pub max_width: Dimension,
    pub max_height: Dimension,
    pub aspect_ratio: Option<f64>,

    pub direction: Direction,
    pub flex_wrap: FlexWrap,
    pub align: Align,
    pub align_self: Option<Align>,
    pub justify: Justify,
    pub gap: f64,
    pub row_gap: Option<f64>,
    pub column_gap: Option<f64>,

    pub padding: Edges,
    pub margin: Edges,

    pub position: Position,
    /// Offsets for positioned nodes.
    pub left: Dimension,
    pub top: Dimension,
    pub right: Dimension,
    pub bottom: Dimension,

    pub overflow: Overflow,

    /// Flex grow factor (how much of remaining space to absorb).
    pub flex_grow: f64,
    /// Flex shrink factor.
    pub flex_shrink: f64,
    /// Flex basis (initial main size before grow/shrink).
    pub flex_basis: Dimension,
    /// Flex item ordering.
    pub order: i32,

    /// Z-index for stacking order (higher = on top).
    /// `None` means auto (paint in insertion order).
    pub z_index: Option<i32>,

    // ── Visual ──────────────────────────────────────────
    pub background: Color,
    pub border_color: Color,
    pub border_style: BorderStyle,
    pub border_width: f64,
    pub border_top_width: f64,
    pub border_right_width: f64,
    pub border_bottom_width: f64,
    pub border_left_width: f64,
    pub corner_radius: f64,
    pub opacity: f64,
    pub box_shadow: Option<Shadow>,
    pub outline_width: f64,
    pub outline_color: Color,

    // ── Transform ───────────────────────────────────────
    pub transform_translate_x: f64,
    pub transform_translate_y: f64,
    pub transform_scale_x: f64,
    pub transform_scale_y: f64,
    pub transform_rotate: f64,
    pub transform_skew_x: f64,
    pub transform_skew_y: f64,

    // ── Filter ──────────────────────────────────────────
    pub filter_blur: f64,
    pub filter_brightness: f64,
    pub filter_contrast: f64,
    pub filter_opacity: f64,

    // ── Text ────────────────────────────────────────────
    pub font_size: f64,
    pub font_weight: FontWeight,
    pub line_height: f64,
    pub color: Color,
    pub text_align: TextAlign,
    pub white_space: WhiteSpace,
    pub text_decoration: TextDecoration,
    pub text_transform: TextTransform,
    pub letter_spacing: f64,
    pub word_spacing: f64,
    pub text_indent: f64,
    pub text_overflow: TextOverflow,
    pub text_shadow: Option<Shadow>,
    pub word_break: WordBreak,

    // ── Interaction ─────────────────────────────────────
    pub cursor: Cursor,
    pub pointer_events: PointerEvents,
    pub user_select: UserSelect,

    // ── Inheritance tracking ────────────────────────────
    /// Bitmask of which properties were explicitly set (vs inherited/default).
    pub written: StyleWritten,
}

impl Default for Style {
    fn default() -> Self {
        Self {
            display: Display::Flex,
            box_sizing: BoxSizing::BorderBox,
            visibility: Visibility::Visible,
            width: Dimension::Auto,
            height: Dimension::Auto,
            min_width: Dimension::Auto,
            min_height: Dimension::Auto,
            max_width: Dimension::Auto,
            max_height: Dimension::Auto,
            aspect_ratio: None,
            direction: Direction::Column,
            flex_wrap: FlexWrap::NoWrap,
            align: Align::Stretch,
            align_self: None,
            justify: Justify::Start,
            gap: 0.0,
            row_gap: None,
            column_gap: None,
            padding: Edges::ZERO,
            margin: Edges::ZERO,
            position: Position::Relative,
            left: Dimension::Auto,
            top: Dimension::Auto,
            right: Dimension::Auto,
            bottom: Dimension::Auto,
            overflow: Overflow::Visible,
            flex_grow: 0.0,
            flex_shrink: 1.0,
            flex_basis: Dimension::Auto,
            order: 0,
            z_index: None,
            background: Color::TRANSPARENT,
            border_color: Color::TRANSPARENT,
            border_style: BorderStyle::None,
            border_width: 0.0,
            border_top_width: 0.0,
            border_right_width: 0.0,
            border_bottom_width: 0.0,
            border_left_width: 0.0,
            corner_radius: 0.0,
            opacity: 1.0,
            box_shadow: None,
            outline_width: 0.0,
            outline_color: Color::TRANSPARENT,
            transform_translate_x: 0.0,
            transform_translate_y: 0.0,
            transform_scale_x: 1.0,
            transform_scale_y: 1.0,
            transform_rotate: 0.0,
            transform_skew_x: 0.0,
            transform_skew_y: 0.0,
            filter_blur: 0.0,
            filter_brightness: 1.0,
            filter_contrast: 1.0,
            filter_opacity: 1.0,
            font_size: 14.0,
            font_weight: FontWeight::NORMAL,
            line_height: 1.3,
            color: Color::WHITE,
            text_align: TextAlign::Left,
            white_space: WhiteSpace::Normal,
            text_decoration: TextDecoration::None,
            text_transform: TextTransform::None,
            letter_spacing: 0.0,
            word_spacing: 0.0,
            text_indent: 0.0,
            text_overflow: TextOverflow::Clip,
            text_shadow: None,
            word_break: WordBreak::Normal,
            cursor: Cursor::Default,
            pointer_events: PointerEvents::Auto,
            user_select: UserSelect::Auto,
            written: StyleWritten(0),
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

    /// Builder: set z-index.
    pub fn z(mut self, z: i32) -> Self {
        self.z_index = Some(z);
        self
    }

    /// Builder: set display none.
    pub fn hidden(mut self) -> Self {
        self.display = Display::None;
        self
    }

    /// Builder: set font weight.
    pub fn bold(mut self) -> Self {
        self.font_weight = FontWeight::BOLD;
        self
    }

    /// Builder: set line-height multiplier.
    pub fn lh(mut self, lh: f64) -> Self {
        self.line_height = lh;
        self
    }

    /// Total effective border width on each side.
    /// Uses per-side widths if set, otherwise falls back to uniform `border_width`.
    pub fn effective_border(&self) -> Edges {
        let bw = self.border_width;
        Edges {
            top: if self.border_top_width > 0.0 {
                self.border_top_width
            } else {
                bw
            },
            right: if self.border_right_width > 0.0 {
                self.border_right_width
            } else {
                bw
            },
            bottom: if self.border_bottom_width > 0.0 {
                self.border_bottom_width
            } else {
                bw
            },
            left: if self.border_left_width > 0.0 {
                self.border_left_width
            } else {
                bw
            },
        }
    }

    /// True when this node is out-of-flow (absolute or fixed).
    pub fn is_out_of_flow(&self) -> bool {
        matches!(self.position, Position::Absolute | Position::Fixed)
    }

    /// True when display is none.
    pub fn is_hidden(&self) -> bool {
        self.display == Display::None
    }

    /// Inherit CSS-inheritable properties from a parent style.
    ///
    /// Only copies properties that (a) are inheritable per CSS spec and
    /// (b) were NOT explicitly set on this node (tracked via `written`).
    pub fn inherit_from(&mut self, parent: &Style) {
        let w = self.written;
        if !w.has(INHERIT_COLOR) {
            self.color = parent.color;
        }
        if !w.has(INHERIT_FONT_SIZE) {
            self.font_size = parent.font_size;
        }
        if !w.has(INHERIT_FONT_WEIGHT) {
            self.font_weight = parent.font_weight;
        }
        if !w.has(INHERIT_LINE_HEIGHT) {
            self.line_height = parent.line_height;
        }
        if !w.has(INHERIT_TEXT_ALIGN) {
            self.text_align = parent.text_align;
        }
        if !w.has(INHERIT_WHITE_SPACE) {
            self.white_space = parent.white_space;
        }
        if !w.has(INHERIT_VISIBILITY) {
            self.visibility = parent.visibility;
        }
        if !w.has(INHERIT_CURSOR) {
            self.cursor = parent.cursor;
        }
        if !w.has(INHERIT_LETTER_SPACING) {
            self.letter_spacing = parent.letter_spacing;
        }
        if !w.has(INHERIT_WORD_SPACING) {
            self.word_spacing = parent.word_spacing;
        }
        if !w.has(INHERIT_TEXT_TRANSFORM) {
            self.text_transform = parent.text_transform;
        }
        if !w.has(INHERIT_TEXT_INDENT) {
            self.text_indent = parent.text_indent;
        }
        if !w.has(INHERIT_WORD_BREAK) {
            self.word_break = parent.word_break;
        }
        if !w.has(INHERIT_DIRECTION) {
            self.direction = parent.direction;
        }
    }
}

// ── Pre-compiled style operations ───────────────────────────────────────────

/// Pre-compiled style mutation — zero string matching at apply time.
///
/// Created at CSS parse time or by the Tailwind class compiler.
/// Each variant maps to one or two `Style` field writes.
/// Applying N ops is N enum matches — no string hashing or parsing.
#[derive(Debug, Clone, PartialEq)]
pub enum StyleOp {
    // ── Display / Box Model ─────────────────────────────
    Display(Display),
    BoxSizing(BoxSizing),
    Visibility(Visibility),

    // ── Dimensions ──────────────────────────────────────
    Width(Dimension),
    Height(Dimension),
    MinWidth(Dimension),
    MinHeight(Dimension),
    MaxWidth(Dimension),
    MaxHeight(Dimension),
    AspectRatio(f64),

    // ── Flex layout ─────────────────────────────────────
    Direction(Direction),
    FlexWrap(FlexWrap),
    Align(Align),
    AlignSelf(Align),
    Justify(Justify),
    Gap(f64),
    RowGap(f64),
    ColumnGap(f64),

    // ── Spacing ─────────────────────────────────────────
    Padding(Edges),
    PaddingX(f64),
    PaddingY(f64),
    PaddingTop(f64),
    PaddingRight(f64),
    PaddingBottom(f64),
    PaddingLeft(f64),
    Margin(Edges),
    MarginX(f64),
    MarginY(f64),
    MarginTop(f64),
    MarginRight(f64),
    MarginBottom(f64),
    MarginLeft(f64),

    // ── Position ────────────────────────────────────────
    Position(Position),
    Left(Dimension),
    Top(Dimension),
    Right(Dimension),
    Bottom(Dimension),
    ZIndex(i32),

    // ── Flex item ───────────────────────────────────────
    FlexGrow(f64),
    FlexShrink(f64),
    FlexBasis(Dimension),
    Order(i32),

    // ── Overflow ────────────────────────────────────────
    Overflow(Overflow),

    // ── Visual ──────────────────────────────────────────
    Background(Color),
    BorderColor(Color),
    BorderStyleOp(BorderStyle),
    BorderWidth(f64),
    BorderTopWidth(f64),
    BorderRightWidth(f64),
    BorderBottomWidth(f64),
    BorderLeftWidth(f64),
    CornerRadius(f64),
    Opacity(f64),
    BoxShadow(Shadow),
    OutlineWidth(f64),
    OutlineColor(Color),

    // ── Transform ───────────────────────────────────────
    TranslateX(f64),
    TranslateY(f64),
    ScaleX(f64),
    ScaleY(f64),
    Rotate(f64),
    SkewX(f64),
    SkewY(f64),

    // ── Filter ──────────────────────────────────────────
    FilterBlur(f64),
    FilterBrightness(f64),
    FilterContrast(f64),
    FilterOpacity(f64),

    // ── Text ────────────────────────────────────────────
    FontSize(f64),
    FontWeight(FontWeight),
    LineHeight(f64),
    TextColor(Color),
    TextAlign(TextAlign),
    WhiteSpace(WhiteSpace),
    TextDecorationOp(TextDecoration),
    TextTransformOp(TextTransform),
    LetterSpacing(f64),
    WordSpacing(f64),
    TextIndent(f64),
    TextOverflowOp(TextOverflow),
    TextShadow(Shadow),
    WordBreakOp(WordBreak),

    // ── Interaction ─────────────────────────────────────
    CursorOp(Cursor),
    PointerEventsOp(PointerEvents),
    UserSelectOp(UserSelect),
}

impl StyleOp {
    /// Apply this pre-compiled operation to a [`Style`].
    #[inline]
    pub fn apply(&self, s: &mut Style) {
        match self {
            Self::Display(d) => s.display = *d,
            Self::BoxSizing(b) => s.box_sizing = *b,
            Self::Visibility(v) => {
                s.visibility = *v;
                s.written.set(INHERIT_VISIBILITY);
            }
            Self::Width(d) => s.width = *d,
            Self::Height(d) => s.height = *d,
            Self::MinWidth(d) => s.min_width = *d,
            Self::MinHeight(d) => s.min_height = *d,
            Self::MaxWidth(d) => s.max_width = *d,
            Self::MaxHeight(d) => s.max_height = *d,
            Self::AspectRatio(v) => s.aspect_ratio = Some(*v),
            Self::Direction(d) => {
                s.direction = *d;
                s.written.set(INHERIT_DIRECTION);
            }
            Self::FlexWrap(w) => s.flex_wrap = *w,
            Self::Align(a) => s.align = *a,
            Self::AlignSelf(a) => s.align_self = Some(*a),
            Self::Justify(j) => s.justify = *j,
            Self::Gap(v) => s.gap = *v,
            Self::RowGap(v) => s.row_gap = Some(*v),
            Self::ColumnGap(v) => s.column_gap = Some(*v),
            Self::Padding(e) => s.padding = *e,
            Self::PaddingX(v) => {
                s.padding.left = *v;
                s.padding.right = *v;
            }
            Self::PaddingY(v) => {
                s.padding.top = *v;
                s.padding.bottom = *v;
            }
            Self::PaddingTop(v) => s.padding.top = *v,
            Self::PaddingRight(v) => s.padding.right = *v,
            Self::PaddingBottom(v) => s.padding.bottom = *v,
            Self::PaddingLeft(v) => s.padding.left = *v,
            Self::Margin(e) => s.margin = *e,
            Self::MarginX(v) => {
                s.margin.left = *v;
                s.margin.right = *v;
            }
            Self::MarginY(v) => {
                s.margin.top = *v;
                s.margin.bottom = *v;
            }
            Self::MarginTop(v) => s.margin.top = *v,
            Self::MarginRight(v) => s.margin.right = *v,
            Self::MarginBottom(v) => s.margin.bottom = *v,
            Self::MarginLeft(v) => s.margin.left = *v,
            Self::Position(p) => s.position = *p,
            Self::Left(d) => s.left = *d,
            Self::Top(d) => s.top = *d,
            Self::Right(d) => s.right = *d,
            Self::Bottom(d) => s.bottom = *d,
            Self::ZIndex(z) => s.z_index = Some(*z),
            Self::FlexGrow(v) => s.flex_grow = *v,
            Self::FlexShrink(v) => s.flex_shrink = *v,
            Self::FlexBasis(d) => s.flex_basis = *d,
            Self::Order(v) => s.order = *v,
            Self::Overflow(o) => s.overflow = *o,
            Self::Background(c) => s.background = *c,
            Self::BorderColor(c) => s.border_color = *c,
            Self::BorderStyleOp(v) => s.border_style = *v,
            Self::BorderWidth(v) => s.border_width = *v,
            Self::BorderTopWidth(v) => s.border_top_width = *v,
            Self::BorderRightWidth(v) => s.border_right_width = *v,
            Self::BorderBottomWidth(v) => s.border_bottom_width = *v,
            Self::BorderLeftWidth(v) => s.border_left_width = *v,
            Self::CornerRadius(v) => s.corner_radius = *v,
            Self::Opacity(v) => s.opacity = *v,
            Self::BoxShadow(v) => s.box_shadow = Some(*v),
            Self::OutlineWidth(v) => s.outline_width = *v,
            Self::OutlineColor(c) => s.outline_color = *c,
            Self::TranslateX(v) => s.transform_translate_x = *v,
            Self::TranslateY(v) => s.transform_translate_y = *v,
            Self::ScaleX(v) => s.transform_scale_x = *v,
            Self::ScaleY(v) => s.transform_scale_y = *v,
            Self::Rotate(v) => s.transform_rotate = *v,
            Self::SkewX(v) => s.transform_skew_x = *v,
            Self::SkewY(v) => s.transform_skew_y = *v,
            Self::FilterBlur(v) => s.filter_blur = *v,
            Self::FilterBrightness(v) => s.filter_brightness = *v,
            Self::FilterContrast(v) => s.filter_contrast = *v,
            Self::FilterOpacity(v) => s.filter_opacity = *v,
            Self::FontSize(v) => {
                s.font_size = *v;
                s.written.set(INHERIT_FONT_SIZE);
            }
            Self::FontWeight(w) => {
                s.font_weight = *w;
                s.written.set(INHERIT_FONT_WEIGHT);
            }
            Self::LineHeight(v) => {
                s.line_height = *v;
                s.written.set(INHERIT_LINE_HEIGHT);
            }
            Self::TextColor(c) => {
                s.color = *c;
                s.written.set(INHERIT_COLOR);
            }
            Self::TextAlign(a) => {
                s.text_align = *a;
                s.written.set(INHERIT_TEXT_ALIGN);
            }
            Self::WhiteSpace(w) => {
                s.white_space = *w;
                s.written.set(INHERIT_WHITE_SPACE);
            }
            Self::TextDecorationOp(v) => s.text_decoration = *v,
            Self::TextTransformOp(v) => {
                s.text_transform = *v;
                s.written.set(INHERIT_TEXT_TRANSFORM);
            }
            Self::LetterSpacing(v) => {
                s.letter_spacing = *v;
                s.written.set(INHERIT_LETTER_SPACING);
            }
            Self::WordSpacing(v) => {
                s.word_spacing = *v;
                s.written.set(INHERIT_WORD_SPACING);
            }
            Self::TextIndent(v) => {
                s.text_indent = *v;
                s.written.set(INHERIT_TEXT_INDENT);
            }
            Self::TextOverflowOp(v) => s.text_overflow = *v,
            Self::TextShadow(v) => s.text_shadow = Some(*v),
            Self::WordBreakOp(v) => {
                s.word_break = *v;
                s.written.set(INHERIT_WORD_BREAK);
            }
            Self::CursorOp(v) => {
                s.cursor = *v;
                s.written.set(INHERIT_CURSOR);
            }
            Self::PointerEventsOp(v) => s.pointer_events = *v,
            Self::UserSelectOp(v) => s.user_select = *v,
        }
    }
}

/// Apply a list of pre-compiled operations to a [`Style`].
#[inline]
pub fn apply_ops(s: &mut Style, ops: &[StyleOp]) {
    for op in ops {
        op.apply(s);
    }
}
