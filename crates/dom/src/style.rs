//! Style — all visual + layout properties for a DOM node.
//!
//! Designed as a single flat struct so the layout solver, painter, and
//! transition system can read/write fields without indirection.
//! Every spatial field is `f64` matching our [`layout`] types exactly.

use any_compute_core::Lerp;
use any_compute_core::animation::Easing;
use any_compute_core::layout::Rect;
use any_compute_core::render::Color;

/// Baseline rem-to-px multiplier (browser default: 1rem = 16px).
pub const REM_PX: f64 = 16.0;
/// Default font size in pixels.
pub const DEFAULT_FONT_SIZE: f64 = 14.0;
/// Default line-height as a unitless multiplier.
pub const DEFAULT_LINE_HEIGHT: f64 = 1.3;
/// Approximate character width as a fraction of font_size (monospace ≈ 0.6, proportional ≈ 0.55).
pub const CHAR_WIDTH_RATIO: f64 = 0.55;
/// Text baseline vertical offset as a fraction of font_size.
pub const TEXT_BASELINE_RATIO: f64 = 0.85;
/// Minimum bar element height in pixels.
pub const MIN_BAR_HEIGHT: f64 = 8.0;
/// Default bar track background (subtle white overlay).
pub const BAR_TRACK_BG: Color = Color::rgba(255, 255, 255, 20);
/// Default easing for transitions/animations when none specified.
pub const DEFAULT_EASING: Easing = Easing::EaseInOut;

// ── CSS enum generator ──────────────────────────────────────────────────────
//
// Every CSS keyword enum follows the same shape:
//   #[derive(Debug, Clone, Copy, PartialEq, Eq)] + Default + from_css(&str).
//
// The macro eliminates per-enum boilerplate. Each entry reads:
//   EnumName [DefaultVariant, FallbackForUnknownCSS] { "css-value" | "alias" => Variant, … }
//
// DefaultVariant   — what `Default::default()` returns.
// FallbackVariant  — what `from_css()` returns for unrecognised input.
// These differ for Direction (default Row, fallback Column) and Align (default Stretch, fallback Start).

macro_rules! css_enums {
    ($( $(#[doc = $doc:literal])* $name:ident [$def:ident, $fb:ident] {
        $( $($css:literal)|+ => $variant:ident ),+ $(,)?
    })*) => {$(
        $(#[doc = $doc])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub enum $name { $( $variant, )+ }
        impl Default for $name { fn default() -> Self { Self::$def } }
        impl $name {
            /// Parse a CSS value string. Unknown values use the type's fallback.
            pub fn from_css(val: &str) -> Self {
                match val { $( $($css)|+ => Self::$variant, )+ _ => Self::$fb, }
            }
            /// Canonical CSS string for this variant (first alias listed in the macro).
            pub fn to_css(self) -> &'static str {
                match self { $( Self::$variant => css_enums!(@first $($css),+), )+ }
            }
        }
    )*};
    // Helper: extract the first literal from a comma-separated list.
    (@first $first:literal $(, $rest:literal)*) => { $first };
}

css_enums! {
    /// Display mode (CSS `display`).
    Display [Flex, Flex] { "flex" => Flex, "block" => Block, "none" => None }

    /// Main axis direction for child layout (flexbox model).
    Direction [Row, Column] { "row" => Row, "column" => Column }

    /// Whether flex children can wrap to new lines.
    FlexWrap [NoWrap, NoWrap] { "nowrap" | "no-wrap" => NoWrap, "wrap" => Wrap, "wrap-reverse" => WrapReverse }

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
    Position [Relative, Relative] { "relative" => Relative, "absolute" => Absolute, "fixed" => Fixed }

    /// Overflow behavior.
    Overflow [Visible, Visible] { "visible" => Visible, "hidden" => Hidden, "scroll" => Scroll, "auto" => Auto }

    /// Text alignment within a text node.
    TextAlign [Left, Left] { "left" | "start" => Left, "center" => Center, "right" | "end" => Right }

    /// Visibility (CSS `visibility`).
    Visibility [Visible, Visible] { "visible" => Visible, "hidden" => Hidden }

    /// White-space handling.
    WhiteSpace [Normal, Normal] { "normal" => Normal, "nowrap" => NoWrap, "pre" => Pre }

    /// Box-sizing model (CSS `box-sizing`).
    BoxSizing [BorderBox, BorderBox] { "border-box" => BorderBox, "content-box" => ContentBox }

    /// Text decoration line (CSS `text-decoration`).
    TextDecoration [None, None] { "none" => None, "underline" => Underline, "overline" => Overline, "line-through" => LineThrough }

    /// Text transform (CSS `text-transform`).
    TextTransform [None, None] { "none" => None, "uppercase" => Uppercase, "lowercase" => Lowercase, "capitalize" => Capitalize }

    /// Cursor style (CSS `cursor`).
    Cursor [Default, Default] {
        "default" => Default, "pointer" => Pointer, "text" => Text, "move" => Move,
        "not-allowed" => NotAllowed, "grab" => Grab, "grabbing" => Grabbing,
        "crosshair" => Crosshair, "help" => Help, "wait" => Wait, "none" => None
    }

    /// Pointer events (CSS `pointer-events`).
    PointerEvents [Auto, Auto] { "auto" => Auto, "none" => None }

    /// User select (CSS `user-select`).
    UserSelect [Auto, Auto] { "auto" => Auto, "none" => None, "text" => Text, "all" => All }

    /// Text overflow (CSS `text-overflow`).
    TextOverflow [Clip, Clip] { "clip" => Clip, "ellipsis" => Ellipsis }

    /// Word break (CSS `word-break` / `overflow-wrap`).
    WordBreak [Normal, Normal] { "normal" => Normal, "break-all" => BreakAll, "keep-all" => KeepAll, "break-word" => BreakWord }

    /// Border style (CSS `border-style`).
    BorderStyle [None, None] {
        "none" => None, "solid" => Solid, "dashed" => Dashed, "dotted" => Dotted,
        "double" => Double, "groove" => Groove, "ridge" => Ridge,
        "inset" => Inset, "outset" => Outset
    }

    /// Object fit (CSS `object-fit`).
    ObjectFit [Fill, Fill] { "fill" => Fill, "contain" => Contain, "cover" => Cover, "scale-down" => ScaleDown, "none" => None }
}

// ── Font weight (struct, not enum — variable 100–900 range) ─────────────────

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

    /// Interpolate font weight (rounds to nearest integer).
    pub fn lerp(self, other: Self, t: f64) -> Self {
        Self((self.0 as f64 + (other.0 as f64 - self.0 as f64) * t).round() as u16)
    }
}

impl Default for FontWeight {
    fn default() -> Self {
        Self::NORMAL
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

    /// Interpolate between two dimensions. Same-variant pairs blend; mixed snap at t=0.5.
    pub fn lerp(self, other: Self, t: f64) -> Self {
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
    /// True when any side is non-zero.
    pub fn any_nonzero(&self) -> bool {
        self.top > 0.0 || self.right > 0.0 || self.bottom > 0.0 || self.left > 0.0
    }

    /// Component-wise linear interpolation.
    pub fn lerp(self, other: Self, t: f64) -> Self {
        Self {
            top: self.top.lerp(other.top, t),
            right: self.right.lerp(other.right, t),
            bottom: self.bottom.lerp(other.bottom, t),
            left: self.left.lerp(other.left, t),
        }
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

impl Shadow {
    /// Component-wise linear interpolation.
    pub fn lerp(self, other: Self, t: f64) -> Self {
        Self {
            x: self.x.lerp(other.x, t),
            y: self.y.lerp(other.y, t),
            blur: self.blur.lerp(other.blur, t),
            spread: self.spread.lerp(other.spread, t),
            color: self.color.lerp(other.color, t),
            inset: if t < 0.5 { self.inset } else { other.inset },
        }
    }
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
pub const INHERIT_FONT_FAMILY: u64 = 1 << 14;

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
    | INHERIT_DIRECTION
    | INHERIT_FONT_FAMILY;

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
    pub font_family: Option<String>,
    pub font_size: f64,
    pub font_weight: FontWeight,
    pub line_height: f64,
    /// When `true`, `line_height` is an absolute px value; otherwise it's a
    /// multiplier of `font_size`.
    pub line_height_absolute: bool,
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
            font_family: None,
            font_size: DEFAULT_FONT_SIZE,
            font_weight: FontWeight::NORMAL,
            line_height: DEFAULT_LINE_HEIGHT,
            line_height_absolute: false,
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

    // ── Interpolation ────────────────────────────────────

    /// Linearly interpolate between two styles.
    ///
    /// Numeric / color / edge fields blend smoothly.
    /// Enum / discrete fields snap at `t = 0.5`.
    pub fn lerp(&self, other: &Style, t: f64) -> Style {
        if t <= 0.0 {
            return self.clone();
        }
        if t >= 1.0 {
            return other.clone();
        }

        let lf = |a: f64, b: f64| a + (b - a) * t;
        macro_rules! snap {
            ($a:expr, $b:expr) => {
                if t < 0.5 { $a } else { $b }
            };
        }
        let opt_f = |a: Option<f64>, b: Option<f64>| match (a, b) {
            (Some(x), Some(y)) => Some(lf(x, y)),
            _ => snap!(a, b),
        };
        let opt_i = |a: Option<i32>, b: Option<i32>| match (a, b) {
            (Some(x), Some(y)) => Some(x + ((y - x) as f64 * t).round() as i32),
            _ => snap!(a, b),
        };
        let opt_shadow = |a: Option<Shadow>, b: Option<Shadow>| match (a, b) {
            (Some(x), Some(y)) => Some(x.lerp(y, t)),
            _ => snap!(a, b),
        };

        Style {
            // Enums — snap
            display: snap!(self.display, other.display),
            box_sizing: snap!(self.box_sizing, other.box_sizing),
            visibility: snap!(self.visibility, other.visibility),
            direction: snap!(self.direction, other.direction),
            flex_wrap: snap!(self.flex_wrap, other.flex_wrap),
            align: snap!(self.align, other.align),
            align_self: snap!(self.align_self, other.align_self),
            justify: snap!(self.justify, other.justify),
            position: snap!(self.position, other.position),
            overflow: snap!(self.overflow, other.overflow),
            text_align: snap!(self.text_align, other.text_align),
            white_space: snap!(self.white_space, other.white_space),
            text_decoration: snap!(self.text_decoration, other.text_decoration),
            text_transform: snap!(self.text_transform, other.text_transform),
            cursor: snap!(self.cursor, other.cursor),
            pointer_events: snap!(self.pointer_events, other.pointer_events),
            user_select: snap!(self.user_select, other.user_select),
            border_style: snap!(self.border_style, other.border_style),
            text_overflow: snap!(self.text_overflow, other.text_overflow),
            word_break: snap!(self.word_break, other.word_break),

            // Dimensions — blend same-variant, snap mixed
            width: self.width.lerp(other.width, t),
            height: self.height.lerp(other.height, t),
            min_width: self.min_width.lerp(other.min_width, t),
            min_height: self.min_height.lerp(other.min_height, t),
            max_width: self.max_width.lerp(other.max_width, t),
            max_height: self.max_height.lerp(other.max_height, t),
            left: self.left.lerp(other.left, t),
            top: self.top.lerp(other.top, t),
            right: self.right.lerp(other.right, t),
            bottom: self.bottom.lerp(other.bottom, t),
            flex_basis: self.flex_basis.lerp(other.flex_basis, t),

            // Numeric — blend
            gap: lf(self.gap, other.gap),
            row_gap: opt_f(self.row_gap, other.row_gap),
            column_gap: opt_f(self.column_gap, other.column_gap),
            flex_grow: lf(self.flex_grow, other.flex_grow),
            flex_shrink: lf(self.flex_shrink, other.flex_shrink),
            order: snap!(self.order, other.order),
            aspect_ratio: opt_f(self.aspect_ratio, other.aspect_ratio),
            z_index: opt_i(self.z_index, other.z_index),

            // Edges — blend each side
            padding: self.padding.lerp(other.padding, t),
            margin: self.margin.lerp(other.margin, t),

            // Visual — colors blend, numeric blend
            background: self.background.lerp(other.background, t),
            border_color: self.border_color.lerp(other.border_color, t),
            border_width: lf(self.border_width, other.border_width),
            border_top_width: lf(self.border_top_width, other.border_top_width),
            border_right_width: lf(self.border_right_width, other.border_right_width),
            border_bottom_width: lf(self.border_bottom_width, other.border_bottom_width),
            border_left_width: lf(self.border_left_width, other.border_left_width),
            corner_radius: lf(self.corner_radius, other.corner_radius),
            opacity: lf(self.opacity, other.opacity),
            box_shadow: opt_shadow(self.box_shadow, other.box_shadow),
            outline_width: lf(self.outline_width, other.outline_width),
            outline_color: self.outline_color.lerp(other.outline_color, t),

            // Transform — blend
            transform_translate_x: lf(self.transform_translate_x, other.transform_translate_x),
            transform_translate_y: lf(self.transform_translate_y, other.transform_translate_y),
            transform_scale_x: lf(self.transform_scale_x, other.transform_scale_x),
            transform_scale_y: lf(self.transform_scale_y, other.transform_scale_y),
            transform_rotate: lf(self.transform_rotate, other.transform_rotate),
            transform_skew_x: lf(self.transform_skew_x, other.transform_skew_x),
            transform_skew_y: lf(self.transform_skew_y, other.transform_skew_y),

            // Filter — blend
            filter_blur: lf(self.filter_blur, other.filter_blur),
            filter_brightness: lf(self.filter_brightness, other.filter_brightness),
            filter_contrast: lf(self.filter_contrast, other.filter_contrast),
            filter_opacity: lf(self.filter_opacity, other.filter_opacity),

            // Text — blend numeric, snap enums (already above)
            font_family: snap!(self.font_family.clone(), other.font_family.clone()),
            font_size: lf(self.font_size, other.font_size),
            font_weight: self.font_weight.lerp(other.font_weight, t),
            line_height: lf(self.line_height, other.line_height),
            line_height_absolute: snap!(self.line_height_absolute, other.line_height_absolute),
            color: self.color.lerp(other.color, t),
            letter_spacing: lf(self.letter_spacing, other.letter_spacing),
            word_spacing: lf(self.word_spacing, other.word_spacing),
            text_indent: lf(self.text_indent, other.text_indent),
            text_shadow: opt_shadow(self.text_shadow, other.text_shadow),

            // Written mask — take the union
            written: StyleWritten(self.written.0 | other.written.0),
        }
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

    /// Whether this node has any non-identity transform.
    pub fn has_transform(&self) -> bool {
        self.transform_translate_x != 0.0
            || self.transform_translate_y != 0.0
            || self.transform_scale_x != 1.0
            || self.transform_scale_y != 1.0
            || self.transform_rotate != 0.0
            || self.transform_skew_x != 0.0
            || self.transform_skew_y != 0.0
    }

    /// Whether border is visible (non-zero width + non-transparent color).
    pub fn has_visible_border(&self) -> bool {
        self.border_color.a > 0 && self.effective_border().any_nonzero()
    }

    /// Apply translate + scale transform to a rectangle (center-relative scaling).
    pub fn transform_rect(&self, r: Rect) -> Rect {
        let cx = r.origin.x + r.size.w / 2.0;
        let cy = r.origin.y + r.size.h / 2.0;
        let sw = r.size.w * self.transform_scale_x;
        let sh = r.size.h * self.transform_scale_y;
        Rect::new(
            cx - sw / 2.0 + self.transform_translate_x,
            cy - sh / 2.0 + self.transform_translate_y,
            sw,
            sh,
        )
    }

    /// Apply opacity pre-multiplication to a color.
    pub fn apply_opacity(&self, c: Color) -> Color {
        if self.opacity >= 1.0 {
            return c;
        }
        let a = (c.a as f64 * self.opacity.clamp(0.0, 1.0)) as u8;
        Color::rgba(c.r, c.g, c.b, a)
    }

    /// Compute the effective text content after `text-transform`.
    pub fn transform_text<'a>(&self, s: &'a str) -> std::borrow::Cow<'a, str> {
        match self.text_transform {
            TextTransform::Uppercase => std::borrow::Cow::Owned(s.to_uppercase()),
            TextTransform::Lowercase => std::borrow::Cow::Owned(s.to_lowercase()),
            TextTransform::Capitalize => {
                let mut result = String::with_capacity(s.len());
                let mut prev_space = true;
                for c in s.chars() {
                    if prev_space && c.is_alphabetic() {
                        result.extend(c.to_uppercase());
                    } else {
                        result.push(c);
                    }
                    prev_space = c.is_whitespace();
                }
                std::borrow::Cow::Owned(result)
            }
            TextTransform::None => std::borrow::Cow::Borrowed(s),
        }
    }

    /// Compute the character width factor including letter-spacing.
    pub fn char_width(&self) -> f64 {
        self.font_size * CHAR_WIDTH_RATIO + self.letter_spacing
    }

    /// Compute text width for a string accounting for letter-spacing and word-spacing.
    pub fn text_width(&self, s: &str) -> f64 {
        let char_w = self.char_width();
        let base = s.len() as f64 * char_w;
        if self.word_spacing != 0.0 {
            let spaces = s.chars().filter(|c| *c == ' ').count() as f64;
            base + spaces * self.word_spacing
        } else {
            base
        }
    }

    /// Inherit CSS-inheritable properties from a parent style.
    ///
    /// Only copies properties that (a) are inheritable per CSS spec and
    /// (b) were NOT explicitly set on this node (tracked via `written`).
    pub fn inherit_from(&mut self, parent: &Style) {
        macro_rules! inh {
            ($($bit:ident => $field:ident),+ $(,)?) => {
                $( if !self.written.has($bit) { self.$field = parent.$field; } )+
            };
        }
        inh! {
            INHERIT_COLOR => color, INHERIT_FONT_SIZE => font_size,
            INHERIT_FONT_WEIGHT => font_weight, INHERIT_LINE_HEIGHT => line_height,
            INHERIT_TEXT_ALIGN => text_align, INHERIT_WHITE_SPACE => white_space,
            INHERIT_VISIBILITY => visibility, INHERIT_CURSOR => cursor,
            INHERIT_LETTER_SPACING => letter_spacing, INHERIT_WORD_SPACING => word_spacing,
            INHERIT_TEXT_TRANSFORM => text_transform, INHERIT_TEXT_INDENT => text_indent,
            INHERIT_WORD_BREAK => word_break, INHERIT_DIRECTION => direction,
        }
        // font-family inherits as Option<String> (non-Copy), handle separately
        if !self.written.has(INHERIT_FONT_FAMILY) {
            self.font_family = parent.font_family.clone();
            self.line_height_absolute = parent.line_height_absolute;
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
    FontFamily(String),
    FontSize(f64),
    FontWeight(FontWeight),
    LineHeight(f64),
    /// Absolute line-height in px (not a multiplier).
    LineHeightPx(f64),
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
            Self::FontFamily(v) => {
                s.font_family = Some(v.clone());
                s.written.set(INHERIT_FONT_FAMILY);
            }
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
                s.line_height_absolute = false;
                s.written.set(INHERIT_LINE_HEIGHT);
            }
            Self::LineHeightPx(v) => {
                s.line_height = *v;
                s.line_height_absolute = true;
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

/// Copy a single CSS property from `src` into `dst`.
///
/// Used by per-property transition blending: each transition computes its
/// own interpolated style, then only the relevant field is copied over.
pub fn copy_css_property(dst: &mut Style, src: &Style, prop: &str) {
    match prop {
        "all" => *dst = src.clone(),
        "background" | "background-color" => dst.background = src.background,
        "color" => dst.color = src.color,
        "border-color" => dst.border_color = src.border_color,
        "border-width" => dst.border_width = src.border_width,
        "border-radius" => dst.corner_radius = src.corner_radius,
        "opacity" => dst.opacity = src.opacity,
        "width" => dst.width = src.width,
        "height" => dst.height = src.height,
        "min-width" => dst.min_width = src.min_width,
        "min-height" => dst.min_height = src.min_height,
        "max-width" => dst.max_width = src.max_width,
        "max-height" => dst.max_height = src.max_height,
        "padding" => dst.padding = src.padding,
        "margin" => dst.margin = src.margin,
        "gap" => dst.gap = src.gap,
        "font-size" => dst.font_size = src.font_size,
        "font-weight" => dst.font_weight = src.font_weight,
        "line-height" => {
            dst.line_height = src.line_height;
            dst.line_height_absolute = src.line_height_absolute;
        }
        "transform" => {
            dst.transform_translate_x = src.transform_translate_x;
            dst.transform_translate_y = src.transform_translate_y;
            dst.transform_scale_x = src.transform_scale_x;
            dst.transform_scale_y = src.transform_scale_y;
            dst.transform_rotate = src.transform_rotate;
            dst.transform_skew_x = src.transform_skew_x;
            dst.transform_skew_y = src.transform_skew_y;
        }
        "filter" => {
            dst.filter_blur = src.filter_blur;
            dst.filter_brightness = src.filter_brightness;
            dst.filter_contrast = src.filter_contrast;
            dst.filter_opacity = src.filter_opacity;
        }
        "box-shadow" => dst.box_shadow = src.box_shadow,
        "text-shadow" => dst.text_shadow = src.text_shadow,
        "letter-spacing" => dst.letter_spacing = src.letter_spacing,
        "word-spacing" => dst.word_spacing = src.word_spacing,
        "text-indent" => dst.text_indent = src.text_indent,
        "outline-color" => dst.outline_color = src.outline_color,
        "outline-width" => dst.outline_width = src.outline_width,
        "left" => dst.left = src.left,
        "top" => dst.top = src.top,
        "right" => dst.right = src.right,
        "bottom" => dst.bottom = src.bottom,
        "flex-grow" => dst.flex_grow = src.flex_grow,
        "flex-shrink" => dst.flex_shrink = src.flex_shrink,
        "flex-basis" => dst.flex_basis = src.flex_basis,
        "border-top-width" => dst.border_top_width = src.border_top_width,
        "border-right-width" => dst.border_right_width = src.border_right_width,
        "border-bottom-width" => dst.border_bottom_width = src.border_bottom_width,
        "border-left-width" => dst.border_left_width = src.border_left_width,
        "row-gap" => dst.row_gap = src.row_gap,
        "column-gap" => dst.column_gap = src.column_gap,
        "z-index" => dst.z_index = src.z_index,
        "aspect-ratio" => dst.aspect_ratio = src.aspect_ratio,
        "visibility" => dst.visibility = src.visibility,
        "font-family" => dst.font_family.clone_from(&src.font_family),
        _ => {} // Unknown property — no-op
    }
}
