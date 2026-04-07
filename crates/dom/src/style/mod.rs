//! Style — all visual + layout properties for a DOM node.
//!
//! Designed as a single flat struct so the layout solver, painter, and
//! transition system can read/write fields without indirection.
//! Every spatial field is `f64` matching our [`layout`] types exactly.

use any_compute_core::Lerp;
use any_compute_core::animation::Easing;
use any_compute_core::layout::Rect;
use any_compute_core::render::Color;

// Re-export layout primitives from core — single source of truth.
pub use any_compute_core::flex::{
    Align, BoxSizing, Dimension, Direction, Display, Edges, FlexStyle, FlexWrap, Justify, Overflow,
    Position,
};

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
/// Line-through vertical offset as a fraction of font_size (≈ middle of x-height).
pub const LINE_THROUGH_RATIO: f64 = 0.3;
/// Extra pixel offset below baseline for underline decoration.
pub const UNDERLINE_OFFSET_PX: f64 = 1.0;
/// Number of characters reserved for ellipsis ("...").
pub const ELLIPSIS_CHARS: f64 = 3.0;
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
    /// Text alignment within a text node.
    TextAlign [Left, Left] { "left" | "start" => Left, "center" => Center, "right" | "end" => Right }

    /// Visibility (CSS `visibility`).
    Visibility [Visible, Visible] { "visible" => Visible, "hidden" => Hidden }

    /// White-space handling.
    WhiteSpace [Normal, Normal] { "normal" => Normal, "nowrap" => NoWrap, "pre" => Pre }

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
}

impl Lerp for FontWeight {
    fn lerp(self, other: Self, t: f64) -> Self {
        Self((self.0 as f64 + (other.0 as f64 - self.0 as f64) * t).round() as u16)
    }
}

impl Default for FontWeight {
    fn default() -> Self {
        Self::NORMAL
    }
}

// ── HTML tag classification ─────────────────────────────────────────────────
//
// Single source of truth for HTML element semantics.
// The macro generates: enum variants, from_str, kind (box/text/bar),
// and semantic queries (is_interactive, is_hidden, is_heading).
//
// Adding a new HTML tag = one line in the macro invocation.

/// Semantic category of an HTML tag used for NodeKind and default behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TagKind {
    /// Container — maps to NodeKind::Box.
    Box,
    /// Text leaf — maps to NodeKind::Text.
    Text,
    /// Progress bar — maps to NodeKind::Bar.
    Bar,
}

/// Generates the `HtmlTag` enum and its `From<&str>`, `kind()`, `is_*()` methods
/// from a compact tag table.
///
/// Syntax per entry: `VariantName "tag-string" kind [flags...]`
/// Flags: `interactive` (cursor:pointer), `hidden` (display:none), `heading`
macro_rules! html_tags {
    ($( $variant:ident $tag:literal $kind:ident $( $flag:ident )* ),* $(,)?) => {
        /// HTML element type — classifies tags for node creation and default styles.
        ///
        /// Convert any tag string via `HtmlTag::from("button")` or `"button".into()`.
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub enum HtmlTag {
            $( $variant, )*
            /// Unrecognized tags become generic containers.
            Custom,
        }

        impl From<&str> for HtmlTag {
            fn from(tag: &str) -> Self {
                match tag {
                    $( $tag => Self::$variant, )*
                    _ => Self::Custom,
                }
            }
        }

        impl HtmlTag {
            /// Tag string for this variant.
            pub fn as_str(self) -> &'static str {
                match self {
                    $( Self::$variant => $tag, )*
                    Self::Custom => "div",
                }
            }

            /// What NodeKind this tag maps to.
            pub fn kind(self) -> TagKind {
                match self {
                    $( Self::$variant => TagKind::$kind, )*
                    Self::Custom => TagKind::Box,
                }
            }

            /// Interactive elements get cursor:pointer by default.
            pub fn is_interactive(self) -> bool {
                match self {
                    $( Self::$variant => html_tags!(@has_flag interactive $( $flag )*), )*
                    Self::Custom => false,
                }
            }

            /// Hidden elements (script, style, head, etc.) get display:none.
            pub fn is_hidden(self) -> bool {
                match self {
                    $( Self::$variant => html_tags!(@has_flag hidden $( $flag )*), )*
                    Self::Custom => false,
                }
            }

            /// Heading elements (h1-h6).
            pub fn is_heading(self) -> bool {
                match self {
                    $( Self::$variant => html_tags!(@has_flag heading $( $flag )*), )*
                    Self::Custom => false,
                }
            }

            /// Inline elements (span, a, b, em, etc.) — flow horizontally in a parent.
            pub fn is_inline(self) -> bool {
                match self {
                    $( Self::$variant => html_tags!(@has_flag inline $( $flag )*), )*
                    Self::Custom => false,
                }
            }

            /// Void elements (br, input, img, hr, meta, link) — cannot have children.
            pub fn is_void(self) -> bool {
                match self {
                    $( Self::$variant => html_tags!(@has_flag void $( $flag )*), )*
                    Self::Custom => false,
                }
            }
        }
    };
    // Flag detection: returns true if the target flag is in the list.
    (@has_flag $target:ident $target2:ident $( $rest:ident )*) => {
        if stringify!($target) == stringify!($target2) { true }
        else { html_tags!(@has_flag $target $( $rest )*) }
    };
    (@has_flag $target:ident) => { false };
}

html_tags! {
    // ── Containers / structural ──
    Div        "div"        Box,
    Section    "section"    Box,
    Article    "article"    Box,
    Aside      "aside"      Box,
    Nav        "nav"        Box,
    Header     "header"     Box,
    Footer     "footer"     Box,
    Main       "main"       Box,
    Body       "body"       Box,
    // ── Interactive ──
    Button     "button"     Box     interactive,
    A          "a"          Text    interactive inline,
    // ── Form elements ──
    Form       "form"       Box,
    Input      "input"      Box     void,
    Textarea   "textarea"   Text,
    Select     "select"     Box     interactive,
    Fieldset   "fieldset"   Box,
    // ── Text / inline ──
    Span       "span"       Text    inline,
    P          "p"          Text,
    Label      "label"      Text    inline,
    Strong     "strong"     Text    inline,
    B          "b"          Text    inline,
    Em         "em"         Text    inline,
    I          "i"          Text    inline,
    Small      "small"      Text    inline,
    // ── Headings ──
    H1         "h1"         Text    heading,
    H2         "h2"         Text    heading,
    H3         "h3"         Text    heading,
    H4         "h4"         Text    heading,
    H5         "h5"         Text    heading,
    H6         "h6"         Text    heading,
    // ── Lists ──
    Ul         "ul"         Box,
    Ol         "ol"         Box,
    Li         "li"         Box,
    // ── Table ──
    Table      "table"      Box,
    Tr         "tr"         Box,
    Td         "td"         Box,
    Th         "th"         Box,
    // ── Media ──
    Img        "img"        Box     void,
    // ── Misc content ──
    Center     "center"     Box,
    Pre        "pre"        Text,
    Code       "code"       Text    inline,
    Blockquote "blockquote" Box,
    Hr         "hr"         Box     void,
    Br         "br"         Box     void,
    Details    "details"    Box,
    Summary    "summary"    Box     interactive,
    // ── Progress / bars ──
    Progress   "progress"   Bar,
    Meter      "meter"      Bar,
    // ── Hidden / metadata ──
    Script     "script"     Box     hidden,
    StyleTag   "style"      Box     hidden,
    Link       "link"       Box     hidden void,
    Meta       "meta"       Box     hidden void,
    Title      "title"      Box     hidden,
    Head       "head"       Box     hidden,
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

impl Lerp for Shadow {
    fn lerp(self, other: Self, t: f64) -> Self {
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

// ── Inheritance system ─────────────────────────────────────────────────────
//
// THE property list lives in `for_each_inheritable!` — the single source of
// truth.  It feeds the list to any `$callback!` macro, so adding a new
// inheritable property means editing exactly ONE place.

/// Single source of truth: invokes `$callback!(CONST => field, …)` for every
/// inheritable CSS property.
macro_rules! for_each_inheritable {
    ($callback:ident) => {
        $callback! {
            INHERIT_COLOR => color,
            INHERIT_FONT_SIZE => font_size,
            INHERIT_FONT_WEIGHT => font_weight,
            INHERIT_LINE_HEIGHT => line_height,
            INHERIT_TEXT_ALIGN => text_align,
            INHERIT_WHITE_SPACE => white_space,
            INHERIT_VISIBILITY => visibility,
            INHERIT_CURSOR => cursor,
            INHERIT_LETTER_SPACING => letter_spacing,
            INHERIT_WORD_SPACING => word_spacing,
            INHERIT_TEXT_TRANSFORM => text_transform,
            INHERIT_TEXT_INDENT => text_indent,
            INHERIT_WORD_BREAK => word_break,
            INHERIT_FONT_FAMILY => font_family,
        }
    };
}

/// Generates `pub const INHERIT_*: u64` bit constants and the combined `INHERIT_ALL` mask.
macro_rules! gen_inherit_consts {
    ( $( $C:ident => $f:ident ),* $(,)? ) => {
        gen_inherit_consts!(@bits 0u32; $($C => $f,)*);
        /// Mask of all inheritable properties.
        pub const INHERIT_ALL: u64 = $($C)|*;
    };
    (@bits $n:expr; $C:ident => $f:ident, $($rest:tt)*) => {
        pub const $C: u64 = 1u64 << $n;
        gen_inherit_consts!(@bits $n + 1u32; $($rest)*);
    };
    (@bits $_n:expr;) => {};
}

for_each_inheritable!(gen_inherit_consts);

// ── Style interpolation macro ─────────────────────────────────────────────
//
// Each field is tagged with its interpolation kind:
//   num      — f64 linear interpolation
//   snap     — discrete snap at t=0.5 (via Clone)
//   lerp     — delegates to Lerp::lerp (Color, Dimension, Edges, FontWeight, Shadow)
//   opt_num  — Option<f64>: lerp when both Some, snap when mixed
//   opt_int  — Option<i32>: rounded integer lerp when both Some
//   opt_lerp — Option<T: Lerp + Copy>: lerp when both Some, snap when mixed

macro_rules! style_lerp_field {
    ($self:ident, $other:ident, $t:ident; $( [$kind:ident] $field:ident ),* $(,)?) => {
        Style {
            $( $field: style_lerp_field!(@one $kind, $self.$field, $other.$field, $t), )*
            written: StyleWritten($self.written.0 | $other.written.0),
        }
    };
    (@one num, $a:expr, $b:expr, $t:expr) => { $a + ($b - $a) * $t };
    (@one snap, $a:expr, $b:expr, $t:expr) => { if $t < 0.5 { $a.clone() } else { $b.clone() } };
    (@one lerp, $a:expr, $b:expr, $t:expr) => { Lerp::lerp($a, $b, $t) };
    (@one opt_num, $a:expr, $b:expr, $t:expr) => {
        match ($a, $b) {
            (Some(x), Some(y)) => Some(x + (y - x) * $t),
            _ => if $t < 0.5 { $a } else { $b },
        }
    };
    (@one opt_int, $a:expr, $b:expr, $t:expr) => {
        match ($a, $b) {
            (Some(x), Some(y)) => Some(x + ((y - x) as f64 * $t).round() as i32),
            _ => if $t < 0.5 { $a } else { $b },
        }
    };
    (@one opt_lerp, $a:expr, $b:expr, $t:expr) => {
        match ($a, $b) {
            (Some(x), Some(y)) => Some(Lerp::lerp(x, y, $t)),
            _ => if $t < 0.5 { $a } else { $b },
        }
    };
}

// ── CSS property copy macro ───────────────────────────────────────────────
//
// Maps CSS property names to Style fields. Multi-field CSS properties (like
// "transform") naturally list multiple fields in braces.

macro_rules! css_property_copy {
    ($dst:ident, $src:ident, $prop:ident;
     $( $($css:literal)|+ => { $($field:ident),+ } ),* $(,)?
    ) => {
        match $prop {
            "all" => *$dst = $src.clone(),
            $( $($css)|+ => { $( $dst.$field = $src.$field.clone(); )+ } )*
            _ => {}
        }
    }
}

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
    pub outline_style: BorderStyle,
    pub outline_offset: f64,

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
            outline_style: BorderStyle::None,
            outline_offset: 0.0,
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
        self.written.set(INHERIT_COLOR);
        self
    }

    /// Builder: set cursor style (marks `written` so inheritance doesn't override).
    pub fn cursor(mut self, c: Cursor) -> Self {
        self.cursor = c;
        self.written.set(INHERIT_CURSOR);
        self
    }

    /// Builder: set font size.
    pub fn font(mut self, size: f64) -> Self {
        self.font_size = size;
        self.written.set(INHERIT_FONT_SIZE);
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
        self.written.set(INHERIT_FONT_WEIGHT);
        self
    }

    /// Builder: set line-height multiplier.
    pub fn lh(mut self, lh: f64) -> Self {
        self.line_height = lh;
        self.written.set(INHERIT_LINE_HEIGHT);
        self
    }

    /// Builder: set text alignment.
    pub fn text_align(mut self, a: TextAlign) -> Self {
        self.text_align = a;
        self.written.set(INHERIT_TEXT_ALIGN);
        self
    }

    /// Builder: set visibility.
    pub fn visibility(mut self, v: Visibility) -> Self {
        self.visibility = v;
        self.written.set(INHERIT_VISIBILITY);
        self
    }

    /// Builder: set width and height in pixels.
    pub fn wh(self, w: f64, h: f64) -> Self {
        self.w(w).h(h)
    }

    /// Builder: set min-width in pixels.
    pub fn min_w(mut self, px: f64) -> Self {
        self.min_width = Dimension::Px(px);
        self
    }

    /// Builder: set min-height in pixels.
    pub fn min_h(mut self, px: f64) -> Self {
        self.min_height = Dimension::Px(px);
        self
    }

    /// Builder: set align-self.
    pub fn align_self(mut self, a: Align) -> Self {
        self.align_self = Some(a);
        self
    }

    /// Builder: set flex-wrap.
    pub fn wrap(mut self, w: FlexWrap) -> Self {
        self.flex_wrap = w;
        self
    }

    // ── Interpolation ────────────────────────────────────

    /// Linearly interpolate between two styles.
    ///
    /// Numeric / color / edge fields blend smoothly.
    /// Enum / discrete fields snap at `t = 0.5`.
    /// Each field's interpolation kind is tagged in the macro invocation.
    pub fn lerp(&self, other: &Style, t: f64) -> Style {
        if t <= 0.0 {
            return self.clone();
        }
        if t >= 1.0 {
            return other.clone();
        }

        style_lerp_field!(self, other, t;
            // Display / Box Model — snap
            [snap] display, [snap] box_sizing, [snap] visibility,
            // Dimensions — variant-aware lerp
            [lerp] width, [lerp] height, [lerp] min_width, [lerp] min_height,
            [lerp] max_width, [lerp] max_height, [opt_num] aspect_ratio,
            // Flex layout — snap enums, lerp numerics
            [snap] direction, [snap] flex_wrap, [snap] align, [snap] align_self,
            [snap] justify, [num] gap, [opt_num] row_gap, [opt_num] column_gap,
            // Spacing — component-wise lerp
            [lerp] padding, [lerp] margin,
            // Position — snap enum, lerp offsets
            [snap] position, [lerp] left, [lerp] top, [lerp] right, [lerp] bottom,
            [snap] overflow,
            // Flex item — lerp numerics, snap discrete
            [num] flex_grow, [num] flex_shrink, [lerp] flex_basis, [snap] order,
            [opt_int] z_index,
            // Visual — lerp colors + numerics
            [lerp] background, [lerp] border_color, [snap] border_style,
            [num] border_width, [num] border_top_width, [num] border_right_width,
            [num] border_bottom_width, [num] border_left_width,
            [num] corner_radius, [num] opacity, [opt_lerp] box_shadow,
            [num] outline_width, [lerp] outline_color, [snap] outline_style, [num] outline_offset,
            // Transform — lerp all
            [num] transform_translate_x, [num] transform_translate_y,
            [num] transform_scale_x, [num] transform_scale_y,
            [num] transform_rotate, [num] transform_skew_x, [num] transform_skew_y,
            // Filter — lerp all
            [num] filter_blur, [num] filter_brightness,
            [num] filter_contrast, [num] filter_opacity,
            // Text — lerp numerics, snap enums
            [snap] font_family, [num] font_size, [lerp] font_weight,
            [num] line_height, [snap] line_height_absolute, [lerp] color,
            [snap] text_align, [snap] white_space,
            [snap] text_decoration, [snap] text_transform,
            [num] letter_spacing, [num] word_spacing, [num] text_indent,
            [snap] text_overflow, [opt_lerp] text_shadow, [snap] word_break,
            // Interaction — snap
            [snap] cursor, [snap] pointer_events, [snap] user_select,
        )
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

    /// Whether `other` differs from `self` in any layout-affecting property.
    /// Visual-only fields (colors, opacity, transforms, shadows, filters,
    /// cursors, decorations) are ignored.
    pub fn differs_in_layout(&self, o: &Style) -> bool {
        self.display != o.display
            || self.box_sizing != o.box_sizing
            || self.width != o.width
            || self.height != o.height
            || self.min_width != o.min_width
            || self.min_height != o.min_height
            || self.max_width != o.max_width
            || self.max_height != o.max_height
            || self.aspect_ratio != o.aspect_ratio
            || self.direction != o.direction
            || self.flex_wrap != o.flex_wrap
            || self.align != o.align
            || self.align_self != o.align_self
            || self.justify != o.justify
            || self.gap != o.gap
            || self.row_gap != o.row_gap
            || self.column_gap != o.column_gap
            || self.padding != o.padding
            || self.margin != o.margin
            || self.position != o.position
            || self.left != o.left
            || self.top != o.top
            || self.right != o.right
            || self.bottom != o.bottom
            || self.overflow != o.overflow
            || self.flex_grow != o.flex_grow
            || self.flex_shrink != o.flex_shrink
            || self.flex_basis != o.flex_basis
            || self.order != o.order
            || self.border_width != o.border_width
            || self.border_top_width != o.border_top_width
            || self.border_right_width != o.border_right_width
            || self.border_bottom_width != o.border_bottom_width
            || self.border_left_width != o.border_left_width
            || self.font_family != o.font_family
            || self.font_size != o.font_size
            || self.font_weight != o.font_weight
            || self.line_height != o.line_height
            || self.line_height_absolute != o.line_height_absolute
            || self.white_space != o.white_space
            || self.text_align != o.text_align
            || self.letter_spacing != o.letter_spacing
            || self.word_spacing != o.word_spacing
            || self.text_indent != o.text_indent
            || self.word_break != o.word_break
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
        let cx = r.origin.x + r.size.w() / 2.0;
        let cy = r.origin.y + r.size.h() / 2.0;
        let sw = r.size.w() * self.transform_scale_x;
        let sh = r.size.h() * self.transform_scale_y;
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
        c.with_alpha(a)
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
        let base = s.chars().count() as f64 * char_w;
        if self.word_spacing != 0.0 {
            let spaces = s.chars().filter(|c| *c == ' ').count() as f64;
            base + spaces * self.word_spacing
        } else {
            base
        }
    }

    /// Horizontal offset for text-align within available width.
    pub fn text_align_offset(&self, text_w: f64, avail_w: f64) -> f64 {
        match self.text_align {
            TextAlign::Center => (avail_w - text_w).max(0.0) / 2.0,
            TextAlign::Right => (avail_w - text_w).max(0.0),
            TextAlign::Left => 0.0,
        }
    }

    /// Inherit CSS-inheritable properties from a parent style.
    ///
    /// Only copies properties that (a) are inheritable per CSS spec and
    /// (b) were NOT explicitly set on this node (tracked via `written`).
    /// Property list driven by `for_each_inheritable!` — single source of truth.
    pub fn inherit_from(&mut self, parent: &Style) {
        macro_rules! do_inherit {
            ($( $C:ident => $f:ident ),* $(,)?) => {
                $( if !self.written.has($C) { self.$f = parent.$f.clone(); } )*
            };
        }
        for_each_inheritable!(do_inherit);
        // line_height_absolute piggybacks on font-family inheritance
        if !self.written.has(INHERIT_FONT_FAMILY) {
            self.line_height_absolute = parent.line_height_absolute;
        }
    }
}

// ── FlexStyle bridge ────────────────────────────────────────────────────────

impl FlexStyle for Style {
    fn display(&self) -> Display {
        self.display
    }
    fn box_sizing(&self) -> BoxSizing {
        self.box_sizing
    }
    fn direction(&self) -> Direction {
        self.direction
    }
    fn flex_wrap(&self) -> FlexWrap {
        self.flex_wrap
    }
    fn align(&self) -> Align {
        self.align
    }
    fn self_align(&self) -> Option<Align> {
        self.align_self
    }
    fn justify(&self) -> Justify {
        self.justify
    }
    fn position(&self) -> Position {
        self.position
    }
    fn overflow(&self) -> Overflow {
        self.overflow
    }
    fn width(&self) -> Dimension {
        self.width
    }
    fn height(&self) -> Dimension {
        self.height
    }
    fn min_width(&self) -> Dimension {
        self.min_width
    }
    fn min_height(&self) -> Dimension {
        self.min_height
    }
    fn max_width(&self) -> Dimension {
        self.max_width
    }
    fn max_height(&self) -> Dimension {
        self.max_height
    }
    fn aspect_ratio(&self) -> Option<f64> {
        self.aspect_ratio
    }
    fn padding(&self) -> Edges {
        self.padding
    }
    fn margin(&self) -> Edges {
        self.margin
    }
    fn border_widths(&self) -> Edges {
        self.effective_border()
    }
    fn gap(&self) -> f64 {
        self.gap
    }
    fn row_gap(&self) -> Option<f64> {
        self.row_gap
    }
    fn column_gap(&self) -> Option<f64> {
        self.column_gap
    }
    fn flex_grow(&self) -> f64 {
        self.flex_grow
    }
    fn flex_shrink(&self) -> f64 {
        self.flex_shrink
    }
    fn flex_basis(&self) -> Dimension {
        self.flex_basis
    }
    fn order(&self) -> i32 {
        self.order
    }
    fn left(&self) -> Dimension {
        self.left
    }
    fn top(&self) -> Dimension {
        self.top
    }
    fn right(&self) -> Dimension {
        self.right
    }
    fn bottom(&self) -> Dimension {
        self.bottom
    }
}

// ── Pre-compiled style operations ───────────────────────────────────────────

/// Pre-compiled style mutation — zero string matching at apply time.
///
/// Created at CSS parse time or by the Tailwind class compiler.
/// Each variant maps to one or two `Style` field writes.
/// Applying N ops is N enum matches — no string hashing or parsing.
mod ops;
pub use ops::*;
