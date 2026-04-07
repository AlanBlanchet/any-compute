use super::*;
use any_compute_core::render::Color;

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
    OutlineStyleOp(BorderStyle),
    OutlineOffset(f64),

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
            Self::Direction(d) => s.direction = *d,
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
            Self::OutlineStyleOp(v) => s.outline_style = *v,
            Self::OutlineOffset(v) => s.outline_offset = *v,
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

    /// Returns `true` when this op changes a property that affects layout
    /// (box model, flex, position, text metrics). Visual-only props (colors,
    /// opacity, transforms, filters, shadows, decorations, cursors) return `false`.
    #[inline]
    pub fn affects_layout(&self) -> bool {
        !matches!(
            self,
            Self::Background(_)
                | Self::BorderColor(_)
                | Self::BorderStyleOp(_)
                | Self::CornerRadius(_)
                | Self::Opacity(_)
                | Self::BoxShadow(_)
                | Self::OutlineWidth(_)
                | Self::OutlineColor(_)
                | Self::OutlineStyleOp(_)
                | Self::OutlineOffset(_)
                | Self::TranslateX(_)
                | Self::TranslateY(_)
                | Self::ScaleX(_)
                | Self::ScaleY(_)
                | Self::Rotate(_)
                | Self::SkewX(_)
                | Self::SkewY(_)
                | Self::FilterBlur(_)
                | Self::FilterBrightness(_)
                | Self::FilterContrast(_)
                | Self::FilterOpacity(_)
                | Self::TextColor(_)
                | Self::TextDecorationOp(_)
                | Self::TextTransformOp(_)
                | Self::TextOverflowOp(_)
                | Self::TextShadow(_)
                | Self::CursorOp(_)
                | Self::PointerEventsOp(_)
                | Self::UserSelectOp(_)
                | Self::Visibility(_)
                | Self::ZIndex(_)
        )
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
/// Multi-field CSS properties list all their fields in braces.
pub fn copy_css_property(dst: &mut Style, src: &Style, prop: &str) {
    css_property_copy!(dst, src, prop;
        "background" | "background-color" => { background },
        "color" => { color },
        "border-color" => { border_color },
        "border-width" => { border_width },
        "border-radius" => { corner_radius },
        "opacity" => { opacity },
        "width" => { width },
        "height" => { height },
        "min-width" => { min_width },
        "min-height" => { min_height },
        "max-width" => { max_width },
        "max-height" => { max_height },
        "padding" => { padding },
        "margin" => { margin },
        "gap" => { gap },
        "font-size" => { font_size },
        "font-weight" => { font_weight },
        "line-height" => { line_height, line_height_absolute },
        "transform" => {
            transform_translate_x, transform_translate_y,
            transform_scale_x, transform_scale_y,
            transform_rotate, transform_skew_x, transform_skew_y
        },
        "filter" => { filter_blur, filter_brightness, filter_contrast, filter_opacity },
        "box-shadow" => { box_shadow },
        "text-shadow" => { text_shadow },
        "letter-spacing" => { letter_spacing },
        "word-spacing" => { word_spacing },
        "text-indent" => { text_indent },
        "outline-color" => { outline_color },
        "outline-width" => { outline_width },
        "outline-style" => { outline_style },
        "outline-offset" => { outline_offset },
        "left" => { left },
        "top" => { top },
        "right" => { right },
        "bottom" => { bottom },
        "flex-grow" => { flex_grow },
        "flex-shrink" => { flex_shrink },
        "flex-basis" => { flex_basis },
        "border-top-width" => { border_top_width },
        "border-right-width" => { border_right_width },
        "border-bottom-width" => { border_bottom_width },
        "border-left-width" => { border_left_width },
        "row-gap" => { row_gap },
        "column-gap" => { column_gap },
        "z-index" => { z_index },
        "aspect-ratio" => { aspect_ratio },
        "visibility" => { visibility },
        "font-family" => { font_family },
    );
}
