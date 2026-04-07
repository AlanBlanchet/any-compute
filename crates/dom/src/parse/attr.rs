use crate::css::StyleSheet;
use crate::style::*;
use any_compute_core::render::Color;

pub(super) fn resolve_style(
    tag: &str,
    attrs: &[(String, String)],
    sheet: Option<&StyleSheet>,
) -> Style {
    match sheet {
        Some(sheet) => sheet.resolve(
            tag,
            &find_attr(attrs, "class").unwrap_or_default(),
            find_attr(attrs, "id").as_deref(),
            attrs,
        ),
        None => {
            let mut s = Style::default();
            apply_style_attrs(&mut s, attrs);
            s
        }
    }
}

pub(crate) fn find_attr(attrs: &[(String, String)], key: &str) -> Option<String> {
    attrs.iter().find(|(k, _)| k == key).map(|(_, v)| v.clone())
}

/// Apply attribute key/value pairs onto an existing [`Style`].
///
/// Only explicitly-present keys are modified — unmentioned fields stay put.
/// This is the shared workhorse for both HTML attribute parsing and CSS
/// class resolution.  Internally delegates to [`compile_attr`] + [`StyleOp::apply`]
/// so there is exactly one attribute → style mapping in the whole crate.
pub(crate) fn apply_style_attrs(s: &mut Style, attrs: &[(String, String)]) {
    for (key, val) in attrs {
        if key == "style" {
            // Inline CSS: `style="prop: val; prop2: val2; ..."`
            apply_inline_css(s, val);
            continue;
        }
        if let Some(op) = compile_attr(key, val) {
            op.apply(s);
        }
    }
}

/// Compile a single CSS `property: value` declaration into [`StyleOp`]s.
///
/// Handles multi-op shorthands (transform, filter, box/text-shadow),
/// longhand expansion via [`expand_css_property`], and [`compile_attr`].
/// This is the single pipeline shared by inline `style="..."` and CSS rules.
pub(crate) fn compile_declaration(prop: &str, value: &str) -> Vec<StyleOp> {
    match prop {
        "transform" => parse_transform(value),
        "filter" => parse_filter(value),
        "box-shadow" => parse_shadow(value)
            .into_iter()
            .map(StyleOp::BoxShadow)
            .collect(),
        "text-shadow" => parse_shadow(value)
            .into_iter()
            .map(StyleOp::TextShadow)
            .collect(),
        _ => crate::css::expand_css_property(prop, value)
            .into_iter()
            .filter_map(|(k, v)| compile_attr(&k, &v))
            .collect(),
    }
}

/// CSS spec: `display:flex` implies `flex-direction:row` by default.
/// Our ua.css defaults block elements to column, so inject Row when flex is
/// requested without an explicit direction.
pub(crate) fn inject_flex_row(ops: &mut Vec<StyleOp>) {
    let has_flex = ops
        .iter()
        .any(|op| matches!(op, StyleOp::Display(crate::style::Display::Flex)));
    let has_dir = ops.iter().any(|op| matches!(op, StyleOp::Direction(_)));
    if has_flex && !has_dir {
        ops.push(StyleOp::Direction(crate::style::Direction::Row));
    }
}

/// Parse and apply an inline CSS `style="..."` attribute value.
fn apply_inline_css(s: &mut Style, css_text: &str) {
    let mut ops = Vec::new();
    for decl in css_text.split(';') {
        let decl = decl.trim();
        if decl.is_empty() {
            continue;
        }
        let Some((prop, value)) = decl.split_once(':') else {
            continue;
        };
        let prop = prop.trim().to_ascii_lowercase();
        ops.extend(compile_declaration(&prop, value.trim()));
    }
    inject_flex_row(&mut ops);
    for op in ops {
        op.apply(s);
    }
}

/// Compile a single attribute key/value pair into a pre-resolved [`StyleOp`].
///
/// Returns `None` for unrecognized keys or unparseable values.
/// This is the single source of truth for the `attr name → Style field` mapping —
/// both the HTML parser and the CSS engine delegate here.
pub(crate) fn compile_attr(key: &str, val: &str) -> Option<StyleOp> {
    use StyleOp::*;
    match key {
        // ── Display / Box Model ─────────────────────────
        "display" => Some(Display(crate::style::Display::from_css(val))),
        "box-sizing" => Some(BoxSizing(crate::style::BoxSizing::from_css(val))),
        "visibility" => Some(Visibility(crate::style::Visibility::from_css(val))),

        // ── Dimensions ──────────────────────────────────
        "w" | "width" => parse_dimension(val).map(Width),
        "h" | "height" => parse_dimension(val).map(Height),
        "min-w" | "min-width" => parse_dimension(val).map(MinWidth),
        "min-h" | "min-height" => parse_dimension(val).map(MinHeight),
        "max-w" | "max-width" => parse_dimension(val).map(MaxWidth),
        "max-h" | "max-height" => parse_dimension(val).map(MaxHeight),
        "aspect-ratio" => parse_aspect_ratio(val).map(AspectRatio),

        // ── Flex layout ─────────────────────────────────
        "direction" | "dir" | "flex-direction" => {
            Some(Direction(crate::style::Direction::from_css(val)))
        }
        "flex-wrap" => Some(FlexWrap(crate::style::FlexWrap::from_css(val))),
        "align" | "align-items" => Some(Align(crate::style::Align::from_css(val))),
        "align-self" => Some(AlignSelf(crate::style::Align::from_css(val))),
        "justify" | "justify-content" => Some(Justify(crate::style::Justify::from_css(val))),
        "gap" => parse_px(val).map(Gap),
        "row-gap" => parse_px(val).map(RowGap),
        "column-gap" => parse_px(val).map(ColumnGap),

        // ── Spacing ─────────────────────────────────────
        "pad" | "padding" => parse_px(val).map(|v| Padding(Edges::all(v))),
        "pad-x" | "padding-x" => parse_px(val).map(PaddingX),
        "pad-y" | "padding-y" => parse_px(val).map(PaddingY),
        "padding-top" => parse_px(val).map(PaddingTop),
        "padding-right" => parse_px(val).map(PaddingRight),
        "padding-bottom" => parse_px(val).map(PaddingBottom),
        "padding-left" => parse_px(val).map(PaddingLeft),
        "margin" => parse_px(val).map(|v| Margin(Edges::all(v))),
        "margin-x" => parse_px(val).map(MarginX),
        "margin-y" => parse_px(val).map(MarginY),
        "margin-top" => parse_px(val).map(MarginTop),
        "margin-right" => parse_px(val).map(MarginRight),
        "margin-bottom" => parse_px(val).map(MarginBottom),
        "margin-left" => parse_px(val).map(MarginLeft),

        // ── Position ────────────────────────────────────
        "position" => Some(Position(crate::style::Position::from_css(val))),
        "left" => parse_dimension(val).map(Left),
        "top" => parse_dimension(val).map(Top),
        "right" => parse_dimension(val).map(Right),
        "bottom" => parse_dimension(val).map(Bottom),
        "z-index" => val.trim().parse::<i32>().ok().map(ZIndex),

        // ── Flex item ───────────────────────────────────
        "grow" | "flex-grow" => val.parse().ok().map(FlexGrow),
        "shrink" | "flex-shrink" => val.parse().ok().map(FlexShrink),
        "flex-basis" => parse_dimension(val).map(FlexBasis),
        "order" => val.trim().parse::<i32>().ok().map(Order),

        // ── Overflow ────────────────────────────────────
        "overflow" => Some(Overflow(crate::style::Overflow::from_css(val))),

        // ── Visual ──────────────────────────────────────
        "bg" | "background" | "background-color" => parse_color(val).map(Background),
        "border-color" => parse_color(val).map(BorderColor),
        "border-style" => Some(BorderStyleOp(crate::style::BorderStyle::from_css(val))),
        "border-width" => parse_px(val).map(BorderWidth),
        "border-top-width" => parse_px(val).map(BorderTopWidth),
        "border-right-width" => parse_px(val).map(BorderRightWidth),
        "border-bottom-width" => parse_px(val).map(BorderBottomWidth),
        "border-left-width" => parse_px(val).map(BorderLeftWidth),
        "radius" | "border-radius" => parse_px(val).map(CornerRadius),
        "opacity" => val.parse().ok().map(Opacity),
        "box-shadow" => parse_shadow(val).map(BoxShadow),
        "outline-width" => parse_px(val).map(OutlineWidth),
        "outline-color" => parse_color(val).map(OutlineColor),

        // ── Transform (individual properties) ───────────
        "translate-x" => parse_px(val).map(TranslateX),
        "translate-y" => parse_px(val).map(TranslateY),
        "scale-x" => val.parse().ok().map(ScaleX),
        "scale-y" => val.parse().ok().map(ScaleY),
        "rotate" => parse_angle(val).map(Rotate),
        "skew-x" => parse_angle(val).map(SkewX),
        "skew-y" => parse_angle(val).map(SkewY),

        // ── Filter (individual properties) ──────────────
        "filter-blur" => parse_px(val).map(FilterBlur),
        "filter-brightness" => val.parse().ok().map(FilterBrightness),
        "filter-contrast" => val.parse().ok().map(FilterContrast),
        "filter-opacity" => val.parse().ok().map(FilterOpacity),

        // ── Text ────────────────────────────────────────
        "font-family" => {
            let family = val
                .trim()
                .trim_matches(|c| c == '\'' || c == '"')
                .to_string();
            Some(FontFamily(family))
        }
        "font" | "font-size" => parse_px(val).map(FontSize),
        "font-weight" => crate::style::FontWeight::from_css(val).map(FontWeight),
        "line-height" => parse_line_height(val),
        "color" => parse_color(val).map(TextColor),
        "text-align" => Some(TextAlign(crate::style::TextAlign::from_css(val))),
        "white-space" => Some(WhiteSpace(crate::style::WhiteSpace::from_css(val))),
        "text-decoration" => Some(TextDecorationOp(crate::style::TextDecoration::from_css(
            val,
        ))),
        "text-transform" => Some(TextTransformOp(crate::style::TextTransform::from_css(val))),
        "letter-spacing" => parse_px(val).map(LetterSpacing),
        "word-spacing" => parse_px(val).map(WordSpacing),
        "text-indent" => parse_px(val).map(TextIndent),
        "text-overflow" => Some(TextOverflowOp(crate::style::TextOverflow::from_css(val))),
        "text-shadow" => parse_shadow(val).map(TextShadow),
        "word-break" | "overflow-wrap" => Some(WordBreakOp(crate::style::WordBreak::from_css(val))),

        // ── Interaction ─────────────────────────────────
        "cursor" => Some(CursorOp(crate::style::Cursor::from_css(val))),
        "pointer-events" => Some(PointerEventsOp(crate::style::PointerEvents::from_css(val))),
        "user-select" => Some(UserSelectOp(crate::style::UserSelect::from_css(val))),

        // Non-style attributes (class, id, tag, value, text) are skipped.
        _ => None,
    }
}

/// Parse line-height: bare number (multiplier) or px value.
fn parse_line_height(val: &str) -> Option<StyleOp> {
    let val = val.trim();
    // "normal" → use default multiplier
    if val == "normal" {
        return Some(StyleOp::LineHeight(DEFAULT_LINE_HEIGHT));
    }
    // Absolute: px/rem/em → store as LineHeightPx
    if val.ends_with("px") || val.ends_with("rem") || val.ends_with("em") {
        return parse_px(val).map(StyleOp::LineHeightPx);
    }
    // Percentage: e.g. "150%" → 1.5 multiplier
    if let Some(pct) = val.strip_suffix('%') {
        return pct
            .trim()
            .parse::<f64>()
            .ok()
            .map(|v| StyleOp::LineHeight(v / 100.0));
    }
    // Bare number = multiplier
    val.parse::<f64>().ok().map(StyleOp::LineHeight)
}

// ── Value parsers ───────────────────────────────────────────────────────────

use crate::style::REM_PX;

/// Strip a CSS length unit and convert to f64 pixels.
///
/// Handles `rem` (× [`REM_PX`]), `em` (× `REM_PX`), `px`, and bare numbers.
/// Single source of truth for unit conversion — every length parser delegates here.
pub(crate) fn parse_px(val: &str) -> Option<f64> {
    let val = val.trim();
    if let Some(num) = val.strip_suffix("rem") {
        return num.trim().parse::<f64>().ok().map(|v| v * REM_PX);
    }
    if let Some(num) = val.strip_suffix("em") {
        return num.trim().parse::<f64>().ok().map(|v| v * REM_PX);
    }
    // CSS pt → px: 1pt = 4/3 px
    if let Some(num) = val.strip_suffix("pt") {
        return num.trim().parse::<f64>().ok().map(|v| v * (4.0 / 3.0));
    }
    val.strip_suffix("px").unwrap_or(val).trim().parse().ok()
}

pub(crate) fn parse_dimension(val: &str) -> Option<Dimension> {
    let val = val.trim();
    if val == "auto" {
        return Some(Dimension::Auto);
    }
    // calc() expressions: `calc(100% - 20px)` → Dimension::Calc { percent, px }
    if let Some(inner) = val.strip_prefix("calc(").and_then(|v| v.strip_suffix(')')) {
        return parse_calc_expr(inner);
    }
    if let Some(pct) = val.strip_suffix('%') {
        return pct.trim().parse::<f64>().ok().map(Dimension::Percent);
    }
    parse_px(val).map(Dimension::Px)
}

/// Parse a `calc()` expression body into `Dimension::Calc { percent, px }`.
///
/// Handles: `100% - 20px`, `50% + 1rem`, `100% - 2 * 16px`, etc.
/// Splits on `+` / `-` tokens and accumulates percent and px components.
fn parse_calc_expr(expr: &str) -> Option<Dimension> {
    let expr = expr.trim();
    let mut percent = 0.0_f64;
    let mut px = 0.0_f64;
    let mut sign = 1.0_f64;

    // Tokenize around + and - while respecting whitespace-delimited operators.
    // CSS calc requires spaces around + and - (to distinguish from negative numbers).
    let mut i = 0;
    let bytes = expr.as_bytes();
    let len = bytes.len();

    while i < len {
        // Skip whitespace
        while i < len && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= len {
            break;
        }

        // Check for operator
        if bytes[i] == b'+' {
            sign = 1.0;
            i += 1;
            continue;
        }
        if bytes[i] == b'-' {
            sign = -1.0;
            i += 1;
            continue;
        }

        // Read a term (number + optional unit)
        let start = i;
        // consume digits, dots, minus (for negative numbers like -20px)
        while i < len && !bytes[i].is_ascii_whitespace() && bytes[i] != b'+' {
            // Stop at a space-preceded + or - (operator boundary)
            if bytes[i] == b'-' && i > start && bytes[i - 1].is_ascii_whitespace() {
                break;
            }
            i += 1;
        }
        let term = expr[start..i].trim();
        if term.is_empty() {
            continue;
        }

        if let Some(pct_val) = term.strip_suffix('%') {
            if let Ok(v) = pct_val.trim().parse::<f64>() {
                percent += sign * v;
            }
        } else if let Some(v) = parse_px(term) {
            px += sign * v;
        }
        // Reset sign to positive for the next term (explicit operator will override)
        sign = 1.0;
    }

    // If we got only one component, return the simpler variant
    if percent == 0.0 && px != 0.0 {
        Some(Dimension::Px(px))
    } else if percent != 0.0 && px == 0.0 {
        Some(Dimension::Percent(percent))
    } else if percent != 0.0 || px != 0.0 {
        Some(Dimension::Calc { percent, px })
    } else {
        None
    }
}

/// Parse a CSS time value to seconds: `300ms` → 0.3, `1.5s` → 1.5.
pub(crate) fn parse_time(val: &str) -> Option<f64> {
    let val = val.trim();
    if let Some(ms) = val.strip_suffix("ms") {
        return ms.trim().parse::<f64>().ok().map(|v| v / 1000.0);
    }
    if let Some(s) = val.strip_suffix('s') {
        return s.trim().parse().ok();
    }
    // Bare number — assume seconds (CSS spec compliant)
    val.parse().ok()
}

/// Parse an angle value (CSS `deg`, `rad`, `turn`, or bare number = degrees).
pub(crate) fn parse_angle(val: &str) -> Option<f64> {
    let val = val.trim();
    if let Some(deg) = val.strip_suffix("deg") {
        return deg.trim().parse().ok();
    }
    if let Some(rad) = val.strip_suffix("rad") {
        return rad
            .trim()
            .parse::<f64>()
            .ok()
            .map(|r| r * 180.0 / std::f64::consts::PI);
    }
    if let Some(turn) = val.strip_suffix("turn") {
        return turn.trim().parse::<f64>().ok().map(|t| t * 360.0);
    }
    // Bare number = degrees
    val.parse().ok()
}

/// Parse CSS `aspect-ratio`: `16 / 9`, `16/9`, or bare number.
pub(crate) fn parse_aspect_ratio(val: &str) -> Option<f64> {
    let val = val.trim();
    if val == "auto" {
        return None;
    }
    if let Some((num, den)) = val.split_once('/') {
        let n: f64 = num.trim().parse().ok()?;
        let d: f64 = den.trim().parse().ok()?;
        if d != 0.0 { Some(n / d) } else { None }
    } else {
        val.parse().ok()
    }
}

/// Parse CSS `box-shadow` / `text-shadow`: `[inset] x y [blur] [spread] [color]`.
pub(crate) fn parse_shadow(val: &str) -> Option<crate::style::Shadow> {
    let val = val.trim();
    if val == "none" {
        return None;
    }
    let inset = val.contains("inset");
    let cleaned = val.replace("inset", "");
    let parts: Vec<&str> = cleaned.split_whitespace().collect();
    if parts.len() < 2 {
        return None;
    }
    let mut nums = Vec::new();
    let mut color = None;
    for part in &parts {
        if let Some(px) = parse_px(part) {
            nums.push(px);
        } else if let Some(c) = parse_color(part) {
            color = Some(c);
        }
    }
    if nums.len() < 2 {
        return None;
    }
    Some(crate::style::Shadow {
        x: nums[0],
        y: nums[1],
        blur: nums.get(2).copied().unwrap_or(0.0),
        spread: nums.get(3).copied().unwrap_or(0.0),
        color: color.unwrap_or(Color::BLACK),
        inset,
    })
}

/// Iterate over CSS function calls in a value string, yielding `(name, args)` pairs.
/// E.g. `"blur(4px) brightness(1.2)"` yields `[("blur","4px"), ("brightness","1.2")]`.
fn scan_css_functions(val: &str, mut cb: impl FnMut(&str, &str)) {
    let bytes = val.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= bytes.len() {
            break;
        }
        let fn_start = i;
        while i < bytes.len() && bytes[i] != b'(' {
            i += 1;
        }
        if i >= bytes.len() {
            break;
        }
        let name = val[fn_start..i].trim();
        i += 1; // skip '('
        let arg_start = i;
        while i < bytes.len() && bytes[i] != b')' {
            i += 1;
        }
        if i >= bytes.len() {
            break;
        }
        let args = val[arg_start..i].trim();
        i += 1; // skip ')'
        cb(name, args);
    }
}

/// Parse CSS `transform` shorthand into individual ops.
pub(crate) fn parse_transform(val: &str) -> Vec<StyleOp> {
    use StyleOp::*;
    let mut ops = Vec::new();
    scan_css_functions(val.trim(), |name, args| {
        let parts: Vec<&str> = args.split(',').map(|s| s.trim()).collect();
        match name {
            "translateX" => {
                if let Some(v) = parse_px(parts[0]) {
                    ops.push(TranslateX(v));
                }
            }
            "translateY" => {
                if let Some(v) = parse_px(parts[0]) {
                    ops.push(TranslateY(v));
                }
            }
            "translate" => {
                if let Some(x) = parts.first().and_then(|s| parse_px(s)) {
                    ops.push(TranslateX(x));
                }
                if let Some(y) = parts.get(1).and_then(|s| parse_px(s)) {
                    ops.push(TranslateY(y));
                }
            }
            "scaleX" => {
                if let Ok(v) = parts[0].parse::<f64>() {
                    ops.push(ScaleX(v));
                }
            }
            "scaleY" => {
                if let Ok(v) = parts[0].parse::<f64>() {
                    ops.push(ScaleY(v));
                }
            }
            "scale" => {
                if let Ok(x) = parts[0].parse::<f64>() {
                    ops.push(ScaleX(x));
                    let y = parts
                        .get(1)
                        .and_then(|s| s.parse::<f64>().ok())
                        .unwrap_or(x);
                    ops.push(ScaleY(y));
                }
            }
            "rotate" => {
                if let Some(v) = parse_angle(parts[0]) {
                    ops.push(Rotate(v));
                }
            }
            "skewX" => {
                if let Some(v) = parse_angle(parts[0]) {
                    ops.push(SkewX(v));
                }
            }
            "skewY" => {
                if let Some(v) = parse_angle(parts[0]) {
                    ops.push(SkewY(v));
                }
            }
            "skew" => {
                if let Some(x) = parts.first().and_then(|s| parse_angle(s)) {
                    ops.push(SkewX(x));
                }
                if let Some(y) = parts.get(1).and_then(|s| parse_angle(s)) {
                    ops.push(SkewY(y));
                }
            }
            _ => {} // matrix, perspective, etc. — silently ignored
        }
    });
    ops
}

/// Parse a percentage-or-number argument from a CSS filter function.
fn parse_filter_amount(arg: &str) -> Option<f64> {
    arg.strip_suffix('%')
        .and_then(|s| s.parse::<f64>().ok())
        .map(|v| v / 100.0)
        .or_else(|| arg.parse().ok())
}

/// Parse CSS `filter` shorthand into individual ops.
pub(crate) fn parse_filter(val: &str) -> Vec<StyleOp> {
    use StyleOp::*;
    let mut ops = Vec::new();
    let val = val.trim();
    if val == "none" {
        return ops;
    }
    scan_css_functions(val, |name, arg| {
        match name {
            "blur" => {
                if let Some(v) = parse_px(arg) {
                    ops.push(FilterBlur(v));
                }
            }
            "brightness" => {
                if let Some(v) = parse_filter_amount(arg) {
                    ops.push(FilterBrightness(v));
                }
            }
            "contrast" => {
                if let Some(v) = parse_filter_amount(arg) {
                    ops.push(FilterContrast(v));
                }
            }
            "opacity" => {
                if let Some(v) = parse_filter_amount(arg) {
                    ops.push(FilterOpacity(v));
                }
            }
            _ => {} // saturate, grayscale, sepia, hue-rotate, invert, drop-shadow — silently ignored
        }
    });
    ops
}

/// Parse `#rrggbb`, `#rgb`, `#rrggbbaa`, `rgb(r,g,b)`, `rgba(r,g,b,a)`, or named colors.
pub(crate) fn parse_color(val: &str) -> Option<Color> {
    let val = val.trim();

    // Hex colors.
    if let Some(hex) = val.strip_prefix('#') {
        return parse_hex_color(hex);
    }

    // rgb(r,g,b) / rgba(r,g,b,a)
    if let Some(inner) = val
        .strip_prefix("rgb(")
        .or_else(|| val.strip_prefix("rgba("))
    {
        let inner = inner.strip_suffix(')')?.trim();
        // Modern CSS also supports space-separated `rgb(r g b / a)` syntax.
        let parts: Vec<&str> = if inner.contains(',') {
            inner.split(',').map(|s| s.trim()).collect()
        } else {
            // `rgb(r g b / a)` or `rgb(r g b)`
            let (rgb_part, alpha_part) = if let Some(idx) = inner.find('/') {
                (&inner[..idx], Some(inner[idx + 1..].trim()))
            } else {
                (inner, None)
            };
            let mut v: Vec<&str> = rgb_part.split_whitespace().collect();
            if let Some(a) = alpha_part {
                v.push(a);
            }
            v
        };
        return match parts.len() {
            3 => Some(Color::rgb(
                parts[0].parse().ok()?,
                parts[1].parse().ok()?,
                parts[2].parse().ok()?,
            )),
            4 => {
                let r: u8 = parts[0].parse().ok()?;
                let g: u8 = parts[1].parse().ok()?;
                let b: u8 = parts[2].parse().ok()?;
                // CSS spec: alpha is 0.0-1.0 (float) or 0%-100%.
                // Legacy engines also accept 0-255 integer.
                let a_str = parts[3].trim();
                let a: u8 = if a_str.contains('.') {
                    // Float 0.0-1.0 → map to 0-255
                    let f: f64 = a_str.parse().ok()?;
                    (f.clamp(0.0, 1.0) * 255.0).round() as u8
                } else if let Some(pct) = a_str.strip_suffix('%') {
                    let f: f64 = pct.trim().parse().ok()?;
                    (f.clamp(0.0, 100.0) / 100.0 * 255.0).round() as u8
                } else {
                    // Integer: if > 1 treat as 0-255 directly (legacy),
                    // if 0 or 1 treat as literal (CSS clamps > 1 to 1).
                    let v: f64 = a_str.parse().ok()?;
                    if v > 1.0 {
                        (v.clamp(0.0, 255.0)).round() as u8
                    } else {
                        (v.clamp(0.0, 1.0) * 255.0).round() as u8
                    }
                };
                Some(Color::rgba(r, g, b, a))
            }
            _ => None,
        };
    }

    // Named colors (small common set).
    match val {
        "white" => Some(Color::WHITE),
        "black" => Some(Color::BLACK),
        "transparent" => Some(Color::TRANSPARENT),
        "red" => Some(Color::rgb(255, 0, 0)),
        "green" => Some(Color::rgb(0, 128, 0)),
        "blue" => Some(Color::rgb(0, 0, 255)),
        "yellow" => Some(Color::rgb(255, 255, 0)),
        "cyan" => Some(Color::rgb(0, 255, 255)),
        "magenta" => Some(Color::rgb(255, 0, 255)),
        "gray" | "grey" => Some(Color::rgb(128, 128, 128)),
        _ => None,
    }
}

fn parse_hex_color(hex: &str) -> Option<Color> {
    let hex = hex.trim();
    match hex.len() {
        3 => {
            // #rgb → #rrggbb
            let r = u8::from_str_radix(&hex[0..1], 16).ok()? * 17;
            let g = u8::from_str_radix(&hex[1..2], 16).ok()? * 17;
            let b = u8::from_str_radix(&hex[2..3], 16).ok()? * 17;
            Some(Color::rgb(r, g, b))
        }
        6 => {
            let r = u8::from_str_radix(&hex[0..2], 16).ok()?;
            let g = u8::from_str_radix(&hex[2..4], 16).ok()?;
            let b = u8::from_str_radix(&hex[4..6], 16).ok()?;
            Some(Color::rgb(r, g, b))
        }
        8 => {
            let r = u8::from_str_radix(&hex[0..2], 16).ok()?;
            let g = u8::from_str_radix(&hex[2..4], 16).ok()?;
            let b = u8::from_str_radix(&hex[4..6], 16).ok()?;
            let a = u8::from_str_radix(&hex[6..8], 16).ok()?;
            Some(Color::rgba(r, g, b, a))
        }
        _ => None,
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────
