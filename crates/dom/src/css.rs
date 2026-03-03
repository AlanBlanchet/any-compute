//! CSS parser → [`StyleSheet`] — converts CSS text into pre-baked [`Style`] lookups.
//!
//! Zero external dependencies.  Parses a purposefully small CSS subset that
//! maps 1:1 onto our [`Style`] system.  The stylesheet is resolved once at
//! parse time; runtime lookups are O(1) HashMap gets + cheap Style construction.
//!
//! ## Supported selectors
//!
//! | Selector     | Example        | Stored in       |
//! |-------------|----------------|-----------------|
//! | Class       | `.card`        | `classes`       |
//! | Tag         | `div`          | `tags`          |
//! | Id          | `#main`        | `ids`           |
//! | Comma group | `.a, .b`       | both separately |
//!
//! ## Supported properties
//!
//! Every CSS property that has a corresponding [`Style`] field is supported.
//! Standard CSS names are normalized to our attribute names automatically:
//! `background-color` → `bg`, `flex-direction` → `direction`, etc.
//!
//! Shorthand properties `padding` and `margin` with 1–4 values are expanded
//! to individual sides.
//!
//! ## Usage
//!
//! ```
//! use any_compute_dom::css::StyleSheet;
//! let sheet = StyleSheet::parse(".title { font-size: 22px; color: #cdd2f4; }");
//! let style = sheet.class("title");
//! assert_eq!(style.font_size, 22.0);
//! ```

use std::collections::HashMap;

use crate::parse::{apply_style_attrs, compile_attr, parse_px};
use crate::style::{Style, StyleOp, apply_ops};

// ── StyleSheet ──────────────────────────────────────────────────────────────

/// Pre-parsed CSS stylesheet.
///
/// Class / tag / id rules are baked to attribute pair lists.  Looking up a
/// class is one HashMap probe + a sequential attribute application — zero
/// string parsing at resolve time beyond `f64::parse` and hex→u8.
pub struct StyleSheet {
    classes: HashMap<String, Vec<StyleOp>>,
    tags: HashMap<String, Vec<StyleOp>>,
    ids: HashMap<String, Vec<StyleOp>>,
}

/// Browser-like user-agent defaults for HTML tags.
///
/// Applied as the lowest-specificity base before any CSS classes or IDs.
/// Mimics the browser's built-in stylesheet so tags like `<p>`, `<h1>`, etc.
/// look correct without requiring any external CSS.
const UA_CSS: &str = r#"
/* Block elements: column direction */
div, section, article, aside, nav, header, footer, main, form, fieldset, details, summary {
    flex-direction: column;
}

/* Headings */
h1 { font-size: 32px; font-weight: bold; margin-top: 21.44px; margin-bottom: 21.44px; }
h2 { font-size: 24px; font-weight: bold; margin-top: 19.92px; margin-bottom: 19.92px; }
h3 { font-size: 18.72px; font-weight: bold; margin-top: 18.72px; margin-bottom: 18.72px; }
h4 { font-size: 16px; font-weight: bold; margin-top: 21.28px; margin-bottom: 21.28px; }
h5 { font-size: 13.28px; font-weight: bold; margin-top: 22.18px; margin-bottom: 22.18px; }
h6 { font-size: 10.72px; font-weight: bold; margin-top: 24.98px; margin-bottom: 24.98px; }

/* Paragraph */
p { margin-top: 16px; margin-bottom: 16px; }

/* Body */
body { margin: 8px; }

/* Lists */
ul, ol { margin-top: 16px; margin-bottom: 16px; padding-left: 40px; }
li { flex-direction: row; }

/* Code / pre */
pre, code { font-size: 13px; }
pre { margin-top: 16px; margin-bottom: 16px; overflow: scroll; }

/* HR */
hr { height: 1px; margin-top: 8px; margin-bottom: 8px; }

/* Table */
table { flex-direction: column; }
tr { flex-direction: row; }
td, th { padding: 4px; }
th { font-weight: bold; }

/* Strong / Bold */
strong, b { font-weight: bold; }

/* Small */
small { font-size: 13px; }

/* Blockquote */
blockquote { margin-top: 16px; margin-bottom: 16px; margin-left: 40px; margin-right: 40px; }
"#;

impl StyleSheet {
    /// Parse a CSS string into a `StyleSheet`.
    ///
    /// Fault-tolerant: malformed rules are silently skipped.  Never panics.
    pub fn parse(css: &str) -> Self {
        let rules = parse_rules(css);
        let mut classes: HashMap<String, Vec<StyleOp>> = HashMap::new();
        let mut tags: HashMap<String, Vec<StyleOp>> = HashMap::new();
        let mut ids: HashMap<String, Vec<StyleOp>> = HashMap::new();

        for (selector, decls) in rules {
            let sel = selector.trim();
            if sel.is_empty() {
                continue;
            }
            if let Some(name) = sel.strip_prefix('.') {
                classes.entry(name.to_string()).or_default().extend(decls);
            } else if let Some(name) = sel.strip_prefix('#') {
                ids.entry(name.to_string()).or_default().extend(decls);
            } else if let Some(dot) = sel.find('.') {
                // tag.class → store under class (tag+class compounds not needed yet)
                classes
                    .entry(sel[dot + 1..].to_string())
                    .or_default()
                    .extend(decls);
            } else {
                tags.entry(sel.to_string()).or_default().extend(decls);
            }
        }

        Self { classes, tags, ids }
    }

    /// Parse CSS with UA (user-agent) defaults baked in.
    ///
    /// UA styles are the lowest specificity — any class, id, or even tag
    /// rule from `css` will override them.
    pub fn parse_with_ua(css: &str) -> Self {
        let combined = format!("{UA_CSS}\n{css}");
        Self::parse(&combined)
    }

    /// Look up a name in a map and apply its ops onto a default [`Style`].
    fn lookup(map: &HashMap<String, Vec<StyleOp>>, name: &str) -> Style {
        let mut s = Style::default();
        if let Some(ops) = map.get(name) {
            apply_ops(&mut s, ops);
        }
        s
    }

    /// Resolve a single class name into a [`Style`].
    /// Unknown class → `Style::default()`.
    pub fn class(&self, name: &str) -> Style {
        Self::lookup(&self.classes, name)
    }

    /// Resolve multiple class names, merging in order (later overrides earlier).
    pub fn classes(&self, names: &[&str]) -> Style {
        let mut s = Style::default();
        for name in names {
            if let Some(ops) = self.classes.get(*name) {
                apply_ops(&mut s, ops);
            }
        }
        s
    }

    /// Apply a class's declarations on top of an existing style.
    pub fn apply(&self, style: &mut Style, name: &str) {
        if let Some(ops) = self.classes.get(name) {
            apply_ops(style, ops);
        }
    }

    /// Resolve a tag selector.
    pub fn tag(&self, name: &str) -> Style {
        Self::lookup(&self.tags, name)
    }

    /// Resolve an id selector.
    pub fn id(&self, name: &str) -> Style {
        Self::lookup(&self.ids, name)
    }

    /// Full cascade: tag < class(es) < id < inline attrs.
    /// Used by `parse_with_css` internally.
    pub fn resolve(
        &self,
        tag: &str,
        class_list: &str,
        id: Option<&str>,
        inline: &[(String, String)],
    ) -> Style {
        let mut s = Style::default();
        // 1. Tag
        if let Some(ops) = self.tags.get(tag) {
            apply_ops(&mut s, ops);
        }
        // 2. Classes (in order)
        for cls in class_list.split_ascii_whitespace() {
            if let Some(ops) = self.classes.get(cls) {
                apply_ops(&mut s, ops);
            }
        }
        // 3. Id
        if let Some(id) = id {
            if let Some(ops) = self.ids.get(id) {
                apply_ops(&mut s, ops);
            }
        }
        // 4. Inline attributes (highest specificity)
        apply_style_attrs(&mut s, inline);
        s
    }
}

// ── CSS tokenizer ───────────────────────────────────────────────────────────

fn parse_rules(css: &str) -> Vec<(String, Vec<StyleOp>)> {
    let cleaned = strip_comments(css);
    let bytes = cleaned.as_bytes();
    let mut rules = Vec::new();
    let mut i = 0;

    while i < bytes.len() {
        // Skip whitespace
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= bytes.len() {
            break;
        }

        // Read selector (everything before '{')
        while i < bytes.len() && bytes[i] != b'{' {
            i += 1;
        }
        if i >= bytes.len() {
            break; // no opening brace found — skip trailing text
        }
        let selector = cleaned[..i]
            .rsplit('}')
            .next()
            .unwrap_or("")
            .trim()
            .to_string();
        i += 1; // skip '{'

        // Read declarations (between '{' and '}')
        let decl_start = i;
        let mut depth = 1u32;
        while i < bytes.len() && depth > 0 {
            match bytes[i] {
                b'{' => depth += 1,
                b'}' => depth -= 1,
                _ => {}
            }
            if depth > 0 {
                i += 1;
            }
        }
        if depth != 0 {
            break; // unclosed brace — stop parsing, keep rules so far
        }
        let body = &cleaned[decl_start..i];
        i += 1; // skip '}'

        let decls = compile_declarations(body);

        // Handle comma-separated selectors
        for sel in selector.split(',') {
            let sel = sel.trim();
            if !sel.is_empty() {
                rules.push((sel.to_string(), decls.clone()));
            }
        }
    }

    rules
}

fn strip_comments(css: &str) -> String {
    let mut out = String::with_capacity(css.len());
    let bytes = css.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if i + 1 < bytes.len() && bytes[i] == b'/' && bytes[i + 1] == b'*' {
            i += 2;
            while i + 1 < bytes.len() && !(bytes[i] == b'*' && bytes[i + 1] == b'/') {
                i += 1;
            }
            if i + 1 < bytes.len() {
                i += 2; // skip '*/'
            }
        } else {
            out.push(bytes[i] as char);
            i += 1;
        }
    }
    out
}

/// Parse declarations and compile each to pre-resolved [`StyleOp`]s.
/// CSS shorthand expansion happens here; resolve-time is pure enum-match.
fn compile_declarations(body: &str) -> Vec<StyleOp> {
    let mut ops = Vec::new();
    for decl in body.split(';') {
        let decl = decl.trim();
        if decl.is_empty() {
            continue;
        }
        let Some((prop, value)) = decl.split_once(':') else {
            continue;
        };
        let prop = prop.trim().to_ascii_lowercase();
        let value = value.trim();
        for (key, val) in expand_css_property(&prop, value) {
            if let Some(op) = compile_attr(&key, &val) {
                ops.push(op);
            }
        }
    }
    ops
}

// ── Property normalization + shorthand expansion ────────────────────────────

/// Normalize a CSS length value to a bare-number string in px.
/// Delegates to [`parse_px`] — the single unit-conversion source of truth.
fn norm_val(v: &str) -> String {
    parse_px(v).map(|n| n.to_string()).unwrap_or_else(|| v.to_string())
}

/// Map CSS property names to our attribute names and expand shorthands.
fn expand_css_property(prop: &str, value: &str) -> Vec<(String, String)> {
    match prop {
        // ── Shorthands with 1–4 values ──
        "padding" | "margin" => expand_box_shorthand(prop, value),

        // ── Border shorthand: `border: 1px solid #color` ──
        "border" => expand_border(value),

        // ── Name aliases: standard CSS → our attr names ──
        "background-color" | "background" => vec![("bg".into(), value.into())],
        "flex-direction" => vec![("direction".into(), value.into())],
        "align-items" => vec![("align".into(), value.into())],
        "align-self" => vec![("align-self".into(), value.into())],
        "justify-content" => vec![("justify".into(), value.into())],
        "border-radius" => vec![("radius".into(), norm_val(value))],

        // ── Flex shorthand: `flex: N` → flex-grow: N ──
        "flex" => {
            let parts: Vec<&str> = value.split_whitespace().collect();
            match parts.len() {
                1 => vec![("flex-grow".into(), parts[0].into())],
                n if n >= 2 => vec![
                    ("flex-grow".into(), parts[0].into()),
                    ("flex-shrink".into(), parts[1].into()),
                ],
                _ => vec![],
            }
        }

        // ── Individual sides (already named correctly) ──
        "padding-top" | "padding-right" | "padding-bottom" | "padding-left" | "margin-top"
        | "margin-right" | "margin-bottom" | "margin-left" => {
            vec![(prop.into(), norm_val(value))]
        }

        // ── Dimension properties: normalize px ──
        "width" | "height" | "min-width" | "min-height" | "max-width" | "max-height" | "gap"
        | "row-gap" | "column-gap" | "border-width" | "border-top-width"
        | "border-right-width" | "border-bottom-width" | "border-left-width" | "left" | "top"
        | "right" | "bottom" | "flex-basis" => {
            vec![(prop.into(), norm_val(value))]
        }

        // ── Font-size: normalize px ──
        "font-size" => vec![(prop.into(), norm_val(value))],

        // ── New properties (pass through directly) ──
        "display" | "box-sizing" | "visibility" | "flex-wrap" | "font-weight" | "line-height"
        | "text-align" | "white-space" | "z-index" => vec![(prop.into(), value.into())],

        // ── Everything else passes through unchanged ──
        _ => vec![(prop.into(), value.into())],
    }
}

/// Expand CSS `border` shorthand: `1px solid #color` → width + style(ignored) + color.
fn expand_border(value: &str) -> Vec<(String, String)> {
    let parts: Vec<&str> = value.split_whitespace().collect();
    let mut result = Vec::new();
    for part in &parts {
        if let Some(px) = crate::parse::parse_px(part) {
            result.push(("border-width".into(), px.to_string()));
        } else if part.starts_with('#') || part.starts_with("rgb") || crate::parse::parse_color(part).is_some() {
            result.push(("border-color".into(), (*part).into()));
        }
        // "solid", "dashed", etc. are silently ignored (we only render solid)
    }
    result
}

/// Expand `padding`/`margin` shorthand with 1–4 space-separated values.
fn expand_box_shorthand(prop: &str, value: &str) -> Vec<(String, String)> {
    let parts: Vec<String> = value.split_whitespace().map(|p| norm_val(p)).collect();
    match parts.len() {
        1 => vec![(prop.into(), parts[0].clone())],
        2 => {
            // V H → top/bottom=V, left/right=H
            vec![
                (format!("{prop}-top"), parts[0].clone()),
                (format!("{prop}-right"), parts[1].clone()),
                (format!("{prop}-bottom"), parts[0].clone()),
                (format!("{prop}-left"), parts[1].clone()),
            ]
        }
        3 => {
            // T H B
            vec![
                (format!("{prop}-top"), parts[0].clone()),
                (format!("{prop}-right"), parts[1].clone()),
                (format!("{prop}-bottom"), parts[2].clone()),
                (format!("{prop}-left"), parts[1].clone()),
            ]
        }
        4 => {
            // T R B L
            vec![
                (format!("{prop}-top"), parts[0].clone()),
                (format!("{prop}-right"), parts[1].clone()),
                (format!("{prop}-bottom"), parts[2].clone()),
                (format!("{prop}-left"), parts[3].clone()),
            ]
        }
        _ => vec![],
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use any_compute_core::layout::Size;
    use any_compute_core::render::Color;

    use crate::style::*;

    #[test]
    fn basic_class() {
        let sheet = StyleSheet::parse(".title { font-size: 22px; color: #cdd2f4; }");
        let s = sheet.class("title");
        assert_eq!(s.font_size, 22.0);
        assert_eq!(s.color, Color::rgb(205, 210, 244));
    }

    #[test]
    fn unknown_class_returns_default() {
        let sheet = StyleSheet::parse(".a { gap: 8; }");
        assert_eq!(sheet.class("nonexistent"), Style::default());
    }

    #[test]
    fn multi_property_rule() {
        let css = r#"
            .card {
                flex-grow: 1;
                background: #313244;
                border-radius: 12px;
                padding: 16px;
                gap: 6;
            }
        "#;
        let s = StyleSheet::parse(css).class("card");
        assert_eq!(s.flex_grow, 1.0);
        assert_eq!(s.background, Color::rgb(49, 50, 68));
        assert_eq!(s.corner_radius, 12.0);
        assert_eq!(s.padding, Edges::all(16.0));
        assert_eq!(s.gap, 6.0);
    }

    #[test]
    fn comma_selectors() {
        let sheet = StyleSheet::parse(".a, .b { gap: 5; }");
        assert_eq!(sheet.class("a").gap, 5.0);
        assert_eq!(sheet.class("b").gap, 5.0);
    }

    #[test]
    fn comments_stripped() {
        let css = "/* header */ .x { /* inside */ font-size: 18; }";
        let s = StyleSheet::parse(css).class("x");
        assert_eq!(s.font_size, 18.0);
    }

    #[test]
    fn multi_class_merge() {
        let css = ".base { gap: 8; font-size: 14; } .override { font-size: 22; }";
        let s = StyleSheet::parse(css).classes(&["base", "override"]);
        assert_eq!(s.gap, 8.0); // from .base
        assert_eq!(s.font_size, 22.0); // overridden
    }

    #[test]
    fn shorthand_padding_two_values() {
        let css = ".x { padding: 16px 12px; }";
        let s = StyleSheet::parse(css).class("x");
        assert_eq!(s.padding.top, 16.0);
        assert_eq!(s.padding.right, 12.0);
        assert_eq!(s.padding.bottom, 16.0);
        assert_eq!(s.padding.left, 12.0);
    }

    #[test]
    fn shorthand_padding_four_values() {
        let css = ".x { padding: 1 2 3 4; }";
        let s = StyleSheet::parse(css).class("x");
        assert_eq!(s.padding.top, 1.0);
        assert_eq!(s.padding.right, 2.0);
        assert_eq!(s.padding.bottom, 3.0);
        assert_eq!(s.padding.left, 4.0);
    }

    #[test]
    fn shorthand_margin_two_values() {
        let css = ".x { margin: 10px 20px; }";
        let s = StyleSheet::parse(css).class("x");
        assert_eq!(s.margin.top, 10.0);
        assert_eq!(s.margin.right, 20.0);
        assert_eq!(s.margin.bottom, 10.0);
        assert_eq!(s.margin.left, 20.0);
    }

    #[test]
    fn css_name_normalization() {
        let css = r#"
            .x {
                background-color: #ff0000;
                flex-direction: row;
                align-items: center;
                justify-content: space-between;
            }
        "#;
        let s = StyleSheet::parse(css).class("x");
        assert_eq!(s.background, Color::rgb(255, 0, 0));
        assert_eq!(s.direction, Direction::Row);
        assert_eq!(s.align, Align::Center);
        assert_eq!(s.justify, Justify::SpaceBetween);
    }

    #[test]
    fn tag_selector() {
        let sheet = StyleSheet::parse("div { gap: 4; }");
        assert_eq!(sheet.tag("div").gap, 4.0);
    }

    #[test]
    fn id_selector() {
        let sheet = StyleSheet::parse("#main { width: 800; }");
        let s = sheet.id("main");
        assert_eq!(s.width, Dimension::Px(800.0));
    }

    #[test]
    fn full_cascade_resolve() {
        let css = r#"
            div { font-size: 10; }
            .big { font-size: 20; }
            #special { color: #ff0000; }
        "#;
        let sheet = StyleSheet::parse(css);
        let s = sheet.resolve("div", "big", Some("special"), &[("gap".into(), "5".into())]);
        assert_eq!(s.font_size, 20.0); // class overrides tag
        assert_eq!(s.color, Color::rgb(255, 0, 0)); // id
        assert_eq!(s.gap, 5.0); // inline
    }

    #[test]
    fn apply_on_existing_style() {
        let css = ".accent { color: #a6e3a1; }";
        let sheet = StyleSheet::parse(css);
        let mut s = Style::default().font(16.0);
        sheet.apply(&mut s, "accent");
        assert_eq!(s.font_size, 16.0); // untouched
        assert_eq!(s.color, Color::rgb(166, 227, 161)); // applied
    }

    #[test]
    fn flex_shorthand() {
        let css = ".x { flex: 2; }";
        let s = StyleSheet::parse(css).class("x");
        assert_eq!(s.flex_grow, 2.0);
    }

    #[test]
    fn integration_parse_with_css() {
        let css = ".container { width: 400; height: 300; gap: 8; }";
        let html = r#"<div class="container"><span>Hi</span></div>"#;
        let sheet = StyleSheet::parse(css);
        let mut tree = crate::parse::parse_with_css(html, &sheet);
        tree.layout(Size::new(400.0, 300.0));
        assert_eq!(tree.arena[0].style.width, Dimension::Px(400.0));
        assert_eq!(tree.arena[0].style.gap, 8.0);
    }

    #[test]
    fn css_plus_inline_override() {
        let css = ".base { width: 100; height: 50; }";
        let html = r#"<div class="base" w="200"></div>"#;
        let sheet = StyleSheet::parse(css);
        let tree = crate::parse::parse_with_css(html, &sheet);
        // Inline w="200" overrides CSS width: 100
        assert_eq!(tree.arena[0].style.width, Dimension::Px(200.0));
        assert_eq!(tree.arena[0].style.height, Dimension::Px(50.0));
    }

    // ── Fault-tolerance ──────────────────────────────────────────────

    #[test]
    fn garbage_css_does_not_crash() {
        let sh = StyleSheet::parse("{{{{ not css }} color: ;; }}}");
        assert_eq!(sh.class("nonexistent"), Style::default());
    }

    #[test]
    fn bad_values_silently_ignored() {
        let sh = StyleSheet::parse(".x { font-size: banana; color: nope; width: zzz; gap: ; }");
        let s = sh.class("x");
        assert_eq!(s.font_size, 14.0); // default
        assert_eq!(s.color, Color::WHITE); // default
        assert_eq!(s.width, Dimension::Auto); // default
        assert_eq!(s.gap, 0.0); // default
    }

    // ── Properties not covered elsewhere ─────────────────────────────

    #[test]
    fn transparent_background() {
        let s = StyleSheet::parse(".x { background: transparent; }").class("x");
        assert_eq!(s.background, Color::TRANSPARENT);
    }

    #[test]
    fn overflow_scroll() {
        let s = StyleSheet::parse(".x { overflow: scroll; }").class("x");
        assert_eq!(s.overflow, Overflow::Scroll);
    }

    #[test]
    fn direction_column() {
        let s = StyleSheet::parse(".x { flex-direction: column; }").class("x");
        assert_eq!(s.direction, Direction::Column);
    }

    #[test]
    fn dimension_width_height() {
        let s = StyleSheet::parse(".x { width: 220; height: 56; }").class("x");
        assert_eq!(s.width, Dimension::Px(220.0));
        assert_eq!(s.height, Dimension::Px(56.0));
    }

    #[test]
    fn corner_radius_from_css() {
        let s = StyleSheet::parse(".x { border-radius: 8px; }").class("x");
        assert_eq!(s.corner_radius, 8.0);
    }

    // ── Pixel-level CSS visual correctness ──────────────────────────

    use crate::tree::Tree;
    use any_compute_core::render::{PixelBuffer, RenderList};

    /// Build a single-node tree from CSS, layout, paint, and rasterize to pixels.
    fn css_to_pixels(css: &str, class: &str, vw: f64, vh: f64) -> PixelBuffer {
        let sheet = StyleSheet::parse(css);
        let s = sheet.class(class);
        let mut tree = Tree::new(s.w(vw).h(vh));
        tree.layout(Size::new(vw, vh));
        let mut list = RenderList::default();
        tree.paint(&mut list);
        let mut buf = PixelBuffer::new(vw as u32, vh as u32, Color::BLACK);
        buf.paint(&list);
        buf
    }

    #[test]
    fn pixel_css_border_radius_clips_corner() {
        // CSS border-radius must visibly clip the top-left corner pixel.
        let buf = css_to_pixels(
            ".box { background: #ffffff; border-radius: 20px; }",
            "box",
            100.0,
            100.0,
        );
        // Center is white (filled).
        assert_eq!(buf.pixel(50, 50), Color::WHITE);
        // Top-left corner is still the clear color (outside radius curve).
        assert_eq!(buf.pixel(0, 0), Color::BLACK);
        // Top-center is filled (well inside radius).
        assert_eq!(buf.pixel(50, 0), Color::WHITE);
    }

    #[test]
    fn pixel_css_background_color_exact() {
        // CSS hex color must produce the exact RGB pixel values.
        let buf = css_to_pixels(".x { background: #a6e3a1; }", "x", 40.0, 40.0);
        assert_eq!(buf.pixel(20, 20), Color::rgb(166, 227, 161));
    }

    #[test]
    fn pixel_css_transparent_background_untouched() {
        // `background: transparent` should leave pixels at the clear color.
        let buf = css_to_pixels(
            ".x { background: transparent; width: 50; height: 50; }",
            "x",
            50.0,
            50.0,
        );
        assert_eq!(buf.pixel(25, 25), Color::BLACK); // clear color
    }

    #[test]
    fn pixel_css_card_rounded_corners() {
        // Realistic card component: all four corners must be clipped.
        let buf = css_to_pixels(
            ".card { background: #313244; border-radius: 12px; }",
            "card",
            200.0,
            120.0,
        );
        let fill = Color::rgb(49, 50, 68);
        // Center is fill.
        assert_eq!(buf.pixel(100, 60), fill);
        // All corners are clipped.
        assert_eq!(buf.pixel(0, 0), Color::BLACK);
        assert_eq!(buf.pixel(199, 0), Color::BLACK);
        assert_eq!(buf.pixel(0, 119), Color::BLACK);
        assert_eq!(buf.pixel(199, 119), Color::BLACK);
    }

    #[test]
    fn pixel_css_nested_layout_paint() {
        // Build a parent + child via the full CSS + HTML pipeline, then rasterize.
        let css = r#"
            .parent { background: #1e1e2e; width: 200; height: 100; }
            .child  { background: #89b4fa; width: 80; height: 40; }
        "#;
        let html = r#"<div class="parent"><div class="child"></div></div>"#;
        let sheet = StyleSheet::parse(css);
        let mut tree = crate::parse::parse_with_css(html, &sheet);
        tree.layout(Size::new(200.0, 100.0));
        let mut list = RenderList::default();
        tree.paint(&mut list);
        let mut buf = PixelBuffer::new(200, 100, Color::BLACK);
        buf.paint(&list);
        let parent_bg = Color::rgb(30, 30, 46);
        let child_bg = Color::rgb(137, 180, 250);
        // Child occupies top-left 80×40 inside parent.
        assert_eq!(buf.pixel(10, 10), child_bg);
        // Parent area outside child.
        assert_eq!(buf.pixel(150, 80), parent_bg);
    }

    // ── Tailwind CSS visual comparison tests ────────────────────────────
    //
    // These parse real compiled Tailwind CSS output (tailwind.css) through
    // our StyleSheet::parse() engine, then verify both computed Style values
    // and rendered pixel output match expected results.

    /// Lazily parse the bundled Tailwind CSS subset — reused across tests.
    fn tailwind() -> StyleSheet {
        StyleSheet::parse(include_str!("tailwind.css"))
    }

    // ── Property correctness (Style assertions) ─────────────────────

    #[test]
    fn tw_padding_rem_values() {
        let tw = tailwind();
        // .p-4 { padding: 1rem; } → 16px all sides
        let s = tw.class("p-4");
        assert_eq!(s.padding, Edges::all(16.0));
        // .p-2 { padding: 0.5rem; } → 8px
        assert_eq!(tw.class("p-2").padding, Edges::all(8.0));
        // .p-8 { padding: 2rem; } → 32px
        assert_eq!(tw.class("p-8").padding, Edges::all(32.0));
    }

    #[test]
    fn tw_padding_px_axis() {
        let tw = tailwind();
        // .px-4 → padding-left: 1rem; padding-right: 1rem;
        let s = tw.class("px-4");
        assert_eq!(s.padding.left, 16.0);
        assert_eq!(s.padding.right, 16.0);
        // .py-2 → padding-top: 0.5rem; padding-bottom: 0.5rem;
        let s = tw.class("py-2");
        assert_eq!(s.padding.top, 8.0);
        assert_eq!(s.padding.bottom, 8.0);
    }

    #[test]
    fn tw_padding_individual_sides() {
        let tw = tailwind();
        assert_eq!(tw.class("pt-4").padding.top, 16.0);
        assert_eq!(tw.class("pr-4").padding.right, 16.0);
        assert_eq!(tw.class("pb-4").padding.bottom, 16.0);
        assert_eq!(tw.class("pl-4").padding.left, 16.0);
    }

    #[test]
    fn tw_margin_rem_values() {
        let tw = tailwind();
        assert_eq!(tw.class("m-4").margin, Edges::all(16.0));
        assert_eq!(tw.class("m-2").margin, Edges::all(8.0));
        // .mx-4 → margin-left: 1rem; margin-right: 1rem;
        let s = tw.class("mx-4");
        assert_eq!(s.margin.left, 16.0);
        assert_eq!(s.margin.right, 16.0);
    }

    #[test]
    fn tw_gap_scale() {
        let tw = tailwind();
        assert_eq!(tw.class("gap-0").gap, 0.0);
        assert_eq!(tw.class("gap-1").gap, 4.0); // 0.25rem
        assert_eq!(tw.class("gap-2").gap, 8.0); // 0.5rem
        assert_eq!(tw.class("gap-4").gap, 16.0); // 1rem
        assert_eq!(tw.class("gap-8").gap, 32.0); // 2rem
    }

    #[test]
    fn tw_width_height_rem() {
        let tw = tailwind();
        assert_eq!(tw.class("w-4").width, Dimension::Px(16.0));
        assert_eq!(tw.class("w-8").width, Dimension::Px(32.0));
        assert_eq!(tw.class("w-64").width, Dimension::Px(256.0));
        assert_eq!(tw.class("h-16").height, Dimension::Px(64.0));
        assert_eq!(tw.class("h-full").height, Dimension::Percent(100.0));
        assert_eq!(tw.class("w-full").width, Dimension::Percent(100.0));
        assert_eq!(tw.class("w-1\\/2").width, Dimension::Percent(50.0));
    }

    #[test]
    fn tw_flex_direction() {
        let tw = tailwind();
        assert_eq!(tw.class("flex-row").direction, Direction::Row);
        assert_eq!(tw.class("flex-col").direction, Direction::Column);
    }

    #[test]
    fn tw_alignment() {
        let tw = tailwind();
        assert_eq!(tw.class("items-center").align, Align::Center);
        assert_eq!(tw.class("items-end").align, Align::End);
        assert_eq!(tw.class("items-stretch").align, Align::Stretch);
        assert_eq!(tw.class("justify-center").justify, Justify::Center);
        assert_eq!(
            tw.class("justify-between").justify,
            Justify::SpaceBetween
        );
        assert_eq!(
            tw.class("justify-evenly").justify,
            Justify::SpaceEvenly
        );
    }

    #[test]
    fn tw_flex_grow_shrink() {
        let tw = tailwind();
        assert_eq!(tw.class("grow").flex_grow, 1.0);
        assert_eq!(tw.class("grow-0").flex_grow, 0.0);
        assert_eq!(tw.class("shrink").flex_shrink, 1.0);
        assert_eq!(tw.class("shrink-0").flex_shrink, 0.0);
    }

    #[test]
    fn tw_border_radius() {
        let tw = tailwind();
        assert_eq!(tw.class("rounded-none").corner_radius, 0.0);
        assert_eq!(tw.class("rounded-sm").corner_radius, 2.0); // 0.125rem
        assert_eq!(tw.class("rounded").corner_radius, 4.0); // 0.25rem
        assert_eq!(tw.class("rounded-lg").corner_radius, 8.0); // 0.5rem
        assert_eq!(tw.class("rounded-full").corner_radius, 9999.0);
    }

    #[test]
    fn tw_opacity() {
        let tw = tailwind();
        assert_eq!(tw.class("opacity-0").opacity, 0.0);
        assert_eq!(tw.class("opacity-50").opacity, 0.5);
        assert_eq!(tw.class("opacity-100").opacity, 1.0);
    }

    #[test]
    fn tw_font_size() {
        let tw = tailwind();
        assert_eq!(tw.class("text-xs").font_size, 12.0); // 0.75rem
        assert_eq!(tw.class("text-sm").font_size, 14.0); // 0.875rem
        assert_eq!(tw.class("text-base").font_size, 16.0); // 1rem
        assert_eq!(tw.class("text-lg").font_size, 18.0); // 1.125rem
        assert_eq!(tw.class("text-xl").font_size, 20.0); // 1.25rem
        assert_eq!(tw.class("text-2xl").font_size, 24.0); // 1.5rem
    }

    #[test]
    fn tw_background_colors() {
        let tw = tailwind();
        assert_eq!(tw.class("bg-white").background, Color::WHITE);
        assert_eq!(tw.class("bg-black").background, Color::BLACK);
        assert_eq!(
            tw.class("bg-red-500").background,
            Color::rgb(239, 68, 68)
        );
        assert_eq!(
            tw.class("bg-blue-500").background,
            Color::rgb(59, 130, 246)
        );
        assert_eq!(
            tw.class("bg-green-500").background,
            Color::rgb(34, 197, 94)
        );
        assert_eq!(
            tw.class("bg-slate-900").background,
            Color::rgb(15, 23, 42)
        );
    }

    #[test]
    fn tw_text_colors() {
        let tw = tailwind();
        assert_eq!(tw.class("text-white").color, Color::WHITE);
        assert_eq!(tw.class("text-black").color, Color::BLACK);
        assert_eq!(
            tw.class("text-red-500").color,
            Color::rgb(239, 68, 68)
        );
        assert_eq!(
            tw.class("text-gray-400").color,
            Color::rgb(156, 163, 175)
        );
    }

    #[test]
    fn tw_border_width_and_color() {
        let tw = tailwind();
        assert_eq!(tw.class("border").border_width, 1.0);
        assert_eq!(tw.class("border-2").border_width, 2.0);
        assert_eq!(tw.class("border-4").border_width, 4.0);
        assert_eq!(
            tw.class("border-red-500").border_color,
            Color::rgb(239, 68, 68)
        );
    }

    #[test]
    fn tw_position_overflow() {
        let tw = tailwind();
        assert_eq!(tw.class("relative").position, Position::Relative);
        assert_eq!(tw.class("absolute").position, Position::Absolute);
        assert_eq!(tw.class("overflow-hidden").overflow, Overflow::Hidden);
        assert_eq!(tw.class("overflow-scroll").overflow, Overflow::Scroll);
    }

    #[test]
    fn tw_multi_class_composition() {
        let tw = tailwind();
        // Simulating: class="flex-row items-center gap-4 p-4 bg-slate-800 rounded-lg"
        let s = tw.classes(&[
            "flex-row",
            "items-center",
            "gap-4",
            "p-4",
            "bg-slate-800",
            "rounded-lg",
        ]);
        assert_eq!(s.direction, Direction::Row);
        assert_eq!(s.align, Align::Center);
        assert_eq!(s.gap, 16.0);
        assert_eq!(s.padding, Edges::all(16.0));
        assert_eq!(s.background, Color::rgb(30, 41, 59));
        assert_eq!(s.corner_radius, 8.0);
    }

    // ── Pixel-level Tailwind visual tests ───────────────────────────

    /// Build a tree from Tailwind-styled HTML, layout, paint, and rasterize.
    fn tw_to_pixels(html: &str, vw: f64, vh: f64) -> PixelBuffer {
        let tw = tailwind();
        let mut tree = crate::parse::parse_with_css(html, &tw);
        tree.layout(Size::new(vw, vh));
        let mut list = RenderList::default();
        tree.paint(&mut list);
        let mut buf = PixelBuffer::new(vw as u32, vh as u32, Color::BLACK);
        buf.paint(&list);
        buf
    }

    #[test]
    fn tw_pixel_card_bg_matches() {
        // Tailwind bg-blue-500 must produce exact pixel color.
        let buf = tw_to_pixels(
            r#"<div class="bg-blue-500 w-64 h-32 rounded-lg"></div>"#,
            256.0,
            128.0,
        );
        let blue500 = Color::rgb(59, 130, 246);
        assert_eq!(buf.pixel(128, 64), blue500);
        // Corner is clipped by rounded-lg (8px radius).
        assert_eq!(buf.pixel(0, 0), Color::BLACK);
    }

    #[test]
    fn tw_pixel_nested_layout() {
        // Parent: dark bg, column layout with padding.
        // Child: blue card inside.
        let html = r#"
            <div class="bg-slate-900 w-96 h-48 p-4 flex-col">
                <div class="bg-blue-500 w-full h-16 rounded"></div>
            </div>
        "#;
        let buf = tw_to_pixels(html, 384.0, 192.0);
        let parent_bg = Color::rgb(15, 23, 42);
        let child_bg = Color::rgb(59, 130, 246);
        // Padding area (top-left corner) is parent bg.
        assert_eq!(buf.pixel(8, 8), parent_bg);
        // Child inside padding at (16, 16).
        assert_eq!(buf.pixel(24, 24), child_bg);
        // Far bottom-right is parent bg (below child).
        assert_eq!(buf.pixel(200, 170), parent_bg);
    }

    #[test]
    fn tw_pixel_diff_identical_renders() {
        // Two identical renders should produce zero diff.
        let html = r#"<div class="bg-red-500 w-32 h-32 rounded-full"></div>"#;
        let a = tw_to_pixels(html, 128.0, 128.0);
        let b = tw_to_pixels(html, 128.0, 128.0);
        assert_eq!(a.diff(&b, 0), 0);
        assert_eq!(a.diff_ratio(&b, 0), 0.0);
    }

    #[test]
    fn tw_pixel_diff_different_colors() {
        // Different backgrounds → many differing pixels.
        let a = tw_to_pixels(
            r#"<div class="bg-red-500 w-32 h-32"></div>"#,
            128.0,
            128.0,
        );
        let b = tw_to_pixels(
            r#"<div class="bg-blue-500 w-32 h-32"></div>"#,
            128.0,
            128.0,
        );
        let ratio = a.diff_ratio(&b, 0);
        // Most pixels differ (entire fill area).
        assert!(ratio > 0.9, "diff ratio should be >90%, got {ratio}");
    }

    #[test]
    fn tw_pixel_radius_visual_equivalence() {
        // Tailwind rounded-2xl (1rem = 16px) should match our raw CSS with radius: 16px.
        let tw_buf = tw_to_pixels(
            r#"<div class="bg-white w-48 h-48 rounded-2xl"></div>"#,
            192.0,
            192.0,
        );
        let raw_buf = css_to_pixels(
            ".box { background: #ffffff; border-radius: 16px; }",
            "box",
            192.0,
            192.0,
        );
        // Zero pixel difference — our Tailwind CSS parse and raw CSS produce identical output.
        assert_eq!(tw_buf.diff(&raw_buf, 0), 0);
    }

    #[test]
    fn tw_full_color_palette_parse_count() {
        // Verify the Tailwind CSS file parsed all expected classes.
        let tw = tailwind();
        // Spot-check: every Tailwind color family (22 backgrounds × 11 shades = 242+).
        // We check a handful to confirm they're all present.
        for name in &[
            "bg-slate-500",
            "bg-gray-700",
            "bg-zinc-900",
            "bg-red-300",
            "bg-orange-400",
            "bg-amber-600",
            "bg-yellow-200",
            "bg-lime-500",
            "bg-green-700",
            "bg-emerald-400",
            "bg-teal-600",
            "bg-cyan-300",
            "bg-sky-500",
            "bg-blue-800",
            "bg-indigo-400",
            "bg-violet-600",
            "bg-purple-300",
            "bg-fuchsia-500",
            "bg-pink-700",
            "bg-rose-400",
        ] {
            let s = tw.class(name);
            assert_ne!(
                s.background,
                Color::TRANSPARENT,
                "Tailwind class '{name}' should set a non-transparent background"
            );
        }
    }
}
