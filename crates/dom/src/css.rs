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

use crate::parse::apply_style_attrs;
use crate::style::Style;

// ── StyleSheet ──────────────────────────────────────────────────────────────

/// Pre-parsed CSS stylesheet.
///
/// Class / tag / id rules are baked to attribute pair lists.  Looking up a
/// class is one HashMap probe + a sequential attribute application — zero
/// string parsing at resolve time beyond `f64::parse` and hex→u8.
pub struct StyleSheet {
    classes: HashMap<String, Vec<(String, String)>>,
    tags: HashMap<String, Vec<(String, String)>>,
    ids: HashMap<String, Vec<(String, String)>>,
}

impl StyleSheet {
    /// Parse a CSS string into a `StyleSheet`.
    ///
    /// Fault-tolerant: malformed rules are silently skipped.  Never panics.
    pub fn parse(css: &str) -> Self {
        let rules = parse_rules(css);
        let mut classes: HashMap<String, Vec<(String, String)>> = HashMap::new();
        let mut tags: HashMap<String, Vec<(String, String)>> = HashMap::new();
        let mut ids: HashMap<String, Vec<(String, String)>> = HashMap::new();

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

    /// Resolve a single class name into a [`Style`].
    /// Unknown class → `Style::default()`.
    pub fn class(&self, name: &str) -> Style {
        let mut s = Style::default();
        if let Some(attrs) = self.classes.get(name) {
            apply_style_attrs(&mut s, attrs);
        }
        s
    }

    /// Resolve multiple class names, merging in order (later overrides earlier).
    pub fn classes(&self, names: &[&str]) -> Style {
        let mut s = Style::default();
        for name in names {
            if let Some(attrs) = self.classes.get(*name) {
                apply_style_attrs(&mut s, attrs);
            }
        }
        s
    }

    /// Apply a class's declarations on top of an existing style.
    pub fn apply(&self, style: &mut Style, name: &str) {
        if let Some(attrs) = self.classes.get(name) {
            apply_style_attrs(style, attrs);
        }
    }

    /// Resolve a tag selector.
    pub fn tag(&self, name: &str) -> Style {
        let mut s = Style::default();
        if let Some(attrs) = self.tags.get(name) {
            apply_style_attrs(&mut s, attrs);
        }
        s
    }

    /// Resolve an id selector.
    pub fn id(&self, name: &str) -> Style {
        let mut s = Style::default();
        if let Some(attrs) = self.ids.get(name) {
            apply_style_attrs(&mut s, attrs);
        }
        s
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
        if let Some(attrs) = self.tags.get(tag) {
            apply_style_attrs(&mut s, attrs);
        }
        // 2. Classes (in order)
        for cls in class_list.split_ascii_whitespace() {
            if let Some(attrs) = self.classes.get(cls) {
                apply_style_attrs(&mut s, attrs);
            }
        }
        // 3. Id
        if let Some(id) = id {
            if let Some(attrs) = self.ids.get(id) {
                apply_style_attrs(&mut s, attrs);
            }
        }
        // 4. Inline attributes (highest specificity)
        apply_style_attrs(&mut s, inline);
        s
    }
}

// ── CSS tokenizer ───────────────────────────────────────────────────────────

fn parse_rules(css: &str) -> Vec<(String, Vec<(String, String)>)> {
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

        let decls = parse_declarations(body);

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

/// Parse "prop: value; prop: value;" into normalized attribute pairs.
fn parse_declarations(body: &str) -> Vec<(String, String)> {
    let mut attrs = Vec::new();
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
        attrs.extend(expand_css_property(&prop, value));
    }
    attrs
}

// ── Property normalization + shorthand expansion ────────────────────────────

/// Normalize CSS px values — strip `px` suffix since our system is px-native.
fn norm_val(v: &str) -> String {
    v.strip_suffix("px").unwrap_or(v).to_string()
}

/// Map CSS property names to our attribute names and expand shorthands.
fn expand_css_property(prop: &str, value: &str) -> Vec<(String, String)> {
    match prop {
        // ── Shorthands with 1–4 values ──
        "padding" | "margin" => expand_box_shorthand(prop, value),

        // ── Name aliases: standard CSS → our attr names ──
        "background-color" | "background" => vec![("bg".into(), value.into())],
        "flex-direction" => vec![("direction".into(), value.into())],
        "align-items" => vec![("align".into(), value.into())],
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
        | "border-width" | "left" | "top" => vec![(prop.into(), norm_val(value))],

        // ── Font-size: normalize px ──
        "font-size" => vec![(prop.into(), norm_val(value))],

        // ── Everything else passes through unchanged ──
        _ => vec![(prop.into(), value.into())],
    }
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
}
