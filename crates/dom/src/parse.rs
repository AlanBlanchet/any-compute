//! Parse HTML-like markup into our arena-based [`Tree`].
//!
//! Zero external dependencies — a purpose-built scanner that maps a small
//! subset of HTML/CSS onto our [`Style`] + [`NodeKind`] model.
//!
//! ## Supported elements
//!
//! | Markup tag   | Maps to          | Notes                                    |
//! |-------------|------------------|------------------------------------------|
//! | `<div>`     | `NodeKind::Box`  | Container, default column layout         |
//! | `<span>`    | `NodeKind::Text` | Inline text (body = text content)        |
//! | `<p>`       | `NodeKind::Text` | Paragraph text                           |
//! | `<progress>`| `NodeKind::Bar`  | `value` attr → fraction, `color` → fill  |
//! | any other   | `NodeKind::Box`  | Unknown tags become generic containers   |
//!
//! ## Supported attributes
//!
//! Style attributes mirror our [`Style`] builder names for zero-friction
//! mapping.  CSS-like inline `style="..."` is **not** parsed — instead use
//! direct attributes which are friendlier and type-safe:
//!
//! ```html
//! <div w="200" h="100" bg="#1e1e2e" direction="row" gap="8" pad="12">
//!   <span font="16" color="#cdd2f4">Hello</span>
//!   <progress value="0.7" color="#a6e3a1" h="8" radius="4" />
//! </div>
//! ```
//!
//! ## Usage
//!
//! ```
//! use any_compute_dom::parse::parse;
//! let tree = parse(r#"<div w="400" h="300"><span>Hello</span></div>"#);
//! assert_eq!(tree.arena.len(), 2);
//! ```

use any_compute_core::render::Color;

use crate::css::StyleSheet;
use crate::style::*;
use crate::tree::*;

/// Parse error with position context.
///
/// Kept for external consumers — the built-in parsers are fault-tolerant
/// and never return errors, but downstream code may still define custom
/// parse errors using this type.
#[derive(Debug, Clone, PartialEq)]
pub struct ParseError {
    pub offset: usize,
    pub message: String,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "parse error at byte {}: {}", self.offset, self.message)
    }
}

impl std::error::Error for ParseError {}

/// Parse HTML-like markup into a [`Tree`].
///
/// Fault-tolerant: malformed tags/attributes are silently skipped.
/// Empty input produces a default root node.  Never panics.
pub fn parse(input: &str) -> Tree {
    let tokens = tokenize(input);
    build_tree(&tokens, None)
}

/// Parse HTML-like markup with CSS class resolution via a [`StyleSheet`].
///
/// Fault-tolerant: malformed markup is silently skipped.  Never panics.
pub fn parse_with_css(input: &str, sheet: &StyleSheet) -> Tree {
    let tokens = tokenize(input);
    build_tree(&tokens, Some(sheet))
}

// ── Tokenizer ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
enum Token {
    /// `<tag attr="val" ...>` or `<tag ... />`
    OpenTag {
        name: String,
        attrs: Vec<(String, String)>,
        self_closing: bool,
    },
    /// `</tag>`
    CloseTag { name: String },
    /// Text content between tags (trimmed, non-empty).
    Text { content: String },
}

fn tokenize(input: &str) -> Vec<Token> {
    let bytes = input.as_bytes();
    let mut tokens = Vec::new();
    let mut i = 0;

    while i < bytes.len() {
        if bytes[i] == b'<' {
            if i + 1 < bytes.len() && bytes[i + 1] == b'/' {
                // Close tag.
                i += 2;
                let name_start = i;
                while i < bytes.len() && bytes[i] != b'>' {
                    i += 1;
                }
                if i >= bytes.len() {
                    break; // unclosed close tag — skip
                }
                let name = input[name_start..i].trim().to_ascii_lowercase();
                tokens.push(Token::CloseTag { name });
                i += 1; // skip '>'
            } else {
                // Open tag.
                i += 1;
                // Tag name.
                let name_start = i;
                while i < bytes.len()
                    && !bytes[i].is_ascii_whitespace()
                    && bytes[i] != b'>'
                    && bytes[i] != b'/'
                {
                    i += 1;
                }
                let name = input[name_start..i].trim().to_ascii_lowercase();

                // Attributes.
                let mut attrs = Vec::new();
                loop {
                    // Skip whitespace.
                    while i < bytes.len() && bytes[i].is_ascii_whitespace() {
                        i += 1;
                    }
                    if i >= bytes.len() {
                        break; // unclosed open tag — use attrs gathered so far
                    }
                    if bytes[i] == b'>' || bytes[i] == b'/' {
                        break;
                    }
                    // Attribute name.
                    let attr_start = i;
                    while i < bytes.len()
                        && bytes[i] != b'='
                        && !bytes[i].is_ascii_whitespace()
                        && bytes[i] != b'>'
                        && bytes[i] != b'/'
                    {
                        i += 1;
                    }
                    let attr_name = input[attr_start..i].to_ascii_lowercase();
                    // Skip '='
                    if i < bytes.len() && bytes[i] == b'=' {
                        i += 1;
                    }
                    // Value — quoted or bare.
                    let value = if i < bytes.len() && (bytes[i] == b'"' || bytes[i] == b'\'') {
                        let quote = bytes[i];
                        i += 1;
                        let val_start = i;
                        while i < bytes.len() && bytes[i] != quote {
                            i += 1;
                        }
                        let val = input[val_start..i].to_string();
                        if i < bytes.len() {
                            i += 1;
                        } // skip closing quote
                        val
                    } else {
                        // Bare value — until whitespace or '>' or '/'.
                        let val_start = i;
                        while i < bytes.len()
                            && !bytes[i].is_ascii_whitespace()
                            && bytes[i] != b'>'
                            && bytes[i] != b'/'
                        {
                            i += 1;
                        }
                        input[val_start..i].to_string()
                    };
                    if !attr_name.is_empty() {
                        attrs.push((attr_name, value));
                    }
                }

                let self_closing = i < bytes.len() && bytes[i] == b'/';
                if self_closing {
                    i += 1;
                }
                if i < bytes.len() && bytes[i] == b'>' {
                    i += 1;
                }

                tokens.push(Token::OpenTag {
                    name,
                    attrs,
                    self_closing,
                });
            }
        } else {
            // Text content.
            let start = i;
            while i < bytes.len() && bytes[i] != b'<' {
                i += 1;
            }
            let text = input[start..i].trim();
            if !text.is_empty() {
                tokens.push(Token::Text {
                    content: text.to_string(),
                });
            }
        }
    }

    tokens
}

// ── Tree builder ────────────────────────────────────────────────────────────

fn build_tree(tokens: &[Token], sheet: Option<&StyleSheet>) -> Tree {
    if tokens.is_empty() {
        return Tree::new(Style::default());
    }

    // The first token must be an open tag — it becomes the root.
    // If not, create a default root and treat everything as children.
    let (root_name, root_attrs, root_self_closing, start_idx) = match &tokens[0] {
        Token::OpenTag {
            name,
            attrs,
            self_closing,
            ..
        } => (name.clone(), attrs.clone(), *self_closing, 1),
        _ => {
            // No root open tag — wrap everything in a default container.
            (String::from("div"), Vec::new(), false, 0)
        }
    };

    let root_style = resolve_style(&root_name, &root_attrs, sheet);
    let mut tree = Tree::new(root_style);
    let root_id = tree.root;

    // Transform root kind if it maps to text/bar (rare but possible).
    set_kind(&mut tree, root_id, &root_name, &root_attrs);
    apply_tag(&mut tree, root_id, &root_attrs);

    if root_self_closing {
        return tree;
    }

    // Stack of (NodeId, tag_name) for nesting.
    let mut stack: Vec<(NodeId, String)> = vec![(root_id, root_name.clone())];

    let mut ti = start_idx;
    while ti < tokens.len() {
        match &tokens[ti] {
            Token::OpenTag {
                name,
                attrs,
                self_closing,
                ..
            } => {
                let parent = stack.last().map(|(id, _)| *id).unwrap_or(tree.root);
                let id = spawn_child(&mut tree, parent, name, attrs, sheet);

                if !self_closing {
                    stack.push((id, name.clone()));
                }
            }
            Token::CloseTag { name, .. } => {
                // Pop the stack until we find a matching open tag.
                // Unmatched close tags are silently skipped.
                if let Some(pos) = stack.iter().rposition(|(_, n)| n == name) {
                    stack.truncate(pos);
                }
            }
            Token::Text { content, .. } => {
                // Text between tags → Text node child.
                let parent = stack.last().map(|(id, _)| *id).unwrap_or(tree.root);
                // Check if parent is already a Text node — if so, replace its content.
                if matches!(tree.slot(parent).kind, NodeKind::Text(_)) {
                    tree.slot_mut(parent).kind = NodeKind::Text(content.clone());
                } else {
                    let parent_style = tree.slot(parent).style.clone();
                    tree.add_text(
                        parent,
                        content.as_str(),
                        Style {
                            font_size: parent_style.font_size,
                            color: parent_style.color,
                            ..Style::default()
                        },
                    );
                }
            }
        }
        ti += 1;
    }

    tree
}

// ── Tag → NodeKind mapping ──────────────────────────────────────────────────

enum TagMapping {
    Box,
    Text,
    Bar,
}

fn map_tag(name: &str) -> TagMapping {
    match name {
        "span" | "p" | "h1" | "h2" | "h3" | "h4" | "h5" | "h6" | "label" | "text" => {
            TagMapping::Text
        }
        "progress" | "bar" | "meter" => TagMapping::Bar,
        _ => TagMapping::Box, // div, section, header, footer, nav, ...
    }
}

/// Create a child node from tag + attributes and apply tag/data-tag.
fn spawn_child(
    tree: &mut Tree,
    parent: NodeId,
    tag: &str,
    attrs: &[(String, String)],
    sheet: Option<&StyleSheet>,
) -> NodeId {
    let style = resolve_style(tag, attrs, sheet);
    let id = match map_tag(tag) {
        TagMapping::Box => tree.add_box(parent, style),
        TagMapping::Text => {
            let text = find_attr(attrs, "text").unwrap_or_default();
            tree.add_text(parent, text, style)
        }
        TagMapping::Bar => {
            let frac: f64 = find_attr(attrs, "value")
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.0);
            let fill = find_attr(attrs, "color")
                .and_then(|v| parse_color(&v))
                .unwrap_or(Color::WHITE);
            tree.add_bar(parent, frac, fill, style)
        }
    };
    apply_tag(tree, id, attrs);
    id
}

/// Set kind on an existing node (used for the root which Tree::new always creates as Box).
fn set_kind(tree: &mut Tree, id: NodeId, tag: &str, attrs: &[(String, String)]) {
    match map_tag(tag) {
        TagMapping::Text => {
            let text = find_attr(attrs, "text").unwrap_or_default();
            tree.slot_mut(id).kind = NodeKind::Text(text);
        }
        TagMapping::Bar => {
            let frac: f64 = find_attr(attrs, "value")
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.0);
            let fill = find_attr(attrs, "color")
                .and_then(|v| parse_color(&v))
                .unwrap_or(Color::WHITE);
            tree.slot_mut(id).kind = NodeKind::Bar {
                fraction: frac,
                fill,
            };
        }
        TagMapping::Box => {} // already Box by default
    }
}

/// Apply data-tag / tag attribute if present.
fn apply_tag(tree: &mut Tree, id: NodeId, attrs: &[(String, String)]) {
    if let Some(tag_val) = find_attr(attrs, "data-tag").or_else(|| find_attr(attrs, "tag")) {
        tree.tag(id, tag_val);
    }
}

// ── Style resolution ────────────────────────────────────────────────────────

/// Resolve style for an element: CSS (tag < classes < id) then inline attrs.
fn resolve_style(tag: &str, attrs: &[(String, String)], sheet: Option<&StyleSheet>) -> Style {
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
/// class resolution.
pub(crate) fn apply_style_attrs(s: &mut Style, attrs: &[(String, String)]) {
    for (key, val) in attrs {
        match key.as_str() {
            "w" | "width" => {
                if let Some(d) = parse_dimension(val) {
                    s.width = d;
                }
            }
            "h" | "height" => {
                if let Some(d) = parse_dimension(val) {
                    s.height = d;
                }
            }
            "min-w" | "min-width" => {
                if let Some(d) = parse_dimension(val) {
                    s.min_width = d;
                }
            }
            "min-h" | "min-height" => {
                if let Some(d) = parse_dimension(val) {
                    s.min_height = d;
                }
            }
            "max-w" | "max-width" => {
                if let Some(d) = parse_dimension(val) {
                    s.max_width = d;
                }
            }
            "max-h" | "max-height" => {
                if let Some(d) = parse_dimension(val) {
                    s.max_height = d;
                }
            }
            "direction" | "dir" | "flex-direction" => {
                s.direction = match val.as_str() {
                    "row" => Direction::Row,
                    _ => Direction::Column,
                }
            }
            "align" | "align-items" => {
                s.align = match val.as_str() {
                    "center" => Align::Center,
                    "end" | "flex-end" => Align::End,
                    "stretch" => Align::Stretch,
                    _ => Align::Start,
                }
            }
            "justify" | "justify-content" => {
                s.justify = match val.as_str() {
                    "center" => Justify::Center,
                    "end" | "flex-end" => Justify::End,
                    "space-between" => Justify::SpaceBetween,
                    "space-around" => Justify::SpaceAround,
                    "space-evenly" => Justify::SpaceEvenly,
                    _ => Justify::Start,
                }
            }
            "gap" => {
                if let Ok(v) = val.parse() {
                    s.gap = v;
                }
            }
            "pad" | "padding" => {
                if let Ok(v) = val.parse() {
                    s.padding = Edges::all(v);
                }
            }
            "pad-x" | "padding-x" => {
                if let Ok(v) = val.parse::<f64>() {
                    s.padding.left = v;
                    s.padding.right = v;
                }
            }
            "pad-y" | "padding-y" => {
                if let Ok(v) = val.parse::<f64>() {
                    s.padding.top = v;
                    s.padding.bottom = v;
                }
            }
            "padding-top" => {
                if let Ok(v) = val.parse() {
                    s.padding.top = v;
                }
            }
            "padding-right" => {
                if let Ok(v) = val.parse() {
                    s.padding.right = v;
                }
            }
            "padding-bottom" => {
                if let Ok(v) = val.parse() {
                    s.padding.bottom = v;
                }
            }
            "padding-left" => {
                if let Ok(v) = val.parse() {
                    s.padding.left = v;
                }
            }
            "margin" => {
                if let Ok(v) = val.parse() {
                    s.margin = Edges::all(v);
                }
            }
            "margin-x" => {
                if let Ok(v) = val.parse::<f64>() {
                    s.margin.left = v;
                    s.margin.right = v;
                }
            }
            "margin-y" => {
                if let Ok(v) = val.parse::<f64>() {
                    s.margin.top = v;
                    s.margin.bottom = v;
                }
            }
            "margin-top" => {
                if let Ok(v) = val.parse() {
                    s.margin.top = v;
                }
            }
            "margin-right" => {
                if let Ok(v) = val.parse() {
                    s.margin.right = v;
                }
            }
            "margin-bottom" => {
                if let Ok(v) = val.parse() {
                    s.margin.bottom = v;
                }
            }
            "margin-left" => {
                if let Ok(v) = val.parse() {
                    s.margin.left = v;
                }
            }
            "bg" | "background" | "background-color" => {
                if let Some(c) = parse_color(val) {
                    s.background = c;
                }
            }
            "border-color" => {
                if let Some(c) = parse_color(val) {
                    s.border_color = c;
                }
            }
            "border-width" => {
                if let Ok(v) = val.parse() {
                    s.border_width = v;
                }
            }
            "radius" | "border-radius" => {
                if let Ok(v) = val.parse() {
                    s.corner_radius = v;
                }
            }
            "opacity" => {
                if let Ok(v) = val.parse() {
                    s.opacity = v;
                }
            }
            "font" | "font-size" => {
                if let Ok(v) = val.parse() {
                    s.font_size = v;
                }
            }
            "color" => {
                if let Some(c) = parse_color(val) {
                    s.color = c;
                }
            }
            "grow" | "flex-grow" => {
                if let Ok(v) = val.parse() {
                    s.flex_grow = v;
                }
            }
            "shrink" | "flex-shrink" => {
                if let Ok(v) = val.parse() {
                    s.flex_shrink = v;
                }
            }
            "position" => {
                s.position = match val.as_str() {
                    "absolute" => Position::Absolute,
                    _ => Position::Relative,
                }
            }
            "overflow" => {
                s.overflow = match val.as_str() {
                    "hidden" => Overflow::Hidden,
                    "scroll" => Overflow::Scroll,
                    _ => Overflow::Visible,
                }
            }
            "left" => {
                if let Some(d) = parse_dimension(val) {
                    s.left = d;
                    s.position = Position::Absolute;
                }
            }
            "top" => {
                if let Some(d) = parse_dimension(val) {
                    s.top = d;
                    s.position = Position::Absolute;
                }
            }
            // Non-style attributes (class, id, tag, value, text) are skipped.
            _ => {}
        }
    }
}

// ── Value parsers ───────────────────────────────────────────────────────────

pub(crate) fn parse_dimension(val: &str) -> Option<Dimension> {
    let val = val.trim();
    if val == "auto" {
        return Some(Dimension::Auto);
    }
    if let Some(pct) = val.strip_suffix('%') {
        return pct.trim().parse::<f64>().ok().map(Dimension::Percent);
    }
    // Strip "px" suffix if present.
    let num = val.strip_suffix("px").unwrap_or(val);
    num.trim().parse::<f64>().ok().map(Dimension::Px)
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
        let parts: Vec<&str> = inner.split(',').map(|s| s.trim()).collect();
        return match parts.len() {
            3 => Some(Color::rgb(
                parts[0].parse().ok()?,
                parts[1].parse().ok()?,
                parts[2].parse().ok()?,
            )),
            4 => Some(Color::rgba(
                parts[0].parse().ok()?,
                parts[1].parse().ok()?,
                parts[2].parse().ok()?,
                parts[3].parse().ok()?,
            )),
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

#[cfg(test)]
mod tests {
    use super::*;
    use any_compute_core::layout::Size;

    #[test]
    fn parse_simple_div() {
        let tree = parse(r#"<div w="400" h="300"></div>"#);
        assert_eq!(tree.arena.len(), 1);
        assert_eq!(tree.arena[0].style.width, Dimension::Px(400.0));
        assert_eq!(tree.arena[0].style.height, Dimension::Px(300.0));
    }

    #[test]
    fn parse_nested_with_text() {
        let tree = parse(r#"<div w="400" h="300"><span font="16">Hello</span></div>"#);
        assert_eq!(tree.arena.len(), 2);
        match &tree.arena[1].kind {
            NodeKind::Text(s) => assert_eq!(s, "Hello"),
            other => panic!("expected Text, got {:?}", other),
        }
    }

    #[test]
    fn parse_bar() {
        let tree = parse(r##"<div><progress value="0.7" color="#00ff00" h="8" /></div>"##);
        assert_eq!(tree.arena.len(), 2);
        match &tree.arena[1].kind {
            NodeKind::Bar { fraction, fill } => {
                assert!((fraction - 0.7).abs() < 1e-10);
                assert_eq!(*fill, Color::rgb(0, 255, 0));
            }
            other => panic!("expected Bar, got {:?}", other),
        }
    }

    #[test]
    fn parse_percentage_dimensions() {
        let tree = parse(r#"<div w="50%" h="100%"></div>"#);
        assert_eq!(tree.arena[0].style.width, Dimension::Percent(50.0));
        assert_eq!(tree.arena[0].style.height, Dimension::Percent(100.0));
    }

    #[test]
    fn parse_hex_colors() {
        assert_eq!(parse_color("#ff0000"), Some(Color::rgb(255, 0, 0)));
        assert_eq!(parse_color("#f00"), Some(Color::rgb(255, 0, 0)));
        assert_eq!(parse_color("#ff000080"), Some(Color::rgba(255, 0, 0, 128)));
    }

    #[test]
    fn parse_named_colors() {
        assert_eq!(parse_color("white"), Some(Color::WHITE));
        assert_eq!(parse_color("transparent"), Some(Color::TRANSPARENT));
    }

    #[test]
    fn parse_rgb_function() {
        assert_eq!(
            parse_color("rgb(100,200,50)"),
            Some(Color::rgb(100, 200, 50))
        );
        assert_eq!(
            parse_color("rgba(100,200,50,128)"),
            Some(Color::rgba(100, 200, 50, 128))
        );
    }

    #[test]
    fn parse_tag_attribute() {
        let tree = parse(r#"<div w="100" h="50" data-tag="my-btn"></div>"#);
        assert_eq!(tree.arena[0].tag.as_deref(), Some("my-btn"));
    }

    #[test]
    fn parse_self_closing() {
        let tree = parse(
            r#"<div w="400" h="300"><span font="14" text="hi" /><div w="10" h="10" /></div>"#,
        );
        assert_eq!(tree.arena.len(), 3);
    }

    #[test]
    fn parse_direction_and_flex() {
        let tree = parse(r#"<div direction="row" gap="8"><div grow="1" /><div w="50" /></div>"#);
        assert_eq!(tree.arena[0].style.direction, Direction::Row);
        assert_eq!(tree.arena[0].style.gap, 8.0);
        assert_eq!(tree.arena[1].style.flex_grow, 1.0);
    }

    #[test]
    fn parsed_tree_layouts_correctly() {
        let mut tree =
            parse(r##"<div w="400" h="300" bg="#000"><div w="200" h="100" bg="#fff" /></div>"##);
        tree.layout(Size::new(400.0, 300.0));
        assert_eq!(tree.arena[1].rect.size.w, 200.0);
        assert_eq!(tree.arena[1].rect.size.h, 100.0);
    }

    #[test]
    fn parse_complex_layout() {
        let markup = r##"
            <div w="800" h="600" direction="row" bg="#1e1e2e">
                <div w="200" bg="#181825" pad="12" gap="8">
                    <span font="16" color="#cdd2f4">Sidebar</span>
                </div>
                <div grow="1" pad="24" gap="16">
                    <span font="22" color="#cdd2f4">Main Content</span>
                    <progress value="0.65" color="#a6e3a1" h="8" radius="4" />
                </div>
            </div>
        "##;
        let mut tree = parse(markup);
        tree.layout(Size::new(800.0, 600.0));
        // Root is row layout.
        assert_eq!(tree.arena[0].style.direction, Direction::Row);
        // Sidebar has fixed width 200.
        assert_eq!(tree.arena[1].rect.size.w, 200.0);
        // Main content is flex-grow, should take remaining space.
        assert!(tree.arena[4].rect.size.w > 500.0, "main should be >500px");
    }

    #[test]
    fn empty_input_does_not_crash() {
        let tree = parse("");
        // Empty input → default root node.
        assert_eq!(tree.arena.len(), 1);
    }

    #[test]
    fn unclosed_tag_does_not_crash() {
        let tree = parse("<div");
        // Unclosed tag → still produces a tree with best-effort parsing.
        assert!(!tree.arena.is_empty());
    }

    #[test]
    fn broken_html_does_not_crash() {
        let tree = parse("<broken attr=");
        let _ = tree;
    }

    #[test]
    fn malformed_attr_values_ignored() {
        let tree = parse(r#"<div w="notanumber" h="300"></div>"#);
        assert_eq!(tree.arena[0].style.width, Dimension::Auto); // bad → default
        assert_eq!(tree.arena[0].style.height, Dimension::Px(300.0));
    }
}
