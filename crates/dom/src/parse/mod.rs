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
    let mut tree = build_tree(&tokens, Some(sheet));
    tree.set_sheet(std::sync::Arc::new(sheet.clone()));
    tree
}

/// Extracted raw text blocks from `<style>` and `<script>` elements.
///
/// Returned by [`extract_resources`] and used by the page runtime to
/// feed CSS to the stylesheet parser and JS to the VM.
#[derive(Debug, Default)]
pub struct PageResources {
    /// Concatenated `<style>` content (in document order).
    pub css: String,
    /// Concatenated `<script>` content (in document order).
    pub scripts: Vec<String>,
}

/// Extract `<style>` / `<script>` raw text from HTML source.
///
/// Scans for raw-text elements, extracts their bodies, and removes them
/// from the input so the DOM parser only sees structural HTML.
/// Returns the cleaned HTML and the extracted resources.
pub fn extract_resources(input: &str) -> (String, PageResources) {
    let mut clean = String::with_capacity(input.len());
    let mut res = PageResources::default();
    let mut pos = 0;
    let lower = input.to_ascii_lowercase();
    let bytes = lower.as_bytes();
    let max_iterations = bytes.len() + 1; // safety: at most one iteration per byte
    let mut iter_count = 0;

    while pos < bytes.len() {
        iter_count += 1;
        if iter_count > max_iterations {
            // Bail out — something is wrong with the HTML structure.
            log::warn!("extract_resources: bailing after {iter_count} iterations at pos {pos}");
            clean.push_str(&input[pos..]);
            break;
        }

        if let Some(tag_start) = find_raw_tag_start(&lower[pos..]) {
            // Copy everything before this tag to clean output.
            clean.push_str(&input[pos..pos + tag_start.offset]);

            let is_style = tag_start.is_style;
            let open_end = pos + tag_start.offset + tag_start.open_len;
            let close_tag = if is_style { "</style>" } else { "</script>" };

            // Find matching close tag.
            let body_end = lower[open_end..]
                .find(close_tag)
                .map(|i| open_end + i)
                .unwrap_or(bytes.len());
            let close_end = if body_end + close_tag.len() <= bytes.len() {
                body_end + close_tag.len()
            } else {
                bytes.len()
            };

            let body = input[open_end..body_end].trim();
            if !body.is_empty() {
                if is_style {
                    if !res.css.is_empty() {
                        res.css.push('\n');
                    }
                    res.css.push_str(body);
                } else {
                    res.scripts.push(body.to_string());
                }
            }

            pos = close_end;
        } else {
            clean.push_str(&input[pos..]);
            break;
        }
    }

    (clean, res)
}

struct RawTagStart {
    offset: usize,
    open_len: usize,
    is_style: bool,
}

/// Find the next `<style` or `<script` tag in lowercase input.
fn find_raw_tag_start(s: &str) -> Option<RawTagStart> {
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'<' {
            for (prefix, is_style) in &[("<style", true), ("<script", false)] {
                if s[i..].starts_with(prefix) {
                    let after = i + prefix.len();
                    // Must be followed by whitespace or '>'.
                    if after < bytes.len()
                        && (bytes[after].is_ascii_whitespace() || bytes[after] == b'>')
                    {
                        // Find the end of this open tag '>'
                        if let Some(gt) = s[after..].find('>') {
                            return Some(RawTagStart {
                                offset: i,
                                open_len: prefix.len() + gt + 1,
                                is_style: *is_style,
                            });
                        }
                    }
                }
            }
        }
        i += 1;
    }
    None
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

// ── HTML entity decoding ────────────────────────────────────────────────

/// Decode HTML entities (`&amp;`, `&lt;`, `&#39;`, `&#x2F;`, etc.) in-place.
/// Covers named entities common in real-world pages plus numeric/hex forms.
fn decode_entities(input: &str) -> String {
    if !input.contains('&') {
        return input.to_string();
    }
    let mut out = String::with_capacity(input.len());
    let mut chars = input.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '&' {
            out.push(c);
            continue;
        }
        // Collect entity up to ';' (max 10 chars to avoid runaway)
        let mut entity = String::new();
        let mut found_semi = false;
        for _ in 0..10 {
            match chars.peek() {
                Some(&';') => {
                    chars.next();
                    found_semi = true;
                    break;
                }
                Some(&c) if c.is_alphanumeric() || c == '#' => {
                    entity.push(c);
                    chars.next();
                }
                _ => break,
            }
        }
        if found_semi {
            if let Some(decoded) = resolve_entity(&entity) {
                out.push_str(decoded);
                continue;
            }
            // Numeric entities: &#NNN; or &#xHHH;
            if let Some(ch) = decode_numeric_entity(&entity) {
                out.push(ch);
                continue;
            }
        }
        // Unrecognized — emit raw
        out.push('&');
        out.push_str(&entity);
        if found_semi {
            out.push(';');
        }
    }
    out
}

fn resolve_entity(name: &str) -> Option<&'static str> {
    Some(match name {
        "amp" => "&",
        "lt" => "<",
        "gt" => ">",
        "quot" => "\"",
        "apos" => "'",
        "nbsp" => "\u{00A0}",
        "ndash" => "\u{2013}",
        "mdash" => "\u{2014}",
        "laquo" => "\u{00AB}",
        "raquo" => "\u{00BB}",
        "copy" => "\u{00A9}",
        "reg" => "\u{00AE}",
        "trade" => "\u{2122}",
        "bull" => "\u{2022}",
        "hellip" => "\u{2026}",
        "lsquo" => "\u{2018}",
        "rsquo" => "\u{2019}",
        "ldquo" => "\u{201C}",
        "rdquo" => "\u{201D}",
        "euro" => "\u{20AC}",
        "pound" => "\u{00A3}",
        "yen" => "\u{00A5}",
        "cent" => "\u{00A2}",
        "times" => "\u{00D7}",
        "divide" => "\u{00F7}",
        "rarr" => "\u{2192}",
        "larr" => "\u{2190}",
        "uarr" => "\u{2191}",
        "darr" => "\u{2193}",
        "para" => "\u{00B6}",
        "sect" => "\u{00A7}",
        "middot" => "\u{00B7}",
        "deg" => "\u{00B0}",
        "plusmn" => "\u{00B1}",
        "frac12" => "\u{00BD}",
        "frac14" => "\u{00BC}",
        "frac34" => "\u{00BE}",
        "iexcl" => "\u{00A1}",
        "iquest" => "\u{00BF}",
        // Accented letters common in French/Spanish/German
        "eacute" => "\u{00E9}",
        "egrave" => "\u{00E8}",
        "ecirc" => "\u{00EA}",
        "agrave" => "\u{00E0}",
        "aacute" => "\u{00E1}",
        "acirc" => "\u{00E2}",
        "iuml" => "\u{00EF}",
        "ocirc" => "\u{00F4}",
        "uacute" => "\u{00FA}",
        "ugrave" => "\u{00F9}",
        "uuml" => "\u{00FC}",
        "ccedil" => "\u{00E7}",
        "ntilde" => "\u{00F1}",
        "szlig" => "\u{00DF}",
        "Eacute" => "\u{00C9}",
        "Egrave" => "\u{00C8}",
        "Agrave" => "\u{00C0}",
        _ => return None,
    })
}

fn decode_numeric_entity(entity: &str) -> Option<char> {
    let s = entity.strip_prefix('#')?;
    let code = if let Some(hex) = s.strip_prefix('x').or_else(|| s.strip_prefix('X')) {
        u32::from_str_radix(hex, 16).ok()?
    } else {
        s.parse::<u32>().ok()?
    };
    char::from_u32(code)
}

fn tokenize(input: &str) -> Vec<Token> {
    let bytes = input.as_bytes();
    let mut tokens = Vec::new();
    let mut i = 0;

    while i < bytes.len() {
        if bytes[i] == b'<' {
            // Skip markup declarations and processing instructions:
            //   <!-- comment -->   |   <!DOCTYPE ...>   |   <?xml ...?>
            if let Some(skip) = skip_markup_special(&input[i..]) {
                i += skip;
                continue;
            }
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
                    content: decode_entities(text),
                });
            }
        }
    }

    tokens
}

/// Skip HTML comments (`<!-- -->`), declarations (`<!DOCTYPE>`, `<![CDATA[]]>`),
/// and processing instructions (`<?...?>`).  Returns bytes consumed, or `None`.
fn skip_markup_special(s: &str) -> Option<usize> {
    if s.starts_with("<!--") {
        return Some(find_end(s, 4, "-->"));
    }
    if s.starts_with("<![CDATA[") {
        return Some(find_end(s, 9, "]]>"));
    }
    if s.starts_with("<!") {
        return Some(find_end(s, 2, ">"));
    }
    if s.starts_with("<?") {
        return Some(find_end(s, 2, "?>"));
    }
    None
}

/// Scan forward from `start` in `s` for `end_marker`, returning total bytes consumed.
fn find_end(s: &str, start: usize, end_marker: &str) -> usize {
    match s[start..].find(end_marker) {
        Some(pos) => start + pos + end_marker.len(),
        None => s.len(), // unterminated — consume rest
    }
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
    populate_identity(&mut tree, root_id, &root_name, &root_attrs);

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

                if !self_closing && !HtmlTag::from(name.as_str()).is_void() {
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

    // Post-pass: auto-set flex-direction to Row for parents with only inline children.
    // This emulates HTML inline flow — <span>, <a>, <b>, etc. siblings flow horizontally.
    auto_inline_direction(&mut tree);

    tree
}

// ── Inline auto-row layout ──────────────────────────────────────────────────

/// When a Box node's visible children are ALL inline-flagged elements,
/// auto-set its flex-direction to Row. This emulates HTML inline flow
/// where `<span>`, `<a>`, `<b>`, etc. siblings flow horizontally.
///
/// Only applies when:
/// - The parent is a Box node
/// - The parent doesn't have an explicitly-set direction via CSS
/// - ALL visible children are inline-type elements (span, a, b, em, etc.)
/// - There is more than one visible child (single child doesn't need row)
fn auto_inline_direction(tree: &mut Tree) {
    use any_compute_core::flex::{Direction, FlexWrap};
    let ids: Vec<NodeId> = (0..tree.arena.len()).map(NodeId).collect();
    for id in ids {
        let slot = tree.slot(id);
        if !matches!(slot.kind, NodeKind::Box) {
            continue;
        }
        // Skip if direction was already set to Row (e.g. by CSS `display:flex`)
        if slot.style.direction == Direction::Row {
            continue;
        }
        let children = slot.children.clone();
        if children.len() < 2 {
            continue;
        }
        // Check: all visible children are inline-tagged OR text-leaf boxes
        // (Box with a single Text child — common wrapper pattern like <div><a>...</a></div>)
        let all_inline = children.iter().all(|&cid| {
            let cs = tree.slot(cid);
            if cs.style.is_hidden() {
                return true;
            }
            let tag = HtmlTag::from(cs.element.as_str());
            if tag.is_inline() {
                return true;
            }
            // Text-leaf box: a Box whose only visible child is a Text/inline node
            if matches!(cs.kind, NodeKind::Box) {
                let visible: Vec<_> = cs
                    .children
                    .iter()
                    .filter(|&&gc| !tree.slot(gc).style.is_hidden())
                    .collect();
                return visible.len() == 1
                    && matches!(tree.slot(*visible[0]).kind, NodeKind::Text(..));
            }
            false
        });
        if all_inline {
            let s = &mut tree.slot_mut(id).style;
            s.direction = Direction::Row;
            s.flex_wrap = FlexWrap::Wrap;
            if s.gap == 0.0 {
                s.gap = 4.0; // Prevent inline text from merging
            }
        }
    }
}

// ── Tag → NodeKind mapping ──────────────────────────────────────────────────
// Uses the HtmlTag enum from style.rs — single source of truth for tag semantics.

/// Create a child node from tag + attributes and apply tag/data-tag.
/// Extract bar fraction + fill from element attributes.
fn bar_from_attrs(attrs: &[(String, String)]) -> (f64, Color) {
    let frac = find_attr(attrs, "value")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.0);
    let fill = find_attr(attrs, "color")
        .and_then(|v| parse_color(&v))
        .unwrap_or(Color::WHITE);
    (frac, fill)
}

fn spawn_child(
    tree: &mut Tree,
    parent: NodeId,
    tag: &str,
    attrs: &[(String, String)],
    sheet: Option<&StyleSheet>,
) -> NodeId {
    let html_tag = HtmlTag::from(tag);
    // "bar" is our custom alias, not an HTML tag
    let kind = if tag == "bar" || tag == "text" {
        HtmlTag::from(match tag {
            "bar" => "progress",
            "text" => "span",
            _ => tag,
        })
    } else {
        html_tag
    };

    // ── Input-type specialisation ───────────────────────────────────────
    // <input type="hidden"> → display:none
    // <input type="submit|button" value="X"> → button-like with label text
    // <input type="text|..."> → visible input box with size-based width
    if tag == "input" {
        let itype = find_attr(attrs, "type")
            .unwrap_or_default()
            .to_ascii_lowercase();
        if itype == "hidden" {
            let mut style = resolve_style(tag, attrs, sheet);
            style.display = Display::None;
            let id = tree.add_box(parent, style);
            populate_identity(tree, id, tag, attrs);
            return id;
        }
        if itype == "submit" || itype == "button" {
            let label = find_attr(attrs, "value").unwrap_or_default();
            let mut style = resolve_style("button", attrs, sheet);
            apply_style_attrs(&mut style, attrs);
            let id = tree.add_text(parent, &label, style);
            populate_identity(tree, id, tag, attrs);
            return id;
        }
        // Text / search / password / default — render as visible box
        // `size` attribute sets width in character units (~8px per char)
        if let Some(sz) = find_attr(attrs, "size") {
            if let Ok(n) = sz.trim().parse::<f64>() {
                let mut patched_attrs = attrs.to_vec();
                let w = (n * CHAR_WIDTH_RATIO * DEFAULT_FONT_SIZE).round();
                patched_attrs.push(("width".into(), format!("{w}")));
                let style = resolve_style(tag, &patched_attrs, sheet);
                let id = tree.add_box(parent, style);
                apply_tag(tree, id, attrs);
                populate_identity(tree, id, tag, attrs);
                return id;
            }
        }
    }

    let mut style = resolve_style(tag, attrs, sheet);

    // ── HTML-specific attribute overrides ────────────────────────────────
    // `align` on td/th → text-align (HTML spec), not flex align-items
    if matches!(tag, "td" | "th" | "center" | "p") {
        if let Some(val) = find_attr(attrs, "align") {
            style.text_align = TextAlign::from_css(&val);
        }
    }

    let id = match kind.kind() {
        TagKind::Box => tree.add_box(parent, style),
        TagKind::Text => {
            let text = find_attr(attrs, "text").unwrap_or_default();
            tree.add_text(parent, text, style)
        }
        TagKind::Bar => {
            let (frac, fill) = bar_from_attrs(attrs);
            tree.add_bar(parent, frac, fill, style)
        }
    };
    apply_tag(tree, id, attrs);
    populate_identity(tree, id, tag, attrs);
    id
}

/// Store element tag name + class list + id on a slot for runtime restyle matching.
fn populate_identity(tree: &mut Tree, id: NodeId, tag: &str, attrs: &[(String, String)]) {
    let slot = tree.slot_mut(id);
    slot.element = tag.to_string();
    slot.class_list = find_attr(attrs, "class")
        .map(|c| c.split_whitespace().map(String::from).collect())
        .unwrap_or_default();
    slot.id = find_attr(attrs, "id").map(|s| s.to_string());
    slot.base_style = slot.style.clone();
}

/// Set kind on an existing node (used for the root which Tree::new always creates as Box).
fn set_kind(tree: &mut Tree, id: NodeId, tag: &str, attrs: &[(String, String)]) {
    let html_tag = HtmlTag::from(tag);
    let kind = if tag == "bar" || tag == "text" {
        HtmlTag::from(match tag {
            "bar" => "progress",
            "text" => "span",
            _ => tag,
        })
    } else {
        html_tag
    };
    match kind.kind() {
        TagKind::Text => {
            let text = find_attr(attrs, "text").unwrap_or_default();
            tree.slot_mut(id).kind = NodeKind::Text(text);
        }
        TagKind::Bar => {
            let (frac, fill) = bar_from_attrs(attrs);
            tree.slot_mut(id).kind = NodeKind::Bar {
                fraction: frac,
                fill,
            };
        }
        TagKind::Box => {} // already Box by default
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
mod attr;
pub use attr::*;

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
        assert_eq!(tree.arena[1].rect.size.w(), 200.0);
        assert_eq!(tree.arena[1].rect.size.h(), 100.0);
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
        assert_eq!(tree.arena[1].rect.size.w(), 200.0);
        // Main content is flex-grow, should take remaining space.
        assert!(tree.arena[4].rect.size.w() > 500.0, "main should be >500px");
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
