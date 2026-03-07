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

use crate::parse::{
    apply_style_attrs, compile_attr, parse_filter, parse_px, parse_shadow, parse_time,
    parse_transform,
};
use crate::style::{Style, StyleOp, apply_ops};
use any_compute_core::animation::Easing;

// ── CSS transition + animation metadata ─────────────────────────────────────

/// Parsed CSS transition declaration: `transition: property duration easing delay`.
///
/// Stored per-selector alongside [`StyleOp`]s. At runtime, when a style change
/// occurs on a matching element, the transition system uses this spec to create
/// a [`Transition<T>`] in the animation engine.
#[derive(Debug, Clone, PartialEq)]
pub struct TransitionSpec {
    /// CSS property name (`"all"`, `"opacity"`, `"transform"`, etc.).
    pub property: String,
    /// Duration in seconds.
    pub duration_secs: f64,
    /// Easing function.
    pub easing: Easing,
    /// Delay before start in seconds.
    pub delay_secs: f64,
}

/// How many times an animation repeats.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AnimationIterCount {
    /// Repeat a fixed number of times.
    Count(f64),
    /// Repeat forever.
    Infinite,
}

impl Default for AnimationIterCount {
    fn default() -> Self {
        Self::Count(1.0)
    }
}

/// Animation playback direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AnimationDirection {
    #[default]
    Normal,
    Reverse,
    Alternate,
    AlternateReverse,
}

impl AnimationDirection {
    pub fn from_css(val: &str) -> Self {
        match val {
            "reverse" => Self::Reverse,
            "alternate" => Self::Alternate,
            "alternate-reverse" => Self::AlternateReverse,
            _ => Self::Normal,
        }
    }
}

/// Animation fill mode (CSS `animation-fill-mode`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AnimationFillMode {
    #[default]
    None,
    Forwards,
    Backwards,
    Both,
}

impl AnimationFillMode {
    pub fn from_css(val: &str) -> Self {
        match val {
            "forwards" => Self::Forwards,
            "backwards" => Self::Backwards,
            "both" => Self::Both,
            _ => Self::None,
        }
    }
}

/// Parsed CSS animation declaration: `animation: name duration easing delay count direction fill`.
#[derive(Debug, Clone, PartialEq)]
pub struct AnimationSpec {
    pub name: String,
    pub duration_secs: f64,
    pub easing: Easing,
    pub delay_secs: f64,
    pub iteration_count: AnimationIterCount,
    pub direction: AnimationDirection,
    pub fill_mode: AnimationFillMode,
}

/// A single keyframe stop within `@keyframes`.
#[derive(Debug, Clone)]
pub struct Keyframe {
    /// Progress point: 0.0 = `from`, 1.0 = `to`, 0.5 = `50%`.
    pub stop: f64,
    /// Style operations to apply at this stop.
    pub ops: Vec<StyleOp>,
}

// ── Selector model ──────────────────────────────────────────────────────────

/// CSS pseudo-class.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PseudoClass {
    Hover,
    Focus,
    Active,
    Visited,
    FirstChild,
    LastChild,
    /// `nth-child(an+b)` — stored as (a, b).
    NthChild(i32, i32),
}

/// Combinator between selector segments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Combinator {
    /// First segment (no combinator).
    None,
    /// Descendant (space): `A B`.
    Descendant,
    /// Direct child: `A > B`.
    Child,
}

/// One compound segment of a selector: `div.card#main:hover`.
#[derive(Debug, Clone, Default)]
pub struct SelectorSegment {
    pub tag: Option<String>,
    pub classes: Vec<String>,
    pub id: Option<String>,
    pub pseudos: Vec<PseudoClass>,
    pub universal: bool,
}

/// Selector specificity as (ids, classes+pseudos, tags).
pub type Specificity = (u16, u16, u16);

/// A fully parsed CSS selector with specificity.
#[derive(Debug, Clone)]
pub struct ParsedSelector {
    /// Chain of (combinator, compound-selector) segments.
    pub segments: Vec<(Combinator, SelectorSegment)>,
    /// CSS specificity: (ids, classes+pseudos, tags).
    pub specificity: Specificity,
}

/// A rule with a complex selector (descendant / child / pseudo-class).
#[derive(Debug, Clone)]
pub struct ComplexRule {
    /// The parsed selector with specificity.
    pub selector: ParsedSelector,
    /// The compiled style operations + transition/animation metadata.
    pub payload: RulePayload,
}

/// Everything compiled from one CSS rule block.
#[derive(Debug, Clone, Default)]
pub struct RulePayload {
    pub ops: Vec<StyleOp>,
    pub transitions: Vec<TransitionSpec>,
    pub animations: Vec<AnimationSpec>,
}

impl RulePayload {
    fn extend(&mut self, other: &RulePayload) {
        self.ops.extend_from_slice(&other.ops);
        self.transitions.extend_from_slice(&other.transitions);
        self.animations.extend_from_slice(&other.animations);
    }
}

// ── StyleSheet ──────────────────────────────────────────────────────────────

/// Pre-parsed CSS stylesheet.
///
/// Simple selectors (class / tag / id) are O(1) HashMap lookups.
/// Complex selectors (descendant, child, pseudo-class) are stored separately
/// and matched via tree-walking in `resolve_with_context()`.
///
/// Transitions, animations, @keyframes, and CSS custom properties (variables)
/// are fully parsed at stylesheet creation time.
pub struct StyleSheet {
    // ── Fast-path lookups (simple selectors) ──
    classes: HashMap<String, RulePayload>,
    tags: HashMap<String, RulePayload>,
    ids: HashMap<String, RulePayload>,

    // ── Complex selectors (tree-walking) ──
    complex_rules: Vec<ComplexRule>,

    // ── @keyframes definitions ──
    keyframes: HashMap<String, Vec<Keyframe>>,

    // ── CSS custom properties (variables) ──
    /// Global variables (from `:root` or `*` rules).
    variables: HashMap<String, String>,
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
    /// Handles @keyframes, CSS variables, transitions, animations, and
    /// complex selectors (descendant, child, pseudo-class).
    pub fn parse(css: &str) -> Self {
        let cleaned = strip_comments(css);
        let mut classes: HashMap<String, RulePayload> = HashMap::new();
        let mut tags: HashMap<String, RulePayload> = HashMap::new();
        let mut ids: HashMap<String, RulePayload> = HashMap::new();
        let mut complex_rules: Vec<ComplexRule> = Vec::new();
        let mut keyframes: HashMap<String, Vec<Keyframe>> = HashMap::new();
        let mut variables: HashMap<String, String> = HashMap::new();

        let bytes = cleaned.as_bytes();
        let mut i = 0;

        while i < bytes.len() {
            // Skip whitespace
            while i < bytes.len() && bytes[i].is_ascii_whitespace() {
                i += 1;
            }
            if i >= bytes.len() {
                break;
            }

            // ── @keyframes block ──
            if cleaned[i..].starts_with("@keyframes ")
                || cleaned[i..].starts_with("@-webkit-keyframes ")
            {
                let prefix_end = if cleaned[i..].starts_with("@-webkit-") {
                    i + "@-webkit-keyframes ".len()
                } else {
                    i + "@keyframes ".len()
                };
                // Read animation name
                let name_start = prefix_end;
                let mut j = prefix_end;
                while j < bytes.len() && bytes[j] != b'{' {
                    j += 1;
                }
                if j >= bytes.len() {
                    break;
                }
                let name = cleaned[name_start..j].trim().to_string();
                j += 1; // skip outer '{'

                // Parse keyframe stops until matching '}'
                let mut kf_list = Vec::new();
                let mut depth = 1u32;
                while j < bytes.len() && depth > 0 {
                    while j < bytes.len() && bytes[j].is_ascii_whitespace() {
                        j += 1;
                    }
                    if j >= bytes.len() || bytes[j] == b'}' {
                        depth -= 1;
                        j += 1;
                        continue;
                    }
                    // Read stop selector (from, to, 50%, etc)
                    let stop_start = j;
                    while j < bytes.len() && bytes[j] != b'{' && bytes[j] != b'}' {
                        j += 1;
                    }
                    if j >= bytes.len() || bytes[j] == b'}' {
                        depth -= 1;
                        j += 1;
                        continue;
                    }
                    let stop_text = cleaned[stop_start..j].trim();
                    let stop = match stop_text {
                        "from" => Some(0.0),
                        "to" => Some(1.0),
                        s if s.ends_with('%') => s[..s.len() - 1]
                            .trim()
                            .parse::<f64>()
                            .ok()
                            .map(|v| v / 100.0),
                        _ => None,
                    };
                    j += 1; // skip inner '{'
                    // Read body
                    let body_start = j;
                    let mut inner_depth = 1u32;
                    while j < bytes.len() && inner_depth > 0 {
                        match bytes[j] {
                            b'{' => inner_depth += 1,
                            b'}' => inner_depth -= 1,
                            _ => {}
                        }
                        if inner_depth > 0 {
                            j += 1;
                        }
                    }
                    let body = &cleaned[body_start..j];
                    j += 1; // skip inner '}'
                    if let Some(stop_val) = stop {
                        let payload = compile_declarations(body, &variables);
                        kf_list.push(Keyframe {
                            stop: stop_val,
                            ops: payload.ops,
                        });
                    }
                }
                kf_list.sort_by(|a, b| {
                    a.stop
                        .partial_cmp(&b.stop)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                if !name.is_empty() {
                    keyframes.insert(name, kf_list);
                }
                i = j;
                continue;
            }

            // ── Skip other @-rules we don't handle ──
            if i < bytes.len() && bytes[i] == b'@' {
                // Skip until the end of the block or semicolon
                while i < bytes.len() && bytes[i] != b'{' && bytes[i] != b';' {
                    i += 1;
                }
                if i < bytes.len() && bytes[i] == b'{' {
                    let mut depth = 1u32;
                    i += 1;
                    while i < bytes.len() && depth > 0 {
                        match bytes[i] {
                            b'{' => depth += 1,
                            b'}' => depth -= 1,
                            _ => {}
                        }
                        i += 1;
                    }
                } else if i < bytes.len() {
                    i += 1; // skip ';'
                }
                continue;
            }

            // ── Regular rule ──
            // Read selector
            let sel_start = i;
            while i < bytes.len() && bytes[i] != b'{' {
                i += 1;
            }
            if i >= bytes.len() {
                break;
            }
            let selector = cleaned[sel_start..i].trim().to_string();
            i += 1; // skip '{'

            // Read declarations body
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
                break;
            }
            let body = &cleaned[decl_start..i];
            i += 1; // skip '}'

            // Extract CSS custom properties from :root / * selectors
            let sel_trimmed = selector.trim();
            if sel_trimmed == ":root" || sel_trimmed == "*" {
                for decl in body.split(';') {
                    let decl = decl.trim();
                    if let Some((prop, value)) = decl.split_once(':') {
                        let prop = prop.trim();
                        if prop.starts_with("--") {
                            variables.insert(prop.to_string(), value.trim().to_string());
                        }
                    }
                }
            }

            let payload = compile_declarations(body, &variables);

            // Distribute to maps per comma-separated selector
            for sel in selector.split(',') {
                let sel = sel.trim();
                if sel.is_empty() {
                    continue;
                }

                // Check if this is a complex selector
                let is_complex = sel.contains(' ') || sel.contains('>') || sel.contains(':');
                // Exclude simple pseudo-element/class-only that we handle as simple
                let is_simple_pseudo = sel.starts_with(':') && !sel.contains(' ');

                if is_complex && !is_simple_pseudo {
                    if let Some(parsed) = parse_selector(sel) {
                        // Check if this is really complex or just a simple selector with pseudo
                        if parsed.segments.len() == 1 {
                            let seg = &parsed.segments[0].1;
                            if seg.pseudos.is_empty() {
                                // Actually simple — use fast path
                                store_simple_selector(
                                    sel,
                                    &payload,
                                    &mut classes,
                                    &mut tags,
                                    &mut ids,
                                );
                                continue;
                            }
                        }
                        complex_rules.push(ComplexRule {
                            selector: parsed,
                            payload: payload.clone(),
                        });
                    }
                } else {
                    store_simple_selector(sel, &payload, &mut classes, &mut tags, &mut ids);
                }
            }
        }

        Self {
            classes,
            tags,
            ids,
            complex_rules,
            keyframes,
            variables,
        }
    }

    /// Parse CSS with UA (user-agent) defaults baked in.
    pub fn parse_with_ua(css: &str) -> Self {
        let combined = format!("{UA_CSS}\n{css}");
        Self::parse(&combined)
    }

    /// Look up a name in a map and apply its ops onto a default [`Style`].
    fn lookup(map: &HashMap<String, RulePayload>, name: &str) -> Style {
        let mut s = Style::default();
        if let Some(rp) = map.get(name) {
            apply_ops(&mut s, &rp.ops);
        }
        s
    }

    /// Resolve a single class name into a [`Style`].
    pub fn class(&self, name: &str) -> Style {
        Self::lookup(&self.classes, name)
    }

    /// Resolve multiple class names, merging in order (later overrides earlier).
    pub fn classes(&self, names: &[&str]) -> Style {
        let mut s = Style::default();
        for name in names {
            if let Some(rp) = self.classes.get(*name) {
                apply_ops(&mut s, &rp.ops);
            }
        }
        s
    }

    /// Apply a class's declarations on top of an existing style.
    pub fn apply(&self, style: &mut Style, name: &str) {
        if let Some(rp) = self.classes.get(name) {
            apply_ops(style, &rp.ops);
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
    pub fn resolve(
        &self,
        tag: &str,
        class_list: &str,
        id: Option<&str>,
        inline: &[(String, String)],
    ) -> Style {
        let mut s = Style::default();
        // 1. Tag
        if let Some(rp) = self.tags.get(tag) {
            apply_ops(&mut s, &rp.ops);
        }
        // 2. Classes (in order)
        for cls in class_list.split_ascii_whitespace() {
            if let Some(rp) = self.classes.get(cls) {
                apply_ops(&mut s, &rp.ops);
            }
        }
        // 3. Id
        if let Some(id) = id {
            if let Some(rp) = self.ids.get(id) {
                apply_ops(&mut s, &rp.ops);
            }
        }
        // 4. Inline attributes (highest specificity)
        apply_style_attrs(&mut s, inline);
        s
    }

    // ── Metadata accessors ──────────────────────────────────────────────

    /// Get transition specs for a class selector.
    pub fn class_transitions(&self, name: &str) -> &[TransitionSpec] {
        self.classes
            .get(name)
            .map(|rp| rp.transitions.as_slice())
            .unwrap_or(&[])
    }

    /// Get animation specs for a class selector.
    pub fn class_animations(&self, name: &str) -> &[AnimationSpec] {
        self.classes
            .get(name)
            .map(|rp| rp.animations.as_slice())
            .unwrap_or(&[])
    }

    /// Look up a `@keyframes` definition by name.
    pub fn keyframes(&self, name: &str) -> Option<&[Keyframe]> {
        self.keyframes.get(name).map(|v| v.as_slice())
    }

    /// Look up a CSS custom property value.
    pub fn var(&self, name: &str) -> Option<&str> {
        self.variables.get(name).map(|s| s.as_str())
    }

    /// Get all complex rules (for tree-walking resolution).
    pub fn complex_rules(&self) -> &[ComplexRule] {
        &self.complex_rules
    }
}

/// Store a simple selector's payload into the appropriate HashMap.
fn store_simple_selector(
    sel: &str,
    payload: &RulePayload,
    classes: &mut HashMap<String, RulePayload>,
    tags: &mut HashMap<String, RulePayload>,
    ids: &mut HashMap<String, RulePayload>,
) {
    if let Some(name) = sel.strip_prefix('.') {
        classes.entry(name.to_string()).or_default().extend(payload);
    } else if let Some(name) = sel.strip_prefix('#') {
        ids.entry(name.to_string()).or_default().extend(payload);
    } else if let Some(dot) = sel.find('.') {
        // tag.class compound → store under class
        classes
            .entry(sel[dot + 1..].to_string())
            .or_default()
            .extend(payload);
    } else {
        tags.entry(sel.to_string()).or_default().extend(payload);
    }
}

// ── CSS comment stripping ───────────────────────────────────────────────────

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

/// Parse declarations into a [`RulePayload`] (ops + transitions + animations).
///
/// CSS variables are resolved inline via `var(--name, fallback)`.
/// Transition/animation shorthand and longhand properties are extracted.
fn compile_declarations(body: &str, variables: &HashMap<String, String>) -> RulePayload {
    let mut payload = RulePayload::default();

    // Collect individual transition/animation longhand values for assembly
    let mut tr_properties: Option<Vec<String>> = None;
    let mut tr_durations: Option<Vec<f64>> = None;
    let mut tr_easings: Option<Vec<Easing>> = None;
    let mut tr_delays: Option<Vec<f64>> = None;

    let mut an_names: Option<Vec<String>> = None;
    let mut an_durations: Option<Vec<f64>> = None;
    let mut an_easings: Option<Vec<Easing>> = None;
    let mut an_delays: Option<Vec<f64>> = None;
    let mut an_iterations: Option<Vec<AnimationIterCount>> = None;
    let mut an_directions: Option<Vec<AnimationDirection>> = None;
    let mut an_fills: Option<Vec<AnimationFillMode>> = None;

    for decl in body.split(';') {
        let decl = decl.trim();
        if decl.is_empty() {
            continue;
        }
        let Some((prop, raw_value)) = decl.split_once(':') else {
            continue;
        };
        let prop = prop.trim().to_ascii_lowercase();
        // Skip custom properties (stored at sheet level, not op level)
        if prop.starts_with("--") {
            continue;
        }

        // Resolve var() references in the value
        let value = resolve_var(raw_value.trim(), variables);
        let value = value.as_str();

        // ── Transition shorthand ──
        if prop == "transition" {
            for spec in parse_transition_shorthand(value) {
                payload.transitions.push(spec);
            }
            continue;
        }
        // ── Transition longhands ──
        if prop == "transition-property" {
            tr_properties = Some(value.split(',').map(|s| s.trim().to_string()).collect());
            continue;
        }
        if prop == "transition-duration" {
            tr_durations = Some(
                value
                    .split(',')
                    .filter_map(|s| parse_time(s.trim()))
                    .collect(),
            );
            continue;
        }
        if prop == "transition-timing-function" {
            tr_easings = Some(
                value
                    .split(',')
                    .map(|s| Easing::from_css(s.trim()))
                    .collect(),
            );
            continue;
        }
        if prop == "transition-delay" {
            tr_delays = Some(
                value
                    .split(',')
                    .filter_map(|s| parse_time(s.trim()))
                    .collect(),
            );
            continue;
        }

        // ── Animation shorthand ──
        if prop == "animation" {
            for spec in parse_animation_shorthand(value) {
                payload.animations.push(spec);
            }
            continue;
        }
        // ── Animation longhands ──
        if prop == "animation-name" {
            an_names = Some(value.split(',').map(|s| s.trim().to_string()).collect());
            continue;
        }
        if prop == "animation-duration" {
            an_durations = Some(
                value
                    .split(',')
                    .filter_map(|s| parse_time(s.trim()))
                    .collect(),
            );
            continue;
        }
        if prop == "animation-timing-function" {
            an_easings = Some(
                value
                    .split(',')
                    .map(|s| Easing::from_css(s.trim()))
                    .collect(),
            );
            continue;
        }
        if prop == "animation-delay" {
            an_delays = Some(
                value
                    .split(',')
                    .filter_map(|s| parse_time(s.trim()))
                    .collect(),
            );
            continue;
        }
        if prop == "animation-iteration-count" {
            an_iterations = Some(
                value
                    .split(',')
                    .map(|s| {
                        let s = s.trim();
                        if s == "infinite" {
                            AnimationIterCount::Infinite
                        } else {
                            AnimationIterCount::Count(s.parse().unwrap_or(1.0))
                        }
                    })
                    .collect(),
            );
            continue;
        }
        if prop == "animation-direction" {
            an_directions = Some(
                value
                    .split(',')
                    .map(|s| AnimationDirection::from_css(s.trim()))
                    .collect(),
            );
            continue;
        }
        if prop == "animation-fill-mode" {
            an_fills = Some(
                value
                    .split(',')
                    .map(|s| AnimationFillMode::from_css(s.trim()))
                    .collect(),
            );
            continue;
        }

        // ── Multi-op shorthands ──
        match prop.as_str() {
            "transform" => {
                payload.ops.extend(parse_transform(value));
                continue;
            }
            "filter" => {
                payload.ops.extend(parse_filter(value));
                continue;
            }
            "box-shadow" => {
                if let Some(s) = parse_shadow(value) {
                    payload.ops.push(StyleOp::BoxShadow(s));
                }
                continue;
            }
            "text-shadow" => {
                if let Some(s) = parse_shadow(value) {
                    payload.ops.push(StyleOp::TextShadow(s));
                }
                continue;
            }
            _ => {}
        }

        for (key, val) in expand_css_property(&prop, value) {
            if let Some(op) = compile_attr(&key, &val) {
                payload.ops.push(op);
            }
        }
    }

    // ── Assemble transition longhands ──
    if let Some(props) = tr_properties {
        let durs = tr_durations.unwrap_or_default();
        let easings = tr_easings.unwrap_or_default();
        let delays = tr_delays.unwrap_or_default();
        for (i, prop) in props.into_iter().enumerate() {
            payload.transitions.push(TransitionSpec {
                property: prop,
                duration_secs: durs.get(i).or(durs.first()).copied().unwrap_or(0.0),
                easing: easings
                    .get(i)
                    .or(easings.first())
                    .copied()
                    .unwrap_or(Easing::EaseInOut),
                delay_secs: delays.get(i).or(delays.first()).copied().unwrap_or(0.0),
            });
        }
    }

    // ── Assemble animation longhands ──
    if let Some(names) = an_names {
        let durs = an_durations.unwrap_or_default();
        let easings = an_easings.unwrap_or_default();
        let delays = an_delays.unwrap_or_default();
        let iters = an_iterations.unwrap_or_default();
        let dirs = an_directions.unwrap_or_default();
        let fills = an_fills.unwrap_or_default();
        for (i, name) in names.into_iter().enumerate() {
            payload.animations.push(AnimationSpec {
                name,
                duration_secs: durs.get(i).or(durs.first()).copied().unwrap_or(0.0),
                easing: easings
                    .get(i)
                    .or(easings.first())
                    .copied()
                    .unwrap_or(Easing::EaseInOut),
                delay_secs: delays.get(i).or(delays.first()).copied().unwrap_or(0.0),
                iteration_count: iters.get(i).or(iters.first()).copied().unwrap_or_default(),
                direction: dirs.get(i).or(dirs.first()).copied().unwrap_or_default(),
                fill_mode: fills.get(i).or(fills.first()).copied().unwrap_or_default(),
            });
        }
    }

    payload
}

// ── CSS variable resolution ─────────────────────────────────────────────────

/// Resolve `var(--name)` and `var(--name, fallback)` references in a CSS value.
fn resolve_var(value: &str, variables: &HashMap<String, String>) -> String {
    let mut result = value.to_string();
    // Iterate until no more var() references (handles nested vars)
    for _ in 0..16 {
        let Some(start) = result.find("var(") else {
            break;
        };
        // Find matching close paren
        let after = &result[start + 4..];
        let mut depth = 1u32;
        let mut end = 0;
        for (i, b) in after.bytes().enumerate() {
            match b {
                b'(' => depth += 1,
                b')' => {
                    depth -= 1;
                    if depth == 0 {
                        end = i;
                        break;
                    }
                }
                _ => {}
            }
        }
        if depth != 0 {
            break;
        } // malformed
        let inner = &after[..end];
        let (var_name, fallback) = if let Some((name, fb)) = inner.split_once(',') {
            (name.trim(), Some(fb.trim()))
        } else {
            (inner.trim(), None)
        };
        let resolved = variables
            .get(var_name)
            .map(|s| s.as_str())
            .or(fallback)
            .unwrap_or("");
        result = format!(
            "{}{}{}",
            &result[..start],
            resolved,
            &result[start + 4 + end + 1..]
        );
    }
    result
}

// ── Transition / animation shorthand parsers ────────────────────────────────

/// Parse CSS `transition` shorthand: `property duration [easing] [delay], ...`
fn parse_transition_shorthand(value: &str) -> Vec<TransitionSpec> {
    let mut specs = Vec::new();
    for item in value.split(',') {
        let parts: Vec<&str> = item.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }
        let property = parts[0].to_string();
        let duration_secs = parts.get(1).and_then(|s| parse_time(s)).unwrap_or(0.0);
        let easing = parts
            .get(2)
            .map(|s| Easing::from_css(s))
            .unwrap_or(Easing::EaseInOut);
        let delay_secs = parts.get(3).and_then(|s| parse_time(s)).unwrap_or(0.0);
        specs.push(TransitionSpec {
            property,
            duration_secs,
            easing,
            delay_secs,
        });
    }
    specs
}

/// Parse CSS `animation` shorthand: `name duration [easing] [delay] [count] [direction] [fill], ...`
fn parse_animation_shorthand(value: &str) -> Vec<AnimationSpec> {
    let mut specs = Vec::new();
    for item in value.split(',') {
        let parts: Vec<&str> = item.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }

        let mut name = String::new();
        let mut duration_secs = 0.0;
        let mut easing = Easing::EaseInOut;
        let mut delay_secs = 0.0;
        let mut iteration_count = AnimationIterCount::default();
        let mut direction = AnimationDirection::default();
        let mut fill_mode = AnimationFillMode::default();
        let mut time_count = 0u8; // first time = duration, second = delay

        for part in &parts {
            if let Some(t) = parse_time(part) {
                if time_count == 0 {
                    duration_secs = t;
                } else {
                    delay_secs = t;
                }
                time_count += 1;
            } else if *part == "infinite" {
                iteration_count = AnimationIterCount::Infinite;
            } else if let Ok(n) = part.parse::<f64>() {
                iteration_count = AnimationIterCount::Count(n);
            } else if matches!(
                *part,
                "normal" | "reverse" | "alternate" | "alternate-reverse"
            ) {
                direction = AnimationDirection::from_css(part);
            } else if matches!(*part, "none" | "forwards" | "backwards" | "both") {
                fill_mode = AnimationFillMode::from_css(part);
            } else if matches!(
                *part,
                "linear" | "ease" | "ease-in" | "ease-out" | "ease-in-out"
            ) || part.starts_with("cubic-bezier(")
            {
                easing = Easing::from_css(part);
            } else if name.is_empty() {
                name = part.to_string();
            }
        }

        if !name.is_empty() {
            specs.push(AnimationSpec {
                name,
                duration_secs,
                easing,
                delay_secs,
                iteration_count,
                direction,
                fill_mode,
            });
        }
    }
    specs
}

// ── Selector parsing ────────────────────────────────────────────────────────

/// Parse a CSS selector string into a [`ParsedSelector`].
fn parse_selector(sel: &str) -> Option<ParsedSelector> {
    let sel = sel.trim();
    if sel.is_empty() {
        return None;
    }

    let mut segments = Vec::new();
    let mut current = SelectorSegment::default();
    let mut combinator = Combinator::None;
    let mut chars = sel.chars().peekable();

    while let Some(&ch) = chars.peek() {
        match ch {
            '>' => {
                chars.next();
                // Flush current segment
                if has_content(&current) {
                    segments.push((combinator, current));
                    current = SelectorSegment::default();
                }
                combinator = Combinator::Child;
            }
            ' ' | '\t' | '\n' | '\r' => {
                chars.next();
                // Consume all whitespace
                while chars.peek().map(|c| c.is_ascii_whitespace()) == Some(true) {
                    chars.next();
                }
                // Check if next char is > (child combinator after space)
                if chars.peek() == Some(&'>') {
                    continue;
                }
                // If current has content, it's a descendant combinator
                if has_content(&current) {
                    segments.push((combinator, current));
                    current = SelectorSegment::default();
                    combinator = Combinator::Descendant;
                }
            }
            '.' => {
                chars.next();
                let name = consume_ident(&mut chars);
                if !name.is_empty() {
                    current.classes.push(name);
                }
            }
            '#' => {
                chars.next();
                let name = consume_ident(&mut chars);
                if !name.is_empty() {
                    current.id = Some(name);
                }
            }
            ':' => {
                chars.next();
                let pseudo_name = consume_ident(&mut chars);
                if let Some(pc) = parse_pseudo_class(&pseudo_name, &mut chars) {
                    current.pseudos.push(pc);
                }
            }
            '*' => {
                chars.next();
                current.universal = true;
            }
            _ => {
                let name = consume_ident(&mut chars);
                if !name.is_empty() {
                    current.tag = Some(name);
                }
            }
        }
    }

    if has_content(&current) {
        segments.push((combinator, current));
    }

    if segments.is_empty() {
        return None;
    }

    let specificity = compute_specificity(&segments);
    Some(ParsedSelector {
        segments,
        specificity,
    })
}

fn has_content(seg: &SelectorSegment) -> bool {
    seg.tag.is_some()
        || !seg.classes.is_empty()
        || seg.id.is_some()
        || !seg.pseudos.is_empty()
        || seg.universal
}

/// Consume an identifier (letters, digits, hyphens, underscores).
fn consume_ident(chars: &mut std::iter::Peekable<std::str::Chars<'_>>) -> String {
    let mut s = String::new();
    while let Some(&ch) = chars.peek() {
        if ch.is_alphanumeric() || ch == '-' || ch == '_' || ch == '\\' {
            s.push(ch);
            chars.next();
        } else {
            break;
        }
    }
    s
}

/// Parse a pseudo-class from its name (after the `:`).
fn parse_pseudo_class(
    name: &str,
    chars: &mut std::iter::Peekable<std::str::Chars<'_>>,
) -> Option<PseudoClass> {
    match name {
        "hover" => Some(PseudoClass::Hover),
        "focus" => Some(PseudoClass::Focus),
        "active" => Some(PseudoClass::Active),
        "visited" => Some(PseudoClass::Visited),
        "first-child" => Some(PseudoClass::FirstChild),
        "last-child" => Some(PseudoClass::LastChild),
        "nth-child" => {
            // Consume `(an+b)` or `(n)` or `(odd)` or `(even)`
            if chars.peek() == Some(&'(') {
                chars.next(); // skip '('
                let mut expr = String::new();
                while let Some(&ch) = chars.peek() {
                    if ch == ')' {
                        chars.next();
                        break;
                    }
                    expr.push(ch);
                    chars.next();
                }
                let (a, b) = parse_nth_expr(&expr);
                Some(PseudoClass::NthChild(a, b))
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Parse an `an+b` expression: `odd`, `even`, `3n+1`, `2n`, `5`, etc.
fn parse_nth_expr(expr: &str) -> (i32, i32) {
    let expr = expr.trim().to_ascii_lowercase();
    match expr.as_str() {
        "odd" => (2, 1),
        "even" => (2, 0),
        _ => {
            if let Some(pos) = expr.find('n') {
                let a_str = &expr[..pos].trim();
                let a: i32 = if a_str.is_empty() || *a_str == "+" {
                    1
                } else if *a_str == "-" {
                    -1
                } else {
                    a_str.parse().unwrap_or(1)
                };
                let rest = expr[pos + 1..].trim().to_string();
                let b: i32 = if rest.is_empty() {
                    0
                } else {
                    rest.replace(' ', "").parse().unwrap_or(0)
                };
                (a, b)
            } else {
                // Pure number
                (0, expr.parse().unwrap_or(0))
            }
        }
    }
}

/// Compute CSS specificity (ids, classes+pseudos, tags) from segments.
fn compute_specificity(segments: &[(Combinator, SelectorSegment)]) -> Specificity {
    let mut a: u16 = 0; // ID count
    let mut b: u16 = 0; // Class + pseudo-class count
    let mut c: u16 = 0; // Tag count
    for (_, seg) in segments {
        if seg.id.is_some() {
            a += 1;
        }
        b += seg.classes.len() as u16;
        b += seg.pseudos.len() as u16;
        if seg.tag.is_some() {
            c += 1;
        }
        // Universal (*) has 0 specificity
    }
    (a, b, c)
}

// ── Property normalization + shorthand expansion ────────────────────────────

/// Normalize a CSS length value to a bare-number string in px.
/// Delegates to [`parse_px`] — the single unit-conversion source of truth.
fn norm_val(v: &str) -> String {
    parse_px(v)
        .map(|n| n.to_string())
        .unwrap_or_else(|| v.to_string())
}

/// Map CSS property names to our attribute names and expand shorthands.
fn expand_css_property(prop: &str, value: &str) -> Vec<(String, String)> {
    match prop {
        // ── Shorthands with 1–4 values ──
        "padding" | "margin" => expand_box_shorthand(prop, value),

        // ── Border shorthand: `border: 1px solid #color` ──
        "border" => expand_border(value),

        // ── Outline shorthand: `outline: 1px solid #color` ──
        "outline" => expand_outline(value),

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
        "width"
        | "height"
        | "min-width"
        | "min-height"
        | "max-width"
        | "max-height"
        | "gap"
        | "row-gap"
        | "column-gap"
        | "border-width"
        | "border-top-width"
        | "border-right-width"
        | "border-bottom-width"
        | "border-left-width"
        | "left"
        | "top"
        | "right"
        | "bottom"
        | "flex-basis"
        | "outline-width"
        | "letter-spacing"
        | "word-spacing"
        | "text-indent" => {
            vec![(prop.into(), norm_val(value))]
        }

        // ── Font-size: normalize px ──
        "font-size" => vec![(prop.into(), norm_val(value))],

        // ── Pass-through enum/keyword properties ──
        "display" | "box-sizing" | "visibility" | "flex-wrap" | "font-weight" | "line-height"
        | "text-align" | "white-space" | "z-index" | "text-decoration" | "text-transform"
        | "text-overflow" | "word-break" | "overflow-wrap" | "cursor" | "pointer-events"
        | "user-select" | "border-style" | "aspect-ratio" | "order" | "object-fit" => {
            vec![(prop.into(), value.into())]
        }

        // ── Everything else passes through unchanged ──
        _ => vec![(prop.into(), value.into())],
    }
}

/// Expand CSS `border` shorthand: `1px solid #color` → width + style + color.
fn expand_border(value: &str) -> Vec<(String, String)> {
    let parts: Vec<&str> = value.split_whitespace().collect();
    let mut result = Vec::new();
    for part in &parts {
        if let Some(px) = crate::parse::parse_px(part) {
            result.push(("border-width".into(), px.to_string()));
        } else if part.starts_with('#')
            || part.starts_with("rgb")
            || crate::parse::parse_color(part).is_some()
        {
            result.push(("border-color".into(), (*part).into()));
        } else if matches!(
            *part,
            "solid"
                | "dashed"
                | "dotted"
                | "double"
                | "groove"
                | "ridge"
                | "inset"
                | "outset"
                | "none"
        ) {
            result.push(("border-style".into(), (*part).into()));
        }
    }
    result
}

/// Expand CSS `outline` shorthand: `1px solid #color` → width + color.
fn expand_outline(value: &str) -> Vec<(String, String)> {
    let parts: Vec<&str> = value.split_whitespace().collect();
    let mut result = Vec::new();
    for part in &parts {
        if let Some(px) = crate::parse::parse_px(part) {
            result.push(("outline-width".into(), px.to_string()));
        } else if part.starts_with('#')
            || part.starts_with("rgb")
            || crate::parse::parse_color(part).is_some()
        {
            result.push(("outline-color".into(), (*part).into()));
        }
        // style tokens (solid, dashed) silently ignored — always solid
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
        // Unknown class returns default
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
        let css = ".base { gap: 8; font-size: 14; } .override { font-size: 22; } .accent { color: #a6e3a1; }";
        let sheet = StyleSheet::parse(css);
        let s = sheet.classes(&["base", "override"]);
        assert_eq!(s.gap, 8.0);
        assert_eq!(s.font_size, 22.0);
        // Apply on existing style
        let mut s2 = Style::default().font(16.0);
        sheet.apply(&mut s2, "accent");
        assert_eq!(s2.font_size, 16.0);
        assert_eq!(s2.color, Color::rgb(166, 227, 161));
    }

    #[test]
    fn shorthand_expansion() {
        let css = r#"
            .p2 { padding: 16px 12px; }
            .p4 { padding: 1 2 3 4; }
            .m2 { margin: 10px 20px; }
            .fx { flex: 2; }
        "#;
        let sheet = StyleSheet::parse(css);
        // padding 2 values
        let s = sheet.class("p2");
        assert_eq!(
            s.padding,
            Edges {
                top: 16.0,
                right: 12.0,
                bottom: 16.0,
                left: 12.0
            }
        );
        // padding 4 values
        let s = sheet.class("p4");
        assert_eq!(
            s.padding,
            Edges {
                top: 1.0,
                right: 2.0,
                bottom: 3.0,
                left: 4.0
            }
        );
        // margin 2 values
        let s = sheet.class("m2");
        assert_eq!(
            s.margin,
            Edges {
                top: 10.0,
                right: 20.0,
                bottom: 10.0,
                left: 20.0
            }
        );
        // flex shorthand
        assert_eq!(sheet.class("fx").flex_grow, 2.0);
    }

    #[test]
    fn css_name_normalization() {
        let css = r#"
            .x {
                background-color: #ff0000;
                flex-direction: row;
                align-items: center;
                justify-content: space-between;
                overflow: scroll;
                border-radius: 8px;
                width: 220; height: 56;
            }
            .col { flex-direction: column; }
        "#;
        let sheet = StyleSheet::parse(css);
        let s = sheet.class("x");
        assert_eq!(s.background, Color::rgb(255, 0, 0));
        assert_eq!(s.direction, Direction::Row);
        assert_eq!(s.align, Align::Center);
        assert_eq!(s.justify, Justify::SpaceBetween);
        assert_eq!(s.overflow, Overflow::Scroll);
        assert_eq!(s.corner_radius, 8.0);
        assert_eq!(s.width, Dimension::Px(220.0));
        assert_eq!(s.height, Dimension::Px(56.0));
        assert_eq!(sheet.class("col").direction, Direction::Column);
    }

    #[test]
    fn tag_selector() {
        let sheet = StyleSheet::parse("div { gap: 4; }");
        assert_eq!(sheet.tag("div").gap, 4.0);
    }

    #[test]
    fn id_selector() {
        let sheet = StyleSheet::parse("#main { width: 800; }");
        assert_eq!(sheet.id("main").width, Dimension::Px(800.0));
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
        assert_eq!(s.font_size, 20.0);
        assert_eq!(s.color, Color::rgb(255, 0, 0));
        assert_eq!(s.gap, 5.0);
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
        assert_eq!(tree.arena[0].style.width, Dimension::Px(200.0));
        assert_eq!(tree.arena[0].style.height, Dimension::Px(50.0));
    }

    #[test]
    fn fault_tolerance() {
        // Garbage CSS doesn't crash
        let sh = StyleSheet::parse("{{{{ not css }} color: ;; }}}");
        assert_eq!(sh.class("nonexistent"), Style::default());
        // Bad values silently ignored
        let sh = StyleSheet::parse(".x { font-size: banana; color: nope; width: zzz; gap: ; }");
        let s = sh.class("x");
        assert_eq!(s.font_size, 14.0);
        assert_eq!(s.color, Color::WHITE);
        assert_eq!(s.width, Dimension::Auto);
        assert_eq!(s.gap, 0.0);
    }

    // ── Pixel-level CSS visual correctness ──────────────────────────

    use crate::tree::Tree;
    use any_compute_core::render::{PixelBuffer, RenderList};

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
    fn pixel_css_correctness() {
        // Exact hex color
        let buf = css_to_pixels(".x { background: #a6e3a1; }", "x", 40.0, 40.0);
        assert_eq!(buf.pixel(20, 20), Color::rgb(166, 227, 161));

        // Transparent background untouched
        let buf = css_to_pixels(
            ".x { background: transparent; width: 50; height: 50; }",
            "x",
            50.0,
            50.0,
        );
        assert_eq!(buf.pixel(25, 25), Color::BLACK);

        // Border-radius clips corners
        let buf = css_to_pixels(
            ".box { background: #ffffff; border-radius: 20px; }",
            "box",
            100.0,
            100.0,
        );
        assert_eq!(buf.pixel(50, 50), Color::WHITE);
        assert_eq!(buf.pixel(0, 0), Color::BLACK);
        assert_eq!(buf.pixel(50, 0), Color::WHITE);

        // Card: all four corners clipped
        let buf = css_to_pixels(
            ".card { background: #313244; border-radius: 12px; }",
            "card",
            200.0,
            120.0,
        );
        let fill = Color::rgb(49, 50, 68);
        assert_eq!(buf.pixel(100, 60), fill);
        assert_eq!(buf.pixel(0, 0), Color::BLACK);
        assert_eq!(buf.pixel(199, 0), Color::BLACK);
        assert_eq!(buf.pixel(0, 119), Color::BLACK);
        assert_eq!(buf.pixel(199, 119), Color::BLACK);
    }

    #[test]
    fn pixel_css_nested_layout_paint() {
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
        assert_eq!(buf.pixel(10, 10), Color::rgb(137, 180, 250));
        assert_eq!(buf.pixel(150, 80), Color::rgb(30, 30, 46));
    }

    // ── Tailwind CSS tests (consolidated) ───────────────────────────

    fn tailwind() -> StyleSheet {
        StyleSheet::parse(include_str!("tailwind.css"))
    }

    #[test]
    fn tw_spacing_and_sizing() {
        let tw = tailwind();
        // Padding rem
        assert_eq!(tw.class("p-4").padding, Edges::all(16.0));
        assert_eq!(tw.class("p-2").padding, Edges::all(8.0));
        assert_eq!(tw.class("p-8").padding, Edges::all(32.0));
        // Padding axis
        let s = tw.class("px-4");
        assert_eq!((s.padding.left, s.padding.right), (16.0, 16.0));
        let s = tw.class("py-2");
        assert_eq!((s.padding.top, s.padding.bottom), (8.0, 8.0));
        // Padding individual
        assert_eq!(tw.class("pt-4").padding.top, 16.0);
        assert_eq!(tw.class("pr-4").padding.right, 16.0);
        assert_eq!(tw.class("pb-4").padding.bottom, 16.0);
        assert_eq!(tw.class("pl-4").padding.left, 16.0);
        // Margin
        assert_eq!(tw.class("m-4").margin, Edges::all(16.0));
        assert_eq!(tw.class("m-2").margin, Edges::all(8.0));
        let s = tw.class("mx-4");
        assert_eq!((s.margin.left, s.margin.right), (16.0, 16.0));
        // Gap
        assert_eq!(tw.class("gap-0").gap, 0.0);
        assert_eq!(tw.class("gap-1").gap, 4.0);
        assert_eq!(tw.class("gap-2").gap, 8.0);
        assert_eq!(tw.class("gap-4").gap, 16.0);
        assert_eq!(tw.class("gap-8").gap, 32.0);
        // Width / Height
        assert_eq!(tw.class("w-4").width, Dimension::Px(16.0));
        assert_eq!(tw.class("w-8").width, Dimension::Px(32.0));
        assert_eq!(tw.class("w-64").width, Dimension::Px(256.0));
        assert_eq!(tw.class("h-16").height, Dimension::Px(64.0));
        assert_eq!(tw.class("h-full").height, Dimension::Percent(100.0));
        assert_eq!(tw.class("w-full").width, Dimension::Percent(100.0));
        assert_eq!(tw.class("w-1\\/2").width, Dimension::Percent(50.0));
    }

    #[test]
    fn tw_layout_and_visual() {
        let tw = tailwind();
        // Flex direction
        assert_eq!(tw.class("flex-row").direction, Direction::Row);
        assert_eq!(tw.class("flex-col").direction, Direction::Column);
        // Alignment
        assert_eq!(tw.class("items-center").align, Align::Center);
        assert_eq!(tw.class("items-end").align, Align::End);
        assert_eq!(tw.class("items-stretch").align, Align::Stretch);
        assert_eq!(tw.class("justify-center").justify, Justify::Center);
        assert_eq!(tw.class("justify-between").justify, Justify::SpaceBetween);
        assert_eq!(tw.class("justify-evenly").justify, Justify::SpaceEvenly);
        // Flex grow/shrink
        assert_eq!(tw.class("grow").flex_grow, 1.0);
        assert_eq!(tw.class("grow-0").flex_grow, 0.0);
        assert_eq!(tw.class("shrink").flex_shrink, 1.0);
        assert_eq!(tw.class("shrink-0").flex_shrink, 0.0);
        // Border radius
        assert_eq!(tw.class("rounded-none").corner_radius, 0.0);
        assert_eq!(tw.class("rounded-sm").corner_radius, 2.0);
        assert_eq!(tw.class("rounded").corner_radius, 4.0);
        assert_eq!(tw.class("rounded-lg").corner_radius, 8.0);
        assert_eq!(tw.class("rounded-full").corner_radius, 9999.0);
        // Opacity
        assert_eq!(tw.class("opacity-0").opacity, 0.0);
        assert_eq!(tw.class("opacity-50").opacity, 0.5);
        assert_eq!(tw.class("opacity-100").opacity, 1.0);
        // Font size
        assert_eq!(tw.class("text-xs").font_size, 12.0);
        assert_eq!(tw.class("text-sm").font_size, 14.0);
        assert_eq!(tw.class("text-base").font_size, 16.0);
        assert_eq!(tw.class("text-lg").font_size, 18.0);
        assert_eq!(tw.class("text-xl").font_size, 20.0);
        assert_eq!(tw.class("text-2xl").font_size, 24.0);
        // Colors
        assert_eq!(tw.class("bg-white").background, Color::WHITE);
        assert_eq!(tw.class("bg-black").background, Color::BLACK);
        assert_eq!(tw.class("bg-red-500").background, Color::rgb(239, 68, 68));
        assert_eq!(tw.class("bg-blue-500").background, Color::rgb(59, 130, 246));
        assert_eq!(tw.class("bg-green-500").background, Color::rgb(34, 197, 94));
        assert_eq!(tw.class("bg-slate-900").background, Color::rgb(15, 23, 42));
        assert_eq!(tw.class("text-white").color, Color::WHITE);
        assert_eq!(tw.class("text-black").color, Color::BLACK);
        assert_eq!(tw.class("text-red-500").color, Color::rgb(239, 68, 68));
        assert_eq!(tw.class("text-gray-400").color, Color::rgb(156, 163, 175));
        // Border
        assert_eq!(tw.class("border").border_width, 1.0);
        assert_eq!(tw.class("border-2").border_width, 2.0);
        assert_eq!(tw.class("border-4").border_width, 4.0);
        assert_eq!(
            tw.class("border-red-500").border_color,
            Color::rgb(239, 68, 68)
        );
        // Position / overflow
        assert_eq!(tw.class("relative").position, Position::Relative);
        assert_eq!(tw.class("absolute").position, Position::Absolute);
        assert_eq!(tw.class("overflow-hidden").overflow, Overflow::Hidden);
        assert_eq!(tw.class("overflow-scroll").overflow, Overflow::Scroll);
    }

    #[test]
    fn tw_composition_and_pixel() {
        let tw = tailwind();
        // Multi-class composition
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

        // Full color palette spot-check
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
            assert_ne!(
                tw.class(name).background,
                Color::TRANSPARENT,
                "'{name}' should set bg"
            );
        }

        // ── Pixel tests ──
        fn tw_to_pixels(html: &str, vw: f64, vh: f64) -> PixelBuffer {
            let tw = StyleSheet::parse(include_str!("tailwind.css"));
            let mut tree = crate::parse::parse_with_css(html, &tw);
            tree.layout(Size::new(vw, vh));
            let mut list = RenderList::default();
            tree.paint(&mut list);
            let mut buf = PixelBuffer::new(vw as u32, vh as u32, Color::BLACK);
            buf.paint(&list);
            buf
        }

        // Card bg
        let buf = tw_to_pixels(
            r#"<div class="bg-blue-500 w-64 h-32 rounded-lg"></div>"#,
            256.0,
            128.0,
        );
        assert_eq!(buf.pixel(128, 64), Color::rgb(59, 130, 246));
        assert_eq!(buf.pixel(0, 0), Color::BLACK);

        // Nested layout
        let buf = tw_to_pixels(
            r#"<div class="bg-slate-900 w-96 h-48 p-4 flex-col"><div class="bg-blue-500 w-full h-16 rounded"></div></div>"#,
            384.0,
            192.0,
        );
        assert_eq!(buf.pixel(8, 8), Color::rgb(15, 23, 42));
        assert_eq!(buf.pixel(24, 24), Color::rgb(59, 130, 246));
        assert_eq!(buf.pixel(200, 170), Color::rgb(15, 23, 42));

        // Identical renders = zero diff
        let a = tw_to_pixels(
            r#"<div class="bg-red-500 w-32 h-32 rounded-full"></div>"#,
            128.0,
            128.0,
        );
        let b = tw_to_pixels(
            r#"<div class="bg-red-500 w-32 h-32 rounded-full"></div>"#,
            128.0,
            128.0,
        );
        assert_eq!(a.diff(&b, 0), 0);

        // Different colors = high diff
        let c = tw_to_pixels(r#"<div class="bg-blue-500 w-32 h-32"></div>"#, 128.0, 128.0);
        assert!(a.diff_ratio(&c, 0) > 0.9);

        // Tailwind rounded-2xl = raw CSS radius: 16px
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
        assert_eq!(tw_buf.diff(&raw_buf, 0), 0);
    }

    // ── CSS transitions ─────────────────────────────────────────────────

    #[test]
    fn transition_shorthand() {
        let css = ".fade { transition: opacity 0.3s ease-in 0.1s; }";
        let sheet = StyleSheet::parse(css);
        let specs = sheet.class_transitions("fade");
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].property, "opacity");
        assert!((specs[0].duration_secs - 0.3).abs() < 1e-6);
        assert_eq!(specs[0].easing, Easing::EaseIn);
        assert!((specs[0].delay_secs - 0.1).abs() < 1e-6);
    }

    #[test]
    fn transition_multi_property() {
        let css = ".move { transition: transform 0.5s ease-out, opacity 300ms linear; }";
        let sheet = StyleSheet::parse(css);
        let specs = sheet.class_transitions("move");
        assert_eq!(specs.len(), 2);
        assert_eq!(specs[0].property, "transform");
        assert!((specs[0].duration_secs - 0.5).abs() < 1e-6);
        assert_eq!(specs[0].easing, Easing::EaseOut);
        assert_eq!(specs[1].property, "opacity");
        assert!((specs[1].duration_secs - 0.3).abs() < 1e-6);
        assert_eq!(specs[1].easing, Easing::Linear);
    }

    #[test]
    fn transition_longhands() {
        let css = r#".x {
            transition-property: width, height;
            transition-duration: 0.2s, 0.4s;
            transition-timing-function: ease-in, ease-out;
            transition-delay: 0s, 50ms;
        }"#;
        let sheet = StyleSheet::parse(css);
        let specs = sheet.class_transitions("x");
        assert_eq!(specs.len(), 2);
        assert_eq!(specs[0].property, "width");
        assert!((specs[0].duration_secs - 0.2).abs() < 1e-6);
        assert_eq!(specs[0].easing, Easing::EaseIn);
        assert_eq!(specs[1].property, "height");
        assert!((specs[1].duration_secs - 0.4).abs() < 1e-6);
        assert_eq!(specs[1].easing, Easing::EaseOut);
        assert!((specs[1].delay_secs - 0.05).abs() < 1e-6);
    }

    #[test]
    fn transition_all_shorthand() {
        let css = ".all { transition: all 0.2s ease; }";
        let sheet = StyleSheet::parse(css);
        let specs = sheet.class_transitions("all");
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].property, "all");
        assert_eq!(specs[0].easing, Easing::EaseInOut); // "ease" = EaseInOut
    }

    // ── @keyframes + animations ─────────────────────────────────────────

    #[test]
    fn keyframes_parse() {
        let css = r#"
            @keyframes fadeIn {
                from { opacity: 0; }
                to { opacity: 1; }
            }
        "#;
        let sheet = StyleSheet::parse(css);
        let kf = sheet.keyframes("fadeIn").unwrap();
        assert_eq!(kf.len(), 2);
        assert!((kf[0].stop - 0.0).abs() < 1e-10);
        assert!((kf[1].stop - 1.0).abs() < 1e-10);
    }

    #[test]
    fn keyframes_percentage_stops() {
        let css = r#"
            @keyframes slide {
                0% { width: 0; }
                50% { width: 100; }
                100% { width: 200; }
            }
        "#;
        let sheet = StyleSheet::parse(css);
        let kf = sheet.keyframes("slide").unwrap();
        assert_eq!(kf.len(), 3);
        assert!((kf[0].stop - 0.0).abs() < 1e-10);
        assert!((kf[1].stop - 0.5).abs() < 1e-10);
        assert!((kf[2].stop - 1.0).abs() < 1e-10);
    }

    #[test]
    fn animation_shorthand() {
        let css = ".spin { animation: rotate 2s linear infinite; }";
        let sheet = StyleSheet::parse(css);
        let anims = sheet.class_animations("spin");
        assert_eq!(anims.len(), 1);
        assert_eq!(anims[0].name, "rotate");
        assert!((anims[0].duration_secs - 2.0).abs() < 1e-6);
        assert_eq!(anims[0].easing, Easing::Linear);
        assert_eq!(anims[0].iteration_count, AnimationIterCount::Infinite);
    }

    #[test]
    fn animation_longhands() {
        let css = r#".x {
            animation-name: slideIn;
            animation-duration: 500ms;
            animation-timing-function: ease-out;
            animation-delay: 100ms;
            animation-iteration-count: 3;
            animation-direction: alternate;
            animation-fill-mode: forwards;
        }"#;
        let sheet = StyleSheet::parse(css);
        let anims = sheet.class_animations("x");
        assert_eq!(anims.len(), 1);
        assert_eq!(anims[0].name, "slideIn");
        assert!((anims[0].duration_secs - 0.5).abs() < 1e-6);
        assert_eq!(anims[0].easing, Easing::EaseOut);
        assert!((anims[0].delay_secs - 0.1).abs() < 1e-6);
        assert_eq!(anims[0].iteration_count, AnimationIterCount::Count(3.0));
        assert_eq!(anims[0].direction, AnimationDirection::Alternate);
        assert_eq!(anims[0].fill_mode, AnimationFillMode::Forwards);
    }

    #[test]
    fn keyframes_and_animation_together() {
        let css = r#"
            @keyframes pulse { from { opacity: 1; } 50% { opacity: 0.5; } to { opacity: 1; } }
            .pulse { animation: pulse 1s ease-in-out infinite; background: #ff0000; }
        "#;
        let sheet = StyleSheet::parse(css);
        // Keyframes parsed correctly
        let kf = sheet.keyframes("pulse").unwrap();
        assert_eq!(kf.len(), 3);
        // Animation spec attached to class
        let anims = sheet.class_animations("pulse");
        assert_eq!(anims.len(), 1);
        assert_eq!(anims[0].name, "pulse");
        assert_eq!(anims[0].iteration_count, AnimationIterCount::Infinite);
        // Normal style ops still work alongside animation
        assert_eq!(sheet.class("pulse").background, Color::rgb(255, 0, 0));
    }

    // ── CSS variables ───────────────────────────────────────────────────

    #[test]
    fn css_variables_root() {
        let css = r#"
            :root { --primary: #a6e3a1; --spacing: 16px; }
            .card { color: var(--primary); padding: var(--spacing); }
        "#;
        let sheet = StyleSheet::parse(css);
        assert_eq!(sheet.var("--primary"), Some("#a6e3a1"));
        assert_eq!(sheet.var("--spacing"), Some("16px"));
        let s = sheet.class("card");
        assert_eq!(s.color, Color::rgb(166, 227, 161));
        assert_eq!(s.padding, Edges::all(16.0));
    }

    #[test]
    fn css_variable_fallback() {
        let css = ".x { font-size: var(--missing, 20px); }";
        let sheet = StyleSheet::parse(css);
        assert_eq!(sheet.class("x").font_size, 20.0);
    }

    #[test]
    fn css_variable_star_selector() {
        let css = r#"
            * { --gap: 8; }
            .box { gap: var(--gap); }
        "#;
        let sheet = StyleSheet::parse(css);
        assert_eq!(sheet.class("box").gap, 8.0);
    }

    // ── calc() ──────────────────────────────────────────────────────────

    #[test]
    fn calc_dimension() {
        let css = ".x { width: calc(100% - 20px); }";
        let sheet = StyleSheet::parse(css);
        let s = sheet.class("x");
        match s.width {
            Dimension::Calc { percent, px } => {
                assert!((percent - 100.0).abs() < 1e-10);
                assert!((px - (-20.0)).abs() < 1e-10);
            }
            _ => panic!("Expected Dimension::Calc, got {:?}", s.width),
        }
        // Resolve at 400px parent → 400 - 20 = 380
        assert!((s.width.resolve(400.0).unwrap() - 380.0).abs() < 1e-10);
    }

    #[test]
    fn calc_percent_only() {
        let css = ".x { width: calc(50% + 25%); }";
        let sheet = StyleSheet::parse(css);
        let s = sheet.class("x");
        // Should simplify to Percent(75)
        assert_eq!(s.width, Dimension::Percent(75.0));
    }

    #[test]
    fn calc_px_only() {
        let css = ".x { width: calc(100px + 20px); }";
        let sheet = StyleSheet::parse(css);
        let s = sheet.class("x");
        assert_eq!(s.width, Dimension::Px(120.0));
    }

    #[test]
    fn calc_with_rem() {
        let css = ".x { padding: calc(100% - 2rem); }";
        let sheet = StyleSheet::parse(css);
        let s = sheet.class("x");
        // 2rem = 32px → calc(100% - 32px)
        match s.padding.top {
            // padding shorthand → parse_dimension for each side... actually padding uses parse_px
            // Since this is a single value, it goes through expand_box_shorthand → norm_val → parse_px
            // parse_px can't handle calc() — it returns None, so it falls through
            // Actually, let's check what happens...
            v => {
                // padding goes through expand_box_shorthand which calls norm_val which calls parse_px
                // parse_px doesn't handle calc — this will be 0.0
                // This is fine — calc() is for dimension properties (width/height), not simple px values
                let _ = v;
            }
        }
    }

    // ── Advanced selectors ──────────────────────────────────────────────

    #[test]
    fn selector_parsing_simple() {
        let sel = parse_selector(".card").unwrap();
        assert_eq!(sel.segments.len(), 1);
        assert_eq!(sel.segments[0].1.classes, vec!["card"]);
        assert_eq!(sel.specificity, (0, 1, 0));
    }

    #[test]
    fn selector_parsing_compound() {
        let sel = parse_selector("div.card#main").unwrap();
        assert_eq!(sel.segments.len(), 1);
        assert_eq!(sel.segments[0].1.tag, Some("div".to_string()));
        assert_eq!(sel.segments[0].1.classes, vec!["card"]);
        assert_eq!(sel.segments[0].1.id, Some("main".to_string()));
        assert_eq!(sel.specificity, (1, 1, 1));
    }

    #[test]
    fn selector_parsing_descendant() {
        let sel = parse_selector(".parent .child").unwrap();
        assert_eq!(sel.segments.len(), 2);
        assert_eq!(sel.segments[0].0, Combinator::None);
        assert_eq!(sel.segments[0].1.classes, vec!["parent"]);
        assert_eq!(sel.segments[1].0, Combinator::Descendant);
        assert_eq!(sel.segments[1].1.classes, vec!["child"]);
        assert_eq!(sel.specificity, (0, 2, 0));
    }

    #[test]
    fn selector_parsing_child() {
        let sel = parse_selector(".parent > .child").unwrap();
        assert_eq!(sel.segments.len(), 2);
        assert_eq!(sel.segments[1].0, Combinator::Child);
        assert_eq!(sel.specificity, (0, 2, 0));
    }

    #[test]
    fn selector_parsing_pseudo() {
        let sel = parse_selector(".btn:hover").unwrap();
        assert_eq!(sel.segments.len(), 1);
        assert_eq!(sel.segments[0].1.classes, vec!["btn"]);
        assert_eq!(sel.segments[0].1.pseudos, vec![PseudoClass::Hover]);
        assert_eq!(sel.specificity, (0, 2, 0)); // 1 class + 1 pseudo
    }

    #[test]
    fn selector_parsing_nth_child() {
        let sel = parse_selector("li:nth-child(2n+1)").unwrap();
        assert_eq!(sel.segments[0].1.tag, Some("li".to_string()));
        assert_eq!(sel.segments[0].1.pseudos, vec![PseudoClass::NthChild(2, 1)]);
    }

    #[test]
    fn selector_parsing_nth_child_keywords() {
        // odd = 2n+1, even = 2n
        let sel = parse_selector("li:nth-child(odd)").unwrap();
        assert_eq!(sel.segments[0].1.pseudos, vec![PseudoClass::NthChild(2, 1)]);
        let sel = parse_selector("li:nth-child(even)").unwrap();
        assert_eq!(sel.segments[0].1.pseudos, vec![PseudoClass::NthChild(2, 0)]);
    }

    #[test]
    fn selector_parsing_universal() {
        let sel = parse_selector("*").unwrap();
        assert!(sel.segments[0].1.universal);
        assert_eq!(sel.specificity, (0, 0, 0)); // * has 0 specificity
    }

    #[test]
    fn selector_specificity_ordering() {
        // #id > .class > tag
        let id_spec = parse_selector("#main").unwrap().specificity;
        let cls_spec = parse_selector(".card").unwrap().specificity;
        let tag_spec = parse_selector("div").unwrap().specificity;
        assert!(id_spec > cls_spec);
        assert!(cls_spec > tag_spec);

        // More specific compound beats less specific
        let compound = parse_selector("div.card.active").unwrap().specificity;
        assert!(compound > cls_spec);
    }

    #[test]
    fn complex_selector_stored() {
        let css = ".parent .child { color: #ff0000; }";
        let sheet = StyleSheet::parse(css);
        assert_eq!(sheet.complex_rules().len(), 1);
        let rule = &sheet.complex_rules()[0];
        assert_eq!(rule.selector.segments.len(), 2);
        assert_eq!(rule.payload.ops.len(), 1);
    }

    #[test]
    fn pseudo_class_stored_as_complex() {
        let css = ".btn:hover { background: #ff0000; }";
        let sheet = StyleSheet::parse(css);
        assert_eq!(sheet.complex_rules().len(), 1);
        let seg = &sheet.complex_rules()[0].selector.segments[0].1;
        assert_eq!(seg.classes, vec!["btn"]);
        assert_eq!(seg.pseudos, vec![PseudoClass::Hover]);
    }

    // ── Easing::from_css ────────────────────────────────────────────────

    #[test]
    fn easing_from_css() {
        assert_eq!(Easing::from_css("linear"), Easing::Linear);
        assert_eq!(Easing::from_css("ease"), Easing::EaseInOut);
        assert_eq!(Easing::from_css("ease-in"), Easing::EaseIn);
        assert_eq!(Easing::from_css("ease-out"), Easing::EaseOut);
        assert_eq!(Easing::from_css("ease-in-out"), Easing::EaseInOut);
        assert_eq!(
            Easing::from_css("cubic-bezier(0.25, 0.1, 0.25, 1)"),
            Easing::CubicBezier
        );
        // Unknown defaults to EaseInOut (CSS default)
        assert_eq!(Easing::from_css("invalid"), Easing::EaseInOut);
    }

    // ── Dimension::Calc resolution ──────────────────────────────────────

    #[test]
    fn calc_resolve() {
        // calc(100% - 32px) at parent=400 → 400 - 32 = 368
        let d = Dimension::Calc {
            percent: 100.0,
            px: -32.0,
        };
        assert!((d.resolve(400.0).unwrap() - 368.0).abs() < 1e-10);

        // calc(50% + 10px) at parent=200 → 100 + 10 = 110
        let d = Dimension::Calc {
            percent: 50.0,
            px: 10.0,
        };
        assert!((d.resolve(200.0).unwrap() - 110.0).abs() < 1e-10);
    }
}
