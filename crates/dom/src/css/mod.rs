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

use crate::parse::{apply_style_attrs, compile_declaration, inject_flex_row, parse_px, parse_time};
use crate::style::{DEFAULT_EASING, Style, apply_ops};
use any_compute_core::animation::Easing;

// ── CSS transition + animation metadata ─────────────────────────────────────

/// Parsed CSS transition declaration: `transition: property duration easing delay`.
///
/// Stored per-selector alongside [`StyleOp`]s. At runtime, when a style change
/// occurs on a matching element, the transition system uses this spec to create
/// a [`Transition<T>`] in the animation engine.

mod rule;
pub use rule::*;

#[derive(Clone)]
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

/// Browser-like user-agent defaults for HTML tags — loaded from `ua.css`.
/// Built-in user-agent stylesheet defaults for HTML elements (button, h1, etc.).
pub const UA_CSS: &str = include_str!("../ua.css");

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
            let start_i = i; // guard against infinite loops
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
                    // Parse comma-separated stops (e.g. "0%, 100%")
                    let payload = compile_declarations(body, &variables);
                    for part in stop_text.split(',') {
                        let part = part.trim();
                        let stop = match part {
                            "from" => Some(0.0),
                            "to" => Some(1.0),
                            s if s.ends_with('%') => s[..s.len() - 1]
                                .trim()
                                .parse::<f64>()
                                .ok()
                                .map(|v| v / 100.0),
                            _ => None,
                        };
                        if let Some(stop_val) = stop {
                            kf_list.push(Keyframe {
                                stop: stop_val,
                                ops: payload.ops.clone(),
                            });
                        }
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

            // Safety: ensure progress — if i didn't advance, skip one byte.
            if i == start_i {
                i += 1;
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
        // 0. Universal selector `*`
        if let Some(rp) = self.tags.get("*") {
            apply_ops(&mut s, &rp.ops);
        }
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

        payload.ops.extend(compile_declaration(&prop, value));
    }

    inject_flex_row(&mut payload.ops);

    // ── Assemble transition longhands ──
    if let Some(props) = tr_properties {
        let durs = tr_durations.unwrap_or_default();
        let easings = tr_easings.unwrap_or_default();
        let delays = tr_delays.unwrap_or_default();
        for (i, prop) in props.into_iter().enumerate() {
            payload.transitions.push(TransitionSpec {
                property: prop,
                duration_secs: nth_or_first(&durs, i, 0.0),
                easing: nth_or_first(&easings, i, DEFAULT_EASING),
                delay_secs: nth_or_first(&delays, i, 0.0),
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
                duration_secs: nth_or_first(&durs, i, 0.0),
                easing: nth_or_first(&easings, i, DEFAULT_EASING),
                delay_secs: nth_or_first(&delays, i, 0.0),
                iteration_count: nth_or_first(&iters, i, AnimationIterCount::default()),
                direction: nth_or_first(&dirs, i, AnimationDirection::default()),
                fill_mode: nth_or_first(&fills, i, AnimationFillMode::default()),
            });
        }
    }

    payload
}

// ── CSS longhand assembly helpers ────────────────────────────────────────────

/// CSS longhand list lookup: `slice[i]`, falling back to `slice[0]`, then `default`.
///
/// Matches CSS spec behavior for shorthand lists: when fewer values are specified
/// than properties, the list cycles from the beginning.
fn nth_or_first<T: Copy>(slice: &[T], i: usize, default: T) -> T {
    slice.get(i).or(slice.first()).copied().unwrap_or(default)
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
            .unwrap_or(DEFAULT_EASING);
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
        let mut easing = DEFAULT_EASING;
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
            '[' => {
                // Attribute selector — skip until matching ']'
                chars.next();
                while let Some(&c) = chars.peek() {
                    chars.next();
                    if c == ']' {
                        break;
                    }
                }
            }
            '+' | '~' => {
                // Adjacent / general sibling combinators
                chars.next();
                if has_content(&current) {
                    segments.push((combinator, current));
                    current = SelectorSegment::default();
                }
                combinator = Combinator::Descendant; // treat as descendant
            }
            ',' => {
                // Shouldn't reach here (split by caller), but skip to be safe
                chars.next();
            }
            _ => {
                let name = consume_ident(&mut chars);
                if name.is_empty() {
                    // Unknown char — skip it to avoid infinite loop
                    chars.next();
                } else {
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
pub(crate) fn expand_css_property(prop: &str, value: &str) -> Vec<(String, String)> {
    match prop {
        // ── Shorthands with 1–4 values ──
        "padding" | "margin" => expand_box_shorthand(prop, value),

        // ── Border shorthand: `border: 1px solid #color` ──
        "border" => expand_border(value),

        // ── Per-side border shorthands: `border-top: 1px solid #color` ──
        "border-top" | "border-right" | "border-bottom" | "border-left" => {
            expand_side_border(prop, value)
        }

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
        | "outline-offset"
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
        | "user-select" | "border-style" | "outline-style" | "aspect-ratio" | "order"
        | "object-fit" => {
            vec![(prop.into(), value.into())]
        }

        // ── Everything else passes through unchanged ──
        _ => vec![(prop.into(), value.into())],
    }
}

/// Expand CSS `border` shorthand: `1px solid #color` → width + style + color.
/// Expand a CSS box-decoration shorthand (`border`, `outline`) into longhand pairs.
/// `prefix` is e.g. `"border"` or `"outline"`. `include_style` enables `border-style`.
fn expand_box_decl(prefix: &str, value: &str, include_style: bool) -> Vec<(String, String)> {
    let mut result = Vec::new();
    for part in value.split_whitespace() {
        if let Some(px) = crate::parse::parse_px(part) {
            result.push((format!("{prefix}-width"), px.to_string()));
        } else if part.starts_with('#')
            || part.starts_with("rgb")
            || crate::parse::parse_color(part).is_some()
        {
            result.push((format!("{prefix}-color"), part.into()));
        } else if include_style
            && matches!(
                part,
                "solid"
                    | "dashed"
                    | "dotted"
                    | "double"
                    | "groove"
                    | "ridge"
                    | "inset"
                    | "outset"
                    | "none"
            )
        {
            result.push((format!("{prefix}-style"), part.into()));
        }
    }
    result
}

fn expand_border(value: &str) -> Vec<(String, String)> {
    expand_box_decl("border", value, true)
}

/// Expand `border-top/right/bottom/left: 1px solid #color` into longhands.
/// Width goes to side-specific (`border-top-width`), color/style go to global
/// (`border-color`, `border-style`) since the style system has no per-side color.
fn expand_side_border(prop: &str, value: &str) -> Vec<(String, String)> {
    let mut result = Vec::new();
    for part in value.split_whitespace() {
        if let Some(px) = crate::parse::parse_px(part) {
            result.push((format!("{prop}-width"), px.to_string()));
        } else if part.starts_with('#')
            || part.starts_with("rgb")
            || crate::parse::parse_color(part).is_some()
        {
            result.push(("border-color".into(), part.into()));
        } else if matches!(
            part,
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
            result.push(("border-style".into(), part.into()));
        }
    }
    result
}

fn expand_outline(value: &str) -> Vec<(String, String)> {
    expand_box_decl("outline", value, true)
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
#[path = "../css_tests.rs"]
mod tests;
