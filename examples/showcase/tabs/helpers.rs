//! Shared helpers for tab modules — single source for CSS shorthand and formatting.

use any_compute_core::render::Color;
use any_compute_dom::css::StyleSheet;
use any_compute_dom::style::Style;
use any_compute_dom::theme;
use any_compute_dom::tree::*;

/// Fallback HTML shown when a page panics during load or render.
pub const ERROR_PAGE_HTML: &str = "<h1>Error</h1><p>Page failed to load</p>";

/// Resolve one CSS class from sheet.
pub fn s(sheet: &StyleSheet, class: &str) -> Style {
    sheet.class(class)
}

/// Resolve + merge multiple CSS classes from sheet.
pub fn sm(sheet: &StyleSheet, classes: &[&str]) -> Style {
    sheet.classes(classes)
}

/// Badge with a color variant: `badge(sheet, "green")` → `sm(sheet, &["badge", "badge-green"])`.
pub fn badge(sheet: &StyleSheet, variant: &str) -> Style {
    sm(sheet, &["badge", &format!("badge-{variant}")])
}

/// Build an `ai-{prefix}-{label}` tag, normalizing the label to lowercase + dashes.
pub fn ai_tag(prefix: &str, label: &str) -> String {
    format!("ai-{prefix}-{}", label.to_lowercase().replace(' ', "-"))
}

/// Key-value card: label + value in a small card surface.
pub fn kv_card(t: &mut Tree, parent: NodeId, sheet: &StyleSheet, label: &str, value: &str) {
    let c = t.add_box(parent, sheet.class("card-sm"));
    t.add_text(c, label, sheet.class("label"));
    t.add_text(c, value, sheet.classes(&["font-14", "text"]));
}

/// Format a numeric value with human-readable SI suffixes.
/// e.g. `format_human(1_500_000.0, "ops/s")` → `"1.5M ops/s"`.
pub fn format_human(value: f64, unit: &str) -> String {
    if value >= 1_000_000.0 {
        format!("{:.1}M {unit}", value / 1_000_000.0)
    } else if value >= 1_000.0 {
        format!("{:.1}K {unit}", value / 1_000.0)
    } else {
        format!("{:.0} {unit}", value)
    }
}

/// Format ops/sec using `format_human`.
pub fn format_ops(ops: f64) -> String {
    format_human(ops, "ops/s")
}

/// Build a horizontal subtab bar with pill buttons.
///
/// Tags each button as `"{tag_prefix}{index}"`.
/// Empty `labels` is a no-op (tabs without subtabs).
pub fn build_subtab_bar(
    sheet: &StyleSheet,
    t: &mut Tree,
    parent: NodeId,
    labels: &[&str],
    active: usize,
    tag_prefix: &str,
) {
    if labels.is_empty() {
        return;
    }
    let bar = t.add_box(parent, s(sheet, "subtab-bar"));
    for (i, &label) in labels.iter().enumerate() {
        let is_active = i == active;
        let btn_style = s(sheet, "subtab-btn").bg(if is_active {
            theme::SURFACE0
        } else {
            Color::TRANSPARENT
        });
        let btn = t.add_box(bar, btn_style);
        let text_color = if is_active {
            theme::ACCENT
        } else {
            theme::TEXT_DIM
        };
        t.add_text(btn, label, s(sheet, "font-11").color(text_color));
        t.tag(btn, &format!("{tag_prefix}{i}"));
    }
}
