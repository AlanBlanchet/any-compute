//! Shared helpers for tab modules — single source for CSS shorthand and formatting.

use any_compute_dom::css::StyleSheet;
use any_compute_dom::style::Style;
use any_compute_dom::tree::*;

/// Resolve one CSS class from sheet.
pub fn s(sheet: &StyleSheet, class: &str) -> Style {
    sheet.class(class)
}

/// Resolve + merge multiple CSS classes from sheet.
pub fn sm(sheet: &StyleSheet, classes: &[&str]) -> Style {
    sheet.classes(classes)
}

/// Key-value card: label + value in a small card surface.
pub fn kv_card(t: &mut Tree, parent: NodeId, sheet: &StyleSheet, label: &str, value: &str) {
    let c = t.add_box(parent, sheet.class("card-sm"));
    t.add_text(c, label, sheet.class("label"));
    t.add_text(c, value, sheet.classes(&["font-14", "text"]));
}

/// Format ops/sec: "1.2M ops/s", "45.3K ops/s", "800 ops/s".
pub fn format_ops(ops: f64) -> String {
    if ops >= 1_000_000.0 {
        format!("{:.1}M ops/s", ops / 1_000_000.0)
    } else if ops >= 1_000.0 {
        format!("{:.1}K ops/s", ops / 1_000.0)
    } else {
        format!("{:.0} ops/s", ops)
    }
}
