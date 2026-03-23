//! DOM Website tab — renders the parsed HTML/CSS website tree.
//!
//! Shows a fully parsed website with animations, hover, z-index, scroll,
//! flexbox layout, CSS transitions and @keyframes — all rendered through
//! our Tree→RenderList→Gpu pipeline.

use any_compute_dom::css::StyleSheet;
use any_compute_dom::tree::*;
use any_compute_dom::Overflow;

use super::helpers::{s, sm};
use crate::AppData;

pub fn build(sheet: &StyleSheet, t: &mut Tree, parent: NodeId, data: &AppData) {
    t.add_text(parent, "DOM Website Demo", s(sheet, "title"));
    t.add_text(
        parent,
        "Full HTML/CSS parsing — animations, hover, z-index, overflow, transitions, flexbox",
        s(sheet, "subtitle"),
    );
    t.add_box(parent, s(sheet, "spacer-12"));

    // Info badges
    let info = t.add_box(parent, sm(sheet, &["row-gap-8"]));
    if let Some(tree) = &data.dom_tree {
        let nodes = tree.arena.len();
        t.add_text(info, &format!("{nodes} nodes"), sm(sheet, &["badge", "badge-green"]));
    }
    t.add_text(info, "HTML parsed", sm(sheet, &["badge", "badge-blue"]));
    t.add_text(info, "CSS matched", sm(sheet, &["badge", "badge-blue"]));
    t.add_text(info, "Animations active", sm(sheet, &["badge", "badge-yellow"]));

    t.add_box(parent, s(sheet, "spacer-12"));

    // The parsed DOM tree is rendered separately in the event loop
    // (it has its own animations/hover/focus). Show a container for it.
    let container = t.add_box(
        parent,
        s(sheet, "card").h(500.0).overflow(Overflow::Hidden),
    );
    t.add_text(
        container,
        "Interactive website rendered below — hover cards, click buttons, watch animations",
        s(sheet, "subtitle"),
    );

    // Feature checklist
    t.add_box(parent, s(sheet, "spacer-12"));
    t.add_text(parent, "Features Exercised", sm(sheet, &["heading", "text"]));
    let features = [
        ("Flexbox", "Row, column, wrap, gap, grow, shrink, align"),
        ("CSS Parsing", "Selectors, cascade, specificity, variables, calc()"),
        ("Animations", "@keyframes pulse, slide-in, color-cycle, glow"),
        ("Transitions", "hover border-color, background 0.3s ease"),
        ("Events", "Hover, active, focus, click dispatch with tags"),
        ("Z-Index", "Stacking layers with z-index: 1, 2, 3"),
        ("Overflow", "Scrollable nested containers"),
        ("Data Grid", "Table-like layout with header + rows"),
    ];
    for (label, desc) in &features {
        let row = t.add_box(parent, sm(sheet, &["row-gap-8"]));
        t.add_text(row, *label, sm(sheet, &["font-13", "blue"]).w(120.0));
        t.add_text(row, *desc, s(sheet, "body"));
    }
}
