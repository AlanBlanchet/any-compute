use super::node::{NodeId, NodeKind, Slot};
use super::Tree;
use crate::style::*;
use any_compute_core::interaction::TextInput;
use any_compute_core::render::Color;

// ═══════════════════════════════════════════════════════════════════════════
// ── Address bar builder (shared by showcase + visual tests) ─────────────
// ═══════════════════════════════════════════════════════════════════════════

/// Style params for the address bar.  Callers supply their own font size and
/// color palette — no dependency on CSS helpers or the `theme` module.
pub struct AddressBarStyle {
    pub font_size: f64,
    pub bg: Color,
    pub border_focused: Color,
    pub border_unfocused: Color,
    pub text_color: Color,
    pub protocol_color: Color,
    pub selection_bg: Color,
    pub selection_text: Color,
    pub cursor_color: Color,
}

impl Default for AddressBarStyle {
    fn default() -> Self {
        use crate::theme;
        Self {
            font_size: 9.0,
            bg: theme::SURFACE0,
            border_focused: theme::BLUE,
            border_unfocused: theme::SURFACE_BRIGHT,
            text_color: theme::SUBTEXT0,
            protocol_color: theme::OVERLAY0,
            selection_bg: theme::BLUE,
            selection_text: theme::SIDEBAR_BG,
            cursor_color: theme::BLUE,
        }
    }
}

/// Build an editable address bar showing `input` state.
///
/// Uses inline text-splitting for selection highlights: the text is split
/// into before / selected / after segments so the highlight background is
/// on the text node itself, guaranteeing pixel-perfect alignment regardless
/// of font metrics.
///
/// Used by both the showcase browser tab and the visual regression tests.
impl Tree {
    pub fn build_address_bar(
        &mut self,
        parent: NodeId,
        input: &TextInput,
        style: &AddressBarStyle,
    ) -> NodeId {
        let addr = self.add_box(
            parent,
            Style::default()
                .grow(1.0)
                .h(28.0)
                .pad_xy(10.0, 0.0)
                .row()
                .align(Align::Center)
                .bg(style.bg)
                .radius(6.0)
                .overflow(Overflow::Hidden)
                .border(
                    1.0,
                    if input.focused {
                        style.border_focused
                    } else {
                        style.border_unfocused
                    },
                ),
        );
        self.tag(addr, "browser-input");

        let font = Style::default().font(style.font_size);

        if input.focused {
            let parts = input.parts();

            if parts.has_selection {
                if !parts.before.is_empty() {
                    self.add_text(addr, parts.before, font.clone().color(style.text_color));
                }
                self.add_text(
                    addr,
                    parts.selected,
                    font.clone()
                        .color(style.selection_text)
                        .bg(style.selection_bg),
                );
                if !parts.after.is_empty() {
                    self.add_text(addr, parts.after, font.color(style.text_color));
                }
            } else {
                let (before_text, after_text) = input.text.split_at(input.cursor);
                if !before_text.is_empty() {
                    self.add_text(addr, before_text, font.clone().color(style.text_color));
                }
                self.add_box(addr, Style::default().w(1.0).h(14.0).bg(style.cursor_color));
                if !after_text.is_empty() {
                    self.add_text(addr, after_text, font.color(style.text_color));
                }
            }
        } else if let Some(idx) = input.text.find("://") {
            let (proto, rest) = input.text.split_at(idx + 3);
            self.add_text(addr, proto, font.clone().color(style.protocol_color));
            self.add_text(addr, rest, font.color(style.text_color));
        } else {
            self.add_text(addr, &input.text, font.color(style.text_color));
        }

        addr
    }
}

impl any_compute_core::visual::Graphable for Tree {
    fn to_graph(&self) -> any_compute_core::visual::VisualGraph<2> {
        use any_compute_core::layout::V;
        use any_compute_core::visual::{VNode, VisualGraph};

        // Recursive builder: creates a VisualGraph for a node's children,
        // nesting sub-graphs for box nodes that themselves have children.
        // Depth-limited to avoid stack overflow on deeply nested DOM trees.
        fn build_subgraph(
            arena: &[Slot],
            ids: &[NodeId],
            label: &str,
            depth: usize,
        ) -> VisualGraph<2> {
            let mut g = VisualGraph::new(label);
            let mut map = std::collections::HashMap::new();
            for &id in ids {
                let slot = &arena[id.0];
                let tag = slot.tag.as_deref().unwrap_or("");
                let lbl = match &slot.kind {
                    NodeKind::Text(s) => {
                        let preview: String = s.chars().take(16).collect();
                        if s.len() > 16 {
                            format!("\"{preview}…\"")
                        } else {
                            format!("\"{preview}\"")
                        }
                    }
                    NodeKind::Box => {
                        if !tag.is_empty() {
                            tag.to_string()
                        } else {
                            format!("box-{}", id.0)
                        }
                    }
                    NodeKind::Bar { .. } => format!("bar-{}", id.0),
                };
                let node_tag = match &slot.kind {
                    NodeKind::Text(_) => "constant",
                    NodeKind::Box if !slot.children.is_empty() => "custom",
                    NodeKind::Box => "input",
                    NodeKind::Bar { .. } => "reduce",
                };
                let mut vnode = VNode::new(V([0.0, 0.0]), &lbl, node_tag);
                // Nest children as a sub-graph (depth-limited)
                if matches!(slot.kind, NodeKind::Box) && !slot.children.is_empty() && depth < 8 {
                    let sub = build_subgraph(arena, &slot.children, &lbl, depth + 1);
                    vnode = vnode.children(sub);
                }
                let gi = g.add(vnode);
                map.insert(id, gi);
            }
            // Edges between siblings (parent→child in the DOM)
            for &id in ids {
                if let Some(&gi) = map.get(&id) {
                    for child_id in &arena[id.0].children {
                        if let Some(&ci) = map.get(child_id) {
                            g.edge(gi, ci);
                        }
                    }
                }
            }
            g.auto_layout();
            g
        }

        // Start from root's children
        let root_children = &self.arena[0].children;
        build_subgraph(&self.arena, root_children, "DOM", 0)
    }
}
