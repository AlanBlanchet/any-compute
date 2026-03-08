//! Arena-based scene graph — owns the node tree, solves layout, paints, dispatches events.
//!
//! Uses a flat `Vec<Slot>` arena so the whole tree is one allocation,
//! cache-friendly, and trivially serialisable.  Node IDs are indices.

use any_compute_core::hints::Hints;
use any_compute_core::interaction::{
    DispatchResult, EventContext, InputEvent, Phase,
};
use any_compute_core::layout::{Point, Rect, Size};
use any_compute_core::render::{Border, Color, Primitive, RenderList};

use super::style::*;
// Re-import specific items we use in match arms for clarity.
use super::style::{BoxSizing, Overflow, Visibility};

// ── Node identity ───────────────────────────────────────────────────────────

/// Lightweight handle into the arena.  Cheap to copy, compare, hash.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub usize);

/// What kind of content a node holds.
#[derive(Debug, Clone)]
pub enum NodeKind {
    /// Container (like a `<div>`) — has children, no intrinsic content.
    Box,
    /// Text leaf — has intrinsic size based on `content × font_size`.
    Text(String),
    /// Horizontal bar — intrinsic height, width comes from parent/flex.
    /// Stores a fill fraction `[0..1]` and bar color.
    Bar { fraction: f64, fill: Color },
}

// ── Arena slot ──────────────────────────────────────────────────────────────

/// One node in the arena.
#[derive(Debug, Clone)]
pub struct Slot {
    pub kind: NodeKind,
    pub style: Style,
    pub hints: Hints,
    pub parent: Option<NodeId>,
    pub children: Vec<NodeId>,
    /// Computed layout rect (filled by `Tree::layout`).
    pub rect: Rect,
    /// Scroll offset for `Overflow::Scroll` containers.
    pub scroll: Point,
    /// Optional click handler tag — matched by the host.
    pub tag: Option<String>,
}

impl Slot {
    /// Create a new slot with sensible defaults for spatial/hint fields.
    pub fn new(kind: NodeKind, style: Style, parent: Option<NodeId>) -> Self {
        Self {
            kind,
            style,
            hints: Hints::default(),
            parent,
            children: Vec::new(),
            rect: Rect::ZERO,
            scroll: Point::ZERO,
            tag: None,
        }
    }
}

// ── Tree ────────────────────────────────────────────────────────────────────

/// The DOM — a flat arena of [`Slot`]s.
pub struct Tree {
    pub arena: Vec<Slot>,
    pub root: NodeId,
}

impl Tree {
    /// Read-only access to a slot by id.
    #[inline]
    pub fn slot(&self, id: NodeId) -> &Slot {
        &self.arena[id.0]
    }

    /// Mutable access to a slot by id.
    #[inline]
    pub fn slot_mut(&mut self, id: NodeId) -> &mut Slot {
        &mut self.arena[id.0]
    }
}

impl Tree {
    /// Start building a tree with a root node.
    pub fn new(root_style: Style) -> Self {
        Self {
            arena: vec![Slot::new(NodeKind::Box, root_style, None)],
            root: NodeId(0),
        }
    }

    // ── Mutation ─────────────────────────────────────────

    /// Allocate a Box node as child of `parent`. Returns its `NodeId`.
    pub fn add_box(&mut self, parent: NodeId, style: Style) -> NodeId {
        self.add_node(parent, NodeKind::Box, style)
    }

    /// Allocate a Text node.
    pub fn add_text(&mut self, parent: NodeId, content: impl Into<String>, style: Style) -> NodeId {
        self.add_node(parent, NodeKind::Text(content.into()), style)
    }

    /// Allocate a Bar node (progress / throughput).
    pub fn add_bar(&mut self, parent: NodeId, fraction: f64, fill: Color, style: Style) -> NodeId {
        self.add_node(parent, NodeKind::Bar { fraction, fill }, style)
    }

    /// Tag a node for event matching.
    pub fn tag(&mut self, id: NodeId, tag: impl Into<String>) {
        self.slot_mut(id).tag = Some(tag.into());
    }

    /// Set hints on a node.
    pub fn set_hints(&mut self, id: NodeId, hints: Hints) {
        self.slot_mut(id).hints = hints;
    }

    fn add_node(&mut self, parent: NodeId, kind: NodeKind, style: Style) -> NodeId {
        let id = NodeId(self.arena.len());
        self.arena.push(Slot::new(kind, style, Some(parent)));
        self.slot_mut(parent).children.push(id);
        id
    }

    // ── Layout ──────────────────────────────────────────

    /// Solve layout for the whole tree, given the viewport size.
    pub fn layout(&mut self, viewport: Size) {
        let root = self.root;
        self.layout_node(
            root, viewport.w, viewport.h, viewport.w, viewport.h, 0.0, 0.0,
        );
    }

    /// Layout a single node.
    ///
    /// * `avail_w/h` — allocated space from flex distribution (used for auto-width
    ///   fallback and as an upper bound).
    /// * `resolve_w/h` — parent's content dimensions (used for resolving
    ///   percentage / calc dimensions so they aren't double-resolved).
    fn layout_node(
        &mut self,
        id: NodeId,
        avail_w: f64,
        avail_h: f64,
        resolve_w: f64,
        resolve_h: f64,
        ox: f64,
        oy: f64,
    ) {
        // Skip hidden nodes entirely.
        let style = self.slot(id).style.clone();
        if style.is_hidden() {
            self.slot_mut(id).rect = Rect::ZERO;
            return;
        }

        let pad_h = style.padding.horizontal();
        let pad_v = style.padding.vertical();
        let margin_h = style.margin.horizontal();
        let margin_v = style.margin.vertical();
        let bdr = style.effective_border();
        let bdr_h = bdr.horizontal();
        let bdr_v = bdr.vertical();

        // In border-box mode, width/height *include* padding + border.
        // Content area = resolved size − padding − border.
        // Percentages resolve against the parent's content area (resolve_w/h)
        // so they are not double-resolved through the flex allocation.
        let outer_w = style.width.resolve(resolve_w).unwrap_or(avail_w) + margin_h;
        let outer_h_hint = style.height.resolve(resolve_h);

        let content_w = match style.box_sizing {
            BoxSizing::BorderBox => (outer_w - margin_h - pad_h - bdr_h).max(0.0),
            BoxSizing::ContentBox => (outer_w - margin_h).max(0.0),
        };

        // Determine intrinsic height for text / bar.
        let intrinsic_h = match &self.slot(id).kind {
            NodeKind::Text(s) => {
                let font = style.font_size;
                let chars = s.len().max(1) as f64;
                let lines = (chars * font * 0.55 / content_w.max(1.0)).ceil().max(1.0);
                lines * font * style.line_height
            }
            NodeKind::Bar { .. } => style.font_size.max(8.0),
            NodeKind::Box => 0.0,
        };

        // Recursively layout children to know content height.
        let children: Vec<NodeId> = self.slot(id).children.clone();
        let flow_children: Vec<NodeId> = children
            .iter()
            .copied()
            .filter(|c| {
                let cs = &self.slot(*c).style;
                !cs.is_out_of_flow() && !cs.is_hidden()
            })
            .collect();

        let is_row = style.direction == Direction::Row;
        let child_avail_w = content_w;
        let child_avail_h = outer_h_hint
            .map(|h| {
                let deduct = match style.box_sizing {
                    BoxSizing::BorderBox => pad_v + bdr_v,
                    BoxSizing::ContentBox => 0.0,
                };
                (h - deduct).max(0.0)
            })
            .unwrap_or((avail_h - margin_v - pad_v - bdr_v).max(0.0));

        // Compute main-axis gap (row-gap / column-gap overrides).
        let main_gap = if is_row {
            style.column_gap.unwrap_or(style.gap)
        } else {
            style.row_gap.unwrap_or(style.gap)
        };

        // First pass: measure children.
        let total_gap = if flow_children.len() > 1 {
            main_gap * (flow_children.len() - 1) as f64
        } else {
            0.0
        };

        // Compute flex totals.
        let total_grow: f64 = flow_children
            .iter()
            .map(|c| self.slot(*c).style.flex_grow)
            .sum();

        let mut child_sizes: Vec<(NodeId, f64, f64)> = Vec::with_capacity(flow_children.len());
        let mut used_main = total_gap;

        for &cid in &flow_children {
            // Extract resolved values first, then drop the borrow so
            // intrinsic_width can read the arena without conflict.
            let (explicit_w, explicit_h, margin_main, child_align_self) = {
                let cs = &self.slot(cid).style;
                (
                    cs.width.resolve(child_avail_w),
                    cs.height.resolve(child_avail_h),
                    if is_row {
                        cs.margin.horizontal()
                    } else {
                        cs.margin.vertical()
                    },
                    cs.align_self,
                )
            };
            let cross_align = child_align_self.unwrap_or(style.align);
            let cw = if is_row {
                explicit_w.unwrap_or_else(|| self.intrinsic_width(cid))
            } else {
                match (explicit_w, cross_align) {
                    (Some(w), _) => w,
                    (None, Align::Stretch) => child_avail_w,
                    (None, _) => self.intrinsic_width(cid),
                }
            };
            let ch = if is_row {
                match (explicit_h, cross_align) {
                    (Some(h), _) => h,
                    (None, Align::Stretch) => child_avail_h,
                    (None, _) => 0.0,
                }
            } else {
                explicit_h.unwrap_or(0.0)
            };
            child_sizes.push((cid, cw, ch));
            used_main += if is_row { cw } else { ch } + margin_main;
        }

        // Distribute remaining space among flex-grow children.
        let main_budget = if is_row { child_avail_w } else { child_avail_h };
        let remaining = (main_budget - used_main).max(0.0);

        if total_grow > 0.0 && remaining > 0.0 {
            for (cid, cw, ch) in &mut child_sizes {
                let grow = self.slot(*cid).style.flex_grow;
                if grow > 0.0 {
                    let share = remaining * grow / total_grow;
                    if is_row {
                        *cw += share;
                    } else {
                        *ch += share;
                    }
                }
            }
        }

        // Flex-shrink: when children overflow a *definite* main axis, shrink
        // proportionally.  If main_budget is zero the container has auto/
        // indefinite size — it will wrap to content, so shrink must not fire.
        let overflow = used_main - main_budget;
        if overflow > 0.0 && main_budget > 0.0 {
            let total_shrink: f64 = child_sizes
                .iter()
                .map(|(cid, _, _)| self.slot(*cid).style.flex_shrink)
                .sum();
            if total_shrink > 0.0 {
                for (cid, cw, ch) in &mut child_sizes {
                    let shrink = self.slot(*cid).style.flex_shrink;
                    if shrink > 0.0 {
                        let share = overflow * shrink / total_shrink;
                        // Respect min-width / min-height constraints.
                        if is_row {
                            let min = self
                                .slot(*cid)
                                .style
                                .min_width
                                .resolve(child_avail_w)
                                .unwrap_or(0.0);
                            *cw = (*cw - share).max(min);
                        } else {
                            let min = self
                                .slot(*cid)
                                .style
                                .min_height
                                .resolve(child_avail_h)
                                .unwrap_or(0.0);
                            *ch = (*ch - share).max(min);
                        }
                    }
                }
            }
        }

        // Second pass: position children.
        let inner_x = ox + style.margin.left + style.padding.left + bdr.left;
        let inner_y = oy + style.margin.top + style.padding.top + bdr.top;
        let scroll = self.slot(id).scroll;
        let mut cursor_x = inner_x - scroll.x;
        let mut cursor_y = inner_y - scroll.y;

        // Justify offset.
        let total_child_main: f64 = child_sizes
            .iter()
            .map(|(cid, w, h)| {
                let cs = &self.slot(*cid).style;
                if is_row {
                    *w + cs.margin.horizontal()
                } else {
                    *h + cs.margin.vertical()
                }
            })
            .sum::<f64>()
            + total_gap;

        let justify_offset = match style.justify {
            Justify::Center => {
                ((if is_row { child_avail_w } else { child_avail_h }) - total_child_main).max(0.0)
                    / 2.0
            }
            Justify::End => {
                ((if is_row { child_avail_w } else { child_avail_h }) - total_child_main).max(0.0)
            }
            _ => 0.0,
        };

        if is_row {
            cursor_x += justify_offset;
        } else {
            cursor_y += justify_offset;
        }

        let mut max_cross = 0.0_f64;
        for (cid, cw, ch) in &child_sizes {
            let cs = &self.slot(*cid).style;
            let cm = cs.margin;
            let cx = cursor_x + cm.left;
            let cy = cursor_y + cm.top;

            // Cross-axis alignment.
            // Per-child align-self overrides parent align.
            let effective_align = cs.align_self.unwrap_or(style.align);
            let (fx, fy) = if is_row {
                let aligned_y = match effective_align {
                    Align::Center => cy + (child_avail_h - ch) / 2.0,
                    Align::End => cy + child_avail_h - ch,
                    _ => cy,
                };
                (cx, aligned_y)
            } else {
                let aligned_x = match effective_align {
                    Align::Center => cx + (child_avail_w - cw) / 2.0,
                    Align::End => cx + child_avail_w - cw,
                    _ => cx,
                };
                (aligned_x, cy)
            };

            self.layout_node(*cid, *cw, *ch, child_avail_w, child_avail_h, fx, fy);

            let child_rect = self.slot(*cid).rect;
            if is_row {
                cursor_x += child_rect.size.w + cm.horizontal() + main_gap;
                max_cross = max_cross.max(child_rect.size.h + cm.vertical());
            } else {
                cursor_y += child_rect.size.h + cm.vertical() + main_gap;
                max_cross = max_cross.max(child_rect.size.w + cm.horizontal());
            }
        }

        // Layout out-of-flow children (absolute / fixed).
        for &cid in &children {
            let cs = &self.slot(cid).style;
            if !cs.is_out_of_flow() || cs.is_hidden() {
                continue;
            }
            let ax = inner_x + cs.left.resolve(content_w).unwrap_or(0.0);
            let ay = inner_y + cs.top.resolve(child_avail_h).unwrap_or(0.0);
            let aw = cs.width.resolve(content_w).unwrap_or(content_w);
            let ah = cs.height.resolve(child_avail_h).unwrap_or(0.0);
            self.layout_node(cid, aw, ah, content_w, child_avail_h, ax, ay);
        }

        // Compute own height from children if auto.
        let children_h = cursor_y - (inner_y - scroll.y) - main_gap.max(0.0);
        let _children_w = cursor_x - (inner_x - scroll.x) - main_gap.max(0.0);

        let _insets = match style.box_sizing {
            BoxSizing::BorderBox => pad_h + bdr_h,
            BoxSizing::ContentBox => 0.0,
        };
        let final_w = style
            .width
            .resolve(resolve_w)
            .unwrap_or((avail_w - margin_h).max(0.0));
        let final_h = outer_h_hint.unwrap_or_else(|| {
            let content = intrinsic_h.max(if is_row { max_cross } else { children_h });
            (content + pad_v + bdr_v).max(avail_h)
        });

        // Apply min/max constraints.
        let final_w = Dimension::clamp(final_w, style.min_width, style.max_width, resolve_w);
        let final_h = Dimension::clamp(final_h, style.min_height, style.max_height, resolve_h);

        self.slot_mut(id).rect = Rect::new(ox, oy, final_w, final_h);
    }

    // ── Intrinsic sizing ─────────────────────────────────

    /// Estimate the min-content width of a subtree (for main-axis measurement
    /// of row children that have no explicit width).
    fn intrinsic_width(&self, id: NodeId) -> f64 {
        let slot = self.slot(id);
        let s = &slot.style;
        let pad_h = s.padding.horizontal();
        match &slot.kind {
            NodeKind::Text(t) => t.len() as f64 * s.font_size * 0.55 + pad_h,
            NodeKind::Bar { .. } => pad_h,
            NodeKind::Box => {
                let row = s.direction == Direction::Row;
                let gap = if slot.children.len() > 1 {
                    s.gap * (slot.children.len() - 1) as f64
                } else {
                    0.0
                };
                let children: Vec<NodeId> = slot.children.clone();
                let content: f64 = if row {
                    children
                        .iter()
                        .map(|c| {
                            let cs = &self.slot(*c).style;
                            cs.width
                                .resolve(0.0)
                                .unwrap_or_else(|| self.intrinsic_width(*c))
                                + cs.margin.horizontal()
                        })
                        .sum::<f64>()
                        + gap
                } else {
                    children
                        .iter()
                        .map(|c| {
                            let cs = &self.slot(*c).style;
                            cs.width
                                .resolve(0.0)
                                .unwrap_or_else(|| self.intrinsic_width(*c))
                                + cs.margin.horizontal()
                        })
                        .fold(0.0_f64, f64::max)
                };
                content + pad_h
            }
        }
    }

    // ── Paint ───────────────────────────────────────────

    /// Walk the tree and emit primitives into the render list.
    pub fn paint(&self, list: &mut RenderList) {
        self.paint_node(self.root, list);
    }

    fn paint_node(&self, id: NodeId, list: &mut RenderList) {
        let slot = self.slot(id);
        let r = slot.rect;
        let s = &slot.style;

        // Display::None → skip entirely (no children either).
        if s.is_hidden() {
            return;
        }

        if s.opacity <= 0.0 || s.visibility == Visibility::Hidden {
            // Invisible but still occupies space; still paint children though
            // (visibility is not inherited in our model, only display:none is).
            self.paint_children(id, list);
            return;
        }

        // Clip for scrollable containers.
        let needs_clip = !matches!(s.overflow, Overflow::Visible);
        if needs_clip {
            list.push(Primitive::PushClip { bounds: r });
        }

        // Background.
        if s.background.a > 0 {
            let bw = s.effective_border();
            let has_border = s.border_color.a > 0
                && (bw.top > 0.0 || bw.right > 0.0 || bw.bottom > 0.0 || bw.left > 0.0);
            let border = if has_border {
                let max_bw = bw.top.max(bw.right).max(bw.bottom).max(bw.left);
                Some(Border {
                    color: s.border_color,
                    width: max_bw,
                })
            } else {
                None
            };
            list.push(Primitive::Rect {
                bounds: r,
                fill: s.background,
                border,
                corner_radius: s.corner_radius,
            });
        }

        // Kind-specific paint.
        match &slot.kind {
            NodeKind::Text(content) => {
                let tx = r.origin.x + s.padding.left;
                let ty = r.origin.y + s.padding.top + s.font_size * 0.85;
                list.push(Primitive::Text {
                    anchor: Point::new(tx, ty),
                    content: content.clone(),
                    font_size: s.font_size,
                    color: s.color,
                });
            }
            NodeKind::Bar { fraction, fill } => {
                let bar_h = (r.size.h - s.padding.vertical()).max(0.0);
                let track_w = (r.size.w - s.padding.horizontal()).max(0.0);
                let bx = r.origin.x + s.padding.left;
                let by = r.origin.y + s.padding.top;
                // Track background.
                list.push(Primitive::Rect {
                    bounds: Rect::new(bx, by, track_w, bar_h),
                    fill: Color::rgba(255, 255, 255, 20),
                    border: None,
                    corner_radius: s.corner_radius,
                });
                // Fill.
                let fill_w = track_w * fraction.clamp(0.0, 1.0);
                if fill_w > 0.0 {
                    list.push(Primitive::Rect {
                        bounds: Rect::new(bx, by, fill_w, bar_h),
                        fill: *fill,
                        border: None,
                        corner_radius: s.corner_radius,
                    });
                }
            }
            NodeKind::Box => {}
        }

        // Paint children in z-index order.
        self.paint_children(id, list);

        if needs_clip {
            list.push(Primitive::PopClip);
        }
    }

    /// Paint children sorted by z-index. Children without z-index use
    /// insertion order (stable sort preserves source order for equal z).
    fn paint_children(&self, id: NodeId, list: &mut RenderList) {
        let children = &self.slot(id).children;
        if children.is_empty() {
            return;
        }

        // Fast path: if no child has a z-index set, paint in insertion order.
        let any_z = children
            .iter()
            .any(|c| self.slot(*c).style.z_index.is_some());
        if !any_z {
            for &child_id in children {
                self.paint_node(child_id, list);
            }
            return;
        }

        // Sort by z-index (stable — preserves insertion order for equal z).
        let mut sorted: Vec<NodeId> = children.clone();
        sorted.sort_by_key(|c| self.slot(*c).style.z_index.unwrap_or(0));
        for child_id in sorted {
            self.paint_node(child_id, list);
        }
    }

    // ── Event dispatch ──────────────────────────────────

    /// Find the deepest node at `pos` and return its `NodeId`.
    pub fn hit_test(&self, pos: Point) -> Option<NodeId> {
        self.hit_test_node(self.root, pos)
    }

    fn hit_test_node(&self, id: NodeId, pos: Point) -> Option<NodeId> {
        let slot = self.slot(id);
        if !slot.rect.contains(pos) {
            return None;
        }
        // Deepest child wins (reverse for z-order: last child = on top).
        for &child_id in slot.children.iter().rev() {
            if let Some(hit) = self.hit_test_node(child_id, pos) {
                return Some(hit);
            }
        }
        Some(id)
    }

    /// Dispatch a click and return the tag of the clicked node (if any).
    pub fn click(&self, pos: Point) -> Option<&str> {
        let mut id = self.hit_test(pos)?;
        // Walk up until we find a tagged node.
        loop {
            if let Some(ref tag) = self.slot(id).tag {
                return Some(tag.as_str());
            }
            id = self.slot(id).parent?;
        }
    }

    /// Build the ancestor path (root → target) for a given node.
    fn ancestor_path(&self, target: NodeId) -> Vec<NodeId> {
        let mut path = Vec::new();
        let mut cur = Some(target);
        while let Some(id) = cur {
            path.push(id);
            cur = self.slot(id).parent;
        }
        path.reverse();
        path
    }

    /// Collect tags along a path (root → target order).
    fn collect_tags(&self, path: &[NodeId]) -> Vec<String> {
        path.iter()
            .filter_map(|&id| self.slot(id).tag.clone())
            .collect()
    }

    /// Find the deepest tagged node from target upward (same as `click` walk).
    pub fn tag_at(&self, pos: Point) -> Option<String> {
        let mut id = self.hit_test(pos)?;
        loop {
            if let Some(ref tag) = self.slot(id).tag {
                return Some(tag.clone());
            }
            id = self.slot(id).parent?;
        }
    }

    /// Full capture → target → bubble dispatch.
    ///
    /// For pointer events, hit-tests to find the target.
    /// Returns a [`DispatchResult`] with the tag chain from root → target.
    /// The host inspects `result.target_tag()` or `result.bubble_tags()` to dispatch actions.
    pub fn dispatch(&self, event: InputEvent) -> DispatchResult {
        let target = event.pos().and_then(|p| self.hit_test(p));
        let Some(target) = target else {
            return DispatchResult::default();
        };

        let path = self.ancestor_path(target);
        let tags = self.collect_tags(&path);
        let mut ctx = EventContext::new(event);

        // Capture phase: root → target-1
        ctx.phase = Phase::Capture;
        for &id in &path[..path.len().saturating_sub(1)] {
            if ctx.stopped {
                break;
            }
            // Per-node handler hook point: if handlers were stored on Slot,
            // we would invoke them here.  For tag-based dispatch the host
            // processes the returned DispatchResult instead.
            let _ = id;
        }

        // Target phase.
        if !ctx.stopped {
            ctx.phase = Phase::Target;
            let _ = target;
        }

        // Bubble phase: target-1 → root (reverse).
        if !ctx.stopped {
            ctx.phase = Phase::Bubble;
            for &id in path.iter().rev().skip(1) {
                if ctx.stopped {
                    break;
                }
                let _ = id;
            }
        }

        DispatchResult {
            tags,
            stopped: ctx.stopped,
            default_prevented: ctx.default_prevented,
        }
    }

    /// Apply a scroll delta to a node (or the nearest scrollable ancestor).
    pub fn scroll(&mut self, pos: Point, delta: Point) {
        let Some(mut id) = self.hit_test(pos) else {
            return;
        };
        loop {
            if self.slot(id).style.overflow == Overflow::Scroll {
                let s = &mut self.slot_mut(id).scroll;
                s.x = (s.x - delta.x).max(0.0);
                s.y = (s.y - delta.y).max(0.0);
                return;
            }
            match self.slot(id).parent {
                Some(p) => id = p,
                None => return,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_tree_layout() {
        let mut tree = Tree::new(Style::default().w(400.0).h(300.0).bg(Color::BLACK));
        let child = tree.add_box(
            tree.root,
            Style::default().w(200.0).h(100.0).bg(Color::WHITE),
        );
        tree.layout(Size::new(400.0, 300.0));
        assert_eq!(tree.slot(child).rect.size.w, 200.0);
        assert_eq!(tree.slot(child).rect.size.h, 100.0);
    }

    #[test]
    fn text_node_has_intrinsic_height() {
        let mut tree = Tree::new(Style::default().w(300.0).h(200.0));
        let txt = tree.add_text(tree.root, "Hello world", Style::default().font(16.0));
        tree.layout(Size::new(300.0, 200.0));
        assert!(tree.slot(txt).rect.size.h > 0.0);
    }

    #[test]
    fn hit_test_finds_child() {
        let mut tree = Tree::new(Style::default().w(400.0).h(300.0));
        let child = tree.add_box(tree.root, Style::default().w(100.0).h(50.0));
        tree.layout(Size::new(400.0, 300.0));
        let hit = tree.hit_test(Point::new(50.0, 25.0));
        assert_eq!(hit, Some(child));
    }

    #[test]
    fn click_returns_tag() {
        let mut tree = Tree::new(Style::default().w(400.0).h(300.0));
        let child = tree.add_box(tree.root, Style::default().w(100.0).h(50.0));
        tree.tag(child, "my-button");
        tree.layout(Size::new(400.0, 300.0));
        assert_eq!(tree.click(Point::new(50.0, 25.0)), Some("my-button"));
        assert_eq!(tree.click(Point::new(350.0, 250.0)), None);
    }

    #[test]
    fn flex_grow_distributes_space() {
        let mut tree = Tree::new(Style::default().w(300.0).h(100.0).row());
        let _a = tree.add_box(tree.root, Style::default().w(50.0).h(100.0));
        let b = tree.add_box(tree.root, Style::default().h(100.0).grow(1.0));
        tree.layout(Size::new(300.0, 100.0));
        let bw = tree.slot(b).rect.size.w;
        assert!(
            bw > 200.0,
            "flex child should consume remaining space, got {}",
            bw
        );
    }

    #[test]
    fn paint_produces_primitives() {
        let mut tree = Tree::new(Style::default().w(400.0).h(300.0).bg(Color::BLACK));
        tree.add_text(
            tree.root,
            "Hi",
            Style::default().font(14.0).color(Color::WHITE),
        );
        tree.add_bar(
            tree.root,
            0.5,
            Color::rgb(0, 255, 0),
            Style::default().h(10.0),
        );
        tree.layout(Size::new(400.0, 300.0));
        let mut list = RenderList::default();
        tree.paint(&mut list);
        assert!(
            list.len() >= 3,
            "expected ≥3 primitives, got {}",
            list.len()
        );
    }

    /// Regression: row-direction tab buttons inside a column sidebar must
    /// stretch to the sidebar width so clicks anywhere on the row register.
    #[test]
    fn row_button_in_column_stretches_width() {
        // Sidebar: column, 220×600, padding 16 12
        let mut tree = Tree::new(Style::default().w(800.0).h(600.0).row());
        let sidebar = tree.add_box(
            tree.root,
            Style {
                width: Dimension::Px(220.0),
                padding: Edges::xy(12.0, 16.0),
                gap: 8.0,
                ..Style::default()
            },
        );
        // Tab button: row with height=36, no explicit width.
        let btn = tree.add_box(
            sidebar,
            Style {
                height: Dimension::Px(36.0),
                direction: Direction::Row,
                padding: Edges::xy(12.0, 0.0),
                align: Align::Center,
                ..Style::default()
            },
        );
        tree.add_text(btn, "Hardware", Style::default().font(13.0));
        tree.tag(btn, "tab-0");

        tree.layout(Size::new(800.0, 600.0));

        let btn_r = tree.slot(btn).rect;
        // Button must fill the sidebar's inner width (220 − 12 − 12 = 196).
        assert!(
            (btn_r.size.w - 196.0).abs() < 1.0,
            "tab button should stretch to 196px, got {}",
            btn_r.size.w
        );
        // Click in the middle of the button must return the tag.
        let mid_x = btn_r.origin.x + btn_r.size.w / 2.0;
        let mid_y = btn_r.origin.y + btn_r.size.h / 2.0;
        assert_eq!(tree.click(Point::new(mid_x, mid_y)), Some("tab-0"));
        // Click near the right edge (x ≈ 190) must also work.
        assert_eq!(
            tree.click(Point::new(btn_r.origin.x + 180.0, mid_y)),
            Some("tab-0")
        );
    }

    /// Text in a row parent must get intrinsic width, not 0.
    #[test]
    fn text_in_row_has_intrinsic_width() {
        let mut tree = Tree::new(Style::default().w(400.0).h(100.0).row());
        let txt = tree.add_text(tree.root, "Hello", Style::default().font(14.0));
        tree.layout(Size::new(400.0, 100.0));
        let w = tree.slot(txt).rect.size.w;
        assert!(w > 10.0, "text in row should have intrinsic width, got {w}");
    }

    #[test]
    fn dispatch_returns_tag_chain() {
        use any_compute_core::interaction::{Button, InputEvent};
        let mut tree = Tree::new(Style::default().w(400.0).h(300.0));
        let sidebar = tree.add_box(tree.root, Style::default().w(200.0).h(300.0));
        tree.tag(sidebar, "sidebar");
        let btn = tree.add_box(sidebar, Style::default().w(100.0).h(50.0));
        tree.tag(btn, "tab-0");
        tree.layout(Size::new(400.0, 300.0));

        let result = tree.dispatch(InputEvent::PointerDown {
            pos: Point::new(50.0, 25.0),
            button: Button::Primary,
        });
        assert_eq!(result.tags, vec!["sidebar", "tab-0"]);
        assert_eq!(result.target_tag(), Some("tab-0"));
    }

    #[test]
    fn dispatch_miss_returns_empty() {
        use any_compute_core::interaction::{Button, InputEvent};
        let mut tree = Tree::new(Style::default().w(400.0).h(300.0));
        tree.layout(Size::new(400.0, 300.0));
        let result = tree.dispatch(InputEvent::PointerDown {
            pos: Point::new(500.0, 500.0),
            button: Button::Primary,
        });
        assert!(result.tags.is_empty());
    }

    #[test]
    fn tag_at_finds_deepest() {
        let mut tree = Tree::new(Style::default().w(400.0).h(300.0));
        let c = tree.add_box(tree.root, Style::default().w(200.0).h(100.0));
        tree.tag(c, "container");
        let inner = tree.add_box(c, Style::default().w(100.0).h(50.0));
        tree.tag(inner, "inner-btn");
        tree.layout(Size::new(400.0, 300.0));
        assert_eq!(
            tree.tag_at(Point::new(50.0, 25.0)).as_deref(),
            Some("inner-btn")
        );
    }

    #[test]
    fn row_with_fixed_and_grow_respects_min_width() {
        // Sidebar (200px, min-width 200px) + main (flex-grow 1) in an 800px row.
        let mut root_style = Style::default().w(800.0).h(600.0);
        root_style.direction = Direction::Row;
        let mut t = Tree::new(root_style);
        let root = t.root;
        let mut sb_style = Style::default().w(200.0);
        sb_style.min_width = Dimension::Px(200.0);
        let sidebar = t.add_box(root, sb_style);
        let main = t.add_box(root, Style::default().grow(1.0));
        // Give main a child to create intrinsic width.
        t.add_text(main, "Dashboard", Style::default().font(16.0));
        t.layout(Size::new(800.0, 600.0));
        let sb_w = t.slot(sidebar).rect.size.w;
        let mn_w = t.slot(main).rect.size.w;
        assert!(sb_w >= 200.0, "sidebar should be >= 200px but was {sb_w}");
        assert!(
            (sb_w + mn_w - 800.0).abs() < 1.0,
            "sidebar ({sb_w}) + main ({mn_w}) should sum to ~800"
        );
    }

    /// End-to-end layout of the visual_cmp dashboard through parse_with_css.
    #[test]
    fn visual_cmp_layout_dimensions() {
        use crate::css::StyleSheet;
        use crate::parse::parse_with_css;

        let css = r#"
* { box-sizing: border-box; }
.root { flex-direction: row; width: 800px; height: 600px; background: #1e1e2e; }
.sidebar { width: 200px; min-width: 200px; background: #181825; padding: 16px; gap: 10px; }
.main { flex-grow: 1; }
.header { flex-direction: row; height: 48px; min-height: 48px; background: #313244;
          padding: 0px 20px; align-items: center; font-size: 16px; color: #cdd2f4; }
.content { flex-grow: 1; padding: 20px; gap: 16px; }
.cards-row { flex-direction: row; gap: 12px; }
.card { flex-grow: 1; background: #313244; border-radius: 12px; padding: 16px; gap: 8px; }
.card-title { font-size: 14px; color: #89b4fa; }
.card-body { font-size: 12px; color: #cdd2f4; }
.bar-row { gap: 6px; }
.bar-track { height: 8px; background: #333333; border-radius: 4px; }
.bar-fill-green { height: 8px; width: 70%; background: #a6e3a1; border-radius: 4px; }
.bar-fill-blue  { height: 8px; width: 45%; background: #89b4fa; border-radius: 4px; }
.bar-fill-red   { height: 8px; width: 85%; background: #f38ba8; border-radius: 4px; }
.color-swatch { width: 40px; height: 40px; border-radius: 6px; }
.nested-row { flex-direction: row; gap: 8px; }
.opacity-box { width: 60px; height: 40px; background: #89b4fa; border-radius: 6px; }
"#;
        let html = r#"
<div class="root">
  <div class="sidebar" tag="sidebar">
    <span>Sidebar</span>
  </div>
  <div class="main" tag="main">
    <div class="header" tag="header">Dashboard</div>
    <div class="content" tag="content">
      <div class="cards-row" tag="cards-row">
        <div class="card" tag="card1"><span class="card-title">Title</span><span class="card-body">Body text</span></div>
        <div class="card"><span class="card-title">Title</span><span class="card-body">Body text</span></div>
        <div class="card"><span class="card-title">Title</span><span class="card-body">Body text</span></div>
      </div>
      <div class="bar-row" tag="bar-row">
        <div class="bar-track" tag="track1"><div class="bar-fill-green" tag="fill-green"></div></div>
        <div class="bar-track"><div class="bar-fill-blue" tag="fill-blue"></div></div>
        <div class="bar-track"><div class="bar-fill-red" tag="fill-red"></div></div>
      </div>
      <div class="nested-row">
        <div class="color-swatch" tag="swatch"></div>
        <div class="color-swatch"></div>
        <div class="color-swatch"></div>
      </div>
      <div class="nested-row">
        <div class="opacity-box" tag="obox"></div>
        <div class="opacity-box"></div>
        <div class="opacity-box"></div>
      </div>
    </div>
  </div>
</div>
"#;
        let sheet = StyleSheet::parse(css);
        let mut tree = parse_with_css(html, &sheet);
        tree.layout(Size::new(800.0, 600.0));

        // Walk the tree and print all node rects for debugging.
        for (i, slot) in tree.arena.iter().enumerate() {
            let tag = slot.tag.as_deref().unwrap_or("");
            let r = &slot.rect;
            let kind = match &slot.kind {
                NodeKind::Box => "box",
                NodeKind::Text(s) => s.as_str(),
                NodeKind::Bar { .. } => "bar",
            };
            println!(
                "[{i:2}] {tag:12} {kind:20} x={:6.1} y={:6.1} w={:6.1} h={:6.1}",
                r.origin.x, r.origin.y, r.size.w, r.size.h,
            );
        }

        let by_tag = |t: &str| -> &Slot {
            tree.arena
                .iter()
                .find(|s| s.tag.as_deref() == Some(t))
                .unwrap_or_else(|| panic!("missing tag '{t}'"))
        };

        let sidebar = by_tag("sidebar");
        let header = by_tag("header");
        let content = by_tag("content");
        let swatch = by_tag("swatch");
        let obox = by_tag("obox");
        let track = by_tag("track1");
        let fill_green = by_tag("fill-green");
        let fill_blue = by_tag("fill-blue");
        let fill_red = by_tag("fill-red");
        let card = by_tag("card1");

        println!("\n=== Key dimensions ===");
        println!("sidebar: w={:.1}", sidebar.rect.size.w);
        println!("header:  h={:.1}", header.rect.size.h);
        println!(
            "content: w={:.1} h={:.1}",
            content.rect.size.w, content.rect.size.h
        );
        println!(
            "swatch:  w={:.1} h={:.1}",
            swatch.rect.size.w, swatch.rect.size.h
        );
        println!(
            "opacity: w={:.1} h={:.1}",
            obox.rect.size.w, obox.rect.size.h
        );
        println!(
            "card1:   w={:.1} h={:.1}",
            card.rect.size.w, card.rect.size.h
        );
        println!(
            "bar-track w={:.1}, fills: green={:.1} ({:.1}%) blue={:.1} ({:.1}%) red={:.1} ({:.1}%)",
            track.rect.size.w,
            fill_green.rect.size.w,
            fill_green.rect.size.w / track.rect.size.w * 100.0,
            fill_blue.rect.size.w,
            fill_blue.rect.size.w / track.rect.size.w * 100.0,
            fill_red.rect.size.w,
            fill_red.rect.size.w / track.rect.size.w * 100.0,
        );

        // Assertions.
        assert!(
            (sidebar.rect.size.w - 200.0).abs() < 1.0,
            "sidebar should be 200px, got {:.1}",
            sidebar.rect.size.w
        );
        assert!(
            (header.rect.size.h - 48.0).abs() < 1.0,
            "header should be 48px, got {:.1}",
            header.rect.size.h
        );
        assert!(
            (swatch.rect.size.w - 40.0).abs() < 1.0,
            "swatch should be 40px, got {:.1}",
            swatch.rect.size.w
        );
        assert!(
            (obox.rect.size.w - 60.0).abs() < 1.0,
            "opacity-box should be 60px, got {:.1}",
            obox.rect.size.w
        );
        assert!(
            (fill_green.rect.size.w / track.rect.size.w - 0.70).abs() < 0.02,
            "green fill should be 70%, got {:.1}%",
            fill_green.rect.size.w / track.rect.size.w * 100.0
        );
    }
}
