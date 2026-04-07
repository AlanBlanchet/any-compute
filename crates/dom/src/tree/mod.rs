//! Arena-based scene graph — owns the node tree, solves layout, paints, dispatches events.
//!
//! Uses a flat `Vec<Slot>` arena so the whole tree is one allocation,
//! cache-friendly, and trivially serialisable.  Node IDs are indices.

use any_compute_core::hints::Hints;
use any_compute_core::interaction::{
    DispatchResult, EventContext, InputEvent, Modifiers, Phase,
};
use any_compute_core::layout::{Point, Rect, Size};
use any_compute_core::render::{Border, Color, Primitive, RenderList};

use crate::css::{AnimationFillMode, StyleSheet};
use crate::style::*;
// Re-import specific items we use in match arms for clarity.
use crate::style::{
    BoxSizing, Cursor, Overflow, PointerEvents, TextDecoration, TextOverflow, Visibility,
    WhiteSpace,
};

use std::sync::Arc;
use std::time::Instant;

/// True for characters that form part of a "word" (for double-click selection).
#[inline]
fn is_word_char(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_' || b == b'-'
}

/// Selection highlight color (semi-transparent blue, browser-like).
const SELECTION_BG: Color = Color::rgba(100, 149, 237, 100);

// ═══════════════════════════════════════════════════════════════════════════
// ── Dirty tracking (re-exported from core) ──────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

pub use any_compute_core::tree::Dirty;

/// Classify a CSS transition/animation property as layout-affecting or visual-only.
/// Kept in sync with `Style::differs_in_layout()` and `StyleOp::affects_layout()`.
fn property_affects_layout(prop: &str) -> bool {
    matches!(
        prop,
        "display"
            | "box-sizing"
            | "width"
            | "height"
            | "min-width"
            | "min-height"
            | "max-width"
            | "max-height"
            | "aspect-ratio"
            | "direction"
            | "flex-direction"
            | "flex-wrap"
            | "align-items"
            | "align-self"
            | "justify-content"
            | "gap"
            | "row-gap"
            | "column-gap"
            | "padding"
            | "padding-top"
            | "padding-right"
            | "padding-bottom"
            | "padding-left"
            | "margin"
            | "margin-top"
            | "margin-right"
            | "margin-bottom"
            | "margin-left"
            | "position"
            | "left"
            | "top"
            | "right"
            | "bottom"
            | "overflow"
            | "flex-grow"
            | "flex-shrink"
            | "flex-basis"
            | "order"
            | "border-width"
            | "border-top-width"
            | "border-right-width"
            | "border-bottom-width"
            | "border-left-width"
            | "font-family"
            | "font-size"
            | "font-weight"
            | "line-height"
            | "white-space"
            | "text-align"
            | "letter-spacing"
            | "word-spacing"
            | "text-indent"
            | "word-break"
            | "all"
    )
}

// ── Animation runtime state ─────────────────────────────────────────────────

/// Shared timing fields for transitions and keyframe animations.
///
/// Both [`ActiveTransition`] and [`ActiveAnimation`] embed this so
/// delay/duration/easing logic is defined exactly once.

mod animation;
mod node;
mod address_bar;

pub use animation::*;
pub use node::*;
pub use address_bar::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextSelection {
    pub node: NodeId,
    pub start: usize,
    pub end: usize,
}

/// The DOM — a flat arena of [`Slot`]s with optional CSS for live restyle.
pub struct Tree {
    pub arena: Vec<Slot>,
    pub root: NodeId,
    /// Shared stylesheet — when present, enables live hover/active restyle.
    pub sheet: Option<Arc<StyleSheet>>,
    /// Last viewport used by `layout()` — reused by `tick()` for auto re-layout.
    viewport: Size,
    /// Currently hovered node (if any).
    hovered: Option<NodeId>,
    /// Currently focused node (if any).
    focused: Option<NodeId>,
    /// Global flag: at least one node has `Dirty::LAYOUT`.
    pub needs_layout: bool,
    /// Global flag: at least one node has `Dirty::PAINT`.
    pub needs_paint: bool,
    /// Active text selection (double-click word select, or click-drag).
    pub selection: Option<TextSelection>,
    /// Last PointerDown time + target for double-click detection.
    last_click: Option<(Instant, NodeId)>,
    /// Anchor char index for click-drag text selection.
    drag_anchor: Option<(NodeId, usize)>,
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

    /// Walk from `start` to root, calling `f` on each slot along the path.
    fn walk_ancestors(&mut self, start: NodeId, mut f: impl FnMut(&mut Slot)) {
        let mut id = start;
        loop {
            f(&mut self.arena[id.0]);
            match self.arena[id.0].parent {
                Some(p) => id = p,
                None => break,
            }
        }
    }

    /// Set a flag on ancestor path + restyle. Combines the common
    /// `walk_ancestors(…) + restyle_path(…)` pattern.
    fn set_state_and_restyle(&mut self, node: NodeId, f: impl FnMut(&mut Slot)) {
        self.walk_ancestors(node, f);
        self.restyle_path(node);
    }

    /// Mark a node (and ancestors) dirty, propagating upward.
    /// `LAYOUT` implies `PAINT`. Ancestors get the same flags so that
    /// top-down traversal can discover dirty subtrees early.
    fn mark_dirty(&mut self, id: NodeId, flags: Dirty) {
        let flags = if flags.contains(Dirty::LAYOUT) {
            flags | Dirty::PAINT
        } else {
            flags
        };
        if flags.contains(Dirty::LAYOUT) {
            self.needs_layout = true;
        }
        if flags.contains(Dirty::PAINT) {
            self.needs_paint = true;
        }
        let mut cur = id;
        loop {
            let slot = &mut self.arena[cur.0];
            let before = slot.dirty;
            slot.dirty |= flags;
            // Stop propagating if ancestor already had these flags.
            if slot.dirty == before {
                break;
            }
            match slot.parent {
                Some(p) => cur = p,
                None => break,
            }
        }
    }
}

impl Tree {
    /// Start building a tree with a root node.
    pub fn new(root_style: Style) -> Self {
        Self {
            arena: vec![Slot::new(NodeKind::Box, root_style, None)],
            root: NodeId(0),
            sheet: None,
            viewport: Size::ZERO,
            hovered: None,
            focused: None,
            needs_layout: true,
            needs_paint: true,
            selection: None,
            last_click: None,
            drag_anchor: None,
        }
    }

    /// Attach a stylesheet for live restyle (hover/active pseudo-class updates).
    pub fn set_sheet(&mut self, sheet: Arc<StyleSheet>) {
        self.sheet = Some(sheet);
    }

    /// Initialize CSS @keyframes animations for all nodes that have animation specs.
    ///
    /// Call once after parse + layout to start animation playback.
    pub fn start_animations(&mut self) {
        let sheet = match &self.sheet {
            Some(s) => Arc::clone(s),
            None => return,
        };

        for i in 0..self.arena.len() {
            let slot = &self.arena[i];
            let mut anims = Vec::new();

            // Collect animation specs from all classes on this node
            for cls in &slot.class_list {
                for spec in sheet.class_animations(cls) {
                    if let Some(kfs) = sheet.keyframes(&spec.name) {
                        anims.push(ActiveAnimation {
                            name: spec.name.clone(),
                            keyframes: kfs.to_vec(),
                            timing: Timing {
                                elapsed: 0.0,
                                duration: spec.duration_secs,
                                delay: spec.delay_secs,
                                easing: spec.easing,
                            },
                            iteration_count: spec.iteration_count,
                            direction: spec.direction,
                            fill_mode: spec.fill_mode,
                            iterations_done: 0.0,
                        });
                    }
                }
            }

            self.arena[i].animations = anims;
        }
    }

    /// Advance all active transitions and animations by `dt` seconds.
    ///
    /// When animations modify layout-affecting properties the tree is
    /// automatically re-laid-out using the last viewport from `layout()`.
    /// Returns a [`TickResult`] so the caller knows whether to redraw.
    pub fn tick(&mut self, dt: f64) -> TickResult {
        let mut result = TickResult::default();
        let mut dirty_nodes: Vec<(NodeId, Dirty)> = Vec::new();

        for i in 0..self.arena.len() {
            let slot = &mut self.arena[i];

            // ── Transitions ─────────────────────────────────────────
            if !slot.transitions.is_empty() {
                // Start from the target (to) style — completed properties stay at target.
                let to_style = slot.transitions[0].to.clone();
                let mut blended_result = to_style;
                let mut all_done = true;
                let mut any_layout = false;

                for tr in &mut slot.transitions {
                    tr.timing.elapsed += dt;
                    if !tr.finished() {
                        all_done = false;
                        result.active = true;
                        if property_affects_layout(&tr.property) {
                            any_layout = true;
                        }
                        let blended = tr.from.lerp(&tr.to, tr.progress());
                        super::style::copy_css_property(
                            &mut blended_result,
                            &blended,
                            &tr.property,
                        );
                    }
                }

                slot.style = blended_result;
                if all_done {
                    slot.transitions.clear();
                }
                let id = NodeId(i);
                if any_layout {
                    dirty_nodes.push((id, Dirty::LAYOUT | Dirty::PAINT));
                } else {
                    dirty_nodes.push((id, Dirty::PAINT));
                }
            }

            // ── Animations ──────────────────────────────────────────
            let mut anim_dirty = false;
            for anim in &mut slot.animations {
                if !anim.finished() {
                    anim.timing.elapsed += dt;
                    anim_dirty = true;
                    result.active = true;
                }
            }

            // Apply keyframe ops on top of the current style
            if anim_dirty {
                let mut any_layout = false;
                for anim in &slot.animations {
                    if !anim.finished()
                        || matches!(
                            anim.fill_mode,
                            AnimationFillMode::Forwards | AnimationFillMode::Both
                        )
                    {
                        // Check if any keyframe op affects layout.
                        if !any_layout {
                            for kf in &anim.keyframes {
                                if kf.ops.iter().any(|op| op.affects_layout()) {
                                    any_layout = true;
                                    break;
                                }
                            }
                        }
                        anim.apply_to(&mut slot.style);
                    }
                }
                let id = NodeId(i);
                if any_layout {
                    dirty_nodes.push((id, Dirty::LAYOUT | Dirty::PAINT));
                } else {
                    dirty_nodes.push((id, Dirty::PAINT));
                }
            }
        }

        // Propagate dirty flags.
        for (id, flags) in dirty_nodes {
            self.mark_dirty(id, flags);
        }

        // Auto re-layout when animations changed layout-affecting properties.
        if self.needs_layout && self.viewport != Size::ZERO {
            self.layout(self.viewport);
        }

        result
    }

    /// Whether any node has active animations/transitions (use for redraw scheduling).
    pub fn has_active_animations(&self) -> bool {
        self.arena
            .iter()
            .any(|s| !s.transitions.is_empty() || s.animations.iter().any(|a| !a.finished()))
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

    /// Create an element node by tag name (e.g. `"button"`, `"div"`, `"h1"`).
    ///
    /// Resolves user-agent defaults for the tag so the node looks correct
    /// without any manual styling.
    ///
    /// ```ignore
    /// let btn = tree.add_element(parent, "button");
    /// tree.add_text(btn, "Click me", Style::default());
    /// ```
    pub fn add_element(&mut self, parent: NodeId, tag: &str) -> NodeId {
        self.add_element_with(parent, tag, |s| s)
    }

    /// Like [`add_element`](Self::add_element) but applies a closure to
    /// customize the UA-default style before node creation.  The closure
    /// acts as inline-style overrides (highest CSS specificity).
    ///
    /// ```ignore
    /// let btn = tree.add_element_with(parent, "button", |s| s.bg(RED).h(36));
    /// ```
    pub fn add_element_with(
        &mut self,
        parent: NodeId,
        tag: &str,
        f: impl FnOnce(Style) -> Style,
    ) -> NodeId {
        let mut el = tag.to_dom();
        el.style = f(el.style);
        let id = self.add_node(parent, el.kind, el.style);
        self.slot_mut(id).element = tag.to_string();
        id
    }

    /// Tag a node for event matching.
    pub fn tag(&mut self, id: NodeId, tag: impl Into<String>) {
        self.slot_mut(id).tag = Some(tag.into());
    }

    /// Set hints on a node.
    pub fn set_hints(&mut self, id: NodeId, hints: Hints) {
        self.slot_mut(id).hints = hints;
    }

    /// Focus a node (for `:focus` pseudo-class). Blurs the previous node.
    pub fn focus(&mut self, id: NodeId) {
        if let Some(old) = self.focused {
            self.arena[old.0].focused = false;
        }
        self.arena[id.0].focused = true;
        self.focused = Some(id);
        self.restyle_if_sheet(id);
        if let Some(old) = self.focused {
            if old != id {
                self.restyle_if_sheet(old);
            }
        }
    }

    /// Remove focus from the currently focused node.
    pub fn blur(&mut self) {
        if let Some(old) = self.focused.take() {
            self.arena[old.0].focused = false;
            self.restyle_if_sheet(old);
        }
    }

    /// Make a node editable — sets an initial value and marks it for text input.
    pub fn set_editable(&mut self, id: NodeId, initial: &str) {
        self.arena[id.0].value = Some(initial.to_string());
        self.arena[id.0].caret = initial.len();
        self.sync_value_text(id);
    }

    /// Current value of an editable node (`None` if not editable).
    pub fn value(&self, id: NodeId) -> Option<&str> {
        self.arena[id.0].value.as_deref()
    }

    /// Caret byte position in the editable value.
    pub fn caret(&self, id: NodeId) -> usize {
        self.arena[id.0].caret
    }

    /// Sync visible text child to match the editable `value`.
    /// Finds (or creates) the first Text child and replaces its content.
    fn sync_value_text(&mut self, id: NodeId) {
        let text = match &self.arena[id.0].value {
            Some(v) => v.clone(),
            None => return,
        };
        // Find existing text child.
        let existing = self.arena[id.0]
            .children
            .iter()
            .copied()
            .find(|c| matches!(self.arena[c.0].kind, NodeKind::Text(_)));
        if let Some(tid) = existing {
            self.arena[tid.0].kind = NodeKind::Text(text);
            self.arena[tid.0].dirty |= Dirty::LAYOUT | Dirty::PAINT;
        } else {
            // Create a text child inheriting parent style.
            let style = self.arena[id.0].style.clone();
            self.add_text(id, &text, style);
        }
    }

    /// Handle editing key (Backspace, Delete, arrows, Home, End).
    /// Returns `true` if the value was modified or caret moved.
    fn handle_edit_key(&mut self, id: NodeId, key: &str, mods: Modifiers) -> bool {
        let slot = &mut self.arena[id.0];
        let val = match &mut slot.value {
            Some(v) => v,
            None => return false,
        };
        let caret = &mut slot.caret;
        match key {
            "Backspace" => {
                if *caret > 0 {
                    // Find char boundary before caret.
                    let prev = val[..*caret]
                        .char_indices()
                        .next_back()
                        .map(|(i, _)| i)
                        .unwrap_or(0);
                    val.drain(prev..*caret);
                    *caret = prev;
                    true
                } else {
                    false
                }
            }
            "Delete" => {
                if *caret < val.len() {
                    let next = val[*caret..]
                        .char_indices()
                        .nth(1)
                        .map(|(i, _)| *caret + i)
                        .unwrap_or(val.len());
                    val.drain(*caret..next);
                    true
                } else {
                    false
                }
            }
            "ArrowLeft" | "Left" => {
                if *caret > 0 {
                    *caret = val[..*caret]
                        .char_indices()
                        .next_back()
                        .map(|(i, _)| i)
                        .unwrap_or(0);
                }
                true
            }
            "ArrowRight" | "Right" => {
                if *caret < val.len() {
                    *caret = val[*caret..]
                        .char_indices()
                        .nth(1)
                        .map(|(i, _)| *caret + i)
                        .unwrap_or(val.len());
                }
                true
            }
            "Home" => {
                *caret = 0;
                true
            }
            "End" => {
                *caret = val.len();
                true
            }
            // Select all
            "a" if mods.ctrl || mods.meta => {
                *caret = val.len();
                true
            }
            _ => false,
        }
    }

    fn add_node(&mut self, parent: NodeId, kind: NodeKind, mut style: Style) -> NodeId {
        // CSS inheritance: copy inheritable properties from parent when not explicitly set.
        style.inherit_from(&self.arena[parent.0].style);
        let id = NodeId(self.arena.len());
        self.arena.push(Slot::new(kind, style, Some(parent)));
        self.slot_mut(parent).children.push(id);
        id
    }

    // ── Layout ──────────────────────────────────────────

    /// Pre-measure all text nodes using an external font measurement function.
    ///
    /// `f(text, font_size) -> width` should return the glyph-shaped text width
    /// (e.g. via `Gpu::measure_text`).  CSS letter-spacing and word-spacing are
    /// added automatically. Call before `layout()` for accurate flex centering.
    pub fn measure_text_nodes(&mut self, mut f: impl FnMut(&str, f64) -> f64) {
        for slot in &mut self.arena {
            if let NodeKind::Text(ref text) = slot.kind {
                let mut w = f(text, slot.style.font_size);
                if slot.style.letter_spacing != 0.0 {
                    w += text.chars().count() as f64 * slot.style.letter_spacing;
                }
                if slot.style.word_spacing != 0.0 {
                    let spaces = text.chars().filter(|c| *c == ' ').count() as f64;
                    w += spaces * slot.style.word_spacing;
                }
                slot.measured_text_width = w;
            }
        }
    }

    /// Solve layout for the whole tree, given the viewport size.
    pub fn layout(&mut self, viewport: Size) {
        self.viewport = viewport;
        let root = self.root;
        self.layout_node(
            root,
            viewport.w(),
            viewport.h(),
            viewport.w(),
            viewport.h(),
            0.0,
            0.0,
        );
        // Layout done — clear LAYOUT flags, promote to PAINT (positions may have changed).
        // Also refresh z_sorted caches for all dirty nodes.
        for i in 0..self.arena.len() {
            if self.arena[i].dirty.contains(Dirty::LAYOUT) {
                self.arena[i].dirty.remove(Dirty::LAYOUT);
                self.arena[i].dirty.insert(Dirty::PAINT);
            }
            // Rebuild z_sorted cache while we have &mut self.
            if self.arena[i].dirty.contains(Dirty::Z_ORDER) || self.arena[i].z_sorted.is_empty() {
                let children = self.arena[i].children.clone();
                let any_z = children
                    .iter()
                    .any(|c| self.arena[c.0].style.z_index.is_some());
                let mut sorted = children;
                if any_z {
                    sorted.sort_by_key(|c| self.arena[c.0].style.z_index.unwrap_or(0));
                }
                self.arena[i].z_sorted = sorted;
                self.arena[i].dirty.remove(Dirty::Z_ORDER);
            }
        }
        self.needs_layout = false;
        self.needs_paint = true;
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
        let outer_h_hint = style.height.resolve(resolve_h).or_else(|| {
            // aspect-ratio: derive height from width when height is auto
            style.aspect_ratio.map(|ar| (outer_w - margin_h) / ar)
        });

        let content_w = match style.box_sizing {
            BoxSizing::BorderBox => (outer_w - margin_h - pad_h - bdr_h).max(0.0),
            BoxSizing::ContentBox => (outer_w - margin_h).max(0.0),
        };

        // Determine intrinsic height for text / bar.
        let intrinsic_h = self.intrinsic_height(id, content_w);

        // Recursively layout children to know content height.
        let children: Vec<NodeId> = self.slot(id).children.clone();
        let mut flow_children: Vec<NodeId> = children
            .iter()
            .copied()
            .filter(|c| {
                let cs = &self.slot(*c).style;
                !cs.is_out_of_flow() && !cs.is_hidden()
            })
            .collect();
        // CSS `order` property: sort by order (stable — preserves DOM order for ties).
        flow_children.sort_by_key(|c| self.slot(*c).style.order);

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

        // First pass: measure children (initial sizes before grow/shrink).
        let mut child_sizes: Vec<(NodeId, f64, f64)> = Vec::with_capacity(flow_children.len());
        for &cid in &flow_children {
            let (explicit_w, explicit_h, flex_basis, child_align_self) = {
                let cs = &self.slot(cid).style;
                let basis_ctx = if is_row { child_avail_w } else { child_avail_h };
                (
                    cs.width.resolve(child_avail_w),
                    cs.height.resolve(child_avail_h),
                    cs.flex_basis.resolve(basis_ctx),
                    cs.align_self,
                )
            };
            let cross_align = child_align_self.unwrap_or(style.align);
            let cw = if is_row {
                flex_basis
                    .or(explicit_w)
                    .unwrap_or_else(|| self.intrinsic_width(cid))
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
                    (None, _) => self.intrinsic_height(cid, cw),
                }
            } else {
                flex_basis
                    .or(explicit_h)
                    .unwrap_or_else(|| self.intrinsic_height(cid, cw))
            };
            child_sizes.push((cid, cw, ch));
        }

        // Break children into flex lines.
        let main_budget = if is_row { child_avail_w } else { child_avail_h };
        let wraps = style.flex_wrap != FlexWrap::NoWrap;
        let mut lines: Vec<Vec<usize>> = vec![vec![]];

        if wraps && main_budget > 0.0 {
            let mut line_used = 0.0_f64;
            for (i, &(cid, cw, ch)) in child_sizes.iter().enumerate() {
                let cs = &self.slot(cid).style;
                let m = if is_row {
                    cs.margin.horizontal()
                } else {
                    cs.margin.vertical()
                };
                let child_main = if is_row { cw } else { ch } + m;
                let gap_add = if lines.last().unwrap().is_empty() {
                    0.0
                } else {
                    main_gap
                };
                if !lines.last().unwrap().is_empty()
                    && line_used + gap_add + child_main > main_budget
                {
                    lines.push(vec![]);
                    line_used = 0.0;
                }
                let gap_add = if lines.last().unwrap().is_empty() {
                    0.0
                } else {
                    main_gap
                };
                line_used += child_main + gap_add;
                lines.last_mut().unwrap().push(i);
            }
        } else {
            lines[0] = (0..child_sizes.len()).collect();
        }
        if style.flex_wrap == FlexWrap::WrapReverse {
            lines.reverse();
        }

        // Per-line: grow/shrink + position children.
        let inner_x = ox + style.margin.left + style.padding.left + bdr.left;
        let inner_y = oy + style.margin.top + style.padding.top + bdr.top;
        let scroll = self.slot(id).scroll;
        let cross_gap = if is_row {
            style.row_gap.unwrap_or(style.gap)
        } else {
            style.column_gap.unwrap_or(style.gap)
        };
        let mut cross_cursor = if is_row {
            inner_y - scroll.y
        } else {
            inner_x - scroll.x
        };
        let mut total_cross = 0.0_f64;
        let mut last_cursor_main = if is_row {
            inner_x - scroll.x
        } else {
            inner_y - scroll.y
        };

        for line_indices in &lines {
            if line_indices.is_empty() {
                continue;
            }

            // Grow/shrink within this line.
            let line_gap = if line_indices.len() > 1 {
                main_gap * (line_indices.len() - 1) as f64
            } else {
                0.0
            };
            let mut line_used = line_gap;
            let mut line_grow = 0.0_f64;
            for &idx in line_indices {
                let (cid, cw, ch) = child_sizes[idx];
                let cs = &self.slot(cid).style;
                let m = if is_row {
                    cs.margin.horizontal()
                } else {
                    cs.margin.vertical()
                };
                line_used += if is_row { cw } else { ch } + m;
                line_grow += cs.flex_grow;
            }

            let remaining = (main_budget - line_used).max(0.0);
            if line_grow > 0.0 && remaining > 0.0 {
                for &idx in line_indices {
                    let grow = self.slot(child_sizes[idx].0).style.flex_grow;
                    if grow > 0.0 {
                        let share = remaining * grow / line_grow;
                        if is_row {
                            child_sizes[idx].1 += share;
                        } else {
                            child_sizes[idx].2 += share;
                        }
                    }
                }
            }

            // Flex-shrink (only for non-wrapping — wrapped lines don't overflow main axis).
            // Iterate: when an item is clamped at its min, redistribute the remaining
            // overflow among unclamped items until fully absorbed or all items are clamped.
            let mut remaining_overflow = line_used - main_budget;
            if remaining_overflow > 0.0 && main_budget > 0.0 && !wraps {
                for _ in 0..line_indices.len() {
                    if remaining_overflow <= 0.0 {
                        break;
                    }
                    let active_shrink: f64 = line_indices
                        .iter()
                        .filter_map(|&idx| {
                            let cid = child_sizes[idx].0;
                            let shrink = self.slot(cid).style.flex_shrink;
                            let min = if is_row {
                                self.slot(cid).style.min_width.resolve(child_avail_w)
                            } else {
                                self.slot(cid).style.min_height.resolve(child_avail_h)
                            }
                            .unwrap_or(0.0);
                            let cur = if is_row {
                                child_sizes[idx].1
                            } else {
                                child_sizes[idx].2
                            };
                            (shrink > 0.0 && cur > min + 0.001).then_some(shrink)
                        })
                        .sum();
                    if active_shrink <= 0.0 {
                        break;
                    }
                    let mut absorbed = 0.0_f64;
                    for &idx in line_indices {
                        let cid = child_sizes[idx].0;
                        let shrink = self.slot(cid).style.flex_shrink;
                        if shrink <= 0.0 {
                            continue;
                        }
                        let min = if is_row {
                            self.slot(cid).style.min_width.resolve(child_avail_w)
                        } else {
                            self.slot(cid).style.min_height.resolve(child_avail_h)
                        }
                        .unwrap_or(0.0);
                        let cur = if is_row {
                            child_sizes[idx].1
                        } else {
                            child_sizes[idx].2
                        };
                        if cur <= min + 0.001 {
                            continue;
                        }
                        let share = remaining_overflow * shrink / active_shrink;
                        let new_val = (cur - share).max(min);
                        let actual = cur - new_val;
                        if is_row {
                            child_sizes[idx].1 = new_val;
                        } else {
                            child_sizes[idx].2 = new_val;
                        }
                        absorbed += actual;
                    }
                    remaining_overflow -= absorbed;
                }
            }

            // Justify: compute per-line offsets.
            let line_child_main: f64 = line_indices
                .iter()
                .map(|&idx| {
                    let (cid, cw, ch) = child_sizes[idx];
                    let cs = &self.slot(cid).style;
                    if is_row {
                        cw + cs.margin.horizontal()
                    } else {
                        ch + cs.margin.vertical()
                    }
                })
                .sum();
            let line_total_with_gap = line_child_main + line_gap;
            let n = line_indices.len() as f64;
            let line_remaining = (main_budget - line_child_main).max(0.0);

            let (j_offset, j_gap) = match style.justify {
                Justify::Center => ((main_budget - line_total_with_gap).max(0.0) / 2.0, main_gap),
                Justify::End => ((main_budget - line_total_with_gap).max(0.0), main_gap),
                Justify::SpaceBetween if n > 1.0 => (0.0, line_remaining / (n - 1.0)),
                Justify::SpaceAround if n > 0.0 => {
                    let g = line_remaining / n;
                    (g / 2.0, g)
                }
                Justify::SpaceEvenly if n > 0.0 => {
                    let g = line_remaining / (n + 1.0);
                    (g, g)
                }
                _ => (0.0, main_gap),
            };

            let main_start = if is_row {
                inner_x - scroll.x
            } else {
                inner_y - scroll.y
            };
            let mut cursor_main = main_start + j_offset;
            let mut line_max_cross = 0.0_f64;

            // Pre-compute actual line cross dimension for correct Center/End alignment.
            // CSS spec: for single-line containers with a definite cross size,
            // the line cross = container inner cross.  For multi-line or auto-size,
            // it's the intrinsic max of items in the line.
            let intrinsic_cross: f64 = line_indices
                .iter()
                .map(|&idx| {
                    let (cid, cw, ch) = child_sizes[idx];
                    let cs = &self.slot(cid).style;
                    if is_row {
                        ch + cs.margin.vertical()
                    } else {
                        cw + cs.margin.horizontal()
                    }
                })
                .fold(0.0_f64, f64::max);
            let line_cross = if is_row {
                intrinsic_cross.max(child_avail_h)
            } else {
                intrinsic_cross.max(child_avail_w)
            };

            for &idx in line_indices {
                let (cid, cw, ch) = child_sizes[idx];
                let cs = &self.slot(cid).style;
                let cm = cs.margin;
                let effective_align = cs.align_self.unwrap_or(style.align);

                let (fx, fy) = if is_row {
                    let cx = cursor_main + cm.left;
                    let align_cross = line_cross - cm.vertical();
                    let cy = cross_cursor + cm.top;
                    let aligned_y = match effective_align {
                        Align::Center => cy + (align_cross - ch) / 2.0,
                        Align::End => cy + align_cross - ch,
                        _ => cy,
                    };
                    (cx, aligned_y)
                } else {
                    let cy = cursor_main + cm.top;
                    let align_cross = line_cross - cm.horizontal();
                    let cx = cross_cursor + cm.left;
                    let aligned_x = match effective_align {
                        Align::Center => cx + (align_cross - cw) / 2.0,
                        Align::End => cx + align_cross - cw,
                        _ => cx,
                    };
                    (aligned_x, cy)
                };

                self.layout_node(cid, cw, ch, child_avail_w, child_avail_h, fx, fy);

                let child_rect = self.slot(cid).rect;
                if is_row {
                    cursor_main += child_rect.size.w() + cm.horizontal() + j_gap;
                    line_max_cross = line_max_cross.max(child_rect.size.h() + cm.vertical());
                } else {
                    cursor_main += child_rect.size.h() + cm.vertical() + j_gap;
                    line_max_cross = line_max_cross.max(child_rect.size.w() + cm.horizontal());
                }
            }

            last_cursor_main = cursor_main;
            cross_cursor += line_max_cross + cross_gap;
            total_cross += line_max_cross;
        }

        // Add cross gaps between lines.
        if lines.len() > 1 {
            total_cross += cross_gap * (lines.len() - 1) as f64;
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
        let children_main = last_cursor_main
            - (if is_row {
                inner_x - scroll.x
            } else {
                inner_y - scroll.y
            })
            - main_gap.max(0.0);

        let final_w = style
            .width
            .resolve(resolve_w)
            .unwrap_or((avail_w - margin_h).max(0.0));
        let final_h = outer_h_hint.unwrap_or_else(|| {
            let content = intrinsic_h.max(if is_row { total_cross } else { children_main });
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
    /// Intrinsic height of a content node (text or bar) given available width.
    /// Returns 0 for Box nodes (their height comes from children during layout).
    fn intrinsic_height(&self, id: NodeId, avail_w: f64) -> f64 {
        let slot = self.slot(id);
        let s = &slot.style;
        match &slot.kind {
            NodeKind::Text(text) => {
                let text_w = slot.text_w(text) + s.text_indent;
                let lines = if s.white_space == WhiteSpace::NoWrap {
                    1.0
                } else {
                    (text_w / avail_w.max(1.0)).ceil().max(1.0)
                };
                let lh = if s.line_height_absolute {
                    s.line_height
                } else {
                    s.font_size * s.line_height
                };
                lines * lh
            }
            NodeKind::Bar { .. } => s.font_size.max(MIN_BAR_HEIGHT),
            NodeKind::Box => 0.0,
        }
    }

    fn intrinsic_width(&self, id: NodeId) -> f64 {
        let slot = self.slot(id);
        let s = &slot.style;
        let pad_h = s.padding.horizontal();
        match &slot.kind {
            NodeKind::Text(t) => slot.text_w(t) + s.text_indent + pad_h,
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

    /// Call after `paint()` to clear dirty flags so the next frame can skip
    /// unchanged nodes.  Separate from `paint()` to keep it `&self`.
    pub fn post_paint(&mut self) {
        for slot in &mut self.arena {
            slot.dirty = Dirty::empty();
        }
        self.needs_paint = false;
    }

    /// Whether the tree has pending visual changes that require a repaint.
    #[inline]
    pub fn needs_paint(&self) -> bool {
        self.needs_paint
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

        // Apply transform to bounds (translate + scale around center).
        let r = if s.has_transform() {
            s.transform_rect(r)
        } else {
            r
        };

        // Clip for scrollable containers.
        let needs_clip = !matches!(s.overflow, Overflow::Visible);
        if needs_clip {
            list.push(Primitive::PushClip { bounds: r });
        }

        // Box-shadow (drawn before the main rect).
        if let Some(sh) = &s.box_shadow {
            let shadow_bounds = Rect::new(
                r.origin.x + sh.x - sh.spread,
                r.origin.y + sh.y - sh.spread,
                r.size.w() + sh.spread * 2.0,
                r.size.h() + sh.spread * 2.0,
            );
            list.push(Primitive::Rect {
                bounds: shadow_bounds,
                fill: s.apply_opacity(sh.color),
                border: None,
                corner_radius: s.corner_radius + sh.spread,
            });
        }

        // Background + border.
        let bg = s.apply_opacity(s.background);
        let bw = s.effective_border();
        let has_border = s.has_visible_border();
        if bg.a > 0 || has_border {
            let border = if has_border {
                let bc = s.apply_opacity(s.border_color);
                Some(Border {
                    color: bc,
                    top: bw.top,
                    right: bw.right,
                    bottom: bw.bottom,
                    left: bw.left,
                })
            } else {
                None
            };
            list.push(Primitive::Rect {
                bounds: r,
                fill: bg,
                border,
                corner_radius: s.corner_radius,
            });
        }

        // Outline (drawn outside the border box, after background).
        if s.outline_width > 0.0 && s.outline_color.a > 0 && s.outline_style != BorderStyle::None {
            let ow = s.outline_width;
            let off = s.outline_offset + ow;
            let outline_bounds = Rect::new(
                r.origin.x - off,
                r.origin.y - off,
                r.size.w() + off * 2.0,
                r.size.h() + off * 2.0,
            );
            let cr = if s.corner_radius > 0.0 {
                s.corner_radius + off
            } else {
                0.0
            };
            list.push(Primitive::Rect {
                bounds: outline_bounds,
                fill: Color::TRANSPARENT,
                border: Some(Border::uniform(s.apply_opacity(s.outline_color), ow)),
                corner_radius: cr,
            });
        }

        // Kind-specific paint.
        match &slot.kind {
            NodeKind::Text(content) => {
                let display_text = s.transform_text(content);
                let text_w = self.slot(id).text_w(&display_text);
                let avail_w = r.size.w() - s.padding.left - s.padding.right;

                // text-overflow: ellipsis — truncate when text overflows
                let final_text = if s.text_overflow == TextOverflow::Ellipsis
                    && text_w > avail_w
                    && avail_w > 0.0
                {
                    let char_w = s.char_width();
                    let max_chars =
                        ((avail_w - char_w * ELLIPSIS_CHARS).max(0.0) / char_w) as usize;
                    let truncated: String = display_text.chars().take(max_chars).collect();
                    std::borrow::Cow::Owned(format!("{}...", truncated))
                } else {
                    display_text
                };

                // text-align: compute horizontal offset
                let align_offset = s.text_align_offset(text_w, avail_w);

                let tx = r.origin.x + s.padding.left + s.text_indent + align_offset;
                let ty = r.origin.y + s.padding.top + s.font_size * TEXT_BASELINE_RATIO;
                let painted_text = final_text.into_owned();

                // Selection highlight (drawn behind text)
                if let Some(sel) = &self.selection {
                    if sel.node == id && sel.start < sel.end {
                        let char_count = painted_text.chars().count().max(1);
                        let cw = text_w / char_count as f64;
                        let sel_start_chars = painted_text[..sel.start.min(painted_text.len())]
                            .chars()
                            .count();
                        let sel_end_chars = painted_text[..sel.end.min(painted_text.len())]
                            .chars()
                            .count();
                        let sel_x = tx + sel_start_chars as f64 * cw;
                        let sel_w = (sel_end_chars - sel_start_chars) as f64 * cw;
                        let sel_y = r.origin.y + s.padding.top;
                        let sel_h = s.font_size * s.line_height;
                        list.push(Primitive::Rect {
                            bounds: Rect::new(sel_x, sel_y, sel_w, sel_h),
                            fill: SELECTION_BG,
                            border: None,
                            corner_radius: 2.0,
                        });
                    }
                }

                // text-shadow (drawn behind text)
                if let Some(sh) = &s.text_shadow {
                    list.push(Primitive::Text {
                        anchor: Point::new(tx + sh.x, ty + sh.y),
                        content: painted_text.clone(),
                        font_size: s.font_size,
                        color: s.apply_opacity(sh.color),
                    });
                }

                list.push(Primitive::Text {
                    anchor: Point::new(tx, ty),
                    content: painted_text,
                    font_size: s.font_size,
                    color: s.apply_opacity(s.color),
                });

                // text-decoration
                if s.text_decoration != TextDecoration::None {
                    let line_y = match s.text_decoration {
                        TextDecoration::Underline => {
                            ty + s.font_size * (1.0 - TEXT_BASELINE_RATIO) + UNDERLINE_OFFSET_PX
                        }
                        TextDecoration::Overline => r.origin.y + s.padding.top,
                        TextDecoration::LineThrough => ty - s.font_size * LINE_THROUGH_RATIO,
                        TextDecoration::None => unreachable!(),
                    };
                    let line_end = tx + text_w.min(avail_w);
                    list.push_line(tx, line_y, line_end, line_y, s.apply_opacity(s.color), 1.0);
                }
            }
            NodeKind::Bar { fraction, fill } => {
                let bar_h = (r.size.h() - s.padding.vertical()).max(0.0);
                let track_w = (r.size.w() - s.padding.horizontal()).max(0.0);
                let bx = r.origin.x + s.padding.left;
                let by = r.origin.y + s.padding.top;
                // Track background.
                list.push(Primitive::Rect {
                    bounds: Rect::new(bx, by, track_w, bar_h),
                    fill: BAR_TRACK_BG,
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

        // Scrollbar thumb for scroll containers
        if s.overflow == Overflow::Scroll && !slot.children.is_empty() {
            let scroll_y = slot.scroll.y;
            // Compute content height from children extents (relative to parent)
            let mut content_bottom = 0.0_f64;
            for &cid in &slot.children {
                let cr = self.slot(cid).rect;
                let child_bottom = cr.origin.y + cr.size.h() + scroll_y - r.origin.y;
                content_bottom = content_bottom.max(child_bottom);
            }
            let visible_h = r.size.h();
            if content_bottom > visible_h + 1.0 {
                let bar_w = 6.0;
                let track_h = visible_h;
                let ratio = (visible_h / content_bottom).min(1.0);
                let thumb_h = (track_h * ratio).max(20.0);
                let scroll_ratio = scroll_y / (content_bottom - visible_h).max(1.0);
                let thumb_y = r.origin.y + scroll_ratio * (track_h - thumb_h);
                let thumb_x = r.origin.x + r.size.w() - bar_w - 2.0;
                list.push(Primitive::Rect {
                    bounds: Rect::new(thumb_x, thumb_y, bar_w, thumb_h),
                    fill: Color::rgba(180, 180, 200, 80),
                    border: None,
                    corner_radius: bar_w / 2.0,
                });
            }
        }

        if needs_clip {
            list.push(Primitive::PopClip);
        }
    }

    /// Paint children sorted by z-index. Children without z-index use
    /// insertion order (stable sort preserves source order for equal z).
    /// Return children sorted by z-index (ascending). If no child has z-index
    /// set, returns a simple clone in insertion order (no allocation for fast path).
    fn z_sorted_children(&self, id: NodeId) -> &[NodeId] {
        &self.slot(id).z_sorted
    }

    fn paint_children(&self, id: NodeId, list: &mut RenderList) {
        for &child_id in self.z_sorted_children(id) {
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
        let s = &slot.style;

        // pointer-events: none → skip this node and its children
        if s.pointer_events == PointerEvents::None {
            return None;
        }

        // Apply transform to bounds for hit-testing
        let test_rect = if s.has_transform() {
            s.transform_rect(slot.rect)
        } else {
            slot.rect
        };

        if !test_rect.contains(pos) {
            return None;
        }
        // Highest z-index first (reverse z-sorted order) — last painted = first hit.
        for &child_id in self.z_sorted_children(id).iter().rev() {
            if let Some(hit) = self.hit_test_node(child_id, pos) {
                return Some(hit);
            }
        }
        Some(id)
    }

    /// Walk up from the hit node to find the nearest tagged ancestor.
    fn find_tag_node(&self, pos: Point) -> Option<NodeId> {
        let mut id = self.hit_test(pos)?;
        loop {
            if self.slot(id).tag.is_some() {
                return Some(id);
            }
            id = self.slot(id).parent?;
        }
    }

    /// Walk up from `start` to find the nearest editable ancestor (has `value`).
    fn find_editable_ancestor(&self, start: NodeId) -> Option<NodeId> {
        let mut id = start;
        loop {
            if self.arena[id.0].value.is_some() {
                return Some(id);
            }
            id = self.arena[id.0].parent?;
        }
    }

    /// Dispatch a click and return the tag of the clicked node (if any).
    pub fn click(&self, pos: Point) -> Option<&str> {
        self.find_tag_node(pos)
            .and_then(|id| self.slot(id).tag.as_deref())
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

    /// Like `click`, but returns an owned String.
    pub fn tag_at(&self, pos: Point) -> Option<String> {
        self.click(pos).map(str::to_string)
    }

    /// Resolve cursor style at a position — walks from hit target up to root.
    pub fn cursor_at(&self, pos: Point) -> Cursor {
        let Some(mut id) = self.hit_test(pos) else {
            return Cursor::Default;
        };
        loop {
            let c = self.slot(id).style.cursor;
            if c != Cursor::Default {
                return c;
            }
            match self.slot(id).parent {
                Some(p) => id = p,
                None => return Cursor::Default,
            }
        }
    }

    /// Find a tagged node's laid-out rect, or `None` if no node has the tag.
    pub fn tagged_rect(&self, tag: &str) -> Option<Rect> {
        self.arena
            .iter()
            .find(|s| s.tag.as_deref() == Some(tag))
            .map(|s| s.rect)
    }

    /// Full capture → target → bubble dispatch.
    ///
    /// For pointer events, hit-tests to find the target, then automatically
    /// updates `:hover` / `:active` state on the affected nodes and restyles.
    /// Returns a [`DispatchResult`] with the tag chain from root → target.
    /// `result.restyled` is `true` only when visual state actually changed.
    pub fn dispatch(&mut self, event: InputEvent) -> DispatchResult {
        let target = event.pos().and_then(|p| self.hit_test(p));
        let mut restyled = false;

        // ── Update hover / active state ─────────────────────────────
        match &event {
            InputEvent::PointerMove { pos, .. } => {
                let prev = self.hovered;
                let next = target;
                if prev != next {
                    if let Some(old) = prev {
                        self.set_state_and_restyle(old, |s| s.hovered = false);
                    }
                    if let Some(new) = next {
                        self.set_state_and_restyle(new, |s| s.hovered = true);
                    }
                    self.hovered = next;
                    restyled = true;
                }
                // Click-drag text selection: extend from anchor
                if let Some((text_id, anchor)) = self.drag_anchor {
                    let cur = self.char_index_at(text_id, pos.x);
                    let (lo, hi) = if cur < anchor {
                        (cur, anchor)
                    } else {
                        (anchor, cur)
                    };
                    if lo != hi {
                        self.selection = Some(TextSelection {
                            node: text_id,
                            start: lo,
                            end: hi,
                        });
                    }
                }
            }
            InputEvent::PointerDown { pos, .. } => {
                if let Some(t) = target {
                    self.set_state_and_restyle(t, |s| s.active = true);
                    restyled = true;

                    // Auto-focus editable nodes on click.
                    let editable = self.find_editable_ancestor(t);
                    if let Some(eid) = editable {
                        self.focus(eid);
                        // Place caret at click position.
                        let text_child = self.arena[eid.0]
                            .children
                            .iter()
                            .copied()
                            .find(|c| matches!(self.arena[c.0].kind, NodeKind::Text(_)));
                        if let Some(tid) = text_child {
                            let idx = self.char_index_at(tid, pos.x);
                            self.arena[eid.0].caret = idx;
                        }
                    } else if self.focused.is_some() {
                        self.blur();
                    }

                    // Double-click detection → word selection on text nodes
                    let now = Instant::now();
                    let is_double = self.last_click.is_some_and(|(prev_t, prev_n)| {
                        prev_n == t && now.duration_since(prev_t).as_millis() < 500
                    });
                    if is_double {
                        self.select_word_at(t, *pos);
                        self.last_click = None;
                        self.drag_anchor = None;
                    } else {
                        self.selection = None;
                        self.last_click = Some((now, t));
                        // Set drag anchor for click-drag text selection
                        let text_id = self.find_text_child(t).unwrap_or(t);
                        if matches!(self.arena[text_id.0].kind, NodeKind::Text(_))
                            && self.arena[text_id.0].style.user_select != UserSelect::None
                        {
                            let idx = self.char_index_at(text_id, pos.x);
                            self.drag_anchor = Some((text_id, idx));
                        } else {
                            self.drag_anchor = None;
                        }
                    }
                } else {
                    self.selection = None;
                    self.last_click = None;
                    self.drag_anchor = None;
                }
            }
            InputEvent::PointerUp { .. } => {
                self.drag_anchor = None;
                // Collect all nodes that were active — we need to restyle each
                let active_ids: Vec<NodeId> = self
                    .arena
                    .iter()
                    .enumerate()
                    .filter(|(_, s)| s.active)
                    .map(|(i, _)| NodeId(i))
                    .collect();

                if !active_ids.is_empty() {
                    // Clear active flag first
                    for &id in &active_ids {
                        self.arena[id.0].active = false;
                    }
                    // Restyle every node that lost :active — not just the target path
                    for &id in &active_ids {
                        self.restyle_if_sheet(id);
                    }
                    restyled = true;
                }
            }

            // ── Text editing for focused editable nodes ─────────────
            InputEvent::TextInput { text } => {
                if let Some(fid) = self.focused {
                    if self.arena[fid.0].value.is_some() {
                        let slot = &mut self.arena[fid.0];
                        let val = slot.value.as_mut().unwrap();
                        let caret = slot.caret.min(val.len());
                        val.insert_str(caret, text);
                        slot.caret = caret + text.len();
                        self.sync_value_text(fid);
                        self.needs_layout = true;
                        self.needs_paint = true;
                    }
                }
            }
            InputEvent::KeyDown { key, modifiers } => {
                if let Some(fid) = self.focused {
                    if self.arena[fid.0].value.is_some() {
                        let handled = self.handle_edit_key(fid, key, *modifiers);
                        if handled {
                            self.sync_value_text(fid);
                            self.needs_layout = true;
                            self.needs_paint = true;
                        }
                    }
                }
            }

            InputEvent::Scroll { pos, delta } => {
                self.scroll(*pos, *delta);
            }

            _ => {}
        }

        // ── Standard dispatch ───────────────────────────────────────
        let Some(target) = target else {
            return DispatchResult {
                restyled,
                ..Default::default()
            };
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

        // Resolve cursor: walk from target up, find first non-default.
        // Text nodes implicitly use a text cursor (browser default).
        let cursor = {
            let mut cursor_str = String::new();
            let mut cid = target;
            loop {
                let slot = self.slot(cid);
                let c = slot.style.cursor;
                if c != Cursor::Default {
                    cursor_str = c.to_css().to_string();
                    break;
                }
                // Text nodes default to text cursor (CSS spec).
                if matches!(slot.kind, NodeKind::Text(_))
                    && slot.style.user_select != UserSelect::None
                {
                    cursor_str = "text".to_string();
                    break;
                }
                match slot.parent {
                    Some(p) => cid = p,
                    None => break,
                }
            }
            cursor_str
        };

        DispatchResult {
            tags,
            stopped: ctx.stopped,
            default_prevented: ctx.default_prevented,
            restyled,
            cursor,
        }
    }

    /// Restyle a node and its ancestor path — re-resolve base style + pseudo-class overrides.
    fn restyle_path(&mut self, leaf: NodeId) {
        let sheet = match &self.sheet {
            Some(s) => Arc::clone(s),
            None => return,
        };
        let mut id = leaf;
        loop {
            self.restyle_node(id, &sheet);
            match self.slot(id).parent {
                Some(p) => id = p,
                None => break,
            }
        }
    }

    /// Restyle a single node if a stylesheet is available.
    fn restyle_if_sheet(&mut self, id: NodeId) {
        if let Some(sheet) = self.sheet.clone() {
            self.restyle_node(id, &sheet);
        }
    }

    /// Re-resolve a single node's style: base + pseudo-class overrides from complex rules.
    /// If the node has CSS transition specs, property changes become animated transitions
    /// instead of instant snaps.
    fn restyle_node(&mut self, id: NodeId, sheet: &StyleSheet) {
        let slot = &self.arena[id.0];
        let old_style = slot.style.clone();
        let mut new_style = slot.base_style.clone();
        let hovered = slot.hovered;
        let active = slot.active;

        // Apply matching complex rules (pseudo-classes)
        for rule in sheet.complex_rules() {
            if self.matches_complex_rule(id, rule, hovered, active) {
                apply_ops(&mut new_style, &rule.payload.ops);
            }
        }

        if old_style == new_style {
            self.arena[id.0].transitions.clear();
            return;
        }

        // Style changed — classify as layout or paint-only.
        let is_layout_change = old_style.differs_in_layout(&new_style);

        // Collect transition specs from all classes on this node
        let slot = &self.arena[id.0];
        let mut transition_specs = Vec::new();
        for cls in &slot.class_list {
            for spec in sheet.class_transitions(cls) {
                transition_specs.push(spec.clone());
            }
        }

        if transition_specs.is_empty() {
            // No transitions — snap directly
            self.arena[id.0].style = new_style;
            self.arena[id.0].transitions.clear();
        } else {
            // Create one ActiveTransition per spec, each with its own timing.
            // If an "all" spec exists, it covers every property; individual specs
            // override the "all" timing for their specific property.
            let all_spec = transition_specs.iter().find(|s| s.property == "all");

            let mut transitions = Vec::new();
            // Gather the set of individually-specified properties
            let specific: Vec<_> = transition_specs
                .iter()
                .filter(|s| s.property != "all")
                .collect();

            if !specific.is_empty() {
                for spec in &specific {
                    transitions.push(ActiveTransition::from_spec(
                        spec,
                        old_style.clone(),
                        new_style.clone(),
                    ));
                }
                // If there's also an "all" spec, add it for remaining properties
                if let Some(a) = all_spec {
                    transitions.push(ActiveTransition::from_spec(a, old_style, new_style));
                }
            } else if let Some(a) = all_spec {
                // Only "all" — single transition
                transitions.push(ActiveTransition::from_spec(a, old_style, new_style));
            }

            self.arena[id.0].transitions = transitions;
        }

        // Propagate dirty flags based on what changed.
        let flags = if is_layout_change {
            Dirty::LAYOUT | Dirty::PAINT
        } else {
            Dirty::PAINT
        };
        self.mark_dirty(id, flags);
    }

    /// Check if a node matches a complex selector rule given current pseudo-class state.
    fn matches_complex_rule(
        &self,
        id: NodeId,
        rule: &super::css::ComplexRule,
        hovered: bool,
        active: bool,
    ) -> bool {
        use super::css::{Combinator, PseudoClass};

        let segs = &rule.selector.segments;
        if segs.is_empty() {
            return false;
        }

        // Match from right to left (last segment must match the target node)
        let (_, ref last_seg) = segs[segs.len() - 1];

        // Check pseudo-classes on the final segment
        for pc in &last_seg.pseudos {
            match pc {
                PseudoClass::Hover if !hovered => return false,
                PseudoClass::Active if !active => return false,
                PseudoClass::Focus => {
                    if !self.slot(id).focused {
                        return false;
                    }
                }
                PseudoClass::FirstChild => {
                    if let Some(pid) = self.slot(id).parent {
                        let children = &self.slot(pid).children;
                        if children.first() != Some(&id) {
                            return false;
                        }
                    }
                }
                PseudoClass::LastChild => {
                    if let Some(pid) = self.slot(id).parent {
                        let children = &self.slot(pid).children;
                        if children.last() != Some(&id) {
                            return false;
                        }
                    }
                }
                PseudoClass::NthChild(a, b) => {
                    if let Some(pid) = self.slot(id).parent {
                        let children = &self.slot(pid).children;
                        // CSS nth-child is 1-indexed
                        let idx = children
                            .iter()
                            .position(|&c| c == id)
                            .map(|i| i as i32 + 1)
                            .unwrap_or(0);
                        let matches = if *a == 0 {
                            idx == *b
                        } else {
                            let diff = idx - b;
                            diff % a == 0 && diff / a >= 0
                        };
                        if !matches {
                            return false;
                        }
                    }
                }
                _ => {} // :visited, :hover/active already handled
            }
        }

        // Check tag/class/id on the final segment
        if !self.segment_matches_node(id, last_seg) {
            return false;
        }

        // Walk remaining segments right-to-left
        if segs.len() == 1 {
            return true;
        }

        let mut cur = id;
        for i in (0..segs.len() - 1).rev() {
            let (ref comb, ref seg) = segs[i];
            match comb {
                Combinator::Child => {
                    let Some(pid) = self.slot(cur).parent else {
                        return false;
                    };
                    if !self.segment_matches_node(pid, seg) {
                        return false;
                    }
                    cur = pid;
                }
                Combinator::Descendant | Combinator::None => {
                    let mut found = false;
                    let mut ancestor = self.slot(cur).parent;
                    while let Some(aid) = ancestor {
                        if self.segment_matches_node(aid, seg) {
                            cur = aid;
                            found = true;
                            break;
                        }
                        ancestor = self.slot(aid).parent;
                    }
                    if !found {
                        return false;
                    }
                }
            }
        }

        true
    }

    /// Check if a single selector segment matches a node (tag/class/id, not pseudos).
    fn segment_matches_node(&self, id: NodeId, seg: &super::css::SelectorSegment) -> bool {
        let slot = &self.arena[id.0];

        if seg.universal {
            // `*` matches everything
        } else if let Some(ref tag) = seg.tag {
            if slot.element != *tag {
                return false;
            }
        }

        for cls in &seg.classes {
            if !slot.class_list.iter().any(|c| c == cls) {
                return false;
            }
        }

        if let Some(ref seg_id) = seg.id {
            match &slot.id {
                Some(node_id) if node_id == seg_id => {}
                _ => return false,
            }
        }

        true
    }

    /// Compute the character index at a screen x-position within a text node.
    /// Uses `measured_text_width` for proportional accuracy when available,
    /// falling back to `style.char_width()` monospace estimate.
    fn char_index_at(&self, text_id: NodeId, x: f64) -> usize {
        let slot = &self.arena[text_id.0];
        let NodeKind::Text(ref content) = slot.kind else {
            return 0;
        };
        let s = &slot.style;
        let display = s.transform_text(content);
        let char_count = display.chars().count();
        if char_count == 0 {
            return 0;
        }
        let r = slot.rect;
        let total_w = slot.text_w(&display);
        let avail_w = r.size.w() - s.padding.left - s.padding.right;
        let align_offset = s.text_align_offset(total_w, avail_w);
        let text_x = r.origin.x + s.padding.left + s.text_indent + align_offset;
        let char_w = total_w / char_count as f64;
        let click_offset = (x - text_x).max(0.0);
        let char_idx = (click_offset / char_w).round() as usize;
        // Convert character index back to byte offset for consistency
        display
            .char_indices()
            .nth(char_idx.min(char_count))
            .map(|(i, _)| i)
            .unwrap_or(display.len())
    }

    /// Double-click word selection: find the text node at `target`, compute
    /// which character the click lands on, then select the whole word.
    fn select_word_at(&mut self, target: NodeId, pos: Point) {
        // Walk to find the text node (target might be a box parent of text)
        let text_id = self.find_text_child(target).unwrap_or(target);
        let slot = &self.arena[text_id.0];
        if slot.style.user_select == UserSelect::None {
            return;
        }
        let NodeKind::Text(ref content) = slot.kind else {
            return;
        };
        let text = content.clone();

        let char_idx = self
            .char_index_at(text_id, pos.x)
            .min(text.len().saturating_sub(1));

        // Find word boundaries around char_idx
        let bytes = text.as_bytes();
        let mut start = char_idx;
        let mut end = char_idx;
        while start > 0 && is_word_char(bytes[start - 1]) {
            start -= 1;
        }
        while end < bytes.len() && is_word_char(bytes[end]) {
            end += 1;
        }
        // If we clicked on a non-word char, select just that char
        if start == end && char_idx < bytes.len() {
            end = char_idx + 1;
        }

        if start < end {
            self.selection = Some(TextSelection {
                node: text_id,
                start,
                end,
            });
        }
    }

    /// Find the first Text child of a node (for selection on box containers).
    fn find_text_child(&self, id: NodeId) -> Option<NodeId> {
        if matches!(self.arena[id.0].kind, NodeKind::Text(_)) {
            return Some(id);
        }
        for &child in &self.arena[id.0].children.clone() {
            if matches!(self.arena[child.0].kind, NodeKind::Text(_)) {
                return Some(child);
            }
        }
        None
    }

    /// Apply a scroll delta to a node (or the nearest scrollable ancestor).
    pub fn scroll(&mut self, pos: Point, delta: Point) {
        let Some(mut id) = self.hit_test(pos) else {
            return;
        };
        loop {
            if self.slot(id).style.overflow == Overflow::Scroll {
                // Compute content extent to clamp scroll
                let slot = self.slot(id);
                let visible_h = slot.rect.size.h();
                let visible_w = slot.rect.size.w();
                let mut content_h = 0.0_f64;
                let mut content_w = 0.0_f64;
                for &cid in &slot.children {
                    let cr = self.slot(cid).rect;
                    let cy = cr.origin.y + cr.size.h() + slot.scroll.y - slot.rect.origin.y;
                    let cx = cr.origin.x + cr.size.w() + slot.scroll.x - slot.rect.origin.x;
                    content_h = content_h.max(cy);
                    content_w = content_w.max(cx);
                }
                let max_y = (content_h - visible_h).max(0.0);
                let max_x = (content_w - visible_w).max(0.0);
                let s = &mut self.slot_mut(id).scroll;
                s.x = (s.x - delta.x).clamp(0.0, max_x);
                s.y = (s.y - delta.y).clamp(0.0, max_y);
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
#[path = "../tree_tests.rs"]
mod tests;
