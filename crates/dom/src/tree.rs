//! Arena-based scene graph — owns the node tree, solves layout, paints, dispatches events.
//!
//! Uses a flat `Vec<Slot>` arena so the whole tree is one allocation,
//! cache-friendly, and trivially serialisable.  Node IDs are indices.

use any_compute_core::hints::Hints;
use any_compute_core::interaction::{DispatchResult, EventContext, InputEvent, Phase};
use any_compute_core::layout::{Point, Rect, Size};
use any_compute_core::render::{Border, Color, Primitive, RenderList};

use super::css::{AnimationDirection, AnimationFillMode, AnimationIterCount, Keyframe, StyleSheet};
use super::style::*;
// Re-import specific items we use in match arms for clarity.
use super::style::{
    BoxSizing, Cursor, Overflow, PointerEvents, TextDecoration, TextOverflow, Visibility,
    WhiteSpace,
};

use std::sync::Arc;

// ═══════════════════════════════════════════════════════════════════════════
// ── Dirty tracking ──────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

bitflags::bitflags! {
    /// Per-node dirty flags — gate which phases need re-running.
    ///
    /// Layout-affecting changes (dimensions, padding, flex) set `LAYOUT`.
    /// Visual-only changes (color, opacity, background) set `PAINT`.
    /// `LAYOUT` implies `PAINT` (anything that moved must also be repainted).
    /// `Z_ORDER` means the cached z-sorted child list must be rebuilt.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct Dirty: u8 {
        const LAYOUT  = 0b0001;
        const PAINT   = 0b0010;
        const Z_ORDER = 0b0100;
    }
}

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
#[derive(Debug, Clone)]
pub struct Timing {
    pub elapsed: f64,
    pub duration: f64,
    pub delay: f64,
    pub easing: any_compute_core::animation::Easing,
}

impl Timing {
    /// Seconds of active playback (after delay has passed).
    #[inline]
    pub fn active_time(&self) -> f64 {
        (self.elapsed - self.delay).max(0.0)
    }

    /// Whether enough time has elapsed to be past the delay.
    #[inline]
    pub fn started(&self) -> bool {
        self.elapsed >= self.delay
    }

    /// Linear progress in [0,1] — delay-aware, clamped, **no** easing.
    pub fn raw_progress(&self) -> f64 {
        if !self.started() {
            return 0.0;
        }
        if self.duration <= 0.0 {
            return 1.0;
        }
        (self.active_time() / self.duration).clamp(0.0, 1.0)
    }

    /// Progress with easing curve applied.
    pub fn eased_progress(&self) -> f64 {
        self.easing.apply(self.raw_progress())
    }

    /// True when the single iteration is complete.
    pub fn finished(&self) -> bool {
        self.elapsed >= self.delay + self.duration
    }
}

/// Result of a [`Tree::tick`] call.
#[derive(Debug, Clone, Copy, Default)]
pub struct TickResult {
    /// Any animations/transitions still running — caller should redraw.
    pub active: bool,
}

/// Active CSS property transition — interpolates one `StyleOp` from→to.
#[derive(Debug, Clone)]
pub struct ActiveTransition {
    pub property: String,
    pub from: Style,
    pub to: Style,
    pub timing: Timing,
}

impl ActiveTransition {
    /// Normalized progress \in [0,1] with easing applied.
    pub fn progress(&self) -> f64 {
        self.timing.eased_progress()
    }

    /// True if this transition has completed.
    pub fn finished(&self) -> bool {
        self.timing.finished()
    }

    /// Build from a `TransitionSpec` + before/after snapshots.
    fn from_spec(spec: &super::css::TransitionSpec, from: Style, to: Style) -> Self {
        Self {
            property: spec.property.clone(),
            from,
            to,
            timing: Timing {
                elapsed: 0.0,
                duration: spec.duration_secs,
                delay: spec.delay_secs,
                easing: spec.easing,
            },
        }
    }
}

/// Active CSS @keyframes animation on a node.
#[derive(Debug, Clone)]
pub struct ActiveAnimation {
    pub name: String,
    pub keyframes: Vec<Keyframe>,
    pub timing: Timing,
    pub iteration_count: AnimationIterCount,
    pub direction: AnimationDirection,
    pub fill_mode: AnimationFillMode,
    pub iterations_done: f64,
}

impl ActiveAnimation {
    /// Current normalized progress \in [0,1] within the current iteration.
    pub fn progress(&self) -> f64 {
        if !self.timing.started() {
            return match self.fill_mode {
                AnimationFillMode::Backwards | AnimationFillMode::Both => 0.0,
                _ => 0.0,
            };
        }
        let active = self.timing.active_time();
        if self.timing.duration <= 0.0 {
            return 1.0;
        }
        let raw_iter = active / self.timing.duration;
        let iter_frac = raw_iter.fract();
        let iter_num = raw_iter.floor();

        // Check if finished
        match self.iteration_count {
            AnimationIterCount::Count(n) if iter_num >= n => {
                return match self.fill_mode {
                    AnimationFillMode::Forwards | AnimationFillMode::Both => 1.0,
                    _ => 0.0,
                };
            }
            _ => {}
        }

        let t = match self.direction {
            AnimationDirection::Normal => iter_frac,
            AnimationDirection::Reverse => 1.0 - iter_frac,
            AnimationDirection::Alternate => {
                if (iter_num as u64) % 2 == 0 {
                    iter_frac
                } else {
                    1.0 - iter_frac
                }
            }
            AnimationDirection::AlternateReverse => {
                if (iter_num as u64) % 2 == 0 {
                    1.0 - iter_frac
                } else {
                    iter_frac
                }
            }
        };
        self.timing.easing.apply(t)
    }

    /// True if this animation has completed all iterations.
    pub fn finished(&self) -> bool {
        if !self.timing.started() {
            return false;
        }
        match self.iteration_count {
            AnimationIterCount::Infinite => false,
            AnimationIterCount::Count(n) => self.timing.active_time() >= self.timing.duration * n,
        }
    }

    /// Apply current keyframe interpolation to a style.
    pub fn apply_to(&self, style: &mut Style) {
        let t = self.progress();
        if self.keyframes.is_empty() {
            return;
        }

        // Single-pass keyframe bracket search: find largest stop <= t (prev)
        // and smallest stop >= t (next).
        let mut prev = &self.keyframes[0];
        let mut next = self.keyframes.last().unwrap();
        for kf in &self.keyframes {
            if kf.stop <= t {
                prev = kf;
            }
            if kf.stop >= t && kf.stop < next.stop {
                next = kf;
            }
        }

        if (next.stop - prev.stop).abs() < f64::EPSILON {
            // Exact match — apply directly
            apply_ops(style, &prev.ops);
        } else {
            // Interpolate between surrounding keyframes via full Style::lerp.
            // Clone the base style, apply each keyframe's ops independently,
            // then lerp.  Properties not touched by keyframes stay identical
            // in both copies, so the lerp is a no-op for them.
            let local_t = (t - prev.stop) / (next.stop - prev.stop);
            let mut prev_style = style.clone();
            apply_ops(&mut prev_style, &prev.ops);
            let mut next_style = style.clone();
            apply_ops(&mut next_style, &next.ops);
            *style = prev_style.lerp(&next_style, local_t);
        }
    }
}

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
    /// Style before any pseudo-class overrides — reset target for restyle.
    pub base_style: Style,
    pub hints: Hints,
    pub parent: Option<NodeId>,
    pub children: Vec<NodeId>,
    /// Computed layout rect (filled by `Tree::layout`).
    pub rect: Rect,
    /// Scroll offset for `Overflow::Scroll` containers.
    pub scroll: Point,
    /// Optional click handler tag — matched by the host.
    pub tag: Option<String>,
    /// HTML `id` attribute — for CSS `#id` selector matching.
    pub id: Option<String>,
    /// HTML element tag name (e.g. "div", "span") — for CSS restyle.
    pub element: String,
    /// CSS class list — for CSS restyle on pseudo-class changes.
    pub class_list: Vec<String>,
    /// Current hover state — set by dispatch, used for `:hover` restyle.
    pub hovered: bool,
    /// Current active (pressed) state — set by dispatch, used for `:active` restyle.
    pub active: bool,
    /// Current focused state — set by `Tree::focus()`, used for `:focus` restyle.
    pub focused: bool,
    /// Accurate text width from font shaping (0.0 = not measured, use estimate).
    pub measured_text_width: f64,
    /// Active CSS transitions (property-level interpolation on style change).
    pub transitions: Vec<ActiveTransition>,
    /// Active CSS @keyframes animations.
    pub animations: Vec<ActiveAnimation>,
    /// Per-node dirty flags — gate layout/paint traversal.
    pub dirty: Dirty,
    /// Cached z-sorted children — invalidated when `Dirty::Z_ORDER` is set.
    z_sorted: Vec<NodeId>,
}

impl Slot {
    /// Create a new slot with sensible defaults for spatial/hint fields.
    pub fn new(kind: NodeKind, style: Style, parent: Option<NodeId>) -> Self {
        Self {
            kind,
            base_style: style.clone(),
            style,
            hints: Hints::default(),
            parent,
            children: Vec::new(),
            rect: Rect::ZERO,
            scroll: Point::ZERO,
            tag: None,
            id: None,
            element: String::new(),
            class_list: Vec::new(),
            hovered: false,
            active: false,
            focused: false,
            measured_text_width: 0.0,
            transitions: Vec::new(),
            animations: Vec::new(),
            dirty: Dirty::LAYOUT | Dirty::PAINT | Dirty::Z_ORDER,
            z_sorted: Vec::new(),
        }
    }

    /// Return measured text width if available, else CSS-estimated width.
    pub fn text_w(&self, text: &str) -> f64 {
        if self.measured_text_width > 0.0 {
            self.measured_text_width
        } else {
            self.style.text_width(text)
        }
    }
}

// ── Tree ────────────────────────────────────────────────────────────────────

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

    fn add_node(&mut self, parent: NodeId, kind: NodeKind, style: Style) -> NodeId {
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
                    w += text.len() as f64 * slot.style.letter_spacing;
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
            root, viewport.w(), viewport.h(), viewport.w(), viewport.h(), 0.0, 0.0,
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
            let overflow = line_used - main_budget;
            if overflow > 0.0 && main_budget > 0.0 && !wraps {
                let total_shrink: f64 = line_indices
                    .iter()
                    .map(|&idx| self.slot(child_sizes[idx].0).style.flex_shrink)
                    .sum();
                if total_shrink > 0.0 {
                    for &idx in line_indices {
                        let cid = child_sizes[idx].0;
                        let shrink = self.slot(cid).style.flex_shrink;
                        if shrink > 0.0 {
                            let share = overflow * shrink / total_shrink;
                            if is_row {
                                let min = self
                                    .slot(cid)
                                    .style
                                    .min_width
                                    .resolve(child_avail_w)
                                    .unwrap_or(0.0);
                                child_sizes[idx].1 = (child_sizes[idx].1 - share).max(min);
                            } else {
                                let min = self
                                    .slot(cid)
                                    .style
                                    .min_height
                                    .resolve(child_avail_h)
                                    .unwrap_or(0.0);
                                child_sizes[idx].2 = (child_sizes[idx].2 - share).max(min);
                            }
                        }
                    }
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
                let max_bw = bw.top.max(bw.right).max(bw.bottom).max(bw.left);
                Some(Border {
                    color: s.apply_opacity(s.border_color),
                    width: max_bw,
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
        if s.outline_width > 0.0 && s.outline_color.a > 0 {
            let ow = s.outline_width;
            let outline_bounds = Rect::new(
                r.origin.x - ow,
                r.origin.y - ow,
                r.size.w() + ow * 2.0,
                r.size.h() + ow * 2.0,
            );
            list.push(Primitive::Rect {
                bounds: outline_bounds,
                fill: Color::TRANSPARENT,
                border: Some(Border {
                    color: s.apply_opacity(s.outline_color),
                    width: ow,
                }),
                corner_radius: s.corner_radius + ow,
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
                let align_offset = match s.text_align {
                    TextAlign::Center => (avail_w - text_w).max(0.0) / 2.0,
                    TextAlign::Right => (avail_w - text_w).max(0.0),
                    TextAlign::Left => 0.0,
                };

                let tx = r.origin.x + s.padding.left + s.text_indent + align_offset;
                let ty = r.origin.y + s.padding.top + s.font_size * TEXT_BASELINE_RATIO;
                let painted_text = final_text.into_owned();

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
                    list.push(Primitive::Line {
                        from: Point::new(tx, line_y),
                        to: Point::new(line_end, line_y),
                        stroke: s.apply_opacity(s.color),
                        width: 1.0,
                    });
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
            InputEvent::PointerMove { .. } => {
                let prev = self.hovered;
                let next = target;
                if prev != next {
                    // Un-hover previous path
                    if let Some(old) = prev {
                        self.walk_ancestors(old, |s| s.hovered = false);
                    }
                    // Hover new path
                    if let Some(new) = next {
                        self.walk_ancestors(new, |s| s.hovered = true);
                    }
                    self.hovered = next;
                    // Restyle affected paths
                    if let Some(old) = prev {
                        self.restyle_path(old);
                    }
                    if let Some(new) = next {
                        self.restyle_path(new);
                    }
                    restyled = true;
                }
            }
            InputEvent::PointerDown { .. } => {
                if let Some(t) = target {
                    self.walk_ancestors(t, |s| s.active = true);
                    self.restyle_path(t);
                    restyled = true;
                }
            }
            InputEvent::PointerUp { .. } => {
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
                if matches!(slot.kind, NodeKind::Text(_)) && slot.style.user_select != UserSelect::None {
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
#[path = "tree_tests.rs"]
mod tests;
