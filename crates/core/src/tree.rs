//! Generic arena tree — framework-agnostic node storage, dirty tracking, traversal.
//!
//! This module defines the **universal tree infrastructure** that any tree-based
//! system (DOM, JS scene graph, 3D hierarchy, etc.) builds on. It owns:
//!
//! - [`NodeId`] — lightweight arena handle
//! - [`Dirty`] — per-node change flags that gate layout/paint phases
//! - [`NodeStyle`] — trait for anything that provides layout-relevant info
//! - [`Arena`] — flat `Vec<Node<S>>` arena with traversal helpers
//!
//! ## Design principle
//!
//! HTML/CSS is just one **surface** that maps onto this tree. A JavaScript
//! engine, a 3D scene graph, or a native widget toolkit can all use the same
//! arena + dirty tracking + traversal without touching DOM code.

use crate::layout::{Point, Rect};

// ═══════════════════════════════════════════════════════════════════════════
// ── NodeId ──────────────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

/// Lightweight handle into the arena.  Cheap to copy, compare, hash.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub usize);

// ═══════════════════════════════════════════════════════════════════════════
// ── Dirty flags ─────────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

/// Per-node dirty flags — gate which phases need re-running.
///
/// `LAYOUT` implies `PAINT` (anything that moved must also be repainted).
/// `Z_ORDER` means the cached z-sorted child list must be rebuilt.
/// Implemented without `bitflags` crate to keep core dep-free for this module.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dirty(u8);

impl Dirty {
    pub const LAYOUT: Self = Self(0b0001);
    pub const PAINT: Self = Self(0b0010);
    pub const Z_ORDER: Self = Self(0b0100);

    pub const fn empty() -> Self {
        Self(0)
    }
    pub const fn all() -> Self {
        Self(0b0111)
    }

    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }
    #[inline]
    pub fn insert(&mut self, other: Self) {
        self.0 |= other.0;
    }
    #[inline]
    pub fn remove(&mut self, other: Self) {
        self.0 &= !other.0;
    }
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

impl Default for Dirty {
    fn default() -> Self {
        Self::empty()
    }
}

impl std::ops::BitOr for Dirty {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl std::ops::BitOrAssign for Dirty {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ── NodeStyle trait ─────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

/// Trait for anything that provides style / layout information for a tree node.
///
/// DOM's `Style` implements this. A JS engine's style object would too.
/// A 3D transform node could also implement this with different semantics.
pub trait NodeStyle: Clone + std::fmt::Debug {
    /// Whether this node should be laid out (`false` = display:none equivalent).
    fn is_visible(&self) -> bool;

    /// Stacking order hint. `None` = document order.
    fn z_order(&self) -> Option<i32>;

    /// Whether this node is positioned out of the normal flow
    /// (absolute/fixed in CSS, free-floating in a 3D scene, etc.).
    fn is_out_of_flow(&self) -> bool;
}

// ═══════════════════════════════════════════════════════════════════════════
// ── Node ────────────────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

/// One node in the arena — the minimal scaffolding every tree consumer shares.
///
/// Framework-specific data (DOM: hover/active/transitions, JS: bindings,
/// 3D: transform matrix) is stored in `data: D`.
#[derive(Debug, Clone)]
pub struct Node<D> {
    /// Framework-specific payload.
    pub data: D,
    pub parent: Option<NodeId>,
    pub children: Vec<NodeId>,
    /// Computed layout rect (filled by the layout phase).
    pub rect: Rect,
    /// Optional event-matching tag — looked up by the host.
    pub tag: Option<String>,
    /// Per-node dirty flags — gate layout/paint traversal.
    pub dirty: Dirty,
    /// Cached z-sorted children — invalidated when `Dirty::Z_ORDER` is set.
    z_sorted: Vec<NodeId>,
}

impl<D> Node<D> {
    pub fn new(data: D, parent: Option<NodeId>) -> Self {
        Self {
            data,
            parent,
            children: Vec::new(),
            rect: Rect::ZERO,
            tag: None,
            dirty: Dirty::LAYOUT | Dirty::PAINT | Dirty::Z_ORDER,
            z_sorted: Vec::new(),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ── Arena ───────────────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

/// A flat arena of `Node<D>` — the universal tree container.
///
/// All tree operations (add, remove, traverse, dirty propagation) live here.
/// Layout and paint are left to the consumer since they depend on the
/// framework-specific data in `D`.
pub struct Arena<D> {
    pub nodes: Vec<Node<D>>,
    pub root: NodeId,
    pub needs_layout: bool,
    pub needs_paint: bool,
}

impl<D> Arena<D> {
    /// Create a new arena with a single root node.
    pub fn new(root_data: D) -> Self {
        Self {
            nodes: vec![Node::new(root_data, None)],
            root: NodeId(0),
            needs_layout: true,
            needs_paint: true,
        }
    }

    /// Read-only access to a node by id.
    #[inline]
    pub fn node(&self, id: NodeId) -> &Node<D> {
        &self.nodes[id.0]
    }

    /// Mutable access to a node by id.
    #[inline]
    pub fn node_mut(&mut self, id: NodeId) -> &mut Node<D> {
        &mut self.nodes[id.0]
    }

    /// Allocate a new child node under `parent`. Returns the new `NodeId`.
    pub fn add(&mut self, parent: NodeId, data: D) -> NodeId {
        let id = NodeId(self.nodes.len());
        self.nodes.push(Node::new(data, Some(parent)));
        self.nodes[parent.0].children.push(id);
        id
    }

    /// Tag a node for event matching.
    pub fn tag(&mut self, id: NodeId, tag: impl Into<String>) {
        self.nodes[id.0].tag = Some(tag.into());
    }

    /// Walk from `start` to root, calling `f` on each node along the path.
    pub fn walk_ancestors(&mut self, start: NodeId, mut f: impl FnMut(&mut Node<D>)) {
        let mut id = start;
        loop {
            f(&mut self.nodes[id.0]);
            match self.nodes[id.0].parent {
                Some(p) => id = p,
                None => break,
            }
        }
    }

    /// Walk from `start` to root (read-only).
    pub fn walk_ancestors_ref(&self, start: NodeId, mut f: impl FnMut(&Node<D>)) {
        let mut id = start;
        loop {
            f(&self.nodes[id.0]);
            match self.nodes[id.0].parent {
                Some(p) => id = p,
                None => break,
            }
        }
    }

    /// Build the ancestor path (root → target) for a given node.
    pub fn ancestor_path(&self, target: NodeId) -> Vec<NodeId> {
        let mut path = Vec::new();
        let mut cur = Some(target);
        while let Some(id) = cur {
            path.push(id);
            cur = self.nodes[id.0].parent;
        }
        path.reverse();
        path
    }

    /// Collect tags along a path (root → target order).
    pub fn collect_tags(&self, path: &[NodeId]) -> Vec<String> {
        path.iter()
            .filter_map(|&id| self.nodes[id.0].tag.clone())
            .collect()
    }

    /// Mark a node (and ancestors) dirty, propagating upward.
    /// `LAYOUT` implies `PAINT`.
    pub fn mark_dirty(&mut self, id: NodeId, flags: Dirty) {
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
            let node = &mut self.nodes[cur.0];
            let before = node.dirty;
            node.dirty |= flags;
            if node.dirty == before {
                break;
            }
            match node.parent {
                Some(p) => cur = p,
                None => break,
            }
        }
    }

    /// Rebuild the z-sorted child cache for a node.
    ///
    /// Requires a closure that extracts z-index from the node data,
    /// keeping this generic over any framework's style type.
    pub fn rebuild_z_order(&mut self, id: NodeId, z_fn: impl Fn(&D) -> Option<i32>) {
        let children = self.nodes[id.0].children.clone();
        let any_z = children
            .iter()
            .any(|c| z_fn(&self.nodes[c.0].data).is_some());
        let mut sorted = children;
        if any_z {
            sorted.sort_by_key(|c| z_fn(&self.nodes[c.0].data).unwrap_or(0));
        }
        self.nodes[id.0].z_sorted = sorted;
        self.nodes[id.0].dirty.remove(Dirty::Z_ORDER);
    }

    /// Get the cached z-sorted children (must call `rebuild_z_order` first).
    pub fn z_sorted_children(&self, id: NodeId) -> &[NodeId] {
        &self.nodes[id.0].z_sorted
    }

    /// Hit-test: find the deepest node at `pos`.
    ///
    /// Takes a `visible_fn` to check visibility and a `pointer_fn` to check
    /// if pointer events are enabled, keeping it framework-agnostic.
    pub fn hit_test(
        &self,
        id: NodeId,
        pos: Point,
        visible_fn: &impl Fn(&D) -> bool,
        transform_fn: &impl Fn(&D, Rect) -> Rect,
    ) -> Option<NodeId> {
        let node = &self.nodes[id.0];
        if !visible_fn(&node.data) {
            return None;
        }
        let test_rect = transform_fn(&node.data, node.rect);
        if !test_rect.contains(pos) {
            return None;
        }
        // Highest z-index first (reverse z-sorted order).
        for &child_id in self.z_sorted_children(id).iter().rev() {
            if let Some(hit) = self.hit_test(child_id, pos, visible_fn, transform_fn) {
                return Some(hit);
            }
        }
        Some(id)
    }

    /// Clear all dirty flags after a paint pass.
    pub fn post_paint(&mut self) {
        for node in &mut self.nodes {
            node.dirty = Dirty::empty();
        }
        self.needs_layout = false;
        self.needs_paint = false;
    }

    /// Number of nodes in the arena.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ── Tests ───────────────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone)]
    struct TestData {
        visible: bool,
        z: Option<i32>,
    }

    impl Default for TestData {
        fn default() -> Self {
            Self {
                visible: true,
                z: None,
            }
        }
    }

    impl NodeStyle for TestData {
        fn is_visible(&self) -> bool {
            self.visible
        }
        fn z_order(&self) -> Option<i32> {
            self.z
        }
        fn is_out_of_flow(&self) -> bool {
            false
        }
    }

    #[test]
    fn arena_add_and_traverse() {
        let mut arena = Arena::new(TestData::default());
        let child1 = arena.add(arena.root, TestData::default());
        let child2 = arena.add(arena.root, TestData::default());
        let grandchild = arena.add(child1, TestData::default());

        assert_eq!(arena.len(), 4);
        assert_eq!(arena.node(arena.root).children.len(), 2);
        assert_eq!(arena.node(child1).children.len(), 1);
        assert_eq!(arena.node(child2).children.len(), 0);
        assert_eq!(arena.node(grandchild).parent, Some(child1));
    }

    #[test]
    fn ancestor_path_root_to_target() {
        let mut arena = Arena::new(TestData::default());
        let c = arena.add(arena.root, TestData::default());
        let gc = arena.add(c, TestData::default());

        let path = arena.ancestor_path(gc);
        assert_eq!(path, vec![arena.root, c, gc]);
    }

    #[test]
    fn dirty_propagation() {
        let mut arena = Arena::new(TestData::default());
        let child = arena.add(arena.root, TestData::default());
        // Clear initial dirty
        arena.post_paint();
        assert!(!arena.needs_layout);
        assert!(!arena.needs_paint);

        arena.mark_dirty(child, Dirty::LAYOUT);
        assert!(arena.needs_layout);
        assert!(arena.needs_paint);
        assert!(arena.node(child).dirty.contains(Dirty::LAYOUT));
        assert!(arena.node(child).dirty.contains(Dirty::PAINT));
        // Parent also marked
        assert!(arena.node(arena.root).dirty.contains(Dirty::LAYOUT));
    }

    #[test]
    fn dirty_flags_bitops() {
        let mut d = Dirty::empty();
        assert!(d.is_empty());
        d |= Dirty::LAYOUT;
        assert!(d.contains(Dirty::LAYOUT));
        assert!(!d.contains(Dirty::PAINT));
        d.insert(Dirty::PAINT);
        assert!(d.contains(Dirty::LAYOUT | Dirty::PAINT));
        d.remove(Dirty::LAYOUT);
        assert!(!d.contains(Dirty::LAYOUT));
        assert!(d.contains(Dirty::PAINT));
    }

    #[test]
    fn collect_tags_along_path() {
        let mut arena = Arena::new(TestData::default());
        arena.tag(arena.root, "root");
        let mid = arena.add(arena.root, TestData::default());
        // mid has no tag
        let leaf = arena.add(mid, TestData::default());
        arena.tag(leaf, "leaf");

        let path = arena.ancestor_path(leaf);
        let tags = arena.collect_tags(&path);
        assert_eq!(tags, vec!["root".to_string(), "leaf".to_string()]);
    }

    #[test]
    fn z_order_rebuild() {
        let mut arena = Arena::new(TestData::default());
        let a = arena.add(
            arena.root,
            TestData {
                visible: true,
                z: Some(2),
            },
        );
        let b = arena.add(
            arena.root,
            TestData {
                visible: true,
                z: Some(1),
            },
        );
        let c = arena.add(
            arena.root,
            TestData {
                visible: true,
                z: None,
            },
        );

        arena.rebuild_z_order(arena.root, |d| d.z);
        let sorted = arena.z_sorted_children(arena.root);
        // z=None(0) first, z=1 second, z=2 third
        assert_eq!(sorted[0], c);
        assert_eq!(sorted[1], b);
        assert_eq!(sorted[2], a);
    }
}
