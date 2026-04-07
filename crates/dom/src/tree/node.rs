use any_compute_core::hints::Hints;
use any_compute_core::layout::{Point, Rect};
use any_compute_core::render::Color;
use any_compute_core::tree::Dirty;
use super::animation::{ActiveAnimation, ActiveTransition};
use super::Tree;
use crate::css::StyleSheet;
use crate::style::*;

pub use any_compute_core::tree::NodeId;

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

// ── Element abstraction ─────────────────────────────────────────────────────

/// A typed DOM element with user-agent default styles resolved.
///
/// Created via `"button".to_dom()` or `HtmlTag::Button.to_dom()`.
/// Add to a tree with [`Tree::add_element`] or [`DomElement::add_to`].
pub struct DomElement {
    pub tag: HtmlTag,
    pub kind: NodeKind,
    pub style: Style,
}

impl DomElement {
    /// Insert this element into `tree` as a child of `parent`.
    pub fn add_to(self, tree: &mut Tree, parent: NodeId) -> NodeId {
        let id = tree.add_node(parent, self.kind, self.style);
        tree.slot_mut(id).element = self.tag.as_str().to_string();
        id
    }
}

/// Convert to a [`DomElement`] with user-agent default styles.
///
/// CSS specificity `(a,b,c,d)` governs how these defaults interact
/// with other rules:
///   - **d** (element selector, lowest) — UA tag defaults live here
///   - **c** (class/attribute/pseudo) — `.my-btn { ... }` overrides UA
///   - **b** (ID selector) — `#submit { ... }` overrides classes
///   - **a** (inline style, highest) — direct `Style` builder overrides all
///
/// ```ignore
/// // Get a fully-styled button element:
/// let el = "button".to_dom();
/// // → DomElement with cursor:pointer, padding, centered text, border
///
/// // Add to tree:
/// let btn = el.add_to(&mut tree, parent);
///
/// // Or use Tree's convenience method:
/// let btn = tree.add_element(parent, "button");
/// ```
pub trait ToDom {
    fn to_dom(&self) -> DomElement;
}

impl ToDom for str {
    fn to_dom(&self) -> DomElement {
        let tag = HtmlTag::from(self);
        tag.to_dom()
    }
}

impl ToDom for HtmlTag {
    fn to_dom(&self) -> DomElement {
        use std::sync::OnceLock;
        static UA_SHEET: OnceLock<StyleSheet> = OnceLock::new();
        let sheet = UA_SHEET.get_or_init(|| StyleSheet::parse(crate::css::UA_CSS));

        let style = sheet.tag(self.as_str());
        let kind = match self.kind() {
            TagKind::Box => NodeKind::Box,
            TagKind::Text => NodeKind::Text(String::new()),
            TagKind::Bar => NodeKind::Bar {
                fraction: 0.0,
                fill: Color::WHITE,
            },
        };
        DomElement {
            tag: *self,
            kind,
            style,
        }
    }
}

impl<T: ToDom + ?Sized> ToDom for &T {
    fn to_dom(&self) -> DomElement {
        (**self).to_dom()
    }
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
    pub(super) z_sorted: Vec<NodeId>,
    /// Editable text value — for `<input>`, `<textarea>`, or `contenteditable`.
    /// When `Some`, the node accepts `TextInput` / `KeyDown` when focused.
    pub value: Option<String>,
    /// Caret position inside `value` (byte offset).
    pub caret: usize,
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
            value: None,
            caret: 0,
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
