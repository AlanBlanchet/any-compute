use super::graph::VisualGraph;
use super::style::{NodeStyle, tag_color};
use crate::layout::{Rect, V};
use crate::render::{Color, RenderList};

// ═══════════════════════════════════════════════════════════════════════════
// ── GraphNode trait — any object can be a visual node ───────────────────
// ═══════════════════════════════════════════════════════════════════════════

/// Trait for any object that can appear as a node in a visual graph.
///
/// Implement this to place arbitrary objects on a 2D or 3D canvas.
/// Default methods provide sensible behavior — override only what you need.
pub trait GraphNode<const D: usize = 2>: Send {
    /// Node label (displayed in header).
    fn label(&self) -> &str;

    /// Operation type tag (drives coloring via [`tag_color`]).
    fn tag(&self) -> &str {
        ""
    }

    /// Position in parent coordinate space.
    fn position(&self) -> V<D>;

    /// Visual size (width, height).
    fn size(&self) -> V<2> {
        V([120.0, 50.0])
    }

    /// Input port labels.
    fn inputs(&self) -> &[String] {
        &[]
    }

    /// Output port labels.
    fn outputs(&self) -> &[String] {
        &[]
    }

    /// Optional sub-graph (for hierarchical expansion).
    fn children(&self) -> Option<&VisualGraph<D>> {
        None
    }

    /// Custom body rendering inside the node box.
    fn render_body(&self, _list: &mut RenderList, _inner: Rect) {}
}

// ═══════════════════════════════════════════════════════════════════════════
// ── VNode — concrete positioned node ────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

/// A concrete node in a visual graph.
///
/// Implements [`GraphNode`] so it integrates with the rendering pipeline.
/// Use `VNode::new()` for quick construction or the builder pattern.
pub struct VNode<const D: usize> {
    pub pos: V<D>,
    pub dims: V<2>,
    pub label: String,
    pub tag: String,
    pub ports_in: Vec<String>,
    pub ports_out: Vec<String>,
    pub style: NodeStyle,
    pub sub: Option<VisualGraph<D>>,
}

impl<const D: usize> VNode<D> {
    /// Create a node at a given position with a label and tag.
    pub fn new(pos: V<D>, label: impl Into<String>, tag: impl Into<String>) -> Self {
        let tag_s: String = tag.into();
        let header_color = tag_color(&tag_s);
        Self {
            pos,
            dims: V([120.0, 50.0]),
            label: label.into(),
            tag: tag_s,
            ports_in: Vec::new(),
            ports_out: Vec::new(),
            style: NodeStyle {
                header: header_color,
                ..NodeStyle::default()
            },
            sub: None,
        }
    }

    /// Set visual size.
    pub fn size(mut self, w: f64, h: f64) -> Self {
        self.dims = V([w, h]);
        self
    }

    /// Add input ports.
    pub fn ins(mut self, ports: &[&str]) -> Self {
        self.ports_in = ports.iter().map(|s| s.to_string()).collect();
        self
    }

    /// Add output ports.
    pub fn outs(mut self, ports: &[&str]) -> Self {
        self.ports_out = ports.iter().map(|s| s.to_string()).collect();
        self
    }

    /// Attach a sub-graph (enables hierarchical expansion).
    pub fn children(mut self, sub: VisualGraph<D>) -> Self {
        self.sub = Some(sub);
        self
    }
}

impl<const D: usize> GraphNode<D> for VNode<D> {
    fn label(&self) -> &str {
        &self.label
    }
    fn tag(&self) -> &str {
        &self.tag
    }
    fn position(&self) -> V<D> {
        self.pos
    }
    fn size(&self) -> V<2> {
        self.dims
    }
    fn inputs(&self) -> &[String] {
        &self.ports_in
    }
    fn outputs(&self) -> &[String] {
        &self.ports_out
    }
    fn children(&self) -> Option<&VisualGraph<D>> {
        self.sub.as_ref()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ── VEdge — connection between nodes ────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

/// A directed edge between two nodes.
pub struct VEdge {
    /// Source node index.
    pub from: usize,
    /// Source output port index.
    pub from_port: usize,
    /// Target node index.
    pub to: usize,
    /// Target input port index.
    pub to_port: usize,
    /// Edge color.
    pub color: Color,
    /// Line width.
    pub width: f64,
}

impl VEdge {
    pub fn new(from: usize, to: usize) -> Self {
        Self {
            from,
            from_port: 0,
            to,
            to_port: 0,
            color: Color::rgb(150, 150, 170),
            width: 1.5,
        }
    }

    pub fn ports(mut self, from_port: usize, to_port: usize) -> Self {
        self.from_port = from_port;
        self.to_port = to_port;
        self
    }

    pub fn color(mut self, c: impl Into<Color>) -> Self {
        self.color = c.into();
        self
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ── From conversions — wrap anything into a VNode ───────────────────────
// ═══════════════════════════════════════════════════════════════════════════

impl<const D: usize> From<(&str, &str)> for VNode<D>
where
    V<D>: Default,
{
    /// Create a node from (label, tag) at the origin.
    fn from((label, tag): (&str, &str)) -> Self {
        Self::new(V::default(), label, tag)
    }
}

impl<const D: usize> From<&str> for VNode<D>
where
    V<D>: Default,
{
    /// Create a node from just a label at the origin.
    fn from(label: &str) -> Self {
        Self::new(V::default(), label, "")
    }
}
