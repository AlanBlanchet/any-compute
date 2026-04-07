use super::node::{GraphNode, VEdge};
use super::view::GraphView;
use crate::layout::{Point, Rect, V};
use crate::render::{Color, RenderList, Renderable};

// ═══════════════════════════════════════════════════════════════════════════
// ── VisualGraph<D> — the container ──────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

/// A visual graph of positioned nodes and edges.
///
/// Generic over dimension: `VisualGraph<2>` for 2D layouts,
/// `VisualGraph<3>` for 3D spatial placement.
///
/// Nodes can contain sub-graphs, enabling infinite hierarchical zoom.
pub struct VisualGraph<const D: usize> {
    pub(super) nodes: Vec<Box<dyn GraphNode<D>>>,
    /// Layout-assigned positions (override each node's default position).
    pub(super) positions: Vec<Option<V<D>>>,
    pub edges: Vec<VEdge>,
    pub label: String,
}

impl<const D: usize> VisualGraph<D> {
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            nodes: Vec::new(),
            positions: Vec::new(),
            edges: Vec::new(),
            label: label.into(),
        }
    }

    /// Add a node and return its index.
    pub fn add(&mut self, node: impl GraphNode<D> + 'static) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(Box::new(node));
        self.positions.push(None);
        idx
    }

    /// Connect two nodes (output port 0 → input port 0).
    pub fn edge(&mut self, from: usize, to: usize) -> &mut Self {
        self.edges.push(VEdge::new(from, to));
        self
    }

    /// Connect with specific ports.
    pub fn edge_ports(
        &mut self,
        from: usize,
        from_port: usize,
        to: usize,
        to_port: usize,
    ) -> &mut Self {
        self.edges
            .push(VEdge::new(from, to).ports(from_port, to_port));
        self
    }

    /// Number of nodes.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the graph is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Access a node by index.
    pub fn node(&self, idx: usize) -> &dyn GraphNode<D> {
        &*self.nodes[idx]
    }

    /// Effective position of a node (layout override or node default).
    pub fn node_pos(&self, idx: usize) -> V<D> {
        self.positions[idx].unwrap_or_else(|| self.nodes[idx].position())
    }

    /// Set a node's position (layout override).
    pub fn set_pos(&mut self, idx: usize, pos: V<D>) {
        self.positions[idx] = Some(pos);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ── Auto-layout — simple layered graph layout ───────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

impl VisualGraph<2> {
    /// Bounding rectangle for a node at its current position.
    pub fn node_rect(&self, idx: usize) -> Rect {
        let pos = self.node_pos(idx);
        Rect::from_parts(Point::new(pos.0[0], pos.0[1]), self.nodes[idx].size())
    }

    /// Automatically position nodes in a left-to-right layered layout.
    ///
    /// Assigns each node to a layer based on longest path from inputs,
    /// then distributes nodes evenly within each layer.
    pub fn auto_layout(&mut self) {
        let n = self.nodes.len();
        if n == 0 {
            return;
        }

        // Build adjacency: who does each node feed?
        let mut incoming: Vec<Vec<usize>> = vec![vec![]; n];
        for e in &self.edges {
            incoming[e.to].push(e.from);
        }

        // Assign layers via longest path (reverse topo order).
        let mut layer_of = vec![0usize; n];
        let mut changed = true;
        while changed {
            changed = false;
            for i in 0..n {
                for &src in &incoming[i] {
                    let new_layer = layer_of[src] + 1;
                    if new_layer > layer_of[i] {
                        layer_of[i] = new_layer;
                        changed = true;
                    }
                }
            }
        }

        // Group nodes by layer.
        let max_layer = layer_of.iter().copied().max().unwrap_or(0);
        let mut layers: Vec<Vec<usize>> = vec![vec![]; max_layer + 1];
        for (i, &l) in layer_of.iter().enumerate() {
            layers[l].push(i);
        }

        // Position: horizontal by layer, vertical by index within layer.
        let h_gap = 180.0;
        let v_gap = 80.0;
        let max_per_layer = layers.iter().map(|l| l.len()).max().unwrap_or(1);
        let total_h = max_per_layer as f64 * v_gap;

        for (li, layer) in layers.iter().enumerate() {
            let x = 40.0 + li as f64 * h_gap;
            let layer_h = layer.len() as f64 * v_gap;
            let y_offset = (total_h - layer_h) / 2.0;
            for (ni, &node_idx) in layer.iter().enumerate() {
                let y = 40.0 + y_offset + ni as f64 * v_gap;
                self.set_pos(node_idx, V([x, y]));
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ── Rendering — VisualGraph<2> → RenderList ─────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

/// Compute a port's screen position given the node bounds.
pub(super) fn port_pos(bounds: Rect, is_output: bool, port_idx: usize, port_count: usize) -> Point {
    let x = if is_output {
        bounds.origin.x + bounds.size.x
    } else {
        bounds.origin.x
    };
    let spacing = bounds.size.y / (port_count as f64 + 1.0);
    let y = bounds.origin.y + spacing * (port_idx as f64 + 1.0);
    Point::new(x, y)
}

/// Render a background grid that follows zoom/pan.
///
/// Uses adaptive spacing so grid cells stay 25–100 px in screen space
/// regardless of zoom, and snaps line coordinates to whole pixels to
/// prevent sub-pixel aliasing from making lines flicker.
pub(super) fn render_grid(
    list: &mut RenderList,
    view: &GraphView,
    y_off: f64,
    xform: &dyn Fn(Point) -> Point,
) {
    let color = Color::rgba(255, 255, 255, 12);

    // Adaptive spacing: keep cells between ~25 and ~100 screen-pixels.
    let base = 50.0;
    let screen_cell = base * view.zoom;
    let spacing = if screen_cell < 25.0 {
        base * (25.0 / screen_cell).log2().ceil().exp2()
    } else {
        base
    };

    // Generous symmetric extent in screen space.
    let extent = 4000.0;
    // Visible bounds in graph space (inverse transform)
    let inv = |sx: f64, sy: f64| -> (f64, f64) {
        (
            (sx / view.zoom) - view.pan.0[0],
            ((sy - y_off) / view.zoom) - view.pan.0[1],
        )
    };
    let (vx0, vy0) = inv(-extent, y_off - extent);
    let (vx1, vy1) = inv(extent, extent);
    let x_range = (vx0 / spacing).floor() as i64..=(vx1 / spacing).ceil() as i64;
    let y_range = (vy0 / spacing).floor() as i64..=(vy1 / spacing).ceil() as i64;

    // Snap helper: round to nearest 0.5 for crisp 1px lines.
    let snap = |v: f64| -> f64 { (v * 2.0).round() / 2.0 };

    for gx in x_range {
        let x = gx as f64 * spacing;
        let p0 = xform(Point::new(x, vy0));
        let p1 = xform(Point::new(x, vy1));
        list.push_line(snap(p0.x), p0.y, snap(p1.x), p1.y, color, 1.0);
    }
    for gy in y_range {
        let y = gy as f64 * spacing;
        let p0 = xform(Point::new(vx0, y));
        let p1 = xform(Point::new(vx1, y));
        list.push_line(p0.x, snap(p0.y), p1.x, snap(p1.y), color, 1.0);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ── Graphable trait — any component → VisualGraph ───────────────────────
// ═══════════════════════════════════════════════════════════════════════════

/// Any type that can describe itself as a visual graph.
///
/// This is the universal "render me as a node diagram" entry point.
/// Implement `to_graph()` → get `render_graph()` and `capture_graph()` for free.
///
/// ## Examples
///
/// - `Network` (neural net) — shows layers as nodes, data flow as edges
/// - `Scene` — shows objects, lights, camera as connected nodes
/// - `Tree` (DOM) — shows parent→child structure
/// - Any DAG / state machine / pipeline
pub trait Graphable {
    /// Produce a visual graph representation of this component.
    fn to_graph(&self) -> VisualGraph<2>;

    /// Render the graph into a `RenderList` (generic entry point).
    ///
    /// Uses the default `GraphView` (zoom=1, no pan, top-level only).
    /// Override for custom rendering behavior.
    fn render_graph(&self) -> RenderList {
        let g = self.to_graph();
        let mut list = RenderList::default();
        let view = GraphView::default();
        g.render(&mut list, &view);
        list
    }

    /// Capture the graph into a `PixelBuffer` for testing / export.
    fn capture_graph(&self, width: u32, height: u32) -> crate::render::PixelBuffer {
        use crate::render::PixelBuffer;
        let list = self.render_graph();
        let mut pb = PixelBuffer::new(width, height, Color::rgb(25, 28, 36));
        pb.paint(&list);
        pb
    }
}
