use crate::layout::{Point, Rect, V};
use crate::render::{Border, Color, Primitive, RenderList, Renderable};
use super::graph::{port_pos, render_grid, VisualGraph};
use super::style::{NodeStyle, tag_color};

// ═══════════════════════════════════════════════════════════════════════════
// ── GraphView — interactive viewport state ──────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

/// Interactive viewport state for navigating a visual graph.
///
/// Tracks zoom level, pan offset, which sub-graph we're viewing (breadcrumb),
/// and which nodes are expanded to show their children inline.
#[derive(Clone, Debug)]
pub struct GraphView {
    /// Zoom factor (1.0 = 100%).
    pub zoom: f64,
    /// Pan offset in graph coordinates.
    pub pan: V<2>,
    /// Breadcrumb trail: stack of node indices we've "entered".
    /// Empty = viewing the root graph.
    pub breadcrumb: Vec<usize>,
    /// Set of node indices whose children are shown expanded (inline).
    /// Nodes NOT in this set show a collapsed indicator instead.
    pub expanded: Vec<usize>,
}

impl Default for GraphView {
    fn default() -> Self {
        Self {
            zoom: 1.0,
            pan: V([0.0, 0.0]),
            breadcrumb: Vec::new(),
            expanded: Vec::new(),
        }
    }
}

impl GraphView {
    /// Enter a child node (push onto breadcrumb).
    pub fn enter(&mut self, node_idx: usize) {
        self.breadcrumb.push(node_idx);
        self.expanded.clear();
        self.pan = V([0.0, 0.0]);
        self.zoom = 1.0;
    }

    /// Go back one level (pop breadcrumb). Returns the node we left.
    pub fn back(&mut self) -> Option<usize> {
        let popped = self.breadcrumb.pop();
        if popped.is_some() {
            self.expanded.clear();
            self.pan = V([0.0, 0.0]);
            self.zoom = 1.0;
        }
        popped
    }

    /// Go back to root.
    pub fn to_root(&mut self) {
        self.breadcrumb.clear();
        self.expanded.clear();
        self.pan = V([0.0, 0.0]);
        self.zoom = 1.0;
    }

    /// Navigate to a specific breadcrumb depth (0 = root).
    pub fn go_to_depth(&mut self, depth: usize) {
        if depth < self.breadcrumb.len() {
            self.breadcrumb.truncate(depth);
            self.expanded.clear();
            self.pan = V([0.0, 0.0]);
            self.zoom = 1.0;
        }
    }

    /// Toggle expand/collapse of a node's children.
    pub fn toggle(&mut self, node_idx: usize) {
        if let Some(pos) = self.expanded.iter().position(|&i| i == node_idx) {
            self.expanded.remove(pos);
        } else {
            self.expanded.push(node_idx);
        }
    }

    /// Is this node expanded?
    pub fn is_expanded(&self, node_idx: usize) -> bool {
        self.expanded.contains(&node_idx)
    }

    /// Current breadcrumb depth.
    pub fn depth(&self) -> usize {
        self.breadcrumb.len()
    }

    /// Apply zoom + pan transform to a point.
    pub fn transform(&self, p: Point) -> Point {
        Point::new(
            (p.x + self.pan.0[0]) * self.zoom,
            (p.y + self.pan.0[1]) * self.zoom,
        )
    }

    /// Apply zoom + pan to a rect.
    pub fn transform_rect(&self, r: Rect) -> Rect {
        let origin = self.transform(r.origin);
        Rect::from_parts(
            origin,
            V([r.size.0[0] * self.zoom, r.size.0[1] * self.zoom]),
        )
    }

    /// Zoom relative to a point in screen space (e.g. cursor position).
    ///
    /// Adjusts both `zoom` and `pan` so that `screen_pos` maps to the
    /// same graph-space coordinate before and after the zoom change.
    pub fn zoom_at(&mut self, screen_pos: Point, factor: f64, min: f64, max: f64) {
        let old_zoom = self.zoom;
        let new_zoom = (old_zoom * factor).clamp(min, max);
        // Anchor: graph_pt = screen / old_zoom - pan
        // After:  screen   = (graph_pt + new_pan) * new_zoom
        // Solve:  new_pan  = screen / new_zoom - graph_pt
        //       = screen / new_zoom - screen / old_zoom + pan
        self.pan.0[0] += screen_pos.x * (1.0 / new_zoom - 1.0 / old_zoom);
        self.pan.0[1] += screen_pos.y * (1.0 / new_zoom - 1.0 / old_zoom);
        self.zoom = new_zoom;
    }

    /// Hit-test: which node index is under this screen-space point?
    pub fn hit_test(&self, graph: &VisualGraph<2>, screen_pos: Point) -> Option<usize> {
        // Reverse transform: screen → graph coords
        let gx = screen_pos.x / self.zoom - self.pan.0[0];
        let gy = screen_pos.y / self.zoom - self.pan.0[1];
        // Check nodes in reverse order (topmost first)
        for idx in (0..graph.len()).rev() {
            let r = graph.node_rect(idx);
            if gx >= r.origin.x
                && gx <= r.origin.x + r.size.0[0]
                && gy >= r.origin.y
                && gy <= r.origin.y + r.size.0[1]
            {
                return Some(idx);
            }
        }
        None
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ── Rendering with GraphView — zoom, pan, collapsed children ────────────
// ═══════════════════════════════════════════════════════════════════════════

impl Renderable<GraphView> for VisualGraph<2> {
    fn render(&self, list: &mut RenderList, view: &GraphView) {
        // Resolve which sub-graph to render based on breadcrumb
        let target = self.resolve_breadcrumb(&view.breadcrumb);

        // Render breadcrumb bar if we're inside a sub-graph
        if !view.breadcrumb.is_empty() {
            self.render_breadcrumb(list, view);
        }

        // Vertical offset: leave room for breadcrumb bar
        let y_offset = if view.breadcrumb.is_empty() {
            0.0
        } else {
            28.0
        };

        // Render the resolved sub-graph with zoom/pan
        target.render_nodes_edges(list, view, y_offset);
    }
}

impl VisualGraph<2> {
    /// Walk the breadcrumb path to find the sub-graph to render.
    pub fn resolve_breadcrumb_pub(&self, breadcrumb: &[usize]) -> &VisualGraph<2> {
        self.resolve_breadcrumb(breadcrumb)
    }

    /// Walk the breadcrumb path to find the sub-graph we should render.
    fn resolve_breadcrumb(&self, breadcrumb: &[usize]) -> &VisualGraph<2> {
        let mut current = self;
        for &idx in breadcrumb {
            if idx < current.len() {
                if let Some(child) = current.node(idx).children() {
                    current = child;
                } else {
                    break;
                }
            }
        }
        current
    }

    /// Render the breadcrumb navigation bar with bevel-style segments.
    fn render_breadcrumb(&self, list: &mut RenderList, view: &GraphView) {
        let bar_h = 24.0;
        let font = 10.0;
        let skew = 10.0; // diagonal separator width
        let pad_x = 12.0; // horizontal text padding
        let bg = Color::rgb(30, 32, 40);

        // Background bar
        list.push_rect(0.0, 0.0, 4000.0, bar_h, bg);

        /// Draw a breadcrumb segment with a tilted `/` right edge.
        /// The shape is a parallelogram: left edge vertical, right edge diagonal.
        fn tilted_segment(
            list: &mut RenderList,
            x: f64,
            w: f64,
            h: f64,
            skew: f64,
            fill: Color,
            label: &str,
            font: f64,
            is_first: bool,
        ) -> f64 {
            // Draw the parallelogram as two triangles:
            //   left-top ---- right-top+skew
            //    |                 /
            //   left-bot ---- right-bot
            let lx = if is_first { x } else { x };
            let rx = x + w;
            list.push_triangle(
                [
                    Point::new(lx, 0.0),
                    Point::new(rx + skew, 0.0),
                    Point::new(lx, h),
                ],
                fill,
            );
            list.push_triangle(
                [
                    Point::new(lx, h),
                    Point::new(rx + skew, 0.0),
                    Point::new(rx, h),
                ],
                fill,
            );
            // Text centered both vertically and horizontally.
            // The 0.3 * font offset compensates for text baseline vs cell center.
            let text_w = label.len() as f64 * (font * 0.55);
            let tx = x + (w + skew - text_w) / 2.0;
            let ty = (h - font) / 2.0 + font * 0.15;
            list.push_text(tx, ty, label, font, Color::rgb(240, 240, 250));
            // Next segment starts overlapping the diagonal
            rx + 2.0
        }

        let mut x = 0.0;
        // Root segment
        let root_color = Color::rgb(60, 65, 85);
        let root_w = self.label.len() as f64 * 6.0 + pad_x * 2.0;
        x = tilted_segment(
            list,
            x,
            root_w,
            bar_h,
            skew,
            root_color,
            &self.label,
            font,
            true,
        );

        // Breadcrumb segments — each colored by node tag
        let mut current = self;
        for &idx in &view.breadcrumb {
            if idx < current.len() {
                let node = current.node(idx);
                let name = node.label();
                let color = if node.tag().is_empty() {
                    Color::rgb(80, 85, 100)
                } else {
                    tag_color(node.tag())
                };
                let seg_w = name.len() as f64 * 6.0 + pad_x * 2.0;
                x = tilted_segment(list, x, seg_w, bar_h, skew, color, name, font, false);
                if let Some(child) = node.children() {
                    current = child;
                }
            }
        }
    }

    /// Render nodes and edges with zoom/pan transform, respecting expanded state.
    fn render_nodes_edges(&self, list: &mut RenderList, view: &GraphView, y_off: f64) {
        let header_h = 22.0 * view.zoom;
        let font = 11.0 * view.zoom;
        let port_r = 4.0 * view.zoom;
        let style = NodeStyle::default();

        let xform = |p: Point| -> Point {
            Point::new(
                (p.x + view.pan.0[0]) * view.zoom,
                (p.y + view.pan.0[1]) * view.zoom + y_off,
            )
        };
        let xform_rect = |r: Rect| -> Rect {
            let o = xform(r.origin);
            Rect::from_parts(o, V([r.size.0[0] * view.zoom, r.size.0[1] * view.zoom]))
        };

        // ── Background grid ──────────────────────────────────────────
        render_grid(list, view, y_off, &xform);

        // ── Edges ────────────────────────────────────────────────────
        for e in &self.edges {
            let src = self.nodes[e.from].as_ref();
            let dst = self.nodes[e.to].as_ref();
            let src_bounds = xform_rect(self.node_rect(e.from));
            let dst_bounds = xform_rect(self.node_rect(e.to));
            let src_ports = src.outputs().len().max(1);
            let dst_ports = dst.inputs().len().max(1);
            let p1 = port_pos(src_bounds, true, e.from_port, src_ports);
            let p2 = port_pos(dst_bounds, false, e.to_port, dst_ports);

            let mid_x = (p1.x + p2.x) / 2.0;
            let w = (e.width * view.zoom).max(0.75);
            list.push_line(p1.x, p1.y, mid_x, p1.y, e.color, w);
            list.push_line(mid_x, p1.y, mid_x, p2.y, e.color, w);
            list.push_line(mid_x, p2.y, p2.x, p2.y, e.color, w);
        }

        // ── Nodes ────────────────────────────────────────────────────
        for (idx, node) in self.nodes.iter().enumerate() {
            let n = node.as_ref();
            let bounds = xform_rect(self.node_rect(idx));
            let (x, y) = (bounds.origin.x, bounds.origin.y);
            let (w, h) = (bounds.size.0[0], bounds.size.0[1]);
            let z = view.zoom;
            let tag = n.tag();
            let header_color = if tag.is_empty() {
                style.header
            } else {
                tag_color(tag)
            };
            let has_children = n.children().is_some();
            let is_expanded = view.is_expanded(idx);

            // Body background — expanded nodes get a taller box
            let body_h = if is_expanded {
                if let Some(sub) = n.children() {
                    h + sub.len() as f64 * 30.0 * z
                } else {
                    h
                }
            } else {
                h
            };

            // Drop shadow
            list.primitives.push(Primitive::Rect {
                bounds: Rect::from_parts(Point::new(x + 2.0 * z, y + 2.0 * z), V([w, body_h])),
                fill: Color::rgba(0, 0, 0, 40),
                border: None,
                corner_radius: style.corner_radius * z,
            });

            // Body
            list.primitives.push(Primitive::Rect {
                bounds: Rect::from_parts(Point::new(x, y), V([w, body_h])),
                fill: style.fill,
                border: Some(Border::uniform(style.border, 1.0)),
                corner_radius: style.corner_radius * z,
            });

            // Header bar with left-side color accent
            let accent_w = 4.0 * z;
            list.primitives.push(Primitive::Rect {
                bounds: Rect::from_parts(Point::new(x, y), V([accent_w, header_h])),
                fill: header_color,
                border: None,
                corner_radius: 0.0,
            });
            list.primitives.push(Primitive::Rect {
                bounds: Rect::from_parts(Point::new(x, y), V([w, header_h])),
                fill: header_color.with_alpha(60),
                border: None,
                corner_radius: style.corner_radius * z,
            });

            // Title text — vertically centered in header
            let title_y = y + (header_h - font) / 2.0;
            list.push_text(
                x + (accent_w + 4.0 * z),
                title_y,
                n.label(),
                font,
                style.text,
            );

            // Expand icon for nodes with children (clickable enter indicator)
            if has_children {
                let icon_s = font * 0.9;
                let icon_x = x + w - icon_s - 4.0 * z;
                let icon_y = y + (header_h - icon_s) / 2.0;
                // Circle background
                list.push_circle(
                    icon_x + icon_s / 2.0,
                    icon_y + icon_s / 2.0,
                    icon_s / 2.0 + 1.0,
                    Color::rgba(255, 255, 255, 30),
                );
                list.push_text(
                    icon_x,
                    icon_y,
                    "\u{25B6}",
                    icon_s,
                    Color::rgb(200, 200, 220),
                );
            }

            // Tag subtitle — dim text below header
            let mut content_y = y + header_h + 4.0 * z;
            if !tag.is_empty() {
                list.push_text(
                    x + (accent_w + 4.0 * z),
                    content_y,
                    tag,
                    (font - z).max(6.0 * z),
                    Color::rgb(130, 130, 150),
                );
                content_y += (font - z).max(6.0 * z) + 2.0 * z;
            }

            // Port labels (input names on left, output names on right)
            let port_font = (font - 2.0 * z).max(6.0 * z);
            let port_text_color = Color::rgb(160, 170, 185);
            let inputs = n.inputs();
            let outputs = n.outputs();
            let port_rows = inputs.len().max(outputs.len());
            let port_row_h = if port_rows > 0 {
                ((body_h - header_h - 8.0 * z) / port_rows as f64).min(14.0 * z)
            } else {
                0.0
            };
            for (i, label) in inputs.iter().enumerate() {
                let py = content_y + i as f64 * port_row_h;
                list.push_text(
                    x + (accent_w + 4.0 * z),
                    py,
                    label,
                    port_font,
                    port_text_color,
                );
            }
            for (i, label) in outputs.iter().enumerate() {
                let py = content_y + i as f64 * port_row_h;
                let tw = label.len() as f64 * port_font * 0.55;
                list.push_text(x + w - tw - 4.0 * z, py, label, port_font, port_text_color);
            }

            // Port circles
            let in_count = inputs.len().max(1);
            let out_count = outputs.len().max(1);
            let in_color = Color::rgb(80, 190, 120);
            let out_color = Color::rgb(200, 90, 90);
            for i in 0..in_count {
                let p = port_pos(bounds, false, i, in_count);
                list.push_circle(p.x, p.y, port_r, in_color);
            }
            for i in 0..out_count {
                let p = port_pos(bounds, true, i, out_count);
                list.push_circle(p.x, p.y, port_r, out_color);
            }

            // Expanded children: render sub-graph inline below header
            if is_expanded {
                if let Some(sub) = n.children() {
                    let child_view = GraphView {
                        zoom: z * 0.75,
                        pan: V([0.0, 0.0]),
                        breadcrumb: Vec::new(),
                        expanded: Vec::new(),
                    };
                    let child_y = y + header_h + 20.0 * z;
                    let child_x = x + 10.0 * z;
                    let mut child_list = RenderList::default();
                    sub.render_nodes_edges(&mut child_list, &child_view, 0.0);
                    for p in child_list.iter() {
                        list.push(p.offset(child_x, child_y));
                    }
                }
            }

            // Custom body rendering
            let inner = Rect::from_parts(
                Point::new(x + 4.0 * z, y + header_h + 2.0 * z),
                V([w - 8.0 * z, h - header_h - 6.0 * z]),
            );
            n.render_body(list, inner);
        }
    }
}

impl Renderable<()> for VisualGraph<2> {
    fn render(&self, list: &mut RenderList, _viewport: &()) {
        let view = GraphView::default();
        self.render_nodes_edges(list, &view, 0.0);
    }
}

