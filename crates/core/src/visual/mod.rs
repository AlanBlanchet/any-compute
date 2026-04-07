//! Visual graph — render any DAG as interactive nodes + edges.
//!
//! ## Design
//!
//! `VisualGraph<D>` is a dimension-generic (`D=2` or `D=3`) container of
//! positioned nodes and edges.  Every node can contain a child graph,
//! enabling **infinite hierarchical zoom** (a ResNet block expands into
//! its conv → bn → relu chain, each of those can expand further, etc.).
//!
//! Any object can become a node via the [`GraphNode`] trait → implement
//! one method, get positioning, rendering, ports, and hierarchy for free.
//!
//! ## Interaction
//!
//! `GraphView` holds zoom/pan/breadcrumb state.  Children are collapsed
//! by default; clicking a node with children "enters" it (pushes onto
//! the breadcrumb trail).  Back-navigation pops the trail.
//!
//! ## Trait cascade
//!
//! ```text
//! Graphable ──► to_graph() → VisualGraph<2>
//!              ──► render_graph() → RenderList  (blanket)
//!              ──► capture_graph(w, h) → PixelBuffer  (blanket)
//! ```
//!
//! Implement `Graphable` on any type to make it visualizable as a graph.

mod graph;
mod node;
mod style;
mod view;

pub use graph::*;
pub use node::*;
pub use style::*;
pub use view::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::{Point, V};
    use crate::render::{Color, PixelBuffer, Primitive, RenderList, Renderable};

    /// Build a standard 3-node DAG for reuse across tests.
    fn three_node_dag() -> VisualGraph<2> {
        let mut g = VisualGraph::<2>::new("test");
        let a = g.add(VNode::new(V([0.0, 0.0]), "Input", "input").outs(&["x"]));
        let b = g.add(
            VNode::new(V([200.0, 0.0]), "ReLU", "relu")
                .ins(&["x"])
                .outs(&["y"]),
        );
        let c = g.add(VNode::new(V([400.0, 0.0]), "Output", "output").ins(&["y"]));
        g.edge(a, b);
        g.edge(b, c);
        g
    }

    /// Build a hierarchical graph for sub-graph tests.
    fn nested_dag() -> VisualGraph<2> {
        let mut sub = VisualGraph::<2>::new("block");
        let s0 = sub.add(VNode::new(V([0.0, 0.0]), "Conv", "conv"));
        let s1 = sub.add(VNode::new(V([180.0, 0.0]), "BN", "bn"));
        sub.edge(s0, s1);

        let mut g = VisualGraph::<2>::new("net");
        g.add(
            VNode::new(V([0.0, 0.0]), "Block", "residual")
                .size(140.0, 60.0)
                .children(sub),
        );
        g.add(VNode::new(V([200.0, 0.0]), "FC", "linear"));
        g.edge(0, 1);
        g
    }

    /// Helper: extract all text labels from a render list.
    fn texts(list: &RenderList) -> Vec<String> {
        list.iter()
            .filter_map(|p| {
                if let Primitive::Text { content, .. } = p {
                    Some(content.clone())
                } else {
                    None
                }
            })
            .collect()
    }

    /// Pixel + structural test: renders a 3-node graph, verifies pixel colors
    /// for each header (different tag → different color), edge lines bridge
    /// nodes, and ports exist at correct positions.
    #[test]
    fn full_dag_pixel_structure_and_edges() {
        let g = three_node_dag();
        let mut list = RenderList::default();
        g.render(&mut list, &());

        // Should produce ≥20 primitives (3 nodes × ~6 + edge lines)
        assert!(list.len() >= 20, "primitives: {}", list.len());

        // Edges produce line segments bridging right-of-A to left-of-B
        // Filter out background grid lines (alpha 12)
        let lines: Vec<_> = list
            .iter()
            .filter_map(|p| match p {
                Primitive::Line {
                    from,
                    to,
                    width,
                    stroke,
                    ..
                } if stroke.a > 12 => Some((*from, *to, *width)),
                _ => None,
            })
            .collect();
        assert!(
            lines.len() >= 4,
            "2 edges × ≥2 segments, got {}",
            lines.len()
        );
        // First edge starts near x≈120 (right side of Input node)
        assert!(lines[0].0.x > 100.0 && lines[0].0.x < 140.0);

        // Pixel test: headers have distinct colors per tag
        let mut pb = PixelBuffer::new(600, 100, Color::rgb(0, 0, 0));
        pb.paint(&list);
        let input_c = pb.pixel(60, 11);
        let relu_c = pb.pixel(260, 11);
        let output_c = pb.pixel(460, 11);
        assert_ne!(input_c, relu_c);
        assert_ne!(relu_c, output_c);
        assert!(input_c.g > input_c.r, "input should be green-dominant");
    }

    /// Diamond DAG: A→B, A→C, B→D, C→D. Verifies auto_layout layer assignment
    /// AND that zoom_at preserves the point under cursor.
    #[test]
    fn diamond_layout_and_zoom_at_invariance() {
        let mut g = VisualGraph::<2>::new("diamond");
        let a = g.add(VNode::<2>::from("A"));
        let b = g.add(VNode::<2>::from("B"));
        let c = g.add(VNode::<2>::from("C"));
        let d = g.add(VNode::<2>::from("D"));
        g.edge(a, b);
        g.edge(a, c);
        g.edge(b, d);
        g.edge(c, d);
        g.auto_layout();

        // Layer assignment: A leftmost, D rightmost, B&C same layer
        let ax = g.node_pos(a).0[0];
        let bx = g.node_pos(b).0[0];
        let cx = g.node_pos(c).0[0];
        let dx = g.node_pos(d).0[0];
        assert!(ax < bx && ax < cx, "A must be leftmost");
        assert!(dx > bx && dx > cx, "D must be rightmost");
        assert!((bx - cx).abs() < 1.0, "B and C same layer");
        assert!(
            (g.node_pos(b).0[1] - g.node_pos(c).0[1]).abs() > 10.0,
            "B&C different y"
        );

        // zoom_at preserves anchor: 10 cumulative zooms, the cursor point
        // should always map back to the same graph coordinate
        let cursor = Point::new(200.0, 100.0);
        let mut view = GraphView::default();
        let gx_before = cursor.x / view.zoom - view.pan.0[0];
        let gy_before = cursor.y / view.zoom - view.pan.0[1];
        for _ in 0..10 {
            view.zoom_at(cursor, 1.15, 0.1, 10.0);
        }
        let gx_after = cursor.x / view.zoom - view.pan.0[0];
        let gy_after = cursor.y / view.zoom - view.pan.0[1];
        assert!(
            (gx_before - gx_after).abs() < 0.01,
            "x drift: {}",
            gx_before - gx_after
        );
        assert!(
            (gy_before - gy_after).abs() < 0.01,
            "y drift: {}",
            gy_before - gy_after
        );

        // zoom_at + unzoom returns to ~original state
        for _ in 0..10 {
            view.zoom_at(cursor, 1.0 / 1.15, 0.1, 10.0);
        }
        assert!(
            (view.zoom - 1.0).abs() < 0.01,
            "zoom should return to ~1, got {}",
            view.zoom
        );
    }

    /// Hit-test across zoom levels + pan, verifying correct node identification.
    #[test]
    fn hit_test_with_zoom_pan_and_miss() {
        let g = three_node_dag();

        // 1x zoom: direct hit
        let v1 = GraphView::default();
        assert_eq!(
            v1.hit_test(&g, Point::new(60.0, 25.0)),
            Some(0),
            "hit Input at 1x"
        );
        assert_eq!(
            v1.hit_test(&g, Point::new(170.0, 25.0)),
            None,
            "miss between nodes"
        );

        // 2x zoom: Input at (0,0)->(240,100), ReLU at (400,0)->(640,100)
        let v2 = GraphView {
            zoom: 2.0,
            ..GraphView::default()
        };
        assert_eq!(
            v2.hit_test(&g, Point::new(120.0, 50.0)),
            Some(0),
            "hit Input at 2x"
        );
        assert_eq!(
            v2.hit_test(&g, Point::new(500.0, 50.0)),
            Some(1),
            "hit ReLU at 2x"
        );
        assert_eq!(v2.hit_test(&g, Point::new(300.0, 50.0)), None, "miss at 2x");

        // Pan: shift everything right by 100 graph-coords → Input now at screen (100, 0)
        let v3 = GraphView {
            pan: V([100.0, 0.0]),
            ..GraphView::default()
        };
        assert_eq!(
            v3.hit_test(&g, Point::new(160.0, 25.0)),
            Some(0),
            "hit Input with pan"
        );
        assert_eq!(
            v3.hit_test(&g, Point::new(50.0, 25.0)),
            None,
            "miss before panned Input"
        );
    }

    /// Full breadcrumb lifecycle: enter → render sub-graph → back → render root,
    /// verifying both navigation state AND rendered content labels.
    #[test]
    fn breadcrumb_enter_render_and_back() {
        let g = nested_dag();
        let mut view = GraphView::default();

        // Root: should show "Block" and "FC", not "Conv"
        let mut root_list = RenderList::default();
        g.render(&mut root_list, &view);
        let root_t = texts(&root_list);
        assert!(root_t.iter().any(|t| t == "Block"), "root has Block");
        assert!(root_t.iter().any(|t| t == "FC"), "root has FC");
        assert!(!root_t.iter().any(|t| t == "Conv"), "root hides Conv");

        // Enter Block → sub-graph with Conv, BN
        view.enter(0);
        assert_eq!(view.depth(), 1);
        let sub = g.resolve_breadcrumb_pub(&view.breadcrumb);
        assert_eq!(sub.len(), 2);
        assert_eq!(sub.node(0).label(), "Conv");

        // Rendered breadcrumb shows "net" and "Block" labels
        let mut sub_list = RenderList::default();
        g.render(&mut sub_list, &view);
        let sub_t = texts(&sub_list);
        assert!(
            sub_t.iter().any(|t| t.contains("net")),
            "breadcrumb root label"
        );
        assert!(sub_t.iter().any(|t| t.contains("Block")), "breadcrumb path");
        assert!(sub_t.iter().any(|t| t == "Conv"), "sub-graph shows Conv");

        // Back → root
        assert_eq!(view.back(), Some(0));
        assert_eq!(view.depth(), 0);
        let root2 = g.resolve_breadcrumb_pub(&view.breadcrumb);
        assert_eq!(root2.len(), 2);
    }

    /// Expand/collapse: verify toggle toggles, collapsed hides children,
    /// expanded shows children inline with more primitives.
    #[test]
    fn expand_collapse_renders_children_inline() {
        let g = nested_dag();
        let mut view = GraphView::default();

        let mut collapsed = RenderList::default();
        g.render(&mut collapsed, &view);
        let collapsed_t = texts(&collapsed);
        assert!(
            !collapsed_t.iter().any(|t| t == "Conv"),
            "collapsed hides Conv"
        );

        view.toggle(0);
        assert!(view.is_expanded(0));
        let mut expanded = RenderList::default();
        g.render(&mut expanded, &view);
        let expanded_t = texts(&expanded);
        assert!(
            expanded_t.iter().any(|t| t == "Conv"),
            "expanded shows Conv"
        );
        assert!(
            expanded.len() > collapsed.len(),
            "expanded has more primitives"
        );

        view.toggle(0);
        assert!(!view.is_expanded(0));
    }

    /// Graphable trait: blanket render_graph + capture_graph on a custom type.
    /// Verifies the full cascade: to_graph → render → pixels.
    #[test]
    fn graphable_blanket_full_cascade() {
        struct TestGraphable;
        impl Graphable for TestGraphable {
            fn to_graph(&self) -> VisualGraph<2> {
                three_node_dag()
            }
        }

        let d = TestGraphable;
        let list = d.render_graph();
        assert!(list.len() > 10);

        let pb = d.capture_graph(600, 100);
        assert_eq!(pb.width, 600);
        // Headers should be visible (not all background)
        let bg = Color::rgb(25, 28, 36);
        let non_bg = (0..600).filter(|&x| pb.pixel(x, 11) != bg).count();
        assert!(non_bg > 30, "capture should have visible content");
    }

    /// Tag colors: pairwise comparison across all 16 known tags.
    /// Same color only allowed for semantically related tags.
    #[test]
    fn tag_colors_pairwise_distinct() {
        let tags = [
            "input",
            "constant",
            "output",
            "relu",
            "sigmoid",
            "tanh",
            "conv",
            "linear",
            "gemm",
            "batchnorm",
            "add",
            "reduce",
            "reshape",
            "residual",
            "dropout",
            "unknown",
        ];
        let colors: Vec<Color> = tags.iter().map(|t| tag_color(t)).collect();
        for i in 0..tags.len() {
            for j in (i + 1)..tags.len() {
                if colors[i] == colors[j] {
                    let same_cat = matches!(
                        (tags[i], tags[j]),
                        ("add", "sub")
                            | ("add", "mul")
                            | ("add", "div")
                            | ("sub", "mul")
                            | ("sub", "div")
                            | ("mul", "div")
                            | ("relu", "sigmoid")
                            | ("relu", "tanh")
                            | ("sigmoid", "tanh")
                            | ("batchnorm", _)
                            | (_, "batchnorm")
                            | ("reshape", "concat")
                            | ("reshape", "slice")
                            | ("reshape", "gather")
                            | ("concat", "slice")
                            | ("concat", "gather")
                            | ("slice", "gather")
                            | ("constant", "const")
                            | ("input", "placeholder")
                            | ("linear", "gemm")
                            | ("linear", "matmul")
                            | ("gemm", "matmul")
                    );
                    assert!(
                        same_cat,
                        "'{}' and '{}' share color but aren't related",
                        tags[i], tags[j]
                    );
                }
            }
        }
    }
}
