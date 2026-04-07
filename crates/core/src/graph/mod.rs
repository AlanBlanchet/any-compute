//! Computational graph — lazy evaluation, automatic fusion, and graph export.\n
/// Generates `fn name(&mut self, input: NodeId) -> NodeId` wrappers for Graph.
macro_rules! graph_unary {
    ($($method:ident => $op:expr),* $(,)?) => {
        impl Graph {
            $(pub fn $method(&mut self, input: NodeId) -> NodeId { self.unary(input, $op) })*
        }
    }
}

macro_rules! graph_binary {
    ($($method:ident => $op:expr),* $(,)?) => {
        impl Graph {
            $(pub fn $method(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId { self.binary(lhs, rhs, $op) })*
        }
    }
}

macro_rules! graph_reduce {
    ($($method:ident => $op:expr),* $(,)?) => {
        impl Graph {
            $(pub fn $method(&mut self, input: NodeId) -> NodeId { self.reduce(input, $op) })*
        }
    }
}

/// Generates `fn name(self) -> Self` forwarding wrappers for LazyMut.
macro_rules! lazy_forward {
    ($($method:ident $(=> $op:expr)?),* $(,)?) => {
        impl<'g> LazyMut<'g> {
            $(pub fn $method(self) -> Self {
                let id = self.graph.$method(self.id);
                Self { graph: self.graph, id }
            })*
        }
    }
}

mod graph_impl;
mod lazy;
mod node;
pub(crate) mod onnx;

pub use graph_impl::*;
pub use lazy::*;
pub use node::*;

#[cfg(test)]
mod tests {
    use super::*;
    
    use crate::compute::Device;
    use crate::kernel::ReduceOp;
    
    
    use std::collections::HashMap;

    fn dev() -> Device {
        Device::cpu()
    }

    fn empty() -> HashMap<String, Vec<f64>> {
        HashMap::new()
    }

    // ── Graph building + eval ────────────────────────────────────────────

    #[test]
    fn constant_passthrough() {
        let mut g = Graph::new();
        let c = g.constant(vec![1.0, 2.0, 3.0]);
        let result = g.eval(c, &dev(), &empty());
        assert_eq!(result, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn unary_chain() {
        let mut g = Graph::new();
        let x = g.constant(vec![1.0, 4.0, 9.0]);
        let y = g.sqrt(x);
        let z = g.scale(y, 2.0);
        let result = g.eval(z, &dev(), &empty());
        assert_eq!(result, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn binary_ops() {
        let mut g = Graph::new();
        let a = g.constant(vec![1.0, 2.0, 3.0]);
        let b = g.constant(vec![4.0, 5.0, 6.0]);
        let c = g.add(a, b);
        let result = g.eval(c, &dev(), &empty());
        assert_eq!(result, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn reduce_sum() {
        let mut g = Graph::new();
        let x = g.constant(vec![1.0, 2.0, 3.0, 4.0]);
        let s = g.sum(x);
        let result = g.eval(s, &dev(), &empty());
        assert_eq!(result, vec![10.0]);
    }

    #[test]
    fn input_binding() {
        let mut g = Graph::new();
        let x = g.input("x");
        let y = g.scale(x, 3.0);
        let mut inputs = HashMap::new();
        inputs.insert("x".to_string(), vec![1.0, 2.0, 3.0]);
        let result = g.eval(y, &dev(), &inputs);
        assert_eq!(result, vec![3.0, 6.0, 9.0]);
    }

    #[test]
    fn gemm_through_graph() {
        let mut g = Graph::new();
        let a = g.constant(vec![1.0, 0.0, 0.0, 1.0]); // 2x2 identity
        let b = g.constant(vec![5.0, 6.0, 7.0, 8.0]);
        let c = g.gemm(a, b, 2, 2, 2);
        let result = g.eval(c, &dev(), &empty());
        assert_eq!(result, vec![5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn scan_prefix_sum() {
        let mut g = Graph::new();
        let x = g.constant(vec![1.0, 2.0, 3.0]);
        let s = g.scan(x, ReduceOp::Sum);
        let result = g.eval(s, &dev(), &empty());
        assert_eq!(result, vec![1.0, 3.0, 6.0]);
    }

    #[test]
    fn concat_slices() {
        let mut g = Graph::new();
        let a = g.constant(vec![1.0, 2.0]);
        let b = g.constant(vec![3.0, 4.0]);
        let c = g.concat(vec![a, b]);
        let result = g.eval(c, &dev(), &empty());
        assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0]);

        // Slice back
        let s = g.slice(c, 1, 3);
        let result2 = g.eval(s, &dev(), &empty());
        assert_eq!(result2, vec![2.0, 3.0]);
    }

    #[test]
    fn diamond_dag() {
        // x -> sqrt -> (a, b) -> a + b
        let mut g = Graph::new();
        let x = g.constant(vec![4.0, 9.0, 16.0]);
        let sq = g.sqrt(x);
        let doubled = g.add(sq, sq); // same node used twice
        let result = g.eval(doubled, &dev(), &empty());
        assert_eq!(result, vec![4.0, 6.0, 8.0]);
    }

    // ── Lazy builder ─────────────────────────────────────────────────────

    #[test]
    fn lazy_chain() {
        let mut g = Graph::new();
        let result = Lazy::constant(&mut g, vec![1.0, 4.0, 9.0])
            .sqrt()
            .scale(2.0)
            .eval(&dev(), &empty());
        assert_eq!(result, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn lazy_reduce() {
        let mut g = Graph::new();
        let result = Lazy::constant(&mut g, vec![1.0, 2.0, 3.0, 4.0])
            .sum()
            .eval(&dev(), &empty());
        assert_eq!(result, vec![10.0]);
    }

    // ── Fusion detection ─────────────────────────────────────────────────

    #[test]
    fn fusion_groups_detected() {
        let mut g = Graph::new();
        let x = g.constant(vec![1.0, 2.0]);
        let a = g.sqrt(x); // element-wise
        let b = g.abs(a); // element-wise, single consumer of a
        let c = g.neg(b); // element-wise, single consumer of b
        let _d = g.sum(c); // reduce — breaks the chain

        let groups = g.fusion_groups();
        // Should find one fusable chain: [sqrt, abs, neg]
        assert!(!groups.is_empty());
        let chain = &groups[0];
        assert!(chain.len() >= 2); // at least sqrt+abs
    }

    // ── DOT export ───────────────────────────────────────────────────────

    #[test]
    fn dot_export() {
        let mut g = Graph::new();
        let x = g.constant(vec![1.0, 2.0]);
        let y = g.sqrt(x);
        let _z = g.sum(y);
        let dot = g.to_dot();
        assert!(dot.contains("digraph"));
        assert!(dot.contains("Sqrt"));
    }

    // ── ONNX export ─────────────────────────────────────────────────────

    #[test]
    fn onnx_produces_bytes() {
        let mut g = Graph::new();
        let x = g.input("x");
        let y = g.sqrt(x);
        let z = g.scale(y, 2.0);
        g.name(z, "output");
        let bytes = g.to_onnx("test_model");
        // Basic sanity: non-empty, starts with valid protobuf field tag
        assert!(bytes.len() > 20);
    }

    #[test]
    fn onnx_roundtrip_structure() {
        let mut g = Graph::new();
        let a = g.input("a");
        let b = g.input("b");
        let c = g.add(a, b);
        let d = g.relu(c);
        g.name(d, "result");
        let bytes = g.to_onnx("add_relu");
        // Should contain "Add" and "Relu" op type strings
        let text = String::from_utf8_lossy(&bytes);
        assert!(text.contains("Add"));
        assert!(text.contains("Relu"));
        assert!(text.contains("any-compute"));
    }

    // ── Graph from Buffer / V / Matrix ──────────────────────────────────

    #[test]
    fn graph_from_buffer() {
        let buf = crate::buffer::Buffer::new(vec![2.0, 3.0, 4.0]);
        let mut g = Graph::new();
        let x = g.from_buffer(&buf);
        let y = g.sqrt(x);
        let result = g.eval_to_buffer(y, &dev(), &empty());
        assert!((result.data()[0] - 2.0f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn graph_from_vector() {
        let v = crate::layout::V3::new3(9.0, 16.0, 25.0);
        let mut g = Graph::new();
        let x = g.from_vector(&v);
        let y = g.sqrt(x);
        let result = g.eval(y, &dev(), &empty());
        assert!((result[0] - 3.0).abs() < 1e-10);
        assert!((result[1] - 4.0).abs() < 1e-10);
        assert!((result[2] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn graph_node_count() {
        let mut g = Graph::new();
        assert!(g.is_empty());
        let x = g.constant(vec![1.0]);
        let _y = g.sqrt(x);
        assert_eq!(g.len(), 2);
    }

    // ── Lazy bridge from Buffer / V / Matrix ────────────────────────────

    #[test]
    fn buffer_lazy_bridge() {
        let buf = crate::buffer::Buffer::new(vec![1.0, 4.0, 9.0]);
        let mut g = Graph::new();
        let result = buf.lazy(&mut g).sqrt().scale(2.0).eval(&dev(), &empty());
        assert_eq!(result, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn vector_lazy_bridge() {
        let v = crate::layout::V3::new3(4.0, 16.0, 64.0);
        let mut g = Graph::new();
        let result = v.lazy(&mut g).sqrt().eval(&dev(), &empty());
        assert!((result[0] - 2.0).abs() < 1e-10);
        assert!((result[1] - 4.0).abs() < 1e-10);
        assert!((result[2] - 8.0).abs() < 1e-10);
    }

    #[test]
    fn matrix_lazy_bridge() {
        let m = crate::layout::Mat2 {
            data: [[4.0, 9.0], [16.0, 25.0]],
        };
        let mut g = Graph::new();
        let result = m.lazy(&mut g).sqrt().eval(&dev(), &empty());
        assert_eq!(result, vec![2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn lazy_binary_ops() {
        let mut g = Graph::new();
        let b = g.constant(vec![10.0, 20.0, 30.0]);
        let result = LazyMut::constant(&mut g, vec![1.0, 2.0, 3.0])
            .add_node(b)
            .eval(&dev(), &empty());
        assert_eq!(result, vec![11.0, 22.0, 33.0]);
    }

    #[test]
    fn lazy_new_unary_ops() {
        let mut g = Graph::new();
        let result = LazyMut::constant(&mut g, vec![1.7, 2.3, -0.5])
            .floor()
            .eval(&dev(), &empty());
        assert_eq!(result, vec![1.0, 2.0, -1.0]);
    }

    #[test]
    fn buffer_new_unary_ops() {
        let buf = crate::buffer::Buffer::new(vec![1.7, 2.3, -0.5]);
        assert_eq!(buf.floor().data(), &[1.0, 2.0, -1.0]);
        assert_eq!(buf.ceil().data(), &[2.0, 3.0, 0.0]);
        let buf2 = crate::buffer::Buffer::new(vec![4.0, 0.25]);
        assert!((buf2.rsqrt().data()[0] - 0.5).abs() < 1e-10);
        assert!((buf2.rsqrt().data()[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn eval_with_progress() {
        let mut g = Graph::new();
        let x = g.constant(vec![1.0, 4.0, 9.0]);
        let s = g.sqrt(x);
        let y = g.scale(s, 2.0);

        let mut steps = Vec::new();
        g.eval_with(
            y,
            &dev(),
            &empty(),
            Some(|p: EvalProgress| {
                steps.push((p.step, p.total, p.op_kind.clone(), p.input_size));
            }),
        );
        // sqrt+scale are fused → only 2 callbacks: constant + fused chain
        assert_eq!(steps.len(), 2);
        assert!(steps[0].2.starts_with("constant"));
        assert!(steps[1].2.starts_with("fused:"));
        assert!(steps[1].2.contains("2×unary")); // 2 ops fused
    }

    #[test]
    fn plan_output() {
        let mut g = Graph::new();
        let x = g.constant(vec![1.0, 2.0]);
        let y = g.sqrt(x);
        let _z = g.abs(y);
        let plan = g.plan(_z);
        assert!(plan.contains("3 steps"));
        assert!(plan.contains("unary:Sqrt"));
        assert!(plan.contains("unary:Abs"));
        // Sqrt→Abs should be a fusable group
        assert!(plan.contains("Fusable groups: 1"));
    }

    #[test]
    fn fusion_produces_correct_results() {
        let mut g = Graph::new();
        let x = g.constant(vec![4.0, 16.0, 25.0]);
        // sqrt → abs → neg: 3 unary ops fused into one pass
        let s = g.sqrt(x);
        let a = g.abs(s);
        let n = g.neg(a);

        let result = g.eval(n, &dev(), &empty());
        assert_eq!(result, vec![-2.0, -4.0, -5.0]);

        // Verify fusion actually happened
        let groups = g.fusion_groups();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].len(), 3);
    }

    #[test]
    fn to_visual_converts_graph() {
        let mut g = Graph::new();
        let x = g.input("x");
        let w = g.constant(vec![1.0; 4]);
        let y = g.gemm(x, w, 1, 1, 4);
        let out = g.relu(y);

        let vg = g.to_visual(out);
        assert_eq!(vg.len(), 4); // input, constant, gemm, relu
        assert!(!vg.edges.is_empty());

        // Should be renderable.
        let mut list = crate::render::RenderList::default();
        crate::render::Renderable::render(&vg, &mut list, &());
        assert!(list.len() > 5);
    }
}
