use crate::compute::Device;
use crate::kernel::{BinaryOp, ReduceOp};
use std::collections::HashMap;

use super::{EvalProgress, Graph, NodeId};

// ═══════════════════════════════════════════════════════════════════════════
// ── Lazy<'g> — ergonomic builder that records into a Graph ──────────────
// ═══════════════════════════════════════════════════════════════════════════

/// A lazy value that records operations into a [`Graph`].
///
/// Every arithmetic method returns a new `Lazy` without computing anything.
/// Call [`Lazy::eval`] to materialize the result on a [`Device`].
///
/// ```
/// use any_compute_core::graph::{Graph, Lazy};
/// use any_compute_core::compute::Device;
/// use std::collections::HashMap;
///
/// let mut g = Graph::new();
/// let x = Lazy::constant(&mut g, vec![1.0, 4.0, 9.0]);
/// let y = x.sqrt().scale(2.0);
/// let result = y.eval(&Device::cpu(), &HashMap::new());
/// assert_eq!(result, vec![2.0, 4.0, 6.0]);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Lazy<'g> {
    pub graph: &'g Graph,
    pub id: NodeId,
}

/// Mutable lazy builder — owns a mutable reference to the graph.
///
/// This is the primary way to build computation graphs ergonomically.
/// Each method records an operation and returns a new `LazyMut`.
pub struct LazyMut<'g> {
    graph: &'g mut Graph,
    id: NodeId,
}

impl<'g> LazyMut<'g> {
    /// Construct from an existing graph and node ID (used by `Buffer::lazy()`).
    pub fn from_raw(graph: &'g mut Graph, id: NodeId) -> Self {
        Self { graph, id }
    }

    /// Create a lazy value from constant data.
    pub fn constant(graph: &'g mut Graph, data: Vec<f64>) -> Self {
        let id = graph.constant(data);
        Self { graph, id }
    }

    /// Create a lazy value from a named input placeholder.
    pub fn input(graph: &'g mut Graph, name: &str) -> Self {
        let id = graph.input(name);
        Self { graph, id }
    }

    /// The node ID in the graph.
    pub fn node_id(&self) -> NodeId {
        self.id
    }

    /// Get the graph reference.
    pub fn graph(&self) -> &Graph {
        self.graph
    }

    /// Evaluate this node on a device.
    pub fn eval(&self, device: &Device, inputs: &HashMap<String, Vec<f64>>) -> Vec<f64> {
        self.graph.eval(self.id, device, inputs)
    }

    /// Evaluate with a progress callback (see [`Graph::eval_with`]).
    pub fn eval_with(
        &self,
        device: &Device,
        inputs: &HashMap<String, Vec<f64>>,
        on_progress: impl FnMut(EvalProgress<'_>),
    ) -> Vec<f64> {
        self.graph
            .eval_with(self.id, device, inputs, Some(on_progress))
    }

    /// Return a textual execution plan (see [`Graph::plan`]).
    pub fn plan(&self) -> String {
        self.graph.plan(self.id)
    }

    pub fn scale(self, s: f64) -> Self {
        let id = self.graph.scale(self.id, s);
        Self {
            graph: self.graph,
            id,
        }
    }
    pub fn offset(self, o: f64) -> Self {
        let id = self.graph.offset(self.id, o);
        Self {
            graph: self.graph,
            id,
        }
    }

    // ── Reduce ops ───────────────────────────────────────────────────────

    pub fn reduce_min(self) -> Self {
        let id = self.graph.min(self.id);
        Self {
            graph: self.graph,
            id,
        }
    }
    pub fn reduce_max(self) -> Self {
        let id = self.graph.max(self.id);
        Self {
            graph: self.graph,
            id,
        }
    }

    // ── Scan ─────────────────────────────────────────────────────────────

    pub fn prefix_sum(self) -> Self {
        let id = self.graph.scan(self.id, ReduceOp::Sum);
        Self {
            graph: self.graph,
            id,
        }
    }

    // ── Slice/reshape ────────────────────────────────────────────────────

    pub fn slice(self, start: usize, end: usize) -> Self {
        let id = self.graph.slice(self.id, start, end);
        Self {
            graph: self.graph,
            id,
        }
    }

    pub fn reshape(self, shape: Vec<usize>) -> Self {
        let id = self.graph.reshape(self.id, shape);
        Self {
            graph: self.graph,
            id,
        }
    }

    // ── Binary ops (take a second NodeId) ────────────────────────────────

    /// Combine this lazy value with another node via a binary op.
    pub fn combine(self, other: NodeId, op: BinaryOp) -> Self {
        let id = self.graph.binary(self.id, other, op);
        Self {
            graph: self.graph,
            id,
        }
    }

    pub fn add_node(self, other: NodeId) -> Self {
        self.combine(other, BinaryOp::Add)
    }
    pub fn sub_node(self, other: NodeId) -> Self {
        self.combine(other, BinaryOp::Sub)
    }
    pub fn mul_node(self, other: NodeId) -> Self {
        self.combine(other, BinaryOp::Mul)
    }
    pub fn div_node(self, other: NodeId) -> Self {
        self.combine(other, BinaryOp::Div)
    }

    // ── Gemm ─────────────────────────────────────────────────────────────

    pub fn gemm(self, other: NodeId, m: usize, n: usize, k: usize) -> Self {
        let id = self.graph.gemm(self.id, other, m, n, k);
        Self {
            graph: self.graph,
            id,
        }
    }
}

// ── Macro-generated LazyMut forwarding ───────────────────────────────────

for_each_unary!(lazy_forward);
for_each_reduce!(lazy_forward);

// ═══════════════════════════════════════════════════════════════════════════
// ── Lazy builder via Graph methods ──────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

impl<'g> Lazy<'g> {
    pub fn constant(graph: &'g mut Graph, data: Vec<f64>) -> LazyMut<'g> {
        LazyMut::constant(graph, data)
    }

    pub fn input(graph: &'g mut Graph, name: &str) -> LazyMut<'g> {
        LazyMut::input(graph, name)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ── From conversions for Graph building ─────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

impl Graph {
    /// Create a lazy constant from a Buffer.
    pub fn from_buffer(&mut self, buf: &crate::buffer::Buffer) -> NodeId {
        self.constant(buf.data().to_vec())
    }

    /// Create a lazy constant from a V<N>.
    pub fn from_vector<const N: usize>(&mut self, v: &crate::layout::V<N>) -> NodeId {
        self.constant(v.0.to_vec())
    }

    /// Create a lazy constant from a Matrix<R,C>.
    pub fn from_matrix<const R: usize, const C: usize>(
        &mut self,
        m: &crate::layout::Matrix<R, C>,
    ) -> NodeId {
        let data: Vec<f64> = m.data.iter().flat_map(|row| row.iter().copied()).collect();
        self.constant(data)
    }

    /// Evaluate and convert the result back to a Buffer.
    pub fn eval_to_buffer(
        &self,
        output: NodeId,
        device: &Device,
        inputs: &HashMap<String, Vec<f64>>,
    ) -> crate::buffer::Buffer {
        let data = self.eval(output, device, inputs);
        crate::buffer::Buffer::on(device, data)
    }
}

// ── lazy() bridge methods on core types ──────────────────────────────────

impl<const N: usize> crate::layout::V<N> {
    /// Enter lazy mode: record operations in a [`Graph`] instead of executing.
    pub fn lazy<'g>(&self, graph: &'g mut Graph) -> LazyMut<'g> {
        let id = graph.from_vector(self);
        LazyMut::from_raw(graph, id)
    }
}

impl<const R: usize, const C: usize> crate::layout::Matrix<R, C> {
    /// Enter lazy mode: record operations in a [`Graph`] instead of executing.
    pub fn lazy<'g>(&self, graph: &'g mut Graph) -> LazyMut<'g> {
        let id = graph.from_matrix(self);
        LazyMut::from_raw(graph, id)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ── ONNX protobuf encoder (hand-rolled, no codegen dependency) ──────────
// ═══════════════════════════════════════════════════════════════════════════
