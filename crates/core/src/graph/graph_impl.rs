use crate::compute::Device;
use crate::kernel::{BinaryOp, ReduceOp, Scalar, UnaryOp};
use std::collections::HashMap;

use super::{EvalProgress, GraphOp, NodeId, onnx};

// ═══════════════════════════════════════════════════════════════════════════
// ── Graph — the DAG of operations ───────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

/// A node in the computational graph.
#[derive(Debug, Clone)]
pub(super) struct Node {
    pub(super) op: GraphOp,
    /// Human-readable label (optional).
    pub(super) name: Option<String>,
    /// How many downstream nodes reference this one (computed during eval).
    pub(super) ref_count: u32,
}

/// Directed acyclic graph of lazy operations.
///
/// Build a graph by adding nodes, then evaluate it on a Device, export to
/// ONNX, or visualize as DOT.
#[derive(Debug, Clone)]
pub struct Graph {
    pub(super) nodes: Vec<Node>,
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

impl Graph {
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    /// Number of nodes in the graph.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Add a node and return its handle.
    fn push(&mut self, op: GraphOp) -> NodeId {
        let id = NodeId(self.nodes.len() as u32);
        self.nodes.push(Node {
            op,
            name: None,
            ref_count: 0,
        });
        id
    }

    /// Name a node (for debugging / DOT / ONNX output names).
    pub fn name(&mut self, id: NodeId, label: &str) {
        self.nodes[id.0 as usize].name = Some(label.to_string());
    }

    // ── Leaf constructors ────────────────────────────────────────────────

    /// Add a constant data node.
    pub fn constant(&mut self, data: Vec<f64>) -> NodeId {
        self.push(GraphOp::Constant(data))
    }

    /// Add a named input placeholder (bound at eval time).
    pub fn input(&mut self, name: &str) -> NodeId {
        self.push(GraphOp::Input(name.to_string()))
    }

    // ── Operation constructors ───────────────────────────────────────────

    pub fn unary(&mut self, input: NodeId, op: UnaryOp) -> NodeId {
        self.push(GraphOp::Unary { input, op })
    }

    pub fn binary(&mut self, lhs: NodeId, rhs: NodeId, op: BinaryOp) -> NodeId {
        self.push(GraphOp::Binary { lhs, rhs, op })
    }

    pub fn reduce(&mut self, input: NodeId, op: ReduceOp) -> NodeId {
        self.push(GraphOp::Reduce { input, op })
    }

    pub fn scan(&mut self, input: NodeId, op: ReduceOp) -> NodeId {
        self.push(GraphOp::Scan { input, op })
    }

    pub fn gemm(&mut self, lhs: NodeId, rhs: NodeId, m: usize, n: usize, k: usize) -> NodeId {
        self.push(GraphOp::Gemm { lhs, rhs, m, n, k })
    }

    pub fn gather(&mut self, data: NodeId, indices: NodeId) -> NodeId {
        self.push(GraphOp::Gather { data, indices })
    }

    pub fn reshape(&mut self, input: NodeId, shape: Vec<usize>) -> NodeId {
        self.push(GraphOp::Reshape { input, shape })
    }

    pub fn concat(&mut self, nodes: Vec<NodeId>) -> NodeId {
        self.push(GraphOp::Concat(nodes))
    }

    pub fn slice(&mut self, input: NodeId, start: usize, end: usize) -> NodeId {
        self.push(GraphOp::Slice { input, start, end })
    }

    pub fn custom(&mut self, input: NodeId, name: &str) -> NodeId {
        self.push(GraphOp::Custom {
            input,
            name: name.to_string(),
        })
    }

    // ── Convenience chains (see macro invocations below) ──────────────────

    pub fn scale(&mut self, input: NodeId, s: f64) -> NodeId {
        self.unary(input, UnaryOp::Scale(Scalar::from(s)))
    }
    pub fn offset(&mut self, input: NodeId, o: f64) -> NodeId {
        self.unary(input, UnaryOp::Offset(Scalar::from(o)))
    }

    // ── Evaluation ───────────────────────────────────────────────────────

    /// Evaluate the graph, computing the value at `output`.
    ///
    /// `inputs` maps placeholder names to their data.
    /// Returns the computed data for the specified output node.
    pub fn eval(
        &self,
        output: NodeId,
        device: &Device,
        inputs: &HashMap<String, Vec<f64>>,
    ) -> Vec<f64> {
        self.eval_with(output, device, inputs, None::<fn(EvalProgress<'_>)>)
    }

    /// Evaluate with a progress callback.
    ///
    /// The callback fires before each node is computed, with information
    /// about which step/total, what operation, and input data sizes.
    ///
    /// ```ignore
    /// graph.eval_with(output, &dev, &inputs, Some(|p: EvalProgress| {
    ///     println!("[{}/{}] {} ({})", p.step, p.total, p.label, p.op_kind);
    /// }));
    /// ```
    pub fn eval_with(
        &self,
        output: NodeId,
        device: &Device,
        inputs: &HashMap<String, Vec<f64>>,
        mut on_progress: Option<impl FnMut(EvalProgress<'_>)>,
    ) -> Vec<f64> {
        // Topological order via post-order DFS.
        let order = self.topo_order(output);
        let total = order.len();
        let mut cache: HashMap<u32, Vec<f64>> = HashMap::new();

        // Pre-compute fusion chains: map chain-head input → (chain-tail id, [UnaryOp]).
        // Only pure-unary chains are fused (binary ops break chains).
        let fusions = self.fusion_groups();
        let mut fuse_head: HashMap<u32, (u32, Vec<UnaryOp>)> = HashMap::new();
        let mut fuse_skip: std::collections::HashSet<u32> = std::collections::HashSet::new();
        for group in &fusions {
            // Verify all nodes in chain are unary.
            let all_unary = group
                .iter()
                .all(|nid| matches!(self.nodes[nid.0 as usize].op, GraphOp::Unary { .. }));
            if !all_unary || group.len() < 2 {
                continue;
            }
            let ops: Vec<UnaryOp> = group
                .iter()
                .map(|nid| match &self.nodes[nid.0 as usize].op {
                    GraphOp::Unary { op, .. } => *op,
                    _ => unreachable!(),
                })
                .collect();
            let head = group[0].0;
            let tail = group[group.len() - 1].0;
            // The input to the chain head is what we read from cache.
            let head_input = match &self.nodes[head as usize].op {
                GraphOp::Unary { input, .. } => input.0,
                _ => unreachable!(),
            };
            fuse_head.insert(head, (tail, ops));
            // Mark all nodes except head as skippable (head drives the fused dispatch).
            for nid in &group[1..] {
                fuse_skip.insert(nid.0);
            }
            // But if tail == output, we must not skip it from the cache.
            // (handled by inserting under tail id below)
            let _ = head_input; // used implicitly via cache lookup
        }

        for (step_0, &id) in order.iter().enumerate() {
            // Skip nodes that are interior to a fused chain.
            if fuse_skip.contains(&id) {
                continue;
            }

            let node = &self.nodes[id as usize];

            // Fire progress callback before computing this node.
            if let Some(ref mut cb) = on_progress {
                let label_owned;
                let label = match &node.name {
                    Some(n) => n.as_str(),
                    None => {
                        label_owned = format!("n{id}");
                        &label_owned
                    }
                };
                let first_input_size = node
                    .op
                    .inputs()
                    .first()
                    .and_then(|nid| cache.get(&nid.0))
                    .map_or(0, |v| v.len());
                let op_kind = if fuse_head.contains_key(&id) {
                    let (_, ops) = &fuse_head[&id];
                    format!("fused:{}×unary", ops.len())
                } else {
                    node.op.kind_str()
                };
                cb(EvalProgress {
                    step: step_0 + 1,
                    total,
                    label,
                    op_kind,
                    input_size: first_input_size,
                });
            }

            // Check if this node is the head of a fused chain.
            if let Some((tail, ops)) = fuse_head.get(&id) {
                let input_id = match &node.op {
                    GraphOp::Unary { input, .. } => input.0,
                    _ => unreachable!(),
                };
                let a = &cache[&input_id];
                let result = device.fused_unary(a, ops);
                // Store under the tail node id so downstream refs find it.
                cache.insert(*tail, result);
                continue;
            }

            let result = match &node.op {
                GraphOp::Constant(data) => data.clone(),
                GraphOp::Input(name) => inputs
                    .get(name)
                    .unwrap_or_else(|| panic!("unbound input: {name}"))
                    .clone(),
                GraphOp::Unary { input, op } => {
                    let a = &cache[&input.0];
                    device.unary(a, *op)
                }
                GraphOp::Binary { lhs, rhs, op } => {
                    let a = &cache[&lhs.0];
                    let b = &cache[&rhs.0];
                    device.binary(a, b, *op)
                }
                GraphOp::Reduce { input, op } => {
                    let a = &cache[&input.0];
                    vec![device.reduce(a, *op)]
                }
                GraphOp::Scan { input, op } => {
                    let a = &cache[&input.0];
                    device.scan(a, *op)
                }
                GraphOp::Gemm { lhs, rhs, m, n, k } => {
                    let a = &cache[&lhs.0];
                    let b = &cache[&rhs.0];
                    device.gemm(a, b, *m, *n, *k)
                }
                GraphOp::Gather { data, indices } => {
                    let d = &cache[&data.0];
                    let idx: Vec<usize> = cache[&indices.0].iter().map(|v| *v as usize).collect();
                    device.gather(d, &idx)
                }
                GraphOp::Reshape { input, .. } => {
                    // Reshape is metadata-only — data doesn't change.
                    cache[&input.0].clone()
                }
                GraphOp::Concat(nodes) => {
                    let mut out = Vec::new();
                    for nid in nodes {
                        out.extend_from_slice(&cache[&nid.0]);
                    }
                    out
                }
                GraphOp::Slice { input, start, end } => {
                    let a = &cache[&input.0];
                    a[*start..*end].to_vec()
                }
                GraphOp::Custom { input, .. } => {
                    // Custom ops pass through — they're opaque.
                    cache[&input.0].clone()
                }
            };
            cache.insert(id, result);
        }

        cache.remove(&output.0).unwrap_or_default()
    }

    /// Return a textual execution plan for evaluating `output`.
    ///
    /// Shows the topological order, fusable groups, and each step's operation.
    pub fn plan(&self, output: NodeId) -> String {
        let order = self.topo_order(output);
        let fusions = self.fusion_groups();
        let fused_set: HashMap<u32, usize> = fusions
            .iter()
            .enumerate()
            .flat_map(|(gi, group)| group.iter().map(move |nid| (nid.0, gi)))
            .collect();

        let mut lines = Vec::new();
        lines.push(format!("Execution plan: {} steps", order.len()));
        if !fusions.is_empty() {
            lines.push(format!("Fusable groups: {}", fusions.len()));
            for (i, g) in fusions.iter().enumerate() {
                let names: Vec<_> = g.iter().map(|nid| nid.label(self)).collect();
                lines.push(format!("  group {i}: {}", names.join(" → ")));
            }
        }
        lines.push(String::new());
        for (i, &id) in order.iter().enumerate() {
            let node = &self.nodes[id as usize];
            let label = node.name.as_deref().unwrap_or("_");
            let fuse_tag = fused_set
                .get(&id)
                .map_or(String::new(), |g| format!(" [fuse:{g}]"));
            lines.push(format!(
                "  {:>3}. {label}: {}{fuse_tag}",
                i + 1,
                node.op.kind_str()
            ));
        }
        lines.join("\n")
    }

    /// Topological order of all ancestors of `output` (post-order DFS).
    fn topo_order(&self, output: NodeId) -> Vec<u32> {
        let mut visited = vec![false; self.nodes.len()];
        let mut order = Vec::new();
        self.topo_dfs(output.0, &mut visited, &mut order);
        order
    }

    fn topo_dfs(&self, id: u32, visited: &mut [bool], order: &mut Vec<u32>) {
        if visited[id as usize] {
            return;
        }
        visited[id as usize] = true;
        for dep in self.nodes[id as usize].op.inputs() {
            self.topo_dfs(dep.0, visited, order);
        }
        order.push(id);
    }

    // ── Fusion ───────────────────────────────────────────────────────────

    /// Count how many nodes reference each node.
    fn ref_counts(&self) -> Vec<u32> {
        let mut counts = vec![0u32; self.nodes.len()];
        for node in &self.nodes {
            for dep in node.op.inputs() {
                counts[dep.0 as usize] += 1;
            }
        }
        counts
    }

    /// Identify chains of element-wise ops that can be fused into a single pass.
    ///
    /// Returns groups where each group is a sequence of node IDs that form a
    /// fusable chain. A chain breaks when:
    /// - The op is not element-wise
    /// - The node has multiple consumers (ref_count > 1)
    pub fn fusion_groups(&self) -> Vec<Vec<NodeId>> {
        let refs = self.ref_counts();
        let mut visited = vec![false; self.nodes.len()];
        let mut groups = Vec::new();

        for i in 0..self.nodes.len() {
            if visited[i] || !self.nodes[i].op.is_elementwise() {
                continue;
            }
            // Walk backwards along single-consumer element-wise chains.
            let mut chain = vec![NodeId(i as u32)];
            visited[i] = true;

            // Walk forward: find downstream element-wise ops where we're the sole input
            let mut cursor = i;
            loop {
                // Find a downstream node that has `cursor` as input
                let next = self.nodes.iter().enumerate().find(|(j, n)| {
                    !visited[*j]
                        && n.op.is_elementwise()
                        && n.op.inputs().contains(&NodeId(cursor as u32))
                        && refs[cursor] == 1
                });
                match next {
                    Some((j, _)) => {
                        chain.push(NodeId(j as u32));
                        visited[j] = true;
                        cursor = j;
                    }
                    None => break,
                }
            }

            if chain.len() > 1 {
                groups.push(chain);
            }
        }
        groups
    }

    // ── Export: DOT ──────────────────────────────────────────────────────

    /// Export the graph as a Graphviz DOT string.
    pub fn to_dot(&self) -> String {
        let mut out = String::from("digraph ComputeGraph {\n  rankdir=TB;\n  node [shape=box];\n");
        for (i, node) in self.nodes.iter().enumerate() {
            let fallback = node.op.label();
            let label = node.name.as_deref().unwrap_or(&fallback);
            out.push_str(&format!("  n{i} [label=\"{label}\"];\n"));
            for dep in node.op.inputs() {
                out.push_str(&format!("  n{} -> n{i};\n", dep.0));
            }
        }
        out.push_str("}\n");
        out
    }

    // ── Export: ONNX ─────────────────────────────────────────────────────

    /// Export the graph to ONNX protobuf bytes.
    ///
    /// ONNX is a platform-neutral interchange format for computational graphs.
    /// This produces a self-contained ModelProto that can be loaded by
    /// ONNX Runtime, TensorRT, CoreML, etc.
    ///
    /// We emit raw protobuf without a codegen dependency — the ONNX spec is
    /// stable enough that hand-encoding the ~10 message types is simpler and
    /// avoids pulling in `prost` or `protobuf` crates.
    pub fn to_onnx(&self, model_name: &str) -> Vec<u8> {
        onnx::encode(self, model_name)
    }

    /// Convenience: write ONNX to a file.
    pub fn save_onnx(&self, path: impl AsRef<std::path::Path>, model_name: &str) {
        let bytes = self.to_onnx(model_name);
        std::fs::write(path, bytes).expect("failed to write ONNX");
    }

    // ── Export: Visual ───────────────────────────────────────────────────

    /// Convert this computational graph into a visual graph for rendering.
    ///
    /// Each node becomes a `VNode<2>` with auto-layout.  The resulting
    /// visual graph can be rendered via `Renderable<()>`.
    pub fn to_visual(&self, output: NodeId) -> crate::visual::VisualGraph<2> {
        use crate::layout::V;
        use crate::visual::{VNode, VisualGraph};

        let order = self.topo_order(output);
        let mut vg = VisualGraph::<2>::new("graph");
        let mut id_to_idx: HashMap<u32, usize> = HashMap::new();

        for &id in &order {
            let node = &self.nodes[id as usize];
            let label = node
                .name
                .as_deref()
                .unwrap_or(&format!("n{id}"))
                .to_string();
            let tag = node.op.kind_tag();
            let mut vn = VNode::new(V([0.0, 0.0]), label, tag);

            // Set ports based on inputs/outputs.
            let inputs = node.op.inputs();
            if !inputs.is_empty() {
                vn.ports_in = (0..inputs.len()).map(|i| format!("in{i}")).collect();
            }
            vn.ports_out = vec!["out".to_string()];

            let idx = vg.add(vn);
            id_to_idx.insert(id, idx);
        }

        // Add edges.
        for &id in &order {
            let node = &self.nodes[id as usize];
            let dst = id_to_idx[&id];
            for (port, dep) in node.op.inputs().iter().enumerate() {
                if let Some(&src) = id_to_idx.get(&dep.0) {
                    vg.edge_ports(src, 0, dst, port);
                }
            }
        }

        vg.auto_layout();
        vg
    }
}

// ── Macro-generated Graph convenience methods ────────────────────────────

for_each_unary!(graph_unary);

for_each_binary!(graph_binary);

for_each_reduce!(graph_reduce);
