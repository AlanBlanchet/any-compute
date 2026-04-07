use super::Graph;
use crate::kernel::{BinaryOp, ReduceOp, UnaryOp};

// ═══════════════════════════════════════════════════════════════════════════
// ── EvalProgress — progress feedback during graph evaluation ────────────
// ═══════════════════════════════════════════════════════════════════════════

/// Snapshot of evaluation progress passed to the [`Graph::eval_with`] callback.
pub struct EvalProgress<'a> {
    /// 1-based step index within the topological evaluation order.
    pub step: usize,
    /// Total number of nodes to evaluate.
    pub total: usize,
    /// Human-readable node label (user-given name or `nX`).
    pub label: &'a str,
    /// String description of the operation kind (e.g. `"unary:sqrt"`, `"binary:add"`).
    pub op_kind: String,
    /// Size of the (first) input data, or 0 for sources.
    pub input_size: usize,
}

// ═══════════════════════════════════════════════════════════════════════════
// ── NodeId — typed handle into the graph ────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

/// Opaque handle to a node in a computational graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub(crate) u32);

impl NodeId {
    /// Resolve the human-readable name of this node in a graph.
    pub fn label(self, graph: &Graph) -> String {
        graph.nodes[self.0 as usize]
            .name
            .clone()
            .unwrap_or_else(|| format!("n{}", self.0))
    }
}

impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "n{}", self.0)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ── GraphOp — the operation vocabulary ──────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

/// The set of operations a graph node can represent.
#[derive(Debug, Clone)]
pub enum GraphOp {
    /// Leaf: constant data baked into the graph.
    Constant(Vec<f64>),
    /// Leaf: named input placeholder — bound at eval time.
    Input(String),
    /// Element-wise unary: out[i] = op(a[i]).
    Unary { input: NodeId, op: UnaryOp },
    /// Element-wise binary: out[i] = op(a[i], b[i]).
    Binary {
        lhs: NodeId,
        rhs: NodeId,
        op: BinaryOp,
    },
    /// Reduction to scalar.
    Reduce { input: NodeId, op: ReduceOp },
    /// Inclusive prefix scan.
    Scan { input: NodeId, op: ReduceOp },
    /// Matrix multiply: C[m×n] = A[m×k] × B[k×n].
    Gemm {
        lhs: NodeId,
        rhs: NodeId,
        m: usize,
        n: usize,
        k: usize,
    },
    /// Gather: out[i] = data[indices[i]].
    Gather { data: NodeId, indices: NodeId },
    /// Reshape (no data movement — just a metadata annotation).
    Reshape { input: NodeId, shape: Vec<usize> },
    /// Concatenation of multiple tensors along axis 0.
    Concat(Vec<NodeId>),
    /// Slice [start..end).
    Slice {
        input: NodeId,
        start: usize,
        end: usize,
    },
    /// Custom user-provided fn (opaque — blocks fusion + export).
    Custom { input: NodeId, name: String },
}

impl GraphOp {
    /// All input node IDs this operation depends on.
    pub(super) fn inputs(&self) -> Vec<NodeId> {
        match self {
            Self::Constant(_) | Self::Input(_) => vec![],
            Self::Unary { input, .. }
            | Self::Reduce { input, .. }
            | Self::Scan { input, .. }
            | Self::Reshape { input, .. }
            | Self::Slice { input, .. }
            | Self::Custom { input, .. } => vec![*input],
            Self::Binary { lhs, rhs, .. } | Self::Gemm { lhs, rhs, .. } => vec![*lhs, *rhs],
            Self::Gather { data, indices } => vec![*data, *indices],
            Self::Concat(nodes) => nodes.clone(),
        }
    }

    /// Whether this is element-wise (fusable with adjacent element-wise ops).
    pub(super) fn is_elementwise(&self) -> bool {
        matches!(self, Self::Unary { .. } | Self::Binary { .. })
    }

    /// Short label for DOT / debug output.
    pub(super) fn label(&self) -> String {
        match self {
            Self::Constant(d) => {
                if d.len() <= 4 {
                    format!("const{d:?}")
                } else {
                    format!("const[{}]", d.len())
                }
            }
            Self::Input(name) => format!("input:{name}"),
            Self::Unary { op, .. } => format!("{op:?}"),
            Self::Binary { op, .. } => format!("{op:?}"),
            Self::Reduce { op, .. } => format!("reduce:{op:?}"),
            Self::Scan { op, .. } => format!("scan:{op:?}"),
            Self::Gemm { m, n, k, .. } => format!("gemm({m}×{k}·{k}×{n})"),
            Self::Gather { .. } => "gather".into(),
            Self::Reshape { shape, .. } => format!("reshape{shape:?}"),
            Self::Concat(nodes) => format!("concat({})", nodes.len()),
            Self::Slice { start, end, .. } => format!("slice[{start}..{end})"),
            Self::Custom { name, .. } => format!("custom:{name}"),
        }
    }

    /// Compact kind string for progress reporting (e.g. `"unary:sqrt"`, `"binary:add"`).
    pub(super) fn kind_str(&self) -> String {
        match self {
            Self::Constant(_) => "constant".into(),
            Self::Input(name) => format!("input:{name}"),
            Self::Unary { op, .. } => format!("unary:{op:?}"),
            Self::Binary { op, .. } => format!("binary:{op:?}"),
            Self::Reduce { op, .. } => format!("reduce:{op:?}"),
            Self::Scan { op, .. } => format!("scan:{op:?}"),
            Self::Gemm { .. } => "gemm".into(),
            Self::Gather { .. } => "gather".into(),
            Self::Reshape { .. } => "reshape".into(),
            Self::Concat(_) => "concat".into(),
            Self::Slice { .. } => "slice".into(),
            Self::Custom { name, .. } => format!("custom:{name}"),
        }
    }

    /// Short tag for visual node coloring (matches [`visual::tag_color`] keys).
    pub(super) fn kind_tag(&self) -> &'static str {
        match self {
            Self::Constant(_) => "constant",
            Self::Input(_) => "input",
            Self::Unary { op, .. } => match op {
                UnaryOp::Relu => "relu",
                UnaryOp::Sigmoid => "sigmoid",
                UnaryOp::Tanh => "tanh",
                _ => "relu", // generic unary → activation color
            },
            Self::Binary { .. } => "add",
            Self::Reduce { .. } => "reduce",
            Self::Scan { .. } => "reduce",
            Self::Gemm { .. } => "gemm",
            Self::Gather { .. } => "gather",
            Self::Reshape { .. } => "reshape",
            Self::Concat(_) => "concat",
            Self::Slice { .. } => "slice",
            Self::Custom { .. } => "custom",
        }
    }
}
