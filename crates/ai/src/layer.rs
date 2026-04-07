//! Neural network layer trait and implementations.

use any_compute_core::graph::{Graph, NodeId};
use any_compute_core::kernel::UnaryOp;

/// A neural network layer that produces a sub-graph (one forward pass).
///
/// Every layer records its operations into a [`Graph`] and returns the
/// output `NodeId`.  Layers can be composed via [`Network`](super::Network).
pub trait Layer {
    /// Build this layer in the graph.  `input` is the previous layer's output.
    fn forward(&self, graph: &mut Graph, input: NodeId) -> NodeId;

    /// Human-readable name for visualization.
    fn name(&self) -> &str;

    /// Tag for visual graph coloring.
    fn tag(&self) -> &str {
        "custom"
    }

    /// Input → output dimensions (when known).
    fn dims(&self) -> Option<(usize, usize)> {
        None
    }

    /// Extract child layer info (for containers like Residual/Sequential).
    fn children_info(&self, _graph: &Graph, _parent_start: u32) -> Vec<super::LayerInfo> {
        Vec::new()
    }
}

/// Fully-connected linear layer: `output = input × weights + bias`.
pub struct Linear {
    pub weights: Vec<f64>,
    pub bias: Vec<f64>,
    pub in_features: usize,
    pub out_features: usize,
}

impl Linear {
    pub fn new(in_f: usize, out_f: usize) -> Self {
        let bound = (6.0 / (in_f + out_f) as f64).sqrt();
        let mut rng_state = 42u64;
        let weights: Vec<f64> = (0..in_f * out_f)
            .map(|_| {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let u = (rng_state >> 33) as f64 / (1u64 << 31) as f64;
                u * 2.0 * bound - bound
            })
            .collect();
        let bias = vec![0.0; out_f];
        Self {
            weights,
            bias,
            in_features: in_f,
            out_features: out_f,
        }
    }
}

impl Layer for Linear {
    fn forward(&self, g: &mut Graph, input: NodeId) -> NodeId {
        let w = g.constant(self.weights.clone());
        g.name(w, "weights");
        let b = g.constant(self.bias.clone());
        g.name(b, "bias");
        let mm = g.gemm(input, w, 1, self.out_features, self.in_features);
        g.name(mm, "matmul");
        let out = g.add(mm, b);
        g.name(out, "linear_out");
        out
    }
    fn name(&self) -> &str {
        "Linear"
    }
    fn tag(&self) -> &str {
        "linear"
    }
    fn dims(&self) -> Option<(usize, usize)> {
        Some((self.in_features, self.out_features))
    }
}

/// Activation layer — wraps any [`UnaryOp`].
pub struct Activation {
    pub op: UnaryOp,
    label: String,
    in_features: Option<usize>,
    out_features: Option<usize>,
}

impl Activation {
    /// Custom named activation/operation (for graph visualization).
    pub fn new(label: &str, in_features: usize, out_features: usize) -> Self {
        Self {
            op: UnaryOp::Relu,
            label: label.into(),
            in_features: Some(in_features),
            out_features: Some(out_features),
        }
    }
    pub fn relu() -> Self {
        Self {
            op: UnaryOp::Relu,
            label: "ReLU".into(),
            in_features: None,
            out_features: None,
        }
    }
    pub fn sigmoid() -> Self {
        Self {
            op: UnaryOp::Sigmoid,
            label: "Sigmoid".into(),
            in_features: None,
            out_features: None,
        }
    }
    pub fn tanh_act() -> Self {
        Self {
            op: UnaryOp::Tanh,
            label: "Tanh".into(),
            in_features: None,
            out_features: None,
        }
    }
}

impl Layer for Activation {
    fn forward(&self, g: &mut Graph, input: NodeId) -> NodeId {
        let out = g.unary(input, self.op);
        g.name(out, &self.label);
        out
    }
    fn name(&self) -> &str {
        &self.label
    }
    fn tag(&self) -> &str {
        match self.label.as_str() {
            "Attention" | "CausalAttn" => "reduce",
            "GELU" => "relu",
            _ => "relu",
        }
    }
    fn dims(&self) -> Option<(usize, usize)> {
        match (self.in_features, self.out_features) {
            (Some(i), Some(o)) => Some((i, o)),
            _ => None,
        }
    }
}

/// Batch normalization: `output = (input - mean) * gamma + beta`.
pub struct BatchNorm {
    pub features: usize,
    pub gamma: Vec<f64>,
    pub beta: Vec<f64>,
    pub running_mean: Vec<f64>,
    pub running_var: Vec<f64>,
    pub eps: f64,
}

impl BatchNorm {
    pub fn new(features: usize) -> Self {
        Self {
            features,
            gamma: vec![1.0; features],
            beta: vec![0.0; features],
            running_mean: vec![0.0; features],
            running_var: vec![1.0; features],
            eps: 1e-5,
        }
    }
}

impl Layer for BatchNorm {
    fn forward(&self, g: &mut Graph, input: NodeId) -> NodeId {
        let mean = g.constant(self.running_mean.clone());
        g.name(mean, "bn_mean");
        let centered = g.sub(input, mean);
        let var = g.constant(
            self.running_var
                .iter()
                .map(|v| (v + self.eps).sqrt())
                .collect(),
        );
        g.name(var, "bn_std");
        let normed = g.div(centered, var);
        let gamma = g.constant(self.gamma.clone());
        g.name(gamma, "gamma");
        let scaled = g.mul(normed, gamma);
        let beta = g.constant(self.beta.clone());
        g.name(beta, "beta");
        let out = g.add(scaled, beta);
        g.name(out, "bn_out");
        out
    }
    fn name(&self) -> &str {
        "BatchNorm"
    }
    fn tag(&self) -> &str {
        "batchnorm"
    }
    fn dims(&self) -> Option<(usize, usize)> {
        Some((self.features, self.features))
    }
}

/// Residual (skip) connection: `output = layer(input) + input`.
pub struct Residual {
    pub layers: Vec<Box<dyn Layer>>,
}

impl Residual {
    pub fn new(layers: Vec<Box<dyn Layer>>) -> Self {
        Self { layers }
    }
}

impl Layer for Residual {
    fn forward(&self, g: &mut Graph, input: NodeId) -> NodeId {
        let mut x = input;
        for layer in &self.layers {
            x = layer.forward(g, x);
        }
        let out = g.add(x, input);
        g.name(out, "residual_add");
        out
    }
    fn name(&self) -> &str {
        "Residual"
    }
    fn tag(&self) -> &str {
        "residual"
    }
    fn children_info(&self, _graph: &Graph, _parent_start: u32) -> Vec<super::LayerInfo> {
        self.layers
            .iter()
            .map(|l| super::LayerInfo::from_layer(l.as_ref(), (0, 0), Vec::new()))
            .collect()
    }
}

/// Sequential container — chains layers in order.
pub struct Sequential {
    pub layers: Vec<Box<dyn Layer>>,
    pub label: String,
}

impl Sequential {
    pub fn new(label: impl Into<String>, layers: Vec<Box<dyn Layer>>) -> Self {
        Self {
            layers,
            label: label.into(),
        }
    }
}

impl Layer for Sequential {
    fn forward(&self, g: &mut Graph, input: NodeId) -> NodeId {
        let mut x = input;
        for layer in &self.layers {
            x = layer.forward(g, x);
        }
        x
    }
    fn name(&self) -> &str {
        &self.label
    }
    fn tag(&self) -> &str {
        "custom"
    }
}
