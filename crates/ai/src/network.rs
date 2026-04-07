//! Neural network builder — `Network`, `LayerInfo`, architecture constructors.

use any_compute_core::compute::Device;
use any_compute_core::graph::{Graph, NodeId};
use any_compute_core::layout::V;
use any_compute_core::visual::{Graphable, VNode, VisualGraph};
use std::collections::HashMap;

use super::layer::*;

/// Metadata about a layer in the network — name, tag, graph node range, sub-layers.
#[derive(Clone, Debug)]
pub struct LayerInfo {
    pub name: String,
    pub tag: String,
    /// Input and output dimensions (when known).
    pub dims: Option<(usize, usize)>,
    /// Range of graph node IDs owned by this layer (start, end exclusive).
    pub node_range: (u32, u32),
    /// Sub-layer info (for Residual/Sequential containers).
    pub children: Vec<LayerInfo>,
}

impl LayerInfo {
    /// Build from a layer trait object.
    pub fn from_layer(layer: &dyn Layer, node_range: (u32, u32), children: Vec<LayerInfo>) -> Self {
        Self {
            name: layer.name().to_string(),
            tag: layer.tag().to_string(),
            dims: layer.dims(),
            node_range,
            children,
        }
    }

    /// Format a display label: "Linear 4→8" when dims known, else just name.
    pub fn display_label(&self) -> String {
        if let Some((i, o)) = self.dims {
            format!("{} {}→{}", self.name, i, o)
        } else {
            self.name.clone()
        }
    }

    /// Label with instance number for repeated layers (e.g. "ResBlock #2 8→8").
    pub fn numbered_label(&self, n: usize) -> String {
        let base = self.display_label();
        if n > 1 { format!("{base} #{n}") } else { base }
    }

    /// Port label for a given side: "[4]" when dim known, else fallback.
    pub fn port_label(&self, output: bool) -> String {
        match (self.dims, output) {
            (Some((d, _)), false) => format!("[{d}]"),
            (Some((_, d)), true) => format!("[{d}]"),
            (None, false) => "in".into(),
            (None, true) => "out".into(),
        }
    }

    /// Convert to a [`VNode<2>`] with dimension-aware labels and ports.
    pub fn to_vnode(&self) -> VNode<2> {
        let label = self.display_label();
        let ip = self.port_label(false);
        let op = self.port_label(true);
        VNode::new(V([0.0, 0.0]), &label, &self.tag)
            .ins(&[&ip])
            .outs(&[&op])
    }
}

/// A complete neural network: input specification + layer stack + graph.
///
/// Builds the computational graph on construction, ready for eval or
/// visual export.  Preserves layer structure for high-level visualization.
pub struct Network {
    pub graph: Graph,
    pub input: NodeId,
    pub output: NodeId,
    pub label: String,
    pub layer_info: Vec<LayerInfo>,
}

impl Network {
    /// Build a network from a sequential list of layers.
    pub fn build(
        label: impl Into<String>,
        in_features: usize,
        layers: Vec<Box<dyn Layer>>,
    ) -> Self {
        let mut g = Graph::new();
        let input = g.input("x");
        g.name(input, &format!("input[{in_features}]"));

        let mut x = input;
        let mut infos = Vec::new();
        for layer in &layers {
            let start = g.len() as u32;
            x = layer.forward(&mut g, x);
            let end = g.len() as u32;
            infos.push(LayerInfo::from_layer(
                layer.as_ref(),
                (start, end),
                layer.children_info(&g, start),
            ));
        }

        Self {
            graph: g,
            input,
            output: x,
            label: label.into(),
            layer_info: infos,
        }
    }

    /// Forward pass: evaluate the network on input data.
    pub fn forward(&self, data: &[f64], device: &Device) -> Vec<f64> {
        let mut inputs = HashMap::new();
        inputs.insert("x".to_string(), data.to_vec());
        self.graph.eval(self.output, device, &inputs)
    }

    /// Convert to a layer-level visual graph.
    pub fn to_visual(&self) -> VisualGraph<2> {
        let mut vg = VisualGraph::<2>::new(&self.label);

        let in_dim = self.layer_info.iter().find_map(|i| i.dims.map(|d| d.0));
        let in_label = in_dim.map_or("x".to_string(), |d| format!("[{d}]"));

        let inp = vg.add(
            VNode::new(V([0.0, 0.0]), "Input", "input")
                .ins(&[])
                .outs(&[&in_label]),
        );

        let mut prev = inp;
        let mut type_counts: HashMap<String, usize> = HashMap::new();
        for info in &self.layer_info {
            let count = type_counts.entry(info.name.clone()).or_insert(0);
            *count += 1;
            let label = info.numbered_label(*count);

            let mut node = VNode::new(V([0.0, 0.0]), &label, &info.tag)
                .ins(&[&info.port_label(false)])
                .outs(&[&info.port_label(true)]);

            if !info.children.is_empty() {
                let mut sub = VisualGraph::<2>::new(&label);
                let mut sub_prev = None;
                let mut child_counts: HashMap<String, usize> = HashMap::new();
                for child in &info.children {
                    let cc = child_counts.entry(child.name.clone()).or_insert(0);
                    *cc += 1;
                    let ci = sub.add(
                        VNode::new(V([0.0, 0.0]), &child.numbered_label(*cc), &child.tag)
                            .ins(&[&child.port_label(false)])
                            .outs(&[&child.port_label(true)]),
                    );
                    if let Some(p) = sub_prev {
                        sub.edge(p, ci);
                    }
                    sub_prev = Some(ci);
                }
                sub.auto_layout();
                node = node.children(sub);
            }

            let idx = vg.add(node);
            vg.edge(prev, idx);
            prev = idx;
        }

        let out_dim = self
            .layer_info
            .iter()
            .rev()
            .find_map(|i| i.dims.map(|d| d.1));
        let out_label = out_dim.map_or("y".to_string(), |d| format!("[{d}]"));

        let out = vg.add(
            VNode::new(V([0.0, 0.0]), "Output", "output")
                .ins(&[&out_label])
                .outs(&[]),
        );
        vg.edge(prev, out);

        vg.auto_layout();
        vg
    }

    /// Textual execution plan.
    pub fn plan(&self) -> String {
        self.graph.plan(self.output)
    }
}

impl Graphable for Network {
    fn to_graph(&self) -> VisualGraph<2> {
        self.to_visual()
    }
}

// ── Shared building blocks for network builders ─────────────────────────

fn mlp_block(dim: usize, hidden: usize) -> Box<dyn Layer> {
    Box::new(Residual::new(vec![
        Box::new(BatchNorm::new(dim)),
        Box::new(Linear::new(dim, hidden)),
        Box::new(Activation::new("GELU", hidden, hidden)),
        Box::new(Linear::new(hidden, dim)),
    ]))
}

fn attn_block(dim: usize, heads: usize, label: &str) -> Box<dyn Layer> {
    let head_dim = dim / heads.max(1);
    Box::new(Residual::new(vec![
        Box::new(BatchNorm::new(dim)),
        Box::new(Linear::new(dim, dim * 3)),
        Box::new(Activation::new(label, dim, head_dim)),
        Box::new(Linear::new(dim, dim)),
    ]))
}

fn transformer_blocks(
    layers: &mut Vec<Box<dyn Layer>>,
    dim: usize,
    depth: usize,
    heads: usize,
    attn_label: &str,
) {
    let mlp_hidden = dim * 4;
    for _ in 0..depth {
        layers.push(attn_block(dim, heads, attn_label));
        layers.push(mlp_block(dim, mlp_hidden));
    }
}

fn output_head(layers: &mut Vec<Box<dyn Layer>>, dim: usize, out: usize) {
    layers.push(Box::new(BatchNorm::new(dim)));
    layers.push(Box::new(Linear::new(dim, out)));
}

// ── Network builders ────────────────────────────────────────────────────

/// Build a ResNet-style network.
pub fn resnet(in_features: usize, hidden: usize, blocks: usize, out_classes: usize) -> Network {
    let mut layers: Vec<Box<dyn Layer>> = Vec::new();
    layers.push(Box::new(Linear::new(in_features, hidden)));
    layers.push(Box::new(Activation::relu()));
    for _ in 0..blocks {
        layers.push(Box::new(Residual::new(vec![
            Box::new(Linear::new(hidden, hidden)),
            Box::new(BatchNorm::new(hidden)),
            Box::new(Activation::relu()),
            Box::new(Linear::new(hidden, hidden)),
            Box::new(BatchNorm::new(hidden)),
        ])));
        layers.push(Box::new(Activation::relu()));
    }
    layers.push(Box::new(Linear::new(hidden, out_classes)));
    Network::build("ResNet", in_features, layers)
}

/// Build a Vision Transformer (ViT) network graph.
pub fn vit(
    patch_dim: usize,
    embed_dim: usize,
    depth: usize,
    heads: usize,
    classes: usize,
) -> Network {
    let mut layers: Vec<Box<dyn Layer>> = Vec::new();
    layers.push(Box::new(Linear::new(patch_dim, embed_dim)));
    layers.push(Box::new(Activation::relu()));
    transformer_blocks(&mut layers, embed_dim, depth, heads, "Attention");
    output_head(&mut layers, embed_dim, classes);
    Network::build("ViT", patch_dim, layers)
}

/// Build a GPT-style causal language model graph.
pub fn gpt(vocab: usize, embed_dim: usize, depth: usize, heads: usize) -> Network {
    let mut layers: Vec<Box<dyn Layer>> = Vec::new();
    layers.push(Box::new(Linear::new(vocab, embed_dim)));
    transformer_blocks(&mut layers, embed_dim, depth, heads, "CausalAttn");
    output_head(&mut layers, embed_dim, vocab);
    Network::build("GPT", vocab, layers)
}

/// Build an MLP-Mixer network graph.
pub fn mixer(patches: usize, channels: usize, depth: usize, classes: usize) -> Network {
    let mut layers: Vec<Box<dyn Layer>> = Vec::new();
    layers.push(Box::new(Linear::new(patches, channels)));
    let ch_hidden = channels * 4;
    for _ in 0..depth {
        layers.push(mlp_block(channels, channels));
        layers.push(mlp_block(channels, ch_hidden));
    }
    output_head(&mut layers, channels, classes);
    Network::build("MLP-Mixer", patches, layers)
}

#[cfg(test)]
mod tests {
    use super::*;
    use any_compute_core::compute::Device;
    use any_compute_core::visual::Graphable;
    use std::collections::HashMap;

    fn dev() -> Device {
        Device::cpu()
    }
    fn empty() -> HashMap<String, Vec<f64>> {
        HashMap::new()
    }

    #[test]
    fn linear_layer_forward() {
        let lin = Linear::new(3, 2);
        let mut g = Graph::new();
        let x = g.input("x");
        let out = lin.forward(&mut g, x);

        let mut inputs = HashMap::new();
        inputs.insert("x".to_string(), vec![1.0, 0.0, 0.0]);
        let result = g.eval(out, &dev(), &inputs);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn resnet_builds_and_evals() {
        let net = resnet(4, 8, 2, 3);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let output = net.forward(&input, &dev());
        assert_eq!(output.len(), 3);

        let vg = net.to_visual();
        assert!(vg.len() >= 5, "expected ≥5 layer nodes, got {}", vg.len());
    }

    #[test]
    fn network_plan() {
        let net = resnet(4, 8, 1, 2);
        let plan = net.plan();
        assert!(plan.contains("steps"));
    }

    #[test]
    fn residual_skip_connection() {
        let block = Residual::new(vec![
            Box::new(Linear::new(4, 4)),
            Box::new(Activation::relu()),
        ]);
        let mut g = Graph::new();
        let x = g.constant(vec![1.0, 2.0, 3.0, 4.0]);
        let out = block.forward(&mut g, x);
        let result = g.eval(out, &dev(), &empty());
        assert_eq!(result.len(), 4);
        assert_ne!(result, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn sequential_chains_layers() {
        let seq = Sequential::new(
            "block",
            vec![
                Box::new(Linear::new(3, 4)),
                Box::new(Activation::relu()),
                Box::new(Linear::new(4, 2)),
            ],
        );
        let mut g = Graph::new();
        let x = g.input("x");
        let out = seq.forward(&mut g, x);

        let mut inputs = HashMap::new();
        inputs.insert("x".to_string(), vec![1.0, 0.5, -1.0]);
        let result = g.eval(out, &dev(), &inputs);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn network_graphable_labels_and_pixels() {
        use any_compute_core::render::Color;

        let net = resnet(2, 4, 2, 2);
        let g = net.to_graph();
        assert!(
            g.len() > 3,
            "resnet graph should have several nodes, got {}",
            g.len()
        );

        let prims = net.render_graph();
        assert!(prims.len() > 20, "too few primitives: {}", prims.len());

        let texts: Vec<String> = prims
            .iter()
            .filter_map(|p| {
                if let any_compute_core::render::Primitive::Text { content, .. } = p {
                    Some(content.clone())
                } else {
                    None
                }
            })
            .collect();
        assert!(texts.iter().any(|t| t == "Linear" || t.contains("linear")));
        assert!(texts.iter().any(|t| t == "ReLU" || t.contains("relu")));
        assert!(
            texts
                .iter()
                .any(|t| t == "Residual" || t.contains("residual"))
        );
        assert!(texts.iter().any(|t| t == "Input" || t == "Output"));

        let pb = net.capture_graph(400, 200);
        let bg = Color::rgb(25, 28, 36);
        let non_bg = (0..400).filter(|&x| pb.pixel(x, 50) != bg).count();
        assert!(non_bg > 10, "capture should render content");
    }
}
