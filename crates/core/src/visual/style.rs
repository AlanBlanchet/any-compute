use crate::render::Color;

// ═══════════════════════════════════════════════════════════════════════════
// ── Style & theming ─────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

/// Visual styling for a node.
#[derive(Debug, Clone)]
pub struct NodeStyle {
    pub fill: Color,
    pub header: Color,
    pub border: Color,
    pub text: Color,
    pub corner_radius: f64,
    pub port_radius: f64,
}

impl Default for NodeStyle {
    fn default() -> Self {
        Self {
            fill: Color::rgb(40, 44, 52),
            header: Color::rgb(60, 65, 80),
            border: Color::rgb(80, 85, 100),
            text: Color::rgb(220, 220, 230),
            corner_radius: 6.0,
            port_radius: 4.0,
        }
    }
}

/// Map an operation tag to a consistent color.
pub fn tag_color(tag: &str) -> Color {
    match tag {
        "input" | "placeholder" => Color::rgb(80, 160, 80),
        "constant" | "const" => Color::rgb(100, 130, 180),
        "output" => Color::rgb(200, 80, 80),
        "relu" | "sigmoid" | "tanh" => Color::rgb(200, 140, 40),
        "conv" | "conv2d" => Color::rgb(60, 140, 200),
        "linear" | "gemm" | "matmul" => Color::rgb(140, 80, 200),
        "batchnorm" | "bn" | "norm" => Color::rgb(180, 100, 60),
        "add" | "sub" | "mul" | "div" => Color::rgb(80, 180, 180),
        "reduce" | "sum" | "mean" | "pool" => Color::rgb(200, 80, 160),
        "reshape" | "concat" | "slice" | "gather" => Color::rgb(160, 160, 80),
        "residual" | "skip" => Color::rgb(200, 200, 80),
        "dropout" | "custom" => Color::rgb(120, 120, 120),
        _ => Color::rgb(100, 100, 120),
    }
}

