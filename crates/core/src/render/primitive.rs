use crate::layout::{Point, Rect};
use super::{Color, Viewport};

/// A single draw command — references layout types for all spatial data.
#[derive(Debug, Clone)]
pub enum Primitive {
    /// Filled/stroked rectangle — references [`Rect`] for bounds.
    Rect {
        bounds: Rect,
        fill: Color,
        border: Option<Border>,
        corner_radius: f64,
    },
    /// Text at a position — references [`Point`] for anchor.
    Text {
        anchor: Point,
        content: String,
        font_size: f64,
        color: Color,
    },
    /// Line between two points — references [`Point`].
    Line {
        from: Point,
        to: Point,
        stroke: Color,
        width: f64,
    },
    /// Clip region — references [`Rect`].
    PushClip {
        bounds: Rect,
    },
    PopClip,
    /// Filled triangle — three screen-space vertices.
    Triangle {
        vertices: [Point; 3],
        fill: Color,
    },
}

#[derive(Debug, Clone, Copy)]
pub struct Border {
    pub color: Color,
    pub top: f64,
    pub right: f64,
    pub bottom: f64,
    pub left: f64,
}

impl Border {
    /// Create a border with equal width on all sides.
    pub fn uniform(color: Color, width: f64) -> Self {
        Self {
            color,
            top: width,
            right: width,
            bottom: width,
            left: width,
        }
    }

    /// Whether all four sides have the same width.
    pub fn is_uniform(&self) -> bool {
        self.top == self.right && self.right == self.bottom && self.bottom == self.left
    }
}

impl Primitive {
    /// Translate all spatial coordinates by `(dx, dy)`.
    pub fn offset(&self, dx: f64, dy: f64) -> Self {
        let d = Point::new(dx, dy);
        match self {
            Primitive::Rect {
                bounds,
                fill,
                border,
                corner_radius,
            } => Primitive::Rect {
                bounds: Rect::from_parts(bounds.origin + d, bounds.size),
                fill: *fill,
                border: *border,
                corner_radius: *corner_radius,
            },
            Primitive::Text {
                anchor,
                content,
                font_size,
                color,
            } => Primitive::Text {
                anchor: *anchor + d,
                content: content.clone(),
                font_size: *font_size,
                color: *color,
            },
            Primitive::Line {
                from,
                to,
                stroke,
                width,
            } => Primitive::Line {
                from: *from + d,
                to: *to + d,
                stroke: *stroke,
                width: *width,
            },
            Primitive::PushClip { bounds } => Primitive::PushClip {
                bounds: Rect::from_parts(bounds.origin + d, bounds.size),
            },
            Primitive::PopClip => Primitive::PopClip,
            Primitive::Triangle { vertices, fill } => Primitive::Triangle {
                vertices: [vertices[0] + d, vertices[1] + d, vertices[2] + d],
                fill: *fill,
            },
        }
    }
}

/// Ordered list of draw commands — the output of layout/paint phase.
#[derive(Debug, Clone, Default)]
pub struct RenderList {
    pub primitives: Vec<Primitive>,
}

impl RenderList {
    pub fn push(&mut self, p: Primitive) {
        self.primitives.push(p);
    }

    /// Push a line between two points.
    pub fn push_line(&mut self, x1: f64, y1: f64, x2: f64, y2: f64, stroke: Color, width: f64) {
        let dx = x2 - x1;
        let dy = y2 - y1;
        if dx * dx + dy * dy < 1.0 {
            return;
        }
        self.primitives.push(Primitive::Line {
            from: Point::new(x1, y1),
            to: Point::new(x2, y2),
            stroke,
            width,
        });
    }

    /// Push a circular dot at a position.
    pub fn push_dot(&mut self, x: f64, y: f64, size: f64, fill: Color) {
        self.primitives.push(Primitive::Rect {
            bounds: Rect::new(x - size / 2.0, y - size / 2.0, size, size),
            fill,
            border: None,
            corner_radius: size / 2.0,
        });
    }

    /// Push a filled rectangle.
    pub fn push_rect(&mut self, x: f64, y: f64, w: f64, h: f64, fill: Color) {
        self.primitives.push(Primitive::Rect {
            bounds: Rect::new(x, y, w, h),
            fill,
            border: None,
            corner_radius: 0.0,
        });
    }

    /// Push a filled triangle from three screen-space points.
    pub fn push_triangle(&mut self, pts: [Point; 3], fill: Color) {
        self.primitives.push(Primitive::Triangle {
            vertices: pts,
            fill,
        });
    }

    /// Push a text label at a position.
    pub fn push_text(&mut self, x: f64, y: f64, text: &str, font_size: f64, color: Color) {
        self.primitives.push(Primitive::Text {
            anchor: Point::new(x, y),
            content: text.to_string(),
            font_size,
            color,
        });
    }

    /// Push a clip region, render inside `f`, then pop.
    pub fn clipped(&mut self, bounds: Rect, f: impl FnOnce(&mut Self)) {
        self.push(Primitive::PushClip { bounds });
        f(self);
        self.push(Primitive::PopClip);
    }

    /// Composite another render list into this one, clipped and offset to `vp`.
    pub fn composite(&mut self, vp: Rect, source: &Self) {
        self.clipped(vp, |l| {
            for p in source.iter() {
                l.push(p.offset(vp.origin.x, vp.origin.y));
            }
        });
    }

    /// Push a rounded rectangle.
    pub fn push_rounded_rect(
        &mut self,
        x: f64,
        y: f64,
        w: f64,
        h: f64,
        fill: Color,
        corner_radius: f64,
    ) {
        self.primitives.push(Primitive::Rect {
            bounds: Rect::new(x, y, w, h),
            fill,
            border: None,
            corner_radius,
        });
    }

    /// Push a bordered rectangle.
    pub fn push_bordered_rect(
        &mut self,
        x: f64,
        y: f64,
        w: f64,
        h: f64,
        fill: Color,
        border: Border,
        corner_radius: f64,
    ) {
        self.primitives.push(Primitive::Rect {
            bounds: Rect::new(x, y, w, h),
            fill,
            border: Some(border),
            corner_radius,
        });
    }

    /// Push a circle (via a rounded rect where corner_radius = size/2).
    pub fn push_circle(&mut self, cx: f64, cy: f64, radius: f64, fill: Color) {
        let d = radius * 2.0;
        self.primitives.push(Primitive::Rect {
            bounds: Rect::new(cx - radius, cy - radius, d, d),
            fill,
            border: None,
            corner_radius: radius,
        });
    }

    /// Push a polyline (sequence of connected line segments).
    pub fn push_polyline(&mut self, points: &[Point], stroke: Color, width: f64) {
        for pair in points.windows(2) {
            self.push_line(pair[0].x, pair[0].y, pair[1].x, pair[1].y, stroke, width);
        }
    }

    /// Push a grid of horizontal and vertical lines.
    pub fn push_grid(
        &mut self,
        x: f64,
        y: f64,
        w: f64,
        h: f64,
        cols: usize,
        rows: usize,
        stroke: Color,
        width: f64,
    ) {
        for i in 0..=cols {
            let gx = x + w * i as f64 / cols as f64;
            self.push_line(gx, y, gx, y + h, stroke, width);
        }
        for i in 0..=rows {
            let gy = y + h * i as f64 / rows as f64;
            self.push_line(x, gy, x + w, gy, stroke, width);
        }
    }

    /// Push X and Y axes with labels at the origin.
    pub fn push_axes(&mut self, ox: f64, oy: f64, w: f64, h: f64, stroke: Color, width: f64) {
        // X axis
        self.push_line(ox, oy, ox + w, oy, stroke, width);
        // Y axis
        self.push_line(ox, oy, ox, oy - h, stroke, width);
    }

    /// Push a bar chart: evenly spaced vertical bars.
    pub fn push_bars(
        &mut self,
        x: f64,
        y: f64,
        w: f64,
        h: f64,
        values: &[f64],
        fill: Color,
        gap: f64,
    ) {
        if values.is_empty() {
            return;
        }
        let max_val = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if max_val <= 0.0 {
            return;
        }
        let n = values.len() as f64;
        let bar_w = (w - gap * (n - 1.0)) / n;
        for (i, &v) in values.iter().enumerate() {
            let bar_h = (v / max_val) * h;
            let bx = x + i as f64 * (bar_w + gap);
            let by = y + h - bar_h;
            self.push_rect(bx, by, bar_w, bar_h, fill);
        }
    }

    /// Push a line chart with grid, data lines, and dots.
    ///
    /// Draws horizontal grid lines, connected line segments, and dots at each
    /// data point inside the bounding box `(x, y, w, h)` with inner padding.
    pub fn push_line_chart(
        &mut self,
        x: f64,
        y: f64,
        w: f64,
        h: f64,
        data: &[f64],
        stroke: Color,
        dot_radius: f64,
    ) {
        if data.is_empty() {
            return;
        }
        let pad = 8.0;
        let cw = w - pad * 2.0;
        let ch = h - pad * 2.0;
        let max_val = data.iter().copied().fold(0.0_f64, f64::max).max(0.01);
        let min_val = data.iter().copied().fold(f64::INFINITY, f64::min);
        let range = (max_val - min_val).max(0.01);

        // Horizontal grid
        let gc = Color::rgba(255, 255, 255, 15);
        for i in 0..=4 {
            let gy = y + pad + ch * (1.0 - i as f64 / 4.0);
            self.push_line(x + pad, gy, x + pad + cw, gy, gc, 1.0);
        }

        let n = data.len();
        let denom = (n - 1).max(1) as f64;
        let px = |i: usize| x + pad + i as f64 / denom * cw;
        let py = |v: f64| y + pad + ch * (1.0 - (v - min_val) / range);

        // Line segments
        for i in 1..n {
            self.push_line(px(i - 1), py(data[i - 1]), px(i), py(data[i]), stroke, 2.0);
        }

        // Dots
        for i in 0..n {
            self.push_dot(px(i), py(data[i]), dot_radius, stroke);
        }
    }

    /// Offset all primitives by (dx, dy).
    pub fn translate(&mut self, dx: f64, dy: f64) {
        for p in &mut self.primitives {
            *p = p.offset(dx, dy);
        }
    }

    /// Append another RenderList's primitives.
    pub fn extend(&mut self, other: &RenderList) {
        self.primitives.extend_from_slice(&other.primitives);
    }

    /// Project a 3D line through a viewport and push it if both endpoints are visible.
    pub fn push_projected_line<P>(
        &mut self,
        vp: &impl Viewport<P>,
        a: P,
        b: P,
        color: Color,
        width: f64,
    ) {
        if let (Some((x1, y1, _)), Some((x2, y2, _))) = (vp.project(a), vp.project(b)) {
            self.push_line(x1, y1, x2, y2, color, width);
        }
    }

    pub fn clear(&mut self) {
        self.primitives.clear();
    }

    pub fn len(&self) -> usize {
        self.primitives.len()
    }

    pub fn is_empty(&self) -> bool {
        self.primitives.is_empty()
    }

    pub fn iter(&self) -> std::slice::Iter<'_, Primitive> {
        self.primitives.iter()
    }
}

