struct VertexInput { @location(0) position: vec2<f32> };
struct InstanceInput {
    @location(1) bounds:         vec4<f32>,
    @location(2) color:          vec4<f32>,
    @location(3) params:         vec4<f32>,
    @location(4) border_color:   vec4<f32>,
    @location(5) border_widths:  vec4<f32>,
};
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color:          vec4<f32>,
    @location(1) uv:             vec2<f32>,
    @location(2) rect_size:      vec2<f32>,
    @location(3) params:         vec4<f32>,
    @location(4) border_color:   vec4<f32>,
    @location(5) border_widths:  vec4<f32>,
};
struct Uniforms { screen_size: vec2<f32> }
@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@vertex
fn vs_main(model: VertexInput, instance: InstanceInput) -> VertexOutput {
    var out: VertexOutput;
    // params.y encodes rotation angle (radians). 0 = axis-aligned rect.
    let angle = instance.params.y;
    let center = vec2<f32>(
        instance.bounds.x + instance.bounds.z * 0.5,
        instance.bounds.y + instance.bounds.w * 0.5
    );
    let local = (model.position - vec2<f32>(0.5)) * instance.bounds.zw;
    let cos_a = cos(angle);
    let sin_a = sin(angle);
    let rotated = vec2<f32>(
        local.x * cos_a - local.y * sin_a,
        local.x * sin_a + local.y * cos_a
    );
    let pos = center + rotated;
    let clip_x = (pos.x / uniforms.screen_size.x) * 2.0 - 1.0;
    let clip_y = 1.0 - (pos.y / uniforms.screen_size.y) * 2.0;
    out.clip_position = vec4<f32>(clip_x, clip_y, 0.0, 1.0);
    out.color = instance.color;
    out.uv = model.position;
    out.rect_size = instance.bounds.zw;
    out.params = instance.params;
    out.border_color = instance.border_color;
    out.border_widths = instance.border_widths;
    return out;
}

fn sdf_round_rect(p: vec2<f32>, half: vec2<f32>, r: f32) -> f32 {
    let q = abs(p) - half + vec2<f32>(r);
    return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - r;
}

/// 2D cross product (z-component of cross in 3D).
fn cross2d(a: vec2<f32>, b: vec2<f32>) -> f32 {
    return a.x * b.y - a.y * b.x;
}

/// Point-in-triangle test using sign of cross products.
fn point_in_triangle(p: vec2<f32>, v0: vec2<f32>, v1: vec2<f32>, v2: vec2<f32>) -> bool {
    let d0 = cross2d(v1 - v0, p - v0);
    let d1 = cross2d(v2 - v1, p - v1);
    let d2 = cross2d(v0 - v2, p - v2);
    let has_neg = (d0 < 0.0) || (d1 < 0.0) || (d2 < 0.0);
    let has_pos = (d0 > 0.0) || (d1 > 0.0) || (d2 > 0.0);
    return !(has_neg && has_pos);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Triangle mode: params.z > 0.5 — vertices stored in border_color/border_widths
    if in.params.z > 0.5 {
        let v0 = in.border_color.xy;
        let v1 = in.border_color.zw;
        let v2 = in.border_widths.xy;
        if !point_in_triangle(in.uv, v0, v1, v2) {
            discard;
        }
        let col = in.color;
        return vec4<f32>(col.rgb * col.a, col.a);
    }

    let p = in.uv * in.rect_size - in.rect_size * 0.5;
    let half = in.rect_size * 0.5;
    let radius = min(in.params.x, min(half.x, half.y));

    // Per-side border widths: top, right, bottom, left.
    let bt = in.border_widths.x;
    let br = in.border_widths.y;
    let bb = in.border_widths.z;
    let bl = in.border_widths.w;
    let has_border = bt + br + bb + bl > 0.0;

    let d = sdf_round_rect(p, half, radius);
    if d > 0.5 { discard; }
    let aa = 1.0 - smoothstep(-0.5, 0.5, d);

    var col = in.color;
    if has_border {
        // Compute an inner SDF shrunk by the average border width on each axis.
        // This gives the correct rounded-rect inset shape at corners instead
        // of purely axis-aligned edge checks.
        let inset_x = (bl + br) * 0.5;
        let inset_y = (bt + bb) * 0.5;
        let inner_half = half - vec2<f32>(inset_x, inset_y);
        let inner_r = max(radius - min(inset_x, inset_y), 0.0);
        // Shift the sampling point by asymmetric border offsets
        let inner_p = p + vec2<f32>((bl - br) * 0.5, (bt - bb) * 0.5);
        let d_inner = sdf_round_rect(inner_p, max(inner_half, vec2<f32>(0.0)), inner_r);

        // Pixel is in the border when it's outside the inner rounded rect
        if d_inner > 0.0 {
            col = in.border_color;
        }
    }

    return vec4<f32>(col.rgb * col.a * aa, col.a * aa);
}
