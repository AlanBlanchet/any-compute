struct VertexInput { @location(0) position: vec2<f32> };
struct InstanceInput {
    @location(1) bounds:       vec4<f32>,
    @location(2) color:        vec4<f32>,
    @location(3) params:       vec4<f32>,
    @location(4) border_color: vec4<f32>,
};
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color:        vec4<f32>,
    @location(1) uv:           vec2<f32>,
    @location(2) rect_size:    vec2<f32>,
    @location(3) params:       vec4<f32>,
    @location(4) border_color: vec4<f32>,
};
struct Uniforms { screen_size: vec2<f32> }
@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@vertex
fn vs_main(model: VertexInput, instance: InstanceInput) -> VertexOutput {
    var out: VertexOutput;
    let pos = vec2<f32>(
        instance.bounds.x + model.position.x * instance.bounds.z,
        instance.bounds.y + model.position.y * instance.bounds.w
    );
    let clip_x = (pos.x / uniforms.screen_size.x) * 2.0 - 1.0;
    let clip_y = 1.0 - (pos.y / uniforms.screen_size.y) * 2.0;
    out.clip_position = vec4<f32>(clip_x, clip_y, 0.0, 1.0);
    out.color = instance.color;
    out.uv = model.position;
    out.rect_size = instance.bounds.zw;
    out.params = instance.params;
    out.border_color = instance.border_color;
    return out;
}

fn sdf_round_rect(p: vec2<f32>, half: vec2<f32>, r: f32) -> f32 {
    let q = abs(p) - half + vec2<f32>(r);
    return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - r;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let p = in.uv * in.rect_size - in.rect_size * 0.5;
    let half = in.rect_size * 0.5;
    let radius = min(in.params.x, min(half.x, half.y));
    let border_w = in.params.y;

    let d = sdf_round_rect(p, half, radius);
    if d > 0.5 { discard; }
    let aa = 1.0 - smoothstep(-0.5, 0.5, d);

    var col = in.color;
    if border_w > 0.0 {
        let ir = max(radius - border_w, 0.0);
        let inner_d = sdf_round_rect(p, half - vec2<f32>(border_w), ir);
        if inner_d > 0.0 { col = in.border_color; }
    }

    return vec4<f32>(col.rgb * col.a * aa, col.a * aa);
}
