// Element-wise map compute shader.
// Placeholder {{BODY}} is replaced at runtime with the per-element expression.

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size({{WORKGROUP_SIZE}})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= arrayLength(&input) {
        return;
    }
    let v = input[i];
    output[i] = {{BODY}};
}
