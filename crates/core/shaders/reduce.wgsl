// Workgroup-level reduction shader.
// Placeholder {{OP}} is replaced with the reduction expression (e.g., shared[lid.x] + shared[lid.x + stride]).
// {{WORKGROUP_SIZE}} controls both workgroup dispatch and shared memory allocation.

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

var<workgroup> shared: array<f32, {{WORKGROUP_SIZE}}>;

@compute @workgroup_size({{WORKGROUP_SIZE}})
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let i = wid.x * {{WORKGROUP_SIZE}}u + lid.x;
    shared[lid.x] = select(0.0, input[i], i < arrayLength(&input));
    workgroupBarrier();

    for (var stride = {{HALF_WORKGROUP}}u; stride > 0u; stride >>= 1u) {
        if lid.x < stride {
            shared[lid.x] = {{OP}};
        }
        workgroupBarrier();
    }

    if lid.x == 0u {
        output[wid.x] = shared[0];
    }
}
