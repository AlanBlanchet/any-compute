// Tiled matrix multiply (GEMM) compute shader.
// {{TILE_SIZE}} controls the tile dimension; shared memory = TILE^2 elements.

struct Uniforms {
    M: u32,
    N: u32,
    K: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;

const TILE: u32 = {{TILE_SIZE}}u;
var<workgroup> tileA: array<f32, {{TILE_ELEMENTS}}>;
var<workgroup> tileB: array<f32, {{TILE_ELEMENTS}}>;

@compute @workgroup_size({{TILE_SIZE}}, {{TILE_SIZE}})
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = gid.y;
    let col = gid.x;
    var sum: f32 = 0.0;

    let numTiles = (uniforms.K + TILE - 1u) / TILE;

    for (var t: u32 = 0u; t < numTiles; t++) {
        let aCol = t * TILE + lid.x;
        let bRow = t * TILE + lid.y;

        if row < uniforms.M && aCol < uniforms.K {
            tileA[lid.y * TILE + lid.x] = A[row * uniforms.K + aCol];
        } else {
            tileA[lid.y * TILE + lid.x] = 0.0;
        }

        if bRow < uniforms.K && col < uniforms.N {
            tileB[lid.y * TILE + lid.x] = B[bRow * uniforms.N + col];
        } else {
            tileB[lid.y * TILE + lid.x] = 0.0;
        }

        workgroupBarrier();

        for (var k: u32 = 0u; k < TILE; k++) {
            sum += tileA[lid.y * TILE + k] * tileB[k * TILE + lid.x];
        }

        workgroupBarrier();
    }

    if row < uniforms.M && col < uniforms.N {
        C[row * uniforms.N + col] = sum;
    }
}
