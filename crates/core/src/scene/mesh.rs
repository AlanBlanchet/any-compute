use crate::buffer::Buffer;
use crate::compute::Device;
use crate::layout::{AABB, Region, V, V3};
use crate::ops::{AsF64s, DeviceAware, Export, Summary, json_array, json_f64};

// ═══════════════════════════════════════════════════════════════════════════
// ── Mesh ─────────────────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

/// Triangle mesh — flat buffer storage for device-friendly dispatch.
///
/// Vertex positions are stored as a flat `Buffer` of `[x0,y0,z0, x1,y1,z1, …]`.
/// This lets operations like transform, bounding-box, and normal recomputation
/// dispatch through Device (SIMD/GPU) on the entire mesh at once.
#[derive(Debug, Clone)]
pub struct Mesh {
    /// Flat vertex positions: len = vertex_count * 3.
    pub vertices: Buffer,
    /// Triangle indices (3 per face). If empty, treat as non-indexed (sequential).
    pub indices: Vec<u32>,
    /// Per-vertex normals: len = vertex_count * 3. Empty = auto-compute on demand.
    pub normals: Buffer,
    /// Per-vertex UVs: len = vertex_count * 2. Empty = no UVs.
    pub uvs: Buffer,
    /// Material index for this mesh.
    pub material: usize,
}

impl Mesh {
    /// Angle at step `i` of `count` around a full circle.
    #[inline]
    fn ring_angle(i: u32, count: u32) -> f64 {
        std::f64::consts::TAU * i as f64 / count as f64
    }

    /// Generate grid indices for a (rows+1) x (cols+1) vertex grid.
    fn grid_indices(rows: u32, cols: u32) -> Vec<u32> {
        let stride = cols + 1;
        let mut idxs = Vec::with_capacity((rows * cols * 6) as usize);
        for r in 0..rows {
            for c in 0..cols {
                let a = r * stride + c;
                let b = a + stride;
                idxs.extend_from_slice(&[a, a + 1, b, a + 1, b + 1, b]);
            }
        }
        idxs
    }

    /// Create a mesh from vertex positions (flat f64 array, 3 per vertex).
    pub fn new(vertices: Vec<f64>) -> Self {
        Self {
            vertices: Buffer::new(vertices),
            indices: Vec::new(),
            normals: Buffer::new(Vec::new()),
            uvs: Buffer::new(Vec::new()),
            material: 0,
        }
    }

    /// Create a mesh on a specific device.
    pub fn on(device: &Device, vertices: Vec<f64>) -> Self {
        Self {
            vertices: Buffer::on(device, vertices),
            indices: Vec::new(),
            normals: Buffer::on(device, Vec::new()),
            uvs: Buffer::on(device, Vec::new()),
            material: 0,
        }
    }

    pub fn with_indices(mut self, indices: Vec<u32>) -> Self {
        self.indices = indices;
        self
    }

    pub fn with_normals(mut self, normals: Vec<f64>) -> Self {
        self.normals = Buffer::new(normals);
        self
    }

    pub fn with_uvs(mut self, uvs: Vec<f64>) -> Self {
        self.uvs = Buffer::new(uvs);
        self
    }

    pub fn with_material(mut self, idx: usize) -> Self {
        self.material = idx;
        self
    }

    /// Number of vertices.
    pub fn vertex_count(&self) -> usize {
        self.vertices.len() / 3
    }

    /// Number of triangles.
    pub fn tri_count(&self) -> usize {
        if self.indices.is_empty() {
            self.vertex_count() / 3
        } else {
            self.indices.len() / 3
        }
    }

    /// Read vertex at index as V<3>.
    pub fn vertex(&self, i: usize) -> V3 {
        let d = self.vertices.data();
        let base = i * 3;
        V([d[base], d[base + 1], d[base + 2]])
    }

    /// Iterate over face triples (index triples into vertex array).
    pub fn faces(&self) -> Vec<[usize; 3]> {
        if self.indices.is_empty() {
            let vc = self.vertex_count();
            (0..vc / 3).map(|f| [f * 3, f * 3 + 1, f * 3 + 2]).collect()
        } else {
            self.indices
                .chunks_exact(3)
                .map(|c| [c[0] as usize, c[1] as usize, c[2] as usize])
                .collect()
        }
    }

    /// Compute axis-aligned bounding box from vertex data.
    pub fn bounds(&self) -> AABB {
        let d = self.vertices.data();
        if d.len() < 3 {
            return AABB::ZERO;
        }
        let mut lo = V([d[0], d[1], d[2]]);
        let mut hi = lo;
        for i in (3..d.len()).step_by(3) {
            let p = V([d[i], d[i + 1], d[i + 2]]);
            lo = lo.comp_min(p);
            hi = hi.comp_max(p);
        }
        Region::from_parts(lo, hi - lo)
    }

    /// Recompute per-vertex normals from face normals (area-weighted).
    ///
    /// This allocates a normals buffer of the same length as vertices.
    pub fn compute_normals(&mut self) {
        let vc = self.vertex_count();
        let mut norms = vec![0.0f64; vc * 3];

        for [i0, i1, i2] in self.faces() {
            let n = (self.vertex(i1) - self.vertex(i0)).cross(self.vertex(i2) - self.vertex(i0));
            for vi in [i0, i1, i2] {
                norms[vi * 3] += n.0[0];
                norms[vi * 3 + 1] += n.0[1];
                norms[vi * 3 + 2] += n.0[2];
            }
        }

        for i in 0..vc {
            let n = V([norms[i * 3], norms[i * 3 + 1], norms[i * 3 + 2]]).normalized();
            norms[i * 3] = n.0[0];
            norms[i * 3 + 1] = n.0[1];
            norms[i * 3 + 2] = n.0[2];
        }

        self.normals = Buffer::new(norms);
    }

    /// Generate a cube mesh with center and half-extent.
    pub fn cube(center: V3, half: f64) -> Self {
        let [cx, cy, cz] = center.0;
        let c = [
            [cx - half, cy - half, cz - half],
            [cx + half, cy - half, cz - half],
            [cx + half, cy + half, cz - half],
            [cx - half, cy + half, cz - half],
            [cx - half, cy - half, cz + half],
            [cx + half, cy - half, cz + half],
            [cx + half, cy + half, cz + half],
            [cx - half, cy + half, cz + half],
        ];
        let verts: Vec<f64> = c.iter().flat_map(|p| p.iter().copied()).collect();
        #[rustfmt::skip]
        let indices = vec![
            0,2,1, 0,3,2, 1,6,5, 1,2,6, 5,7,4, 5,6,7,
            4,3,0, 4,7,3, 3,6,2, 3,7,6, 4,1,5, 4,0,1,
        ];
        Self::new(verts).with_indices(indices)
    }

    /// Generate a UV sphere mesh.
    pub fn sphere(center: V3, radius: f64, rings: u32, sectors: u32) -> Self {
        let [cx, cy, cz] = center.0;
        let mut verts = Vec::new();

        for r in 0..=rings {
            let phi = std::f64::consts::PI * r as f64 / rings as f64;
            for s in 0..=sectors {
                let theta = Self::ring_angle(s, sectors);
                verts.extend_from_slice(&[
                    cx + radius * phi.sin() * theta.cos(),
                    cy + radius * phi.cos(),
                    cz + radius * phi.sin() * theta.sin(),
                ]);
            }
        }

        Self::new(verts).with_indices(Self::grid_indices(rings, sectors))
    }

    /// Generate a cylinder mesh.
    pub fn cylinder(base: V3, radius: f64, height: f64, segments: u32) -> Self {
        let [bx, by, bz] = base.0;
        let mut verts = Vec::new();
        let mut idxs = Vec::new();

        // Center vertices: bottom (0) and top (1)
        verts.extend_from_slice(&[bx, by, bz]);
        verts.extend_from_slice(&[bx, by + height, bz]);

        // Ring vertices: bottom ring then top ring (shared angles)
        let ring_start = 2u32;
        for y_offset in [0.0, height] {
            for i in 0..segments {
                let a = Self::ring_angle(i, segments);
                verts.extend_from_slice(&[
                    bx + radius * a.cos(),
                    by + y_offset,
                    bz + radius * a.sin(),
                ]);
            }
        }
        let top_ring = ring_start + segments;

        for i in 0..segments {
            let next = (i + 1) % segments;
            idxs.extend_from_slice(&[0, ring_start + i, ring_start + next]); // bottom cap
            idxs.extend_from_slice(&[1, top_ring + next, top_ring + i]); // top cap
            let (bl, br, tl, tr) = (
                ring_start + i,
                ring_start + next,
                top_ring + i,
                top_ring + next,
            );
            idxs.extend_from_slice(&[bl, tl, br, tl, tr, br]); // side
        }

        Self::new(verts).with_indices(idxs)
    }

    /// Generate a torus mesh.
    pub fn torus(center: V3, major_r: f64, minor_r: f64, major_seg: u32, minor_seg: u32) -> Self {
        let [cx, cy, cz] = center.0;
        let mut verts = Vec::new();

        for i in 0..=major_seg {
            let theta = Self::ring_angle(i, major_seg);
            for j in 0..=minor_seg {
                let phi = Self::ring_angle(j, minor_seg);
                let r = major_r + minor_r * phi.cos();
                verts.extend_from_slice(&[
                    cx + r * theta.cos(),
                    cy + minor_r * phi.sin(),
                    cz + r * theta.sin(),
                ]);
            }
        }

        Self::new(verts).with_indices(Self::grid_indices(major_seg, minor_seg))
    }

    /// Parse a Wavefront OBJ string into a Mesh.
    ///
    /// Supports: `v` (vertex), `vn` (normal), `vt` (tex coord), `f` (face).
    /// Face indices are 1-based and can be `v`, `v/vt`, `v/vt/vn`, or `v//vn`.
    pub fn from_obj(src: &str) -> Self {
        let mut positions: Vec<[f64; 3]> = Vec::new();
        let mut normals_src: Vec<[f64; 3]> = Vec::new();
        let mut uvs_src: Vec<[f64; 2]> = Vec::new();
        // Collect raw face specs to decide indexed vs expanded after parsing.
        let mut raw_faces: Vec<Vec<(Option<usize>, Option<usize>, Option<usize>)>> = Vec::new();

        for line in src.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut parts = line.split_whitespace();
            let Some(cmd) = parts.next() else { continue };
            match cmd {
                "v" => {
                    let coords: Vec<f64> = parts.filter_map(|s| s.parse().ok()).collect();
                    if coords.len() >= 3 {
                        positions.push([coords[0], coords[1], coords[2]]);
                    }
                }
                "vn" => {
                    let coords: Vec<f64> = parts.filter_map(|s| s.parse().ok()).collect();
                    if coords.len() >= 3 {
                        normals_src.push([coords[0], coords[1], coords[2]]);
                    }
                }
                "vt" => {
                    let coords: Vec<f64> = parts.filter_map(|s| s.parse().ok()).collect();
                    if coords.len() >= 2 {
                        uvs_src.push([coords[0], coords[1]]);
                    }
                }
                "f" => {
                    let face: Vec<_> = parts.map(|s| Self::parse_face_vertex(s)).collect();
                    raw_faces.push(face);
                }
                _ => {}
            }
        }

        let has_attribs = !normals_src.is_empty() || !uvs_src.is_empty();

        if !has_attribs {
            // Simple indexed path: share positions, build index buffer.
            let verts: Vec<f64> = positions.iter().flat_map(|p| p.iter().copied()).collect();
            let mut indices = Vec::new();
            for face in &raw_faces {
                for i in 1..face.len().saturating_sub(1) {
                    for &fi in &[0, i, i + 1] {
                        if let Some(vi) = face[fi].0 {
                            indices.push(vi as u32);
                        }
                    }
                }
            }
            let mut mesh = Self::new(verts).with_indices(indices);
            mesh.compute_normals();
            mesh
        } else {
            // Expanded path: per-face vertices with normals/UVs.
            let mut verts = Vec::new();
            let mut norm_buf = Vec::new();
            let mut uv_buf = Vec::new();
            for face in &raw_faces {
                for i in 1..face.len().saturating_sub(1) {
                    for &fi in &[0, i, i + 1] {
                        let (vi, vti, vni) = face[fi];
                        if let Some(pos) = vi.and_then(|i| positions.get(i)) {
                            verts.extend_from_slice(pos);
                        }
                        if let Some(n) = vni.and_then(|i| normals_src.get(i)) {
                            norm_buf.extend_from_slice(n);
                        }
                        if let Some(uv) = vti.and_then(|i| uvs_src.get(i)) {
                            uv_buf.extend_from_slice(uv);
                        }
                    }
                }
            }
            let mut mesh = Self::new(verts);
            if !norm_buf.is_empty() {
                mesh.normals = norm_buf.into();
            } else {
                mesh.compute_normals();
            }
            if !uv_buf.is_empty() {
                mesh.uvs = uv_buf.into();
            }
            mesh
        }
    }

    /// Parse a face vertex spec like "1/2/3", "1//3", "1/2", or "1".
    /// Returns (position_idx, tex_idx, normal_idx) — all 0-based.
    fn parse_face_vertex(s: &str) -> (Option<usize>, Option<usize>, Option<usize>) {
        let parts: Vec<&str> = s.split('/').collect();
        let vi = parts
            .first()
            .and_then(|s| s.parse::<usize>().ok())
            .map(|i| i - 1);
        let vti = parts.get(1).and_then(|s| {
            if s.is_empty() {
                None
            } else {
                s.parse::<usize>().ok().map(|i| i - 1)
            }
        });
        let vni = parts
            .get(2)
            .and_then(|s| s.parse::<usize>().ok())
            .map(|i| i - 1);
        (vi, vti, vni)
    }
}

// ── Basic physics ───────────────────────────────────────────────────────────

/// Rigid body state for simple physics simulation.
#[derive(Debug, Clone, Copy)]
pub struct RigidBody {
    pub position: V3,
    pub velocity: V3,
    pub mass: f64,
    pub restitution: f64,
}

impl Default for RigidBody {
    fn default() -> Self {
        Self {
            position: V3::ZERO,
            velocity: V3::ZERO,
            mass: 1.0,
            restitution: 0.5,
        }
    }
}

impl RigidBody {
    /// Step physics by dt seconds with gravity.
    pub fn step(&mut self, dt: f64, gravity: V3) {
        self.velocity = self.velocity + gravity * dt;
        self.position = self.position + self.velocity * dt;
    }

    /// Bounce off a horizontal plane at y = plane_y.
    /// Bodies with tiny velocity at the plane are put to rest.
    pub fn bounce_plane(&mut self, plane_y: f64) {
        if self.position.0[1] < plane_y {
            self.position.0[1] = plane_y;
            self.velocity.0[1] = -self.velocity.0[1] * self.restitution;
            // Sleep: if bounce velocity is negligible, rest on the plane
            if self.velocity.0[1].abs() < 0.05 {
                self.velocity.0[1] = 0.0;
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ── Ops trait impls ─────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

impl AsF64s for Mesh {
    fn as_f64s(&self) -> &[f64] {
        self.vertices.data()
    }
}

impl Summary for Mesh {
    fn summary(&self) -> String {
        let b = self.bounds();
        format!(
            "Mesh(verts={}, tris={}, bounds=[{:.1}..{:.1}])",
            self.vertex_count(),
            self.tri_count(),
            b.origin,
            b.origin + b.size
        )
    }
}

impl DeviceAware for Mesh {
    fn to_device(&self, device: &crate::compute::Device) -> Self {
        Self {
            vertices: Buffer::on(device, self.vertices.data().to_vec()),
            indices: self.indices.clone(),
            normals: Buffer::on(device, self.normals.data().to_vec()),
            uvs: Buffer::on(device, self.uvs.data().to_vec()),
            material: self.material,
        }
    }
    fn device_name(&self) -> String {
        self.vertices.device().name().to_string()
    }
}

impl Export for Mesh {
    fn to_json(&self) -> String {
        let d = self.vertices.data();
        let verts = json_array(d.iter().map(|v| json_f64(*v)));
        let idxs = json_array(self.indices.iter().map(|i| i.to_string()));
        format!(
            "{{\"vertex_count\":{},\"tri_count\":{},\"vertices\":{},\"indices\":{}}}",
            self.vertex_count(),
            self.tri_count(),
            verts,
            idxs
        )
    }

    fn to_csv(&self) -> String {
        self.to_obj()
    }
}

impl Mesh {
    /// Export to Wavefront OBJ string.
    pub fn to_obj(&self) -> String {
        let d = self.vertices.data();
        let mut out = String::new();
        for chunk in d.chunks_exact(3) {
            out.push_str(&format!("v {} {} {}\n", chunk[0], chunk[1], chunk[2]));
        }
        for face in self.faces() {
            out.push_str(&format!(
                "f {} {} {}\n",
                face[0] + 1,
                face[1] + 1,
                face[2] + 1
            ));
        }
        out
    }
}
