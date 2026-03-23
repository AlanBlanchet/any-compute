//! 3D scene graph — meshes, cameras, lights, materials.
//!
//! Built entirely on [`V<3>`](crate::layout::V) and [`Buffer`](crate::buffer::Buffer),
//! so all heavy computation (normals, transforms, ray intersections) dispatches
//! through [`Device`](crate::compute::Device) automatically.
//!
//! ## Design
//!
//! - `Mesh` stores geometry as flat `Buffer`s (vertices, normals, uvs) + index buffer.
//!   All per-vertex operations (transform, normal recomputation) batch through Device.
//! - `Camera` and `Light` are value types built on `V<3>`.
//! - `Material` describes surface appearance — referenced by index, not embedded.
//! - `Scene` is a flat collection (not a tree) — keeps dispatch simple.

use crate::buffer::Buffer;
use crate::compute::Device;
use crate::layout::{Region, V, V3, V4, AABB};
use crate::Lerp;

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
        let verts = self.vertices.data();
        let vc = self.vertex_count();
        let mut norms = vec![0.0f64; vc * 3];

        let face_iter: Box<dyn Iterator<Item = [usize; 3]>> = if self.indices.is_empty() {
            Box::new((0..vc / 3).map(|f| [f * 3, f * 3 + 1, f * 3 + 2]))
        } else {
            Box::new(self.indices.chunks_exact(3).map(|c| [c[0] as usize, c[1] as usize, c[2] as usize]))
        };

        for [i0, i1, i2] in face_iter {
            let a = V([verts[i0 * 3], verts[i0 * 3 + 1], verts[i0 * 3 + 2]]);
            let b = V([verts[i1 * 3], verts[i1 * 3 + 1], verts[i1 * 3 + 2]]);
            let c = V([verts[i2 * 3], verts[i2 * 3 + 1], verts[i2 * 3 + 2]]);
            let n = (b - a).cross(c - a); // area-weighted (not normalized)
            for vi in [i0, i1, i2] {
                norms[vi * 3] += n.0[0];
                norms[vi * 3 + 1] += n.0[1];
                norms[vi * 3 + 2] += n.0[2];
            }
        }

        // Normalize each accumulated normal.
        for i in 0..vc {
            let n = V([norms[i * 3], norms[i * 3 + 1], norms[i * 3 + 2]]).normalized();
            norms[i * 3] = n.0[0];
            norms[i * 3 + 1] = n.0[1];
            norms[i * 3 + 2] = n.0[2];
        }

        self.normals = Buffer::new(norms);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ── Camera ───────────────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

/// Projection mode.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Projection {
    /// Field of view in radians, aspect ratio.
    Perspective { fov: f64, aspect: f64, near: f64, far: f64 },
    /// Orthographic half-extents.
    Orthographic { width: f64, height: f64, near: f64, far: f64 },
}

/// Camera — position, target, up + projection.
#[derive(Debug, Clone)]
pub struct Camera {
    pub eye: V3,
    pub target: V3,
    pub up: V3,
    pub projection: Projection,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            eye: V([0.0, 0.0, 5.0]),
            target: V3::ZERO,
            up: V([0.0, 1.0, 0.0]),
            projection: Projection::Perspective {
                fov: std::f64::consts::FRAC_PI_4,
                aspect: 16.0 / 9.0,
                near: 0.1,
                far: 1000.0,
            },
        }
    }
}

impl Camera {
    /// View direction (normalized).
    pub fn forward(&self) -> V3 {
        (self.target - self.eye).normalized()
    }

    /// Right vector (normalized).
    pub fn right(&self) -> V3 {
        self.forward().cross(self.up).normalized()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ── Light ────────────────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

/// Light source.
#[derive(Debug, Clone)]
pub enum Light {
    /// Direction toward the light (infinite distance).
    Directional { direction: V3, color: V3, intensity: f64 },
    /// Point light with position and falloff.
    Point { position: V3, color: V3, intensity: f64, range: f64 },
    /// Ambient light (uniform everywhere).
    Ambient { color: V3, intensity: f64 },
}

// ═══════════════════════════════════════════════════════════════════════════
// ── Material ─────────────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

/// PBR-style material.
#[derive(Debug, Clone)]
pub struct Material {
    /// Base color (linear RGB, 0..1).
    pub albedo: V3,
    /// 0 = dielectric, 1 = metal.
    pub metallic: f64,
    /// 0 = mirror, 1 = rough.
    pub roughness: f64,
    /// Emission color (linear RGB, 0..1). Zero = no emission.
    pub emissive: V3,
    /// Index of refraction (1.0 = air, 1.5 = glass).
    pub ior: f64,
    /// Opacity (1.0 = opaque, 0.0 = fully transparent).
    pub alpha: f64,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            albedo: V([0.8, 0.8, 0.8]),
            metallic: 0.0,
            roughness: 0.5,
            emissive: V3::ZERO,
            ior: 1.5,
            alpha: 1.0,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ── Transform ────────────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

/// Affine transform: position + rotation (quaternion V4) + scale.
#[derive(Debug, Clone, Copy)]
pub struct Transform {
    pub position: V3,
    /// Rotation as unit quaternion (x, y, z, w).
    pub rotation: V4,
    pub scale: V3,
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            position: V3::ZERO,
            rotation: V([0.0, 0.0, 0.0, 1.0]), // identity quaternion
            scale: V([1.0, 1.0, 1.0]),
        }
    }
}

impl Lerp for Transform {
    fn lerp(self, other: Self, t: f64) -> Self {
        Self {
            position: self.position.lerp(other.position, t),
            rotation: self.rotation.lerp(other.rotation, t).normalized(),
            scale: self.scale.lerp(other.scale, t),
        }
    }
}

impl Transform {
    /// Apply this transform to a 3D point.
    pub fn apply(&self, p: V3) -> V3 {
        let scaled = V([p.0[0] * self.scale.0[0], p.0[1] * self.scale.0[1], p.0[2] * self.scale.0[2]]);
        self.rotate(scaled) + self.position
    }

    /// Rotate a vector by the quaternion.
    pub fn rotate(&self, v: V3) -> V3 {
        let q = self.rotation;
        let u = V([q.0[0], q.0[1], q.0[2]]); // vector part
        let s = q.0[3]; // scalar part
        // Rodrigues via quaternion: v' = 2(u·v)u + (s²-u·u)v + 2s(u×v)
        let uv = u.dot(v);
        let uu = u.dot(u);
        u * (2.0 * uv) + v * (s * s - uu) + u.cross(v) * (2.0 * s)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ── SceneObject + Scene ──────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

/// An object in the scene: mesh + transform.
#[derive(Debug, Clone)]
pub struct SceneObject {
    pub mesh: Mesh,
    pub transform: Transform,
}

/// Flat scene — objects, materials, lights, camera.
///
/// No recursion, no tree. The flat list keeps Device dispatch simple:
/// iterate objects, transform vertices, collect render commands.
#[derive(Debug, Clone)]
pub struct Scene {
    pub objects: Vec<SceneObject>,
    pub materials: Vec<Material>,
    pub lights: Vec<Light>,
    pub camera: Camera,
}

impl Default for Scene {
    fn default() -> Self {
        Self {
            objects: Vec::new(),
            materials: vec![Material::default()],
            lights: vec![Light::Ambient {
                color: V([1.0, 1.0, 1.0]),
                intensity: 0.3,
            }],
            camera: Camera::default(),
        }
    }
}

impl Scene {
    pub fn add(&mut self, mesh: Mesh) -> usize {
        let idx = self.objects.len();
        self.objects.push(SceneObject { mesh, transform: Transform::default() });
        idx
    }

    pub fn add_transformed(&mut self, mesh: Mesh, transform: Transform) -> usize {
        let idx = self.objects.len();
        self.objects.push(SceneObject { mesh, transform });
        idx
    }

    pub fn add_material(&mut self, mat: Material) -> usize {
        let idx = self.materials.len();
        self.materials.push(mat);
        idx
    }

    pub fn add_light(&mut self, light: Light) {
        self.lights.push(light);
    }

    /// World-space bounding box of all objects.
    pub fn bounds(&self) -> AABB {
        if self.objects.is_empty() {
            return AABB::ZERO;
        }
        let mut lo = V([f64::INFINITY; 3]);
        let mut hi = V([f64::NEG_INFINITY; 3]);
        for obj in &self.objects {
            let b = obj.mesh.bounds();
            // Transform the 8 AABB corners and expand.
            let corners = aabb_corners(&b);
            for c in &corners {
                let tc = obj.transform.apply(*c);
                lo = lo.comp_min(tc);
                hi = hi.comp_max(tc);
            }
        }
        Region::from_parts(lo, hi - lo)
    }
}

/// 8 corners of a 3D AABB.
fn aabb_corners(b: &AABB) -> [V3; 8] {
    let o = b.origin;
    let s = b.size;
    [
        o,
        o + V([s.0[0], 0.0, 0.0]),
        o + V([0.0, s.0[1], 0.0]),
        o + V([0.0, 0.0, s.0[2]]),
        o + V([s.0[0], s.0[1], 0.0]),
        o + V([s.0[0], 0.0, s.0[2]]),
        o + V([0.0, s.0[1], s.0[2]]),
        o + s,
    ]
}

// ═══════════════════════════════════════════════════════════════════════════
// ── Ray ──────────────────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

/// Ray for intersection testing.
#[derive(Debug, Clone, Copy)]
pub struct Ray {
    pub origin: V3,
    pub direction: V3,
}

impl Ray {
    pub fn new(origin: V3, direction: V3) -> Self {
        Self { origin, direction: direction.normalized() }
    }

    /// Point along the ray at parameter t.
    pub fn at(&self, t: f64) -> V3 {
        self.origin + self.direction * t
    }

    /// Ray–AABB intersection (slab method). Returns Some(t_near) or None.
    pub fn intersect_aabb(&self, aabb: &AABB) -> Option<f64> {
        let inv = V([1.0 / self.direction.0[0], 1.0 / self.direction.0[1], 1.0 / self.direction.0[2]]);
        let t0 = (aabb.origin - self.origin).zip(inv, |a, b| a * b);
        let t1 = (aabb.end() - self.origin).zip(inv, |a, b| a * b);
        let tmin_v = t0.comp_min(t1);
        let tmax_v = t0.comp_max(t1);
        let tmin = tmin_v.0[0].max(tmin_v.0[1]).max(tmin_v.0[2]);
        let tmax = tmax_v.0[0].min(tmax_v.0[1]).min(tmax_v.0[2]);
        if tmax >= tmin.max(0.0) { Some(tmin.max(0.0)) } else { None }
    }

    /// Ray–triangle intersection (Möller–Trumbore). Returns Some(t) or None.
    pub fn intersect_tri(&self, a: V3, b: V3, c: V3) -> Option<f64> {
        let edge1 = b - a;
        let edge2 = c - a;
        let h = self.direction.cross(edge2);
        let det = edge1.dot(h);
        if det.abs() < 1e-12 {
            return None;
        }
        let inv_det = 1.0 / det;
        let s = self.origin - a;
        let u = s.dot(h) * inv_det;
        if !(0.0..=1.0).contains(&u) {
            return None;
        }
        let q = s.cross(edge1);
        let v = self.direction.dot(q) * inv_det;
        if v < 0.0 || u + v > 1.0 {
            return None;
        }
        let t = edge2.dot(q) * inv_det;
        if t > 1e-8 { Some(t) } else { None }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ── Tests ────────────────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn tri_mesh() -> Mesh {
        // Single triangle: (0,0,0), (1,0,0), (0,1,0)
        Mesh::new(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    }

    #[test]
    fn mesh_vertex_count() {
        let m = tri_mesh();
        assert_eq!(m.vertex_count(), 3);
        assert_eq!(m.tri_count(), 1);
    }

    #[test]
    fn mesh_vertex_access() {
        let m = tri_mesh();
        assert_eq!(m.vertex(0), V([0.0, 0.0, 0.0]));
        assert_eq!(m.vertex(1), V([1.0, 0.0, 0.0]));
    }

    #[test]
    fn mesh_bounds() {
        let m = tri_mesh();
        let b = m.bounds();
        assert_eq!(b.origin, V([0.0, 0.0, 0.0]));
        assert_eq!(b.size, V([1.0, 1.0, 0.0]));
    }

    #[test]
    fn mesh_compute_normals() {
        let mut m = tri_mesh();
        m.compute_normals();
        assert_eq!(m.normals.len(), 9); // 3 verts × 3 components
        // All vertex normals for a flat XY triangle should point in +Z.
        let n = V([m.normals.data()[0], m.normals.data()[1], m.normals.data()[2]]);
        assert!((n.0[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn camera_forward_right() {
        let cam = Camera::default();
        let fwd = cam.forward();
        assert!((fwd.0[2] - (-1.0)).abs() < 1e-10);
        let r = cam.right();
        assert!((r.0[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn transform_identity() {
        let t = Transform::default();
        let p = V([3.0, 4.0, 5.0]);
        let tp = t.apply(p);
        assert!((tp.0[0] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn transform_translate() {
        let t = Transform { position: V([10.0, 0.0, 0.0]), ..Transform::default() };
        let p = V([1.0, 2.0, 3.0]);
        let tp = t.apply(p);
        assert!((tp.0[0] - 11.0).abs() < 1e-10);
    }

    #[test]
    fn scene_default() {
        let s = Scene::default();
        assert!(s.objects.is_empty());
        assert_eq!(s.materials.len(), 1);
        assert_eq!(s.lights.len(), 1);
    }

    #[test]
    fn scene_add_mesh() {
        let mut s = Scene::default();
        let idx = s.add(tri_mesh());
        assert_eq!(idx, 0);
        assert_eq!(s.objects.len(), 1);
    }

    #[test]
    fn ray_aabb_hit() {
        let aabb = Region::new3(0.0, 0.0, 0.0, 2.0, 2.0, 2.0);
        let ray = Ray::new(V([1.0, 1.0, -5.0]), V([0.0, 0.0, 1.0]));
        assert!(ray.intersect_aabb(&aabb).is_some());
    }

    #[test]
    fn ray_aabb_miss() {
        let aabb = Region::new3(0.0, 0.0, 0.0, 2.0, 2.0, 2.0);
        let ray = Ray::new(V([5.0, 5.0, -5.0]), V([0.0, 0.0, 1.0]));
        assert!(ray.intersect_aabb(&aabb).is_none());
    }

    #[test]
    fn ray_tri_hit() {
        let a = V([0.0, 0.0, 0.0]);
        let b = V([2.0, 0.0, 0.0]);
        let c = V([0.0, 2.0, 0.0]);
        let ray = Ray::new(V([0.5, 0.5, -1.0]), V([0.0, 0.0, 1.0]));
        let t = ray.intersect_tri(a, b, c);
        assert!(t.is_some());
        assert!((t.unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn ray_tri_miss() {
        let a = V([0.0, 0.0, 0.0]);
        let b = V([1.0, 0.0, 0.0]);
        let c = V([0.0, 1.0, 0.0]);
        let ray = Ray::new(V([5.0, 5.0, -1.0]), V([0.0, 0.0, 1.0]));
        assert!(ray.intersect_tri(a, b, c).is_none());
    }

    #[test]
    fn transform_lerp() {
        let a = Transform::default();
        let b = Transform { position: V([10.0, 0.0, 0.0]), ..Transform::default() };
        let mid = a.lerp(b, 0.5);
        assert!((mid.position.0[0] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn aabb_intersects() {
        let a = Region::new3(0.0, 0.0, 0.0, 2.0, 2.0, 2.0);
        let b = Region::new3(1.0, 1.0, 1.0, 2.0, 2.0, 2.0);
        assert!(a.intersects(&b));
        let c = Region::new3(5.0, 5.0, 5.0, 1.0, 1.0, 1.0);
        assert!(!a.intersects(&c));
    }
}
