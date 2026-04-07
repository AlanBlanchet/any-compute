use super::camera::Camera;
use super::lighting::{Light, Material, shade_face};
use super::mesh::Mesh;
use crate::Lerp;
use crate::layout::{AABB, Region, V, V3, V4};
use crate::render::Color;

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
        let scaled = V([
            p.0[0] * self.scale.0[0],
            p.0[1] * self.scale.0[1],
            p.0[2] * self.scale.0[2],
        ]);
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
        self.objects.push(SceneObject {
            mesh,
            transform: Transform::default(),
        });
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

    /// Shade a face using this scene's lights — convenience for `shade_face()`.
    pub fn shade_face(&self, normal: V3, base: Color) -> Color {
        shade_face(normal, base, &self.lights)
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
            let corners = b.corners();
            for c in &corners {
                let tc = obj.transform.apply(*c);
                lo = lo.comp_min(tc);
                hi = hi.comp_max(tc);
            }
        }
        Region::from_parts(lo, hi - lo)
    }
}

impl crate::visual::Graphable for Scene {
    fn to_graph(&self) -> crate::visual::VisualGraph<2> {
        use crate::visual::{VNode, VisualGraph};
        let mut g = VisualGraph::new("Scene");
        let scene_node = g.add(VNode::new(V([0.0, 0.0]), "Scene", "custom"));
        for (i, obj) in self.objects.iter().enumerate() {
            let label = format!("Object {} ({}v)", i, obj.mesh.vertex_count());
            let idx = g.add(VNode::new(V([0.0, 0.0]), &label, "input"));
            g.edge(idx, scene_node);
        }
        for (i, light) in self.lights.iter().enumerate() {
            let label = match light {
                Light::Directional { .. } => format!("Dir Light {i}"),
                Light::Point { .. } => format!("Point Light {i}"),
                Light::Ambient { .. } => format!("Ambient {i}"),
            };
            let idx = g.add(VNode::new(V([0.0, 0.0]), &label, "constant"));
            g.edge(idx, scene_node);
        }
        let cam = g.add(VNode::new(V([0.0, 0.0]), "Camera", "output"));
        g.edge(scene_node, cam);
        g.auto_layout();
        g
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ── Ops trait impls ─────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

impl crate::ops::Summary for Scene {
    fn summary(&self) -> String {
        let total_tris: usize = self.objects.iter().map(|o| o.mesh.tri_count()).sum();
        format!(
            "Scene(objects={}, tris={}, lights={}, materials={})",
            self.objects.len(),
            total_tris,
            self.lights.len(),
            self.materials.len()
        )
    }
}
