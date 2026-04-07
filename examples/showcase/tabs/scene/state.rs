use super::*;
// ── Colors (semantic aliases for theme palette) ─────────────────────────

pub(super) const C_GROUND: Color = Color::rgb(88, 91, 112);
pub(super) const C_CUBE: Color = theme::BLUE;
pub(super) const C_SPHERE: Color = theme::RED;
pub(super) const C_CYLINDER: Color = theme::GREEN;
pub(super) const C_TORUS: Color = theme::MAUVE;
pub(super) const C_AABB: Color = theme::YELLOW;

// ── View dimensions ─────────────────────────────────────────────────────

pub(super) const VP_W: f64 = 700.0;
pub(super) const VP_H: f64 = 460.0;

// ── Scene info ──────────────────────────────────────────────────────────

/// Pre-computed scene + interactive state.
#[derive(Debug, Clone)]
pub struct SceneInfo {
    pub scene: Scene,
    pub bodies: Vec<RigidBody>,
    pub mesh_colors: Vec<Color>,
    pub drag_active: bool,
    pub drag_last: (f64, f64),
    /// Raw primitives for the 3D viewport — built per frame, emitted after layout.
    pub viewport_prims: RenderList,
    /// Viewport pixel dimensions (updated after layout from the scene-viewport node).
    pub vp_size: (f64, f64),
    /// Currently selected object index (None = nothing selected).
    pub selected: Option<usize>,
}

impl Default for SceneInfo {
    fn default() -> Self {
        let mut scene = Scene::default();

        // Materials
        let mat_ground = scene.add_material(Material {
            albedo: V([0.3, 0.3, 0.4]),
            roughness: 0.9,
            ..Default::default()
        });
        let mat_cube = scene.add_material(Material {
            albedo: V([0.5, 0.7, 1.0]),
            metallic: 0.6,
            roughness: 0.2,
            ..Default::default()
        });
        let mat_sphere = scene.add_material(Material {
            albedo: V([0.95, 0.5, 0.65]),
            metallic: 1.0,
            roughness: 0.1,
            emissive: V([0.1, 0.0, 0.0]),
            ..Default::default()
        });
        let mat_cyl = scene.add_material(Material {
            albedo: V([0.6, 0.9, 0.6]),
            roughness: 0.4,
            ..Default::default()
        });
        let mat_torus = scene.add_material(Material {
            albedo: V([0.8, 0.6, 0.95]),
            roughness: 0.3,
            metallic: 0.4,
            ..Default::default()
        });

        // Ground plane (double-sided so it's visible from below too)
        let ground = Mesh::new(vec![
            // top face
            -6.0, 0.0, -6.0, 6.0, 0.0, -6.0, 6.0, 0.0, 6.0, -6.0, 0.0, -6.0, 6.0, 0.0, 6.0, -6.0,
            0.0, 6.0, // bottom face (reversed winding)
            -6.0, 0.0, -6.0, 6.0, 0.0, 6.0, 6.0, 0.0, -6.0, -6.0, 0.0, -6.0, -6.0, 0.0, 6.0, 6.0,
            0.0, 6.0,
        ])
        .with_material(mat_ground);
        scene.add(ground);

        // Cube
        let mut cube = Mesh::cube(V([0.0, 1.0, 0.0]), 0.5);
        cube.material = mat_cube;
        cube.compute_normals();
        scene.add(cube);

        // Sphere (UV sphere for smooth faces)
        let mut sphere = Mesh::sphere(V([-2.5, 1.0, 0.0]), 0.8, 12, 16);
        sphere.material = mat_sphere;
        sphere.compute_normals();
        scene.add(sphere);

        // Cylinder
        let mut cyl = Mesh::cylinder(V([3.0, 0.0, 0.0]), 0.6, 2.0, 16);
        cyl.material = mat_cyl;
        cyl.compute_normals();
        scene.add(cyl);

        // Torus
        let mut torus = Mesh::torus(V([0.0, 2.5, -2.5]), 0.8, 0.3, 16, 8);
        torus.material = mat_torus;
        torus.compute_normals();
        scene.add(torus);

        // Lights
        scene.add_light(Light::Directional {
            direction: V([0.5, -1.0, -0.3]).normalized(),
            color: V([1.0, 0.95, 0.9]),
            intensity: 1.0,
        });
        scene.add_light(Light::Point {
            position: V([-2.0, 3.0, 2.0]),
            color: V([0.6, 0.8, 1.0]),
            intensity: 2.0,
            range: 10.0,
        });
        scene.add_light(Light::Ambient {
            color: V([1.0, 1.0, 1.0]),
            intensity: 0.15,
        });

        // Camera
        scene.camera = Camera {
            eye: V([4.0, 4.0, 8.0]),
            target: V([0.0, 0.8, 0.0]),
            up: V([0.0, 1.0, 0.0]),
            projection: Projection::Perspective {
                fov: std::f64::consts::FRAC_PI_4,
                aspect: VP_W / VP_H,
                near: 0.1,
                far: 100.0,
            },
        };

        // Physics bodies (mapped to scene objects: cube=1, sphere=2)
        let mesh_colors = vec![C_GROUND, C_CUBE, C_SPHERE, C_CYLINDER, C_TORUS];
        let bodies = vec![
            RigidBody {
                position: V([0.0, 1.0, 0.0]),
                velocity: V([0.0, 3.0, 0.0]),
                mass: 1.0,
                restitution: 0.6,
            },
            RigidBody {
                position: V([-2.5, 1.0, 0.0]),
                velocity: V([0.0, 2.0, 0.0]),
                mass: 0.5,
                restitution: 0.8,
            },
        ];

        SceneInfo {
            scene,
            bodies,
            mesh_colors,
            drag_active: false,
            drag_last: (0.0, 0.0),
            viewport_prims: RenderList::default(),
            vp_size: (VP_W, VP_H),
            selected: None,
        }
    }
}

impl SceneInfo {
    /// Step physics simulation by dt.
    pub fn tick(&mut self, dt: f64) {
        let gravity = V([0.0, -9.8, 0.0]);
        for body in &mut self.bodies {
            body.step(dt, gravity);
            body.bounce_plane(0.0);
        }
        // Sync physics positions to scene objects (cube=1, sphere=2)
        let mapping = [(1usize, 0usize), (2, 1)];
        for &(obj_idx, body_idx) in &mapping {
            if let (Some(body), Some(obj)) = (
                self.bodies.get(body_idx),
                self.scene.objects.get_mut(obj_idx),
            ) {
                obj.transform.position = body.position;
            }
        }
    }

    /// Handle mouse drag for camera orbit.
    pub fn on_drag(&mut self, dx: f64, dy: f64) {
        self.scene.camera.orbit(dx * 0.005, -dy * 0.005);
    }

    /// Handle scroll for camera zoom.
    pub fn on_zoom(&mut self, delta: f64) {
        self.scene.camera.zoom(if delta > 0.0 { 0.9 } else { 1.1 });
    }
}

// ── Face rendering ──────────────────────────────────────────────────────

pub(super) struct ProjectedFace {
    pub(super) obj: usize,
    pub(super) face: usize,
    pub(super) depth: f64,
    /// Screen-space triangle vertices.
    pub(super) v: [(f64, f64); 3],
    pub(super) color: Color,
}

// ── Viewport primitive helpers ──────────────────────────────────────────

impl SceneInfo {
    pub(super) fn push_aabb_wireframe(
        &self,
        list: &mut RenderList,
        vp: &impl Viewport<V<3>>,
        aabb: &any_compute_core::layout::AABB,
        xf: &Transform,
        color: Color,
    ) {
        let c = aabb.corners();
        let edges = [
            [0, 1],
            [1, 4],
            [4, 2],
            [2, 0],
            [3, 5],
            [5, 6],
            [6, 7],
            [7, 3],
            [0, 3],
            [1, 5],
            [4, 6],
            [2, 7],
        ];
        for [a, b] in edges {
            list.push_projected_line(vp, xf.apply(c[a]), xf.apply(c[b]), color, 1.0);
        }
    }
}

// ── Object names for editor ─────────────────────────────────────────────

pub(super) const OBJECT_NAMES: &[&str] = &["Ground", "Cube", "Sphere", "Cylinder", "Torus"];
pub(super) const OBJECT_COLORS: &[Color] = &[C_GROUND, C_CUBE, C_SPHERE, C_CYLINDER, C_TORUS];
