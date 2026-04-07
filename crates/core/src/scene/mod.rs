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

mod camera;
mod lighting;
mod mesh;
mod ray;
mod transform;

pub use camera::*;
pub use lighting::*;
pub use mesh::*;
pub use ray::*;
pub use transform::*;

// ═══════════════════════════════════════════════════════════════════════════
// ── Tests ────────────────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Lerp;
    use crate::layout::{Region, V, V3};

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
        let n = V([
            m.normals.data()[0],
            m.normals.data()[1],
            m.normals.data()[2],
        ]);
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
        let t = Transform {
            position: V([10.0, 0.0, 0.0]),
            ..Transform::default()
        };
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
        let b = Transform {
            position: V([10.0, 0.0, 0.0]),
            ..Transform::default()
        };
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

    #[test]
    fn obj_parse_cube() {
        let obj = "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3\nf 1 3 4\n";
        let m = Mesh::from_obj(obj);
        assert_eq!(m.vertex_count(), 4);
        assert_eq!(m.tri_count(), 2);
        assert_eq!(m.indices.len(), 6);
    }

    #[test]
    fn obj_parse_face_vtn() {
        // OBJ format with vertex/texture/normal indices
        let obj = "v 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nvn 0 0 1\nf 1/1/1 2/1/1 3/1/1\n";
        let m = Mesh::from_obj(obj);
        assert_eq!(m.vertex_count(), 3);
        assert_eq!(m.tri_count(), 1);
    }

    #[test]
    fn obj_parse_polygon_fan() {
        // Quad face should be triangulated
        let obj = "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n";
        let m = Mesh::from_obj(obj);
        assert_eq!(m.tri_count(), 2); // quad → 2 triangles
    }

    #[test]
    fn sphere_mesh() {
        let s = Mesh::sphere(V([0.0, 0.0, 0.0]), 1.0, 8, 12);
        assert!(s.vertex_count() > 50);
        assert!(s.tri_count() > 80);
        let b = s.bounds();
        assert!((b.origin.0[0] + 1.0).abs() < 0.3);
    }

    #[test]
    fn cylinder_mesh() {
        let c = Mesh::cylinder(V([0.0, 0.0, 0.0]), 1.0, 2.0, 12);
        assert!(c.vertex_count() > 20);
        assert!(c.tri_count() > 20);
    }

    #[test]
    fn torus_mesh() {
        let t = Mesh::torus(V([0.0, 0.0, 0.0]), 2.0, 0.5, 12, 6);
        assert!(t.vertex_count() > 50);
        assert!(t.tri_count() > 100);
    }

    #[test]
    fn camera_project_visible() {
        let cam = Camera {
            eye: V([0.0, 0.0, 5.0]),
            target: V3::ZERO,
            up: V([0.0, 1.0, 0.0]),
            projection: Projection::Perspective {
                fov: std::f64::consts::FRAC_PI_4,
                aspect: 1.0,
                near: 0.1,
                far: 100.0,
            },
        };
        // Origin should be visible (ahead of camera)
        let ndc = cam.project(V3::ZERO);
        assert!(ndc.is_some());
        let ndc = ndc.unwrap();
        assert!((ndc.0[0]).abs() < 0.01);
        assert!((ndc.0[1]).abs() < 0.01);
    }

    #[test]
    fn camera_project_behind() {
        let cam = Camera::default(); // looks at origin from z=5
        // Point behind the camera
        let ndc = cam.project(V([0.0, 0.0, 10.0]));
        assert!(ndc.is_none());
    }

    #[test]
    fn camera_project_to_screen() {
        let cam = Camera {
            eye: V([0.0, 0.0, 5.0]),
            target: V3::ZERO,
            up: V([0.0, 1.0, 0.0]),
            projection: Projection::Perspective {
                fov: std::f64::consts::FRAC_PI_4,
                aspect: 1.0,
                near: 0.1,
                far: 100.0,
            },
        };
        let screen = cam.project_to_screen(V3::ZERO, 800.0, 600.0);
        assert!(screen.is_some());
        let (sx, sy, _) = screen.unwrap();
        assert!((sx - 400.0).abs() < 10.0);
        assert!((sy - 300.0).abs() < 10.0);
    }

    #[test]
    fn camera_orbit() {
        let mut cam = Camera {
            eye: V([0.0, 0.0, 5.0]),
            target: V3::ZERO,
            up: V([0.0, 1.0, 0.0]),
            projection: Projection::Perspective {
                fov: std::f64::consts::FRAC_PI_4,
                aspect: 1.0,
                near: 0.1,
                far: 100.0,
            },
        };
        let dist_before = (cam.eye - cam.target).magnitude();
        cam.orbit(0.1, 0.0);
        let dist_after = (cam.eye - cam.target).magnitude();
        // Distance should be preserved
        assert!((dist_before - dist_after).abs() < 0.1);
        // Eye should have moved
        assert!(cam.eye.0[0].abs() > 0.01);
    }

    #[test]
    fn camera_orbit_identity() {
        // Zero rotation must NOT move the eye.
        let mut cam = Camera {
            eye: V([4.0, 4.0, 8.0]),
            target: V([0.0, 0.8, 0.0]),
            up: V([0.0, 1.0, 0.0]),
            projection: Projection::Perspective {
                fov: std::f64::consts::FRAC_PI_4,
                aspect: 16.0 / 9.0,
                near: 0.1,
                far: 100.0,
            },
        };
        let eye_before = cam.eye;
        cam.orbit(0.0, 0.0);
        assert!((cam.eye - eye_before).magnitude() < 1e-10);
    }

    #[test]
    fn camera_orbit_no_sign_flip() {
        // Repeated small orbits must keep the eye on the same side of target.
        let mut cam = Camera {
            eye: V([0.0, 3.0, 5.0]),
            target: V3::ZERO,
            up: V([0.0, 1.0, 0.0]),
            projection: Projection::Perspective {
                fov: std::f64::consts::FRAC_PI_4,
                aspect: 1.0,
                near: 0.1,
                far: 100.0,
            },
        };
        for _ in 0..10 {
            cam.orbit(0.01, 0.01);
            // Z must stay positive (same side as initial eye)
            assert!(cam.eye.0[2] > 0.0, "eye flipped to z={}", cam.eye.0[2]);
        }
    }

    #[test]
    fn camera_zoom() {
        let mut cam = Camera::default();
        let dist_before = (cam.eye - cam.target).magnitude();
        cam.zoom(0.5);
        let dist_after = (cam.eye - cam.target).magnitude();
        assert!(dist_after < dist_before);
    }

    #[test]
    fn rigid_body_gravity() {
        let mut body = RigidBody::default();
        body.position = V([0.0, 5.0, 0.0]);
        body.step(1.0, V([0.0, -9.8, 0.0]));
        assert!(body.position.0[1] < 5.0);
        assert!(body.velocity.0[1] < 0.0);
    }

    #[test]
    fn rigid_body_bounce() {
        let mut body = RigidBody {
            position: V([0.0, -0.1, 0.0]),
            velocity: V([0.0, -5.0, 0.0]),
            mass: 1.0,
            restitution: 0.8,
        };
        body.bounce_plane(0.0);
        assert!(body.position.0[1] >= 0.0);
        assert!(body.velocity.0[1] > 0.0);
        assert!((body.velocity.0[1] - 4.0).abs() < 0.01);
    }

    #[test]
    fn scene_graphable_topology_labels_and_pixels() {
        use crate::render::{Color, Renderable};
        use crate::visual::Graphable;

        let mut scene = Scene::default();
        scene.add(Mesh::cube(V3::ZERO, 1.0));
        scene.add(Mesh::cube(V([3.0, 0.0, 0.0]), 0.5));
        scene.add_light(Light::Directional {
            direction: V([0.0, -1.0, 0.0]),
            color: V([1.0, 1.0, 1.0]),
            intensity: 1.0,
        });

        let g = scene.to_graph();
        // 2 objects + 2 lights (ambient default + directional) + 1 scene node + 1 camera = 6
        assert_eq!(g.len(), 6, "scene graph node count");

        // Verify labels via render
        let mut list = crate::render::RenderList::default();
        g.render(&mut list, &());
        let texts: Vec<String> = list
            .iter()
            .filter_map(|p| {
                if let crate::render::Primitive::Text { content, .. } = p {
                    Some(content.clone())
                } else {
                    None
                }
            })
            .collect();
        assert!(texts.iter().any(|t| t.contains("Scene")));
        assert!(texts.iter().any(|t| t.contains("Camera")));
        assert!(texts.iter().any(|t| t.contains("Object 0")));

        // Pixel verification: capture_graph produces visible content
        let pb = scene.capture_graph(600, 200);
        let bg = Color::rgb(25, 28, 36);
        let non_bg = (0..600).filter(|&x| pb.pixel(x, 50) != bg).count();
        assert!(
            non_bg > 20,
            "graph should render content, got {non_bg} non-bg pixels"
        );
    }
}
