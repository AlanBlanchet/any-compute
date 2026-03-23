//! 3D Scene tab — mesh, camera, transform, ray intersection, normals.
//!
//! Builds a scene with primitives, renders a wireframe-style ASCII overview,
//! and shows stats (vertex count, tri count, AABB, ray hits).

use any_compute_core::layout::{V, AABB};
use any_compute_core::scene::*;
use any_compute_dom::css::StyleSheet;
use any_compute_dom::style::*;
use any_compute_dom::tree::*;
use any_compute_dom::theme;

use super::helpers::{s, sm, kv_card};
use crate::AppData;

/// Pre-computed scene info (built once, displayed every frame).
#[derive(Debug, Clone)]
pub struct SceneInfo {
    pub vertex_count: usize,
    pub tri_count: usize,
    pub object_count: usize,
    pub bounds: AABB,
    pub ray_hit: bool,
    pub ray_t: f64,
    pub camera_pos: String,
    pub normals_computed: bool,
}

impl Default for SceneInfo {
    fn default() -> Self {
        // Build a demo scene
        let mut scene = Scene::default();

        // Ground plane (2 triangles)
        let ground = Mesh::new(vec![
            -5.0, 0.0, -5.0,  5.0, 0.0, -5.0,  5.0, 0.0, 5.0,
            -5.0, 0.0, -5.0,  5.0, 0.0, 5.0,  -5.0, 0.0, 5.0,
        ]).with_material(0);
        scene.add(ground);

        // Cube (12 triangles)
        let cube = make_cube(V([0.0, 1.0, 0.0]), 1.0);
        scene.add(cube);

        // Pyramid (4 triangles)
        let pyramid = make_pyramid(V([3.0, 0.0, 0.0]), 1.5, 2.0);
        scene.add(pyramid);

        // Compute bounds
        let bounds = scene.bounds();

        // Ray test: shoot from camera toward origin
        let ray = Ray::new(V([0.0, 2.0, 5.0]), V([0.0, -0.3, -1.0]));
        let hit = ray.intersect_aabb(&bounds);

        // Normals on cube
        let mut obj = scene.objects[1].mesh.clone();
        obj.compute_normals();

        Self {
            vertex_count: scene.objects.iter().map(|o| o.mesh.vertex_count()).sum(),
            tri_count: scene.objects.iter().map(|o| o.mesh.tri_count()).sum(),
            object_count: scene.objects.len(),
            bounds,
            ray_hit: hit.is_some(),
            ray_t: hit.unwrap_or(0.0),
            camera_pos: format!("{:.1}, {:.1}, {:.1}", 0.0, 2.0, 5.0),
            normals_computed: obj.normals.len() > 0,
        }
    }
}

fn make_cube(center: V<3>, size: f64) -> Mesh {
    let h = size / 2.0;
    let c = center;
    // 8 corners
    let v = [
        [c.0[0]-h, c.0[1]-h, c.0[2]-h], [c.0[0]+h, c.0[1]-h, c.0[2]-h],
        [c.0[0]+h, c.0[1]+h, c.0[2]-h], [c.0[0]-h, c.0[1]+h, c.0[2]-h],
        [c.0[0]-h, c.0[1]-h, c.0[2]+h], [c.0[0]+h, c.0[1]-h, c.0[2]+h],
        [c.0[0]+h, c.0[1]+h, c.0[2]+h], [c.0[0]-h, c.0[1]+h, c.0[2]+h],
    ];
    let verts: Vec<f64> = v.iter().flat_map(|p| p.iter().copied()).collect();
    let indices = vec![
        0,1,2, 0,2,3, // front
        1,5,6, 1,6,2, // right
        5,4,7, 5,7,6, // back
        4,0,3, 4,3,7, // left
        3,2,6, 3,6,7, // top
        4,5,1, 4,1,0, // bottom
    ];
    Mesh::new(verts).with_indices(indices)
}

fn make_pyramid(base_center: V<3>, base_size: f64, height: f64) -> Mesh {
    let h = base_size / 2.0;
    let c = base_center;
    let apex = [c.0[0], c.0[1] + height, c.0[2]];
    let bl = [c.0[0]-h, c.0[1], c.0[2]-h];
    let br = [c.0[0]+h, c.0[1], c.0[2]-h];
    let fr = [c.0[0]+h, c.0[1], c.0[2]+h];
    let fl = [c.0[0]-h, c.0[1], c.0[2]+h];
    // 4 side faces + 2 base triangles = 6
    let verts: Vec<f64> = [
        // front
        fl, fr, apex,
        // right
        fr, br, apex,
        // back
        br, bl, apex,
        // left
        bl, fl, apex,
        // base
        bl, br, fr,
        bl, fr, fl,
    ]
    .iter()
    .flat_map(|p| p.iter().copied())
    .collect();
    Mesh::new(verts)
}

pub fn build(sheet: &StyleSheet, t: &mut Tree, parent: NodeId, data: &AppData) {
    let info = &data.scene_info;

    t.add_text(parent, "3D Scene", s(sheet, "title"));
    t.add_text(parent, "Mesh, Camera, Transform, Ray intersection, Normals computation", s(sheet, "subtitle"));
    t.add_box(parent, s(sheet, "spacer-12"));

    // Scene stats
    let stats = t.add_box(parent, sm(sheet, &["row-gap-16"]));

    let stat_card = |t: &mut Tree, p: NodeId, label: &str, value: &str, color: &str| {
        let c = t.add_box(p, s(sheet, "card-sm").w(160.0));
        t.add_text(c, label, s(sheet, "label"));
        t.add_text(c, value, sm(sheet, &["font-24", color]));
    };

    stat_card(t, stats, "Vertices", &info.vertex_count.to_string(), "green");
    stat_card(t, stats, "Triangles", &info.tri_count.to_string(), "blue");
    stat_card(t, stats, "Objects", &info.object_count.to_string(), "mauve");

    t.add_box(parent, s(sheet, "spacer-12"));

    // Scene objects
    t.add_text(parent, "Scene Objects", sm(sheet, &["heading", "text"]));
    let objects = [
        ("Ground Plane", "2 tris", "Flat XZ plane"),
        ("Cube", "12 tris", "Indexed mesh, 8 vertices"),
        ("Pyramid", "6 tris", "Non-indexed, face list"),
    ];
    for (name, tris, desc) in &objects {
        let row = t.add_box(parent, sm(sheet, &["row-gap-8"]));
        let _dot = t.add_box(row, Style::default().w(8.0).h(8.0).radius(4.0).bg(theme::ACCENT));
        t.add_text(row, *name, sm(sheet, &["font-13", "text"]).w(120.0));
        t.add_text(row, *tris, sm(sheet, &["font-13", "blue"]).w(60.0));
        t.add_text(row, *desc, s(sheet, "body"));
    }

    t.add_box(parent, s(sheet, "spacer-12"));

    // AABB
    t.add_text(parent, "Bounding Box (AABB)", sm(sheet, &["heading", "text"]));
    let b = &info.bounds;
    let aabb_row = t.add_box(parent, sm(sheet, &["row-gap-16"]));
    let origin_str = format!("({:.1}, {:.1}, {:.1})", b.origin.0[0], b.origin.0[1], b.origin.0[2]);
    let size_str = format!("({:.1}, {:.1}, {:.1})", b.size.0[0], b.size.0[1], b.size.0[2]);
    kv_card(t, aabb_row, sheet, "Origin", &origin_str);
    kv_card(t, aabb_row, sheet, "Size", &size_str);

    t.add_box(parent, s(sheet, "spacer-12"));

    // Ray intersection
    t.add_text(parent, "Ray Casting", sm(sheet, &["heading", "text"]));
    let ray_row = t.add_box(parent, sm(sheet, &["row-gap-16"]));
    kv_card(t, ray_row, sheet, "Camera", &info.camera_pos);
    kv_card(t, ray_row, sheet, "AABB Hit", if info.ray_hit { "Yes" } else { "No" });
    if info.ray_hit {
        kv_card(t, ray_row, sheet, "t", &format!("{:.3}", info.ray_t));
    }

    t.add_box(parent, s(sheet, "spacer-12"));

    // Features
    t.add_text(parent, "3D Features", sm(sheet, &["heading", "text"]));
    let features = [
        ("Mesh", "Flat Buffer vertex storage, indexed + non-indexed"),
        ("Camera", "Perspective + Orthographic projection"),
        ("Transform", "Position + Quaternion rotation + Scale, Lerp"),
        ("Light", "Directional, Point, Ambient with intensity"),
        ("Material", "PBR: albedo, metallic, roughness, emissive, IOR"),
        ("Ray", "AABB slab intersection + Moller-Trumbore triangle"),
        ("Normals", &format!("Area-weighted smooth normals: {}", if info.normals_computed { "Computed" } else { "Pending" })),
    ];
    for (label, desc) in &features {
        let row = t.add_box(parent, sm(sheet, &["row-gap-8"]));
        t.add_text(row, *label, sm(sheet, &["font-13", "teal"]).w(100.0));
        t.add_text(row, *desc, s(sheet, "body"));
    }
}
