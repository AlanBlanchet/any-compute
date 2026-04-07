//! 3D Scene tab — interactive solid-face 3D viewport with camera controls.
//!
//! Uses `Camera::project_to_screen()` for proper projection, filled faces
//! with flat shading, mouse-drag orbiting, scroll-zoom, and physics.
//!
//! Rendering is split: DOM nodes for chrome (badges, legend, info panels)
//! and raw `Primitive`s for the 3D viewport (faces + lines), eliminating
//! ~900 DOM nodes per frame.

use any_compute_core::layout::{Point, V};
use any_compute_core::render::{Color, RenderList, Viewport};
use any_compute_core::scene::*;
use any_compute_dom::css::StyleSheet;
use any_compute_dom::style::*;
use any_compute_dom::theme;
use any_compute_dom::tree::*;

use super::helpers::{badge, s, sm};


mod state;
mod ui;

pub use state::*;

// ── Build ───────────────────────────────────────────────────────────────

pub fn build(
    sheet: &StyleSheet,
    t: &mut Tree,
    parent: NodeId,
    data: &mut crate::AppData,
    _time: f64,
) {
    let info = &mut data.scene_info;
    let (vp_w, vp_h) = info.vp_size;
    let camera = &info.scene.camera;
    let vp = camera.view(vp_w, vp_h);

    // Stats badges
    let stats = t.add_box(parent, sm(sheet, &["row-gap-12", "wrap"]));
    let total_verts: usize = info
        .scene
        .objects
        .iter()
        .map(|o| o.mesh.vertex_count())
        .sum();
    let total_tris: usize = info.scene.objects.iter().map(|o| o.mesh.tri_count()).sum();
    t.add_text(
        stats,
        &format!("{total_verts} verts"),
        badge(sheet, "green"),
    );
    t.add_text(stats, &format!("{total_tris} tris"), badge(sheet, "blue"));
    t.add_text(
        stats,
        &format!("{} objects", info.scene.objects.len()),
        badge(sheet, "yellow"),
    );
    t.add_text(
        stats,
        &format!("{} lights", info.scene.lights.len()),
        badge(sheet, "yellow"),
    );
    t.add_text(
        stats,
        &format!("{} materials", info.scene.materials.len()),
        badge(sheet, "red"),
    );
    t.add_text(
        stats,
        "WASD move \u{2022} Q/E up/down \u{2022} Drag rotate \u{2022} Scroll zoom",
        sm(sheet, &["font-9", "text-dim"]),
    );

    // ── Main row: hierarchy + 3D viewport + inspector ────────────
    let main_row = t.add_box(parent, Style::default().row().grow(1.0));

    // ── Hierarchy panel (left) ──────────────────────────────────────
    info.build_hierarchy(sheet, t, main_row);

    // ── 3D Viewport (DOM shell only — content via raw primitives) ───
    let viewport = t.add_box(main_row, sheet.class("scene-panel").grow(1.0));
    t.tag(viewport, "scene-viewport");

    // ── Inspector panel (right) ─────────────────────────────────────
    info.build_inspector(sheet, t, main_row);

    // Legend (DOM nodes — small)
    let legend = t.add_box(viewport, sm(sheet, &["row-gap-8"]).abs(10.0, 10.0));
    for (label, color) in [
        ("Cube", C_CUBE),
        ("Sphere", C_SPHERE),
        ("Cylinder", C_CYLINDER),
        ("Torus", C_TORUS),
        ("AABB", C_AABB),
    ] {
        let _ = t.add_box(legend, Style::default().w(8.0).h(8.0).radius(4.0).bg(color));
        t.add_text(legend, label, sm(sheet, &["font-9", "text-dim"]));
    }

    // Axis labels (DOM text — only 3 nodes)
    let proj = |p| vp.project(p);
    let ax_len = 0.8;
    let axes = [
        (V([ax_len, 0.0, 0.0]), C_SPHERE, "X"),
        (V([0.0, ax_len, 0.0]), C_CYLINDER, "Y"),
        (V([0.0, 0.0, ax_len]), theme::ACCENT, "Z"),
    ];
    for (dir, color, label) in axes {
        if let Some((lx, ly, _)) = proj(dir) {
            t.add_text(
                viewport,
                label,
                sm(sheet, &["font-9"]).color(color).abs(lx + 4.0, ly - 4.0),
            );
        }
    }

    // ── Build raw primitives for the viewport ───────────────────────
    let mut prims = RenderList::default();

    // Grid (lines instead of dot-boxes)
    let gc = Color::rgba(255, 255, 255, 15);
    for i in -6..=6 {
        let z = i as f64;
        prims.push_projected_line(&vp, V([-6.0, 0.0, z]), V([6.0, 0.0, z]), gc, 1.0);
        prims.push_projected_line(&vp, V([z, 0.0, -6.0]), V([z, 0.0, 6.0]), gc, 1.0);
    }

    // Collect + depth-sort faces
    let mut faces = Vec::<ProjectedFace>::new();
    for (obj_idx, obj) in info.scene.objects.iter().enumerate() {
        let mesh = &obj.mesh;
        let vc = mesh.vertex_count();
        let base_color = info.mesh_colors.get(obj_idx).copied().unwrap_or(C_CUBE);
        let mut face_idx = 0usize;

        for [i0, i1, i2] in mesh.faces() {
            if i0 >= vc || i1 >= vc || i2 >= vc {
                continue;
            }
            let a = obj.transform.apply(mesh.vertex(i0));
            let b = obj.transform.apply(mesh.vertex(i1));
            let c = obj.transform.apply(mesh.vertex(i2));

            let normal = (b - a).cross(c - a).normalized();
            let center = (a + b + c) * (1.0 / 3.0);
            let view_dir = (center - vp.camera.eye).normalized();
            let ndv = normal.dot(view_dir);
            // Hard cull truly back-facing geometry; faces near the silhouette
            // fade via alpha to prevent pop-in/pop-out flickering.
            if ndv > 0.0 {
                continue;
            }

            let pa = proj(a);
            let pb = proj(b);
            let pc = proj(c);

            if let (Some((ax, ay, az)), Some((bx, by, bz)), Some((cx2, cy2, cz))) = (pa, pb, pc) {
                let min_x = ax.min(bx).min(cx2);
                let min_y = ay.min(by).min(cy2);
                let max_x = ax.max(bx).max(cx2);
                let max_y = ay.max(by).max(cy2);
                if (max_x - min_x) < 1.0
                    || (max_y - min_y) < 1.0
                    || max_x < 0.0
                    || max_y < 0.0
                    || min_x > vp_w
                    || min_y > vp_h
                {
                    continue;
                }
                let face_color = info.scene.shade_face(normal, base_color);
                faces.push(ProjectedFace {
                    obj: obj_idx,
                    face: face_idx,
                    depth: (az + bz + cz) / 3.0,
                    v: [(ax, ay), (bx, by), (cx2, cy2)],
                    color: face_color,
                });
            }
            face_idx += 1;
        }
    }

    // Depth sort: painter's algorithm (farthest first).
    // Group by object at similar depths to avoid interleaving artifacts.
    const DEPTH_QUANT: f64 = 256.0; // ~0.004 units precision
    faces.sort_by(|a, b| {
        let da = (a.depth * DEPTH_QUANT).round() as i64;
        let db = (b.depth * DEPTH_QUANT).round() as i64;
        db.cmp(&da) // farthest first (painter's)
            .then(a.obj.cmp(&b.obj)) // same-depth: group by object
            .then(a.face.cmp(&b.face)) // within object: stable face order
    });
    log::debug!(
        "scene: {} faces after cull, {} grid lines, vp={vp_w}x{vp_h}",
        faces.len(),
        prims.len(),
    );
    for f in &faces {
        prims.push_triangle(
            [
                Point::new(f.v[0].0, f.v[0].1),
                Point::new(f.v[1].0, f.v[1].1),
                Point::new(f.v[2].0, f.v[2].1),
            ],
            f.color,
        );
    }

    // Light indicators
    for light in &info.scene.lights {
        match light {
            Light::Point { position, .. } => {
                if let Some((lx, ly, _)) = proj(*position) {
                    if lx > 0.0 && ly > 0.0 && lx < vp_w && ly < vp_h {
                        prims.push_dot(lx, ly, 10.0, C_AABB);
                    }
                }
            }
            Light::Directional { direction, .. } => {
                let ax = vp_w - 50.0;
                let ay = 30.0;
                let dl = 20.0;
                prims.push_line(
                    ax,
                    ay,
                    ax + direction.x * dl,
                    ay - direction.y * dl,
                    C_AABB,
                    2.0,
                );
            }
            _ => {}
        }
    }

    // AABB wireframe for selected object only
    if let Some(sel) = info.selected {
        if sel < info.scene.objects.len() && sel > 0 {
            let obj = &info.scene.objects[sel];
            info.push_aabb_wireframe(
                &mut prims,
                &vp,
                &obj.mesh.bounds(),
                &obj.transform,
                C_AABB.with_alpha(80),
            );
        }
    }

    // Physics velocity arrows
    for (i, body) in info.bodies.iter().enumerate() {
        if let Some((px, py, _)) = proj(body.position) {
            let col = info.mesh_colors.get(i + 1).copied().unwrap_or(C_CUBE);
            let tip = body.position + body.velocity * 0.15;
            if let Some((tx, ty, _)) = proj(tip) {
                prims.push_line(px, py, tx, ty, col.with_alpha(100), 2.0);
            }
        }
    }

    // Axis lines
    for (dir, color, _) in axes {
        prims.push_projected_line(&vp, V([0.0; 3]), dir, color, 2.0);
    }

    info.viewport_prims = prims;
    log::debug!(
        "scene: total viewport prims={}, bodies=[y={:.2}, y={:.2}]",
        info.viewport_prims.len(),
        info.bodies.get(0).map_or(0.0, |b| b.position.y),
        info.bodies.get(1).map_or(0.0, |b| b.position.y),
    );
}
