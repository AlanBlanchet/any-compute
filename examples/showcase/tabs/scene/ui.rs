use super::*;
// ── Hierarchy panel (left — just names) ─────────────────────────────────

impl SceneInfo {
    pub(super) fn build_hierarchy(&self, sheet: &StyleSheet, t: &mut Tree, parent: NodeId) {
        let panel = t.add_box(
            parent,
            Style::default()
                .w(140.0)
                .bg(theme::SIDEBAR_BG)
                .overflow(Overflow::Scroll)
                .pad(6.0),
        );

        t.add_text(panel, "Hierarchy", s(sheet, "subheading").pad_xy(0.0, 4.0));

        for (i, _obj) in self.scene.objects.iter().enumerate() {
            let name = *OBJECT_NAMES.get(i).unwrap_or(&"Object");
            let color = OBJECT_COLORS.get(i).copied().unwrap_or(C_CUBE);
            let is_sel = self.selected == Some(i);

            let row_bg = if is_sel {
                theme::SURFACE_BRIGHT
            } else {
                Color::TRANSPARENT
            };
            let row = t.add_box(
                panel,
                Style::default()
                    .row()
                    .align(Align::Center)
                    .gap(6.0)
                    .pad_xy(6.0, 4.0)
                    .radius(4.0)
                    .bg(row_bg),
            );
            t.add_box(row, Style::default().w(6.0).h(6.0).radius(3.0).bg(color));
            let text_c = if is_sel { theme::TEXT } else { theme::SUBTEXT0 };
            t.add_text(row, name, s(sheet, "font-11").color(text_c));
            t.tag(row, &format!("scene-obj-{i}"));
        }

        t.add_text(panel, "Camera", s(sheet, "subheading").pad_xy(0.0, 6.0));
        let cam_sel = self.selected == Some(100);
        let cam_row = t.add_box(
            panel,
            Style::default()
                .row()
                .pad_xy(6.0, 4.0)
                .radius(4.0)
                .bg(if cam_sel {
                    theme::SURFACE_BRIGHT
                } else {
                    Color::TRANSPARENT
                }),
        );
        t.add_text(
            cam_row,
            "Main Camera",
            s(sheet, "font-11").color(if cam_sel {
                theme::TEXT
            } else {
                theme::SUBTEXT0
            }),
        );
        t.tag(cam_row, "scene-obj-100");

        t.add_text(panel, "Lights", s(sheet, "subheading").pad_xy(0.0, 6.0));
        for (i, _light) in self.scene.lights.iter().enumerate() {
            let idx = 200 + i;
            let is_sel = self.selected == Some(idx);
            let lrow = t.add_box(
                panel,
                Style::default()
                    .row()
                    .pad_xy(6.0, 4.0)
                    .radius(4.0)
                    .bg(if is_sel {
                        theme::SURFACE_BRIGHT
                    } else {
                        Color::TRANSPARENT
                    }),
            );
            let lbl = match _light {
                Light::Directional { .. } => "Directional",
                Light::Point { .. } => "Point",
                Light::Ambient { .. } => "Ambient",
            };
            t.add_text(
                lrow,
                lbl,
                s(sheet, "font-11").color(if is_sel { theme::TEXT } else { theme::SUBTEXT0 }),
            );
            t.tag(lrow, &format!("scene-obj-{idx}"));
        }
    }

    // ── Inspector panel (right — details for selected) ──────────────

    pub(super) fn build_inspector(&self, sheet: &StyleSheet, t: &mut Tree, parent: NodeId) {
        let panel = t.add_box(
            parent,
            Style::default()
                .w(180.0)
                .bg(theme::SIDEBAR_BG)
                .overflow(Overflow::Scroll)
                .pad(8.0),
        );

        t.add_text(panel, "Inspector", s(sheet, "subheading").pad_xy(0.0, 4.0));

        let sel = match self.selected {
            Some(s) => s,
            None => {
                t.add_text(panel, "Select an object", s(sheet, "label"));
                return;
            }
        };

        if sel < self.scene.objects.len() {
            let obj = &self.scene.objects[sel];
            let name = *OBJECT_NAMES.get(sel).unwrap_or(&"Object");
            let color = OBJECT_COLORS.get(sel).copied().unwrap_or(C_CUBE);

            let hdr = t.add_box(panel, Style::default().row().align(Align::Center).gap(6.0));
            t.add_box(hdr, Style::default().w(10.0).h(10.0).radius(5.0).bg(color));
            t.add_text(hdr, name, s(sheet, "heading"));

            let card = t.add_box(panel, s(sheet, "card"));
            let pos = obj.transform.position;
            t.add_text(
                card,
                &format!("X {:.2}  Y {:.2}  Z {:.2}", pos.0[0], pos.0[1], pos.0[2]),
                s(sheet, "detail"),
            );
            let rot = obj.transform.rotation;
            t.add_text(
                card,
                &format!(
                    "R {:.1}° {:.1}° {:.1}°",
                    rot.0[0].to_degrees(),
                    rot.0[1].to_degrees(),
                    rot.0[2].to_degrees()
                ),
                s(sheet, "detail"),
            );
            let sc = obj.transform.scale;
            t.add_text(
                card,
                &format!("S {:.2}  {:.2}  {:.2}", sc.0[0], sc.0[1], sc.0[2]),
                s(sheet, "detail"),
            );

            let mesh_card = t.add_box(panel, s(sheet, "card"));
            t.add_text(mesh_card, "Mesh", s(sheet, "font-9").color(theme::ACCENT));
            t.add_text(
                mesh_card,
                &format!("{} vertices", obj.mesh.vertex_count()),
                s(sheet, "detail"),
            );
            t.add_text(
                mesh_card,
                &format!("{} triangles", obj.mesh.tri_count()),
                s(sheet, "detail"),
            );

            let mat = self.scene.materials.get(obj.mesh.material);
            if let Some(mat) = mat {
                let mat_card = t.add_box(panel, s(sheet, "card"));
                t.add_text(
                    mat_card,
                    "Material",
                    s(sheet, "font-9").color(theme::ACCENT),
                );
                t.add_text(
                    mat_card,
                    &format!("roughness {:.2}", mat.roughness),
                    s(sheet, "detail"),
                );
                t.add_text(
                    mat_card,
                    &format!("metallic  {:.2}", mat.metallic),
                    s(sheet, "detail"),
                );
            }

            if sel > 0 {
                let aabb = obj.mesh.bounds();
                let aabb_card = t.add_box(panel, s(sheet, "card"));
                t.add_text(aabb_card, "Bounds", s(sheet, "font-9").color(theme::ACCENT));
                t.add_text(
                    aabb_card,
                    &format!(
                        "origin ({:.1}, {:.1}, {:.1})",
                        aabb.origin.0[0], aabb.origin.0[1], aabb.origin.0[2]
                    ),
                    s(sheet, "detail"),
                );
                t.add_text(
                    aabb_card,
                    &format!(
                        "size ({:.1}, {:.1}, {:.1})",
                        aabb.size.0[0], aabb.size.0[1], aabb.size.0[2]
                    ),
                    s(sheet, "detail"),
                );
            }
        } else if sel == 100 {
            t.add_text(panel, "Main Camera", s(sheet, "heading"));
            let card = t.add_box(panel, s(sheet, "card"));
            let eye = self.scene.camera.eye;
            let tgt = self.scene.camera.target;
            t.add_text(
                card,
                &format!("eye  ({:.1}, {:.1}, {:.1})", eye.0[0], eye.0[1], eye.0[2]),
                s(sheet, "detail"),
            );
            t.add_text(
                card,
                &format!("tgt  ({:.1}, {:.1}, {:.1})", tgt.0[0], tgt.0[1], tgt.0[2]),
                s(sheet, "detail"),
            );
            if let Projection::Perspective { fov, near, far, .. } = self.scene.camera.projection {
                t.add_text(
                    card,
                    &format!("fov {:.0}°", fov.to_degrees()),
                    s(sheet, "detail"),
                );
                t.add_text(
                    card,
                    &format!("near {:.2}  far {:.0}", near, far),
                    s(sheet, "detail"),
                );
            }
        } else if sel >= 200 {
            let li = sel - 200;
            if let Some(light) = self.scene.lights.get(li) {
                let card = t.add_box(panel, s(sheet, "card"));
                match light {
                    Light::Directional {
                        direction,
                        color,
                        intensity,
                    } => {
                        t.add_text(card, "Directional", s(sheet, "heading"));
                        t.add_text(
                            card,
                            &format!(
                                "dir ({:.2}, {:.2}, {:.2})",
                                direction.0[0], direction.0[1], direction.0[2]
                            ),
                            s(sheet, "detail"),
                        );
                        t.add_text(
                            card,
                            &format!("intensity {intensity:.2}"),
                            s(sheet, "detail"),
                        );
                        t.add_text(
                            card,
                            &format!(
                                "color ({:.2}, {:.2}, {:.2})",
                                color.0[0], color.0[1], color.0[2]
                            ),
                            s(sheet, "detail"),
                        );
                    }
                    Light::Point {
                        position,
                        color,
                        intensity,
                        range,
                    } => {
                        t.add_text(card, "Point Light", s(sheet, "heading"));
                        t.add_text(
                            card,
                            &format!(
                                "pos ({:.1}, {:.1}, {:.1})",
                                position.0[0], position.0[1], position.0[2]
                            ),
                            s(sheet, "detail"),
                        );
                        t.add_text(
                            card,
                            &format!("intensity {intensity:.2}  range {range:.1}"),
                            s(sheet, "detail"),
                        );
                        t.add_text(
                            card,
                            &format!(
                                "color ({:.2}, {:.2}, {:.2})",
                                color.0[0], color.0[1], color.0[2]
                            ),
                            s(sheet, "detail"),
                        );
                    }
                    Light::Ambient { color, intensity } => {
                        t.add_text(card, "Ambient", s(sheet, "heading"));
                        t.add_text(
                            card,
                            &format!("intensity {intensity:.2}"),
                            s(sheet, "detail"),
                        );
                        t.add_text(
                            card,
                            &format!(
                                "color ({:.2}, {:.2}, {:.2})",
                                color.0[0], color.0[1], color.0[2]
                            ),
                            s(sheet, "detail"),
                        );
                    }
                }
            }
        }
    }
}

