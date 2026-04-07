use super::*;
// ═══════════════════════════════════════════════════════════════════════════
// Shared panel helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Build a standard panel: info bar + scrollable body. Returns `(info_row, scroll_box)`.
fn panel_scaffold(sheet: &StyleSheet, t: &mut Tree, parent: NodeId) -> (NodeId, NodeId) {
    let info = t.add_box(parent, sm(sheet, &["row", "gap-12"]).align(Align::Center));
    let body = t.add_box(parent, s(sheet, "source-box").overflow(Overflow::Scroll));
    (info, body)
}

/// Empty-state placeholder when a panel has no data.
fn empty_state(sheet: &StyleSheet, t: &mut Tree, parent: NodeId, msg: &str) {
    t.add_text(parent, msg, sm(sheet, &["font-11", "text-dim"]).pad(12.0));
}

// ═══════════════════════════════════════════════════════════════════════════
// Elements subtab — DOM tree inspector
// ═══════════════════════════════════════════════════════════════════════════

pub(super) fn build_elements(sheet: &StyleSheet, t: &mut Tree, parent: NodeId, state: &BrowserState) {
    let Some(page) = &state.page else {
        empty_state(sheet, t, parent, "No page loaded");
        return;
    };
    let (info, tree_box) = panel_scaffold(sheet, t, parent);
    t.add_text(
        info,
        &format!("{} nodes", state.node_count()),
        badge(sheet, "green"),
    );

    // DOM tree with expand/collapse, classes, ids
    fn render_dom_node(
        sheet: &StyleSheet,
        t: &mut Tree,
        container: NodeId,
        arena: &[Slot],
        id: NodeId,
        depth: usize,
    ) {
        if depth > 8 {
            return;
        }
        let slot = &arena[id.0];
        let indent = depth as f64 * 14.0;
        let row = t.add_box(
            container,
            Style::default()
                .row()
                .pad_xy(4.0, 1.0)
                .gap(2.0)
                .align(Align::Center),
        );

        let (color, label) = match &slot.kind {
            NodeKind::Text(s) => {
                let preview: String = s.chars().take(50).collect();
                let ellipsis = if s.len() > 50 { "…" } else { "" };
                (theme::GREEN, format!("\"{preview}{ellipsis}\""))
            }
            NodeKind::Box => {
                let el = if slot.element.is_empty() {
                    "div"
                } else {
                    &slot.element
                };
                let mut lbl = format!("<{el}");
                if let Some(id_attr) = &slot.id {
                    lbl.push_str(&format!(" id=\"{id_attr}\""));
                }
                if !slot.class_list.is_empty() {
                    lbl.push_str(&format!(" class=\"{}\"", slot.class_list.join(" ")));
                }
                lbl.push('>');
                (theme::BLUE, lbl)
            }
            NodeKind::Bar { .. } => (theme::YELLOW, "<bar>".into()),
        };

        // Collapse arrow
        let has_children = !slot.children.is_empty();
        let arrow = if has_children { "▼" } else { " " };
        let mut arrow_s = s(sheet, "font-9").color(theme::OVERLAY0);
        arrow_s.padding.left = indent;
        t.add_text(row, arrow, arrow_s.w(indent + 12.0));

        // Tag label
        t.add_text(row, &label, s(sheet, "font-11").color(color));

        // Dimensions (small, right-aligned)
        if matches!(slot.kind, NodeKind::Box) {
            let r = &slot.rect;
            let dim = format!(
                "{}×{}",
                r.size.w().round() as i32,
                r.size.h().round() as i32
            );
            t.add_text(
                row,
                &dim,
                s(sheet, "font-9").color(theme::OVERLAY0).grow(1.0),
            );
        }

        for &child in &slot.children {
            render_dom_node(sheet, t, container, arena, child, depth + 1);
        }
    }

    render_dom_node(sheet, t, tree_box, &page.tree().arena, NodeId(0), 0);

    // ── Computed Styles section ──────────────────────────────
    let styles_hdr = t.add_box(
        parent,
        Style::default()
            .row()
            .pad_xy(8.0, 6.0)
            .bg(Color::rgba(255, 255, 255, 8)),
    );
    t.add_text(
        styles_hdr,
        "Computed Styles",
        s(sheet, "font-11").color(theme::TEXT),
    );

    let styles_box = t.add_box(
        parent,
        Style::default().gap(0.0).pad_xy(8.0, 4.0),
    );
    // Show root element's key styles
    if let Some(page) = &state.page {
        let root = &page.tree().arena[0];
        let rs = &root.style;
        let props = [
            ("display", format!("{:?}", rs.display)),
            ("font-size", format!("{}px", rs.font_size)),
            ("color", format_color(rs.color)),
            ("background", format_color(rs.background)),
            ("padding", format_edges(&rs.padding)),
            ("margin", format_edges(&rs.margin)),
            ("border-radius", format!("{}px", rs.corner_radius)),
            ("overflow", format!("{:?}", rs.overflow)),
        ];
        for (prop, value) in &props {
            let row = t.add_box(styles_box, Style::default().row().gap(8.0).pad_xy(0.0, 1.0));
            t.add_text(row, *prop, s(sheet, "font-9").color(theme::MAUVE).w(100.0));
            t.add_text(row, value, s(sheet, "font-9").color(theme::TEXT));
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Console subtab — JS logs panel
// ═══════════════════════════════════════════════════════════════════════════

pub(super) fn build_console(sheet: &StyleSheet, t: &mut Tree, parent: NodeId, state: &BrowserState) {
    let (info, log_box) = panel_scaffold(sheet, t, parent);
    t.add_text(
        info,
        &format!("{} entries", state.console_logs.len()),
        badge(sheet, "blue"),
    );

    // Count by level
    let errs = state
        .console_logs
        .iter()
        .filter(|(l, _)| l == "error")
        .count();
    let warns = state
        .console_logs
        .iter()
        .filter(|(l, _)| l == "warn")
        .count();
    if errs > 0 {
        t.add_text(info, &format!("{errs} errors"), badge(sheet, "red"));
    }
    if warns > 0 {
        t.add_text(info, &format!("{warns} warnings"), badge(sheet, "yellow"));
    }

    for (idx, (level, msg)) in state.console_logs.iter().enumerate() {
        let (bg, level_color, prefix) = match level.as_str() {
            "error" => (Color::rgba(255, 50, 50, 15), theme::RED, "✕"),
            "warn" => (Color::rgba(255, 200, 50, 10), theme::YELLOW, "⚠"),
            _ => (Color::TRANSPARENT, theme::BLUE, "ⓘ"),
        };
        let row = t.add_box(
            log_box,
            Style::default()
                .row()
                .gap(8.0)
                .pad_xy(8.0, 3.0)
                .bg(bg)
                .align(Align::Center),
        );
        // Level icon
        t.add_text(row, prefix, s(sheet, "font-11").color(level_color).w(16.0));
        // Message
        t.add_text(row, msg, s(sheet, "font-11").color(theme::TEXT).grow(1.0));
        // Line number
        t.add_text(
            row,
            &format!(":{}", idx + 1),
            s(sheet, "font-9").color(theme::OVERLAY0),
        );
    }
    if state.console_logs.is_empty() {
        empty_state(sheet, t, log_box, "No console output");
    }

    // ── Console input prompt ────────────────────────────────
    let prompt_row = t.add_box(
        parent,
        Style::default()
            .row()
            .h(28.0)
            .pad_xy(8.0, 0.0)
            .gap(6.0)
            .align(Align::Center)
            .bg(Color::rgba(255, 255, 255, 5)),
    );
    t.add_text(prompt_row, ">", s(sheet, "font-11").color(theme::ACCENT));
    t.add_text(
        prompt_row,
        "Type JS expression...",
        s(sheet, "font-11").color(theme::OVERLAY0),
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Network subtab — request log
// ═══════════════════════════════════════════════════════════════════════════

pub(super) fn build_network(sheet: &StyleSheet, t: &mut Tree, parent: NodeId, state: &BrowserState) {
    let (info, net_box) = panel_scaffold(sheet, t, parent);
    t.add_text(
        info,
        &format!("{} requests", state.network_log.len()),
        badge(sheet, "blue"),
    );

    // Summary badges
    let total_size: usize = state.network_log.iter().map(|(_, _, _, sz)| sz).sum();
    if total_size > 0 {
        t.add_text(
            info,
            &format_human(total_size as f64, "B"),
            badge(sheet, "green"),
        );
    }

    // Column widths
    const COLS: &[(&str, f64)] = &[
        ("Status", 50.0),
        ("Method", 50.0),
        ("URL", 0.0), // grows
        ("Size", 60.0),
        ("Type", 50.0),
    ];

    // Header
    let hdr = t.add_box(
        net_box,
        Style::default()
            .row()
            .gap(4.0)
            .pad_xy(8.0, 4.0)
            .bg(Color::rgba(255, 255, 255, 5)),
    );
    for &(label, w) in COLS {
        let mut st = s(sheet, "font-9").color(theme::OVERLAY0);
        if w > 0.0 {
            st = st.w(w);
        } else {
            st = st.grow(1.0).min_w(80.0);
        }
        t.add_text(hdr, label, st);
    }

    for (method, url, status, size) in &state.network_log {
        let row = t.add_box(
            net_box,
            Style::default()
                .row()
                .gap(4.0)
                .pad_xy(8.0, 2.0)
                .align(Align::Center),
        );

        // Status code (colored badge)
        let status_color = match *status {
            200..=299 => theme::GREEN,
            300..=399 => theme::YELLOW,
            _ => theme::RED,
        };
        let status_box = t.add_box(
            row,
            Style::default()
                .w(50.0)
                .h(18.0)
                .bg(status_color.with_alpha(25))
                .radius(3.0)
                .align(Align::Center)
                .justify(Justify::Center),
        );
        t.add_text(
            status_box,
            &status.to_string(),
            s(sheet, "font-9").color(status_color),
        );

        // Method
        t.add_text(row, method, s(sheet, "font-9").color(theme::MAUVE).w(50.0));

        // URL (truncated, grows)
        let url_display: String = url.chars().take(60).collect();
        let ellipsis = if url.len() > 60 { "…" } else { "" };
        t.add_text(
            row,
            &format!("{url_display}{ellipsis}"),
            s(sheet, "font-9").color(theme::TEXT).grow(1.0).min_w(80.0),
        );

        // Size
        t.add_text(
            row,
            &format_human(*size as f64, "B"),
            s(sheet, "font-9").color(theme::SUBTEXT0).w(60.0),
        );

        // Type
        let rtype = if method == "FILE" {
            "document"
        } else {
            "fetch"
        };
        t.add_text(
            row,
            rtype,
            s(sheet, "font-9").color(theme::OVERLAY0).w(50.0),
        );
    }
    if state.network_log.is_empty() {
        empty_state(sheet, t, net_box, "No network activity");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Source subtab
// ═══════════════════════════════════════════════════════════════════════════

/// HTML source viewer with line numbers.
pub(super) fn build_source(sheet: &StyleSheet, t: &mut Tree, parent: NodeId, state: &BrowserState) {
    // Info bar
    let info = t.add_box(parent, sm(sheet, &["row", "gap-12"]).align(Align::Center));
    t.add_text(
        info,
        &format!("{} lines", state.html.lines().count()),
        badge(sheet, "blue"),
    );
    t.add_text(
        info,
        &format!("{} chars", state.html.len()),
        badge(sheet, "green"),
    );
    t.add_text(
        info,
        &format!("{} nodes", state.node_count()),
        badge(sheet, "yellow"),
    );

    // Source listing (limit to 500 lines to avoid huge trees)
    let source_box = t.add_box(parent, s(sheet, "source-box"));
    for (i, line) in state.html.lines().take(500).enumerate() {
        let row = t.add_box(source_box, s(sheet, "source-row"));
        t.add_text(row, &format!("{:>4}", i + 1), s(sheet, "line-num"));
        if !line.is_empty() {
            t.add_text(row, line, s(sheet, "source-text"));
        }
    }
    let total = state.html.lines().count();
    if total > 500 {
        t.add_text(
            source_box,
            &format!("... {total} total lines (showing first 500)"),
            s(sheet, "font-9").color(theme::OVERLAY0).pad(8.0),
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Style formatting helpers
// ═══════════════════════════════════════════════════════════════════════════

fn format_color(c: Color) -> String {
    if c.a == 255 {
        format!("#{:02x}{:02x}{:02x}", c.r, c.g, c.b)
    } else if c.a == 0 {
        "transparent".into()
    } else {
        format!("rgba({},{},{},{:.2})", c.r, c.g, c.b, c.a as f64 / 255.0)
    }
}

fn format_edges(e: &Edges) -> String {
    if e.top == e.right && e.right == e.bottom && e.bottom == e.left {
        format!("{}px", e.top)
    } else if e.top == e.bottom && e.left == e.right {
        format!("{}px {}px", e.top, e.left)
    } else {
        format!("{}px {}px {}px {}px", e.top, e.right, e.bottom, e.left)
    }
}
