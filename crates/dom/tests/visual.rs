//! Visual regression tests — render isolated components via headless GPU,
//! capture PNGs to `out/visual/`, and assert pixel properties.
//!
//! Run: `cargo test -p any-compute-dom --features gpu --test visual`
//! Or:  `make test-visual`
//!
//! PNGs are saved to `out/visual/` for human inspection. Tests assert
//! structural properties (colors present, regions non-empty, correct tags)
//! rather than exact pixel-perfect snapshots — so they stay stable across
//! GPU drivers.
//!
//! Each test function builds a component tree from scratch, rendering in
//! isolation. State mutations (focus, selection, typing) are applied via
//! the same `TextInput` / `InputEvent` APIs used in production.

// Entire test suite requires GPU — skip compilation without `gpu` feature.
#![cfg(feature = "gpu")]

use any_compute_core::interaction::TextInput;
use any_compute_core::layout::{Point, Size};
use any_compute_core::render::Color;
use any_compute_dom::harness::{Capture, TestHarness};
use any_compute_dom::style::*;
use any_compute_dom::theme;
use any_compute_dom::tree::*;

// ═══════════════════════════════════════════════════════════════════════════
// ── Helpers ─────────────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

const VIEW_W: u32 = 600;
const VIEW_H: u32 = 100;

fn out_dir() -> std::path::PathBuf {
    let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("out")
        .join("visual");
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Build a URL bar in the given state and return a harness ready for capture.
fn build_url_bar(input: &TextInput) -> TestHarness {
    let root_style = Style::default()
        .w(VIEW_W as f64)
        .h(VIEW_H as f64)
        .bg(theme::BG);
    let mut tree = Tree::new(root_style);

    // Nav bar row
    let bar = tree.add_box(
        tree.root,
        Style::default()
            .w(VIEW_W as f64)
            .h(42.0)
            .row()
            .gap(8.0)
            .pad(8.0)
            .align(Align::Center)
            .bg(theme::BG)
            .radius(8.0),
    );

    // Reload button
    let reload = tree.add_box(
        bar,
        Style::default()
            .w(28.0)
            .h(28.0)
            .align(Align::Center)
            .justify(Justify::Center)
            .bg(theme::SURFACE0)
            .radius(6.0),
    );
    tree.tag(reload, "browser-reload");
    tree.add_text(
        reload,
        "↻",
        Style::default().font(14.0).color(theme::SUBTEXT0),
    );

    // Address bar — shared builder
    tree.build_address_bar(bar, input, &AddressBarStyle::default());

    TestHarness::from_tree(tree, (VIEW_W, VIEW_H))
}

/// Assert that a region has at least some pixels of the expected color.
fn has_color_in_region(
    cap: &Capture,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
    target: Color,
    tol: u8,
) -> bool {
    cap.count_color(x, y, w, h, target, tol) > 0
}

// ═══════════════════════════════════════════════════════════════════════════
// ── URL bar visual states ───────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn url_bar_unfocused() {
    let input = TextInput::new("anc://dashboard");
    let mut h = build_url_bar(&input);
    let cap = h.capture();
    cap.save_png(&out_dir().join("url_unfocused.png"));

    // The URL text area should have SURFACE0 background (dark)
    assert!(
        has_color_in_region(&cap, 50, 5, 400, 35, theme::SURFACE0, 10),
        "unfocused URL bar should have SURFACE0 bg"
    );
    // Should NOT have BLUE border (unfocused uses SURFACE_BRIGHT)
    // The text should be visible (not transparent)
    let center_px = cap.pixel(200, 20);
    assert_ne!(
        center_px,
        Color::TRANSPARENT,
        "center of URL bar should not be transparent"
    );
}

#[test]
fn url_bar_focused_select_all() {
    let mut input = TextInput::new("anc://dashboard");
    input.focus(); // Focuses + selects all (Chrome-like)
    let mut h = build_url_bar(&input);
    let cap = h.capture();
    cap.save_png(&out_dir().join("url_focused_select_all.png"));

    // Selection should show BLUE highlight
    assert!(
        has_color_in_region(&cap, 50, 5, 400, 35, theme::BLUE, 15),
        "focused+selected URL bar should have BLUE selection highlight"
    );
}

#[test]
fn url_bar_focused_cursor_only() {
    let mut input = TextInput::new("anc://dashboard");
    input.focused = true;
    input.cursor = 6; // After "anc://"
    input.sel = None;
    let mut h = build_url_bar(&input);
    let cap = h.capture();
    cap.save_png(&out_dir().join("url_focused_cursor.png"));

    // Should have BLUE border (focused)
    assert!(
        has_color_in_region(&cap, 45, 5, 2, 35, theme::BLUE, 15)
            || has_color_in_region(&cap, 45, 0, 510, 2, theme::BLUE, 15),
        "focused URL bar should have BLUE border"
    );
    // Cursor bar should be BLUE (1px wide)
    // Text before and after cursor should be visible
}

#[test]
fn url_bar_partial_selection() {
    let mut input = TextInput::new("anc://dashboard");
    input.focused = true;
    input.cursor = 15; // end
    input.sel = Some(6); // select "dashboard"
    let mut h = build_url_bar(&input);
    let cap = h.capture();
    cap.save_png(&out_dir().join("url_partial_selection.png"));

    // Should have both non-selected text AND blue selection
    assert!(
        has_color_in_region(&cap, 50, 5, 400, 35, theme::BLUE, 15),
        "partial selection should show BLUE region"
    );
}

#[test]
fn url_bar_after_typing() {
    let mut input = TextInput::new("anc://dashboard");
    input.focus();
    // Simulate selecting all and typing new URL
    input.insert("https://example.com");
    let mut h = build_url_bar(&input);
    let cap = h.capture();
    cap.save_png(&out_dir().join("url_after_typing.png"));

    // The old text should be gone, new text visible
    // Cursor should be at end (no selection)
    assert_eq!(input.text, "https://example.com");
    assert_eq!(input.sel, None);
    assert_eq!(input.cursor, input.text.len());
}

#[test]
fn url_bar_backspace_deletes_selection() {
    let mut input = TextInput::new("anc://dashboard");
    input.focus(); // selects all
    input.backspace(); // should delete all selected text
    assert_eq!(input.text, "");
    assert_eq!(input.cursor, 0);

    let mut h = build_url_bar(&input);
    let cap = h.capture();
    cap.save_png(&out_dir().join("url_after_backspace.png"));

    // Empty URL bar should be mostly SURFACE0 bg
    assert!(
        has_color_in_region(&cap, 60, 8, 300, 25, theme::SURFACE0, 10),
        "empty URL bar should show SURFACE0 background"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// ── State diff: unfocused → focused should change ───────────────────────
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn focus_changes_visual() {
    // Unfocused
    let input_unfocused = TextInput::new("anc://dashboard");
    let mut h1 = build_url_bar(&input_unfocused);
    let cap1 = h1.capture();
    cap1.save_png(&out_dir().join("focus_diff_before.png"));

    // Focused (selects all)
    let mut input_focused = TextInput::new("anc://dashboard");
    input_focused.focus();
    let mut h2 = build_url_bar(&input_focused);
    let cap2 = h2.capture();
    cap2.save_png(&out_dir().join("focus_diff_after.png"));

    // There should be a visual difference (border color, selection)
    let diff = cap1.diff_count(&cap2, 5);
    assert!(
        diff > 50,
        "focusing the URL bar should produce visible change, but only {} pixels differed",
        diff,
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// ── Triangle rendering ──────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn triangle_renders_with_fill() {
    let root_style = Style::default().w(200.0).h(200.0).bg(Color::BLACK);
    let mut tree = Tree::new(root_style);
    tree.layout(Size::new(200.0, 200.0));

    let mut h = TestHarness::from_tree(tree, (200, 200));

    // Manually inject a triangle into the render list
    let mut list = any_compute_core::render::RenderList::default();
    // Background
    list.push_rect(0.0, 0.0, 200.0, 200.0, Color::BLACK);
    // Red triangle covering most of the viewport
    list.push_triangle(
        [
            Point::new(100.0, 10.0),
            Point::new(10.0, 190.0),
            Point::new(190.0, 190.0),
        ],
        Color::rgb(255, 0, 0),
    );

    let (w, h_px, rgba) = h.gpu.capture(&list);
    let cap = Capture {
        width: w,
        height: h_px,
        rgba,
    };
    cap.save_png(&out_dir().join("triangle_red.png"));

    // Center of triangle should be red
    let center = cap.pixel(100, 120);
    assert!(
        center.r > 200 && center.g < 50 && center.b < 50,
        "center of triangle should be red, got {:?}",
        center
    );
    // Top-left corner should be black (outside triangle)
    let corner = cap.pixel(5, 5);
    assert!(
        corner.r < 20 && corner.g < 20 && corner.b < 20,
        "corner should be black, got {:?}",
        corner
    );
}

#[test]
fn triangle_alpha_blending() {
    // Test that triangles with alpha < 255 blend correctly
    let mut list = any_compute_core::render::RenderList::default();
    list.push_rect(0.0, 0.0, 200.0, 200.0, Color::rgb(0, 0, 255)); // blue bg
    list.push_triangle(
        [
            Point::new(100.0, 10.0),
            Point::new(10.0, 190.0),
            Point::new(190.0, 190.0),
        ],
        Color::rgba(255, 0, 0, 128), // 50% red over blue
    );

    let root_style = Style::default().w(200.0).h(200.0);
    let mut tree = Tree::new(root_style);
    tree.layout(Size::new(200.0, 200.0));
    let mut h = TestHarness::from_tree(tree, (200, 200));

    let (w, h_px, rgba) = h.gpu.capture(&list);
    let cap = Capture {
        width: w,
        height: h_px,
        rgba,
    };
    cap.save_png(&out_dir().join("triangle_alpha.png"));

    // Center should be a blend of red + blue (purplish)
    let center = cap.pixel(100, 120);
    assert!(
        center.r > 80 && center.b > 80,
        "center should blend red+blue, got {:?}",
        center
    );
    // Corner should be pure blue
    let corner = cap.pixel(5, 5);
    assert!(
        corner.b > 200 && corner.r < 30,
        "corner should be blue, got {:?}",
        corner
    );
}
