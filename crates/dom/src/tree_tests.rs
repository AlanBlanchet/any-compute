use super::*;

#[test]
fn basic_tree_layout() {
    let mut tree = Tree::new(Style::default().w(400.0).h(300.0).bg(Color::BLACK));
    let child = tree.add_box(
        tree.root,
        Style::default().w(200.0).h(100.0).bg(Color::WHITE),
    );
    tree.layout(Size::new(400.0, 300.0));
    assert_eq!(tree.slot(child).rect.size.w(), 200.0);
    assert_eq!(tree.slot(child).rect.size.h(), 100.0);
}

#[test]
fn text_node_has_intrinsic_height() {
    let mut tree = Tree::new(Style::default().w(300.0).h(200.0));
    let txt = tree.add_text(tree.root, "Hello world", Style::default().font(16.0));
    tree.layout(Size::new(300.0, 200.0));
    assert!(tree.slot(txt).rect.size.h() > 0.0);
}

#[test]
fn hit_test_finds_child() {
    let mut tree = Tree::new(Style::default().w(400.0).h(300.0));
    let child = tree.add_box(tree.root, Style::default().w(100.0).h(50.0));
    tree.layout(Size::new(400.0, 300.0));
    let hit = tree.hit_test(Point::new(50.0, 25.0));
    assert_eq!(hit, Some(child));
}

#[test]
fn click_returns_tag() {
    let mut tree = Tree::new(Style::default().w(400.0).h(300.0));
    let child = tree.add_box(tree.root, Style::default().w(100.0).h(50.0));
    tree.tag(child, "my-button");
    tree.layout(Size::new(400.0, 300.0));
    assert_eq!(tree.click(Point::new(50.0, 25.0)), Some("my-button"));
    assert_eq!(tree.click(Point::new(350.0, 250.0)), None);
}

#[test]
fn flex_grow_distributes_space() {
    let mut tree = Tree::new(Style::default().w(300.0).h(100.0).row());
    let _a = tree.add_box(tree.root, Style::default().w(50.0).h(100.0));
    let b = tree.add_box(tree.root, Style::default().h(100.0).grow(1.0));
    tree.layout(Size::new(300.0, 100.0));
    let bw = tree.slot(b).rect.size.w();
    assert!(
        bw > 200.0,
        "flex child should consume remaining space, got {}",
        bw
    );
}

#[test]
fn paint_produces_primitives() {
    let mut tree = Tree::new(Style::default().w(400.0).h(300.0).bg(Color::BLACK));
    tree.add_text(
        tree.root,
        "Hi",
        Style::default().font(14.0).color(Color::WHITE),
    );
    tree.add_bar(
        tree.root,
        0.5,
        Color::rgb(0, 255, 0),
        Style::default().h(10.0),
    );
    tree.layout(Size::new(400.0, 300.0));
    let mut list = RenderList::default();
    tree.paint(&mut list);
    assert!(
        list.len() >= 3,
        "expected ≥3 primitives, got {}",
        list.len()
    );
}

/// Regression: row-direction tab buttons inside a column sidebar must
/// stretch to the sidebar width so clicks anywhere on the row register.
#[test]
fn row_button_in_column_stretches_width() {
    // Sidebar: column, 220×600, padding 16 12
    let mut tree = Tree::new(Style::default().w(800.0).h(600.0).row());
    let sidebar = tree.add_box(
        tree.root,
        Style {
            width: Dimension::Px(220.0),
            padding: Edges::xy(12.0, 16.0),
            gap: 8.0,
            ..Style::default()
        },
    );
    // Tab button: row with height=36, no explicit width.
    let btn = tree.add_box(
        sidebar,
        Style {
            height: Dimension::Px(36.0),
            direction: Direction::Row,
            padding: Edges::xy(12.0, 0.0),
            align: Align::Center,
            ..Style::default()
        },
    );
    tree.add_text(btn, "Hardware", Style::default().font(13.0));
    tree.tag(btn, "tab-0");

    tree.layout(Size::new(800.0, 600.0));

    let btn_r = tree.slot(btn).rect;
    // Button must fill the sidebar's inner width (220 − 12 − 12 = 196).
    assert!(
        (btn_r.size.w() - 196.0).abs() < 1.0,
        "tab button should stretch to 196px, got {}",
        btn_r.size.w()
    );
    // Click in the middle of the button must return the tag.
    let mid_x = btn_r.origin.x + btn_r.size.w() / 2.0;
    let mid_y = btn_r.origin.y + btn_r.size.h() / 2.0;
    assert_eq!(tree.click(Point::new(mid_x, mid_y)), Some("tab-0"));
    // Click near the right edge (x ≈ 190) must also work.
    assert_eq!(
        tree.click(Point::new(btn_r.origin.x + 180.0, mid_y)),
        Some("tab-0")
    );
}

/// Text in a row parent must get intrinsic width, not 0.
#[test]
fn text_in_row_has_intrinsic_width() {
    let mut tree = Tree::new(Style::default().w(400.0).h(100.0).row());
    let txt = tree.add_text(tree.root, "Hello", Style::default().font(14.0));
    tree.layout(Size::new(400.0, 100.0));
    let w = tree.slot(txt).rect.size.w();
    assert!(w > 10.0, "text in row should have intrinsic width, got {w}");
}

#[test]
fn dispatch_returns_tag_chain() {
    use any_compute_core::interaction::{Button, InputEvent};
    let mut tree = Tree::new(Style::default().w(400.0).h(300.0));
    let sidebar = tree.add_box(tree.root, Style::default().w(200.0).h(300.0));
    tree.tag(sidebar, "sidebar");
    let btn = tree.add_box(sidebar, Style::default().w(100.0).h(50.0));
    tree.tag(btn, "tab-0");
    tree.layout(Size::new(400.0, 300.0));

    let result = tree.dispatch(InputEvent::PointerDown {
        pos: Point::new(50.0, 25.0),
        button: Button::Primary,
    });
    assert_eq!(result.tags, vec!["sidebar", "tab-0"]);
    assert_eq!(result.target_tag(), Some("tab-0"));
}

#[test]
fn dispatch_miss_returns_empty() {
    use any_compute_core::interaction::{Button, InputEvent};
    let mut tree = Tree::new(Style::default().w(400.0).h(300.0));
    tree.layout(Size::new(400.0, 300.0));
    let result = tree.dispatch(InputEvent::PointerDown {
        pos: Point::new(500.0, 500.0),
        button: Button::Primary,
    });
    assert!(result.tags.is_empty());
}

#[test]
fn tag_at_finds_deepest() {
    let mut tree = Tree::new(Style::default().w(400.0).h(300.0));
    let c = tree.add_box(tree.root, Style::default().w(200.0).h(100.0));
    tree.tag(c, "container");
    let inner = tree.add_box(c, Style::default().w(100.0).h(50.0));
    tree.tag(inner, "inner-btn");
    tree.layout(Size::new(400.0, 300.0));
    assert_eq!(
        tree.tag_at(Point::new(50.0, 25.0)).as_deref(),
        Some("inner-btn")
    );
}

#[test]
fn row_with_fixed_and_grow_respects_min_width() {
    // Sidebar (200px, min-width 200px) + main (flex-grow 1) in an 800px row.
    let mut root_style = Style::default().w(800.0).h(600.0);
    root_style.direction = Direction::Row;
    let mut t = Tree::new(root_style);
    let root = t.root;
    let mut sb_style = Style::default().w(200.0);
    sb_style.min_width = Dimension::Px(200.0);
    let sidebar = t.add_box(root, sb_style);
    let main = t.add_box(root, Style::default().grow(1.0));
    // Give main a child to create intrinsic width.
    t.add_text(main, "Dashboard", Style::default().font(16.0));
    t.layout(Size::new(800.0, 600.0));
    let sb_w = t.slot(sidebar).rect.size.w();
    let mn_w = t.slot(main).rect.size.w();
    assert!(sb_w >= 200.0, "sidebar should be >= 200px but was {sb_w}");
    assert!(
        (sb_w + mn_w - 800.0).abs() < 1.0,
        "sidebar ({sb_w}) + main ({mn_w}) should sum to ~800"
    );
}

/// End-to-end layout of the visual_cmp dashboard through parse_with_css.
#[test]
fn visual_cmp_layout_dimensions() {
    use crate::css::StyleSheet;
    use crate::parse::parse_with_css;

    let css = r#"
* { box-sizing: border-box; }
.root { flex-direction: row; width: 800px; height: 600px; background: #1e1e2e; }
.sidebar { width: 200px; min-width: 200px; background: #181825; padding: 16px; gap: 10px; }
.main { flex-grow: 1; }
.header { flex-direction: row; height: 48px; min-height: 48px; background: #313244;
          padding: 0px 20px; align-items: center; font-size: 16px; color: #cdd2f4; }
.content { flex-grow: 1; padding: 20px; gap: 16px; }
.cards-row { flex-direction: row; gap: 12px; }
.card { flex-grow: 1; background: #313244; border-radius: 12px; padding: 16px; gap: 8px; }
.card-title { font-size: 14px; color: #89b4fa; }
.card-body { font-size: 12px; color: #cdd2f4; }
.bar-row { gap: 6px; }
.bar-track { height: 8px; background: #333333; border-radius: 4px; }
.bar-fill-green { height: 8px; width: 70%; background: #a6e3a1; border-radius: 4px; }
.bar-fill-blue  { height: 8px; width: 45%; background: #89b4fa; border-radius: 4px; }
.bar-fill-red   { height: 8px; width: 85%; background: #f38ba8; border-radius: 4px; }
.color-swatch { width: 40px; height: 40px; border-radius: 6px; }
.nested-row { flex-direction: row; gap: 8px; }
.opacity-box { width: 60px; height: 40px; background: #89b4fa; border-radius: 6px; }
"#;
    let html = r#"
<div class="root">
  <div class="sidebar" tag="sidebar">
    <span>Sidebar</span>
  </div>
  <div class="main" tag="main">
    <div class="header" tag="header">Dashboard</div>
    <div class="content" tag="content">
      <div class="cards-row" tag="cards-row">
        <div class="card" tag="card1"><span class="card-title">Title</span><span class="card-body">Body text</span></div>
        <div class="card"><span class="card-title">Title</span><span class="card-body">Body text</span></div>
        <div class="card"><span class="card-title">Title</span><span class="card-body">Body text</span></div>
      </div>
      <div class="bar-row" tag="bar-row">
        <div class="bar-track" tag="track1"><div class="bar-fill-green" tag="fill-green"></div></div>
        <div class="bar-track"><div class="bar-fill-blue" tag="fill-blue"></div></div>
        <div class="bar-track"><div class="bar-fill-red" tag="fill-red"></div></div>
      </div>
      <div class="nested-row">
        <div class="color-swatch" tag="swatch"></div>
        <div class="color-swatch"></div>
        <div class="color-swatch"></div>
      </div>
      <div class="nested-row">
        <div class="opacity-box" tag="obox"></div>
        <div class="opacity-box"></div>
        <div class="opacity-box"></div>
      </div>
    </div>
  </div>
</div>
"#;
    let sheet = StyleSheet::parse(css);
    let mut tree = parse_with_css(html, &sheet);
    tree.layout(Size::new(800.0, 600.0));

    // Walk the tree and print all node rects for debugging.
    for (i, slot) in tree.arena.iter().enumerate() {
        let tag = slot.tag.as_deref().unwrap_or("");
        let r = &slot.rect;
        let kind = match &slot.kind {
            NodeKind::Box => "box",
            NodeKind::Text(s) => s.as_str(),
            NodeKind::Bar { .. } => "bar",
        };
        println!(
            "[{i:2}] {tag:12} {kind:20} x={:6.1} y={:6.1} w={:6.1} h={:6.1}",
            r.origin.x, r.origin.y, r.size.w(), r.size.h(),
        );
    }

    let by_tag = |t: &str| -> &Slot {
        tree.arena
            .iter()
            .find(|s| s.tag.as_deref() == Some(t))
            .unwrap_or_else(|| panic!("missing tag '{t}'"))
    };

    let sidebar = by_tag("sidebar");
    let header = by_tag("header");
    let content = by_tag("content");
    let swatch = by_tag("swatch");
    let obox = by_tag("obox");
    let track = by_tag("track1");
    let fill_green = by_tag("fill-green");
    let fill_blue = by_tag("fill-blue");
    let fill_red = by_tag("fill-red");
    let card = by_tag("card1");

    println!("\n=== Key dimensions ===");
    println!("sidebar: w={:.1}", sidebar.rect.size.w());
    println!("header:  h={:.1}", header.rect.size.h());
    println!(
        "content: w={:.1} h={:.1}",
        content.rect.size.w(), content.rect.size.h()
    );
    println!(
        "swatch:  w={:.1} h={:.1}",
        swatch.rect.size.w(), swatch.rect.size.h()
    );
    println!(
        "opacity: w={:.1} h={:.1}",
        obox.rect.size.w(), obox.rect.size.h()
    );
    println!(
        "card1:   w={:.1} h={:.1}",
        card.rect.size.w(), card.rect.size.h()
    );
    println!(
        "bar-track w={:.1}, fills: green={:.1} ({:.1}%) blue={:.1} ({:.1}%) red={:.1} ({:.1}%)",
        track.rect.size.w(),
        fill_green.rect.size.w(),
        fill_green.rect.size.w() / track.rect.size.w() * 100.0,
        fill_blue.rect.size.w(),
        fill_blue.rect.size.w() / track.rect.size.w() * 100.0,
        fill_red.rect.size.w(),
        fill_red.rect.size.w() / track.rect.size.w() * 100.0,
    );

    // Assertions.
    assert!(
        (sidebar.rect.size.w() - 200.0).abs() < 1.0,
        "sidebar should be 200px, got {:.1}",
        sidebar.rect.size.w()
    );
    assert!(
        (header.rect.size.h() - 48.0).abs() < 1.0,
        "header should be 48px, got {:.1}",
        header.rect.size.h()
    );
    assert!(
        (swatch.rect.size.w() - 40.0).abs() < 1.0,
        "swatch should be 40px, got {:.1}",
        swatch.rect.size.w()
    );
    assert!(
        (obox.rect.size.w() - 60.0).abs() < 1.0,
        "opacity-box should be 60px, got {:.1}",
        obox.rect.size.w()
    );
    assert!(
        (fill_green.rect.size.w() / track.rect.size.w() - 0.70).abs() < 0.02,
        "green fill should be 70%, got {:.1}%",
        fill_green.rect.size.w() / track.rect.size.w() * 100.0
    );
}

// ── Hover / Active restyle tests ────────────────────────────────────

fn tree_with_hover_css() -> Tree {
    use crate::css::StyleSheet;
    use crate::parse::parse_with_css;

    let css = r#"
.btn { width: 100px; height: 50px; background: #333333; }
.btn:hover { background: #ff0000; }
.btn:active { background: #00ff00; }
.container { width: 400px; height: 300px; }
"#;
    let html = r#"<div class="container"><div class="btn" tag="btn"></div></div>"#;
    let sheet = StyleSheet::parse(css);
    let mut tree = parse_with_css(html, &sheet);
    tree.layout(Size::new(400.0, 300.0));
    tree
}

#[test]
fn hover_changes_background_color() {
    let mut tree = tree_with_hover_css();
    let btn_id = tree
        .arena
        .iter()
        .position(|s| s.tag.as_deref() == Some("btn"))
        .unwrap();
    let btn = NodeId(btn_id);

    // Before hover: base style
    let before = tree.slot(btn).style.background;
    assert_eq!(before, Color::rgb(0x33, 0x33, 0x33), "base should be #333");

    // Hover over the button
    tree.dispatch(InputEvent::PointerMove {
        pos: Point::new(50.0, 25.0),
    });

    let after = tree.slot(btn).style.background;
    assert_eq!(
        after,
        Color::rgb(0xff, 0x00, 0x00),
        "hover should be #ff0000"
    );

    // Move away — should revert
    tree.dispatch(InputEvent::PointerMove {
        pos: Point::new(350.0, 250.0),
    });

    let reverted = tree.slot(btn).style.background;
    assert_eq!(
        reverted,
        Color::rgb(0x33, 0x33, 0x33),
        "should revert to base after unhover"
    );
}

#[test]
fn active_changes_background_color() {
    use any_compute_core::interaction::Button;
    let mut tree = tree_with_hover_css();
    let btn_id = tree
        .arena
        .iter()
        .position(|s| s.tag.as_deref() == Some("btn"))
        .unwrap();
    let btn = NodeId(btn_id);

    // Pointer down → active
    tree.dispatch(InputEvent::PointerDown {
        pos: Point::new(50.0, 25.0),
        button: Button::Primary,
    });
    let active_bg = tree.slot(btn).style.background;
    assert_eq!(
        active_bg,
        Color::rgb(0x00, 0xff, 0x00),
        ":active should be #00ff00"
    );

    // Pointer up → revert (hover still applies since cursor is on btn)
    tree.dispatch(InputEvent::PointerUp {
        pos: Point::new(50.0, 25.0),
        button: Button::Primary,
    });
    let after_up = tree.slot(btn).style.background;
    // Note: dispatch doesn't auto-hover on up, so this snaps to base
    assert_ne!(
        after_up,
        Color::rgb(0x00, 0xff, 0x00),
        "should no longer be :active"
    );
}

#[test]
fn hover_state_tracks_correctly() {
    let mut tree = tree_with_hover_css();
    let btn_id = tree
        .arena
        .iter()
        .position(|s| s.tag.as_deref() == Some("btn"))
        .unwrap();
    let btn = NodeId(btn_id);

    assert!(!tree.slot(btn).hovered, "should start unhovered");

    tree.dispatch(InputEvent::PointerMove {
        pos: Point::new(50.0, 25.0),
    });
    assert!(
        tree.slot(btn).hovered,
        "should be hovered after move onto it"
    );

    tree.dispatch(InputEvent::PointerMove {
        pos: Point::new(350.0, 250.0),
    });
    assert!(
        !tree.slot(btn).hovered,
        "should be unhovered after move off"
    );
}

// ── Universal selector tests ────────────────────────────────────────

#[test]
fn universal_selector_applies_box_sizing() {
    use crate::css::StyleSheet;
    use crate::parse::parse_with_css;

    let css = "* { box-sizing: border-box; }\n.box { width: 100px; height: 100px; padding: 10px; background: #888; }";
    let html = r#"<div class="box" tag="b"></div>"#;
    let sheet = StyleSheet::parse(css);
    let mut tree = parse_with_css(html, &sheet);
    tree.layout(Size::new(400.0, 300.0));

    let b = tree
        .arena
        .iter()
        .find(|s| s.tag.as_deref() == Some("b"))
        .unwrap();
    assert_eq!(
        b.style.box_sizing,
        BoxSizing::BorderBox,
        "* should set border-box"
    );
    // With border-box, total width stays 100px (content = 80, padding = 10+10)
    assert!(
        (b.rect.size.w() - 100.0).abs() < 1.0,
        "border-box width should be 100px, got {:.1}",
        b.rect.size.w()
    );
}

// ── Animation tests ─────────────────────────────────────────────────

#[test]
fn start_animations_creates_active_animations() {
    use crate::css::StyleSheet;
    use crate::parse::parse_with_css;

    let css = r#"
@keyframes pulse { from { opacity: 1; } to { opacity: 0.5; } }
.animated { width: 100px; height: 100px; animation: pulse 2s ease-in-out infinite; }
.root { width: 400px; height: 300px; }
"#;
    let html = r#"<div class="root"><div class="animated" tag="anim"></div></div>"#;
    let sheet = StyleSheet::parse(css);
    let mut tree = parse_with_css(html, &sheet);
    tree.layout(Size::new(400.0, 300.0));

    // Before start_animations — no active animations
    let anim_id = tree
        .arena
        .iter()
        .position(|s| s.tag.as_deref() == Some("anim"))
        .unwrap();
    assert!(tree.arena[anim_id].animations.is_empty());

    tree.start_animations();
    assert_eq!(tree.arena[anim_id].animations.len(), 1);
    assert_eq!(tree.arena[anim_id].animations[0].name, "pulse");
}

#[test]
fn tick_advances_animations() {
    use crate::css::StyleSheet;
    use crate::parse::parse_with_css;

    let css = r#"
@keyframes fade { from { opacity: 1; } to { opacity: 0.2; } }
.anim { width: 100px; height: 100px; animation: fade 1s linear; }
.root { width: 400px; height: 300px; }
"#;
    let html = r#"<div class="root"><div class="anim" tag="a"></div></div>"#;
    let sheet = StyleSheet::parse(css);
    let mut tree = parse_with_css(html, &sheet);
    tree.layout(Size::new(400.0, 300.0));
    tree.start_animations();

    let idx = tree
        .arena
        .iter()
        .position(|s| s.tag.as_deref() == Some("a"))
        .unwrap();
    let _before = tree.arena[idx].style.opacity;

    // Tick halfway
    let result = tree.tick(0.5);
    assert!(result.active, "should still be running at t=0.5");

    // Tick past end
    tree.tick(0.6);
    let finished = !tree.has_active_animations();
    assert!(finished, "animation should finish after total 1.1s");
}

// ── Element + class identity stored on parse ────────────────────────

#[test]
fn parse_stores_element_and_class_list() {
    use crate::css::StyleSheet;
    use crate::parse::parse_with_css;

    let css = ".btn { width: 100px; height: 50px; }";
    let html =
        r#"<div class="container main"><button class="btn primary" tag="b">Click</button></div>"#;
    let sheet = StyleSheet::parse(css);
    let tree = parse_with_css(html, &sheet);

    // Root → div.container.main
    assert_eq!(tree.slot(tree.root).element, "div");
    assert_eq!(tree.slot(tree.root).class_list, vec!["container", "main"]);

    // button.btn.primary
    let btn = tree
        .arena
        .iter()
        .find(|s| s.tag.as_deref() == Some("b"))
        .unwrap();
    assert_eq!(btn.element, "button");
    assert_eq!(btn.class_list, vec!["btn", "primary"]);
}

// ── Transition interpolation tests ──────────────────────────────────

#[test]
fn transition_interpolates_style_on_hover() {
    use crate::css::StyleSheet;
    use crate::parse::parse_with_css;

    let css = r#"
.box { width: 200px; height: 100px; background: #000000; transition: all 1s linear; }
.box:hover { background: #ffffff; }
.root { width: 400px; height: 300px; }
"#;
    let html = r#"<div class="root"><div class="box" tag="b"></div></div>"#;
    let sheet = StyleSheet::parse(css);
    let mut tree = parse_with_css(html, &sheet);
    tree.layout(Size::new(400.0, 300.0));

    let idx = tree
        .arena
        .iter()
        .position(|s| s.tag.as_deref() == Some("b"))
        .unwrap();
    let id = NodeId(idx);

    // Base: background is black
    assert_eq!(tree.slot(id).style.background, Color::rgb(0, 0, 0));

    // Hover → creates transition
    tree.dispatch(InputEvent::PointerMove {
        pos: Point::new(100.0, 50.0),
    });
    assert!(
        !tree.arena[idx].transitions.is_empty(),
        "hover should create transition"
    );

    // Tick 50% through the 1s transition
    tree.tick(0.5);
    let mid = tree.slot(id).style.background;
    // Midpoint should be approximately gray (half between black and white)
    assert!(
        mid.r > 100 && mid.r < 200,
        "mid-transition red should be ~128, got {}",
        mid.r
    );

    // Tick to completion
    tree.tick(0.6);
    let done = tree.slot(id).style.background;
    assert_eq!(
        done,
        Color::rgb(255, 255, 255),
        "after transition should be white"
    );
    assert!(
        tree.arena[idx].transitions.is_empty(),
        "transition should be removed"
    );
}

#[test]
fn transition_reverse_hover_starts_from_current() {
    use crate::css::StyleSheet;
    use crate::parse::parse_with_css;

    let css = r#"
.box { width: 200px; height: 100px; background: #000000; transition: all 0.4s linear; }
.box:hover { background: #ffffff; }
.root { width: 400px; height: 300px; }
"#;
    let html = r#"<div class="root"><div class="box" tag="b"></div></div>"#;
    let sheet = StyleSheet::parse(css);
    let mut tree = parse_with_css(html, &sheet);
    tree.layout(Size::new(400.0, 300.0));

    let idx = tree
        .arena
        .iter()
        .position(|s| s.tag.as_deref() == Some("b"))
        .unwrap();
    let id = NodeId(idx);

    // Hover → transition starts
    tree.dispatch(InputEvent::PointerMove {
        pos: Point::new(100.0, 50.0),
    });
    tree.tick(0.2); // 50% through 0.4s transition

    let mid = tree.slot(id).style.background;
    // Should be somewhere between black and white
    assert!(
        mid.r > 50,
        "mid-transition should be past black, got r={}",
        mid.r
    );

    // Unhover → reverse transition from current interpolated point
    tree.dispatch(InputEvent::PointerMove {
        pos: Point::new(350.0, 250.0),
    });
    assert!(
        !tree.arena[idx].transitions.is_empty(),
        "unhover should create reverse transition"
    );

    // The "from" of the new transition should be the current mid-color
    let from_r = tree.arena[idx].transitions[0].from.background.r;
    assert!(
        from_r > 50,
        "reverse from should start from mid-color, got r={}",
        from_r
    );

    // The "to" should be the base black
    let to_bg = tree.arena[idx].transitions[0].to.background;
    assert_eq!(
        to_bg,
        Color::rgb(0, 0, 0),
        "reverse target should be base color"
    );
}

#[test]
fn no_transition_snaps_immediately() {
    use crate::css::StyleSheet;
    use crate::parse::parse_with_css;

    let css = r#"
.box { width: 200px; height: 100px; background: #333333; }
.box:hover { background: #ff0000; }
.root { width: 400px; height: 300px; }
"#;
    let html = r#"<div class="root"><div class="box" tag="b"></div></div>"#;
    let sheet = StyleSheet::parse(css);
    let mut tree = parse_with_css(html, &sheet);
    tree.layout(Size::new(400.0, 300.0));

    let idx = tree
        .arena
        .iter()
        .position(|s| s.tag.as_deref() == Some("b"))
        .unwrap();
    let id = NodeId(idx);

    // Hover without transition spec → should snap immediately
    tree.dispatch(InputEvent::PointerMove {
        pos: Point::new(100.0, 50.0),
    });
    assert!(
        tree.arena[idx].transitions.is_empty(),
        "no transition spec → no transition"
    );
    assert_eq!(
        tree.slot(id).style.background,
        Color::rgb(0xff, 0x00, 0x00),
        "should snap to hover color"
    );
}
