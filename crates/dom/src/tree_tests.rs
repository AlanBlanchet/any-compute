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
            r.origin.x,
            r.origin.y,
            r.size.w(),
            r.size.h(),
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
        content.rect.size.w(),
        content.rect.size.h()
    );
    println!(
        "swatch:  w={:.1} h={:.1}",
        swatch.rect.size.w(),
        swatch.rect.size.h()
    );
    println!(
        "opacity: w={:.1} h={:.1}",
        obox.rect.size.w(),
        obox.rect.size.h()
    );
    println!(
        "card1:   w={:.1} h={:.1}",
        card.rect.size.w(),
        card.rect.size.h()
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

// ── Hover state ─────────────────────────────────────────────────────────

#[test]
fn hover_sets_and_clears_state() {
    let mut tree = Tree::new(Style::default().w(400.0).h(300.0));
    let child = tree.add_box(tree.root, Style::default().w(100.0).h(50.0));
    tree.layout(Size::new(400.0, 300.0));

    assert!(!tree.slot(child).hovered);
    tree.dispatch(InputEvent::PointerMove {
        pos: Point::new(50.0, 25.0),
    });
    assert!(
        tree.slot(child).hovered,
        "child should be hovered after PointerMove"
    );

    // Move away
    tree.dispatch(InputEvent::PointerMove {
        pos: Point::new(350.0, 250.0),
    });
    assert!(
        !tree.slot(child).hovered,
        "child should lose hover when pointer leaves"
    );
}

// ── Active state ────────────────────────────────────────────────────────

#[test]
fn active_sets_on_pointer_down_clears_on_up() {
    use any_compute_core::interaction::Button;
    let mut tree = Tree::new(Style::default().w(400.0).h(300.0));
    let child = tree.add_box(tree.root, Style::default().w(100.0).h(50.0));
    tree.layout(Size::new(400.0, 300.0));

    tree.dispatch(InputEvent::PointerDown {
        pos: Point::new(50.0, 25.0),
        button: Button::Primary,
    });
    assert!(
        tree.slot(child).active,
        "child should be active after PointerDown"
    );

    tree.dispatch(InputEvent::PointerUp {
        pos: Point::new(50.0, 25.0),
        button: Button::Primary,
    });
    assert!(
        !tree.slot(child).active,
        "child should lose active after PointerUp"
    );
}

// ── Focus / blur ────────────────────────────────────────────────────────

#[test]
fn focus_and_blur() {
    let mut tree = Tree::new(Style::default().w(400.0).h(300.0));
    let child = tree.add_box(tree.root, Style::default().w(100.0).h(50.0));
    tree.layout(Size::new(400.0, 300.0));

    tree.focus(child);
    assert!(tree.slot(child).focused);
    assert_eq!(tree.focused, Some(child));

    tree.blur();
    assert!(!tree.slot(child).focused);
    assert_eq!(tree.focused, None);
}

// ── Editable text input ─────────────────────────────────────────────────

#[test]
fn editable_text_insert() {
    let mut tree = Tree::new(Style::default().w(400.0).h(300.0));
    let input = tree.add_box(tree.root, Style::default().w(200.0).h(30.0));
    tree.set_editable(input, "hello");
    tree.layout(Size::new(400.0, 300.0));

    tree.focus(input);
    tree.dispatch(InputEvent::TextInput {
        text: " world".into(),
    });
    assert_eq!(tree.value(input), Some("hello world"));
}

#[test]
fn editable_backspace() {
    let mut tree = Tree::new(Style::default().w(400.0).h(300.0));
    let input = tree.add_box(tree.root, Style::default().w(200.0).h(30.0));
    tree.set_editable(input, "abc");
    tree.layout(Size::new(400.0, 300.0));

    tree.focus(input);
    tree.dispatch(InputEvent::KeyDown {
        key: "Backspace".into(),
        modifiers: Modifiers::default(),
    });
    assert_eq!(tree.value(input), Some("ab"));
}

// ── Scroll ──────────────────────────────────────────────────────────────

#[test]
fn scroll_overflow_container() {
    let mut tree = Tree::new(Style::default().w(400.0).h(300.0));
    let container = tree.add_box(
        tree.root,
        Style {
            overflow: Overflow::Scroll,
            ..Style::default().w(200.0).h(100.0)
        },
    );
    // Tall child to create scrollable content
    tree.add_box(container, Style::default().w(200.0).h(500.0));
    tree.layout(Size::new(400.0, 300.0));

    tree.scroll(Point::new(100.0, 50.0), Point::new(0.0, -50.0));
    assert!(
        tree.slot(container).scroll.y > 0.0,
        "scroll offset should be positive after scrolling down"
    );
}

// ── Dispatch restyled flag ──────────────────────────────────────────────

#[test]
fn dispatch_hover_reports_restyled() {
    let mut tree = Tree::new(Style::default().w(400.0).h(300.0));
    tree.add_box(tree.root, Style::default().w(100.0).h(50.0));
    tree.layout(Size::new(400.0, 300.0));

    let r = tree.dispatch(InputEvent::PointerMove {
        pos: Point::new(50.0, 25.0),
    });
    assert!(r.restyled, "first hover should report restyled");

    // Same position again — no state change
    let r2 = tree.dispatch(InputEvent::PointerMove {
        pos: Point::new(50.0, 25.0),
    });
    assert!(!r2.restyled, "repeated hover should not restyle");
}

// ── Graphable tests ─────────────────────────────────────────────────────

#[test]
fn graphable_arena_structure_labels_and_render() {
    use any_compute_core::render::Renderable;
    use any_compute_core::visual::Graphable;

    // Build a tree with mixed node types
    let mut tree = Tree::new(Style::default().w(400.0).h(300.0));
    let sidebar = tree.add_box(tree.root, Style::default().w(200.0).h(300.0));
    tree.tag(sidebar, "sidebar");
    tree.add_text(sidebar, "Hello", Style::default().font(14.0));
    let nested = tree.add_box(sidebar, Style::default().w(100.0).h(100.0));
    tree.add_box(nested, Style::default().w(50.0).h(50.0));
    tree.add_bar(
        tree.root,
        0.5,
        Color::rgb(0, 200, 0),
        Style::default().h(10.0),
    );
    tree.layout(Size::new(400.0, 300.0));

    let g = tree.to_graph();
    // root's children: sidebar + bar = 2 top-level nodes
    // sidebar's children (Hello, nested) live in its sub-graph
    assert_eq!(g.len(), 2, "top-level graph = root children");
    // sidebar node should have a sub-graph with 2 children
    let sidebar_node = g.node(0);
    assert!(sidebar_node.children().is_some(), "sidebar has sub-graph");
    assert_eq!(sidebar_node.children().unwrap().len(), 2);

    // Labels from render include tagged nodes (top-level only)
    let mut list = RenderList::default();
    g.render(&mut list, &());
    let texts: Vec<String> = list
        .iter()
        .filter_map(|p| {
            if let any_compute_core::render::Primitive::Text { content, .. } = p {
                Some(content.clone())
            } else {
                None
            }
        })
        .collect();
    assert!(texts.iter().any(|t| t.contains("sidebar")));
    // "Hello" is in sidebar sub-graph, so verify it exists there
    let sub = sidebar_node.children().unwrap();
    let sub_labels: Vec<_> = (0..sub.len())
        .map(|i| sub.node(i).label().to_string())
        .collect();
    assert!(sub_labels.iter().any(|l| l.contains("Hello")));

    // render_graph blanket method works
    let prims = tree.render_graph();
    assert!(prims.len() > 20, "render_graph: {}", prims.len());
}

// ── Visual regression: text width (Unicode / emoji) ─────────────────────

#[test]
fn text_width_counts_chars_not_bytes() {
    // Emoji characters are 4 bytes in UTF-8 but should count as 1 character.
    let s = Style::default();
    let ascii_w = s.text_width("A");
    let emoji_w = s.text_width("\u{1F310}"); // 🌐 (4 bytes)
    assert!(
        (emoji_w - ascii_w).abs() < 0.01,
        "emoji width ({emoji_w}) should equal single char width ({ascii_w})"
    );

    // Multi-char string: "AB" vs "🌐🎮" — both 2 chars
    let two_ascii = s.text_width("AB");
    let two_emoji = s.text_width("\u{1F310}\u{1F3AE}");
    assert!(
        (two_emoji - two_ascii).abs() < 0.01,
        "two emoji ({two_emoji}) should equal two chars ({two_ascii})"
    );
}

#[test]
fn flex_center_text_pixel() {
    // A 100x100 box with align-items:center + justify-content:center containing
    // a text child — the text rect should be centered, not left-aligned.
    let mut tree = Tree::new(
        Style::default()
            .w(100.0)
            .h(100.0)
            .bg(Color::rgb(30, 30, 46))
            .align(Align::Center)
            .justify(Justify::Center),
    );
    let txt = tree.add_text(
        tree.root,
        "X",
        Style {
            font_size: 14.0,
            color: Color::WHITE,
            ..Style::default()
        },
    );
    tree.layout(Size::new(100.0, 100.0));
    let txt_rect = tree.slot(txt).rect;
    // Intrinsic width of "X" at font_size 14 ≈ 14 * CHAR_WIDTH_RATIO ≈ 8.4
    // Centered in 100px → x should be ≈ (100 - 8.4) / 2 ≈ 45.8
    assert!(
        txt_rect.origin.x > 30.0,
        "text x ({}) should be centered (>30), not left-aligned",
        txt_rect.origin.x
    );
    assert!(
        txt_rect.origin.x < 60.0,
        "text x ({}) should be near center (<60)",
        txt_rect.origin.x
    );
}

#[test]
fn flex_center_emoji_pixel() {
    // Same centering test but with an emoji — the emoji should be centered,
    // not pushed left by inflated byte-length width.
    let mut tree = Tree::new(
        Style::default()
            .w(100.0)
            .h(100.0)
            .bg(Color::rgb(30, 30, 46))
            .align(Align::Center)
            .justify(Justify::Center),
    );
    let txt = tree.add_text(
        tree.root,
        "\u{1F310}", // 🌐
        Style {
            font_size: 14.0,
            color: Color::WHITE,
            ..Style::default()
        },
    );
    tree.layout(Size::new(100.0, 100.0));
    let txt_rect = tree.slot(txt).rect;
    // Single char at font_size 14 → width ≈ 8.4, centered at ≈ 45.8
    assert!(
        txt_rect.origin.x > 30.0,
        "emoji x ({}) should be centered (>30), not left-aligned at byte-width",
        txt_rect.origin.x
    );
    assert!(
        txt_rect.origin.x < 60.0,
        "emoji x ({}) should be near center (<60)",
        txt_rect.origin.x
    );

    // Compare with ASCII: positions should be identical
    let mut tree2 = Tree::new(
        Style::default()
            .w(100.0)
            .h(100.0)
            .align(Align::Center)
            .justify(Justify::Center),
    );
    let txt2 = tree2.add_text(
        tree2.root,
        "X",
        Style {
            font_size: 14.0,
            ..Style::default()
        },
    );
    tree2.layout(Size::new(100.0, 100.0));
    let txt2_rect = tree2.slot(txt2).rect;
    assert!(
        (txt_rect.origin.x - txt2_rect.origin.x).abs() < 1.0,
        "emoji x ({}) should match ASCII x ({})",
        txt_rect.origin.x,
        txt2_rect.origin.x
    );
}

#[test]
fn page_load_external_html_no_crash() {
    // Loading complex HTML (from external sites) should not crash.
    let complex_html = r#"
        <html>
        <head><style>
            body { margin: 0; font-family: sans-serif; }
            .container { display: flex; gap: 10px; }
            .box { width: 50px; height: 50px; background: red; border-radius: 5px; }
        </style></head>
        <body>
            <div class="container">
                <div class="box">1</div>
                <div class="box">2</div>
                <div class="box">3</div>
            </div>
            <script>
                var x = document.getElementById("nonexistent");
            </script>
        </body>
        </html>
    "#;
    let page = crate::page::Page::load(complex_html);
    assert!(
        page.tree().arena.len() > 3,
        "page should have multiple nodes"
    );

    // Layout should not crash
    let mut page = page;
    page.layout(Size::new(800.0, 600.0));

    // Paint should not crash
    let mut list = RenderList::default();
    page.paint(&mut list);
    assert!(list.len() > 0, "page should produce render primitives");
}

#[test]
fn page_load_malformed_html_no_crash() {
    // Malformed HTML (unclosed tags, bad attributes) should parse without panic
    let bad_html = r#"
        <div><span>unclosed
        <div class=no-quotes>
        <img src="" />
        <div style="background: invalid;">text</div>
        <script>var x = 1 +</script>
        </div>
    "#;
    let page = crate::page::Page::load(bad_html);
    // Just need it to not crash
    let mut page = page;
    page.layout(Size::new(400.0, 300.0));
    let mut list = RenderList::default();
    page.paint(&mut list);
}

#[test]
fn graphable_page_tree() {
    use any_compute_core::visual::Graphable;
    // A loaded page's tree should produce a valid graph via the Graphable trait
    let html = r#"
        <div class="root">
            <div class="header">Title</div>
            <div class="content">
                <div class="card">Card 1</div>
                <div class="card">Card 2</div>
            </div>
        </div>
    "#;
    let page = crate::page::Page::load(html);
    let graph = page.tree().to_graph();
    // Graph should have nodes for root's children (header + content = 2)
    assert!(
        graph.len() >= 2,
        "graph should have >=2 nodes, got {}",
        graph.len()
    );
    // render_graph should produce primitives
    let prims = page.tree().render_graph();
    assert!(prims.len() > 5, "render_graph should produce primitives");
}

// ═══════════════════════════════════════════════════════════════════════════
// ── Cursor behavior tests ───────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn cursor_pointer_on_styled_element() {
    let mut tree = Tree::new(Style::default().w(300.0).h(200.0));
    let btn = tree.add_box(
        tree.root,
        Style::default()
            .w(100.0)
            .h(40.0)
            .bg(Color::WHITE)
            .cursor(Cursor::Pointer),
    );
    tree.tag(btn, "btn");
    tree.layout(Size::new(300.0, 200.0));

    assert_eq!(tree.cursor_at(Point::new(50.0, 20.0)), Cursor::Pointer);
    assert_eq!(tree.cursor_at(Point::new(250.0, 150.0)), Cursor::Default);
}

#[test]
fn cursor_inherits_from_parent() {
    let mut tree = Tree::new(Style::default().w(300.0).h(200.0));
    let parent = tree.add_box(
        tree.root,
        Style::default().w(200.0).h(100.0).cursor(Cursor::Pointer),
    );
    let _child = tree.add_box(
        parent,
        Style::default().w(80.0).h(30.0).bg(Color::rgb(255, 0, 0)),
    );
    tree.layout(Size::new(300.0, 200.0));

    // Child has no explicit cursor → inherits Pointer from parent
    assert_eq!(tree.cursor_at(Point::new(40.0, 15.0)), Cursor::Pointer);
}

#[test]
fn cursor_child_overrides_parent() {
    let mut tree = Tree::new(Style::default().w(300.0).h(200.0));
    let parent = tree.add_box(
        tree.root,
        Style::default().w(200.0).h(100.0).cursor(Cursor::Pointer),
    );
    let _child = tree.add_box(
        parent,
        Style::default().w(80.0).h(30.0).cursor(Cursor::Text),
    );
    tree.layout(Size::new(300.0, 200.0));

    // Child's Text cursor overrides parent's Pointer
    assert_eq!(tree.cursor_at(Point::new(40.0, 15.0)), Cursor::Text);
}

#[test]
fn cursor_css_parsed() {
    use crate::css::StyleSheet;
    use crate::parse::parse_with_css;
    let css = ".clickable { cursor: pointer; width: 100px; height: 40px; background: red; }";
    let html = r#"<div class="clickable">Click me</div>"#;
    let sheet = StyleSheet::parse(css);
    let mut tree = parse_with_css(html, &sheet);
    tree.layout(Size::new(300.0, 200.0));

    assert_eq!(tree.cursor_at(Point::new(50.0, 20.0)), Cursor::Pointer);
}

// ═══════════════════════════════════════════════════════════════════════════
// ── Click routing with nested/overlapping elements ──────────────────────
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn click_routes_to_topmost_child() {
    let mut tree = Tree::new(Style::default().w(300.0).h(200.0));
    let outer = tree.add_box(tree.root, Style::default().w(200.0).h(100.0));
    tree.tag(outer, "outer");
    let inner = tree.add_box(outer, Style::default().w(80.0).h(40.0));
    tree.tag(inner, "inner");
    tree.layout(Size::new(300.0, 200.0));

    // Click inside inner → inner tag
    assert_eq!(tree.click(Point::new(40.0, 20.0)), Some("inner"));
    // Click outside inner but inside outer → outer tag
    assert_eq!(tree.click(Point::new(150.0, 70.0)), Some("outer"));
}

// ═══════════════════════════════════════════════════════════════════════════
// ── Hover state tests ───────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn hover_sets_state_on_target() {
    use any_compute_core::interaction::InputEvent;
    let mut tree = Tree::new(Style::default().w(300.0).h(200.0));
    let child = tree.add_box(tree.root, Style::default().w(100.0).h(50.0));
    tree.layout(Size::new(300.0, 200.0));

    tree.dispatch(InputEvent::PointerMove {
        pos: Point::new(50.0, 25.0),
    });
    assert!(tree.slot(child).hovered);

    // Move away
    tree.dispatch(InputEvent::PointerMove {
        pos: Point::new(250.0, 150.0),
    });
    assert!(!tree.slot(child).hovered);
}

// ═══════════════════════════════════════════════════════════════════════════
// ── Scroll behavior tests ───────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn scroll_moves_content() {
    // Container 100x50 with overflow:scroll, child 100x200 (overflows)
    let mut container_s = Style::default().w(100.0).h(50.0).bg(Color::BLACK);
    container_s.overflow = Overflow::Scroll;
    let mut tree = Tree::new(Style::default().w(200.0).h(200.0).bg(Color::WHITE));
    let container = tree.add_box(tree.root, container_s);
    let child = tree.add_box(container, Style::default().w(100.0).h(200.0));
    tree.tag(child, "tall-child");
    tree.layout(Size::new(200.0, 200.0));

    // Before scroll: child top should be reachable
    assert_eq!(
        tree.tag_at(Point::new(50.0, 10.0)),
        Some("tall-child".into())
    );

    // Scroll down — content shifts up, revealing lower portions
    tree.scroll(Point::new(50.0, 25.0), Point::new(0.0, 30.0));
    tree.layout(Size::new(200.0, 200.0));

    // After scroll: hit test still works within container bounds
    assert_eq!(
        tree.tag_at(Point::new(50.0, 10.0)),
        Some("tall-child".into())
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// ── Sidebar centering test ──────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn sidebar_icons_centered() {
    use crate::css::StyleSheet;
    let css = r#"
        .sidebar {
            width: 44px; min-width: 44px; box-sizing: border-box;
            background: #111; padding: 8px 4px; gap: 4px;
            flex-shrink: 0; overflow: hidden; align-items: center;
        }
        .tab-btn {
            width: 32px; height: 32px; min-width: 32px; min-height: 32px;
            border-radius: 8px; align-items: center; justify-content: center;
            flex-shrink: 0; cursor: pointer;
        }
        .brand-icon {
            width: 24px; height: 24px; min-width: 24px; min-height: 24px;
            background: blue; border-radius: 6px; flex-shrink: 0;
        }
        .row { flex-direction: row; }
        .grow { flex-grow: 1; }
    "#;
    let sheet = StyleSheet::parse(css);
    let mut t = Tree::new(sheet.class("row").w(200.0).h(200.0));
    let sb = t.add_box(t.root, sheet.class("sidebar"));
    let brand = t.add_box(sb, sheet.class("brand-icon"));
    t.tag(brand, "brand");
    for i in 0..4 {
        let btn = t.add_box(sb, sheet.class("tab-btn"));
        t.tag(btn, &format!("tab-{i}"));
        // Add text child like showcase does
        t.add_text(btn, "X", Style::default().font(14.0));
    }
    let _main = t.add_box(t.root, Style::default().grow(1.0));
    t.layout(Size::new(200.0, 200.0));

    // Sidebar rect
    let sb_rect = t.slot(sb).rect;

    let brand_rect = t.slot(brand).rect;
    let brand_center = brand_rect.origin.x + brand_rect.size.w() / 2.0;
    let sidebar_center = sb_rect.origin.x + sb_rect.size.w() / 2.0;
    assert!(
        (brand_center - sidebar_center).abs() < 1.0,
        "brand NOT centered: icon_center={brand_center}, sidebar_center={sidebar_center}, \
         brand_x={}, brand_w={}, sidebar_x={}, sidebar_w={}",
        brand_rect.origin.x,
        brand_rect.size.w(),
        sb_rect.origin.x,
        sb_rect.size.w()
    );

    for i in 0..4u32 {
        let tag = format!("tab-{i}");
        let r = t.tagged_rect(&tag).expect(&tag);
        let btn_center = r.origin.x + r.size.w() / 2.0;
        assert!(
            (btn_center - sidebar_center).abs() < 1.0,
            "{tag}: NOT centered: btn_center={btn_center}, sidebar_center={sidebar_center}, \
             btn_x={}, btn_w={}, sidebar_x={}, sidebar_w={}",
            r.origin.x,
            r.size.w(),
            sb_rect.origin.x,
            sb_rect.size.w()
        );
    }
}

// ── ToDom / add_element tests ───────────────────────────────────────────────

#[test]
fn to_dom_button_has_ua_defaults() {
    let el = "button".to_dom();
    assert_eq!(el.tag, HtmlTag::Button);
    assert!(matches!(el.kind, NodeKind::Box));
    // UA defaults from ua.css: cursor:pointer, padding, centered
    assert_eq!(el.style.cursor, Cursor::Pointer);
    assert_eq!(el.style.justify, Justify::Center);
    assert_eq!(el.style.align, Align::Center);
    assert_eq!(el.style.direction, Direction::Row);
    assert!(el.style.border_width > 0.0 || el.style.border_top_width > 0.0);
}

#[test]
fn to_dom_heading_has_font_size() {
    let h1 = "h1".to_dom();
    assert_eq!(h1.tag, HtmlTag::H1);
    assert!(matches!(h1.kind, NodeKind::Text(_)));
    assert!((h1.style.font_size - 32.0).abs() < 0.1);
    assert_eq!(h1.style.font_weight, FontWeight::BOLD);

    let h2 = "h2".to_dom();
    assert!((h2.style.font_size - 24.0).abs() < 0.1);
}

#[test]
fn to_dom_unknown_tag_is_div() {
    let el = "custom-widget".to_dom();
    assert_eq!(el.tag, HtmlTag::Custom);
    assert!(matches!(el.kind, NodeKind::Box));
}

#[test]
fn add_element_creates_node_with_ua_style() {
    let mut tree = Tree::new(Style::default().w(400.0).h(300.0));
    let root = tree.root;
    let btn = tree.add_element(root, "button");
    assert_eq!(tree.slot(btn).style.cursor, Cursor::Pointer);
    assert_eq!(tree.slot(btn).element, "button");
}

#[test]
fn add_element_with_overrides() {
    let mut tree = Tree::new(Style::default().w(400.0).h(300.0));
    let root = tree.root;
    let red = Color::rgba(255, 0, 0, 255);
    let btn = tree.add_element_with(root, "button", |s| s.h(40.0).bg(red));
    // Override applied
    assert_eq!(tree.slot(btn).style.height, Dimension::Px(40.0));
    assert_eq!(tree.slot(btn).style.background, red);
    // UA defaults still present for un-overridden fields
    assert_eq!(tree.slot(btn).style.cursor, Cursor::Pointer);
    assert_eq!(tree.slot(btn).style.justify, Justify::Center);
}

#[test]
fn to_dom_via_html_tag_enum() {
    let el = HtmlTag::Button.to_dom();
    assert_eq!(el.style.cursor, Cursor::Pointer);
    let el2 = HtmlTag::A.to_dom();
    assert_eq!(el2.style.cursor, Cursor::Pointer);
}

#[test]
fn to_dom_css_specificity_layering() {
    // UA defaults (specificity d=element) can be overridden by class (c)
    // and class by inline (a). Verify the cascade works.
    let css = "button { background: rgb(0,100,200); } .primary { background: rgb(255,0,0); }";
    let sheet = StyleSheet::parse_with_ua(css);

    // Tag-only lookup: user CSS overrides UA
    let tag_style = sheet.tag("button");
    assert_eq!(tag_style.background, Color::rgba(0, 100, 200, 255));

    // Class overrides tag (higher specificity)
    let mut style = sheet.tag("button");
    sheet.apply(&mut style, "primary");
    assert_eq!(style.background, Color::rgba(255, 0, 0, 255));
    // But non-overridden UA properties remain
    assert_eq!(style.cursor, Cursor::Pointer);
}
