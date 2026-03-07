//! Scenario runner — replay scripted interactions against the DOM engine
//! and capture GPU screenshots at marked points.
//!
//! No real mouse or keyboard is touched.  Works fully headless (`Gpu::init_headless`).
//!
//! ```sh
//! cargo run --bin anv-scenario --features window
//! ```

use any_compute_bench::gpu::Gpu;
use any_compute_core::interaction::Scenario;
use any_compute_core::layout::{Point, Size};
use any_compute_core::render::RenderList;
use any_compute_dom::css::StyleSheet;
use any_compute_dom::parse::parse_with_css;
use std::path::PathBuf;

const W: f64 = 800.0;
const H: f64 = 600.0;

/// Demo CSS — same dashboard from visual_cmp, extended with hover/active states.
const CSS: &str = r#"
* { box-sizing: border-box; }
.root {
  flex-direction: row;
  width: 800px; height: 600px;
  background: #1e1e2e;
}
.sidebar {
  width: 200px; min-width: 200px;
  background: #181825;
  padding: 16px; gap: 10px;
}
.nav-item {
  height: 36px; min-height: 36px;
  border-radius: 8px;
  padding: 0px 12px;
  align-items: center;
  font-size: 13px;
}
.nav-active  { background: #89b4fa; color: #181825; }
.nav-inactive { background: transparent; color: #9399b2; }
.main { flex-grow: 1; }
.header {
  flex-direction: row;
  height: 48px; min-height: 48px;
  background: #313244;
  padding: 0px 20px;
  align-items: center;
  font-size: 16px; color: #cdd2f4;
}
.content {
  flex-grow: 1;
  padding: 20px; gap: 16px;
}
.cards-row {
  flex-direction: row; gap: 12px;
}
.card {
  flex-grow: 1;
  background: #313244;
  border-radius: 12px;
  padding: 16px; gap: 8px;
}
.card-title { font-size: 14px; color: #89b4fa; }
.card-body  { font-size: 12px; color: #cdd2f4; }
"#;

/// Dashboard HTML with `tag` attributes for hit-testing.
const HTML: &str = r#"
<div class="root">
  <div class="sidebar">
    <div class="nav-item nav-active" tag="nav-dash">Dashboard</div>
    <div class="nav-item nav-inactive" tag="nav-settings">Settings</div>
    <div class="nav-item nav-inactive" tag="nav-about">About</div>
  </div>
  <div class="main">
    <div class="header">Dashboard</div>
    <div class="content">
      <div class="cards-row">
        <div class="card" tag="card-cpu">
          <span class="card-title">Processor</span>
          <span class="card-body">8 cores / 16 threads</span>
        </div>
        <div class="card" tag="card-mem">
          <span class="card-title">Memory</span>
          <span class="card-body">32 GB available</span>
        </div>
        <div class="card" tag="card-gpu">
          <span class="card-title">GPU</span>
          <span class="card-body">Vulkan backend</span>
        </div>
      </div>
    </div>
  </div>
</div>
"#;

pub fn main() {
    env_logger::init();

    let out = PathBuf::from("out/scenario");
    std::fs::create_dir_all(&out).unwrap();

    // ── Parse & layout ──────────────────────────────────────────────────
    let sheet = StyleSheet::parse(CSS);
    let mut tree = parse_with_css(HTML, &sheet);
    tree.layout(Size::new(W, H));

    // ── Build scenario ──────────────────────────────────────────────────
    let scenario = Scenario::new()
        .capture()
        .click(Point::new(100.0, 78.0))
        .assert_tag(Point::new(100.0, 78.0), "nav-settings")
        .capture()
        .hover(Point::new(350.0, 150.0))
        .assert_tag(Point::new(350.0, 150.0), "card-cpu")
        .capture()
        .click(Point::new(650.0, 150.0))
        .assert_tag(Point::new(650.0, 150.0), "card-gpu")
        .capture();

    // ── Replay step-by-step with inline capture ──────────────────────────
    let mut gpu = Gpu::init_headless(W as u32, H as u32);
    let mut captures = 0usize;
    let mut assertions = 0usize;
    let mut failures = 0usize;

    for (i, action) in scenario.actions.iter().enumerate() {
        let r = tree.replay_step(action, i);

        if let Some(pass) = r.assertion {
            assertions += 1;
            if !pass {
                failures += 1;
                eprintln!("FAIL step {}: {:?}", r.index, r.action);
            }
        }
        if r.capture {
            tree.layout(Size::new(W, H));
            let mut list = RenderList::default();
            tree.paint(&mut list);
            let path = out.join(format!("capture_{captures}.png"));
            gpu.capture_png(&list, &path);
            println!("  saved {}", path.display());
            captures += 1;
        }
    }

    let total = scenario.actions.len();
    println!("Scenario: {total} steps, {assertions} assertions, {failures} failures");
    println!("{captures} screenshots saved to {}", out.display());
    if failures > 0 {
        std::process::exit(1);
    }
}
