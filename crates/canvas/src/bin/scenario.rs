//! Scenario runner — replay scripted interactions and capture GPU screenshots.
//!
//! Fully headless — no real mouse or keyboard interaction needed.
//!
//! ```sh
//! cargo run -p any-compute-canvas --bin anv-scenario --features gpu
//! ```

use any_compute_canvas::gpu::Gpu;
use any_compute_canvas::scenario::{replay_step, Scenario};
use any_compute_canvas::{DEFAULT_VIEWPORT, PALETTE_CSS};
use any_compute_core::layout::Point;
use any_compute_core::render::RenderList;
use any_compute_dom::css::StyleSheet;
use any_compute_dom::parse::parse_with_css;
use std::path::PathBuf;

/// CSS loaded from the single-source fixture file.
const CSS: &str = include_str!("../../fixtures/visual_test.css");

/// HTML loaded from the single-source fixture file.
const HTML: &str = include_str!("../../fixtures/visual_test.html");

pub fn main() {
    env_logger::init();

    let out = PathBuf::from("out/scenario");
    std::fs::create_dir_all(&out).unwrap();

    let full_css = format!("{PALETTE_CSS}\n{CSS}");
    let sheet = StyleSheet::parse(&full_css);
    let mut tree = parse_with_css(HTML, &sheet);
    tree.layout(DEFAULT_VIEWPORT);

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

    let mut gpu = Gpu::init_headless(
        DEFAULT_VIEWPORT.w as u32,
        DEFAULT_VIEWPORT.h as u32,
    );
    let mut captures = 0usize;
    let mut assertions = 0usize;
    let mut failures = 0usize;

    for (i, action) in scenario.actions.iter().enumerate() {
        let r = replay_step(&mut tree, action, i);

        if let Some(pass) = r.assertion {
            assertions += 1;
            if !pass {
                failures += 1;
                eprintln!("FAIL step {}: {:?}", r.index, r.action);
            }
        }
        if r.capture {
            tree.layout(DEFAULT_VIEWPORT);
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
