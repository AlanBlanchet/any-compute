//! Headless snapshot capture — programmatic hover/click/transition without touching the real cursor.
//!
//! Produces PNGs under `out/snapshots/` for visual inspection:
//!   - `01_initial.png`            — base state, no interactions
//!   - `02_hover_card.png`         — after hovering over first card
//!   - `03_mid_transition.png`     — mid-transition (partial interpolation)
//!   - `04_hover_nav.png`          — sidebar nav hover
//!   - `05_hover_button.png`       — button hover
//!   - `06_hover_swatch.png`       — swatch border-radius transition complete
//!   - `07_active_button.png`      — button :active state
//!   - `08_unhover_card.png`       — card after leaving hover
//!
//! Run: `cargo run -p any-compute-canvas --bin anv-snapshot --features gpu`

use any_compute_canvas::harness::TestHarness;
use std::path::{Path, PathBuf};

const CSS: &str = include_str!("../../../../examples/dom/playground.css");
const HTML: &str = include_str!("../../../../examples/dom/playground.html");

fn snap(h: &mut TestHarness, dir: &Path, name: &str) {
    let p = dir.join(name);
    h.capture_png(&p);
    println!("  → {}", p.display());
}

fn main() {
    let dir = PathBuf::from("out/snapshots");
    std::fs::create_dir_all(&dir).unwrap();

    let mut h = TestHarness::from_css_html(CSS, HTML, (800, 600));
    h.tree.start_animations();

    println!("Headless snapshot capture");

    // 1. Initial state
    snap(&mut h, &dir, "01_initial.png");

    // 2. Hover over second card ("Memory" card, roughly center)
    h.hover((430.0, 130.0));
    snap(&mut h, &dir, "02_hover_card.png");

    // 3. Mid-transition: hover a card with transition, then tick partially
    h.hover((300.0, 130.0)); // hover first card (Processor)
    h.tree.tick(0.07); // 70ms into 200ms transition
    snap(&mut h, &dir, "03_mid_transition.png");

    // Let transition complete
    h.tree.tick(0.2);

    // 4. Hover sidebar nav item (Settings)
    h.hover((100.0, 105.0));
    h.tree.tick(0.3); // complete any transition
    snap(&mut h, &dir, "04_hover_nav.png");

    // 5. Hover a button (OK)
    h.hover((230.0, 210.0));
    h.tree.tick(0.3);
    snap(&mut h, &dir, "05_hover_button.png");

    // 6. Hover a swatch (border-radius transition)
    h.hover((200.0, 370.0));
    h.tree.tick(0.3);
    snap(&mut h, &dir, "06_hover_swatch.png");

    // 7. Active state on primary button (OK)
    h.pointer_down((230.0, 210.0));
    snap(&mut h, &dir, "07_active_button.png");
    h.pointer_up((230.0, 210.0));

    // 8. Move cursor away — card should revert
    h.hover((750.0, 580.0));
    h.tree.tick(0.3);
    snap(&mut h, &dir, "08_unhover_card.png");

    println!("Done — {} snapshots in {}", 8, dir.display());
}
