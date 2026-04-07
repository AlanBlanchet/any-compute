//! Headless snapshot renderer — renders every showcase tab + browser pages to PNG.
//!
//! Usage: `cargo run -p showcase --bin snapshot`
//!
//! Outputs PNGs to `out/snapshots/` for visual inspection.

mod app;
mod shared;
mod tabs;

use any_compute_core::animation::{Transition, TransitionManager};
use any_compute_dom::gpu::Gpu;
use app::{AppData, COMBINED_CSS, WEBSITE_HTML, build_sheet};
use std::io::Read;
use std::path::Path;
use std::time::Duration;

// ── Main ────────────────────────────────────────────────────────────────

fn main() {
    let out_dir = Path::new("out/snapshots");
    std::fs::create_dir_all(out_dir).unwrap();

    let sheet = build_sheet();
    let mut gpu = Gpu::init_headless(1400, 900);
    let mut data = AppData::new();
    // Seed with representative values for headless rendering
    data.fps = 60;
    data.frame_times = vec![16.0; 30];
    data.compute_results = vec![
        ("Sqrt 10000".into(), 42000.0),
        ("Sum 10000".into(), 85000.0),
        ("Sqrt 100000".into(), 4200.0),
    ];
    data.live_ac_ops = 15000.0;
    data.live_rayon_ops = 8000.0;
    data.live_std_ops = 3000.0;

    let w = 1400.0;
    let h = 900.0;

    eprintln!("=== Showcase Snapshot Renderer ===\n");

    // ── Tab 0: Browser (dashboard) ──────────────────────────────────────
    eprintln!("[Tab 0] Browser — Dashboard");
    data.tab = 0;
    data.browser.html = WEBSITE_HTML.to_string();
    data.browser.reload(&COMBINED_CSS);
    data.render_frame(
        &sheet,
        &mut gpu,
        w,
        h,
        &out_dir.join("tab0_browser_dashboard.png"),
    );

    // Browser subtabs
    for sub in 0..4 {
        let names = ["elements", "console", "network", "source"];
        data.subtabs[0] = sub;
        eprintln!("[Tab 0] Browser — DevTools: {}", names[sub]);
        data.render_frame(
            &sheet,
            &mut gpu,
            w,
            h,
            &out_dir.join(format!("tab0_browser_{}.png", names[sub])),
        );
    }

    // Browser: devtools closed
    data.browser.toggle_devtools();
    data.subtabs[0] = 0;
    eprintln!("[Tab 0] Browser — DevTools closed");
    data.render_frame(
        &sheet,
        &mut gpu,
        w,
        h,
        &out_dir.join("tab0_browser_no_devtools.png"),
    );
    data.browser.toggle_devtools(); // re-open

    // ── Tab 0: Browser with external HTML pages ─────────────────────────
    let test_pages: &[(&str, &str)] = &[
        (
            "simple",
            r##"<!DOCTYPE html>
<html>
<head><title>Simple Test</title></head>
<body style="background:#1e1e2e;color:#cdd6f4;font-family:sans-serif;padding:20px;">
  <h1>Hello World</h1>
  <p>This is a <a href="#">link</a> inside a paragraph.</p>
  <div style="display:flex;gap:10px;margin:20px 0;">
    <button style="padding:8px 16px;background:#89b4fa;border:none;border-radius:6px;color:#1e1e2e;">Primary</button>
    <button style="padding:8px 16px;background:#a6e3a1;border:none;border-radius:6px;color:#1e1e2e;">Success</button>
    <button style="padding:8px 16px;background:#f38ba8;border:none;border-radius:6px;color:#1e1e2e;">Danger</button>
  </div>
  <ul>
    <li>Item one</li>
    <li>Item two</li>
    <li>Item three</li>
  </ul>
</body>
</html>"##,
        ),
        (
            "table",
            r##"<!DOCTYPE html>
<html>
<head><title>Table Test</title></head>
<body style="background:#1e1e2e;color:#cdd6f4;padding:20px;">
  <h2>Data Table</h2>
  <table style="border-collapse:collapse;width:100%;">
    <tr style="background:#313244;">
      <th style="padding:8px;text-align:left;">Name</th>
      <th style="padding:8px;text-align:left;">Value</th>
      <th style="padding:8px;text-align:left;">Status</th>
    </tr>
    <tr><td style="padding:8px;">Alpha</td><td style="padding:8px;">42</td><td style="padding:8px;color:#a6e3a1;">Active</td></tr>
    <tr style="background:#181825;"><td style="padding:8px;">Beta</td><td style="padding:8px;">17</td><td style="padding:8px;color:#f38ba8;">Error</td></tr>
    <tr><td style="padding:8px;">Gamma</td><td style="padding:8px;">93</td><td style="padding:8px;color:#f9e2af;">Pending</td></tr>
  </table>
</body>
</html>"##,
        ),
        (
            "nested",
            r##"<!DOCTYPE html>
<html>
<head><title>Nested Layout</title>
<style>
.container { display:flex; gap:16px; padding:16px; }
.card { background:#313244; border-radius:8px; padding:16px; flex:1; }
.card h3 { color:#89b4fa; margin:0 0 8px 0; }
.card p { color:#a6adc8; margin:0; font-size:14px; }
.header { background:#181825; padding:12px 20px; margin-bottom:16px; }
.header h1 { color:#cdd6f4; margin:0; font-size:22px; }
</style>
</head>
<body style="background:#1e1e2e;color:#cdd6f4;">
  <div class="header"><h1>Dashboard</h1></div>
  <div class="container">
    <div class="card"><h3>Users</h3><p>1,234 active</p></div>
    <div class="card"><h3>Revenue</h3><p>$45,678</p></div>
    <div class="card"><h3>Orders</h3><p>892 today</p></div>
  </div>
  <div class="container">
    <div class="card" style="flex:2;"><h3>Activity</h3><p>Recent sign-ups and purchases shown here.</p></div>
    <div class="card"><h3>Alerts</h3><p>3 warnings</p></div>
  </div>
</body>
</html>"##,
        ),
    ];

    for (name, html) in test_pages {
        eprintln!("[Tab 0] Browser — page: {name}");
        data.browser.html = html.to_string();
        data.browser.reload(&COMBINED_CSS);
        data.subtabs[0] = 0;
        data.render_frame(
            &sheet,
            &mut gpu,
            w,
            h,
            &out_dir.join(format!("tab0_browser_page_{name}.png")),
        );
    }

    // ── Tab 0: Browser with live-fetched google.com ─────────────────────
    eprintln!("[Tab 0] Browser — fetching google.com...");
    match ureq::get("https://www.google.com").call() {
        Ok(mut resp) => {
            let ct = resp
                .headers()
                .get("content-type")
                .and_then(|v| v.to_str().ok())
                .map(str::to_owned);
            let mut bytes = Vec::new();
            if resp.body_mut().as_reader().read_to_end(&mut bytes).is_ok() {
                let html = shared::decode_http_body(&bytes, ct.as_deref());
                eprintln!("  fetched {} bytes", html.len());
                data.browser.html = html;
                data.browser.reload(&COMBINED_CSS);
                data.subtabs[0] = 0; // Elements
                data.render_frame(
                    &sheet,
                    &mut gpu,
                    w,
                    h,
                    &out_dir.join("tab0_browser_google.png"),
                );
                // Also render without devtools for better viewport
                data.browser.toggle_devtools();
                data.render_frame(
                    &sheet,
                    &mut gpu,
                    w,
                    h,
                    &out_dir.join("tab0_browser_google_full.png"),
                );
                data.browser.toggle_devtools();
            }
        }
        Err(e) => eprintln!("  failed to fetch google.com: {e}"),
    }

    // Restore dashboard for later
    data.browser.html = WEBSITE_HTML.to_string();
    data.browser.reload(&COMBINED_CSS);

    // ── Tab 1: 3D Scene ─────────────────────────────────────────────────
    eprintln!("[Tab 1] 3D Scene");
    data.tab = 1;
    data.transitions = TransitionManager::default();
    let mut t = Transition::new(0.0, 1.0, Duration::ZERO);
    t.start();
    data.transitions.add("tab-1", t);
    data.render_frame(&sheet, &mut gpu, w, h, &out_dir.join("tab1_scene.png"));

    // ── Tab 2: Compute ──────────────────────────────────────────────────
    eprintln!("[Tab 2] Compute — Benchmarks");
    data.tab = 2;
    data.transitions = TransitionManager::default();
    let mut t = Transition::new(0.0, 1.0, Duration::ZERO);
    t.start();
    data.transitions.add("tab-2", t);
    data.subtabs[2] = 0;
    data.render_frame(
        &sheet,
        &mut gpu,
        w,
        h,
        &out_dir.join("tab2_compute_bench.png"),
    );

    data.subtabs[2] = 1;
    eprintln!("[Tab 2] Compute — Live");
    data.render_frame(
        &sheet,
        &mut gpu,
        w,
        h,
        &out_dir.join("tab2_compute_live.png"),
    );

    // ── Tab 3: AI ───────────────────────────────────────────────────────
    eprintln!("[Tab 3] AI");
    data.tab = 3;
    data.transitions = TransitionManager::default();
    let mut t = Transition::new(0.0, 1.0, Duration::ZERO);
    t.start();
    data.transitions.add("tab-3", t);
    data.render_frame(&sheet, &mut gpu, w, h, &out_dir.join("tab3_ai.png"));

    eprintln!(
        "\n=== Done! {} snapshots saved to {} ===",
        std::fs::read_dir(out_dir).unwrap().count(),
        out_dir.display()
    );
}
