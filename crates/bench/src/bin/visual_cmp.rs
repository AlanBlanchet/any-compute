//! Visual comparison binary — renders the same layout as `visual_test.html`
//! through our CSS engine + WGPU, so we can screenshot-compare against a browser.

use any_compute_bench::gpu::Gpu;
use any_compute_core::layout::Size;
use any_compute_core::render::RenderList;
use any_compute_dom::css::StyleSheet;
use any_compute_dom::parse::parse_with_css;
use std::sync::Arc;
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

const W: f64 = 800.0;
const H: f64 = 600.0;

/// Same CSS as visual_test.html — single source of truth for the comparison.
const TEST_CSS: &str = r#"
* {
  box-sizing: border-box;
}
.root {
  flex-direction: row;
  width: 800px;
  height: 600px;
  background: #1e1e2e;
}
.sidebar {
  width: 200px;
  min-width: 200px;
  background: #181825;
  padding: 16px;
  gap: 10px;
}
.brand-row {
  flex-direction: row;
  gap: 10px;
  align-items: center;
}
.brand-icon {
  width: 28px;
  height: 28px;
  min-width: 28px;
  min-height: 28px;
  background: #89b4fa;
  border-radius: 6px;
}
.brand-label {
  font-size: 14px;
  color: #cdd2f4;
}
.nav-item {
  height: 36px;
  min-height: 36px;
  border-radius: 8px;
  padding: 0px 12px;
  align-items: center;
  color: #9399b2;
  font-size: 13px;
}
.nav-active {
  background: #89b4fa;
  color: #181825;
}
.nav-inactive {
  background: transparent;
  color: #9399b2;
}
.main {
  flex-grow: 1;
}
.header {
  flex-direction: row;
  height: 48px;
  min-height: 48px;
  background: #313244;
  padding: 0px 20px;
  align-items: center;
  font-size: 16px;
  color: #cdd2f4;
}
.content {
  flex-grow: 1;
  padding: 20px;
  gap: 16px;
}
.cards-row {
  flex-direction: row;
  gap: 12px;
}
.card {
  flex-grow: 1;
  background: #313244;
  border-radius: 12px;
  padding: 16px;
  gap: 8px;
}
.card-title {
  font-size: 14px;
  color: #89b4fa;
}
.card-body {
  font-size: 12px;
  color: #cdd2f4;
}
.border-box {
  width: 120px;
  height: 60px;
  background: #45475a;
  border: 2px solid #89b4fa;
  border-radius: 8px;
  align-items: center;
  justify-content: center;
  font-size: 11px;
  color: #cdd2f4;
}
.nested-row {
  flex-direction: row;
  gap: 8px;
}
.color-swatch {
  width: 40px;
  height: 40px;
  border-radius: 6px;
}
.swatch-green  { background: #a6e3a1; }
.swatch-blue   { background: #89b4fa; }
.swatch-red    { background: #f38ba8; }
.swatch-yellow { background: #f9e2af; }
.swatch-mauve  { background: #cba6f7; }
.bar-row {
  gap: 6px;
}
.bar-track {
  height: 8px;
  background: rgba(255, 255, 255, 20);
  border-radius: 4px;
  overflow: hidden;
}
.bar-fill-green {
  height: 8px;
  width: 70%;
  background: #a6e3a1;
  border-radius: 4px;
}
.bar-fill-blue {
  height: 8px;
  width: 45%;
  background: #89b4fa;
  border-radius: 4px;
}
.bar-fill-red {
  height: 8px;
  width: 85%;
  background: #f38ba8;
  border-radius: 4px;
}
.opacity-box {
  width: 60px;
  height: 40px;
  background: #89b4fa;
  border-radius: 6px;
}
.opacity-50 { opacity: 0.5; }
.opacity-25 { opacity: 0.25; }
"#;

/// Same structure as visual_test.html body — parsed via our HTML+CSS engine.
const TEST_HTML: &str = r#"
<div class="root">
  <div class="sidebar">
    <div class="brand-row">
      <div class="brand-icon"></div>
      <span class="brand-label">any-compute</span>
    </div>
    <div class="nav-item nav-active">Dashboard</div>
    <div class="nav-item nav-inactive">Settings</div>
    <div class="nav-item nav-inactive">About</div>
  </div>
  <div class="main">
    <div class="header">Dashboard</div>
    <div class="content">
      <div class="cards-row">
        <div class="card">
          <span class="card-title">Processor</span>
          <span class="card-body">8 cores / 16 threads</span>
        </div>
        <div class="card">
          <span class="card-title">Memory</span>
          <span class="card-body">32 GB available</span>
        </div>
        <div class="card">
          <span class="card-title">GPU</span>
          <span class="card-body">Vulkan backend</span>
        </div>
      </div>
      <div class="nested-row">
        <div class="color-swatch swatch-green"></div>
        <div class="color-swatch swatch-blue"></div>
        <div class="color-swatch swatch-red"></div>
        <div class="color-swatch swatch-yellow"></div>
        <div class="color-swatch swatch-mauve"></div>
        <div class="border-box">bordered</div>
      </div>
      <div class="bar-row">
        <div class="bar-track"><div class="bar-fill-green"></div></div>
        <div class="bar-track"><div class="bar-fill-blue"></div></div>
        <div class="bar-track"><div class="bar-fill-red"></div></div>
      </div>
      <div class="nested-row">
        <div class="opacity-box"></div>
        <div class="opacity-box opacity-50"></div>
        <div class="opacity-box opacity-25"></div>
      </div>
    </div>
  </div>
</div>
"#;

pub fn main() {
    env_logger::init();

    let sheet = StyleSheet::parse(TEST_CSS);
    let mut tree = parse_with_css(TEST_HTML, &sheet);
    tree.layout(Size::new(W, H));
    let mut list = RenderList::default();
    tree.paint(&mut list);

    println!(
        "Visual test: {} nodes, {} primitives",
        tree.arena.len(),
        list.len()
    );

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let window = Arc::new(
        WindowBuilder::new()
            .with_title("any-compute — Visual CSS Test")
            .with_inner_size(winit::dpi::LogicalSize::new(W, H))
            .with_resizable(false)
            .build(&event_loop)
            .unwrap(),
    );

    let mut gpu = Gpu::init(window.clone());

    let _ = event_loop.run(move |event, elwt| match event {
        Event::WindowEvent {
            event: wevent,
            window_id,
        } if window_id == window.id() => match wevent {
            WindowEvent::CloseRequested => elwt.exit(),
            WindowEvent::RedrawRequested => {
                gpu.paint(&list);
                window.request_redraw();
            }
            _ => {}
        },
        _ => {}
    });
}
