//! Visual comparison binary — renders the same layout as the browser reference
//! through our CSS engine + WGPU for screenshot comparison.

use any_compute_canvas::gpu::Gpu;
use any_compute_canvas::{DEFAULT_VIEWPORT, PALETTE_CSS};
use any_compute_core::render::RenderList;
use any_compute_dom::css::StyleSheet;
use any_compute_dom::parse::parse_with_css;
use std::sync::Arc;
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

/// CSS loaded from the single-source fixture file.
const TEST_CSS: &str = include_str!("../../fixtures/visual_test.css");

/// HTML body loaded from the single-source fixture file.
const TEST_HTML: &str = include_str!("../../fixtures/visual_test.html");

pub fn main() {
    env_logger::init();
    let (w, h) = (DEFAULT_VIEWPORT.w, DEFAULT_VIEWPORT.h);
    let css = format!("{PALETTE_CSS}\n{TEST_CSS}");
    let sheet = StyleSheet::parse(&css);
    let mut tree = parse_with_css(TEST_HTML, &sheet);
    tree.layout(DEFAULT_VIEWPORT);
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
            .with_inner_size(winit::dpi::LogicalSize::new(w, h))
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
