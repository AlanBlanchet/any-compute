//! Interactive DOM playground — test all UI features in a live window.
//!
//! Exercises: flex layout, borders, border-radius, padding, gap, colors,
//! opacity, CSS transitions, @keyframes animations, hover, active, clicks.
//!
//! Run: `cargo run -p dom-example` or `make dom`

use any_compute_canvas::gpu::Gpu;
use any_compute_canvas::winit::{
    self,
    event::{ElementState, Event, MouseButton, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};
use any_compute_canvas::{DEFAULT_VIEWPORT, PALETTE_CSS};
use any_compute_core::interaction::{Button, HoverState, InputEvent};
use any_compute_core::layout::Point;
use any_compute_core::render::RenderList;
use any_compute_dom::css::StyleSheet;
use any_compute_dom::parse::parse_with_css;
use std::sync::Arc;

const CSS: &str = include_str!("playground.css");
const HTML: &str = include_str!("playground.html");

fn main() {
    env_logger::init();

    let (w, h) = (DEFAULT_VIEWPORT.w, DEFAULT_VIEWPORT.h);
    let full_css = format!("{PALETTE_CSS}\n{CSS}");
    let sheet = StyleSheet::parse(&full_css);
    let mut tree = parse_with_css(HTML, &sheet);
    tree.layout(DEFAULT_VIEWPORT);

    println!(
        "DOM playground: {} nodes",
        tree.arena.len(),
    );

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Wait);

    let window = Arc::new(
        WindowBuilder::new()
            .with_title("any-compute — DOM Playground")
            .with_inner_size(winit::dpi::LogicalSize::new(w, h))
            .with_resizable(false)
            .build(&event_loop)
            .unwrap(),
    );

    let mut gpu = Gpu::init(window.clone());
    let mut hover = HoverState::default();
    let mut cursor = Point::ZERO;

    let _ = event_loop.run(move |event, elwt| match event {
        Event::Resumed => window.request_redraw(),
        Event::WindowEvent {
            event: wevent,
            window_id,
        } if window_id == window.id() => match wevent {
            WindowEvent::CloseRequested => elwt.exit(),

            WindowEvent::CursorMoved { position, .. } => {
                cursor = Point::new(position.x, position.y);
                let result = tree.dispatch(InputEvent::PointerMove { pos: cursor });
                if let Some(delta) = hover.update(result.target_tag().map(String::from)) {
                    if let Some(ref tag) = delta.entered {
                        println!("  hover → {tag}");
                    }
                    window.request_redraw();
                }
            }

            WindowEvent::MouseInput { state, button, .. } => {
                let btn = match button {
                    MouseButton::Left => Button::Primary,
                    MouseButton::Right => Button::Secondary,
                    MouseButton::Middle => Button::Middle,
                    _ => Button::Primary,
                };
                let event = match state {
                    ElementState::Pressed => InputEvent::PointerDown {
                        pos: cursor,
                        button: btn,
                    },
                    ElementState::Released => InputEvent::PointerUp {
                        pos: cursor,
                        button: btn,
                    },
                };
                let result = tree.dispatch(event);
                if state == ElementState::Released {
                    if let Some(tag) = result.target_tag() {
                        println!("  click → {tag}");
                    }
                }
                window.request_redraw();
            }

            WindowEvent::RedrawRequested => {
                tree.layout(DEFAULT_VIEWPORT);
                let mut list = RenderList::default();
                tree.paint(&mut list);
                gpu.paint(&list);
            }
            _ => {}
        },
        _ => {}
    });
}
