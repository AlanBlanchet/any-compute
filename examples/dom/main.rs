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
    window::{CursorIcon, WindowBuilder},
};
use any_compute_canvas::{DEFAULT_VIEWPORT, PALETTE_CSS};
use any_compute_core::interaction::{Button, InputEvent};
use any_compute_core::layout::Point;
use any_compute_core::render::RenderList;
use any_compute_dom::css::StyleSheet;
use any_compute_dom::parse::parse_with_css;
use std::sync::Arc;
use std::time::Instant;

const CSS: &str = include_str!("playground.css");
const HTML: &str = include_str!("playground.html");
/// Cap frame dt to ~30fps to prevent transitions snapping after idle.
const MAX_FRAME_DT: f64 = 0.032;

fn main() {
    env_logger::init();

    let (w, h) = (DEFAULT_VIEWPORT.w, DEFAULT_VIEWPORT.h);
    let full_css = format!("{PALETTE_CSS}\n{CSS}");
    let sheet = StyleSheet::parse(&full_css);
    let mut tree = parse_with_css(HTML, &sheet);
    tree.layout(DEFAULT_VIEWPORT);
    tree.start_animations();

    println!("DOM playground: {} nodes", tree.arena.len());

    let event_loop = EventLoop::new().unwrap();
    // Use Poll for continuous animation ticking; falls back to Wait when idle.
    event_loop.set_control_flow(ControlFlow::Poll);

    let window = Arc::new(
        WindowBuilder::new()
            .with_title("any-compute — DOM Playground")
            .with_inner_size(winit::dpi::LogicalSize::new(w, h))
            .with_resizable(false)
            .build(&event_loop)
            .unwrap(),
    );

    let mut gpu = Gpu::init(window.clone());
    let mut cursor = Point::ZERO;
    let mut last_frame = Instant::now();
    let mut needs_repaint = true;
    let mut current_cursor = CursorIcon::Default;

    let _ = event_loop.run(move |event, elwt| match event {
        Event::Resumed => {
            needs_repaint = true;
            window.request_redraw();
        }
        Event::AboutToWait => {
            // Tick animations each frame
            let now = Instant::now();
            // Cap dt to avoid huge spikes after idle (transitions would snap-finish).
            let dt = (now - last_frame).as_secs_f64().min(MAX_FRAME_DT);
            last_frame = now;

            // tick() auto re-layouts when animations touch layout properties.
            if tree.tick(dt).active {
                needs_repaint = true;
            }

            if needs_repaint {
                window.request_redraw();
            }

            // Switch to Wait when no animations running to save CPU
            if tree.has_active_animations() || needs_repaint {
                elwt.set_control_flow(ControlFlow::Poll);
            } else {
                elwt.set_control_flow(ControlFlow::Wait);
            }
        }
        Event::WindowEvent {
            event: wevent,
            window_id,
        } if window_id == window.id() => match wevent {
            WindowEvent::CloseRequested => elwt.exit(),

            WindowEvent::CursorMoved { position, .. } => {
                cursor = Point::new(position.x, position.y);
                let result = tree.dispatch(InputEvent::PointerMove { pos: cursor });
                // Apply OS cursor from dispatch result.
                let icon = match result.cursor.as_str() {
                    "pointer" => CursorIcon::Pointer,
                    "text" => CursorIcon::Text,
                    "move" => CursorIcon::Move,
                    "not-allowed" => CursorIcon::NotAllowed,
                    "grab" => CursorIcon::Grab,
                    "grabbing" => CursorIcon::Grabbing,
                    "crosshair" => CursorIcon::Crosshair,
                    "help" => CursorIcon::Help,
                    "wait" => CursorIcon::Wait,
                    _ => CursorIcon::Default,
                };
                if icon != current_cursor {
                    window.set_cursor_icon(icon);
                    current_cursor = icon;
                }
                // Only repaint if hover target actually changed
                if result.restyled {
                    needs_repaint = true;
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
                if result.restyled {
                    needs_repaint = true;
                }
            }

            WindowEvent::RedrawRequested => {
                let mut list = RenderList::default();
                tree.paint(&mut list);
                gpu.paint(&list);
                needs_repaint = false;
            }
            _ => {}
        },
        _ => {}
    });
}
