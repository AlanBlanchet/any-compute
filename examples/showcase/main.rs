//! Unified showcase — every feature of any-compute in one tabbed window.
//!
//! Run: `cargo run -p showcase` or `make showcase`
//!
//! Tabs:
//!   0. Browser      — Split: Preview (left) + DevTools (right, Elements default)
//!   1. 3D Scene     — mesh, camera, transform, ray intersection, normals
//!   2. Compute      — Benchmarks | Live metrics
//!   3. AI           — Planner (left) + tabbed center (Training | Models | Datasets)

mod app;
mod tabs;

use any_compute_core::interaction::{Button, InputEvent};
use any_compute_core::layout::{Point, Size};
use any_compute_core::render::RenderList;
use any_compute_dom::gpu::Gpu;
use any_compute_dom::page::Page;
use any_compute_dom::tree::*;
use any_compute_dom::winit;
use app::{AppData, COMBINED_CSS, TAB_LABELS, WEBSITE_HTML, build_sheet};
use std::sync::Arc;
use std::time::Instant;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use winit::event::{ElementState, Event, Modifiers, MouseButton, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{CursorIcon, WindowBuilder};

// ═══════════════════════════════════════════════════════════════════════════

mod shared;
use shared::*;

fn main() {
    // Logging — dual output: stderr (compact) + log file (verbose, timestamped).
    // The log file is written to `showcase.log` in the working directory.
    // Set RUST_LOG env to control level, e.g. RUST_LOG=debug cargo run -p showcase
    let log_file = tracing_appender::rolling::never(".", "showcase.log");
    let (file_writer, _guard) = tracing_appender::non_blocking(log_file);
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                "warn,showcase=debug,any_compute_dom=info,any_compute_core=info,cosmic_text=warn"
                    .parse()
                    .unwrap()
            }),
        )
        .with(
            tracing_subscriber::fmt::layer()
                .with_writer(std::io::stderr)
                .compact(),
        )
        .with(
            tracing_subscriber::fmt::layer()
                .with_writer(file_writer)
                .with_ansi(false)
                .with_thread_ids(true)
                .with_target(true),
        )
        .init();

    log::info!("Showcase starting — log file: showcase.log");

    let sheet = build_sheet();
    let state = Shared::new();

    // Build browser page for tab 0 with default website content.
    {
        state.write(|d| {
            d.browser.html = WEBSITE_HTML.to_string();
            d.browser.reload(&COMBINED_CSS);
        });
    }

    state.spawn_hw_detect();
    state.spawn_compute_bench();
    state.spawn_live_sim();

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let window = Arc::new(
        WindowBuilder::new()
            .with_title("any-compute — Showcase")
            .with_inner_size(winit::dpi::LogicalSize::new(VIEWPORT.w(), VIEWPORT.h()))
            .with_maximized(true)
            .build(&event_loop)
            .unwrap(),
    );

    let mut gpu = Gpu::init(window.clone());

    // Measure text in the DOM page with real font metrics
    state.write(|d| {
        if let Some(page) = &mut d.browser.page {
            page.measure_text_nodes(|text, font_size| gpu.measure_text(text, font_size));
            page.layout(Size::new(VIEWPORT.w() - 220.0, VIEWPORT.h() - 56.0));
            page.start_animations();
            d.browser.needs_measure = false;
        }
    });

    let mut cursor_pos = Point::ZERO;
    let mut last_frame = Instant::now();
    let mut last_tree: Option<Tree> = None;
    let mut list = RenderList::default();
    let mut current_cursor = CursorIcon::Default;
    let mut modifiers = Modifiers::default();
    let mut last_hover_tag: Option<String> = None;
    let _ = event_loop.run(move |event, elwt| match event {
        Event::AboutToWait => {
            window.request_redraw();
        }
        Event::WindowEvent {
            event: wevent,
            window_id,
        } if window_id == window.id() => match wevent {
            WindowEvent::CloseRequested => {
                state.write(|d| d.live_running = false);
                elwt.exit();
            }
            WindowEvent::Resized(sz) => gpu.resize(sz.width, sz.height),

            WindowEvent::CursorMoved { position, .. } => {
                cursor_pos = Point::new(position.x, position.y);

                // Scene drag (tab 1): orbit camera — checked FIRST so hover
                // tracking is suppressed while dragging.
                let scene_dragging = state.on_scene() && state.read(|d| d.scene_info.drag_active);
                if scene_dragging {
                    state.write(|d| {
                        let (lx, ly) = d.scene_info.drag_last;
                        let dx = position.x - lx;
                        let dy = position.y - ly;
                        d.scene_info.on_drag(dx, dy);
                        d.scene_info.drag_last = (position.x, position.y);
                    });
                    if CursorIcon::Grabbing != current_cursor {
                        window.set_cursor_icon(CursorIcon::Grabbing);
                        current_cursor = CursorIcon::Grabbing;
                    }
                    return;
                }

                // AI Graph drag (tab 3): pan the graph view
                let graph_dragging = state.on_graph() && state.read(|d| d.ai.drag_active);
                if graph_dragging {
                    state.write(|d| {
                        let (lx, ly) = d.ai.drag_last;
                        let dx = position.x - lx;
                        let dy = position.y - ly;
                        d.ai.on_drag(dx, dy);
                        d.ai.drag_last = (position.x, position.y);
                    });
                    if CursorIcon::Grabbing != current_cursor {
                        window.set_cursor_icon(CursorIcon::Grabbing);
                        current_cursor = CursorIcon::Grabbing;
                    }
                    return;
                }

                // Hover tracking on sidebar tags — only fire on change
                if let Some(tree) = &last_tree {
                    let tag = tree.tag_at(cursor_pos);
                    if tag != last_hover_tag {
                        state.handle_hover(tag.clone());
                        last_hover_tag = tag;
                    }
                }

                // Determine cursor icon from browser DOM page, showcase tree, or default
                let desired = if state.on_browser_preview() {
                    last_tree
                        .as_ref()
                        .and_then(|t| t.tagged_rect(BVP))
                        .filter(|vp| vp.contains(Point::new(cursor_pos.x, cursor_pos.y)))
                        .map(|vp| {
                            let dom_pos = AppData::to_dom_pos(cursor_pos, vp);
                            state.write(|d| {
                                d.browser.cursor = dom_pos;
                                if let Some(page) = &mut d.browser.page {
                                    let result =
                                        page.dispatch(InputEvent::PointerMove { pos: dom_pos });
                                    match result.cursor.as_str() {
                                        "pointer" => CursorIcon::Pointer,
                                        "text" => CursorIcon::Text,
                                        "not-allowed" => CursorIcon::NotAllowed,
                                        _ => CursorIcon::Default,
                                    }
                                } else {
                                    CursorIcon::Default
                                }
                            })
                        })
                        .unwrap_or_else(|| {
                            // Cursor is outside browser viewport — check showcase tree
                            last_tree.as_ref().map_or(CursorIcon::Default, |t| {
                                use any_compute_dom::style::Cursor;
                                match t.cursor_at(cursor_pos) {
                                    Cursor::Pointer => CursorIcon::Pointer,
                                    Cursor::Text => CursorIcon::Text,
                                    Cursor::NotAllowed => CursorIcon::NotAllowed,
                                    Cursor::Grab => CursorIcon::Grab,
                                    _ => CursorIcon::Default,
                                }
                            })
                        })
                } else {
                    // Non-browser tabs: resolve cursor from showcase tree
                    last_tree.as_ref().map_or(CursorIcon::Default, |t| {
                        use any_compute_dom::style::Cursor;
                        match t.cursor_at(cursor_pos) {
                            Cursor::Pointer => CursorIcon::Pointer,
                            Cursor::Text => CursorIcon::Text,
                            Cursor::NotAllowed => CursorIcon::NotAllowed,
                            Cursor::Grab => CursorIcon::Grab,
                            _ => CursorIcon::Default,
                        }
                    })
                };
                if desired != current_cursor {
                    window.set_cursor_icon(desired);
                    current_cursor = desired;
                }
            }

            WindowEvent::CursorLeft { .. } => {
                // Clear all hover effects when cursor exits the window
                if last_hover_tag.is_some() {
                    state.handle_hover(None);
                    last_hover_tag = None;
                }
                // Also clear DOM page hover
                if state.on_browser_preview() {
                    state.write(|d| {
                        if let Some(page) = &mut d.browser.page {
                            page.dispatch(InputEvent::PointerMove {
                                pos: Point::new(-1.0, -1.0),
                            });
                        }
                    });
                }
            }

            WindowEvent::MouseInput {
                state: elem_state,
                button,
                ..
            } => {
                let btn = match button {
                    MouseButton::Left => Button::Primary,
                    MouseButton::Right => Button::Secondary,
                    MouseButton::Middle => Button::Middle,
                    _ => Button::Primary,
                };

                if elem_state == ElementState::Released {
                    // End scene drag + graph drag
                    let graph_click = state.write(|d| {
                        if d.scene_info.drag_active {
                            d.scene_info.drag_active = false;
                        }
                        let click = if d.ai.drag_active {
                            d.ai.drag_active = false;
                            d.ai.click_pos.take().filter(|&(sx, sy)| {
                                (cursor_pos.x - sx).abs() < 4.0 && (cursor_pos.y - sy).abs() < 4.0
                            })
                        } else {
                            d.ai.click_pos.take();
                            None
                        };
                        click
                    });
                    // Graph single-click: enter node or click breadcrumb
                    if let Some(_) = graph_click {
                        if let Some(vp) = last_tree
                            .as_ref()
                            .and_then(|t| t.tagged_rect(tabs::graph::VIEWPORT_TAG))
                        {
                            let local = Point::new(cursor_pos.x - vp.x(), cursor_pos.y - vp.y());
                            state.write(|d| {
                                if !d.ai.view.breadcrumb.is_empty() && local.y < 28.0 {
                                    if local.x < 100.0 {
                                        d.ai.view.to_root();
                                    } else {
                                        d.ai.view.back();
                                    }
                                    return;
                                }
                                if let Some(info) = d.ai.cached_visual() {
                                    let vg = &info.visual;
                                    let target = vg.resolve_breadcrumb_pub(&d.ai.view.breadcrumb);
                                    if let Some(idx) = d.ai.view.hit_test(target, local) {
                                        if target.node(idx).children().is_some() {
                                            d.ai.view.enter(idx);
                                        }
                                    }
                                }
                            });
                        }
                    }
                    if current_cursor == CursorIcon::Grabbing {
                        window.set_cursor_icon(CursorIcon::Default);
                        current_cursor = CursorIcon::Default;
                    }

                    // Check sidebar tags first
                    if let Some(tree) = &last_tree {
                        if let Some(tag) = tree.tag_at(cursor_pos) {
                            // Browser-input special handling: position cursor on re-click
                            if tag == "browser-input" {
                                let already = state.read(|d| d.browser.input.focused);
                                if already {
                                    if let Some(rect) = tree.tagged_rect("browser-input") {
                                        let pad_border = 11.0; // 10px padding + 1px border
                                        let local_x =
                                            (cursor_pos.x - rect.x() - pad_border).max(0.0);
                                        let cw = 9.0 * any_compute_dom::style::CHAR_WIDTH_RATIO;
                                        state.write(|d| {
                                            let ci = (local_x / cw).round() as usize;
                                            let byte_off = d
                                                .browser
                                                .input
                                                .text
                                                .char_indices()
                                                .nth(ci)
                                                .map(|(i, _)| i)
                                                .unwrap_or(d.browser.input.text.len());
                                            d.browser.input.sel = None;
                                            d.browser.input.cursor = byte_off;
                                        });
                                    }
                                } else {
                                    state.write(|d| d.browser.focus());
                                }
                            } else {
                                state.handle_click(&tag);
                            }
                        }
                    }
                    // DOM page dispatch
                    if state.on_browser_preview() {
                        if let Some(vp) = last_tree.as_ref().and_then(|t| t.tagged_rect(BVP)) {
                            state.write(|d| {
                                if let Some(page) = &mut d.browser.page {
                                    let result = page.dispatch(InputEvent::PointerUp {
                                        pos: AppData::to_dom_pos(cursor_pos, vp),
                                        button: btn,
                                    });
                                    if let Some(tag) = result.target_tag() {
                                        log::debug!("browser click → {tag}");
                                    }
                                }
                            });
                        }
                    }
                } else {
                    // Start scene drag if on scene tab and clicking in the content area
                    if state.on_scene() && AppData::in_content_area(cursor_pos) {
                        state.write(|d| {
                            d.scene_info.drag_active = true;
                            d.scene_info.drag_last = (cursor_pos.x, cursor_pos.y);
                        });
                    }

                    // AI Graph: single-click to enter node or click breadcrumb to go back
                    if state.on_graph() && AppData::in_content_area(cursor_pos) {
                        let vp_rect = last_tree
                            .as_ref()
                            .and_then(|t| t.tagged_rect(tabs::graph::VIEWPORT_TAG));
                        if let Some(vp) = vp_rect {
                            if vp.contains(cursor_pos) {
                                state.write(|d| {
                                    d.ai.drag_active = true;
                                    d.ai.drag_last = (cursor_pos.x, cursor_pos.y);
                                    d.ai.click_pos = Some((cursor_pos.x, cursor_pos.y));
                                });
                            }
                        }
                    }

                    if state.on_browser_preview() {
                        if let Some(vp) = last_tree.as_ref().and_then(|t| t.tagged_rect(BVP)) {
                            state.write(|d| {
                                if let Some(page) = &mut d.browser.page {
                                    page.dispatch(InputEvent::PointerDown {
                                        pos: AppData::to_dom_pos(cursor_pos, vp),
                                        button: btn,
                                    });
                                }
                            });
                        }
                    }
                }
            }

            WindowEvent::MouseWheel { delta, .. } => {
                let dy = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => y as f64 * 40.0,
                    winit::event::MouseScrollDelta::PixelDelta(p) => p.y,
                };
                // Scene tab: zoom camera only when cursor is in scene viewport
                if state.on_scene() && AppData::in_content_area(cursor_pos) {
                    let in_viewport = last_tree
                        .as_ref()
                        .and_then(|t| t.tagged_rect("scene-viewport"))
                        .map_or(false, |vp| vp.contains(cursor_pos));
                    if in_viewport {
                        state.write(|d| d.scene_info.on_zoom(dy));
                        return;
                    }
                }
                // AI Graph tab: zoom the graph view relative to cursor
                if state.on_graph() && AppData::in_content_area(cursor_pos) {
                    let vp_rect = last_tree
                        .as_ref()
                        .and_then(|t| t.tagged_rect(tabs::graph::VIEWPORT_TAG));
                    if let Some(vp) = vp_rect {
                        // Only zoom when cursor is inside the graph viewport
                        if vp.contains(cursor_pos) {
                            let local = Point::new(cursor_pos.x - vp.x(), cursor_pos.y - vp.y());
                            state.write(|d| {
                                let factor = if dy > 0.0 { 1.1 } else { 1.0 / 1.1 };
                                d.ai.on_zoom((local.x, local.y), factor);
                            });
                            return;
                        }
                    }
                }
                // DOM tab: dispatch scroll to DOM page or DevTools panel
                if state.on_browser_preview() {
                    // DevTools panel scroll — use tree's built-in scroll method
                    if let Some(dt) = last_tree.as_ref().and_then(|t| t.tagged_rect(BDT)) {
                        if dt.contains(Point::new(cursor_pos.x, cursor_pos.y)) {
                            state.write(|d| {
                                // Compute max scroll from content extent
                                d.browser.devtools_scroll =
                                    (d.browser.devtools_scroll - dy).max(0.0);
                            });
                            // Clamp against actual content extent from laid-out tree
                            if let Some(tree) = &last_tree {
                                if let Some(slot) =
                                    tree.arena.iter().find(|s| s.tag.as_deref() == Some(BDT))
                                {
                                    let visible_h = slot.rect.size.h();
                                    let mut content_h = 0.0_f64;
                                    for &cid in &slot.children {
                                        let cr = tree.arena[cid.0].rect;
                                        let cy = cr.origin.y + cr.size.h() + slot.scroll.y
                                            - slot.rect.origin.y;
                                        content_h = content_h.max(cy);
                                    }
                                    let max_y = (content_h - visible_h).max(0.0);
                                    state.write(|d| {
                                        d.browser.devtools_scroll =
                                            d.browser.devtools_scroll.min(max_y);
                                    });
                                }
                            }
                            return;
                        }
                    }
                    // Browser viewport scroll
                    if let Some(vp) = last_tree.as_ref().and_then(|t| t.tagged_rect(BVP)) {
                        if vp.contains(Point::new(cursor_pos.x, cursor_pos.y)) {
                            state.write(|d| {
                                if let Some(page) = &mut d.browser.page {
                                    let local =
                                        Point::new(cursor_pos.x - vp.x(), cursor_pos.y - vp.y());
                                    page.dispatch(InputEvent::Scroll {
                                        pos: local,
                                        delta: Point::new(0.0, dy),
                                    });
                                }
                            });
                            return; // don't also scroll the shell
                        }
                    }
                }
                // Scroll the shell content
                state.write(|d| d.scroll_target = (d.scroll_target - dy).max(0.0));
            }

            WindowEvent::ModifiersChanged(new_mods) => {
                modifiers = new_mods;
            }

            WindowEvent::KeyboardInput {
                event:
                    winit::event::KeyEvent {
                        logical_key,
                        state: ElementState::Pressed,
                        text,
                        ..
                    },
                ..
            } => {
                let editing = state.read(|d| d.browser.input.focused && d.tab == 0);

                if editing {
                    let shift = modifiers.state().shift_key();
                    let ctrl = modifiers.state().control_key();
                    let mods = any_compute_core::interaction::Modifiers {
                        shift,
                        ctrl,
                        ..Default::default()
                    };

                    // Enter/Escape are browser-level, not text editing
                    match &logical_key {
                        Key::Named(NamedKey::Enter) => {
                            let fetch_url = state.write(|d| d.browser.navigate(&COMBINED_CSS));
                            if let Some(url) = fetch_url {
                                let s2 = state.clone();
                                std::thread::spawn(move || {
                                    let result = std::panic::catch_unwind(
                                        std::panic::AssertUnwindSafe(|| {
                                            let html = match ureq::get(&url).call() {
                                                Ok(mut resp) => {
                                                    let ct = resp
                                                        .headers()
                                                        .get("content-type")
                                                        .and_then(|v| v.to_str().ok())
                                                        .map(str::to_owned);
                                                    // Read up to 200KB to avoid OOM
                                                    let mut buf = Vec::with_capacity(200_000);
                                                    let mut reader = resp.body_mut().as_reader();
                                                    let mut chunk = [0u8; 8192];
                                                    loop {
                                                        match std::io::Read::read(
                                                            &mut reader,
                                                            &mut chunk,
                                                        ) {
                                                            Ok(0) => break,
                                                            Ok(n) => {
                                                                buf.extend_from_slice(&chunk[..n]);
                                                                if buf.len() > 200_000 {
                                                                    break;
                                                                }
                                                            }
                                                            Err(_) => break,
                                                        }
                                                    }
                                                    decode_http_body(&buf, ct.as_deref())
                                                }
                                                Err(e) => format!(
                                                    "<h1>Error</h1><p>{}</p>",
                                                    html_escape(&e.to_string())
                                                ),
                                            };
                                            s2.write(|d| {
                                                d.browser.load_fetched(html, &COMBINED_CSS)
                                            });
                                        }),
                                    );
                                    if let Err(e) = result {
                                        log::error!(
                                            "fetch thread panicked: {:?}",
                                            e.downcast_ref::<&str>()
                                        );
                                        s2.write(|d| {
                                            d.browser.html = ERROR_PAGE_HTML.into();
                                            d.browser.reload(&COMBINED_CSS);
                                        });
                                    }
                                });
                            }
                            state.write(|d| d.browser.input.blur());
                        }
                        Key::Named(NamedKey::Escape) => {
                            state.write(|d| d.browser.input.blur());
                        }
                        _ => {
                            // Map logical key to string for TextInput::handle_key
                            let key_str = match &logical_key {
                                Key::Character(c) if ctrl => c.as_str().to_string(),
                                Key::Named(n) => format!("{n:?}"),
                                _ => String::new(),
                            };
                            let handled =
                                state.write(|d| d.browser.input.handle_key(&key_str, mods));
                            if !handled && !ctrl {
                                if let Some(txt) = &text {
                                    state.write(|d| d.browser.input.handle_text(txt.as_str()));
                                }
                            }
                        }
                    }
                } else {
                    // Track held keys for WASD camera controls on scene tab
                    if let Key::Character(c) = &logical_key {
                        let k = c.as_str().to_lowercase();
                        if state.on_scene()
                            && matches!(k.as_str(), "w" | "a" | "s" | "d" | "q" | "e")
                        {
                            state.write(|d| {
                                d.keys_held.insert(k);
                            });
                            // Don't fall through to tab switching
                        } else if let Ok(n) = c.as_str().parse::<usize>() {
                            if n > 0 && n <= TAB_LABELS.len() {
                                state.handle_click(&AppData::tab_tag(n - 1));
                            }
                        }
                    } else if let Key::Named(NamedKey::Tab) = &logical_key {
                        let next = state.read(|d| (d.tab + 1) % TAB_LABELS.len());
                        state.handle_click(&AppData::tab_tag(next));
                    } else if let Key::Named(NamedKey::Space) = &logical_key {
                        if state.on_scene() {
                            state.write(|d| {
                                d.keys_held.insert("space".into());
                            });
                        }
                    } else if let Key::Named(NamedKey::Shift) = &logical_key {
                        if state.on_scene() {
                            state.write(|d| {
                                d.keys_held.insert("shift".into());
                            });
                        }
                    }
                }
            }

            WindowEvent::KeyboardInput {
                event:
                    winit::event::KeyEvent {
                        logical_key,
                        state: ElementState::Released,
                        ..
                    },
                ..
            } => match &logical_key {
                Key::Character(c) => {
                    let k = c.as_str().to_lowercase();
                    state.write(|d| {
                        d.keys_held.remove(&k);
                    });
                }
                Key::Named(NamedKey::Space) => {
                    state.write(|d| {
                        d.keys_held.remove("space");
                    });
                }
                Key::Named(NamedKey::Shift) => {
                    state.write(|d| {
                        d.keys_held.remove("shift");
                    });
                }
                _ => {}
            },

            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = (now - last_frame).as_secs_f64().min(MAX_FRAME_DT);
                last_frame = now;

                // Tick transitions
                state.write(|d| {
                    d.transitions.start_all();
                    // Record frame time
                    d.frame_times.push(dt * 1000.0);
                    if d.frame_times.len() > 120 {
                        d.frame_times.remove(0);
                    }
                    // Tick DOM page animations
                    if let Some(page) = &mut d.browser.page {
                        page.tick(dt);
                    }
                    // Tick scene physics + WASD camera
                    if d.tab == 1 {
                        d.scene_info.tick(dt);
                        let speed = 5.0 * dt;
                        if d.keys_held.contains("w") {
                            d.scene_info.scene.camera.fly(speed);
                        }
                        if d.keys_held.contains("s") {
                            d.scene_info.scene.camera.fly(-speed);
                        }
                        if d.keys_held.contains("a") {
                            d.scene_info.scene.camera.strafe(-speed);
                        }
                        if d.keys_held.contains("d") {
                            d.scene_info.scene.camera.strafe(speed);
                        }
                        if d.keys_held.contains("q") || d.keys_held.contains("space") {
                            d.scene_info.scene.camera.elevate(speed);
                        }
                        if d.keys_held.contains("e") || d.keys_held.contains("shift") {
                            d.scene_info.scene.camera.elevate(-speed);
                        }
                    }
                    // Tick AI training simulation
                    if d.tab == 3 && d.ai.training {
                        d.ai.tick_training(dt);
                    }
                    // Poll dataset provider loading
                    if d.tab == 3 {
                        d.ai.tick_ds_fetch();
                    }
                    d.dt = dt;
                });

                let sz = window.inner_size();
                if sz.width > 0 && sz.height > 0 {
                    let w = sz.width as f64;
                    let h = sz.height as f64;
                    let mut tree = state.build_tree(&sheet, w, h);
                    tree.measure_text_nodes(|text, font_size| gpu.measure_text(text, font_size));
                    tree.layout(Size::new(w, h));

                    list.clear();
                    tree.paint(&mut list);
                    tree.post_paint();

                    // On scene tab, push viewport primitives offset by the viewport node's position
                    if state.on_scene() {
                        if let Some(vp) = tree.tagged_rect("scene-viewport") {
                            // Update viewport size for next frame's camera aspect
                            state.write(|d| {
                                d.scene_info.vp_size = (vp.w(), vp.h());
                                // Update camera aspect ratio
                                if let any_compute_core::scene::Projection::Perspective {
                                    aspect,
                                    ..
                                } = &mut d.scene_info.scene.camera.projection
                                {
                                    *aspect = vp.w() / vp.h().max(1.0);
                                }
                            });
                            let prims = state.read(|d| d.scene_info.viewport_prims.clone());
                            list.composite(vp, &prims);
                        }
                    }

                    // Graph mode: composite graph prims into viewport (any tab)
                    if state.read(|d| d.graph_mode) {
                        if let Some(vp) = tree.tagged_rect(tabs::graph::VIEWPORT_TAG) {
                            let prims = state.read(|d| d.ai.prims.clone());
                            list.composite(vp, &prims);
                        }
                    }

                    // AI tab (non-graph-mode): composite loss chart or model graph
                    if !state.read(|d| d.graph_mode) && state.read(|d| d.tab == 3) {
                        let is_graph =
                            state.read(|d| d.ai.center == tabs::graph::CenterView::ModelGraph);
                        if is_graph {
                            if let Some(vp) = tree.tagged_rect(tabs::graph::VIEWPORT_TAG) {
                                let prims = state.read(|d| d.ai.prims.clone());
                                list.composite(vp, &prims);
                            }
                        } else if let Some(vp) = tree.tagged_rect("ai-loss-chart") {
                            let prims = state.read(|d| d.ai.prims.clone());
                            list.composite(vp, &prims);
                        }

                        // Embedding scatter plot
                        if let Some(vp) = tree.tagged_rect("ai-ds-scatter") {
                            state.write(|d| d.ai.build_scatter_prims(vp.w(), vp.h()));
                            let prims = state.read(|d| d.ai.scatter_prims.clone());
                            list.composite(vp, &prims);
                        }
                    }

                    // On Browser → Preview subtab, overlay the Page below the nav bar
                    if state.on_browser_preview() {
                        // Re-measure text if the page was reloaded
                        let needs = state.read(|d| d.browser.needs_measure);
                        if needs {
                            let ok = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                state.write(|d| {
                                    if let Some(page) = &mut d.browser.page {
                                        page.measure_text_nodes(|text, font_size| {
                                            gpu.measure_text(text, font_size)
                                        });
                                        page.start_animations();
                                    }
                                    d.browser.needs_measure = false;
                                });
                            }));
                            if ok.is_err() {
                                log::error!("measure_text_nodes panicked — replacing page");
                                state.write(|d| {
                                    d.browser.page = Some(Page::load(ERROR_PAGE_HTML));
                                    d.browser.needs_measure = false;
                                });
                            }
                        }

                        if let Some(vp) = tree.tagged_rect(BVP) {
                            let dom_list =
                                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                    state.write(|d| {
                                        d.browser.page.as_mut().map(|page| {
                                            page.layout(Size::new(vp.w(), vp.h()));
                                            let mut dl = RenderList::default();
                                            page.paint(&mut dl);
                                            page.post_paint();
                                            dl
                                        })
                                    })
                                }));
                            match dom_list {
                                Ok(Some(dl)) => list.composite(vp, &dl),
                                Ok(None) => {}
                                Err(_) => {
                                    log::error!("page layout/paint panicked — replacing page");
                                    state.write(|d| {
                                        d.browser.page = Some(Page::load(ERROR_PAGE_HTML));
                                        d.browser.needs_measure = true;
                                    });
                                }
                            }
                        }
                    }

                    gpu.paint(&list);
                    last_tree = Some(tree);

                    // Update window title with FPS
                    let fps = state.read(|d| d.fps);
                    let nodes = list.len();
                    window.set_title(&format!(
                        "any-compute — Showcase | {fps} FPS | {nodes} prims"
                    ));
                }
            }
            _ => {}
        },
        _ => {}
    });
}
