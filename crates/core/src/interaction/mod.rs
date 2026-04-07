//! Input / interaction model — framework-agnostic event types, propagation, hover tracking.
//!
//! All spatial data references [`layout::Point`] and [`layout::Rect`] — never raw x/y.
//! Supports web-like event propagation: capture → target → bubble.
//!
//! ## Architecture
//!
//! - [`InputEvent`] — single enum covering pointer, keyboard, focus/blur, scroll.
//! - [`EventContext`] — wraps an event with propagation state (phase, stopped, default_prevented).
//! - [`DispatchResult`] — returned from tree dispatch: the tag chain from root → target, whether
//!   propagation was stopped, and whether the default was prevented.
//! - [`HoverState`] — tracks the currently hovered tag across frames, emits enter/leave deltas.

mod dispatch;
mod event;
mod state;
mod text_input;

pub use dispatch::*;
pub use event::*;
pub use state::*;
pub use text_input::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::Point;

    #[test]
    fn event_context_propagation() {
        let mut ctx = EventContext::new(InputEvent::Focus);
        assert!(!ctx.stopped);
        assert!(!ctx.default_prevented);
        assert_eq!(ctx.phase, Phase::Capture);

        ctx.stop_propagation();
        assert!(ctx.stopped);

        ctx.prevent_default();
        assert!(ctx.default_prevented);
    }

    #[test]
    fn modifiers_default() {
        let m = Modifiers::default();
        assert!(!m.shift && !m.ctrl && !m.alt && !m.meta);
    }

    #[test]
    fn input_event_variants() {
        let _ = InputEvent::PointerDown {
            pos: Point::ZERO,
            button: Button::Primary,
        };
        let _ = InputEvent::PointerUp {
            pos: Point::ZERO,
            button: Button::Secondary,
        };
        let _ = InputEvent::PointerMove { pos: Point::ZERO };
        let _ = InputEvent::PointerEnter { pos: Point::ZERO };
        let _ = InputEvent::PointerLeave { pos: Point::ZERO };
        let _ = InputEvent::Scroll {
            pos: Point::ZERO,
            delta: Point::new(0.0, -10.0),
        };
        let _ = InputEvent::KeyDown {
            key: "a".into(),
            modifiers: Modifiers::default(),
        };
        let _ = InputEvent::KeyUp {
            key: "a".into(),
            modifiers: Modifiers::default(),
        };
        let _ = InputEvent::Focus;
        let _ = InputEvent::Blur;
    }

    #[test]
    fn input_event_pos_extraction() {
        let e = InputEvent::PointerDown {
            pos: Point::new(10.0, 20.0),
            button: Button::Primary,
        };
        assert_eq!(e.pos(), Some(Point::new(10.0, 20.0)));
        assert!(e.is_pointer());

        let k = InputEvent::KeyDown {
            key: "a".into(),
            modifiers: Modifiers::default(),
        };
        assert_eq!(k.pos(), None);
        assert!(!k.is_pointer());
    }

    #[test]
    fn dispatch_result_tags() {
        let r = DispatchResult {
            tags: vec!["root".into(), "sidebar".into(), "tab-0".into()],
            ..Default::default()
        };
        assert_eq!(r.target_tag(), Some("tab-0"));
        let bubble: Vec<&str> = r.bubble_tags().collect();
        assert_eq!(bubble, vec!["tab-0", "sidebar", "root"]);
    }

    #[test]
    fn hover_state_delta() {
        let mut hs = HoverState::default();
        assert!(hs.update(None).is_none());
        let d = hs.update(Some("tab-0".into())).unwrap();
        assert_eq!(d.left, None);
        assert_eq!(d.entered.as_deref(), Some("tab-0"));

        let d = hs.update(Some("tab-1".into())).unwrap();
        assert_eq!(d.left.as_deref(), Some("tab-0"));
        assert_eq!(d.entered.as_deref(), Some("tab-1"));

        assert!(hs.update(Some("tab-1".into())).is_none());

        let d = hs.update(None).unwrap();
        assert_eq!(d.left.as_deref(), Some("tab-1"));
        assert_eq!(d.entered, None);
    }

    #[test]
    fn focus_state_tracking() {
        let mut fs = FocusState::default();
        let prev = fs.focus(Some("input-1".into()));
        assert_eq!(prev, None);
        let prev = fs.focus(Some("input-2".into()));
        assert_eq!(prev.as_deref(), Some("input-1"));
        let prev = fs.focus(None);
        assert_eq!(prev.as_deref(), Some("input-2"));
    }

    #[test]
    fn text_input_insert_and_backspace() {
        let mut ti = TextInput::new("hello");
        assert_eq!(ti.cursor, 5);
        ti.insert(" world");
        assert_eq!(ti.text, "hello world");
        ti.backspace();
        assert_eq!(ti.text, "hello worl");
        ti.left(false);
        ti.left(false);
        ti.backspace();
        assert_eq!(ti.text, "hello wrl");
    }

    #[test]
    fn text_input_selection() {
        let mut ti = TextInput::new("abcdef");
        ti.cursor = 2;
        ti.sel = Some(4);
        let parts = ti.parts();
        assert_eq!(parts.before, "ab");
        assert_eq!(parts.selected, "cd");
        assert_eq!(parts.after, "ef");
        assert!(parts.has_selection);

        ti.delete_sel();
        assert_eq!(ti.text, "abef");
        assert_eq!(ti.cursor, 2);
    }

    #[test]
    fn text_input_select_all_and_replace() {
        let mut ti = TextInput::new("old text");
        ti.focused = true;
        ti.select_all();
        assert_eq!(ti.sel, Some(0));
        assert_eq!(ti.cursor, ti.text.len());
        ti.insert("new");
        assert_eq!(ti.text, "new");
    }

    #[test]
    fn text_input_home_end_shift() {
        let mut ti = TextInput::new("abc");
        ti.cursor = 1;
        ti.end(true);
        assert_eq!(ti.sel, Some(1));
        assert_eq!(ti.cursor, 3);
        let parts = ti.parts();
        assert_eq!(parts.selected, "bc");

        ti.home(false);
        assert_eq!(ti.cursor, 0);
        assert_eq!(ti.sel, None);
    }

    #[test]
    fn text_input_handle_key() {
        let mut ti = TextInput::new("test");
        ti.focused = true;
        let ctrl_a = Modifiers {
            ctrl: true,
            ..Default::default()
        };
        assert!(ti.handle_key("a", ctrl_a));
        assert_eq!(ti.sel, Some(0));
        assert_eq!(ti.cursor, 4);
    }

    #[test]
    fn text_input_focus_selects_all() {
        let mut ti = TextInput::new("url");
        ti.focus();
        assert!(ti.focused);
        assert_eq!(ti.sel, Some(0));
        assert_eq!(ti.cursor, 3);
    }
}
