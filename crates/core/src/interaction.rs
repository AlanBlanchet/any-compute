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

use crate::layout::{Point, Rect};

// ── Pointer button ──────────────────────────────────────────────────────────

/// Pointer (mouse / touch) button.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Button {
    Primary,
    Secondary,
    Middle,
}

// ── Input event ─────────────────────────────────────────────────────────────

/// Unified input event — references [`Point`] for all positions.
#[derive(Debug, Clone)]
pub enum InputEvent {
    PointerDown { pos: Point, button: Button },
    PointerUp { pos: Point, button: Button },
    PointerMove { pos: Point },
    PointerEnter { pos: Point },
    PointerLeave { pos: Point },
    Scroll { delta: Point },
    KeyDown { key: String, modifiers: Modifiers },
    KeyUp { key: String, modifiers: Modifiers },
    Focus,
    Blur,
}

impl InputEvent {
    /// Extract position from pointer events; `None` for keyboard/focus.
    pub fn pos(&self) -> Option<Point> {
        match self {
            Self::PointerDown { pos, .. }
            | Self::PointerUp { pos, .. }
            | Self::PointerMove { pos }
            | Self::PointerEnter { pos }
            | Self::PointerLeave { pos } => Some(*pos),
            _ => None,
        }
    }

    /// Is this a pointer-class event?
    pub fn is_pointer(&self) -> bool {
        self.pos().is_some()
    }
}

/// Keyboard modifiers — matches web's modifier key model.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Modifiers {
    pub shift: bool,
    pub ctrl: bool,
    pub alt: bool,
    pub meta: bool,
}

// ── Propagation ─────────────────────────────────────────────────────────────

/// Event propagation phase (mirrors the W3C DOM event model).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// Traveling from root toward target.
    Capture,
    /// On the target itself.
    Target,
    /// Bubbling back up from target to root.
    Bubble,
}

/// Wrapper around an event that carries propagation state.
#[derive(Debug, Clone)]
pub struct EventContext {
    pub event: InputEvent,
    pub phase: Phase,
    /// Set to `true` to stop the event from reaching further listeners.
    pub stopped: bool,
    /// Set to `true` to prevent default behavior.
    pub default_prevented: bool,
}

impl EventContext {
    pub fn new(event: InputEvent) -> Self {
        Self {
            event,
            phase: Phase::Capture,
            stopped: false,
            default_prevented: false,
        }
    }

    pub fn stop_propagation(&mut self) {
        self.stopped = true;
    }

    pub fn prevent_default(&mut self) {
        self.default_prevented = true;
    }
}

/// Outcome of processing an event — tells the host whether to repaint, etc.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventResponse {
    /// Event was ignored.
    Ignored,
    /// State changed — host should repaint.
    Consumed,
}

/// Trait for anything that can receive input events within a bounding rect.
pub trait Interactive {
    /// The bounding area this element occupies.
    fn bounds(&self) -> Rect;

    /// Handle an input event with propagation context.
    fn handle_event(&mut self, ctx: &mut EventContext) -> EventResponse;
}

// ── Dispatch result ─────────────────────────────────────────────────────────

/// Result of dispatching an event through a tree with capture → target → bubble.
///
/// Carries the tag chain (root → target in order), the target node index,
/// and propagation flags.  The host inspects `tags` to decide what action
/// to take and checks `stopped` / `default_prevented` to honour propagation.
#[derive(Debug, Clone, Default)]
pub struct DispatchResult {
    /// Tags encountered along the path from root → target.
    /// Order: outermost first, target last.
    pub tags: Vec<String>,
    /// Whether `stop_propagation()` was called during dispatch.
    pub stopped: bool,
    /// Whether `prevent_default()` was called during dispatch.
    pub default_prevented: bool,
}

impl DispatchResult {
    /// The deepest (innermost) tag — the one closest to (or on) the target.
    pub fn target_tag(&self) -> Option<&str> {
        self.tags.last().map(|s| s.as_str())
    }

    /// Walk tags from innermost → outermost (bubble order).
    pub fn bubble_tags(&self) -> impl Iterator<Item = &str> {
        self.tags.iter().rev().map(|s| s.as_str())
    }
}

// ── Hover tracking ──────────────────────────────────────────────────────────

/// Tracks the currently hovered tag across frames.
///
/// Call [`update`](Self::update) each time the cursor moves.
/// If the hovered tag changes it returns a [`HoverDelta`] with the old
/// and new tag so the host can start transitions.
#[derive(Debug, Default)]
pub struct HoverState {
    /// Currently hovered tag (the deepest tagged node under the cursor).
    pub current: Option<String>,
}

/// What changed between two hover checks.
#[derive(Debug, Clone)]
pub struct HoverDelta {
    /// Tag that was previously hovered (`None` = nothing was hovered).
    pub left: Option<String>,
    /// Tag that is now hovered (`None` = cursor left all tagged nodes).
    pub entered: Option<String>,
}

impl HoverState {
    /// Update the hovered tag.  Returns `Some(delta)` when the tag changes.
    pub fn update(&mut self, new_tag: Option<String>) -> Option<HoverDelta> {
        if self.current == new_tag {
            return None;
        }
        let delta = HoverDelta {
            left: self.current.take(),
            entered: new_tag.clone(),
        };
        self.current = new_tag;
        Some(delta)
    }
}

// ── Focus tracking ──────────────────────────────────────────────────────────

/// Tracks the focused tag for keyboard dispatch.
#[derive(Debug, Default)]
pub struct FocusState {
    pub focused: Option<String>,
}

impl FocusState {
    /// Move focus to a new tag. Returns the previously focused tag.
    pub fn focus(&mut self, tag: Option<String>) -> Option<String> {
        std::mem::replace(&mut self.focused, tag)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert!(hs.update(None).is_none()); // no change
        let d = hs.update(Some("tab-0".into())).unwrap();
        assert_eq!(d.left, None);
        assert_eq!(d.entered.as_deref(), Some("tab-0"));

        let d = hs.update(Some("tab-1".into())).unwrap();
        assert_eq!(d.left.as_deref(), Some("tab-0"));
        assert_eq!(d.entered.as_deref(), Some("tab-1"));

        assert!(hs.update(Some("tab-1".into())).is_none()); // same

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
}
