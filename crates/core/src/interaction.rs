//! Input / interaction model — framework-agnostic event types and hit-testing.
//!
//! All spatial data references [`layout::Point`] and [`layout::Rect`] — never raw x/y.
//! Supports web-like event propagation: capture → target → bubble.

use crate::layout::{Point, Rect};

/// Pointer (mouse / touch) button.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Button {
    Primary,
    Secondary,
    Middle,
}

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

/// Keyboard modifiers — matches web's modifier key model.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Modifiers {
    pub shift: bool,
    pub ctrl: bool,
    pub alt: bool,
    pub meta: bool,
}

/// Event propagation phase (mirrors the DOM event model).
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
        // Ensure all variants construct without panic
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
}
