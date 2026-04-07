//! Event propagation — context, phases, dispatch results.

use super::event::InputEvent;
use crate::layout::Rect;

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
    /// Whether any node was restyled (hover/active state changed).
    /// Callers can skip repaint when `false`.
    pub restyled: bool,
    /// CSS cursor resolved from the hit node (empty = default).
    /// Walk up from the target until a non-default cursor is found.
    pub cursor: String,
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
