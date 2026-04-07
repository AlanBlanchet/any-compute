//! Input event types — pointer, keyboard, focus, scroll.

use crate::layout::Point;

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
    PointerDown {
        pos: Point,
        button: Button,
    },
    PointerUp {
        pos: Point,
        button: Button,
    },
    PointerMove {
        pos: Point,
    },
    PointerEnter {
        pos: Point,
    },
    PointerLeave {
        pos: Point,
    },
    Scroll {
        pos: Point,
        delta: Point,
    },
    KeyDown {
        key: String,
        modifiers: Modifiers,
    },
    KeyUp {
        key: String,
        modifiers: Modifiers,
    },
    /// Character input — fired when a printable character is entered (like
    /// the browser `input` event on `<textarea>`). Distinct from `KeyDown`
    /// which carries key names; this carries the actual text to insert.
    TextInput {
        text: String,
    },
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
            | Self::PointerLeave { pos }
            | Self::Scroll { pos, .. } => Some(*pos),
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
