//! Hover and focus state tracking across frames.

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
