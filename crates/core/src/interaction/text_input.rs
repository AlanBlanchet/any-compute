//! Reusable single-line text editing widget with cursor, selection, and keyboard handling.

use super::event::Modifiers;

/// Reusable single-line text input with cursor, selection, and keyboard handling.
///
/// Captures all text editing state + operations (insert, delete, arrows, Home/End,
/// Ctrl+A) in a single struct. Consumers only need to feed key events and read
/// the resulting text/cursor/selection state for rendering.
#[derive(Debug, Clone)]
pub struct TextInput {
    pub text: String,
    pub cursor: usize,
    /// Selection anchor byte offset. `None` = no selection.
    pub sel: Option<usize>,
    pub focused: bool,
}

impl Default for TextInput {
    fn default() -> Self {
        Self {
            text: String::new(),
            cursor: 0,
            sel: None,
            focused: false,
        }
    }
}

/// Which part of the text is before/in/after the selection (for rendering).
#[derive(Debug, Clone)]
pub struct TextParts<'a> {
    pub before: &'a str,
    pub selected: &'a str,
    pub after: &'a str,
    /// Cursor byte position (for rendering the caret when no selection).
    pub cursor: usize,
    pub has_selection: bool,
}

impl TextInput {
    pub fn new(text: impl Into<String>) -> Self {
        let text = text.into();
        let cursor = text.len();
        Self {
            text,
            cursor,
            sel: None,
            focused: false,
        }
    }

    /// Focus and select all text (Chrome-like behavior).
    pub fn focus(&mut self) {
        self.focused = true;
        self.select_all();
    }

    pub fn blur(&mut self) {
        self.focused = false;
        self.sel = None;
    }

    /// Ordered (lo, hi) of cursor + selection anchor, or just (cursor, cursor).
    pub fn sel_range(&self) -> (usize, usize) {
        match self.sel {
            Some(anchor) => (anchor.min(self.cursor), anchor.max(self.cursor)),
            None => (self.cursor, self.cursor),
        }
    }

    /// Split text into before/selected/after for rendering.
    pub fn parts(&self) -> TextParts<'_> {
        let (lo, hi) = self.sel_range();
        TextParts {
            before: &self.text[..lo],
            selected: &self.text[lo..hi],
            after: &self.text[hi..],
            cursor: self.cursor,
            has_selection: lo < hi,
        }
    }

    /// Delete selected text (if any) and collapse cursor. Returns true if there was a selection.
    pub fn delete_sel(&mut self) -> bool {
        let (lo, hi) = self.sel_range();
        if lo < hi {
            self.text.replace_range(lo..hi, "");
            self.cursor = lo;
            self.sel = None;
            true
        } else {
            self.sel = None;
            false
        }
    }

    /// Next char boundary after `pos`.
    fn next_boundary(&self, pos: usize) -> usize {
        self.text[pos..]
            .char_indices()
            .nth(1)
            .map_or(self.text.len(), |(i, _)| pos + i)
    }

    /// Prev char boundary before `pos`.
    fn prev_boundary(&self, pos: usize) -> usize {
        self.text[..pos]
            .char_indices()
            .next_back()
            .map_or(0, |(i, _)| i)
    }

    /// Ensure selection anchor exists when shift is held.
    fn ensure_sel_anchor(&mut self, shift: bool) {
        if shift && self.sel.is_none() {
            self.sel = Some(self.cursor);
        } else if !shift {
            self.sel = None;
        }
    }

    /// Insert text, replacing selection if active.
    pub fn insert(&mut self, s: &str) {
        self.delete_sel();
        self.text.insert_str(self.cursor, s);
        self.cursor += s.len();
    }

    /// Backspace: delete selection or char before cursor.
    pub fn backspace(&mut self) {
        if !self.delete_sel() && self.cursor > 0 {
            let prev = self.prev_boundary(self.cursor);
            self.text.replace_range(prev..self.cursor, "");
            self.cursor = prev;
        }
    }

    /// Delete key: delete selection or char after cursor.
    pub fn delete(&mut self) {
        if !self.delete_sel() && self.cursor < self.text.len() {
            let next = self.next_boundary(self.cursor);
            self.text.replace_range(self.cursor..next, "");
        }
    }

    /// Move cursor left.
    pub fn left(&mut self, shift: bool) {
        if shift {
            if self.sel.is_none() {
                self.sel = Some(self.cursor);
            }
        } else if self.sel.is_some() {
            self.cursor = self.sel_range().0;
            self.sel = None;
            return;
        }
        if self.cursor > 0 {
            self.cursor = self.prev_boundary(self.cursor);
        }
    }

    /// Move cursor right.
    pub fn right(&mut self, shift: bool) {
        if shift {
            if self.sel.is_none() {
                self.sel = Some(self.cursor);
            }
        } else if self.sel.is_some() {
            self.cursor = self.sel_range().1;
            self.sel = None;
            return;
        }
        if self.cursor < self.text.len() {
            self.cursor = self.next_boundary(self.cursor);
        }
    }

    /// Home: jump to start.
    pub fn home(&mut self, shift: bool) {
        self.ensure_sel_anchor(shift);
        self.cursor = 0;
    }

    /// End: jump to end.
    pub fn end(&mut self, shift: bool) {
        self.ensure_sel_anchor(shift);
        self.cursor = self.text.len();
    }

    /// Select all.
    pub fn select_all(&mut self) {
        self.sel = Some(0);
        self.cursor = self.text.len();
    }

    /// Handle a keyboard event. Returns true if consumed.
    pub fn handle_key(&mut self, key: &str, mods: Modifiers) -> bool {
        if !self.focused {
            return false;
        }
        match key {
            "a" if mods.ctrl => {
                self.select_all();
                true
            }
            "Backspace" => {
                self.backspace();
                true
            }
            "Delete" => {
                self.delete();
                true
            }
            "ArrowLeft" => {
                self.left(mods.shift);
                true
            }
            "ArrowRight" => {
                self.right(mods.shift);
                true
            }
            "Home" => {
                self.home(mods.shift);
                true
            }
            "End" => {
                self.end(mods.shift);
                true
            }
            _ => false,
        }
    }

    /// Handle text input (printable characters). Returns true if consumed.
    pub fn handle_text(&mut self, text: &str) -> bool {
        if !self.focused || text.is_empty() || !text.chars().all(|c| !c.is_control() || c == ' ') {
            return false;
        }
        self.insert(text);
        true
    }

    /// Set text and place cursor at end.
    pub fn set_text(&mut self, text: impl Into<String>) {
        self.text = text.into();
        self.cursor = self.text.len();
        self.sel = None;
    }
}
