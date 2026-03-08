//! Scriptable interaction replay — zero-OS-interaction scenario execution.
//!
//! [`Action`], [`StepResult`], and [`Scenario`] define a scripted sequence of
//! pointer/keyboard/scroll events that are replayed against a [`Tree`] without
//! touching the real mouse, keyboard, or display.
//!
//! ## Usage
//!
//! ```ignore
//! let scenario = Scenario::new()
//!     .click(Point::new(100.0, 50.0))
//!     .capture()
//!     .hover(Point::new(300.0, 200.0))
//!     .assert_tag(Point::new(300.0, 200.0), "card-1")
//!     .capture();
//!
//! let results = replay(&mut tree, &scenario);
//! ```

use any_compute_core::interaction::{Button, DispatchResult, InputEvent};
use any_compute_core::layout::Point;
use any_compute_dom::tree::Tree;

// ── Action ──────────────────────────────────────────────────────────────────

/// A single scripted interaction step.
#[derive(Debug, Clone)]
pub enum Action {
    /// Full click (pointer-down + pointer-up) at the given point.
    Click(Point),
    /// Hover (pointer-move) to the given point.
    Hover(Point),
    /// Scroll at the given point with the given delta.
    Scroll { pos: Point, delta: Point },
    /// Dispatch an arbitrary [`InputEvent`].
    Dispatch(InputEvent),
    /// Assert that `tag_at(pos)` equals `expected` (self-validating scripts).
    AssertTag { pos: Point, expected: String },
    /// Mark this step for screenshot capture by the host.
    Capture,
}

// ── StepResult ──────────────────────────────────────────────────────────────

/// Result of replaying one [`Action`] against a tree.
#[derive(Debug, Clone)]
pub struct StepResult {
    /// Which action was executed (index in the scenario).
    pub index: usize,
    /// The action that was executed.
    pub action: Action,
    /// Dispatch result for click/hover/dispatch actions; `None` for assert/capture.
    pub dispatch: Option<DispatchResult>,
    /// For `AssertTag`: `Some(true)` if matched, `Some(false)` if not, `None` otherwise.
    pub assertion: Option<bool>,
    /// True when this step is a `Capture` — the host should take a screenshot now.
    pub capture: bool,
}

impl StepResult {
    /// Build a result with only dispatch info (click, hover, dispatch actions).
    pub fn dispatched(index: usize, action: Action, dispatch: DispatchResult) -> Self {
        Self {
            index,
            action,
            dispatch: Some(dispatch),
            assertion: None,
            capture: false,
        }
    }

    /// Build a result with no output (scroll, ignored actions).
    pub fn silent(index: usize, action: Action) -> Self {
        Self {
            index,
            action,
            dispatch: None,
            assertion: None,
            capture: false,
        }
    }

    /// Build an assertion result.
    pub fn asserted(index: usize, action: Action, pass: bool) -> Self {
        Self {
            index,
            action,
            dispatch: None,
            assertion: Some(pass),
            capture: false,
        }
    }

    /// Build a capture marker.
    pub fn captured(index: usize) -> Self {
        Self {
            index,
            action: Action::Capture,
            dispatch: None,
            assertion: None,
            capture: true,
        }
    }
}

// ── Scenario ────────────────────────────────────────────────────────────────

/// Ordered sequence of [`Action`]s to replay against a tree.
#[derive(Debug, Clone, Default)]
pub struct Scenario {
    pub actions: Vec<Action>,
}

impl Scenario {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(mut self, action: Action) -> Self {
        self.actions.push(action);
        self
    }

    pub fn click(self, pos: impl Into<Point>) -> Self {
        self.push(Action::Click(pos.into()))
    }

    pub fn hover(self, pos: impl Into<Point>) -> Self {
        self.push(Action::Hover(pos.into()))
    }

    pub fn scroll(self, pos: impl Into<Point>, delta: impl Into<Point>) -> Self {
        self.push(Action::Scroll {
            pos: pos.into(),
            delta: delta.into(),
        })
    }

    pub fn dispatch(self, event: InputEvent) -> Self {
        self.push(Action::Dispatch(event))
    }

    pub fn assert_tag(self, pos: impl Into<Point>, expected: impl Into<String>) -> Self {
        self.push(Action::AssertTag {
            pos: pos.into(),
            expected: expected.into(),
        })
    }

    pub fn capture(self) -> Self {
        self.push(Action::Capture)
    }
}

// ── Replay ──────────────────────────────────────────────────────────────────

/// Execute a single [`Action`] against a tree and return its result.
pub fn replay_step(tree: &mut Tree, action: &Action, index: usize) -> StepResult {
    match action {
        Action::Click(pos) => {
            tree.dispatch(InputEvent::PointerDown {
                pos: *pos,
                button: Button::Primary,
            });
            let d = tree.dispatch(InputEvent::PointerUp {
                pos: *pos,
                button: Button::Primary,
            });
            StepResult::dispatched(index, action.clone(), d)
        }
        Action::Hover(pos) => {
            let d = tree.dispatch(InputEvent::PointerMove { pos: *pos });
            StepResult::dispatched(index, action.clone(), d)
        }
        Action::Scroll { pos, delta } => {
            tree.scroll(*pos, *delta);
            StepResult::silent(index, action.clone())
        }
        Action::Dispatch(event) => {
            let d = tree.dispatch(event.clone());
            StepResult::dispatched(index, action.clone(), d)
        }
        Action::AssertTag { pos, expected } => {
            let pass = tree.tag_at(*pos).as_deref() == Some(expected.as_str());
            StepResult::asserted(index, action.clone(), pass)
        }
        Action::Capture => StepResult::captured(index),
    }
}

/// Replay a full [`Scenario`] against a tree. Returns one [`StepResult`] per action.
pub fn replay(tree: &mut Tree, scenario: &Scenario) -> Vec<StepResult> {
    scenario
        .actions
        .iter()
        .enumerate()
        .map(|(i, action)| replay_step(tree, action, i))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use any_compute_core::layout::Size;
    use any_compute_core::render::Color;
    use any_compute_dom::style::Style;

    fn test_tree() -> Tree {
        let mut tree = Tree::new(Style::default().w(400.0).h(300.0));
        let btn = tree.add_box(tree.root, Style::default().w(100.0).h(50.0));
        tree.tag(btn, "my-button");
        tree.layout(Size::new(400.0, 300.0));
        tree
    }

    #[test]
    fn replay_click_dispatches_and_returns_tag() {
        let mut tree = test_tree();
        let scenario = Scenario::new().click(Point::new(50.0, 25.0));
        let results = replay(&mut tree, &scenario);
        assert_eq!(results.len(), 1);
        let r = &results[0];
        assert_eq!(r.index, 0);
        assert!(!r.capture);
        let d = r.dispatch.as_ref().unwrap();
        assert_eq!(d.target_tag(), Some("my-button"));
    }

    #[test]
    fn replay_assert_tag_passes_and_fails() {
        let mut tree = test_tree();
        let scenario = Scenario::new()
            .assert_tag(Point::new(50.0, 25.0), "my-button")
            .assert_tag(Point::new(50.0, 25.0), "wrong-tag");
        let results = replay(&mut tree, &scenario);
        assert_eq!(results[0].assertion, Some(true));
        assert_eq!(results[1].assertion, Some(false));
    }

    #[test]
    fn replay_capture_sets_flag() {
        let mut tree = Tree::new(Style::default().w(400.0).h(300.0));
        tree.layout(Size::new(400.0, 300.0));

        let scenario = Scenario::new()
            .capture()
            .click(Point::new(10.0, 10.0))
            .capture();
        let results = replay(&mut tree, &scenario);
        assert!(results[0].capture);
        assert!(!results[1].capture);
        assert!(results[2].capture);
    }

    #[test]
    fn replay_hover_dispatches_pointer_move() {
        let mut tree = test_tree();
        let scenario = Scenario::new().hover(Point::new(50.0, 25.0));
        let results = replay(&mut tree, &scenario);
        let d = results[0].dispatch.as_ref().unwrap();
        assert_eq!(d.target_tag(), Some("my-button"));
    }

    #[test]
    fn replay_full_scenario_sequence() {
        let mut tree = Tree::new(Style::default().w(400.0).h(300.0));
        let a = tree.add_box(tree.root, Style::default().w(200.0).h(150.0));
        tree.tag(a, "box-a");
        let b = tree.add_box(tree.root, Style::default().w(200.0).h(150.0));
        tree.tag(b, "box-b");
        tree.layout(Size::new(400.0, 300.0));

        let scenario = Scenario::new()
            .capture()
            .click(Point::new(100.0, 75.0))
            .assert_tag(Point::new(100.0, 75.0), "box-a")
            .hover(Point::new(100.0, 225.0))
            .assert_tag(Point::new(100.0, 225.0), "box-b")
            .capture();

        let results = replay(&mut tree, &scenario);
        assert_eq!(results.len(), 6);
        assert!(results[0].capture);
        assert!(results[1].dispatch.is_some());
        assert_eq!(results[2].assertion, Some(true));
        assert!(results[3].dispatch.is_some());
        assert_eq!(results[4].assertion, Some(true));
        assert!(results[5].capture);
    }
}
