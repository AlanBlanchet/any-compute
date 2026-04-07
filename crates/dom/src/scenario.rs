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

use crate::tree::Tree;
use any_compute_core::interaction::{Button, DispatchResult, InputEvent, Modifiers};
use any_compute_core::layout::Point;
use any_compute_core::render::{Color, PixelBuffer};

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
    /// Assert pixel color at a position in the last captured frame.
    AssertPixel {
        x: u32,
        y: u32,
        expected: Color,
        tolerance: u8,
    },
    /// Assert a rectangular region is uniformly one color in the last captured frame.
    AssertRegion {
        x: u32,
        y: u32,
        w: u32,
        h: u32,
        tolerance: u8,
    },
    /// Simulate a wait (for animation tick). The host decides how to interpret dt.
    Wait(f64),
    /// Type text (dispatches key events for each character).
    Type(String),
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
    /// For `AssertTag`/`AssertPixel`/`AssertRegion`: pass/fail.
    pub assertion: Option<bool>,
    /// True when this step is a `Capture` — the host should take a screenshot now.
    pub capture: bool,
    /// For `Wait` actions: the requested delay in seconds.
    pub wait_dt: Option<f64>,
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
            wait_dt: None,
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
            wait_dt: None,
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
            wait_dt: None,
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
            wait_dt: None,
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

    pub fn assert_pixel(self, x: u32, y: u32, expected: Color, tolerance: u8) -> Self {
        self.push(Action::AssertPixel {
            x,
            y,
            expected,
            tolerance,
        })
    }

    pub fn assert_region(self, x: u32, y: u32, w: u32, h: u32, tolerance: u8) -> Self {
        self.push(Action::AssertRegion {
            x,
            y,
            w,
            h,
            tolerance,
        })
    }

    pub fn wait(self, dt: f64) -> Self {
        self.push(Action::Wait(dt))
    }

    pub fn type_text(self, text: impl Into<String>) -> Self {
        self.push(Action::Type(text.into()))
    }
}

// ── Replay ──────────────────────────────────────────────────────────────────

/// Execute a single [`Action`] against a tree and return its result.
///
/// `last_capture` is the most recent pixel buffer (from a previous Capture step).
/// Pixel-assertion actions check against it.
pub fn replay_step(
    tree: &mut Tree,
    action: &Action,
    index: usize,
    last_capture: Option<&PixelBuffer>,
) -> StepResult {
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
            let d = tree.dispatch(InputEvent::Scroll {
                pos: *pos,
                delta: *delta,
            });
            StepResult::dispatched(index, action.clone(), d)
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
        Action::AssertPixel {
            x,
            y,
            expected,
            tolerance,
        } => {
            let pass = last_capture
                .map(|buf| {
                    let c = buf.pixel(*x, *y);
                    let t = *tolerance as i16;
                    (c.r as i16 - expected.r as i16).abs() <= t
                        && (c.g as i16 - expected.g as i16).abs() <= t
                        && (c.b as i16 - expected.b as i16).abs() <= t
                })
                .unwrap_or(false);
            StepResult::asserted(index, action.clone(), pass)
        }
        Action::AssertRegion {
            x,
            y,
            w,
            h,
            tolerance,
        } => {
            let pass = last_capture
                .map(|buf| buf.region_uniform(*x, *y, *w, *h, *tolerance))
                .unwrap_or(false);
            StepResult::asserted(index, action.clone(), pass)
        }
        Action::Wait(dt) => {
            let mut r = StepResult::silent(index, action.clone());
            r.wait_dt = Some(*dt);
            r
        }
        Action::Type(text) => {
            for ch in text.chars() {
                tree.dispatch(InputEvent::KeyDown {
                    key: ch.to_string(),
                    modifiers: Modifiers::default(),
                });
            }
            StepResult::silent(index, action.clone())
        }
    }
}

/// Replay a full [`Scenario`] against a tree. Returns one [`StepResult`] per action.
///
/// Pixel-assertion actions (`AssertPixel`, `AssertRegion`) check against a
/// `capture_fn` callback. When a `Capture` action is encountered, `capture_fn`
/// is called to produce a PixelBuffer snapshot of the current tree state.
/// If no `capture_fn` is provided, pixel assertions always fail.
pub fn replay(tree: &mut Tree, scenario: &Scenario) -> Vec<StepResult> {
    replay_with(tree, scenario, None::<fn(&Tree) -> PixelBuffer>)
}

/// Replay with an optional capture callback for pixel-level assertions.
pub fn replay_with(
    tree: &mut Tree,
    scenario: &Scenario,
    capture_fn: Option<impl Fn(&Tree) -> PixelBuffer>,
) -> Vec<StepResult> {
    let mut results = Vec::with_capacity(scenario.actions.len());
    let mut last_capture: Option<PixelBuffer> = None;

    for (i, action) in scenario.actions.iter().enumerate() {
        let result = replay_step(tree, action, i, last_capture.as_ref());

        // If this step is a capture, invoke the callback to get the pixel buffer.
        if result.capture {
            if let Some(ref f) = capture_fn {
                last_capture = Some(f(tree));
            }
        }

        results.push(result);
    }
    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::style::Style;
    use any_compute_core::layout::Size;

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
