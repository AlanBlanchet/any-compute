use crate::style::StyleOp;
use any_compute_core::animation::Easing;

#[derive(Debug, Clone, PartialEq)]
pub struct TransitionSpec {
    /// CSS property name (`"all"`, `"opacity"`, `"transform"`, etc.).
    pub property: String,
    /// Duration in seconds.
    pub duration_secs: f64,
    /// Easing function.
    pub easing: Easing,
    /// Delay before start in seconds.
    pub delay_secs: f64,
}

/// How many times an animation repeats.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AnimationIterCount {
    /// Repeat a fixed number of times.
    Count(f64),
    /// Repeat forever.
    Infinite,
}

impl Default for AnimationIterCount {
    fn default() -> Self {
        Self::Count(1.0)
    }
}

css_enums! {
    /// Animation playback direction.
    AnimationDirection [Normal, Normal] {
        "normal" => Normal, "reverse" => Reverse,
        "alternate" => Alternate, "alternate-reverse" => AlternateReverse
    }

    /// Animation fill mode (CSS `animation-fill-mode`).
    AnimationFillMode [None, None] {
        "none" => None, "forwards" => Forwards,
        "backwards" => Backwards, "both" => Both
    }
}

/// Parsed CSS animation declaration: `animation: name duration easing delay count direction fill`.
#[derive(Debug, Clone, PartialEq)]
pub struct AnimationSpec {
    pub name: String,
    pub duration_secs: f64,
    pub easing: Easing,
    pub delay_secs: f64,
    pub iteration_count: AnimationIterCount,
    pub direction: AnimationDirection,
    pub fill_mode: AnimationFillMode,
}

/// A single keyframe stop within `@keyframes`.
#[derive(Debug, Clone)]
pub struct Keyframe {
    /// Progress point: 0.0 = `from`, 1.0 = `to`, 0.5 = `50%`.
    pub stop: f64,
    /// Style operations to apply at this stop.
    pub ops: Vec<StyleOp>,
}

// ── Selector model ──────────────────────────────────────────────────────────

/// CSS pseudo-class.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PseudoClass {
    Hover,
    Focus,
    Active,
    Visited,
    FirstChild,
    LastChild,
    /// `nth-child(an+b)` — stored as (a, b).
    NthChild(i32, i32),
}

/// Combinator between selector segments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Combinator {
    /// First segment (no combinator).
    None,
    /// Descendant (space): `A B`.
    Descendant,
    /// Direct child: `A > B`.
    Child,
}

/// One compound segment of a selector: `div.card#main:hover`.
#[derive(Debug, Clone, Default)]
pub struct SelectorSegment {
    pub tag: Option<String>,
    pub classes: Vec<String>,
    pub id: Option<String>,
    pub pseudos: Vec<PseudoClass>,
    pub universal: bool,
}

/// Selector specificity as (ids, classes+pseudos, tags).
pub type Specificity = (u16, u16, u16);

/// A fully parsed CSS selector with specificity.
#[derive(Debug, Clone)]
pub struct ParsedSelector {
    /// Chain of (combinator, compound-selector) segments.
    pub segments: Vec<(Combinator, SelectorSegment)>,
    /// CSS specificity: (ids, classes+pseudos, tags).
    pub specificity: Specificity,
}

/// A rule with a complex selector (descendant / child / pseudo-class).
#[derive(Debug, Clone)]
pub struct ComplexRule {
    /// The parsed selector with specificity.
    pub selector: ParsedSelector,
    /// The compiled style operations + transition/animation metadata.
    pub payload: RulePayload,
}

/// Everything compiled from one CSS rule block.
#[derive(Debug, Clone, Default)]
pub struct RulePayload {
    pub ops: Vec<StyleOp>,
    pub transitions: Vec<TransitionSpec>,
    pub animations: Vec<AnimationSpec>,
}

impl RulePayload {
    pub fn extend(&mut self, other: &RulePayload) {
        self.ops.extend_from_slice(&other.ops);
        self.transitions.extend_from_slice(&other.transitions);
        self.animations.extend_from_slice(&other.animations);
    }
}
