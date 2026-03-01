//! Our DOM — a fully owned, GPU-optimized scene graph.
//!
//! Every node carries its own style, layout solution, hint flags, and optional
//! transitions — the engine reads hints to decide caching, batching, and compute
//! strategy per-subtree.  No external framework, no virtual DOM diff — we own
//! the full pipeline from builder → layout → paint → event dispatch.
//!
//! ## Key types
//!
//! - [`Node`] — polymorphic scene-graph node (Div/Text/Bar/custom via extension)
//! - [`Style`] — flexbox-inspired layout + visual properties
//! - [`Tree`] — arena-based owning container with layout + paint + event traversal
//!
//! ## Layout model
//!
//! Flexbox-like single-axis layout (row/column).
//! Percentage sizing resolved against parent.  Absolute children skip the flow.
//! The solver runs top-down constraints → bottom-up sizing → top-down placement,
//! producing a `Rect` per node stored in the arena.
//!
//! ## Painting
//!
//! `Tree::paint` walks visible nodes in z-order, emitting [`Primitive`]s into
//! our [`RenderList`].  Clipping, scroll offsets, and corner radii are handled.
//!
//! ## Events
//!
//! `Tree::dispatch` does capture → target → bubble using our [`EventContext`]
//! and hit-tests against computed `Rect`s.

pub mod style;
pub mod tree;

pub use style::*;
pub use tree::*;
