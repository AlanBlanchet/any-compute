//! # any-compute-dom
//!
//! Arena-based scene graph with layout, paint, and event dispatch —
//! separated from core so the compute library stays framework-agnostic.
//!
//! ## Modules
//!
//! - [`style`] — flexbox-inspired layout + visual properties
//! - [`tree`]  — arena-based owning container with layout + paint + event traversal
//! - [`parse`] — convert HTML-like markup into our [`Tree`]
//!
//! ## Design
//!
//! All spatial/render types come from `any-compute-core` (layout, render, hints,
//! interaction).  This crate only owns the *scene graph structure* and the
//! *parser* that bridges external DOM representations into our optimized arena.

pub mod css;
pub mod parse;
pub mod style;
pub mod tree;

pub use style::*;
pub use tree::*;
