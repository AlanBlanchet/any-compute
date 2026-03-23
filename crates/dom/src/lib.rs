//! # any-compute-dom
//!
//! Arena-based scene graph with layout, paint, GPU rendering, and event dispatch.
//!
//! ## Modules
//!
//! - [`style`] — flexbox-inspired layout + visual properties
//! - [`tree`]  — arena-based owning container with layout + paint + event traversal
//! - [`parse`] — convert HTML-like markup into our [`Tree`]
//! - [`css`]   — CSS parser with transitions, @keyframes, variables
//! - [`gpu`]   — wgpu renderer (behind `gpu` feature)
//! - [`theme`] — Catppuccin Mocha palette constants (behind `gpu` feature)
//! - [`harness`] — headless test driver (behind `gpu` feature)
//! - [`scenario`] — action replay + assertions (behind `gpu` feature)

#[macro_use]
pub mod style;
pub mod css;
pub mod parse;
pub mod tree;

#[cfg(feature = "gpu")]
pub mod gpu;
#[cfg(feature = "gpu")]
pub mod harness;
#[cfg(feature = "gpu")]
pub mod scenario;
#[cfg(feature = "gpu")]
pub mod theme;

pub use style::*;
pub use tree::*;

/// Compiled Tailwind v3 CSS subset — every utility class that maps to our Style system.
/// Parsed through `StyleSheet::parse()` at runtime; no special Tailwind runtime.
pub const TAILWIND_CSS: &str = include_str!("tailwind.css");

/// Catppuccin Mocha CSS variables — prepend before any app CSS for `var()` resolution.
pub const PALETTE_CSS: &str = include_str!("../fixtures/palette.css");

/// Default viewport for visual tools (playground, dashboard).
pub const DEFAULT_VIEWPORT: any_compute_core::layout::Size =
    any_compute_core::layout::Size::new(800.0, 600.0);

/// Re-export winit so downstream crates don't need a separate dep.
#[cfg(feature = "gpu")]
pub use winit;
