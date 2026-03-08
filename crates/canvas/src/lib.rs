//! # any-compute-canvas
//!
//! GPU renderer, headless capture, and scenario replay for any-compute DOM.
//!
//! ## Modules
//!
//! - [`gpu`]      — WGPU renderer: instanced SDF rects + glyphon text, windowed or headless
//! - [`scenario`] — scriptable interaction replay + `StepResult` for visual testing
//! - [`theme`]    — Catppuccin Mocha palette constants (single source for all crates)

#[cfg(feature = "gpu")]
pub mod gpu;

pub mod scenario;
pub mod theme;

/// Catppuccin Mocha CSS variables — prepend before any app CSS for `var()` resolution.
pub const PALETTE_CSS: &str = include_str!("../fixtures/palette.css");

/// Default viewport for visual tools (visual-cmp, scenario, playground).
pub const DEFAULT_VIEWPORT: any_compute_core::layout::Size =
    any_compute_core::layout::Size::new(800.0, 600.0);

/// Re-export winit so downstream crates don't need a separate dep.
#[cfg(feature = "gpu")]
pub use winit;
