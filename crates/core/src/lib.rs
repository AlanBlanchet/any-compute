//! # any-compute-core
//!
//! Framework-agnostic core for high-performance computation and data visualization.
//!
//! This crate contains:
//! - **Compute** — hardware-abstracted parallel work dispatch (CPU, GPU, WASM)
//! - **Kernel** — low-level compute kernels (SIMD, CUDA, ROCm, MKL, Metal)
//! - **Shader** — cross-platform shader compilation (WGSL, GLSL, SPIR-V)
//! - **Data layer** — virtualized access to massive datasets
//! - **Layout** — positioning, sizing, and constraint solving (`Point`, `Rect`, `Size`)
//! - **Interaction** — input events, gestures, hit-testing (web-like propagation)
//! - **Render primitives** — shapes, colors, text descriptors (no actual rendering)
//! - **Animation** — timing engine, easing, transitions
//! - **Hints** — runtime optimization hints (animated, static, streaming, etc.)
//!
//! **Zero UI-framework dependencies.** RSX, DOM, GPU backends live in sibling crates.
//!
//! ## Design: automatic optimization
//!
//! Elements carry [`hints::Hints`] that describe their runtime behavior.
//! The engine reads these to select the best code path automatically:
//! - Static content → skip diff, cache aggressively
//! - Animated content → pre-allocate interpolation buffers, batch GPU uploads
//! - Streaming data → double-buffer, prefetch ahead of viewport
//!
//! Users never *need* to set hints — defaults are sensible — but they *can*
//! override them for fine-grained control.

pub mod animation;
pub mod bench;
pub mod compute;
pub mod data;
pub mod hints;
pub mod interaction;
pub mod kernel;
pub mod layout;
pub mod render;
pub mod shader;

mod error;
pub use error::Error;

/// Result alias used throughout the crate.
pub type Result<T> = std::result::Result<T, Error>;

/// Trait for types that can be linearly interpolated.
///
/// This is the **single source of truth** for interpolation.
/// `Point`, `Size`, `Rect`, `Color`, and any user type implement this.
/// The animation system only calls `Lerp::lerp` — never reimplements blending.
pub trait Lerp: Sized {
    fn lerp(self, other: Self, t: f64) -> Self;
}

impl Lerp for f64 {
    fn lerp(self, other: Self, t: f64) -> Self {
        self + (other - self) * t
    }
}

impl Lerp for f32 {
    fn lerp(self, other: Self, t: f64) -> Self {
        let t = t as f32;
        self + (other - self) * t
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f64_lerp_endpoints() {
        assert_eq!(0.0f64.lerp(100.0, 0.0), 0.0);
        assert_eq!(0.0f64.lerp(100.0, 1.0), 100.0);
    }

    #[test]
    fn f64_lerp_midpoint() {
        assert!((0.0f64.lerp(100.0, 0.5) - 50.0).abs() < 1e-10);
    }

    #[test]
    fn f32_lerp_endpoints() {
        assert_eq!(0.0f32.lerp(100.0, 0.0), 0.0);
        assert_eq!(0.0f32.lerp(100.0, 1.0), 100.0);
    }

    #[test]
    fn f64_lerp_negative() {
        assert!(((-50.0f64).lerp(50.0, 0.5)).abs() < 1e-10);
    }
}
