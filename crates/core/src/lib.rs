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

// ── Op-registry callback macros ───────────────────────────────────────────
//
// Single source of truth for each op family.  Consumer modules call
// `for_each_X!(their_macro)` to generate all the boilerplate methods.
// Adding a new parameterless unary op = one line here — then Buffer, Graph,
// and LazyMut all pick it up automatically.

/// All parameterless [`UnaryOp`] variants.
macro_rules! for_each_unary {
    ($mac:ident) => {
        $mac!(
            neg     => UnaryOp::Neg,
            abs     => UnaryOp::Abs,
            sqrt    => UnaryOp::Sqrt,
            rsqrt   => UnaryOp::Rsqrt,
            exp     => UnaryOp::Exp,
            log     => UnaryOp::Log,
            sin     => UnaryOp::Sin,
            cos     => UnaryOp::Cos,
            tanh    => UnaryOp::Tanh,
            relu    => UnaryOp::Relu,
            sigmoid => UnaryOp::Sigmoid,
            floor   => UnaryOp::Floor,
            ceil    => UnaryOp::Ceil,
        );
    };
}

/// All [`BinaryOp`] variants.
macro_rules! for_each_binary {
    ($mac:ident) => {
        $mac!(
            add     => BinaryOp::Add,
            sub     => BinaryOp::Sub,
            mul     => BinaryOp::Mul,
            div     => BinaryOp::Div,
            minimum => BinaryOp::Min,
            maximum => BinaryOp::Max,
            power   => BinaryOp::Pow,
        );
    };
}

/// All [`ReduceOp`] variants.
macro_rules! for_each_reduce {
    ($mac:ident) => {
        $mac!(
            sum     => ReduceOp::Sum,
            min     => ReduceOp::Min,
            max     => ReduceOp::Max,
            product => ReduceOp::Product,
            mean    => ReduceOp::Mean,
        );
    };
}

pub mod buffer;
pub mod compute;
pub mod data;
pub mod flex;
pub mod hints;
pub mod interaction;
pub mod propagation;
pub mod tree;

pub use compute::Device;

/// NaN-safe ascending comparison for `f64` values.
#[inline]
pub fn f64_cmp(a: &f64, b: &f64) -> std::cmp::Ordering {
    a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
}

// ── display_enum! ─────────────────────────────────────────────────────────
//
// Generates a `Display` impl for enums that map variants to string literals.
// Avoids hand-writing repetitive `match self { Variant => write!(f, "...") }`.
//
// Usage:
//   display_enum!(MyEnum { Foo => "foo label", Bar => "bar label" });

macro_rules! display_enum {
    ($name:ty { $( $variant:ident => $lit:literal ),* $(,)? }) => {
        impl ::std::fmt::Display for $name {
            fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
                f.write_str(match self { $( Self::$variant => $lit, )* })
            }
        }
    }
}

pub mod graph;
pub mod kernel;
pub mod layout;
pub mod ops;
pub mod render;
pub mod scene;
pub mod shader;
pub mod visual;

mod error;
pub use error::Error;

/// Result alias used throughout the crate.
pub type Result<T> = std::result::Result<T, Error>;

/// Compile-time feature flag report — lets downstream crates query which
/// core features were enabled without duplicating `cfg!` checks.
pub const FEATURES: CoreFeatures = CoreFeatures {
    cuda: cfg!(feature = "cuda"),
    rocm: cfg!(feature = "rocm"),
    mkl: cfg!(feature = "mkl"),
    metal: cfg!(feature = "metal"),
    wgpu: cfg!(feature = "wgpu-backend"),
    shader: cfg!(feature = "shader"),
};

/// Which optional core features are compiled in.
pub struct CoreFeatures {
    pub cuda: bool,
    pub rocm: bool,
    pub mkl: bool,
    pub metal: bool,
    pub wgpu: bool,
    pub shader: bool,
}

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
