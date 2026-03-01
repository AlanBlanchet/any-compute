//! # any-compute-rsx
//!
//! Declarative / React-like UI layer for any-compute, built on [dioxus](https://dioxuslabs.com).
//!
//! This crate **lives in its own directory** and is the *only* place RSX markup exists.
//! Core logic is imported from `any-compute-core`; this crate merely provides
//! component wrappers that translate core primitives into RSX elements.
//!
//! ## Design rules
//! - **No RSX code escapes this crate.** Other crates depend on `any-compute-core`, not this.
//! - Components here are thin adapters: data comes from core, rendering from dioxus.
//! - Hooks wrap core systems (animation, compute) into reactive dioxus primitives.

pub mod components;
pub mod hooks;

pub use dioxus;
