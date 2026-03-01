use thiserror::Error as ThisError;

/// Unified error type for the core crate.
#[derive(Debug, ThisError)]
pub enum Error {
    #[error("layout constraint unsatisfiable: {0}")]
    Layout(String),

    #[error("data access out of range: index {index}, len {len}")]
    OutOfRange { index: usize, len: usize },

    #[error("compute backend error: {0}")]
    Compute(String),

    #[error("animation error: {0}")]
    Animation(String),

    #[error("{0}")]
    Other(String),
}
