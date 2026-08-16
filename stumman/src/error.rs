//! Stummañ error types.
//!
//! GlTrainError wraps glcore::GlError for training-specific context.
//! All fallible paths return Result<T, GlTrainError>.

use thiserror::Error;

/// Every way a Stummañ operation can fail.
#[derive(Debug, Error)]
pub enum GlTrainError {
    /// Two operands disagreed on shape.
    #[error("shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        /// The shape the operation required.
        expected: Vec<usize>,
        /// The shape it actually received.
        got: Vec<usize>,
    },

    /// A backend rejected the request (bad storage length, unsupported device, ...).
    #[error("backend error: {0}")]
    Backend(String),

    /// The operation itself is not valid for this input (wrong rank, empty shape, ...).
    #[error("invalid operation: {0}")]
    InvalidOp(String),

    /// A failure surfaced from the shared glcore layer.
    #[error(transparent)]
    Gl(#[from] glcore::error::GlError),

    /// Anything else, carried verbatim.
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

/// Shorthand for a Stummañ fallible result.
pub type Result<T> = std::result::Result<T, GlTrainError>;
