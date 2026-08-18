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

    /// A registered skill exists but is not implemented on this milestone.
    ///
    /// This is what an M2 stub returns. It is deliberately distinct from
    /// [`GlTrainError::InvalidOp`]: the caller asked for something real and
    /// spelled it correctly, and the answer is "not yet, here is why, here is
    /// which milestone owns it". A stub must never quietly do something else
    /// instead, so this error is the only thing a stub's compute path can
    /// produce.
    #[error("{skill} is not implemented yet ({milestone}): {reason}")]
    Unsupported {
        /// Registry id of the thing that was asked for, e.g. `"dora"`.
        skill: &'static str,
        /// Why it is not implemented, in one line. Names the blocking work.
        reason: &'static str,
        /// The milestone that owns the implementation, e.g. `"M3"`.
        milestone: &'static str,
    },

    /// A checkpoint failed validation, or its bytes could not be parsed.
    #[error("checkpoint error: {0}")]
    Checkpoint(String),

    /// An I/O failure while reading or writing a checkpoint.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// A failure surfaced from the shared glcore layer.
    #[error(transparent)]
    Gl(#[from] glcore::error::GlError),

    /// Anything else, carried verbatim.
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

/// Shorthand for a Stummañ fallible result.
pub type Result<T> = std::result::Result<T, GlTrainError>;
