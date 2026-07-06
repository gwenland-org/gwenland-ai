//! Unified error type for all GL engine crates.

/// The error type shared by every GwenLand AI crate.
#[derive(thiserror::Error, Debug)]
pub enum GlError {
    /// Underlying filesystem / IO failure.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// A model file (GGUF, safetensors, tokenizer.json, ...) failed to parse.
    #[error("Parse error: {0}")]
    Parse(String),

    /// A tensor operation received incompatible shapes.
    #[error("Shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        /// The shape the operation required.
        expected: Vec<usize>,
        /// The shape it actually received.
        got: Vec<usize>,
    },

    /// A tensor uses a dtype the current code path cannot handle.
    #[error("Unsupported dtype: {0:?}")]
    UnsupportedDtype(String),

    /// Engine-level failure (init, load, inference, missing hardware, ...).
    #[error("Engine error: {0}")]
    Engine(String),
}
