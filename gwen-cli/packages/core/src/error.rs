use thiserror::Error;

/// Top-level error type for gwenland-core.
#[derive(Debug, Error)]
pub enum GwenError {
    /// A generic inference backend error.
    #[error("inference backend error: {0}")]
    InferenceBackend(String),

    /// The requested backend was not compiled in.
    #[error("backend '{backend}' not available (compile with --features {backend}-backend)")]
    BackendNotAvailable { backend: String },

    /// The model architecture is not supported by the given backend.
    #[error("model architecture not supported by backend '{backend}': {arch}")]
    UnsupportedArchitecture { backend: String, arch: String },

    /// Model loading failed.
    #[error("model loading failed: {0}")]
    ModelLoad(String),

    // ── candle-backend variants (Requirements: 1.4, 2.5, 5.6, 12.4, 14.5) ──

    /// Dequantisation of a specific tensor failed.
    #[error("dequantization failed for tensor '{tensor_name}': {error}")]
    Dequantization { tensor_name: String, error: String },

    /// A forward-pass operation failed at a known layer and op.
    #[error("inference error at {layer}/{operation}: {error}")]
    InferenceError {
        layer: String,
        operation: String,
        error: String,
    },

    /// System RAM is insufficient to load the model.
    #[error("insufficient memory: need {required_gb:.2} GB, have {available_gb:.2} GB")]
    InsufficientMemory { required_gb: f32, available_gb: f32 },

    /// The model uses an architecture the candle backend does not support.
    #[error("unsupported model architecture: {0}")]
    ArchitectureNotSupported(String),

    /// A candle tensor operation returned an error.
    #[error("candle error: {0}")]
    CandleError(String),
}
