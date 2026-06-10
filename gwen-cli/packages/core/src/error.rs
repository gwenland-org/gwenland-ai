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

    // ── GWEN-213: LoRA bridge error variants ──

    /// A LoRA tensor has the wrong shape.
    #[error("invalid LoRA shape: expected {expected:?}, got {actual:?}")]
    InvalidLoraShape { expected: Vec<usize>, actual: Vec<usize> },

    /// A lora_b tensor is present but no corresponding lora_a (or vice-versa).
    #[error("missing LoRA pair for layer index {layer_idx}")]
    MissingLoraPair { layer_idx: usize },

    /// The adapter delta shape does not match the base weight shape.
    #[error("shape mismatch: adapter {adapter:?} vs base {base:?}")]
    ShapeMismatch { adapter: Vec<usize>, base: Vec<usize> },

    /// The GGUF tensor uses a quantization format the merger does not support.
    #[error("unsupported quantization format: {format}")]
    UnsupportedQuantization { format: String },

    /// Peak memory during merge exceeded the configured budget.
    #[error("memory budget exceeded: required {required} bytes, available {available} bytes")]
    MemoryBudgetExceeded { required: usize, available: usize },

    /// Merged weights contain non-finite values (NaN or Inf).
    #[error("invalid merged weights in layer '{layer_name}' at index {index}")]
    InvalidMergedWeights { layer_name: String, index: usize },
}
