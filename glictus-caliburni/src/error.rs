use thiserror::Error;

/// Unified error type for the GLLM format crate.
///
/// glictus-caliburni is a standalone format library with zero workspace
/// deps by design (ARTX01 guardrail), so it carries its own error enum
/// instead of glcore's `GlError`. Runtimes embedding this crate are
/// expected to wrap `GllmError` into their own error type.
#[derive(Debug, Error)]
pub enum GllmError {
    #[error("Invalid magic bytes: expected GLLM (0x474C4C4D), got 0x{0:08X}")]
    InvalidMagic(u32),

    #[error("Unsupported format version: {major}.{minor}")]
    UnsupportedVersion { major: u8, minor: u8 },

    #[error("Manifest parse error: {0}")]
    ManifestParse(#[from] serde_json::Error),

    #[error("Checksum mismatch for {file}: expected {expected}, got {actual}")]
    ChecksumMismatch {
        file: String,
        expected: String,
        actual: String,
    },

    #[error("Missing execution unit: {0}")]
    MissingExecutionUnit(String),

    #[error("Package is missing its manifest (gllm.json)")]
    MissingManifest,

    #[error("Package is missing its shared component (shared.gllm)")]
    MissingSharedComponent,

    #[error("Invalid package format: {0}")]
    InvalidPackageFormat(String),

    #[error("Integrity error: {0}")]
    IntegrityError(String),

    #[error("Missing required metadata field: {0}")]
    MissingMetadata(String),

    #[error("Unsupported dtype code: 0x{0:04X}")]
    UnsupportedDtype(u16),

    #[error("Layer index out of bounds: {index} (max: {max})")]
    LayerOutOfBounds { index: usize, max: usize },

    #[error("Unknown extension URI: {0}")]
    UnknownExtension(String),

    #[error("Package validation failed: {0}")]
    ValidationError(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("No valid execution plan found (Pvalid = ∅)")]
    NoValidPlan,
}

pub type GllmResult<T> = Result<T, GllmError>;
