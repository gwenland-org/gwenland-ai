use serde::Serialize;
use serde_json::Value;

pub type ToolResult<T> = Result<T, StructuredError>;

#[derive(Debug, Clone, Copy)]
pub enum ErrorCode {
    ModelNotFound,
    InferenceFailed,
    TrainingFailed,
    BenchmarkFailed,
    PublishFailed,
    InvalidInput,
    PermissionDenied,
    SubprocessFailed,
    JobNotFound,
}

impl ErrorCode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ModelNotFound => "MODEL_NOT_FOUND",
            Self::InferenceFailed => "INFERENCE_FAILED",
            Self::TrainingFailed => "TRAINING_FAILED",
            Self::BenchmarkFailed => "BENCHMARK_FAILED",
            Self::PublishFailed => "PUBLISH_FAILED",
            Self::InvalidInput => "INVALID_INPUT",
            Self::PermissionDenied => "PERMISSION_DENIED",
            Self::SubprocessFailed => "SUBPROCESS_FAILED",
            Self::JobNotFound => "JOB_NOT_FOUND",
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct StructuredError {
    pub code: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<Value>,
}

impl StructuredError {
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            code: code.as_str().to_string(),
            message: message.into(),
            details: None,
        }
    }

    pub fn with_details(code: ErrorCode, message: impl Into<String>, details: Value) -> Self {
        Self {
            code: code.as_str().to_string(),
            message: message.into(),
            details: Some(details),
        }
    }
}
