use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

use serde_json::{json, Value};

use crate::error::{ErrorCode, StructuredError, ToolResult};
use crate::runner;
use crate::schema::{LoadInput, LoadOutput};

pub fn descriptor() -> Value {
    json!({
        "name": "gwenland_load",
        "description": "Load a local GGUF/GGQR model through GwenLand and verify it is ready for inference.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "model_path": { "type": "string", "description": "Path to a .gguf or .ggqr model file" },
                "quantization": { "type": "string", "description": "Optional quantization hint: q4_0, q8_0, f16" },
                "context_size": { "type": "number", "description": "Optional max context length; reserved for future core support" }
            },
            "required": ["model_path"],
            "additionalProperties": false
        }
    })
}

pub fn run(arguments: Value) -> ToolResult<LoadOutput> {
    let input: LoadInput = super::parse_args(arguments)?;
    if matches!(input.context_size, Some(0)) {
        return Err(StructuredError::new(
            ErrorCode::InvalidInput,
            "context_size must be greater than zero",
        ));
    }

    let path = validate_model_path(&input.model_path)?;
    let args = vec![
        "--json".to_string(),
        "--non-interactive".to_string(),
        "run".to_string(),
        path.display().to_string(),
        "--prompt".to_string(),
        "hello".to_string(),
        "--max-tokens".to_string(),
        "1".to_string(),
        "--temperature".to_string(),
        "0".to_string(),
    ];
    let output = runner::run_gwenland(&args, ErrorCode::ModelNotFound)?;
    let _ = runner::core_json_data(&output, ErrorCode::ModelNotFound)?;

    let metadata = std::fs::metadata(&path).map_err(|error| {
        StructuredError::with_details(
            ErrorCode::ModelNotFound,
            "failed to read model metadata",
            json!({ "path": path.display().to_string(), "error": error.to_string() }),
        )
    })?;

    Ok(LoadOutput {
        model_id: model_id(&path, metadata.len()),
        model_name: path
            .file_stem()
            .and_then(|name| name.to_str())
            .unwrap_or("model")
            .to_string(),
        parameters: infer_parameters(&path),
        quantization: input
            .quantization
            .unwrap_or_else(|| infer_quantization(&path)),
        memory_mb: metadata.len() as f64 / (1024.0 * 1024.0),
        load_time_ms: output.duration_ms,
    })
}

fn validate_model_path(path: &str) -> ToolResult<PathBuf> {
    let path = PathBuf::from(path);
    if !path.exists() {
        return Err(StructuredError::with_details(
            ErrorCode::ModelNotFound,
            "model file does not exist",
            json!({ "path": path.display().to_string() }),
        ));
    }

    let extension = path
        .extension()
        .and_then(|extension| extension.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    if !matches!(extension.as_str(), "gguf" | "ggqr") {
        return Err(StructuredError::with_details(
            ErrorCode::InvalidInput,
            "model_path must point to a .gguf or .ggqr file",
            json!({ "path": path.display().to_string() }),
        ));
    }

    Ok(path)
}

fn model_id(path: &Path, bytes: u64) -> String {
    let mut hasher = DefaultHasher::new();
    path.display().to_string().hash(&mut hasher);
    bytes.hash(&mut hasher);
    format!("model-{:016x}", hasher.finish())
}

fn infer_quantization(path: &Path) -> String {
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();

    for quant in ["q4_0", "q4_k", "q8_0", "q6_k", "q5_k", "f16", "fp16"] {
        if name.contains(quant) {
            return quant.to_string();
        }
    }
    "auto".to_string()
}

fn infer_parameters(path: &Path) -> String {
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default()
        .replace(['_', '-'], " ");

    for part in name.split_whitespace() {
        let lower = part.to_ascii_lowercase();
        if lower.ends_with('b')
            && lower[..lower.len().saturating_sub(1)]
                .parse::<f64>()
                .is_ok()
        {
            return part.to_string();
        }
    }
    "unknown".to_string()
}
