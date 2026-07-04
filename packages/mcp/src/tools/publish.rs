use std::path::PathBuf;

use serde_json::{json, Value};

use crate::error::{ErrorCode, StructuredError, ToolResult};
use crate::schema::{PublishInput, PublishOutput};

pub fn descriptor() -> Value {
    json!({
        "name": "gwenland_publish",
        "description": "Package a trained adapter for distribution with overwrite protection.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "adapter_path": { "type": "string" },
                "output_path": { "type": "string" },
                "format": { "type": "string", "enum": ["gguf", "safetensors"], "default": "gguf" },
                "metadata": {
                    "type": "object",
                    "properties": {
                        "name": { "type": "string" },
                        "version": { "type": "string" },
                        "description": { "type": "string" }
                    }
                },
                "force": { "type": "boolean", "default": false }
            },
            "required": ["adapter_path", "output_path"],
            "additionalProperties": false
        }
    })
}

pub fn run(arguments: Value) -> ToolResult<PublishOutput> {
    let input: PublishInput = super::parse_args(arguments)?;
    let adapter_path = PathBuf::from(&input.adapter_path);
    if !adapter_path.exists() {
        return Err(StructuredError::with_details(
            ErrorCode::InvalidInput,
            "adapter_path does not exist",
            json!({ "path": adapter_path.display().to_string() }),
        ));
    }

    let format = input.format.unwrap_or_else(|| "gguf".to_string());
    let format = format.to_ascii_lowercase();
    if !matches!(format.as_str(), "gguf" | "safetensors") {
        return Err(StructuredError::with_details(
            ErrorCode::InvalidInput,
            "format must be gguf or safetensors",
            json!({ "format": format }),
        ));
    }

    let output_path = PathBuf::from(&input.output_path);
    if output_path.exists() && !input.force.unwrap_or(false) {
        return Err(StructuredError::with_details(
            ErrorCode::PermissionDenied,
            "output_path already exists; pass force: true to overwrite it",
            json!({ "path": output_path.display().to_string() }),
        ));
    }

    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent).map_err(|error| {
            StructuredError::with_details(
                ErrorCode::PermissionDenied,
                "failed to create output directory",
                json!({ "path": parent.display().to_string(), "error": error.to_string() }),
            )
        })?;
    }

    std::fs::copy(&adapter_path, &output_path).map_err(|error| {
        StructuredError::with_details(
            ErrorCode::PublishFailed,
            "failed to package adapter",
            json!({
                "adapter_path": adapter_path.display().to_string(),
                "output_path": output_path.display().to_string(),
                "error": error.to_string(),
            }),
        )
    })?;

    if let Some(metadata) = input.metadata {
        let metadata_path = output_path.with_extension(format!(
            "{}.metadata.json",
            output_path
                .extension()
                .and_then(|extension| extension.to_str())
                .unwrap_or("adapter")
        ));
        let metadata_json = serde_json::to_string_pretty(&metadata).map_err(|error| {
            StructuredError::new(
                ErrorCode::PublishFailed,
                format!("failed to serialize metadata: {error}"),
            )
        })?;
        std::fs::write(&metadata_path, metadata_json).map_err(|error| {
            StructuredError::with_details(
                ErrorCode::PublishFailed,
                "failed to write metadata sidecar",
                json!({ "path": metadata_path.display().to_string(), "error": error.to_string() }),
            )
        })?;
    }

    let size_bytes = std::fs::metadata(&output_path)
        .map(|metadata| metadata.len())
        .unwrap_or(0);

    Ok(PublishOutput {
        output_path: output_path.display().to_string(),
        size_bytes,
        format,
    })
}
