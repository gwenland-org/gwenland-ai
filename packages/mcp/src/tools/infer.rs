use serde_json::{json, Value};

use crate::error::{ErrorCode, StructuredError, ToolResult};
use crate::runner;
use crate::schema::{InferInput, InferOutput};

pub fn descriptor() -> Value {
    json!({
        "name": "gwenland_infer",
        "description": "Run single-shot inference on a local GwenLand model.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "model_path": { "type": "string" },
                "prompt": { "type": "string" },
                "max_tokens": { "type": "number", "default": 512 },
                "temperature": { "type": "number", "default": 0.7 },
                "system_prompt": { "type": "string" },
                "stream": { "type": "boolean", "default": false }
            },
            "required": ["model_path", "prompt"],
            "additionalProperties": false
        }
    })
}

pub fn run(arguments: Value) -> ToolResult<InferOutput> {
    let input: InferInput = super::parse_args(arguments)?;
    if input.stream.unwrap_or(false) {
        return Err(StructuredError::new(
            ErrorCode::InvalidInput,
            "streaming inference is not supported for MCP tool results; set stream to false",
        ));
    }

    let prompt = match input.system_prompt {
        Some(system_prompt) => format!(
            "<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            input.prompt
        ),
        None => input.prompt,
    };

    let args = vec![
        "--json".to_string(),
        "--non-interactive".to_string(),
        "run".to_string(),
        input.model_path,
        "--prompt".to_string(),
        prompt,
        "--max-tokens".to_string(),
        input.max_tokens.unwrap_or(512).to_string(),
        "--temperature".to_string(),
        input.temperature.unwrap_or(0.7).to_string(),
    ];

    let output = runner::run_gwenland(&args, ErrorCode::InferenceFailed)?;
    let data = runner::core_json_data(&output, ErrorCode::InferenceFailed)?;
    serde_json::from_value::<InferOutput>(data).map_err(|error| {
        StructuredError::with_details(
            ErrorCode::InferenceFailed,
            "failed to parse gwenland inference output",
            json!({ "error": error.to_string(), "stdout": output.stdout, "stderr": output.stderr }),
        )
    })
}
