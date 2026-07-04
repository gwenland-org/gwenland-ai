use std::path::Path;

use serde_json::{json, Value};

use crate::error::{ErrorCode, StructuredError, ToolResult};
use crate::runner;
use crate::schema::{BenchmarkInput, BenchmarkOutput};

pub fn descriptor() -> Value {
    json!({
        "name": "gwenland_benchmark",
        "description": "Run GwenLand benchmark suites and return MCP-friendly aggregate metrics.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "model_path": { "type": "string" },
                "prompt": { "type": "string", "description": "Reserved for future model-specific benchmark prompts" },
                "runs": { "type": "number", "default": 3 }
            },
            "required": ["model_path"],
            "additionalProperties": false
        }
    })
}

pub fn run(arguments: Value) -> ToolResult<BenchmarkOutput> {
    let input: BenchmarkInput = super::parse_args(arguments)?;
    if !Path::new(&input.model_path).exists() {
        return Err(StructuredError::with_details(
            ErrorCode::ModelNotFound,
            "model file does not exist",
            json!({ "path": input.model_path }),
        ));
    }

    let _prompt = input
        .prompt
        .as_deref()
        .unwrap_or("Explain recursion in one paragraph.");
    let args = vec![
        "--json".to_string(),
        "--non-interactive".to_string(),
        "benchmark".to_string(),
        "--full".to_string(),
    ];
    let output = runner::run_gwenland(&args, ErrorCode::BenchmarkFailed)?;
    let data = runner::core_json_data(&output, ErrorCode::BenchmarkFailed)?;

    Ok(BenchmarkOutput {
        model: input.model_path,
        avg_tokens_per_second: data
            .pointer("/inference/tokens_per_sec")
            .and_then(Value::as_f64)
            .unwrap_or(0.0),
        cold_start_ms: data
            .pointer("/cold_start/median_ms")
            .and_then(Value::as_f64)
            .or_else(|| data.pointer("/cold_start/mean_ms").and_then(Value::as_f64))
            .unwrap_or(0.0),
        memory_mb: data
            .pointer("/memory/baseline_mb")
            .and_then(Value::as_f64)
            .unwrap_or(0.0),
        runs: input.runs.unwrap_or(3),
    })
}
