use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::error::StructuredError;

#[derive(Debug, Deserialize)]
pub struct JsonRpcRequest {
    #[allow(dead_code)]
    pub jsonrpc: Option<String>,
    pub id: Option<Value>,
    pub method: String,
    #[serde(default = "default_params")]
    pub params: Value,
}

fn default_params() -> Value {
    Value::Null
}

#[derive(Debug, Deserialize)]
pub struct ToolCallParams {
    pub name: String,
    #[serde(default = "default_params")]
    pub arguments: Value,
}

#[derive(Debug, Deserialize)]
pub struct LoadInput {
    pub model_path: String,
    pub quantization: Option<String>,
    pub context_size: Option<u32>,
}

#[derive(Debug, Serialize)]
pub struct LoadOutput {
    pub model_id: String,
    pub model_name: String,
    pub parameters: String,
    pub quantization: String,
    pub memory_mb: f64,
    pub load_time_ms: u64,
}

#[derive(Debug, Deserialize)]
pub struct InferInput {
    pub model_path: String,
    pub prompt: String,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f64>,
    pub system_prompt: Option<String>,
    pub stream: Option<bool>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct InferOutput {
    pub text: String,
    pub tokens_generated: usize,
    pub tokens_per_second: f64,
    pub time_ms: u64,
}

#[derive(Debug, Deserialize)]
pub struct TrainInput {
    pub base_model: String,
    pub dataset: String,
    pub output_dir: String,
    pub lora_rank: Option<usize>,
    pub learning_rate: Option<f64>,
    pub max_steps: Option<usize>,
    pub batch_size: Option<usize>,
    pub job_id: Option<String>,
    pub force: Option<bool>,
}

#[derive(Debug, Serialize)]
pub struct TrainOutput {
    pub job_id: String,
    pub status: String,
    pub pid: u32,
}

#[derive(Debug, Deserialize)]
pub struct TrainStatusInput {
    pub job_id: String,
}

#[derive(Debug, Serialize)]
pub struct TrainStatusOutput {
    pub job_id: String,
    pub status: String,
    pub step: usize,
    pub max_steps: usize,
    pub loss: f64,
    pub elapsed_ms: u64,
    pub eta_ms: Option<u64>,
}

#[derive(Debug, Deserialize)]
pub struct BenchmarkInput {
    pub model_path: String,
    pub prompt: Option<String>,
    pub runs: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct BenchmarkOutput {
    pub model: String,
    pub avg_tokens_per_second: f64,
    pub cold_start_ms: f64,
    pub memory_mb: f64,
    pub runs: usize,
}

#[derive(Debug, Deserialize)]
pub struct PublishInput {
    pub adapter_path: String,
    pub output_path: String,
    pub format: Option<String>,
    pub metadata: Option<PublishMetadata>,
    pub force: Option<bool>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct PublishMetadata {
    pub name: Option<String>,
    pub version: Option<String>,
    pub description: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct PublishOutput {
    pub output_path: String,
    pub size_bytes: u64,
    pub format: String,
}

pub fn jsonrpc_result(id: Value, result: Value) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": result,
    })
}

pub fn jsonrpc_error(
    id: Value,
    code: i64,
    message: impl Into<String>,
    data: Option<Value>,
) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": {
            "code": code,
            "message": message.into(),
            "data": data,
        },
    })
}

pub fn tool_success<T: Serialize>(payload: &T) -> Value {
    let text = serde_json::to_string_pretty(payload).unwrap_or_else(|_| "{}".to_string());
    json!({
        "content": [
            {
                "type": "text",
                "text": text,
            }
        ],
    })
}

pub fn tool_error(error: &StructuredError) -> Value {
    let text = serde_json::to_string_pretty(error).unwrap_or_else(|_| {
        "{\"code\":\"INVALID_INPUT\",\"message\":\"failed to serialize error\"}".to_string()
    });
    json!({
        "isError": true,
        "content": [
            {
                "type": "text",
                "text": text,
            }
        ],
    })
}
