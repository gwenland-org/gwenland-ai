use serde::de::DeserializeOwned;
use serde_json::{json, Value};

use crate::error::{ErrorCode, StructuredError, ToolResult};

pub mod benchmark;
pub mod infer;
pub mod load;
pub mod publish;
pub mod train;
pub mod train_status;

pub fn list_tools() -> Value {
    json!({
        "tools": [
            load::descriptor(),
            infer::descriptor(),
            train::descriptor(),
            train_status::descriptor(),
            benchmark::descriptor(),
            publish::descriptor(),
        ],
    })
}

pub fn call_tool(name: &str, arguments: Value) -> ToolResult<Value> {
    match name {
        "gwenland_load" => load::run(arguments).map(|output| crate::schema::tool_success(&output)),
        "gwenland_infer" => {
            infer::run(arguments).map(|output| crate::schema::tool_success(&output))
        }
        "gwenland_train" => {
            train::run(arguments).map(|output| crate::schema::tool_success(&output))
        }
        "gwenland_train_status" => {
            train_status::run(arguments).map(|output| crate::schema::tool_success(&output))
        }
        "gwenland_benchmark" => {
            benchmark::run(arguments).map(|output| crate::schema::tool_success(&output))
        }
        "gwenland_publish" => {
            publish::run(arguments).map(|output| crate::schema::tool_success(&output))
        }
        _ => Err(StructuredError::with_details(
            ErrorCode::InvalidInput,
            "unknown GwenLand MCP tool",
            json!({ "tool": name }),
        )),
    }
}

pub(crate) fn parse_args<T: DeserializeOwned>(arguments: Value) -> ToolResult<T> {
    serde_json::from_value(arguments).map_err(|error| {
        StructuredError::with_details(
            ErrorCode::InvalidInput,
            "invalid tool arguments",
            json!({ "error": error.to_string() }),
        )
    })
}
