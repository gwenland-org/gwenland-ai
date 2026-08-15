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

#[cfg(test)]
mod tests {
    use super::*;

    /// The registry initializes and advertises every tool `call_tool` can
    /// dispatch — the two lists drifting apart is the failure this catches.
    #[test]
    fn registry_lists_every_dispatchable_tool() {
        let listed = list_tools();
        let tools = listed["tools"].as_array().expect("tools is an array");
        assert_eq!(tools.len(), 6, "expected six registered tools");

        let names: Vec<&str> = tools.iter().map(|t| t["name"].as_str().expect("tool has a name")).collect();
        for expected in [
            "gwenland_load",
            "gwenland_infer",
            "gwenland_train",
            "gwenland_train_status",
            "gwenland_benchmark",
            "gwenland_publish",
        ] {
            assert!(names.contains(&expected), "{expected} missing from list_tools(); have {names:?}");
        }
    }

    /// Every advertised tool carries the fields an MCP client needs to call it.
    #[test]
    fn every_descriptor_has_a_name_and_input_schema() {
        let listed = list_tools();
        for tool in listed["tools"].as_array().expect("tools is an array") {
            let name = tool["name"].as_str().expect("descriptor has a name");
            assert!(!name.is_empty());
            assert!(tool["inputSchema"].is_object(), "{name} has no inputSchema object");
        }
    }

    /// A known tool resolves by name: dispatching it with arguments that fail
    /// *schema* validation must produce an argument error, not "unknown tool".
    /// Deliberately routed through bad arguments so the tool's real work (model
    /// loading, training) never runs in a unit test.
    #[test]
    fn known_tool_resolves_by_name_and_reports_bad_arguments() {
        let err = call_tool("gwenland_load", json!({ "unexpected": true }))
            .expect_err("wrong arguments must be rejected");
        assert_eq!(err.code, ErrorCode::InvalidInput.as_str());
        assert_ne!(
            err.message, "unknown GwenLand MCP tool",
            "the name resolved to the unknown-tool arm instead of the load tool"
        );
    }

    /// Error case: an unregistered name is refused, and the error echoes back
    /// which name was tried.
    #[test]
    fn unknown_tool_name_is_refused() {
        let err = call_tool("gwenland_not_a_tool", json!({})).expect_err("unknown tool must fail");
        assert_eq!(err.code, ErrorCode::InvalidInput.as_str());
        assert_eq!(err.message, "unknown GwenLand MCP tool");
    }
}
