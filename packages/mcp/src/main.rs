mod error;
mod runner;
mod schema;
mod tools;

use std::io::{self, BufRead, Write};

use serde_json::{json, Value};

use crate::schema::{JsonRpcRequest, ToolCallParams};

fn main() {
    let stdin = io::stdin();
    let mut stdout = io::stdout();

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(line) => line,
            Err(error) => {
                eprintln!("gwenland-mcp: stdin read failed: {error}");
                break;
            }
        };

        if line.trim().is_empty() {
            continue;
        }

        if let Some(response) = handle_line(&line) {
            if writeln!(stdout, "{response}").is_err() {
                break;
            }
            let _ = stdout.flush();
        }
    }
}

fn handle_line(line: &str) -> Option<Value> {
    match serde_json::from_str::<JsonRpcRequest>(line) {
        Ok(request) => handle_request(request),
        Err(error) => Some(schema::jsonrpc_error(
            Value::Null,
            -32700,
            "Parse error",
            Some(json!({ "message": error.to_string() })),
        )),
    }
}

fn handle_request(request: JsonRpcRequest) -> Option<Value> {
    let id = request.id?;

    match request.method.as_str() {
        "initialize" => Some(schema::jsonrpc_result(id, initialize_result())),
        "ping" => Some(schema::jsonrpc_result(id, json!({}))),
        "tools/list" => Some(schema::jsonrpc_result(id, tools::list_tools())),
        "tools/call" => Some(handle_tool_call(id, request.params)),
        _ => Some(schema::jsonrpc_error(
            id,
            -32601,
            "Method not found",
            Some(json!({ "method": request.method })),
        )),
    }
}

fn initialize_result() -> Value {
    json!({
        "protocolVersion": "2024-11-05",
        "capabilities": {
            "tools": {
                "listChanged": false,
            },
        },
        "serverInfo": {
            "name": "gwenland-mcp",
            "version": env!("CARGO_PKG_VERSION"),
        },
    })
}

fn handle_tool_call(id: Value, params: Value) -> Value {
    let params = match serde_json::from_value::<ToolCallParams>(params) {
        Ok(params) => params,
        Err(error) => {
            return schema::jsonrpc_error(
                id,
                -32602,
                "Invalid params",
                Some(json!({ "message": error.to_string() })),
            );
        }
    };

    match tools::call_tool(&params.name, params.arguments) {
        Ok(result) => schema::jsonrpc_result(id, result),
        Err(error) => schema::jsonrpc_result(id, schema::tool_error(&error)),
    }
}
