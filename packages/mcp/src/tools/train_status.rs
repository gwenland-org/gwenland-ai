use serde_json::{json, Value};

use crate::error::ToolResult;
use crate::runner;
use crate::schema::{TrainStatusInput, TrainStatusOutput};

pub fn descriptor() -> Value {
    json!({
        "name": "gwenland_train_status",
        "description": "Check the status of a detached GwenLand training job.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "job_id": { "type": "string" }
            },
            "required": ["job_id"],
            "additionalProperties": false
        }
    })
}

pub fn run(arguments: Value) -> ToolResult<TrainStatusOutput> {
    let input: TrainStatusInput = super::parse_args(arguments)?;
    let record = runner::read_job_record(&input.job_id)?;
    let stdout_tail = runner::read_tail(&record.stdout_log, 64 * 1024);
    let stderr_tail = runner::read_tail(&record.stderr_log, 64 * 1024);
    let logs = format!("{stdout_tail}\n{stderr_tail}");
    let running = runner::pid_running(record.pid);
    let lower_logs = logs.to_ascii_lowercase();
    let status = if running {
        "running"
    } else if lower_logs.contains("error") || lower_logs.contains("failed") {
        "failed"
    } else {
        "completed"
    };

    Ok(TrainStatusOutput {
        job_id: record.job_id,
        status: status.to_string(),
        step: parse_usize_after(&logs, "step").unwrap_or(0),
        max_steps: record.max_steps.unwrap_or(0),
        loss: parse_f64_after(&logs, "loss").unwrap_or(0.0),
        elapsed_ms: runner::now_ms().saturating_sub(record.started_ms),
        eta_ms: None,
    })
}

fn parse_usize_after(logs: &str, label: &str) -> Option<usize> {
    logs.lines()
        .rev()
        .find_map(|line| parse_number_text(line, label))
        .and_then(|value| value.parse::<usize>().ok())
}

fn parse_f64_after(logs: &str, label: &str) -> Option<f64> {
    logs.lines()
        .rev()
        .find_map(|line| parse_number_text(line, label))
        .and_then(|value| value.parse::<f64>().ok())
}

fn parse_number_text<'a>(line: &'a str, label: &str) -> Option<&'a str> {
    let lower = line.to_ascii_lowercase();
    let start = lower.find(label)? + label.len();
    let after = line[start..].trim_start_matches(|ch: char| {
        ch.is_whitespace() || matches!(ch, ':' | '=' | '/' | '[' | '(')
    });
    let end = after
        .find(|ch: char| !(ch.is_ascii_digit() || ch == '.'))
        .unwrap_or(after.len());
    if end == 0 {
        None
    } else {
        Some(&after[..end])
    }
}
