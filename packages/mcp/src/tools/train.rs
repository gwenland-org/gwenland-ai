use std::path::PathBuf;

use serde_json::{json, Value};

use crate::error::{ErrorCode, StructuredError, ToolResult};
use crate::runner::{self, JobRecord};
use crate::schema::{TrainInput, TrainOutput};

pub fn descriptor() -> Value {
    json!({
        "name": "gwenland_train",
        "description": "Start a detached GwenLand LoRA training job and persist its PID for later polling.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "base_model": { "type": "string", "description": "Path or model id for the base model" },
                "dataset": { "type": "string", "description": "Path to JSONL training dataset" },
                "output_dir": { "type": "string", "description": "Directory for checkpoints and adapter output" },
                "lora_rank": { "type": "number", "default": 8 },
                "learning_rate": { "type": "number", "default": 0.0002 },
                "max_steps": { "type": "number", "default": 1000 },
                "batch_size": { "type": "number", "default": 1 },
                "job_id": { "type": "string" },
                "force": { "type": "boolean", "default": false, "description": "Allow writing into an existing output_dir" }
            },
            "required": ["base_model", "dataset", "output_dir"],
            "additionalProperties": false
        }
    })
}

pub fn run(arguments: Value) -> ToolResult<TrainOutput> {
    let input: TrainInput = super::parse_args(arguments)?;
    let job_id = input
        .job_id
        .unwrap_or_else(|| format!("gwen-train-{}", runner::now_ms()));
    validate_job_id(&job_id)?;

    let dataset = PathBuf::from(&input.dataset);
    if !dataset.exists() {
        return Err(StructuredError::with_details(
            ErrorCode::InvalidInput,
            "dataset file does not exist",
            json!({ "path": dataset.display().to_string() }),
        ));
    }

    let output_dir = PathBuf::from(&input.output_dir);
    if output_dir.exists() && !input.force.unwrap_or(false) {
        return Err(StructuredError::with_details(
            ErrorCode::PermissionDenied,
            "output_dir already exists; pass force: true to allow writing into it",
            json!({ "path": output_dir.display().to_string() }),
        ));
    }

    let jobs_dir = runner::jobs_dir()?;
    let stdout_log = jobs_dir.join(format!("{job_id}.stdout.log"));
    let stderr_log = jobs_dir.join(format!("{job_id}.stderr.log"));
    let max_steps = input.max_steps.unwrap_or(1000);

    let args = vec![
        "--json".to_string(),
        "--non-interactive".to_string(),
        "train".to_string(),
        "--model".to_string(),
        input.base_model,
        "--dataset".to_string(),
        input.dataset,
        "--output".to_string(),
        input.output_dir.clone(),
        "--verbose".to_string(),
        "--lora-rank".to_string(),
        input.lora_rank.unwrap_or(8).to_string(),
        "--batch-size".to_string(),
        input.batch_size.unwrap_or(1).to_string(),
        "--max-steps".to_string(),
        max_steps.to_string(),
        "--lr".to_string(),
        input.learning_rate.unwrap_or(2e-4).to_string(),
    ];

    let pid = runner::spawn_gwenland_job(&args, &stdout_log, &stderr_log)?;
    let record = JobRecord {
        job_id: job_id.clone(),
        pid,
        command: args,
        output_dir: input.output_dir,
        stdout_log,
        stderr_log,
        started_ms: runner::now_ms(),
        max_steps: Some(max_steps),
    };
    runner::write_job_record(&record)?;

    Ok(TrainOutput {
        job_id,
        status: "started".to_string(),
        pid,
    })
}

fn validate_job_id(job_id: &str) -> ToolResult<()> {
    if job_id.is_empty()
        || !job_id
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.'))
    {
        return Err(StructuredError::with_details(
            ErrorCode::InvalidInput,
            "job_id may only contain ASCII letters, numbers, '.', '-', and '_'",
            json!({ "job_id": job_id }),
        ));
    }
    Ok(())
}
