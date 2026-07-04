use std::ffi::OsString;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::error::{ErrorCode, StructuredError, ToolResult};

#[derive(Debug)]
pub struct CommandOutput {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: Option<i32>,
    pub duration_ms: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct JobRecord {
    pub job_id: String,
    pub pid: u32,
    pub command: Vec<String>,
    pub output_dir: String,
    pub stdout_log: PathBuf,
    pub stderr_log: PathBuf,
    pub started_ms: u64,
    pub max_steps: Option<usize>,
}

pub fn run_gwenland(args: &[String], failure_code: ErrorCode) -> ToolResult<CommandOutput> {
    let started = Instant::now();
    let output = Command::new(gwenland_bin())
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|error| {
            StructuredError::with_details(
                ErrorCode::SubprocessFailed,
                "failed to spawn gwenland subprocess",
                json!({ "error": error.to_string(), "args": args }),
            )
        })?;

    let result = CommandOutput {
        stdout: String::from_utf8_lossy(&output.stdout).to_string(),
        stderr: String::from_utf8_lossy(&output.stderr).to_string(),
        exit_code: output.status.code(),
        duration_ms: started.elapsed().as_millis() as u64,
    };

    if !output.status.success() {
        return Err(StructuredError::with_details(
            failure_code,
            "gwenland subprocess returned a non-zero exit status",
            json!({
                "exit_code": result.exit_code,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "args": args,
            }),
        ));
    }

    Ok(result)
}

pub fn core_json_data(output: &CommandOutput, failure_code: ErrorCode) -> ToolResult<Value> {
    let line = output
        .stdout
        .lines()
        .rev()
        .find(|line| line.trim_start().starts_with('{'))
        .unwrap_or(output.stdout.trim());

    let value = serde_json::from_str::<Value>(line).map_err(|error| {
        StructuredError::with_details(
            failure_code,
            "gwenland subprocess did not return structured JSON",
            json!({
                "parse_error": error.to_string(),
                "stdout": output.stdout,
                "stderr": output.stderr,
            }),
        )
    })?;

    match value.get("status").and_then(Value::as_str) {
        Some("ok") => Ok(value.get("data").cloned().unwrap_or(Value::Null)),
        Some("error") => Err(core_error_from_value(&value, failure_code)),
        _ => Ok(value),
    }
}

pub fn spawn_gwenland_job(
    args: &[String],
    stdout_log: &Path,
    stderr_log: &Path,
) -> ToolResult<u32> {
    let stdout = File::create(stdout_log).map_err(|error| {
        StructuredError::with_details(
            ErrorCode::PermissionDenied,
            "failed to create training stdout log",
            json!({ "path": stdout_log.display().to_string(), "error": error.to_string() }),
        )
    })?;
    let stderr = File::create(stderr_log).map_err(|error| {
        StructuredError::with_details(
            ErrorCode::PermissionDenied,
            "failed to create training stderr log",
            json!({ "path": stderr_log.display().to_string(), "error": error.to_string() }),
        )
    })?;

    let child = Command::new(gwenland_bin())
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr))
        .spawn()
        .map_err(|error| {
            StructuredError::with_details(
                ErrorCode::TrainingFailed,
                "failed to spawn detached training job",
                json!({ "error": error.to_string(), "args": args }),
            )
        })?;

    Ok(child.id())
}

pub fn jobs_dir() -> ToolResult<PathBuf> {
    let dir = gwen_home().join("jobs");
    fs::create_dir_all(&dir).map_err(|error| {
        StructuredError::with_details(
            ErrorCode::PermissionDenied,
            "failed to create GwenLand jobs directory",
            json!({ "path": dir.display().to_string(), "error": error.to_string() }),
        )
    })?;
    Ok(dir)
}

pub fn write_job_record(record: &JobRecord) -> ToolResult<()> {
    let path = job_record_path(&record.job_id)?;
    if path.exists() {
        return Err(StructuredError::with_details(
            ErrorCode::InvalidInput,
            "training job id already exists",
            json!({ "job_id": record.job_id, "path": path.display().to_string() }),
        ));
    }

    let json = serde_json::to_string_pretty(record).map_err(|error| {
        StructuredError::new(
            ErrorCode::TrainingFailed,
            format!("failed to serialize training job record: {error}"),
        )
    })?;

    fs::write(&path, json).map_err(|error| {
        StructuredError::with_details(
            ErrorCode::PermissionDenied,
            "failed to write training job record",
            json!({ "path": path.display().to_string(), "error": error.to_string() }),
        )
    })
}

pub fn read_job_record(job_id: &str) -> ToolResult<JobRecord> {
    let path = job_record_path(job_id)?;
    let raw = fs::read_to_string(&path).map_err(|error| {
        StructuredError::with_details(
            ErrorCode::JobNotFound,
            "training job record not found",
            json!({ "job_id": job_id, "path": path.display().to_string(), "error": error.to_string() }),
        )
    })?;

    serde_json::from_str(&raw).map_err(|error| {
        StructuredError::with_details(
            ErrorCode::TrainingFailed,
            "training job record is corrupted",
            json!({ "job_id": job_id, "path": path.display().to_string(), "error": error.to_string() }),
        )
    })
}

pub fn job_record_path(job_id: &str) -> ToolResult<PathBuf> {
    Ok(jobs_dir()?.join(format!("{job_id}.json")))
}

pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or(0)
}

pub fn read_tail(path: &Path, max_bytes: usize) -> String {
    let Ok(bytes) = fs::read(path) else {
        return String::new();
    };
    let start = bytes.len().saturating_sub(max_bytes);
    String::from_utf8_lossy(&bytes[start..]).to_string()
}

pub fn pid_running(pid: u32) -> bool {
    #[cfg(windows)]
    {
        let filter = format!("PID eq {pid}");
        let Ok(output) = Command::new("tasklist")
            .args(["/FI", &filter, "/NH"])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .output()
        else {
            return false;
        };
        String::from_utf8_lossy(&output.stdout).contains(&pid.to_string())
    }

    #[cfg(unix)]
    {
        Command::new("kill")
            .args(["-0", &pid.to_string()])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|status| status.success())
            .unwrap_or(false)
    }

    #[cfg(not(any(unix, windows)))]
    {
        let _ = pid;
        false
    }
}

fn gwenland_bin() -> OsString {
    if let Some(bin) = std::env::var_os("GWENLAND_BIN") {
        return bin;
    }

    let bin_name = if cfg!(windows) {
        "gwenland.exe"
    } else {
        "gwenland"
    };
    if let Ok(current_exe) = std::env::current_exe() {
        if let Some(parent) = current_exe.parent() {
            let sibling = parent.join(bin_name);
            if sibling.exists() {
                return sibling.into_os_string();
            }
        }
    }

    OsString::from("gwenland")
}

fn gwen_home() -> PathBuf {
    if let Some(home) = std::env::var_os("GWEN_HOME") {
        return PathBuf::from(home);
    }
    if let Some(home) = std::env::var_os("USERPROFILE") {
        return PathBuf::from(home).join(".gwenland");
    }
    if let Some(home) = std::env::var_os("HOME") {
        return PathBuf::from(home).join(".gwenland");
    }
    PathBuf::from(".gwenland")
}

fn core_error_from_value(value: &Value, fallback_code: ErrorCode) -> StructuredError {
    let Some(error) = value.get("error") else {
        return StructuredError::new(fallback_code, "gwenland returned an error");
    };

    let code = error
        .get("code")
        .and_then(Value::as_str)
        .unwrap_or(fallback_code.as_str())
        .to_string();
    let message = error
        .get("message")
        .and_then(Value::as_str)
        .unwrap_or("gwenland returned an error")
        .to_string();
    let details = error.get("details").cloned();

    StructuredError {
        code,
        message,
        details,
    }
}
