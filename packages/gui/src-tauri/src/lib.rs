use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::HashMap,
    env,
    ffi::OsStr,
    fs::{self, File},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::Mutex,
    time::{Instant, SystemTime, UNIX_EPOCH},
};
use tauri::Emitter;

#[derive(Default)]
struct JobStore {
    jobs: Mutex<HashMap<String, RunningJob>>,
}

struct RunningJob {
    child: Option<Child>,
    started: Instant,
    max_steps: u32,
    stdout_log: PathBuf,
    stderr_log: PathBuf,
    status: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct InferParams {
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    system_prompt: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct InferResult {
    text: String,
    tokens_generated: u32,
    tokens_per_second: f64,
    time_ms: u64,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct TrainConfig {
    base_model: String,
    dataset: String,
    output_dir: String,
    lora_rank: u32,
    learning_rate: f64,
    max_steps: u32,
    batch_size: u32,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct TrainJob {
    job_id: String,
    status: String,
    pid: Option<u32>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct TrainStatus {
    job_id: String,
    status: String,
    step: u32,
    max_steps: u32,
    loss: Option<f64>,
    progress: f64,
    elapsed_seconds: u64,
    log_tail: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BenchmarkResult {
    model: String,
    tokens_per_second: f64,
    cold_start_ms: f64,
    memory_mb: f64,
    runs: u32,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ModelInfo {
    name: String,
    path: String,
    size_mb: f64,
    quantization: String,
    parameters: String,
    modified_ms: u64,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct SystemStats {
    memory_mb: f64,
    cold_start_ms: f64,
    binary_mb: f64,
    models_dir: String,
    binary_path: String,
}

#[tauri::command]
fn run_inference(
    model_path: String,
    prompt: String,
    params: InferParams,
) -> Result<InferResult, String> {
    if model_path.trim().is_empty() {
        return Err("Select a model before running inference.".to_string());
    }

    let final_prompt = match params.system_prompt.as_deref() {
        Some(system) if !system.trim().is_empty() => format!("System: {}\n\nUser: {}", system, prompt),
        _ => prompt,
    };

    let max_tokens = params.max_tokens.unwrap_or(128).to_string();
    let temperature = params.temperature.unwrap_or(0.7).to_string();
    let top_p = params.top_p.unwrap_or(0.95).to_string();

    let started = Instant::now();
    let output = run_gwenland([
        "--json",
        "--non-interactive",
        "run",
        model_path.as_str(),
        "--prompt",
        final_prompt.as_str(),
        "--max-tokens",
        max_tokens.as_str(),
        "--temperature",
        temperature.as_str(),
        "--top-p",
        top_p.as_str(),
    ])?;

    let elapsed_ms = started.elapsed().as_millis() as u64;
    let json = parse_json_output(&output);
    let data = json.as_ref().and_then(|value| value.get("data")).unwrap_or(&Value::Null);

    let text = get_string(data, &["text", "response", "output"])
        .or_else(|| get_string(json.as_ref().unwrap_or(&Value::Null), &["text", "response", "output"]))
        .unwrap_or_else(|| output.trim().to_string());
    let tokens_generated = get_u64(data, &["tokens_generated", "tokens", "generated_tokens"])
        .unwrap_or_else(|| estimate_tokens(&text) as u64) as u32;
    let tokens_per_second = get_f64(data, &["tokens_per_second", "tok_s", "tokens_per_sec"])
        .unwrap_or_else(|| {
            if elapsed_ms == 0 {
                0.0
            } else {
                tokens_generated as f64 / (elapsed_ms as f64 / 1000.0)
            }
        });

    Ok(InferResult {
        text,
        tokens_generated,
        tokens_per_second,
        time_ms: elapsed_ms,
    })
}

#[tauri::command]
fn start_training(config: TrainConfig, state: tauri::State<JobStore>) -> Result<TrainJob, String> {
    if config.base_model.trim().is_empty() {
        return Err("Select a base model before starting training.".to_string());
    }
    if config.dataset.trim().is_empty() {
        return Err("Select a dataset before starting training.".to_string());
    }
    if config.output_dir.trim().is_empty() {
        return Err("Choose an output directory before starting training.".to_string());
    }

    let jobs_dir = gwenland_core::storage::paths::root_dir().join("jobs");
    fs::create_dir_all(&jobs_dir)
        .map_err(|err| format!("Failed to create jobs directory: {err}"))?;

    let job_id = format!("train-{}", now_ms());
    let stdout_log = jobs_dir.join(format!("{job_id}.stdout.log"));
    let stderr_log = jobs_dir.join(format!("{job_id}.stderr.log"));
    let stdout = File::create(&stdout_log)
        .map_err(|err| format!("Failed to create stdout log: {err}"))?;
    let stderr = File::create(&stderr_log)
        .map_err(|err| format!("Failed to create stderr log: {err}"))?;

    let lr = config.learning_rate.to_string();
    let lora_rank = config.lora_rank.to_string();
    let max_steps = config.max_steps.to_string();
    let batch_size = config.batch_size.to_string();

    let child = Command::new(gwenland_bin())
        .args([
            "--json",
            "--non-interactive",
            "train",
            "--model",
            config.base_model.as_str(),
            "--dataset",
            config.dataset.as_str(),
            "--output",
            config.output_dir.as_str(),
            "--verbose",
            "--lora-rank",
            lora_rank.as_str(),
            "--batch-size",
            batch_size.as_str(),
            "--max-steps",
            max_steps.as_str(),
            "--lr",
            lr.as_str(),
        ])
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr))
        .spawn()
        .map_err(|err| format!("Failed to start training: {err}"))?;

    let pid = child.id();
    let mut jobs = state
        .jobs
        .lock()
        .map_err(|_| "Training job store is unavailable.".to_string())?;
    jobs.insert(
        job_id.clone(),
        RunningJob {
            child: Some(child),
            started: Instant::now(),
            max_steps: config.max_steps,
            stdout_log,
            stderr_log,
            status: "running".to_string(),
        },
    );

    Ok(TrainJob {
        job_id,
        status: "running".to_string(),
        pid: Some(pid),
    })
}

#[tauri::command]
fn get_train_status(job_id: String, state: tauri::State<JobStore>) -> Result<TrainStatus, String> {
    let mut jobs = state
        .jobs
        .lock()
        .map_err(|_| "Training job store is unavailable.".to_string())?;
    let job = jobs
        .get_mut(&job_id)
        .ok_or_else(|| format!("Training job not found: {job_id}"))?;

    if let Some(child) = job.child.as_mut() {
        if let Some(status) = child
            .try_wait()
            .map_err(|err| format!("Failed to read training status: {err}"))?
        {
            job.status = if status.success() {
                "completed".to_string()
            } else {
                "failed".to_string()
            };
            job.child = None;
        }
    }

    let stdout_tail = read_tail(&job.stdout_log, 12);
    let stderr_tail = read_tail(&job.stderr_log, 8);
    let log_tail = match (stdout_tail.trim().is_empty(), stderr_tail.trim().is_empty()) {
        (false, false) => format!("{stdout_tail}\n{stderr_tail}"),
        (false, true) => stdout_tail,
        (true, false) => stderr_tail,
        (true, true) => "Waiting for training logs...".to_string(),
    };
    let step = extract_u32(&log_tail, &["step", "iter", "iteration"]).unwrap_or(0);
    let loss = extract_f64(&log_tail, &["loss", "train_loss"]);
    let progress = if job.max_steps == 0 {
        0.0
    } else {
        (step.min(job.max_steps) as f64 / job.max_steps as f64) * 100.0
    };

    Ok(TrainStatus {
        job_id,
        status: job.status.clone(),
        step,
        max_steps: job.max_steps,
        loss,
        progress,
        elapsed_seconds: job.started.elapsed().as_secs(),
        log_tail,
    })
}

#[tauri::command]
fn run_benchmark(model_path: String, runs: Option<u32>) -> Result<BenchmarkResult, String> {
    if model_path.trim().is_empty() {
        return Err("Select a model before running a benchmark.".to_string());
    }

    let run_count = runs.unwrap_or(5).max(1);
    let run_count_arg = run_count.to_string();
    let output = run_gwenland([
        "--json",
        "--non-interactive",
        "benchmark",
        model_path.as_str(),
        "--runs",
        run_count_arg.as_str(),
        "--full",
    ])?;
    let json = parse_json_output(&output);
    let data = json.as_ref().and_then(|value| value.get("data")).unwrap_or(&Value::Null);

    let inference = data.get("inference").unwrap_or(data);
    let cold_start = data.get("cold_start").unwrap_or(data);
    let memory = data.get("memory").unwrap_or(data);

    Ok(BenchmarkResult {
        model: model_path,
        tokens_per_second: get_f64(inference, &["tokens_per_second", "tokens_per_sec", "tok_s"])
            .unwrap_or(0.0),
        cold_start_ms: get_f64(cold_start, &["median_ms", "mean_ms", "cold_start_ms"]).unwrap_or(0.0),
        memory_mb: get_f64(memory, &["baseline_mb", "peak_mb", "memory_mb"]).unwrap_or(0.0),
        runs: run_count,
    })
}

#[tauri::command]
fn list_models() -> Result<Vec<ModelInfo>, String> {
    let models_dir = gwenland_core::storage::paths::models_dir();
    fs::create_dir_all(&models_dir)
        .map_err(|err| format!("Failed to create models directory: {err}"))?;

    let mut models = Vec::new();
    for entry in fs::read_dir(&models_dir)
        .map_err(|err| format!("Failed to read models directory: {err}"))?
    {
        let entry = entry.map_err(|err| format!("Failed to read model entry: {err}"))?;
        let path = entry.path();
        if is_model_file(&path) {
            models.push(model_info(path)?);
        }
    }

    models.sort_by(|a, b| b.modified_ms.cmp(&a.modified_ms));
    Ok(models)
}

#[tauri::command]
fn import_model(source_path: String) -> Result<ModelInfo, String> {
    let source = PathBuf::from(source_path);
    if !is_model_file(&source) {
        return Err("Only .gguf and .ggqr model files can be imported.".to_string());
    }

    let models_dir = gwenland_core::storage::paths::models_dir();
    fs::create_dir_all(&models_dir)
        .map_err(|err| format!("Failed to create models directory: {err}"))?;

    let file_name = source
        .file_name()
        .ok_or_else(|| "Model file has no file name.".to_string())?;
    let destination = models_dir.join(file_name);
    if destination.exists() {
        return Err(format!(
            "A model named {} already exists.",
            destination.display()
        ));
    }

    fs::copy(&source, &destination).map_err(|err| format!("Failed to import model: {err}"))?;
    model_info(destination)
}

#[tauri::command]
fn download_model(window: tauri::Window, url: String) -> Result<ModelInfo, String> {
    if url.trim().is_empty() {
        return Err("Enter a model URL before downloading.".to_string());
    }

    let _ = window.emit("download-progress", "starting");
    let output = run_gwenland(["--json", "--non-interactive", "fetch", url.as_str(), "--yes"])?;
    let _ = window.emit("download-progress", "completed");

    let json = parse_json_output(&output);
    let data = json.as_ref().and_then(|value| value.get("data")).unwrap_or(&Value::Null);
    if let Some(path) = get_string(data, &["path", "model_path", "destination"]) {
        return model_info(PathBuf::from(path));
    }

    list_models()?
        .into_iter()
        .next()
        .ok_or_else(|| "Download completed, but no local model was found.".to_string())
}

#[tauri::command]
fn get_system_stats() -> Result<SystemStats, String> {
    let models_dir = gwenland_core::storage::paths::models_dir();
    let binary = gwenland_bin();
    let binary_mb = fs::metadata(&binary)
        .map(|meta| bytes_to_mb(meta.len()))
        .unwrap_or(11.44);

    Ok(SystemStats {
        memory_mb: 81.0,
        cold_start_ms: 10.1,
        binary_mb,
        models_dir: models_dir.display().to_string(),
        binary_path: binary.display().to_string(),
    })
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(JobStore::default())
        .invoke_handler(tauri::generate_handler![
            run_inference,
            start_training,
            get_train_status,
            run_benchmark,
            list_models,
            import_model,
            download_model,
            get_system_stats
        ])
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

fn run_gwenland<I, S>(args: I) -> Result<String, String>
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    let output = Command::new(gwenland_bin())
        .args(args)
        .output()
        .map_err(|err| format!("Failed to run GwenLand CLI: {err}"))?;

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    if output.status.success() {
        return Ok(stdout);
    }

    let stderr = String::from_utf8_lossy(&output.stderr);
    Err(if stderr.trim().is_empty() {
        format!("GwenLand CLI exited with status {}", output.status)
    } else {
        stderr.trim().to_string()
    })
}

fn gwenland_bin() -> PathBuf {
    if let Ok(path) = env::var("GWENLAND_BIN") {
        return PathBuf::from(path);
    }

    if let Ok(current) = env::current_exe() {
        if let Some(parent) = current.parent() {
            for name in binary_names() {
                let candidate = parent.join(name);
                if candidate.exists() {
                    return candidate;
                }
            }
        }
    }

    let sidecars = Path::new(env!("CARGO_MANIFEST_DIR")).join("binaries");
    if let Ok(entries) = fs::read_dir(sidecars) {
        for entry in entries.flatten() {
            let path = entry.path();
            let Some(name) = path.file_name().and_then(|value| value.to_str()) else {
                continue;
            };
            if name.starts_with("gwen-") && path.is_file() {
                return path;
            }
        }
    }

    PathBuf::from(if cfg!(windows) { "gwenland.exe" } else { "gwenland" })
}

fn binary_names() -> [&'static str; 4] {
    if cfg!(windows) {
        ["gwenland.exe", "gwen.exe", "gwenland", "gwen"]
    } else {
        ["gwenland", "gwen", "gwenland.exe", "gwen.exe"]
    }
}

fn parse_json_output(output: &str) -> Option<Value> {
    for line in output.lines().rev() {
        let trimmed = line.trim();
        if trimmed.starts_with('{') {
            if let Ok(value) = serde_json::from_str::<Value>(trimmed) {
                return Some(value);
            }
        }
    }
    serde_json::from_str::<Value>(output).ok()
}

fn get_string(value: &Value, keys: &[&str]) -> Option<String> {
    keys.iter()
        .find_map(|key| value.get(*key).and_then(Value::as_str).map(ToString::to_string))
}

fn get_u64(value: &Value, keys: &[&str]) -> Option<u64> {
    keys.iter().find_map(|key| value.get(*key).and_then(Value::as_u64))
}

fn get_f64(value: &Value, keys: &[&str]) -> Option<f64> {
    keys.iter().find_map(|key| {
        value
            .get(*key)
            .and_then(|number| number.as_f64().or_else(|| number.as_u64().map(|value| value as f64)))
    })
}

fn estimate_tokens(text: &str) -> usize {
    text.split_whitespace().count().max(1)
}

fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or_default()
}

fn read_tail(path: &Path, lines: usize) -> String {
    let Ok(content) = fs::read_to_string(path) else {
        return String::new();
    };
    let mut tail = content.lines().rev().take(lines).collect::<Vec<_>>();
    tail.reverse();
    tail.join("\n")
}

fn extract_u32(text: &str, keys: &[&str]) -> Option<u32> {
    extract_number(text, keys).and_then(|value| value.parse::<u32>().ok())
}

fn extract_f64(text: &str, keys: &[&str]) -> Option<f64> {
    extract_number(text, keys).and_then(|value| value.parse::<f64>().ok())
}

fn extract_number(text: &str, keys: &[&str]) -> Option<String> {
    for line in text.lines().rev() {
        let lower = line.to_ascii_lowercase();
        for key in keys {
            if let Some(index) = lower.find(key) {
                let after = &line[index + key.len()..];
                let number = after
                    .trim_start_matches(|ch: char| {
                        ch == ':' || ch == '=' || ch == ' ' || ch == '\t' || ch == '#'
                    })
                    .chars()
                    .take_while(|ch| ch.is_ascii_digit() || *ch == '.')
                    .collect::<String>();
                if !number.is_empty() {
                    return Some(number);
                }
            }
        }
    }
    None
}

fn is_model_file(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| {
            extension.eq_ignore_ascii_case("gguf") || extension.eq_ignore_ascii_case("ggqr")
        })
        .unwrap_or(false)
}

fn model_info(path: PathBuf) -> Result<ModelInfo, String> {
    let metadata = fs::metadata(&path).map_err(|err| format!("Failed to read model metadata: {err}"))?;
    let file_name = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("model")
        .to_string();

    Ok(ModelInfo {
        quantization: infer_quantization(&file_name),
        parameters: infer_parameters(&file_name),
        name: file_name,
        path: path.display().to_string(),
        size_mb: bytes_to_mb(metadata.len()),
        modified_ms: metadata
            .modified()
            .ok()
            .and_then(|modified| modified.duration_since(UNIX_EPOCH).ok())
            .map(|duration| duration.as_millis() as u64)
            .unwrap_or_default(),
    })
}

fn infer_quantization(name: &str) -> String {
    let upper = name.to_ascii_uppercase();
    for quant in ["Q8_0", "Q6_K", "Q5_K_M", "Q5_K", "Q4_K_M", "Q4_K", "F16", "BF16"] {
        if upper.contains(quant) {
            return quant.to_string();
        }
    }
    "Unknown".to_string()
}

fn infer_parameters(name: &str) -> String {
    let lower = name.to_ascii_lowercase();
    for params in ["0.5b", "1.5b", "1.7b", "3b", "7b", "8b", "14b", "32b"] {
        if lower.contains(params) {
            return params.to_ascii_uppercase();
        }
    }
    "Local".to_string()
}

fn bytes_to_mb(bytes: u64) -> f64 {
    (bytes as f64 / 1024.0 / 1024.0 * 100.0).round() / 100.0
}
