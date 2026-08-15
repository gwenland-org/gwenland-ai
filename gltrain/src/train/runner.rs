// @INFO: Pre-flight checks + subprocess spawn for `gwen train`.
// @EDITABLE: check_model_on_hub() uses a HEAD request — adjust timeout or retry if needed.
// @DANGER: script temp file must remain in scope for the entire subprocess lifetime.
//          Assigning it to `_script` (underscore) is intentional: keep alive, not unused.

use anyhow::{Context, Result, bail};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use tokio::io::{AsyncBufReadExt, BufReader};

use crate::dataset::validate::{ValidateOptions, run_validation};
use crate::storage::registry::{ModelEntry, ModelRegistry};
use crate::train::config::{ResumeMode, TrainConfig};
use crate::train::script::write_train_script;
use crate::train::vram::{estimate_train_time, estimate_vram, vram_suggestions};

/// Subset of CLI args that can override YAML fields.
/// Defined here so config.rs can reference it without a TUI dependency.
pub struct TrainOverrides {
    pub model: Option<String>,
    pub dataset: Option<PathBuf>,
    pub output: Option<PathBuf>,
    pub name: Option<String>,
}

// ── public entry point ───────────────────────────────────────────────────────

/// Run pre-flight checks and, on success, return a ready-to-read Child.
/// The caller owns the returned Child and tempfile; both must stay alive for the
/// duration of training.
/// @INFO: Used by the TUI path. Headless path uses run_train() instead.
pub async fn preflight_and_spawn(
    config: &TrainConfig,
    verbose: bool,
    custom_script: Option<&Path>,
) -> Result<(tokio::process::Child, tempfile::NamedTempFile)> {
    config.validate()?;

    let val_opts = ValidateOptions {
        strict: false,
        fix: false,
        inplace: false,
    };
    let val = run_validation(&config.dataset, &val_opts)
        .map_err(|e| anyhow::anyhow!("dataset validation failed: {}", e))?;
    if val.error_count > 0 {
        bail!(
            "{} dataset error{} found. Run `gwen dataset validate -i {}` to inspect.",
            val.error_count,
            if val.error_count == 1 { "" } else { "s" },
            config.dataset.display()
        );
    }

    if !is_local_model_path(&config.model) {
        let registry = ModelRegistry::load()?;
        if registry.find(&config.model).is_none() {
            check_model_on_hub(&config.model).await?;
        }
    } else {
        let resolved = resolve_local_model_path(&config.model);
        if !resolved.exists() {
            bail!("local model file not found: {}", resolved.display());
        }
    }

    check_output_dir(&config.output)?;

    let script = write_train_script(custom_script)?;
    let child = spawn_child(config, script.path(), verbose)?;
    Ok((child, script))
}

pub async fn run_train(
    config: &TrainConfig,
    dry_run: bool,
    verbose: bool,
    custom_script: Option<&Path>,
) -> Result<()> {
    run_train_with_opts(
        config,
        dry_run,
        verbose,
        false,
        custom_script,
        false,
        None,
        ResumeMode::None,
    )
        .await
}

pub async fn run_train_with_opts(
    config: &TrainConfig,
    dry_run: bool,
    verbose: bool,
    json_output: bool,
    custom_script: Option<&Path>,
    gdtqp: bool,
    max_steps: Option<usize>,
    resume: ResumeMode,
) -> Result<()> {
    // 1. Validate config fields
    config.validate()?;

    // 2. Dataset validation — reuse JIN-169 implementation
    let val_opts = ValidateOptions {
        strict: false,
        fix: false,
        inplace: false,
    };
    let val = run_validation(&config.dataset, &val_opts)
        .map_err(|e| anyhow::anyhow!("dataset validation failed: {}", e))?;
    if val.error_count > 0 {
        bail!(
            "{} dataset error{} found. Run `gwen dataset validate -i {}` to inspect.",
            val.error_count,
            if val.error_count == 1 { "" } else { "s" },
            config.dataset.display()
        );
    }

    // 3. Check model — skip registry/HF for local file paths
    if !is_local_model_path(&config.model) {
        let registry = ModelRegistry::load()?;
        if registry.find(&config.model).is_none() {
            check_model_on_hub(&config.model).await?;
        }
    } else {
        let resolved = resolve_local_model_path(&config.model);
        if !resolved.exists() {
            bail!("local model file not found: {}", resolved.display());
        }
    }

    // 4. Check output dir + disk space
    check_output_dir(&config.output)?;

    // 5. --dry-run for local GGUF: run a real native 1-step pass and report
    //    memory + loss. For remote/HF models, fall back to the estimation table.
    if dry_run && is_local_model_path(&config.model) {
        let gguf_path = resolve_local_model_path(&config.model);
        // Dry-run is a 1-step memory/loss probe; never resume into it.
        let native_cfg = train_config_to_native(config, true, gdtqp, max_steps, ResumeMode::None);
        crate::train::native_runner::run_native_local(&native_cfg, &gguf_path, None, None)?;
        return Ok(());
    }
    if dry_run {
        emit_dry_run_report(config, &val, json_output)?;
        return Ok(());
    }

    // 6. Dispatch: native Rust path for local GGUF, Python script path otherwise
    if is_local_model_path(&config.model) {
        let gguf_path = resolve_local_model_path(&config.model);
        let native_cfg = train_config_to_native(config, false, gdtqp, max_steps, resume);
        let (tx, _rx) = std::sync::mpsc::channel::<String>();
        std::thread::spawn(move || {
            while let Ok(msg) = _rx.recv() {
                println!("{}", msg);
            }
        });
        crate::train::native_runner::run_native_local(&native_cfg, &gguf_path, None, Some(tx))?;
    } else {
        let _script = write_train_script(custom_script)?;
        spawn_training(config, _script.path(), verbose).await?;
    }

    // 7. Update registry on success
    register_trained_model(config)?;

    Ok(())
}

// ── dry-run report ────────────────────────────────────────────────────────────

fn emit_dry_run_report(
    config: &TrainConfig,
    val: &crate::dataset::validate::ValidationResult,
    json_output: bool,
) -> Result<()> {
    if json_output {
        return emit_dry_run_json(config, val);
    }
    print_dry_run_report(config, val)
}

fn emit_dry_run_json(
    config: &TrainConfig,
    val: &crate::dataset::validate::ValidationResult,
) -> Result<()> {
    let est = estimate_vram(config);
    let time_str = estimate_train_time(config, val.valid);
    let total_steps = (val.valid / config.batch_size.max(1) as usize) * config.epochs as usize;

    let free_gb = if config.output.exists() || std::fs::create_dir_all(&config.output).is_ok() {
        crate::platform::hardware::check_disk_space(&config.output)
            .map(|(_, avail)| avail as f64 / (1024.0 * 1024.0 * 1024.0))
    } else {
        None
    };

    let valid = est.available_gb.map(|_| est.fits).unwrap_or(true);

    let obj = serde_json::json!({
        "command": "train",
        "valid": valid,
        "vram_gb": (est.total_gb as f64 * 10.0).round() / 10.0,
        "vram_available_gb": est.available_gb.map(|v| (v as f64 * 10.0).round() / 10.0),
        "vram_fits": est.fits,
        "duration_est": time_str,
        "total_steps": total_steps,
        "disk_free_gb": free_gb.map(|g| (g * 10.0).round() / 10.0),
        "dataset_samples": val.valid,
        "model": config.model,
    });
    println!("{}", obj);
    if !valid {
        std::process::exit(crate::dry_run::EXIT_INSUFFICIENT_RESOURCES);
    }
    Ok(())
}

fn print_dry_run_report(
    config: &TrainConfig,
    val: &crate::dataset::validate::ValidationResult,
) -> Result<()> {
    let divider = "━".repeat(50);
    let line = "─".repeat(44);

    // Check output dir free space
    let free_gb = if config.output.exists() || std::fs::create_dir_all(&config.output).is_ok() {
        crate::platform::hardware::check_disk_space(&config.output)
            .map(|(_, avail)| avail as f64 / (1024.0 * 1024.0 * 1024.0))
    } else {
        None
    };

    println!("\ngwen train --dry-run");
    println!("{}", divider);

    // ── Pre-flight summary ────────────────────────────────────────────────────
    let config_label = "config";
    println!("  {:<16} {:<24} ✓", config_label, "(loaded)");

    let dataset_str = config.dataset.display().to_string();
    let dataset_short = if dataset_str.len() > 22 {
        &dataset_str[dataset_str.len() - 22..]
    } else {
        &dataset_str
    };
    println!(
        "  {:<16} {:<24} ✓  ({} valid samples)",
        "Dataset", dataset_short, val.valid
    );

    let model_short = if config.model.len() > 24 {
        &config.model[..24]
    } else {
        &config.model
    };
    println!("  {:<16} {:<24} ✓", "Base model", model_short);

    let output_str = config.output.display().to_string();
    let output_short = if output_str.len() > 22 {
        &output_str[output_str.len() - 22..]
    } else {
        &output_str
    };
    if let Some(gb) = free_gb {
        println!(
            "  {:<16} {:<24} ✓  ({:.0} GB free)",
            "Output dir", output_short, gb
        );
    } else {
        println!("  {:<16} {:<24} ✓", "Output dir", output_short);
    }

    println!();

    // ── VRAM breakdown ────────────────────────────────────────────────────────
    let est = estimate_vram(config);

    println!("  VRAM Breakdown");
    println!("  {}", line);
    println!(
        "  {:<20} {:<26} {:.1} GB",
        "Base model", est.model_label, est.base_gb
    );
    let lora_desc = format!("r={}, target: {}", config.lora_r, config.lora_target);
    println!(
        "  {:<20} {:<26} {:.1} GB",
        "LoRA adapters", lora_desc, est.lora_gb
    );
    let act_desc = format!("batch={}, seq={}", config.batch_size, config.max_seq_len);
    println!(
        "  {:<20} {:<26} {:.1} GB",
        "Activations", act_desc, est.activation_gb
    );

    let opt_label = if config.optimizer.contains("8bit") {
        "AdamW 8-bit states"
    } else {
        "AdamW states"
    };
    println!(
        "  {:<20} {:<26} {:.1} GB",
        "Optimizer", opt_label, est.optimizer_gb
    );
    println!(
        "  {:<20} {:<26} {:.1} GB",
        "Safety buffer", "+20% overhead", est.safety_gb
    );
    println!("  {}", line);
    println!("  {:<46} {:.1} GB", "Total estimated", est.total_gb);

    if let Some(avail) = est.available_gb {
        let gpu_name = est.gpu_name.as_deref().unwrap_or("GPU");
        if est.fits {
            println!(
                "  {:<20} {:<20} {:.1} GB   ✓ fits!",
                "Available VRAM",
                format!("{} (detected)", gpu_name),
                avail
            );
        } else {
            println!(
                "  {:<20} {:<20} {:.1} GB   ✗ insufficient!",
                "Available VRAM",
                format!("{} (detected)", gpu_name),
                avail
            );
        }
    } else {
        println!("  {:<46} unknown (no GPU detected)", "Available VRAM");
    }

    println!();

    // ── Training estimate ─────────────────────────────────────────────────────
    let time_str = estimate_train_time(config, val.valid);
    let total_steps = (val.valid / config.batch_size.max(1) as usize) * config.epochs as usize;

    println!("  Training Estimate");
    println!("  {}", line);
    println!("  {:<20} {}", "Epochs", config.epochs);
    println!("  {:<20} {}", "Steps", total_steps);
    println!(
        "  {:<20} ~{}  (based on T4 baseline)",
        "Est. time", time_str
    );
    println!("{}", divider);

    // ── Result ────────────────────────────────────────────────────────────────
    if est.available_gb.map(|_| !est.fits).unwrap_or(false) {
        println!("  ✗ Not enough VRAM. Adjust config and retry.");
        let suggestions = vram_suggestions(config, &est);
        if !suggestions.is_empty() {
            println!();
            println!("  Suggestions:");
            for s in &suggestions {
                println!("    {}", s);
            }
        }
        println!("{}", divider);
        std::process::exit(1);
    } else {
        println!("  ✓ Ready to train. Remove --dry-run to start.");
        let config_path = "(your config)";
        println!("    Run: gwen train -c {}", config_path);
    }

    println!();
    Ok(())
}

// ── pre-flight helpers ───────────────────────────────────────────────────────

/// Returns true when `model` looks like a local file path rather than a HF model ID.
/// Triggers on: leading `.` (relative), `/` (absolute), `~` (home), or `.gguf` suffix.
fn is_local_model_path(model: &str) -> bool {
    model.starts_with('.')
        || model.starts_with('/')
        || model.starts_with('~')
        || model.ends_with(".gguf")
}

/// Resolve a local model path: expand `~` to home dir and make relative paths absolute
/// relative to the current working directory.
fn resolve_local_model_path(model: &str) -> PathBuf {
    if let Some(rest) = model.strip_prefix("~/") {
        if let Some(home) = dirs::home_dir() {
            return home.join(rest);
        }
    }
    PathBuf::from(model)
}

/// HEAD request to HF Hub to verify the model exists without downloading it.
async fn check_model_on_hub(model_id: &str) -> Result<()> {
    let url = format!("https://huggingface.co/{}", model_id);
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .context("failed to build HTTP client")?;

    let resp = client
        .head(&url)
        .send()
        .await
        .with_context(|| format!("could not reach HF Hub to verify model '{}'", model_id))?;

    if resp.status() == reqwest::StatusCode::NOT_FOUND {
        bail!(
            "model '{}' not found in local registry or HF Hub. \
             Check the model ID or run `gwen fetch -m {}`.",
            model_id,
            model_id
        );
    }
    if !resp.status().is_success() && resp.status() != reqwest::StatusCode::UNAUTHORIZED {
        // @INFO: 401 from HF just means the model is gated — it still exists.
        bail!(
            "HF Hub returned {} for model '{}'. Check network connectivity.",
            resp.status(),
            model_id
        );
    }
    Ok(())
}

/// Ensure the output directory can be created and has ≥10 GB free.
fn check_output_dir(output: &Path) -> Result<()> {
    if !output.exists() {
        std::fs::create_dir_all(output)
            .with_context(|| format!("cannot create output directory: {}", output.display()))?;
    }

    // @EDITABLE: minimum free space threshold — lower if users are on small drives
    const MIN_FREE_BYTES: u64 = 10 * 1024 * 1024 * 1024; // 10 GB

    let free = crate::platform::hardware::check_disk_space(output);
    if let Some((free_bytes, _total)) = free {
        if free_bytes < MIN_FREE_BYTES {
            bail!(
                "insufficient disk space at '{}': {:.1} GB free, need at least 10 GB.",
                output.display(),
                free_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
            );
        }
    }
    // If check_disk_space returns None we allow it — platform may not support the query.
    Ok(())
}

// ── subprocess ───────────────────────────────────────────────────────────────

/// Spawn the training subprocess and return the Child handle.
/// Stdout is piped; stderr is inherited when verbose, null otherwise.
/// @INFO: Called by both the headless path and the TUI path (which takes stdout ownership).
pub fn spawn_child(
    config: &TrainConfig,
    script: &Path,
    verbose: bool,
) -> Result<tokio::process::Child> {
    tokio::process::Command::new("python3")
        .arg(script)
        .args(config.to_python_args())
        .stdout(Stdio::piped())
        .stderr(if verbose {
            Stdio::inherit()
        } else {
            Stdio::null()
        })
        .spawn()
        .context("failed to spawn python3 — is Python 3 installed and on PATH?")
}

async fn spawn_training(config: &TrainConfig, script: &Path, verbose: bool) -> Result<()> {
    let mut child = spawn_child(config, script, verbose)?;

    let stdout = child
        .stdout
        .take()
        .context("could not capture subprocess stdout")?;

    let mut lines = BufReader::new(stdout).lines();

    while let Some(line) = lines
        .next_line()
        .await
        .context("error reading subprocess stdout")?
    {
        match serde_json::from_str::<serde_json::Value>(&line) {
            Ok(val) => handle_json_line(&val)?,
            Err(_) => {
                if verbose {
                    eprintln!("  [py] {}", line);
                }
            }
        }
    }

    let status = child
        .wait()
        .await
        .context("failed to wait for training subprocess")?;
    if !status.success() {
        bail!(
            "training subprocess exited with code {}",
            status.code().unwrap_or(-1)
        );
    }
    Ok(())
}

/// Handle a single parsed JSON line from base_train.py stdout.
fn handle_json_line(val: &serde_json::Value) -> Result<()> {
    if let Some(err_msg) = val.get("error") {
        let message = val
            .get("message")
            .and_then(|m| m.as_str())
            .unwrap_or("unknown error");
        eprintln!("error: {}", message);
        bail!("training failed: {}", err_msg);
    }

    match val.get("event").and_then(|e| e.as_str()) {
        Some("done") => {
            let out = val.get("output").and_then(|o| o.as_str()).unwrap_or("");
            println!("  + Training complete -> {}", out);
        }
        Some("interrupted") => {
            let msg = val
                .get("message")
                .and_then(|m| m.as_str())
                .unwrap_or("interrupted");
            println!("  ! {}", msg);
        }
        _ => {
            // Step log line — only print in verbose mode (caller filters via flag)
            if let Some(step) = val.get("step").and_then(|s| s.as_u64()) {
                let loss = val.get("loss").and_then(|l| l.as_f64()).unwrap_or(0.0);
                println!("  step {:>6} | loss {:.4}", step, loss);
            }
        }
    }
    Ok(())
}

// ── registry update ───────────────────────────────────────────────────────────

/// Add a minimal registry entry for the freshly trained model.
fn register_trained_model(config: &TrainConfig) -> Result<()> {
    let mut registry = ModelRegistry::load()?;

    let id = config
        .name
        .clone()
        .unwrap_or_else(|| config.model.replace('/', "_"));

    let entry = ModelEntry {
        id: id.clone(),
        source: config.model.clone(),
        format: "lora".into(),
        quant: if config.qlora {
            "qlora".into()
        } else {
            "full".into()
        },
        size_bytes: 0,
        downloaded_at: chrono::Utc::now().to_rfc3339(),
        sha256: String::new(),
        path: config.output.clone(),
    };

    registry.upsert(entry);
    registry.save().with_context(|| {
        format!(
            "training succeeded but could not update model registry for '{}'",
            id
        )
    })?;
    Ok(())
}

/// Convert the YAML-backed `TrainConfig` to the candle-native `NewTrainConfig`.
///
/// When `dry_run` is set, `max_steps` is capped to 1 so the native runner runs
/// a single forward/backward/step and reports memory + loss, then exits.
fn train_config_to_native(
    cfg: &TrainConfig,
    dry_run: bool,
    gdtqp: bool,
    max_steps: Option<usize>,
    resume: ResumeMode,
) -> crate::train::config::NewTrainConfig {
    crate::train::config::NewTrainConfig {
        model_id: cfg.model.clone(),
        dataset_path: cfg.dataset.clone(),
        epochs: cfg.epochs as usize,
        batch_size: cfg.batch_size as usize,
        grad_accum: cfg.grad_accum as usize,
        lr: cfg.learning_rate,
        max_grad_norm: cfg.max_grad_norm,
        lora: crate::train::config::LoraConfig {
            r: cfg.lora_r as usize,
            alpha: cfg.lora_alpha as f32,
            dropout: cfg.lora_dropout,
            target_modules: cfg
                .lora_target
                .split(',')
                .map(|s| s.trim().to_string())
                .collect(),
        },
        dry_run,
        max_steps: if dry_run { Some(1) } else { max_steps },
        output_path: cfg.output.clone(),
        custom_script: None,
        gdtqp,
        resume_checkpoint: resume,
    }
}
