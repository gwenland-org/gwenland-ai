// @INFO: CLI entry point for `gwen train`.
//
//   Three dispatch paths, selected in order:
//
//   1. --dry-run  → estimate VRAM/time from HF metadata, print table, exit 0.
//                   No weights, no subprocess, no GPU required.
//
//   2. --custom-script → legacy Python subprocess (base_train.py or user script).
//                        Kept for backwards-compatibility; prints a deprecation
//                        warning to stderr, then delegates to preflight_and_spawn.
//
//   3. (default) native Candle path → load JSONL, tokenize, build LoraLayer,
//                                     run TrainingLoop. Launches the TUI panel
//                                     once the model is loaded and the first
//                                     batch is ready so the user can watch
//                                     real-time loss without a subprocess.
//
// @EDITABLE: Add --resume, --checkpoint, --grad-accum, --lora-r flags here.
//            They all map into NewTrainConfig and require no structural changes.
// @DANGER:   The native path creates a VarMap that must stay alive for the
//            entire training run. Do not move it into a closure or sub-scope.

use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context, Result};
use clap::Args;
use gwenland_core::train::config::{LoraConfig, NewTrainConfig, TrainConfig};
use gwenland_core::train::dry_run::{self, DryRunResult};
use gwenland_core::train::native_runner;
use gwenland_core::train::runner::{preflight_and_spawn, run_train_with_opts, TrainOverrides};

use crate::tui::train_panel::{events_from_child, events_from_native_rx, run_train_tui};

// ── Clap args ─────────────────────────────────────────────────────────────────

/// Why clap::Args and not Parser:
/// TrainArgs is registered as a sub-command in main.rs's top-level Parser.
/// Using Args (not Parser) keeps the derive tree consistent with every other
/// command in this crate and lets main.rs own the --help text width/colour.
#[derive(Args, Debug)]
#[command(about = "Fine-tune a model (LoRA/QLoRA)")]
pub struct TrainArgs {
    /// Training config YAML [legacy]
    #[arg(short = 'c', long, value_name = "CONFIG")]
    pub config: Option<PathBuf>,

    /// HuggingFace model ID
    #[arg(short = 'm', long, value_name = "HF_MODEL_ID")]
    pub model: Option<String>,

    /// Path to JSONL dataset
    #[arg(short = 'd', long, value_name = "PATH")]
    pub dataset: Option<PathBuf>,

    /// Training epochs [default: 3]
    #[arg(long, value_name = "N")]
    pub epochs: Option<usize>,

    /// Learning rate [default: 1e-4]
    #[arg(long, value_name = "F")]
    pub lr: Option<f64>,

    /// Output directory [default: ./gwen-output]
    #[arg(short = 'o', long, value_name = "PATH")]
    pub output: Option<PathBuf>,

    /// Estimate VRAM without training
    #[arg(long)]
    pub dry_run: bool,

    /// Stream logs to stdout
    #[arg(short = 'v', long)]
    pub verbose: bool,

    /// Custom Python script [deprecated]
    #[arg(long, value_name = "SCRIPT")]
    pub custom_script: Option<PathBuf>,

    /// Override model display name in local registry
    #[arg(short = 'n', long, value_name = "NAME")]
    pub name: Option<String>,
}

// ── public entry point ────────────────────────────────────────────────────────

/// Top-level async entry point called from main.rs.
/// All errors are printed to stderr and exit 1 — the inner function returns
/// Result so that the `?` operator keeps error paths readable.
pub async fn run_train_cmd(args: TrainArgs, mode: gwenland_core::engine::GwenMode) {
    if let Err(e) = run_train_inner(args, mode).await {
        eprintln!("error: {:?}", e);
        std::process::exit(1);
    }
}

// ── dispatch ──────────────────────────────────────────────────────────────────

async fn run_train_inner(
    args: TrainArgs,
    mode: gwenland_core::engine::GwenMode,
) -> Result<()> {
    // Global --dry-run (GwenMode) and command-local --dry-run are equivalent.
    let dry_run = args.dry_run || mode.dry_run;

    // ── path 1: dry-run ───────────────────────────────────────────────────────
    //
    // Why before the config-YAML load: the dry-run only needs a TrainConfig
    // built from flags — it never reads a YAML or validates python-path fields.
    // Checking dry_run first keeps the flag fast on machines with no Python.
    if dry_run {
        let config = build_train_config_from_args(&args)?;
        let result = dry_run::run(&config)
            .context("dry-run analysis failed")?;
        print_dry_run_table(&result, &config);
        return Ok(());
    }

    // ── path 2: legacy Python subprocess ─────────────────────────────────────
    //
    // Why before the native path: callers who explicitly pass --custom-script
    // or who have a --config YAML intend the Python path. Routing them to the
    // native Candle path would silently change behaviour.
    if args.custom_script.is_some() || args.config.is_some() {
        return run_legacy_path(&args, &mode).await;
    }

    // ── path 3: native Candle training ────────────────────────────────────────
    run_native_path(args, mode).await
}

// ── legacy Python path ────────────────────────────────────────────────────────

async fn run_legacy_path(
    args: &TrainArgs,
    mode: &gwenland_core::engine::GwenMode,
) -> Result<()> {
    // Deprecation notice — stderr only, never stdout.
    // The user sees this once per invocation; CI pipelines can grep for it.
    if args.custom_script.is_some() {
        eprintln!("⚠  --custom-script is legacy mode.");
        eprintln!("   The native Rust training path is now the default.");
        eprintln!("   Remove --custom-script to use Candle LoRA training.");
        eprintln!("   See: gwen train --help");
    }

    // Load YAML config — required for the Python path.
    let config_path = args.config.as_ref().context(
        "the legacy Python path requires a config YAML (-c / --config)",
    )?;
    let mut config = TrainConfig::from_yaml(config_path)?;

    let overrides = TrainOverrides {
        model:   args.model.clone(),
        dataset: args.dataset.clone(),
        output:  args.output.clone(),
        name:    args.name.clone(),
    };
    config.apply_overrides(&overrides);

    // Interactive TTY + not verbose → launch TUI panel over the subprocess.
    // Why atty::is: GwenMode::new already sets non_interactive when stdout is
    // piped, but ratatui::init() panics if the terminal is a raw pipe. The
    // atty check is the last safety net.
    let use_tui = !args.verbose
        && mode.is_tui()
        && atty::is(atty::Stream::Stdout);

    if use_tui {
        let (child, script) =
            preflight_and_spawn(&config, false, args.custom_script.as_deref()).await?;
        let (rx, pid) = events_from_child(child)?;
        run_train_tui(rx, pid, &config, script)?;
    } else {
        run_train_with_opts(
            &config,
            false, // dry_run already handled above
            args.verbose,
            mode.json,
            args.custom_script.as_deref(),
        )
        .await?;
    }

    Ok(())
}

// ── native Candle training path ───────────────────────────────────────────────

/// Why a separate async fn:
/// The native path is the only one that needs a `VarMap` alive for the whole
/// run. Keeping it in its own scope makes the lifetime explicit and prevents
/// accidental moves into closures.
///
/// Why all candle/hf_hub/tokenizers calls are in gwen-core's native_runner:
/// Those crates are direct deps of gwen-core, not gwen-tui. gwen-tui calls one
/// function (`native_runner::run_native`) and handles only the TUI wiring.
/// This keeps the crate boundary clean and avoids adding ML crates to the TUI
/// binary unconditionally.
async fn run_native_path(
    args: TrainArgs,
    mode: gwenland_core::engine::GwenMode,
) -> Result<()> {
    let model_id = args.model.clone().context(
        "native training requires --model / -m <HF_MODEL_ID>",
    )?;
    let dataset_path = args.dataset.clone().context(
        "native training requires --dataset / -d <PATH>",
    )?;
    let output = args
        .output
        .clone()
        .unwrap_or_else(|| PathBuf::from("./gwen-output"));

    let mut config = NewTrainConfig::default();
    config.model_id     = model_id;
    config.dataset_path = dataset_path;
    config.output_path  = output;
    if let Some(e) = args.epochs { config.epochs = e; }
    if let Some(l) = args.lr    { config.lr      = l; }

    // Why TUI after config built, not after model loaded:
    // native_runner::run_native is sync and blocks until training finishes.
    // We need to decide the TUI vs headless path before blocking. All
    // fallible setup (tokenizer fetch, dataset load, model init) is inside
    // run_native; if it fails the terminal is never touched by ratatui.
    let use_tui = !args.verbose && mode.is_tui() && atty::is(atty::Stream::Stdout);

    if use_tui {
        let tui_config = synthetic_train_config(&config);

        // ── anonymous pipe bridge ──────────────────────────────────────────────
        //
        // Why std::sync::mpsc instead of an OS pipe:
        // `run_native` emits JSON strings via Sender<String>. The TUI reads
        // Receiver<TrainEvent>. `events_from_native_rx` spawns a converter thread
        // that parses the strings into events — zero overhead, no OS pipe handles,
        // no platform-specific code.
        //
        // Why std::thread not tokio::spawn: native_runner::run_native is
        // synchronous; blocking inside a tokio task would stall the runtime.
        let (progress_tx, progress_rx) = std::sync::mpsc::channel::<String>();
        let config_for_thread = config.clone();
        let (result_tx, result_rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = result_tx.send(native_runner::run_native(&config_for_thread, progress_tx));
        });

        // Convert the raw JSON string channel into a parsed TrainEvent channel
        // that run_train_tui can consume directly.
        let event_rx = events_from_native_rx(progress_rx);

        let dummy_script = tempfile::NamedTempFile::new()
            .context("failed to create dummy tempfile for TUI")?;

        // Pass None for pid — the training thread cannot be suspended via
        // SIGSTOP without deadlocking the AdamW optimizer state.
        let _ = run_train_tui(event_rx, None, &tui_config, dummy_script);

        match result_rx.recv() {
            Ok(Ok(result)) => eprintln!(
                "[train] done — {} steps, final loss {:.4}, elapsed {:?}",
                result.total_steps, result.final_loss, result.elapsed
            ),
            Ok(Err(e)) => return Err(e.context("training loop failed")),
            Err(_) => eprintln!("[train] training thread disconnected"),
        }
    } else {
        // Headless: run_native blocks and emits JSON progress to stdout.
        // Provide a disconnected sender — training still proceeds; the
        // `tx.send().ok()` calls inside TrainingLoop silently discard the error.
        let (progress_tx, _progress_rx) = std::sync::mpsc::channel::<String>();
        let result = native_runner::run_native(&config, progress_tx)
            .context("native training failed")?;

        eprintln!(
            "[train] done — {} steps, final loss {:.4}, elapsed {:?}",
            result.total_steps, result.final_loss, result.elapsed
        );
    }

    Ok(())
}

// ── dry-run table ─────────────────────────────────────────────────────────────

/// Print the dry-run analysis as a formatted table to stdout.
///
/// Layout matches the existing `print_dry_run_report` style in runner.rs so
/// both the Python-path and native-path dry-run outputs are visually consistent.
///
/// Why stdout and not stderr:
/// The dry-run table is the primary output of `gwen train --dry-run`. It is
/// designed to be read by humans and piped through `less`. Diagnostic warnings
/// that come from DryRunResult.warnings are appended at the bottom where they
/// don't interrupt the table parse.
fn print_dry_run_table(result: &DryRunResult, config: &TrainConfig) {
    let divider = "━".repeat(52);
    let line    = "─".repeat(46);

    println!("\ngwen train --dry-run  (native Candle path)");
    println!("{}", divider);

    // ── Dataset ───────────────────────────────────────────────────────────────
    println!("  {:<22} {}", "Dataset samples",   result.dataset_samples);
    println!("  {:<22} {} tokens", "Avg token length", result.avg_token_length);
    println!("  {:<22} {} tokens", "Total tokens",     result.total_tokens);
    println!();

    // ── Model ─────────────────────────────────────────────────────────────────
    let model_short = shorten(&config.model, 28);
    println!("  {:<22} {}", "Model", model_short);
    println!(
        "  {:<22} {:.2}B params",
        "Model parameters",
        result.model_params as f64 / 1e9
    );
    println!(
        "  {:<22} {} trainable",
        "LoRA parameters",
        fmt_params(result.lora_params)
    );
    println!();

    // ── VRAM breakdown ────────────────────────────────────────────────────────
    println!("  VRAM estimate");
    println!("  {}", line);
    println!(
        "  {:<28} {:>8.1} MB",
        "Base model (bf16)",
        result.vram.model_mb
    );
    println!(
        "  {:<28} {:>8.1} MB",
        "Activations",
        result.vram.activations_mb
    );
    println!(
        "  {:<28} {:>8.1} MB",
        "Optimizer states (AdamW)",
        result.vram.optimizer_mb
    );
    println!("  {}", line);
    println!(
        "  {:<28} {:>8.1} MB  (×1.2 buffer)",
        "Total",
        result.vram.total_mb
    );
    println!();

    // ── Time estimates ────────────────────────────────────────────────────────
    println!("  Training time estimates  (epochs={})", config.epochs);
    println!("  {}", line);
    println!(
        "  {:<28} {}",
        "CPU",
        fmt_duration(result.time_cpu)
    );
    println!(
        "  {:<28} {}",
        "NVIDIA T4",
        fmt_duration(result.time_t4)
    );
    println!(
        "  {:<28} {}",
        "NVIDIA RTX 3080",
        fmt_duration(result.time_rtx3080)
    );
    println!("{}", divider);

    // ── Result ────────────────────────────────────────────────────────────────
    if result.warnings.is_empty() {
        println!("  ✓ Ready to train — remove --dry-run to start.");
    } else {
        println!("  ⚠  {} warning(s):", result.warnings.len());
        for w in &result.warnings {
            println!("     • {}", w);
        }
        println!();
        println!("  Training may proceed, but review warnings above.");
    }

    println!();
}

// ── config helpers ────────────────────────────────────────────────────────────

/// Build a `TrainConfig` from CLI flags for the dry-run path.
///
/// Why TrainConfig and not NewTrainConfig:
/// `dry_run::run` takes `&TrainConfig` because it needs the YAML-style flat
/// fields (lora_r, lora_target as CSV string, etc.) to call the existing
/// `build_lora_config` helper. Using the same type avoids a parallel code path.
fn build_train_config_from_args(args: &TrainArgs) -> Result<TrainConfig> {
    let model = args.model.clone().context(
        "--dry-run requires --model/-m <HF_MODEL_ID>",
    )?;
    let dataset = args.dataset.clone().context(
        "--dry-run requires --dataset/-d <PATH>",
    )?;
    let output = args
        .output
        .clone()
        .unwrap_or_else(|| PathBuf::from("./gwen-output"));

    let defaults = NewTrainConfig::default();
    let lora_defaults = LoraConfig::default();

    Ok(TrainConfig {
        model,
        dataset,
        output,
        name: args.name.clone(),
        epochs:        args.epochs.map(|e| e as u32).unwrap_or(defaults.epochs as u32),
        batch_size:    defaults.batch_size as u32,
        grad_accum:    defaults.grad_accum as u32,
        learning_rate: args.lr.unwrap_or(defaults.lr),
        max_seq_len:   1024,
        lora_r:        lora_defaults.r as u32,
        lora_alpha:    lora_defaults.alpha as u32,
        lora_dropout:  lora_defaults.dropout,
        lora_target:   lora_defaults.target_modules.join(","),
        qlora:         false,
        optimizer:     "adamw".to_string(),
        scheduler:     "cosine".to_string(),
        fp16:          false,
        weight_decay:  0.01,
    })
}

/// Minimal `TrainConfig` for TUI state initialisation on the native path.
///
/// Why: TuiState::new reads config.epochs and config.learning_rate. We fill
/// those from NewTrainConfig. All other fields are harmless defaults.
fn synthetic_train_config(config: &NewTrainConfig) -> TrainConfig {
    TrainConfig {
        model:         config.model_id.clone(),
        dataset:       config.dataset_path.clone(),
        output:        config.output_path.clone(),
        name:          None,
        epochs:        config.epochs as u32,
        batch_size:    config.batch_size as u32,
        grad_accum:    config.grad_accum as u32,
        learning_rate: config.lr,
        max_seq_len:   1024,
        lora_r:        config.lora.r as u32,
        lora_alpha:    config.lora.alpha as u32,
        lora_dropout:  config.lora.dropout,
        lora_target:   config.lora.target_modules.join(","),
        qlora:         false,
        optimizer:     "adamw".to_string(),
        scheduler:     "cosine".to_string(),
        fp16:          false,
        weight_decay:  0.01,
    }
}

// ── display helpers ───────────────────────────────────────────────────────────

fn shorten(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("…{}", &s[s.len() - max + 1..])
    }
}

fn fmt_params(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1e6)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1e3)
    } else {
        n.to_string()
    }
}

fn fmt_duration(d: Duration) -> String {
    let secs = d.as_secs();
    if secs < 60 {
        format!("{}s", secs)
    } else if secs < 3600 {
        format!("{}m {:02}s", secs / 60, secs % 60)
    } else {
        format!("{}h {:02}m", secs / 3600, (secs % 3600) / 60)
    }
}
