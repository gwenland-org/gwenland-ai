// @INFO: CLI entry point for `gwen train` and its subcommands.
//
//   Dispatch tree:
//
//   gwen train export-adapter  → export LoRA adapter from a checkpoint
//   gwen train merge-adapter   → merge adapter SafeTensors into a GGUF base
//   gwen train [flags]         → fine-tune a model (LoRA/QLoRA) — default path
//
//   The default `gwen train` path has three internal branches:
//   1. --dry-run  → estimate VRAM/time from HF metadata, print table, exit 0.
//   2. --custom-script → legacy Python subprocess (deprecated).
//   3. (default) native Candle path → TrainingLoop with optional TUI panel.
//
// @EDITABLE: Add --resume, --checkpoint, --grad-accum, --lora-r flags to TrainArgs.
//            They map into NewTrainConfig and require no structural changes.
// @DANGER:   The native path creates a VarMap that must stay alive for the
//            entire training run. Do not move it into a closure or sub-scope.

use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context, Result};
use clap::{Args, Subcommand};
use gwenland_core::train::config::{LoraConfig, NewTrainConfig, TrainConfig};
use gwenland_core::train::dry_run::{self, DryRunResult};
use gwenland_core::train::lora_cli;
use gwenland_core::train::native_runner;
use gwenland_core::train::runner::{preflight_and_spawn, run_train_with_opts, TrainOverrides};

use crate::tui::train_panel::{events_from_child, events_from_native_rx, run_train_tui};

// ── Top-level TrainArgs ───────────────────────────────────────────────────────

#[derive(Args, Debug)]
#[command(
    about = "Train a LoRA adapter or run the full train→export→merge pipeline",
    after_help = "One-shot workflow:\n  gwen train --auto-merge --base-model ./qwen3.gguf --dataset ./data.jsonl\n\nThis trains, exports the adapter, and merges into the base model automatically."
)]
pub struct TrainArgs {
    #[command(subcommand)]
    pub subcommand: Option<TrainSubcommand>,

    // ── Training flags (used when no subcommand is given) ─────────────────────

    #[arg(short = 'c', long, value_name = "CONFIG", global = false,
          help = "Training config YAML (legacy path; use native flags instead)")]
    pub config: Option<PathBuf>,

    #[arg(short = 'm', long, value_name = "HF_MODEL_ID", global = false,
          help = "HuggingFace model ID to fine-tune (e.g. mistralai/Mistral-7B-v0.1)")]
    pub model: Option<String>,

    #[arg(short = 'd', long, value_name = "PATH", global = false,
          help = "Path to JSONL training dataset")]
    pub dataset: Option<PathBuf>,

    #[arg(long, value_name = "N", global = false,
          help = "Number of training epochs (default: 3)")]
    pub epochs: Option<usize>,

    #[arg(long, value_name = "F", global = false,
          help = "Learning rate (default: 1e-4)")]
    pub lr: Option<f64>,

    #[arg(short = 'o', long, value_name = "PATH", global = false,
          help = "Output directory for checkpoints and adapter (default: ./gwen-output)")]
    pub output: Option<PathBuf>,

    #[arg(long, global = false,
          help = "Estimate VRAM and training time without running training")]
    pub dry_run: bool,

    #[arg(short = 'v', long, global = false,
          help = "Stream training logs to stdout instead of showing the TUI panel")]
    pub verbose: bool,

    #[arg(long, value_name = "SCRIPT", global = false,
          help = "Custom Python training script [deprecated — use native path]")]
    pub custom_script: Option<PathBuf>,

    #[arg(short = 'n', long, value_name = "NAME", global = false,
          help = "Override the model display name stored in the local registry")]
    pub name: Option<String>,

    // ── --auto-merge flags ────────────────────────────────────────────────────

    #[arg(long, global = false,
          help = "After training, automatically export and merge adapter into base model")]
    pub auto_merge: bool,

    #[arg(long, value_name = "PATH", global = false,
          help = "Base GGUF model path for --auto-merge (required when --auto-merge is set)")]
    pub base_model: Option<PathBuf>,
}

// ── Subcommands ───────────────────────────────────────────────────────────────

#[derive(Subcommand, Debug)]
pub enum TrainSubcommand {
    /// Export a trained LoRA checkpoint to SafeTensors adapter format
    ExportAdapter(ExportAdapterArgs),

    /// Merge a LoRA adapter into a base GGUF model, producing a new merged GGUF
    MergeAdapter(MergeAdapterArgs),
}

// ── export-adapter args ───────────────────────────────────────────────────────

#[derive(Args, Debug)]
#[command(
    about = "Export a trained LoRA checkpoint to SafeTensors adapter format",
    after_help = "Examples:\n  gwen train export-adapter --checkpoint ./checkpoints/epoch3.st --output ./adapter.st\n  gwen train export-adapter --checkpoint ./ckpt.st --output ./out.st --dry-run"
)]
pub struct ExportAdapterArgs {
    #[arg(long, value_name = "PATH", required = true,
          help = "Path to candle .safetensors training checkpoint")]
    pub checkpoint: PathBuf,

    #[arg(long, value_name = "PATH", required = true,
          help = "Output path for the exported adapter file (.safetensors)")]
    pub output: PathBuf,

    #[arg(long,
          help = "Validate checkpoint and count adapters without writing output")]
    pub dry_run: bool,
}

// ── merge-adapter args ────────────────────────────────────────────────────────

#[derive(Args, Debug)]
#[command(
    about = "Merge a LoRA adapter into a base GGUF model, producing a new merged GGUF",
    after_help = "Examples:\n  gwen train merge-adapter --base ./qwen3.gguf --adapter ./adapter.st --output ./merged.gguf\n  gwen train merge-adapter --base ./model.gguf --adapter ./adapter.st --output ./out.gguf --memory-budget 4294967296\n  gwen train merge-adapter --base ./model.gguf --adapter ./adapter.st --output ./out.gguf --dry-run"
)]
pub struct MergeAdapterArgs {
    #[arg(long, value_name = "PATH", required = true,
          help = "Base GGUF model file to merge into (must be Q8_0-quantized)")]
    pub base: PathBuf,

    #[arg(long, value_name = "PATH", required = true,
          help = "LoRA adapter SafeTensors file (from export-adapter)")]
    pub adapter: PathBuf,

    #[arg(long, value_name = "PATH", required = true,
          help = "Output path for the merged GGUF model (overwritten if exists)")]
    pub output: PathBuf,

    #[arg(long, value_name = "BYTES",
          help = "Memory budget in bytes for merge operation (default: 2GB = 2147483648)")]
    pub memory_budget: Option<usize>,

    #[arg(long,
          help = "Validate all paths without executing the merge")]
    pub dry_run: bool,
}

// ── public entry point ────────────────────────────────────────────────────────

/// Top-level async entry point called from main.rs.
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
    // Route to subcommand first — subcommands are mutually exclusive with
    // training flags and each has its own --dry-run semantics.
    if let Some(sub) = args.subcommand {
        return match sub {
            TrainSubcommand::ExportAdapter(a) => run_export_adapter(a),
            TrainSubcommand::MergeAdapter(a) => run_merge_adapter(a),
        };
    }

    // No subcommand: normal training flow.
    let dry_run = args.dry_run || mode.dry_run;

    // ── path 1: config-driven flow (incl. native local-GGUF dry-run) ─────────
    // A --config run carries the model in the YAML, so its dry-run must go
    // through run_train_with_opts (which performs a real native 1-step pass for
    // local GGUF models). Only the no-config dry-run uses the HF estimation
    // table below.
    if args.custom_script.is_some() || args.config.is_some() {
        return run_legacy_path(&args, &mode).await;
    }

    // ── path 2: dry-run estimation (no config; needs --model/--dataset) ──────
    if dry_run {
        let config = build_train_config_from_args(&args)?;
        let result = dry_run::run(&config)
            .context("dry-run analysis failed")?;
        print_dry_run_table(&result, &config);
        return Ok(());
    }

    // ── path 3: native Candle training ────────────────────────────────────────
    run_native_path(args, mode).await
}

// ── Task 8.1: export-adapter handler ─────────────────────────────────────────

/// Handle `gwen train export-adapter`.
///
/// @INFO All ML work (VarMap load, adapter extraction, SafeTensors write) is
/// delegated to `gwenland_core::train::lora_cli::export_adapter` so that the
/// tui crate never needs to import candle or ML types directly.
fn run_export_adapter(args: ExportAdapterArgs) -> Result<()> {
    if args.dry_run {
        eprintln!("[export-adapter] dry-run: validating checkpoint...");
    } else {
        eprintln!(
            "[export-adapter] exporting from {} → {}",
            args.checkpoint.display(),
            args.output.display()
        );
    }

    match lora_cli::export_adapter(&args.checkpoint, &args.output, args.dry_run) {
        Ok(count) if args.dry_run => {
            eprintln!("[export-adapter] validation passed: {count} adapter pair(s) found");
            Ok(())
        }
        Ok(count) => {
            eprintln!(
                "[export-adapter] adapter exported: {} ({count} pair(s))",
                args.output.display()
            );
            Ok(())
        }
        Err(e) => {
            eprintln!("[export-adapter] error: {e}");
            std::process::exit(1);
        }
    }
}

// ── Task 8.2: merge-adapter handler ──────────────────────────────────────────

/// Handle `gwen train merge-adapter`.
///
/// @DANGER The output file is overwritten without prompting. The warning is
/// printed to stderr so users see it even in non-interactive sessions.
/// @INFO Merge progress lines (`[gwen-merge] merged layer: ...`) are emitted
/// by the core pipeline and arrive on stderr automatically.
fn run_merge_adapter(args: MergeAdapterArgs) -> Result<()> {
    // Overwrite warning — always shown, not gated on --yes.
    if args.output.exists() && !args.dry_run {
        eprintln!(
            "[merge-adapter] warning: output file already exists and will be overwritten: {}",
            args.output.display()
        );
    }

    if args.dry_run {
        eprintln!("[merge-adapter] dry-run: validating paths...");
    } else {
        eprintln!(
            "[merge-adapter] merging {} into {}...",
            args.adapter.display(),
            args.base.display()
        );
    }

    match lora_cli::merge_adapter(
        &args.base,
        &args.adapter,
        &args.output,
        args.memory_budget,
        args.dry_run,
    ) {
        Ok(()) if args.dry_run => {
            eprintln!("[merge-adapter] validation passed");
            Ok(())
        }
        Ok(()) => {
            eprintln!("[merge-adapter] merged model: {}", args.output.display());
            Ok(())
        }
        Err(e) => {
            eprintln!("[merge-adapter] error: {e}");
            std::process::exit(1);
        }
    }
}

// ── legacy Python path ────────────────────────────────────────────────────────

async fn run_legacy_path(
    args: &TrainArgs,
    mode: &gwenland_core::engine::GwenMode,
) -> Result<()> {
    if args.custom_script.is_some() {
        eprintln!("⚠  --custom-script is legacy mode.");
        eprintln!("   The native Rust training path is now the default.");
        eprintln!("   Remove --custom-script to use Candle LoRA training.");
        eprintln!("   See: gwen train --help");
    }

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

    let dry_run = args.dry_run || mode.dry_run;

    // Dry-run never enters the interactive TUI — it runs a single native step
    // (for local GGUF) or the estimation report and exits.
    let use_tui = !dry_run
        && !args.verbose
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
            dry_run,
            args.verbose,
            mode.json,
            args.custom_script.as_deref(),
        )
        .await?;
    }

    Ok(())
}

// ── native Candle training path ───────────────────────────────────────────────

/// Run native Candle LoRA training, with optional --auto-merge at the end.
///
/// @INFO VarMap stays alive for the full training run inside native_runner.
/// The auto-merge step runs after the training thread joins successfully.
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

    // Validate auto-merge args before starting training so the user sees the
    // error immediately rather than after a potentially long training run.
    if args.auto_merge && args.base_model.is_none() {
        anyhow::bail!("--base-model is required when --auto-merge is set");
    }

    let mut config = NewTrainConfig::default();
    config.model_id     = model_id;
    config.dataset_path = dataset_path;
    config.output_path  = output.clone();
    if let Some(e) = args.epochs { config.epochs = e; }
    if let Some(l) = args.lr    { config.lr      = l; }

    let use_tui = !args.verbose && mode.is_tui() && atty::is(atty::Stream::Stdout);

    let train_result = if use_tui {
        let tui_config = synthetic_train_config(&config);
        let (progress_tx, progress_rx) = std::sync::mpsc::channel::<String>();
        let config_for_thread = config.clone();
        let (result_tx, result_rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = result_tx.send(native_runner::run_native(&config_for_thread, progress_tx));
        });

        let event_rx = events_from_native_rx(progress_rx);
        let dummy_script = tempfile::NamedTempFile::new()
            .context("failed to create dummy tempfile for TUI")?;
        let _ = run_train_tui(event_rx, None, &tui_config, dummy_script);

        match result_rx.recv() {
            Ok(Ok(r)) => {
                eprintln!(
                    "[train] done — {} steps, final loss {:.4}, elapsed {:?}",
                    r.total_steps, r.final_loss, r.elapsed
                );
                Ok(r)
            }
            Ok(Err(e)) => Err(e.context("training loop failed")),
            Err(_) => {
                eprintln!("[train] training thread disconnected");
                return Ok(());
            }
        }
    } else {
        let (progress_tx, _progress_rx) = std::sync::mpsc::channel::<String>();
        let r = native_runner::run_native(&config, progress_tx)
            .context("native training failed")?;
        eprintln!(
            "[train] done — {} steps, final loss {:.4}, elapsed {:?}",
            r.total_steps, r.final_loss, r.elapsed
        );
        Ok(r)
    }?;

    // ── Task 8.3: --auto-merge ────────────────────────────────────────────────
    if args.auto_merge {
        // base_model is validated non-None above.
        let base_model = args.base_model.unwrap();

        // The checkpoint is the last SafeTensors file written by the training
        // loop. native_runner writes to {output_path}/checkpoint.safetensors.
        let checkpoint_path = output.join("checkpoint.safetensors");
        let adapter_path    = output.join("adapter.safetensors");

        // Derive merged output path: {base_model_stem}.lora_merged.gguf
        let merged_path = {
            let stem = base_model
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("model");
            base_model
                .parent()
                .unwrap_or_else(|| std::path::Path::new("."))
                .join(format!("{stem}.lora_merged.gguf"))
        };

        eprintln!("[auto-merge] step 1/2: exporting adapter...");
        if let Err(e) = lora_cli::export_adapter(&checkpoint_path, &adapter_path, false) {
            eprintln!("[auto-merge] export failed: {e}");
            std::process::exit(1);
        }
        eprintln!("[auto-merge] adapter: {}", adapter_path.display());

        eprintln!("[auto-merge] step 2/2: merging into base model...");
        if let Err(e) = lora_cli::merge_adapter(&base_model, &adapter_path, &merged_path, None, false) {
            eprintln!("[auto-merge] merge failed: {e}");
            std::process::exit(1);
        }
        eprintln!("[auto-merge] complete: {}", merged_path.display());
    }

    Ok(())
}

// ── dry-run table ─────────────────────────────────────────────────────────────

fn print_dry_run_table(result: &DryRunResult, config: &TrainConfig) {
    let divider = "━".repeat(52);
    let line    = "─".repeat(46);

    println!("\ngwen train --dry-run  (native Candle path)");
    println!("{}", divider);

    println!("  {:<22} {}", "Dataset samples",   result.dataset_samples);
    println!("  {:<22} {} tokens", "Avg token length", result.avg_token_length);
    println!("  {:<22} {} tokens", "Total tokens",     result.total_tokens);
    println!();

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

    println!("  VRAM estimate");
    println!("  {}", line);
    println!("  {:<28} {:>8.1} MB", "Base model (bf16)", result.vram.model_mb);
    println!("  {:<28} {:>8.1} MB", "Activations", result.vram.activations_mb);
    println!("  {:<28} {:>8.1} MB", "Optimizer states (AdamW)", result.vram.optimizer_mb);
    println!("  {}", line);
    println!("  {:<28} {:>8.1} MB  (×1.2 buffer)", "Total", result.vram.total_mb);
    println!();

    println!("  Training time estimates  (epochs={})", config.epochs);
    println!("  {}", line);
    println!("  {:<28} {}", "CPU", fmt_duration(result.time_cpu));
    println!("  {:<28} {}", "NVIDIA T4", fmt_duration(result.time_t4));
    println!("  {:<28} {}", "NVIDIA RTX 3080", fmt_duration(result.time_rtx3080));
    println!("{}", divider);

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
        max_grad_norm: 1.0,
    })
}

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
        max_grad_norm: config.max_grad_norm,
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
