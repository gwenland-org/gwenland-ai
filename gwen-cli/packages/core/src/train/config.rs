// @INFO: TrainConfig — loaded from YAML, CLI flags override after parse.
// @EDITABLE: Add hyperparams here; update to_python_args() accordingly.
// @DANGER: apply_overrides() must be called after from_yaml() before validate(). Order matters.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

// ── LoRA config ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LoraConfig {
    pub r: usize,
    pub alpha: f32,
    pub dropout: f64,
    pub target_modules: Vec<String>,
}

impl Default for LoraConfig {
    fn default() -> Self {
        Self {
            r: 8,
            alpha: 16.0,
            dropout: 0.05,
            target_modules: vec!["q_proj".to_string(), "v_proj".to_string()],
        }
    }
}

// ── TrainResult ──────────────────────────────────────────────────────────────
//
// Lives here (not in training_loop.rs) so that native_runner.rs can reference
// it without the `candle` feature gate. training_loop.rs re-exports it from
// this location.

/// Summary returned after a completed native training run.
#[derive(Debug, Clone)]
pub struct TrainResult {
    /// Mean cross-entropy loss over the final accumulation window.
    pub final_loss: f32,
    /// Total optimiser steps taken (= total_batches / grad_accum).
    pub total_steps: usize,
    /// Wall-clock time from `TrainingLoop::run()` entry to return.
    pub elapsed: std::time::Duration,
}

// ── New TrainConfig (candle-native, distinct from the YAML-backed one below) ─

#[derive(Debug, Clone)]
pub struct NewTrainConfig {
    pub model_id: String,
    pub dataset_path: PathBuf,
    pub epochs: usize,
    pub batch_size: usize,
    pub grad_accum: usize,
    pub lr: f64,
    pub max_grad_norm: f64,
    pub lora: LoraConfig,
    pub dry_run: bool,
    /// Hard cap on optimiser steps. `Some(n)` stops after `n` steps (used by the
    /// native 1-step dry-run); `None` runs the full schedule.
    pub max_steps: Option<usize>,
    pub output_path: PathBuf,
    pub custom_script: Option<PathBuf>,
}

impl Default for NewTrainConfig {
    fn default() -> Self {
        Self {
            model_id: String::new(),
            dataset_path: PathBuf::new(),
            epochs: 3,
            batch_size: 1,
            grad_accum: 16,
            lr: 1e-4,
            max_grad_norm: 1.0,
            lora: LoraConfig::default(),
            dry_run: false,
            max_steps: None,
            output_path: PathBuf::new(),
            custom_script: None,
        }
    }
}

// ── Cli stub for From<Cli> — wired to the real TrainArgs in gwen-tui later ──

/// Mirrors the fields of `gwen_tui::commands::train::TrainArgs` that are
/// relevant to NewTrainConfig. Kept here so the impl compiles without a
/// circular dependency; the TUI crate will convert its TrainArgs → Cli before
/// calling NewTrainConfig::from(cli).
pub struct Cli {
    pub model: Option<String>,
    pub dataset: Option<PathBuf>,
    pub output: Option<PathBuf>,
    pub dry_run: bool,
    pub custom_script: Option<PathBuf>,
}

impl From<Cli> for NewTrainConfig {
    fn from(cli: Cli) -> Self {
        let mut cfg = NewTrainConfig::default();
        if let Some(m) = cli.model      { cfg.model_id     = m; }
        if let Some(d) = cli.dataset    { cfg.dataset_path = d; }
        if let Some(o) = cli.output     { cfg.output_path  = o; }
        cfg.dry_run       = cli.dry_run;
        cfg.custom_script = cli.custom_script;
        cfg
    }
}

// ── default fns (required by serde) ─────────────────────────────────────────

fn default_epochs() -> u32 { 3 }
fn default_batch() -> u32 { 1 }
fn default_grad_accum() -> u32 { 16 }
fn default_lr() -> f64 { 1e-4 }
fn default_seq_len() -> u32 { 1024 }
fn default_lora_r() -> u32 { 8 }
fn default_lora_alpha() -> u32 { 16 }
fn default_lora_dropout() -> f64 { 0.05 }
fn default_lora_target() -> String { "q_proj,v_proj".to_string() }
fn default_true() -> bool { true }
fn default_optimizer() -> String { "adamw_8bit".to_string() }
fn default_scheduler() -> String { "cosine".to_string() }
fn default_weight_decay() -> f64 { 0.01 }
fn default_max_grad_norm() -> f64 { 1.0 }

// ── struct ───────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize, Serialize)]
pub struct TrainConfig {
    pub model: String,
    pub dataset: PathBuf,
    pub output: PathBuf,
    pub name: Option<String>,

    #[serde(default = "default_epochs")]
    pub epochs: u32,
    #[serde(default = "default_batch")]
    pub batch_size: u32,
    #[serde(default = "default_grad_accum")]
    pub grad_accum: u32,
    #[serde(default = "default_lr")]
    pub learning_rate: f64,
    #[serde(default = "default_seq_len")]
    pub max_seq_len: u32,
    #[serde(default = "default_lora_r")]
    pub lora_r: u32,
    #[serde(default = "default_lora_alpha")]
    pub lora_alpha: u32,
    #[serde(default = "default_lora_dropout")]
    pub lora_dropout: f64,
    #[serde(default = "default_lora_target")]
    pub lora_target: String,
    #[serde(default = "default_true")]
    pub qlora: bool,
    #[serde(default = "default_optimizer")]
    pub optimizer: String,
    #[serde(default = "default_scheduler")]
    pub scheduler: String,
    #[serde(default = "default_true")]
    pub fp16: bool,
    #[serde(default = "default_weight_decay")]
    pub weight_decay: f64,
    #[serde(default = "default_max_grad_norm")]
    pub max_grad_norm: f64,
}

impl TrainConfig {
    pub fn from_yaml(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("cannot read config: {}", path.display()))?;
        serde_yaml::from_str(&raw)
            .with_context(|| format!("invalid YAML in config: {}", path.display()))
    }

    /// Apply CLI flag overrides. Called after from_yaml(), before validate().
    pub fn apply_overrides(&mut self, args: &crate::train::runner::TrainOverrides) {
        if let Some(m) = &args.model { self.model = m.clone(); }
        if let Some(d) = &args.dataset { self.dataset = d.clone(); }
        if let Some(o) = &args.output { self.output = o.clone(); }
        if let Some(n) = &args.name { self.name = Some(n.clone()); }
    }

    /// Check that the required string fields are not empty.
    pub fn validate(&self) -> Result<()> {
        if self.model.trim().is_empty() {
            bail!("config.model is required (pass -m or set `model:` in config YAML)");
        }
        if self.dataset.as_os_str().is_empty() {
            bail!("config.dataset is required (pass -d or set `dataset:` in config YAML)");
        }
        if self.output.as_os_str().is_empty() {
            bail!("config.output is required (pass -o or set `output:` in config YAML)");
        }
        Ok(())
    }

    /// Convert config into ordered CLI args for base_train.py.
    pub fn to_python_args(&self) -> Vec<String> {
        let mut args = vec![
            "--model-name".into(), self.model.clone(),
            "--dataset".into(), self.dataset.display().to_string(),
            "--output-dir".into(), self.output.display().to_string(),
            "--epochs".into(), self.epochs.to_string(),
            "--batch-size".into(), self.batch_size.to_string(),
            "--grad-accum".into(), self.grad_accum.to_string(),
            "--learning-rate".into(), self.learning_rate.to_string(),
            "--max-seq-len".into(), self.max_seq_len.to_string(),
            "--lora-r".into(), self.lora_r.to_string(),
            "--lora-alpha".into(), self.lora_alpha.to_string(),
            "--lora-dropout".into(), self.lora_dropout.to_string(),
            "--lora-target".into(), self.lora_target.clone(),
            "--optimizer".into(), self.optimizer.clone(),
            "--scheduler".into(), self.scheduler.clone(),
            "--weight-decay".into(), self.weight_decay.to_string(),
        ];
        // argparse flags (store_true): only pass when true
        if self.qlora { args.push("--qlora".into()); }
        if self.fp16  { args.push("--fp16".into()); }
        args
    }
}
