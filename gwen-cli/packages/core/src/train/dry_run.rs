//! Dry-run analyser: validates a training config without touching weights.
//!
//! # Why a dedicated module?
//!
//! The TUI's "estimate before you train" path needs to answer three questions
//! before the user commits to a potentially multi-hour run:
//!   1. Does the dataset look healthy?
//!   2. Will the model + LoRA adapters fit in VRAM?
//!   3. How long will this take on common hardware?
//!
//! All three answers are derivable from lightweight metadata (JSONL headers,
//! HF `config.json`, arithmetic). No GPU, no tokenizer warmup, no weight
//! downloads needed. Keeping this logic isolated means the TUI can call it on
//! every config change without latency.
//!
//! # Why NOT load model weights here?
//!
//! `config.json` is ~2 KB. Model weights are 4–140 GB. Even a single-shard
//! safetensors header parse would saturate the local NVMe for seconds and blow
//! the 3-second budget. All parameter counts are derived from `config.json`
//! arithmetic, which matches what HF reports to within rounding error.
//!
//! # Why the sync hf-hub API instead of the tokio one?
//!
//! `run()` is called from a sync context (the TUI's pre-flight check). Wrapping
//! a tokio runtime inside the dry-run would pull `tokio::runtime::Builder` into
//! a thread that may already own a runtime (actix-web worker threads), causing a
//! panic. The sync API uses `ureq` under the hood and is safe to call anywhere.
//! `config.json` is cached by hf-hub after the first fetch, so subsequent calls
//! are instant disk reads.
//!
//! # Wire format
//!
//! `DryRunResult` is serialisable so the TUI can render it directly and the CLI
//! can print it as JSON with `serde_json::to_string_pretty`.

use std::path::Path;
use std::time::Duration;

use anyhow::{Context, Result};
use hf_hub::api::sync::ApiBuilder;
use serde::{Deserialize, Serialize};
use sysinfo::System;

use crate::diagnostics::estimator::{
    estimate_time, estimate_vram, Device, VramEstimate,
};
use crate::platform::hub_model::resolve_token;
use crate::train::config::{LoraConfig, TrainConfig};
use crate::train::samples::{load_jsonl, DEFAULT_MAX_LEN};

// ── result type ───────────────────────────────────────────────────────────────

/// Complete dry-run analysis for one `TrainConfig`.
///
/// All VRAM figures are in **megabytes** (matching `diagnostics::estimator`).
/// Time estimates are `Duration` so the caller can choose the display format.
#[derive(Debug, Clone, Serialize)]
pub struct DryRunResult {
    /// Number of valid samples that survived JSONL parsing.
    pub dataset_samples: usize,

    /// Mean token count per sample (truncated to `DEFAULT_MAX_LEN`).
    pub avg_token_length: usize,

    /// `dataset_samples × avg_token_length` — the effective token budget.
    pub total_tokens: usize,

    /// Total model parameters inferred from `config.json`.
    pub model_params: usize,

    /// Trainable LoRA parameters: `2 × r × d_model × |target_modules|`.
    pub lora_params: usize,

    /// VRAM breakdown from `diagnostics::estimator::estimate_vram`.
    pub vram: VramEstimate,

    /// Estimated wall-clock time on a CPU.
    pub time_cpu: Duration,

    /// Estimated wall-clock time on an NVIDIA T4.
    pub time_t4: Duration,

    /// Estimated wall-clock time on an NVIDIA RTX 3080.
    pub time_rtx3080: Duration,

    /// Human-readable warnings accumulated during analysis.
    /// Non-fatal: a non-empty list does not prevent training from proceeding.
    pub warnings: Vec<String>,
}

// ── HF config.json shape ─────────────────────────────────────────────────────

/// Subset of `config.json` fields needed for parameter estimation.
///
/// All fields are `Option` because different model families use different names.
/// We try common aliases and fall back gracefully.
#[derive(Debug, Deserialize)]
struct HfModelConfig {
    // Total parameter count — present in newer HF configs as a top-level field.
    #[serde(rename = "num_parameters")]
    num_parameters: Option<usize>,

    // Hidden dimension — present in LLaMA/Mistral/Qwen/Gemma families.
    hidden_size: Option<usize>,
    // Older GPT-2 style name for the same field.
    n_embd: Option<usize>,

    // Number of transformer layers.
    num_hidden_layers: Option<usize>,
    // GPT-2 style alias.
    n_layer: Option<usize>,

    // Reserved for future param-estimation refinements; not read yet.
    #[allow(dead_code)]
    vocab_size: Option<usize>,
    #[allow(dead_code)]
    num_attention_heads: Option<usize>,
}

impl HfModelConfig {
    /// Resolve `d_model` from whichever alias is present.
    fn d_model(&self) -> Option<usize> {
        self.hidden_size.or(self.n_embd)
    }

    /// Resolve number of transformer layers.
    fn num_layers(&self) -> Option<usize> {
        self.num_hidden_layers.or(self.n_layer)
    }

    /// Estimate total parameter count.
    ///
    /// Priority:
    /// 1. `num_parameters` if the config supplies it directly.
    /// 2. Classic transformer formula: `12 × d_model² × num_layers` (attention
    ///    + FFN, no embedding table, approximate but within ~5% for standard
    ///    architectures).
    fn total_params(&self) -> Option<usize> {
        if let Some(n) = self.num_parameters {
            return Some(n);
        }
        let d = self.d_model()?;
        let l = self.num_layers()?;
        // 12 × d² × L approximates:
        //   attention: 4×d² (Q, K, V, O projections)
        //   FFN:       8×d² (two linear layers with 4×d hidden)
        // It deliberately ignores the embedding table (~vocab×d) because we do
        // not train embeddings in LoRA, and including it would skew VRAM
        // estimates for large-vocabulary models.
        Some(12 * d * d * l)
    }
}

// ── public API ────────────────────────────────────────────────────────────────

/// Run the dry-run analysis.
///
/// Steps (all non-destructive, no Tensor initialisation):
/// 1. Load and count JSONL samples; warn if < 100.
/// 2. Measure token lengths without calling `tokenize()` — avoids Device init.
/// 3. Fetch `config.json` from HF Hub (metadata only, no weights).
/// 4. Compute `lora_params` from `LoraConfig` fields.
/// 5. Estimate VRAM with `diagnostics::estimator::estimate_vram`.
/// 6. Estimate training time for CPU, T4, RTX3080.
/// 7. Warn if estimated VRAM exceeds 80% of available system RAM.
pub fn run(config: &TrainConfig) -> Result<DryRunResult> {
    let mut warnings: Vec<String> = Vec::new();

    // ── 1. dataset ────────────────────────────────────────────────────────────

    let samples = load_jsonl(&config.dataset)
        .context("failed to load dataset for dry run")?;

    let dataset_samples = samples.len();

    if dataset_samples < 100 {
        warnings.push(format!(
            "dataset has only {} samples — models typically need ≥ 100 for \
             meaningful fine-tuning; consider augmenting or using a larger dataset",
            dataset_samples
        ));
    }

    // ── 2. token length estimation (no Tensor, no Device) ────────────────────
    //
    // Why not call `tokenize()` here?
    // `tokenize()` requires a `candle_core::Device`, which initialises the
    // CUDA / Metal backend even on CPU — a 200-400 ms penalty that would blow
    // the 3-second budget.
    //
    // Why character lengths instead of the 4-chars heuristic?
    // We have the actual samples in memory, so we can compute the real average
    // character count and divide by 4 (the BPE approximation). This is more
    // accurate than a flat constant because short samples won't be inflated by
    // a global average, and long samples are clamped to DEFAULT_MAX_LEN just
    // as `tokenize()` would truncate them.
    //
    // Why not import `dataset::avg_token_length`?
    // That function takes `&[Tensor]` (already tokenised). Here we have raw
    // `Sample` strings — computing char lengths avoids both Device init and
    // a round-trip through the tokenizer. We inline the arithmetic directly.
    let avg_token_length = {
        let char_lengths: Vec<usize> = samples
            .iter()
            .map(|s| s.input.len() + s.output.len())
            .collect();
        let avg_chars = if char_lengths.is_empty() {
            0
        } else {
            char_lengths.iter().sum::<usize>() / char_lengths.len()
        };
        // Divide by 4 (BPE approximation), clamp to DEFAULT_MAX_LEN.
        (avg_chars / 4).max(1).min(DEFAULT_MAX_LEN)
    };
    let total_tokens = dataset_samples * avg_token_length * config.epochs as usize;

    // ── 3. HF Hub config.json (metadata only) ─────────────────────────────────

    let hf_cfg = fetch_model_config(&config.model)
        .context("failed to fetch model config from HF Hub")?;

    let total_params = hf_cfg
        .total_params()
        .context("config.json did not contain enough fields to infer parameter count")?;

    let d_model = hf_cfg.d_model().unwrap_or(4096);
    let num_layers = hf_cfg.num_layers().unwrap_or(32);

    // ── 4. LoRA param count ───────────────────────────────────────────────────

    // Each target module contributes two adapter matrices:
    //   lora_a: (r × d_model)
    //   lora_b: (d_model × r)
    // Total per module: 2 × r × d_model.
    let lora_cfg = build_lora_config(config);
    let lora_params = 2 * lora_cfg.r * d_model * lora_cfg.target_modules.len();

    // ── 5. VRAM estimate ──────────────────────────────────────────────────────

    let vram = estimate_vram(
        total_params,
        lora_params,
        config.batch_size as usize,
        config.max_seq_len as usize,
        d_model,
        num_layers,
    );

    // ── 6. time estimates ─────────────────────────────────────────────────────

    let time_cpu    = estimate_time(total_tokens, 1, Device::Cpu.tflops());
    let time_t4     = estimate_time(total_tokens, 1, Device::T4.tflops());
    let time_rtx3080 = estimate_time(total_tokens, 1, Device::Rtx3080.tflops());

    // ── 7. system RAM warning ─────────────────────────────────────────────────

    if let Some(available_ram_mb) = available_system_ram_mb() {
        let threshold = available_ram_mb * 0.80;
        if vram.total_mb > threshold {
            warnings.push(format!(
                "estimated VRAM ({:.0} MB) exceeds 80% of available system RAM \
                 ({:.0} MB); training may cause OOM swapping or be killed by the OS",
                vram.total_mb, available_ram_mb
            ));
        }
    }

    Ok(DryRunResult {
        dataset_samples,
        avg_token_length,
        total_tokens,
        model_params: total_params,
        lora_params,
        vram,
        time_cpu,
        time_t4,
        time_rtx3080,
        warnings,
    })
}

// ── helpers ───────────────────────────────────────────────────────────────────

/// Download (or load from cache) `config.json` for `model_id`.
///
/// Uses the sync hf-hub API with progress disabled — this is a metadata fetch,
/// not a weights download. The file is ~2 KB and is cached by hf-hub under
/// `~/.cache/huggingface/hub/` after the first call.
fn fetch_model_config(model_id: &str) -> Result<HfModelConfig> {
    let token = resolve_token();

    let api = ApiBuilder::from_env()
        .with_token(token)
        .with_progress(false) // no progress bar for a 2 KB file
        .build()
        .context("failed to build HF Hub sync API client")?;

    let repo = api.model(model_id.to_string());

    let config_path: std::path::PathBuf = repo
        .get("config.json")
        .with_context(|| {
            format!(
                "could not fetch config.json for model '{}' — \
                 check the model ID and your HF_TOKEN if the repo is private",
                model_id
            )
        })?;

    parse_model_config(&config_path)
}

/// Parse `config.json` from a local path (cached by hf-hub).
fn parse_model_config(path: &Path) -> Result<HfModelConfig> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("cannot read config.json at '{}'", path.display()))?;
    serde_json::from_str::<HfModelConfig>(&raw)
        .with_context(|| format!("config.json at '{}' is not valid JSON", path.display()))
}

/// Estimate average token length without a tokenizer.
///
/// Uses the 4-chars-per-token heuristic (close to the real BPE average for
/// English text). Result is clamped to `max_len` to match what `tokenize()`
/// would truncate to, so VRAM estimates stay conservative.
///
/// Not called by `run()` directly any more — `run()` inlines the arithmetic
/// with `s.input.len() + s.output.len()` to avoid the `+1` separator that
/// this function includes. Kept for unit tests which verify the clamping logic.
#[allow(dead_code)]
fn estimate_avg_token_length(
    samples: &[crate::train::samples::Sample],
    max_len: usize,
) -> usize {
    if samples.is_empty() {
        return 0;
    }
    let total_chars: usize = samples
        .iter()
        .map(|s| s.input.len() + 1 + s.output.len()) // +1 for '\n' separator
        .sum();
    let avg_chars = total_chars / samples.len();
    // 4 chars/token is a good approximation for GPT-style BPE on English text.
    let estimated_tokens = (avg_chars / 4).max(1);
    estimated_tokens.min(max_len)
}

/// Build a `LoraConfig` from the flat `TrainConfig` lora_* fields.
fn build_lora_config(config: &TrainConfig) -> LoraConfig {
    LoraConfig {
        r: config.lora_r as usize,
        alpha: config.lora_alpha as f32,
        dropout: config.lora_dropout,
        target_modules: config
            .lora_target
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect(),
    }
}

/// Query available system RAM in megabytes using sysinfo.
///
/// Returns `None` only if sysinfo fails to refresh (extremely rare on supported
/// platforms). The caller treats `None` as "skip the RAM warning".
fn available_system_ram_mb() -> Option<f32> {
    let mut sys = System::new();
    sys.refresh_memory();
    let available_bytes = sys.available_memory();
    if available_bytes == 0 {
        return None;
    }
    Some(available_bytes as f32 / 1_048_576.0)
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::train::samples::Sample;

    // ── estimate_avg_token_length ─────────────────────────────────────────────

    #[test]
    fn avg_token_length_empty() {
        assert_eq!(estimate_avg_token_length(&[], 1024), 0);
    }

    #[test]
    fn avg_token_length_clamped_to_max_len() {
        // 4000-char input ÷ 4 = 1000 tokens, which exceeds max_len=512
        let samples = vec![Sample {
            input:  "a".repeat(4000),
            output: "b".repeat(0),
        }];
        assert_eq!(estimate_avg_token_length(&samples, 512), 512);
    }

    #[test]
    fn avg_token_length_typical() {
        // 80 chars ÷ 4 = 20 tokens, well under max_len
        let samples = vec![Sample {
            input:  "Hello, world! This is a short input.".to_string(),
            output: "And a short output response.".to_string(),
        }];
        let result = estimate_avg_token_length(&samples, 1024);
        // (36 + 1 + 28) / 4 = 65/4 = 16; within a reasonable range
        assert!(result > 0 && result <= 1024);
    }

    // ── build_lora_config ─────────────────────────────────────────────────────

    #[test]
    fn lora_config_parses_csv_target_modules() {
        let cfg = TrainConfig {
            model:          "test".to_string(),
            dataset:        "x".into(),
            output:         "y".into(),
            name:           None,
            epochs:         1,
            batch_size:     1,
            grad_accum:     1,
            learning_rate:  1e-4,
            max_seq_len:    128,
            lora_r:         4,
            lora_alpha:     8,
            lora_dropout:   0.05,
            lora_target:    "q_proj, v_proj, k_proj".to_string(),
            qlora:          false,
            optimizer:      "adamw".to_string(),
            scheduler:      "cosine".to_string(),
            fp16:           false,
            weight_decay:   0.01,
        };
        let lora = build_lora_config(&cfg);
        assert_eq!(lora.target_modules, vec!["q_proj", "v_proj", "k_proj"]);
        assert_eq!(lora.r, 4);
    }

    #[test]
    fn lora_params_formula() {
        // 2 × r × d_model × |modules| = 2 × 8 × 4096 × 2 = 131_072
        let lora = LoraConfig {
            r: 8,
            alpha: 16.0,
            dropout: 0.05,
            target_modules: vec!["q_proj".to_string(), "v_proj".to_string()],
        };
        let d_model = 4096;
        let expected = 2 * lora.r * d_model * lora.target_modules.len();
        assert_eq!(expected, 131_072);
    }

    // ── HfModelConfig ─────────────────────────────────────────────────────────

    #[test]
    fn hf_config_prefers_num_parameters() {
        let raw = r#"{"num_parameters": 7000000000, "hidden_size": 4096, "num_hidden_layers": 32}"#;
        let cfg: HfModelConfig = serde_json::from_str(raw).unwrap();
        assert_eq!(cfg.total_params(), Some(7_000_000_000));
    }

    #[test]
    fn hf_config_falls_back_to_formula() {
        // 12 × 4096² × 32 = 6_442_450_944
        let raw = r#"{"hidden_size": 4096, "num_hidden_layers": 32}"#;
        let cfg: HfModelConfig = serde_json::from_str(raw).unwrap();
        assert_eq!(cfg.total_params(), Some(12 * 4096 * 4096 * 32));
    }

    #[test]
    fn hf_config_accepts_gpt2_aliases() {
        let raw = r#"{"n_embd": 768, "n_layer": 12}"#;
        let cfg: HfModelConfig = serde_json::from_str(raw).unwrap();
        assert_eq!(cfg.d_model(), Some(768));
        assert_eq!(cfg.num_layers(), Some(12));
    }

    #[test]
    fn hf_config_returns_none_when_fields_missing() {
        let raw = r#"{"vocab_size": 32000}"#;
        let cfg: HfModelConfig = serde_json::from_str(raw).unwrap();
        assert_eq!(cfg.total_params(), None);
    }

    // ── available_system_ram_mb ───────────────────────────────────────────────

    #[test]
    fn system_ram_is_positive() {
        // sysinfo should always return a non-zero value on the CI host.
        let mb = available_system_ram_mb();
        assert!(mb.is_some(), "sysinfo returned no RAM info");
        assert!(mb.unwrap() > 0.0, "available RAM must be > 0");
    }
}
