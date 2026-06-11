// train/native_runner.rs — Native Candle LoRA training orchestrator.
//
// Why this module exists:
//   gwen-tui cannot directly reference candle_core / candle_nn / hf_hub /
//   tokenizers — those are deps of gwen-core. This module is the single place
//   that wires: dataset loading → tokenisation → LoRA init → training loop.
//   gwen-tui calls run_native() and receives a TrainResult. That is the entire
//   public contract; the crate boundary is the API surface.
//
// Candle is now unconditional (Cycle 6) so there are no feature guards here.

use std::sync::mpsc::Sender;

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};

use crate::platform::hub_model::resolve_token;
use crate::train::config::{NewTrainConfig, TrainResult};
use crate::train::dataset::{batch, load_jsonl, tokenize, DEFAULT_MAX_LEN};
use crate::train::lora::LoraLayer;
use crate::train::layered_training_loop::LayeredTrainingLoop;
use crate::train::training_loop::TrainingLoop;

/// Run one complete native LoRA training job.
///
/// Steps:
///   1. Fetch tokenizer.json from HF Hub (cached after first call).
///   2. Load and tokenise the JSONL dataset.
///   3. Build VarMap + LoraLayer.
///   4. Run TrainingLoop, emitting JSON progress events to stdout AND via `tx`
///      so the TUI can read them without a subprocess.
///
/// `tx` parameter:
///   mpsc::Sender is Send + Clone; using it avoids lifetime tangles that come
///   with Box<dyn Fn> callbacks. If the TUI exits, send() returns Err which
///   is silently ignored — a dead receiver is not a training error.
/// Run native LoRA training directly against a local GGUF file using
/// `LayeredTrainingLoop` — no Python, no HF model download.
///
/// Tokenizer is still fetched from HF Hub by repo ID derived from the GGUF
/// filename (e.g. `Qwen3-1.7B-Q8_0.gguf` → `Qwen/Qwen3-1.7B`), or overridden
/// via `tokenizer_repo` if the caller knows the exact repo.
pub fn run_native_local(
    config: &NewTrainConfig,
    gguf_path: &std::path::Path,
    tokenizer_repo: Option<&str>,
    tx: Option<Sender<String>>,
) -> Result<TrainResult> {
    // ── 1. Tokenizer from HF Hub ──────────────────────────────────────────────
    let repo = tokenizer_repo
        .map(|s| s.to_string())
        .unwrap_or_else(|| infer_tokenizer_repo(gguf_path));

    eprintln!("[train] fetching tokenizer from HF Hub: {}", repo);

    let tokenizer = {
        use hf_hub::api::sync::ApiBuilder;
        let token = resolve_token();
        let api = ApiBuilder::from_env()
            .with_token(token)
            .with_progress(true)
            .build()
            .context("failed to build HF Hub sync client")?;
        let tok_path = api
            .model(repo.clone())
            .get("tokenizer.json")
            .with_context(|| format!("failed to fetch tokenizer.json for '{}'", repo))?;
        tokenizers::Tokenizer::from_file(&tok_path)
            .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {}", e))?
    };

    // ── 2. Dataset ────────────────────────────────────────────────────────────
    let samples = load_jsonl(&config.dataset_path).context("failed to load JSONL dataset")?;
    eprintln!("[train] {} samples from {}", samples.len(), config.dataset_path.display());

    let device = Device::Cpu;
    let token_tensors = tokenize(&samples, &tokenizer, DEFAULT_MAX_LEN, &device)
        .context("tokenisation failed")?;
    let batches = batch(token_tensors, config.batch_size);
    eprintln!("[train] {} batches (batch_size={})", batches.len(), config.batch_size);

    // ── 3. VarMap — populated by LayeredTrainingLoop::new ─────────────────────
    // The loop reads all architecture dims (vocab, hidden, layers) from the GGUF
    // at runtime and builds the persistent embedding + output head + per-layer
    // adapters itself, so we pass an empty VarMap here.
    let varmap = VarMap::new();

    // ── 4. LayeredTrainingLoop ────────────────────────────────────────────────
    eprintln!("[train] opening GGUF: {}", gguf_path.display());
    let mut training_loop = LayeredTrainingLoop::new(
        config.clone(), gguf_path, batches, varmap, tx,
    ).context("failed to initialise LayeredTrainingLoop")?;

    eprintln!("[train] starting layered training ({} epochs)…", config.epochs);
    training_loop.run().context("layered training loop failed")
}

/// Infer the HF Hub repo ID from a GGUF filename.
/// `Qwen3-1.7B-Q8_0.gguf` → `Qwen/Qwen3-1.7B`
/// `llama-3.1-8b-instruct-q4_k.gguf` → `meta-llama/Llama-3.1-8B-Instruct` (best-effort)
fn infer_tokenizer_repo(gguf_path: &std::path::Path) -> String {
    let stem = gguf_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");

    // Strip trailing quant suffix: `-Q8_0`, `-q4_k_m`, etc.
    let base = if let Some(pos) = stem.to_lowercase().rfind("-q") {
        &stem[..pos]
    } else {
        stem
    };

    // Known prefixes → HF org
    if base.to_lowercase().starts_with("qwen") {
        return format!("Qwen/{}", base);
    }
    if base.to_lowercase().starts_with("llama") {
        return format!("meta-llama/{}", base);
    }
    if base.to_lowercase().starts_with("mistral") {
        return format!("mistralai/{}", base);
    }
    if base.to_lowercase().starts_with("phi") {
        return format!("microsoft/{}", base);
    }
    // Fallback: user must set tokenizer_repo manually
    eprintln!("[train] warning: could not infer HF repo for '{}'; defaulting to '{}'", stem, base);
    base.to_string()
}

pub fn run_native(config: &NewTrainConfig, tx: Sender<String>) -> Result<TrainResult> {
    // ── 1. Tokenizer ──────────────────────────────────────────────────────────
    //
    // Sync hf_hub is used here because run_native is called from a std::thread
    // (not an async task). block_on on an existing tokio context would panic.

    let tokenizer = {
        use hf_hub::api::sync::ApiBuilder;
        let token = resolve_token();
        let api = ApiBuilder::from_env()
            .with_token(token)
            .with_progress(true)
            .build()
            .context("failed to build HF Hub sync client")?;
        let tok_path = api
            .model(config.model_id.clone())
            .get("tokenizer.json")
            .with_context(|| {
                format!(
                    "failed to fetch tokenizer.json for '{}'",
                    config.model_id
                )
            })?;
        tokenizers::Tokenizer::from_file(&tok_path)
            .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {}", e))?
    };

    // ── 2. Dataset ────────────────────────────────────────────────────────────

    let samples =
        load_jsonl(&config.dataset_path).context("failed to load JSONL dataset")?;

    eprintln!(
        "[train] {} samples loaded from {}",
        samples.len(),
        config.dataset_path.display()
    );

    // Device::Cpu always available; GPU selection belongs behind a --device
    // flag mapped to Device::new_cuda(0) / Device::Metal(0). Adding that flag
    // later requires no change to this function's signature.
    let device = Device::Cpu;

    let token_tensors =
        tokenize(&samples, &tokenizer, DEFAULT_MAX_LEN, &device).context("tokenisation failed")?;

    let batches = batch(token_tensors, config.batch_size);

    eprintln!(
        "[train] {} batches (batch_size={})",
        batches.len(),
        config.batch_size
    );

    // ── 3. Model ──────────────────────────────────────────────────────────────
    //
    // VarMap must outlive both LoraLayer (which holds Var references) and
    // TrainingLoop (which checkpoints from it). Keeping it here satisfies the
    // borrow checker without Arc or unsafe.

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    // Base weight placeholder — zero tensor stands in for the real frozen
    // weight until the model-loader module is written. LoRA adapters (lora_a,
    // lora_b) are real trainable Vars; only the frozen base is synthetic.
    let d_model = 4096_usize;
    let base_weight = Tensor::zeros((d_model, d_model), DType::F32, &device)
        .context("failed to allocate base weight placeholder")?;

    let model = LoraLayer::new(d_model, d_model, base_weight, &config.lora, vb)
        .context("failed to initialise LoraLayer")?;

    eprintln!(
        "[train] LoRA adapter ready ({} trainable params)",
        model.trainable_params()
    );

    // ── 4. Training loop ──────────────────────────────────────────────────────

    let mut training_loop =
        TrainingLoop::new(config.clone(), model, batches, varmap, Some(tx))
            .context("failed to initialise TrainingLoop")?;

    training_loop.run().context("training loop failed")
}
