// engine/inference/runner.rs — Token-by-token generation loop.
//
// This is the public entry point for native inference. The caller provides an
// `InferenceConfig` and a callback (or None for raw stdout). Tokens are
// produced one at a time, flushed immediately for a streaming feel.
//
// Memory guard integration:
//   The MemoryGuard is checked every 16 tokens (GWEN-208). If RAM usage
//   exceeds the configured threshold the loop stops gracefully and prints a
//   warning. The partial output is returned to the caller so it can persist
//   the session fragment.
//
// Architecture dispatch:
//   Because candle-transformers uses different types per architecture, we
//   delegate to per-architecture helpers at the bottom of this file. Each
//   helper shares the same token-loop skeleton; only the model construction
//   and forward-pass call differ.

use anyhow::{Context, Result};
use candle_transformers::generation::LogitsProcessor;
use std::io::Write;
use sysinfo::System;

use crate::engine::inference::{
    loader::{self, LoadedModel},
    model_dispatch::{detect_from_gguf, ModelKind},
    sampler::SamplerConfig,
};
use crate::engine::memory_guard::MemoryGuard;

// ── public config ─────────────────────────────────────────────────────────────

/// Everything `run_inference` needs to generate a response.
#[derive(Debug, Clone)]
pub struct InferenceConfig {
    /// Model name or path (resolved by loader::resolve_model_path).
    pub model: String,
    /// HF model ID used to fetch the tokenizer (e.g. "Qwen/Qwen3-8B").
    pub model_id_for_tokenizer: String,
    /// Prompt text (system + user combined if interactive mode).
    pub prompt: String,
    pub sampler: SamplerConfig,
    /// RAM threshold at which generation is halted (0–100 %).
    pub auto_stop_pct: u8,
    /// Print device / model banner to stderr before generating.
    pub show_banner: bool,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            model: String::new(),
            model_id_for_tokenizer: String::new(),
            prompt: String::new(),
            sampler: SamplerConfig::default(),
            auto_stop_pct: 90,
            show_banner: true,
        }
    }
}

/// Result returned after a generation run completes (or is interrupted).
pub struct InferenceResult {
    pub generated_text: String,
    pub token_count: usize,
    /// True if the run was halted early by the MemoryGuard.
    pub memory_stopped: bool,
}

// ── public entry point ─────────────────────────────────────────────────────────

/// Run native inference.
///
/// Tokens are printed to stdout one-by-one with a flush after each, giving
/// a streaming feel even though we are in-process.
///
/// `callback` — if Some, called with each decoded token string instead of
///              printing directly. Used by the SSE proxy to push tokens into
///              the channel without owning stdout.
pub fn run_inference(
    cfg: &InferenceConfig,
    callback: Option<&mut dyn FnMut(String)>,
) -> Result<InferenceResult> {
    let memory_guard = MemoryGuard::new(cfg.auto_stop_pct);
    let mut sys = System::new_all();

    let loaded = loader::load(&cfg.model, &cfg.model_id_for_tokenizer)
        .context("failed to load model")?;

    let kind = detect_from_gguf(&loaded.gguf_path)
        .context("failed to detect model architecture")?;

    if cfg.show_banner {
        eprintln!("  ❖ Architecture: {}", kind.as_str());
        eprintln!("  ❖ Max tokens:   {}", cfg.sampler.max_tokens);
        eprintln!("  ❖ Temperature:  {}", cfg.sampler.temperature);
        eprintln!("  ❖ Auto-stop:    {}% RAM", cfg.auto_stop_pct);
        eprintln!();
    }

    cfg.sampler.validate()?;

    dispatch_generation(cfg, loaded, kind, memory_guard, &mut sys, callback)
}

// ── architecture dispatch ─────────────────────────────────────────────────────

fn dispatch_generation(
    cfg: &InferenceConfig,
    loaded: LoadedModel,
    kind: ModelKind,
    memory_guard: MemoryGuard,
    sys: &mut System,
    callback: Option<&mut dyn FnMut(String)>,
) -> Result<InferenceResult> {
    // All architectures share the same token loop; only the forward-pass
    // wrapper differs. We abstract it via a closure that owns the model.
    //
    // Why a closure rather than a trait? The model types from candle-transformers
    // don't share a common trait for the forward pass, and boxing the closure is
    // simpler than defining a new trait just for this dispatch.

    match kind {
        ModelKind::LLaMA3  => run_quantized_loop(cfg, loaded, memory_guard, sys, callback),
        ModelKind::Mistral => run_quantized_loop(cfg, loaded, memory_guard, sys, callback),
        ModelKind::Qwen    => run_quantized_loop(cfg, loaded, memory_guard, sys, callback),
        ModelKind::Phi3    => run_quantized_loop(cfg, loaded, memory_guard, sys, callback),
    }
}

// ── GGUF quantized generation loop ────────────────────────────────────────────

/// Token-by-token generation using candle's built-in GGUF model loader.
///
/// `ModelWeights` in candle-transformers supports multiple architectures
/// through the `quantized::llama::ModelWeights` type which reads any
/// llama-compatible GGUF. For Qwen2/Qwen3, Mistral, and Phi-3, the GGUF
/// metadata tells candle which rope / attention variant to use — the loading
/// path is identical.
fn run_quantized_loop(
    cfg: &InferenceConfig,
    loaded: LoadedModel,
    memory_guard: MemoryGuard,
    sys: &mut System,
    mut callback: Option<&mut dyn FnMut(String)>,
) -> Result<InferenceResult> {
    use candle_core::quantized::gguf_file;
    use candle_transformers::models::quantized_llama::ModelWeights;
    use std::fs::File;
    use std::io::BufReader;

    // Open and fully parse the GGUF file.
    let f = File::open(&loaded.gguf_path)
        .with_context(|| format!("cannot open GGUF: {}", loaded.gguf_path.display()))?;
    let mut reader = BufReader::new(f);

    let model_content = gguf_file::Content::read(&mut reader)
        .context("failed to parse GGUF weights")?;

    let mut model = ModelWeights::from_gguf(model_content, &mut reader, &loaded.device)
        .context("failed to build model from GGUF")?;

    // Tokenise prompt.
    let tokenizer = loaded.tokenizer;
    let encoding = tokenizer
        .encode(cfg.prompt.as_str(), true)
        .map_err(|e| anyhow::anyhow!("tokenizer encode error: {}", e))?;
    let mut tokens: Vec<u32> = encoding.get_ids().to_vec();

    // Build the candle LogitsProcessor from our sampler config — it handles
    // temperature and top-p internally so we delegate to it rather than
    // duplicating the maths from sampler.rs.
    let mut logits_processor = LogitsProcessor::new(
        rand::random::<u64>(),
        Some(cfg.sampler.temperature as f64),
        Some(cfg.sampler.top_p as f64),
    );

    let eos_token = tokenizer
        .token_to_id("</s>")
        .or_else(|| tokenizer.token_to_id("<|endoftext|>"))
        .or_else(|| tokenizer.token_to_id("<|im_end|>"))
        .unwrap_or(2); // fallback EOS id used by most LLaMA models

    let mut generated_text = String::new();
    let mut token_count = 0usize;
    let mut memory_stopped = false;

    let prompt_len = tokens.len();

    for _ in 0..cfg.sampler.max_tokens {
        // Forward pass — the model mutates its KV-cache internally.
        let input = candle_core::Tensor::new(tokens.as_slice(), &loaded.device)?
            .unsqueeze(0)?; // shape: [1, seq_len]

        let logits = model.forward(&input, prompt_len + token_count)?;
        // logits shape: [1, 1, vocab_size] — squeeze to [vocab_size]
        let logits = logits.squeeze(0)?.squeeze(0)?;

        // After the first token we only feed the newly generated token.
        // Slice tokens to just the last one for subsequent iterations.
        let next_token = logits_processor.sample(&logits)?;

        tokens = vec![next_token]; // KV cache tracks history; feed single token next

        if next_token == eos_token {
            break;
        }

        let token_str = tokenizer
            .decode(&[next_token], true)
            .map_err(|e| anyhow::anyhow!("tokenizer decode error: {}", e))?;

        generated_text.push_str(&token_str);
        token_count += 1;

        // Emit token — either via callback (SSE proxy) or direct stdout.
        if let Some(ref mut cb) = callback {
            cb(token_str);
        } else {
            print!("{}", token_str);
            let _ = std::io::stdout().flush();
        }

        // GWEN-208: RAM check every 16 tokens to avoid sysinfo overhead per token.
        if token_count % 16 == 0 {
            sys.refresh_memory();
            if memory_guard.check(sys) {
                let used_pct = memory_used_pct(sys);
                eprintln!(
                    "\n  ⚠ RAM threshold reached ({}%). Stopping inference.",
                    used_pct
                );
                memory_stopped = true;
                break;
            }
        }
    }

    // Newline after the streamed tokens so the shell prompt appears cleanly.
    if callback.is_none() {
        println!();
    }

    Ok(InferenceResult {
        generated_text,
        token_count,
        memory_stopped,
    })
}

fn memory_used_pct(sys: &System) -> u8 {
    let total = sys.total_memory();
    if total == 0 {
        return 0;
    }
    ((sys.used_memory() as f64 / total as f64) * 100.0) as u8
}

// ── dry-run output ─────────────────────────────────────────────────────────────

/// Print the dry-run table to stdout without executing anything.
pub fn print_dry_run(cfg: &InferenceConfig) {
    use crate::engine::inference::loader::{estimate_vram_gb, resolve_model_path, select_device};
    use crate::engine::inference::model_dispatch::detect_from_gguf;

    println!("gwenland run (dry run)");

    let (model_path, arch_str, vram_str) = match resolve_model_path(&cfg.model) {
        Ok(p) => {
            let arch = detect_from_gguf(&p)
                .map(|k| k.as_str().to_string())
                .unwrap_or_else(|_| "unknown".to_string());
            let vram = estimate_vram_gb(&p);
            (p.display().to_string(), arch, format!("{:.1} GB", vram))
        }
        Err(e) => (
            format!("NOT FOUND — {}", e),
            "unknown".to_string(),
            "unknown".to_string(),
        ),
    };

    let (_, device_label) = select_device();

    println!("  Model:        {}", model_path);
    println!("  Architecture: {}", arch_str);
    println!("  Device:       {}", device_label);
    println!("  VRAM est.:    {}", vram_str);
    println!("  Max tokens:   {}", cfg.sampler.max_tokens);
    println!("  Temperature:  {}", cfg.sampler.temperature);
    println!("  Auto-stop:    {}% RAM threshold", cfg.auto_stop_pct);
    println!("  Status:       Ready (not executed)");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::inference::loader::resolve_model_path;

    #[test]
    fn test_dry_run_output_format() {
        // Just ensure print_dry_run doesn't panic on a missing model.
        let cfg = InferenceConfig {
            model: "nonexistent-model".to_string(),
            model_id_for_tokenizer: "Qwen/Qwen3-8B".to_string(),
            ..Default::default()
        };
        print_dry_run(&cfg); // should not panic
    }

    #[test]
    fn test_model_resolution_order() {
        // A path starting with "./" should be treated as an explicit path (not searched).
        let result = resolve_model_path("./no_such_file.gguf");
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("does not exist"), "got: {}", msg);
    }

    #[test]
    fn test_model_resolution_named_not_found() {
        // A bare name that doesn't exist should suggest gwen fetch.
        let result = resolve_model_path("totally-fake-model-zzz");
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("gwen fetch"), "got: {}", msg);
    }
}
