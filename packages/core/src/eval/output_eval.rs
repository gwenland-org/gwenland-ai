// eval/output_eval.rs — Phase 2 output-based evaluation.
//
// Cycle 6: Ollama /api/generate replaced with native candle inference.
// Each sample is run through the inference runner in blocking mode so the
// tokio executor is not starved by candle's CPU-bound forward pass.
//
// Why still a separate module from metrics.rs?
// Phase 1 (metrics) is synchronous and pure-Rust (no model weights needed).
// Phase 2 is inference-bound and much slower per-sample. Keeping them split
// lets Phase 1 give fast feedback while Phase 2 runs through the full dataset.

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::eval::metrics::EvalSample;
use crate::engine::inference::runner::{run_inference, InferenceConfig};
use crate::engine::inference::sampler::SamplerConfig;

// ── result type ────────────────────────────────────────────────────────────────

/// Result for a single (input, expected, generated) triple.
///
/// `is_match` is true when `generated.trim().to_lowercase()` contains
/// `expected.trim().to_lowercase()`. Intentionally lenient: a model that
/// answers "The capital is Paris." when expected "Paris" is scored as correct.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampleResult {
    pub input: String,
    pub expected: String,
    pub generated: String,
    #[serde(rename = "match")]
    pub is_match: bool,
}

// ── progress callback ──────────────────────────────────────────────────────────

/// Called after each sample: (done_count, total_count, matched_count).
pub type OutputProgressCallback = Box<dyn Fn(usize, usize, usize) + Send>;

// ── public entry point ─────────────────────────────────────────────────────────

/// Run inference on each sample and score against expected output.
///
/// `model`       — model name or path (resolved by inference::loader)
/// `max_samples` — cap on how many samples to evaluate (spec default: 50)
/// `callback`    — optional live-update hook for the TUI progress panel
pub async fn run_output_eval(
    samples: &[EvalSample],
    model: &str,
    max_samples: usize,
    callback: Option<OutputProgressCallback>,
) -> Result<Vec<SampleResult>> {
    let eval_samples = &samples[..samples.len().min(max_samples)];
    let total = eval_samples.len();
    let mut results = Vec::with_capacity(total);
    let mut matched = 0usize;

    for (idx, sample) in eval_samples.iter().enumerate() {
        let generated = run_single_inference(model, &sample.input)
            .unwrap_or_else(|e| format!("[error: {}]", e));

        let is_match = is_substring_match(&generated, &sample.output);
        if is_match {
            matched += 1;
        }

        results.push(SampleResult {
            input: sample.input.clone(),
            expected: sample.output.clone(),
            generated,
            is_match,
        });

        if let Some(cb) = &callback {
            cb(idx + 1, total, matched);
        }
    }

    Ok(results)
}

// ── helpers ────────────────────────────────────────────────────────────────────

/// Run a single non-streaming inference call for evaluation.
/// Uses a tight token limit (128) since eval only needs a short answer.
fn run_single_inference(model: &str, prompt: &str) -> Result<String> {
    let cfg = InferenceConfig {
        model: model.to_string(),
        model_id_for_tokenizer: model.to_string(),
        prompt: prompt.to_string(),
        sampler: SamplerConfig {
            temperature: 0.0, // greedy for deterministic eval scoring
            max_tokens: 128,
            ..Default::default()
        },
        auto_stop_pct: 90,
        show_banner: false,
    };

    // run_inference is sync/blocking — safe to call from a std::thread or
    // via tokio::task::spawn_blocking.
    let result = run_inference(&cfg, None)?;
    Ok(result.generated_text)
}

/// Case-insensitive substring match: true if `generated` contains `expected`.
fn is_substring_match(generated: &str, expected: &str) -> bool {
    let generated_lc = generated.trim().to_lowercase();
    let exp = expected.trim().to_lowercase();
    if exp.is_empty() {
        return true;
    }
    generated_lc.contains(exp.as_str())
}

/// Compute the fraction of samples where `is_match` is true.
pub fn exact_match_rate(results: &[SampleResult]) -> f64 {
    if results.is_empty() {
        return 0.0;
    }
    let matched = results.iter().filter(|r| r.is_match).count();
    matched as f64 / results.len() as f64
}
