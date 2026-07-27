//! `glbench kl-div` — per-position KL-divergence between `glproc::runner::Runner`
//! (the native-quantized, known-good reference — `gwen run`'s engine) and a
//! `.gllm` package (`GllmEngine`), teacher-forced over the identical token
//! sequence. Gated behind `gllm-bench`, same as [`crate::ppl`].
//!
//! # Vetted against `RESEARCH_REQUIREMENTS.md`'s 8 mandatory questions
//!
//! 1. **What engineering problem does this solve?** Aggregate scores (PPL,
//!    cross-entropy) can move only slightly even when one tensor in one
//!    layer is completely wrong — that is exactly what happened with the
//!    Q6_K dequant bug (`notes/issues/gllm-e2e-garbage-output.md`,
//!    resolved): the bug was only actually localized once someone wrote a
//!    one-off differential dump (`glictus-caliburni/examples/diff_dump.rs`)
//!    comparing per-position logits between the two engines by hand. This
//!    command promotes that technique from a throwaway diagnostic script
//!    into a standing, reusable glbench capability.
//! 2. **Who benefits?** Anyone changing glictus-caliburni's runtime,
//!    converter, or dequant kernels, who needs to know *before* shipping
//!    whether a change silently diverged from the reference math — the
//!    exact question the Q6_K investigation spent days answering by hand.
//! 3. **Used in production/research?** Yes — `llama.cpp` ships
//!    `--kl-divergence-base` for precisely this purpose (compare a
//!    quantized build's logits against a full-precision reference); it is
//!    the community's standard companion metric to perplexity, not a
//!    novelty invented here.
//! 4. **How is it calculated?** Standard `KL(P‖Q) = Σ P(i)·(log P(i) − log
//!    Q(i))`, `P` = the oracle (`glproc::runner::Runner`)'s softmax
//!    distribution, `Q` = the `.gllm` package's, at each teacher-forced
//!    position — full vocabulary, no temperature, no truncation (same "raw
//!    logits" discipline `glcore::trace` already documents for the
//!    behavioral signals).
//! 5. **Reproducible?** Yes — fixed embedded text (the same
//!    [`crate::ppl::WIKITEXT2_SAMPLE`] `ppl` uses, for an apples-to-apples
//!    comparison between the two commands), deterministic forward pass on
//!    both sides, no sampling anywhere in the path.
//! 6. **Actionable insight?** Yes, directly: it names the exact token
//!    position where the two engines' beliefs about the next token
//!    diverge — the same kind of signal `diff_dump.rs`'s per-layer norm
//!    table gives for hidden states, but at the output distribution
//!    instead, and without needing to hand-write a new diagnostic script
//!    each time.
//! 7. **Lightweight?** Yes — reuses `GllmEngine::score_sequence` (already
//!    built for `ppl`) and `glproc::runner::Runner::forward_into` (already
//!    built for `gwen run`). No new `GlEngine`-trait surface, no new
//!    capture mechanism; one log-softmax pass per position per engine.
//! 8. **Aligns with philosophy?** Yes — purely observational, computes a
//!    number from two engines' own already-produced logits, mutates
//!    neither.
//!
//! # Why this drives `glproc::runner::Runner` and `GllmEngine` directly,
//! not through `EngineAdapter`
//!
//! Same reason as [`crate::ppl`]: a `.gllm` package has no tokenizer yet
//! (ARTX1 OQ3), so `GllmEngine` cannot go through `glcore::Runtime`'s
//! `load` (extension-based dispatch expects a `.gguf`/`.safetensors` file).
//! `--gguf` supplies both the tokenizer source *and* the oracle model in
//! one file, exactly as `run_package_e2e.rs` and `ppl` already do.
//!
//! # Why one full pass, not `ppl`'s sliding window
//!
//! `ppl`'s sliding window exists to score documents longer than the
//! context length would otherwise allow. KL-divergence here is scoped to
//! "does the .gllm forward pass match the reference on real text" — a
//! single teacher-forced pass over `--tokens` tokens (default 64, capped
//! deliberately low: every position costs a full forward pass on *both*
//! engines, and `.gllm`'s per-token per-layer mmap/unmap overhead — see
//! `architecture/mensura-veritatis-v3/ARTX4-Benchmark.md` — makes this the
//! slower of glbench's two `.gllm`-scoring commands) is enough to answer
//! that question and keeps the implementation to exactly what's needed,
//! not a second copy of the windowing logic.

use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use glcore::engine_trait::GlEngine;
use glcore::format::gguf::GgufFile;
use gltokenizer::Tokenizer;
use glictus_caliburni::runtime::GllmEngine;

use crate::export::json::Json;
use crate::ppl::{log_softmax, WIKITEXT2_SAMPLE};

/// CLI arguments for `kl-div`.
pub struct KlDivArgs {
    pub model: PathBuf,
    pub gguf: PathBuf,
    pub tokens: usize,
    pub out: Option<PathBuf>,
}

impl Default for KlDivArgs {
    fn default() -> Self {
        KlDivArgs { model: PathBuf::new(), gguf: PathBuf::new(), tokens: 64, out: None }
    }
}

/// The JSON-serializable summary of one `kl-div` run.
#[derive(Debug, Clone, PartialEq)]
pub struct KlDivSession {
    pub model_path: String,
    pub gguf_path: String,
    pub tokens_compared: usize,
    pub kl_mean: f64,
    pub kl_max: f64,
    pub kl_max_position: usize,
    pub per_position_kl: Vec<f64>,
    pub timestamp: String,
}

impl KlDivSession {
    pub fn to_json(&self) -> Json {
        Json::obj([
            ("model_path", Json::s(self.model_path.clone())),
            ("gguf_path", Json::s(self.gguf_path.clone())),
            ("tokens_compared", Json::n(self.tokens_compared as f64)),
            ("kl_mean", Json::n(self.kl_mean)),
            ("kl_max", Json::n(self.kl_max)),
            ("kl_max_position", Json::n(self.kl_max_position as f64)),
            (
                "per_position_kl",
                Json::Arr(self.per_position_kl.iter().map(|&v| Json::n(v)).collect()),
            ),
            ("timestamp", Json::s(self.timestamp.clone())),
        ])
    }

    pub fn from_json(v: &Json) -> Result<KlDivSession, String> {
        let per_position_kl = v
            .get("per_position_kl")
            .and_then(|a| a.as_arr())
            .map(|a| a.iter().filter_map(|x| x.as_f64()).collect())
            .unwrap_or_default();

        Ok(KlDivSession {
            model_path: v.get("model_path").and_then(|s| s.as_str()).unwrap_or("").to_string(),
            gguf_path: v.get("gguf_path").and_then(|s| s.as_str()).unwrap_or("").to_string(),
            tokens_compared: v.get("tokens_compared").and_then(|n| n.as_f64()).unwrap_or(0.0) as usize,
            kl_mean: v.get("kl_mean").and_then(|n| n.as_f64()).unwrap_or(0.0),
            kl_max: v.get("kl_max").and_then(|n| n.as_f64()).unwrap_or(0.0),
            kl_max_position: v.get("kl_max_position").and_then(|n| n.as_f64()).unwrap_or(0.0) as usize,
            per_position_kl,
            timestamp: v.get("timestamp").and_then(|s| s.as_str()).unwrap_or("").to_string(),
        })
    }
}

/// Run `glbench kl-div --model <package-dir> --gguf <original.gguf> [...]`.
pub fn run_kl_divergence(args: KlDivArgs) -> Result<(), String> {
    eprintln!("loading tokenizer + oracle model from {} ...", args.gguf.display());
    let gguf_path_str = args.gguf.to_str().ok_or("--gguf path is not valid UTF-8")?;
    let gguf = GgufFile::open(gguf_path_str).map_err(|e| format!("opening {}: {e}", args.gguf.display()))?;
    let tokenizer =
        Tokenizer::from_gguf_path(&args.gguf.to_string_lossy()).map_err(|e| format!("building tokenizer from {}: {e}", args.gguf.display()))?;

    let all_tokens = tokenizer.encode(WIKITEXT2_SAMPLE, false);
    if all_tokens.len() <= args.tokens {
        return Err(format!(
            "embedded WikiText-2 sample tokenized to only {} tokens, need more than --tokens ({})",
            all_tokens.len(),
            args.tokens
        ));
    }
    let tokens = &all_tokens[..args.tokens];

    let model = glproc::loader::load_gguf(&gguf).map_err(|e| format!("loading glproc oracle model: {e}"))?;
    let mut runner = glproc::runner::Runner::new(&model);
    let mut oracle_logits = Vec::with_capacity(tokens.len());
    for (pos, &token) in tokens.iter().enumerate() {
        runner
            .forward_into(token, pos)
            .map_err(|e| format!("oracle forward pass at position {pos}: {e}"))?;
        oracle_logits.push(runner.logits().to_vec());
    }

    eprintln!("loading .gllm package from {} ...", args.model.display());
    let mut engine = GllmEngine::new();
    engine.init().map_err(|e| format!("initializing gllm engine: {e}"))?;
    engine
        .load_model(args.model.to_str().ok_or("--model path is not valid UTF-8")?)
        .map_err(|e| format!("loading {}: {e}", args.model.display()))?;
    let candidate_logits = engine
        .score_sequence(tokens)
        .map_err(|e| format!("scoring .gllm package: {e}"))?;

    if oracle_logits.len() != candidate_logits.len() {
        return Err(format!(
            "position count mismatch: oracle={} candidate={}",
            oracle_logits.len(),
            candidate_logits.len()
        ));
    }

    let per_position_kl: Vec<f64> = oracle_logits
        .iter()
        .zip(&candidate_logits)
        .map(|(o, c)| kl_divergence(o, c))
        .collect::<Result<Vec<_>, String>>()?;

    let kl_mean = mean(&per_position_kl);
    let (kl_max_position, kl_max) = per_position_kl
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, &v)| (i, v))
        .unwrap_or((0, 0.0));

    let session = KlDivSession {
        model_path: args.model.display().to_string(),
        gguf_path: args.gguf.display().to_string(),
        tokens_compared: tokens.len(),
        kl_mean,
        kl_max,
        kl_max_position,
        per_position_kl,
        timestamp: iso8601_now(),
    };

    print!("{}", render_table(&session));

    if let Some(out) = &args.out {
        std::fs::write(out, session.to_json().to_pretty()).map_err(|e| format!("writing {}: {e}", out.display()))?;
        eprintln!("wrote {}", out.display());
    }

    Ok(())
}

/// `KL(oracle ‖ candidate) = Σ_i P_oracle(i) · (log P_oracle(i) − log P_candidate(i))`,
/// computed entirely in log-space via [`log_softmax`] until the final
/// exponentiation, the standard numerically-stable formulation. Terms where
/// the oracle assigns (rounding-to-)zero probability are skipped — they
/// contribute `0 · anything = 0` to the true sum, but `0 · -inf` would
/// otherwise poison the computation in floating point.
fn kl_divergence(oracle_logits: &[f32], candidate_logits: &[f32]) -> Result<f64, String> {
    if oracle_logits.len() != candidate_logits.len() {
        return Err(format!(
            "vocab size mismatch: oracle={} candidate={}",
            oracle_logits.len(),
            candidate_logits.len()
        ));
    }
    let oracle_log_probs = log_softmax(oracle_logits);
    let candidate_log_probs = log_softmax(candidate_logits);

    Ok(oracle_log_probs
        .iter()
        .zip(&candidate_log_probs)
        .map(|(&p_log, &q_log)| {
            let p = (p_log as f64).exp();
            if p <= 0.0 {
                0.0
            } else {
                p * (p_log as f64 - q_log as f64)
            }
        })
        .sum())
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn render_table(s: &KlDivSession) -> String {
    let mut out = String::new();
    let rule = "\u{2500}".repeat(46);

    out.push_str(&format!("glbench kl-div: {}\n", s.model_path));
    out.push_str(&format!("  oracle (glproc::runner::Runner)  {}\n", s.gguf_path));
    out.push_str(&format!("{rule}\n"));
    out.push_str(&format!("  Tokens compared    {}\n", s.tokens_compared));
    out.push_str(&format!("  KL mean            {:.6}\n", s.kl_mean));
    out.push_str(&format!("  KL max             {:.6}  (position {})\n", s.kl_max, s.kl_max_position));
    out.push_str(&format!("{rule}\n"));
    out.push_str("  Read: KL == 0 means the two engines agree exactly at that\n");
    out.push_str("  position. A few nats is a real, large disagreement -- for\n");
    out.push_str("  reference, this is the class of signal that localized the\n");
    out.push_str("  Q6_K dequant bug (notes/issues/gllm-e2e-garbage-output.md).\n");
    out
}

/// Manual ISO 8601 UTC timestamp (no `chrono` — glbench's zero-dep rule,
/// same implementation as [`crate::ppl::run_ppl`]'s and `quant_info.rs`'s).
fn iso8601_now() -> String {
    let secs = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    civil_from_unix(secs)
}

fn civil_from_unix(unix_secs: u64) -> String {
    let days = (unix_secs / 86400) as i64;
    let secs_of_day = unix_secs % 86400;
    let (hour, minute, second) = (secs_of_day / 3600, (secs_of_day % 3600) / 60, secs_of_day % 60);

    let z = days + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if month <= 2 { y + 1 } else { y };

    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kl_divergence_of_identical_distributions_is_zero() {
        let logits = vec![1.0f32, 2.0, 0.5, -1.0, 3.0];
        let kl = kl_divergence(&logits, &logits).unwrap();
        assert!(kl.abs() < 1e-5, "KL(P||P) should be ~0, got {kl}");
    }

    #[test]
    fn kl_divergence_is_nonnegative_for_different_distributions() {
        let p = vec![2.0f32, 0.1, 0.1, 0.1];
        let q = vec![0.1f32, 2.0, 0.1, 0.1];
        let kl = kl_divergence(&p, &q).unwrap();
        assert!(kl > 0.0, "KL should be strictly positive for different distributions, got {kl}");
    }

    #[test]
    fn kl_divergence_matches_hand_computed_two_point_case() {
        // P = [log(0.9), log(0.1)] as raw "logits" isn't quite right since
        // log_softmax renormalizes -- instead pick logits whose softmax is
        // exactly known: [ln(9), 0] -> softmax = [0.9, 0.1].
        let p_logits = vec![9f32.ln(), 0.0];
        let q_logits = vec![1f32.ln(), 0.0]; // softmax = [0.5, 0.5]
        let kl = kl_divergence(&p_logits, &q_logits).unwrap();
        // KL(P||Q) = 0.9*ln(0.9/0.5) + 0.1*ln(0.1/0.5) = 0.9*ln(1.8) + 0.1*ln(0.2)
        let expected = 0.9 * (1.8f64).ln() + 0.1 * (0.2f64).ln();
        assert!((kl - expected).abs() < 1e-4, "got {kl}, want {expected}");
    }

    #[test]
    fn kl_divergence_rejects_vocab_size_mismatch() {
        let a = vec![1.0f32, 2.0];
        let b = vec![1.0f32, 2.0, 3.0];
        assert!(kl_divergence(&a, &b).is_err());
    }

    #[test]
    fn kl_div_session_serializes() {
        let session = KlDivSession {
            model_path: "model.gllm".into(),
            gguf_path: "model.gguf".into(),
            tokens_compared: 64,
            kl_mean: 0.1234,
            kl_max: 2.5,
            kl_max_position: 12,
            per_position_kl: vec![0.01, 0.02, 2.5],
            timestamp: "2026-07-23T00:00:00Z".into(),
        };
        let back = KlDivSession::from_json(&session.to_json()).unwrap();
        assert_eq!(session, back);
    }

    #[test]
    fn iso8601_epoch_is_1970() {
        assert_eq!(civil_from_unix(0), "1970-01-01T00:00:00Z");
    }
}
