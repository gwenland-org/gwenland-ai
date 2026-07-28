//! `glbench ppl` — perplexity of a `.gllm` package on an embedded WikiText-2
//! sample, via teacher-forced log-probabilities (Wave 2, gated behind the
//! `gllm-bench` feature — the same approved dependency path Wave 1 used for
//! `quant-info`, now actually exercising glictus-caliburni's runtime).
//!
//! # Root cause of the garbage-output bug is fixed (2026-07-23) — re-validate before citing
//!
//! The forward pass this command drives (`GllmEngine` / `GllmRuntime` /
//! `GlprocBackend`) previously produced garbage output end-to-end (multilingual
//! noise, not remotely coherent) on Qwen2.5-0.5B-Instruct. Root cause found and
//! fixed: `glcore::format::gguf::dequant_q6_k` used the wrong (naive linear)
//! nibble order for Q6_K, silently corrupting `ffn_down.weight` in every layer
//! of a real Q4_K_M model — see `notes/issues/gllm-e2e-garbage-output.md`
//! (resolved) and `architecture/mensura-veritatis-v3/ARTX2-Quant.md` for the
//! full audit. Fixed in `converter.rs`'s `dequantize_for_gquant` (PR #16,
//! merged), verified via `diff_dump.rs` (layer-0 divergence vs the known-good
//! `glproc::runner::Runner` path dropped 300x) and `run_package_e2e` (garbage
//! → coherent, factually correct text).
//!
//! **This does not mean a PPL number from this command is validated** — it
//! means the known reason numbers were meaningless is gone. Nobody has re-run
//! `glbench ppl` against a real GQ4A/GQ2A package since the fix (Pridwen
//! §12's "Known Unknowns" — GQ4A/GQ2A PPL vs Q4_K_M — are still unmeasured;
//! see `architecture/mensura-veritatis-v3/ARTX4-Benchmark.md`). Run it, and
//! ideally cross-check with a KL-divergence comparison against
//! `glproc::runner::Runner`'s logits on the same sequence, before citing any
//! number anywhere.
//!
//! # Why `--gguf` is required alongside `--model`
//!
//! A `.gllm` package does not package a tokenizer yet (ARTX1 OQ3 is decided
//! — a `GLLMTokenizer.gllm` unit — but the converter does not emit one). The
//! only tokenizer this workspace has is `glcore::tokenizer::GllmTokenizer`,
//! built from GGUF metadata — so this command asks for the *original* source
//! GGUF the package was converted from, exactly as
//! `glictus-caliburni/examples/run_package_e2e.rs` does.
//!
//! # Why the sliding window is a sequential loop, not one batched call
//!
//! `GllmEngine::score_sequence` (added for this command) forwards one token
//! at a time — the whole runtime stack (`ActivationBuffer`, `KvCacheSlot`) is
//! architected around single-position autoregressive state, with no
//! `[seq_len, dim]` batched path anywhere. A sliding window of `context`
//! tokens is `context` sequential forward passes; there is no way to ask for
//! fewer today.

use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use glcore::engine_trait::GlEngine;
use glcore::format::gguf::GgufFile;
use glcore::tokenizer::GllmTokenizer;
use glictus_caliburni::runtime::GllmEngine;

use crate::export::json::Json;

/// CLI arguments for `ppl`.
pub struct PplArgs {
    pub model: PathBuf,
    pub gguf: PathBuf,
    pub context: usize,
    pub stride: usize,
    pub out: Option<PathBuf>,
}

impl Default for PplArgs {
    fn default() -> Self {
        PplArgs {
            model: PathBuf::new(),
            gguf: PathBuf::new(),
            context: 512,
            stride: 256,
            out: None,
        }
    }
}

/// The JSON-serializable summary of one `ppl` run.
#[derive(Debug, Clone, PartialEq)]
pub struct PplSession {
    pub model_path: String,
    pub context_len: usize,
    pub stride: usize,
    pub dataset: &'static str,
    pub total_tokens: usize,
    pub evaluated_tokens: usize,
    pub perplexity: f64,
    pub cross_entropy_mean: f64,
    pub per_window_ce: Vec<f64>,
    pub timestamp: String,
}

impl PplSession {
    pub fn to_json(&self) -> Json {
        Json::obj([
            ("model_path", Json::s(self.model_path.clone())),
            ("context_len", Json::n(self.context_len as f64)),
            ("stride", Json::n(self.stride as f64)),
            ("dataset", Json::s(self.dataset)),
            ("total_tokens", Json::n(self.total_tokens as f64)),
            ("evaluated_tokens", Json::n(self.evaluated_tokens as f64)),
            ("perplexity", Json::n(self.perplexity)),
            ("cross_entropy_mean", Json::n(self.cross_entropy_mean)),
            (
                "per_window_ce",
                Json::Arr(self.per_window_ce.iter().map(|&v| Json::n(v)).collect()),
            ),
            ("timestamp", Json::s(self.timestamp.clone())),
        ])
    }

    pub fn from_json(v: &Json) -> Result<PplSession, String> {
        let per_window_ce = v
            .get("per_window_ce")
            .and_then(|a| a.as_arr())
            .map(|a| a.iter().filter_map(|x| x.as_f64()).collect())
            .unwrap_or_default();

        Ok(PplSession {
            model_path: v.get("model_path").and_then(|s| s.as_str()).unwrap_or("").to_string(),
            context_len: v.get("context_len").and_then(|n| n.as_f64()).unwrap_or(0.0) as usize,
            stride: v.get("stride").and_then(|n| n.as_f64()).unwrap_or(0.0) as usize,
            dataset: "wikitext2-sample-embedded",
            total_tokens: v.get("total_tokens").and_then(|n| n.as_f64()).unwrap_or(0.0) as usize,
            evaluated_tokens: v.get("evaluated_tokens").and_then(|n| n.as_f64()).unwrap_or(0.0) as usize,
            perplexity: v.get("perplexity").and_then(|n| n.as_f64()).unwrap_or(0.0),
            cross_entropy_mean: v.get("cross_entropy_mean").and_then(|n| n.as_f64()).unwrap_or(0.0),
            per_window_ce,
            timestamp: v.get("timestamp").and_then(|s| s.as_str()).unwrap_or("").to_string(),
        })
    }
}

/// First article of the WikiText-2 test split (Valkyria Chronicles III),
/// lightly trimmed. Stable across builds: real English prose, not lorem
/// ipsum, so a BPE tokenizer produces a real distribution of common/rare
/// tokens rather than an artificially easy or hard sequence.
pub const WIKITEXT2_SAMPLE: &str = " Valkyria Chronicles III ( Japanese : \u{6226}\u{5834}\u{306e}\u{30f4}\u{30a1}\u{30eb}\u{30ad}\u{30e5}\u{30ea}\u{30a2}3 , lit . Sen j\u{014d} no Valkyria 3 : Gallian Chronicles ) , commonly referred to as Valkyria Chronicles III outside Japan , is a tactical role @-@ playing video game developed by Sega and Media.Vision for the PlayStation Portable . Released in January 2011 in Japan , it is the third game in the Valkyria series . Employing the same fusion of tactical and real @-@ time gameplay as its predecessors , the story runs parallel to the first game and follows the \" Nameless \" , a penal military unit serving the nation of Gallia during the Second Europan War who perform secret black operations and are pitted against an elite enemy unit known as \" Calamaty Raven \" .\n The game began development in 2010 , carrying over a large portion of the work done on Valkyria Chronicles II . While it retained the standard features of the series , it also underwent multiple adjustments , such as making the game more forgiving for series newcomers . Character designer Raita Honjou and composer Hitoshi Sakimoto both returned from previous entries , along with Valkyria Chronicles II director Takeshi Ozawa . A large team of writers handled the script . The game 's opening theme was sung by May 'n .\n It met with positive sales in Japan , and was praised by both Japanese and western critics . After release , it received downloadable content , along with an expanded edition in November of that year . It was also adapted into manga and an original video animation series . Due to low sales of Valkyria Chronicles II , Valkyria Chronicles III was not localized , but a fan translation compatible with the game 's expanded edition was released in 2014 . Media.Vision would return to the franchise with the development of Valkyria : Azure Revolution for the PlayStation 4 .\n As with the previous Valkyria Chronicles games , Valkyria Chronicles III is a tactical role @-@ playing game where players take control of a military unit and take part in missions against enemy forces . Stories are told through comic book @-@ like panels with animated character portraits , with characters speaking partially in voiced speech bubbles and partially in unvoiced , separate text . The player progresses through a series of linear missions , gradually unlocked as maps that can be freely scanned through and replayed as they are unlocked . The route to each story location on the map varies depending on an individual player 's approach : when one option is selected , the other is sealed off to the player . Outside missions , the player characters rest in a camp , where units can be customized and character growth occurs . Alongside the main story missions are character @-@ specific sub missions relating to different squad members . After the game 's completion , additional episodes are unlocked , some of them having a higher difficulty than those found in the rest of the game . There are also love simulation elements related to the game 's two main heroines , although they take a very minor role .\n The game 's battle system , the BliTZ system , is carried over directly from Valkyria Chronicles . During missions , players select each unit using a top @-@ down perspective of the battlefield map : once a character is selected , the player moves the character around the battlefield in third @-@ person . A character can only act once per @-@ turn , but characters can be granted multiple turns at the expense of other characters ' turns . Each character has a field and pursuit range , which determines the area of effect for regular attacks as well as counterattacks performed against them . Each side 's units can move only a limited number of times before a turn is over . If a unit is knocked out during a turn , they are sent back to the barracks and can no longer participate for the rest of the mission . These missions can be replayed as many times as they wish , and a Roman numeral system is used to indicate how far into the mission a mission is .\n";

/// Run `glbench ppl --model <package-dir> --gguf <original.gguf> [...]`.
pub fn run_ppl(args: PplArgs) -> Result<(), String> {
    eprintln!(
        "loading tokenizer from {} (packages don't embed one yet, ARTX1 OQ3)...",
        args.gguf.display()
    );
    let gguf = GgufFile::open(args.gguf.to_str().ok_or("--gguf path is not valid UTF-8")?)
        .map_err(|e| format!("opening {}: {e}", args.gguf.display()))?;
    let tokenizer = GllmTokenizer::from_gguf_path(&args.gguf.to_string_lossy())
        .map_err(|e| format!("building tokenizer from {}: {e}", args.gguf.display()))?;

    let tokens = tokenizer.encode(WIKITEXT2_SAMPLE, false);
    let total_tokens = tokens.len();
    if total_tokens <= args.context {
        return Err(format!(
            "embedded WikiText-2 sample tokenized to only {total_tokens} tokens, \
             need more than --context ({})",
            args.context
        ));
    }

    eprintln!("loading .gllm package from {} ...", args.model.display());
    let mut engine = GllmEngine::new();
    engine
        .init()
        .map_err(|e| format!("initializing gllm engine: {e}"))?;
    engine
        .load_model(args.model.to_str().ok_or("--model path is not valid UTF-8")?)
        .map_err(|e| format!("loading {}: {e}", args.model.display()))?;

    let (all_log_probs, per_window_ce) = sliding_window_log_probs(&engine, &tokens, args.context, args.stride)?;
    let evaluated_tokens = all_log_probs.len();
    let cross_entropy_mean = -mean(&all_log_probs);
    let perplexity = cross_entropy_mean.exp();

    let session = PplSession {
        model_path: args.model.display().to_string(),
        context_len: args.context,
        stride: args.stride,
        dataset: "wikitext2-sample-embedded",
        total_tokens,
        evaluated_tokens,
        perplexity,
        cross_entropy_mean,
        per_window_ce,
        timestamp: iso8601_now(),
    };

    print!("{}", render_table(&session, &args.model));

    if let Some(out) = &args.out {
        std::fs::write(out, session.to_json().to_pretty())
            .map_err(|e| format!("writing {}: {e}", out.display()))?;
        eprintln!("wrote {}", out.display());
    }

    Ok(())
}

/// Slide a `context`-token window across `tokens` in steps of `stride`,
/// teacher-forcing each window through `engine` and collecting the
/// log-probability of every "new" token (the last `stride` positions of each
/// window, so overlapping windows never double-count a token) — the
/// standard WikiText-2 sliding-window protocol.
///
/// Returns the flat list of per-token log-probs (for the overall mean) and
/// one cross-entropy value per window (for drift inspection across the
/// document).
fn sliding_window_log_probs(
    engine: &GllmEngine,
    tokens: &[u32],
    context: usize,
    stride: usize,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    let mut all_log_probs = Vec::new();
    let mut per_window_ce = Vec::new();
    let mut window_start = 0usize;

    while window_start + context <= tokens.len() {
        let window = &tokens[window_start..window_start + context];
        // Teacher-forced logits: position i's row predicts token i+1. Score
        // one token further than the window so the window's last token has
        // a target to be scored against.
        let extended_end = (window_start + context + 1).min(tokens.len());
        let scoring_len = extended_end - window_start;
        let all_logits = engine
            .score_sequence(&tokens[window_start..window_start + scoring_len])
            .map_err(|e| format!("scoring window at {window_start}: {e}"))?;

        // Only the last `stride` positions of the window are "new" (not
        // already counted by the previous, overlapping window) — except the
        // very first window, which has no previous overlap to skip.
        let new_start = if window_start == 0 { 0 } else { context - stride };

        let mut window_log_probs = Vec::new();
        for i in new_start..window.len() {
            // logits[i] predicts token at absolute position window_start+i+1.
            let target_abs = window_start + i + 1;
            if target_abs >= tokens.len() {
                break;
            }
            let target = tokens[target_abs] as usize;
            let logits = &all_logits[i];
            let lp = log_softmax(logits)
                .get(target)
                .copied()
                .ok_or_else(|| format!("target token {target} out of vocab range ({})", logits.len()))?;
            window_log_probs.push(lp as f64);
        }

        if !window_log_probs.is_empty() {
            per_window_ce.push(-mean(&window_log_probs));
            all_log_probs.extend(window_log_probs);
        }

        window_start += stride;
    }

    if all_log_probs.is_empty() {
        return Err("sliding window produced no evaluated tokens".to_string());
    }

    Ok((all_log_probs, per_window_ce))
}

/// Numerically stable log-softmax: `log(softmax(x))_i = x_i - max(x) - log(sum(exp(x - max(x))))`.
///
/// `pub(crate)`: shared with [`crate::kl_divergence`], which needs the exact
/// same computation on both engines' logits — a second copy here would be
/// the same "two implementations of one thing" risk this whole workspace
/// just got audited for (`architecture/mensura-veritatis-v3/ARTX2-Quant.md`).
pub(crate) fn log_softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f64 = logits.iter().map(|&l| ((l - max) as f64).exp()).sum();
    let log_sum_exp = sum_exp.ln();
    logits.iter().map(|&l| l - max - log_sum_exp as f32).collect()
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn render_table(s: &PplSession, model: &std::path::Path) -> String {
    let mut out = String::new();
    let rule = "\u{2500}".repeat(46);

    out.push_str(&format!("glbench ppl: {}\n", model.display()));
    out.push_str("!! NOT YET RE-VALIDATED — cross-check before citing !!\n");
    out.push_str("   the known garbage-output bug (Q6_K dequant, see\n");
    out.push_str("   notes/issues/gllm-e2e-garbage-output.md) is fixed, but this\n");
    out.push_str("   command hasn't been re-run against a real GQ4A/GQ2A package\n");
    out.push_str("   since — cross-check with glproc::runner::Runner before citing.\n");
    out.push_str(&format!("{rule}\n"));
    out.push_str("  Dataset           wikitext2-sample (embedded)\n");
    out.push_str(&format!(
        "  Tokens            {} total / {} evaluated\n",
        s.total_tokens, s.evaluated_tokens
    ));
    out.push_str(&format!("  Context / Stride  {} / {}\n", s.context_len, s.stride));
    out.push_str(&format!("{rule}\n"));
    out.push_str(&format!("  Cross-entropy     {:.4}\n", s.cross_entropy_mean));
    out.push_str(&format!("  Perplexity        {:.2}\n", s.perplexity));
    out.push_str(&format!("{rule}\n"));
    out
}

/// Manual ISO 8601 UTC timestamp (no `chrono` — glbench's zero-dep rule).
fn iso8601_now() -> String {
    let secs = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    civil_from_unix(secs)
}

/// Convert Unix seconds to an ISO 8601 UTC string (Howard Hinnant's
/// `civil_from_days` algorithm) — the same approach `quant_info.rs` uses, so
/// this is the second, not a third, from-scratch date implementation.
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
    fn log_softmax_matches_reference() {
        let logits = [1.0f32, 2.0, 3.0];
        let lp = log_softmax(&logits);
        // Reference via naive (non-shifted) softmax: exp/sum then ln.
        let sum: f64 = logits.iter().map(|&l| (l as f64).exp()).sum();
        for (i, &l) in logits.iter().enumerate() {
            let reference = ((l as f64).exp() / sum).ln();
            assert!((lp[i] as f64 - reference).abs() < 1e-5, "index {i}: {} vs {reference}", lp[i]);
        }
    }

    #[test]
    fn log_softmax_sums_to_one_in_probability_space() {
        let logits = [0.5f32, -1.2, 3.3, 0.0, -0.4];
        let lp = log_softmax(&logits);
        let sum: f64 = lp.iter().map(|&x| (x as f64).exp()).sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum {sum}");
    }

    #[test]
    fn cross_entropy_to_perplexity() {
        let ce = 3.0f64;
        let ppl = ce.exp();
        assert!((ppl - 20.0855369).abs() < 1e-4, "{ppl}");
    }

    #[test]
    fn ppl_session_serializes() {
        let session = PplSession {
            model_path: "model.gllm".into(),
            context_len: 512,
            stride: 256,
            dataset: "wikitext2-sample-embedded",
            total_tokens: 2041,
            evaluated_tokens: 1785,
            perplexity: 26.74,
            cross_entropy_mean: 3.2847,
            per_window_ce: vec![3.1, 3.3, 3.4],
            timestamp: "2026-07-23T00:00:00Z".into(),
        };
        let back = PplSession::from_json(&session.to_json()).unwrap();
        assert_eq!(session, back);
    }

    #[test]
    fn wikitext_sample_nonempty_and_contains_valkyria() {
        assert!(WIKITEXT2_SAMPLE.len() > 1000, "len={}", WIKITEXT2_SAMPLE.len());
        assert!(WIKITEXT2_SAMPLE.contains("Valkyria"));
    }

    #[test]
    fn iso8601_epoch_is_1970() {
        assert_eq!(civil_from_unix(0), "1970-01-01T00:00:00Z");
    }
}
