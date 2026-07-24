//! Accuracy-vs-performance comparison: throughput facts from a `run` session
//! joined against numerical-accuracy facts from a `kl-div` or `ppl` session —
//! two kinds of session that, before this, could only ever be viewed
//! separately (a reader had to open both JSON files and cross-reference by
//! hand).
//!
//! [Vetted per `RESEARCH_REQUIREMENTS.md`'s 8 mandatory questions]
//! 1. What problem: a faster configuration is only a real win if it didn't
//!    also get less accurate — but `run`'s tok/s and `kl-div`/`ppl`'s
//!    divergence numbers live in separate archives with no shared view.
//! 2. Who benefits: anyone deciding between two quantizations, or checking
//!    that a "speedup" changeset didn't quietly trade away numerical fidelity
//!    — the exact question this whole session's Q6_K investigation started
//!    from (a garbage-output bug that a throughput number alone would never
//!    have shown).
//! 3. Production/research use: reporting a speed/quality Pareto point
//!    together is standard practice in any inference-engine comparison
//!    (this is literally how llama.cpp's own quantization tables are read).
//! 4. How calculated: no new measurement — pure join of two already-written
//!    JSON archives on `model_path`, surfacing decode/prefill tok/s next to
//!    whichever accuracy figure the second archive contains.
//! 5. Reproducible: yes, deterministic given the same two files.
//! 6. Actionable: directly answers "is this configuration actually a better
//!    trade" instead of requiring a reader to open two files and do the
//!    cross-reference by hand.
//! 7. Lightweight: no new engine hook, no new measurement — pure
//!    report-level composition of existing archives.
//! 8. Philosophy: lives in `comparison`, alongside `runs`/`quantization`,
//!    which already compare two sessions along one axis — this compares two
//!    *different kinds* of session along the accuracy/performance axis, and
//!    reports facts side by side, no verdict beyond a model-path mismatch
//!    warning (the one thing that would make the comparison meaningless).

use crate::core::session::BenchmarkSession;
use crate::export::json::Json;

/// The numerical-accuracy figure pulled from a `kl-div` or `ppl` archive.
/// `Unknown` when the JSON matches neither shape — surfaced as a fact
/// ("could not identify accuracy figures"), not silently dropped.
#[derive(Debug, Clone, PartialEq)]
pub enum AccuracyFigure {
    KlDivergence { mean: f64, max: f64, tokens_compared: usize },
    Perplexity { value: f64, evaluated_tokens: usize },
    Unknown,
}

/// The joined report.
#[derive(Debug, Clone, PartialEq)]
pub struct AccuracyVsPerf {
    pub run_model_path: String,
    pub accuracy_model_path: String,
    /// `false` when the two archives look like they were not run against
    /// the same model — the join is still produced, but flagged.
    pub model_paths_match: bool,
    pub engine: String,
    pub decode_tps_mean: f64,
    pub prefill_tps_mean: f64,
    pub accuracy: AccuracyFigure,
}

/// Join a `run` session with a `kl-div`/`ppl` accuracy archive.
pub fn join(run: &BenchmarkSession, accuracy_json: &Json) -> AccuracyVsPerf {
    use crate::comparison::statistics::Stats;

    let decode = Stats::from_samples(&run.measurements.decode_tps_samples());
    let prefill = Stats::from_samples(&run.measurements.prefill_tps_samples());

    let accuracy_model_path = accuracy_json.get("model_path").and_then(|s| s.as_str()).unwrap_or("").to_string();

    let accuracy = if let (Some(mean), Some(max)) =
        (accuracy_json.get("kl_mean").and_then(|n| n.as_f64()), accuracy_json.get("kl_max").and_then(|n| n.as_f64()))
    {
        AccuracyFigure::KlDivergence {
            mean,
            max,
            tokens_compared: accuracy_json.get("tokens_compared").and_then(|n| n.as_f64()).unwrap_or(0.0) as usize,
        }
    } else if let Some(value) = accuracy_json.get("perplexity").and_then(|n| n.as_f64()) {
        AccuracyFigure::Perplexity {
            value,
            evaluated_tokens: accuracy_json.get("evaluated_tokens").and_then(|n| n.as_f64()).unwrap_or(0.0) as usize,
        }
    } else {
        AccuracyFigure::Unknown
    };

    AccuracyVsPerf {
        run_model_path: run.workload.model_path.clone(),
        model_paths_match: paths_match(&run.workload.model_path, &accuracy_model_path),
        accuracy_model_path,
        engine: run.engine.name.clone(),
        decode_tps_mean: decode.mean,
        prefill_tps_mean: prefill.mean,
        accuracy,
    }
}

/// Compare by final path component, not the full path — a `run` archive and
/// a `kl-div` archive are very likely to have been invoked with
/// differently-rooted but equivalent paths (e.g. relative vs absolute, or a
/// `.gguf` file vs a `.gllm` package directory for the same model), and a
/// full-string compare would false-flag the common case.
///
/// Deliberately **not** `Path::file_stem()`: model names routinely contain
/// their own dots (`qwen2.5-0.5b-instruct-q4_k_m`), and `file_stem` splits at
/// the *last* dot regardless of whether it is a real extension — on a bare
/// package-directory name with no extension at all, that would truncate
/// after `0` (the dot inside `0.5b`) instead of leaving the name alone. Only
/// a literal, known trailing `.gguf` is stripped; everything else is
/// compared as the full final path component.
fn paths_match(a: &str, b: &str) -> bool {
    if a.is_empty() || b.is_empty() {
        return false;
    }
    let normalize = |p: &str| -> String {
        let name = std::path::Path::new(p).file_name().and_then(|s| s.to_str()).unwrap_or(p);
        name.strip_suffix(".gguf").unwrap_or(name).to_lowercase()
    };
    normalize(a) == normalize(b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::metrics::{IterationMetrics, MeasurementSet};
    use crate::core::result::SessionMetadata;
    use crate::core::workload::WorkloadSpec;
    use crate::engine::metadata::EngineMetadata;
    use crate::environment::hardware::EnvironmentSnapshot;

    fn sample_run(model_path: &str) -> BenchmarkSession {
        let mut m = MeasurementSet::default();
        m.iterations.push(IterationMetrics {
            prompt_tokens: 10,
            generated_tokens: 20,
            prefill_ms: 5.0,
            decode_ms: 100.0,
            total_ms: 105.0,
        });
        let mut spec = WorkloadSpec::default();
        spec.model_path = model_path.to_string();
        BenchmarkSession::new(
            SessionMetadata::new("test"),
            EnvironmentSnapshot::probe(""),
            EngineMetadata {
                name: "glproc".into(),
                backend: "cpu".into(),
                available: true,
                model_arch: None,
                quantization: None,
                thinking_capable: None,
            },
            spec,
            m,
        )
    }

    #[test]
    fn recognizes_kl_divergence_shape() {
        let run = sample_run("/models/qwen2.5-0.5b.gguf");
        let accuracy = Json::obj([
            ("model_path", Json::s("qwen2.5-0.5b".to_string())),
            ("kl_mean", Json::n(0.998597)),
            ("kl_max", Json::n(7.515453)),
            ("tokens_compared", Json::n(64.0)),
        ]);
        let joined = join(&run, &accuracy);
        assert_eq!(
            joined.accuracy,
            AccuracyFigure::KlDivergence { mean: 0.998597, max: 7.515453, tokens_compared: 64 }
        );
        assert!(joined.model_paths_match);
    }

    #[test]
    fn recognizes_perplexity_shape() {
        let run = sample_run("/models/qwen2.5-0.5b.gguf");
        let accuracy = Json::obj([
            ("model_path", Json::s("qwen2.5-0.5b".to_string())),
            ("perplexity", Json::n(12.4)),
            ("evaluated_tokens", Json::n(512.0)),
        ]);
        let joined = join(&run, &accuracy);
        assert_eq!(joined.accuracy, AccuracyFigure::Perplexity { value: 12.4, evaluated_tokens: 512 });
    }

    #[test]
    fn unrecognized_shape_is_unknown_not_a_panic() {
        let run = sample_run("/models/x.gguf");
        let accuracy = Json::obj([("something_else", Json::n(1.0))]);
        let joined = join(&run, &accuracy);
        assert_eq!(joined.accuracy, AccuracyFigure::Unknown);
    }

    #[test]
    fn mismatched_model_stems_are_flagged_not_hidden() {
        let run = sample_run("/models/qwen2.5-0.5b.gguf");
        let accuracy = Json::obj([("model_path", Json::s("qwen3-1.7b".to_string())), ("perplexity", Json::n(9.0))]);
        let joined = join(&run, &accuracy);
        assert!(!joined.model_paths_match);
    }

    #[test]
    fn matches_across_gguf_vs_package_dir_naming() {
        // A .gguf path (run's oracle) vs a package directory (kl-div's
        // .gllm target) for the "same" model in casual usage — the stem
        // compare should not false-flag this as a mismatch.
        assert!(paths_match("/models/qwen2.5-0.5b-instruct-q4_k_m.gguf", "qwen2.5-0.5b-instruct-q4_k_m"));
    }
}
