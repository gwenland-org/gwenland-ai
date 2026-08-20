// report.rs — EvalReport struct and JSON serialisation.
//
// Why a dedicated report module?
// The TUI command builds an EvalReport by combining Phase 1 (MetricsResult)
// and Phase 2 (Vec<SampleResult>), then either prints it to the terminal or
// writes it to a JSON file. Centralising the struct definition here ensures
// that the JSON schema is owned by gwen-core, not gwen-tui, consistent with
// the crate boundary rule: ML/data logic lives in gwen-core.

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::eval::metrics::MetricsResult;
use crate::eval::output_eval::SampleResult;

// ── report struct ──────────────────────────────────────────────────────────────

/// The complete eval report: model identity, aggregated metrics, and per-sample
/// inference results.
///
/// This is the top-level struct written to `--output <path>` as JSON and is
/// also the source for the terminal summary printed after the TUI exits.
#[derive(Debug, Serialize, Deserialize)]
pub struct EvalReport {
    /// The model ID passed via `--model`.
    pub model: String,
    /// Path to the dataset file passed via `--dataset`.
    pub dataset: String,
    /// Aggregated scalar metrics from both phases.
    pub metrics: ReportMetrics,
    /// Per-sample inference results from Phase 2.
    pub samples: Vec<SampleResult>,
}

/// Flat struct of all scalar metrics for the JSON `"metrics"` key.
///
/// Flattened rather than nesting MetricsResult + exact_match_rate so that the
/// JSON output has a single clean object with all scalars at one level.
#[derive(Debug, Serialize, Deserialize, Default)]
pub struct ReportMetrics {
    pub avg_loss: f64,
    pub perplexity: f64,
    pub tokens_per_sec: f64,
    pub memory_per_token_mb: f64,
    pub exact_match_rate: f64,
}

impl ReportMetrics {
    /// Combine Phase 1 metrics and the Phase 2 match rate into a single struct.
    pub fn from_parts(m: &MetricsResult, exact_match_rate: f64) -> Self {
        Self {
            avg_loss: m.avg_loss,
            perplexity: m.perplexity,
            tokens_per_sec: m.tokens_per_sec,
            memory_per_token_mb: m.memory_per_token_mb,
            exact_match_rate,
        }
    }
}

// ── file I/O ───────────────────────────────────────────────────────────────────

/// Serialise `report` to pretty-printed JSON and write it to `path`.
///
/// Pretty-printing (two-space indent) is used because eval reports are read
/// by humans in editors and by downstream scripts that expect stable formatting.
/// The extra bytes are negligible given that a 50-sample report is < 50 KB.
pub fn write_report(report: &EvalReport, path: &str) -> Result<()> {
    let json = serde_json::to_string_pretty(report)
        .map_err(|e| anyhow::anyhow!("failed to serialise report: {}", e))?;

    std::fs::write(path, json)
        .map_err(|e| anyhow::anyhow!("failed to write report to '{}': {}", path, e))?;

    Ok(())
}
