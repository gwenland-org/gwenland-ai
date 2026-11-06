//! A glbench session against `llama-bench -o json` output.
//!
//! Two tok/s side by side look like a comparison whether or not they measured
//! the same work, and against llama-bench they often did not: its generation
//! test does not sample at all, and starts from an empty context. A T4 run once
//! put those rows next to each other as if they were equal. So each row here
//! says what differs, and a row that differs says so instead of giving a ratio
//! the reader would quote.

use crate::comparison::statistics::Stats;
use crate::core::session::BenchmarkSession;
use crate::export::json::{self, Json};

/// One `llama-bench` test.
#[derive(Debug, Clone, PartialEq)]
pub struct VLLlamaBenchRecord {
    pub n_prompt: u64,
    pub n_gen: u64,
    /// Tokens already in context when the test starts.
    pub n_depth: u64,
    /// Median of the per-repetition rates, or the mean when no samples exist.
    pub tokens_per_second: f64,
    pub model_type: String,
    pub gpu_info: String,
    pub build_commit: String,
}

/// One phase of the head-to-head.
#[derive(Debug, Clone, PartialEq)]
pub struct VLHeadToHeadRow {
    /// `prefill` or `decode`.
    pub phase: &'static str,
    /// glbench median, tokens/second.
    pub ours: f64,
    /// llama-bench median, tokens/second.
    pub theirs: f64,
    /// Every way the two figures measured different work. Empty means a like
    /// for like comparison.
    pub mismatches: Vec<String>,
}

impl VLHeadToHeadRow {
    /// `ours / theirs`; only meaningful when [`Self::mismatches`] is empty.
    pub fn ratio(&self) -> f64 {
        if self.theirs > 0.0 {
            self.ours / self.theirs
        } else {
            0.0
        }
    }
}

/// The whole comparison.
#[derive(Debug, Clone, PartialEq)]
pub struct VLHeadToHead {
    pub rows: Vec<VLHeadToHeadRow>,
    /// Differences that apply to every row.
    pub setup_mismatches: Vec<String>,
    /// llama.cpp build, for the record.
    pub build_commit: String,
}

fn record(v: &Json) -> Option<VLLlamaBenchRecord> {
    let number = |k: &str| v.get(k).and_then(|n| n.as_f64());
    let text = |k: &str| v.get(k).and_then(|s| s.as_str()).unwrap_or_default().to_string();
    let samples: Vec<f64> = v
        .get("samples_ts")
        .and_then(|s| s.as_arr())
        .map(|s| s.iter().filter_map(|n| n.as_f64()).collect())
        .unwrap_or_default();
    let tokens_per_second = if samples.is_empty() {
        number("avg_ts")?
    } else {
        Stats::from_samples(&samples).median
    };
    Some(VLLlamaBenchRecord {
        n_prompt: number("n_prompt")? as u64,
        n_gen: number("n_gen")? as u64,
        n_depth: number("n_depth").unwrap_or(0.0) as u64,
        tokens_per_second,
        model_type: text("model_type"),
        gpu_info: text("gpu_info"),
        build_commit: text("build_commit"),
    })
}

/// Read `llama-bench -o json` output: an array of tests, or one test.
pub fn parse(text: &str) -> Result<Vec<VLLlamaBenchRecord>, String> {
    let value = json::parse(text).map_err(|e| format!("llama-bench output: {e}"))?;
    let records: Vec<VLLlamaBenchRecord> = match value.as_arr() {
        Some(items) => items.iter().filter_map(record).collect(),
        None => record(&value).into_iter().collect(),
    };
    if records.is_empty() {
        return Err("llama-bench output holds no test with n_prompt and n_gen".to_string());
    }
    Ok(records)
}

/// Put a session and llama-bench's tests side by side, naming what differs.
pub fn head_to_head(session: &BenchmarkSession, records: &[VLLlamaBenchRecord]) -> VLHeadToHead {
    let m = &session.measurements;
    let (prompt_tokens, generated) = m
        .iterations
        .first()
        .map_or((0, 0), |i| (i.prompt_tokens, i.generated_tokens));
    let mut rows = Vec::new();

    if let Some(pp) = records.iter().find(|r| r.n_prompt > 0 && r.n_gen == 0) {
        let mut mismatches = Vec::new();
        if pp.n_prompt != prompt_tokens {
            mismatches.push(format!(
                "prompt length: {prompt_tokens} tokens here, {} in llama-bench",
                pp.n_prompt
            ));
        }
        rows.push(VLHeadToHeadRow {
            phase: "prefill",
            ours: Stats::from_samples(&m.prefill_tps_samples()).median,
            theirs: pp.tokens_per_second,
            mismatches,
        });
    }

    if let Some(tg) = records.iter().find(|r| r.n_gen > 0 && r.n_prompt == 0) {
        let mut mismatches = vec![
            "sampling: glbench decode samples each token on the host, llama-bench does not sample"
                .to_string(),
        ];
        if tg.n_depth != prompt_tokens {
            mismatches.push(format!(
                "starting context: {prompt_tokens} tokens here, {} in llama-bench, and decode \
                 cost grows with context",
                tg.n_depth
            ));
        }
        if tg.n_gen != generated {
            mismatches.push(format!(
                "tokens generated: {generated} here, {} in llama-bench",
                tg.n_gen
            ));
        }
        rows.push(VLHeadToHeadRow {
            phase: "decode",
            ours: Stats::from_samples(&m.decode_tps_samples()).median,
            theirs: tg.tokens_per_second,
            mismatches,
        });
    }

    let mut setup_mismatches = Vec::new();
    let first = &records[0];
    if let Some(gpu) = &session.environment.hardware.gpu.name {
        if !first.gpu_info.is_empty() && !first.gpu_info.contains(gpu.as_str()) {
            setup_mismatches.push(format!("device: {gpu} here, {} in llama-bench", first.gpu_info));
        }
    }
    if let Some(quant) = &session.engine.quantization {
        if !first.model_type.is_empty()
            && !first.model_type.to_lowercase().contains(&quant.to_lowercase())
        {
            setup_mismatches.push(format!(
                "quantization: {quant} here, '{}' in llama-bench",
                first.model_type
            ));
        }
    }

    VLHeadToHead { rows, setup_mismatches, build_commit: first.build_commit.clone() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::metrics::{IterationMetrics, MeasurementSet};
    use crate::core::result::SessionMetadata;
    use crate::core::workload::WorkloadSpec;
    use crate::engine::metadata::EngineMetadata;
    use crate::environment::gpu::GpuInfo;
    use crate::environment::hardware::EnvironmentSnapshot;

    /// Trimmed from a real run, `glcuda-t4-wave116-h2h-q8-results.zip`.
    const PREFILL: &str = r#"{
      "build_commit": "4d9176092", "gpu_info": "Tesla T4",
      "model_type": "qwen2 1B Q8_0", "n_prompt": 244, "n_gen": 0, "n_depth": 0,
      "avg_ts": 11335.05,
      "samples_ts": [7999.26, 12128.8, 11935.1, 11848.6, 11658.8]
    }"#;

    fn session() -> BenchmarkSession {
        let mut m = MeasurementSet::default();
        m.iterations.push(IterationMetrics {
            prompt_tokens: 244,
            generated_tokens: 64,
            prefill_ms: 16.0,
            decode_ms: 280.0,
            total_ms: 296.0,
            ttft_ms: None,
        });
        let mut s = BenchmarkSession::new(
            SessionMetadata::new("t"),
            EnvironmentSnapshot::probe(""),
            EngineMetadata {
                name: "glcuda".into(),
                backend: "cuda".into(),
                available: true,
                model_arch: None,
                quantization: Some("Q8_0".into()),
                thinking_capable: None,
            },
            WorkloadSpec::default(),
            m,
        );
        s.environment.hardware.gpu = GpuInfo { name: Some("Tesla T4".into()), ..GpuInfo::default() };
        s
    }

    fn generation(n_gen: u64, n_depth: u64) -> VLLlamaBenchRecord {
        VLLlamaBenchRecord {
            n_prompt: 0,
            n_gen,
            n_depth,
            tokens_per_second: 280.0,
            model_type: "qwen2 1B Q8_0".into(),
            gpu_info: "Tesla T4".into(),
            build_commit: "abc".into(),
        }
    }

    #[test]
    fn a_real_record_parses_and_the_median_ignores_the_cold_first_sample() {
        let records = parse(PREFILL).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].n_prompt, 244);
        assert_eq!(records[0].tokens_per_second, 11848.6);
    }

    #[test]
    fn matching_prefill_is_like_for_like() {
        let h = head_to_head(&session(), &parse(PREFILL).unwrap());
        assert_eq!(h.rows.len(), 1);
        assert!(h.rows[0].mismatches.is_empty(), "{:?}", h.rows[0].mismatches);
        assert!(h.setup_mismatches.is_empty(), "{:?}", h.setup_mismatches);
        assert!(h.rows[0].ratio() > 1.0);
    }

    #[test]
    fn a_different_prompt_length_is_named() {
        let mut records = parse(PREFILL).unwrap();
        records[0].n_prompt = 512;
        let h = head_to_head(&session(), &records);
        assert!(h.rows[0].mismatches[0].contains("512"), "{:?}", h.rows[0].mismatches);
    }

    /// llama-bench never samples, so this row can never be like for like, even
    /// with depth and length matched.
    #[test]
    fn decode_is_never_the_same_work() {
        let h = head_to_head(&session(), &[generation(64, 244)]);
        let row = &h.rows[0];
        assert_eq!(row.phase, "decode");
        assert_eq!(row.mismatches.len(), 1);
        assert!(row.mismatches[0].contains("sampling"));

        let h = head_to_head(&session(), &[generation(128, 0)]);
        assert_eq!(h.rows[0].mismatches.len(), 3, "{:?}", h.rows[0].mismatches);
    }

    #[test]
    fn a_different_quantization_or_device_applies_to_the_whole_comparison() {
        let mut other = generation(64, 244);
        other.model_type = "qwen2 1B Q4_K - Medium".into();
        other.gpu_info = "NVIDIA L4".into();
        let h = head_to_head(&session(), &[other]);
        assert_eq!(h.setup_mismatches.len(), 2, "{:?}", h.setup_mismatches);
    }

    #[test]
    fn output_with_no_usable_test_is_an_error() {
        assert!(parse("[]").is_err());
        assert!(parse("not json").is_err());
    }
}
