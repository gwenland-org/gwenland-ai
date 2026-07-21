//! What to run: the workload specification.
//!
//! A [`WorkloadSpec`] describes the benchmark to execute — which model, which
//! engine, the prompt shape, and the token budgets. It is pure configuration:
//! no results, no timing, no conclusions.

use crate::core::schema::{field_f64, field_str, FromJson, ToJson};
use crate::export::json::Json;

/// The kind of workload to run. glbench measures three fundamental phases plus
/// a sustained-load variant; nothing here optimizes or routes — it only says
/// what to exercise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkloadKind {
    /// Prompt processing (prefill) throughput.
    Prefill,
    /// Token generation (decode) throughput.
    Decode,
    /// Full request: prefill followed by decode.
    EndToEnd,
    /// Sustained repeated requests, for stability/thermal observation.
    Stress,
}

impl WorkloadKind {
    /// Stable lowercase identifier used in archives and CLI flags.
    pub fn as_str(self) -> &'static str {
        match self {
            WorkloadKind::Prefill => "prefill",
            WorkloadKind::Decode => "decode",
            WorkloadKind::EndToEnd => "end_to_end",
            WorkloadKind::Stress => "stress",
        }
    }

    /// Parse from the identifier produced by [`WorkloadKind::as_str`].
    ///
    /// Inherent `Option`-returning parser rather than a `FromStr` impl: the
    /// call sites want `Option`, not a `Result` with an error type, and this
    /// keeps the enum self-contained.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<WorkloadKind> {
        match s {
            "prefill" => Some(WorkloadKind::Prefill),
            "decode" => Some(WorkloadKind::Decode),
            "end_to_end" => Some(WorkloadKind::EndToEnd),
            "stress" => Some(WorkloadKind::Stress),
            _ => None,
        }
    }
}

/// The full description of a benchmark to run.
#[derive(Debug, Clone)]
pub struct WorkloadSpec {
    /// Engine name to run through, e.g. `"glproc"` or `"glcuda"`.
    pub engine: String,
    /// Path to the model file (GGUF or safetensors).
    pub model_path: String,
    /// The prompt text to feed the engine.
    pub prompt: String,
    /// Number of tokens to generate during the measured decode phase.
    pub max_new_tokens: usize,
    /// Timed cold-start iterations, run before warmup, each individually
    /// recorded (never averaged into one figure — see `MeasurementSet::cold`'s
    /// docs for why more than one matters). Reported as median + min/max, the
    /// range itself being part of what a reader needs: "always ~500ms" and
    /// "usually 90ms but once 900ms" look identical as a mean.
    pub cold_iters: usize,
    /// Untimed warmup iterations before measurement begins.
    pub warmup_iters: usize,
    /// Timed measurement iterations; multiple runs feed the statistics.
    pub measure_iters: usize,
    /// Sampling temperature (recorded for reproducibility; 0 = greedy).
    pub temperature: f32,
    /// Fixed RNG seed for deterministic sampling across runs.
    pub seed: u64,
    /// Which phase(s) this workload measures.
    pub kind: WorkloadKind,
    /// Manual override for CoT (thinking-model) detection: `Some(true)` /
    /// `Some(false)` forces the flag; `None` (default) lets the GGUF header
    /// decide. Feeds the entropy assessment, never the engine.
    pub cot_mode: Option<bool>,
    /// Oracle engine to cross-check the first 50 generated tokens against,
    /// e.g. `"glproc"`. `None` (default) skips the check entirely — it loads
    /// and runs a second full engine, roughly doubling the cost of the run,
    /// so it is opt-in rather than automatic. Skipped (not merely trivial)
    /// when equal to `engine`: comparing an engine's output to itself always
    /// matches and would report a check that verified nothing.
    pub verify_against: Option<String>,
}

impl Default for WorkloadSpec {
    fn default() -> Self {
        WorkloadSpec {
            engine: "glproc".to_string(),
            model_path: String::new(),
            prompt: String::new(),
            max_new_tokens: 128,
            cold_iters: 5,
            warmup_iters: 1,
            measure_iters: 3,
            temperature: 0.0, // greedy by default: deterministic timing
            seed: 42,
            kind: WorkloadKind::EndToEnd,
            cot_mode: None,
            verify_against: None,
        }
    }
}

impl ToJson for WorkloadSpec {
    fn to_json(&self) -> Json {
        Json::obj([
            ("engine", Json::s(self.engine.clone())),
            ("model_path", Json::s(self.model_path.clone())),
            ("prompt", Json::s(self.prompt.clone())),
            ("max_new_tokens", Json::n(self.max_new_tokens as f64)),
            ("cold_iters", Json::n(self.cold_iters as f64)),
            ("warmup_iters", Json::n(self.warmup_iters as f64)),
            ("measure_iters", Json::n(self.measure_iters as f64)),
            ("temperature", Json::n(self.temperature as f64)),
            ("seed", Json::n(self.seed as f64)),
            ("kind", Json::s(self.kind.as_str())),
            (
                "cot_mode",
                match self.cot_mode {
                    Some(b) => Json::Bool(b),
                    None => Json::Null,
                },
            ),
            (
                "verify_against",
                match &self.verify_against {
                    Some(s) => Json::s(s.clone()),
                    None => Json::Null,
                },
            ),
        ])
    }
}

impl FromJson for WorkloadSpec {
    fn from_json(v: &Json) -> Result<Self, String> {
        let kind_s = field_str(v, "kind")?;
        Ok(WorkloadSpec {
            engine: field_str(v, "engine")?,
            model_path: field_str(v, "model_path")?,
            prompt: field_str(v, "prompt")?,
            max_new_tokens: field_f64(v, "max_new_tokens")? as usize,
            // Absent from pre-multi-cold-start archives: 0, not the current
            // default of 5 — an archive describes what actually ran, and an
            // old one ran a single untimed-adjacent cold sample, not five.
            cold_iters: v.get("cold_iters").and_then(|n| n.as_f64()).unwrap_or(0.0) as usize,
            warmup_iters: field_f64(v, "warmup_iters")? as usize,
            measure_iters: field_f64(v, "measure_iters")? as usize,
            temperature: field_f64(v, "temperature")? as f32,
            seed: field_f64(v, "seed")? as u64,
            kind: WorkloadKind::from_str(&kind_s)
                .ok_or_else(|| format!("unknown workload kind '{kind_s}'"))?,
            // Optional and absent from pre-v2 archives: absent means "auto".
            cot_mode: v.get("cot_mode").and_then(|b| b.as_bool()),
            // Optional and absent from archives predating this check: absent
            // means "no cross-check was requested", not "was requested and
            // is now missing".
            verify_against: v.get("verify_against").and_then(|s| s.as_str()).map(String::from),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_cold_iters_is_five() {
        assert_eq!(WorkloadSpec::default().cold_iters, 5);
    }

    #[test]
    fn cold_iters_round_trips_through_json() {
        let mut spec = WorkloadSpec::default();
        spec.cold_iters = 7;
        let back = WorkloadSpec::from_json(&spec.to_json()).unwrap();
        assert_eq!(back.cold_iters, 7);
    }

    #[test]
    fn an_archive_with_no_cold_iters_field_reads_as_zero() {
        // Pre-multi-cold-start archives: the field simply is not there. Must
        // read as "this run had no dedicated cold-start phase" (0), not as
        // the current default (5) — an archive describes what actually ran.
        let json = WorkloadSpec::default().to_json();
        let Json::Obj(mut map) = json else { unreachable!() };
        map.remove("cold_iters");
        let back = WorkloadSpec::from_json(&Json::Obj(map)).unwrap();
        assert_eq!(back.cold_iters, 0);
    }

    #[test]
    fn verify_against_defaults_to_none() {
        assert_eq!(WorkloadSpec::default().verify_against, None);
    }

    #[test]
    fn verify_against_round_trips_through_json() {
        let mut spec = WorkloadSpec::default();
        spec.verify_against = Some("glproc".to_string());
        let back = WorkloadSpec::from_json(&spec.to_json()).unwrap();
        assert_eq!(back.verify_against, Some("glproc".to_string()));
    }

    #[test]
    fn an_archive_with_no_verify_against_field_reads_as_none() {
        let json = WorkloadSpec::default().to_json();
        let Json::Obj(mut map) = json else { unreachable!() };
        map.remove("verify_against");
        let back = WorkloadSpec::from_json(&Json::Obj(map)).unwrap();
        assert_eq!(back.verify_against, None);
    }
}
