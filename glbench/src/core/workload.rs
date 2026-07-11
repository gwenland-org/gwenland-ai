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
}

impl Default for WorkloadSpec {
    fn default() -> Self {
        WorkloadSpec {
            engine: "glproc".to_string(),
            model_path: String::new(),
            prompt: String::new(),
            max_new_tokens: 128,
            warmup_iters: 1,
            measure_iters: 3,
            temperature: 0.0, // greedy by default: deterministic timing
            seed: 42,
            kind: WorkloadKind::EndToEnd,
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
            ("warmup_iters", Json::n(self.warmup_iters as f64)),
            ("measure_iters", Json::n(self.measure_iters as f64)),
            ("temperature", Json::n(self.temperature as f64)),
            ("seed", Json::n(self.seed as f64)),
            ("kind", Json::s(self.kind.as_str())),
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
            warmup_iters: field_f64(v, "warmup_iters")? as usize,
            measure_iters: field_f64(v, "measure_iters")? as usize,
            temperature: field_f64(v, "temperature")? as f32,
            seed: field_f64(v, "seed")? as u64,
            kind: WorkloadKind::from_str(&kind_s)
                .ok_or_else(|| format!("unknown workload kind '{kind_s}'"))?,
        })
    }
}
