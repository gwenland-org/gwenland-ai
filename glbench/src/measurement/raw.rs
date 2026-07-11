//! Raw counter passthrough.
//!
//! The measurement layer's contract is to store facts, not conclusions
//! (DESIGN.md). This module is the narrow seam that turns an engine's
//! `InferOutput` into glbench's `IterationMetrics` without interpreting
//! anything — a deliberate, single, auditable conversion point.

use glcore::engine_trait::InferOutput;

use crate::core::metrics::IterationMetrics;

/// Convert an engine's output into raw iteration metrics. One-to-one field
/// copies; no derived rates, no verdicts.
pub fn from_infer_output(out: &InferOutput) -> IterationMetrics {
    IterationMetrics {
        prompt_tokens: out.prompt_tokens as u64,
        generated_tokens: out.tokens_generated as u64,
        prefill_ms: out.prefill_ms,
        decode_ms: out.generation_ms,
        total_ms: out.elapsed_ms as f64,
    }
}
