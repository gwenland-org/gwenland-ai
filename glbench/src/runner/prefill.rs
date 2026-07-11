//! Prefill-phase focus.
//!
//! The engine reports prefill timing separately in its `InferOutput`, so glbench
//! does not need a distinct execution path to measure it — it reads
//! `prefill_ms` / `prompt_tokens` from each iteration. What *does* matter for a
//! prefill benchmark is that the prompt is long enough to be representative:
//! measuring prefill on a five-token prompt reports launch overhead, not
//! throughput. This module holds that guidance.

/// The minimum prompt length (in tokens) below which a prefill number is more
/// launch-overhead than throughput. Mirrors llama-bench's `pp512` convention:
/// short prompts do not saturate the prefill path.
pub const MIN_REPRESENTATIVE_PROMPT: usize = 128;

/// Whether a prompt of `prompt_tokens` is long enough for a meaningful prefill
/// throughput figure.
pub fn is_representative(prompt_tokens: usize) -> bool {
    prompt_tokens >= MIN_REPRESENTATIVE_PROMPT
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_prompt_flagged() {
        assert!(!is_representative(5));
        assert!(is_representative(512));
    }
}
