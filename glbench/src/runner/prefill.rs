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

/// The least time a prefill workload warms up for, on top of `--warmup`.
///
/// One-token requests are so short that a few of them do not bring a GPU up to
/// speed. On a T4, a prefill run with two warmup iterations measured 36.7 ms for
/// its first three iterations and 17 ms after, inside the statistics. Any
/// end-to-end run hides this, because its first decode warms the device for it.
pub const MIN_WARMUP: std::time::Duration = std::time::Duration::from_secs(1);

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
