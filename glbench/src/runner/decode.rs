//! Decode-phase focus.
//!
//! Decode throughput is the headline metric for interactive inference. As with
//! prefill, the engine times it separately (`generation_ms` / `tokens_generated`),
//! so glbench reads rather than re-measures. The one policy that matters here is
//! the token budget: too few generated tokens and per-token variance dominates.

/// A decode measurement wants enough generated tokens that per-token jitter
/// averages out. Below this the decode tok/s is dominated by the first-token
/// and last-token edge effects.
pub const MIN_DECODE_TOKENS: usize = 32;

/// Whether `generated_tokens` is enough for a stable decode throughput figure.
pub fn is_stable_budget(generated_tokens: usize) -> bool {
    generated_tokens >= MIN_DECODE_TOKENS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_budget_flagged() {
        assert!(!is_stable_budget(4));
        assert!(is_stable_budget(128));
    }
}
