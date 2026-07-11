//! Per-phase timeline of a single request.
//!
//! Decomposes one iteration's wall-clock into load-independent phases (prefill,
//! decode) so a renderer can show where time went. The engine reports the two
//! phase durations directly; this is the small struct that carries them plus the
//! derived idle/overhead residual (total minus the accounted phases).

use crate::core::metrics::IterationMetrics;

/// A single request's time broken into phases, all in milliseconds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Timeline {
    /// Prefill (prompt processing) time.
    pub prefill_ms: f64,
    /// Decode (generation) time.
    pub decode_ms: f64,
    /// Wall-clock total.
    pub total_ms: f64,
}

impl Timeline {
    /// Build from a measured iteration.
    pub fn from_iteration(it: &IterationMetrics) -> Timeline {
        Timeline {
            prefill_ms: it.prefill_ms,
            decode_ms: it.decode_ms,
            total_ms: it.total_ms,
        }
    }

    /// Time unaccounted for by prefill + decode (sampling, detokenize, host
    /// overhead). Never negative.
    pub fn overhead_ms(&self) -> f64 {
        (self.total_ms - self.prefill_ms - self.decode_ms).max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overhead_is_residual() {
        let it = IterationMetrics {
            prompt_tokens: 10,
            generated_tokens: 20,
            prefill_ms: 50.0,
            decode_ms: 400.0,
            total_ms: 470.0,
        };
        let tl = Timeline::from_iteration(&it);
        assert!((tl.overhead_ms() - 20.0).abs() < 1e-9);
    }

    #[test]
    fn overhead_never_negative() {
        let it = IterationMetrics {
            prompt_tokens: 0,
            generated_tokens: 0,
            prefill_ms: 100.0,
            decode_ms: 100.0,
            total_ms: 150.0, // less than sum (clock skew)
        };
        assert_eq!(Timeline::from_iteration(&it).overhead_ms(), 0.0);
    }
}
