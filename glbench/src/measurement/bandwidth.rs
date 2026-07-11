//! Effective-bandwidth estimation.
//!
//! For a memory-bound decode step, the effective bandwidth is the weight bytes
//! streamed per token times the token rate. This is an *estimate* derived from
//! the model footprint and observed tok/s — glbench labels it as such and never
//! presents it as a hardware capability (that lives in the capability table).

/// Estimate effective memory bandwidth in GB/s from the per-token weight bytes
/// and the decode token rate. Returns 0 for a non-positive rate.
pub fn effective_gbs(bytes_per_token: u64, tokens_per_second: f64) -> f64 {
    if tokens_per_second <= 0.0 {
        0.0
    } else {
        (bytes_per_token as f64 * tokens_per_second) / 1e9
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seven_b_q8_estimate() {
        // ~7.7 GB weights at 30 tok/s ~= 231 GB/s effective.
        let gbs = effective_gbs(7_700_000_000, 30.0);
        assert!((gbs - 231.0).abs() < 1.0);
    }

    #[test]
    fn zero_rate_zero_bandwidth() {
        assert_eq!(effective_gbs(1_000_000, 0.0), 0.0);
    }
}
