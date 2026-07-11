//! Throughput conversions: counts + durations -> tokens/second.
//!
//! One place for the rate math so "tokens per second" means the same thing
//! everywhere (guarding zero durations identically).

/// Tokens per second from a token count and a millisecond duration.
pub fn tokens_per_second(tokens: u64, ms: f64) -> f64 {
    if ms <= 0.0 {
        0.0
    } else {
        tokens as f64 / (ms / 1e3)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_rate() {
        assert!((tokens_per_second(100, 1000.0) - 100.0).abs() < 1e-9);
    }

    #[test]
    fn zero_time_is_zero() {
        assert_eq!(tokens_per_second(100, 0.0), 0.0);
    }
}
