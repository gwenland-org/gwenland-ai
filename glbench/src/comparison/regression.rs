//! Regression detection: did the candidate get worse than the baseline by more
//! than a stated threshold?

/// The verdict of a regression check on a single "higher is better" metric.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Regression {
    /// Candidate improved beyond the threshold.
    Improved,
    /// Candidate is within ±threshold of the baseline.
    Neutral,
    /// Candidate regressed beyond the threshold.
    Regressed,
}

impl Regression {
    /// Stable identifier.
    pub fn as_str(self) -> &'static str {
        match self {
            Regression::Improved => "improved",
            Regression::Neutral => "neutral",
            Regression::Regressed => "regressed",
        }
    }
}

/// Judge a relative change (positive = better, for a higher-is-better metric)
/// against a symmetric `threshold` (e.g. 0.05 = 5%).
pub fn regression_verdict(relative_change: f64, threshold: f64) -> Regression {
    let t = threshold.abs();
    if relative_change > t {
        Regression::Improved
    } else if relative_change < -t {
        Regression::Regressed
    } else {
        Regression::Neutral
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verdicts() {
        assert_eq!(regression_verdict(0.10, 0.05), Regression::Improved);
        assert_eq!(regression_verdict(-0.10, 0.05), Regression::Regressed);
        assert_eq!(regression_verdict(0.02, 0.05), Regression::Neutral);
    }
}
