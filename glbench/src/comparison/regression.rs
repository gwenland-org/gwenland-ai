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

    /// Parse the stable identifier back, for reading a verdict out of an
    /// archive. `None` on an unknown string rather than a guess.
    /// Inherent `Option`-returning parser rather than a `FromStr` impl,
    /// matching [`crate::core::workload::WorkloadKind::from_str`]: the call
    /// sites want `Option`, not a `Result` with an error type.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<Regression> {
        Some(match s {
            "improved" => Regression::Improved,
            "neutral" => Regression::Neutral,
            "regressed" => Regression::Regressed,
            _ => return None,
        })
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

    #[test]
    fn every_verdict_round_trips_through_its_stable_identifier() {
        for verdict in [Regression::Improved, Regression::Neutral, Regression::Regressed] {
            assert_eq!(Regression::from_str(verdict.as_str()), Some(verdict));
        }
        assert_eq!(Regression::from_str("catastrophic"), None);
    }
}
