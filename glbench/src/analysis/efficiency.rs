//! Efficiency: observed performance as a fraction of a stated ceiling.
//!
//! Thin by design — the ceiling basis lives in [`super::ceiling`]; this module
//! just expresses the ratio in the shapes callers want (percentage, headroom).

/// Efficiency as a fraction and its complement (headroom to the ceiling).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Efficiency {
    /// Observed / ceiling, 0.0..=1.0.
    pub fraction: f64,
}

impl Efficiency {
    /// Build from an observed value and a ceiling, guarding a zero ceiling.
    pub fn new(observed: f64, ceiling: f64) -> Option<Efficiency> {
        if ceiling <= 0.0 {
            return None;
        }
        Some(Efficiency { fraction: (observed / ceiling).clamp(0.0, 1.0) })
    }

    /// As a whole-number percentage.
    pub fn percent(&self) -> f64 {
        self.fraction * 100.0
    }

    /// Remaining headroom to the ceiling, 0.0..=1.0.
    pub fn headroom(&self) -> f64 {
        1.0 - self.fraction
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_of_ceiling() {
        let e = Efficiency::new(160.0, 320.0).unwrap();
        assert!((e.percent() - 50.0).abs() < 1e-9);
        assert!((e.headroom() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn zero_ceiling_is_none() {
        assert!(Efficiency::new(10.0, 0.0).is_none());
    }
}
