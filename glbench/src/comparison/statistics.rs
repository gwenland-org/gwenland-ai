//! Descriptive statistics over sample vectors — pure functions, no state.
//!
//! Shared by the measurement summary and every comparison. Percentiles use
//! linear interpolation between the two nearest ranks (the "linear" / type-7
//! method, matching NumPy's default), so p50 of an even-length sample is the
//! mean of the two middle values rather than an arbitrary pick.

/// A descriptive summary of a sample set.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Stats {
    /// Number of samples.
    pub count: usize,
    /// Arithmetic mean.
    pub mean: f64,
    /// 50th percentile (median).
    pub median: f64,
    /// Minimum sample.
    pub min: f64,
    /// Maximum sample.
    pub max: f64,
    /// Population standard deviation.
    pub std_dev: f64,
    /// 95th percentile.
    pub p95: f64,
    /// 99th percentile.
    pub p99: f64,
}

impl Stats {
    /// Compute statistics over `samples`. An empty input yields all-zeros.
    pub fn from_samples(samples: &[f64]) -> Stats {
        if samples.is_empty() {
            return Stats::default();
        }
        let count = samples.len();
        let mean = samples.iter().sum::<f64>() / count as f64;
        let variance = samples.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / count as f64;
        let std_dev = variance.sqrt();

        let mut sorted = samples.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        Stats {
            count,
            mean,
            median: percentile(&sorted, 50.0),
            min: sorted[0],
            max: sorted[count - 1],
            std_dev,
            p95: percentile(&sorted, 95.0),
            p99: percentile(&sorted, 99.0),
        }
    }

    /// Coefficient of variation (std_dev / mean) — a scale-free measure of
    /// run-to-run noise. Returns 0 when the mean is 0.
    pub fn coefficient_of_variation(&self) -> f64 {
        if self.mean == 0.0 {
            0.0
        } else {
            self.std_dev / self.mean
        }
    }
}

/// The `p`-th percentile (0..=100) of an already-sorted ascending slice, via
/// linear interpolation between closest ranks.
pub fn percentile(sorted: &[f64], p: f64) -> f64 {
    match sorted.len() {
        0 => 0.0,
        1 => sorted[0],
        n => {
            let rank = (p / 100.0) * (n as f64 - 1.0);
            let lo = rank.floor() as usize;
            let hi = rank.ceil() as usize;
            if lo == hi {
                sorted[lo]
            } else {
                let frac = rank - lo as f64;
                sorted[lo] * (1.0 - frac) + sorted[hi] * frac
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stats_of_known_set() {
        let s = Stats::from_samples(&[10.0, 20.0, 30.0, 40.0]);
        assert_eq!(s.count, 4);
        assert!((s.mean - 25.0).abs() < 1e-9);
        assert!((s.median - 25.0).abs() < 1e-9); // interpolated middle
        assert_eq!(s.min, 10.0);
        assert_eq!(s.max, 40.0);
    }

    #[test]
    fn percentile_interpolates() {
        let sorted = [1.0, 2.0, 3.0, 4.0];
        // p95 of 4 points: rank = 0.95*3 = 2.85 -> between idx2(3) and idx3(4).
        assert!((percentile(&sorted, 95.0) - 3.85).abs() < 1e-9);
    }

    #[test]
    fn empty_is_zero() {
        let s = Stats::from_samples(&[]);
        assert_eq!(s, Stats::default());
    }

    #[test]
    fn cv_guards_zero_mean() {
        let s = Stats::from_samples(&[0.0, 0.0]);
        assert_eq!(s.coefficient_of_variation(), 0.0);
    }
}
