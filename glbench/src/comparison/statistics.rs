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
    /// 95% confidence interval half-width for the mean, `mean ± ci95`.
    /// `None` below 3 samples — a t-interval on 1-2 points is not a claim
    /// worth making (0 or 1 degrees of freedom), and this crate will not
    /// print a number that looks precise but is not.
    pub ci95: Option<f64>,
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
            ci95: confidence_interval_95(samples, mean, count),
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

/// 95% confidence-interval half-width for the mean of `samples`, via a
/// Student's-t interval (`mean ± t(0.975, df) * sample_std / sqrt(n)`).
/// `None` below 3 samples.
///
/// Uses the **sample** standard deviation (Bessel-corrected, divide by
/// `n - 1`) here — distinct from [`Stats::std_dev`], which is the
/// population form (divide by `n`) that the rest of this module reports.
/// The correction matters exactly because `n` is always small in a
/// benchmark (a handful of `--iters`): population std dev systematically
/// understates the true spread for a CI, and the difference is largest
/// precisely where it would otherwise be invisible.
///
/// The t critical value has no closed form; this looks it up from a small
/// table of the common small-`df` cases (an ordinary practice for
/// dependency-free small-sample stats, not an approximation of anything
/// this module could compute exactly). `df` beyond the table's range uses
/// 1.96 (the normal-distribution limit as `df -> infinity`), which
/// undershoots the true multiplier by under 5% even at `df = 30` — an
/// acceptable trade against carrying a full t-table.
fn confidence_interval_95(samples: &[f64], mean: f64, count: usize) -> Option<f64> {
    if count < 3 {
        return None;
    }
    let df = count - 1;
    let sample_variance =
        samples.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (count - 1) as f64;
    let sample_std = sample_variance.sqrt();
    let t = t_critical_975(df);
    Some(t * sample_std / (count as f64).sqrt())
}

/// Two-tailed 97.5th-percentile Student's-t critical value (i.e. the
/// multiplier for a 95% confidence interval) at `df` degrees of freedom.
/// Standard table values for small `df`; the normal-distribution limit
/// (1.96) beyond the table.
fn t_critical_975(df: usize) -> f64 {
    const TABLE: [f64; 30] = [
        12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179,
        2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064,
        2.060, 2.056, 2.052, 2.048, 2.045, 2.042,
    ];
    match df {
        0 => f64::NAN, // unreachable: from_samples already refuses count < 3 (df < 2)
        d if d <= TABLE.len() => TABLE[d - 1],
        _ => 1.96,
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

    #[test]
    fn ci95_is_none_below_three_samples() {
        assert_eq!(Stats::from_samples(&[]).ci95, None);
        assert_eq!(Stats::from_samples(&[1.0]).ci95, None);
        assert_eq!(Stats::from_samples(&[1.0, 2.0]).ci95, None);
    }

    #[test]
    fn ci95_matches_a_textbook_example() {
        // n=5, mean=10, deviations [-2,-1,0,1,2]: sample variance (sum-sq /
        // (n-1)) = 10/4 = 2.5, sample std = sqrt(2.5). df=4 -> t=2.776.
        let samples = [8.0, 9.0, 10.0, 11.0, 12.0];
        let s = Stats::from_samples(&samples);
        assert!((s.mean - 10.0).abs() < 1e-9);
        let ci = s.ci95.expect("n=5 must produce a CI");
        let expected = 2.776 * 2.5_f64.sqrt() / (5.0_f64).sqrt();
        assert!((ci - expected).abs() < 1e-3, "got {ci}, expected {expected}");
    }

    #[test]
    fn ci95_uses_sample_not_population_std_dev() {
        // For n=3, population std (divide by n) and sample std (divide by
        // n-1) differ by sqrt(3/2) ≈ 1.2247 — the CI must use the larger
        // (sample) figure, or it silently understates the true interval.
        let samples = [1.0, 2.0, 3.0];
        let s = Stats::from_samples(&samples);
        let sample_std = (1.0_f64).sqrt(); // variance = ((1)^2+(0)^2+(1)^2)/2 = 1
        let expected = t_critical_975(2) * sample_std / (3.0_f64).sqrt();
        assert!((s.ci95.unwrap() - expected).abs() < 1e-9);
    }

    #[test]
    fn t_critical_falls_back_to_the_normal_limit_past_the_table() {
        assert_eq!(t_critical_975(1), 12.706);
        assert_eq!(t_critical_975(30), 2.042);
        assert_eq!(t_critical_975(31), 1.96);
        assert_eq!(t_critical_975(1000), 1.96);
    }

    #[test]
    fn wider_spread_gives_a_wider_interval() {
        let tight = Stats::from_samples(&[10.0, 10.1, 9.9, 10.0, 10.05]);
        let loose = Stats::from_samples(&[10.0, 15.0, 5.0, 12.0, 8.0]);
        assert!(loose.ci95.unwrap() > tight.ci95.unwrap());
    }
}
