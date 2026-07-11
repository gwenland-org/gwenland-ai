//! Performance health score: a single 0..=1 figure summarizing "is this run
//! healthy?" from the facts on hand.
//!
//! Health is a blend of two observable signals: how close decode runs to the
//! hardware ceiling (efficiency), and how stable it is run-to-run (low
//! variance). When no ceiling is known, health reflects stability alone, and
//! the report notes that the number is partial. This is a summary convenience,
//! not a verdict — the raw stats remain the source of truth.

use crate::comparison::statistics::Stats;

/// Compute a health score in 0.0..=1.0 from throughput stats and (optionally)
/// the ceiling efficiency.
pub fn score(decode: &Stats, _prefill: &Stats, ceiling_efficiency: Option<f64>) -> f64 {
    if decode.count == 0 {
        return 0.0;
    }

    // Stability component: 1.0 for perfectly repeatable runs, decaying as the
    // coefficient of variation grows. CV of 0 -> 1.0, CV of 0.25 -> ~0.0.
    let cv = decode.coefficient_of_variation();
    let stability = (1.0 - cv * 4.0).clamp(0.0, 1.0);

    match ceiling_efficiency {
        Some(eff) => {
            // Weight efficiency more heavily than stability: being near the
            // ceiling is the primary "healthy" signal.
            (0.7 * eff + 0.3 * stability).clamp(0.0, 1.0)
        }
        // No ceiling: health is stability only.
        None => stability,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stats(mean: f64, std: f64) -> Stats {
        Stats { count: 3, mean, std_dev: std, ..Default::default() }
    }

    #[test]
    fn zero_samples_zero_health() {
        assert_eq!(score(&Stats::default(), &Stats::default(), Some(0.9)), 0.0);
    }

    #[test]
    fn high_efficiency_stable_is_healthy() {
        let s = stats(30.0, 0.0);
        assert!(score(&s, &s, Some(0.9)) > 0.9);
    }

    #[test]
    fn noisy_run_lowers_health() {
        let stable = score(&stats(30.0, 0.0), &Stats::default(), None);
        let noisy = score(&stats(30.0, 6.0), &Stats::default(), None); // CV 0.2
        assert!(noisy < stable);
    }
}
