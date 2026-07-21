//! Thermal throttling detection via CPU clock speed, start vs. end of a run.
//!
//! glbench has no thermal sensor or PMU access (see
//! `analysis::hypothesis`'s own reasoning on this), so it cannot read a die
//! temperature or a throttle flag directly. What it *can* read — on Linux
//! and Windows — is the CPU's reported clock speed, which sustained thermal
//! throttling suppresses. Comparing that number at the start of a session to
//! the same number at the end is an indirect but honest signal: a real
//! frequency drop is measured, never inferred from timing alone.

/// CPU clock speed (MHz) captured at the start and end of one benchmark
/// session, for thermal-throttle detection.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ThermalSnapshot {
    /// Clock speed before the measured iterations began, if readable.
    pub start_mhz: Option<f64>,
    /// Clock speed after the measured iterations finished, if readable.
    pub end_mhz: Option<f64>,
    /// Mean of one clock reading per measured iteration — a coarser but more
    /// representative figure than the start/end pair alone, since a spike or
    /// dip mid-run that recovers by the end would be invisible to
    /// `throttled()` otherwise. `None` when no readings were taken (e.g. an
    /// OS this crate cannot read clock speed on at all).
    pub avg_mhz: Option<f64>,
}

/// A clock drop past this fraction of the start reading is flagged as
/// throttling — matches the ARTX11 spec's threshold: `end < start * 0.9`.
pub const THROTTLE_DROP_FRACTION: f64 = 0.9;

impl ThermalSnapshot {
    /// Whether the readings indicate the CPU throttled during the run.
    /// `false` whenever either reading is missing — an absent measurement is
    /// not evidence of throttling, and must not be treated as if it were.
    pub fn throttled(&self) -> bool {
        match (self.start_mhz, self.end_mhz) {
            (Some(start), Some(end)) if start > 0.0 => end < start * THROTTLE_DROP_FRACTION,
            _ => false,
        }
    }
}

/// Mean of a set of per-iteration clock readings, ignoring any iteration
/// where the probe failed (rather than treating a missing reading as 0 MHz,
/// which would drag the average down for a reason that has nothing to do
/// with the CPU). `None` for an empty input.
pub fn average_mhz(readings: &[f64]) -> Option<f64> {
    if readings.is_empty() {
        return None;
    }
    Some(readings.iter().sum::<f64>() / readings.len() as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_readings_is_never_throttled() {
        assert!(!ThermalSnapshot::default().throttled());
    }

    #[test]
    fn one_missing_reading_is_never_throttled() {
        assert!(!ThermalSnapshot { start_mhz: Some(3000.0), end_mhz: None, avg_mhz: None }.throttled());
        assert!(!ThermalSnapshot { start_mhz: None, end_mhz: Some(2000.0), avg_mhz: None }.throttled());
    }

    #[test]
    fn a_drop_past_the_threshold_is_throttled() {
        // 3000 -> 2600 is a 13.3% drop, past the 10% threshold.
        let t = ThermalSnapshot { start_mhz: Some(3000.0), end_mhz: Some(2600.0), avg_mhz: None };
        assert!(t.throttled());
    }

    #[test]
    fn a_drop_under_the_threshold_is_not_throttled() {
        // 3000 -> 2750 is an 8.3% drop, under the 10% threshold.
        let t = ThermalSnapshot { start_mhz: Some(3000.0), end_mhz: Some(2750.0), avg_mhz: None };
        assert!(!t.throttled());
    }

    #[test]
    fn a_higher_end_reading_is_not_throttled() {
        // Clock speed rose (e.g. it started conservative and boosted) — not
        // a throttle by any definition.
        let t = ThermalSnapshot { start_mhz: Some(2000.0), end_mhz: Some(3000.0), avg_mhz: None };
        assert!(!t.throttled());
    }

    #[test]
    fn exactly_the_threshold_is_not_throttled() {
        // Strictly less-than: the boundary itself is not a violation.
        let t = ThermalSnapshot { start_mhz: Some(3000.0), end_mhz: Some(2700.0), avg_mhz: None };
        assert!(!t.throttled());
    }

    #[test]
    fn average_mhz_of_empty_readings_is_none() {
        assert_eq!(average_mhz(&[]), None);
    }

    #[test]
    fn average_mhz_computes_the_mean() {
        assert_eq!(average_mhz(&[3000.0, 2900.0, 3100.0]), Some(3000.0));
    }

    #[test]
    fn average_mhz_of_one_reading_is_that_reading() {
        assert_eq!(average_mhz(&[2995.0]), Some(2995.0));
    }
}
