//! [`VLTrainingAttribution`] — where a training step's time actually went.
//!
//! The inference side already has this shape: `glcore::telemetry::PhaseProfile`
//! reports stage totals plus an explicit `unattributed_ms`, because a breakdown
//! that silently sums to less than the whole invites the reader to assume it
//! sums to the whole. Training gets the same treatment for the same reason.
//!
//! The three phases are measured back to back inside `train_step`, so
//! `unattributed_ms` is normally the cost of two clock reads and should be near
//! zero. It is archived anyway: if it ever grows, that is the finding.

use crate::core::schema::ToJson;
use crate::export::json::Json;
use crate::training::step::VLTrainingStep;

/// Nanoseconds per millisecond.
const NS_PER_MS: f64 = 1_000_000.0;

/// Phase breakdown across every observed step.
#[derive(Debug, Clone, PartialEq)]
pub struct VLTrainingAttribution {
    /// Steps this breakdown covers.
    pub steps: usize,
    /// Total forward time.
    pub forward_ms: f64,
    /// Total backward time, including `finish_step`.
    pub backward_ms: f64,
    /// Total optimizer time.
    pub optimizer_ms: f64,
    /// Total step time, as measured from the top of the step to the end of the
    /// update.
    pub total_ms: f64,
    /// `total_ms` minus the three phases. Near zero by construction; archived
    /// so that stops being an assumption.
    pub unattributed_ms: f64,

    /// Forward share of `total_ms`, in [0, 1].
    pub forward_share: f64,
    /// Backward share of `total_ms`.
    pub backward_share: f64,
    /// Optimizer share of `total_ms`.
    pub optimizer_share: f64,

    /// Mean wall-clock time per step, in milliseconds.
    pub mean_step_ms: f64,
}

/// Build the breakdown. `None` for an empty series — no steps, no attribution.
pub fn analyze(steps: &[VLTrainingStep]) -> Option<VLTrainingAttribution> {
    if steps.is_empty() {
        return None;
    }
    let sum = |f: fn(&VLTrainingStep) -> u64| -> f64 {
        steps.iter().map(|s| f(s) as f64).sum::<f64>() / NS_PER_MS
    };

    let forward_ms = sum(|s| s.forward_ns);
    let backward_ms = sum(|s| s.backward_ns);
    let optimizer_ms = sum(|s| s.optimizer_ns);
    let total_ms = sum(|s| s.total_ns);

    // Shares are of the measured total, not of the three phases summed. If the
    // phases do not account for the whole step, the shares must show that gap
    // rather than normalise it away.
    let share = |part: f64| if total_ms > 0.0 { part / total_ms } else { 0.0 };

    Some(VLTrainingAttribution {
        steps: steps.len(),
        forward_ms,
        backward_ms,
        optimizer_ms,
        total_ms,
        unattributed_ms: (total_ms - forward_ms - backward_ms - optimizer_ms).max(0.0),
        forward_share: share(forward_ms),
        backward_share: share(backward_ms),
        optimizer_share: share(optimizer_ms),
        mean_step_ms: total_ms / steps.len() as f64,
    })
}

impl ToJson for VLTrainingAttribution {
    fn to_json(&self) -> Json {
        Json::obj([
            ("steps", Json::n(self.steps as f64)),
            ("forward_ms", Json::n(self.forward_ms)),
            ("backward_ms", Json::n(self.backward_ms)),
            ("optimizer_ms", Json::n(self.optimizer_ms)),
            ("total_ms", Json::n(self.total_ms)),
            ("unattributed_ms", Json::n(self.unattributed_ms)),
            ("forward_share", Json::n(self.forward_share)),
            ("backward_share", Json::n(self.backward_share)),
            ("optimizer_share", Json::n(self.optimizer_share)),
            ("mean_step_ms", Json::n(self.mean_step_ms)),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Millisecond sums from integer nanosecond counts; 1e-9 is far above the
    /// division error.
    const TOL_ATTR: f64 = 1e-9;

    fn step(forward: u64, backward: u64, optimizer: u64, total: u64) -> VLTrainingStep {
        VLTrainingStep {
            index: 0,
            epoch: 0,
            loss: 0.5,
            forward_ns: forward,
            backward_ns: backward,
            optimizer_ns: optimizer,
            total_ns: total,
            grad_count: 2,
            grad_elements: 4,
            grad_l2_norm: 1.0,
            grad_nan: 0,
            grad_inf: 0,
            lr: 1e-3,
        }
    }

    #[test]
    fn an_empty_series_has_no_attribution() {
        assert!(analyze(&[]).is_none());
    }

    #[test]
    fn phases_and_shares_describe_the_measured_total() {
        // 1ms forward, 2ms backward, 1ms optimizer, 4ms total.
        let steps = vec![step(1_000_000, 2_000_000, 1_000_000, 4_000_000)];
        let a = analyze(&steps).unwrap();

        assert!((a.forward_ms - 1.0).abs() < TOL_ATTR);
        assert!((a.backward_ms - 2.0).abs() < TOL_ATTR);
        assert!((a.optimizer_ms - 1.0).abs() < TOL_ATTR);
        assert!((a.total_ms - 4.0).abs() < TOL_ATTR);
        assert!((a.unattributed_ms - 0.0).abs() < TOL_ATTR);

        assert!((a.forward_share - 0.25).abs() < TOL_ATTR);
        assert!((a.backward_share - 0.5).abs() < TOL_ATTR);
        assert!((a.optimizer_share - 0.25).abs() < TOL_ATTR);
        assert!((a.mean_step_ms - 4.0).abs() < TOL_ATTR);
    }

    /// The property this module exists to preserve: a gap between the total and
    /// the phases is reported, not normalised away.
    #[test]
    fn a_gap_between_the_total_and_the_phases_is_surfaced_not_hidden() {
        // Phases sum to 3ms but the step took 5ms.
        let steps = vec![step(1_000_000, 1_000_000, 1_000_000, 5_000_000)];
        let a = analyze(&steps).unwrap();

        assert!((a.unattributed_ms - 2.0).abs() < TOL_ATTR, "got {}", a.unattributed_ms);
        // Shares are of the real total, so they deliberately do not sum to 1.
        let summed = a.forward_share + a.backward_share + a.optimizer_share;
        assert!(
            (summed - 0.6).abs() < TOL_ATTR,
            "shares must not be renormalised to hide the gap, got {summed}"
        );
    }

    #[test]
    fn attribution_accumulates_across_steps() {
        let steps = vec![
            step(1_000_000, 2_000_000, 1_000_000, 4_000_000),
            step(1_000_000, 2_000_000, 1_000_000, 4_000_000),
            step(1_000_000, 2_000_000, 1_000_000, 4_000_000),
        ];
        let a = analyze(&steps).unwrap();
        assert_eq!(a.steps, 3);
        assert!((a.total_ms - 12.0).abs() < TOL_ATTR);
        assert!((a.mean_step_ms - 4.0).abs() < TOL_ATTR, "mean is per step, not total");
        // Shares are scale-free, so they are unchanged by repetition.
        assert!((a.backward_share - 0.5).abs() < TOL_ATTR);
    }

    #[test]
    fn a_zero_duration_series_gives_zero_shares_rather_than_nan() {
        let steps = vec![step(0, 0, 0, 0)];
        let a = analyze(&steps).unwrap();
        for share in [a.forward_share, a.backward_share, a.optimizer_share] {
            assert!(share.is_finite(), "share must not be NaN");
            assert!(share.abs() < TOL_ATTR);
        }
        assert!(a.mean_step_ms.is_finite());
    }

    /// `unattributed_ms` is clamped at zero: the phases are measured inside the
    /// total window, so a negative gap would mean the clock went backwards, and
    /// reporting a negative duration is worse than reporting none.
    #[test]
    fn a_negative_gap_is_clamped_rather_than_reported_as_negative_time() {
        let steps = vec![step(3_000_000, 3_000_000, 3_000_000, 1_000_000)];
        let a = analyze(&steps).unwrap();
        assert!(a.unattributed_ms >= 0.0, "got {}", a.unattributed_ms);
    }
}
