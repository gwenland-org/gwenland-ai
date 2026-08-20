//! [`VLConvergence`] — did the loss actually go down, and how do we know?
//!
//! # Every number here carries the window it was computed over
//!
//! "Plateaued" and "converged" are not properties of a run; they are properties
//! of a run *plus a window and a threshold*. A report that says "plateau
//! detected" without saying over how many steps and below what delta is not a
//! measurement, it is an opinion with a number attached. So every field that
//! depends on a parameter is archived next to that parameter, and the CLI
//! surfaces the defaults rather than hiding them.
//!
//! # No default target loss
//!
//! `target_loss` has no default and never will (research document §14). "Time
//! to reach a good loss" needs someone to say what good means for their model
//! and their data; picking 0.1 or 0.01 here would invent a claim glbench has no
//! basis for. Ask for `--target-loss` or report `steps_to_target` as absent.

use crate::core::schema::ToJson;
use crate::export::json::Json;
use crate::training::step::VLTrainingStep;

/// Default smoothing factor for the EMA. 0.1 weights roughly the last ten
/// steps, which is short enough to track a real descent and long enough to
/// ignore per-step noise. Archived alongside the value it produced.
pub const DEFAULT_EMA_ALPHA: f64 = 0.1;

/// Default plateau window: how many trailing steps are examined.
pub const DEFAULT_PLATEAU_WINDOW: usize = 20;

/// Default plateau threshold: a run is plateaued when the loss range across the
/// window is below this fraction of the window's mean loss. Relative rather
/// than absolute, because an absolute delta means different things at loss 100
/// and loss 0.001.
pub const DEFAULT_PLATEAU_THRESHOLD: f64 = 1e-3;

/// Convergence behaviour of a training run.
#[derive(Debug, Clone, PartialEq)]
pub struct VLConvergence {
    /// Loss at the first observed step.
    pub first_loss: f32,
    /// Loss at the last observed step.
    pub final_loss: f32,
    /// Lowest loss seen at any observed step.
    pub best_loss: f32,
    /// Step index at which `best_loss` occurred.
    pub best_step: usize,

    /// Least-squares slope of loss against step index. Negative means
    /// descending. Units are loss per step.
    pub slope_per_step: f64,
    /// Exponential moving average of the loss at the final step.
    pub ema_final: f64,
    /// The smoothing factor `ema_final` was computed with.
    pub ema_alpha: f64,

    /// Whether the trailing window looks flat.
    pub plateau_detected: bool,
    /// How many trailing steps `plateau_detected` examined.
    pub plateau_window: usize,
    /// The relative-range threshold `plateau_detected` compared against.
    pub plateau_threshold: f64,

    /// Coefficient of variation (std/mean) over the trailing window — run
    /// stability, independent of scale.
    pub cv: f64,
    /// How many trailing steps `cv` covers.
    pub cv_window: usize,

    /// The target the caller asked about. `None` when `--target-loss` was not
    /// given; there is no default.
    pub target_loss: Option<f32>,
    /// First step index whose loss reached `target_loss`. `None` when there is
    /// no target, or when the run never reached it — the two are distinguished
    /// by `target_loss` being `Some`.
    pub steps_to_target: Option<usize>,
}

/// Compute convergence over the observed steps.
///
/// Returns `None` for an empty series: there is no convergence behaviour to
/// describe, and returning a struct full of zeros would let a reader mistake
/// "no data" for "flat at zero".
pub fn analyze(
    steps: &[VLTrainingStep],
    target_loss: Option<f32>,
    ema_alpha: f64,
    plateau_window: usize,
    plateau_threshold: f64,
) -> Option<VLConvergence> {
    if steps.is_empty() {
        return None;
    }

    let losses: Vec<f64> = steps.iter().map(|s| s.loss as f64).collect();
    let first_loss = steps[0].loss;
    let final_loss = steps[steps.len() - 1].loss;

    // Keyed on the step's own `index`, not its position in the array: with
    // `--step-sample N` the array is thinned, and reporting a position as a
    // step number would name a step that never existed.
    let best = steps
        .iter()
        .min_by(|a, b| a.loss.total_cmp(&b.loss))
        .expect("non-empty");

    let mut ema = losses[0];
    for &l in &losses[1..] {
        ema = ema_alpha * l + (1.0 - ema_alpha) * ema;
    }

    let window = plateau_window.min(losses.len());
    let tail = &losses[losses.len() - window..];
    let mean = tail.iter().sum::<f64>() / tail.len() as f64;
    let (lo, hi) = tail
        .iter()
        .fold((f64::MAX, f64::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
    // Relative range against the window mean. Guarded so a window centred on
    // zero reports "flat" rather than dividing by it.
    let relative_range = if mean.abs() > f64::EPSILON {
        (hi - lo) / mean.abs()
    } else {
        0.0
    };
    // A single-step window is trivially flat and says nothing; refuse to call
    // that a plateau.
    let plateau_detected = window > 1 && relative_range < plateau_threshold;

    let variance = tail.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / tail.len() as f64;
    let cv = if mean.abs() > f64::EPSILON {
        variance.sqrt() / mean.abs()
    } else {
        0.0
    };

    let steps_to_target = target_loss.and_then(|t| {
        steps
            .iter()
            .find(|s| s.loss <= t)
            .map(|s| s.index)
    });

    Some(VLConvergence {
        first_loss,
        final_loss,
        best_loss: best.loss,
        best_step: best.index,
        slope_per_step: least_squares_slope(&losses),
        ema_final: ema,
        ema_alpha,
        plateau_detected,
        plateau_window: window,
        plateau_threshold,
        cv,
        cv_window: window,
        target_loss,
        steps_to_target,
    })
}

/// Least-squares slope of `y` against its own index.
///
/// Against the index rather than the archived `step.index`: with `--step-sample
/// N` the archived indices are thinned, and fitting against them would report a
/// slope per *archived* step while calling it a slope per step.
fn least_squares_slope(y: &[f64]) -> f64 {
    let n = y.len() as f64;
    if y.len() < 2 {
        return 0.0;
    }
    let mean_x = (n - 1.0) / 2.0;
    let mean_y = y.iter().sum::<f64>() / n;
    let mut num = 0.0;
    let mut den = 0.0;
    for (i, &v) in y.iter().enumerate() {
        let dx = i as f64 - mean_x;
        num += dx * (v - mean_y);
        den += dx * dx;
    }
    if den.abs() < f64::EPSILON {
        0.0
    } else {
        num / den
    }
}

impl ToJson for VLConvergence {
    fn to_json(&self) -> Json {
        Json::obj([
            ("first_loss", Json::n(self.first_loss as f64)),
            ("final_loss", Json::n(self.final_loss as f64)),
            ("best_loss", Json::n(self.best_loss as f64)),
            ("best_step", Json::n(self.best_step as f64)),
            ("slope_per_step", Json::n(self.slope_per_step)),
            ("ema_final", Json::n(self.ema_final)),
            ("ema_alpha", Json::n(self.ema_alpha)),
            ("plateau_detected", Json::Bool(self.plateau_detected)),
            ("plateau_window", Json::n(self.plateau_window as f64)),
            ("plateau_threshold", Json::n(self.plateau_threshold)),
            ("cv", Json::n(self.cv)),
            ("cv_window", Json::n(self.cv_window as f64)),
            (
                "target_loss",
                self.target_loss.map(|t| Json::n(t as f64)).unwrap_or(Json::Null),
            ),
            (
                "steps_to_target",
                self.steps_to_target
                    .map(|s| Json::n(s as f64))
                    .unwrap_or(Json::Null),
            ),
        ])
    }
}

impl VLConvergence {
    /// Dotted paths this value emits as `null`, with the honest reason.
    ///
    /// `target_loss` absent is `not_applicable` — nobody asked. A target that
    /// was asked for but never reached is `not_observed`: the instrument was
    /// watching and the event did not occur. Collapsing those two into one
    /// status would lose the distinction a reader most needs.
    pub fn null_paths(&self) -> Vec<(&'static str, crate::core::availability::ENAvailability)> {
        use crate::core::availability::ENAvailability;
        let mut out = Vec::new();
        match (self.target_loss, self.steps_to_target) {
            (None, _) => {
                out.push(("target_loss", ENAvailability::NotApplicable));
                out.push(("steps_to_target", ENAvailability::NotApplicable));
            }
            (Some(_), None) => out.push(("steps_to_target", ENAvailability::NotObserved)),
            (Some(_), Some(_)) => {}
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Slopes and EMAs are plain f64 arithmetic over short series; 1e-9 is far
    /// above the accumulated error and far below any difference under test.
    const TOL_CONV: f64 = 1e-9;

    fn series(losses: &[f32]) -> Vec<VLTrainingStep> {
        losses
            .iter()
            .enumerate()
            .map(|(i, &loss)| VLTrainingStep {
                index: i,
                epoch: 0,
                loss,
                forward_ns: 1,
                backward_ns: 1,
                optimizer_ns: 1,
                total_ns: 3,
                grad_count: 2,
                grad_elements: 4,
                grad_l2_norm: 1.0,
                grad_nan: 0,
                grad_inf: 0,
                lr: 1e-3,
            })
            .collect()
    }

    fn analyze_default(steps: &[VLTrainingStep], target: Option<f32>) -> VLConvergence {
        analyze(
            steps,
            target,
            DEFAULT_EMA_ALPHA,
            DEFAULT_PLATEAU_WINDOW,
            DEFAULT_PLATEAU_THRESHOLD,
        )
        .expect("non-empty series")
    }

    #[test]
    fn an_empty_series_has_no_convergence_rather_than_a_zeroed_one() {
        assert!(analyze(
            &[],
            None,
            DEFAULT_EMA_ALPHA,
            DEFAULT_PLATEAU_WINDOW,
            DEFAULT_PLATEAU_THRESHOLD
        )
        .is_none());
    }

    #[test]
    fn a_descending_run_reports_a_negative_slope() {
        let c = analyze_default(&series(&[1.0, 0.8, 0.6, 0.4, 0.2]), None);
        assert!(c.slope_per_step < 0.0, "got {}", c.slope_per_step);
        // Exactly -0.2 per step for this series.
        assert!((c.slope_per_step + 0.2).abs() < 1e-6, "got {}", c.slope_per_step);
        assert!((c.first_loss - 1.0).abs() < 1e-6);
        assert!((c.final_loss - 0.2).abs() < 1e-6);
        assert!((c.best_loss - 0.2).abs() < 1e-6);
        assert_eq!(c.best_step, 4);
    }

    #[test]
    fn a_diverging_run_reports_a_positive_slope() {
        let c = analyze_default(&series(&[0.2, 0.4, 0.6, 0.8]), None);
        assert!(c.slope_per_step > 0.0, "got {}", c.slope_per_step);
        assert_eq!(c.best_step, 0, "the best step is the first when loss climbs");
    }

    #[test]
    fn best_loss_is_the_minimum_not_the_final_value() {
        // Dips at step 2, then climbs back — a real overfitting shape.
        let c = analyze_default(&series(&[1.0, 0.5, 0.1, 0.4, 0.9]), None);
        assert!((c.best_loss - 0.1).abs() < 1e-6);
        assert_eq!(c.best_step, 2);
        assert!((c.final_loss - 0.9).abs() < 1e-6, "final is not best");
    }

    #[test]
    fn a_flat_run_is_a_plateau_and_a_descending_one_is_not() {
        let flat = analyze_default(&series(&[0.5; 30]), None);
        assert!(flat.plateau_detected);
        assert_eq!(flat.plateau_window, DEFAULT_PLATEAU_WINDOW);
        assert!(flat.cv < TOL_CONV, "a constant series has no variation");

        let descending: Vec<f32> = (0..30).map(|i| 1.0 - i as f32 * 0.03).collect();
        let moving = analyze_default(&series(&descending), None);
        assert!(!moving.plateau_detected, "a descending run is not a plateau");
    }

    /// A window of one is trivially flat. Calling that a plateau would report
    /// "converged" for every single-step run.
    #[test]
    fn a_single_step_run_is_not_called_a_plateau() {
        let c = analyze_default(&series(&[0.5]), None);
        assert!(!c.plateau_detected);
        assert_eq!(c.plateau_window, 1);
        // And the slope of a one-point series is 0, not NaN.
        assert!(c.slope_per_step.abs() < TOL_CONV);
    }

    #[test]
    fn the_window_shrinks_to_the_series_when_the_run_is_shorter() {
        let c = analyze_default(&series(&[1.0, 0.9, 0.8]), None);
        assert_eq!(c.plateau_window, 3, "window must not exceed the series");
        assert_eq!(c.cv_window, 3);
    }

    #[test]
    fn every_parameterised_number_is_archived_with_its_parameter() {
        let c = analyze(&series(&[1.0, 0.5]), None, 0.25, 7, 5e-4).unwrap();
        assert!((c.ema_alpha - 0.25).abs() < TOL_CONV);
        assert!((c.plateau_threshold - 5e-4).abs() < TOL_CONV);
        // Window clamped to the series length, and reported as clamped.
        assert_eq!(c.plateau_window, 2);

        let json = c.to_json();
        for parameter in ["ema_alpha", "plateau_window", "plateau_threshold", "cv_window"] {
            assert!(
                json.get(parameter).is_some(),
                "{parameter} must be archived next to the number it produced"
            );
        }
    }

    #[test]
    fn the_ema_follows_the_series_and_uses_the_stated_alpha() {
        // alpha = 1.0 makes the EMA track the last value exactly.
        let c = analyze(&series(&[1.0, 0.5, 0.25]), None, 1.0, 20, 1e-3).unwrap();
        assert!((c.ema_final - 0.25).abs() < 1e-6, "got {}", c.ema_final);

        // alpha = 0.0 pins it to the first value.
        let c = analyze(&series(&[1.0, 0.5, 0.25]), None, 0.0, 20, 1e-3).unwrap();
        assert!((c.ema_final - 1.0).abs() < 1e-6, "got {}", c.ema_final);
    }

    // -----------------------------------------------------------------------
    // Targets — research §14
    // -----------------------------------------------------------------------

    #[test]
    fn steps_to_target_reports_the_first_step_that_reached_it() {
        let c = analyze_default(&series(&[1.0, 0.8, 0.45, 0.3, 0.45]), Some(0.5));
        assert_eq!(c.steps_to_target, Some(2), "first crossing, not the best step");
    }

    /// A target that was asked for but never reached is `not_observed`, not
    /// `not_applicable`. The instrument was watching; the event did not happen.
    #[test]
    fn a_target_that_was_never_reached_is_not_observed_rather_than_not_applicable() {
        use crate::core::availability::ENAvailability;
        let c = analyze_default(&series(&[1.0, 0.9, 0.8]), Some(0.01));
        assert_eq!(c.steps_to_target, None);

        let nulls = c.null_paths();
        assert_eq!(nulls, vec![("steps_to_target", ENAvailability::NotObserved)]);
    }

    #[test]
    fn no_target_makes_both_target_fields_not_applicable() {
        use crate::core::availability::ENAvailability;
        let c = analyze_default(&series(&[1.0, 0.5]), None);
        assert_eq!(c.target_loss, None);
        assert_eq!(
            c.null_paths(),
            vec![
                ("target_loss", ENAvailability::NotApplicable),
                ("steps_to_target", ENAvailability::NotApplicable),
            ]
        );
    }

    #[test]
    fn a_reached_target_leaves_nothing_to_explain() {
        let c = analyze_default(&series(&[1.0, 0.1]), Some(0.5));
        assert_eq!(c.steps_to_target, Some(1));
        assert!(c.null_paths().is_empty());
    }

    #[test]
    fn the_json_projection_nulls_exactly_what_null_paths_declares() {
        for (steps, target) in [
            (series(&[1.0, 0.5]), None),
            (series(&[1.0, 0.9]), Some(0.01f32)),
            (series(&[1.0, 0.1]), Some(0.5f32)),
        ] {
            let c = analyze_default(&steps, target);
            let json = c.to_json();
            let obj = json.as_obj().unwrap();
            let actual: Vec<&str> = obj
                .iter()
                .filter(|(_, v)| matches!(v, Json::Null))
                .map(|(k, _)| k.as_str())
                .collect();
            let mut declared: Vec<&str> = c.null_paths().iter().map(|(p, _)| *p).collect();
            declared.sort_unstable();
            assert_eq!(actual, declared, "target={target:?}");
        }
    }
}
