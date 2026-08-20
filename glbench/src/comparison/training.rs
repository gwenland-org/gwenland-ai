//! Comparing two training runs.
//!
//! # Configuration first, then outcome
//!
//! Two training runs differ in what they were *asked* to do before they differ
//! in what happened. A report that leads with "final loss 0.41 vs 0.38" and
//! leaves the reader to discover that one had rank 8 and the other rank 2 is
//! the same failure as an inference comparison that hides the quantisation.
//!
//! So [`VLTrainingComparison`] reports the configuration delta as a first-class
//! list, and every outcome delta sits next to it. When the configurations are
//! identical the list is empty and the outcome delta is a clean A/B; when they
//! are not, the reader sees why before they see the number.

use crate::comparison::runs::Delta;
use crate::core::schema::ToJson;
use crate::export::json::Json;
use crate::training::session::VLTrainingSession;

/// One configuration field that differs between two runs.
#[derive(Debug, Clone, PartialEq)]
pub struct VLConfigDelta {
    /// Which field.
    pub field: String,
    /// Its value in the baseline.
    pub baseline: String,
    /// Its value in the candidate.
    pub candidate: String,
}

/// Two training runs, compared.
#[derive(Debug, Clone)]
pub struct VLTrainingComparison {
    /// Configuration fields that differ. Empty means a clean A/B.
    pub config_deltas: Vec<VLConfigDelta>,

    /// Final loss.
    pub final_loss: Delta,
    /// Best loss reached.
    pub best_loss: Delta,
    /// Least-squares slope — how fast each run descended.
    pub slope: Delta,
    /// Mean wall-clock time per step.
    pub mean_step_ms: Delta,
    /// Trainable parameter count.
    pub trainable_parameters: Delta,

    /// Observations, in the same voice the inference comparison uses.
    pub notes: Vec<String>,
}

/// Compare two training sessions.
pub fn compare(baseline: &VLTrainingSession, candidate: &VLTrainingSession) -> VLTrainingComparison {
    let mut config_deltas = Vec::new();
    let mut push = |field: &str, a: String, b: String| {
        if a != b {
            config_deltas.push(VLConfigDelta {
                field: field.to_string(),
                baseline: a,
                candidate: b,
            });
        }
    };

    push("optimizer", baseline.optimizer.clone(), candidate.optimizer.clone());
    push("epochs", baseline.epochs.to_string(), candidate.epochs.to_string());
    push(
        "step_sample_n",
        baseline.step_sample_n.to_string(),
        candidate.step_sample_n.to_string(),
    );
    if let (Some(a), Some(b)) = (&baseline.adapter, &candidate.adapter) {
        push("adapter.kind", a.kind.clone(), b.kind.clone());
        push("adapter.rank", a.rank.to_string(), b.rank.to_string());
        push("adapter.alpha", format!("{:.4}", a.alpha), format!("{:.4}", b.alpha));
        push("adapter.d_in", a.d_in.to_string(), b.d_in.to_string());
        push("adapter.d_out", a.d_out.to_string(), b.d_out.to_string());
    }
    // The learning rate lives on the steps, not the config, because a schedule
    // can move it. Compared at the first step, where a schedule has not yet.
    if let (Some(a), Some(b)) = (baseline.steps.first(), candidate.steps.first()) {
        push("lr@step0", format!("{:.6}", a.lr), format!("{:.6}", b.lr));
    }

    let conv = |s: &VLTrainingSession, f: fn(&crate::training::convergence::VLConvergence) -> f64| {
        s.convergence.as_ref().map(f).unwrap_or(0.0)
    };
    let final_loss = Delta {
        baseline: conv(baseline, |c| c.final_loss as f64),
        candidate: conv(candidate, |c| c.final_loss as f64),
    };
    let best_loss = Delta {
        baseline: conv(baseline, |c| c.best_loss as f64),
        candidate: conv(candidate, |c| c.best_loss as f64),
    };
    let slope = Delta {
        baseline: conv(baseline, |c| c.slope_per_step),
        candidate: conv(candidate, |c| c.slope_per_step),
    };
    let mean_step_ms = Delta {
        baseline: baseline.attribution.as_ref().map(|a| a.mean_step_ms).unwrap_or(0.0),
        candidate: candidate.attribution.as_ref().map(|a| a.mean_step_ms).unwrap_or(0.0),
    };
    let trainable_parameters = Delta {
        baseline: baseline.adapter.as_ref().map(|a| a.trainable_parameters as f64).unwrap_or(0.0),
        candidate: candidate.adapter.as_ref().map(|a| a.trainable_parameters as f64).unwrap_or(0.0),
    };

    let mut notes = Vec::new();
    if config_deltas.is_empty() {
        notes.push("Configurations are identical; the deltas are a clean A/B.".to_string());
    } else {
        notes.push(format!(
            "{} configuration field(s) differ — read the outcome deltas as a comparison of \
             two different runs, not of one change.",
            config_deltas.len()
        ));
    }
    notes.push(format!(
        "Final loss {:.6} -> {:.6} ({:+.1}%).",
        final_loss.baseline,
        final_loss.candidate,
        final_loss.relative() * 100.0
    ));
    // Loss is lower-is-better, the opposite of every throughput metric in this
    // crate, so the direction is spelled out rather than left to a sign.
    if final_loss.candidate < final_loss.baseline {
        notes.push("Candidate reached a lower final loss.".to_string());
    } else if final_loss.candidate > final_loss.baseline {
        notes.push("Candidate reached a higher final loss.".to_string());
    }
    if trainable_parameters.baseline != trainable_parameters.candidate {
        notes.push(format!(
            "Candidate trains {:.2}x the parameters.",
            trainable_parameters.ratio()
        ));
    }

    VLTrainingComparison {
        config_deltas,
        final_loss,
        best_loss,
        slope,
        mean_step_ms,
        trainable_parameters,
        notes,
    }
}

impl ToJson for VLTrainingComparison {
    fn to_json(&self) -> Json {
        let delta = |d: &Delta| {
            Json::obj([
                ("baseline", Json::n(d.baseline)),
                ("candidate", Json::n(d.candidate)),
                ("relative", Json::n(d.relative())),
                ("ratio", Json::n(d.ratio())),
            ])
        };
        Json::obj([
            (
                "config_deltas",
                Json::Arr(
                    self.config_deltas
                        .iter()
                        .map(|c| {
                            Json::obj([
                                ("field", Json::s(c.field.clone())),
                                ("baseline", Json::s(c.baseline.clone())),
                                ("candidate", Json::s(c.candidate.clone())),
                            ])
                        })
                        .collect(),
                ),
            ),
            ("final_loss", delta(&self.final_loss)),
            ("best_loss", delta(&self.best_loss)),
            ("slope", delta(&self.slope)),
            ("mean_step_ms", delta(&self.mean_step_ms)),
            ("trainable_parameters", delta(&self.trainable_parameters)),
            (
                "notes",
                Json::Arr(self.notes.iter().map(|n| Json::s(n.clone())).collect()),
            ),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::adapter::VLAdapterObservation;
    use crate::training::step::VLTrainingStep;
    use crate::training::{attribution, convergence};

    fn steps(losses: &[f32], lr: f64) -> Vec<VLTrainingStep> {
        losses
            .iter()
            .enumerate()
            .map(|(i, &loss)| VLTrainingStep {
                index: i,
                epoch: 0,
                loss,
                forward_ns: 1_000_000,
                backward_ns: 2_000_000,
                optimizer_ns: 1_000_000,
                total_ns: 4_000_000,
                grad_count: 2,
                grad_elements: 16,
                grad_l2_norm: 0.5,
                grad_nan: 0,
                grad_inf: 0,
                lr,
            })
            .collect()
    }

    fn session(losses: &[f32], rank: usize, lr: f64) -> VLTrainingSession {
        let s = steps(losses, lr);
        VLTrainingSession {
            steps_archived: s.len(),
            steps_observed: s.len(),
            step_sample_n: 1,
            epochs: 1,
            epoch_losses: vec![losses[losses.len() - 1]],
            optimizer: "adamw".to_string(),
            attribution: attribution::analyze(&s),
            convergence: convergence::analyze(
                &s,
                None,
                convergence::DEFAULT_EMA_ALPHA,
                convergence::DEFAULT_PLATEAU_WINDOW,
                convergence::DEFAULT_PLATEAU_THRESHOLD,
            ),
            memory: None,
            adapter: Some(VLAdapterObservation::lora(64, 64, rank, rank as f32)),
            bit_profiles: Vec::new(),
            post_eval: None,
            steps: s,
        }
    }

    #[test]
    fn identical_configurations_produce_an_empty_delta_list() {
        let a = session(&[1.0, 0.5, 0.25], 4, 1e-3);
        let b = session(&[1.0, 0.4, 0.20], 4, 1e-3);
        let c = compare(&a, &b);

        assert!(c.config_deltas.is_empty(), "got {:?}", c.config_deltas);
        assert!(c.notes.iter().any(|n| n.contains("clean A/B")));
    }

    /// The property this module exists for: a configuration difference is
    /// surfaced before the outcome number.
    #[test]
    fn a_rank_change_is_reported_as_a_configuration_delta() {
        let a = session(&[1.0, 0.5], 2, 1e-3);
        let b = session(&[1.0, 0.4], 8, 1e-3);
        let c = compare(&a, &b);

        let ranks: Vec<_> = c.config_deltas.iter().filter(|d| d.field == "adapter.rank").collect();
        assert_eq!(ranks.len(), 1);
        assert_eq!(ranks[0].baseline, "2");
        assert_eq!(ranks[0].candidate, "8");
        assert!(
            c.notes.iter().any(|n| n.contains("configuration field")),
            "the reader must be warned before reading the deltas: {:?}",
            c.notes
        );
        // Rank also changes the parameter count, and that is reported.
        assert!(c.trainable_parameters.candidate > c.trainable_parameters.baseline);
        assert!(c.notes.iter().any(|n| n.contains("4.00x")), "got {:?}", c.notes);
    }

    #[test]
    fn a_learning_rate_change_is_caught_from_the_first_step() {
        let a = session(&[1.0, 0.5], 4, 1e-3);
        let b = session(&[1.0, 0.5], 4, 1e-2);
        let c = compare(&a, &b);
        assert!(
            c.config_deltas.iter().any(|d| d.field == "lr@step0"),
            "got {:?}",
            c.config_deltas
        );
    }

    /// Loss is lower-is-better, unlike every throughput metric in this crate.
    /// The direction must be stated, not inferred from a sign.
    #[test]
    fn the_direction_of_a_loss_change_is_spelled_out() {
        let better = compare(&session(&[1.0, 0.5], 4, 1e-3), &session(&[1.0, 0.2], 4, 1e-3));
        assert!(
            better.notes.iter().any(|n| n.contains("lower final loss")),
            "got {:?}",
            better.notes
        );

        let worse = compare(&session(&[1.0, 0.2], 4, 1e-3), &session(&[1.0, 0.5], 4, 1e-3));
        assert!(
            worse.notes.iter().any(|n| n.contains("higher final loss")),
            "got {:?}",
            worse.notes
        );
    }

    #[test]
    fn outcome_deltas_carry_the_measured_values() {
        let a = session(&[1.0, 0.6], 4, 1e-3);
        let b = session(&[1.0, 0.3], 4, 1e-3);
        let c = compare(&a, &b);

        assert!((c.final_loss.baseline - 0.6).abs() < 1e-6);
        assert!((c.final_loss.candidate - 0.3).abs() < 1e-6);
        assert!((c.best_loss.candidate - 0.3).abs() < 1e-6);
        // Both descend, so both slopes are negative; the candidate falls faster.
        assert!(c.slope.candidate < c.slope.baseline);
        assert!(c.mean_step_ms.baseline > 0.0);
    }

    #[test]
    fn a_session_with_no_convergence_report_compares_without_panicking() {
        let mut a = session(&[1.0, 0.5], 4, 1e-3);
        a.convergence = None;
        a.attribution = None;
        let b = session(&[1.0, 0.4], 4, 1e-3);

        let c = compare(&a, &b);
        assert!(c.final_loss.baseline.is_finite());
        assert!(c.mean_step_ms.baseline.is_finite());
    }

    #[test]
    fn the_json_projection_carries_the_config_deltas() {
        let c = compare(&session(&[1.0, 0.5], 2, 1e-3), &session(&[1.0, 0.4], 8, 1e-3));
        let json = c.to_json();
        let deltas = json.get("config_deltas").unwrap().as_arr().unwrap();
        assert!(!deltas.is_empty());
        assert!(deltas
            .iter()
            .any(|d| d.get("field").and_then(|f| f.as_str()) == Some("adapter.rank")));
        assert!(json.get("final_loss").unwrap().get("relative").is_some());
    }
}
