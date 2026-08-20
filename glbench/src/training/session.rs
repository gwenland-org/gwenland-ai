//! [`VLTrainingSession`] — the training half of a v2 archive.
//!
//! # Sampling is recorded, not implied
//!
//! `steps_observed`, `steps_archived` and `step_sample_n` sit next to each
//! other on purpose (D-19). A consumer that sees 40 steps in the array and
//! computes a mean over them is right only if those 40 are every step. When
//! they are one in ten, the same arithmetic gives a plausible wrong answer and
//! nothing in the file contradicts it — unless the file says so, which is what
//! these three fields are for.
//!
//! # Nesting, and the direction it goes
//!
//! A training session may hold a post-training [`VLInferenceSession`]
//! (`post_eval`). An inference session may **not** hold a training session —
//! there is no such field, so the type graph is acyclic by construction rather
//! than by convention (D-08).

use crate::core::availability::ENAvailability;
use crate::core::inference::VLInferenceSession;
use crate::core::schema::ToJson;
use crate::export::json::Json;
use crate::training::collector::VLStepBitProfile;
use crate::training::adapter::VLAdapterObservation;
use crate::training::attribution::VLTrainingAttribution;
use crate::training::convergence::VLConvergence;
use crate::training::memory::VLTrainingMemory;
use crate::training::step::{self, VLTrainingStep};

/// A training run, as archived.
#[derive(Debug, Clone)]
pub struct VLTrainingSession {
    /// Archived steps, after D-19 sampling.
    pub steps: Vec<VLTrainingStep>,
    /// Steps the observer was called for, before sampling.
    pub steps_observed: usize,
    /// Steps actually in `steps`. Redundant with `steps.len()` and archived
    /// anyway, so a reader scanning the header does not have to count.
    pub steps_archived: usize,
    /// The D-19 `N` used. 1 means every step is present.
    pub step_sample_n: usize,

    /// Epochs requested.
    pub epochs: usize,
    /// Mean loss per epoch, as `Trainer::train` returned it. Complete
    /// regardless of sampling — it comes from the trainer, not the observer.
    pub epoch_losses: Vec<f32>,

    /// Which optimizer ran.
    pub optimizer: String,

    /// Phase breakdown across the archived steps.
    pub attribution: Option<VLTrainingAttribution>,
    /// Convergence behaviour.
    pub convergence: Option<VLConvergence>,
    /// Memory footprint.
    pub memory: Option<VLTrainingMemory>,
    /// What was being trained.
    pub adapter: Option<VLAdapterObservation>,

    /// Bit profiles of gradients and optimizer state, when a bit scope asked.
    /// Sampled with the same D-19 `N` as the steps, and each tagged with the
    /// step it came from.
    pub bit_profiles: Vec<VLStepBitProfile>,

    /// Inference measured *after* training, in `Unified` mode. Carries
    /// `role: PostTraining` so position is never load-bearing (D-08).
    pub post_eval: Option<VLInferenceSession>,
}

impl VLTrainingSession {
    /// Every dotted path this session emits as `null`, with the honest reason.
    ///
    /// Assembled from the children rather than hard-coded, so a field a child
    /// starts or stops nulling cannot drift away from the map. Paths are
    /// relative to the session root, which is what
    /// [`crate::validation::availability`] walks.
    pub fn null_paths(&self) -> Vec<(String, ENAvailability)> {
        let mut out = Vec::new();

        // F-05: stumman M2 trains one linear layer with no tokenizer, no
        // batching and no data parallelism. The token-denominated and
        // multi-device per-step fields have no subject at all — one entry
        // covers the whole column (D-09's array collapse).
        for path in step::NULL_PATHS {
            out.push((format!("training.steps[].{path}"), ENAvailability::NotApplicable));
        }

        match &self.attribution {
            Some(_) => {}
            None => out.push(("training.attribution".to_string(), ENAvailability::NotObserved)),
        }
        match &self.convergence {
            Some(c) => {
                for (path, status) in c.null_paths() {
                    out.push((format!("training.convergence.{path}"), status));
                }
            }
            None => out.push(("training.convergence".to_string(), ENAvailability::NotObserved)),
        }
        match &self.memory {
            Some(m) => {
                for (path, status) in m.null_paths() {
                    out.push((format!("training.memory.{path}"), status));
                }
            }
            None => out.push(("training.memory".to_string(), ENAvailability::Unavailable)),
        }
        if self.adapter.is_none() {
            out.push(("training.adapter".to_string(), ENAvailability::Unavailable));
        }
        if self.post_eval.is_none() {
            // Only a Unified session has one; for TrainingOnly it is not that
            // the evaluation failed, it is that none was asked for.
            out.push(("training.post_eval".to_string(), ENAvailability::NotApplicable));
        }
        out
    }
}

impl ToJson for VLTrainingSession {
    fn to_json(&self) -> Json {
        let opt = |v: Option<&dyn Fn() -> Json>| match v {
            Some(f) => f(),
            None => Json::Null,
        };
        let _ = opt; // shape kept for symmetry with the sibling modules

        Json::obj([
            (
                "steps",
                Json::Arr(self.steps.iter().map(|s| s.to_json()).collect()),
            ),
            ("steps_observed", Json::n(self.steps_observed as f64)),
            ("steps_archived", Json::n(self.steps_archived as f64)),
            ("step_sample_n", Json::n(self.step_sample_n as f64)),
            ("epochs", Json::n(self.epochs as f64)),
            (
                "epoch_losses",
                Json::Arr(self.epoch_losses.iter().map(|l| Json::n(*l as f64)).collect()),
            ),
            ("optimizer", Json::s(self.optimizer.clone())),
            (
                "attribution",
                self.attribution.as_ref().map(|a| a.to_json()).unwrap_or(Json::Null),
            ),
            (
                "convergence",
                self.convergence.as_ref().map(|c| c.to_json()).unwrap_or(Json::Null),
            ),
            (
                "memory",
                self.memory.as_ref().map(|m| m.to_json()).unwrap_or(Json::Null),
            ),
            (
                "adapter",
                self.adapter.as_ref().map(|a| a.to_json()).unwrap_or(Json::Null),
            ),
            (
                "bit_profiles",
                Json::Arr(
                    self.bit_profiles
                        .iter()
                        .map(|entry| {
                            let b = &entry.scope;
                            let p = &b.profile;
                            Json::obj([
                                ("step_index", Json::n(entry.step_index as f64)),
                                ("scope", Json::s(b.scope.as_str())),
                                ("tensor_name", Json::s(b.tensor_name.clone())),
                                ("count", Json::n(p.count as f64)),
                                ("sign_set_ratio", Json::n(p.sign_set_ratio)),
                                ("exponent_min", Json::n(p.exponent_min as f64)),
                                ("exponent_max", Json::n(p.exponent_max as f64)),
                                ("dynamic_range_used", Json::n(p.dynamic_range_used)),
                                (
                                    "mantissa_entropy_bits",
                                    p.mantissa_entropy_bits.map(Json::n).unwrap_or(Json::Null),
                                ),
                                (
                                    "mantissa_sparse_skipped",
                                    Json::Bool(p.mantissa_sparse_skipped),
                                ),
                                ("zero_count", Json::n(p.zero_count as f64)),
                                ("nan_count", Json::n(p.nan_count as f64)),
                                ("inf_count", Json::n(p.inf_count as f64)),
                            ])
                        })
                        .collect(),
                ),
            ),
            (
                "post_eval",
                self.post_eval.as_ref().map(|p| p.to_json()).unwrap_or(Json::Null),
            ),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::mode::ENInferenceRole;
    use crate::training::convergence;

    fn steps(n: usize) -> Vec<VLTrainingStep> {
        (0..n)
            .map(|i| VLTrainingStep {
                index: i,
                epoch: 0,
                loss: 1.0 - i as f32 * 0.1,
                forward_ns: 1_000,
                backward_ns: 2_000,
                optimizer_ns: 500,
                total_ns: 3_600,
                grad_count: 2,
                grad_elements: 16,
                grad_l2_norm: 0.5,
                grad_nan: 0,
                grad_inf: 0,
                lr: 1e-3,
            })
            .collect()
    }

    fn session(archived: Vec<VLTrainingStep>, observed: usize, n: usize) -> VLTrainingSession {
        VLTrainingSession {
            steps_archived: archived.len(),
            steps: archived,
            steps_observed: observed,
            step_sample_n: n,
            epochs: 1,
            epoch_losses: vec![0.5],
            optimizer: "adamw".to_string(),
            attribution: None,
            convergence: None,
            memory: None,
            adapter: None,
            bit_profiles: Vec::new(),
            post_eval: None,
        }
    }

    /// D-19's whole point: a thinned series must be self-describing.
    #[test]
    fn a_thinned_series_says_so_in_the_archive() {
        let s = session(steps(4), 40, 10);
        let json = s.to_json();
        assert_eq!(json.get("steps_archived").unwrap().as_f64(), Some(4.0));
        assert_eq!(json.get("steps_observed").unwrap().as_f64(), Some(40.0));
        assert_eq!(json.get("step_sample_n").unwrap().as_f64(), Some(10.0));
        assert_eq!(json.get("steps").unwrap().as_arr().unwrap().len(), 4);
    }

    #[test]
    fn a_complete_series_reports_matching_counts_and_a_sample_of_one() {
        let s = session(steps(5), 5, 1);
        assert_eq!(s.steps_archived, s.steps_observed);
        assert_eq!(s.step_sample_n, 1);
    }

    /// F-05: the per-step fields with no subject at M2 are declared once for
    /// the whole column, not once per element.
    #[test]
    fn per_step_null_columns_collapse_to_one_entry_each() {
        let s = session(steps(50), 50, 1);
        let nulls = s.null_paths();
        for path in step::NULL_PATHS {
            let key = format!("training.steps[].{path}");
            let hits = nulls.iter().filter(|(p, _)| *p == key).count();
            assert_eq!(hits, 1, "{key} must be declared once for 50 steps, got {hits}");
            assert!(nulls
                .iter()
                .any(|(p, s)| *p == key && *s == ENAvailability::NotApplicable));
        }
    }

    #[test]
    fn absent_sub_reports_are_declared_with_distinct_reasons() {
        let s = session(steps(2), 2, 1);
        let nulls = s.null_paths();
        let find = |p: &str| nulls.iter().find(|(path, _)| path == p).map(|(_, s)| *s);

        // Nothing to attribute vs nothing collected vs nothing asked for — three
        // different absences, three different statuses.
        assert_eq!(find("training.attribution"), Some(ENAvailability::NotObserved));
        assert_eq!(find("training.memory"), Some(ENAvailability::Unavailable));
        assert_eq!(find("training.post_eval"), Some(ENAvailability::NotApplicable));
    }

    /// The child's own statuses must reach the session map with the right
    /// prefix, or D-10 fails on a path nothing declared.
    #[test]
    fn child_null_paths_are_prefixed_and_carried_up() {
        let mut s = session(steps(3), 3, 1);
        s.convergence = convergence::analyze(
            &s.steps,
            None,
            convergence::DEFAULT_EMA_ALPHA,
            convergence::DEFAULT_PLATEAU_WINDOW,
            convergence::DEFAULT_PLATEAU_THRESHOLD,
        );

        let nulls = s.null_paths();
        assert!(
            nulls
                .iter()
                .any(|(p, st)| p == "training.convergence.target_loss"
                    && *st == ENAvailability::NotApplicable),
            "the convergence child's own status must be carried up, got {nulls:?}"
        );
        // And the parent is no longer declared absent, because it is present.
        assert!(!nulls.iter().any(|(p, _)| p == "training.convergence"));
    }

    #[test]
    fn a_post_eval_is_archived_with_its_role() {
        let mut s = session(steps(2), 2, 1);
        s.post_eval = Some(VLInferenceSession::nested(ENInferenceRole::PostTraining));

        let json = s.to_json();
        let post = json.get("post_eval").unwrap();
        assert_eq!(post.get("role").unwrap().as_str(), Some("post_training"));
        // And it no longer needs a not_applicable entry.
        assert!(!s.null_paths().iter().any(|(p, _)| p == "training.post_eval"));
    }

    #[test]
    fn epoch_losses_are_complete_even_when_steps_are_thinned() {
        let mut s = session(steps(2), 100, 50);
        s.epochs = 4;
        s.epoch_losses = vec![0.9, 0.7, 0.55, 0.5];

        let json = s.to_json();
        let losses = json.get("epoch_losses").unwrap().as_arr().unwrap();
        assert_eq!(
            losses.len(),
            4,
            "epoch losses come from the trainer, so sampling cannot thin them"
        );
    }
}
