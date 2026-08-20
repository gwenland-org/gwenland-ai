//! [`VLInferenceSession`] — an inference run, tagged with the role it plays
//! (D-08).
//!
//! # What this does and does not hold
//!
//! For an `InferenceOnly` session the measured facts stay exactly where v1 put
//! them: `BenchmarkSession::measurements`, `::behavior`, `::analysis`. This type
//! carries only `role: Standalone` and leaves its own fields `None`.
//!
//! That is a compatibility compromise, and it was chosen rather than overlooked.
//! Moving `measurements` down inside this struct would rewrite `runner`,
//! `analysis`, `comparison`, `validation`, both renderers and all three
//! exporters, for a purely cosmetic gain — and the "single source of truth"
//! property (`glbench/DESIGN.md` §4) holds either way.
//!
//! The fields *are* populated for a nested session: in `Unified` mode the outer
//! run carries `role: PreTraining` with its own measurements, and the one inside
//! the training session carries `role: PostTraining`. That is the case position
//! alone could not disambiguate, which is why the role exists.
//!
//! # Recursion guard
//!
//! A training session may contain a `VLInferenceSession`. A
//! `VLInferenceSession` may **not** contain a training session — there is no
//! such field, so the type graph is acyclic by construction rather than by
//! convention.

use crate::analysis::summary::AnalysisReport;
use crate::behavior::BehaviorReport;
use crate::core::metrics::MeasurementSet;
use crate::core::mode::ENInferenceRole;
use crate::core::schema::ToJson;
use crate::export::json::Json;

/// One inference run and the role it plays in its session.
#[derive(Debug, Clone)]
pub struct VLInferenceSession {
    /// Where this run sits relative to training.
    pub role: ENInferenceRole,
    /// Raw measured facts. `Some` only for a nested session; for a standalone
    /// run the facts live at `BenchmarkSession::measurements`.
    pub measurements: Option<MeasurementSet>,
    /// What the model did. `Some` only for a nested session.
    pub behavior: Option<BehaviorReport>,
    /// Derived analysis over this run's measurements. `Some` only for a nested
    /// session.
    pub analysis: Option<AnalysisReport>,
}

impl VLInferenceSession {
    /// The envelope for a plain inference session: a role and nothing else,
    /// because the facts are at the top level of the session.
    pub fn standalone() -> VLInferenceSession {
        VLInferenceSession {
            role: ENInferenceRole::Standalone,
            measurements: None,
            behavior: None,
            analysis: None,
        }
    }

    /// The envelope for a nested run, which carries its own facts.
    pub fn nested(role: ENInferenceRole) -> VLInferenceSession {
        VLInferenceSession { role, measurements: None, behavior: None, analysis: None }
    }

    /// Parse back from JSON.
    ///
    /// Only `role` and `measurements` are reconstructed. `behavior` and
    /// `analysis` follow the same policy as their top-level counterparts in
    /// [`crate::core::session::BenchmarkSession::from_json`]: derived reports
    /// are recomputed on demand rather than trusted from disk, so a parser for
    /// them could only ever agree with the writer.
    pub fn from_json(v: &Json) -> Result<VLInferenceSession, String> {
        use crate::core::schema::FromJson;

        let role_str = v
            .get("role")
            .and_then(|r| r.as_str())
            .ok_or_else(|| "inference session has no 'role' string".to_string())?;
        let role = ENInferenceRole::from_str(role_str)
            .ok_or_else(|| format!("unknown inference role '{role_str}'"))?;

        let measurements = match v.get("measurements") {
            Some(m) if !matches!(m, Json::Null) => Some(MeasurementSet::from_json(m)?),
            _ => None,
        };

        Ok(VLInferenceSession { role, measurements, behavior: None, analysis: None })
    }
}

impl ToJson for VLInferenceSession {
    fn to_json(&self) -> Json {
        Json::obj([
            ("role", Json::s(self.role.as_str())),
            (
                "measurements",
                self.measurements.as_ref().map(|m| m.to_json()).unwrap_or(Json::Null),
            ),
            (
                "behavior",
                self.behavior
                    .as_ref()
                    .map(crate::core::session::behavior_json)
                    .unwrap_or(Json::Null),
            ),
            (
                "analysis",
                self.analysis.as_ref().map(|a| a.to_json()).unwrap_or(Json::Null),
            ),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::metrics::{IterationMetrics, MeasurementSet};

    #[test]
    fn a_standalone_envelope_carries_a_role_and_no_facts_of_its_own() {
        let s = VLInferenceSession::standalone();
        assert_eq!(s.role, ENInferenceRole::Standalone);
        // The facts live at the top level of the session, not here.
        assert!(s.measurements.is_none());
        assert!(s.behavior.is_none());
        assert!(s.analysis.is_none());
    }

    #[test]
    fn a_standalone_envelope_round_trips_through_json() {
        let s = VLInferenceSession::standalone();
        let back = VLInferenceSession::from_json(&s.to_json()).unwrap();
        assert_eq!(back.role, ENInferenceRole::Standalone);
        assert!(back.measurements.is_none());
    }

    #[test]
    fn a_nested_session_round_trips_its_role_and_its_own_measurements() {
        let mut m = MeasurementSet::default();
        m.iterations.push(IterationMetrics {
            prompt_tokens: 8,
            generated_tokens: 16,
            prefill_ms: 4.0,
            decode_ms: 80.0,
            total_ms: 84.0,
        });
        let mut s = VLInferenceSession::nested(ENInferenceRole::PostTraining);
        s.measurements = Some(m);

        let back = VLInferenceSession::from_json(&s.to_json()).unwrap();
        assert_eq!(back.role, ENInferenceRole::PostTraining);
        let measurements = back.measurements.expect("nested measurements must survive");
        assert_eq!(measurements.iterations.len(), 1);
        assert_eq!(measurements.iterations[0].generated_tokens, 16);
    }

    #[test]
    fn both_nested_roles_are_distinguishable_from_the_json_alone() {
        // This is the whole point of D-08: a consumer must not have to infer
        // "before" or "after" from where the object sits in the tree.
        let pre = VLInferenceSession::nested(ENInferenceRole::PreTraining).to_json();
        let post = VLInferenceSession::nested(ENInferenceRole::PostTraining).to_json();
        assert_eq!(pre.get("role").unwrap().as_str(), Some("pre_training"));
        assert_eq!(post.get("role").unwrap().as_str(), Some("post_training"));
    }

    #[test]
    fn an_unknown_role_is_rejected_rather_than_defaulted_to_standalone() {
        let json = Json::obj([("role", Json::s("mid_training"))]);
        let err = VLInferenceSession::from_json(&json).unwrap_err();
        assert!(err.contains("mid_training"), "got {err}");
    }
}
