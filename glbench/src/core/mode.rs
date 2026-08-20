//! What kind of session this is, and what role a nested inference run plays
//! (D-06, D-08).
//!
//! # Three modes, not four
//!
//! A join of two archives is **not** a session mode. It is a derived artifact
//! over two immutable sessions, with its own top-level type
//! ([`crate::storage::join::VLJoinManifest`]) and its own schema. Giving it a
//! mode variant would mean a `BenchmarkSession` that is really two sessions,
//! which is a different thing wearing the same type.
//!
//! # Why a nested inference session carries an explicit role
//!
//! In `Unified` mode there are two inference runs: one before training and one
//! after. Distinguishing them *by position in the tree* — outer means before,
//! `training.post_eval` means after — is a footgun for every consumer, and it
//! survives exactly until someone reshapes the tree. So each carries its role
//! as data ([`ENInferenceRole`]) and position becomes redundant rather than
//! load-bearing.

/// What a session measured.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ENSessionMode {
    /// Inference only. The v1 shape, and what an archive with no `session_mode`
    /// field is read as (D-20).
    InferenceOnly,
    /// Training only — no inference measured alongside it.
    TrainingOnly,
    /// Training with inference measured before and after, in one session.
    Unified,
}

impl ENSessionMode {
    /// Stable wire identifier, `snake_case`.
    pub fn as_str(self) -> &'static str {
        match self {
            ENSessionMode::InferenceOnly => "inference_only",
            ENSessionMode::TrainingOnly => "training_only",
            ENSessionMode::Unified => "unified",
        }
    }

    /// Parse the wire identifier back. `None` on an unknown string.
    /// Inherent `Option`-returning parser rather than a `FromStr` impl,
    /// matching [`crate::core::workload::WorkloadKind::from_str`]: the call
    /// sites want `Option`, not a `Result` with an error type.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<ENSessionMode> {
        Some(match s {
            "inference_only" => ENSessionMode::InferenceOnly,
            "training_only" => ENSessionMode::TrainingOnly,
            "unified" => ENSessionMode::Unified,
            _ => return None,
        })
    }
}

impl Default for ENSessionMode {
    /// A session with no stated mode is an inference session: that is what every
    /// v1 archive is, and reading one must not require a migration (D-20).
    fn default() -> ENSessionMode {
        ENSessionMode::InferenceOnly
    }
}

/// Where an inference run sits relative to training.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ENInferenceRole {
    /// No training in this session; the inference run is the whole point.
    Standalone,
    /// Measured before the training run, as its baseline.
    PreTraining,
    /// Measured after the training run, to see what training did.
    PostTraining,
}

impl ENInferenceRole {
    /// Stable wire identifier, `snake_case`.
    pub fn as_str(self) -> &'static str {
        match self {
            ENInferenceRole::Standalone => "standalone",
            ENInferenceRole::PreTraining => "pre_training",
            ENInferenceRole::PostTraining => "post_training",
        }
    }

    /// Parse the wire identifier back. `None` on an unknown string.
    /// Inherent `Option`-returning parser rather than a `FromStr` impl,
    /// matching [`crate::core::workload::WorkloadKind::from_str`]: the call
    /// sites want `Option`, not a `Result` with an error type.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<ENInferenceRole> {
        Some(match s {
            "standalone" => ENInferenceRole::Standalone,
            "pre_training" => ENInferenceRole::PreTraining,
            "post_training" => ENInferenceRole::PostTraining,
            _ => return None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_session_mode_round_trips_through_its_wire_identifier() {
        for mode in [
            ENSessionMode::InferenceOnly,
            ENSessionMode::TrainingOnly,
            ENSessionMode::Unified,
        ] {
            assert_eq!(ENSessionMode::from_str(mode.as_str()), Some(mode));
        }
    }

    #[test]
    fn every_inference_role_round_trips_through_its_wire_identifier() {
        for role in [
            ENInferenceRole::Standalone,
            ENInferenceRole::PreTraining,
            ENInferenceRole::PostTraining,
        ] {
            assert_eq!(ENInferenceRole::from_str(role.as_str()), Some(role));
        }
    }

    /// The v1-compatibility promise (D-20) rests on this default, so it is
    /// asserted rather than left to the `derive`.
    #[test]
    fn the_default_session_mode_is_inference_only_for_v1_archives() {
        assert_eq!(ENSessionMode::default(), ENSessionMode::InferenceOnly);
    }

    #[test]
    fn unknown_identifiers_are_rejected_rather_than_defaulted() {
        assert_eq!(ENSessionMode::from_str("distillation"), None);
        assert_eq!(ENInferenceRole::from_str("mid_training"), None);
    }
}
