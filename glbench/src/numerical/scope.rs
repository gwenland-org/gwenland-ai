//! Which tensors GLBitProf is pointed at (D-11).
//!
//! The math in [`crate::numerical::bitprof`] is ungated — it is a pure function
//! over `&[f32]`. The *sources* are gated per source, which is what this module
//! arranges:
//!
//! | Scope | Source | Gate |
//! |---|---|---|
//! | `weights` | `.gllm` package tensors | `gllm-bench` |
//! | `gradients` | stumman's `VLGradStore` | `train-bench` (Wave 4) |
//! | `optimizer` | `Optimizer::state_tensors` | `train-bench` (Wave 4) |
//!
//! Gradients and optimizer state are **recognised and refused** here rather
//! than reported as unknown scopes. A user who reads about `--bit-scope
//! gradients` should learn which feature flag they need, not that the option
//! does not exist.

use crate::numerical::bitprof::VLBitProfile;

/// Which family of tensors to profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ENBitScope {
    /// Model weights, decoded from a `.gllm` package.
    Weights,
    /// Per-step gradients. Wave 4.
    Gradients,
    /// Optimizer state tensors. Wave 4.
    Optimizer,
}

impl ENBitScope {
    /// Stable wire identifier.
    pub fn as_str(self) -> &'static str {
        match self {
            ENBitScope::Weights => "weights",
            ENBitScope::Gradients => "gradients",
            ENBitScope::Optimizer => "optimizer",
        }
    }

    /// Parse the `--bit-scope` flag value.
    ///
    /// Inherent `Option`-returning parser rather than a `FromStr` impl,
    /// matching [`crate::core::workload::WorkloadKind::from_str`]: the call
    /// sites want `Option`, not a `Result` with an error type.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<ENBitScope> {
        Some(match s {
            "weights" => ENBitScope::Weights,
            "gradients" => ENBitScope::Gradients,
            "optimizer" => ENBitScope::Optimizer,
            _ => return None,
        })
    }

    /// Whether this build can collect this scope.
    ///
    /// `weights` needs `gllm-bench` for the package reader; the training scopes
    /// need Wave 4, which does not exist in any build yet.
    pub fn availability(self) -> Result<(), String> {
        match self {
            ENBitScope::Weights => {
                if cfg!(feature = "gllm-bench") {
                    Ok(())
                } else {
                    Err("--bit-scope weights requires --features gllm-bench \
                         (the .gllm package reader decodes the tensors)"
                        .to_string())
                }
            }
            ENBitScope::Gradients | ENBitScope::Optimizer => Err(format!(
                "--bit-scope {} requires --features train-bench (Wave 4)",
                self.as_str()
            )),
        }
    }
}

/// One profiled tensor, tagged with where it came from.
#[derive(Debug, Clone)]
pub struct VLBitScope {
    /// Which family this tensor belongs to.
    pub scope: ENBitScope,
    /// The tensor's name as the source knows it.
    pub tensor_name: String,
    /// Its bit profile.
    pub profile: VLBitProfile,
}

/// Profile every decodable weight tensor in a `.gllm` package.
///
/// Reuses `tensor_stats`'s decode path rather than reaching for the bytes
/// directly. That path already dispatches GQ4A/GQ2A through
/// `glproc::kernels::gquant` and everything else through
/// `glcore::format::decode_tensor`, exactly as the runtime does — a second
/// decoder here would be the "two independent implementations of one format"
/// risk `architecture/gl-stack-audit-2026-07/ARTX2-Quant.md` was written about.
///
/// Tensors in a dtype that path cannot decode are skipped rather than guessed
/// at; the returned list says which ones were profiled, and the caller reports
/// the rest.
#[cfg(feature = "gllm-bench")]
pub fn scope_weights(model: &std::path::Path) -> Result<Vec<VLBitScope>, String> {
    scope_weights_filtered(model, |_| true).map(|(scopes, _)| scopes)
}

/// [`scope_weights`], plus the walk summary and a tensor-name filter.
///
/// The summary is returned rather than dropped so the CLI can say how many
/// tensors it could not decode. "Profiled 291 tensors" is a different claim
/// from "profiled 291 of 338"; only the second one is honest when a package
/// holds dtypes this decoder does not read.
#[cfg(feature = "gllm-bench")]
pub fn scope_weights_filtered<K>(
    model: &std::path::Path,
    keep: K,
) -> Result<(Vec<VLBitScope>, crate::tensor_stats::DecodeSummary), String>
where
    K: Fn(&str) -> bool,
{
    let mut scopes = Vec::new();
    let summary = crate::tensor_stats::for_each_decoded_tensor(model, keep, |entry, values| {
        scopes.push(VLBitScope {
            scope: ENBitScope::Weights,
            tensor_name: entry.name.clone(),
            profile: crate::numerical::bitprof::profile(values),
        });
    })?;
    Ok((scopes, summary))
}

/// Without the package reader there is nothing to decode.
#[cfg(not(feature = "gllm-bench"))]
pub fn scope_weights(_model: &std::path::Path) -> Result<Vec<VLBitScope>, String> {
    Err(ENBitScope::Weights.availability().unwrap_err())
}

/// Profile the weights of the model a session was run against.
///
/// Convenience over [`scope_weights`]: a session records its model path in
/// `workload.model_path`, so `--profile bits` on a run needs no second flag.
/// The profiling itself is static — it reads the package, not the session — so
/// this is a path lookup, not a second measurement.
pub fn scope_weights_for_session(
    session: &crate::core::session::BenchmarkSession,
) -> Result<Vec<VLBitScope>, String> {
    let path = session.workload.model_path.trim();
    if path.is_empty() {
        return Err("this session records no model path to profile".to_string());
    }
    scope_weights(std::path::Path::new(path))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_scope_round_trips_through_its_wire_identifier() {
        for scope in [ENBitScope::Weights, ENBitScope::Gradients, ENBitScope::Optimizer] {
            assert_eq!(ENBitScope::from_str(scope.as_str()), Some(scope));
        }
    }

    #[test]
    fn an_unknown_scope_is_rejected_rather_than_defaulted_to_weights() {
        assert_eq!(ENBitScope::from_str("activations"), None);
        assert_eq!(ENBitScope::from_str(""), None);
    }

    /// The Wave 4 scopes must name the flag a user needs, not fail as unknown.
    #[test]
    fn the_training_scopes_refuse_with_a_build_hint_rather_than_an_unknown_option() {
        for scope in [ENBitScope::Gradients, ENBitScope::Optimizer] {
            let err = scope.availability().unwrap_err();
            assert!(err.contains("train-bench"), "got {err}");
            assert!(err.contains("Wave 4"), "got {err}");
            assert!(err.contains(scope.as_str()), "the message must name the scope: {err}");
        }
    }

    #[test]
    fn the_weights_scope_availability_follows_the_gllm_bench_feature() {
        let result = ENBitScope::Weights.availability();
        if cfg!(feature = "gllm-bench") {
            assert!(result.is_ok());
        } else {
            let err = result.unwrap_err();
            assert!(err.contains("gllm-bench"), "got {err}");
        }
    }

    #[test]
    fn a_session_with_no_model_path_is_refused_before_any_file_is_opened() {
        use crate::core::metrics::MeasurementSet;
        use crate::core::result::SessionMetadata;
        use crate::core::session::BenchmarkSession;
        use crate::core::workload::WorkloadSpec;
        use crate::engine::metadata::EngineMetadata;
        use crate::environment::hardware::EnvironmentSnapshot;

        let session = BenchmarkSession::new(
            SessionMetadata::new("no-model"),
            EnvironmentSnapshot::probe(""),
            EngineMetadata {
                name: "glproc".into(),
                backend: "cpu".into(),
                available: true,
                model_arch: None,
                quantization: None,
                thinking_capable: None,
            },
            WorkloadSpec::default(),
            MeasurementSet::default(),
        );
        let err = scope_weights_for_session(&session).unwrap_err();
        assert!(err.contains("no model path"), "got {err}");
    }
}
