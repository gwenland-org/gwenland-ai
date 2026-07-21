//! The two-engine driver for numerical parity: runs the oracle (glproc) and a
//! candidate engine through the identical workload and reports how far the
//! candidate's tokens diverge.
//!
//! [`numerical::compare_tokens`] is the pure comparison; this module is the
//! only piece that actually loads two engines and runs them, because
//! `glbench validate --against glproc` needs both token streams and nothing
//! else in the crate does.

use glcore::GlError;

use crate::core::schema::ToJson;
use crate::core::workload::WorkloadSpec;
use crate::engine::adapter::EngineAdapter;
use crate::export::json::Json;
use crate::validation::numerical::{compare_tokens, NumericalCheck};

/// The result of validating one candidate engine against the oracle.
#[derive(Debug, Clone)]
pub struct ParityReport {
    pub oracle_engine: String,
    pub candidate_engine: String,
    pub model_path: String,
    pub check: NumericalCheck,
}

impl ParityReport {
    /// True when every compared token matched — the bar for "numerically
    /// equivalent to the oracle" under greedy decoding.
    pub fn passed(&self) -> bool {
        self.check.exact()
    }
}

impl ToJson for ParityReport {
    fn to_json(&self) -> Json {
        Json::obj([
            ("oracle_engine", Json::s(self.oracle_engine.clone())),
            ("candidate_engine", Json::s(self.candidate_engine.clone())),
            ("model_path", Json::s(self.model_path.clone())),
            ("matching_prefix", Json::n(self.check.matching_prefix as f64)),
            ("compared", Json::n(self.check.compared as f64)),
            ("agreement", Json::n(self.check.agreement())),
            ("passed", Json::Bool(self.passed())),
        ])
    }
}

/// Run `candidate_engine` against the `glproc` oracle for the same model and
/// prompt, and report the matching token prefix.
///
/// Forces greedy decoding (temperature 0) regardless of what `spec` requests:
/// numerical parity is only a meaningful comparison under deterministic
/// sampling — at any temperature > 0, divergence is expected and would not
/// indicate a bug.
pub fn validate_against_oracle(
    spec: &WorkloadSpec,
    oracle_engine: &str,
    candidate_engine: &str,
) -> Result<ParityReport, GlError> {
    validate_against_oracle_capped(spec, oracle_engine, candidate_engine, spec.max_new_tokens)
}

/// As [`validate_against_oracle`], but bounded to at most `max_tokens`
/// generated tokens regardless of what `spec.max_new_tokens` asks for.
///
/// Exists for the automatic in-`run` cross-check (`WorkloadSpec::verify_against`):
/// that check's purpose is "did the two engines diverge at all", which the
/// first several dozen tokens already answer, so there is no reason to pay
/// for a full-length second generation on every benchmark that opts in.
pub fn validate_against_oracle_capped(
    spec: &WorkloadSpec,
    oracle_engine: &str,
    candidate_engine: &str,
    max_tokens: usize,
) -> Result<ParityReport, GlError> {
    let mut oracle_spec = spec.clone();
    oracle_spec.engine = oracle_engine.to_string();
    oracle_spec.temperature = 0.0;
    oracle_spec.max_new_tokens = max_tokens;

    let mut candidate_spec = spec.clone();
    candidate_spec.engine = candidate_engine.to_string();
    candidate_spec.temperature = 0.0;
    candidate_spec.max_new_tokens = max_tokens;

    let oracle = EngineAdapter::load(&oracle_spec)?;
    let oracle_tokens = oracle.run_tokens(&oracle_spec)?;
    drop(oracle);

    let candidate = EngineAdapter::load(&candidate_spec)?;
    let candidate_tokens = candidate.run_tokens(&candidate_spec)?;

    let check = compare_tokens(&oracle_tokens, &candidate_tokens);

    Ok(ParityReport {
        oracle_engine: oracle_engine.to_string(),
        candidate_engine: candidate_engine.to_string(),
        model_path: spec.model_path.clone(),
        check,
    })
}

/// The token budget the automatic in-`run` cross-check uses — the plan's
/// "first 50 tokens" — regardless of the run's own `--tokens`.
pub const AUTO_VERIFY_TOKEN_CAP: usize = 50;

#[cfg(test)]
mod tests {
    use super::*;

    fn report(oracle: &[u32], candidate: &[u32]) -> ParityReport {
        ParityReport {
            oracle_engine: "glproc".into(),
            candidate_engine: "glcuda".into(),
            model_path: "model.gguf".into(),
            check: compare_tokens(oracle, candidate),
        }
    }

    #[test]
    fn passed_reflects_exact_match() {
        assert!(report(&[1, 2, 3], &[1, 2, 3]).passed());
        assert!(!report(&[1, 2, 3], &[1, 9, 3]).passed());
    }

    #[test]
    fn to_json_round_trips_fields() {
        let r = report(&[1, 2, 3], &[1, 9, 3]);
        let j = r.to_json();
        assert_eq!(j.get("matching_prefix").unwrap().as_f64(), Some(1.0));
        assert_eq!(j.get("compared").unwrap().as_f64(), Some(3.0));
        assert_eq!(j.get("passed").unwrap().as_bool(), Some(false));
    }

    #[test]
    fn auto_verify_token_cap_matches_the_plan() {
        assert_eq!(AUTO_VERIFY_TOKEN_CAP, 50);
    }
}
