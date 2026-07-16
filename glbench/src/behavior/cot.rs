//! CoT-aware entropy assessment — is low entropy expected, or an anomaly?
//!
//! A thinking-capable model (Qwen3, DeepSeek-R1 family) running with its
//! `<think>` mode active is *supposed* to decode long, near-deterministic
//! reasoning stretches: entropy collapses and top-choice probability pins near
//! 1.0 by design. The exact same numbers from a non-thinking model are the
//! signature of a degenerate, looping decode. One measurement, two opposite
//! meanings — so the flag must carry the model context, and this module is
//! where that context is applied.
//!
//! # Threshold — a deliberate deviation from the PRD draft
//!
//! The PRD sketched `session_mean < mean − 2σ` over this run's own entropy
//! distribution. That comparison can never fire (a mean is never two of its own
//! standard deviations below itself), so it is replaced with an absolute,
//! documented gate: a run is **low-entropy** when its mean entropy is under
//! [`LOW_ENTROPY_NATS`] *and* its mean top-choice probability is over
//! [`HIGH_TOP_PROB`]. Absolute thresholds are also what keep the flag
//! comparable across runs and models — a relative-to-self gate would move with
//! the very behavior it is trying to judge. Both constants are recorded in the
//! output so a reader who disagrees can re-derive the flag from the raw
//! entropy numbers, which are always printed alongside it.

use super::entropy::EntropySignal;

/// Mean entropy (nats) below which a run counts as low-entropy. Well under
/// typical free-generation entropy (1.5–3+ nats) and above true collapse
/// (~0.0x nats), so it separates "locked in" from merely "confident".
pub const LOW_ENTROPY_NATS: f64 = 0.5;

/// Mean top-choice probability above which the distribution counts as pinned.
/// Required *with* the entropy gate so a long-tail distribution with a modest
/// favorite is not misread as collapse.
pub const HIGH_TOP_PROB: f64 = 0.90;

/// What the entropy level means, given what kind of model produced it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntropyFlag {
    /// Entropy is not unusually low; nothing to explain.
    Normal,
    /// Low entropy from a thinking-capable model — expected, not an anomaly.
    LowEntropyCotExpected,
    /// Low entropy from a model with no thinking mode — investigate (check
    /// the repetition signal next: looping is the usual culprit).
    LowEntropyAnomaly,
}

impl EntropyFlag {
    /// Stable identifier used in archives and rendered output.
    pub fn as_str(self) -> &'static str {
        match self {
            EntropyFlag::Normal => "NORMAL",
            EntropyFlag::LowEntropyCotExpected => "LOW_ENTROPY_COT_EXPECTED",
            EntropyFlag::LowEntropyAnomaly => "LOW_ENTROPY_ANOMALY",
        }
    }
}

/// The CoT-aware read of a run's entropy. Carries the inputs to its own
/// verdict (thresholds, thinking flag) so the flag is auditable from the
/// archive alone.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CotAssessment {
    /// The verdict.
    pub flag: EntropyFlag,
    /// Whether the model was treated as thinking-capable (after any manual
    /// `cot_mode` override).
    pub thinking_capable: bool,
    /// The mean-entropy gate that was applied, nats.
    pub threshold_nats: f64,
    /// The top-probability gate that was applied.
    pub threshold_top_prob: f64,
}

impl CotAssessment {
    /// Assess a run's entropy in the light of the model's thinking capability.
    ///
    /// `thinking_capable` is the *resolved* flag: GGUF auto-detection (see
    /// [`crate::engine::model_probe`]) already overridden by the workload's
    /// `cot_mode` when the user set one.
    pub fn assess(entropy: &EntropySignal, thinking_capable: bool) -> CotAssessment {
        let low = entropy.mean < LOW_ENTROPY_NATS && entropy.mean_top_prob > HIGH_TOP_PROB;
        let flag = match (low, thinking_capable) {
            (false, _) => EntropyFlag::Normal,
            (true, true) => EntropyFlag::LowEntropyCotExpected,
            (true, false) => EntropyFlag::LowEntropyAnomaly,
        };
        CotAssessment {
            flag,
            thinking_capable,
            threshold_nats: LOW_ENTROPY_NATS,
            threshold_top_prob: HIGH_TOP_PROB,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entropy(mean: f64, top_prob: f64) -> EntropySignal {
        EntropySignal {
            mean,
            std_dev: 0.1,
            min: mean * 0.5,
            max: mean * 2.0,
            p50: mean,
            p95: mean * 1.5,
            mean_top_prob: top_prob,
            tokens: 100,
        }
    }

    #[test]
    fn low_entropy_on_thinking_model_is_expected() {
        let a = CotAssessment::assess(&entropy(0.16, 0.95), true);
        assert_eq!(a.flag, EntropyFlag::LowEntropyCotExpected);
    }

    #[test]
    fn same_numbers_on_plain_model_are_an_anomaly() {
        // The whole point of the module: identical measurement, opposite verdict.
        let a = CotAssessment::assess(&entropy(0.16, 0.95), false);
        assert_eq!(a.flag, EntropyFlag::LowEntropyAnomaly);
    }

    #[test]
    fn normal_entropy_is_normal_regardless_of_model_kind() {
        assert_eq!(CotAssessment::assess(&entropy(2.1, 0.4), true).flag, EntropyFlag::Normal);
        assert_eq!(CotAssessment::assess(&entropy(2.1, 0.4), false).flag, EntropyFlag::Normal);
    }

    #[test]
    fn low_mean_with_spread_out_top_prob_is_not_flagged() {
        // Both gates must fire: entropy under 0.5 nats but top-prob 0.6 means
        // the distribution is not pinned — not the collapse signature.
        let a = CotAssessment::assess(&entropy(0.4, 0.6), false);
        assert_eq!(a.flag, EntropyFlag::Normal);
    }
}
