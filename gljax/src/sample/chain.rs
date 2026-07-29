//! `SamplerChain` — an explicit, ordered, validated sampling pipeline
//! (ARTX14 §1).
//!
//! ⛔ **vLLM and llama.cpp disagree on stage order, and the disagreement is
//! observable, not cosmetic.** Temperature sits on opposite sides of
//! truncation in the two references: applied before top-p (vLLM), it
//! flattens the distribution so the cumulative-probability threshold admits
//! *more* tokens; applied after (llama.cpp), the candidate set is fixed by
//! the untempered distribution and temperature only reshapes weights within
//! it. At `T > 1` these produce materially different candidate pools from
//! identical logits. gljax does not get to pick "the" order, because there
//! is no consensus order to pick — it makes the order **visible and
//! reproducible** instead: `Stage` order is data, not code, meant to be
//! recorded per request.

use std::fmt;

/// Token id — re-exported from [`crate::tok`] rather than redefined, since
/// this module and the tokenizer trait mean the same thing by it.
pub use crate::tok::TokenId;

/// One stage of a [`SamplerChain`]. Order within the chain is what
/// [`SamplerChain::validate`] checks; the variants themselves carry no
/// ordering constraint of their own.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Stage {
    /// Applied to raw logits over a trailing window of generated tokens.
    /// Must precede any truncation stage (see [`Stage::is_truncation`]).
    RepetitionPenalty { penalty: f32, window: usize },
    /// Subtracts a constant from every token seen at least once.
    PresencePenalty { penalty: f32 },
    /// Subtracts `count * penalty` per token, scaled by occurrence count.
    FrequencyPenalty { penalty: f32 },
    /// Per-token additive bias (OpenAI `logit_bias`). Must precede truncation.
    LogitBias,
    /// Mask from ARTX15 (`sample::mask::MaskSource`). Must precede truncation
    /// — a mask applied after truncation could leave nothing left to sample,
    /// forcing a fallback that silently violates the grammar.
    GrammarMask,
    Temperature { t: f32 },
    TopK { k: usize },
    TopP { p: f32 },
    /// Keep tokens with `P(token) >= P(max) * min_p`.
    MinP { p: f32 },
    /// Locally typical sampling: keep tokens whose surprisal is closest to
    /// the distribution's entropy, up to cumulative probability `mass`.
    Typical { mass: f32 },
}

impl Stage {
    /// Whether this stage can *shrink* the candidate set. `validate` uses
    /// this to enforce ARTX14 §1.1/§3.3's ordering rule: penalties and
    /// `GrammarMask` must precede every truncation stage, because applying
    /// them afterward cannot demote or re-admit a token that truncation
    /// already dropped.
    pub fn is_truncation(&self) -> bool {
        matches!(self, Stage::TopK { .. } | Stage::TopP { .. } | Stage::MinP { .. } | Stage::Typical { .. })
    }

    /// Whether this stage must precede every truncation stage (ARTX14 §1.1
    /// for penalties, §3.3 for the grammar mask).
    pub fn must_precede_truncation(&self) -> bool {
        matches!(
            self,
            Stage::RepetitionPenalty { .. }
                | Stage::PresencePenalty { .. }
                | Stage::FrequencyPenalty { .. }
                | Stage::LogitBias
                | Stage::GrammarMask
        )
    }
}

/// An ordered sampling pipeline. Order is DATA, not code — ARTX14 §1.2's
/// explicit design decision, so a generation's exact sampling behavior can
/// be recorded and reproduced rather than inferred from which code path ran.
#[derive(Debug, Clone, PartialEq)]
pub struct SamplerChain {
    pub stages: Vec<Stage>,
    pub seed: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChainError {
    /// A penalty or `GrammarMask` stage appears after a truncation stage.
    PenaltyAfterTruncation { stage_index: usize },
}

impl fmt::Display for ChainError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ChainError::PenaltyAfterTruncation { stage_index } => write!(
                f,
                "stage {stage_index} is a penalty or grammar mask appearing after a \
                 truncation stage — penalties/masks cannot re-admit or demote a token \
                 truncation already removed (ARTX14 SS1.1/SS3.3)"
            ),
        }
    }
}

impl std::error::Error for ChainError {}

/// The subset of OpenAI-style sampling parameters ARTX14's `Stage` enum can
/// express. Not a full request schema (ARTX16 owns that) — just enough to
/// build a default chain.
#[derive(Debug, Clone, PartialEq)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub min_p: Option<f32>,
    pub typical_mass: Option<f32>,
    pub repetition_penalty: Option<(f32, usize)>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub seed: Option<u64>,
}

impl Default for SamplingParams {
    /// `temperature = 0.0` means greedy — see [`SamplerChain::for_params`].
    fn default() -> Self {
        SamplingParams {
            temperature: 0.0,
            top_k: None,
            top_p: None,
            min_p: None,
            typical_mass: None,
            repetition_penalty: None,
            presence_penalty: None,
            frequency_penalty: None,
            seed: None,
        }
    }
}

impl SamplerChain {
    /// `TopK { k: 1 }` alone. ARTX14 §1.2's explicit design decision: "greedy
    /// is `TopK { k: 1 }`, not a separate code path" — `runtime::sample::argmax`
    /// becomes one configuration of this chain rather than a parallel
    /// implementation. See `sample::tests` (in `mod.rs`) for the bit-identical
    /// equivalence this decision requires.
    pub fn greedy() -> Self {
        SamplerChain { stages: vec![Stage::TopK { k: 1 }], seed: None }
    }

    /// `temperature <= 0.0` means greedy, matching the common "0 = greedy"
    /// convention (llama.cpp, vLLM). Otherwise defers to [`Self::openai_default`].
    pub fn for_params(p: &SamplingParams) -> Self {
        if p.temperature <= 0.0 {
            return Self::greedy();
        }
        Self::openai_default(p)
    }

    /// vLLM-compatible ordering: penalties (repetition, frequency, presence)
    /// -> temperature -> logit processors (min-p) -> top-k / top-p -> sample.
    /// gljax's default, because ARTX16 serves an OpenAI-compatible API and
    /// vLLM is the de-facto reference implementation for that surface.
    pub fn openai_default(p: &SamplingParams) -> Self {
        let mut stages = Vec::new();
        if let Some((penalty, window)) = p.repetition_penalty {
            stages.push(Stage::RepetitionPenalty { penalty, window });
        }
        if let Some(penalty) = p.frequency_penalty {
            stages.push(Stage::FrequencyPenalty { penalty });
        }
        if let Some(penalty) = p.presence_penalty {
            stages.push(Stage::PresencePenalty { penalty });
        }
        stages.push(Stage::Temperature { t: p.temperature });
        if let Some(min_p) = p.min_p {
            stages.push(Stage::MinP { p: min_p });
        }
        if let Some(k) = p.top_k {
            stages.push(Stage::TopK { k });
        }
        if let Some(top_p) = p.top_p {
            stages.push(Stage::TopP { p: top_p });
        }
        SamplerChain { stages, seed: p.seed }
    }

    /// llama.cpp-compatible ordering, for cross-checking against that engine
    /// (ARTX12's differential oracle, T3). llama.cpp's real chain also has
    /// `dry`, `top_n_sigma`, and `xtc` stages, none of which `Stage` models
    /// (they are not in ARTX14 §1.2's enum either) — this reproduces the
    /// relative order of the stages that do exist: penalties -> top-k ->
    /// typical -> top-p -> min-p -> temperature.
    pub fn llamacpp_default(p: &SamplingParams) -> Self {
        let mut stages = Vec::new();
        if let Some((penalty, window)) = p.repetition_penalty {
            stages.push(Stage::RepetitionPenalty { penalty, window });
        }
        if let Some(penalty) = p.frequency_penalty {
            stages.push(Stage::FrequencyPenalty { penalty });
        }
        if let Some(penalty) = p.presence_penalty {
            stages.push(Stage::PresencePenalty { penalty });
        }
        if let Some(k) = p.top_k {
            stages.push(Stage::TopK { k });
        }
        if let Some(mass) = p.typical_mass {
            stages.push(Stage::Typical { mass });
        }
        if let Some(top_p) = p.top_p {
            stages.push(Stage::TopP { p: top_p });
        }
        if let Some(min_p) = p.min_p {
            stages.push(Stage::MinP { p: min_p });
        }
        stages.push(Stage::Temperature { t: p.temperature });
        SamplerChain { stages, seed: p.seed }
    }

    /// Refuses chains that are silently wrong rather than reordering them —
    /// ARTX14 §1.2: "silently reordering would make the recorded order a
    /// lie." Checks the one rule §1.1/§3.3 make load-bearing: every penalty
    /// and `GrammarMask` stage must precede every truncation stage.
    pub fn validate(&self) -> Result<(), ChainError> {
        let mut seen_truncation = false;
        for (i, stage) in self.stages.iter().enumerate() {
            if stage.is_truncation() {
                seen_truncation = true;
            } else if stage.must_precede_truncation() && seen_truncation {
                return Err(ChainError::PenaltyAfterTruncation { stage_index: i });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_is_exactly_top_k_one() {
        let chain = SamplerChain::greedy();
        assert_eq!(chain.stages, vec![Stage::TopK { k: 1 }]);
        assert!(chain.validate().is_ok());
    }

    #[test]
    fn for_params_at_zero_temperature_is_greedy() {
        let p = SamplingParams { temperature: 0.0, top_k: Some(40), ..Default::default() };
        assert_eq!(SamplerChain::for_params(&p), SamplerChain::greedy());
    }

    #[test]
    fn openai_default_places_penalties_before_temperature_before_truncation() {
        let p = SamplingParams {
            temperature: 0.7,
            top_k: Some(40),
            top_p: Some(0.9),
            repetition_penalty: Some((1.1, 64)),
            ..Default::default()
        };
        let chain = SamplerChain::openai_default(&p);
        let positions: Vec<&Stage> = chain.stages.iter().collect();
        let rep_pos = positions.iter().position(|s| matches!(s, Stage::RepetitionPenalty { .. })).unwrap();
        let temp_pos = positions.iter().position(|s| matches!(s, Stage::Temperature { .. })).unwrap();
        let topk_pos = positions.iter().position(|s| matches!(s, Stage::TopK { .. })).unwrap();
        let topp_pos = positions.iter().position(|s| matches!(s, Stage::TopP { .. })).unwrap();
        assert!(rep_pos < temp_pos, "penalties must precede temperature");
        assert!(temp_pos < topk_pos, "vLLM order: temperature before top-k/top-p");
        assert!(temp_pos < topp_pos);
        assert!(chain.validate().is_ok());
    }

    #[test]
    fn llamacpp_default_places_temperature_after_truncation() {
        let p = SamplingParams {
            temperature: 0.7,
            top_k: Some(40),
            top_p: Some(0.9),
            ..Default::default()
        };
        let chain = SamplerChain::llamacpp_default(&p);
        let temp_pos = chain.stages.iter().position(|s| matches!(s, Stage::Temperature { .. })).unwrap();
        let topk_pos = chain.stages.iter().position(|s| matches!(s, Stage::TopK { .. })).unwrap();
        assert!(
            topk_pos < temp_pos,
            "llama.cpp order: truncation before temperature (the opposite of vLLM)"
        );
        assert!(chain.validate().is_ok());
    }

    #[test]
    fn the_two_default_orderings_actually_disagree_on_temperature_placement() {
        // Pins ARTX14 SS1.1's central claim directly: the two references are
        // not merely different in style, they place temperature on opposite
        // sides of truncation.
        let p = SamplingParams { temperature: 0.7, top_k: Some(40), ..Default::default() };
        let openai = SamplerChain::openai_default(&p);
        let llamacpp = SamplerChain::llamacpp_default(&p);

        let side = |chain: &SamplerChain| -> bool {
            let temp = chain.stages.iter().position(|s| matches!(s, Stage::Temperature { .. })).unwrap();
            let topk = chain.stages.iter().position(|s| matches!(s, Stage::TopK { .. })).unwrap();
            temp < topk // true = temperature-before-truncation (vLLM), false = after (llama.cpp)
        };
        assert!(side(&openai));
        assert!(!side(&llamacpp));
    }

    #[test]
    fn validate_refuses_a_penalty_placed_after_top_k() {
        let chain = SamplerChain {
            stages: vec![
                Stage::TopK { k: 40 },
                Stage::RepetitionPenalty { penalty: 1.1, window: 64 },
            ],
            seed: None,
        };
        let err = chain.validate().expect_err("must refuse");
        assert_eq!(err, ChainError::PenaltyAfterTruncation { stage_index: 1 });
    }

    #[test]
    fn validate_refuses_a_grammar_mask_placed_after_top_p() {
        let chain = SamplerChain {
            stages: vec![Stage::TopP { p: 0.9 }, Stage::GrammarMask],
            seed: None,
        };
        assert!(chain.validate().is_err());
    }

    #[test]
    fn validate_accepts_temperature_anywhere_relative_to_truncation() {
        // Temperature is not a "must precede truncation" stage in either
        // reference ordering — only penalties and the grammar mask are.
        let after = SamplerChain { stages: vec![Stage::TopK { k: 40 }, Stage::Temperature { t: 0.7 }], seed: None };
        let before = SamplerChain { stages: vec![Stage::Temperature { t: 0.7 }, Stage::TopK { k: 40 }], seed: None };
        assert!(after.validate().is_ok());
        assert!(before.validate().is_ok());
    }

    #[test]
    fn validate_accepts_multiple_truncation_stages_after_all_penalties() {
        let chain = SamplerChain {
            stages: vec![
                Stage::RepetitionPenalty { penalty: 1.1, window: 64 },
                Stage::LogitBias,
                Stage::TopK { k: 40 },
                Stage::TopP { p: 0.9 },
                Stage::MinP { p: 0.05 },
            ],
            seed: None,
        };
        assert!(chain.validate().is_ok());
    }
}
