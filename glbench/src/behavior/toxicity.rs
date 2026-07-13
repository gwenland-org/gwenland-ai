//! Toxicity — **deliberately not implemented.** This file is the plan and the
//! reasoning, so the decision is recorded rather than silently skipped.
//!
//! # Why there is no code here
//!
//! The original proposal was "logit bias toward flagged token clusters": take a
//! list of flagged tokens, measure how much probability mass the model puts on
//! them, report the number as toxicity.
//!
//! That measures **affinity to a word list**. It does not measure toxicity, and
//! the gap between those two things is not academic — it fails in both
//! directions, and both failures are common:
//!
//! - **False positives.** A model answering a medical question ("carcinoma",
//!   "overdose"), a legal one ("assault", "harassment"), or a security one
//!   ("exploit", "payload") puts real mass on flagged tokens while behaving
//!   perfectly. The number goes up; nothing is wrong. Any model tested on a
//!   serious domain would look worse than one tested on recipes.
//!
//! - **False negatives.** Genuinely harmful output — implicit bias, dog
//!   whistles, sarcastic cruelty, confidently stated misinformation, a polite
//!   refusal to help someone in crisis — uses ordinary vocabulary. It puts
//!   *zero* mass on the flagged list. The number stays clean.
//!
//! So the metric would be wrong on the models that are fine, wrong on the models
//! that are not, and — worst of all — it would look objective while being both.
//! A number in a profiler carries authority. Shipping this one would launder a
//! guess as a measurement, which is a deeper violation of Veritas Prima than
//! guessing openly.
//!
//! # What measuring this honestly would actually take
//!
//! Not a column in a profiler. It is a different kind of work, and it belongs in
//! a different tool:
//!
//! 1. **A labelled dataset.** Prompts with known-harmful continuations, and
//!    known-safe ones that use similar vocabulary (the medical/legal/security
//!    cases above), so false positives are measurable rather than invisible.
//!    RealToxicityPrompts, BBQ, and similar exist for this.
//! 2. **A classifier or human evaluator** judging the *generated text*, not the
//!    logits. Toxicity is a property of meaning, and meaning is not recoverable
//!    from token probabilities.
//! 3. **Calibration and reported error rates.** A safety number without a known
//!    false-positive rate is not a safety number.
//! 4. **A stated threat model.** "Toxic" to whom, in what deployment? The answer
//!    changes what to measure.
//!
//! None of that is a benchmark harness's job. glbench measures what the engine
//! *did* — timings, distributions, token statistics. Whether the output is
//! *harmful* is a question about the world, and this crate has no access to the
//! world.
//!
//! # If this is revisited
//!
//! Should a genuine need appear, the honest first step is a metric that says
//! exactly what it measures and claims nothing more — something like
//! `flagged_token_affinity`, with the false-positive problem printed next to the
//! number every time it is shown. That is a legitimate diagnostic (it answers
//! "is the model dwelling on this vocabulary?"). It is simply not a safety
//! metric, and it must never be labelled as one.
//!
//! The scaffolding below is left as a shape to fill in, not an API to call.

/// Placeholder. Intentionally uninhabited: there is no honest implementation to
/// return, and an empty struct returning zeros would be worse than nothing —
/// callers would read "0.0 toxicity" as a clean bill of health.
///
/// See the module docs. Do not implement this as a word-list score.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToxicitySignal {}

impl ToxicitySignal {
    /// Always `None`. Kept so a caller wiring up the signal list gets a
    /// compile-time "not available" rather than a fabricated zero.
    ///
    /// The `_` parameters document the inputs a real implementation would need
    /// (and immediately show why a profiler cannot supply them: it has the
    /// tokens, but not the ground truth about what they mean).
    pub fn compute(_tokens: &[u32]) -> Option<ToxicitySignal> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn toxicity_is_never_reported_as_a_clean_zero() {
        // The critical property. A safety metric that returns 0.0 when it has
        // not measured anything is worse than absent: it reads as "passed".
        // This must stay None so no consumer can mistake silence for safety.
        assert!(ToxicitySignal::compute(&[1, 2, 3]).is_none());
        assert!(ToxicitySignal::compute(&[]).is_none());
    }
}
