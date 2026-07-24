//! Which sampled token ids end generation.
//!
//! This is **not** a reinvention of [`crate::tokenizer::Tokenizer`]'s own
//! `stop_token_ids`/`is_stop_token` — that resolver (metadata EOS plus every
//! known stop-marker string found in the vocab) is already correct and
//! already tested, and stays exactly as it is. What it
//! cannot do is cross the [`crate::engine_trait::GlEngine`] boundary: an
//! engine that has no `Tokenizer` of its own (a `.gllm` package has none —
//! ARTX1 OQ3 is still open) has no way to ask one what its stop ids are.
//! `StoppingCriteria` is that bridge — a thin, tokenizer-independent value
//! any caller can build (from a real `Tokenizer` via [`Self::from_tokenizer`],
//! or from any other source of ids) and hand to an engine through
//! [`crate::engine_trait::InferInput`].
//!
//! # Design constraints
//!
//! - **Post-sampling only.** A caller checks `is_stop` on the token the
//!   sampler already chose, exactly where `glproc::runner::Runner::generate`
//!   already checks its own `is_stop: impl Fn(u32) -> bool` closure. This
//!   type is deliberately shaped to satisfy that closure signature trivially
//!   (`|id| criteria.is_stop(id)`) — `Runner` needs no changes to consume it.
//! - **Never logit-level.** Nothing here touches a logits buffer. Stopping
//!   this way cannot change what the model would have said, only whether
//!   generation keeps going after it said it — the same distinction
//!   `EosTokenCriteria` draws from a `LogitsProcessor` in every mainstream
//!   inference stack.
//! - **Empty means "never stop early," not "stop on token 0."** A default
//!   `StoppingCriteria` must be indistinguishable from "no criteria was
//!   supplied" — anything else would silently change behavior for every
//!   existing caller that doesn't yet populate it.
//! - **O(1) per step.** A `HashSet` lookup, not a scan.
//!
//! # `ignore_eos` and [`Self::merge`]
//!
//! An engine that can draw stop ids from two sources (e.g. `GllmEngine`:
//! manifest-embedded ids *and* a caller-supplied `InferInput::stopping`)
//! should [`Self::merge`] them rather than pick one — real GGUF metadata is
//! routinely *less* complete than a full `Tokenizer`'s vocab-scanned resolve
//! (measured: Qwen2.5-0.5B's GGUF declares only `tokenizer.ggml.eos_token_id
//! = 151645`, never the `<|endoftext|>` = 151643 a real `Tokenizer` also
//! resolves), so neither source alone is safe to treat as authoritative.
//!
//! `ignore_eos` (named after vLLM's identically-purposed sampling parameter)
//! exists for the caller who has *correct* stop ids available but does not
//! want early stopping to happen anyway — throughput benchmarking over a
//! synthetic prompt is the concrete case: a random stop id landing mid-stream
//! would truncate the very measurement being taken. It is a request to
//! *suppress* stopping, not a third source of ids, so [`Self::merge`] ORs it
//! rather than requiring both sides to set it.

use std::collections::HashSet;

/// The set of token ids that end generation when sampled.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StoppingCriteria {
    stop_ids: HashSet<u32>,
    ignore_eos: bool,
}

impl StoppingCriteria {
    /// Build from any source of stop ids.
    pub fn new(ids: impl IntoIterator<Item = u32>) -> Self {
        StoppingCriteria { stop_ids: ids.into_iter().collect(), ignore_eos: false }
    }

    /// Build from a tokenizer's own resolved stop set — the bridge every
    /// caller that already owns a real `Tokenizer` (glproc's engine, glbench,
    /// `run_package_e2e`) should use, rather than re-deriving EOS ids by hand.
    pub fn from_tokenizer(tokenizer: &crate::tokenizer::Tokenizer) -> Self {
        StoppingCriteria::new(tokenizer.stop_token_ids().iter().copied())
    }

    /// Suppress early stopping even though stop ids are known — see the
    /// module docs' `ignore_eos` section. Consumes and returns `self` for
    /// builder-style chaining: `StoppingCriteria::new(ids).ignoring_eos()`.
    pub fn ignoring_eos(mut self) -> Self {
        self.ignore_eos = true;
        self
    }

    /// Union of two criteria's stop ids; `ignore_eos` is true if either side
    /// set it (a request to suppress stopping should never be silently lost
    /// by merging with a criteria that didn't ask for it).
    pub fn merge(&self, other: &StoppingCriteria) -> StoppingCriteria {
        StoppingCriteria {
            stop_ids: self.stop_ids.union(&other.stop_ids).copied().collect(),
            ignore_eos: self.ignore_eos || other.ignore_eos,
        }
    }

    /// True when `token_id` should end generation. O(1). Always `false` when
    /// `ignore_eos` is set, regardless of `token_id`.
    pub fn is_stop(&self, token_id: u32) -> bool {
        !self.ignore_eos && self.stop_ids.contains(&token_id)
    }

    /// True when this criteria stops on nothing — the default, meaning "no
    /// early stop," never "stop immediately." Reflects `stop_ids` only —
    /// `ignore_eos` alone on an otherwise-empty criteria is still "empty"
    /// (there is nothing to ignore), so this stays a pure statement about
    /// the id set, not about whether `is_stop` can ever return true.
    pub fn is_empty(&self) -> bool {
        self.stop_ids.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_criteria_never_stops() {
        let c = StoppingCriteria::default();
        assert!(c.is_empty());
        for id in [0u32, 1, 2, 100, u32::MAX] {
            assert!(!c.is_stop(id), "empty criteria stopped on {id}");
        }
    }

    #[test]
    fn single_id_matches_only_that_id() {
        let c = StoppingCriteria::new([42]);
        assert!(!c.is_empty());
        assert!(c.is_stop(42));
        assert!(!c.is_stop(41));
        assert!(!c.is_stop(43));
    }

    #[test]
    fn multiple_ids_all_match() {
        // Qwen2.5's own generation_config.json: [151645, 151643].
        let c = StoppingCriteria::new([151645, 151643]);
        assert!(c.is_stop(151645), "<|im_end|> must stop");
        assert!(c.is_stop(151643), "<|endoftext|> must stop");
        assert!(!c.is_stop(151644));
    }

    #[test]
    fn new_deduplicates_via_the_underlying_set() {
        let c = StoppingCriteria::new([7, 7, 7]);
        assert!(c.is_stop(7));
        // No public len() is needed to observe this — is_stop is the only
        // contract — but a duplicate-heavy input must not panic or behave
        // differently from a deduplicated one.
        assert!(!c.is_stop(8));
    }

    #[test]
    fn ignoring_eos_suppresses_stopping_even_with_known_ids() {
        let c = StoppingCriteria::new([42]).ignoring_eos();
        assert!(!c.is_stop(42), "ignore_eos must suppress an otherwise-matching id");
    }

    #[test]
    fn ignoring_eos_on_an_empty_criteria_is_still_reported_empty() {
        // is_empty() is a statement about the id set, not about whether
        // is_stop can ever return true — see the module docs.
        let c = StoppingCriteria::default().ignoring_eos();
        assert!(c.is_empty());
        assert!(!c.is_stop(0));
    }

    #[test]
    fn merge_unions_stop_ids_from_both_sides() {
        // The real Qwen2.5-0.5B case: manifest GGUF metadata gives only
        // {151645}, a full Tokenizer resolves {151645, 151643} — neither
        // side alone is complete, so the engine must union them.
        let manifest = StoppingCriteria::new([151_645]);
        let caller = StoppingCriteria::new([151_645, 151_643]);
        let merged = manifest.merge(&caller);
        assert!(merged.is_stop(151_645));
        assert!(merged.is_stop(151_643));
        assert!(!merged.is_stop(151_644));
    }

    #[test]
    fn merge_ors_ignore_eos_from_either_side() {
        let with_flag = StoppingCriteria::new([1]).ignoring_eos();
        let without_flag = StoppingCriteria::new([2]);
        let merged = without_flag.merge(&with_flag);
        // Both ids are present in the union...
        assert!(!merged.is_stop(1), "ignore_eos from either side must suppress the merged result");
        assert!(!merged.is_stop(2));
    }

    #[test]
    fn merge_of_two_untouched_criteria_is_still_untouched() {
        // Guards Runtime::apply_default_stopping's equality check: merging
        // two defaults must produce something indistinguishable from
        // default, not a criteria that looks "touched" for no reason.
        let merged = StoppingCriteria::default().merge(&StoppingCriteria::default());
        assert_eq!(merged, StoppingCriteria::default());
    }
}
