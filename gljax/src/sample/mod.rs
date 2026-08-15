//! Sampling & logits processing (ARTX14).
//!
//! Wave A14.1's scope, per ARTX14 §5's own wave plan: `chain.rs` plus the
//! **host-only path** (option A, ARTX14 §2.1) — the full-logits-on-host
//! sampler that exists as both the correctness oracle ARTX12's harness needs
//! and, this wave, the only implementation. Its gate is deliberately strict:
//! *"greedy through the new chain must be bit-identical to ARTX05's
//! `argmax_f32`, not merely equivalent."* [`tests::greedy_chain_is_bit_identical_to_argmax`]
//! is that gate.
//!
//! ⛔ **Not built this wave** (later waves in ARTX14 §5's own plan, not
//! skipped arbitrarily):
//! - **A14.2** — the device/host split (`ops::top_k`, §2.2's "148x less
//!   transfer" design). Building it needs three new core StableHLO
//!   primitives gljax does not have yet: `iota`, `compare`, and `sort` (a
//!   region-based, variadic op — `top_k` is *not* a core StableHLO op,
//!   confirmed directly against `stablehlo/dialect/StablehloOps.td`, which
//!   is why JAX itself lowers `lax.top_k` to `iota` + `sort` + `slice`
//!   rather than emitting a `top_k` op that doesn't exist in the dialect).
//!   Three unverified-against-a-real-parser primitives is a substantial,
//!   independent unit of work — exactly the kind of thing this sprint has
//!   repeatedly declined to bolt on unmeasured (see `gljax::arch`'s and
//!   `docs/flash_attention_reachability.md`'s own scope notes for the same
//!   pattern elsewhere this sprint).
//! - **A14.3** — sparse *device-side* penalty upload (the host-side
//!   `PenaltyState` and its arithmetic are built, in `penalty.rs` — only the
//!   device path is deferred).
//! - **A14.5** — ARTX11's `gather_candidate_probs` — depends on ARTX11's
//!   speculative decoding machinery, none of which exists in this codebase.

pub mod chain;
pub mod mask;
pub mod penalty;

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

pub use chain::{ChainError, SamplerChain, SamplingParams, Stage, TokenId};
pub use mask::{AllowMask, MaskError, MaskSource, SlotId};
pub use penalty::PenaltyState;

/// A candidate set being narrowed down to one sampled token, host-side.
/// ARTX14 §4's pseudocode sketch (`gljax/src/sample/host.rs`'s role) —
/// folded into this module rather than a separate `host.rs`, since Wave
/// A14.1 doesn't yet have a device half for `host.rs` to be "the tail of."
pub struct Candidates {
    items: Vec<(TokenId, f32)>,
}

impl Candidates {
    /// `items` is `(token_id, logit)` pairs. NaN logits are dropped here,
    /// matching `runtime::sample::argmax`'s exact convention — that function
    /// skips a NaN so it "never wins by accident" rather than letting it
    /// participate in comparisons; dropping it up front produces the
    /// identical outcome (a dropped candidate can't win either) without
    /// needing every downstream stage to special-case NaN.
    pub fn new(items: impl IntoIterator<Item = (TokenId, f32)>) -> Self {
        Candidates { items: items.into_iter().filter(|(_, logit)| !logit.is_nan()).collect() }
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn as_slice(&self) -> &[(TokenId, f32)] {
        &self.items
    }

    pub fn as_mut_slice(&mut self) -> &mut [(TokenId, f32)] {
        &mut self.items
    }

    /// Divides every logit by `t`. `t <= 0.0` is a caller error (temperature
    /// belongs to `Stage::Temperature`, not greedy — greedy is
    /// `Stage::TopK { k: 1 }` per `SamplerChain::greedy`'s docs, which never
    /// calls this).
    pub fn scale(&mut self, t: f32) {
        assert!(t > 0.0, "Candidates::scale: temperature must be > 0, got {t}");
        for (_, logit) in &mut self.items {
            *logit /= t;
        }
    }

    /// Keeps the `k` highest-logit candidates. Ties break toward the
    /// **lower original index** in `self.items` — the same convention
    /// `runtime::sample::argmax` uses, which is what makes
    /// `truncate(1)` bit-identical to it (`slice::sort_by` is stable, so
    /// equal-logit candidates keep their relative order).
    pub fn truncate(&mut self, k: usize) {
        self.items.sort_by(cmp_desc_by_logit);
        self.items.truncate(k);
    }

    /// Nucleus sampling: keep the smallest prefix (by descending
    /// probability) whose cumulative probability is `>= p`. Always keeps at
    /// least one candidate, even if its own probability exceeds `p` alone.
    pub fn nucleus(&mut self, p: f32) {
        if self.items.is_empty() {
            return;
        }
        self.items.sort_by(cmp_desc_by_logit);
        let probs = softmax(&self.items);
        let mut cumulative = 0.0f32;
        let mut cutoff = self.items.len();
        for (i, prob) in probs.iter().enumerate() {
            cumulative += prob;
            if cumulative >= p {
                cutoff = i + 1;
                break;
            }
        }
        self.items.truncate(cutoff.max(1));
    }

    /// Keeps candidates with `P(token) >= P(max) * p`.
    pub fn min_p(&mut self, p: f32) {
        if self.items.is_empty() {
            return;
        }
        self.items.sort_by(cmp_desc_by_logit);
        let probs = softmax(&self.items);
        let threshold = probs[0] * p;
        let cutoff = probs.iter().position(|&pr| pr < threshold).unwrap_or(probs.len());
        self.items.truncate(cutoff.max(1));
    }

    /// Locally typical sampling: keeps candidates whose surprisal
    /// (`-ln P(token)`) is closest to the distribution's entropy, in that
    /// order, up to cumulative probability `mass`.
    pub fn typical(&mut self, mass: f32) {
        if self.items.is_empty() {
            return;
        }
        // Entropy needs the *original* (untruncated-by-this-step) distribution.
        let probs = softmax(&self.items);
        let entropy: f32 = -probs.iter().map(|&p| if p > 0.0 { p * p.ln() } else { 0.0 }).sum::<f32>();

        let mut by_typicality: Vec<usize> = (0..self.items.len()).collect();
        by_typicality.sort_by(|&i, &j| {
            let surprisal_i = -probs[i].ln();
            let surprisal_j = -probs[j].ln();
            (surprisal_i - entropy).abs().partial_cmp(&(surprisal_j - entropy).abs()).unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut cumulative = 0.0f32;
        let mut keep = Vec::with_capacity(self.items.len());
        for &i in &by_typicality {
            keep.push(self.items[i]);
            cumulative += probs[i];
            if cumulative >= mass {
                break;
            }
        }
        self.items = keep;
    }

    /// Samples one token from what remains, weighted by softmax probability.
    /// `None` iff every candidate was filtered out (including the
    /// all-NaN-input case `Candidates::new` already reduces to empty).
    ///
    /// A single remaining candidate is returned directly, without consuming
    /// `rng` — the greedy path (`truncate(1)`) must be deterministic and
    /// independent of any RNG's specific algorithm, which is what makes
    /// [`SamplerChain::greedy`] bit-identical to `runtime::sample::argmax`.
    pub fn sample(&self, rng: &mut impl Rng) -> Option<TokenId> {
        match self.items.as_slice() {
            [] => None,
            [(id, _)] => Some(*id),
            items => {
                let probs = softmax(items);
                let draw: f32 = rng.gen_range(0.0..1.0);
                let mut cumulative = 0.0f32;
                for (i, &p) in probs.iter().enumerate() {
                    cumulative += p;
                    if draw < cumulative {
                        return Some(items[i].0);
                    }
                }
                // Floating-point rounding can leave `draw` a hair past the
                // last cumulative sum — fall back to the last candidate
                // rather than returning None for a non-empty set.
                items.last().map(|(id, _)| *id)
            }
        }
    }
}

fn cmp_desc_by_logit(a: &(TokenId, f32), b: &(TokenId, f32)) -> std::cmp::Ordering {
    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
}

/// Numerically-stable softmax over `items`' logits (subtract the max before
/// exponentiating), mirroring `ops::softmax::softmax`'s device-side formula
/// exactly — same shape of numerics bug either implementation could have,
/// same fix.
fn softmax(items: &[(TokenId, f32)]) -> Vec<f32> {
    let max = items.iter().map(|(_, l)| *l).fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = items.iter().map(|(_, l)| (l - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

/// A seeded RNG for reproducible sampling, or a non-deterministic one when
/// `seed` is `None`. Separate from [`Candidates`] itself so tests (and any
/// caller wanting exact control) can construct and reuse their own `SmallRng`
/// instead of this policy.
pub fn rng_from_seed(seed: Option<u64>) -> SmallRng {
    match seed {
        Some(s) => SmallRng::seed_from_u64(s),
        None => SmallRng::from_entropy(),
    }
}

/// Applies every stage of `chain` to `candidates` (host-only — ARTX14 §2.1's
/// option (A)), then samples one token. Device-side-only stages
/// (`GrammarMask`, the penalty/logit-bias stages once a device path exists)
/// are no-ops here by design — matching the pseudocode's own `_ => {}` arm
/// for "device-side stages already applied" — except this wave has no device
/// path, so callers that need a grammar mask or penalties must apply
/// [`AllowMask::filter`] / [`PenaltyState`]'s `apply_*` methods to
/// `candidates` **before** calling this, exactly mirroring ARTX14 §1.1/§3.3's
/// ordering rule that those stages precede truncation.
pub fn apply_chain_host(
    chain: &SamplerChain,
    mut candidates: Candidates,
    rng: &mut impl Rng,
) -> Result<Option<TokenId>, ChainError> {
    chain.validate()?;
    for stage in &chain.stages {
        match stage {
            Stage::Temperature { t } => candidates.scale(*t),
            Stage::TopK { k } => candidates.truncate(*k),
            Stage::TopP { p } => candidates.nucleus(*p),
            Stage::MinP { p } => candidates.min_p(*p),
            Stage::Typical { mass } => candidates.typical(*mass),
            // Host-applied ahead of this call (penalties, logit bias) or not
            // yet implemented on any path (grammar mask — Wave B7/ARTX15).
            Stage::RepetitionPenalty { .. }
            | Stage::PresencePenalty { .. }
            | Stage::FrequencyPenalty { .. }
            | Stage::LogitBias
            | Stage::GrammarMask => {}
        }
    }
    Ok(candidates.sample(rng))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::sample::argmax;

    /// ⭐ ARTX14 §5's exact Wave A14.1 gate: "Greedy (`TopK{1}`) reproduces
    /// ARTX05's argmax **bit-identically**." Reuses `runtime::sample::argmax`'s
    /// own test vectors (ties, NaN, all-NaN, single-element, empty) directly
    /// rather than inventing new ones — the gate is that these two functions
    /// agree, so the fixtures should be the ones that already pin the
    /// function this one must match.
    #[test]
    fn greedy_chain_is_bit_identical_to_argmax() {
        let cases: &[&[f32]] = &[
            &[0.1, 0.9, 0.3],
            &[-5.0, -1.0, -3.0],
            &[2.0],
            &[],
            &[1.0, 1.0, 1.0],
            &[0.5, 2.0, 2.0],
            &[f32::NAN, 1.0, 2.0],
            &[1.0, f32::NAN, 2.0],
            &[2.0, f32::NAN, 1.0],
            &[f32::NAN, f32::NAN],
        ];
        let mut rng = rng_from_seed(Some(0));
        for logits in cases {
            let want = argmax(logits).map(|i| i as TokenId);
            let candidates = Candidates::new(logits.iter().enumerate().map(|(i, &l)| (i as TokenId, l)));
            let got = apply_chain_host(&SamplerChain::greedy(), candidates, &mut rng).unwrap();
            assert_eq!(got, want, "logits {logits:?}: chain gave {got:?}, argmax gave {want:?}");
        }
    }

    #[test]
    fn temperature_zero_is_rejected_rather_than_producing_infinite_logits() {
        // Confirms scale()'s own precondition — greedy must go through
        // TopK{1}, never Temperature{t:0.0}, which is exactly why
        // SamplerChain::greedy() and SamplerChain::for_params() both route
        // t<=0.0 away from a Temperature stage entirely.
        let result = std::panic::catch_unwind(|| {
            let mut c = Candidates::new([(0u32, 1.0f32)]);
            c.scale(0.0);
        });
        assert!(result.is_err());
    }

    #[test]
    fn top_k_truncates_to_exactly_k_highest_logits() {
        let mut c = Candidates::new([(0u32, 1.0f32), (1, 5.0), (2, 3.0), (3, 4.0)]);
        c.truncate(2);
        let ids: Vec<TokenId> = c.as_slice().iter().map(|(id, _)| *id).collect();
        assert_eq!(ids, vec![1, 3], "must keep the two highest logits, highest first");
    }

    #[test]
    fn nucleus_keeps_the_smallest_prefix_reaching_p() {
        // Two candidates each with ~50% probability (equal logits) — top-p=0.5
        // must keep at least the first, and in practice both given float
        // rounding at the boundary; top-p=0.99 must keep both.
        let mut narrow = Candidates::new([(0u32, 0.0f32), (1, 0.0)]);
        narrow.nucleus(0.3);
        assert_eq!(narrow.len(), 1, "a low p must not keep both equal-probability candidates");

        let mut wide = Candidates::new([(0u32, 0.0f32), (1, 0.0)]);
        wide.nucleus(0.99);
        assert_eq!(wide.len(), 2, "a high p must keep both");
    }

    #[test]
    fn nucleus_always_keeps_at_least_one_candidate() {
        let mut c = Candidates::new([(0u32, 10.0f32), (1, -10.0)]);
        c.nucleus(0.001);
        assert_eq!(c.len(), 1);
    }

    #[test]
    fn min_p_drops_candidates_far_below_the_top_probability() {
        let mut c = Candidates::new([(0u32, 10.0f32), (1, -10.0), (2, 9.0)]);
        c.min_p(0.5);
        let ids: Vec<TokenId> = c.as_slice().iter().map(|(id, _)| *id).collect();
        assert!(ids.contains(&0), "the top candidate must always survive");
        assert!(!ids.contains(&1), "a vastly lower-probability candidate must be dropped");
    }

    #[test]
    fn typical_keeps_at_least_one_candidate_and_never_grows_the_set() {
        let mut c = Candidates::new([(0u32, 1.0f32), (1, 2.0), (2, 0.5), (3, 3.0)]);
        let original_len = c.len();
        c.typical(0.5);
        assert!(!c.is_empty());
        assert!(c.len() <= original_len);
    }

    #[test]
    fn sample_over_a_single_survivor_is_deterministic_across_calls() {
        let c = Candidates::new([(7u32, 1.0f32)]);
        let mut rng_a = rng_from_seed(Some(1));
        let mut rng_b = rng_from_seed(Some(2));
        assert_eq!(c.sample(&mut rng_a), Some(7));
        assert_eq!(c.sample(&mut rng_b), Some(7), "a single survivor must not depend on the RNG");
    }

    #[test]
    fn sample_over_an_empty_candidate_set_is_none() {
        let c = Candidates::new(std::iter::empty());
        let mut rng = rng_from_seed(Some(0));
        assert_eq!(c.sample(&mut rng), None);
    }

    #[test]
    fn sample_is_reproducible_for_a_fixed_seed() {
        let logits = [(0u32, 1.0f32), (1, 1.0), (2, 1.0), (3, 1.0)];
        let mut rng1 = rng_from_seed(Some(42));
        let mut rng2 = rng_from_seed(Some(42));
        let a = Candidates::new(logits).sample(&mut rng1);
        let b = Candidates::new(logits).sample(&mut rng2);
        assert_eq!(a, b, "the same seed must produce the same draw");
    }

    #[test]
    fn apply_chain_host_refuses_an_invalid_chain_rather_than_sampling() {
        let chain = SamplerChain {
            stages: vec![Stage::TopK { k: 1 }, Stage::RepetitionPenalty { penalty: 1.1, window: 8 }],
            seed: None,
        };
        let candidates = Candidates::new([(0u32, 1.0f32)]);
        let mut rng = rng_from_seed(Some(0));
        assert!(apply_chain_host(&chain, candidates, &mut rng).is_err());
    }

    #[test]
    fn apply_chain_host_full_pipeline_stays_within_the_original_candidate_ids() {
        let p = SamplingParams {
            temperature: 0.8,
            top_k: Some(3),
            top_p: Some(0.95),
            ..Default::default()
        };
        let chain = SamplerChain::openai_default(&p);
        let original: Vec<(TokenId, f32)> =
            (0..20).map(|i| (i as TokenId, (i as f32 * 0.37).sin() * 5.0)).collect();
        let ids: std::collections::HashSet<TokenId> = original.iter().map(|(id, _)| *id).collect();
        let mut rng = rng_from_seed(Some(7));
        let got = apply_chain_host(&chain, Candidates::new(original), &mut rng).unwrap();
        assert!(got.is_some());
        assert!(ids.contains(&got.unwrap()), "the sampled id must be one of the originals");
    }
}
