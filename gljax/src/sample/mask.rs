//! `AllowMask` + `MaskSource` — the seam to ARTX15 (ARTX14 §3.3).
//!
//! ⚠️ **Ordering is forced, not chosen.** If truncation (top-K etc.) ran
//! before the grammar mask, the surviving candidates could be *entirely*
//! masked, leaving nothing to sample and forcing a fallback that would
//! silently violate the grammar. The mask must shrink the candidate set
//! before truncation ever runs — [`Stage::GrammarMask`](crate::sample::chain::Stage::GrammarMask)
//! is one of the two stage kinds `SamplerChain::validate` requires to
//! precede every truncation stage.
//!
//! ⚠️ **`MaskSource` is a trait so ARTX14 never depends on ARTX15.** Sampling
//! compiles and ships without any grammar support; structured generation
//! plugs in by implementing this trait, not by ARTX14 importing ARTX15's
//! grammar engine.
//!
//! ⛔ **Scope note.** This wave builds the interface and the always-allow
//! degenerate case; it does not build a real grammar engine (that is Wave
//! B7/ARTX15's job) or the device-side upload path (§3.3's own costed
//! finding: 18.5 KB/slot/step bit-packed, 4.6x the top-K download traffic at
//! 64 slots — a real cost belonging to ARTX15's design, not restated here as
//! if it were solved).

use crate::sample::chain::TokenId;

/// A lightweight per-request identity for a mask/penalty state to be keyed
/// on. ARTX07 (continuous batching / slot accounting) does not exist in this
/// codebase yet — this is a placeholder narrow enough not to assume ARTX07's
/// shape, wide enough that `MaskSource` implementors (Wave B7's grammar
/// engine) don't need a signature change once ARTX07 lands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SlotId(pub usize);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MaskError {
    /// `accept` was called with a token the grammar does not allow in its
    /// current state — a caller bug (the mask should have excluded it).
    TokenNotAllowed(TokenId),
}

impl std::fmt::Display for MaskError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MaskError::TokenNotAllowed(id) => write!(f, "token {id} is not allowed by the current grammar state"),
        }
    }
}

impl std::error::Error for MaskError {}

/// A per-slot, per-step allow-mask over the vocabulary. Bit-packed:
/// `ceil(vocab_size / 64)` `u64` words — 151,936 bits is 18.5 KB, matching
/// ARTX14 §3.3's costed figure exactly (`151_936 / 8 = 18_992` bytes).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AllowMask {
    bits: Vec<u64>,
    vocab_size: usize,
}

impl AllowMask {
    /// Every token disallowed. The safe starting point for a grammar to
    /// selectively enable tokens from.
    pub fn none_allowed(vocab_size: usize) -> Self {
        AllowMask { bits: vec![0u64; vocab_size.div_ceil(64)], vocab_size }
    }

    /// Every token allowed — the degenerate case this wave's own test gate
    /// requires: "a trivial always-allow mask changes nothing" (ARTX14 §5's
    /// Wave A14.4 gate).
    pub fn all_allowed(vocab_size: usize) -> Self {
        let words = vocab_size.div_ceil(64);
        let mut bits = vec![u64::MAX; words];
        // Clear the tail bits past vocab_size in the last word, so
        // `allowed_ids` (which scans exactly `vocab_size` bits) is the only
        // thing that matters, but any future bit-level consumer (e.g. a
        // popcount) doesn't see phantom allowed ids past the vocabulary.
        let used_bits_in_last_word = vocab_size % 64;
        if used_bits_in_last_word != 0 {
            if let Some(last) = bits.last_mut() {
                *last &= (1u64 << used_bits_in_last_word) - 1;
            }
        }
        AllowMask { bits, vocab_size }
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn is_allowed(&self, id: TokenId) -> bool {
        let id = id as usize;
        if id >= self.vocab_size {
            return false;
        }
        (self.bits[id / 64] >> (id % 64)) & 1 == 1
    }

    pub fn allow(&mut self, id: TokenId) {
        let id = id as usize;
        assert!(id < self.vocab_size, "AllowMask::allow: id {id} out of range for vocab {}", self.vocab_size);
        self.bits[id / 64] |= 1 << (id % 64);
    }

    pub fn disallow(&mut self, id: TokenId) {
        let id = id as usize;
        assert!(id < self.vocab_size, "AllowMask::disallow: id {id} out of range for vocab {}", self.vocab_size);
        self.bits[id / 64] &= !(1 << (id % 64));
    }

    /// Applies this mask to a candidate list, dropping every disallowed
    /// entry. This is the host-side equivalent of ARTX14's device-side
    /// `ops::apply_allow_mask` (which sets disallowed logits to `-inf`
    /// instead of removing them) — behaviorally equivalent for what a
    /// downstream truncation stage sees, since a `-inf` logit never survives
    /// any of `chain.rs`'s truncation stages either.
    pub fn filter(&self, candidates: &mut Vec<(TokenId, f32)>) {
        candidates.retain(|(id, _)| self.is_allowed(*id));
    }
}

/// Produced by ARTX15 (a grammar engine), consumed here. `AllowMask`/
/// `MaskSource` know nothing about grammars — only about which tokens are,
/// for now, allowed.
pub trait MaskSource: Send {
    /// Computes the allow-mask for `slot`'s next token, given its history so
    /// far. `out` is provided by the caller so the same buffer can be reused
    /// across steps rather than allocating one per step.
    fn mask_for(&mut self, slot: SlotId, history: &[TokenId], out: &mut AllowMask);

    /// Advances the grammar state after `token` is actually committed.
    /// # Errors
    /// If `token` was not allowed by the mask this source most recently
    /// produced for `slot` — a caller bug, per [`MaskError`].
    fn accept(&mut self, slot: SlotId, token: TokenId) -> Result<(), MaskError>;
}

/// The trivial [`MaskSource`]: allows everything, forever. Exists so
/// sampling has a concrete, always-available implementor with no grammar
/// engine attached — and so this wave's gate ("a trivial always-allow mask
/// changes nothing") has something real to test against.
pub struct AlwaysAllow {
    vocab_size: usize,
}

impl AlwaysAllow {
    pub fn new(vocab_size: usize) -> Self {
        AlwaysAllow { vocab_size }
    }
}

impl MaskSource for AlwaysAllow {
    fn mask_for(&mut self, _slot: SlotId, _history: &[TokenId], out: &mut AllowMask) {
        *out = AllowMask::all_allowed(self.vocab_size);
    }

    fn accept(&mut self, _slot: SlotId, _token: TokenId) -> Result<(), MaskError> {
        Ok(())
    }
}

/// A [`MaskSource`] that only ever allows one specific token — the other
/// half of this wave's gate: "a single-token mask forces that token."
/// Useful on its own (e.g. forcing a specific opening token) and as the
/// simplest possible non-trivial `MaskSource` to test the seam against
/// before a real grammar engine exists.
pub struct ForceToken {
    vocab_size: usize,
    forced: TokenId,
}

impl ForceToken {
    pub fn new(vocab_size: usize, forced: TokenId) -> Self {
        ForceToken { vocab_size, forced }
    }
}

impl MaskSource for ForceToken {
    fn mask_for(&mut self, _slot: SlotId, _history: &[TokenId], out: &mut AllowMask) {
        *out = AllowMask::none_allowed(self.vocab_size);
        out.allow(self.forced);
    }

    fn accept(&mut self, _slot: SlotId, token: TokenId) -> Result<(), MaskError> {
        if token == self.forced {
            Ok(())
        } else {
            Err(MaskError::TokenNotAllowed(token))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_allowed_allows_every_id_up_to_vocab_size_and_nothing_past_it() {
        let mask = AllowMask::all_allowed(10);
        for id in 0..10 {
            assert!(mask.is_allowed(id), "{id}");
        }
        assert!(!mask.is_allowed(10), "past the vocabulary must be disallowed");
        assert!(!mask.is_allowed(1000));
    }

    #[test]
    fn none_allowed_allows_nothing() {
        let mask = AllowMask::none_allowed(10);
        for id in 0..10 {
            assert!(!mask.is_allowed(id), "{id}");
        }
    }

    #[test]
    fn allow_and_disallow_toggle_a_single_bit_without_disturbing_others() {
        let mut mask = AllowMask::none_allowed(128);
        mask.allow(5);
        mask.allow(70); // crosses into the second u64 word
        assert!(mask.is_allowed(5));
        assert!(mask.is_allowed(70));
        assert!(!mask.is_allowed(4));
        assert!(!mask.is_allowed(71));
        mask.disallow(5);
        assert!(!mask.is_allowed(5));
        assert!(mask.is_allowed(70), "disallowing 5 must not disturb 70");
    }

    #[test]
    fn bit_packing_matches_the_documented_18_5_kb_at_full_vocab() {
        let mask = AllowMask::all_allowed(151_936);
        let bytes = mask.bits.len() * 8;
        // 151936 / 8 = 18992 bytes exactly (151936 is a multiple of 64).
        assert_eq!(bytes, 18_992);
    }

    #[test]
    fn filter_drops_every_disallowed_candidate() {
        let mut mask = AllowMask::none_allowed(10);
        mask.allow(2);
        mask.allow(7);
        let mut cand = vec![(1u32, 0.1f32), (2u32, 0.2), (5u32, 0.3), (7u32, 0.4)];
        mask.filter(&mut cand);
        let ids: Vec<u32> = cand.iter().map(|(id, _)| *id).collect();
        assert_eq!(ids, vec![2, 7]);
    }

    /// Wave A14.4's gate, half one: "a trivial always-allow mask changes
    /// nothing."
    #[test]
    fn always_allow_mask_source_changes_no_candidate() {
        let mut source = AlwaysAllow::new(10);
        let mut mask = AllowMask::none_allowed(10);
        source.mask_for(SlotId(0), &[], &mut mask);
        let mut cand: Vec<(TokenId, f32)> = (0..10).map(|i| (i, i as f32)).collect();
        let before = cand.clone();
        mask.filter(&mut cand);
        assert_eq!(cand, before, "always-allow must drop nothing");
        assert!(source.accept(SlotId(0), 3).is_ok());
    }

    /// Wave A14.4's gate, half two: "a single-token mask forces that token."
    #[test]
    fn force_token_mask_source_forces_exactly_one_token() {
        let mut source = ForceToken::new(10, 4);
        let mut mask = AllowMask::none_allowed(10);
        source.mask_for(SlotId(0), &[], &mut mask);
        let mut cand: Vec<(TokenId, f32)> = (0..10).map(|i| (i, i as f32)).collect();
        mask.filter(&mut cand);
        assert_eq!(cand, vec![(4, 4.0)]);
        assert!(source.accept(SlotId(0), 4).is_ok());
        assert!(source.accept(SlotId(0), 5).is_err());
    }
}
