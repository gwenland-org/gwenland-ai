//! `PenaltyState` — per-slot generation history for repetition/presence/
//! frequency penalties (ARTX14 §3.1).
//!
//! ⛔ **Scope note.** ARTX14 designs penalties as a *device-side*, sparse
//! `(indices, values)` upload applied before the top-K reduction (§3.1's own
//! design decision: "a penalty applied on the host after top-K cannot demote
//! a token *out* of the candidate set, which is most of what a penalty is
//! for"). That device path is Wave A14.3, gated on A14.2's device top-K
//! existing first — neither is built here (Wave A14.1's scope is `chain.rs` +
//! the host-only path). What's here is the host-side state itself:
//! `PenaltyState` and the arithmetic each penalty applies. It is correct and
//! useful standalone for the host-only (option A) sampling path this wave
//! does build, and it is the same state a future device upload would
//! serialize from — building it once, here, rather than twice.

use std::collections::{HashMap, VecDeque};

use crate::tok::TokenId;

/// Per-slot token history. ARTX14 §3.1: "lives beside ARTX07's slot state,
/// since it has exactly the same lifecycle: allocated with the slot, freed
/// with it." ARTX07 doesn't exist in this codebase yet, so today this is
/// owned per-request instead — the lifecycle argument still holds, just at a
/// smaller granularity than a slot.
#[derive(Debug, Clone, Default)]
pub struct PenaltyState {
    /// Occurrence counts. Vocab-sized *in principle* but sparse in practice —
    /// a generation touches a few hundred distinct tokens, so a `HashMap`
    /// costs kilobytes, not `vocab_size` entries.
    counts: HashMap<TokenId, u32>,
    /// Emission order, for the windowed repetition penalty.
    order: VecDeque<TokenId>,
    window: usize,
}

impl PenaltyState {
    /// `window` bounds how far back [`Self::in_window`] looks; `0` means
    /// unbounded (every token ever generated counts).
    pub fn new(window: usize) -> Self {
        PenaltyState { counts: HashMap::new(), order: VecDeque::new(), window }
    }

    /// Records one generated token. Call once per accepted token, in
    /// generation order.
    pub fn record(&mut self, token: TokenId) {
        *self.counts.entry(token).or_insert(0) += 1;
        self.order.push_back(token);
        if self.window > 0 {
            while self.order.len() > self.window {
                // ⚠️ Only the windowed repetition penalty's membership test
                // (`in_window`) shrinks with eviction — `counts` (used by
                // presence/frequency, which ARTX14 §3.1's table defines over
                // "any token seen" / "occurrence count", not a window) is
                // deliberately never decremented here. Frequency and presence
                // are whole-history penalties; only repetition is windowed.
                self.order.pop_front();
            }
        }
    }

    /// Whether `token` appears within the trailing `window` (repetition
    /// penalty's membership test — an in/out decision, not a count).
    pub fn in_window(&self, token: TokenId) -> bool {
        self.order.contains(&token)
    }

    /// Whether `token` has been generated at least once (presence penalty).
    pub fn seen(&self, token: TokenId) -> bool {
        self.counts.contains_key(&token)
    }

    /// How many times `token` has been generated (frequency penalty).
    pub fn count(&self, token: TokenId) -> u32 {
        self.counts.get(&token).copied().unwrap_or(0)
    }

    /// Applies the repetition penalty to `candidates` in place: ARTX14 §3.1's
    /// table — divide the logit by `penalty` if positive, multiply if
    /// negative (the standard convention: penalizing a positive logit should
    /// shrink it toward zero, penalizing a negative logit should push it
    /// further negative, and dividing a negative number by a penalty > 1
    /// would do the opposite of that).
    pub fn apply_repetition(&self, candidates: &mut [(TokenId, f32)], penalty: f32) {
        for (id, logit) in candidates.iter_mut() {
            if self.in_window(*id) {
                *logit = if *logit > 0.0 { *logit / penalty } else { *logit * penalty };
            }
        }
    }

    /// Subtracts a constant from every candidate seen at least once.
    pub fn apply_presence(&self, candidates: &mut [(TokenId, f32)], penalty: f32) {
        for (id, logit) in candidates.iter_mut() {
            if self.seen(*id) {
                *logit -= penalty;
            }
        }
    }

    /// Subtracts `count * penalty` from each candidate.
    pub fn apply_frequency(&self, candidates: &mut [(TokenId, f32)], penalty: f32) {
        for (id, logit) in candidates.iter_mut() {
            let count = self.count(*id);
            if count > 0 {
                *logit -= count as f32 * penalty;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_tracks_both_count_and_window_membership() {
        let mut state = PenaltyState::new(0);
        state.record(5);
        state.record(5);
        state.record(7);
        assert_eq!(state.count(5), 2);
        assert_eq!(state.count(7), 1);
        assert_eq!(state.count(9), 0);
        assert!(state.seen(5));
        assert!(!state.seen(9));
    }

    #[test]
    fn window_zero_means_unbounded_membership() {
        let mut state = PenaltyState::new(0);
        for i in 0..1000u32 {
            state.record(i);
        }
        assert!(state.in_window(0), "unbounded window must still remember the first token");
    }

    #[test]
    fn a_bounded_window_forgets_membership_past_its_size() {
        let mut state = PenaltyState::new(2);
        state.record(1);
        state.record(2);
        state.record(3); // evicts 1 from the window
        assert!(!state.in_window(1), "1 must have fallen out of a window of size 2");
        assert!(state.in_window(2));
        assert!(state.in_window(3));
        // Presence/frequency are whole-history, not windowed — 1 must still count.
        assert!(state.seen(1), "presence/frequency must not be windowed");
        assert_eq!(state.count(1), 1);
    }

    #[test]
    fn apply_repetition_shrinks_a_positive_logit_and_grows_a_negative_one_toward_zero() {
        let mut state = PenaltyState::new(0);
        state.record(0);
        state.record(1);
        let mut cand = vec![(0u32, 4.0f32), (1u32, -4.0f32), (2u32, 4.0f32)];
        state.apply_repetition(&mut cand, 2.0);
        assert_eq!(cand[0].1, 2.0, "positive logit divided by penalty");
        assert_eq!(cand[1].1, -8.0, "negative logit multiplied by penalty");
        assert_eq!(cand[2].1, 4.0, "unseen token untouched");
    }

    #[test]
    fn apply_presence_subtracts_a_flat_constant_regardless_of_count() {
        let mut state = PenaltyState::new(0);
        state.record(0);
        state.record(0);
        state.record(0);
        let mut cand = vec![(0u32, 10.0f32)];
        state.apply_presence(&mut cand, 1.5);
        assert_eq!(cand[0].1, 8.5, "presence is flat, not scaled by count");
    }

    #[test]
    fn apply_frequency_scales_by_occurrence_count() {
        let mut state = PenaltyState::new(0);
        state.record(0);
        state.record(0);
        state.record(0);
        let mut cand = vec![(0u32, 10.0f32)];
        state.apply_frequency(&mut cand, 1.0);
        assert_eq!(cand[0].1, 7.0, "seen 3 times, penalty 1.0 -> -3.0");
    }
}
