//! The byte-pair merge engine.
//!
//! # Why this is written the way it is
//!
//! The naive formulation rescans every adjacent pair on every merge and
//! allocates a `String` per candidate, giving `O(n³)` with `O(n²)`
//! allocations. This module does the same *algorithm* — repeatedly merge the
//! best-ranked adjacent pair — with two changes that remove both costs:
//!
//! 1. **Symbols are spans, not strings.** Every symbol is a `(start, len)`
//!    pair into one immutable scratch buffer. Merging two *adjacent* symbols
//!    is `left.len += right.len`, because their bytes are already contiguous.
//!    The concatenation therefore needs no allocation to look up: it is
//!    `&buf[left.start .. left.start + left.len]` after the widening.
//!
//! 2. **Candidates live in a heap, not in a rescan.** After a merge only two
//!    new adjacencies appear (`prev↔merged` and `merged↔next`), so they are
//!    pushed and the rest of the frontier is left alone.
//!
//! Stale heap entries are unavoidable — a pair can be invalidated by a merge
//! that happens before it is popped — so entries record the exact pair they
//! were built from and are validated on pop. This is cheaper than trying to
//! delete from the heap.
//!
//! Result: `O(n log n)` time, and **zero allocation inside the merge loop**.

use std::collections::BinaryHeap;

/// A symbol: a contiguous span of the scratch buffer, in a doubly-linked list.
///
/// `len == 0` marks a symbol that was absorbed into its left neighbour.
#[derive(Clone, Copy)]
struct Sym {
    start: u32,
    len: u32,
    prev: i32,
    next: i32,
}

/// A pending merge. Ordered so that `BinaryHeap`'s max-pop yields the
/// *best* candidate for both vocabulary styles.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Cand {
    /// Pre-normalised priority: higher is better, always.
    ///
    /// SPM ranks by score (higher wins), byte-level by merge rank (lower
    /// wins). Normalising at push time keeps one comparison path here
    /// instead of a style branch in the hot loop.
    key: i64,
    /// Ties break leftmost, so this is stored negated (bigger = further left).
    neg_start: i64,
    left: u32,
    right: u32,
    /// Symbol lengths at the moment this candidate was priced.
    ///
    /// ⚠️ Adjacency alone is NOT enough to detect staleness: a merge can widen
    /// a symbol without unlinking it, so a pair priced as `a|b` can still look
    /// adjacent after `b` has grown into `bc` — and firing it would merge
    /// `abc`, a piece that was never ranked. Comparing the lengths catches
    /// exactly that case.
    llen: u32,
    rlen: u32,
}

impl Ord for Cand {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.key
            .cmp(&other.key)
            .then_with(|| self.neg_start.cmp(&other.neg_start))
    }
}
impl PartialOrd for Cand {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// How a candidate pair is priced. Returns `None` when the pair is not
/// mergeable at all.
pub trait Ranker {
    /// `piece` is the concatenation of the two symbols, borrowed from the
    /// scratch buffer — no allocation has happened. `left_len` is the byte
    /// length of the left symbol.
    ///
    /// ⚠️ `left_len` is not optional bookkeeping: a merge list is a list of
    /// PAIRS, and different splits of the same string are different rules with
    /// different ranks. Keying only on the concatenation silently collapses
    /// them — measured at 152,403 of 280,147 rules lost on llama-bpe.
    fn rank(&self, piece: &str, left_len: usize) -> Option<i64>;
}

/// Reusable working memory. Holding it across calls keeps the encoder
/// allocation-free after warm-up.
pub struct Merger {
    syms: Vec<Sym>,
    heap: BinaryHeap<Cand>,
}

impl Default for Merger {
    fn default() -> Self {
        Self::new()
    }
}

impl Merger {
    pub fn new() -> Self {
        Merger {
            syms: Vec::new(),
            heap: BinaryHeap::new(),
        }
    }

    /// Merge `buf` into pieces, calling `out` with each final piece in order.
    ///
    /// `buf` is treated as immutable; symbols start as one per `char`.
    pub fn run<R: Ranker>(&mut self, buf: &str, ranker: &R, mut out: impl FnMut(&str)) {
        self.syms.clear();
        self.heap.clear();
        if buf.is_empty() {
            return;
        }

        // One symbol per char. Indices are into `self.syms`.
        for (i, (off, ch)) in buf.char_indices().enumerate() {
            let i = i as i32;
            self.syms.push(Sym {
                start: off as u32,
                len: ch.len_utf8() as u32,
                prev: i - 1,
                next: i + 1,
            });
        }
        let n = self.syms.len();
        self.syms[n - 1].next = -1;

        // Seed the frontier.
        for i in 0..n.saturating_sub(1) {
            self.try_push(buf, ranker, i as u32, (i + 1) as u32);
        }

        while let Some(c) = self.heap.pop() {
            let (l, r) = (c.left as usize, c.right as usize);
            // Lazy deletion: the pair must still be adjacent, both alive, and
            // still exactly the pieces that were priced (see `Cand::llen`).
            if self.syms[l].len != c.llen
                || self.syms[r].len != c.rlen
                || self.syms[l].next != r as i32
            {
                continue;
            }

            // Absorb right into left. Their bytes are already contiguous.
            self.syms[l].len += self.syms[r].len;
            self.syms[r].len = 0;

            let after = self.syms[r].next;
            self.syms[l].next = after;
            if after >= 0 {
                self.syms[after as usize].prev = l as i32;
            }

            // Only two adjacencies changed.
            let before = self.syms[l].prev;
            if before >= 0 {
                self.try_push(buf, ranker, before as u32, l as u32);
            }
            if after >= 0 {
                self.try_push(buf, ranker, l as u32, after as u32);
            }
        }

        // Walk the surviving list.
        let mut i = 0i32;
        while i >= 0 {
            let s = self.syms[i as usize];
            if s.len > 0 {
                out(&buf[s.start as usize..(s.start + s.len) as usize]);
            }
            i = s.next;
        }
    }

    #[inline]
    fn try_push<R: Ranker>(&mut self, buf: &str, ranker: &R, l: u32, r: u32) {
        let a = self.syms[l as usize];
        let b = self.syms[r as usize];
        // Contiguity is an invariant of "merge adjacent only"; assert it in
        // debug so a future change that breaks it fails loudly rather than
        // silently hashing the wrong bytes.
        debug_assert_eq!(a.start + a.len, b.start);
        let piece = &buf[a.start as usize..(b.start + b.len) as usize];
        if let Some(key) = ranker.rank(piece, a.len as usize) {
            self.heap.push(Cand {
                key,
                neg_start: -(a.start as i64),
                left: l,
                right: r,
                llen: a.len,
                rlen: b.len,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Byte-level style: lower merge rank wins, so the key is negated.
    struct RankList(HashMap<String, i64>);
    impl Ranker for RankList {
        fn rank(&self, piece: &str, _left_len: usize) -> Option<i64> {
            self.0.get(piece).map(|r| -r)
        }
    }

    fn collect(buf: &str, r: &RankList) -> Vec<String> {
        let mut m = Merger::new();
        let mut v = Vec::new();
        m.run(buf, r, |p| v.push(p.to_string()));
        v
    }

    #[test]
    fn no_merges_yields_chars() {
        let r = RankList(HashMap::new());
        assert_eq!(collect("abc", &r), ["a", "b", "c"]);
    }

    #[test]
    fn applies_merges_in_rank_order() {
        // "bc" (rank 0) must merge before "ab" (rank 1), giving ["a", "bc"].
        let mut m = HashMap::new();
        m.insert("bc".to_string(), 0);
        m.insert("ab".to_string(), 1);
        assert_eq!(collect("abc", &RankList(m)), ["a", "bc"]);
    }

    #[test]
    fn merges_cascade() {
        let mut m = HashMap::new();
        m.insert("ab".to_string(), 0);
        m.insert("abc".to_string(), 1);
        m.insert("abcd".to_string(), 2);
        assert_eq!(collect("abcd", &RankList(m)), ["abcd"]);
    }

    /// Equal rank must resolve leftmost — the tie-break the naive
    /// implementation got right only by accident of scan order.
    #[test]
    fn ties_break_leftmost() {
        let mut m = HashMap::new();
        m.insert("aa".to_string(), 0);
        // "aaa": merging leftmost gives ["aa", "a"]; rightmost gives ["a","aa"].
        assert_eq!(collect("aaa", &RankList(m)), ["aa", "a"]);
    }

    #[test]
    fn handles_multibyte_chars() {
        let mut m = HashMap::new();
        m.insert("日本".to_string(), 0);
        assert_eq!(collect("日本語", &RankList(m)), ["日本", "語"]);
    }

    #[test]
    fn empty_input_is_empty() {
        assert_eq!(collect("", &RankList(HashMap::new())), Vec::<String>::new());
    }

    /// Stale heap entries must be skipped, not applied. Here "ab" and "bc"
    /// both rank; whichever loses becomes stale and must not fire.
    #[test]
    fn stale_candidates_are_skipped() {
        let mut m = HashMap::new();
        m.insert("ab".to_string(), 0);
        m.insert("bc".to_string(), 1);
        // "ab" wins, so "bc" is stale: b is gone. Result: ["ab", "c"].
        assert_eq!(collect("abc", &RankList(m)), ["ab", "c"]);
    }
}
