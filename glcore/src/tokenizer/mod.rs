//! GwenLand's tokenizer — SentencePiece and byte-level BPE, with a
//! zero-allocation merge engine and exact Unicode character classes.
//!
//! Fourteen GGUF vocabulary families are verified **exact** against llama.cpp's
//! reference vectors, and anything this module cannot express is refused at
//! load time rather than approximated. `notes/gltokenizer-gguf-support-audit.md`
//! holds the per-family status; `tests/tokenizer_parity.rs` enforces it.
//!
//! # What this replaces, and why
//!
//! This module *was* the `gltokenizer` crate, which in turn replaced an
//! earlier `glcore::tokenizer` that passed its round-trip tests while
//! producing token ids matching no reference. That implementation is gone; the
//! six defects are kept here because each is a trap the shape of the problem
//! invites, not a one-off mistake.
//!
//! 1. **A compensating error.** `encode` unconditionally prepended `▁`, and
//!    `decode` stripped a leading space back off "so `decode(encode(text))`
//!    round-trips". The round trip was preserved; the ids were wrong.
//!    SentencePiece's `add_dummy_prefix` is a per-model setting — here,
//!    [`Vocab`]'s `add_dummy_prefix`.
//! 2. **The byte-level pre-tokenizer split only on `0x20`.** Real vocabularies
//!    use one of three fixed patterns that also split contractions, digit
//!    runs, punctuation runs and newlines. See [`pretok`].
//! 3. **Only U+0020 was mapped to `▁`.** Tabs and newlines passed through.
//! 4. **Special tokens in input text were shredded** into pieces instead of
//!    resolving to their id. See [`GllmTokenizer::encode`].
//! 5. **Symbols could vanish silently** when a vocabulary had no `unk` token.
//!    Encoding is now lossless or an explicit error.
//! 6. **`O(n³)` merging with an allocation per candidate pair.** Now
//!    `O(n log n)` with no allocation in the loop — see [`bpe`].
//!
//! # Layout
//!
//! No trait objects, no builders. Each module is one decision:
//!
//! | Module | Owns |
//! |---|---|
//! | [`bpe`] | the merge engine — spans in a heap, zero allocation in the loop |
//! | [`pretok`] | the splitter: which bytes may ever merge together |
//! | [`unicode_tables`] | exact `\p{L}` `\p{M}` `\p{N}` `\p{P}`, **generated** |
//! | [`style`] | which encoding convention a vocabulary uses |
//! | [`vocab`] | vocabulary data, plus the `tokenizer.json` loader |
//! | [`gguf`] | GGUF metadata → [`Vocab`], and the pre-tokenizer name table |
//! | [`spm`] | the two SentencePiece-surface encoders |
//! | this file | [`GllmTokenizer`]: dispatch, byte-level encode, decode |
//!
//! The BPE algorithm is standard and implemented directly from its definition.

pub(crate) mod bpe;
pub mod gguf;
pub mod pretok;
pub(crate) mod spm;
pub mod style;
pub(crate) mod unicode_tables;
pub mod vocab;

pub use pretok::{BpeSplit, Passes, PreTok};
pub use style::{Style, SPM_SPACE};
pub use vocab::{Vocab, VocabParts};

// The SentencePiece surface form is written by [`spm`] and read back here, so
// the `<0xNN>` helpers are shared rather than duplicated.
use spm::parse_byte_token;

use std::cell::RefCell;

thread_local! {
    /// Per-thread scratch buffers.
    ///
    /// ⚠️ These were once a `RefCell` field on [`GllmTokenizer`], which made the
    /// whole type `!Sync` and therefore unusable inside the engines (which
    /// require `Sync`). Thread-local storage keeps the buffers reusable —
    /// steady-state encoding still allocates only the output `Vec` — while
    /// leaving `GllmTokenizer` freely shareable across threads, which ARTX16's
    /// multi-slot serving needs.
    static SCRATCH: RefCell<Scratch> = RefCell::new(Scratch::default());
}

#[derive(Debug, thiserror::Error)]
pub enum TokError {
    #[error("vocabulary is empty")]
    EmptyVocab,
    #[error("{what} id {id} is outside a vocabulary of {vocab}")]
    IdOutOfRange {
        what: &'static str,
        id: u32,
        vocab: usize,
    },
    #[error("SPM vocabulary has {scores} scores for {vocab} tokens")]
    ScoreCountMismatch { scores: usize, vocab: usize },
    #[error("malformed GGUF: {0}")]
    Gguf(String),
    #[error("malformed tokenizer.json: {0}")]
    Json(String),
    #[error("tokenizer.json is missing `{0}`")]
    MissingField(&'static str),
    #[error("unsupported tokenizer model `{0}` (only BPE is implemented)")]
    UnsupportedModel(String),
    #[error("unsupported pre_tokenizer `{0}` — refusing rather than guessing, because a mis-split silently changes token ids")]
    UnsupportedPreTokenizer(String),
    #[error("cannot encode {ch:?}: no vocabulary entry, no byte fallback, and no unk token")]
    Unencodable { ch: String },
}

/// Encoder/decoder over a [`Vocab`].
///
/// Scratch buffers are held internally and reused, so steady-state encoding
/// allocates only the output `Vec<u32>`.
pub struct GllmTokenizer {
    v: Vocab,
    /// Identity for the thread-local pre-token cache; see `Scratch::cache_owner`.
    id: u64,
}

#[derive(Default)]
struct Scratch {
    merger: bpe::Merger,
    /// Byte-level: the byte→printable-char remapping of one chunk.
    mapped: String,
    /// SPM: the whole input with spaces replaced by `▁`.
    prepared: String,
    /// Pre-token → token ids, for the byte-level path. See [`PRETOK_CACHE_CAP`].
    cache: std::collections::HashMap<Box<str>, Box<[u32]>>,
    /// Which [`GllmTokenizer`] `cache` was filled for.
    ///
    /// ⛔ Load-bearing for correctness, not an optimisation. The cache lives in
    /// thread-local storage, so a thread that encodes with two different
    /// vocabularies would otherwise read the first one's ids back for the
    /// second — producing plausible output that is silently wrong for the
    /// model actually being run. `0` is "no owner yet"; ids start at 1.
    cache_owner: u64,
    hits: u64,
    misses: u64,
}

/// Maximum pre-tokens held per thread. Word frequency is long-tailed, so a
/// modest cache captures nearly all of the hits; past the cap this stops
/// inserting rather than evicting, because the entries earned early are the
/// frequent ones and churning them is worse than missing the rare ones.
const PRETOK_CACHE_CAP: usize = 16_384;

/// Pre-tokens longer than this are not cached: they are rare, they are the
/// ones least likely to recur, and they are what would make the cap expensive
/// in memory rather than in entries.
const PRETOK_CACHE_MAX_KEY: usize = 64;

/// Distinguishes tokenizers within a thread's cache. Wraps at `u64::MAX`,
/// which would take longer than any process lives.
static NEXT_TOKENIZER_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

thread_local! {
    /// Benchmark switch — see [`GllmTokenizer::set_pretoken_cache`].
    static CACHE_ON: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
}

/// Ranks candidate merges for a byte-level vocabulary: lower merge rank wins,
/// so the key is negated for the max-heap.
struct ByteLevelRanker<'a>(&'a Vocab);
impl bpe::Ranker for ByteLevelRanker<'_> {
    #[inline]
    fn rank(&self, piece: &str, left_len: usize) -> Option<i64> {
        let rules = self.0.merge_ranks.get(piece)?;
        // Only the rule whose split point matches this pair applies.
        rules
            .iter()
            .find(|(l, _)| *l as usize == left_len)
            .map(|(_, r)| -(*r as i64))
    }
}


impl GllmTokenizer {
    pub fn new(v: Vocab) -> Self {
        GllmTokenizer {
            v,
            id: NEXT_TOKENIZER_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        }
    }

    /// Turn the pre-token cache off for the current thread.
    ///
    /// ⚠️ **Diagnostic only.** The cache never changes the ids produced — it
    /// replays ids the merge engine already produced for the identical
    /// pre-token under the identical vocabulary — so switching it off is only
    /// useful for measuring what it is worth. Kept because an in-process A/B
    /// removes the build and thermal differences that make two separate
    /// binaries hard to compare on this hardware.
    pub fn set_pretoken_cache(on: bool) {
        CACHE_ON.with(|c| c.set(on));
    }

    /// `(hits, misses)` for the current thread since the last reset.
    pub fn pretoken_cache_stats() -> (u64, u64) {
        SCRATCH.with(|sc| {
            let s = sc.borrow();
            (s.hits, s.misses)
        })
    }

    /// Drop the current thread's cache and its counters.
    pub fn reset_pretoken_cache() {
        SCRATCH.with(|sc| {
            let mut s = sc.borrow_mut();
            s.cache.clear();
            s.cache_owner = 0;
            s.hits = 0;
            s.misses = 0;
        });
    }

    pub fn vocab(&self) -> &Vocab {
        &self.v
    }

    /// Encode `text`.
    ///
    /// Input is first split on any special tokens present in the vocabulary
    /// (longest match first), so a literal `<|im_start|>` in the text becomes
    /// that token's id rather than being shredded into text pieces.
    ///
    /// Returns an error rather than dropping anything it cannot represent.
    pub fn encode(&self, text: &str, add_bos: bool) -> Result<Vec<u32>, TokError> {
        let mut ids = Vec::new();
        if add_bos {
            if let Some(b) = self.v.bos_id {
                ids.push(b);
            }
        }
        self.encode_into(text, &mut ids)?;
        Ok(ids)
    }

    /// Encode without a BOS decision, appending to `ids`.
    pub fn encode_into(&self, text: &str, ids: &mut Vec<u32>) -> Result<(), TokError> {
        let mut rest = text;
        while !rest.is_empty() {
            match self.find_special(rest) {
                Some((at, len, id)) => {
                    if at > 0 {
                        self.encode_plain(&rest[..at], ids)?;
                    }
                    ids.push(id);
                    rest = &rest[at + len..];
                }
                None => {
                    self.encode_plain(rest, ids)?;
                    break;
                }
            }
        }
        Ok(())
    }

    /// Earliest special-token occurrence, preferring the longest match at a
    /// given position (`specials_by_len` is sorted longest-first).
    ///
    /// ⭐ **This used to be 71 % of a warm encode.** The obvious formulation —
    /// `s.find(tok)` for each special, keep the earliest — searches the entire
    /// remaining text once *per special token*. Qwen2.5 has 22 of them, so a
    /// 120 KiB prompt was scanned for **2.6 MiB** before a single byte was
    /// tokenized, and the cost grows with the vocabulary's special count for
    /// input that usually contains none of them. Measured: 37.6 of 52.8
    /// ns/byte.
    ///
    /// One left-to-right pass instead. Almost every special begins with the
    /// same byte (`<` for the ChatML and Llama families), so a 256-entry table
    /// skips the text until a byte that could begin one, and only then are the
    /// candidates tried. Returning at the first match is what makes it
    /// *earliest*; trying longest-first at that position is what makes it
    /// prefer `<|im_start|>` over a hypothetical `<|im`.
    ///
    /// ⚠️ Byte offsets are safe to slice at because UTF-8 is
    /// self-synchronising: a valid UTF-8 needle cannot match starting inside a
    /// multi-byte character of a valid UTF-8 haystack. The previous
    /// `str::find` relied on exactly the same property.
    fn find_special(&self, s: &str) -> Option<(usize, usize, u32)> {
        if self.v.specials_by_len.is_empty() {
            return None;
        }
        let b = s.as_bytes();
        let table = &self.v.special_first_byte;
        let mut i = 0usize;
        while i < b.len() {
            if !table[b[i] as usize] {
                i += 1;
                continue;
            }
            let rest = &b[i..];
            for (tok, id) in &self.v.specials_by_len {
                if !tok.is_empty() && rest.starts_with(tok.as_bytes()) {
                    return Some((i, tok.len(), *id));
                }
            }
            i += 1;
        }
        None
    }

    fn encode_plain(&self, text: &str, ids: &mut Vec<u32>) -> Result<(), TokError> {
        match self.v.style {
            Style::Spm => self.encode_spm(text, ids),
            Style::ByteLevel => self.encode_byte_level(text, ids),
            Style::SpmBpe => self.encode_spm_bpe(text, ids),
        }
    }


    /// Byte-level: pre-tokenize, remap bytes to printable chars, merge by rank.
    ///
    /// ⭐ Merging dominates: pre-tokenization measures ~2.7 ns/byte against
    /// ~103 ns/byte for the whole encode, so the merge loop is ~97 % of the
    /// cost. Real text is long-tailed, though — the same few hundred words
    /// carry most of a document — so the same pre-token is merged over and
    /// over to the identical result. The cache below replays that result
    /// instead, which is the only change that can move this path materially.
    fn encode_byte_level(&self, text: &str, ids: &mut Vec<u32>) -> Result<(), TokError> {
        SCRATCH.with(|sc| {
        let mut sc = sc.borrow_mut();
        // ⛔ A cache filled for a different vocabulary must never be read.
        // Same thread, two models, plausible-looking garbage — exactly the
        // silent-wrongness class this crate exists to remove.
        if sc.cache_owner != self.id {
            sc.cache.clear();
            sc.cache_owner = self.id;
        }
        let caching = CACHE_ON.with(|c| c.get());
        let Scratch {
            merger,
            mapped,
            cache,
            hits,
            misses,
            ..
        } = &mut *sc;

        let ranker = ByteLevelRanker(&self.v);
        let v = &self.v;
        let mut err = None;

        // ⚠️ Collected, not streamed — and that is measured, not incidental.
        // Buffering costs one 16-byte entry per pre-token (~475 KiB per 120 KiB
        // encoded) for a list read once and dropped, so streaming through the
        // `split` callback looks strictly better. It is not: the closure form
        // measured **8 % slower on the miss path** (cold cache 48-51 → 52-55
        // ns/byte, cache off 110-120 → 122-126), presumably because `merger.run`
        // no longer inlines through two closure layers. Warm improved slightly;
        // cold is the number that matters. Reverted.
        let mut chunks: Vec<&str> = Vec::new();
        v.pretok.split(text, |c| chunks.push(c));

        for chunk in chunks {
            // ⭐ The cache is keyed on the **raw** pre-token, not its
            // byte-mapped form. `byte_to_char` is a fixed bijection that does
            // not depend on the vocabulary, so the two keys are equivalent —
            // and this one is already a slice of the input, so a hit costs no
            // string building at all. Measured at 64 % of a warm encode before
            // the change; the mapping now runs only on the ~12 % that miss.
            if caching {
                if let Some(cached) = cache.get(chunk) {
                    ids.extend_from_slice(cached);
                    *hits += 1;
                    continue;
                }
                *misses += 1;
            }

            mapped.clear();
            for b in chunk.bytes() {
                mapped.push(v.byte_to_char[b as usize]);
            }
            // Where this chunk's ids begin, so they can be captured below.
            let start = ids.len();

            // Llama-3's `ignore_merges`: a pre-token already in the vocabulary
            // is emitted whole, because rank-order BPE may not be able to
            // reach it (see `gguf::ignore_merges_for`).
            if v.ignore_merges {
                if let Some(&id) = v.token_to_id.get(mapped.as_str()) {
                    ids.push(id);
                    if caching && chunk.len() <= PRETOK_CACHE_MAX_KEY && cache.len() < PRETOK_CACHE_CAP {
                        cache.insert(chunk.into(), Box::new([id]));
                    }
                    continue;
                }
            }
            merger.run(mapped, &ranker, |piece| {
                if err.is_some() {
                    return;
                }
                if let Some(&id) = v.token_to_id.get(piece) {
                    ids.push(id);
                } else if let Some(unk) = v.unk_id {
                    ids.push(unk);
                } else {
                    err = Some(TokError::Unencodable {
                        ch: piece.to_string(),
                    });
                }
            });
            if err.is_some() {
                break;
            }
            // ⚠️ Recorded only on the success path. Caching a partial result
            // from a chunk that errored would hand back a truncated id list on
            // the next occurrence, with no error to show for it.
            if caching && chunk.len() <= PRETOK_CACHE_MAX_KEY && cache.len() < PRETOK_CACHE_CAP {
                cache.insert(chunk.into(), ids[start..].into());
            }
        }
        match err {
            Some(e) => Err(e),
            None => Ok(()),
        }
        })
    }

    /// Decode ids to text.
    ///
    /// ⚠️ There is no compensating prefix strip here. `encode` only inserts a
    /// dummy prefix when the vocabulary asks for one, so decode does not need
    /// to undo anything.
    pub fn decode(&self, ids: &[u32], skip_special: bool) -> String {
        let mut bytes = Vec::with_capacity(ids.len() * 4);
        for &id in ids {
            if skip_special && self.v.special_ids.contains(&id) {
                continue;
            }
            self.token_bytes_into(id, &mut bytes);
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Raw bytes a token contributes to the output.
    pub fn token_bytes(&self, id: u32) -> Vec<u8> {
        let mut b = Vec::new();
        self.token_bytes_into(id, &mut b);
        b
    }

    fn token_bytes_into(&self, id: u32, out: &mut Vec<u8>) {
        let Some(tok) = self.v.id_to_token.get(id as usize) else {
            return;
        };
        match self.v.style {
            // SpmBpe shares SPM's *surface form* — `▁` for space, `<0xNN>` for
            // raw bytes — even though its merges come from a list.
            Style::Spm | Style::SpmBpe => {
                if let Some(b) = parse_byte_token(tok) {
                    out.push(b);
                    return;
                }
                for ch in tok.chars() {
                    if ch == SPM_SPACE {
                        out.push(b' ');
                    } else {
                        let mut buf = [0u8; 4];
                        out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                    }
                }
            }
            Style::ByteLevel => {
                for ch in tok.chars() {
                    match self.v.char_to_byte.get(&ch) {
                        Some(&b) => out.push(b),
                        None => {
                            let mut buf = [0u8; 4];
                            out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                        }
                    }
                }
            }
        }
    }

    /// A streaming decoder for this tokenizer (see [`Incremental`]).
    pub fn incremental(&self) -> Incremental<'_> {
        Incremental {
            tok: self,
            pending: Vec::new(),
        }
    }

    // ── convenience surface, mirroring what callers already used ─────────

    /// Load a vocabulary embedded in a GGUF file.
    pub fn from_gguf_path(path: &str) -> Result<Self, TokError> {
        let bytes = std::fs::read(path).map_err(|e| TokError::Gguf(format!("{path}: {e}")))?;
        Ok(GllmTokenizer::new(gguf::vocab_from_gguf(&bytes)?))
    }

    /// Load a HuggingFace `tokenizer.json`.
    pub fn from_hf_json_path(path: &str) -> Result<Self, TokError> {
        let src = std::fs::read_to_string(path).map_err(|e| TokError::Json(format!("{path}: {e}")))?;
        Ok(GllmTokenizer::new(Vocab::from_hf_json(&src)?))
    }

    pub fn vocab_size(&self) -> usize {
        self.v.len()
    }
    pub fn eos_id(&self) -> u32 {
        self.v.eos_id
    }
    pub fn bos_id(&self) -> Option<u32> {
        self.v.bos_id
    }
    pub fn add_bos_default(&self) -> bool {
        self.v.add_bos_default
    }
    pub fn is_stop_token(&self, id: u32) -> bool {
        self.v.stop_ids.contains(&id)
    }
    pub fn stop_token_ids(&self) -> &std::collections::HashSet<u32> {
        &self.v.stop_ids
    }

    /// Text a single token contributes, decoded independently.
    ///
    /// ⛔ **Not correct for streaming.** A multi-byte character routinely spans
    /// several tokens, so decoding one at a time yields `U+FFFD` for emoji and
    /// non-Latin scripts. Use [`GllmTokenizer::incremental`] to emit deltas; this
    /// is for inspection and debugging, where a lone token is the unit of
    /// interest.
    pub fn decode_token_text(&self, id: u32) -> String {
        let mut b = Vec::new();
        self.token_bytes_into(id, &mut b);
        String::from_utf8_lossy(&b).into_owned()
    }

    /// Encode a single-turn ChatML prompt, emitting `<|im_start|>` /
    /// `<|im_end|>` as their token ids.
    ///
    /// Returns `Ok(None)` when this vocabulary has no ChatML markers, so the
    /// caller can fall back to raw completion encoding.
    ///
    /// ⚠️ This is a *convenience for single-turn use*, not a template engine.
    /// ARTX13 §3 places real chat templating in the serving layer, where the
    /// model's own template lives; a hardcoded system prompt cannot be right
    /// for every model.
    pub fn encode_chat(&self, user: &str) -> Result<Option<Vec<u32>>, TokError> {
        let (Some(&im_start), Some(&im_end)) = (
            self.v.token_to_id.get("<|im_start|>"),
            self.v.token_to_id.get("<|im_end|>"),
        ) else {
            return Ok(None);
        };
        let mut ids = Vec::new();
        for (role, text) in [
            ("system", "You are a helpful assistant."),
            ("user", user),
        ] {
            ids.push(im_start);
            self.encode_into(&format!("{role}\n{text}"), &mut ids)?;
            ids.push(im_end);
            self.encode_into("\n", &mut ids)?;
        }
        ids.push(im_start);
        self.encode_into("assistant\n", &mut ids)?;
        Ok(Some(ids))
    }
}

/// Incremental decoder for token-by-token streaming.
///
/// Emitting `decode(&[one_id])` per token is wrong: a multi-byte character
/// commonly spans several tokens, so per-token decoding yields `U+FFFD` for
/// emoji and non-Latin scripts. This buffers an incomplete UTF-8 tail and
/// releases only complete characters.
pub struct Incremental<'a> {
    tok: &'a GllmTokenizer,
    pending: Vec<u8>,
}

impl Incremental<'_> {
    /// Push one token; return the text that is safe to emit now (possibly "").
    pub fn push(&mut self, id: u32) -> String {
        self.tok.token_bytes_into(id, &mut self.pending);
        match std::str::from_utf8(&self.pending) {
            Ok(s) => {
                let out = s.to_string();
                self.pending.clear();
                out
            }
            Err(e) => {
                let good = e.valid_up_to();
                let out = String::from_utf8_lossy(&self.pending[..good]).into_owned();
                match e.error_len() {
                    // Incomplete: keep the tail and wait for more tokens.
                    None => self.pending.drain(..good),
                    // Genuinely invalid: skip the offending byte, or the
                    // stream would stall forever. Byte-level vocabularies can
                    // emit sequences that are not valid UTF-8.
                    Some(bad) => self.pending.drain(..good + bad),
                };
                out
            }
        }
    }

    /// Flush at end of generation. A non-empty tail was a truncated character.
    pub fn finish(&mut self) -> String {
        if self.pending.is_empty() {
            String::new()
        } else {
            self.pending.clear();
            "\u{FFFD}".to_string()
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    // Only the decode side is used outside `spm`, so the encoder's half of the
    // `<0xNN>` pair is imported here rather than at module scope.
    use spm::fmt_byte_token;

    fn spm_vocab(add_dummy_prefix: bool) -> Vocab {
        let mut toks: Vec<String> = vec!["<unk>".into(), "<s>".into(), "</s>".into()];
        for b in 0..=255u16 {
            toks.push(format!("<0x{b:02X}>"));
        }
        for c in "abcdefghijklmnopqrstuvwxyzHW".chars() {
            toks.push(c.to_string());
        }
        toks.push(SPM_SPACE.to_string());
        // Distinct scores so merge ORDER is actually exercised — the previous
        // fixture used uniform 0.0, which disabled the comparison entirely.
        //
        // ⚠️ Every multi-char entry must be reachable by a chain of PAIRWISE
        // merges through other vocabulary entries. A fixture containing
        // "Hello" but not "Hel"/"Hell" cannot ever produce "Hello", because
        // "llo" would first have to form from "ll" or "lo" — neither of which
        // is a token. That is a property of SPM, not a bug, and it is easy to
        // get wrong when hand-writing a vocabulary.
        let merged = [
            ("He".to_string(), 5.0),
            ("Hel".to_string(), 6.0),
            ("Hell".to_string(), 7.0),
            ("Hello".to_string(), 8.0),
            ("Wo".to_string(), 5.0),
            ("Wor".to_string(), 6.0),
            ("Worl".to_string(), 7.0),
            ("World".to_string(), 8.0),
            (format!("{SPM_SPACE}Hello"), 9.0),
            (format!("{SPM_SPACE}World"), 9.0),
        ];
        let base = toks.len();
        for (t, _) in &merged {
            toks.push(t.clone());
        }
        let mut scores = vec![0.0f32; base];
        scores.extend(merged.iter().map(|(_, s)| *s));

        Vocab::from_parts(VocabParts {
            id_to_token: toks,
            scores,
            merges: vec![],
            special_ids: vec![1, 2],
            style: Style::Spm,
            pretok: PreTok::None,
            add_dummy_prefix,
            ignore_merges: false,
            bos_id: Some(1),
            eos_id: 2,
            unk_id: Some(0),
            add_bos_default: true,
        })
        .unwrap()
    }

    #[test]
    fn spm_round_trip_ascii() {
        let t = GllmTokenizer::new(spm_vocab(true));
        for s in ["Hello World", "Hello", "abc def", "a  b", " leading"] {
            let ids = t.encode(s, false).unwrap();
            assert_eq!(t.decode(&ids, true), format!(" {s}"), "input {s:?}");
        }
    }

    /// ⭐ The regression that matters: with `add_dummy_prefix = false` there is
    /// no phantom leading space, and decode does not compensate for one.
    #[test]
    fn no_dummy_prefix_means_no_phantom_space() {
        let t = GllmTokenizer::new(spm_vocab(false));
        let ids = t.encode("Hello", false).unwrap();
        assert_eq!(t.decode(&ids, true), "Hello");

        let with = GllmTokenizer::new(spm_vocab(true));
        let ids2 = with.encode("Hello", false).unwrap();
        // The two configurations MUST produce different ids. The old
        // implementation could not express this difference at all.
        assert_ne!(ids, ids2, "add_dummy_prefix must change the token ids");
    }

    #[test]
    fn spm_byte_fallback() {
        let t = GllmTokenizer::new(spm_vocab(false));
        let ids = t.encode("ab!", false).unwrap();
        assert_eq!(t.decode(&ids, true), "ab!");
    }

    #[test]
    fn spm_prefers_higher_score_merges() {
        let t = GllmTokenizer::new(spm_vocab(false));
        // "Hello" (8.0) must beat "He"(5.0) + "llo"(6.0) as a single token.
        let ids = t.encode("Hello", false).unwrap();
        assert_eq!(ids.len(), 1, "expected one token, got {ids:?}");
        assert_eq!(t.vocab().token_str(ids[0]), Some("Hello"));
    }

    fn byte_level_vocab() -> Vocab {
        let mut toks = vec!["<|endoftext|>".to_string()];
        let (b2c, _) = {
            let v = Vocab::from_parts(VocabParts {
                id_to_token: vec!["x".into()],
                scores: vec![],
                merges: vec![],
                special_ids: vec![],
                style: Style::ByteLevel,
                pretok: PreTok::Bpe(BpeSplit::GPT2),
                add_dummy_prefix: false,
                ignore_merges: false,
                bos_id: None,
                eos_id: 0,
                unk_id: None,
                add_bos_default: false,
            })
            .unwrap();
            (v.byte_to_char, ())
        };
        for c in b2c.iter() {
            toks.push(c.to_string());
        }
        Vocab::from_parts(VocabParts {
            id_to_token: toks,
            scores: vec![],
            merges: vec![],
            special_ids: vec![0],
            style: Style::ByteLevel,
            pretok: PreTok::Bpe(BpeSplit::QWEN2),
            add_dummy_prefix: false,
            ignore_merges: false,
            bos_id: None,
            eos_id: 0,
            unk_id: None,
            add_bos_default: false,
        })
        .unwrap()
    }

    #[test]
    fn byte_level_round_trip() {
        let t = GllmTokenizer::new(byte_level_vocab());
        for s in ["Hello, World!", "don't stop", "日本語", "hi 👋", "a\n\nb", "1234"] {
            let ids = t.encode(s, false).unwrap();
            assert_eq!(t.decode(&ids, false), s, "input {s:?}");
        }
    }

    /// The pre-token cache must be invisible in the output.
    #[test]
    fn pretoken_cache_is_transparent() {
        let t = GllmTokenizer::new(byte_level_vocab());
        for s in [
            "the cat sat on the mat, the cat sat",
            "don't don't don't",
            "日本語 日本語 日本語",
            "a a a a a a a a",
            "",
            "\n\n\n",
            "1234 1234 1234",
        ] {
            GllmTokenizer::set_pretoken_cache(false);
            GllmTokenizer::reset_pretoken_cache();
            let uncached = t.encode(s, false).unwrap();

            GllmTokenizer::set_pretoken_cache(true);
            GllmTokenizer::reset_pretoken_cache();
            // Twice: the first fills the cache, the second reads it.
            let _ = t.encode(s, false).unwrap();
            let cached = t.encode(s, false).unwrap();

            assert_eq!(uncached, cached, "cache changed the ids for {s:?}");
        }
        GllmTokenizer::set_pretoken_cache(true);
    }

    /// ⛔ The failure mode this cache actually risks.
    ///
    /// The cache is thread-local, so two tokenizers used on one thread share
    /// the storage. Without an owner check the second would read the first's
    /// ids back — same pre-token, different vocabulary, plausible-looking
    /// output that is silently wrong for the model being run. Nothing else in
    /// the suite would catch that: both tokenizers work perfectly alone.
    ///
    /// This test fails on any implementation that omits `Scratch::cache_owner`.
    #[test]
    fn pretoken_cache_does_not_leak_between_tokenizers() {
        // Same surface forms, ids shifted by one: identical text, different ids.
        let shifted = {
            let base = byte_level_vocab();
            let mut toks = vec!["<|pad|>".to_string()];
            for i in 0..base.len() {
                toks.push(base.token_str(i as u32).unwrap().to_string());
            }
            Vocab::from_parts(VocabParts {
                id_to_token: toks,
                scores: vec![],
                merges: vec![],
                special_ids: vec![1],
                style: Style::ByteLevel,
                pretok: PreTok::Bpe(BpeSplit::QWEN2),
                add_dummy_prefix: false,
                ignore_merges: false,
                bos_id: None,
                eos_id: 1,
                unk_id: None,
                add_bos_default: false,
            })
            .unwrap()
        };
        let a = GllmTokenizer::new(byte_level_vocab());
        let b = GllmTokenizer::new(shifted);
        const S: &str = "the cat sat on the mat";

        GllmTokenizer::set_pretoken_cache(false);
        GllmTokenizer::reset_pretoken_cache();
        let want_a = a.encode(S, false).unwrap();
        let want_b = b.encode(S, false).unwrap();
        assert_ne!(want_a, want_b, "fixture is broken: vocabs must differ");

        // Interleave them on this thread, repeatedly, so a leak would show.
        GllmTokenizer::set_pretoken_cache(true);
        GllmTokenizer::reset_pretoken_cache();
        for _ in 0..4 {
            assert_eq!(a.encode(S, false).unwrap(), want_a, "tokenizer A polluted");
            assert_eq!(b.encode(S, false).unwrap(), want_b, "tokenizer B polluted");
        }
    }

    /// ⭐ Bug 4: a literal special token in the input must resolve to its id.
    #[test]
    fn special_tokens_in_text_are_not_shredded() {
        let t = GllmTokenizer::new(byte_level_vocab());
        let ids = t.encode("a<|endoftext|>b", false).unwrap();
        assert!(ids.contains(&0), "special id missing from {ids:?}");
        // And it must be a single id, not a run of byte tokens.
        let specials = ids.iter().filter(|&&i| i == 0).count();
        assert_eq!(specials, 1);
    }

    /// ⭐ Bug 5: nothing is dropped silently.
    #[test]
    fn unencodable_is_an_error_not_a_silent_drop() {
        // A vocab with no byte fallback and no unk.
        let v = Vocab::from_parts(VocabParts {
            id_to_token: vec!["a".into()],
            scores: vec![0.0],
            merges: vec![],
            special_ids: vec![],
            style: Style::Spm,
            pretok: PreTok::None,
            add_dummy_prefix: false,
            ignore_merges: false,
            bos_id: None,
            eos_id: 0,
            unk_id: None,
            add_bos_default: false,
        })
        .unwrap();
        let t = GllmTokenizer::new(v);
        assert!(matches!(
            t.encode("z", false),
            Err(TokError::Unencodable { .. })
        ));
    }

    /// ⭐ The streaming invariant from ARTX13 A13.3: concatenated deltas must
    /// equal the whole-sequence decode.
    #[test]
    fn incremental_deltas_concatenate_to_whole_decode() {
        let t = GllmTokenizer::new(byte_level_vocab());
        for s in ["hi 👋 there", "日本語のテキスト", "plain ascii", "mixed 漢字 and 🎉"] {
            let ids = t.encode(s, false).unwrap();
            let mut inc = t.incremental();
            let mut streamed = String::new();
            for &id in &ids {
                streamed.push_str(&inc.push(id));
            }
            streamed.push_str(&inc.finish());
            assert_eq!(streamed, t.decode(&ids, false), "stream mismatch for {s:?}");
            assert!(!streamed.contains('\u{FFFD}'), "replacement char in {s:?}");
        }
    }

    #[test]
    fn byte_token_formatting_round_trips() {
        for b in 0..=255u8 {
            let mut buf = [0u8; 6];
            let s = fmt_byte_token(b, &mut buf);
            assert_eq!(parse_byte_token(s), Some(b));
        }
    }
}
