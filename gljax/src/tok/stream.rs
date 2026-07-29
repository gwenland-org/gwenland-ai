//! Incremental detokenization (ARTX13 §4) — generic over [`Tokenizer`].
//!
//! ⛔ **This duplicates a working algorithm on purpose.**
//! `glcore::tokenizer::GllmTokenizer::incremental` already implements exactly
//! this — buffer an incomplete UTF-8 tail, release only complete characters —
//! correctly, and with its own test coverage. It is hardwired to
//! `&GllmTokenizer` concretely. The whole reason `tok::Tokenizer` exists
//! (this module's parent docs) is to let gljax code hold *any* tokenizer
//! behind one type, including two different vocabularies live at once for
//! ARTX11 — a generic decoder that only works against the concrete
//! `glcore` type would defeat that purpose for the one piece of tokenizer
//! machinery every streaming response actually calls per token. The
//! algorithm below is intentionally the same one `glcore::tokenizer::Incremental`
//! already proved correct — re-expressed against [`Tokenizer::token_bytes`]
//! instead of a concrete field access is the entire diff.
//!
//! # Why per-token decode is wrong at all (ARTX13 §4.1)
//!
//! Two independent reasons, both real production failure modes, not
//! theoretical ones:
//!
//! 1. **Partial UTF-8.** A multi-byte character (emoji, CJK, Cyrillic)
//!    commonly spans two or more tokens. Decoding one token yields an
//!    incomplete UTF-8 sequence; lossy decoding renders that as `U+FFFD`,
//!    and the substitution happens *before* transmission — no client-side
//!    buffering can recover it after the fact.
//! 2. **Context-dependent spacing.** Byte-level BPE's leading-space
//!    convention and SPM's `▁` handling both make a token's rendered text a
//!    function of its neighbours, which a lone token cannot see.
//!
//! Only reason 1 is handled below (the `ByteLevel` strategy — concatenate
//! token bytes, split on UTF-8 boundaries, which is correct because a
//! byte-level token's bytes are context-independent). ARTX13 §4.2's SPM
//! strategy (re-decode a trailing window of N tokens, diff against what was
//! already emitted) is **not implemented** — gljax's only traced model
//! (Qwen2) and every vocabulary in `glcore`'s exact-match table that gljax
//! currently loads are byte-level BPE, so there is no live caller for it, and
//! its window size is explicitly an unmeasured "must be measured, not
//! guessed" open question in ARTX13 §4.2 itself.

use crate::tok::{TokenId, Tokenizer};

/// One per in-flight request. Owned by the request, not the tokenizer —
/// mirrors ARTX13 §4.2's design exactly.
pub struct IncrementalDecoder<'t> {
    tok: &'t dyn Tokenizer,
    /// Bytes produced but not yet emitted: an incomplete (or, for a
    /// byte-level vocabulary, occasionally genuinely invalid) UTF-8 tail.
    pending: Vec<u8>,
}

impl<'t> IncrementalDecoder<'t> {
    pub fn new(tok: &'t dyn Tokenizer) -> Self {
        IncrementalDecoder { tok, pending: Vec::new() }
    }

    /// Push one token; return the text delta that is safe to emit now.
    /// Returns `""` when the token only extends an incomplete character.
    pub fn push(&mut self, id: TokenId) -> String {
        self.pending.extend_from_slice(&self.tok.token_bytes(id));

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
                    // Genuinely invalid (byte-level vocabularies can emit
                    // sequences that are not valid UTF-8 at all) — skip the
                    // offending byte(s) or the stream stalls forever.
                    Some(bad) => self.pending.drain(..good + bad),
                };
                out
            }
        }
    }

    /// Flush at end of generation. A non-empty tail was a truncated
    /// character — emit `U+FFFD` once rather than silently dropping it.
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
    use crate::tok::GllmTokenizerAdapter;

    /// A tokenizer whose vocabulary can actually spell a multi-byte UTF-8
    /// character across two tokens — the scenario this whole module exists
    /// for. "é" is `0xC3 0xA9` in UTF-8. Under GPT-2 byte-level encoding, raw
    /// byte `b` in `33..=126 | 161..=172 | 174..=255` is spelled as the
    /// character `chr(b)` in the vocab string (`glcore/src/tokenizer/vocab.rs`'s
    /// `gpt2_byte_map`) — so byte `0xC3` (195, in `174..=255`) is vocab entry
    /// `"Ã"` and byte `0xA9` (169, in `161..=172`) is `"©"`. (An
    /// earlier version of this fixture used the *SPM* byte-fallback spelling
    /// `"<0xNN>"`, which `Style::ByteLevel` does not recognize — confirmed
    /// against `token_bytes_into`'s two different-per-style code paths, not
    /// guessed twice.)
    fn split_utf8_tokenizer() -> GllmTokenizerAdapter {
        let src = r#"{
          "model": {
            "type": "BPE",
            "vocab": {"Ã": 0, "©": 1, "x": 2},
            "merges": []
          },
          "pre_tokenizer": {"type": "ByteLevel"}
        }"#;
        GllmTokenizerAdapter::from_hf_json(src).expect("must load")
    }

    #[test]
    fn push_withholds_an_incomplete_multibyte_character() {
        let tok = split_utf8_tokenizer();
        let mut dec = IncrementalDecoder::new(&tok);
        let first = dec.push(0); // first byte of "é" only
        assert_eq!(first, "", "an incomplete character must not be emitted early");
    }

    #[test]
    fn push_emits_a_complete_multibyte_character_once_both_tokens_arrive() {
        let tok = split_utf8_tokenizer();
        let mut dec = IncrementalDecoder::new(&tok);
        let _ = dec.push(0);
        let second = dec.push(1);
        assert_eq!(second, "é", "the character must appear exactly once both bytes are in");
    }

    #[test]
    fn push_never_emits_the_unicode_replacement_character() {
        let tok = split_utf8_tokenizer();
        let mut dec = IncrementalDecoder::new(&tok);
        let mut all = String::new();
        for id in [0, 1, 2] {
            all.push_str(&dec.push(id));
        }
        all.push_str(&dec.finish());
        assert!(
            !all.contains('\u{FFFD}'),
            "a correctly split character must never surface U+FFFD: {all:?}"
        );
        assert_eq!(all, "éx");
    }

    #[test]
    fn finish_flushes_a_truncated_character_as_one_replacement_char() {
        let tok = split_utf8_tokenizer();
        let mut dec = IncrementalDecoder::new(&tok);
        let _ = dec.push(0); // first byte only, then generation ends
        let flushed = dec.finish();
        assert_eq!(flushed, "\u{FFFD}");
    }

    #[test]
    fn finish_is_empty_when_nothing_is_pending() {
        let tok = split_utf8_tokenizer();
        let mut dec = IncrementalDecoder::new(&tok);
        let _ = dec.push(0);
        let _ = dec.push(1);
        assert_eq!(dec.finish(), "", "a clean boundary has nothing to flush");
    }

    #[test]
    fn a_single_byte_token_emits_immediately() {
        let tok = split_utf8_tokenizer();
        let mut dec = IncrementalDecoder::new(&tok);
        assert_eq!(dec.push(2), "x");
    }
}
