//! The two SentencePiece-*surface* encoders.
//!
//! Both write spaces as `▁` and fall back to `<0xNN>` byte tokens, and both
//! are therefore decoded by the same path in [`super`]. What separates them is
//! the only thing that decides ids:
//!
//! * [`Style::Spm`] merges any pair whose concatenation is in the vocabulary,
//!   ranked by that token's **score**;
//! * [`Style::SpmBpe`] (Gemma-4, Sarvam) merges only pairs present in the
//!   **merge list**, ranked by position — and does no word-level splitting at
//!   all, so a merge may span what other families call several words.
//!
//! ⚠️ A vocabulary can carry both scores and merges; Gemma-4 ships 262 144 of
//! the former and 514 906 of the latter. Running the wrong encoder over it
//! produces different ids and no error at all, which is why the two live
//! side by side here rather than sharing a parameterised path.

use super::*;

/// Ranks candidate merges for an SPM vocabulary: higher score wins.
///
/// A pair is mergeable only if the concatenation is itself in the vocabulary,
/// which is what makes SPM vocabulary-driven rather than merge-list-driven.
struct SpmRanker<'a>(&'a Vocab);
impl bpe::Ranker for SpmRanker<'_> {
    #[inline]
    fn rank(&self, piece: &str, _left_len: usize) -> Option<i64> {
        // SPM is vocabulary-driven: any split of a vocabulary entry is valid,
        // so the split point carries no information here.
        let id = *self.0.token_to_id.get(piece)?;
        let score = self.0.scores.get(id as usize).copied().unwrap_or(0.0);
        // f32 → ordered i64. Scores are small and finite in practice; scaling
        // keeps the heap integral and avoids a float Ord wrapper.
        Some((score * 1e6) as i64)
    }
}

impl GllmTokenizer {

    /// Gemma-4's shape: SentencePiece surface form, merge-list ranking.
    ///
    /// Spaces become `▁` across the *whole* input first, then the text is cut
    /// at newline runs only — there is no word-level splitting, so a merge may
    /// legitimately span what other families would call several words.
    pub(super) fn encode_spm_bpe(&self, text: &str, ids: &mut Vec<u32>) -> Result<(), TokError> {
        SCRATCH.with(|sc| {
            let mut sc = sc.borrow_mut();
            let Scratch {
                merger, prepared, ..
            } = &mut *sc;

            prepared.clear();
            for ch in text.chars() {
                prepared.push(if ch == ' ' { SPM_SPACE } else { ch });
            }

            let v = &self.v;
            let ranker = ByteLevelRanker(v);
            let mut err = None;

            let mut chunks: Vec<&str> = Vec::new();
            PreTok::Lines.split(prepared, |c| chunks.push(c));

            for chunk in chunks {
                // ⚠️ A run of newlines is looked up whole before merging.
                // Gemma-4's vocabulary carries multi-newline tokens that
                // rank-order merging cannot always reach, the same class of
                // problem `ignore_merges` solves for Llama-3 — but scoped to
                // newline runs rather than to every pre-token.
                if chunk.as_bytes()[0] == b'\n' {
                    if let Some(&id) = v.token_to_id.get(chunk) {
                        ids.push(id);
                        continue;
                    }
                }
                merger.run(chunk, &ranker, |piece| {
                    if err.is_some() {
                        return;
                    }
                    if let Some(&id) = v.token_to_id.get(piece) {
                        ids.push(id);
                        return;
                    }
                    // Byte fallback is `<0xNN>`, not the GPT-2 remap: this
                    // style never encoded its bytes as printable chars.
                    for b in piece.bytes() {
                        let mut buf = [0u8; 6];
                        let key = fmt_byte_token(b, &mut buf);
                        if let Some(&id) = v.token_to_id.get(key) {
                            ids.push(id);
                        } else if let Some(unk) = v.unk_id {
                            ids.push(unk);
                        } else {
                            err = Some(TokError::Unencodable {
                                ch: piece.to_string(),
                            });
                            return;
                        }
                    }
                });
                if err.is_some() {
                    break;
                }
            }
            match err {
                Some(e) => Err(e),
                None => Ok(()),
            }
        })
    }

    /// SPM: map *every* whitespace run's spaces to `▁`, optionally prepend the
    /// dummy prefix, then merge by score.
    pub(super) fn encode_spm(&self, text: &str, ids: &mut Vec<u32>) -> Result<(), TokError> {
        SCRATCH.with(|sc| {
        let mut sc = sc.borrow_mut();
        let Scratch {
            merger, prepared, ..
        } = &mut *sc;

        prepared.clear();
        if self.v.add_dummy_prefix {
            prepared.push(SPM_SPACE);
        }
        // ⚠️ Bug 3: only U+0020 was previously replaced. SentencePiece maps the
        // space character; other whitespace stays literal and reaches the byte
        // fallback, which is what a reference implementation does.
        for ch in text.chars() {
            if ch == ' ' {
                prepared.push(SPM_SPACE);
            } else {
                prepared.push(ch);
            }
        }

        let ranker = SpmRanker(&self.v);
        let mut err = None;
        let v = &self.v;
        merger.run(prepared, &ranker, |piece| {
            if err.is_some() {
                return;
            }
            if let Some(&id) = v.token_to_id.get(piece) {
                ids.push(id);
                return;
            }
            // Byte fallback: <0xNN> per byte.
            for b in piece.bytes() {
                let mut buf = [0u8; 6];
                let key = fmt_byte_token(b, &mut buf);
                if let Some(&id) = v.token_to_id.get(key) {
                    ids.push(id);
                } else if let Some(unk) = v.unk_id {
                    ids.push(unk);
                } else {
                    // ⚠️ Bug 5: this used to drop the symbol silently.
                    err = Some(TokError::Unencodable {
                        ch: piece.to_string(),
                    });
                    return;
                }
            }
        });
        match err {
            Some(e) => Err(e),
            None => Ok(()),
        }
        })
    }
}

/// Write `<0xNN>` into `buf` without allocating.
#[inline]
pub(super) fn fmt_byte_token(b: u8, buf: &mut [u8; 6]) -> &str {
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    buf[0] = b'<';
    buf[1] = b'0';
    buf[2] = b'x';
    buf[3] = HEX[(b >> 4) as usize];
    buf[4] = HEX[(b & 0xf) as usize];
    buf[5] = b'>';
    std::str::from_utf8(buf).expect("ascii")
}

/// Parse `<0xNN>` back to its byte.
pub(super) fn parse_byte_token(tok: &str) -> Option<u8> {
    let h = tok.strip_prefix("<0x")?.strip_suffix('>')?;
    (h.len() == 2)
        .then(|| u8::from_str_radix(h, 16).ok())
        .flatten()
}
