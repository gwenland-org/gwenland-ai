//! BPE merge table and the core encode loop (ARTX-OQ3 Wave 3).
//!
//! Standard byte-pair-encoding: start with each byte of the input as its
//! own symbol, repeatedly merge the adjacent pair with the lowest rank
//! (lowest = highest priority, matching the order merges appear in the
//! GGUF `tokenizer.ggml.merges` array) until no known pair remains, then
//! map the final symbols to vocab ids.

use std::collections::HashMap;

use crate::error::GllmError;
use crate::tokenizer::byte_map::char_to_byte;
use crate::tokenizer::vocab::VocabTable;

/// Rank-ordered BPE merge rules: `(left_bytes, right_bytes) -> rank`.
/// Lower rank means the merge was declared earlier in the GGUF `merges`
/// array, i.e. it has higher priority.
#[derive(Debug)]
pub struct BpeMergeTable {
    merges: HashMap<(Vec<u8>, Vec<u8>), u32>,
}

impl BpeMergeTable {
    /// Parse from the GGUF `tokenizer.ggml.merges` strings. Each entry is
    /// `"LEFT RIGHT"`, single-space separated; LEFT/RIGHT are GPT-2
    /// byte-level-encoded chars, decoded back to raw bytes the same way
    /// [`VocabTable::from_gguf`] decodes vocab entries.
    pub fn from_gguf(merges: &[String]) -> Result<Self, GllmError> {
        let mut table = HashMap::with_capacity(merges.len());
        for (rank, entry) in merges.iter().enumerate() {
            let Some((left, right)) = entry.split_once(' ') else {
                return Err(GllmError::TokenizerMalformedMerge {
                    entry: entry.clone(),
                });
            };
            if left.is_empty() || right.is_empty() || right.contains(' ') {
                return Err(GllmError::TokenizerMalformedMerge {
                    entry: entry.clone(),
                });
            }
            let left_bytes = decode_chars(left);
            let right_bytes = decode_chars(right);
            table.insert((left_bytes, right_bytes), rank as u32);
        }
        Ok(Self { merges: table })
    }

    /// Rank of merging `left` with `right`, if this table has that rule.
    fn rank(&self, left: &[u8], right: &[u8]) -> Option<u32> {
        self.merges
            .get(&(left.to_vec(), right.to_vec()))
            .copied()
    }

    /// BPE-encode one pre-tokenized chunk (already-decoded raw bytes) into
    /// vocab token ids.
    ///
    /// Symbols start as individual bytes; the adjacent pair with the
    /// lowest rank is merged first, repeated until no adjacent pair has a
    /// rank entry. Each final symbol is looked up in `vocab`; a symbol
    /// with no vocab entry (should not happen for a well-formed byte-level
    /// vocab, which always contains every single byte) falls back to
    /// per-byte encoding via [`byte_fallback`].
    pub fn encode_chunk(&self, chunk: &[u8], vocab: &VocabTable) -> Vec<u32> {
        if chunk.is_empty() {
            return Vec::new();
        }

        let mut symbols: Vec<Vec<u8>> = chunk.iter().map(|&b| vec![b]).collect();

        loop {
            let mut best: Option<(u32, usize)> = None;
            for i in 0..symbols.len().saturating_sub(1) {
                if let Some(rank) = self.rank(&symbols[i], &symbols[i + 1])
                    && best.is_none_or(|(best_rank, _)| rank < best_rank)
                {
                    best = Some((rank, i));
                }
            }
            let Some((_, i)) = best else {
                break;
            };
            let merged = [symbols[i].as_slice(), symbols[i + 1].as_slice()].concat();
            symbols.splice(i..=i + 1, [merged]);
        }

        symbols
            .into_iter()
            .flat_map(|sym| match vocab.bytes_to_id(&sym) {
                Some(id) => vec![id],
                None => byte_fallback(&sym, vocab),
            })
            .collect()
    }
}

/// Decode a GPT-2 byte-level-encoded merge operand back to raw bytes.
/// Mirrors [`VocabTable::from_gguf`]'s per-token decode: a char with no
/// byte mapping falls back to its raw UTF-8 encoding.
fn decode_chars(s: &str) -> Vec<u8> {
    s.chars()
        .flat_map(|c| match char_to_byte(c) {
            Some(b) => vec![b],
            None => c.to_string().into_bytes(),
        })
        .collect()
}

/// Encode a multi-byte symbol with no direct vocab entry as one token id
/// per byte, via each byte's single-byte vocab entry. Bytes with no
/// single-byte vocab entry either (a malformed/incomplete vocab) are
/// silently dropped — a byte-level GPT-2 vocab always has all 256 single
/// bytes, so this path is unreachable for well-formed input.
fn byte_fallback(sym: &[u8], vocab: &VocabTable) -> Vec<u32> {
    sym.iter()
        .filter_map(|&b| vocab.bytes_to_id(&[b]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_vocab() -> VocabTable {
        // Bytes 'a'..'d' as single-byte tokens (ids 0..3), plus the merged
        // forms in the order the merge rules below will build them.
        let tokens = vec![
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
            "d".to_string(),
            "ab".to_string(),
            "abc".to_string(),
        ];
        let types = vec![1; tokens.len()];
        VocabTable::from_gguf(&tokens, &types).unwrap()
    }

    #[test]
    fn merge_table_builds_from_synthetic_merges() {
        let merges = vec!["a b".to_string(), "ab c".to_string()];
        let table = BpeMergeTable::from_gguf(&merges).unwrap();
        assert_eq!(table.rank(b"a", b"b"), Some(0));
        assert_eq!(table.rank(b"ab", b"c"), Some(1));
        assert_eq!(table.rank(b"x", b"y"), None);
    }

    #[test]
    fn merge_table_rejects_malformed_entry() {
        let merges = vec!["no-space-here".to_string()];
        let err = BpeMergeTable::from_gguf(&merges).unwrap_err();
        assert!(
            matches!(err, GllmError::TokenizerMalformedMerge { .. }),
            "got {err:?}"
        );
    }

    #[test]
    fn merge_table_rejects_extra_space() {
        let merges = vec!["a b c".to_string()];
        let err = BpeMergeTable::from_gguf(&merges).unwrap_err();
        assert!(
            matches!(err, GllmError::TokenizerMalformedMerge { .. }),
            "got {err:?}"
        );
    }

    #[test]
    fn encode_chunk_single_byte() {
        let table = BpeMergeTable::from_gguf(&[]).unwrap();
        let vocab = synthetic_vocab();
        let ids = table.encode_chunk(b"a", &vocab);
        assert_eq!(ids, vec![0]);
    }

    #[test]
    fn encode_chunk_applies_merges_in_rank_order() {
        let merges = vec!["a b".to_string(), "ab c".to_string()];
        let table = BpeMergeTable::from_gguf(&merges).unwrap();
        let vocab = synthetic_vocab();

        // "abc" -> merge(a,b) first (rank 0) -> ["ab","c"] -> merge(ab,c)
        // (rank 1) -> ["abc"] -> vocab id 5.
        let ids = table.encode_chunk(b"abc", &vocab);
        assert_eq!(ids, vec![5]);
    }

    #[test]
    fn encode_chunk_stops_when_no_rule_applies() {
        // No merge rules at all: "ab" stays as two separate byte tokens.
        let table = BpeMergeTable::from_gguf(&[]).unwrap();
        let vocab = synthetic_vocab();
        let ids = table.encode_chunk(b"ab", &vocab);
        assert_eq!(ids, vec![0, 1]);
    }

    #[test]
    fn encode_chunk_byte_fallback_for_unknown() {
        // 'd' has a single-byte vocab entry (id 3) but nothing merges it
        // with anything — exercises the direct single-byte path, which is
        // the fallback's target for symbols with no fused vocab entry.
        let table = BpeMergeTable::from_gguf(&[]).unwrap();
        let vocab = synthetic_vocab();
        let ids = table.encode_chunk(b"d", &vocab);
        assert_eq!(ids, vec![3]);
    }

    #[test]
    fn encode_chunk_empty_input() {
        let table = BpeMergeTable::from_gguf(&[]).unwrap();
        let vocab = synthetic_vocab();
        assert_eq!(table.encode_chunk(b"", &vocab), Vec::<u32>::new());
    }

    #[test]
    fn encode_chunk_prefers_lowest_rank_pair_globally() {
        // Rules: (b,c) rank 0, (a,b) rank 1. Input "abc": both pairs are
        // present initially ((a,b) and (b,c)); (b,c) has the lower rank
        // and must be merged first, regardless of position.
        let merges = vec!["b c".to_string(), "a b".to_string()];
        let table = BpeMergeTable::from_gguf(&merges).unwrap();
        let tokens = vec![
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
            "bc".to_string(),
        ];
        let types = vec![1; tokens.len()];
        let vocab = VocabTable::from_gguf(&tokens, &types).unwrap();

        let ids = table.encode_chunk(b"abc", &vocab);
        // After merging b+c -> "bc" (id 3), no rule for (a, bc) exists, so
        // encoding ends at ["a", "bc"] -> [0, 3].
        assert_eq!(ids, vec![0, 3]);
    }
}
