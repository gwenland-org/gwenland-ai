//! Vocab table: token id <-> decoded byte sequence (ARTX-OQ3 Wave 2).
//!
//! GGUF vocab strings are encoded through the GPT-2 byte-level map (see
//! [`crate::tokenizer::byte_map`]) — this table undoes that encoding once
//! at load time, so every other stage of the tokenizer works in raw bytes
//! instead of re-decoding chars on every lookup.

use std::collections::HashMap;

use crate::error::GllmError;
use crate::tokenizer::byte_map::char_to_byte;

/// Token id -> raw bytes (and back), plus the GGUF `token_type` tag per id.
///
/// `token_type` values follow the GGUF convention: `1` = normal, `2` =
/// unknown, `3` = control/special, `4` = user-defined byte fallback, `5` =
/// unused, `6` = byte. This table stores whatever the source GGUF says
/// without interpreting it further — interpretation is
/// [`crate::tokenizer::special`]'s job.
#[derive(Debug)]
pub struct VocabTable {
    id_to_bytes: Vec<Vec<u8>>,
    bytes_to_id: HashMap<Vec<u8>, u32>,
    token_type: Vec<i32>,
}

impl VocabTable {
    /// Build from the raw GGUF `tokenizer.ggml.tokens` / `tokenizer.ggml.token_type`
    /// arrays. Each token string is decoded from GPT-2 byte-level chars back
    /// to raw bytes via [`char_to_byte`]; a char with no byte mapping (which
    /// should not occur for a well-formed GGUF vocab) falls back to its raw
    /// UTF-8 encoding rather than failing the whole load.
    pub fn from_gguf(tokens: &[String], types: &[i32]) -> Result<Self, GllmError> {
        if tokens.len() != types.len() {
            return Err(GllmError::TokenizerVocabMismatch {
                tokens: tokens.len(),
                types: types.len(),
            });
        }

        let mut id_to_bytes = Vec::with_capacity(tokens.len());
        let mut bytes_to_id = HashMap::with_capacity(tokens.len());
        for (id, tok) in tokens.iter().enumerate() {
            let bytes: Vec<u8> = tok
                .chars()
                .flat_map(|c| match char_to_byte(c) {
                    Some(b) => vec![b],
                    None => c.to_string().into_bytes(),
                })
                .collect();
            bytes_to_id.entry(bytes.clone()).or_insert(id as u32);
            id_to_bytes.push(bytes);
        }

        Ok(Self {
            id_to_bytes,
            bytes_to_id,
            token_type: types.to_vec(),
        })
    }

    pub fn id_to_bytes(&self, id: u32) -> Option<&[u8]> {
        self.id_to_bytes.get(id as usize).map(Vec::as_slice)
    }

    pub fn bytes_to_id(&self, bytes: &[u8]) -> Option<u32> {
        self.bytes_to_id.get(bytes).copied()
    }

    pub fn vocab_size(&self) -> u32 {
        self.id_to_bytes.len() as u32
    }

    pub fn token_type(&self, id: u32) -> i32 {
        self.token_type.get(id as usize).copied().unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vocab_builds_from_synthetic_tokens() {
        let tokens = vec![
            "H".to_string(),
            "e".to_string(),
            "l".to_string(),
            "o".to_string(),
            "Hello".to_string(),
        ];
        let types = vec![1, 1, 1, 1, 1];
        let vocab = VocabTable::from_gguf(&tokens, &types).unwrap();

        assert_eq!(vocab.vocab_size(), 5);
        assert_eq!(vocab.id_to_bytes(0), Some(b"H".as_slice()));
        assert_eq!(vocab.id_to_bytes(4), Some(b"Hello".as_slice()));
        assert_eq!(vocab.bytes_to_id(b"Hello"), Some(4));
        assert_eq!(vocab.bytes_to_id(b"nope"), None);
    }

    #[test]
    fn vocab_rejects_mismatched_lengths() {
        let tokens = vec!["a".to_string(), "b".to_string()];
        let types = vec![1];
        let err = VocabTable::from_gguf(&tokens, &types).unwrap_err();
        assert!(
            matches!(
                err,
                GllmError::TokenizerVocabMismatch {
                    tokens: 2,
                    types: 1
                }
            ),
            "got {err:?}"
        );
    }

    #[test]
    fn vocab_id_to_bytes_round_trips() {
        // 'Ġ' (U+0120) is GPT-2's byte-level encoding of a leading space
        // (raw byte 0x20) — a realistic case from a real GGUF vocab.
        let tokens = vec!["\u{0120}Hello".to_string()];
        let types = vec![1];
        let vocab = VocabTable::from_gguf(&tokens, &types).unwrap();

        assert_eq!(vocab.id_to_bytes(0), Some(b" Hello".as_slice()));
        assert_eq!(vocab.bytes_to_id(b" Hello"), Some(0));
    }

    #[test]
    fn vocab_token_type_defaults_to_zero_out_of_range() {
        let tokens = vec!["a".to_string()];
        let types = vec![3];
        let vocab = VocabTable::from_gguf(&tokens, &types).unwrap();

        assert_eq!(vocab.token_type(0), 3);
        assert_eq!(vocab.token_type(99), 0);
    }

    #[test]
    fn vocab_duplicate_byte_sequences_keep_first_id() {
        // Not expected in a real vocab, but the table must not panic —
        // first-seen id wins deterministically.
        let tokens = vec!["dup".to_string(), "dup".to_string()];
        let types = vec![1, 1];
        let vocab = VocabTable::from_gguf(&tokens, &types).unwrap();

        assert_eq!(vocab.bytes_to_id(b"dup"), Some(0));
    }
}
