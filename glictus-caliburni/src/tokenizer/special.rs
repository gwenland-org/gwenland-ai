//! Special token matcher (ARTX-OQ3 Wave 4).
//!
//! ChatML control tokens (`<|im_start|>`, `<|im_end|>`, `<|endoftext|>`,
//! ...) must never be split or BPE-merged like ordinary text — they're
//! matched as whole strings before pre-tokenization runs at all. This
//! module builds a small trie over every vocab entry the GGUF marks as
//! `token_type == 3` (control), so [`GllmTokenizer::encode`] can scan the
//! input once and find the next special-token occurrence in O(text len).

use std::collections::HashMap;

use crate::tokenizer::vocab::VocabTable;

/// GGUF `token_type` value marking a control/special token.
const TOKEN_TYPE_CONTROL: i32 = 3;

/// Fallback ids for Qwen2/Qwen3 ChatML models, used only when the GGUF
/// metadata omits the corresponding `tokenizer.ggml.*_token_id` field.
pub const FALLBACK_IM_START: u32 = 151_644;
pub const FALLBACK_IM_END: u32 = 151_645;
pub const FALLBACK_ENDOFTEXT: u32 = 151_643;

#[derive(Debug, Default)]
struct TrieNode {
    children: HashMap<char, TrieNode>,
    /// Set when this node terminates a special-token string.
    token_id: Option<u32>,
}

/// Scans text for whole special-token strings and carries the resolved
/// ChatML control ids for the loaded vocab.
#[derive(Debug)]
pub struct SpecialTokens {
    root: TrieNode,
    pub im_start: u32,
    pub im_end: u32,
    pub eos: u32,
    pub bos: u32,
    pub pad: u32,
}

impl SpecialTokens {
    /// Build from every control-type ([`TOKEN_TYPE_CONTROL`]) entry in
    /// `vocab`, plus the GGUF-declared eos/bos/pad ids. `im_start`/`im_end`
    /// are resolved by scanning the vocab for their literal ChatML strings
    /// (GGUF has no dedicated metadata field for them), falling back to the
    /// hardcoded Qwen2/Qwen3 ids when the vocab doesn't contain them.
    pub fn from_gguf(vocab: &VocabTable, eos_id: u32, bos_id: u32, pad_id: u32) -> Self {
        let mut root = TrieNode::default();
        for id in 0..vocab.vocab_size() {
            if vocab.token_type(id) != TOKEN_TYPE_CONTROL {
                continue;
            }
            let Some(bytes) = vocab.id_to_bytes(id) else {
                continue;
            };
            let Ok(text) = std::str::from_utf8(bytes) else {
                continue;
            };
            insert(&mut root, text, id);
        }

        let im_start = vocab
            .bytes_to_id(b"<|im_start|>")
            .unwrap_or(FALLBACK_IM_START);
        let im_end = vocab.bytes_to_id(b"<|im_end|>").unwrap_or(FALLBACK_IM_END);

        Self {
            root,
            im_start,
            im_end,
            eos: eos_id,
            bos: bos_id,
            pad: pad_id,
        }
    }

    /// Find the next special-token occurrence in `text` at or after byte
    /// offset `from`. Returns `(start, end, token_id)` in byte offsets, or
    /// `None` if no special token appears past `from`. Scans left to right;
    /// at each candidate start position, matches the *longest* special
    /// token beginning there (so e.g. a token that is itself a prefix of a
    /// longer one never shadows it).
    pub fn find_next(&self, text: &str, from: usize) -> Option<(usize, usize, u32)> {
        let bytes_from = text.get(from..)?;
        for (offset, _) in bytes_from.char_indices() {
            let start = from + offset;
            if let Some((end, id)) = self.longest_match_at(text, start) {
                return Some((start, end, id));
            }
        }
        None
    }

    fn longest_match_at(&self, text: &str, start: usize) -> Option<(usize, u32)> {
        let mut node = &self.root;
        let mut best: Option<(usize, u32)> = None;
        for (offset, c) in text[start..].char_indices() {
            match node.children.get(&c) {
                Some(next) => {
                    node = next;
                    if let Some(id) = node.token_id {
                        best = Some((start + offset + c.len_utf8(), id));
                    }
                }
                None => break,
            }
        }
        best
    }
}

fn insert(root: &mut TrieNode, s: &str, id: u32) {
    let mut node = root;
    for c in s.chars() {
        node = node.children.entry(c).or_default();
    }
    node.token_id = Some(id);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vocab_with_special_tokens() -> VocabTable {
        let tokens = vec![
            "hello".to_string(),
            "<|im_start|>".to_string(),
            "<|im_end|>".to_string(),
            "<|endoftext|>".to_string(),
        ];
        let types = vec![1, 3, 3, 3];
        VocabTable::from_gguf(&tokens, &types).unwrap()
    }

    #[test]
    fn special_tokens_finds_im_start() {
        let vocab = vocab_with_special_tokens();
        let special = SpecialTokens::from_gguf(&vocab, 3, 3, 3);

        let text = "before<|im_start|>after";
        let (start, end, id) = special.find_next(text, 0).unwrap();
        assert_eq!(&text[start..end], "<|im_start|>");
        assert_eq!(id, 1);
    }

    #[test]
    fn special_tokens_find_next_returns_none_for_plain_text() {
        let vocab = vocab_with_special_tokens();
        let special = SpecialTokens::from_gguf(&vocab, 3, 3, 3);

        assert_eq!(special.find_next("just plain text", 0), None);
    }

    #[test]
    fn special_tokens_find_next_respects_from_offset() {
        let vocab = vocab_with_special_tokens();
        let special = SpecialTokens::from_gguf(&vocab, 3, 3, 3);

        let text = "<|im_start|>mid<|im_end|>";
        let (start, _, id) = special.find_next(text, 5).unwrap();
        assert_eq!(id, 2); // <|im_end|>, not the earlier <|im_start|>
        assert!(start > 5);
    }

    #[test]
    fn special_tokens_falls_back_to_hardcoded_im_ids_when_absent_from_vocab() {
        let tokens = vec!["hello".to_string()];
        let types = vec![1];
        let vocab = VocabTable::from_gguf(&tokens, &types).unwrap();
        let special = SpecialTokens::from_gguf(&vocab, 151_643, 151_643, 151_643);

        assert_eq!(special.im_start, FALLBACK_IM_START);
        assert_eq!(special.im_end, FALLBACK_IM_END);
    }

    #[test]
    fn special_tokens_carries_eos_bos_pad() {
        let vocab = vocab_with_special_tokens();
        let special = SpecialTokens::from_gguf(&vocab, 10, 20, 30);
        assert_eq!(special.eos, 10);
        assert_eq!(special.bos, 20);
        assert_eq!(special.pad, 30);
    }
}
