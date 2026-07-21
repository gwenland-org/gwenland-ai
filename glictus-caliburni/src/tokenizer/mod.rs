//! Qwen2/Qwen3-compatible byte-level BPE tokenizer (ARTX-OQ3).
//!
//! Reads vocab/merges/special-token metadata directly from a source
//! `.gguf` file via [`gguf_meta`] — `.gllm` packages carry no tokenizer
//! data (see `converter.rs`'s `tokenizer.ggml.tokens` presence warning).
//! The tokenizer and [`crate::package::GllmPackage`] are independent
//! inputs a caller combines at the engine boundary:
//!
//! ```text
//! .gguf file  ->  GgufMeta (metadata-only)  ->  GllmTokenizer
//! .gllm file  ->  GllmPackage                ->  GllmEngine
//! ```

pub mod bpe;
pub mod byte_map;
pub mod gguf_meta;
pub mod pretok;
pub mod special;
pub mod vocab;

use std::path::Path;

use crate::error::GllmError;
use bpe::BpeMergeTable;
use gguf_meta::GgufMeta;
use pretok::Qwen2PreTokenizer;
use special::SpecialTokens;
use vocab::VocabTable;

/// GGUF `token_type` value marking a control/special token — same
/// constant [`special::SpecialTokens::from_gguf`] uses to decide which
/// vocab entries feed its trie.
const TOKEN_TYPE_CONTROL: i32 = 3;

/// Qwen2/Qwen3-compatible byte-level BPE tokenizer.
///
/// Immutable once loaded: encode/decode never mutate the vocab, merge
/// table, or special-token set.
#[derive(Debug)]
pub struct GllmTokenizer {
    vocab: VocabTable,
    merges: BpeMergeTable,
    pretok: Qwen2PreTokenizer,
    special: SpecialTokens,
    add_bos_default: bool,
}

/// Per-call encode behavior. `TokenizerConfig::default()` matches Qwen2's
/// own convention: no BOS (Qwen has no BOS concept — `bos == eos`), EOS
/// appended for generation prompts.
#[derive(Debug, Clone, Copy)]
pub struct TokenizerConfig {
    pub add_bos: bool,
    pub add_eos: bool,
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self {
            add_bos: false,
            add_eos: true,
        }
    }
}

impl GllmTokenizer {
    /// Load a tokenizer from the source `.gguf` file's metadata. Reads
    /// only `tokenizer.ggml.*` keys — tensor data is never touched, so
    /// this is cheap regardless of model size.
    ///
    /// Required fields: `tokenizer.ggml.tokens`, `tokenizer.ggml.merges`,
    /// `tokenizer.ggml.token_type`. `tokenizer.ggml.model` must be `"gpt2"`
    /// — SentencePiece-style vocabs (`"llama"`) are not supported (Qwen2/
    /// Qwen3 always declare `gpt2`; see module docs on scope).
    pub fn from_gguf(path: &Path) -> Result<Self, GllmError> {
        let meta = GgufMeta::read_tokenizer(path)?;

        let model = meta
            .get_str("tokenizer.ggml.model")
            .ok_or(GllmError::TokenizerMissingField {
                field: "tokenizer.ggml.model",
            })?;
        if model != "gpt2" {
            return Err(GllmError::GgufMalformedMetadata {
                detail: format!(
                    "unsupported tokenizer.ggml.model {model:?}: only \"gpt2\" \
                     (byte-level BPE, as used by Qwen2/Qwen3) is supported"
                ),
            });
        }

        let tokens = meta
            .get_arr_str("tokenizer.ggml.tokens")
            .ok_or(GllmError::TokenizerMissingField {
                field: "tokenizer.ggml.tokens",
            })?;
        let types = meta
            .get_arr_i32("tokenizer.ggml.token_type")
            .ok_or(GllmError::TokenizerMissingField {
                field: "tokenizer.ggml.token_type",
            })?;
        let merges_raw = meta
            .get_arr_str("tokenizer.ggml.merges")
            .ok_or(GllmError::TokenizerMissingField {
                field: "tokenizer.ggml.merges",
            })?;

        let vocab = VocabTable::from_gguf(tokens, types)?;
        let merges = BpeMergeTable::from_gguf(merges_raw)?;
        let pretok = Qwen2PreTokenizer::new();

        // Qwen2/Qwen3 ChatML fallback ids (see special.rs) cover the case
        // where the GGUF omits these fields entirely.
        let eos_id = meta
            .get_u32("tokenizer.ggml.eos_token_id")
            .unwrap_or(special::FALLBACK_ENDOFTEXT);
        let bos_id = meta
            .get_u32("tokenizer.ggml.bos_token_id")
            .unwrap_or(eos_id);
        let pad_id = meta
            .get_u32("tokenizer.ggml.padding_token_id")
            .unwrap_or(eos_id);
        let special = SpecialTokens::from_gguf(&vocab, eos_id, bos_id, pad_id);

        let add_bos_default = meta.get_bool("tokenizer.ggml.add_bos_token").unwrap_or(false);

        Ok(Self {
            vocab,
            merges,
            pretok,
            special,
            add_bos_default,
        })
    }

    /// Encode text to token ids.
    ///
    /// Pipeline: scan for special-token strings (`<|im_start|>` etc.),
    /// splitting the input at each one so BPE never sees or merges across
    /// them; pre-tokenize every non-special span with the Qwen2 regex;
    /// BPE-encode each resulting chunk; concatenate in order. BOS/EOS are
    /// added per `config`, not by pretending they're part of the text.
    pub fn encode(&self, text: &str, config: &TokenizerConfig) -> Vec<u32> {
        let mut ids = Vec::new();
        if config.add_bos {
            ids.push(self.special.bos);
        }

        let mut pos = 0;
        while pos < text.len() {
            match self.special.find_next(text, pos) {
                Some((start, end, token_id)) => {
                    if start > pos {
                        self.encode_plain_span(&text[pos..start], &mut ids);
                    }
                    ids.push(token_id);
                    pos = end;
                }
                None => {
                    self.encode_plain_span(&text[pos..], &mut ids);
                    break;
                }
            }
        }

        if config.add_eos {
            ids.push(self.special.eos);
        }
        ids
    }

    /// Pre-tokenize and BPE-encode a span known to contain no special
    /// tokens, appending results to `out`.
    fn encode_plain_span(&self, span: &str, out: &mut Vec<u32>) {
        for chunk in self.pretok.split(span) {
            out.extend(self.merges.encode_chunk(chunk.as_bytes(), &self.vocab));
        }
    }

    /// Decode token ids to a UTF-8 string. Invalid byte sequences (should
    /// not occur for ids this tokenizer itself produced, but a caller can
    /// hand back arbitrary model output) are replaced per
    /// [`String::from_utf8_lossy`] semantics.
    pub fn decode(&self, ids: &[u32]) -> String {
        let bytes: Vec<u8> = ids
            .iter()
            .flat_map(|&id| self.vocab.id_to_bytes(id).unwrap_or(&[]).to_vec())
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Decode token ids to a UTF-8 string, omitting control/special
    /// tokens (`token_type == 3`) — e.g. `<|im_end|>` won't appear in
    /// output meant for display.
    pub fn decode_skip_special(&self, ids: &[u32]) -> String {
        let bytes: Vec<u8> = ids
            .iter()
            .filter(|&&id| self.vocab.token_type(id) != TOKEN_TYPE_CONTROL)
            .flat_map(|&id| self.vocab.id_to_bytes(id).unwrap_or(&[]).to_vec())
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }

    pub fn vocab_size(&self) -> u32 {
        self.vocab.vocab_size()
    }

    pub fn special_tokens(&self) -> &SpecialTokens {
        &self.special
    }

    /// Whether this model's GGUF metadata declares BOS should be
    /// prepended by default (`tokenizer.ggml.add_bos_token`) — Qwen2/Qwen3
    /// declare `false` (or omit the field, same fallback).
    pub fn add_bos_default(&self) -> bool {
        self.add_bos_default
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Small synthetic Qwen2-shaped vocab: single bytes for 'H','e','l','o'
    /// plus GPT-2 leading-space-encoded 'Ġhello'/'Ġworld', a couple of BPE
    /// merges, and the three ChatML control tokens.
    fn build_test_tokenizer() -> GllmTokenizer {
        let tokens = vec![
            "H".to_string(),      // 0
            "e".to_string(),      // 1
            "l".to_string(),      // 2
            "o".to_string(),      // 3
            "He".to_string(),     // 4
            "Hel".to_string(),    // 5
            "Hell".to_string(),   // 6
            "Hello".to_string(),  // 7
            "\u{0120}Hello".to_string(), // 8 = " Hello"
            "<|im_start|>".to_string(),  // 9
            "<|im_end|>".to_string(),    // 10
            "<|endoftext|>".to_string(), // 11
        ];
        let types = vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3];
        let vocab = VocabTable::from_gguf(&tokens, &types).unwrap();

        let merges = vec![
            "H e".to_string(),
            "He l".to_string(),
            "Hel l".to_string(),
            "Hell o".to_string(),
            "\u{0120} Hello".to_string(),
        ];
        let merges = BpeMergeTable::from_gguf(&merges).unwrap();

        let pretok = Qwen2PreTokenizer::new();
        let special = SpecialTokens::from_gguf(&vocab, 11, 11, 11);

        GllmTokenizer {
            vocab,
            merges,
            pretok,
            special,
            add_bos_default: false,
        }
    }

    #[test]
    fn tokenizer_encodes_simple_ascii() {
        let tok = build_test_tokenizer();
        let ids = tok.encode(
            "Hello",
            &TokenizerConfig {
                add_bos: false,
                add_eos: false,
            },
        );
        assert_eq!(ids, vec![7]);
    }

    #[test]
    fn tokenizer_encodes_special_tokens() {
        let tok = build_test_tokenizer();
        let ids = tok.encode(
            "<|im_start|>",
            &TokenizerConfig {
                add_bos: false,
                add_eos: false,
            },
        );
        assert_eq!(ids, vec![9]);
    }

    #[test]
    fn tokenizer_encodes_special_token_embedded_in_text() {
        let tok = build_test_tokenizer();
        let ids = tok.encode(
            "Hello<|im_end|>",
            &TokenizerConfig {
                add_bos: false,
                add_eos: false,
            },
        );
        assert_eq!(ids, vec![7, 10]);
    }

    #[test]
    fn tokenizer_decode_round_trips_ascii() {
        let tok = build_test_tokenizer();
        let cfg = TokenizerConfig {
            add_bos: false,
            add_eos: false,
        };
        let ids = tok.encode("Hello", &cfg);
        assert_eq!(tok.decode(&ids), "Hello");
    }

    #[test]
    fn tokenizer_decode_skip_special_omits_control_tokens() {
        let tok = build_test_tokenizer();
        let ids = vec![9, 7, 10]; // <|im_start|> Hello <|im_end|>
        assert_eq!(tok.decode(&ids), "<|im_start|>Hello<|im_end|>");
        assert_eq!(tok.decode_skip_special(&ids), "Hello");
    }

    #[test]
    fn tokenizer_chatml_template_positions() {
        let tok = build_test_tokenizer();
        let ids = tok.encode(
            "<|im_start|>Hello<|im_end|>",
            &TokenizerConfig {
                add_bos: false,
                add_eos: false,
            },
        );
        assert_eq!(ids, vec![9, 7, 10]);
    }

    #[test]
    fn tokenizer_handles_unicode_text_without_panicking() {
        let tok = build_test_tokenizer();
        // No merges/vocab cover CJK bytes in this synthetic vocab, so every
        // byte falls through byte_fallback and is silently dropped (no
        // single-byte entries beyond 'H','e','l','o' exist) — the
        // assertion here is just that this does not panic.
        let _ = tok.encode(
            "你好",
            &TokenizerConfig {
                add_bos: false,
                add_eos: false,
            },
        );
    }

    #[test]
    fn tokenizer_add_bos_false_for_qwen() {
        let tok = build_test_tokenizer();
        assert!(!tok.add_bos_default());
    }

    #[test]
    fn tokenizer_add_bos_prepends_when_requested() {
        let tok = build_test_tokenizer();
        let ids = tok.encode(
            "Hello",
            &TokenizerConfig {
                add_bos: true,
                add_eos: false,
            },
        );
        assert_eq!(ids, vec![tok.special_tokens().bos, 7]);
    }

    #[test]
    fn tokenizer_add_eos_appends_by_default() {
        let tok = build_test_tokenizer();
        let ids = tok.encode("Hello", &TokenizerConfig::default());
        assert_eq!(ids, vec![7, tok.special_tokens().eos]);
    }

    #[test]
    fn tokenizer_vocab_size_matches_loaded_vocab() {
        let tok = build_test_tokenizer();
        assert_eq!(tok.vocab_size(), 12);
    }

    #[test]
    fn tokenizer_from_gguf_rejects_missing_file() {
        let err = GllmTokenizer::from_gguf(Path::new("/nonexistent/model.gguf")).unwrap_err();
        assert!(matches!(err, GllmError::Io(_)), "got {err:?}");
    }

    /// Real-model integration: opt-in via GWENLAND_TEST_GGUF. Skips loudly
    /// when absent. TOL: zero — token ids must be identical.
    #[test]
    fn tokenizer_matches_reference_qwen25_encode_values() {
        let Ok(path) = std::env::var("GWENLAND_TEST_GGUF") else {
            eprintln!("SKIP: GWENLAND_TEST_GGUF not set (no real GGUF available)");
            return;
        };
        let tok = GllmTokenizer::from_gguf(Path::new(&path)).unwrap();
        let cfg = TokenizerConfig {
            add_bos: false,
            add_eos: false,
        };

        let cases: &[(&str, &[u32])] = &[
            ("Hello", &[9707]),
            (" Hello", &[21927]),
            ("<|im_start|>", &[151644]),
            ("<|im_end|>", &[151645]),
            ("<|endoftext|>", &[151643]),
        ];
        for (text, expected) in cases {
            assert_eq!(tok.encode(text, &cfg), *expected, "mismatch for {text:?}");
        }

        assert!(!tok.add_bos_default(), "Qwen2/Qwen3 must not default-add BOS");

        let sentence = "The quick brown fox jumps over the lazy dog. 你好，世界！";
        let ids = tok.encode(sentence, &cfg);
        assert_eq!(tok.decode(&ids), sentence);
    }
}
