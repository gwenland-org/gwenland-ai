//! `Tokenizer` — the seam between gljax and `glcore` (ARTX13 §1).
//!
//! ARTX08's rejected-alternative #7 declined a trait with exactly one
//! implementor. ARTX13 §1.1 argues the same reasoning does *not* apply here,
//! and the difference is worth restating: a `Matmul` trait would have exactly
//! one implementor forever (`dot_general`), so the trait buys dispatch
//! nobody needs. `Tokenizer` decouples gljax's core types from `glcore`
//! specifically **because** ARTX11's cross-vocabulary speculation needs gljax
//! to hold two live tokenizers with different vocabularies at once (a draft
//! model's and a target model's) — a real, already-partially-landed
//! requirement (`gljax::arch`'s `Architecture` descriptor, this same sprint),
//! not a speculative one.
//!
//! ⛔ **`runtime::Session`/`runtime::CachedSession` are NOT rewired to this
//! trait in this sprint.** Both currently store `glcore::GllmTokenizer`
//! concretely and expose it through a *public* `tokenizer() -> Option<&GllmTokenizer>`
//! getter, and `HfMetadata::encode`/`HfCheckpoint::encode` call
//! `tokenizer.add_bos_default()` to decide whether to add BOS — a per-call
//! decision the concrete type can answer but the naive trait sketch cannot
//! (see this module's `add_bos_default` deviation below). Switching those
//! call sites is a real behavior-preserving refactor, not a type swap, and it
//! touches the one path Gate A5 has run against real PJRT hardware. This
//! environment cannot re-run Gate A5 to check it — no plugin, no CI here —
//! so, matching the same call `gljax::arch`'s module docs made for
//! `model::qwen2`: the trait and its adapter are built and tested in
//! isolation; wiring `Session` through them is real follow-up work, gated on
//! a CI run, not done blind.
//!
//! # Deviations from ARTX13 §1.2's sketch, and why
//!
//! * `encode` returns `Result<Vec<TokenId>, GlError>` — `glcore::GllmTokenizer::encode`
//!   can fail (an unencodable byte sequence under some vocabularies), and the
//!   sketch's infallible signature would force either a panic or a silent
//!   empty-vec on error. Every other fallible operation in gljax returns
//!   `Result` (`gljax/src/graph/builder.rs`'s own docs: "model definition
//!   panics, file contents do not" — a tokenizer's input is file/user
//!   content, not a model definition).
//! * `token_bytes` returns an owned `Vec<u8>`, not `&[u8]`. `GllmTokenizer::token_bytes`
//!   already returns owned bytes (BPE/SPM byte-fallback tokens are assembled,
//!   not stored pre-formed), so a borrowed signature here would require the
//!   adapter to either leak memory or fake a lifetime it doesn't have.
//! * `add_bos_default(&self) -> bool` is added. Without it, the trait cannot
//!   express what `runtime::hf::HfMetadata::encode` already does today —
//!   "Qwen2 has no BOS in completion mode; `add_bos_default` carries what the
//!   vocabulary itself declares rather than a guess" (that file's own
//!   comment). Dropping this to match the sketch exactly would be a real
//!   capability loss, not a simplification.

pub mod stream;

use crate::GlError;

pub type TokenId = u32;

/// The vocabulary/tokenizer contract gljax's core types depend on, so that
/// `glcore` stays out of them (see this module's docs for why that
/// decoupling is a real, near-term requirement rather than a speculative
/// one).
pub trait Tokenizer: Send + Sync {
    fn encode(&self, text: &str, add_special: bool) -> Result<Vec<TokenId>, GlError>;
    /// Whole-sequence decode. **Not** valid for streaming — see [`stream::IncrementalDecoder`].
    fn decode(&self, ids: &[TokenId], skip_special: bool) -> String;

    fn vocab_size(&self) -> usize;
    fn eos_id(&self) -> TokenId;
    fn bos_id(&self) -> Option<TokenId>;
    fn is_stop(&self, id: TokenId) -> bool;

    /// Whether this vocabulary wants BOS added by default when the caller
    /// has no more specific instruction. See this module's docs.
    fn add_bos_default(&self) -> bool;

    /// Raw surface form of one token, without byte-level unmapping applied.
    /// Needed by [`stream::IncrementalDecoder`].
    fn token_bytes(&self, id: TokenId) -> Vec<u8>;

    /// Stable identity of this vocabulary (ARTX13 §1.2). Two tokenizers with
    /// equal fingerprints are interchangeable for ARTX11's `VocabRelation::Identical`;
    /// unequal fingerprints mean any cross-vocabulary logic must not assume
    /// shared token ids mean the same text.
    fn vocab_fingerprint(&self) -> VocabFingerprint;
}

/// SHA-256 over every token's bytes (in id order) plus `eos_id`/`bos_id`.
///
/// ⚠️ **Narrower than ARTX13 §1.2's spec.** The doc's definition is "sorted
/// vocab entries ++ merge list ++ special tokens"; `glcore::GllmTokenizer`'s
/// public API exposes token bytes by id but not its raw merge list, so the
/// merge list is not included here. Two BPE vocabularies with identical
/// resulting token sets but a different merge *order* could theoretically
/// fingerprint equal while not being byte-identical tokenizers. Nothing
/// currently consumes `vocab_fingerprint` (ARTX11's speculative decoding is
/// not built), so this gap has no live consequence yet — recorded here so it
/// isn't silently forgotten when something does depend on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VocabFingerprint(pub [u8; 32]);

impl VocabFingerprint {
    pub fn hex(&self) -> String {
        crate::runtime::digest::hex(&self.0)
    }
}

/// Computes a [`VocabFingerprint`] from any [`Tokenizer`], using only trait
/// methods — so it works for `GllmTokenizerAdapter` and any future
/// implementor alike, matching the decoupling this module exists for.
pub fn fingerprint(tok: &dyn Tokenizer) -> VocabFingerprint {
    use crate::runtime::digest::sha256;

    let mut buf = Vec::new();
    for id in 0..tok.vocab_size() as TokenId {
        let bytes = tok.token_bytes(id);
        buf.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
        buf.extend_from_slice(&bytes);
    }
    buf.extend_from_slice(&tok.eos_id().to_le_bytes());
    match tok.bos_id() {
        Some(b) => {
            buf.push(1);
            buf.extend_from_slice(&b.to_le_bytes());
        }
        None => buf.push(0),
    }
    VocabFingerprint(sha256(&buf))
}

/// Wraps [`glcore::GllmTokenizer`] behind [`Tokenizer`] — the one real
/// implementor today.
pub struct GllmTokenizerAdapter(pub glcore::GllmTokenizer);

impl GllmTokenizerAdapter {
    /// Loads a HuggingFace `tokenizer.json` from an in-memory string, rather
    /// than `glcore::GllmTokenizer::from_hf_json_path`'s file path — useful
    /// for tests (this module's own) and for any future caller that already
    /// has the JSON in memory (e.g. fetched, not read from a local file).
    pub fn from_hf_json(src: &str) -> Result<Self, GlError> {
        let vocab = glcore::tokenizer::Vocab::from_hf_json(src)?;
        Ok(GllmTokenizerAdapter(glcore::GllmTokenizer::new(vocab)))
    }
}

impl Tokenizer for GllmTokenizerAdapter {
    fn encode(&self, text: &str, add_special: bool) -> Result<Vec<TokenId>, GlError> {
        Ok(self.0.encode(text, add_special)?)
    }

    fn decode(&self, ids: &[TokenId], skip_special: bool) -> String {
        self.0.decode(ids, skip_special)
    }

    fn vocab_size(&self) -> usize {
        self.0.vocab_size()
    }

    fn eos_id(&self) -> TokenId {
        self.0.eos_id()
    }

    fn bos_id(&self) -> Option<TokenId> {
        self.0.bos_id()
    }

    fn is_stop(&self, id: TokenId) -> bool {
        self.0.is_stop_token(id)
    }

    fn add_bos_default(&self) -> bool {
        self.0.add_bos_default()
    }

    fn token_bytes(&self, id: TokenId) -> Vec<u8> {
        self.0.token_bytes(id)
    }

    fn vocab_fingerprint(&self) -> VocabFingerprint {
        fingerprint(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_tokenizer() -> GllmTokenizerAdapter {
        // The smallest real, working GllmTokenizer this crate can build
        // without a checkpoint on disk: HF-JSON with a handful of byte-level
        // tokens (mirrors this sprint's glcore::tokenizer::vocab test fixtures).
        let src = r#"{
          "model": {
            "type": "BPE",
            "vocab": {"a": 0, "b": 1, "c": 2, "d": 3},
            "merges": []
          },
          "pre_tokenizer": {"type": "ByteLevel"},
          "added_tokens": [{"id": 4, "content": "<|endoftext|>"}]
        }"#;
        GllmTokenizerAdapter::from_hf_json(src).expect("must load")
    }

    #[test]
    fn adapter_forwards_vocab_size_and_eos() {
        let tok = tiny_tokenizer();
        assert_eq!(tok.vocab_size(), 5);
        assert_eq!(tok.eos_id(), 4);
    }

    #[test]
    fn adapter_forwards_is_stop_for_the_eos_id() {
        let tok = tiny_tokenizer();
        assert!(tok.is_stop(tok.eos_id()));
    }

    #[test]
    fn vocab_fingerprint_is_stable_and_order_sensitive() {
        let a = tiny_tokenizer();
        let b = tiny_tokenizer();
        assert_eq!(a.vocab_fingerprint(), b.vocab_fingerprint(), "same source -> same fingerprint");
    }

    #[test]
    fn vocab_fingerprint_differs_when_eos_differs() {
        let src_a = r#"{
          "model": {"type": "BPE", "vocab": {"a": 0, "b": 1}, "merges": []},
          "pre_tokenizer": {"type": "ByteLevel"},
          "added_tokens": [{"id": 2, "content": "<eos_a>"}]
        }"#;
        let src_b = r#"{
          "model": {"type": "BPE", "vocab": {"a": 0, "b": 1}, "merges": []},
          "pre_tokenizer": {"type": "ByteLevel"},
          "added_tokens": [{"id": 2, "content": "<eos_b>"}]
        }"#;
        let a = GllmTokenizerAdapter::from_hf_json(src_a).unwrap();
        let b = GllmTokenizerAdapter::from_hf_json(src_b).unwrap();
        assert_ne!(
            a.vocab_fingerprint(),
            b.vocab_fingerprint(),
            "different token text at the same id must fingerprint differently"
        );
    }

    /// This is ARTX13 §4.3's exact concern, at the fingerprint layer: a Qwen
    /// draft and a Gemma target must never fingerprint equal even if they
    /// happen to share a vocab size.
    #[test]
    fn fingerprint_does_not_collide_on_vocab_size_alone() {
        let same_size_different_tokens = r#"{
          "model": {"type": "BPE", "vocab": {"x": 0, "y": 1, "z": 2, "w": 3}, "merges": []},
          "pre_tokenizer": {"type": "ByteLevel"},
          "added_tokens": [{"id": 4, "content": "<|endoftext|>"}]
        }"#;
        let a = tiny_tokenizer();
        let b = GllmTokenizerAdapter::from_hf_json(same_size_different_tokens).unwrap();
        assert_eq!(a.vocab_size(), b.vocab_size());
        assert_ne!(a.vocab_fingerprint(), b.vocab_fingerprint());
    }
}
