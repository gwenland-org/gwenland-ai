//! Stummañ Deskiñ: the tokenizer contract the data pipeline encodes through.
//!
//! # Why a trait rather than a direct call into glcore
//!
//! `glcore::tokenizer::GllmTokenizer` is the real one, and [`ABGllmTokenizer`]
//! wraps it. But it can only be constructed from a GGUF or a HuggingFace
//! `tokenizer.json`, neither of which is committed to this repository — a
//! Qwen2.5 vocabulary is ~7 MB and belongs with the model, not in a test
//! fixture. A data pipeline whose tests cannot run without a downloaded model
//! is a data pipeline that does not get tested.
//!
//! So encoding goes through this trait, and [`ABByteTokenizer`] provides a
//! deterministic, in-tree implementation for the tests. Swapping in a real
//! vocabulary is a one-line change at the call site and touches nothing in
//! [`crate::train::chatml`].
//!
//! # Prefix note
//!
//! `AB` (Algorithm Block) is the closest fit in the repo-wide table: a
//! tokenizer is a reusable building block with a defined algorithm, sitting
//! next to `ABLinear`. The table has no tokenizer-specific prefix, and no
//! other entry fits better — `VL` is for data with no behaviour, `RS` for
//! something with a `Drop` story, `PL` for a multi-stage workflow. Flagged for
//! JinXSuper rather than decided unilaterally.

use crate::error::{GlTrainError, Result};

/// The ChatML turn-opening marker.
pub const IM_START: &str = "<|im_start|>";
/// The ChatML turn-closing marker.
pub const IM_END: &str = "<|im_end|>";

/// Text to token ids, and back.
///
/// Traits take no prefix (naming rule 2).
///
/// # `encode` must not add a BOS
///
/// A ChatML prompt is assembled from several independently-encoded segments
/// (see [`crate::train::chatml`]). A tokenizer that prepended a BOS to each
/// one would scatter them through the middle of the sequence. Any BOS belongs
/// to the caller, once, at the front.
pub trait Tokenizer {
    /// Encode `text` to ids, with no BOS and no special-token wrapping.
    fn encode(&self, text: &str) -> Result<Vec<u32>>;

    /// Decode ids back to text. Lossy decoding is acceptable; this is used for
    /// inspection and for tests, never on the training path.
    fn decode(&self, ids: &[u32]) -> String;

    /// Size of the vocabulary, i.e. the logits dimension a model would need.
    fn vocab_size(&self) -> usize;

    /// The id of a special token, by its literal text.
    ///
    /// Returns [`GlTrainError::Data`] rather than an `Option` because every
    /// caller treats a missing ChatML marker as fatal: a prompt assembled
    /// without `<|im_start|>` is not a ChatML prompt, and silently continuing
    /// would train the model on a format it will never see at inference.
    fn special_id(&self, token: &str) -> Result<u32>;

    /// The padding id used to square off a batch.
    ///
    /// Padding positions are masked out of the loss by
    /// [`crate::train::chatml::IGNORE_INDEX`], so this value never reaches a
    /// gradient. It still has to be a real id: it is fed to the embedding
    /// lookup, which will index with it.
    fn pad_id(&self) -> u32;

    /// `<|im_start|>`, resolved once.
    fn im_start(&self) -> Result<u32> {
        self.special_id(IM_START)
    }

    /// `<|im_end|>`, resolved once.
    fn im_end(&self) -> Result<u32> {
        self.special_id(IM_END)
    }
}

/// A byte-level tokenizer with ChatML markers bolted on, for tests.
///
/// Ids `0..=255` are the raw bytes of the UTF-8 encoding. `256` is
/// `<|im_start|>`, `257` is `<|im_end|>`, `258` is the pad. Vocabulary size is
/// 259.
///
/// # What this is and is not for
///
/// It is for proving the *pipeline*: that a JSONL line becomes ids, that the
/// label mask lands on exactly the assistant span, that collation pads to a
/// rectangle. Every one of those properties is independent of which subword
/// algorithm produced the ids.
///
/// It is not for training a model anyone would use. Byte-level ids make
/// sequences roughly 4x longer than Qwen2.5's BPE would, and the embedding
/// table it implies has no relationship to a real one.
///
/// It has two properties the real tokenizer does not, both of which the tests
/// depend on: it needs no files, and `decode(encode(s)) == s` exactly for any
/// input, so a test can assert on recovered text rather than on ids.
#[derive(Debug, Clone, Default)]
pub struct ABByteTokenizer;

impl ABByteTokenizer {
    /// The id of the first non-byte token.
    const SPECIAL_BASE: u32 = 256;
    /// `<|im_start|>`.
    pub const IM_START_ID: u32 = 256;
    /// `<|im_end|>`.
    pub const IM_END_ID: u32 = 257;
    /// The padding id.
    pub const PAD_ID: u32 = 258;

    /// A new tokenizer. Stateless, so this is free.
    pub fn new() -> Self {
        Self
    }
}

impl Tokenizer for ABByteTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Ok(text.bytes().map(u32::from).collect())
    }

    fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::with_capacity(ids.len());
        let mut out = String::new();
        for &id in ids {
            if id < Self::SPECIAL_BASE {
                bytes.push(id as u8);
                continue;
            }
            // A special token interrupts the byte run, so flush what came
            // before it rather than letting the marker text land mid-character.
            if !bytes.is_empty() {
                out.push_str(&String::from_utf8_lossy(&bytes));
                bytes.clear();
            }
            out.push_str(match id {
                Self::IM_START_ID => IM_START,
                Self::IM_END_ID => IM_END,
                Self::PAD_ID => "<|pad|>",
                _ => "<|unk|>",
            });
        }
        if !bytes.is_empty() {
            out.push_str(&String::from_utf8_lossy(&bytes));
        }
        out
    }

    fn vocab_size(&self) -> usize {
        259
    }

    fn special_id(&self, token: &str) -> Result<u32> {
        match token {
            IM_START => Ok(Self::IM_START_ID),
            IM_END => Ok(Self::IM_END_ID),
            "<|pad|>" => Ok(Self::PAD_ID),
            other => Err(GlTrainError::Data(format!(
                "ABByteTokenizer has no special token {other:?}"
            ))),
        }
    }

    fn pad_id(&self) -> u32 {
        Self::PAD_ID
    }
}

/// The real tokenizer: `glcore::tokenizer::GllmTokenizer` behind [`Tokenizer`].
///
/// # How the special ids are found
///
/// By encoding the marker text and requiring exactly one id back.
/// `GllmTokenizer` splits its input on registered special tokens longest-match
/// first before running BPE, so a vocabulary that knows `<|im_start|>` returns
/// it as a single token. One that does not will shred the marker into a dozen
/// byte pieces, and the length check turns that into a named error instead of
/// a prompt that is quietly malformed.
///
/// `Vocab::token_to_id` is `pub(crate)` inside glcore, so this round trip is
/// the only way to ask the question from outside that crate.
pub struct ABGllmTokenizer {
    inner: glcore::tokenizer::GllmTokenizer,
    pad_id: u32,
}

impl ABGllmTokenizer {
    /// Wrap a loaded tokenizer, padding with `pad_id`.
    ///
    /// Pass the vocabulary's own pad token when it has one. Qwen2.5 does not
    /// ship a distinct pad, and its EOS is the conventional substitute —
    /// harmless here because every pad position is masked out of the loss.
    pub fn new(inner: glcore::tokenizer::GllmTokenizer, pad_id: u32) -> Self {
        Self { inner, pad_id }
    }

    /// Load from a GGUF file and pad with its EOS.
    pub fn from_gguf_path(path: &str) -> Result<Self> {
        let inner = glcore::tokenizer::GllmTokenizer::from_gguf_path(path)
            .map_err(|e| GlTrainError::Data(format!("loading tokenizer from {path}: {e:?}")))?;
        let pad_id = inner.eos_id();
        Ok(Self::new(inner, pad_id))
    }

    /// Load from a HuggingFace `tokenizer.json` and pad with its EOS.
    pub fn from_hf_json_path(path: &str) -> Result<Self> {
        let inner = glcore::tokenizer::GllmTokenizer::from_hf_json_path(path)
            .map_err(|e| GlTrainError::Data(format!("loading tokenizer from {path}: {e:?}")))?;
        let pad_id = inner.eos_id();
        Ok(Self::new(inner, pad_id))
    }

    /// The wrapped tokenizer.
    pub fn inner(&self) -> &glcore::tokenizer::GllmTokenizer {
        &self.inner
    }
}

impl Tokenizer for ABGllmTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        // `false`: no BOS. See the trait's note — segments are encoded
        // separately and concatenated, so a per-segment BOS would land in the
        // middle of the sequence.
        self.inner
            .encode(text, false)
            .map_err(|e| GlTrainError::Data(format!("encoding failed: {e:?}")))
    }

    fn decode(&self, ids: &[u32]) -> String {
        self.inner.decode(ids, false)
    }

    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    fn special_id(&self, token: &str) -> Result<u32> {
        let ids = self
            .inner
            .encode(token, false)
            .map_err(|e| GlTrainError::Data(format!("encoding {token:?} failed: {e:?}")))?;
        match ids.as_slice() {
            [id] => Ok(*id),
            other => Err(GlTrainError::Data(format!(
                "{token:?} is not a single token in this vocabulary — it encoded to {} ids. \
                 This vocabulary is not ChatML-aware; a prompt built from it would not match \
                 what the model sees at inference.",
                other.len()
            ))),
        }
    }

    fn pad_id(&self) -> u32 {
        self.pad_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_tokenizer_round_trips_ascii_exactly() {
        let t = ABByteTokenizer::new();
        let s = "fn main() { println!(\"hi\"); }";
        assert_eq!(t.decode(&t.encode(s).unwrap()), s);
    }

    /// The corpus is Indonesian-English with emoji, so multi-byte input is the
    /// normal case here, not an edge case.
    #[test]
    fn byte_tokenizer_round_trips_multibyte_utf8() {
        let t = ABByteTokenizer::new();
        let s = "Kamu adalah Gwen — 漢字 \u{1f60a} café";
        let ids = t.encode(s).unwrap();
        assert_eq!(ids.len(), s.len(), "one id per UTF-8 byte");
        assert_eq!(t.decode(&ids), s);
    }

    #[test]
    fn byte_tokenizer_ids_stay_inside_the_vocabulary() {
        let t = ABByteTokenizer::new();
        for id in t.encode("anything at all \u{1f4a1}").unwrap() {
            assert!(id < t.vocab_size() as u32);
        }
        assert!(ABByteTokenizer::PAD_ID < t.vocab_size() as u32);
    }

    #[test]
    fn special_ids_are_distinct_from_every_byte() {
        let t = ABByteTokenizer::new();
        let start = t.im_start().unwrap();
        let end = t.im_end().unwrap();
        assert_ne!(start, end);
        for id in [start, end, t.pad_id()] {
            assert!(id >= 256, "special {id} collides with a byte id");
        }
    }

    #[test]
    fn decoding_a_special_token_renders_its_marker_text() {
        let t = ABByteTokenizer::new();
        let mut ids = vec![t.im_start().unwrap()];
        ids.extend(t.encode("system").unwrap());
        ids.push(t.im_end().unwrap());
        assert_eq!(t.decode(&ids), "<|im_start|>system<|im_end|>");
    }

    /// A marker that interrupts a multi-byte character must not be spliced
    /// into the middle of it — the byte run is flushed first.
    #[test]
    fn a_special_token_does_not_corrupt_an_adjacent_multibyte_run() {
        let t = ABByteTokenizer::new();
        let mut ids = t.encode("漢").unwrap();
        ids.push(t.im_end().unwrap());
        ids.extend(t.encode("字").unwrap());
        assert_eq!(t.decode(&ids), "漢<|im_end|>字");
    }

    #[test]
    fn an_unknown_special_token_is_an_error_not_a_guess() {
        let t = ABByteTokenizer::new();
        assert!(t.special_id("<|endoftext|>").is_err());
    }
}
