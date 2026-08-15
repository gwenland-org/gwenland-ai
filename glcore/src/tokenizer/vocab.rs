//! Vocabulary data and its loaders.
//!
//! The vocabulary is deliberately *data* — this module never opens a file it
//! was not handed, which is what lets [`Vocab::from_parts`] serve GGUF, the
//! HuggingFace `tokenizer.json` loader below, and any future container without
//! any of them knowing about the others.

use std::collections::HashMap;

use crate::tokenizer::pretok::{BpeSplit, PreTok};
pub use crate::tokenizer::style::Style;
use crate::tokenizer::TokError;

/// Token strings that end generation, across the families GGUF ships.
const STOP_TOKEN_STRINGS: &[&str] = &[
    "<|endoftext|>",
    "<|im_end|>",
    "<|end|>",
    "<eos>",
    "<end_of_turn>",
    "</s>",
    "<|eot_id|>",
    "<|end_of_text|>",
];

/// Everything needed to encode and decode, and nothing else.
pub struct Vocab {
    pub(crate) id_to_token: Vec<String>,
    pub(crate) token_to_id: HashMap<Box<str>, u32>,
    /// SPM merge scores, indexed by id. Empty for byte-level vocabularies.
    pub(crate) scores: Vec<f32>,
    /// Byte-level merge rules, keyed by concatenation and disambiguated by
    /// the left symbol's byte length. ⚠️ A plain concatenation key collapses
    /// distinct rules together; see [`crate::tokenizer::bpe::Ranker::rank`].
    pub(crate) merge_ranks: HashMap<Box<str>, Box<[(u32, u32)]>>,

    pub(crate) style: Style,
    pub(crate) pretok: PreTok,

    /// ⚠️ Whether a `▁` is prepended before encoding (SentencePiece's
    /// `add_dummy_prefix`). This is a **per-model setting**, not a constant:
    /// prepending it unconditionally shifts every token id for models that
    /// disable it, and corrupts chat templates by injecting a space into the
    /// middle of every rendered segment.
    pub(crate) add_dummy_prefix: bool,

    /// Llama-3's `ignore_merges`: when a pre-token exists verbatim in the
    /// vocabulary, emit it directly instead of running the merge loop.
    /// ⚠️ This is NOT an optimisation — it changes results. Rank-order BPE
    /// can be forced down a path that makes the whole-token form
    /// unreachable, so the two disagree on real inputs.
    pub(crate) ignore_merges: bool,

    /// Special tokens, longest-first, for the pre-encode split. Input text
    /// containing one of these must yield that token's id rather than being
    /// shredded into pieces.
    pub(crate) specials_by_len: Vec<(Box<str>, u32)>,
    /// Which bytes can *begin* a special token — the skip table that turns
    /// [`GllmTokenizer::find_special`] from one full scan per special into one
    /// scan total. Usually one byte is set (`<`), so most text skips straight
    /// through.
    pub(crate) special_first_byte: [bool; 256],
    pub(crate) special_ids: std::collections::HashSet<u32>,
    pub(crate) stop_ids: std::collections::HashSet<u32>,

    pub(crate) bos_id: Option<u32>,
    pub(crate) eos_id: u32,
    pub(crate) unk_id: Option<u32>,
    pub(crate) add_bos_default: bool,

    pub(crate) byte_to_char: [char; 256],
    pub(crate) char_to_byte: HashMap<char, u8>,
}

/// The pieces a caller must supply. Named rather than positional because
/// eight bare arguments is how the previous implementation grew a
/// `#[allow(clippy::too_many_arguments)]`.
pub struct VocabParts {
    pub id_to_token: Vec<String>,
    pub scores: Vec<f32>,
    pub merges: Vec<(String, String)>,
    pub special_ids: Vec<u32>,
    pub style: Style,
    pub pretok: PreTok,
    pub add_dummy_prefix: bool,
    pub ignore_merges: bool,
    pub bos_id: Option<u32>,
    pub eos_id: u32,
    pub unk_id: Option<u32>,
    pub add_bos_default: bool,
}

/// GPT-2 `bytes_to_unicode`: printable bytes map to themselves, the rest are
/// shifted into `U+0100+`.
fn gpt2_byte_map() -> ([char; 256], HashMap<char, u8>) {
    let mut b2c = ['\0'; 256];
    let mut c2b = HashMap::with_capacity(256);
    let mut shift = 0u32;
    for b in 0..=255u32 {
        let printable =
            (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
        let c = if printable {
            char::from_u32(b).unwrap()
        } else {
            let c = char::from_u32(256 + shift).unwrap();
            shift += 1;
            c
        };
        b2c[b as usize] = c;
        c2b.insert(c, b as u8);
    }
    (b2c, c2b)
}

impl VocabParts {
    /// Consume these parts into a validated [`Vocab`].
    pub fn into_vocab(self) -> Result<Vocab, TokError> {
        Vocab::from_parts(self)
    }
}

impl Vocab {
    pub fn from_parts(p: VocabParts) -> Result<Self, TokError> {
        if p.id_to_token.is_empty() {
            return Err(TokError::EmptyVocab);
        }
        if p.eos_id as usize >= p.id_to_token.len() {
            return Err(TokError::IdOutOfRange {
                what: "eos",
                id: p.eos_id,
                vocab: p.id_to_token.len(),
            });
        }
        if p.style == Style::Spm && p.scores.len() != p.id_to_token.len() {
            return Err(TokError::ScoreCountMismatch {
                scores: p.scores.len(),
                vocab: p.id_to_token.len(),
            });
        }

        let mut token_to_id = HashMap::with_capacity(p.id_to_token.len());
        for (id, t) in p.id_to_token.iter().enumerate() {
            // First id wins: duplicate surface forms exist in some vocabs and
            // the lower id is the canonical one.
            token_to_id.entry(t.as_str().into()).or_insert(id as u32);
        }

        // Group by concatenation, keeping every rule's (left_len, rank).
        // Collisions are real: llama-bpe has ~2.2 rules per concatenation.
        let mut grouped: HashMap<Box<str>, Vec<(u32, u32)>> = HashMap::new();
        for (rank, (a, b)) in p.merges.iter().enumerate() {
            grouped
                .entry(format!("{a}{b}").into_boxed_str())
                .or_default()
                .push((a.len() as u32, rank as u32));
        }
        let merge_ranks: HashMap<Box<str>, Box<[(u32, u32)]>> = grouped
            .into_iter()
            .map(|(k, mut v)| {
                // Best (lowest) rank first: the common case is one entry, and
                // when there are several the scan stops at the first match.
                v.sort_unstable();
                (k, v.into_boxed_slice())
            })
            .collect();

        let special_ids: std::collections::HashSet<u32> = p.special_ids.into_iter().collect();
        let mut specials_by_len: Vec<(Box<str>, u32)> = special_ids
            .iter()
            .filter_map(|&id| {
                p.id_to_token
                    .get(id as usize)
                    .map(|t| (t.as_str().into(), id))
            })
            .collect();
        // Longest first so `<|im_start|>` wins over a hypothetical `<|im`.
        //
        // ⛔ The tiebreak is not cosmetic, and it is a latent bug without it.
        // This list is built by iterating a `HashSet`, so its input order is
        // whatever the hasher produced; a length-only sort is *stable*, which
        // means two specials of equal length keep that arbitrary order — and
        // `find_special` prefers whichever comes first at an equal position.
        // Any change to the hasher would then move token ids, with nothing to
        // show for it. Found while evaluating a hash swap; the swap was
        // rejected, this stays.
        //
        // `(Reverse(len), text)` is a total order over distinct tokens, so the
        // result cannot depend on the hasher at all.
        specials_by_len.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then_with(|| a.0.cmp(&b.0)));

        let mut stop_ids: std::collections::HashSet<u32> = STOP_TOKEN_STRINGS
            .iter()
            .filter_map(|s| token_to_id.get(*s).copied())
            .collect();
        stop_ids.insert(p.eos_id);

        let mut special_first_byte = [false; 256];
        for (t, _) in &specials_by_len {
            if let Some(&b0) = t.as_bytes().first() {
                special_first_byte[b0 as usize] = true;
            }
        }

        let (byte_to_char, char_to_byte) = gpt2_byte_map();

        Ok(Vocab {
            id_to_token: p.id_to_token,
            token_to_id,
            scores: p.scores,
            merge_ranks,
            style: p.style,
            pretok: p.pretok,
            add_dummy_prefix: p.add_dummy_prefix,
            ignore_merges: p.ignore_merges,
            specials_by_len,
            special_first_byte,
            special_ids,
            stop_ids,
            bos_id: p.bos_id,
            eos_id: p.eos_id,
            unk_id: p.unk_id,
            add_bos_default: p.add_bos_default,
            byte_to_char,
            char_to_byte,
        })
    }

    /// Load from a HuggingFace `tokenizer.json`.
    ///
    /// ⚠️ Unrecognised `pre_tokenizer` configurations are **refused**, not
    /// approximated: a mis-split silently changes token ids, which is exactly
    /// the failure mode this crate exists to eliminate.
    pub fn from_hf_json(src: &str) -> Result<Self, TokError> {
        let v: serde_json::Value =
            serde_json::from_str(src).map_err(|e| TokError::Json(e.to_string()))?;

        let model = v.get("model").ok_or(TokError::MissingField("model"))?;
        let model_type = model.get("type").and_then(|t| t.as_str()).unwrap_or("BPE");
        if model_type != "BPE" {
            return Err(TokError::UnsupportedModel(model_type.to_string()));
        }

        // vocab: { token -> id }
        let vmap = model
            .get("vocab")
            .and_then(|m| m.as_object())
            .ok_or(TokError::MissingField("model.vocab"))?;
        let mut id_to_token = vec![String::new(); vmap.len()];
        for (tok, id) in vmap {
            let id = id.as_u64().ok_or(TokError::MissingField("model.vocab.id"))? as usize;
            if id >= id_to_token.len() {
                id_to_token.resize(id + 1, String::new());
            }
            id_to_token[id] = tok.clone();
        }

        // merges: ["a b", ...] or [["a","b"], ...]
        let merges = model
            .get("merges")
            .and_then(|m| m.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|m| match m {
                        serde_json::Value::String(s) => {
                            s.split_once(' ').map(|(a, b)| (a.to_string(), b.to_string()))
                        }
                        serde_json::Value::Array(p) if p.len() == 2 => {
                            Some((p[0].as_str()?.to_string(), p[1].as_str()?.to_string()))
                        }
                        _ => None,
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        let pretok = parse_pretok(v.get("pre_tokenizer"))?;
        let style = if merges.is_empty() && pretok == PreTok::None {
            Style::Spm
        } else {
            Style::ByteLevel
        };

        // ⛔ `added_tokens` carry BOTH an id and their text, and that text is
        // **not** in `model.vocab`. An earlier version collected only the ids,
        // which left every special token registered-but-textless:
        //
        //   * `id_to_token` stayed at `model.vocab`'s length, so ids past it
        //     were out of range — `Qwen/Qwen2-0.5B` has 151643 vocab entries
        //     and added tokens at 151643..=151645, so loading it failed
        //     outright with `eos id 151645 is outside a vocabulary of 151643`;
        //   * `specials_by_len` in `from_parts` looks each special id up in
        //     `id_to_token`, so they would have matched the empty string;
        //   * `<|endoftext|>` — the EOS of every Qwen2 base model — could
        //     neither be encoded nor decoded.
        //
        // ⚠️ This path had **no test coverage at all**. `tokenizer_parity.rs`'s
        // 14-vocabulary reference suite exercises `from_gguf_path` only, so
        // "14 vocabulary families exact" was never a statement about this
        // function. Found by loading a real HuggingFace checkpoint.
        let added_entries: Vec<(u32, String)> = v
            .get("added_tokens")
            .and_then(|a| a.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|t| {
                        let id = t.get("id")?.as_u64()? as u32;
                        let content = t.get("content")?.as_str()?.to_string();
                        Some((id, content))
                    })
                    .collect()
            })
            .unwrap_or_default();

        for (id, content) in &added_entries {
            let id = *id as usize;
            if id >= id_to_token.len() {
                id_to_token.resize(id + 1, String::new());
            }
            // An added token that also appears in `model.vocab` keeps the
            // vocab spelling; they agree in practice, and `model.vocab` is the
            // one the merge table is written against.
            if id_to_token[id].is_empty() {
                id_to_token[id] = content.clone();
            }
        }

        let added: Vec<u32> = added_entries.iter().map(|(id, _)| *id).collect();

        // ByteLevel's `add_prefix_space`, when present, is the HF spelling of
        // SentencePiece's `add_dummy_prefix`.
        let add_dummy_prefix = find_bytelevel_flag(v.get("pre_tokenizer"), "add_prefix_space")
            .unwrap_or(style == Style::Spm);

        // ⚠️ A heuristic, and a weak one: "the last added token" is positional,
        // not semantic. For `Qwen/Qwen2-0.5B` it picks `<|im_end|>` (151645) —
        // the *chat* terminator — while the model's own `config.json` declares
        // `eos_token_id: 151643` (`<|endoftext|>`). A base model that stops on
        // `<|im_end|>` never stops at all.
        //
        // `tokenizer.json` has no unambiguous EOS field, so this cannot be
        // resolved here. Callers that have the model's `config.json` should
        // prefer its `eos_token_id`, which is what `gljax::runtime::hf` does.
        // Keying off token *names* instead would be the mistake recorded in
        // the 13-of-24 pre-tokenizer table: a name table drifts silently.
        let eos_id = added.last().copied().unwrap_or(0);

        Vocab::from_parts(VocabParts {
            scores: if style == Style::Spm {
                vec![0.0; id_to_token.len()]
            } else {
                Vec::new()
            },
            id_to_token,
            merges,
            special_ids: added,
            style,
            pretok,
            add_dummy_prefix,
            ignore_merges: false,
            bos_id: None,
            eos_id,
            unk_id: None,
            add_bos_default: false,
        })
    }

    pub fn len(&self) -> usize {
        self.id_to_token.len()
    }
    pub fn is_empty(&self) -> bool {
        self.id_to_token.is_empty()
    }
    pub fn eos_id(&self) -> u32 {
        self.eos_id
    }
    pub fn bos_id(&self) -> Option<u32> {
        self.bos_id
    }
    pub fn add_bos_default(&self) -> bool {
        self.add_bos_default
    }
    pub fn is_stop(&self, id: u32) -> bool {
        self.stop_ids.contains(&id)
    }
    /// The splitter this vocabulary was built with.
    ///
    /// Exposed so `examples/tokenizer_profile.rs` can time the split in
    /// isolation — the alternative is guessing which shape a GGUF resolved to,
    /// and guessing is what that tool exists to stop.
    pub fn pretok(&self) -> PreTok {
        self.pretok
    }
    pub fn style(&self) -> Style {
        self.style
    }
    /// Raw surface form of a token id, before any byte unmapping.
    pub fn token_str(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(id as usize).map(String::as_str)
    }
}

/// Recognise the pre-tokenizer families this crate supports (see
/// [`crate::tokenizer::pretok`]); refuse anything else.
fn parse_pretok(v: Option<&serde_json::Value>) -> Result<PreTok, TokError> {
    let Some(v) = v else {
        return Ok(PreTok::None);
    };
    if v.is_null() {
        return Ok(PreTok::None);
    }

    let ty = v.get("type").and_then(|t| t.as_str()).unwrap_or("");
    match ty {
        "Sequence" => {
            let list = v
                .get("pretokenizers")
                .and_then(|p| p.as_array())
                .ok_or(TokError::MissingField("pre_tokenizer.pretokenizers"))?;
            // The shape in the wild is [Split(Regex), ByteLevel]; the Split
            // carries the pattern that matters.
            for p in list {
                if p.get("type").and_then(|t| t.as_str()) == Some("Split") {
                    let re = p
                        .get("pattern")
                        .and_then(|q| q.get("Regex"))
                        .and_then(|r| r.as_str())
                        .ok_or(TokError::MissingField("Split.pattern.Regex"))?;
                    return classify_regex(re);
                }
            }
            Ok(PreTok::Bpe(BpeSplit::GPT2))
        }
        "ByteLevel" => Ok(PreTok::Bpe(BpeSplit::GPT2)),
        "Split" => {
            let re = v
                .get("pattern")
                .and_then(|q| q.get("Regex"))
                .and_then(|r| r.as_str())
                .ok_or(TokError::MissingField("Split.pattern.Regex"))?;
            classify_regex(re)
        }
        "Metaspace" | "" => Ok(PreTok::None),
        other => Err(TokError::UnsupportedPreTokenizer(other.to_string())),
    }
}

/// Map a known pattern onto a scanner shape.
///
/// ⚠️ Matching is structural rather than literal so that harmless whitespace
/// or escaping differences between exporters do not cause a false refusal —
/// but an unknown pattern is still refused rather than guessed.
fn classify_regex(re: &str) -> Result<PreTok, TokError> {
    let modern = re.contains("(?i:") || re.contains("[^\\r\\n\\p{L}\\p{N}]");
    if !modern {
        return if re.contains("\\p{L}") {
            Ok(PreTok::Bpe(BpeSplit::GPT2))
        } else {
            Err(TokError::UnsupportedPreTokenizer(re.to_string()))
        };
    }
    // Digit run: "\p{N}{1,3}" (cl100k / Llama-3) vs bare "\p{N}" (Qwen2).
    let digit_run = if re.contains("\\p{N}{1,3}") {
        3
    } else if re.contains("\\p{N}{1,") {
        // Some exporters use other bounds; read the number rather than guess.
        re.split("\\p{N}{1,")
            .nth(1)
            .and_then(|t| t.split('}').next())
            .and_then(|n| n.parse::<usize>().ok())
            .ok_or_else(|| TokError::UnsupportedPreTokenizer(re.to_string()))?
    } else {
        1
    };
    // Qwen-3.5 widens the letter arm to `[\p{L}\p{M}]`; every other modern
    // pattern here leaves it at `\p{L}`.
    let marks_are_letters = re.contains("\\p{L}\\p{M}") || re.contains("\\p{M}\\p{L}");
    Ok(PreTok::Bpe(BpeSplit {
        modern: true,
        digit_run,
        space_digit: false,
        marks_are_letters,
        ..BpeSplit::QWEN2
    }))
}

fn find_bytelevel_flag(v: Option<&serde_json::Value>, key: &str) -> Option<bool> {
    let v = v?;
    if v.get("type").and_then(|t| t.as_str()) == Some("ByteLevel") {
        return v.get(key).and_then(|b| b.as_bool());
    }
    v.get("pretokenizers")?
        .as_array()?
        .iter()
        .find(|p| p.get("type").and_then(|t| t.as_str()) == Some("ByteLevel"))?
        .get(key)?
        .as_bool()
}

#[cfg(test)]
mod tests {
    use super::*;

    const QWEN_RE: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
    const CL100K_RE: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
    const GPT2_RE: &str = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

    #[test]
    fn classifies_the_three_real_patterns() {
        assert_eq!(classify_regex(QWEN_RE).unwrap(), PreTok::Bpe(BpeSplit::QWEN2));
        assert_eq!(classify_regex(CL100K_RE).unwrap(), PreTok::Bpe(BpeSplit::LLAMA3));
        assert_eq!(classify_regex(GPT2_RE).unwrap(), PreTok::Bpe(BpeSplit::GPT2));
    }

    #[test]
    fn refuses_an_unknown_pattern() {
        assert!(classify_regex("[abc]+").is_err());
    }

    #[test]
    fn empty_vocab_is_refused() {
        let e = Vocab::from_parts(VocabParts {
            id_to_token: vec![],
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
        });
        assert!(matches!(e, Err(TokError::EmptyVocab)));
    }

    #[test]
    fn eos_out_of_range_is_refused() {
        let e = Vocab::from_parts(VocabParts {
            id_to_token: vec!["a".into()],
            scores: vec![],
            merges: vec![],
            special_ids: vec![],
            style: Style::ByteLevel,
            pretok: PreTok::Bpe(BpeSplit::GPT2),
            add_dummy_prefix: false,
            ignore_merges: false,
            bos_id: None,
            eos_id: 99,
            unk_id: None,
            add_bos_default: false,
        });
        assert!(matches!(e, Err(TokError::IdOutOfRange { .. })));
    }

    #[test]
    fn spm_score_count_must_match_vocab() {
        let e = Vocab::from_parts(VocabParts {
            id_to_token: vec!["a".into(), "b".into()],
            scores: vec![0.0],
            merges: vec![],
            special_ids: vec![],
            style: Style::Spm,
            pretok: PreTok::None,
            add_dummy_prefix: true,
            ignore_merges: false,
            bos_id: None,
            eos_id: 0,
            unk_id: None,
            add_bos_default: true,
        });
        assert!(matches!(e, Err(TokError::ScoreCountMismatch { .. })));
    }

    /// ⛔ Regression: `added_tokens` must enter the vocabulary, not just the
    /// special-id set.
    ///
    /// Shaped after `Qwen/Qwen2-0.5B/tokenizer.json`, whose `model.vocab` ends
    /// at id 151642 and whose three added tokens sit at 151643..=151645. Before
    /// this, loading it failed with `eos id 151645 is outside a vocabulary of
    /// 151643`; the added tokens had ids but no text.
    #[test]
    fn hf_json_added_tokens_join_the_vocabulary() {
        let src = r#"{
          "added_tokens": [
            {"id": 3, "content": "<|endoftext|>", "special": true},
            {"id": 4, "content": "<|im_start|>", "special": true},
            {"id": 5, "content": "<|im_end|>", "special": true}
          ],
          "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false},
          "model": {
            "type": "BPE",
            "vocab": {"a": 0, "b": 1, "ab": 2},
            "merges": ["a b"]
          }
        }"#;
        let v = Vocab::from_hf_json(src).expect("must load");

        assert_eq!(v.len(), 6, "3 vocab entries + 3 added tokens");
        assert_eq!(v.id_to_token[3], "<|endoftext|>");
        assert_eq!(v.id_to_token[4], "<|im_start|>");
        assert_eq!(v.id_to_token[5], "<|im_end|>");
        // And they resolve in the other direction, which is what makes them
        // encodable at all.
        assert_eq!(v.token_to_id.get("<|endoftext|>").copied(), Some(3));
    }

    /// A file with no `added_tokens` must be unaffected.
    #[test]
    fn hf_json_without_added_tokens_keeps_the_plain_vocab_length() {
        let src = r#"{
          "pre_tokenizer": {"type": "ByteLevel"},
          "model": {"type": "BPE", "vocab": {"a": 0, "b": 1}, "merges": ["a b"]}
        }"#;
        let v = Vocab::from_hf_json(src).expect("must load");
        assert_eq!(v.len(), 2);
        assert_eq!(v.eos_id(), 0, "no added tokens -> the fallback");
    }

    /// An added token that duplicates a `model.vocab` entry must not clobber
    /// the vocab spelling the merge table was written against.
    #[test]
    fn hf_json_added_token_does_not_overwrite_an_existing_vocab_entry() {
        let src = r#"{
          "added_tokens": [{"id": 1, "content": "OVERWRITTEN", "special": true}],
          "pre_tokenizer": {"type": "ByteLevel"},
          "model": {"type": "BPE", "vocab": {"a": 0, "b": 1}, "merges": []}
        }"#;
        let v = Vocab::from_hf_json(src).expect("must load");
        assert_eq!(v.id_to_token[1], "b");
    }

    /// ⭐ The actual reported failure, reproduced directly: `Qwen/Qwen2-0.5B`
    /// has `added_tokens` at ids *past* `model.vocab`'s own length (151643
    /// entries, added tokens at 151643..=151645). Neither existing
    /// `from_hf_json` test above exercises this — both use an added-token id
    /// already inside the vocab's range. This is the scenario this function's
    /// own doc comment describes failing with "eos id N is outside a
    /// vocabulary of M" before the fix that resizes `id_to_token`.
    #[test]
    fn hf_json_added_token_past_model_vocab_length_extends_id_to_token() {
        let src = r#"{
          "model": {
            "type": "BPE",
            "vocab": {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4},
            "merges": []
          },
          "pre_tokenizer": {"type": "ByteLevel"},
          "added_tokens": [
            {"id": 5, "content": "<|endoftext|>"},
            {"id": 6, "content": "<|im_end|>"}
          ]
        }"#;
        let v = Vocab::from_hf_json(src).expect("must load");
        assert_eq!(v.len(), 7, "id_to_token must extend to cover the added tokens");
        assert_eq!(v.token_str(5), Some("<|endoftext|>"));
        assert_eq!(v.token_str(6), Some("<|im_end|>"));
        // The documented heuristic: EOS defaults to the last added token.
        assert_eq!(v.eos_id(), 6);
    }

    #[test]
    fn hf_json_refuses_a_non_bpe_model_type() {
        let src = r#"{"model": {"type": "WordPiece", "vocab": {}}}"#;
        match Vocab::from_hf_json(src) {
            Err(TokError::UnsupportedModel(_)) => {}
            other => panic!("expected UnsupportedModel, got {}", other.is_ok()),
        }
    }

    /// `merges` appears in the wild as either `["a b", ...]` or
    /// `[["a","b"], ...]` — both must parse to the same pairs.
    #[test]
    fn hf_json_merges_accept_both_the_string_and_array_pair_forms() {
        let string_form = r#"{
          "model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "ab": 2}, "merges": ["a b"]},
          "pre_tokenizer": {"type": "ByteLevel"}
        }"#;
        let array_form = r#"{
          "model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "ab": 2}, "merges": [["a", "b"]]},
          "pre_tokenizer": {"type": "ByteLevel"}
        }"#;
        let v1 = Vocab::from_hf_json(string_form).expect("string-form merges must parse");
        let v2 = Vocab::from_hf_json(array_form).expect("array-form merges must parse");
        assert_eq!(v1.len(), v2.len());
        assert_eq!(v1.style(), v2.style());
    }
}
