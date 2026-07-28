//! Vocabulary data and its loaders.
//!
//! The vocabulary is deliberately *data* — this crate never opens a file it
//! was not handed. GGUF lives in `glcore`, so `glcore` builds a [`Vocab`]
//! from its own parsed metadata via [`Vocab::from_parts`]; only the
//! HuggingFace `tokenizer.json` loader lives here, because it needs nothing
//! but `serde_json`.

use std::collections::HashMap;

use crate::pretok::{BpeSplit, PreTok};
use crate::TokError;

/// Which encoding convention a vocabulary uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Style {
    /// SentencePiece-like: `▁` marks spaces, per-token scores drive merges,
    /// unknown bytes fall back to `<0xNN>`.
    Spm,
    /// GPT-2-like: bytes are remapped to printable chars, an explicit merge
    /// list drives merges.
    ByteLevel,
    /// The hybrid Gemma-4 and Sarvam use: SentencePiece *surface form* — `▁`
    /// for spaces, raw UTF-8 rather than the GPT-2 byte remap, `<0xNN>` byte
    /// fallback — but merges driven by an explicit **merge list**, as
    /// byte-level does, not by per-token scores.
    ///
    /// ⚠️ It is worth naming this rather than bending [`Spm`](Style::Spm),
    /// because the two disagree on the thing that decides token ids: SPM will
    /// merge any pair whose concatenation is in the vocabulary, ranked by
    /// score; this style merges only pairs that appear in the merge list,
    /// ranked by position. A vocabulary carrying both (Gemma-4 ships 262 144
    /// scores *and* 514 906 merges) silently produces different ids depending
    /// on which one is believed.
    SpmBpe,
}

/// The SentencePiece space marker.
pub const SPM_SPACE: char = '\u{2581}';

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
    /// distinct rules together; see [`crate::bpe::Ranker::rank`].
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
        specials_by_len.sort_by_key(|s| std::cmp::Reverse(s.0.len()));

        let mut stop_ids: std::collections::HashSet<u32> = STOP_TOKEN_STRINGS
            .iter()
            .filter_map(|s| token_to_id.get(*s).copied())
            .collect();
        stop_ids.insert(p.eos_id);

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

        let added: Vec<u32> = v
            .get("added_tokens")
            .and_then(|a| a.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|t| t.get("id")?.as_u64().map(|i| i as u32))
                    .collect()
            })
            .unwrap_or_default();

        // ByteLevel's `add_prefix_space`, when present, is the HF spelling of
        // SentencePiece's `add_dummy_prefix`.
        let add_dummy_prefix = find_bytelevel_flag(v.get("pre_tokenizer"), "add_prefix_space")
            .unwrap_or(style == Style::Spm);

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
    pub fn style(&self) -> Style {
        self.style
    }
    /// Raw surface form of a token id, before any byte unmapping.
    pub fn token_str(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(id as usize).map(String::as_str)
    }
}

/// Recognise the pre-tokenizer families this crate supports (see
/// [`crate::pretok`]); refuse anything else.
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
}
