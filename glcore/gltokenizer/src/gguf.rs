//! Minimal, read-only GGUF **metadata** reader — just enough to build a
//! [`Vocab`].
//!
//! This deliberately does not parse tensor data. A vocabulary lives entirely
//! in the key-value block, so reading stops once that block is consumed.
//! Keeping it here rather than depending on a general GGUF crate means
//! `gltokenizer` stays self-sufficient for the two vocabulary sources that
//! actually exist (`tokenizer.json` and GGUF) without pulling in a tensor
//! stack it has no use for.
//!
//! Layout, all little-endian:
//!
//! ```text
//! "GGUF" | version:u32 | tensor_count:u64 | kv_count:u64 | kv*
//! kv  := key:string | type:u32 | value
//! str := len:u64 | bytes
//! arr := elem_type:u32 | len:u64 | elem*
//! ```

use std::collections::HashMap;

use crate::pretok::{BpeSplit, PreTok};
use crate::vocab::{Style, Vocab, VocabParts};
use crate::TokError;

/// A decoded metadata value. Only the shapes a vocabulary needs.
#[derive(Debug, Clone)]
pub enum Meta {
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    ArrStr(Vec<String>),
    ArrF32(Vec<f32>),
    ArrI32(Vec<i32>),
    /// Present but not a shape this reader decodes.
    Other,
}

struct Cursor<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> Cursor<'a> {
    fn need(&self, n: usize) -> Result<(), TokError> {
        if self.i + n > self.b.len() {
            Err(TokError::Gguf("unexpected end of file".into()))
        } else {
            Ok(())
        }
    }
    fn u8(&mut self) -> Result<u8, TokError> {
        self.need(1)?;
        let v = self.b[self.i];
        self.i += 1;
        Ok(v)
    }
    fn u32(&mut self) -> Result<u32, TokError> {
        self.need(4)?;
        let v = u32::from_le_bytes(self.b[self.i..self.i + 4].try_into().unwrap());
        self.i += 4;
        Ok(v)
    }
    fn u64(&mut self) -> Result<u64, TokError> {
        self.need(8)?;
        let v = u64::from_le_bytes(self.b[self.i..self.i + 8].try_into().unwrap());
        self.i += 8;
        Ok(v)
    }
    fn f32(&mut self) -> Result<f32, TokError> {
        Ok(f32::from_bits(self.u32()?))
    }
    fn str(&mut self) -> Result<String, TokError> {
        let n = self.u64()? as usize;
        self.need(n)?;
        let s = String::from_utf8_lossy(&self.b[self.i..self.i + n]).into_owned();
        self.i += n;
        Ok(s)
    }
    /// Skip a value of `ty` without decoding it.
    fn skip(&mut self, ty: u32) -> Result<(), TokError> {
        match ty {
            0 | 1 | 7 => self.i += 1,
            2 | 3 => self.i += 2,
            4..=6 => self.i += 4,
            10..=12 => self.i += 8,
            8 => {
                let n = self.u64()? as usize;
                self.i += n;
            }
            9 => {
                let et = self.u32()?;
                let n = self.u64()?;
                for _ in 0..n {
                    self.skip(et)?;
                }
            }
            other => return Err(TokError::Gguf(format!("unknown value type {other}"))),
        }
        self.need(0)?;
        Ok(())
    }
}

/// Parse the metadata block. Keys are returned verbatim.
pub fn read_metadata(bytes: &[u8]) -> Result<HashMap<String, Meta>, TokError> {
    let mut c = Cursor { b: bytes, i: 0 };
    c.need(4)?;
    if &bytes[..4] != b"GGUF" {
        return Err(TokError::Gguf("not a GGUF file".into()));
    }
    c.i = 4;
    let version = c.u32()?;
    if !(1..=3).contains(&version) {
        return Err(TokError::Gguf(format!("unsupported GGUF version {version}")));
    }
    let _tensor_count = c.u64()?;
    let kv_count = c.u64()?;

    let mut out = HashMap::new();
    for _ in 0..kv_count {
        let key = c.str()?;
        let ty = c.u32()?;
        // Only decode what a vocabulary needs; skip the rest cheaply.
        let want = key.starts_with("tokenizer.ggml.") || key == "general.architecture";
        if !want {
            c.skip(ty)?;
            continue;
        }
        let v = match ty {
            4 => Meta::U32(c.u32()?),
            5 => Meta::I32(c.u32()? as i32),
            6 => Meta::F32(c.f32()?),
            7 => Meta::Bool(c.u8()? != 0),
            8 => Meta::Str(c.str()?),
            9 => {
                let et = c.u32()?;
                let n = c.u64()? as usize;
                match et {
                    8 => {
                        let mut v = Vec::with_capacity(n);
                        for _ in 0..n {
                            v.push(c.str()?);
                        }
                        Meta::ArrStr(v)
                    }
                    6 => {
                        let mut v = Vec::with_capacity(n);
                        for _ in 0..n {
                            v.push(c.f32()?);
                        }
                        Meta::ArrF32(v)
                    }
                    5 | 4 => {
                        let mut v = Vec::with_capacity(n);
                        for _ in 0..n {
                            v.push(c.u32()? as i32);
                        }
                        Meta::ArrI32(v)
                    }
                    other => {
                        for _ in 0..n {
                            c.skip(other)?;
                        }
                        Meta::Other
                    }
                }
            }
            other => {
                c.skip(other)?;
                Meta::Other
            }
        };
        out.insert(key, v);
    }
    Ok(out)
}

/// Map llama.cpp's canonical pre-tokenizer name onto a scanner shape.
///
/// ⭐ GGUF records the pre-tokenizer by *name* (`tokenizer.ggml.pre`), which
/// llama.cpp assigns by hashing the model's regex. That is a far more reliable
/// signal than re-sniffing a regex string, so GGUF vocabularies key off it.
///
/// Unknown names are refused rather than defaulted: a wrong pre-tokenizer
/// silently changes token ids, which is the failure mode this crate exists to
/// remove.
/// ⛔ **This table is grouped by the pre-tokenizer's actual regex, not by
/// model family.** Names that look related routinely land in different
/// groups — `smollm` is not cl100k, `trillion` is not Qwen2, `codeshell` is
/// not GPT-2 — and an earlier version of this table got thirteen of them
/// wrong precisely by grouping on the name. Every entry below was read off
/// llama.cpp's `pre_type` → `regex_exprs` switch, and each group is one
/// pattern this crate implements *exactly*. Adding a name means checking
/// which `case` arm it reaches, not which model it sounds like.
pub fn pretok_from_name(name: &str) -> Result<PreTok, TokError> {
    Ok(PreTok::Bpe(match name {
        // ── GPT-2 arm alone ───────────────────────────────────────────────
        // `'s|…| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)`
        // llama.cpp: GPT2 / MPT / OLMO / JAIS / TRILLION / GRANITE_DOCLING.
        "gpt-2" | "phi-2" | "jina-es" | "jina-de" | "gigachat" | "jina-v2-es" | "jina-v2-de"
        | "a.x-4.0" | "mellum" | "modern-bert" | "exaone4" | "mpt" | "olmo" | "jais"
        | "trillion" | "granite-docling" => BpeSplit::GPT2,

        // ── `\p{N}` first, then the GPT-2 arm ─────────────────────────────
        // Splitting digits off first is why they never absorb a leading
        // space; that is the whole difference from the group above.
        // llama.cpp: STARCODER / REFACT / COMMAND_R / SMOLLM / CODESHELL /
        // EXAONE / MINERVA.
        //
        // ⭐ `command-r` belongs here, **measured 46/46**. A previous note in
        // this file claimed 45/46 with a divergence in *merge application*;
        // that was wrong — the miss was the splitter, and it is this shape.
        "starcoder" | "refact" | "command-r" | "smollm" | "codeshell" | "exaone"
        | "minerva-7b" => BpeSplit::STARCODER,

        // ── cl100k arms, three-digit runs ─────────────────────────────────
        // llama.cpp: LLAMA3 / DBRX / SMAUG / CHATGLM4 — one regex, three
        // pre_types. Only the LLAMA3 arm also sets `ignore_merges`; see
        // [`ignore_merges_for`].
        "llama3" | "llama-v3" | "llama-bpe" | "falcon3" | "falcon-h1" | "pixtral" | "midm-2.0"
        | "lfm2" | "jina-v5-nano" | "dbrx" | "smaug-bpe" | "glm4" | "chatglm-bpe" => {
            BpeSplit::LLAMA3
        }

        // ── cl100k arms, single digits ────────────────────────────────────
        // llama.cpp: QWEN2 / STABLELM2 / HUNYUAN / SOLAR_OPEN.
        "qwen2" | "deepseek-r1-qwen" | "kormo" | "f2llmv2" | "megrez" | "stablelm2"
        | "hunyuan" => BpeSplit::QWEN2,

        // ⚠️ APPROXIMATED, NOT EXACT — kept because it is *measured*, flagged
        // because the shape is not the same function.
        //
        // llama.cpp gives each of these a multi-expression pipeline rather
        // than one arm: `deepseek-llm` splits on `[\r\n]`, then a huge
        // explicit Latin/Greek/Cyrillic letter class, then `\s?[punct]+`,
        // `\s+$`, a CJK arm and `\p{N}+`; `deepseek-coder` uses `[\r\n]`,
        // `\s?\p{L}+`, `\s?\p{P}+`, a CJK arm and `\p{N}`.
        //
        // Both score **46/46** against the reference corpus under the QWEN2
        // arm, which is the same evidence every other entry here rests on. But
        // `\s?\p{L}+` lets *any* whitespace lead a word while the QWEN2 arm's
        // `[^\r\n\p{L}\p{N}]?` explicitly excludes `\r` and `\n`, so the two
        // are known to differ somewhere the corpus does not reach.
        //
        // Refusing a family that passes every vector we have would be its own
        // kind of dishonesty. The claim is therefore narrowed rather than
        // withdrawn, and `audit.rs` reports it in a separate tier.
        "deepseek-llm" | "deepseek-coder" => BpeSplit::QWEN2,

        // ⚠️ `qwen35` is the QWEN2 pattern with `\p{L}` widened to
        // `[\p{L}\p{M}]` in the letter arm (and `\p{M}` excluded from the
        // punctuation arm). This crate does not carry a `\p{M}` table — but
        // its letter class is `char::is_alphabetic`, already a documented
        // superset of `\p{L}` that absorbs most combining marks (see
        // `pretok::is_letter`). That approximation is *closer* to qwen35 than
        // to qwen2, and qwen35 scores **50/50** against the reference corpus,
        // whose vectors include Khmer combining marks.
        //
        // So this is measured, not assumed — but the residual gap is real:
        // marks outside `Other_Alphabetic` (e.g. U+0301 COMBINING ACUTE) are
        // punctuation to us and letters to qwen35. No reference vector covers
        // that, which is exactly why it is written down here.
        "qwen35" => BpeSplit::QWEN2,

        // ── everything else is refused ────────────────────────────────────
        //
        // Names deliberately NOT mapped, with the reason, so the next person
        // does not re-add them by pattern-matching on the family:
        //
        // * `default` — NOT the GPT-2 shape. llama.cpp's fallback arm is a
        //   four-expression pipeline (punct run, GPT-2 arm, `\p{N}+`,
        //   `[0-9][0-9][0-9]`), and it exists for GGUFs that lost their
        //   `tokenizer.ggml.pre` key. llama.cpp itself logs "GENERATION
        //   QUALITY WILL BE DEGRADED" when it lands here. Refusing is the
        //   honest response to missing metadata.
        // * `falcon` — same shape as `default` minus `\p{N}+`, plus a
        //   backtick in the punctuation class. Needs the pipeline.
        // * `bloom`, `gpt3-finnish`, `poro-chat`, `viking` — a single
        //   `" ?[^(\s|.,!?…。，、।۔،)]+"` expression, unrelated to any arm here.
        // * `gpt-4o`, `llama4`, `tekken`, `tiny_aya`, `youtu` — case-split
        //   letter runs written as lookahead groups
        //   (`((?=[\p{L}])([^a-z]))*`), which no axis here expresses.
        // * `deepseek-v3`, `hunyuan-dense`, `joyai-llm` — `\p{P}`/`\p{S}`
        //   classes plus a CJK arm.
        // * `seed-coder` — QWEN2-like but its punctuation arm excludes
        //   `[\r\n]` instead of trailing them.
        // * `bailingmoe` — QWEN2-like but `\s*[\r\n]` without the `+`.
        // * `superbpe`, `chameleon`, `kimi-k2`, `grok-2`, `afmoe` — bespoke.
        other => return Err(TokError::UnsupportedPreTokenizer(other.to_string())),
    }))
}

/// Whether this family emits a pre-token directly when the vocabulary already
/// contains it, instead of running the merge loop (Llama-3's `ignore_merges`).
///
/// ⚠️ Behavioural, not cosmetic: for " Việt" on llama-bpe, rank-order BPE
/// reaches `ĠVi|á»ĩ|t` because `ĠV+i` (rank 31158) fires before `á»+ĩ`
/// (69499), making the whole-token form unreachable. The reference emits the
/// single vocabulary entry.
///
/// ⛔ This is a property of the *pre_type*, not of the regex shape, so it does
/// NOT follow the grouping in [`pretok_from_name`]. `dbrx`, `smaug-bpe`,
/// `glm4` and `chatglm-bpe` share llama3's regex but do **not** set this flag.
/// An earlier version listed `llama4` and `gpt-4o` here — both wrong: they
/// reach llama.cpp's `GPT4O` arm, which sets no such flag (and whose regex
/// this crate refuses anyway).
fn ignore_merges_for(pre_name: &str) -> bool {
    matches!(
        pre_name,
        "llama3" | "llama-v3" | "llama-bpe" | "falcon3" | "falcon-h1" | "pixtral" | "midm-2.0"
            | "lfm2" | "jina-v5-nano"
    )
}

/// Families where llama.cpp forces a BOS token on regardless of what
/// `tokenizer.ggml.add_bos_token` says.
///
/// ⚠️ Metadata is not the last word here. The same `pre_type` arm that sets
/// [`ignore_merges_for`] also sets `add_bos = true` unconditionally, so a
/// llama-3 GGUF whose metadata says `false` still gets a BOS in llama.cpp.
/// Honouring only the metadata would shift every id by one position on the
/// models most likely to be run.
fn force_add_bos(pre_name: &str) -> bool {
    ignore_merges_for(pre_name)
}

/// Build a [`Vocab`] from GGUF bytes.
pub fn vocab_from_gguf(bytes: &[u8]) -> Result<Vocab, TokError> {
    let m = read_metadata(bytes)?;

    let get_str = |k: &str| match m.get(k) {
        Some(Meta::Str(s)) => Some(s.clone()),
        _ => None,
    };
    let get_u32 = |k: &str| match m.get(k) {
        Some(Meta::U32(v)) => Some(*v),
        Some(Meta::I32(v)) if *v >= 0 => Some(*v as u32),
        _ => None,
    };
    let get_bool = |k: &str| match m.get(k) {
        Some(Meta::Bool(b)) => Some(*b),
        _ => None,
    };

    let model = get_str("tokenizer.ggml.model")
        .ok_or(TokError::MissingField("tokenizer.ggml.model"))?;

    let tokens = match m.get("tokenizer.ggml.tokens") {
        Some(Meta::ArrStr(v)) => v.clone(),
        _ => return Err(TokError::MissingField("tokenizer.ggml.tokens")),
    };

    let scores = match m.get("tokenizer.ggml.scores") {
        Some(Meta::ArrF32(v)) => v.clone(),
        _ => Vec::new(),
    };

    let merges: Vec<(String, String)> = match m.get("tokenizer.ggml.merges") {
        Some(Meta::ArrStr(v)) => v
            .iter()
            .filter_map(|s| s.split_once(' ').map(|(a, b)| (a.to_string(), b.to_string())))
            .collect(),
        _ => Vec::new(),
    };

    // token_type: 2 = UNKNOWN, 3 = CONTROL, 4 = USER_DEFINED, 6 = BYTE.
    // CONTROL and USER_DEFINED are the ones that must not be shredded when
    // they appear literally in input text.
    let special_ids: Vec<u32> = match m.get("tokenizer.ggml.token_type") {
        Some(Meta::ArrI32(t)) => t
            .iter()
            .enumerate()
            .filter(|(_, &ty)| ty == 3 || ty == 4)
            .map(|(i, _)| i as u32)
            .collect(),
        _ => Vec::new(),
    };

    let pre_name = get_str("tokenizer.ggml.pre").unwrap_or_else(|| "default".into());
    let (style, pretok, add_dummy_prefix) = match model.as_str() {
        "gpt2" => (Style::ByteLevel, pretok_from_name(&pre_name)?, false),
        "llama" => (
            Style::Spm,
            PreTok::None,
            // SentencePiece's add_dummy_prefix. GGUF spells it
            // `add_space_prefix`; absent means the SPM default, which is on.
            get_bool("tokenizer.ggml.add_space_prefix").unwrap_or(true),
        ),
        // Gemma-4 (and Sarvam-MoE): declared as its own tokenizer model rather
        // than as `gpt2` + a pre-tokenizer name, because it is genuinely a
        // different shape — SentencePiece surface form, merge-list ranking,
        // and no word-level pre-splitting at all.
        //
        // ⚠️ The vocabulary also ships 262 144 `scores`. They are NOT used:
        // llama.cpp reads only `merges` for this model, and believing the
        // scores instead silently produces different ids. See [`Style::SpmBpe`].
        "gemma4" => (Style::SpmBpe, PreTok::Lines, false),
        other => return Err(TokError::UnsupportedModel(other.to_string())),
    };

    let n = tokens.len();
    VocabParts {
        id_to_token: tokens,
        scores: if style == Style::Spm && scores.len() == n {
            scores
        } else if style == Style::Spm {
            vec![0.0; n]
        } else {
            Vec::new()
        },
        merges,
        special_ids,
        style,
        pretok,
        add_dummy_prefix,
        ignore_merges: ignore_merges_for(&pre_name),
        bos_id: get_u32("tokenizer.ggml.bos_token_id"),
        eos_id: get_u32("tokenizer.ggml.eos_token_id").unwrap_or(0),
        unk_id: get_u32("tokenizer.ggml.unknown_token_id"),
        // llama.cpp overrides the metadata to `true` for Gemma-4 as well as
        // for the llama-3 arm, and logs that it is doing so.
        add_bos_default: force_add_bos(&pre_name)
            || style == Style::SpmBpe
            || get_bool("tokenizer.ggml.add_bos_token").unwrap_or(style == Style::Spm),
    }
    .into_vocab()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_non_gguf() {
        assert!(matches!(read_metadata(b"NOPE0000"), Err(TokError::Gguf(_))));
    }

    #[test]
    fn refuses_unknown_pretokenizer_name() {
        assert!(pretok_from_name("some-future-model").is_err());
    }

    #[test]
    fn maps_the_families_we_have_vectors_for() {
        assert_eq!(pretok_from_name("gpt-2").unwrap(), PreTok::Bpe(BpeSplit::GPT2));
        assert_eq!(
            pretok_from_name("llama-bpe").unwrap(),
            PreTok::Bpe(BpeSplit::LLAMA3)
        );
        assert_eq!(
            pretok_from_name("qwen2").unwrap(),
            PreTok::Bpe(BpeSplit::QWEN2)
        );
    }

    /// ⭐ Pins the thirteen entries an earlier table got wrong by grouping on
    /// the model name instead of on llama.cpp's `regex_exprs` arm. Each of
    /// these looks like it belongs somewhere else, which is exactly why it
    /// needs a test rather than a comment.
    #[test]
    fn name_groups_follow_the_regex_not_the_family() {
        // Sound cl100k, are actually the starcoder two-expression shape.
        for n in ["smollm", "exaone", "minerva-7b", "command-r"] {
            assert_eq!(
                pretok_from_name(n).unwrap(),
                PreTok::Bpe(BpeSplit::STARCODER),
                "{n}"
            );
        }
        // Sounds GPT-2-ish, is starcoder.
        assert_eq!(
            pretok_from_name("codeshell").unwrap(),
            PreTok::Bpe(BpeSplit::STARCODER)
        );
        // Sounds Qwen-ish, is plain GPT-2.
        assert_eq!(
            pretok_from_name("trillion").unwrap(),
            PreTok::Bpe(BpeSplit::GPT2)
        );
        // Sounds Qwen-ish, is cl100k (three-digit runs).
        assert_eq!(
            pretok_from_name("chatglm-bpe").unwrap(),
            PreTok::Bpe(BpeSplit::LLAMA3)
        );
    }

    /// ⛔ `default` is llama.cpp's *fallback* arm — a four-expression pipeline,
    /// not the GPT-2 shape it was previously mapped to. It is reached only by
    /// a GGUF missing `tokenizer.ggml.pre`, which llama.cpp itself warns will
    /// degrade generation quality.
    #[test]
    fn missing_pre_metadata_is_refused_not_guessed() {
        assert!(pretok_from_name("default").is_err());
    }

    /// Names whose regex this crate cannot express must stay refused, however
    /// familiar the model is.
    #[test]
    fn inexpressible_patterns_stay_refused() {
        for n in [
            "falcon",       // extra punct-run and 3-digit passes
            "gpt-4o",       // case-split lookahead groups
            "llama4",       // same
            "tekken",       // same
            "bloom",        // a single unrelated expression
            "viking",       // same, plus \p{N}
            "poro-chat",    // same
            "seed-coder",   // punctuation arm excludes [\r\n]
            "bailingmoe",   // \s*[\r\n] without the +
            "deepseek-v3",  // \p{P}/\p{S} classes plus a CJK arm
        ] {
            assert!(pretok_from_name(n).is_err(), "{n} must be refused");
        }
    }

    /// `ignore_merges` tracks llama.cpp's `pre_type`, which does NOT line up
    /// with the regex grouping: dbrx and chatglm share llama3's pattern but
    /// not its flag.
    #[test]
    fn ignore_merges_does_not_follow_the_regex_group() {
        assert!(ignore_merges_for("llama-bpe"));
        assert!(ignore_merges_for("pixtral"));
        for n in ["dbrx", "smaug-bpe", "glm4", "chatglm-bpe", "qwen2", "llama4", "gpt-4o"] {
            assert!(!ignore_merges_for(n), "{n} must not set ignore_merges");
        }
    }
}
