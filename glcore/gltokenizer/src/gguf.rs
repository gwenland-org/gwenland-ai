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
pub fn pretok_from_name(name: &str) -> Result<PreTok, TokError> {
    Ok(PreTok::Bpe(match name {
        // GPT-2 shape, digits may take a leading space.
        "default" | "gpt-2" | "olmo" | "jais" | "mpt" | "bloom" | "gpt3-finnish"
        | "codeshell" | "smaug" | "poro" | "viking" => BpeSplit::GPT2,

        // GPT-2 shape, digits never take a leading space.
        "starcoder" | "refact" => BpeSplit::STARCODER,

        // cl100k shape, three-digit runs.
        "llama3" | "llama-bpe" | "dbrx" | "smollm" | "exaone" | "minerva-7b" | "gpt-4o"
        | "llama4" | "seed-coder" => BpeSplit::LLAMA3,

        // cl100k shape, single digits.
        "qwen2" | "deepseek-llm" | "deepseek-coder" | "deepseek-v3" | "stablelm2"
        | "chatglm-bpe" | "tekken" | "bailingmoe" | "trillion" => BpeSplit::QWEN2,

        // ⚠️ REFUSED DELIBERATELY, not unimplemented.
        //
        // `command-r` pre-tokenizes correctly but diverges on one reference
        // vector (45/46): on a long whitespace run the reference splits
        // `\n    \n     ` + `\n` where rank-order BPE reaches
        // `\n    \n    ` + ` \n`. The cause is in merge application,
        // not in splitting, and is unresolved.
        //
        // Claiming support at 45/46 would mean silently mis-tokenizing
        // whitespace-heavy input such as source code — exactly the failure
        // this crate refuses elsewhere. Re-enable only at 46/46.
        //
        // `falcon` is refused for a different reason: its pattern carries an
        // extra leading `[<punct>$+<=>^~|]+` arm that none of the three
        // shapes here express.

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
fn ignore_merges_for(pre_name: &str) -> bool {
    matches!(pre_name, "llama3" | "llama-bpe" | "llama4" | "gpt-4o")
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
        add_bos_default: get_bool("tokenizer.ggml.add_bos_token").unwrap_or(style == Style::Spm),
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
}
