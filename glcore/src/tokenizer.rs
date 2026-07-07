//! BPE tokenizer, written from scratch.
//!
//! Two vocabulary styles are supported, covering the models GGUF files ship:
//!
//! * **SPM style** (llama family): tokens use `▁` for spaces, merging is
//!   driven by per-token scores, and unknown bytes fall back to `<0xNN>`
//!   byte tokens.
//! * **Byte-level BPE** (gpt2/qwen family): raw bytes are first mapped to
//!   printable unicode chars, then merged using an explicit merge list.
//!
//! Vocabularies load from GGUF metadata ([`Tokenizer::from_gguf`]) or a
//! HuggingFace `tokenizer.json` ([`Tokenizer::from_file`]).

use std::collections::{HashMap, HashSet};

use crate::error::GlError;
use crate::format::gguf::{GgufFile, GgufValue};

/// The SentencePiece "lower one eighth block" space marker.
const SPM_SPACE: char = '\u{2581}'; // ▁

/// Token strings that end generation, across the model families GGUF ships:
/// gpt2/qwen (`<|endoftext|>`, `<|im_end|>`), phi (`<|end|>`), gemma
/// (`<eos>`, `<end_of_turn>`), llama2 (`</s>`), llama3 (`<|eot_id|>`,
/// `<|end_of_text|>`). Resolved against the vocab at load time — absent
/// strings are simply skipped.
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

/// Which encoding convention the vocabulary uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Style {
    /// SentencePiece-like: `▁` marks spaces, scores drive merges.
    Spm,
    /// GPT-2-like: bytes are remapped to printable chars, merges are explicit.
    ByteLevel,
}

/// Byte-pair-encoding tokenizer with SPM and byte-level BPE support.
pub struct Tokenizer {
    vocab: HashMap<String, u32>,
    id_to_token: Vec<String>,
    merges: Vec<(String, String)>,
    merge_ranks: HashMap<(String, String), usize>,
    /// Per-token merge scores (SPM style); empty when absent.
    scores: Vec<f32>,
    /// Ids of control/special tokens (BOS, EOS, chat markers, ...).
    special_ids: HashSet<u32>,
    /// Ids that end generation: the metadata EOS plus every
    /// [`STOP_TOKEN_STRINGS`] entry present in the vocab.
    stop_token_ids: HashSet<u32>,
    special_tokens: HashMap<String, u32>,
    style: Style,
    bos_id: u32,
    eos_id: u32,
    unk_id: Option<u32>,
    /// Whether this model expects a BOS token prepended to prompts.
    add_bos_default: bool,
    /// GPT-2 byte → printable char table (byte-level style only).
    byte_to_char: Vec<char>,
    char_to_byte: HashMap<char, u8>,
}

/// Build the GPT-2 `bytes_to_unicode` table: printable bytes map to
/// themselves, the rest are shifted into the `U+0100+` range.
fn gpt2_byte_map() -> (Vec<char>, HashMap<char, u8>) {
    let mut byte_to_char = vec!['\0'; 256];
    let mut char_to_byte = HashMap::new();
    let mut shift = 0u32;
    for b in 0..=255u32 {
        let printable = (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
        let c = if printable {
            char::from_u32(b)
        } else {
            let c = char::from_u32(256 + shift);
            shift += 1;
            c
        }
        .unwrap_or('\0');
        byte_to_char[b as usize] = c;
        char_to_byte.insert(c, b as u8);
    }
    (byte_to_char, char_to_byte)
}

impl Tokenizer {
    fn build(
        id_to_token: Vec<String>,
        merges: Vec<(String, String)>,
        scores: Vec<f32>,
        special_ids: HashSet<u32>,
        style: Style,
        bos_id: u32,
        eos_id: u32,
        unk_id: Option<u32>,
    ) -> Self {
        let mut vocab = HashMap::with_capacity(id_to_token.len());
        for (id, tok) in id_to_token.iter().enumerate() {
            vocab.insert(tok.clone(), id as u32);
        }
        let merge_ranks = merges
            .iter()
            .enumerate()
            .map(|(rank, (a, b))| ((a.clone(), b.clone()), rank))
            .collect();
        let special_tokens = special_ids
            .iter()
            .filter_map(|&id| {
                id_to_token
                    .get(id as usize)
                    .map(|tok| (tok.clone(), id))
            })
            .collect();
        let (byte_to_char, char_to_byte) = gpt2_byte_map();
        let mut stop_token_ids: HashSet<u32> = STOP_TOKEN_STRINGS
            .iter()
            .filter_map(|s| vocab.get(*s).copied())
            .collect();
        stop_token_ids.insert(eos_id);
        Tokenizer {
            vocab,
            id_to_token,
            merges,
            merge_ranks,
            scores,
            special_ids,
            stop_token_ids,
            special_tokens,
            style,
            bos_id,
            eos_id,
            unk_id,
            add_bos_default: style == Style::Spm,
            byte_to_char,
            char_to_byte,
        }
    }

    /// Load the tokenizer embedded in a GGUF file's metadata.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, GlError> {
        let tokens_val = gguf
            .get_meta("tokenizer.ggml.tokens")
            .and_then(GgufValue::as_array)
            .ok_or_else(|| {
                GlError::Parse("GGUF has no tokenizer.ggml.tokens metadata".into())
            })?;
        let mut id_to_token = Vec::with_capacity(tokens_val.len());
        for v in tokens_val {
            id_to_token.push(
                v.as_str()
                    .ok_or_else(|| {
                        GlError::Parse("tokenizer.ggml.tokens contains a non-string".into())
                    })?
                    .to_string(),
            );
        }

        let scores: Vec<f32> = gguf
            .get_meta("tokenizer.ggml.scores")
            .and_then(GgufValue::as_array)
            .map(|arr| arr.iter().filter_map(GgufValue::as_f32).collect())
            .unwrap_or_default();

        let merges: Vec<(String, String)> = gguf
            .get_meta("tokenizer.ggml.merges")
            .and_then(GgufValue::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(GgufValue::as_str)
                    .filter_map(|m| {
                        m.split_once(' ')
                            .map(|(a, b)| (a.to_string(), b.to_string()))
                    })
                    .collect()
            })
            .unwrap_or_default();

        // token_type 3 = CONTROL, 4 = USER_DEFINED — both treated as special.
        let special_ids: HashSet<u32> = gguf
            .get_meta("tokenizer.ggml.token_type")
            .and_then(GgufValue::as_array)
            .map(|arr| {
                arr.iter()
                    .enumerate()
                    .filter(|(_, v)| matches!(v, GgufValue::I32(3) | GgufValue::I32(4)))
                    .map(|(i, _)| i as u32)
                    .collect()
            })
            .unwrap_or_default();

        let model = gguf
            .get_meta("tokenizer.ggml.model")
            .and_then(GgufValue::as_str)
            .unwrap_or("");
        let style = match model {
            "llama" => Style::Spm,
            "gpt2" => Style::ByteLevel,
            _ if !merges.is_empty() => Style::ByteLevel,
            _ => Style::Spm,
        };

        let meta_id = |key: &str| gguf.get_meta(key).and_then(GgufValue::as_u64);
        let bos_id = meta_id("tokenizer.ggml.bos_token_id").unwrap_or(1) as u32;
        let eos_id = meta_id("tokenizer.ggml.eos_token_id").unwrap_or(2) as u32;
        let unk_id = meta_id("tokenizer.ggml.unknown_token_id").map(|v| v as u32);

        let mut specials = special_ids;
        specials.insert(bos_id);
        specials.insert(eos_id);

        let mut tk = Self::build(
            id_to_token,
            merges,
            scores,
            specials,
            style,
            bos_id,
            eos_id,
            unk_id,
        );
        // Respect the model's explicit BOS preference when recorded.
        if let Some(GgufValue::Bool(add)) = gguf.get_meta("tokenizer.ggml.add_bos_token") {
            tk.add_bos_default = *add;
        }
        Ok(tk)
    }

    /// Load a HuggingFace-format `tokenizer.json`.
    pub fn from_file(path: &str) -> Result<Self, GlError> {
        let text = std::fs::read_to_string(path)?;
        let root: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| GlError::Parse(format!("tokenizer.json: invalid JSON: {e}")))?;

        let model = root
            .get("model")
            .ok_or_else(|| GlError::Parse("tokenizer.json: missing 'model'".into()))?;
        let vocab_obj = model
            .get("vocab")
            .and_then(|v| v.as_object())
            .ok_or_else(|| GlError::Parse("tokenizer.json: missing 'model.vocab'".into()))?;

        let mut pairs: Vec<(String, u32)> = Vec::with_capacity(vocab_obj.len());
        let mut max_id = 0u32;
        for (tok, idv) in vocab_obj {
            let id = idv
                .as_u64()
                .ok_or_else(|| GlError::Parse("tokenizer.json: non-integer token id".into()))?
                as u32;
            max_id = max_id.max(id);
            pairs.push((tok.clone(), id));
        }

        let mut id_to_token = vec![String::new(); max_id as usize + 1];
        for (tok, id) in &pairs {
            id_to_token[*id as usize] = tok.clone();
        }

        let merges: Vec<(String, String)> = model
            .get("merges")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|m| {
                        if let Some(s) = m.as_str() {
                            s.split_once(' ')
                                .map(|(a, b)| (a.to_string(), b.to_string()))
                        } else if let Some(pair) = m.as_array() {
                            match (pair.first().and_then(|x| x.as_str()), pair.get(1).and_then(|x| x.as_str())) {
                                (Some(a), Some(b)) => Some((a.to_string(), b.to_string())),
                                _ => None,
                            }
                        } else {
                            None
                        }
                    })
                    .collect()
            })
            .unwrap_or_default();

        let mut special_ids = HashSet::new();
        let mut bos_id = None;
        let mut eos_id = None;
        if let Some(added) = root.get("added_tokens").and_then(|v| v.as_array()) {
            for t in added {
                let id = match t.get("id").and_then(|v| v.as_u64()) {
                    Some(id) => id as u32,
                    None => continue,
                };
                let content = t.get("content").and_then(|v| v.as_str()).unwrap_or("");
                if t.get("special").and_then(|v| v.as_bool()).unwrap_or(false) {
                    special_ids.insert(id);
                    // extend the token table if the added token sits past the base vocab
                    if id as usize >= id_to_token.len() {
                        id_to_token.resize(id as usize + 1, String::new());
                    }
                    if id_to_token[id as usize].is_empty() {
                        id_to_token[id as usize] = content.to_string();
                    }
                    match content {
                        "<s>" | "<|startoftext|>" | "<|begin_of_text|>" => bos_id = Some(id),
                        "</s>" | "<|endoftext|>" | "<|end_of_text|>" | "<|im_end|>" => {
                            if eos_id.is_none() {
                                eos_id = Some(id);
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        let uses_spm = id_to_token.iter().any(|t| t.contains(SPM_SPACE));
        let style = if uses_spm { Style::Spm } else { Style::ByteLevel };

        Ok(Self::build(
            id_to_token,
            merges,
            Vec::new(),
            special_ids,
            style,
            bos_id.unwrap_or(1),
            eos_id.unwrap_or(2),
            None,
        ))
    }

    /// Encode text into token ids, optionally prepending the BOS token.
    pub fn encode(&self, text: &str, add_bos: bool) -> Vec<u32> {
        let mut ids = Vec::new();
        if add_bos {
            ids.push(self.bos_id);
        }
        match self.style {
            Style::Spm => self.encode_spm(text, &mut ids),
            Style::ByteLevel => self.encode_byte_level(text, &mut ids),
        }
        ids
    }

    /// SPM encoding: map spaces to `▁`, then greedily merge the adjacent
    /// pair with the best score (llama.cpp's bigram strategy). Characters
    /// that never reach the vocab fall back to `<0xNN>` byte tokens.
    fn encode_spm(&self, text: &str, ids: &mut Vec<u32>) {
        let prepared: String = format!("{SPM_SPACE}{}", text.replace(' ', "\u{2581}"));
        let mut symbols: Vec<String> = prepared.chars().map(|c| c.to_string()).collect();

        loop {
            let mut best: Option<(usize, f32)> = None;
            for i in 0..symbols.len().saturating_sub(1) {
                let cat = format!("{}{}", symbols[i], symbols[i + 1]);
                if let Some(&id) = self.vocab.get(&cat) {
                    let score = self
                        .scores
                        .get(id as usize)
                        .copied()
                        // No scores? Prefer longer merges, then leftmost.
                        .unwrap_or(cat.chars().count() as f32);
                    if best.map_or(true, |(_, s)| score > s) {
                        best = Some((i, score));
                    }
                }
            }
            match best {
                Some((i, _)) => {
                    let merged = format!("{}{}", symbols[i], symbols[i + 1]);
                    symbols[i] = merged;
                    symbols.remove(i + 1);
                }
                None => break,
            }
        }

        for sym in symbols {
            if let Some(&id) = self.vocab.get(&sym) {
                ids.push(id);
            } else {
                for byte in sym.bytes() {
                    let byte_tok = format!("<0x{byte:02X}>");
                    if let Some(&id) = self.vocab.get(&byte_tok) {
                        ids.push(id);
                    } else if let Some(unk) = self.unk_id {
                        ids.push(unk);
                    }
                }
            }
        }
    }

    /// Byte-level BPE: remap bytes to printable chars, chunk on spaces (a
    /// simplified GPT-2 pre-tokenizer), then merge by explicit merge rank.
    fn encode_byte_level(&self, text: &str, ids: &mut Vec<u32>) {
        for chunk in split_with_leading_space(text) {
            let mapped: String = chunk
                .bytes()
                .map(|b| self.byte_to_char[b as usize])
                .collect();
            let mut symbols: Vec<String> = mapped.chars().map(|c| c.to_string()).collect();

            loop {
                let mut best: Option<(usize, usize)> = None; // (index, rank)
                for i in 0..symbols.len().saturating_sub(1) {
                    let key = (symbols[i].clone(), symbols[i + 1].clone());
                    if let Some(&rank) = self.merge_ranks.get(&key) {
                        if best.map_or(true, |(_, r)| rank < r) {
                            best = Some((i, rank));
                        }
                    }
                }
                match best {
                    Some((i, _)) => {
                        let merged = format!("{}{}", symbols[i], symbols[i + 1]);
                        symbols[i] = merged;
                        symbols.remove(i + 1);
                    }
                    None => break,
                }
            }

            for sym in symbols {
                if let Some(&id) = self.vocab.get(&sym) {
                    ids.push(id);
                } else if let Some(unk) = self.unk_id {
                    ids.push(unk);
                }
            }
        }
    }

    /// Decode token ids back to text.
    ///
    /// With `skip_special = true`, control tokens (BOS/EOS/chat markers) are
    /// dropped from the output.
    pub fn decode(&self, ids: &[u32], skip_special: bool) -> String {
        let mut bytes: Vec<u8> = Vec::new();
        for &id in ids {
            if skip_special && self.special_ids.contains(&id) {
                continue;
            }
            self.token_bytes(id, &mut bytes);
        }
        let text = String::from_utf8_lossy(&bytes).into_owned();
        // SPM encoding always injects one `▁` prefix; undo it symmetrically
        // so decode(encode(text)) round-trips.
        if self.style == Style::Spm {
            text.strip_prefix(' ').map(str::to_string).unwrap_or(text)
        } else {
            text
        }
    }

    /// Raw vocabulary string for a token id (no byte remapping).
    pub fn decode_token(&self, id: u32) -> &str {
        self.id_to_token
            .get(id as usize)
            .map(String::as_str)
            .unwrap_or("")
    }

    /// Display text for a single token — what streaming should print.
    /// Unlike [`Tokenizer::decode`], no leading-space stripping happens.
    pub fn decode_token_text(&self, id: u32) -> String {
        let mut bytes = Vec::new();
        self.token_bytes(id, &mut bytes);
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Append the raw bytes a token represents to `out`.
    fn token_bytes(&self, id: u32, out: &mut Vec<u8>) {
        let tok = match self.id_to_token.get(id as usize) {
            Some(t) => t,
            None => return,
        };
        match self.style {
            Style::Spm => {
                if let Some(b) = parse_byte_token(tok) {
                    out.push(b);
                } else {
                    for c in tok.chars() {
                        if c == SPM_SPACE {
                            out.push(b' ');
                        } else {
                            let mut buf = [0u8; 4];
                            out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
                        }
                    }
                }
            }
            Style::ByteLevel => {
                for c in tok.chars() {
                    if let Some(&b) = self.char_to_byte.get(&c) {
                        out.push(b);
                    } else {
                        // Special/added tokens are stored verbatim.
                        let mut buf = [0u8; 4];
                        out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
                    }
                }
            }
        }
    }

    /// Number of entries in the vocabulary.
    pub fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }

    /// End-of-sequence token id.
    pub fn eos_id(&self) -> u32 {
        self.eos_id
    }

    /// True when `token_id` ends generation — the metadata EOS or any known
    /// stop marker (`<|im_end|>`, `<|endoftext|>`, `</s>`, ...) found in
    /// this model's vocab. O(1) lookup; safe to call per decoded token.
    pub fn is_stop_token(&self, token_id: u32) -> bool {
        self.stop_token_ids.contains(&token_id)
    }

    /// Every token id that ends generation for this model.
    pub fn stop_token_ids(&self) -> &HashSet<u32> {
        &self.stop_token_ids
    }

    /// Beginning-of-sequence token id.
    pub fn bos_id(&self) -> u32 {
        self.bos_id
    }

    /// Whether prompts should get a BOS token by default
    /// (`tokenizer.ggml.add_bos_token`; SPM-style models default to true).
    pub fn add_bos_default(&self) -> bool {
        self.add_bos_default
    }

    /// Special token name → id map (control tokens like BOS/EOS).
    pub fn special_tokens(&self) -> &HashMap<String, u32> {
        &self.special_tokens
    }

    /// The merge list this tokenizer was built with (may be empty for SPM).
    pub fn merges(&self) -> &[(String, String)] {
        &self.merges
    }
}

/// Parse an SPM byte-fallback token like `<0x0A>` into its byte value.
fn parse_byte_token(tok: &str) -> Option<u8> {
    let hex = tok.strip_prefix("<0x")?.strip_suffix('>')?;
    if hex.len() != 2 {
        return None;
    }
    u8::from_str_radix(hex, 16).ok()
}

/// Split text into chunks, each carrying at most one leading space —
/// a simplified GPT-2 pre-tokenizer that preserves every input byte.
fn split_with_leading_space(text: &str) -> Vec<&str> {
    let mut chunks = Vec::new();
    let bytes = text.as_bytes();
    let mut start = 0;
    for (i, &b) in bytes.iter().enumerate() {
        if b == b' ' && i > start {
            chunks.push(&text[start..i]);
            start = i;
        }
    }
    if start < text.len() {
        chunks.push(&text[start..]);
    }
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tiny SPM-style tokenizer: chars + a few merged words + byte tokens.
    fn spm_tokenizer() -> Tokenizer {
        let mut tokens: Vec<String> = vec!["<unk>".into(), "<s>".into(), "</s>".into()];
        // byte fallback tokens
        for b in 0..=255u16 {
            tokens.push(format!("<0x{b:02X}>"));
        }
        // single chars
        for c in "abcdefghijklmnopqrstuvwxyzHW".chars() {
            tokens.push(c.to_string());
        }
        tokens.push(SPM_SPACE.to_string());
        // merged pieces, higher score = merged first
        tokens.push(format!("{SPM_SPACE}Hello"));
        tokens.push(format!("{SPM_SPACE}World"));
        tokens.push("He".into());
        tokens.push("llo".into());
        tokens.push("Hello".into());
        tokens.push("World".into());
        tokens.push(format!("{SPM_SPACE}H"));
        tokens.push(format!("{SPM_SPACE}W"));
        let n = tokens.len();
        let mut specials = HashSet::new();
        specials.insert(1);
        specials.insert(2);
        Tokenizer::build(
            tokens,
            Vec::new(),
            vec![0.0; n], // uniform scores → longest-merge fallback not needed
            specials,
            Style::Spm,
            1,
            2,
            Some(0),
        )
    }

    #[test]
    fn spm_round_trip_ascii() {
        let tk = spm_tokenizer();
        for text in ["Hello World", "Hello", "abc def", "a  b", " leading"] {
            let ids = tk.encode(text, true);
            assert_eq!(ids[0], tk.bos_id());
            assert_eq!(tk.decode(&ids, true), text, "round-trip failed: {text:?}");
        }
    }

    #[test]
    fn spm_byte_fallback() {
        let tk = spm_tokenizer();
        // '!' is not in the vocab — must round-trip via <0x21>
        let ids = tk.encode("ab!", false);
        assert_eq!(tk.decode(&ids, true), "ab!");
    }

    #[test]
    fn byte_level_round_trip() {
        // Byte-level vocab: every mapped single byte is a token; no merges.
        let (byte_to_char, _) = gpt2_byte_map();
        let mut tokens: Vec<String> = vec!["<|endoftext|>".into()];
        for b in 0..=255usize {
            tokens.push(byte_to_char[b].to_string());
        }
        // one merge: "H" + "i" -> "Hi"
        tokens.push("Hi".into());
        let merges = vec![("H".to_string(), "i".to_string())];
        let mut specials = HashSet::new();
        specials.insert(0);
        let tk = Tokenizer::build(
            tokens,
            merges,
            Vec::new(),
            specials,
            Style::ByteLevel,
            0,
            0,
            None,
        );
        for text in ["Hi there", "hello world!", "tab\tand\nnewline", "  double"] {
            let ids = tk.encode(text, false);
            assert_eq!(tk.decode(&ids, true), text, "round-trip failed: {text:?}");
        }
        // the merge actually fires: "Hi" is one token
        let ids = tk.encode("Hi", false);
        assert_eq!(ids.len(), 1);
    }

    #[test]
    fn eos_is_a_stop_token() {
        let tk = spm_tokenizer();
        assert!(tk.is_stop_token(tk.eos_id()));
        // "</s>" is in the vocab (id 2 = eos here) — the resolver must not
        // have added anything spurious for absent markers like <|im_end|>.
        assert!(!tk.is_stop_token(tk.bos_id()));
        assert!(!tk.is_stop_token(9999));
    }

    #[test]
    fn qwen_style_stop_markers_resolve_from_vocab() {
        // Byte-level vocab carrying both qwen stop tokens; eos metadata
        // points at <|endoftext|> but <|im_end|> must stop generation too.
        let (byte_to_char, _) = gpt2_byte_map();
        let mut tokens: Vec<String> = vec!["<|endoftext|>".into(), "<|im_end|>".into()];
        for b in 0..=255usize {
            tokens.push(byte_to_char[b].to_string());
        }
        let mut specials = HashSet::new();
        specials.insert(0);
        specials.insert(1);
        let tk = Tokenizer::build(
            tokens,
            Vec::new(),
            Vec::new(),
            specials,
            Style::ByteLevel,
            0,
            0,
            None,
        );
        assert!(tk.is_stop_token(0), "<|endoftext|> must stop");
        assert!(tk.is_stop_token(1), "<|im_end|> must stop");
        assert!(!tk.is_stop_token(2));
        assert_eq!(tk.stop_token_ids().len(), 2);
    }

    #[test]
    fn decode_skips_specials() {
        let tk = spm_tokenizer();
        let mut ids = vec![tk.bos_id()];
        ids.extend(tk.encode("Hello", false));
        ids.push(tk.eos_id());
        assert_eq!(tk.decode(&ids, true), "Hello");
    }
}
