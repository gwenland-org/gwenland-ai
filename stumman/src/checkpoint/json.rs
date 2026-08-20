//! Stummañ Pik: a minimal, dependency-free JSON reader and writer.
//!
//! Stummañ's dependency budget is four crates (`glcore`, `glproc`, `anyhow`,
//! `thiserror`) and none of them is a serialization library. Two things in this
//! sub-system are JSON: the safetensors header, whose structure is fully
//! specified by the format, and `manifest.json`, whose structure this crate
//! defines. Both are flat and small, so they need a JSON value model rather
//! than a serialization framework.
//!
//! `glbench/src/export/json.rs` solved the same problem under the same
//! constraint and this follows its shape: a `Json` enum, a `BTreeMap` for
//! objects so key order is stable across writes, and a recursive-descent
//! parser. It is deliberately narrower than that one: no pretty-printer, since
//! nothing here is meant to be diffed by eye, and numbers print in the shortest
//! round-trip form Rust gives.
//!
//! Not a general-purpose JSON library. It exists so checkpoints round-trip.

use std::collections::BTreeMap;

/// A JSON value.
#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    /// `null`.
    Null,
    /// `true` / `false`.
    Bool(bool),
    /// All numbers are held as `f64`. Integer-valued numbers print with no
    /// fractional part, which is what safetensors offsets and shapes need.
    Num(f64),
    /// A string.
    Str(String),
    /// An array.
    Arr(Vec<Json>),
    /// An object. `BTreeMap` so serialization is deterministic: a checkpoint
    /// written twice from the same state must produce identical bytes.
    Obj(BTreeMap<String, Json>),
}

impl Json {
    /// A string value.
    pub fn s(v: impl Into<String>) -> Json {
        Json::Str(v.into())
    }

    /// A numeric value.
    pub fn n(v: impl Into<f64>) -> Json {
        Json::Num(v.into())
    }

    /// An array of `usize`, the shape/offset case.
    pub fn usizes(v: impl IntoIterator<Item = usize>) -> Json {
        Json::Arr(v.into_iter().map(|x| Json::Num(x as f64)).collect())
    }

    /// An object from key/value pairs.
    pub fn obj<const N: usize>(pairs: [(&str, Json); N]) -> Json {
        Json::Obj(pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
    }

    /// Borrow as an object.
    pub fn as_obj(&self) -> Option<&BTreeMap<String, Json>> {
        match self {
            Json::Obj(m) => Some(m),
            _ => None,
        }
    }

    /// Borrow as an array.
    pub fn as_arr(&self) -> Option<&[Json]> {
        match self {
            Json::Arr(a) => Some(a),
            _ => None,
        }
    }

    /// Read as `f64`.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Json::Num(n) => Some(*n),
            _ => None,
        }
    }

    /// Read as `usize`. Rejects a negative or fractional number rather than
    /// truncating it, since every `usize` here is a length or an offset.
    pub fn as_usize(&self) -> Option<usize> {
        match self {
            Json::Num(n) if *n >= 0.0 && n.fract() == 0.0 => Some(*n as usize),
            _ => None,
        }
    }

    /// Read as `&str`.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }

    /// Read as `bool`.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Json::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Look up a key, if this is an object.
    pub fn get(&self, key: &str) -> Option<&Json> {
        self.as_obj().and_then(|m| m.get(key))
    }

    /// Serialize compactly, with no insignificant whitespace.
    pub fn to_compact(&self) -> String {
        let mut out = String::new();
        self.write(&mut out);
        out
    }

    fn write(&self, out: &mut String) {
        match self {
            Json::Null => out.push_str("null"),
            Json::Bool(true) => out.push_str("true"),
            Json::Bool(false) => out.push_str("false"),
            Json::Num(n) => write_number(out, *n),
            Json::Str(s) => write_string(out, s),
            Json::Arr(a) => {
                out.push('[');
                for (i, v) in a.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    v.write(out);
                }
                out.push(']');
            }
            Json::Obj(m) => {
                out.push('{');
                for (i, (k, v)) in m.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    write_string(out, k);
                    out.push(':');
                    v.write(out);
                }
                out.push('}');
            }
        }
    }
}

/// Integer-valued numbers print without a fractional part. A safetensors
/// `data_offsets` of `[0.0, 16.0]` would be legal JSON but no other reader
/// expects it, and shapes are counts.
fn write_number(out: &mut String, n: f64) {
    if n.is_finite() && n.fract() == 0.0 && n.abs() < 9.0e15 {
        out.push_str(&format!("{}", n as i64));
    } else if n.is_finite() {
        out.push_str(&format!("{n}"));
    } else {
        // JSON has no NaN or Infinity. Emitting a bare `NaN` would produce a
        // file no parser accepts, so null is the only honest encoding.
        out.push_str("null");
    }
}

fn write_string(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Parse a JSON document. Returns the message a caller can put in an error.
pub fn parse(input: &str) -> Result<Json, String> {
    let mut p = Parser {
        b: input.as_bytes(),
        i: 0,
    };
    p.skip_ws();
    let v = p.value()?;
    p.skip_ws();
    if p.i != p.b.len() {
        return Err(format!("trailing content at byte {}", p.i));
    }
    Ok(v)
}

struct Parser<'a> {
    b: &'a [u8],
    i: usize,
}

impl Parser<'_> {
    fn skip_ws(&mut self) {
        while let Some(c) = self.b.get(self.i) {
            if matches!(c, b' ' | b'\t' | b'\n' | b'\r') {
                self.i += 1;
            } else {
                break;
            }
        }
    }

    fn peek(&self) -> Option<u8> {
        self.b.get(self.i).copied()
    }

    fn eat(&mut self, c: u8) -> Result<(), String> {
        if self.peek() == Some(c) {
            self.i += 1;
            Ok(())
        } else {
            Err(format!(
                "expected '{}' at byte {}",
                c as char, self.i
            ))
        }
    }

    fn literal(&mut self, word: &str, v: Json) -> Result<Json, String> {
        if self.b[self.i..].starts_with(word.as_bytes()) {
            self.i += word.len();
            Ok(v)
        } else {
            Err(format!("invalid literal at byte {}", self.i))
        }
    }

    fn value(&mut self) -> Result<Json, String> {
        match self.peek() {
            Some(b'{') => self.object(),
            Some(b'[') => self.array(),
            Some(b'"') => Ok(Json::Str(self.string()?)),
            Some(b't') => self.literal("true", Json::Bool(true)),
            Some(b'f') => self.literal("false", Json::Bool(false)),
            Some(b'n') => self.literal("null", Json::Null),
            Some(_) => self.number(),
            None => Err("unexpected end of input".into()),
        }
    }

    fn object(&mut self) -> Result<Json, String> {
        self.eat(b'{')?;
        let mut m = BTreeMap::new();
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.i += 1;
            return Ok(Json::Obj(m));
        }
        loop {
            self.skip_ws();
            let k = self.string()?;
            self.skip_ws();
            self.eat(b':')?;
            self.skip_ws();
            let v = self.value()?;
            m.insert(k, v);
            self.skip_ws();
            match self.peek() {
                Some(b',') => self.i += 1,
                Some(b'}') => {
                    self.i += 1;
                    return Ok(Json::Obj(m));
                }
                _ => return Err(format!("expected ',' or '}}' at byte {}", self.i)),
            }
        }
    }

    fn array(&mut self) -> Result<Json, String> {
        self.eat(b'[')?;
        let mut a = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.i += 1;
            return Ok(Json::Arr(a));
        }
        loop {
            self.skip_ws();
            a.push(self.value()?);
            self.skip_ws();
            match self.peek() {
                Some(b',') => self.i += 1,
                Some(b']') => {
                    self.i += 1;
                    return Ok(Json::Arr(a));
                }
                _ => return Err(format!("expected ',' or ']' at byte {}", self.i)),
            }
        }
    }

    fn string(&mut self) -> Result<String, String> {
        self.eat(b'"')?;
        let mut s = String::new();
        loop {
            let c = self
                .peek()
                .ok_or_else(|| "unterminated string".to_string())?;
            self.i += 1;
            match c {
                b'"' => return Ok(s),
                b'\\' => {
                    let e = self
                        .peek()
                        .ok_or_else(|| "unterminated escape".to_string())?;
                    self.i += 1;
                    match e {
                        b'"' => s.push('"'),
                        b'\\' => s.push('\\'),
                        b'/' => s.push('/'),
                        b'b' => s.push('\u{8}'),
                        b'f' => s.push('\u{c}'),
                        b'n' => s.push('\n'),
                        b'r' => s.push('\r'),
                        b't' => s.push('\t'),
                        b'u' => {
                            let hex = self
                                .b
                                .get(self.i..self.i + 4)
                                .ok_or_else(|| "truncated \\u escape".to_string())?;
                            let code = u32::from_str_radix(
                                std::str::from_utf8(hex).map_err(|_| "bad \\u escape")?,
                                16,
                            )
                            .map_err(|_| "bad \\u escape".to_string())?;
                            self.i += 4;
                            // Lone surrogates cannot appear in a Rust char.
                            // Nothing this crate writes produces one, so
                            // rejecting beats silently substituting U+FFFD.
                            s.push(
                                char::from_u32(code)
                                    .ok_or_else(|| format!("\\u{code:04x} is not a character"))?,
                            );
                        }
                        other => return Err(format!("unknown escape '\\{}'", other as char)),
                    }
                }
                // Multi-byte UTF-8 arrives here one byte at a time, so collect
                // the continuation bytes rather than treating each as a char.
                c if c < 0x80 => s.push(c as char),
                _ => {
                    let start = self.i - 1;
                    while self.peek().is_some_and(|c| c & 0xC0 == 0x80) {
                        self.i += 1;
                    }
                    s.push_str(
                        std::str::from_utf8(&self.b[start..self.i])
                            .map_err(|_| "invalid UTF-8 in string".to_string())?,
                    );
                }
            }
        }
    }

    fn number(&mut self) -> Result<Json, String> {
        let start = self.i;
        if self.peek() == Some(b'-') {
            self.i += 1;
        }
        while self
            .peek()
            .is_some_and(|c| c.is_ascii_digit() || matches!(c, b'.' | b'e' | b'E' | b'+' | b'-'))
        {
            self.i += 1;
        }
        std::str::from_utf8(&self.b[start..self.i])
            .ok()
            .and_then(|s| s.parse::<f64>().ok())
            .map(Json::Num)
            .ok_or_else(|| format!("invalid number at byte {start}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integers_print_without_a_fractional_part() {
        // A safetensors reader expects [0, 16], not [0.0, 16.0].
        assert_eq!(Json::usizes([0usize, 16]).to_compact(), "[0,16]");
        assert_eq!(Json::n(2.5).to_compact(), "2.5");
    }

    #[test]
    fn object_keys_serialize_in_a_stable_order() {
        // Two checkpoints written from the same state must be byte-identical.
        let a = Json::obj([("z", Json::n(1)), ("a", Json::n(2))]);
        let b = Json::obj([("a", Json::n(2)), ("z", Json::n(1))]);
        assert_eq!(a.to_compact(), b.to_compact());
        assert_eq!(a.to_compact(), r#"{"a":2,"z":1}"#);
    }

    #[test]
    fn strings_with_quotes_and_control_characters_round_trip() {
        let awkward = "a\"b\\c\nd\te\u{1}f";
        let text = Json::s(awkward).to_compact();
        assert_eq!(parse(&text).unwrap().as_str().unwrap(), awkward);
    }

    #[test]
    fn non_ascii_strings_round_trip() {
        // The crate is named "Stummañ", so this is not a hypothetical.
        for s in ["Stummañ", "Kevskrid — Gwellaer", "日本語"] {
            let text = Json::s(s).to_compact();
            assert_eq!(parse(&text).unwrap().as_str().unwrap(), s, "for {s:?}");
        }
    }

    #[test]
    fn a_nested_document_round_trips() {
        let doc = Json::obj([
            ("name", Json::s("lora_a")),
            ("shape", Json::usizes([4usize, 2])),
            ("rslora", Json::Bool(false)),
            ("alpha", Json::n(8.0)),
            ("base", Json::Null),
        ]);
        let back = parse(&doc.to_compact()).unwrap();
        assert_eq!(back, doc);
        assert_eq!(back.get("shape").unwrap().as_arr().unwrap().len(), 2);
        assert_eq!(back.get("rslora").unwrap().as_bool(), Some(false));
    }

    #[test]
    fn as_usize_rejects_a_negative_or_fractional_number() {
        assert_eq!(Json::n(4.0).as_usize(), Some(4));
        assert_eq!(Json::n(-1.0).as_usize(), None);
        assert_eq!(Json::n(1.5).as_usize(), None);
    }

    #[test]
    fn malformed_documents_are_rejected_rather_than_guessed_at() {
        for bad in [
            "{",
            "{\"a\":}",
            "[1,]",
            "{\"a\":1}{\"b\":2}",
            "\"unterminated",
            "tru",
            "",
        ] {
            assert!(parse(bad).is_err(), "parsed malformed input {bad:?}");
        }
    }

    #[test]
    fn whitespace_between_tokens_is_ignored() {
        let v = parse("  {\n  \"a\" : [ 1 , 2 ]\t}  ").unwrap();
        assert_eq!(v.get("a").unwrap().as_arr().unwrap().len(), 2);
    }

    /// The header of a real safetensors file, as the format specifies it.
    #[test]
    fn a_safetensors_header_round_trips() {
        let header = r#"{"w":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]},"__metadata__":{"format":"pt"}}"#;
        let v = parse(header).unwrap();
        let w = v.get("w").unwrap();
        assert_eq!(w.get("dtype").unwrap().as_str(), Some("F32"));
        assert_eq!(w.get("shape").unwrap().as_arr().unwrap()[1].as_usize(), Some(2));
        assert_eq!(
            w.get("data_offsets").unwrap().as_arr().unwrap()[1].as_usize(),
            Some(16)
        );
        // Re-serialization is semantically exact, not byte-exact: object keys
        // come back sorted because `Json::Obj` is a BTreeMap, which is what
        // makes a checkpoint written twice byte-identical. Asserting byte
        // equality here would be asserting against that guarantee.
        assert_eq!(parse(&v.to_compact()).unwrap(), v);
        assert!(
            v.to_compact().starts_with(r#"{"__metadata__""#),
            "keys must be normalized to sorted order: {}",
            v.to_compact()
        );
    }
}
