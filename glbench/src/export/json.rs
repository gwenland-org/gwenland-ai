//! A minimal, dependency-free JSON writer and reader.
//!
//! glbench forbids external serialization crates (DESIGN.md), so this is a
//! hand-rolled JSON value model just large enough to serialize a
//! [`crate::core::session::BenchmarkSession`] and read it back for comparison.
//! It is deliberately small: object/array/string/number/bool/null, with a
//! pretty-printer and a recursive-descent parser. Not a general-purpose JSON
//! library — it exists so archives round-trip.

use std::collections::BTreeMap;
use std::fmt::Write as _;

/// A JSON value. Objects use a `BTreeMap` so key order is stable across writes
/// — deterministic output matters for diffing archived sessions.
#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    /// All numbers are held as f64; integer-valued numbers print without a
    /// fractional part.
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(BTreeMap<String, Json>),
}

impl Json {
    /// Build an object from key/value pairs.
    pub fn obj<const N: usize>(pairs: [(&str, Json); N]) -> Json {
        let mut m = BTreeMap::new();
        for (k, v) in pairs {
            m.insert(k.to_string(), v);
        }
        Json::Obj(m)
    }

    /// Convenience constructor for a string value.
    pub fn s(v: impl Into<String>) -> Json {
        Json::Str(v.into())
    }

    /// Convenience constructor for a numeric value.
    pub fn n(v: impl Into<f64>) -> Json {
        Json::Num(v.into())
    }

    /// Borrow this value as an object map, if it is one.
    pub fn as_obj(&self) -> Option<&BTreeMap<String, Json>> {
        match self {
            Json::Obj(m) => Some(m),
            _ => None,
        }
    }

    /// Borrow this value as an array, if it is one.
    pub fn as_arr(&self) -> Option<&[Json]> {
        match self {
            Json::Arr(a) => Some(a),
            _ => None,
        }
    }

    /// Read this value as an f64, if it is a number.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Json::Num(n) => Some(*n),
            _ => None,
        }
    }

    /// Read this value as a string slice, if it is a string.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }

    /// Read this value as a bool, if it is one.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Json::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Look up a key in an object value.
    pub fn get(&self, key: &str) -> Option<&Json> {
        self.as_obj().and_then(|m| m.get(key))
    }

    /// Serialize to a pretty-printed JSON string (2-space indent).
    pub fn to_pretty(&self) -> String {
        let mut out = String::new();
        self.write_pretty(&mut out, 0);
        out.push('\n');
        out
    }

    fn write_pretty(&self, out: &mut String, indent: usize) {
        match self {
            Json::Null => out.push_str("null"),
            Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Json::Num(n) => write_number(out, *n),
            Json::Str(s) => write_json_string(out, s),
            Json::Arr(a) => {
                if a.is_empty() {
                    out.push_str("[]");
                    return;
                }
                out.push_str("[\n");
                for (i, v) in a.iter().enumerate() {
                    pad(out, indent + 1);
                    v.write_pretty(out, indent + 1);
                    if i + 1 < a.len() {
                        out.push(',');
                    }
                    out.push('\n');
                }
                pad(out, indent);
                out.push(']');
            }
            Json::Obj(m) => {
                if m.is_empty() {
                    out.push_str("{}");
                    return;
                }
                out.push_str("{\n");
                let len = m.len();
                for (i, (k, v)) in m.iter().enumerate() {
                    pad(out, indent + 1);
                    write_json_string(out, k);
                    out.push_str(": ");
                    v.write_pretty(out, indent + 1);
                    if i + 1 < len {
                        out.push(',');
                    }
                    out.push('\n');
                }
                pad(out, indent);
                out.push('}');
            }
        }
    }
}

fn pad(out: &mut String, indent: usize) {
    for _ in 0..indent {
        out.push_str("  ");
    }
}

fn write_number(out: &mut String, n: f64) {
    if !n.is_finite() {
        // JSON has no NaN/Infinity; null is the only faithful encoding.
        out.push_str("null");
    } else if n.fract() == 0.0 && n.abs() < 1e15 {
        let _ = write!(out, "{}", n as i64);
    } else {
        let _ = write!(out, "{n}");
    }
}

fn write_json_string(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Parse a JSON string into a [`Json`] value. Returns an error message on
/// malformed input. Accepts the subset this writer produces plus standard
/// whitespace and escapes.
pub fn parse(input: &str) -> Result<Json, String> {
    let mut p = Parser { bytes: input.as_bytes(), pos: 0 };
    p.skip_ws();
    let v = p.parse_value()?;
    p.skip_ws();
    if p.pos != p.bytes.len() {
        return Err(format!("trailing characters at byte {}", p.pos));
    }
    Ok(v)
}

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl Parser<'_> {
    fn skip_ws(&mut self) {
        while self.pos < self.bytes.len() {
            match self.bytes[self.pos] {
                b' ' | b'\t' | b'\n' | b'\r' => self.pos += 1,
                _ => break,
            }
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn parse_value(&mut self) -> Result<Json, String> {
        self.skip_ws();
        match self.peek() {
            Some(b'{') => self.parse_object(),
            Some(b'[') => self.parse_array(),
            Some(b'"') => Ok(Json::Str(self.parse_string()?)),
            Some(b't') | Some(b'f') => self.parse_bool(),
            Some(b'n') => self.parse_null(),
            Some(c) if c == b'-' || c.is_ascii_digit() => self.parse_number(),
            other => Err(format!("unexpected token {other:?} at byte {}", self.pos)),
        }
    }

    fn parse_object(&mut self) -> Result<Json, String> {
        self.pos += 1; // '{'
        let mut m = BTreeMap::new();
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(Json::Obj(m));
        }
        loop {
            self.skip_ws();
            let key = self.parse_string()?;
            self.skip_ws();
            if self.peek() != Some(b':') {
                return Err(format!("expected ':' at byte {}", self.pos));
            }
            self.pos += 1;
            let val = self.parse_value()?;
            m.insert(key, val);
            self.skip_ws();
            match self.peek() {
                Some(b',') => {
                    self.pos += 1;
                }
                Some(b'}') => {
                    self.pos += 1;
                    break;
                }
                _ => return Err(format!("expected ',' or '}}' at byte {}", self.pos)),
            }
        }
        Ok(Json::Obj(m))
    }

    fn parse_array(&mut self) -> Result<Json, String> {
        self.pos += 1; // '['
        let mut a = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(Json::Arr(a));
        }
        loop {
            let val = self.parse_value()?;
            a.push(val);
            self.skip_ws();
            match self.peek() {
                Some(b',') => {
                    self.pos += 1;
                }
                Some(b']') => {
                    self.pos += 1;
                    break;
                }
                _ => return Err(format!("expected ',' or ']' at byte {}", self.pos)),
            }
        }
        Ok(Json::Arr(a))
    }

    fn parse_string(&mut self) -> Result<String, String> {
        if self.peek() != Some(b'"') {
            return Err(format!("expected '\"' at byte {}", self.pos));
        }
        self.pos += 1;
        let mut s = String::new();
        while let Some(c) = self.peek() {
            self.pos += 1;
            match c {
                b'"' => return Ok(s),
                b'\\' => {
                    let esc = self.peek().ok_or("unterminated escape")?;
                    self.pos += 1;
                    match esc {
                        b'"' => s.push('"'),
                        b'\\' => s.push('\\'),
                        b'/' => s.push('/'),
                        b'n' => s.push('\n'),
                        b'r' => s.push('\r'),
                        b't' => s.push('\t'),
                        b'b' => s.push('\u{08}'),
                        b'f' => s.push('\u{0c}'),
                        b'u' => {
                            let hex = self
                                .bytes
                                .get(self.pos..self.pos + 4)
                                .ok_or("truncated \\u escape")?;
                            let code = u32::from_str_radix(
                                std::str::from_utf8(hex).map_err(|_| "bad \\u hex")?,
                                16,
                            )
                            .map_err(|_| "bad \\u hex")?;
                            self.pos += 4;
                            s.push(char::from_u32(code).unwrap_or('\u{fffd}'));
                        }
                        other => return Err(format!("bad escape \\{}", other as char)),
                    }
                }
                // Multi-byte UTF-8: copy the continuation bytes through.
                c if c < 0x80 => s.push(c as char),
                _ => {
                    let start = self.pos - 1;
                    while let Some(&b) = self.bytes.get(self.pos) {
                        if b & 0xC0 == 0x80 {
                            self.pos += 1;
                        } else {
                            break;
                        }
                    }
                    let chunk = std::str::from_utf8(&self.bytes[start..self.pos])
                        .map_err(|_| "invalid utf-8 in string")?;
                    s.push_str(chunk);
                }
            }
        }
        Err("unterminated string".into())
    }

    fn parse_bool(&mut self) -> Result<Json, String> {
        if self.bytes[self.pos..].starts_with(b"true") {
            self.pos += 4;
            Ok(Json::Bool(true))
        } else if self.bytes[self.pos..].starts_with(b"false") {
            self.pos += 5;
            Ok(Json::Bool(false))
        } else {
            Err(format!("invalid literal at byte {}", self.pos))
        }
    }

    fn parse_null(&mut self) -> Result<Json, String> {
        if self.bytes[self.pos..].starts_with(b"null") {
            self.pos += 4;
            Ok(Json::Null)
        } else {
            Err(format!("invalid literal at byte {}", self.pos))
        }
    }

    fn parse_number(&mut self) -> Result<Json, String> {
        let start = self.pos;
        while let Some(c) = self.peek() {
            match c {
                b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E' => self.pos += 1,
                _ => break,
            }
        }
        let text = std::str::from_utf8(&self.bytes[start..self.pos]).map_err(|_| "bad number")?;
        text.parse::<f64>()
            .map(Json::Num)
            .map_err(|_| format!("invalid number '{text}'"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_nested() {
        let v = Json::obj([
            ("name", Json::s("qwen")),
            ("ok", Json::Bool(true)),
            ("count", Json::n(42.0)),
            ("ratio", Json::n(1.27)),
            ("empty", Json::Arr(vec![])),
            (
                "list",
                Json::Arr(vec![Json::n(1.0), Json::n(2.0), Json::s("three")]),
            ),
            ("nothing", Json::Null),
        ]);
        let text = v.to_pretty();
        let back = parse(&text).expect("parse");
        assert_eq!(v, back);
    }

    #[test]
    fn integers_print_without_fraction() {
        assert_eq!(Json::n(29.0).to_pretty().trim(), "29");
        assert_eq!(Json::n(1.5).to_pretty().trim(), "1.5");
    }

    #[test]
    fn escapes_and_unicode() {
        let v = Json::s("line\nbreak \"quote\" τ");
        let round = parse(&v.to_pretty()).unwrap();
        assert_eq!(v, round);
    }

    #[test]
    fn non_finite_becomes_null() {
        assert_eq!(Json::n(f64::NAN).to_pretty().trim(), "null");
        assert_eq!(Json::n(f64::INFINITY).to_pretty().trim(), "null");
    }
}
