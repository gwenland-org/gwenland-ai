//! Pre-tokenizer: splits raw text into the chunks BPE runs on independently.
//!
//! # Why hand-written instead of a regex engine
//!
//! The byte-level families in the wild use a small, closed set of patterns
//! that differ in one parameter. Written out:
//!
//! ```text
//! GPT-2     's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
//! Qwen2     (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
//! cl100k    (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
//! ```
//!
//! Qwen2 and cl100k differ **only** in the digit run: `\p{N}` versus
//! `\p{N}{1,3}`. Llama-3 uses the cl100k pattern. So the family collapses to
//! two shapes plus one integer, which a scanner expresses directly.
//!
//! Pulling in a regex engine to interpret three fixed patterns would add a
//! heavy dependency to a deliberately lean crate, and `regex` cannot express
//! the `\s+(?!\S)` lookahead these patterns rely on anyway.
//!
//! Patterns outside this family are **refused at load time**, never
//! approximated — a mis-split silently changes token ids.

/// Which pre-tokenizer shape a vocabulary uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreTok {
    /// No pre-tokenization: the whole string is one chunk (SPM path).
    None,
    /// A byte-level BPE splitter, parameterised by the three axes that
    /// actually vary between the patterns in the wild (see [`BpeSplit`]).
    Bpe(BpeSplit),
    /// Split on newline runs only: `[^\n]+|[\n]+`.
    ///
    /// The Gemma-4 shape. There is no word-level splitting at all — merges run
    /// across whole lines, which is why `▁` has to mark spaces (they are
    /// ordinary characters to the merge engine). The newline split exists so
    /// no symbol can ever span a line break.
    Lines,
}

/// The three independent axes the real patterns differ on.
///
/// Enumerating named families instead would need a variant per model and
/// still not say *what* differs; these three fields say exactly that, and the
/// reference vectors decide each one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BpeSplit {
    /// Case-insensitive contractions, a `[^\r\n\p{L}\p{N}]?` lead
    /// before letter runs, a `[\r\n]*` tail on punctuation runs, and a
    /// `\s*[\r\n]+` arm. GPT-2-era patterns have none of these;
    /// cl100k-era patterns have all of them.
    pub modern: bool,
    /// Maximum digits kept in one chunk: 1 (Qwen2), 3 (cl100k/Llama-3), or
    /// `usize::MAX` (GPT-2 `\p{N}+`).
    pub digit_run: usize,
    /// Whether a digit run may absorb one leading space — GPT-2's
    /// `" ?\p{N}+"`. ⚠️ This is the axis that separates gpt-2/mpt from
    /// starcoder/refact, which are otherwise identical.
    pub space_digit: bool,
}

impl BpeSplit {
    /// GPT-2, MPT: `" ?\p{N}+"`, unbounded runs, case-sensitive.
    pub const GPT2: Self = Self { modern: false, digit_run: usize::MAX, space_digit: true };
    /// StarCoder, Refact: as GPT-2 but digits never take a leading space.
    pub const STARCODER: Self = Self { modern: false, digit_run: usize::MAX, space_digit: false };
    /// cl100k / Llama-3: modern arms, three-digit runs.
    pub const LLAMA3: Self = Self { modern: true, digit_run: 3, space_digit: false };
    /// Qwen2 and friends: modern arms, single digits.
    pub const QWEN2: Self = Self { modern: true, digit_run: 1, space_digit: false };
}

/// `\p{N}` — Unicode general category N (Nd, Nl, No).
///
/// `char::is_numeric` is defined as exactly this category, so it is an exact
/// match rather than an approximation.
#[inline]
fn is_num(c: char) -> bool {
    c.is_numeric()
}

/// `\p{L}` — Unicode general category L (Lu, Ll, Lt, Lm, Lo).
///
/// ⚠️ KNOWN IMPRECISION. `char::is_alphabetic` is the `Alphabetic` *property*,
/// which is `L` plus category `Nl` plus `Other_Alphabetic`. It is therefore a
/// strict superset of `\p{L}`:
///
/// * `Nl` — letter-like numerals, e.g. Roman numerals `Ⅷ` (U+2167).
/// * `Other_Alphabetic` — mostly combining marks used by Indic, Hebrew and
///   Arabic scripts, e.g. Devanagari vowel signs.
///
/// For Latin, CJK, Cyrillic, Greek and Hangul text the two agree exactly, so
/// this affects a narrow band of scripts. `tests::alphabetic_divergence_is_pinned`
/// pins the difference so it stays a recorded decision rather than an
/// accident, and issue-tracked work will replace this with an exact table.
#[inline]
fn is_letter(c: char) -> bool {
    c.is_alphabetic()
}

/// `\s` — Unicode whitespace.
#[inline]
fn is_space(c: char) -> bool {
    c.is_whitespace()
}

/// Everything that is neither whitespace, letter, nor number.
#[inline]
fn is_punct(c: char) -> bool {
    !is_space(c) && !is_letter(c) && !is_num(c)
}

/// The contraction suffixes both patterns special-case.
const CONTRACTIONS: [&str; 7] = ["s", "t", "re", "ve", "m", "ll", "d"];

impl PreTok {
    /// Split `text`, calling `out` with each chunk in order.
    ///
    /// Chunks are borrowed from `text`; nothing is allocated.
    pub fn split<'a>(&self, text: &'a str, mut out: impl FnMut(&'a str)) {
        match self {
            PreTok::None => {
                if !text.is_empty() {
                    out(text);
                }
            }
            PreTok::Bpe(s) => scan(text, *s, &mut out),
            PreTok::Lines => {
                let b = text.as_bytes();
                let mut i = 0;
                while i < b.len() {
                    let nl = b[i] == b'\n';
                    let start = i;
                    while i < b.len() && (b[i] == b'\n') == nl {
                        i += 1;
                    }
                    out(&text[start..i]);
                }
            }
        }
    }

}

fn scan<'a>(text: &'a str, sp: BpeSplit, out: &mut impl FnMut(&'a str)) {
    let (modern, digit_run, space_digit) = (sp.modern, sp.digit_run, sp.space_digit);
    {
        let b = text.as_bytes();
        let mut i = 0usize;

        while i < b.len() {
            let start = i;

            // ── contractions: 's 't 're 've 'm 'll 'd ──────────────────────
            // Modern patterns match these case-insensitively.
            if b[i] == b'\'' {
                if let Some(len) = match_contraction(text, i, modern) {
                    i += len;
                    out(&text[start..i]);
                    continue;
                }
            }

            let c = char_at(text, i);

            // ── letters, with one optional leading non-letter/non-digit ────
            // GPT-2:  " ?\p{L}+"          (only a space may lead)
            // modern: "[^\r\n\p{L}\p{N}]?\p{L}+"  (any non-newline non-alnum)
            let lead_ok = if modern {
                !matches!(c, '\r' | '\n') && !is_letter(c) && !is_num(c)
            } else {
                c == ' '
            };
            if lead_ok {
                let j = i + c.len_utf8();
                if j < b.len() && is_letter(char_at(text, j)) {
                    i = j;
                    i = take_while(text, i, is_letter);
                    out(&text[start..i]);
                    continue;
                }
            }
            if is_letter(c) {
                i = take_while(text, i, is_letter);
                out(&text[start..i]);
                continue;
            }

            // ── digits ────────────────────────────────────────────────────
            // ⚠️ GPT-2 spells this " ?\p{N}+" — a digit run may absorb one
            // leading space. The modern patterns spell it bare (`\p{N}` /
            // `\p{N}{1,3}`) with no leading space, so this arm is GPT-2 only.
            // Missing it turns " 4" into " " + "4", which changes ids.
            if space_digit && c == ' ' {
                let j = i + 1;
                if j < b.len() && is_num(char_at(text, j)) {
                    i = take_while(text, j, is_num);
                    out(&text[start..i]);
                    continue;
                }
            }
            if is_num(c) {
                let mut count = 0;
                while i < b.len() && count < digit_run {
                    let d = char_at(text, i);
                    if !is_num(d) {
                        break;
                    }
                    i += d.len_utf8();
                    count += 1;
                }
                out(&text[start..i]);
                continue;
            }

            // ── punctuation run, optionally with a trailing newline run ────
            // GPT-2:  " ?[^\s\p{L}\p{N}]+"
            // modern: " ?[^\s\p{L}\p{N}]+[\r\n]*"
            let punct_lead = c == ' ' && {
                let j = i + 1;
                j < b.len() && is_punct(char_at(text, j))
            };
            if punct_lead || is_punct(c) {
                if punct_lead {
                    i += 1;
                }
                i = take_while(text, i, is_punct);
                if modern {
                    i = take_while(text, i, |c| c == '\r' || c == '\n');
                }
                out(&text[start..i]);
                continue;
            }

            // ── whitespace ────────────────────────────────────────────────
            if is_space(c) {
                // modern: "\s*[\r\n]+" — whitespace ending in newlines is one
                // chunk. Try it before the generic rules.
                if modern {
                    let ws_end = take_while(text, i, is_space);
                    let nl_start = last_run_start(text, i, ws_end, |c| c == '\r' || c == '\n');
                    if let Some(ns) = nl_start {
                        if ns == i {
                            // The whole run is newlines (possibly preceded by
                            // other whitespace already consumed as `\s*`).
                            i = ws_end;
                            out(&text[start..i]);
                            continue;
                        }
                        // "\s*[\r\n]+": emit up to and including the newlines.
                        i = ws_end;
                        out(&text[start..i]);
                        continue;
                    }
                }

                // "\s+(?!\S)" — a whitespace run that is NOT followed by a
                // non-space keeps all of it; otherwise the final space is left
                // to lead the next word. This lookahead is why a plain regex
                // crate cannot express these patterns.
                let end = take_while(text, i, is_space);
                if end == b.len() {
                    i = end;
                } else {
                    // Leave exactly one trailing space for the next chunk.
                    let last = prev_char_start(text, end);
                    i = if last > start { last } else { end };
                }
                out(&text[start..i]);
                continue;
            }

            // Unreachable in practice; advance to guarantee termination.
            i += c.len_utf8();
            out(&text[start..i]);
        }
    }
}

#[inline]
fn char_at(s: &str, i: usize) -> char {
    s[i..].chars().next().expect("index on char boundary")
}

#[inline]
fn take_while(s: &str, mut i: usize, f: impl Fn(char) -> bool) -> usize {
    while i < s.len() {
        let c = char_at(s, i);
        if !f(c) {
            break;
        }
        i += c.len_utf8();
    }
    i
}

#[inline]
fn prev_char_start(s: &str, mut i: usize) -> usize {
    i -= 1;
    while !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

/// Start of the trailing run in `[from, to)` matching `f`, if the run is
/// non-empty and reaches `to`.
fn last_run_start(s: &str, from: usize, to: usize, f: impl Fn(char) -> bool) -> Option<usize> {
    if to == from {
        return None;
    }
    let mut i = to;
    let mut found = false;
    while i > from {
        let p = prev_char_start(s, i);
        if !f(char_at(s, p)) {
            break;
        }
        i = p;
        found = true;
    }
    found.then_some(i)
}

/// Length of the contraction at `i` (including the apostrophe), if any.
fn match_contraction(s: &str, i: usize, ci: bool) -> Option<usize> {
    let rest = &s[i + 1..];
    for c in CONTRACTIONS {
        let hit = if ci {
            rest.len() >= c.len() && rest[..c.len()].eq_ignore_ascii_case(c)
        } else {
            rest.starts_with(c)
        };
        if hit {
            return Some(1 + c.len());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn go(p: PreTok, s: &str) -> Vec<String> {
        let mut v = Vec::new();
        p.split(s, |c| v.push(c.to_string()));
        v
    }

    const QWEN: PreTok = PreTok::Bpe(BpeSplit::QWEN2);
    const CL100K: PreTok = PreTok::Bpe(BpeSplit::LLAMA3);

    /// Gemma-4 cuts at newline runs and nowhere else — a whole sentence stays
    /// one chunk, which is what lets its merges cross word boundaries.
    #[test]
    fn lines_split_on_newline_runs_only() {
        assert_eq!(go(PreTok::Lines, "a b\nc"), ["a b", "\n", "c"]);
        assert_eq!(go(PreTok::Lines, "a\n\n\nb"), ["a", "\n\n\n", "b"]);
        assert_eq!(go(PreTok::Lines, "\n"), ["\n"]);
        assert_eq!(go(PreTok::Lines, "no newlines at all"), ["no newlines at all"]);
        assert_eq!(go(PreTok::Lines, ""), Vec::<String>::new());
        // A carriage return is NOT a split point: the pattern is `[\n]+`.
        assert_eq!(go(PreTok::Lines, "a\r\nb"), ["a\r", "\n", "b"]);
    }

    #[test]
    fn none_is_identity() {
        assert_eq!(go(PreTok::None, "a b"), ["a b"]);
        assert_eq!(go(PreTok::None, ""), Vec::<String>::new());
    }

    #[test]
    fn words_carry_a_leading_space() {
        assert_eq!(go(PreTok::Bpe(BpeSplit::GPT2), "Hello World"), ["Hello", " World"]);
        assert_eq!(go(QWEN, "Hello World"), ["Hello", " World"]);
    }

    /// The bug the old implementation had: it split only on ASCII space, so
    /// contractions stayed glued to their stem.
    #[test]
    fn contractions_split() {
        assert_eq!(go(PreTok::Bpe(BpeSplit::GPT2), "don't"), ["don", "'t"]);
        assert_eq!(go(QWEN, "don't"), ["don", "'t"]);
    }

    #[test]
    fn contractions_are_case_insensitive_only_in_modern() {
        assert_eq!(go(QWEN, "DON'T"), ["DON", "'T"]);
        // GPT-2's pattern is case-sensitive: 'T is punctuation + letter.
        assert_eq!(go(PreTok::Bpe(BpeSplit::GPT2), "DON'T"), ["DON", "'", "T"]);
    }

    /// Qwen2 keeps digits single-wide; cl100k groups up to three.
    #[test]
    fn digit_runs_differ_by_family() {
        assert_eq!(go(QWEN, "12345"), ["1", "2", "3", "4", "5"]);
        assert_eq!(go(CL100K, "12345"), ["123", "45"]);
        assert_eq!(go(PreTok::Bpe(BpeSplit::GPT2), "12345"), ["12345"]);
    }

    /// Newlines were not a split point at all in the old implementation.
    #[test]
    fn newlines_split() {
        assert_eq!(go(QWEN, "a\nb"), ["a", "\n", "b"]);
        assert_eq!(go(QWEN, "a\n\nb"), ["a", "\n\n", "b"]);
    }

    #[test]
    fn punctuation_runs_group() {
        assert_eq!(go(QWEN, "wow!!!"), ["wow", "!!!"]);
        assert_eq!(go(QWEN, "a ... b"), ["a", " ...", " b"]);
    }

    /// `\s+(?!\S)`: a run of spaces before a word leaves exactly one to lead
    /// the word; a trailing run at end-of-string is kept whole.
    #[test]
    fn whitespace_lookahead() {
        assert_eq!(go(QWEN, "a   b"), ["a", "  ", " b"]);
        assert_eq!(go(QWEN, "a   "), ["a", "   "]);
    }

    #[test]
    fn cjk_is_letters() {
        assert_eq!(go(QWEN, "日本語 text"), ["日本語", " text"]);
    }

    #[test]
    fn emoji_is_punctuation() {
        // Emoji are neither \p{L} nor \p{N}, so they join the punctuation arm.
        assert_eq!(go(QWEN, "hi 👋"), ["hi", " 👋"]);
    }

    #[test]
    fn every_chunk_is_reachable_and_lossless() {
        for p in [PreTok::Bpe(BpeSplit::GPT2), QWEN, CL100K] {
            for s in [
                "Hello, World! 123",
                "  leading",
                "trailing  ",
                "don't stop\n\nnow",
                "日本語とEnglish 42",
                "",
                "\n",
                "   ",
                "a\t\tb",
            ] {
                let parts = go(p, s);
                assert_eq!(parts.concat(), s, "lossy split of {s:?} under {p:?}");
            }
        }
    }

    /// ⚠️ Pins the documented `is_letter` imprecision (see its doc comment) so
    /// the divergence from `\p{L}` is a recorded decision, not an accident.
    /// If an exact `\p{L}` table lands, this test is the one that must change.
    #[test]
    fn alphabetic_divergence_is_pinned() {
        // U+2167 ROMAN NUMERAL EIGHT is category Nl: alphabetic, not \p{L}.
        assert!(is_letter('\u{2167}'), "Nl currently counts as a letter");
        assert!(is_num('\u{2167}'), "and is also \\p{{N}} — genuinely ambiguous");
        // U+093E DEVANAGARI VOWEL SIGN AA is Mc + Other_Alphabetic.
        assert!(is_letter('\u{093E}'), "Other_Alphabetic currently counts as a letter");
    }
}
