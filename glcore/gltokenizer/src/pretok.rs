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
//!
//! # Relation to Peek2 (arXiv 2601.05833)
//!
//! Liu Zai's *Peek2: A Regex-free implementation of pretokenizers for
//! Byte-level BPE* (2026-01-09) reaches the same conclusion independently: a
//! hand-written scanner reproduces cl100k-family presegmentation **exactly**,
//! in stable `O(n)`, and is worth ~1.11× end-to-end on byte-level BPE.
//!
//! ⚠️ Worth being precise about what that endorses and what it does not. This
//! module was regex-free and single-pass before the paper was read; that part
//! is convergent design, not an implementation of it. What the paper's framing
//! *did* change is the character-class layer: category tests now come from
//! precomputed [`crate::unicode_tables`] — an ASCII bitmap plus range binary
//! search — instead of `char::is_alphabetic`, which is the `Alphabetic`
//! property and a strict superset of `\p{L}`. That swap is a **precision**
//! fix, not a speed one; the 1.11× in the paper is measured against a regex
//! baseline this crate never had.

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

/// Extra passes some patterns wrap around the main arm.
///
/// ⚠️ These are **pipeline stages, not alternatives**. llama.cpp applies a
/// pattern's expressions in sequence, each one refining the pieces the
/// previous produced — so an earlier stage changes what a later stage *sees*,
/// not merely where the extra cuts land.
///
/// That distinction is the whole reason falcon could not be approximated by
/// any single arm. On `"\n ="`, cutting `=` out first leaves `"\n "` at the end
/// of its segment, so the main arm's `\s+(?!\S)` lookahead now succeeds and
/// keeps the whole whitespace run; one-pass scanning splits it. Measured:
/// `[1212, 40]` versus `[193, 204, 40]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Passes {
    /// Before the main arm: cut runs of `[\p{P}\$\+<=>\^~\|]`.
    pub punct_runs: bool,
    /// …with a backtick also in that class. falcon has one, `default` does not.
    pub punct_runs_backtick: bool,
    /// After the main arm: cut `\p{N}+` runs (`default` only).
    pub number_runs: bool,
    /// After the main arm: cut `[0-9][0-9][0-9]` groups. ASCII digits only —
    /// the pattern is spelled `[0-9]`, not `\p{N}`.
    pub digit_triples: bool,
}

impl Passes {
    /// The common case: one expression, no wrapping stages.
    pub const NONE: Self = Self {
        punct_runs: false,
        punct_runs_backtick: false,
        number_runs: false,
        digit_triples: false,
    };
}

/// The axes the real patterns differ on.
///
/// Enumerating named families instead would need a variant per model and
/// still not say *what* differs; these fields say exactly that, and the
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
    /// Qwen-3.5 widens the letter arm from `\p{L}` to `[\p{L}\p{M}]`, and
    /// correspondingly drops `\p{M}` from the punctuation arm. Combining marks
    /// therefore attach to the word they modify instead of splitting it.
    pub marks_are_letters: bool,
    /// Stages wrapped around the main arm; see [`Passes`].
    pub passes: Passes,
}

impl BpeSplit {
    /// GPT-2, MPT: `" ?\p{N}+"`, unbounded runs, case-sensitive.
    pub const GPT2: Self = Self {
        modern: false,
        digit_run: usize::MAX,
        space_digit: true,
        marks_are_letters: false,
        passes: Passes::NONE,
    };
    /// StarCoder, Refact: as GPT-2 but digits never take a leading space.
    pub const STARCODER: Self = Self { space_digit: false, ..Self::GPT2 };
    /// cl100k / Llama-3: modern arms, three-digit runs.
    pub const LLAMA3: Self = Self {
        modern: true,
        digit_run: 3,
        space_digit: false,
        marks_are_letters: false,
        passes: Passes::NONE,
    };
    /// Qwen2 and friends: modern arms, single digits.
    pub const QWEN2: Self = Self { digit_run: 1, ..Self::LLAMA3 };
    /// Qwen-3.5: Qwen2 with combining marks folded into the letter class.
    pub const QWEN35: Self = Self { marks_are_letters: true, ..Self::QWEN2 };
    /// Falcon: GPT-2's arm wrapped in a punctuation-run pass and a
    /// three-digit pass.
    pub const FALCON: Self = Self {
        passes: Passes {
            punct_runs: true,
            punct_runs_backtick: true,
            number_runs: false,
            digit_triples: true,
        },
        ..Self::GPT2
    };
    /// llama.cpp's `default` fallback arm — falcon's shape without the
    /// backtick and with an extra `\p{N}+` pass.
    ///
    /// ⚠️ Reached only by a GGUF missing `tokenizer.ggml.pre`, which llama.cpp
    /// warns will degrade generation quality. Supported so such a model loads
    /// *identically to llama.cpp* rather than not at all — but see
    /// `gguf::pretok_from_name`, which still refuses it by default.
    pub const DEFAULT: Self = Self {
        passes: Passes {
            punct_runs: true,
            punct_runs_backtick: false,
            number_runs: true,
            digit_triples: true,
        },
        ..Self::GPT2
    };
}

use crate::unicode_tables::{is_letter, is_mark, is_number, is_punctuation};

/// `\s` — Unicode whitespace.
///
/// `char::is_whitespace` is the `White_Space` property, which is what `\s`
/// means in the regex flavours these patterns were written for.
#[inline]
fn is_space(c: char) -> bool {
    c.is_whitespace()
}

/// The pattern's letter class: `\p{L}`, or `[\p{L}\p{M}]` under qwen35.
#[inline]
fn is_word(sp: BpeSplit, c: char) -> bool {
    is_letter(c) || (sp.marks_are_letters && is_mark(c))
}

/// `[^\s\p{L}\p{N}]` — the patterns' punctuation arm, which is a *complement*
/// class and therefore also catches symbols, emoji and (outside qwen35)
/// combining marks. Not to be confused with [`is_falcon_punct`].
#[inline]
fn is_punct(sp: BpeSplit, c: char) -> bool {
    !is_space(c) && !is_word(sp, c) && !is_number(c)
}

/// `[\p{P}\$\+<=>\^~\|]` plus a backtick for falcon.
///
/// ⚠️ Genuinely `\p{P}`, not the complement class above: emoji and most
/// symbols are excluded. Getting this wrong is invisible — it just moves the
/// segment boundaries, and therefore the ids.
#[inline]
fn is_falcon_punct(c: char, backtick: bool) -> bool {
    is_punctuation(c)
        || matches!(c, '$' | '+' | '<' | '=' | '>' | '^' | '~' | '|')
        || (backtick && c == '`')
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
            PreTok::Bpe(s) => run_pipeline(text, *s, &mut out),
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

/// Apply the pattern's stages in order: optional punctuation-run pass, the
/// main arm, then the optional numeric passes.
///
/// ⚠️ The main arm runs **per segment**, and that is load-bearing rather than
/// an implementation convenience: `\s+(?!\S)` looks ahead only within the
/// segment it is scanning, so where the earlier pass cut determines whether a
/// trailing whitespace run is kept whole or has its last space split off.
fn run_pipeline<'a>(text: &'a str, sp: BpeSplit, out: &mut impl FnMut(&'a str)) {
    let after = &mut |chunk: &'a str| numeric_passes(chunk, sp, out);
    if sp.passes.punct_runs {
        split_runs(text, |c| is_falcon_punct(c, sp.passes.punct_runs_backtick), &mut |seg| {
            scan(seg, sp, after)
        });
    } else {
        scan(text, sp, after);
    }
}

/// The trailing `\p{N}+` and `[0-9][0-9][0-9]` passes.
fn numeric_passes<'a>(chunk: &'a str, sp: BpeSplit, out: &mut impl FnMut(&'a str)) {
    let triples = &mut |s: &'a str| {
        if sp.passes.digit_triples {
            split_digit_triples(s, out);
        } else {
            out(s);
        }
    };
    if sp.passes.number_runs {
        split_runs(chunk, is_number, triples);
    } else {
        triples(chunk);
    }
}

/// Split `text` into alternating runs that do and do not satisfy `f`,
/// preserving order and losing nothing.
///
/// This is what "apply a regex as a splitting stage" means when the regex is a
/// single character-class `+`: matched spans become their own pieces, and the
/// gaps between them become pieces too.
fn split_runs<'a>(text: &'a str, f: impl Fn(char) -> bool, out: &mut impl FnMut(&'a str)) {
    let mut i = 0usize;
    while i < text.len() {
        let start = i;
        let matching = f(char_at(text, i));
        while i < text.len() && f(char_at(text, i)) == matching {
            i += char_at(text, i).len_utf8();
        }
        out(&text[start..i]);
    }
}

/// `[0-9][0-9][0-9]` as a splitting stage: greedily cut successive groups of
/// exactly three ASCII digits, leaving the surrounding text as its own pieces.
///
/// ⚠️ Not `\p{N}` — the expression is spelled with an ASCII range, so `½` and
/// Devanagari digits do not participate.
fn split_digit_triples<'a>(text: &'a str, out: &mut impl FnMut(&'a str)) {
    let b = text.as_bytes();
    let ascii_digit = |k: usize| k < b.len() && b[k].is_ascii_digit();
    let (mut i, mut gap) = (0usize, 0usize);
    while i < b.len() {
        if ascii_digit(i) && ascii_digit(i + 1) && ascii_digit(i + 2) {
            if gap < i {
                out(&text[gap..i]);
            }
            out(&text[i..i + 3]);
            i += 3;
            gap = i;
        } else {
            i += 1;
        }
    }
    if gap < b.len() {
        out(&text[gap..]);
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
            // ⚠️ The lead class stays `[^\r\n\p{L}\p{N}]` even under qwen35 —
            // only the *run* widens to `[\p{L}\p{M}]`. A mark can therefore
            // serve as either, which the regex resolves by backtracking and
            // this scanner resolves by falling through to the run arm.
            let lead_ok = if modern {
                !matches!(c, '\r' | '\n') && !is_letter(c) && !is_number(c)
            } else {
                c == ' '
            };
            if lead_ok {
                let j = i + c.len_utf8();
                if j < b.len() && is_word(sp, char_at(text, j)) {
                    i = j;
                    i = take_while(text, i, |c| is_word(sp, c));
                    out(&text[start..i]);
                    continue;
                }
            }
            if is_word(sp, c) {
                i = take_while(text, i, |c| is_word(sp, c));
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
                if j < b.len() && is_number(char_at(text, j)) {
                    i = take_while(text, j, is_number);
                    out(&text[start..i]);
                    continue;
                }
            }
            if is_number(c) {
                let mut count = 0;
                while i < b.len() && count < digit_run {
                    let d = char_at(text, i);
                    if !is_number(d) {
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
                j < b.len() && is_punct(sp, char_at(text, j))
            };
            if punct_lead || is_punct(sp, c) {
                if punct_lead {
                    i += 1;
                }
                i = take_while(text, i, |c| is_punct(sp, c));
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

    /// ⭐ The approximation this crate used to run on is gone. `is_letter` was
    /// `char::is_alphabetic` — the `Alphabetic` *property*, a strict superset
    /// of `\p{L}` that also contains `Nl` and `Other_Alphabetic`. These two
    /// codepoints are exactly where the two disagree, and the old code got
    /// both wrong.
    #[test]
    fn letter_class_is_now_exactly_p_l() {
        // U+2167 ROMAN NUMERAL EIGHT is category Nl: `Alphabetic`, but \p{N}.
        assert!(!is_letter('\u{2167}'), "Nl is a number, not \\p{{L}}");
        assert!(is_number('\u{2167}'));
        // U+093E DEVANAGARI VOWEL SIGN AA is Mc + Other_Alphabetic.
        assert!(!is_letter('\u{093E}'), "Mc is a mark, not \\p{{L}}");
        assert!(is_mark('\u{093E}'));
        // U+0301 COMBINING ACUTE is Mn and NOT Other_Alphabetic — the case the
        // old superset missed in the opposite direction.
        assert!(!is_letter('\u{0301}'));
        assert!(is_mark('\u{0301}'));
    }

    /// `\p{P}` is a real category, not "everything that is not a letter,
    /// number or space" — the complement class the other arms use.
    #[test]
    fn punctuation_category_excludes_symbols() {
        for c in ['.', ',', '!', '?', '-', '(', '"', '\u{2014}'] {
            assert!(is_punctuation(c), "{c:?} is \\p{{P}}");
        }
        // Symbols (Sm/Sc/Sk/So) are NOT \p{P}. falcon lists the ASCII ones it
        // wants explicitly, which is why they need `is_falcon_punct`.
        for c in ['$', '+', '<', '=', '>', '^', '~', '|', '`', '\u{1F600}'] {
            assert!(!is_punctuation(c), "{c:?} is \\p{{S}}, not \\p{{P}}");
        }
        for c in ['$', '+', '<', '=', '>', '^', '~', '|', '`'] {
            assert!(is_falcon_punct(c, true), "{c:?} is in falcon's class");
        }
        assert!(!is_falcon_punct('`', false), "`default` omits the backtick");
        assert!(!is_falcon_punct('\u{1F600}', true), "emoji stay out");
    }

    /// ⭐ qwen35's one difference from qwen2, now exact rather than inherited
    /// from an over-broad letter class.
    #[test]
    fn marks_attach_to_words_only_under_qwen35() {
        const QWEN35: PreTok = PreTok::Bpe(BpeSplit::QWEN35);

        // ⚠️ ONE mark is a weak discriminator: qwen2's lead class
        // `[^\r\n\p{L}\p{N}]?` already admits a mark, so it attaches to the
        // *following* word rather than splitting off. The piece boundaries
        // still differ, but only by where the mark lands.
        let one = "e\u{0301}t";
        assert_eq!(go(QWEN35, one), [one], "qwen35: one word");
        assert_eq!(go(QWEN, one), ["e", "\u{0301}t"], "qwen2: mark leads `t`");

        // TWO marks separate them cleanly: qwen2's lead is a single optional
        // character, so the second mark has nowhere to go but the punctuation
        // arm. qwen35's `[\p{L}\p{M}]+` run absorbs both.
        let two = "e\u{0301}\u{0301}t";
        assert_eq!(go(QWEN35, two), [two], "qwen35: still one word");
        assert_eq!(
            go(QWEN, two),
            ["e", "\u{0301}\u{0301}", "t"],
            "qwen2: marks become a punctuation run and split the word"
        );
    }

    /// falcon's stages, isolated. The `"\n ="` case is the one no single arm
    /// could reproduce: cutting `=` first changes what `\s+(?!\S)` sees.
    #[test]
    fn falcon_pipeline_stages() {
        const FALCON: PreTok = PreTok::Bpe(BpeSplit::FALCON);
        assert_eq!(go(FALCON, "\n ="), ["\n ", "="]);
        assert_eq!(go(PreTok::Bpe(BpeSplit::GPT2), "\n ="), ["\n", " ="]);
        // Three-digit pass, after the GPT-2 arm has taken " 1234567" whole.
        assert_eq!(go(FALCON, " 1234567"), [" ", "123", "456", "7"]);
        // Emoji are not in falcon's punctuation class, so they are not cut out
        // by the first stage.
        assert_eq!(go(FALCON, "hi \u{1F44B}"), ["hi", " \u{1F44B}"]);
    }

    /// ⭐ The property a regex engine cannot promise.
    ///
    /// `\s+(?!\S)` and the alternations around it are exactly the shape that
    /// makes a backtracking NFA go quadratic — the input below is built to
    /// trigger that: long uniform runs where every arm nearly matches, so an
    /// engine must try and abandon each in turn at every position.
    ///
    /// This scanner cannot backtrack: `scan` advances `i` monotonically and
    /// never revisits a byte, so the work is bounded by the input length by
    /// construction. Asserting it here rather than inferring it from timings
    /// keeps the guarantee out of reach of measurement noise — see
    /// `examples/tokbench.rs`, whose numbers this machine cannot resolve.
    #[test]
    fn linear_on_pathological_input() {
        for p in [
            PreTok::Bpe(BpeSplit::GPT2),
            QWEN,
            CL100K,
            PreTok::Bpe(BpeSplit::QWEN35),
            PreTok::Bpe(BpeSplit::FALCON),
            PreTok::Bpe(BpeSplit::DEFAULT),
        ] {
            for seed in [" ", " \n", "a ", "1", "!", " a1! ", "\u{0301}"] {
                for n in [1usize, 64, 4096] {
                    let s = seed.repeat(n);
                    let parts = go(p, &s);
                    // Losslessness is the invariant that actually matters: a
                    // scanner that terminates but drops or duplicates bytes is
                    // worse than one that hangs.
                    assert_eq!(parts.concat(), s, "lossy for {seed:?}x{n} under {p:?}");
                    assert!(
                        parts.len() <= s.len(),
                        "more pieces than bytes for {seed:?}x{n} under {p:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn digit_triples_are_ascii_only() {
        let mut v = Vec::new();
        // Devanagari digits are \p{N} but not [0-9].
        split_digit_triples("\u{0967}\u{0968}\u{0969}", &mut |s| v.push(s.to_string()));
        assert_eq!(v, ["\u{0967}\u{0968}\u{0969}"]);
    }
}
