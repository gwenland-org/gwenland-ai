//! Qwen2 pre-tokenizer (ARTX-OQ3 Wave 4).
//!
//! Splits raw text into chunks BPE merges never cross — contractions,
//! runs of letters, short runs of digits, punctuation runs, and
//! whitespace each become their own chunk. This is the exact pattern
//! Qwen2/Qwen3 GGUF models declare via `tokenizer.ggml.pre = "qwen2"`
//! (sourced from llama.cpp's `unicode.cpp` regex set for that pre-tokenizer
//! id), not a general-purpose word splitter.
//!
//! Special-token strings (`<|im_start|>` etc.) are never seen by this
//! splitter — the caller ([`crate::tokenizer::GllmTokenizer::encode`])
//! extracts those via [`crate::tokenizer::special::SpecialTokens`] first.
//!
//! ## No lookaround in the `regex` crate
//!
//! The reference pattern's `\s+(?!\S)` alternative — "a whitespace run
//! not immediately followed by non-whitespace" — needs lookahead, which
//! Rust's `regex` crate deliberately does not support (it guarantees
//! linear-time matching; llama.cpp's own `unicode.cpp` sidesteps the same
//! limitation with a hand-rolled scanner instead of a regex engine at
//! all). We drop that alternative from the pattern — the final `\s+`
//! catch-all already matches everything it would — and instead fix up
//! the *effect* of the lookahead in a post-processing pass: whenever a
//! whitespace-only match is longer than one char and more text follows,
//! its last character is handed to the start of the next chunk instead
//! (mirroring llama.cpp's `pos += num_whitespaces - 1`). This produces
//! byte-identical splits to the lookahead pattern without needing one.

use regex::Regex;
use std::sync::LazyLock;

const QWEN2_PATTERN: &str =
    r"(?:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+";

static PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(QWEN2_PATTERN).expect("hardcoded Qwen2 pattern must compile"));

/// Splits text into BPE-ready chunks using the Qwen2 pre-tokenizer regex.
#[derive(Debug, Default)]
pub struct Qwen2PreTokenizer;

impl Qwen2PreTokenizer {
    pub fn new() -> Self {
        Self
    }

    /// Split `text` into pre-tokenized chunks, in order, covering the
    /// entire input (the pattern's final `\s+` alternative guarantees no
    /// character is ever unmatched).
    pub fn split<'a>(&self, text: &'a str) -> Vec<&'a str> {
        let raw: Vec<(usize, usize)> = PATTERN
            .find_iter(text)
            .map(|m| (m.start(), m.end()))
            .collect();
        fixup_trailing_whitespace(text, raw)
            .into_iter()
            .map(|(s, e)| &text[s..e])
            .collect()
    }
}

/// Applies the `\s+(?!\S)` fixup described in the module docs: a
/// whitespace-only span longer than one char, immediately followed by
/// another span, gives up its last char to the start of that next span
/// (matches llama.cpp's `pos += num_whitespaces - 1` on the same rule).
/// Spans are consumed strictly left to right and each input span is
/// folded into the output exactly once, so no "already handled" lookback
/// is needed.
fn fixup_trailing_whitespace(text: &str, spans: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    let mut out: Vec<(usize, usize)> = Vec::with_capacity(spans.len());
    let mut i = 0;
    while i < spans.len() {
        let (start, end) = spans[i];
        let has_next = i + 1 < spans.len();
        let is_whitespace_only = end > start && text[start..end].chars().all(char::is_whitespace);
        let last_char_start = text[start..end]
            .char_indices()
            .last()
            .map(|(off, _)| start + off);

        if is_whitespace_only && has_next && last_char_start.is_some_and(|lcs| lcs > start) {
            let last_char_start = last_char_start.unwrap();
            out.push((start, last_char_start));
            let (_, next_end) = spans[i + 1];
            out.push((last_char_start, next_end));
            i += 2;
        } else {
            out.push((start, end));
            i += 1;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pretok() -> Qwen2PreTokenizer {
        Qwen2PreTokenizer::new()
    }

    #[test]
    fn pretok_splits_ascii_words() {
        assert_eq!(pretok().split("Hello world"), vec!["Hello", " world"]);
    }

    #[test]
    fn pretok_splits_numbers() {
        assert_eq!(pretok().split("abc123def"), vec!["abc", "123", "def"]);
    }

    #[test]
    fn pretok_handles_contractions() {
        assert_eq!(pretok().split("don't"), vec!["don", "'t"]);
    }

    #[test]
    fn pretok_handles_cjk() {
        // \p{L}+ greedily consumes runs of letters including CJK — with no
        // whitespace between the two, they stay one chunk (matches
        // reference GPT-2/Qwen2 regex behavior; word-level CJK segmentation
        // is not this layer's job).
        let chunks = pretok().split("你好");
        assert_eq!(chunks, vec!["你好"]);
    }

    #[test]
    fn pretok_handles_mixed_whitespace() {
        let chunks = pretok().split("a  b\n\nc");
        assert_eq!(chunks.concat(), "a  b\n\nc");
        assert!(chunks.iter().all(|c| !c.is_empty()));
    }

    #[test]
    fn pretok_double_space_keeps_one_space_as_own_chunk_and_donates_the_other() {
        // This is the exact case the dropped `\s+(?!\S)` lookahead existed
        // for: of the two spaces between "a" and "b", the first stands
        // alone (nothing left to attach to) and the second attaches as
        // "b"'s leading-space prefix — never two separate one-space chunks
        // and never a fused two-space chunk.
        assert_eq!(pretok().split("a  b"), vec!["a", " ", " b"]);
    }

    #[test]
    fn pretok_trailing_whitespace_run_has_nothing_to_donate_to() {
        // No span follows the whitespace run, so the fixup must not fire —
        // all trailing whitespace stays in one chunk.
        assert_eq!(pretok().split("a   "), vec!["a", "   "]);
    }

    #[test]
    fn pretok_single_space_between_words_is_never_split() {
        // Single-space runs are length 1, so `end > start` on the
        // *remaining* portion after donating a char would be empty — must
        // stay attached as the next word's leading-space prefix, not
        // fragment into a zero-length chunk.
        assert_eq!(pretok().split("a b"), vec!["a", " b"]);
    }

    #[test]
    fn pretok_empty_string() {
        assert_eq!(pretok().split(""), Vec::<&str>::new());
    }

    #[test]
    fn pretok_splits_numbers_over_three_digits_into_groups() {
        // \p{N}{1,3} caps each numeric chunk at 3 digits.
        assert_eq!(pretok().split("1234"), vec!["123", "4"]);
    }

    #[test]
    fn pretok_covers_entire_input_with_no_gaps() {
        let text = "Mixed: Hello, world! 123 你好 don't\tstop.";
        let chunks = pretok().split(text);
        assert_eq!(chunks.concat(), text);
    }
}
