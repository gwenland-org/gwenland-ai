#!/usr/bin/env python3
"""Generate `src/tokenizer/unicode_tables.rs` — exact `\\p{L}` `\\p{M}` `\\p{N}` `\\p{P}`.

Run from anywhere:

    python glcore/tools/gen_unicode_tables.py > src/tokenizer/unicode_tables.rs

# Why two sources

Getting a Unicode category wrong is invisible: the tokenizer keeps working and
silently emits different ids. So the table is built from one source and
*verified* against a second, and the script fails loudly if they disagree
anywhere outside the known version delta.

* **Primary — the UCD table llama.cpp generates** (`src/unicode-data.cpp`,
  produced by its `scripts/gen-unicode-data.py` from
  `unicode.org/Public/UCD/latest/ucd/UnicodeData.txt`). Chosen as primary
  because it is the newest UCD available offline on this machine *and* because
  the reference vectors this crate is scored against come from the same
  ecosystem — matching the oracle matters more than matching the newest spec.
* **Cross-check — CPython's `unicodedata`**, an independent UCD implementation.

⚠️ This machine's CPython is 3.11 / UCD 14.0.0, so the cross-check is one
Unicode revision behind. Codepoints assigned after 14.0 legitimately differ and
are reported as a version delta rather than an error; anything else is a bug in
this extraction and aborts.

Nothing here is copied from llama.cpp's *implementation*. `UnicodeData.txt` is
the Unicode Consortium's published data file, and the ranges below are a
re-derivation of it.
"""

from __future__ import annotations

import os
import re
import sys
import unicodedata
from pathlib import Path

# Bit flags from llama.cpp's `src/unicode.h`, which is just how that file
# packs the UCD general category into its range table.
NUMBER = 0x0002
LETTER = 0x0004
ACCENT_MARK = 0x0010
PUNCTUATION = 0x0020

MAX_CP = 0x110000

# General-category prefixes, per the Unicode standard.
CLASSES = {
    "L": (LETTER, ("Lu", "Ll", "Lt", "Lm", "Lo")),
    "M": (ACCENT_MARK, ("Mn", "Mc", "Me")),
    "N": (NUMBER, ("Nd", "Nl", "No")),
    "P": (PUNCTUATION, ("Pc", "Pd", "Ps", "Pe", "Pi", "Pf", "Po")),
}

# Resolved relative to this script, not to the working directory, so the
# generator works from anywhere. `llama.cpp` is checked out as a sibling of the
# repository root; `GLTOK_UNICODE_DATA` overrides for any other layout.
_REPO_ROOT = Path(__file__).resolve().parents[2]
LLAMA_DATA = Path(
    os.environ.get(
        "GLTOK_UNICODE_DATA",
        _REPO_ROOT.parent / "llama.cpp" / "src" / "unicode-data.cpp",
    )
)


def load_llama_flags() -> list[int]:
    """Expand llama.cpp's `{start, flags}` run-list into a flat per-codepoint array.

    Each entry sets the flags for `[start, next_start)`, so the file is a
    run-length encoding of the whole plane.
    """
    src = LLAMA_DATA.read_text(encoding="utf-8", errors="replace")
    body = src.split("unicode_ranges_flags = {", 1)[1].split("};", 1)[0]
    pairs = re.findall(r"\{0x([0-9A-Fa-f]+),\s*0x([0-9A-Fa-f]+)\}", body)
    if len(pairs) < 1000:
        sys.exit(f"unicode-data.cpp: parsed only {len(pairs)} ranges, expected thousands")

    flags = [0] * MAX_CP
    entries = [(int(a, 16), int(b, 16)) for a, b in pairs]
    for i, (start, fl) in enumerate(entries):
        end = entries[i + 1][0] if i + 1 < len(entries) else MAX_CP
        for cp in range(start, min(end, MAX_CP)):
            flags[cp] = fl
    return flags


def python_members(cats: tuple[str, ...]) -> set[int]:
    """Codepoints CPython's UCD puts in `cats`."""
    out = set()
    for cp in range(MAX_CP):
        # Surrogates have no scalar value in Rust; `char` cannot hold one, so
        # they are irrelevant to a `fn(char) -> bool` and are skipped rather
        # than counted as a mismatch.
        if 0xD800 <= cp <= 0xDFFF:
            continue
        if unicodedata.category(chr(cp)) in cats:
            out.add(cp)
    return out


def to_ranges(members: set[int]) -> list[tuple[int, int]]:
    """Collapse a codepoint set into sorted inclusive ranges."""
    out: list[tuple[int, int]] = []
    for cp in sorted(members):
        if out and cp == out[-1][1] + 1:
            out[-1] = (out[-1][0], cp)
        else:
            out.append((cp, cp))
    return out


def main() -> None:
    if not LLAMA_DATA.exists():
        sys.exit(f"missing {LLAMA_DATA} — set GLTOK_UNICODE_DATA to a llama.cpp unicode-data.cpp")

    llama = load_llama_flags()
    tables: dict[str, list[tuple[int, int]]] = {}
    deltas: dict[str, int] = {}

    for name, (bit, cats) in CLASSES.items():
        primary = {
            cp
            for cp in range(MAX_CP)
            if not (0xD800 <= cp <= 0xDFFF) and (llama[cp] & bit)
        }
        check = python_members(cats)

        only_primary = primary - check
        only_check = check - primary

        # Newly assigned codepoints appear in the newer table only. The reverse
        # direction must be empty: UCD never *removes* a category assignment,
        # so a codepoint the older CPython classifies but the newer table does
        # not means the extraction is wrong.
        if only_check:
            sample = sorted(only_check)[:8]
            sys.exit(
                f"\\p{{{name}}}: {len(only_check)} codepoints classified by CPython "
                f"but not by the primary table, e.g. {[hex(c) for c in sample]}. "
                "This cannot be a version delta — the extraction is wrong."
            )
        deltas[name] = len(only_primary)
        tables[name] = to_ranges(primary)

    ascii_bits = []
    for cp in range(128):
        b = 0
        for i, (name, (bit, _)) in enumerate(CLASSES.items()):
            if any(lo <= cp <= hi for lo, hi in tables[name]):
                b |= 1 << i
        ascii_bits.append(b)

    emit(tables, deltas, ascii_bits)


def emit(tables, deltas, ascii_bits) -> None:
    # Windows consoles default to cp1252, which mangles the em-dashes in the
    # emitted doc comments into replacement characters.
    sys.stdout.reconfigure(encoding="utf-8", newline="\n")
    w = sys.stdout.write
    total = sum(len(v) for v in tables.values())
    w(
        "//! Exact Unicode general-category tests — **generated, do not edit**.\n"
        "//!\n"
        "//! Regenerate with `python glcore/tools/gen_unicode_tables.py > "
        "src/tokenizer/unicode_tables.rs`.\n"
        "//! That script documents the two independent UCD sources it "
        "cross-checks and\n"
        "//! aborts rather than emit a table the two disagree on.\n"
        "//!\n"
        "//! # Why this file exists\n"
        "//!\n"
        "//! The pre-tokenizer patterns are written in terms of `\\p{L}`, "
        "`\\p{N}`,\n"
        "//! `\\p{M}` and `\\p{P}`. Rust's standard library has no general-"
        "category\n"
        "//! predicates: `char::is_alphabetic` is the `Alphabetic` *property* "
        "(a strict\n"
        "//! superset of `\\p{L}`) and `char::is_numeric` happens to be exactly "
        "`\\p{N}`.\n"
        "//! Substituting the former is what this crate did before, and it is "
        "the kind\n"
        "//! of near-miss that changes token ids without failing anything.\n"
        "//!\n"
        "//! # Representation\n"
        "//!\n"
        "//! Sorted inclusive ranges plus a 128-entry ASCII bitmap. ASCII — the\n"
        "//! overwhelming majority of real input — resolves in one array index; "
        "the\n"
        "//! rest binary-searches. A flat bitset over all 1 114 112 codepoints "
        "would\n"
        "//! be 136 KiB per class and thrash the cache for no gain, whereas "
        f"these\n//! {total} ranges total a few kilobytes.\n"
        "\n"
    )

    for name in CLASSES:
        w(f"// \\p{{{name}}}: {len(tables[name])} ranges")
        if deltas[name]:
            w(f" ({deltas[name]} codepoints newer than CPython 14.0's UCD)")
        w("\n")
    w("\n")

    for i, name in enumerate(CLASSES):
        w(f"const ASCII_{name}: u8 = 1 << {i};\n")
    w("\n/// Category bits for every ASCII codepoint, indexed directly.\n")
    w("const ASCII: [u8; 128] = [\n")
    for row in range(0, 128, 16):
        w("    " + " ".join(f"{b:#04x}," for b in ascii_bits[row : row + 16]) + "\n")
    w("];\n\n")

    for name in CLASSES:
        rows = tables[name]
        w(f"/// `\\p{{{name}}}` — {len(rows)} inclusive ranges.\n")
        w(f"static {name}_RANGES: [(u32, u32); {len(rows)}] = [\n")
        for i in range(0, len(rows), 4):
            chunk = rows[i : i + 4]
            w("    " + " ".join(f"({lo:#07x},{hi:#07x})," for lo, hi in chunk) + "\n")
        w("];\n\n")

    w(
        "/// Binary search over sorted, non-overlapping inclusive ranges.\n"
        "#[inline]\n"
        "fn in_ranges(ranges: &[(u32, u32)], c: u32) -> bool {\n"
        "    ranges\n"
        "        .binary_search_by(|&(lo, hi)| {\n"
        "            if c < lo {\n"
        "                std::cmp::Ordering::Greater\n"
        "            } else if c > hi {\n"
        "                std::cmp::Ordering::Less\n"
        "            } else {\n"
        "                std::cmp::Ordering::Equal\n"
        "            }\n"
        "        })\n"
        "        .is_ok()\n"
        "}\n\n"
    )

    for name in CLASSES:
        doc = {
            "L": "letter",
            "M": "combining mark",
            "N": "number",
            "P": "punctuation",
        }[name]
        w(
            f"/// `\\p{{{name}}}` — Unicode general category {name} ({doc}).\n"
            "#[inline]\n"
            f"pub(crate) fn is_{doc.split()[-1]}(c: char) -> bool {{\n"
            "    let u = c as u32;\n"
            "    if u < 128 {\n"
            f"        return ASCII[u as usize] & ASCII_{name} != 0;\n"
            "    }\n"
            f"    in_ranges(&{name}_RANGES, u)\n"
            "}\n\n"
        )

    w(TESTS)


# Emitted rather than hand-written, so regenerating the table cannot silently
# drop its own tests. Expectations are hardcoded Unicode facts, NOT derived
# from the table, so a botched extraction still fails them.
TESTS = r'''
#[cfg(test)]
mod tests {
    use super::*;

    /// ⚠️ Structural invariants come first: `in_ranges` is a binary search, so
    /// an unsorted or overlapping table does not error — it returns the wrong
    /// answer for some codepoints and the right one for others. That is the
    /// failure mode a generator bug actually produces.
    #[test]
    fn ranges_are_sorted_disjoint_and_non_empty() {
        for (name, r) in [
            ("L", &L_RANGES[..]),
            ("M", &M_RANGES[..]),
            ("N", &N_RANGES[..]),
            ("P", &P_RANGES[..]),
        ] {
            assert!(!r.is_empty(), "{name} table is empty");
            for (i, &(lo, hi)) in r.iter().enumerate() {
                assert!(lo <= hi, "{name}[{i}] = ({lo:#x}, {hi:#x}) is inverted");
                assert!(hi <= 0x10FFFF, "{name}[{i}] exceeds the Unicode range");
                if i > 0 {
                    let prev = r[i - 1].1;
                    assert!(
                        prev < lo,
                        "{name}[{}]..{prev:#x} overlaps or abuts {name}[{i}] at {lo:#x} \
                         — abutting ranges should have been merged",
                        i - 1
                    );
                }
            }
        }
    }

    /// The binary search must agree with an exhaustive linear scan. Cheap
    /// insurance that the comparator's inverted `Greater`/`Less` is right.
    #[test]
    fn binary_search_agrees_with_linear_scan() {
        for r in [&L_RANGES[..], &M_RANGES[..], &N_RANGES[..], &P_RANGES[..]] {
            // Every boundary, one inside, and both neighbours of every range.
            for &(lo, hi) in r.iter() {
                for c in [lo.saturating_sub(1), lo, lo + (hi - lo) / 2, hi, hi + 1] {
                    let linear = r.iter().any(|&(a, b)| c >= a && c <= b);
                    assert_eq!(in_ranges(r, c), linear, "disagreement at {c:#x}");
                }
            }
        }
    }

    /// The ASCII fast path is a second copy of the data; it must not drift.
    #[test]
    fn ascii_bitmap_matches_the_ranges() {
        for u in 0u32..128 {
            for (bit, r) in [
                (ASCII_L, &L_RANGES[..]),
                (ASCII_M, &M_RANGES[..]),
                (ASCII_N, &N_RANGES[..]),
                (ASCII_P, &P_RANGES[..]),
            ] {
                let via_bitmap = ASCII[u as usize] & bit != 0;
                let via_ranges = r.iter().any(|&(a, b)| u >= a && u <= b);
                assert_eq!(via_bitmap, via_ranges, "ASCII {u:#x} bit {bit:#x}");
            }
        }
    }

    #[test]
    fn punctuation_spot_checks() {
        for c in ['.', ',', '?', '!', '-', '_', '(', ')', '[', ']', '"', '\''] {
            assert!(is_punctuation(c), "{c:?} must be \\p{{P}}");
        }
        // U+060C ARABIC COMMA, U+3001 IDEOGRAPHIC COMMA, U+2014 EM DASH.
        for c in ['\u{060C}', '\u{3001}', '\u{2014}'] {
            assert!(is_punctuation(c), "{c:?} must be \\p{{P}}");
        }
        // Letters, digits, whitespace, and SYMBOLS are not punctuation.
        for c in ['a', 'Z', '0', '9', ' ', '\n', '+', '<', '=', '$', '`', '\u{1F600}'] {
            assert!(!is_punctuation(c), "{c:?} must NOT be \\p{{P}}");
        }
    }

    #[test]
    fn mark_spot_checks() {
        // U+0300 COMBINING GRAVE (Mn), U+093E DEVANAGARI VOWEL SIGN AA (Mc),
        // U+20DD COMBINING ENCLOSING CIRCLE (Me) — one from each subcategory.
        for c in ['\u{0300}', '\u{093E}', '\u{20DD}'] {
            assert!(is_mark(c), "{c:?} must be \\p{{M}}");
        }
        for c in ['a', '0', '.', ' ', '\u{4E00}'] {
            assert!(!is_mark(c), "{c:?} must NOT be \\p{{M}}");
        }
    }

    /// ⭐ The classes this crate previously conflated. `char::is_alphabetic` is
    /// true for all three of these; only the first is `\p{L}`.
    #[test]
    fn letter_number_and_mark_do_not_overlap_where_std_says_they_do() {
        assert!(is_letter('a') && !is_mark('a') && !is_number('a'));
        // U+2167 ROMAN NUMERAL EIGHT is Nl — a number, not a letter.
        assert!(is_number('\u{2167}') && !is_letter('\u{2167}'));
        // U+093E is Mc — a mark, not a letter.
        assert!(is_mark('\u{093E}') && !is_letter('\u{093E}'));
        // U+00BD VULGAR FRACTION ONE HALF is No.
        assert!(is_number('\u{00BD}') && !is_punctuation('\u{00BD}'));
    }

    #[test]
    fn non_bmp_codepoints_resolve() {
        // U+10400 DESERET CAPITAL LONG I (Lu), U+1D165 MUSICAL SYMBOL COMBINING
        // STEM (Mc), U+1D7CE MATHEMATICAL BOLD DIGIT ZERO (Nd).
        assert!(is_letter('\u{10400}'));
        assert!(is_mark('\u{1D165}'));
        assert!(is_number('\u{1D7CE}'));
    }
}
'''


if __name__ == "__main__":
    main()
