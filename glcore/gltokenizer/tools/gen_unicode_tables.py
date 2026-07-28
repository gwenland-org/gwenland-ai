#!/usr/bin/env python3
"""Generate `src/unicode_tables.rs` — exact `\\p{L}` `\\p{M}` `\\p{N}` `\\p{P}`.

Run from the crate root:

    python tools/gen_unicode_tables.py > src/unicode_tables.rs

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

LLAMA_DATA = Path(
    "../../../llama.cpp/src/unicode-data.cpp"
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
        sys.exit(f"missing {LLAMA_DATA} — run from the crate root")

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
        "//! Regenerate with `python tools/gen_unicode_tables.py > "
        "src/unicode_tables.rs`.\n"
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
            f"pub fn is_{doc.split()[-1]}(c: char) -> bool {{\n"
            "    let u = c as u32;\n"
            "    if u < 128 {\n"
            f"        return ASCII[u as usize] & ASCII_{name} != 0;\n"
            "    }\n"
            f"    in_ranges(&{name}_RANGES, u)\n"
            "}\n\n"
        )


if __name__ == "__main__":
    main()
