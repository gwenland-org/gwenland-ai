//! GPT-2 byte-to-unicode mapping (ARTX-OQ3 Wave 2).
//!
//! GGUF vocab strings store each token as a sequence of these mapped
//! Unicode chars, not raw bytes — this is GPT-2's `bytes_to_unicode`
//! trick so that byte values which aren't valid standalone UTF-8 (or are
//! whitespace/control chars that would confuse a plain string vocab) still
//! round-trip through a JSON/text-safe vocabulary. Printable bytes map to
//! themselves; the other 94 bytes are shifted into the `U+0100+` range.
//!
//! The mapping is a pure function of the byte value, so it's built once
//! into a compile-time-sized table and looked up in O(1) both ways.

use std::sync::LazyLock;

/// True for byte values GPT-2 treats as "already printable" — these map
/// to themselves. Matches the reference `bytes_to_unicode` ranges exactly:
/// `!`..`~`, then the Latin-1 supplement printable ranges.
fn is_printable(b: u8) -> bool {
    (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b)
}

/// `byte_to_char[b]` = the Unicode char GPT-2's byte-level BPE uses to
/// represent raw byte `b`. Built once, shared for the process lifetime.
static BYTE_TO_CHAR: LazyLock<[char; 256]> = LazyLock::new(|| {
    let mut table = ['\0'; 256];
    let mut shift: u32 = 0;
    for b in 0..=255u16 {
        let c = if is_printable(b as u8) {
            char::from_u32(b as u32).expect("printable byte range is valid Unicode")
        } else {
            let c = char::from_u32(256 + shift).expect("0x100.. is valid Unicode");
            shift += 1;
            c
        };
        table[b as usize] = c;
    }
    table
});

/// Reverse of [`BYTE_TO_CHAR`]: maps a mapped char back to its source byte.
static CHAR_TO_BYTE: LazyLock<std::collections::HashMap<char, u8>> = LazyLock::new(|| {
    BYTE_TO_CHAR
        .iter()
        .enumerate()
        .map(|(b, &c)| (c, b as u8))
        .collect()
});

/// Map a raw byte to its GPT-2 byte-level-BPE Unicode representation.
pub fn byte_to_char(b: u8) -> char {
    BYTE_TO_CHAR[b as usize]
}

/// Map a GPT-2 byte-level-BPE Unicode char back to its source byte.
/// `None` if `c` was never produced by [`byte_to_char`].
pub fn char_to_byte(c: char) -> Option<u8> {
    CHAR_TO_BYTE.get(&c).copied()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_map_is_bijective() {
        for b in 0..=255u16 {
            let b = b as u8;
            let c = byte_to_char(b);
            assert_eq!(char_to_byte(c), Some(b), "byte {b} did not round-trip");
        }
    }

    #[test]
    fn byte_map_printable_ascii_maps_to_self() {
        for b in b'!'..=b'~' {
            assert_eq!(byte_to_char(b), b as char, "byte {b} should map to itself");
        }
    }

    #[test]
    fn byte_map_control_bytes_are_shifted() {
        // Space (0x20) and NUL (0x00) are not in the printable ranges, so
        // they must NOT map to themselves — they land at U+0100+.
        assert_ne!(byte_to_char(0x20), 0x20 as char);
        assert_ne!(byte_to_char(0x00), 0x00 as char);
        assert!(byte_to_char(0x20) as u32 >= 0x100);
        assert!(byte_to_char(0x00) as u32 >= 0x100);
    }

    #[test]
    fn char_to_byte_rejects_unmapped_char() {
        assert_eq!(char_to_byte('\u{10FFFF}'), None);
    }
}
