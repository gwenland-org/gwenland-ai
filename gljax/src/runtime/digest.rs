//! SHA-256, for compile-cache keys.
//!
//! ⚠️ **This is a cache key, not a security primitive.** It exists to answer
//! "have I compiled this exact module for this exact plugin before"; nothing
//! downstream trusts it to resist an adversary.
//!
//! # Where the hash actually lives
//!
//! Both functions below are thin wrappers over [`glcore::hash`]; the algorithm
//! itself is not implemented here.
//!
//! This file used to carry its own ~90-line SHA-256, written because ARTX01
//! §5.4 caps gljax's dependency list at `libloading` + `log` + `glcore` and the
//! only other implementation in the workspace was behind the `sha2` crate. Its
//! own module docs recorded the exit: *"if a dependency is ever acceptable
//! here, delete this file and call the audited one."*
//!
//! That is now what happened, without a dependency. glbench v3 D-15 moved a
//! std-only SHA-256 into `glcore` — which gljax already depends on
//! unconditionally — so the second implementation had no reason left to exist.
//! Net implementations of SHA-256 in the workspace: one.
//!
//! The published-vector tests stay here rather than moving with the code. They
//! now cover the delegation, which is the thing this file is still responsible
//! for: `glcore` proves the algorithm, these prove gljax reaches it.

/// SHA-256 of `data`.
pub fn sha256(data: &[u8]) -> [u8; 32] {
    glcore::hash::sha256(data)
}

/// Lowercase hex, the form cache filenames use.
pub fn hex(digest: &[u8; 32]) -> String {
    glcore::hash::hex(digest)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The published NIST/FIPS-180-4 vectors. A hand-written digest that is not
    /// tested against them is just a hash-shaped function — and after the move
    /// to `glcore`, these also prove the delegation reaches the right one.
    #[test]
    fn matches_the_published_sha256_test_vectors() {
        let cases: [(&[u8], &str); 4] = [
            (
                b"",
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            ),
            (
                b"abc",
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            ),
            (
                b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
                "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
            ),
            (
                b"abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu",
                "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1",
            ),
        ];
        for (input, expected) in cases {
            assert_eq!(hex(&sha256(input)), expected, "input len {}", input.len());
        }
    }

    /// The block-boundary cases are where padding bugs live: exactly 55, 56,
    /// 63, 64 and 119 bytes each exercise a different padding branch.
    #[test]
    fn handles_every_padding_boundary() {
        for len in [54usize, 55, 56, 57, 63, 64, 65, 118, 119, 120] {
            let data = vec![b'a'; len];
            let digest = sha256(&data);
            assert_eq!(digest.len(), 32);
            // A million 'a's is the fourth published vector; the boundary
            // lengths just have to not panic and to differ from each other.
            assert_ne!(digest, [0u8; 32], "len {len}");
        }
        // Distinct inputs, distinct digests — the property a cache key needs.
        assert_ne!(sha256(&[b'a'; 55]), sha256(&[b'a'; 56]));
        assert_ne!(sha256(&[b'a'; 63]), sha256(&[b'a'; 64]));
    }

    #[test]
    fn one_million_a_vector() {
        // FIPS-180-4's long vector — catches multi-block accumulation errors.
        let data = vec![b'a'; 1_000_000];
        assert_eq!(
            hex(&sha256(&data)),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
        );
    }
}
