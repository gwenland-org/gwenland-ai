//! SHA-256, the workspace's single hash primitive.
//!
//! # Why this lives in `glcore`
//!
//! Two consumers need SHA-256 and neither can depend on the other:
//! `glictus-caliburni` verifies `.gllm` execution units before mmap, and
//! `glbench` stamps a content digest into every archive it writes. Both depend
//! on `glcore`, so this is the one place a shared primitive can sit without
//! inventing a new crate or a cyclic edge.
//!
//! The alternative — a second copy in the second consumer — is the exact
//! failure mode `architecture/gl-stack-audit-2026-07/ARTX2-Quant.md` catalogues:
//! seven independent Q6_K decoders, one of them wrong for months, each looking
//! correct in isolation.
//!
//! # Why hand-written rather than the `sha2` crate
//!
//! `glbench`'s dependency rule is zero external crates (`glbench/DESIGN.md` §9),
//! and its archive integrity cannot be feature-gated — a digest that exists only
//! in some builds is not an integrity guarantee. A `sha2` edge here would reach
//! every crate in the workspace through `glcore`. ~150 lines against the
//! published FIPS-180-4 vectors is the cheaper answer, and the vectors are what
//! make it trustworthy: a hand-written digest that is not tested against them is
//! just a hash-shaped function.
//!
//! # Threat model
//!
//! This is an **accident detector**, not a signature. It catches partial writes,
//! silent corruption, and an archive edited after the fact. Anyone who can edit
//! a file can also recompute its digest; nothing here resists an adversary.

use std::fmt::Write as _;
use std::io::Read;
use std::path::Path;

/// Round constants: the first 32 bits of the fractional parts of the cube roots
/// of the first 64 primes (FIPS-180-4 §4.2.2).
const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

/// Initial hash value: the first 32 bits of the fractional parts of the square
/// roots of the first 8 primes (FIPS-180-4 §5.3.3).
const H0: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

/// Read chunk size for file hashing (64 KiB keeps memory flat on GB files).
const HASH_CHUNK_SIZE: usize = 64 * 1024;

/// Digest length in bytes.
const DIGEST_LEN: usize = 32;

/// How many leading digest bytes [`sha256_128_hex`] keeps. 128 bits gives ~2^64
/// collision resistance under the birthday bound — ample for accident
/// detection, which is the whole threat model above.
const TRUNCATED_LEN: usize = 16;

/// Incremental SHA-256 state.
///
/// Private on purpose: the public surface of this module is three free
/// functions. Streaming exists so [`sha256_file`] can hash a multi-gigabyte
/// `.gllm` layer without materialising it, not because callers need a hasher
/// object.
struct Sha256Ctx {
    /// Running hash state.
    h: [u32; 8],
    /// Partial block not yet compressed.
    buf: [u8; 64],
    /// Bytes currently held in `buf`.
    buf_len: usize,
    /// Total bytes fed in, for the length suffix.
    total_len: u64,
}

impl Sha256Ctx {
    fn new() -> Sha256Ctx {
        Sha256Ctx { h: H0, buf: [0u8; 64], buf_len: 0, total_len: 0 }
    }

    /// Feed bytes in. Compresses every full 64-byte block and keeps the
    /// remainder buffered.
    fn update(&mut self, mut data: &[u8]) {
        self.total_len = self.total_len.wrapping_add(data.len() as u64);

        // Top up a partial block first.
        if self.buf_len > 0 {
            let take = (64 - self.buf_len).min(data.len());
            self.buf[self.buf_len..self.buf_len + take].copy_from_slice(&data[..take]);
            self.buf_len += take;
            data = &data[take..];
            if self.buf_len < 64 {
                // Everything we were handed fits in the partial block. Return
                // here rather than falling through: the tail below rewrites
                // `buf_len` from the leftover of an *empty* slice, which would
                // silently discard the bytes just buffered.
                return;
            }
            let block = self.buf;
            compress(&mut self.h, &block);
            self.buf_len = 0;
        }

        // Then whole blocks straight out of the input.
        let mut chunks = data.chunks_exact(64);
        for block in &mut chunks {
            let mut b = [0u8; 64];
            b.copy_from_slice(block);
            compress(&mut self.h, &b);
        }

        // Whatever is left is shorter than a block; buffer it.
        let rest = chunks.remainder();
        self.buf[..rest.len()].copy_from_slice(rest);
        self.buf_len = rest.len();
    }

    /// Apply the padding and emit the digest.
    fn finalize(mut self) -> [u8; DIGEST_LEN] {
        // FIPS-180-4 §5.1.1: append 0x80, then zeros, then the length in bits
        // as a big-endian u64, so the total is a multiple of 64 bytes.
        let bit_len = self.total_len.wrapping_mul(8);
        self.buf[self.buf_len] = 0x80;
        self.buf_len += 1;

        // The length suffix needs the last 8 bytes; if the 0x80 pushed us past
        // byte 56 they no longer fit, so flush a zero-filled block first. This
        // is the branch every padding bug lives in — see the boundary test.
        if self.buf_len > 56 {
            for byte in self.buf.iter_mut().skip(self.buf_len) {
                *byte = 0;
            }
            let block = self.buf;
            compress(&mut self.h, &block);
            self.buf_len = 0;
        }
        for byte in self.buf.iter_mut().take(56).skip(self.buf_len) {
            *byte = 0;
        }
        self.buf[56..].copy_from_slice(&bit_len.to_be_bytes());
        let block = self.buf;
        compress(&mut self.h, &block);

        let mut out = [0u8; DIGEST_LEN];
        for (i, word) in self.h.iter().enumerate() {
            out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
        }
        out
    }
}

/// The SHA-256 compression function over one 64-byte block.
fn compress(h: &mut [u32; 8], block: &[u8; 64]) {
    let mut w = [0u32; 64];
    for (i, word) in w.iter_mut().take(16).enumerate() {
        *word = u32::from_be_bytes([
            block[i * 4],
            block[i * 4 + 1],
            block[i * 4 + 2],
            block[i * 4 + 3],
        ]);
    }
    for i in 16..64 {
        let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
        let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16]
            .wrapping_add(s0)
            .wrapping_add(w[i - 7])
            .wrapping_add(s1);
    }

    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = *h;
    for i in 0..64 {
        let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let ch = (e & f) ^ ((!e) & g);
        let t1 = hh
            .wrapping_add(s1)
            .wrapping_add(ch)
            .wrapping_add(K[i])
            .wrapping_add(w[i]);
        let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let t2 = s0.wrapping_add(maj);

        hh = g;
        g = f;
        f = e;
        e = d.wrapping_add(t1);
        d = c;
        c = b;
        b = a;
        a = t1.wrapping_add(t2);
    }
    for (dst, src) in h.iter_mut().zip([a, b, c, d, e, f, g, hh]) {
        *dst = dst.wrapping_add(src);
    }
}

/// Render bytes as lowercase hex.
pub fn hex(digest: &[u8]) -> String {
    digest.iter().fold(
        String::with_capacity(digest.len() * 2),
        |mut out, byte| {
            // write! to a String is infallible.
            let _ = write!(out, "{byte:02x}");
            out
        },
    )
}

/// SHA-256 of a byte slice, as raw digest bytes.
///
/// The hex-string forms below are what most callers want. This one exists for
/// callers that fold a digest into a larger key rather than printing it —
/// gljax's compile cache concatenates several digests before hashing again,
/// and going through hex and back would be a round trip for nothing.
pub fn sha256(data: &[u8]) -> [u8; DIGEST_LEN] {
    let mut ctx = Sha256Ctx::new();
    ctx.update(data);
    ctx.finalize()
}

/// SHA-256 of a byte slice, as 64 lowercase hex characters.
pub fn sha256_bytes(data: &[u8]) -> String {
    hex(&sha256(data))
}

/// SHA-256 of a file, streamed, as 64 lowercase hex characters.
///
/// Reads in 64 KiB chunks so memory stays flat regardless of file size — the
/// `.gllm` layer files this was written for are routinely gigabytes.
pub fn sha256_file(path: &Path) -> std::io::Result<String> {
    let mut file = std::fs::File::open(path)?;
    let mut ctx = Sha256Ctx::new();
    let mut buf = vec![0u8; HASH_CHUNK_SIZE];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        ctx.update(&buf[..n]);
    }
    Ok(hex(&ctx.finalize()))
}

/// SHA-256 truncated to its first 128 bits, as 32 lowercase hex characters.
///
/// This is `glbench`'s archive content digest (`sha256-128`). Truncation is a
/// deliberate size/strength trade for a field a human reads in a diff; see the
/// threat model in this module's docs before reaching for it elsewhere.
pub fn sha256_128_hex(data: &[u8]) -> String {
    let mut ctx = Sha256Ctx::new();
    ctx.update(data);
    hex(&ctx.finalize()[..TRUNCATED_LEN])
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// SHA-256 of b"hello" — the vector this primitive carried when it lived in
    /// `glictus-caliburni::checksum`. Kept so the move is provably faithful.
    const SHA256_HELLO: &str = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824";

    #[test]
    fn sha256_bytes_matches_the_hello_vector_from_glictus_caliburni() {
        assert_eq!(sha256_bytes(b"hello"), SHA256_HELLO);
    }

    /// The published NIST/FIPS-180-4 vectors. A hand-written digest that is not
    /// tested against them is just a hash-shaped function.
    #[test]
    fn sha256_bytes_matches_the_published_fips_180_4_vectors() {
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
            assert_eq!(sha256_bytes(input), expected, "input len {}", input.len());
        }
    }

    /// FIPS-180-4's long vector — catches multi-block accumulation errors that
    /// the short vectors cannot reach.
    #[test]
    fn sha256_bytes_matches_the_one_million_a_vector() {
        let data = vec![b'a'; 1_000_000];
        assert_eq!(
            sha256_bytes(&data),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
        );
    }

    /// Streaming must agree with one-shot at every chunk size, especially ones
    /// that split a 64-byte block. This is the property [`sha256_file`] rests on.
    #[test]
    fn incremental_update_matches_one_shot_at_every_chunk_size() {
        let data: Vec<u8> = (0..1000u32).map(|i| (i % 251) as u8).collect();
        let expected = sha256_bytes(&data);
        for chunk in [1usize, 7, 63, 64, 65, 127, 128, 333] {
            let mut ctx = Sha256Ctx::new();
            for part in data.chunks(chunk) {
                ctx.update(part);
            }
            assert_eq!(
                hex(&ctx.finalize()),
                expected,
                "chunk size {chunk} disagreed with one-shot"
            );
        }
    }

    /// 55/56 and 63/64 are the exact lengths where the length suffix stops
    /// fitting in the final block and padding needs an extra one.
    #[test]
    fn handles_every_padding_boundary() {
        for len in [54usize, 55, 56, 57, 63, 64, 65, 118, 119, 120] {
            let digest = sha256_bytes(&vec![b'a'; len]);
            assert_eq!(digest.len(), 64, "len {len}");
        }
        // Distinct inputs across a padding boundary must give distinct digests.
        assert_ne!(sha256_bytes(&[b'a'; 55]), sha256_bytes(&[b'a'; 56]));
        assert_ne!(sha256_bytes(&[b'a'; 63]), sha256_bytes(&[b'a'; 64]));
    }

    #[test]
    fn sha256_file_matches_sha256_bytes_on_the_same_content() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("data.bin");
        let contents = b"some gllm payload bytes";
        fs::write(&path, contents).unwrap();
        assert_eq!(sha256_file(&path).unwrap(), sha256_bytes(contents));
    }

    /// Larger than one 64 KiB read, so the streaming loop runs more than once.
    #[test]
    fn sha256_file_streams_a_file_larger_than_one_read_chunk() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("big.bin");
        let contents: Vec<u8> = (0..(HASH_CHUNK_SIZE * 2 + 12345))
            .map(|i| (i % 253) as u8)
            .collect();
        fs::write(&path, &contents).unwrap();
        assert_eq!(sha256_file(&path).unwrap(), sha256_bytes(&contents));
    }

    #[test]
    fn sha256_128_hex_is_the_first_16_bytes_of_the_full_digest() {
        let full = sha256_bytes(b"hello");
        let truncated = sha256_128_hex(b"hello");
        assert_eq!(truncated.len(), 32);
        assert_eq!(truncated, &full[..32]);
        assert!(truncated.bytes().all(|b| b.is_ascii_hexdigit()));
    }

    /// The raw accessor and the hex form must describe the same digest — they
    /// are two views of one computation, not two computations.
    #[test]
    fn sha256_raw_bytes_and_the_hex_form_agree() {
        for input in [b"".as_slice(), b"hello", b"abc"] {
            assert_eq!(hex(&sha256(input)), sha256_bytes(input));
        }
        assert_eq!(sha256(b"hello").len(), DIGEST_LEN);
    }

    #[test]
    fn sha256_128_hex_separates_inputs_that_differ_by_one_byte() {
        assert_ne!(sha256_128_hex(b"archive-a"), sha256_128_hex(b"archive-b"));
    }
}
