//! Archive content digest (D-15, D-16, D-17).
//!
//! # What this guarantees, and what it does not
//!
//! A `sha256-128` digest over the archive's own canonical JSON. It detects a
//! partial write, silent corruption, and an archive edited after the fact.
//!
//! It is **not** a signature. Anyone who can edit the file can recompute the
//! digest, and this module hands them the function to do it with. "Content
//! digest" reads like tamper-proofing to someone who has not thought about it,
//! so it is spelled out here, in the archive's `algorithm` field, and in
//! `glcore::hash`'s own docs.
//!
//! # The sentinel
//!
//! A digest cannot cover itself, so the field is set to 32 ASCII `0`s while
//! hashing and replaced afterwards. That is only sound if the serialisation is
//! a deterministic function of the value — and it is: `Json::Obj` is a
//! `BTreeMap`, so objects always emit in sorted key order regardless of
//! insertion order.
//!
//! Replacing the sentinel **re-serialises rather than splicing the string**. A
//! textual splice would depend on `00000000000000000000000000000000` appearing
//! exactly once in the document, and no hex-string field elsewhere in the
//! archive can be *proven* never to hold it. Re-serialising is one extra pass
//! over a 10–500 KB document and removes the class of bug entirely.
//!
//! # Naming
//!
//! The field says `sha256-128` and the docs say "128-bit content digest".
//! "SHA-128" is not a standard algorithm name and does not appear anywhere.

use std::collections::BTreeMap;

use crate::core::schema::ToJson;
use crate::export::json::Json;

/// The algorithm identifier written into every archive.
///
/// Carried in the file rather than assumed, so a future native GwenLand 128-bit
/// primitive can replace SHA-256 truncation without a schema change.
pub const ALGORITHM: &str = "sha256-128";

/// The placeholder the digest field holds while its own document is hashed.
/// 32 ASCII zeros — the same width as a real digest, so the document being
/// hashed has the same shape as the document written.
pub const SENTINEL: &str = "00000000000000000000000000000000";

/// Digest length in lowercase hex characters.
pub const DIGEST_HEX_LEN: usize = 32;

/// The integrity block of an archive.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VLIntegrity {
    /// Always [`ALGORITHM`] for archives this build writes.
    pub algorithm: String,
    /// 32 lowercase hex characters.
    pub digest: String,
}

impl VLIntegrity {
    /// The block as written before hashing: real algorithm, sentinel digest.
    pub fn sentinel() -> VLIntegrity {
        VLIntegrity { algorithm: ALGORITHM.to_string(), digest: SENTINEL.to_string() }
    }

    /// Parse back from JSON.
    pub fn from_json(v: &Json) -> Result<VLIntegrity, String> {
        let algorithm = v
            .get("algorithm")
            .and_then(|a| a.as_str())
            .ok_or_else(|| "integrity block has no 'algorithm' string".to_string())?
            .to_string();
        let digest = v
            .get("digest")
            .and_then(|d| d.as_str())
            .ok_or_else(|| "integrity block has no 'digest' string".to_string())?
            .to_string();
        Ok(VLIntegrity { algorithm, digest })
    }
}

impl ToJson for VLIntegrity {
    fn to_json(&self) -> Json {
        Json::obj([
            ("algorithm", Json::s(self.algorithm.clone())),
            ("digest", Json::s(self.digest.clone())),
            // Stated in the archive itself, not only in the docs: a reader who
            // finds a digest field will assume more of it than it can carry.
            (
                "note",
                Json::s(
                    "128-bit content digest: detects accidental modification. \
                     Not a signature — anyone who can edit this file can recompute it.",
                ),
            ),
        ])
    }
}

/// Why verification did not return `Ok`.
///
/// A plain error type, not a prefixed one: the naming convention exempts error
/// types, and `EN`/`VL` would say less than the word "error" already does.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DigestError {
    /// No integrity block. A v1 archive, which is an *absence*, not a failure —
    /// callers report it as [`crate::core::availability::ENAvailability::DoesNotExist`].
    Absent,
    /// The recomputed digest does not match the recorded one.
    Mismatch {
        /// What the archive claims.
        expected: String,
        /// What the content actually hashes to.
        actual: String,
    },
    /// The integrity block is present but malformed.
    Malformed(String),
}

impl std::fmt::Display for DigestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DigestError::Absent => write!(f, "no integrity block (v1 archive)"),
            DigestError::Mismatch { expected, actual } => write!(
                f,
                "content digest mismatch: archive records {expected}, content hashes to {actual}"
            ),
            DigestError::Malformed(why) => write!(f, "malformed integrity block: {why}"),
        }
    }
}

/// Replace the whole `integrity` block with the canonical sentinel block,
/// returning the modified value.
///
/// Shared by the write and verify paths so the two can never disagree about
/// what "the document being hashed" means — the single most likely way for a
/// sentinel scheme to break, and the one this module got wrong first: sealing
/// started from `integrity: null` while verifying started from a populated
/// block, so the two hashed documents differed by the block's own keys.
///
/// Replacing the *entire* block rather than patching its digest field makes
/// both paths hash a byte-identical stand-in regardless of what the input
/// carried. The consequence, stated rather than hidden: the digest does not
/// cover the integrity block itself. That block has no free-form content — the
/// `note` is a constant this build writes — and its two meaningful fields are
/// validated explicitly in [`verify`] before any hashing happens.
fn with_sentinel_digest(value: &Json) -> Json {
    let mut root: BTreeMap<String, Json> = match value.as_obj() {
        Some(m) => m.clone(),
        None => return value.clone(),
    };
    root.insert("integrity".to_string(), VLIntegrity::sentinel().to_json());
    Json::Obj(root)
}

/// Compute the digest of a document, with its own digest field neutralised.
///
/// The input is the *complete* value including an `integrity` block; whatever
/// that block's digest currently holds is replaced by the sentinel first, so
/// computing twice on the same content gives the same answer.
pub fn compute(value: &Json) -> String {
    let canonical = with_sentinel_digest(value).to_pretty();
    glcore::hash::sha256_128_hex(canonical.as_bytes())
}

/// Return the value with a correct, self-consistent integrity block.
pub fn seal(value: &Json) -> Json {
    let digest = compute(value);
    let mut root: BTreeMap<String, Json> = match value.as_obj() {
        Some(m) => m.clone(),
        None => return value.clone(),
    };
    let integrity = VLIntegrity { algorithm: ALGORITHM.to_string(), digest };
    root.insert("integrity".to_string(), integrity.to_json());
    Json::Obj(root)
}

/// Verify a parsed archive against its recorded digest.
///
/// Recomputes over the parsed value rather than the raw file text, so
/// insignificant whitespace differences are not reported as corruption — the
/// claim is about content, not bytes on disk.
pub fn verify(value: &Json) -> Result<(), DigestError> {
    let block = match value.get("integrity") {
        None => return Err(DigestError::Absent),
        Some(Json::Null) => return Err(DigestError::Absent),
        Some(b) => b,
    };
    let integrity = VLIntegrity::from_json(block).map_err(DigestError::Malformed)?;

    if integrity.algorithm != ALGORITHM {
        return Err(DigestError::Malformed(format!(
            "unknown digest algorithm '{}' (this build computes {ALGORITHM})",
            integrity.algorithm
        )));
    }
    if integrity.digest.len() != DIGEST_HEX_LEN
        || !integrity.digest.bytes().all(|b| b.is_ascii_hexdigit())
    {
        return Err(DigestError::Malformed(format!(
            "digest '{}' is not {DIGEST_HEX_LEN} hex characters",
            integrity.digest
        )));
    }

    // The recorded algorithm was already checked to equal ALGORITHM above, so
    // the canonical sentinel block is exactly what the writer hashed.
    let actual = compute(value);
    if actual != integrity.digest {
        return Err(DigestError::Mismatch { expected: integrity.digest, actual });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small stand-in for a session document: an integrity slot plus some
    /// content to protect.
    fn document(label: &str) -> Json {
        Json::obj([
            ("integrity", VLIntegrity::sentinel().to_json()),
            (
                "metadata",
                Json::obj([("label", Json::s(label)), ("schema_version", Json::n(2.0))]),
            ),
            ("measurements", Json::obj([("decode_ms", Json::n(4000.0))])),
        ])
    }

    #[test]
    fn a_sealed_document_verifies() {
        let sealed = seal(&document("run-a"));
        assert_eq!(verify(&sealed), Ok(()));
    }

    /// Same content in, same digest out — the property `--verify` rests on.
    #[test]
    fn the_digest_is_stable_across_two_computations_of_the_same_content() {
        let doc = document("run-a");
        assert_eq!(compute(&doc), compute(&doc));
        // And the sentinel state of the input does not change the answer: a
        // sealed document hashes to the digest it already carries.
        let sealed = seal(&doc);
        assert_eq!(compute(&sealed), compute(&doc));
    }

    #[test]
    fn a_changed_field_changes_the_digest() {
        assert_ne!(compute(&document("run-a")), compute(&document("run-b")));
    }

    #[test]
    fn editing_the_content_of_a_sealed_document_is_detected() {
        let sealed = seal(&document("run-a"));
        let mut root = sealed.as_obj().unwrap().clone();
        root.insert(
            "measurements".to_string(),
            Json::obj([("decode_ms", Json::n(4001.0))]),
        );
        let tampered = Json::Obj(root);

        match verify(&tampered) {
            Err(DigestError::Mismatch { .. }) => {}
            other => panic!("expected Mismatch, got {other:?}"),
        }
    }

    /// The failure mode a naive implementation has: editing the digest field
    /// itself must not make the document verify.
    #[test]
    fn editing_the_digest_field_itself_is_detected_rather_than_accidentally_passing() {
        let sealed = seal(&document("run-a"));
        let mut root = sealed.as_obj().unwrap().clone();
        let mut integrity = root.get("integrity").unwrap().as_obj().unwrap().clone();
        // Flip one hex character of the recorded digest.
        let recorded = integrity.get("digest").unwrap().as_str().unwrap().to_string();
        let flipped: String = recorded
            .chars()
            .enumerate()
            .map(|(i, c)| if i == 0 { if c == 'a' { 'b' } else { 'a' } } else { c })
            .collect();
        assert_ne!(flipped, recorded);
        integrity.insert("digest".to_string(), Json::s(flipped.clone()));
        root.insert("integrity".to_string(), Json::Obj(integrity));

        match verify(&Json::Obj(root)) {
            Err(DigestError::Mismatch { expected, .. }) => assert_eq!(expected, flipped),
            other => panic!("expected Mismatch, got {other:?}"),
        }
    }

    /// Setting the digest back to the sentinel must not verify either — that is
    /// the one value a careless "hash the document as-is" scheme would accept.
    #[test]
    fn a_digest_reset_to_the_sentinel_does_not_verify() {
        let sealed = seal(&document("run-a"));
        let mut root = sealed.as_obj().unwrap().clone();
        root.insert("integrity".to_string(), VLIntegrity::sentinel().to_json());
        match verify(&Json::Obj(root)) {
            Err(DigestError::Mismatch { expected, .. }) => assert_eq!(expected, SENTINEL),
            other => panic!("expected Mismatch, got {other:?}"),
        }
    }

    #[test]
    fn a_v1_archive_with_no_integrity_block_reports_absence_not_corruption() {
        let v1 = Json::obj([("metadata", Json::obj([("label", Json::s("old"))]))]);
        assert_eq!(verify(&v1), Err(DigestError::Absent));
        // A null block means the same thing.
        let nulled = Json::obj([("integrity", Json::Null)]);
        assert_eq!(verify(&nulled), Err(DigestError::Absent));
    }

    #[test]
    fn an_unknown_algorithm_is_malformed_rather_than_a_mismatch() {
        let sealed = seal(&document("run-a"));
        let mut root = sealed.as_obj().unwrap().clone();
        let mut integrity = root.get("integrity").unwrap().as_obj().unwrap().clone();
        integrity.insert("algorithm".to_string(), Json::s("blake3-128"));
        root.insert("integrity".to_string(), Json::Obj(integrity));

        match verify(&Json::Obj(root)) {
            Err(DigestError::Malformed(why)) => assert!(why.contains("blake3-128"), "got {why}"),
            other => panic!("expected Malformed, got {other:?}"),
        }
    }

    #[test]
    fn a_digest_of_the_wrong_shape_is_malformed() {
        for bad in ["", "abc", "zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz"] {
            let mut root = seal(&document("run-a")).as_obj().unwrap().clone();
            let mut integrity = root.get("integrity").unwrap().as_obj().unwrap().clone();
            integrity.insert("digest".to_string(), Json::s(bad));
            root.insert("integrity".to_string(), Json::Obj(integrity));
            match verify(&Json::Obj(root)) {
                Err(DigestError::Malformed(_)) => {}
                other => panic!("digest {bad:?}: expected Malformed, got {other:?}"),
            }
        }
    }

    #[test]
    fn a_sealed_document_records_the_algorithm_it_used() {
        let sealed = seal(&document("run-a"));
        let integrity = VLIntegrity::from_json(sealed.get("integrity").unwrap()).unwrap();
        assert_eq!(integrity.algorithm, ALGORITHM);
        assert_eq!(integrity.digest.len(), DIGEST_HEX_LEN);
        assert_ne!(integrity.digest, SENTINEL);
    }
}
