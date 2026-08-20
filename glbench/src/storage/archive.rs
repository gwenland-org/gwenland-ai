//! Reading and writing session archives.
//!
//! An archive is a single JSON file per session (STORAGE RULE: no database, no
//! cloud, user-managed files). This module is the read/write seam over the
//! session's JSON projection; it carries the schema/version stamp implicitly via
//! the session metadata, and refuses to read a file whose schema is newer than
//! this build understands.
//!
//! # Finalisation (v3)
//!
//! [`write`] is the finalisation point, not just a serializer. Before any bytes
//! reach disk it:
//!
//! 1. annotates the session's `availability` map for every `null` it emits,
//! 2. checks the mode-consistency table and the D-10 invariant,
//! 3. seals the document with a `sha256-128` content digest,
//! 4. marks the file read-only.
//!
//! Steps 1–2 are here rather than in `export` on purpose: by export time the
//! archive already exists and a check has nothing left to prevent. Step 4 is
//! advisory — a read-only flag is trivially cleared, so it guards against
//! *accident*; the digest is the actual guarantee (D-17).

use std::fs;
use std::path::Path;

use crate::core::schema::SCHEMA_VERSION;
use crate::core::session::BenchmarkSession;
use crate::export::json;
use crate::storage::digest::{self, DigestError};
use crate::validation::availability::ENNullSemantics;
use crate::validation::integrity::ValidationReport;

/// Write a session to `path`, finalising it first under the default (strict)
/// null-semantics policy.
pub fn write(session: &BenchmarkSession, path: &Path) -> Result<(), String> {
    write_with_policy(session, path, ENNullSemantics::Strict).map(|_| ())
}

/// Write a session under an explicit null-semantics policy.
///
/// Returns the finalisation report so a caller can surface warnings. Under
/// [`ENNullSemantics::Strict`] a D-10 or mode-consistency violation is an error
/// and nothing is written; under [`ENNullSemantics::Lenient`] the same
/// violations are warnings and the archive is written anyway.
pub fn write_with_policy(
    session: &BenchmarkSession,
    path: &Path,
    policy: ENNullSemantics,
) -> Result<ValidationReport, String> {
    // Work on a copy: writing an archive must not mutate the caller's session.
    let mut session = session.clone();
    // Install the sentinel block first. `integrity` is filled by the seal a few
    // lines below, so annotating it as a missing value here would produce an
    // entry that the written archive immediately contradicts — D-10's mirror
    // case, self-inflicted.
    session.integrity = Some(crate::storage::digest::VLIntegrity::sentinel());
    session.annotate_availability()?;

    let value = session.to_json();
    let mut report =
        crate::validation::availability::check_value(&value, &session.availability, policy);
    for finding in session.check_mode_consistency().findings {
        report.findings.push(finding);
    }

    if !report.passed() {
        let detail = report
            .findings
            .iter()
            .map(|f| format!("  [{}] {}: {}", f.severity.as_str(), f.check, f.message))
            .collect::<Vec<_>>()
            .join("\n");
        return Err(format!(
            "refusing to write {}: session failed finalisation\n{detail}\n\
             (re-run with --null-semantics lenient to write it anyway)",
            path.display()
        ));
    }

    write_text(&digest::seal(&value).to_pretty(), path)?;
    Ok(report)
}

/// Write already-rendered archive text to `path` and mark it read-only.
///
/// Shared with [`crate::storage::join`], which seals its own document rather
/// than a session. Clears an existing read-only flag first: the flag exists to
/// stop an *accident*, and glbench overwriting its own `--out` target is not
/// one. The digest is what detects a change nobody intended.
pub fn write_text(text: &str, path: &Path) -> Result<(), String> {
    clear_readonly(path);
    fs::write(path, text).map_err(|e| format!("writing {}: {e}", path.display()))?;
    set_readonly(path);
    Ok(())
}

/// Best-effort read-only flag. Failure is not an error: the flag is advisory
/// (D-17), and a filesystem that will not carry it does not invalidate the
/// digest that actually guarantees integrity.
fn set_readonly(path: &Path) {
    if let Ok(meta) = fs::metadata(path) {
        let mut perms = meta.permissions();
        perms.set_readonly(true);
        let _ = fs::set_permissions(path, perms);
    }
}

/// Clear the advisory read-only flag if the file already exists.
///
/// `set_readonly(false)` is flagged by clippy because on Unix it restores every
/// write bit rather than the ones the file had. That is the intended effect
/// here and it widens nothing: the only files this runs on are archives
/// `write_text` itself created with `fs::write` and then marked read-only
/// moments earlier, so clearing the flag returns them to exactly the state
/// `fs::write` produced. There is no portable std alternative — `PermissionsExt`
/// is Unix-only and this repo's primary machine is Windows (D-17).
#[allow(clippy::permissions_set_readonly_false)]
fn clear_readonly(path: &Path) {
    if let Ok(meta) = fs::metadata(path) {
        let mut perms = meta.permissions();
        if perms.readonly() {
            perms.set_readonly(false);
            let _ = fs::set_permissions(path, perms);
        }
    }
}

/// Read a session back from a JSON archive at `path`, verifying its digest.
///
/// A digest mismatch is reported through [`read_verified`]'s findings rather
/// than refusing the read: refusing to show a modified archive would make the
/// tool useless exactly when a user most needs to see what changed.
pub fn read(path: &Path) -> Result<BenchmarkSession, String> {
    read_verified(path, true).map(|(session, _)| session)
}

/// Read a session and report what verification found.
///
/// `verify` is the `--verify` / `--no-verify` switch: default-on for `inspect`
/// and `compare`. An absent integrity block is a v1 archive and produces an
/// `Info` finding, never an error (D-20).
pub fn read_verified(
    path: &Path,
    verify: bool,
) -> Result<(BenchmarkSession, ValidationReport), String> {
    use crate::validation::integrity::Severity;

    let text = fs::read_to_string(path).map_err(|e| format!("reading {}: {e}", path.display()))?;
    let value = json::parse(&text).map_err(|e| format!("parsing {}: {e}", path.display()))?;

    // Refuse archives from a future schema — their shape may not match.
    if let Some(v) = value
        .get("metadata")
        .and_then(|m| m.get("schema_version"))
        .and_then(|n| n.as_f64())
    {
        if (v as u32) > SCHEMA_VERSION {
            return Err(format!(
                "archive {} uses schema v{} but this glbench understands v{SCHEMA_VERSION}",
                path.display(),
                v as u32
            ));
        }
    }

    let mut report = ValidationReport::default();
    if verify {
        match digest::verify(&value) {
            Ok(()) => {}
            Err(DigestError::Absent) => report.push(
                Severity::Info,
                "integrity",
                format!(
                    "{}: no content digest (v1 archive) — modification cannot be detected",
                    path.display()
                ),
            ),
            Err(e) => report.push(Severity::Error, "integrity", format!("{}: {e}", path.display())),
        }
    }

    let session = BenchmarkSession::from_json(&value).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok((session, report))
}

/// The digest an archive *records*, without checking whether its content still
/// matches.
///
/// Deliberately does not verify. [`crate::storage::join`] compares this against
/// what it recorded at join time, and that comparison is exactly what it needs
/// when the two disagree — a function that errored on a mismatch would hide the
/// value the caller is trying to read.
pub fn recorded_digest(path: &Path) -> Result<String, DigestError> {
    parse_archive(path)?
        .get("integrity")
        .and_then(|i| i.get("digest"))
        .and_then(|d| d.as_str())
        .map(String::from)
        .ok_or(DigestError::Absent)
}

/// Check an archive file's content against its own recorded digest.
pub fn verify_file(path: &Path) -> Result<(), DigestError> {
    digest::verify(&parse_archive(path)?)
}

/// Read and parse an archive file as raw JSON.
fn parse_archive(path: &Path) -> Result<json::Json, DigestError> {
    let text = fs::read_to_string(path)
        .map_err(|e| DigestError::Malformed(format!("reading {}: {e}", path.display())))?;
    json::parse(&text)
        .map_err(|e| DigestError::Malformed(format!("parsing {}: {e}", path.display())))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::availability::ENAvailability;
    use crate::core::metrics::{IterationMetrics, MeasurementSet};
    use crate::core::mode::{ENInferenceRole, ENSessionMode};
    use crate::core::result::SessionMetadata;
    use crate::core::workload::WorkloadSpec;
    use crate::engine::metadata::EngineMetadata;
    use crate::environment::hardware::EnvironmentSnapshot;
    use crate::export::json::Json;

    fn sample() -> BenchmarkSession {
        let mut m = MeasurementSet::default();
        m.iterations.push(IterationMetrics {
            prompt_tokens: 100,
            generated_tokens: 128,
            prefill_ms: 100.0,
            decode_ms: 4000.0,
            total_ms: 4100.0,
        });
        BenchmarkSession::new(
            SessionMetadata::new("test-run"),
            EnvironmentSnapshot::probe(""),
            EngineMetadata {
                name: "glproc".into(),
                backend: "cpu".into(),
                available: true,
                model_arch: Some("qwen2".into()),
                quantization: Some("Q8_0".into()),
                thinking_capable: Some(false),
            },
            WorkloadSpec::default(),
            m,
        )
    }

    #[test]
    fn round_trip_through_disk() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("run.json");
        let session = sample();
        write(&session, &path).unwrap();
        let back = read(&path).unwrap();
        assert_eq!(back.metadata.label, "test-run");
        assert_eq!(back.engine.name, "glproc");
        assert_eq!(back.measurements.iterations.len(), 1);
        assert_eq!(back.measurements.iterations[0].generated_tokens, 128);
    }

    /// Gate 1: a v2 archive round-trips and verifies.
    #[test]
    fn a_v2_archive_round_trips_and_verifies_clean() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("run.json");
        write(&sample(), &path).unwrap();

        let (back, report) = read_verified(&path, true).unwrap();
        assert!(report.passed(), "findings: {:?}", report.findings);
        assert!(report.findings.is_empty(), "a sealed v2 archive verifies silently");
        assert_eq!(back.metadata.schema_version, 2);
        assert_eq!(back.metadata.session_mode, ENSessionMode::InferenceOnly);
        assert_eq!(
            back.inference.as_ref().map(|i| i.role),
            Some(ENInferenceRole::Standalone)
        );
        assert!(back.integrity.is_some(), "the archive must carry its digest");
    }

    /// The finalisation pass must leave the caller's session untouched — a
    /// writer that mutates its input turns "archive it" into "archive it and
    /// change it".
    #[test]
    fn writing_does_not_mutate_the_session_it_was_given() {
        let dir = tempfile::tempdir().unwrap();
        let session = sample();
        assert!(session.availability.is_empty());
        write(&session, &dir.path().join("run.json")).unwrap();
        assert!(session.availability.is_empty());
        assert!(session.integrity.is_none());
    }

    #[test]
    fn every_null_in_a_written_archive_carries_an_availability_entry() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("run.json");
        write(&sample(), &path).unwrap();

        let text = fs::read_to_string(&path).unwrap();
        let report = crate::validation::availability::check(&text);
        assert!(
            report.passed(),
            "a written archive must satisfy D-10 on its own: {:?}",
            report.findings
        );
    }

    /// D-17: advisory, and documented as such. The flag being set is asserted;
    /// its being clearable is the reason the digest exists.
    #[test]
    fn a_written_archive_is_marked_read_only_and_can_still_be_rewritten() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("run.json");
        write(&sample(), &path).unwrap();
        assert!(fs::metadata(&path).unwrap().permissions().readonly());

        // glbench overwriting its own --out target is not the accident the flag
        // guards against, so a second write must succeed rather than fail.
        write(&sample(), &path).unwrap();
        assert!(fs::metadata(&path).unwrap().permissions().readonly());

        // Let the temp dir clean up on Windows.
        clear_readonly(&path);
    }

    #[test]
    fn a_body_edit_is_reported_as_an_error_but_the_session_still_reads() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("run.json");
        write(&sample(), &path).unwrap();

        // Edit the archive as a user's editor would.
        clear_readonly(&path);
        let text = fs::read_to_string(&path).unwrap();
        let edited = text.replace("\"test-run\"", "\"edited-run\"");
        assert_ne!(edited, text, "the edit must actually change the file");
        fs::write(&path, edited).unwrap();

        let (session, report) = read_verified(&path, true).unwrap();
        assert!(!report.passed(), "a modified archive must produce an error finding");
        assert!(report.findings[0].message.contains("digest mismatch"));
        // Still rendered: refusing to show it would be useless exactly when the
        // user most needs to see what changed.
        assert_eq!(session.metadata.label, "edited-run");
    }

    #[test]
    fn no_verify_skips_the_check_entirely() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("run.json");
        write(&sample(), &path).unwrap();
        clear_readonly(&path);
        let text = fs::read_to_string(&path).unwrap();
        fs::write(&path, text.replace("\"test-run\"", "\"edited-run\"")).unwrap();

        let (_, report) = read_verified(&path, false).unwrap();
        assert!(report.findings.is_empty(), "--no-verify must not check");
    }

    #[test]
    fn strict_mode_refuses_to_write_a_session_with_an_unexplained_null() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("run.json");

        // A mode/content disagreement the annotator cannot paper over: the mode
        // says inference-only but the inference envelope carries the wrong role.
        let mut session = sample();
        session.inference = Some(crate::core::inference::VLInferenceSession::nested(
            ENInferenceRole::PostTraining,
        ));

        let err = write(&session, &path).unwrap_err();
        assert!(err.contains("failed finalisation"), "got {err}");
        assert!(!path.exists(), "nothing may be written when finalisation fails");
    }

    #[test]
    fn lenient_mode_writes_the_same_session_and_reports_the_violation() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("run.json");
        let mut session = sample();
        // Claim a status on a field that unconditionally carries a value.
        crate::core::availability::set(
            &mut session.availability,
            "metadata.label",
            ENAvailability::Unsupported,
        )
        .unwrap();

        assert!(write(&session, &path).is_err());
        assert!(!path.exists());

        let report = write_with_policy(&session, &path, ENNullSemantics::Lenient).unwrap();
        assert!(path.exists(), "lenient mode writes anyway");
        assert!(report.findings.iter().any(|f| f.message.contains("carries a value")));

        clear_readonly(&path);
    }

    #[test]
    fn an_archive_from_a_future_schema_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("future.json");
        let value = Json::obj([(
            "metadata",
            Json::obj([("schema_version", Json::n((SCHEMA_VERSION + 1) as f64))]),
        )]);
        fs::write(&path, value.to_pretty()).unwrap();

        let err = read(&path).unwrap_err();
        assert!(err.contains("understands v"), "got {err}");
    }

    #[test]
    fn recorded_digest_reports_absence_for_an_archive_with_no_integrity_block() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v1.json");
        fs::write(&path, Json::obj([("metadata", Json::Null)]).to_pretty()).unwrap();
        assert_eq!(recorded_digest(&path), Err(DigestError::Absent));
    }

    #[test]
    fn recorded_digest_returns_the_digest_of_a_sealed_archive() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("run.json");
        write(&sample(), &path).unwrap();

        let recorded = recorded_digest(&path).unwrap();
        let session = read(&path).unwrap();
        assert_eq!(Some(recorded), session.integrity.map(|i| i.digest));
    }
}
