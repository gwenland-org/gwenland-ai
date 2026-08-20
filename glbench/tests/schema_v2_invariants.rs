//! The invariants glbench v3 Wave 1 exists to hold.
//!
//! These are the design's own tests 1, 2, 3 and 6 (§11). Each one is here
//! because the invariant it covers is *silently* violable: nothing else in the
//! suite would go red if it broke, and the archive would still look right.
//!
//! Tests 4 (observer overhead) and 5 (GLBitProf known answers) belong to Waves
//! 3 and 2 respectively and are deliberately absent.
//!
//! No test here asserts on a timing threshold. A number belongs in
//! `glbench/benches/` where the repeat count and noise floor are handled
//! properly; a timing assertion in a `#[test]` is a flaky test wearing a
//! measurement's clothes.

use std::fs;
use std::path::{Path, PathBuf};

use glbench::core::availability::{self, ENAvailability, VLAvailabilityMap};
use glbench::core::inference::VLInferenceSession;
use glbench::core::metrics::{IterationMetrics, MeasurementSet};
use glbench::core::mode::{ENInferenceRole, ENSessionMode};
use glbench::core::result::SessionMetadata;
use glbench::core::session::BenchmarkSession;
use glbench::core::workload::WorkloadSpec;
use glbench::engine::metadata::EngineMetadata;
use glbench::environment::hardware::EnvironmentSnapshot;
use glbench::export::json::{self, Json};
use glbench::storage::archive;
use glbench::storage::digest::{self, DigestError};
use glbench::validation::availability::ENNullSemantics;

/// The checked-in v1 archive. Its shape was derived from the pre-v3 writer's
/// own source, not from the v3 writer — a fixture the current writer produced
/// could not prove anything about v1.
fn v1_fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/v1_archive.json")
}

fn session(label: &str) -> BenchmarkSession {
    let mut m = MeasurementSet::default();
    m.iterations.push(IterationMetrics {
        prompt_tokens: 96,
        generated_tokens: 128,
        prefill_ms: 412.5,
        decode_ms: 6203.0,
        total_ms: 6615.5,
    });
    BenchmarkSession::new(
        SessionMetadata::new(label),
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

/// Clear the advisory read-only flag so a test can edit or delete the file.
///
/// See `archive::clear_readonly` for why the lint is allowed: these are files
/// the test itself just wrote, and this restores them to how `fs::write` left
/// them.
#[allow(clippy::permissions_set_readonly_false)]
fn make_writable(path: &Path) {
    let mut perms = fs::metadata(path).unwrap().permissions();
    perms.set_readonly(false);
    fs::set_permissions(path, perms).unwrap();
}

// ---------------------------------------------------------------------------
// Test 1 — No null without a status (D-10)
// ---------------------------------------------------------------------------

#[test]
fn test1_a_null_with_no_availability_entry_fails_and_names_the_path() {
    // A session shape with one deliberate, unexplained null at a known path.
    let doc = Json::obj([
        ("metadata", Json::obj([("label", Json::s("deliberate"))])),
        (
            "measurements",
            Json::obj([
                ("decode_ms", Json::n(6203.0)),
                // The defect: a value that is absent for no stated reason.
                ("energy_joules", Json::Null),
            ]),
        ),
    ]);

    let report = glbench::validation::availability::check_value(
        &doc,
        &VLAvailabilityMap::new(),
        ENNullSemantics::Strict,
    );

    assert!(!report.passed(), "an unexplained null must fail the session");
    assert_eq!(report.findings.len(), 1, "findings: {:?}", report.findings);
    assert!(
        report.findings[0]
            .message
            .contains("measurements.energy_joules"),
        "the finding must name the exact path, got: {}",
        report.findings[0].message
    );
}

#[test]
fn test1_mirror_a_status_on_a_non_null_field_also_fails() {
    let doc = Json::obj([(
        "measurements",
        Json::obj([("decode_ms", Json::n(6203.0))]),
    )]);
    let mut map = VLAvailabilityMap::new();
    availability::set(&mut map, "measurements.decode_ms", ENAvailability::Unsupported).unwrap();

    let report =
        glbench::validation::availability::check_value(&doc, &map, ENNullSemantics::Strict);

    assert!(!report.passed(), "a status on a field that carries a value must fail");
    assert!(
        report.findings[0].message.contains("measurements.decode_ms"),
        "got: {}",
        report.findings[0].message
    );
}

#[test]
fn test1_the_check_runs_at_finalisation_not_at_export() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("run.json");

    // A session the mode table rejects: mode says inference-only, the envelope
    // says this run happened after training.
    let mut broken = session("broken");
    broken.inference = Some(VLInferenceSession::nested(ENInferenceRole::PostTraining));

    // Export renders it without complaint — by export time there is nothing
    // left to prevent.
    let rendered = broken.to_json().to_pretty();
    assert!(rendered.contains("post_training"));

    // Finalisation refuses, and writes nothing.
    let err = archive::write(&broken, &path).unwrap_err();
    assert!(err.contains("failed finalisation"), "got {err}");
    assert!(!path.exists(), "a rejected session must leave no file behind");
}

// ---------------------------------------------------------------------------
// Test 2 — Digest is stable and detects a single-byte edit
// ---------------------------------------------------------------------------

#[test]
fn test2a_write_read_verify_succeeds() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("run.json");
    archive::write(&session("digest-stable"), &path).unwrap();

    let (back, report) = archive::read_verified(&path, true).unwrap();
    assert!(report.passed(), "findings: {:?}", report.findings);
    assert!(report.findings.is_empty(), "a sealed archive verifies silently");
    assert_eq!(back.metadata.label, "digest-stable");

    make_writable(&path);
}

#[test]
fn test2a_the_digest_is_stable_across_two_writes_of_the_same_session() {
    let dir = tempfile::tempdir().unwrap();
    let s = session("stable");
    let a = dir.path().join("a.json");
    let b = dir.path().join("b.json");
    archive::write(&s, &a).unwrap();
    archive::write(&s, &b).unwrap();

    // Same session value in, same digest out. Everything the archive records
    // about *when* it was written lives in the session itself, so two writes of
    // one value are byte-identical.
    assert_eq!(
        fs::read_to_string(&a).unwrap(),
        fs::read_to_string(&b).unwrap(),
        "two writes of the same session must produce the same archive"
    );

    make_writable(&a);
    make_writable(&b);
}

#[test]
fn test2b_flipping_one_character_in_the_body_is_detected() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("run.json");
    archive::write(&session("body-flip"), &path).unwrap();
    make_writable(&path);

    // One character, in a value nobody would notice changing.
    let text = fs::read_to_string(&path).unwrap();
    let edited = text.replace("\"prompt_tokens\": 96", "\"prompt_tokens\": 97");
    assert_ne!(edited, text, "the edit must actually change the file");
    fs::write(&path, &edited).unwrap();

    let value = json::parse(&edited).unwrap();
    match digest::verify(&value) {
        Err(DigestError::Mismatch { .. }) => {}
        other => panic!("expected Mismatch, got {other:?}"),
    }

    // And the read path surfaces it as an error while still rendering.
    let (session, report) = archive::read_verified(&path, true).unwrap();
    assert!(!report.passed());
    assert_eq!(session.measurements.iterations[0].prompt_tokens, 97);
}

#[test]
fn test2c_flipping_a_character_inside_the_digest_field_is_detected_not_passed() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("run.json");
    archive::write(&session("digest-flip"), &path).unwrap();
    make_writable(&path);

    let text = fs::read_to_string(&path).unwrap();
    let value = json::parse(&text).unwrap();
    let recorded = value
        .get("integrity")
        .and_then(|i| i.get("digest"))
        .and_then(|d| d.as_str())
        .unwrap()
        .to_string();

    // Flip one hex character of the recorded digest, keeping it well-formed —
    // the case a naive "hash whatever is in the file" scheme accepts.
    let flipped: String = recorded
        .chars()
        .enumerate()
        .map(|(i, c)| if i == 0 { if c == '0' { '1' } else { '0' } } else { c })
        .collect();
    assert_ne!(flipped, recorded);
    assert_eq!(flipped.len(), 32);
    fs::write(&path, text.replace(&recorded, &flipped)).unwrap();

    let edited = json::parse(&fs::read_to_string(&path).unwrap()).unwrap();
    match digest::verify(&edited) {
        Err(DigestError::Mismatch { expected, .. }) => assert_eq!(expected, flipped),
        other => panic!("expected Mismatch, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Test 3 — Mode consistency
// ---------------------------------------------------------------------------

#[test]
fn test3_inference_only_requires_a_standalone_role_and_no_training() {
    let mut s = session("inference-only");
    s.metadata.session_mode = ENSessionMode::InferenceOnly;
    availability::set(&mut s.availability, "training", ENAvailability::NotApplicable).unwrap();

    assert_eq!(
        s.inference.as_ref().map(|i| i.role),
        Some(ENInferenceRole::Standalone)
    );
    let report = s.check_mode_consistency();
    assert!(report.passed(), "findings: {:?}", report.findings);
}

#[test]
fn test3_inference_only_rejects_a_wrong_role() {
    let mut s = session("wrong-role");
    s.metadata.session_mode = ENSessionMode::InferenceOnly;
    availability::set(&mut s.availability, "training", ENAvailability::NotApplicable).unwrap();
    s.inference = Some(VLInferenceSession::nested(ENInferenceRole::PreTraining));

    let report = s.check_mode_consistency();
    assert!(!report.passed());
    assert!(
        report.findings[0].message.contains("expected 'standalone'"),
        "got {}",
        report.findings[0].message
    );
}

#[test]
fn test3_inference_only_requires_the_training_not_applicable_entry() {
    let mut s = session("no-entry");
    s.metadata.session_mode = ENSessionMode::InferenceOnly;
    // Deliberately no availability entry for `training`.
    let report = s.check_mode_consistency();
    assert!(!report.passed());
    assert!(
        report.findings[0].message.contains("availability['training']"),
        "got {}",
        report.findings[0].message
    );

    // The wrong status is caught too, not just the missing one.
    availability::set(&mut s.availability, "training", ENAvailability::Unavailable).unwrap();
    let report = s.check_mode_consistency();
    assert!(!report.passed());
    assert!(report.findings[0].message.contains("expected 'not_applicable'"));
}

#[test]
fn test3_training_only_rejects_an_inference_block_and_needs_its_own_entry() {
    let mut s = session("training-only");
    s.metadata.session_mode = ENSessionMode::TrainingOnly;
    availability::set(&mut s.availability, "inference", ENAvailability::NotApplicable).unwrap();

    // `BenchmarkSession::new` installs a standalone inference envelope, which is
    // exactly wrong for this mode.
    let report = s.check_mode_consistency();
    assert!(!report.passed());
    let messages: Vec<_> = report.findings.iter().map(|f| f.message.as_str()).collect();
    assert!(
        messages.iter().any(|m| m.contains("carries an inference block")),
        "got {messages:?}"
    );
    // Without `train-bench` there is no training block to hold, so that is
    // reported as well — a default build genuinely cannot produce this mode.
    assert!(
        messages.iter().any(|m| m.contains("no training block")),
        "got {messages:?}"
    );

    // Removing the inference block clears that half of the complaint.
    s.inference = None;
    let report = s.check_mode_consistency();
    let messages: Vec<_> = report.findings.iter().map(|f| f.message.as_str()).collect();
    assert!(
        !messages.iter().any(|m| m.contains("carries an inference block")),
        "got {messages:?}"
    );
}

#[test]
fn test3_unified_requires_the_outer_role_to_be_pre_training() {
    let mut s = session("unified");
    s.metadata.session_mode = ENSessionMode::Unified;

    // Standalone is wrong here: a unified session's outer run is the baseline.
    let report = s.check_mode_consistency();
    let messages: Vec<_> = report.findings.iter().map(|f| f.message.as_str()).collect();
    assert!(
        messages.iter().any(|m| m.contains("expected 'pre_training'")),
        "got {messages:?}"
    );

    s.inference = Some(VLInferenceSession::nested(ENInferenceRole::PreTraining));
    let report = s.check_mode_consistency();
    let messages: Vec<_> = report.findings.iter().map(|f| f.message.as_str()).collect();
    assert!(
        !messages.iter().any(|m| m.contains("pre_training")),
        "the role complaint must clear, got {messages:?}"
    );
    // A unified session still needs a training block, which a default build
    // cannot produce — so the mode is unreachable here, honestly.
    assert!(messages.iter().any(|m| m.contains("no training block")));
}

#[test]
fn test3_a_violation_is_caught_at_finalisation_for_every_mode() {
    let dir = tempfile::tempdir().unwrap();
    for (i, mode) in [
        ENSessionMode::InferenceOnly,
        ENSessionMode::TrainingOnly,
        ENSessionMode::Unified,
    ]
    .into_iter()
    .enumerate()
    {
        let path = dir.path().join(format!("m{i}.json"));
        let mut s = session("mode-check");
        s.metadata.session_mode = mode;
        // The same deliberate violation in every mode: an outer role that no
        // mode in the table accepts.
        s.inference = Some(VLInferenceSession::nested(ENInferenceRole::PostTraining));

        let err = archive::write(&s, &path).unwrap_err();
        assert!(
            err.contains("failed finalisation"),
            "mode {}: got {err}",
            mode.as_str()
        );
        assert!(!path.exists(), "mode {}: nothing may be written", mode.as_str());
    }
}

// ---------------------------------------------------------------------------
// Test 6 — v1 archive still reads
// ---------------------------------------------------------------------------

#[test]
fn test6_a_v1_archive_reads_without_error() {
    let path = v1_fixture();
    assert!(path.exists(), "fixture missing: {}", path.display());

    let session = archive::read(&path).expect("a v1 archive must still read");
    assert_eq!(session.metadata.schema_version, 1);
    assert_eq!(session.metadata.label, "qwen2.5-0.5b-glproc-q8-v1");
    assert_eq!(session.measurements.iterations.len(), 2);
    assert_eq!(session.engine.name, "glproc");
}

#[test]
fn test6_a_v1_archive_reads_as_inference_only_with_an_empty_map_and_no_digest() {
    let path = v1_fixture();
    let (session, report) = archive::read_verified(&path, true).unwrap();

    // The three D-20 defaults.
    assert_eq!(session.metadata.session_mode, ENSessionMode::InferenceOnly);
    assert!(
        session.availability.is_empty(),
        "a v1 archive has no availability block: {:?}",
        session.availability
    );
    assert!(session.integrity.is_none(), "a v1 archive has no digest");

    // Absence, reported as such — not corruption, and not an error.
    assert!(report.passed(), "a v1 archive must not fail verification");
    assert_eq!(report.findings.len(), 1);
    assert!(
        report.findings[0].message.contains("no content digest"),
        "got {}",
        report.findings[0].message
    );

    // And the same absence expressed in the availability vocabulary.
    let value = json::parse(&fs::read_to_string(&path).unwrap()).unwrap();
    assert_eq!(digest::verify(&value), Err(DigestError::Absent));
    assert_eq!(ENAvailability::DoesNotExist.as_str(), "does_not_exist");
}

#[test]
fn test6_the_fixture_really_is_v1_shaped_not_a_v2_archive_in_disguise() {
    // The fixture's whole value is that it was NOT produced by the v3 writer.
    // If any v2 key crept in, it would be testing the writer against itself.
    let value = json::parse(&fs::read_to_string(v1_fixture()).unwrap()).unwrap();
    let root = value.as_obj().unwrap();
    for added_in_v2 in ["availability", "inference", "integrity", "training"] {
        assert!(
            !root.contains_key(added_in_v2),
            "fixture carries the v2 key '{added_in_v2}'"
        );
    }
    let metadata = value.get("metadata").unwrap().as_obj().unwrap();
    for added_in_v2 in ["session_mode", "host_identifier", "collection_profile"] {
        assert!(
            !metadata.contains_key(added_in_v2),
            "fixture metadata carries the v2 key '{added_in_v2}'"
        );
    }
    // And it carries every key the v1 writer emitted.
    for v1_key in [
        "metadata",
        "environment",
        "engine",
        "workload",
        "measurements",
        "telemetry",
        "behavior",
        "analysis",
        "comparison",
        "validation",
    ] {
        assert!(root.contains_key(v1_key), "fixture is missing the v1 key '{v1_key}'");
    }
}

#[test]
fn test6_a_v1_archive_can_be_re_archived_as_v2() {
    // Reading v1 and writing it back must produce a valid v2 archive: the
    // upgrade path exists even though no migration tool is shipped.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("upgraded.json");

    let session = archive::read(&v1_fixture()).unwrap();
    archive::write(&session, &path).unwrap();

    let (back, report) = archive::read_verified(&path, true).unwrap();
    assert!(report.passed(), "findings: {:?}", report.findings);
    assert_eq!(back.metadata.label, "qwen2.5-0.5b-glproc-q8-v1");
    // The rewritten archive stamps the schema version it was read with, which
    // is v1 — the metadata records the producing run, not this rewrite.
    assert!(back.integrity.is_some(), "the rewrite is sealed");
    assert!(!back.availability.is_empty(), "the rewrite is annotated");

    make_writable(&path);
}

// ---------------------------------------------------------------------------
// Round-trip of the new envelope itself
// ---------------------------------------------------------------------------

#[test]
fn the_v2_envelope_survives_a_full_write_read_cycle() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("run.json");

    let mut s = session("envelope");
    s.metadata.host_identifier = Some("ci-runner-3".to_string());
    s.metadata.collection_profile = Some("bits+weights".to_string());
    availability::set_with_note(
        &mut s.availability,
        "analysis.roofline.ceiling_gbs",
        ENAvailability::Estimated,
        "engine::capability::lookup",
    )
    .unwrap();

    archive::write(&s, &path).unwrap();
    let back = archive::read(&path).unwrap();

    assert_eq!(back.metadata.host_identifier.as_deref(), Some("ci-runner-3"));
    assert_eq!(back.metadata.collection_profile.as_deref(), Some("bits+weights"));
    assert_eq!(back.metadata.session_mode, ENSessionMode::InferenceOnly);
    let entry = back
        .availability
        .get("analysis.roofline.ceiling_gbs")
        .expect("the explicit entry must survive");
    assert_eq!(entry.status, ENAvailability::Estimated);
    assert_eq!(entry.note.as_deref(), Some("engine::capability::lookup"));

    make_writable(&path);
}
