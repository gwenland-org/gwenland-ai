//! The `glbench join` / `glbench inspect` command plumbing.
//!
//! `storage::join`'s own tests cover the manifest and the drift check. What is
//! only reachable through the binary is the argument handling and the dispatch
//! that tells a join manifest from a session — and the v3 two-source
//! constraint, which is enforced at the CLI rather than in the type.

use std::fs;
use std::path::Path;
use std::process::Command;

use glbench::core::metrics::{IterationMetrics, MeasurementSet};
use glbench::core::result::SessionMetadata;
use glbench::core::session::BenchmarkSession;
use glbench::core::workload::WorkloadSpec;
use glbench::engine::metadata::EngineMetadata;
use glbench::environment::hardware::EnvironmentSnapshot;
use glbench::storage::archive;

/// The binary under test, as cargo built it for this suite.
const GLBENCH: &str = env!("CARGO_BIN_EXE_glbench");

fn write_session(path: &Path, label: &str, decode_ms: f64) {
    let mut m = MeasurementSet::default();
    m.iterations.push(IterationMetrics {
        prompt_tokens: 96,
        generated_tokens: 128,
        prefill_ms: 410.0,
        decode_ms,
        total_ms: 410.0 + decode_ms,
    });
    let session = BenchmarkSession::new(
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
    );
    archive::write(&session, path).unwrap();
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

struct Run {
    ok: bool,
    stdout: String,
    stderr: String,
}

fn glbench(args: &[&str]) -> Run {
    let out = Command::new(GLBENCH).args(args).output().expect("running glbench");
    Run {
        ok: out.status.success(),
        stdout: String::from_utf8_lossy(&out.stdout).to_string(),
        stderr: String::from_utf8_lossy(&out.stderr).to_string(),
    }
}

#[test]
fn join_writes_a_third_file_and_leaves_both_sources_untouched() {
    let dir = tempfile::tempdir().unwrap();
    let a = dir.path().join("a.json");
    let b = dir.path().join("b.json");
    let out = dir.path().join("join.json");
    write_session(&a, "baseline", 6200.0);
    write_session(&b, "candidate", 5100.0);
    let before = (fs::read_to_string(&a).unwrap(), fs::read_to_string(&b).unwrap());

    let run = glbench(&[
        "join",
        a.to_str().unwrap(),
        b.to_str().unwrap(),
        "--out",
        out.to_str().unwrap(),
        "--label",
        "q8-vs-q4",
    ]);
    assert!(run.ok, "stderr: {}", run.stderr);
    assert!(out.exists(), "the join file must be written");

    // Neither source moved.
    assert_eq!(fs::read_to_string(&a).unwrap(), before.0);
    assert_eq!(fs::read_to_string(&b).unwrap(), before.1);

    // The join records both digests.
    let manifest = glbench::storage::join::read(&out).unwrap();
    assert_eq!(manifest.label, "q8-vs-q4");
    assert_eq!(manifest.sources.len(), 2);
    assert!(manifest.sources.iter().all(|s| s.digest.is_some()));

    for p in [&a, &b, &out] {
        make_writable(p);
    }
}

#[test]
fn inspect_recognises_a_join_and_re_verifies_its_sources() {
    let dir = tempfile::tempdir().unwrap();
    let a = dir.path().join("a.json");
    let b = dir.path().join("b.json");
    let out = dir.path().join("join.json");
    write_session(&a, "baseline", 6200.0);
    write_session(&b, "candidate", 5100.0);

    assert!(glbench(&[
        "join",
        a.to_str().unwrap(),
        b.to_str().unwrap(),
        "--out",
        out.to_str().unwrap(),
    ])
    .ok);

    let run = glbench(&["inspect", out.to_str().unwrap()]);
    assert!(run.ok, "stderr: {}", run.stderr);
    assert!(run.stdout.contains("join:"), "stdout: {}", run.stdout);
    assert!(run.stdout.contains("baseline"), "stdout: {}", run.stdout);

    for p in [&a, &b, &out] {
        make_writable(p);
    }
}

/// The reason the digests are recorded at all.
#[test]
fn inspect_of_a_join_fails_once_a_source_has_been_re_measured() {
    let dir = tempfile::tempdir().unwrap();
    let a = dir.path().join("a.json");
    let b = dir.path().join("b.json");
    let out = dir.path().join("join.json");
    write_session(&a, "baseline", 6200.0);
    write_session(&b, "candidate", 5100.0);
    assert!(glbench(&[
        "join",
        a.to_str().unwrap(),
        b.to_str().unwrap(),
        "--out",
        out.to_str().unwrap(),
    ])
    .ok);

    // Re-run the baseline into the same path: a properly sealed archive, but a
    // different one than the join described.
    make_writable(&a);
    write_session(&a, "baseline", 9999.0);

    let run = glbench(&["inspect", out.to_str().unwrap()]);
    assert!(!run.ok, "drift must fail the command");
    assert!(
        run.stderr.contains("has changed since the join was written"),
        "stderr: {}",
        run.stderr
    );

    for p in [&a, &b, &out] {
        make_writable(p);
    }
}

#[test]
fn join_rejects_anything_other_than_exactly_two_sources() {
    let dir = tempfile::tempdir().unwrap();
    let a = dir.path().join("a.json");
    let b = dir.path().join("b.json");
    let c = dir.path().join("c.json");
    let out = dir.path().join("join.json");
    for (p, label) in [(&a, "one"), (&b, "two"), (&c, "three")] {
        write_session(p, label, 6000.0);
    }

    for args in [
        vec!["join", a.to_str().unwrap(), "--out", out.to_str().unwrap()],
        vec![
            "join",
            a.to_str().unwrap(),
            b.to_str().unwrap(),
            c.to_str().unwrap(),
            "--out",
            out.to_str().unwrap(),
        ],
    ] {
        let run = glbench(&args);
        assert!(!run.ok, "expected a rejection for {args:?}");
        assert!(
            run.stderr.contains("exactly 2 archive paths"),
            "the message must name the v3 constraint, got: {}",
            run.stderr
        );
        assert!(
            run.stderr.contains("N-way join"),
            "the message must point at the extension note, got: {}",
            run.stderr
        );
    }
    assert!(!out.exists());

    for p in [&a, &b, &c] {
        make_writable(p);
    }
}

#[test]
fn join_needs_an_out_path() {
    let dir = tempfile::tempdir().unwrap();
    let a = dir.path().join("a.json");
    let b = dir.path().join("b.json");
    write_session(&a, "one", 6000.0);
    write_session(&b, "two", 5000.0);

    let run = glbench(&["join", a.to_str().unwrap(), b.to_str().unwrap()]);
    assert!(!run.ok);
    assert!(run.stderr.contains("--out"), "stderr: {}", run.stderr);

    for p in [&a, &b] {
        make_writable(p);
    }
}

#[test]
fn inspect_reports_a_modified_session_and_still_renders_it() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("run.json");
    write_session(&path, "tampered", 6200.0);

    make_writable(&path);
    let text = fs::read_to_string(&path).unwrap();
    fs::write(&path, text.replace("\"prompt_tokens\": 96", "\"prompt_tokens\": 97")).unwrap();

    let run = glbench(&["inspect", path.to_str().unwrap()]);
    assert!(!run.ok, "a digest mismatch must set a non-zero exit code");
    assert!(run.stderr.contains("digest mismatch"), "stderr: {}", run.stderr);
    // Still rendered — the user needs to see what changed.
    assert!(!run.stdout.is_empty(), "the session must still be printed");

    // --no-verify skips the check entirely.
    let run = glbench(&["inspect", path.to_str().unwrap(), "--no-verify"]);
    assert!(run.ok, "stderr: {}", run.stderr);
    assert!(!run.stderr.contains("digest mismatch"));
}

#[test]
fn validate_availability_passes_on_an_archive_this_build_wrote() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("run.json");
    write_session(&path, "annotated", 6200.0);

    let run = glbench(&["validate", "--availability", path.to_str().unwrap()]);
    assert!(run.ok, "stdout: {} stderr: {}", run.stdout, run.stderr);
    assert!(
        run.stdout.contains("every null carries an availability status"),
        "stdout: {}",
        run.stdout
    );

    make_writable(&path);
}

/// A v1 archive predates the vocabulary, so every one of its nulls is
/// unexplained. That is the check working, not the archive being broken — and
/// it is exactly what `validate --availability` exists to report.
#[test]
fn validate_availability_names_every_unexplained_null_in_a_v1_archive() {
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/v1_archive.json");
    let run = glbench(&["validate", "--availability", fixture.to_str().unwrap()]);

    assert!(!run.ok, "a v1 archive cannot satisfy D-10");
    for path in [
        "telemetry",
        "behavior",
        "measurements.model_bytes",
        "environment.hardware.gpu.name",
    ] {
        assert!(
            run.stdout.contains(path),
            "the report must name '{path}', stdout: {}",
            run.stdout
        );
    }
}

#[test]
fn null_semantics_rejects_a_value_that_is_neither_strict_nor_lenient() {
    let run = glbench(&[
        "run",
        "--engine",
        "glproc",
        "--model",
        "nonexistent.gguf",
        "--null-semantics",
        "relaxed",
    ]);
    assert!(!run.ok);
    assert!(
        run.stderr.contains("strict|lenient"),
        "stderr: {}",
        run.stderr
    );
}
