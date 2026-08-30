//! Gate 4: a real LoRA training run archives as a valid v2 session.
//!
//! Everything here runs actual training on gltrain — no fixtures, no mocks. The
//! claim being tested is the one Gate 4 makes: that a run produces an archive
//! which passes the D-10 null-semantics check in **strict** mode, with every
//! field M2 cannot fill carrying an honest status rather than a zero.
//!
//! Requires `--features train-bench`.

use std::fs;
use std::path::Path;

use glbench::core::availability::ENAvailability;
use glbench::core::mode::{ENInferenceRole, ENSessionMode};
use glbench::export::json;
use glbench::numerical::scope::ENBitScope;
use glbench::storage::archive;
use glbench::training::runner::{self, TrainArgs};
use glbench::validation::availability::{self, ENNullSemantics};

/// Small enough to keep the suite fast, large enough that the adapter has a
/// real shape and the loss actually moves.
fn args() -> TrainArgs {
    TrainArgs {
        d_in: 8,
        d_out: 8,
        rank: 2,
        samples: 4,
        epochs: 3,
        ..TrainArgs::default()
    }
}

#[allow(clippy::permissions_set_readonly_false)]
fn make_writable(path: &Path) {
    let mut perms = fs::metadata(path).unwrap().permissions();
    perms.set_readonly(false);
    fs::set_permissions(path, perms).unwrap();
}

/// The headline Gate 4 claim.
#[test]
fn a_real_training_run_writes_an_archive_that_passes_strict_d10() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("train.json");

    let session = runner::run(&args()).expect("training runs");
    // Strict is the default: if any null lacked a status, this returns Err and
    // writes nothing.
    archive::write(&session, &path).expect("a training session must archive under strict D-10");

    let text = fs::read_to_string(&path).unwrap();
    let report = availability::check(&text);
    assert!(
        report.passed(),
        "the written archive must satisfy D-10 on its own: {:?}",
        report.findings
    );

    make_writable(&path);
}

/// F-05: the fields gltrain M2 cannot fill must be present, null, and
/// explained — not omitted, and never zero.
#[test]
fn fields_with_no_subject_at_m2_carry_an_honest_status() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("train.json");
    let session = runner::run(&args()).expect("training runs");
    archive::write(&session, &path).unwrap();

    let value = json::parse(&fs::read_to_string(&path).unwrap()).unwrap();
    let map = glbench::core::availability::from_json(value.get("availability")).unwrap();

    // Per-step token and synchronisation fields: one entry each, covering the
    // whole column (D-09's array collapse).
    for path in ["training.steps[].tokens", "training.steps[].sync_ms"] {
        let entry = map
            .get(path)
            .unwrap_or_else(|| panic!("{path} must be declared"));
        assert_eq!(
            entry.status,
            ENAvailability::NotApplicable,
            "{path} has no subject at M2, so it is not_applicable"
        );
    }

    // And they really are null in the data, not absent and not 0.
    let steps = value
        .get("training")
        .and_then(|t| t.get("steps"))
        .and_then(|s| s.as_arr())
        .expect("steps array");
    assert!(!steps.is_empty());
    for step in steps {
        for field in ["tokens", "sync_ms"] {
            assert!(
                matches!(step.get(field), Some(json::Json::Null)),
                "{field} must be null in every step"
            );
        }
    }

    make_writable(&path);
}

/// §6.2's mode table, on a session produced by a real run rather than by hand.
#[test]
fn a_training_only_session_satisfies_the_mode_table() {
    let session = runner::run(&args()).expect("training runs");
    assert_eq!(session.metadata.session_mode, ENSessionMode::TrainingOnly);
    assert!(session.inference.is_none());
    assert!(session.training.is_some());

    let report = session.check_mode_consistency();
    assert!(report.passed(), "findings: {:?}", report.findings);
}

/// Gate 4: a unified session has both inference roles populated and labelled.
#[test]
fn a_unified_session_carries_both_roles_and_archives_cleanly() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("unified.json");

    let mut a = args();
    a.mode = ENSessionMode::Unified;
    let session = runner::run(&a).expect("training runs");

    assert_eq!(
        session.inference.as_ref().map(|i| i.role),
        Some(ENInferenceRole::PreTraining),
    );
    assert_eq!(
        session.training.as_ref().unwrap().post_eval.as_ref().map(|p| p.role),
        Some(ENInferenceRole::PostTraining),
    );
    assert!(session.check_mode_consistency().passed());

    archive::write(&session, &path).expect("a unified session must archive");
    let text = fs::read_to_string(&path).unwrap();
    assert!(availability::check(&text).passed());

    // Both roles are readable from the JSON alone — position is never
    // load-bearing (D-08).
    let value = json::parse(&text).unwrap();
    assert_eq!(
        value.get("inference").and_then(|i| i.get("role")).and_then(|r| r.as_str()),
        Some("pre_training")
    );
    assert_eq!(
        value
            .get("training")
            .and_then(|t| t.get("post_eval"))
            .and_then(|p| p.get("role"))
            .and_then(|r| r.as_str()),
        Some("post_training")
    );

    make_writable(&path);
}

/// Research §14: convergence numbers are reported with their window and
/// threshold, or they are opinions with a number attached.
#[test]
fn convergence_is_archived_with_the_window_and_threshold_it_used() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("train.json");
    let mut a = args();
    a.target_loss = Some(0.5);
    let session = runner::run(&a).expect("training runs");
    archive::write(&session, &path).unwrap();

    let value = json::parse(&fs::read_to_string(&path).unwrap()).unwrap();
    let c = value
        .get("training")
        .and_then(|t| t.get("convergence"))
        .expect("convergence present");

    for parameter in ["ema_alpha", "plateau_window", "plateau_threshold", "cv_window"] {
        assert!(
            c.get(parameter).is_some(),
            "{parameter} must be archived next to the number it produced"
        );
    }
    assert_eq!(c.get("target_loss").and_then(|t| t.as_f64()), Some(0.5));
    // The run either reached it or it did not; either way the field exists.
    assert!(c.get("steps_to_target").is_some());

    make_writable(&path);
}

/// A target nobody asked for must not silently become a default.
#[test]
fn no_target_loss_leaves_the_target_fields_null_and_explained() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("train.json");
    let session = runner::run(&args()).expect("training runs");
    archive::write(&session, &path).unwrap();

    let text = fs::read_to_string(&path).unwrap();
    let value = json::parse(&text).unwrap();
    let c = value.get("training").and_then(|t| t.get("convergence")).unwrap();

    assert!(matches!(c.get("target_loss"), Some(json::Json::Null)));
    assert!(matches!(c.get("steps_to_target"), Some(json::Json::Null)));

    let map = glbench::core::availability::from_json(value.get("availability")).unwrap();
    assert_eq!(
        map.get("training.convergence.target_loss").map(|e| e.status),
        Some(ENAvailability::NotApplicable),
        "nobody asked for a target, so it is not_applicable"
    );
    assert!(availability::check(&text).passed());

    make_writable(&path);
}

/// D-19 through the whole pipeline: a thinned archive says it is thinned.
#[test]
fn a_sampled_run_archives_the_sampling_it_used() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("train.json");

    let mut a = args();
    a.samples = 5;
    a.epochs = 2;
    a.step_sample_n = 4;
    let session = runner::run(&a).expect("training runs");
    archive::write(&session, &path).unwrap();

    let value = json::parse(&fs::read_to_string(&path).unwrap()).unwrap();
    let t = value.get("training").unwrap();
    assert_eq!(t.get("steps_observed").and_then(|v| v.as_f64()), Some(10.0));
    assert_eq!(t.get("step_sample_n").and_then(|v| v.as_f64()), Some(4.0));
    let archived = t.get("steps").and_then(|s| s.as_arr()).unwrap().len();
    assert!(archived < 10, "sampling must thin the array");
    assert_eq!(
        t.get("steps_archived").and_then(|v| v.as_f64()),
        Some(archived as f64),
        "the count must match the array a reader can see"
    );

    make_writable(&path);
}

/// Gradient bit profiling end to end, on gradients from a real backward pass.
#[test]
fn a_gradient_bit_scope_archives_profiles_from_real_gradients() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("train.json");

    let mut a = args();
    a.bit_scopes = vec![ENBitScope::Gradients, ENBitScope::Optimizer];
    let session = runner::run(&a).expect("training runs");
    archive::write(&session, &path).unwrap();

    let text = fs::read_to_string(&path).unwrap();
    assert!(availability::check(&text).passed(), "bit profiles must not break D-10");

    let value = json::parse(&text).unwrap();
    let profiles = value
        .get("training")
        .and_then(|t| t.get("bit_profiles"))
        .and_then(|b| b.as_arr())
        .expect("bit_profiles array");
    assert!(!profiles.is_empty());

    let scopes: Vec<&str> = profiles
        .iter()
        .filter_map(|p| p.get("scope").and_then(|s| s.as_str()))
        .collect();
    assert!(scopes.contains(&"gradients"));
    assert!(scopes.contains(&"optimizer"));

    // Every profile says which step it describes — gradients change each step,
    // so an untagged one measures an unknown moment.
    assert!(profiles.iter().all(|p| p.get("step_index").is_some()));

    // A profile of real f32 data has a real element count.
    assert!(profiles
        .iter()
        .all(|p| p.get("count").and_then(|c| c.as_f64()).unwrap_or(0.0) > 0.0));

    make_writable(&path);
}

/// A dropped entry is healed by the finalisation annotator — but with the
/// *fallback* status, not the one the runner declared.
///
/// Worth pinning because the difference is easy to miss and matters. The runner
/// declares `training.steps[].tokens` as `not_applicable` (F-05: tokens have no
/// subject at M2). If that declaration goes missing, `classify_null`'s generic
/// training-subtree rule fills in `unavailable` instead — a weaker and vaguer
/// claim that still satisfies D-10. The archive stays valid; it just says less.
#[test]
fn a_dropped_entry_is_healed_by_annotation_but_with_the_weaker_fallback_status() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("train.json");

    let mut session = runner::run(&args()).expect("training runs");
    assert_eq!(
        session.availability.get("training.steps[].tokens").map(|e| e.status),
        Some(ENAvailability::NotApplicable),
        "the runner declares the precise status"
    );
    session.availability.remove("training.steps[].tokens");

    // Still writes: annotation refills the gap rather than failing.
    archive::write(&session, &path).expect("annotation heals a dropped entry");
    let text = fs::read_to_string(&path).unwrap();
    assert!(availability::check(&text).passed());

    let value = json::parse(&text).unwrap();
    let map = glbench::core::availability::from_json(value.get("availability")).unwrap();
    assert_eq!(
        map.get("training.steps[].tokens").map(|e| e.status),
        Some(ENAvailability::Unavailable),
        "the fallback is honest but vaguer than the declaration it replaced"
    );

    make_writable(&path);
}

/// The violation annotation *cannot* heal: a status on a field that carries a
/// value. Strict refuses it; lenient writes and reports it.
#[test]
fn strict_mode_refuses_a_status_claiming_a_populated_field_is_absent() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("train.json");

    let mut session = runner::run(&args()).expect("training runs");
    // `steps_observed` is a count the run definitely produced.
    glbench::core::availability::set(
        &mut session.availability,
        "training.steps_observed",
        ENAvailability::Unsupported,
    )
    .unwrap();

    let err = archive::write(&session, &path).unwrap_err();
    assert!(err.contains("training.steps_observed"), "got {err}");
    assert!(err.contains("carries a value"), "got {err}");
    assert!(!path.exists(), "nothing may be written when finalisation fails");

    // Lenient writes it anyway, and says what was wrong.
    let report =
        archive::write_with_policy(&session, &path, ENNullSemantics::Lenient).unwrap();
    assert!(path.exists());
    assert!(report
        .findings
        .iter()
        .any(|f| f.message.contains("training.steps_observed")));

    make_writable(&path);
}

// ---------------------------------------------------------------------------
// Wave 5 — rendering and export
// ---------------------------------------------------------------------------

/// Facts survive a round trip; derivations are recomputed from them.
#[test]
fn a_training_archive_reads_back_its_steps_and_re_derives_its_reports() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("train.json");
    let session = runner::run(&args()).expect("training runs");
    let original = session.training.as_ref().unwrap().clone();
    archive::write(&session, &path).unwrap();

    let back = archive::read(&path).unwrap();
    let t = back.training.as_ref().expect("training reconstructed");

    // Facts, verbatim.
    assert_eq!(t.steps.len(), original.steps.len());
    assert_eq!(t.steps, original.steps);
    assert_eq!(t.steps_observed, original.steps_observed);
    assert_eq!(t.step_sample_n, original.step_sample_n);
    assert_eq!(t.optimizer, original.optimizer);
    assert_eq!(t.epoch_losses, original.epoch_losses);

    // Derivations, recomputed — and they must agree with the originals,
    // because re-deriving over the same measurements is deterministic.
    let a = t.attribution.as_ref().expect("attribution re-derived");
    let a0 = original.attribution.as_ref().unwrap();
    assert_eq!(a.steps, a0.steps);
    assert!((a.total_ms - a0.total_ms).abs() < 1e-9);

    let c = t.convergence.as_ref().expect("convergence re-derived");
    let c0 = original.convergence.as_ref().unwrap();
    assert!((c.final_loss - c0.final_loss).abs() < 1e-9);
    assert!((c.slope_per_step - c0.slope_per_step).abs() < 1e-9);

    make_writable(&path);
}

/// `glbench export --format training-csv`, through the real archive.
#[test]
fn training_steps_export_as_csv_with_real_step_indices() {
    use glbench::export::csv;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("train.json");
    let mut a = args();
    a.samples = 5;
    a.epochs = 2;
    a.step_sample_n = 4;
    let session = runner::run(&a).expect("training runs");
    archive::write(&session, &path).unwrap();

    let back = archive::read(&path).unwrap();
    let rows = csv::render_training(&back);
    assert!(rows.starts_with(csv::TRAINING_HEADER), "{rows}");

    let lines: Vec<&str> = rows.lines().skip(1).collect();
    assert_eq!(lines.len(), 4, "one row per archived step: {rows}");

    // The `step` column carries the real index, not a row number — a thinned
    // series must plot against the right x-axis.
    let indices: Vec<&str> = lines.iter().map(|l| l.split(',').nth(1).unwrap()).collect();
    assert_eq!(indices, vec!["0", "4", "8", "9"]);

    make_writable(&path);
}

#[test]
fn a_session_with_no_training_exports_no_training_csv() {
    use glbench::export::csv;
    use glbench::core::metrics::MeasurementSet;
    use glbench::core::result::SessionMetadata;
    use glbench::core::session::BenchmarkSession;
    use glbench::core::workload::WorkloadSpec;
    use glbench::engine::metadata::EngineMetadata;
    use glbench::environment::hardware::EnvironmentSnapshot;

    let plain = BenchmarkSession::new(
        SessionMetadata::new("inference"),
        EnvironmentSnapshot::probe(""),
        EngineMetadata {
            name: "glproc".into(),
            backend: "cpu".into(),
            available: true,
            model_arch: None,
            quantization: None,
            thinking_capable: None,
        },
        WorkloadSpec::default(),
        MeasurementSet::default(),
    );
    assert!(
        csv::render_training(&plain).is_empty(),
        "a header with no rows would suggest a run that trained nothing"
    );
}

/// The Markdown report must carry the training numbers *and* the caveats that
/// make them readable (Gate 5).
#[test]
fn the_markdown_report_carries_the_training_sections_and_their_caveats() {
    use glbench::export::markdown;

    let mut a = args();
    a.target_loss = Some(0.5);
    let session = runner::run(&a).expect("training runs");
    let md = markdown::render(&session);

    for heading in [
        "## Training",
        "### Epoch loss",
        "### Loss curve",
        "### Convergence",
        "### Step time",
        "### Memory",
    ] {
        assert!(md.contains(heading), "missing {heading}");
    }
    // Sampling is stated even at N=1, so a reader never infers completeness.
    assert!(md.contains("sample N="), "{md}");
    // A plateau claim without its window and threshold is an opinion.
    assert!(md.contains("relative threshold"), "{md}");
    assert!(md.contains("Stability (CV)"), "{md}");
    // The availability map is rendered, so nulls are explained in the report
    // and not only in the JSON.
    assert!(md.contains("## Availability"), "{md}");
    assert!(md.contains("not_applicable"), "{md}");
}

/// A target nobody set must not read as a target of zero.
#[test]
fn the_markdown_report_says_no_target_was_given_rather_than_showing_one() {
    use glbench::export::markdown;
    let session = runner::run(&args()).expect("training runs");
    let md = markdown::render(&session);
    assert!(
        md.contains("`--target-loss` has no default"),
        "the report must explain the absence: {md}"
    );
}

/// The archive is sealed like any other v2 session — training does not get a
/// weaker integrity guarantee.
#[test]
fn a_training_archive_is_sealed_and_verifies() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("train.json");
    let session = runner::run(&args()).expect("training runs");
    archive::write(&session, &path).unwrap();

    let (back, report) = archive::read_verified(&path, true).unwrap();
    assert!(report.passed(), "findings: {:?}", report.findings);
    assert!(report.findings.is_empty(), "a sealed archive verifies silently");
    assert_eq!(back.metadata.session_mode, ENSessionMode::TrainingOnly);
    assert!(back.integrity.is_some());

    make_writable(&path);
}
