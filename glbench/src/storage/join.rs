//! [`VLJoinManifest`] — a comparison of two archives, as a third file (D-18).
//!
//! # Why a third file rather than a field
//!
//! The obvious design is to write the comparison back into one of the two
//! sessions. That makes the session a moving target: it was sealed with a
//! digest, and re-writing it either invalidates that digest or forces a
//! re-seal, at which point "this archive is the run I measured" stops being
//! true.
//!
//! So a join references its sources and never opens either for writing. It
//! records each source's content digest, and `glbench inspect` on the join
//! re-verifies them — which is the entire reason the digests are stored. Three
//! files for one logical comparison is the cost; two independently verifiable
//! source archives is what it buys.
//!
//! # Why `sources` is a `Vec` for a thing that is always two
//!
//! An N-way join is a schema-compatible extension of a list and a schema break
//! for a pair. It costs one field type now. v3 rejects anything other than
//! exactly two, at the CLI, with a message that says so.

use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::comparison::regression::Regression;
use crate::comparison::runs::{self, ComparisonReport, Delta};
use crate::core::availability::{self, ENAvailability, VLAvailabilityMap};
use crate::core::mode::ENSessionMode;
use crate::core::schema::{ToJson, GLBENCH_VERSION, SCHEMA_VERSION};
use crate::export::json::{self, Json};
use crate::storage::archive;
use crate::storage::digest::{self, DigestError, VLIntegrity};
use crate::validation::integrity::{Severity, ValidationReport};

/// How many sources a v3 join takes.
pub const V3_SOURCE_COUNT: usize = 2;

/// One side of a join.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VLJoinSource {
    /// The path as given on the command line.
    pub path: String,
    /// The source's own `metadata.label`.
    pub label: String,
    /// The source's content digest, or `None` for a v1 archive that has none.
    /// `None` is why drift on that source cannot be detected, and the join says
    /// so through its availability map rather than by silently skipping it.
    pub digest: Option<String>,
    /// The source's session mode.
    pub session_mode: ENSessionMode,
}

impl VLJoinSource {
    /// Parse back from JSON.
    pub fn from_json(v: &Json) -> Result<VLJoinSource, String> {
        let path = v
            .get("path")
            .and_then(|p| p.as_str())
            .ok_or_else(|| "join source has no 'path'".to_string())?
            .to_string();
        let label = v
            .get("label")
            .and_then(|l| l.as_str())
            .unwrap_or_default()
            .to_string();
        let digest = v.get("digest").and_then(|d| d.as_str()).map(String::from);
        let mode_str = v
            .get("session_mode")
            .and_then(|m| m.as_str())
            .unwrap_or("inference_only");
        let session_mode = ENSessionMode::from_str(mode_str)
            .ok_or_else(|| format!("unknown session_mode '{mode_str}' in join source"))?;
        Ok(VLJoinSource { path, label, digest, session_mode })
    }
}

impl ToJson for VLJoinSource {
    fn to_json(&self) -> Json {
        Json::obj([
            ("path", Json::s(self.path.clone())),
            ("label", Json::s(self.label.clone())),
            (
                "digest",
                self.digest.clone().map(Json::s).unwrap_or(Json::Null),
            ),
            ("session_mode", Json::s(self.session_mode.as_str())),
        ])
    }
}

/// A comparison of two archives, with enough recorded about each source to
/// detect that one has changed since.
#[derive(Debug, Clone)]
pub struct VLJoinManifest {
    /// Archive schema version, shared with [`crate::core::session::BenchmarkSession`].
    pub schema_version: u32,
    /// Unix epoch seconds when the join was created.
    pub created_unix: u64,
    /// glbench version that produced it.
    pub glbench_version: String,
    /// Human label for the join itself.
    pub label: String,
    /// The joined sources. Exactly two in v3.
    pub sources: Vec<VLJoinSource>,
    /// The comparison over them.
    pub comparison: ComparisonReport,
    /// Why any `null` here has no value (D-09).
    pub availability: VLAvailabilityMap,
    /// The join's own content digest — a join is an archive too.
    pub integrity: Option<VLIntegrity>,
}

impl VLJoinManifest {
    /// JSON projection, before sealing.
    pub fn to_json(&self) -> Json {
        Json::obj([
            ("schema_version", Json::n(self.schema_version as f64)),
            ("created_unix", Json::n(self.created_unix as f64)),
            ("glbench_version", Json::s(self.glbench_version.clone())),
            ("label", Json::s(self.label.clone())),
            (
                "sources",
                Json::Arr(self.sources.iter().map(|s| s.to_json()).collect()),
            ),
            ("comparison", self.comparison.to_json()),
            ("availability", availability::to_json(&self.availability)),
            (
                "integrity",
                self.integrity.as_ref().map(|i| i.to_json()).unwrap_or(Json::Null),
            ),
        ])
    }

    /// Parse back from JSON.
    pub fn from_json(v: &Json) -> Result<VLJoinManifest, String> {
        let sources_json = v
            .get("sources")
            .and_then(|s| s.as_arr())
            .ok_or_else(|| "join manifest has no 'sources' array".to_string())?;
        let mut sources = Vec::with_capacity(sources_json.len());
        for (i, source) in sources_json.iter().enumerate() {
            sources.push(VLJoinSource::from_json(source).map_err(|e| format!("sources[{i}]: {e}"))?);
        }

        Ok(VLJoinManifest {
            schema_version: v
                .get("schema_version")
                .and_then(|n| n.as_f64())
                .unwrap_or(SCHEMA_VERSION as f64) as u32,
            created_unix: v.get("created_unix").and_then(|n| n.as_f64()).unwrap_or(0.0) as u64,
            glbench_version: v
                .get("glbench_version")
                .and_then(|s| s.as_str())
                .unwrap_or_default()
                .to_string(),
            label: v.get("label").and_then(|s| s.as_str()).unwrap_or_default().to_string(),
            sources,
            comparison: comparison_from_json(v.get("comparison"))?,
            availability: availability::from_json(v.get("availability"))?,
            integrity: match v.get("integrity") {
                Some(i) if !matches!(i, Json::Null) => Some(VLIntegrity::from_json(i)?),
                _ => None,
            },
        })
    }
}

/// Reconstruct the comparison from JSON.
///
/// Local rather than a `FromJson` impl on [`ComparisonReport`], matching how
/// `session.rs` reconstructs engine and environment: only the fields a reader
/// of a join actually needs are restored, and the type stays free of a parser
/// nothing else asks for.
fn comparison_from_json(v: Option<&Json>) -> Result<ComparisonReport, String> {
    let v = v.ok_or_else(|| "join manifest has no 'comparison'".to_string())?;
    let delta = |key: &str| -> Delta {
        let d = v.get(key);
        Delta {
            baseline: d.and_then(|x| x.get("baseline")).and_then(|n| n.as_f64()).unwrap_or(0.0),
            candidate: d.and_then(|x| x.get("candidate")).and_then(|n| n.as_f64()).unwrap_or(0.0),
        }
    };
    let regression = v
        .get("regression")
        .and_then(|r| r.as_str())
        .and_then(Regression::from_str)
        .unwrap_or(Regression::Neutral);

    Ok(ComparisonReport {
        baseline_label: v
            .get("baseline_label")
            .and_then(|s| s.as_str())
            .unwrap_or_default()
            .to_string(),
        candidate_label: v
            .get("candidate_label")
            .and_then(|s| s.as_str())
            .unwrap_or_default()
            .to_string(),
        decode_tps: delta("decode_tps"),
        prefill_tps: delta("prefill_tps"),
        regression,
        notes: v
            .get("notes")
            .and_then(|n| n.as_arr())
            .map(|items| items.iter().filter_map(|i| i.as_str().map(String::from)).collect())
            .unwrap_or_default(),
    })
}

/// Read two archives, compare them, and build the manifest.
///
/// Digest drift on a source is recorded as a finding, not an abort: the whole
/// point of noticing is to tell the user, and refusing to compare would leave
/// them with a mismatch they cannot look at.
pub fn build(
    a_path: &Path,
    b_path: &Path,
    label: Option<&str>,
    threshold: f64,
) -> Result<(VLJoinManifest, ValidationReport), String> {
    let mut report = ValidationReport::default();

    let (a_session, a_source) = read_source(a_path, &mut report)?;
    let (b_session, b_source) = read_source(b_path, &mut report)?;

    let comparison = runs::compare(&a_session, &b_session, threshold);
    let label = label
        .map(String::from)
        .unwrap_or_else(|| format!("{} vs {}", a_source.label, b_source.label));

    let mut map = VLAvailabilityMap::new();
    for (i, source) in [&a_source, &b_source].into_iter().enumerate() {
        if source.digest.is_none() {
            // One entry per index would collide under path normalisation
            // (`sources[]`), so say it once and name the file in the note.
            availability::set_with_note(
                &mut map,
                "sources[].digest",
                ENAvailability::DoesNotExist,
                "a v1 source archive carries no content digest, so drift on it cannot be detected",
            )?;
            report.push(
                Severity::Warning,
                "join",
                format!(
                    "source {} ({}) is a v1 archive with no digest — later modification to it \
                     will not be detectable from this join",
                    i, source.path
                ),
            );
        }
    }

    let manifest = VLJoinManifest {
        schema_version: SCHEMA_VERSION,
        created_unix: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
        glbench_version: GLBENCH_VERSION.to_string(),
        label,
        sources: vec![a_source, b_source],
        comparison,
        availability: map,
        integrity: None,
    };
    Ok((manifest, report))
}

/// Read one source archive, verifying it and recording what it is.
fn read_source(
    path: &Path,
    report: &mut ValidationReport,
) -> Result<(crate::core::session::BenchmarkSession, VLJoinSource), String> {
    let session = archive::read(path)?;
    // Record what the file says about itself, then check whether it is telling
    // the truth. Two separate questions: a source can carry a digest that no
    // longer matches its content, and the join wants to record the digest
    // either way so later drift is still measurable against something.
    let digest = match archive::recorded_digest(path) {
        Ok(d) => Some(d),
        Err(DigestError::Absent) => None,
        Err(e) => {
            report.push(Severity::Error, "join", format!("{}: {e}", path.display()));
            None
        }
    };
    if digest.is_some() {
        if let Err(e) = archive::verify_file(path) {
            report.push(Severity::Error, "join", format!("{}: {e}", path.display()));
        }
    }
    let source = VLJoinSource {
        path: path.display().to_string(),
        label: session.metadata.label.clone(),
        digest,
        session_mode: session.metadata.session_mode,
    };
    Ok((session, source))
}

/// Write a join manifest, sealing it with its own content digest.
pub fn write(manifest: &VLJoinManifest, path: &Path) -> Result<(), String> {
    let sealed = digest::seal(&manifest.to_json());
    archive::write_text(&sealed.to_pretty(), path)
}

/// Read a join manifest back.
pub fn read(path: &Path) -> Result<VLJoinManifest, String> {
    let text =
        std::fs::read_to_string(path).map_err(|e| format!("reading {}: {e}", path.display()))?;
    let value = json::parse(&text).map_err(|e| format!("parsing {}: {e}", path.display()))?;
    VLJoinManifest::from_json(&value).map_err(|e| format!("{}: {e}", path.display()))
}

/// Whether a parsed document is a join manifest rather than a session.
///
/// Keyed on `sources`, which a session never has and a join always does.
pub fn looks_like_join(value: &Json) -> bool {
    value.get("sources").and_then(|s| s.as_arr()).is_some()
}

/// Re-verify a join's sources against the digests it recorded.
///
/// This is what makes recording them worth anything: a source edited after the
/// join was written shows up here as an `Error` finding rather than as a
/// comparison quietly describing a file that no longer exists in that form.
pub fn verify_sources(manifest: &VLJoinManifest) -> ValidationReport {
    let mut report = ValidationReport::default();
    for source in &manifest.sources {
        let Some(recorded) = &source.digest else {
            report.push(
                Severity::Info,
                "join",
                format!(
                    "{}: v1 source, no digest was recorded, so drift on it is undetectable",
                    source.path
                ),
            );
            continue;
        };
        let path = Path::new(&source.path);
        match archive::recorded_digest(path) {
            Ok(actual) if &actual == recorded => {
                // Same digest recorded — now confirm the file still hashes to it,
                // which catches an edit that left the digest field alone.
                if let Err(e) = archive::verify_file(path) {
                    report.push(Severity::Error, "join", format!("{}: {e}", source.path));
                }
            }
            Ok(actual) => report.push(
                Severity::Error,
                "join",
                format!(
                    "{}: source has changed since the join was written \
                     (join records {recorded}, file now records {actual})",
                    source.path
                ),
            ),
            Err(DigestError::Absent) => report.push(
                Severity::Error,
                "join",
                format!(
                    "{}: source no longer carries a digest, but the join recorded {recorded}",
                    source.path
                ),
            ),
            Err(e) => report.push(Severity::Error, "join", format!("{}: {e}", source.path)),
        }
    }
    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::metrics::{IterationMetrics, MeasurementSet};
    use crate::core::result::SessionMetadata;
    use crate::core::session::BenchmarkSession;
    use crate::core::workload::WorkloadSpec;
    use crate::engine::metadata::EngineMetadata;
    use crate::environment::hardware::EnvironmentSnapshot;

    fn session(label: &str, decode_ms: f64) -> BenchmarkSession {
        let mut m = MeasurementSet::default();
        m.iterations.push(IterationMetrics {
            prompt_tokens: 100,
            generated_tokens: 128,
            prefill_ms: 100.0,
            decode_ms,
            total_ms: 100.0 + decode_ms,
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

    /// Write two sealed source archives into a fresh directory.
    fn two_sources(dir: &Path) -> (std::path::PathBuf, std::path::PathBuf) {
        let a = dir.join("a.json");
        let b = dir.join("b.json");
        archive::write(&session("baseline", 4000.0), &a).unwrap();
        archive::write(&session("candidate", 3200.0), &b).unwrap();
        (a, b)
    }

    #[test]
    fn a_join_records_both_sources_with_their_digests_and_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let (a, b) = two_sources(dir.path());

        let (manifest, report) = build(&a, &b, Some("q8-vs-q4"), 0.05).unwrap();
        assert!(report.passed(), "findings: {:?}", report.findings);
        assert_eq!(manifest.sources.len(), V3_SOURCE_COUNT);
        assert_eq!(manifest.label, "q8-vs-q4");
        assert_eq!(manifest.sources[0].label, "baseline");
        assert_eq!(manifest.sources[1].label, "candidate");
        assert!(manifest.sources.iter().all(|s| s.digest.is_some()));

        let out = dir.path().join("join.json");
        write(&manifest, &out).unwrap();
        let back = read(&out).unwrap();
        assert_eq!(back.label, "q8-vs-q4");
        assert_eq!(back.sources, manifest.sources);
        assert_eq!(back.comparison.baseline_label, manifest.comparison.baseline_label);
        assert_eq!(back.comparison.decode_tps.baseline, manifest.comparison.decode_tps.baseline);
    }

    #[test]
    fn a_join_seals_itself_with_its_own_digest() {
        let dir = tempfile::tempdir().unwrap();
        let (a, b) = two_sources(dir.path());
        let (manifest, _) = build(&a, &b, None, 0.05).unwrap();
        let out = dir.path().join("join.json");
        write(&manifest, &out).unwrap();

        let text = std::fs::read_to_string(&out).unwrap();
        let value = json::parse(&text).unwrap();
        assert_eq!(digest::verify(&value), Ok(()));
    }

    #[test]
    fn neither_source_is_modified_by_a_join() {
        let dir = tempfile::tempdir().unwrap();
        let (a, b) = two_sources(dir.path());
        let before = (
            std::fs::read_to_string(&a).unwrap(),
            std::fs::read_to_string(&b).unwrap(),
        );

        let (manifest, _) = build(&a, &b, None, 0.05).unwrap();
        write(&manifest, &dir.path().join("join.json")).unwrap();

        assert_eq!(std::fs::read_to_string(&a).unwrap(), before.0);
        assert_eq!(std::fs::read_to_string(&b).unwrap(), before.1);
    }

    #[test]
    fn source_verification_passes_while_the_sources_are_untouched() {
        let dir = tempfile::tempdir().unwrap();
        let (a, b) = two_sources(dir.path());
        let (manifest, _) = build(&a, &b, None, 0.05).unwrap();
        let report = verify_sources(&manifest);
        assert!(report.passed(), "findings: {:?}", report.findings);
    }

    /// The whole reason the digests are recorded.
    #[test]
    fn a_source_edited_after_the_join_is_reported_as_drift() {
        let dir = tempfile::tempdir().unwrap();
        let (a, b) = two_sources(dir.path());
        let (manifest, _) = build(&a, &b, None, 0.05).unwrap();

        // Re-measure the baseline and overwrite it: same path, different run.
        archive::write(&session("baseline", 9999.0), &a).unwrap();

        let report = verify_sources(&manifest);
        assert!(!report.passed(), "drift must be an error");
        let msg = &report.findings[0].message;
        assert!(msg.contains("has changed since the join was written"), "got {msg}");
        assert!(msg.contains("a.json"), "the finding must name the file, got {msg}");
    }

    #[test]
    fn a_join_is_distinguishable_from_a_session_by_its_shape() {
        let dir = tempfile::tempdir().unwrap();
        let (a, b) = two_sources(dir.path());
        let (manifest, _) = build(&a, &b, None, 0.05).unwrap();

        assert!(looks_like_join(&manifest.to_json()));
        assert!(!looks_like_join(&session("plain", 1.0).to_json()));
    }

    #[test]
    fn a_v1_source_with_no_digest_is_recorded_as_such_rather_than_skipped() {
        let dir = tempfile::tempdir().unwrap();
        let (a, b) = two_sources(dir.path());

        // Strip b's integrity block, making it a v1-shaped archive.
        let text = std::fs::read_to_string(&b).unwrap();
        let mut root = json::parse(&text).unwrap().as_obj().unwrap().clone();
        root.remove("integrity");
        let mut metadata = root.get("metadata").unwrap().as_obj().unwrap().clone();
        metadata.insert("schema_version".to_string(), Json::n(1.0));
        root.insert("metadata".to_string(), Json::Obj(metadata));
        archive::write_text(&Json::Obj(root).to_pretty(), &b).unwrap();

        let (manifest, report) = build(&a, &b, None, 0.05).unwrap();
        assert!(manifest.sources[0].digest.is_some());
        assert!(manifest.sources[1].digest.is_none());
        // Recorded honestly in both the map and the findings.
        assert_eq!(
            manifest.availability.get("sources[].digest").map(|e| e.status),
            Some(ENAvailability::DoesNotExist)
        );
        assert!(report.findings.iter().any(|f| f.message.contains("no digest")));
        // Still only a warning: a v1 source is comparable, just not verifiable.
        assert!(report.passed());

        let sources_report = verify_sources(&manifest);
        assert!(sources_report.passed());
        assert!(sources_report
            .findings
            .iter()
            .any(|f| f.message.contains("undetectable")));
    }

    #[test]
    fn the_default_label_names_both_sides() {
        let dir = tempfile::tempdir().unwrap();
        let (a, b) = two_sources(dir.path());
        let (manifest, _) = build(&a, &b, None, 0.05).unwrap();
        assert_eq!(manifest.label, "baseline vs candidate");
    }
}
