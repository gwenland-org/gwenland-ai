//! Reading and writing session archives.
//!
//! An archive is a single JSON file per session (STORAGE RULE: no database, no
//! cloud, user-managed files). This module is the read/write seam over the
//! session's JSON projection; it carries the schema/version stamp implicitly via
//! the session metadata, and refuses to read a file whose schema is newer than
//! this build understands.

use std::fs;
use std::path::Path;

use crate::core::schema::SCHEMA_VERSION;
use crate::core::session::BenchmarkSession;
use crate::export::json;

/// Write a session to `path` as pretty-printed JSON.
pub fn write(session: &BenchmarkSession, path: &Path) -> Result<(), String> {
    let text = session.to_json().to_pretty();
    fs::write(path, text).map_err(|e| format!("writing {}: {e}", path.display()))
}

/// Read a session back from a JSON archive at `path`.
pub fn read(path: &Path) -> Result<BenchmarkSession, String> {
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
    BenchmarkSession::from_json(&value).map_err(|e| format!("{}: {e}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::metrics::{IterationMetrics, MeasurementSet};
    use crate::core::result::SessionMetadata;
    use crate::core::workload::WorkloadSpec;
    use crate::engine::metadata::EngineMetadata;
    use crate::environment::hardware::EnvironmentSnapshot;

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
}
