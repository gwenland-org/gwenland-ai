//! A lightweight manifest over a directory of archives.
//!
//! Not a database (STORAGE RULE) — this just enumerates the `.json` session
//! files in a user-managed directory and reads their headline facts, so the CLI
//! can list or trend over them without a persistent store. Each entry is read
//! lazily from disk on demand.

use std::fs;
use std::path::{Path, PathBuf};

use crate::comparison::statistics::Stats;
use crate::core::workload::WorkloadKind;
use crate::storage::archive;

/// A summarized entry for one archived session.
#[derive(Debug, Clone)]
pub struct ManifestEntry {
    /// Path to the archive file.
    pub path: PathBuf,
    /// Session label.
    pub label: String,
    /// Engine name.
    pub engine: String,
    /// Mean decode throughput, tokens/second. `None` for a prefill workload,
    /// whose single generated token is not a rate.
    pub decode_tps: Option<f64>,
    /// Mean prefill throughput, tokens/second.
    pub prefill_tps: f64,
    /// `production`, `instrumented` or `unknown`.
    pub measurement_mode: &'static str,
}

/// Enumerate and summarize every `.json` archive in `dir`, sorted by filename.
/// Files that fail to parse are skipped (a directory may hold unrelated JSON).
pub fn list(dir: &Path) -> Result<Vec<ManifestEntry>, String> {
    let mut paths: Vec<PathBuf> = fs::read_dir(dir)
        .map_err(|e| format!("reading dir {}: {e}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|x| x == "json").unwrap_or(false))
        .collect();
    paths.sort();

    let mut entries = Vec::new();
    for path in paths {
        if let Ok(session) = archive::read(&path) {
            let decode_tps = (session.workload.kind != WorkloadKind::Prefill)
                .then(|| Stats::from_samples(&session.measurements.decode_tps_samples()).mean);
            let prefill_tps = Stats::from_samples(&session.measurements.prefill_tps_samples()).mean;
            entries.push(ManifestEntry {
                path,
                label: session.metadata.label,
                engine: session.engine.name,
                decode_tps,
                prefill_tps,
                measurement_mode: session.metadata.measurement_mode.as_str(),
            });
        }
    }
    Ok(entries)
}
