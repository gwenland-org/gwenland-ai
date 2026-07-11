//! [`BenchmarkSession`] — the single source of truth.
//!
//! This is a *data model only*: no business logic lives here. Every subsystem
//! reads or fills a field of a session, and every renderer/exporter consumes
//! one. The runner produces the metadata/environment/engine/workload/measurement
//! fields; analysis/comparison/validation fill their report fields; export and
//! render turn the whole thing into bytes. Keeping logic out of this struct is
//! what lets the pipeline stages compose without coupling.

use crate::analysis::summary::AnalysisReport;
use crate::comparison::runs::ComparisonReport;
use crate::core::metrics::MeasurementSet;
use crate::core::result::SessionMetadata;
use crate::core::schema::{field, ToJson};
use crate::core::workload::WorkloadSpec;
use crate::engine::metadata::EngineMetadata;
use crate::environment::hardware::EnvironmentSnapshot;
use crate::export::json::Json;
use crate::validation::integrity::ValidationReport;

/// The complete record of one benchmark run and everything derived from it.
#[derive(Debug, Clone)]
pub struct BenchmarkSession {
    /// Identifying header.
    pub metadata: SessionMetadata,
    /// Where it ran (machine + build).
    pub environment: EnvironmentSnapshot,
    /// Which engine and model.
    pub engine: EngineMetadata,
    /// What was run.
    pub workload: WorkloadSpec,
    /// Raw measured facts.
    pub measurements: MeasurementSet,
    /// Derived analysis (filled after the run; `None` until then).
    pub analysis: Option<AnalysisReport>,
    /// Comparison against another session (filled only by `compare`).
    pub comparison: Option<ComparisonReport>,
    /// Validation findings (filled after the run).
    pub validation: Option<ValidationReport>,
}

impl BenchmarkSession {
    /// Assemble a session from the facts the runner gathered. Reports start
    /// empty and are attached by the analysis/validation passes.
    pub fn new(
        metadata: SessionMetadata,
        environment: EnvironmentSnapshot,
        engine: EngineMetadata,
        workload: WorkloadSpec,
        measurements: MeasurementSet,
    ) -> BenchmarkSession {
        BenchmarkSession {
            metadata,
            environment,
            engine,
            workload,
            measurements,
            analysis: None,
            comparison: None,
            validation: None,
        }
    }

    /// The JSON projection of the whole session — the archive format.
    pub fn to_json(&self) -> Json {
        Json::obj([
            ("metadata", self.metadata.to_json()),
            ("environment", self.environment.to_json()),
            ("engine", self.engine.to_json()),
            ("workload", self.workload.to_json()),
            ("measurements", self.measurements.to_json()),
            ("analysis", opt(&self.analysis)),
            ("comparison", opt(&self.comparison)),
            ("validation", opt(&self.validation)),
        ])
    }

    /// Parse a session back from its JSON archive. Only the fields needed for
    /// comparison and re-rendering are reconstructed; derived reports are
    /// recomputed on demand rather than trusted from disk, so they are not
    /// required to be present.
    pub fn from_json(v: &Json) -> Result<BenchmarkSession, String> {
        use crate::core::metrics::MeasurementSet;
        use crate::core::result::SessionMetadata;
        use crate::core::schema::FromJson;
        use crate::core::workload::WorkloadSpec;

        let metadata = SessionMetadata::from_json(field(v, "metadata")?)?;
        let workload = WorkloadSpec::from_json(field(v, "workload")?)?;
        let measurements = MeasurementSet::from_json(field(v, "measurements")?)?;
        let engine = engine_from_json(field(v, "engine")?)?;
        let environment = environment_from_json(field(v, "environment")?)?;

        Ok(BenchmarkSession {
            metadata,
            environment,
            engine,
            workload,
            measurements,
            analysis: None,
            comparison: None,
            validation: None,
        })
    }
}

/// Encode an optional report as its JSON or null.
fn opt<T: ToJson>(v: &Option<T>) -> Json {
    match v {
        Some(inner) => inner.to_json(),
        None => Json::Null,
    }
}

/// Reconstruct engine metadata from JSON (the fields comparison needs).
fn engine_from_json(v: &Json) -> Result<EngineMetadata, String> {
    Ok(EngineMetadata {
        name: v.get("name").and_then(|s| s.as_str()).unwrap_or("").to_string(),
        backend: v.get("backend").and_then(|s| s.as_str()).unwrap_or("").to_string(),
        available: v.get("available").and_then(|b| b.as_bool()).unwrap_or(false),
        model_arch: v.get("model_arch").and_then(|s| s.as_str()).map(String::from),
        quantization: v.get("quantization").and_then(|s| s.as_str()).map(String::from),
    })
}

/// Reconstruct the environment snapshot from JSON. Only the fields the
/// comparison/analysis layers read are restored; the rest default.
fn environment_from_json(v: &Json) -> Result<EnvironmentSnapshot, String> {
    use crate::environment::cpu::CpuInfo;
    use crate::environment::gpu::GpuInfo;
    use crate::environment::hardware::HardwareSnapshot;
    use crate::environment::memory::MemoryInfo;
    use crate::environment::runtime::RuntimeInfo;
    use crate::environment::storage::StorageInfo;

    let hw = v.get("hardware");
    let cpu = hw.and_then(|h| h.get("cpu"));
    let gpu = hw.and_then(|h| h.get("gpu"));
    let storage = hw.and_then(|h| h.get("storage"));
    let rt = v.get("runtime");

    let hardware = HardwareSnapshot {
        cpu: CpuInfo {
            logical_cores: cpu.and_then(|c| c.get("logical_cores")).and_then(|n| n.as_f64()).unwrap_or(0.0)
                as usize,
            model: cpu.and_then(|c| c.get("model")).and_then(|s| s.as_str()).map(String::from),
            mhz: cpu.and_then(|c| c.get("mhz")).and_then(|n| n.as_f64()),
        },
        gpu: GpuInfo {
            name: gpu.and_then(|g| g.get("name")).and_then(|s| s.as_str()).map(String::from),
            backend: gpu.and_then(|g| g.get("backend")).and_then(|s| s.as_str()).map(String::from),
            compute: gpu.and_then(|g| g.get("compute")).and_then(|s| s.as_str()).map(String::from),
            total_memory_bytes: gpu
                .and_then(|g| g.get("total_memory_bytes"))
                .and_then(|n| n.as_f64())
                .map(|n| n as u64),
            peak_bandwidth_gbs: gpu.and_then(|g| g.get("peak_bandwidth_gbs")).and_then(|n| n.as_f64()),
            peak_compute_tops: gpu.and_then(|g| g.get("peak_compute_tops")).and_then(|n| n.as_f64()),
        },
        memory: MemoryInfo::default(),
        storage: StorageInfo {
            model_file_bytes: storage
                .and_then(|s| s.get("model_file_bytes"))
                .and_then(|n| n.as_f64())
                .map(|n| n as u64),
        },
    };

    let runtime = RuntimeInfo {
        os: rt.and_then(|r| r.get("os")).and_then(|s| s.as_str()).unwrap_or("").to_string(),
        arch: rt.and_then(|r| r.get("arch")).and_then(|s| s.as_str()).unwrap_or("").to_string(),
        glbench_version: rt
            .and_then(|r| r.get("glbench_version"))
            .and_then(|s| s.as_str())
            .unwrap_or("")
            .to_string(),
        build_profile: "unknown",
    };

    Ok(EnvironmentSnapshot { hardware, runtime })
}
