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
    /// What the engine reported about its own internals: stage timings, kernel
    /// selection, memory split, MoE routing. `None` when the engine collects
    /// none — which means *not measured*, never *zero*.
    pub telemetry: Option<glcore::telemetry::EngineTelemetry>,
    /// What the model *did*: repetition, entropy, stalls, perplexity. Comes
    /// from a separate traced run (tracing perturbs timing, so it must not
    /// share a run with the measured iterations). `None` when not captured.
    pub behavior: Option<crate::behavior::BehaviorReport>,
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
            telemetry: None,
            behavior: None,
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
            (
                "telemetry",
                self.telemetry.as_ref().map(telemetry_json).unwrap_or(Json::Null),
            ),
            (
                "behavior",
                self.behavior.as_ref().map(behavior_json).unwrap_or(Json::Null),
            ),
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
            // Telemetry and behavior are not read back from the archive:
            // reconstructing them adds a parser that can only ever agree with
            // the writer. `inspect` re-renders the measured facts; the profile
            // and behavior sections are live-run views. Revisit if archives
            // need diffing (drift analysis will want exactly that).
            telemetry: None,
            behavior: None,
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

/// JSON projection of the behavioral signals — the CI-readable form.
///
/// Absent signals are written as `null`, never as zeros. A CI job asserting
/// "repetition ratio > 0.6" must fail loudly on a run that never measured
/// repetition, not silently pass on a fabricated 0.0.
fn behavior_json(b: &crate::behavior::BehaviorReport) -> Json {
    let rep = match &b.repetition {
        Some(r) => Json::obj([
            ("unique_1gram_ratio", Json::Num(r.unique_1gram_ratio)),
            ("unique_2gram_ratio", Json::Num(r.unique_2gram_ratio)),
            ("unique_3gram_ratio", Json::Num(r.unique_3gram_ratio)),
            ("max_token_run", Json::Num(r.max_token_run as f64)),
            ("looks_degenerate", Json::Bool(r.looks_degenerate())),
            ("tokens", Json::Num(r.tokens as f64)),
        ]),
        None => Json::Null,
    };
    let ent = match &b.entropy {
        Some(e) => Json::obj([
            ("mean_nats", Json::Num(e.mean)),
            ("std_dev", Json::Num(e.std_dev)),
            ("min", Json::Num(e.min)),
            ("max", Json::Num(e.max)),
            ("p50", Json::Num(e.p50)),
            ("p95", Json::Num(e.p95)),
            ("mean_top_prob", Json::Num(e.mean_top_prob)),
        ]),
        None => Json::Null,
    };
    let stall = match &b.stall {
        Some(s) => Json::obj([
            ("mean_ms", Json::Num(s.mean_ms)),
            ("std_dev_ms", Json::Num(s.std_dev_ms)),
            ("p50_ms", Json::Num(s.p50_ms)),
            ("p99_ms", Json::Num(s.p99_ms)),
            ("max_ms", Json::Num(s.max_ms)),
            ("stall_count", Json::Num(s.stall_count as f64)),
            ("jitter", Json::Num(s.jitter)),
        ]),
        None => Json::Null,
    };
    let ood = match &b.ood {
        Some(o) => Json::obj([
            ("perplexity", Json::Num(o.perplexity)),
            ("mean_logprob", Json::Num(o.mean_logprob)),
            ("min_logprob", Json::Num(o.min_logprob)),
            ("p95_surprise", Json::Num(o.p95_surprise)),
        ]),
        None => Json::Null,
    };
    let hall = match &b.hallucination {
        Some(h) => Json::obj([
            ("top_choice_rate", Json::Num(h.top_choice_rate)),
            ("mean_rank", Json::Num(h.mean_rank)),
            ("max_rank", Json::Num(h.max_rank as f64)),
            ("mean_confidence_gap", Json::Num(h.mean_confidence_gap)),
            ("uncertain_offpick_rate", Json::Num(h.uncertain_offpick_rate)),
        ]),
        None => Json::Null,
    };

    Json::obj([
        ("repetition", rep),
        ("entropy", ent),
        ("stall", stall),
        ("ood", ood),
        // Named for what it is. The struct's docs spell out that this is a
        // confidence/rank proxy and NOT a hallucination detector; the key is
        // kept honest here too.
        ("confidence_divergence", hall),
        // Toxicity is deliberately absent, not zero. See behavior::toxicity.
        ("toxicity", Json::Null),
    ])
}

/// JSON projection of the engine's telemetry.
///
/// Lives here, not in glcore, on purpose: glcore's telemetry module is a pure
/// data vocabulary with no serialization framework, and glbench is the consumer
/// that happens to want JSON. Putting the writer here keeps every backend free
/// of a format dependency it never asked for.
///
/// Derived values (share, entropy, load balance) are written alongside the raw
/// counters rather than instead of them: a consumer that disagrees with our
/// definition of "hotspot share" can recompute from `total_ms`, but only if we
/// did not throw it away.
fn telemetry_json(t: &glcore::telemetry::EngineTelemetry) -> Json {
    let phase = |p: &glcore::telemetry::PhaseProfile| {
        Json::obj([
            ("total_ms", Json::Num(p.total_ms)),
            ("unattributed_ms", Json::Num(p.unattributed_ms())),
            (
                "stages",
                Json::Arr(
                    p.stages
                        .iter()
                        .map(|s| {
                            Json::obj([
                                ("name", Json::Str(s.name.clone())),
                                ("total_ms", Json::Num(s.total_ms)),
                                ("calls", Json::Num(s.calls as f64)),
                                (
                                    "share",
                                    s.share_of(p.total_ms).map(Json::Num).unwrap_or(Json::Null),
                                ),
                                (
                                    "bytes_read",
                                    s.bytes_read.map(|b| Json::Num(b as f64)).unwrap_or(Json::Null),
                                ),
                                (
                                    "macs",
                                    s.macs.map(|m| Json::Num(m as f64)).unwrap_or(Json::Null),
                                ),
                                // Derived, but written alongside the raw counts
                                // rather than instead of them: a consumer that
                                // disagrees with our definition can recompute.
                                ("gb_per_s", s.gb_per_s().map(Json::Num).unwrap_or(Json::Null)),
                                (
                                    "gmac_per_s",
                                    s.gmac_per_s().map(Json::Num).unwrap_or(Json::Null),
                                ),
                            ])
                        })
                        .collect(),
                ),
            ),
        ])
    };

    let backend = {
        match &t.backend {
            Some(b) => Json::obj([
                ("simd_path", Json::Str(b.simd_path.clone())),
                ("threads", Json::Num(b.threads as f64)),
                (
                    "kernels",
                    Json::Arr(
                        b.kernels
                            .iter()
                            .map(|(role, kernel)| {
                                Json::obj([
                                    ("role", Json::Str(role.clone())),
                                    ("kernel", Json::Str(kernel.clone())),
                                ])
                            })
                            .collect(),
                    ),
                ),
            ]),
            None => Json::Null,
        }
    };

    let memory = match &t.memory {
        Some(m) => Json::obj([
            ("model_bytes", Json::Num(m.model_bytes as f64)),
            ("kv_cache_bytes", Json::Num(m.kv_cache_bytes as f64)),
            ("scratch_bytes", Json::Num(m.scratch_bytes as f64)),
        ]),
        None => Json::Null,
    };

    let moe = {
        match &t.moe {
            Some(m) => {
                let (min, max, mean) = m.load_balance().unwrap_or((0, 0, 0.0));
                Json::obj([
                    ("num_experts", Json::Num(m.num_experts as f64)),
                    ("top_k", Json::Num(m.num_experts_per_tok as f64)),
                    ("moe_layers", Json::Num(m.moe_layers as f64)),
                    ("experts_touched", Json::Num(m.experts_touched() as f64)),
                    ("load_min", Json::Num(min as f64)),
                    ("load_max", Json::Num(max as f64)),
                    ("load_mean", Json::Num(mean)),
                    (
                        "routing_entropy",
                        m.routing_entropy().map(Json::Num).unwrap_or(Json::Null),
                    ),
                    (
                        "expert_load",
                        Json::Arr(
                            m.expert_load.iter().map(|&c| Json::Num(c as f64)).collect(),
                        ),
                    ),
                ])
            }
            None => Json::Null,
        }
    };

    Json::obj([
        ("prefill", t.prefill.as_ref().map(phase).unwrap_or(Json::Null)),
        ("decode", t.decode.as_ref().map(phase).unwrap_or(Json::Null)),
        ("backend", backend),
        ("memory", memory),
        ("moe", moe),
    ])
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
            physical_cores: cpu
                .and_then(|c| c.get("physical_cores"))
                .and_then(|n| n.as_f64())
                .map(|n| n as usize),
            model: cpu.and_then(|c| c.get("model")).and_then(|s| s.as_str()).map(String::from),
            mhz: cpu.and_then(|c| c.get("mhz")).and_then(|n| n.as_f64()),
            // Read back from the archive, never re-measured: the ceiling is a
            // fact about the machine that RAN the benchmark, and re-probing on
            // whatever machine is `inspect`ing it would silently rewrite history.
            read_bandwidth_gbs: cpu
                .and_then(|c| c.get("read_bandwidth_gbs"))
                .and_then(|n| n.as_f64()),
            // Archived ISA flags are re-read from the record, not re-probed:
            // an archive is a fact about the machine that RAN it, and probing
            // the machine now `inspect`ing it would silently rewrite history.
            isa: cpu
                .and_then(|c| c.get("isa"))
                .map(|i| {
                    let flag = |k: &str| i.get(k).and_then(|b| b.as_bool()).unwrap_or(false);
                    crate::environment::cpu::IsaSupport {
                        avx2: flag("avx2"),
                        fma: flag("fma"),
                        f16c: flag("f16c"),
                        avx512f: flag("avx512f"),
                        avx512bw: flag("avx512bw"),
                        avx512_vnni: flag("avx512_vnni"),
                        avx_vnni: flag("avx_vnni"),
                    }
                })
                .unwrap_or_default(),
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
