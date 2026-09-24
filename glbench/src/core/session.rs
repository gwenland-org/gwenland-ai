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
use crate::core::availability::{self, ENAvailability, VLAvailabilityMap};
use crate::core::inference::VLInferenceSession;
use crate::core::metrics::MeasurementSet;
use crate::core::mode::{ENInferenceRole, ENSessionMode};
use crate::core::result::SessionMetadata;
use crate::core::schema::{field, field_f64, field_str, ToJson};
use crate::core::workload::WorkloadSpec;
use crate::storage::digest::VLIntegrity;
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

    // --- v3 (schema v2) ---
    /// The inference run and the role it plays. For an `InferenceOnly`
    /// session this is the envelope only — the facts stay in the v1 fields
    /// above. See [`VLInferenceSession`] for why they were not moved.
    pub inference: Option<VLInferenceSession>,
    /// The training run (Wave 4). Gated behind `train-bench`, so a default
    /// build never compiles gltrain.
    #[cfg(feature = "train-bench")]
    pub training: Option<crate::training::VLTrainingSession>,
    /// Why every `null` in this session has no value (D-09). Empty for a v1
    /// archive; never absent in one this build writes.
    pub availability: VLAvailabilityMap,
    /// Content digest, filled by [`crate::storage::archive::write`]. `None`
    /// before the session is sealed, and in every v1 archive.
    pub integrity: Option<VLIntegrity>,
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
            // An inference session in the mode `SessionMetadata::new` defaults
            // to, so the §6.2 mode-consistency table holds from construction
            // rather than only after some later fix-up call.
            inference: Some(VLInferenceSession::standalone()),
            #[cfg(feature = "train-bench")]
            training: None,
            availability: VLAvailabilityMap::new(),
            integrity: None,
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
            ("inference", opt(&self.inference)),
            ("training", self.training_json()),
            ("availability", availability::to_json(&self.availability)),
            ("integrity", opt(&self.integrity)),
        ])
    }

    /// The `training` slot: whatever the training observer produced, or null.
    #[cfg(feature = "train-bench")]
    fn training_json(&self) -> Json {
        self.training.as_ref().map(|t| t.to_json()).unwrap_or(Json::Null)
    }

    /// Without `train-bench` the slot is still emitted, as null. The schema
    /// keeps the field either way so a reader never has to distinguish "this
    /// build cannot train" from "this run did not"; `availability` says which.
    #[cfg(not(feature = "train-bench"))]
    fn training_json(&self) -> Json {
        Json::Null
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
            // Telemetry is measured evidence, not a derived report. It must
            // survive archive reads so inspect, export, and cross-run drift
            // analysis see the same bottleneck facts as the live run.
            telemetry: match v.get("telemetry") {
                Some(t) if !matches!(t, Json::Null) => Some(telemetry_from_json(t)?),
                _ => None,
            },
            // Behavior remains a live-run view for now. Its archive reader is
            // separate scope because it has a much wider derived schema than
            // the raw engine telemetry fixed in this wave.
            behavior: None,
            analysis: None,
            comparison: None,
            validation: None,
            // v3 fields. Each has a v1 reading (D-20): absent `inference`
            // means a standalone inference run, absent `availability` is an
            // empty map, and absent `integrity` is an absence rather than a
            // verification failure.
            inference: match v.get("inference") {
                Some(i) if !matches!(i, Json::Null) => Some(VLInferenceSession::from_json(i)?),
                _ => Some(VLInferenceSession::standalone()),
            },
            // Reconstructed, unlike `telemetry` and `behavior`: a training
            // session's steps are measured facts that `export` needs to
            // re-render, and its derived reports are recomputed from them
            // rather than parsed. See `VLTrainingSession::from_json`.
            #[cfg(feature = "train-bench")]
            training: match v.get("training") {
                Some(t) if !matches!(t, Json::Null) => {
                    Some(crate::training::VLTrainingSession::from_json(t)?)
                }
                _ => None,
            },
            availability: availability::from_json(v.get("availability"))?,
            integrity: match v.get("integrity") {
                Some(i) if !matches!(i, Json::Null) => Some(VLIntegrity::from_json(i)?),
                _ => None,
            },
        })
    }

    /// Declare why every `null` this session emits has no value (D-09/D-10).
    ///
    /// Called from [`crate::storage::archive::write`] — the finalisation point,
    /// deliberately not from export, so a malformed archive is never written in
    /// the first place.
    ///
    /// # Why the annotation is conditional rather than a fixed list
    ///
    /// Which fields are null varies per run and per machine: a GPU counter is
    /// null here and a number on a CUDA box; `spike_ratio` is null only when no
    /// spike occurred. A fixed list would trip D-10's mirror check — a status
    /// on a field that turned out to carry a value — on the first machine that
    /// differs. So each rule is applied only to paths that are *actually* null
    /// in this session's own JSON, and an explicit declaration always wins.
    pub fn annotate_availability(&mut self) -> Result<(), String> {
        let value = self.to_json();
        let nulls = crate::validation::availability::null_paths(&value);

        for path in nulls {
            if self.availability.contains_key(&path) {
                continue; // an explicit declaration wins over the default
            }
            let (status, note) = classify_null(&path, self.metadata.session_mode);
            match note {
                Some(note) => {
                    availability::set_with_note(&mut self.availability, &path, status, note)?
                }
                None => availability::set(&mut self.availability, &path, status)?,
            }
        }

        // The mode table also requires an entry for a subtree that is *absent*
        // rather than null, which the walk above never reaches.
        match self.metadata.session_mode {
            ENSessionMode::InferenceOnly => {
                if !self.availability.contains_key("training") {
                    availability::set(
                        &mut self.availability,
                        "training",
                        ENAvailability::NotApplicable,
                    )?;
                }
            }
            ENSessionMode::TrainingOnly => {
                if !self.availability.contains_key("inference") {
                    availability::set(
                        &mut self.availability,
                        "inference",
                        ENAvailability::NotApplicable,
                    )?;
                }
            }
            ENSessionMode::Unified => {}
        }
        Ok(())
    }

    /// Check the mode-consistency table (design §6.2).
    ///
    /// | Mode | `inference` | `training` |
    /// |---|---|---|
    /// | `InferenceOnly` | `Some` (`Standalone`) | `None` |
    /// | `TrainingOnly` | `None` | `Some` |
    /// | `Unified` | `Some` (`PreTraining`) | `Some` |
    ///
    /// Runs at finalisation alongside [`Self::annotate_availability`]. A
    /// session whose mode and contents disagree is one whose consumers will
    /// disagree about what they are reading.
    pub fn check_mode_consistency(&self) -> ValidationReport {
        use crate::validation::integrity::Severity;

        let mut report = ValidationReport::default();
        let check = "session_mode";
        let mode = self.metadata.session_mode;
        let role = self.inference.as_ref().map(|i| i.role);
        let has_training = self.has_training();

        match mode {
            ENSessionMode::InferenceOnly => {
                match role {
                    Some(ENInferenceRole::Standalone) => {}
                    Some(other) => report.push(
                        Severity::Error,
                        check,
                        format!(
                            "inference_only session has inference role '{}', expected 'standalone'",
                            other.as_str()
                        ),
                    ),
                    None => report.push(
                        Severity::Error,
                        check,
                        "inference_only session has no inference block",
                    ),
                }
                if has_training {
                    report.push(
                        Severity::Error,
                        check,
                        "inference_only session carries a training block",
                    );
                }
                self.require_entry(&mut report, "training", ENAvailability::NotApplicable);
            }
            ENSessionMode::TrainingOnly => {
                if let Some(role) = role {
                    report.push(
                        Severity::Error,
                        check,
                        format!(
                            "training_only session carries an inference block (role '{}')",
                            role.as_str()
                        ),
                    );
                }
                if !has_training {
                    report.push(
                        Severity::Error,
                        check,
                        "training_only session has no training block",
                    );
                }
                self.require_entry(&mut report, "inference", ENAvailability::NotApplicable);
            }
            ENSessionMode::Unified => {
                match role {
                    Some(ENInferenceRole::PreTraining) => {}
                    Some(other) => report.push(
                        Severity::Error,
                        check,
                        format!(
                            "unified session has outer inference role '{}', expected 'pre_training'",
                            other.as_str()
                        ),
                    ),
                    None => report.push(
                        Severity::Error,
                        check,
                        "unified session has no inference block",
                    ),
                }
                if !has_training {
                    report.push(Severity::Error, check, "unified session has no training block");
                }
            }
        }
        report
    }

    /// Whether a training block is present.
    #[cfg(feature = "train-bench")]
    fn has_training(&self) -> bool {
        self.training.is_some()
    }

    /// Without `train-bench` there is no training block to have.
    #[cfg(not(feature = "train-bench"))]
    fn has_training(&self) -> bool {
        false
    }

    /// Report a missing or wrong mode-table availability entry.
    fn require_entry(&self, report: &mut ValidationReport, path: &str, want: ENAvailability) {
        use crate::validation::integrity::Severity;
        match self.availability.get(path) {
            Some(entry) if entry.status == want => {}
            Some(entry) => report.push(
                Severity::Error,
                "session_mode",
                format!(
                    "availability['{path}'] is '{}', expected '{}' for a {} session",
                    entry.status.as_str(),
                    want.as_str(),
                    self.metadata.session_mode.as_str()
                ),
            ),
            None => report.push(
                Severity::Error,
                "session_mode",
                format!(
                    "a {} session must record availability['{path}'] = '{}'",
                    self.metadata.session_mode.as_str(),
                    want.as_str()
                ),
            ),
        }
    }
}

/// Why a given `null` path has no value.
///
/// Keyed on the path because the availability map *is* keyed on paths — that is
/// the axis an archive genuinely varies along, not an incidental name. Rules run
/// most specific first, and the fallback is `Unavailable`: the weakest honest
/// claim ("it could exist; this run did not collect it"), never `Unsupported`,
/// which would assert something about the platform that nobody measured.
fn classify_null(path: &str, mode: ENSessionMode) -> (ENAvailability, Option<&'static str>) {
    // Toxicity is deliberately unimplemented, not merely uncollected — see
    // behavior::toxicity for why glbench refuses to ship a keyword matcher.
    if path == "behavior.toxicity" {
        return (
            ENAvailability::DoesNotExist,
            Some("deliberately unimplemented; see glbench::behavior::toxicity"),
        );
    }
    // The whole training subtree is meaningless for an inference-only session.
    if path == "training" || path.starts_with("training.") {
        if mode == ENSessionMode::InferenceOnly {
            return (ENAvailability::NotApplicable, None);
        }
        return (ENAvailability::Unavailable, None);
    }
    if path == "inference" || path.starts_with("inference.") {
        if mode == ENSessionMode::TrainingOnly {
            return (ENAvailability::NotApplicable, None);
        }
        // A standalone envelope's own fields are empty by design: the facts
        // live at the top level of the session rather than duplicated inside.
        return (
            ENAvailability::NotApplicable,
            Some("standalone session: the measured facts are the top-level fields"),
        );
    }
    if path.starts_with("environment.hardware.gpu.") {
        return (
            ENAvailability::Unsupported,
            Some("no GPU device reported this counter on the benchmarking machine"),
        );
    }
    if path.starts_with("environment.hardware.thermal.") {
        return (
            ENAvailability::Unsupported,
            Some("no thermal counter available on this platform"),
        );
    }
    // Signals that need a traced run. A measured run deliberately does not
    // trace: tracing perturbs the timings it would sit beside.
    if path == "behavior" || path.starts_with("behavior.") {
        return (
            ENAvailability::Unavailable,
            Some("needs a traced run; tracing perturbs timing so it never shares a measured run"),
        );
    }
    if path == "telemetry" || path.starts_with("telemetry.") {
        return (
            ENAvailability::Unavailable,
            Some("the engine reported no telemetry for this phase"),
        );
    }
    if path == "comparison" {
        return (
            ENAvailability::NotApplicable,
            Some("filled only by 'glbench compare' and 'glbench ab'"),
        );
    }
    if path == "integrity" {
        return (
            ENAvailability::Unavailable,
            Some("the digest is written when the archive is sealed"),
        );
    }
    (ENAvailability::Unavailable, None)
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
pub(crate) fn behavior_json(b: &crate::behavior::BehaviorReport) -> Json {
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
            ("p95_ms", Json::Num(s.p95_ms)),
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
    let anomaly = match &b.anomaly {
        Some(a) => Json::obj([
            (
                "quarter_gap_ms",
                Json::Arr(a.quarter_gap_ms.iter().map(|&q| Json::Num(q)).collect()),
            ),
            ("drift_frac", Json::Num(a.drift_frac)),
            ("drift_monotonic", Json::Bool(a.drift_is_monotonic())),
            (
                "spike_ratio",
                a.spike_ratio.map(Json::Num).unwrap_or(Json::Null),
            ),
            (
                "spike_token",
                a.spike_token.map(|t| Json::Num(t as f64)).unwrap_or(Json::Null),
            ),
            ("samples", Json::Num(a.samples as f64)),
        ]),
        None => Json::Null,
    };
    let cot = match &b.cot {
        Some(c) => Json::obj([
            ("flag", Json::s(c.flag.as_str())),
            ("thinking_capable", Json::Bool(c.thinking_capable)),
            ("threshold_nats", Json::Num(c.threshold_nats)),
            ("threshold_top_prob", Json::Num(c.threshold_top_prob)),
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
        ("anomaly", anomaly),
        // The CoT-aware entropy read; carries its own thresholds so the flag
        // is auditable from the archive alone.
        ("cot", cot),
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

    let launches = match &t.launches {
        Some(profile) => Json::obj([
            ("timing_source", Json::Str(profile.timing_source.clone())),
            ("coverage", Json::Str(profile.coverage.clone())),
            ("observed_launches", Json::Num(profile.observed_launches as f64)),
            ("timed_launches", Json::Num(profile.timed_launches as f64)),
            ("total_ms", Json::Num(profile.total_ms())),
            ("entries", Json::Arr(profile.entries.iter().map(|entry| Json::obj([
                ("name", Json::Str(entry.name.clone())),
                ("kind", Json::Str(entry.kind.clone())),
                ("total_ms", Json::Num(entry.total_ms)),
                ("launches", Json::Num(entry.launches as f64)),
            ])).collect())),
        ]),
        None => Json::Null,
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
        ("launches", launches),
        ("memory", memory),
        ("moe", moe),
    ])
}

/// Reconstruct the raw telemetry fields written by [`telemetry_json`].
/// Derived convenience values such as `share` and `gb_per_s` are deliberately
/// ignored and recomputed from the raw counters by consumers.
fn telemetry_from_json(v: &Json) -> Result<glcore::telemetry::EngineTelemetry, String> {
    use glcore::telemetry::{
        BackendTelemetry, EngineTelemetry, LaunchProfile, LaunchTiming, MemoryTelemetry,
        MoeTelemetry, PhaseProfile, StageTiming,
    };

    fn required_u64(v: &Json, key: &str, path: &str) -> Result<u64, String> {
        let value = field_f64(v, key).map_err(|error| format!("{path}: {error}"))?;
        if !value.is_finite() || value < 0.0 || value.fract() != 0.0 || value > u64::MAX as f64 {
            return Err(format!("{path}.{key} is not a non-negative integer"));
        }
        Ok(value as u64)
    }

    fn optional_u64(v: &Json, key: &str, path: &str) -> Result<Option<u64>, String> {
        match v.get(key) {
            None | Some(Json::Null) => Ok(None),
            Some(value) => {
                let number = value
                    .as_f64()
                    .ok_or_else(|| format!("{path}.{key} is not a number or null"))?;
                if !number.is_finite()
                    || number < 0.0
                    || number.fract() != 0.0
                    || number > u64::MAX as f64
                {
                    return Err(format!("{path}.{key} is not a non-negative integer"));
                }
                Ok(Some(number as u64))
            }
        }
    }

    fn phase(v: &Json, path: &str) -> Result<PhaseProfile, String> {
        let stages = field(v, "stages")
            .map_err(|error| format!("{path}: {error}"))?
            .as_arr()
            .ok_or_else(|| format!("{path}.stages is not an array"))?;
        let mut parsed = Vec::with_capacity(stages.len());
        for (index, stage) in stages.iter().enumerate() {
            let stage_path = format!("{path}.stages[{index}]");
            parsed.push(StageTiming {
                name: field_str(stage, "name")
                    .map_err(|error| format!("{stage_path}: {error}"))?,
                total_ms: field_f64(stage, "total_ms")
                    .map_err(|error| format!("{stage_path}: {error}"))?,
                calls: required_u64(stage, "calls", &stage_path)?,
                bytes_read: optional_u64(stage, "bytes_read", &stage_path)?,
                macs: optional_u64(stage, "macs", &stage_path)?,
            });
        }
        Ok(PhaseProfile {
            stages: parsed,
            total_ms: field_f64(v, "total_ms").map_err(|error| format!("{path}: {error}"))?,
        })
    }

    fn optional_phase(root: &Json, key: &str) -> Result<Option<PhaseProfile>, String> {
        match root.get(key) {
            None | Some(Json::Null) => Ok(None),
            Some(value) => phase(value, &format!("telemetry.{key}")).map(Some),
        }
    }

    let backend = match v.get("backend") {
        None | Some(Json::Null) => None,
        Some(value) => {
            let kernels = field(value, "kernels")?
                .as_arr()
                .ok_or_else(|| "telemetry.backend.kernels is not an array".to_string())?;
            let mut parsed = Vec::with_capacity(kernels.len());
            for (index, kernel) in kernels.iter().enumerate() {
                let role = field_str(kernel, "role")
                    .map_err(|error| format!("telemetry.backend.kernels[{index}]: {error}"))?;
                let name = field_str(kernel, "kernel")
                    .map_err(|error| format!("telemetry.backend.kernels[{index}]: {error}"))?;
                parsed.push((role, name));
            }
            Some(BackendTelemetry {
                simd_path: field_str(value, "simd_path")?,
                threads: required_u64(value, "threads", "telemetry.backend")? as usize,
                kernels: parsed,
            })
        }
    };

    let memory = match v.get("memory") {
        None | Some(Json::Null) => None,
        Some(value) => Some(MemoryTelemetry {
            model_bytes: required_u64(value, "model_bytes", "telemetry.memory")?,
            kv_cache_bytes: required_u64(value, "kv_cache_bytes", "telemetry.memory")?,
            scratch_bytes: required_u64(value, "scratch_bytes", "telemetry.memory")?,
        }),
    };

    let launches = match v.get("launches") {
        None | Some(Json::Null) => None,
        Some(value) => {
            let raw_entries = field(value, "entries")?.as_arr()
                .ok_or_else(|| "telemetry.launches.entries is not an array".to_string())?;
            let mut entries = Vec::with_capacity(raw_entries.len());
            for (index, entry) in raw_entries.iter().enumerate() {
                let path = format!("telemetry.launches.entries[{index}]");
                entries.push(LaunchTiming {
                    name: field_str(entry, "name").map_err(|error| format!("{path}: {error}"))?,
                    kind: field_str(entry, "kind").map_err(|error| format!("{path}: {error}"))?,
                    total_ms: field_f64(entry, "total_ms").map_err(|error| format!("{path}: {error}"))?,
                    launches: required_u64(entry, "launches", &path)?,
                });
            }
            let observed_launches = required_u64(value, "observed_launches", "telemetry.launches")?;
            let timed_launches = required_u64(value, "timed_launches", "telemetry.launches")?;
            if timed_launches > observed_launches {
                return Err("telemetry.launches.timed_launches exceeds observed_launches".to_string());
            }
            Some(LaunchProfile {
                entries,
                timing_source: field_str(value, "timing_source")?,
                coverage: field_str(value, "coverage")?,
                observed_launches,
                timed_launches,
            })
        }
    };

    let moe = match v.get("moe") {
        None | Some(Json::Null) => None,
        Some(value) => {
            let load = field(value, "expert_load")?
                .as_arr()
                .ok_or_else(|| "telemetry.moe.expert_load is not an array".to_string())?;
            let mut expert_load = Vec::with_capacity(load.len());
            for (index, count) in load.iter().enumerate() {
                let number = count.as_f64().ok_or_else(|| {
                    format!("telemetry.moe.expert_load[{index}] is not a number")
                })?;
                if !number.is_finite()
                    || number < 0.0
                    || number.fract() != 0.0
                    || number > u64::MAX as f64
                {
                    return Err(format!(
                        "telemetry.moe.expert_load[{index}] is not a non-negative integer"
                    ));
                }
                expert_load.push(number as u64);
            }
            Some(MoeTelemetry {
                num_experts: required_u64(value, "num_experts", "telemetry.moe")? as usize,
                num_experts_per_tok: required_u64(value, "top_k", "telemetry.moe")? as usize,
                expert_load,
                moe_layers: required_u64(value, "moe_layers", "telemetry.moe")? as usize,
            })
        }
    };

    Ok(EngineTelemetry {
        prefill: optional_phase(v, "prefill")?,
        decode: optional_phase(v, "decode")?,
        backend,
        launches,
        memory,
        moe,
    })
}

/// Reconstruct engine metadata from JSON (the fields comparison needs).
fn engine_from_json(v: &Json) -> Result<EngineMetadata, String> {
    Ok(EngineMetadata {
        name: v.get("name").and_then(|s| s.as_str()).unwrap_or("").to_string(),
        backend: v.get("backend").and_then(|s| s.as_str()).unwrap_or("").to_string(),
        available: v.get("available").and_then(|b| b.as_bool()).unwrap_or(false),
        model_arch: v.get("model_arch").and_then(|s| s.as_str()).map(String::from),
        quantization: v.get("quantization").and_then(|s| s.as_str()).map(String::from),
        thinking_capable: v.get("thinking_capable").and_then(|b| b.as_bool()),
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
    use crate::environment::thermal::ThermalSnapshot;

    let hw = v.get("hardware");
    let cpu = hw.and_then(|h| h.get("cpu"));
    let gpu = hw.and_then(|h| h.get("gpu"));
    let storage = hw.and_then(|h| h.get("storage"));
    let thermal = hw.and_then(|h| h.get("thermal"));
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
        // Read back, not re-probed: like the ISA flags above, throttling is a
        // fact about the machine that RAN the benchmark.
        thermal: ThermalSnapshot {
            start_mhz: thermal.and_then(|t| t.get("start_mhz")).and_then(|n| n.as_f64()),
            end_mhz: thermal.and_then(|t| t.get("end_mhz")).and_then(|n| n.as_f64()),
            avg_mhz: thermal.and_then(|t| t.get("avg_mhz")).and_then(|n| n.as_f64()),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::bottleneck::Bottleneck;
    use crate::analysis::ceiling::CeilingBasis;
    use crate::analysis::summary::AnalysisReport;
    use crate::comparison::statistics::Stats;
    use crate::core::metrics::{IterationMetrics, MeasurementSet};
    use crate::core::workload::WorkloadSpec;
    use crate::engine::metadata::EngineMetadata;
    use crate::environment::hardware::EnvironmentSnapshot;

    /// The cardinal rule this crate's own docs state (metrics.rs: "measurement
    /// stores raw numbers, never conclusions"): `measurements` and `analysis`
    /// must be two genuinely separate JSON objects, neither leaking into the
    /// other. This asserts it holds, rather than trusting the doc comment.
    #[test]
    fn measurements_and_analysis_are_separate_objects_with_no_cross_contamination() {
        let mut m = MeasurementSet::default();
        m.iterations.push(IterationMetrics {
            prompt_tokens: 10,
            generated_tokens: 20,
            prefill_ms: 5.0,
            decode_ms: 100.0,
            total_ms: 105.0,
        });
        let session = BenchmarkSession {
            metadata: SessionMetadata::new("test"),
            environment: EnvironmentSnapshot::probe(""),
            engine: EngineMetadata {
                name: "glproc".into(),
                backend: "cpu".into(),
                available: true,
                model_arch: None,
                quantization: None,
                thinking_capable: None,
            },
            workload: WorkloadSpec::default(),
            measurements: m,
            telemetry: None,
            behavior: None,
            analysis: Some(AnalysisReport {
                decode_tps: Stats::from_samples(&[200.0]),
                prefill_tps: Stats::from_samples(&[2000.0]),
                health: 0.9,
                bottleneck: Bottleneck::MemoryBound,
                ceiling_efficiency: Some(0.8),
                ceiling_basis: CeilingBasis::Measured,
                roofline: None,
                hypotheses: vec!["consistent with X".to_string()],
                notes: vec!["an observation".to_string()],
            }),
            comparison: None,
            validation: None,
            inference: Some(VLInferenceSession::standalone()),
            #[cfg(feature = "train-bench")]
            training: None,
            availability: VLAvailabilityMap::new(),
            integrity: None,
        };

        let json = session.to_json();
        let measurements = json.get("measurements").unwrap().as_obj().unwrap();
        let analysis = json.get("analysis").unwrap().as_obj().unwrap();

        // Facts a reader must find under `measurements`, never under `analysis`.
        for raw_field in ["iterations", "peak_memory_bytes", "model_bytes", "cold", "energy_joules"] {
            assert!(measurements.contains_key(raw_field), "measurements missing {raw_field}");
            assert!(!analysis.contains_key(raw_field), "analysis leaked raw field {raw_field}");
        }
        // Conclusions a reader must find under `analysis`, never under
        // `measurements` — a health score or a bottleneck verdict inside the
        // raw facts would be exactly the fact/conclusion mixing DESIGN.md
        // forbids.
        for derived_field in ["health", "bottleneck", "ceiling_efficiency", "ceiling_basis", "hypotheses"] {
            assert!(analysis.contains_key(derived_field), "analysis missing {derived_field}");
            assert!(!measurements.contains_key(derived_field), "measurements leaked derived field {derived_field}");
        }
    }

    #[test]
    fn telemetry_survives_a_full_session_json_round_trip() {
        use glcore::telemetry::{
            BackendTelemetry, EngineTelemetry, LaunchProfile, LaunchTiming, MemoryTelemetry,
            MoeTelemetry, PhaseProfile, StageTiming,
        };

        let mut session = BenchmarkSession::new(
            SessionMetadata::new("telemetry-round-trip"),
            EnvironmentSnapshot::probe(""),
            EngineMetadata {
                name: "glcuda".into(),
                backend: "cuda".into(),
                available: true,
                model_arch: Some("qwen2".into()),
                quantization: Some("Q8_0".into()),
                thinking_capable: Some(false),
            },
            WorkloadSpec::default(),
            MeasurementSet::default(),
        );
        let stage = StageTiming {
            name: "attention_core".into(),
            total_ms: 12.5,
            calls: 24,
            bytes_read: Some(4096),
            macs: Some(8192),
        };
        session.telemetry = Some(EngineTelemetry {
            prefill: Some(PhaseProfile { stages: vec![stage.clone()], total_ms: 13.0 }),
            decode: Some(PhaseProfile { stages: vec![stage], total_ms: 14.0 }),
            backend: Some(BackendTelemetry {
                simd_path: "sm_75".into(),
                threads: 128,
                kernels: vec![("attention".into(), "gl_attn_rows_qk4_f32".into())],
            }),
            launches: Some(LaunchProfile {
                entries: vec![
                    LaunchTiming {
                        name: "gl_attn_rows_qk4_f32".into(), kind: "kernel".into(),
                        total_ms: 8.25, launches: 24,
                    },
                    LaunchTiming {
                        name: "cuda_graph_replay".into(), kind: "graph_replay".into(),
                        total_ms: 3.5, launches: 8,
                    },
                ],
                timing_source: "cuda_events_on_launch_stream".into(),
                coverage: "direct launches and graph replays".into(),
                observed_launches: 33,
                timed_launches: 32,
            }),
            memory: Some(MemoryTelemetry {
                model_bytes: 1_000,
                kv_cache_bytes: 2_000,
                scratch_bytes: 3_000,
            }),
            moe: Some(MoeTelemetry {
                num_experts: 4,
                num_experts_per_tok: 2,
                expert_load: vec![10, 20, 30, 40],
                moe_layers: 8,
            }),
        });

        let back = BenchmarkSession::from_json(&session.to_json()).unwrap();
        assert_eq!(back.telemetry, session.telemetry);
    }
}
