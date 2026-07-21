//! The run planner — turns a [`WorkloadSpec`] into an ordered plan of phases
//! and drives it, producing a finished [`BenchmarkSession`].
//!
//! This is the top-level orchestration glbench's `run` command calls. It owns
//! no timing math (that is [`crate::measurement`]) and no inference (that is the
//! [`crate::engine::adapter`]); it sequences warmup → measured iterations →
//! snapshot assembly, then attaches the analysis and validation passes.

use glcore::GlError;

use crate::analysis::summary;
use crate::behavior::BehaviorReport;
use crate::core::metrics::MeasurementSet;
use crate::core::result::SessionMetadata;
use crate::core::session::BenchmarkSession;
use crate::core::workload::WorkloadSpec;
use crate::engine::adapter::EngineAdapter;
use crate::engine::model_probe::ModelProbe;
use crate::environment::hardware::EnvironmentSnapshot;
use crate::environment::power::EnergyMeter;
use crate::validation::integrity;

/// A progress callback: `(phase, iteration, total)` — lets the CLI print a
/// heartbeat without the runner knowing about output.
pub type Progress<'a> = &'a dyn Fn(&str, usize, usize);

/// Execute a full benchmark for `spec` and return the finished session with
/// analysis + validation attached. `progress` is invoked before each phase/iter.
pub fn run(spec: &WorkloadSpec, progress: Progress<'_>) -> Result<BenchmarkSession, GlError> {
    // 1. Environment snapshot (before load, so memory reflects the idle baseline).
    let mut environment = EnvironmentSnapshot::probe(&spec.model_path);

    // 2. Load the engine + model, and probe the model file's own header —
    //    finally filling the arch/quant fields the adapter leaves None, plus
    //    the thinking-capability fact the CoT-aware entropy read needs.
    progress("load", 0, 1);
    let adapter = EngineAdapter::load(spec)?;
    let probe = ModelProbe::probe(&spec.model_path);
    let mut engine_meta = adapter.metadata().clone();
    engine_meta.model_arch = probe.arch.clone();
    engine_meta.quantization = probe.quantization.clone();
    // The workload's manual override wins over auto-detection (decision
    // record: dual approach, GGUF auto-detect + config override).
    engine_meta.thinking_capable = spec.cot_mode.or(probe.thinking_capable);

    // Attach the engine's GPU facts to the hardware snapshot.
    environment.hardware = environment.hardware.clone().with_gpu(adapter.gpu().clone());

    // 3a. Cold-start phase — a dedicated, timed phase before warmup, every
    //     iteration recorded (not just the first): a single cold sample
    //     cannot tell "this model always pays ~500ms to page in" apart from
    //     "the OS happened to schedule something else during this one run".
    //     A median + range over several can. Runs before warmup precisely
    //     because warmup's job is to WARM the cache — running it first would
    //     make every "cold" reading after it not actually cold.
    let mut cold = Vec::with_capacity(spec.cold_iters);
    for i in 0..spec.cold_iters {
        progress("cold", i, spec.cold_iters);
        cold.push(adapter.run_once(spec)?);
    }

    // 3b. Warmup — kept out of the statistics, to pay JIT/cache-warming costs
    //     before measuring. Distinct from cold-start: warmup's numbers are
    //     never recorded at all, because its entire purpose is to leave the
    //     machine in the warm state the measured phase assumes.
    for i in 0..spec.warmup_iters {
        progress("warmup", i, spec.warmup_iters);
        adapter.run_once(spec)?;
    }

    // 4. Measured iterations, bracketed by the energy meter so Joules cover
    //    exactly the work the tok/s figures describe. RAPL is package-level
    //    and Linux-only; where unreadable the meter is None and the report
    //    simply carries no energy figure (never a TDP estimate).
    let meter = EnergyMeter::start();
    let mut measurements = MeasurementSet::default();
    // One clock reading per measured iteration, for `avg_mhz` below — coarser
    // than continuous sampling would be, but representative of the run
    // without needing a background thread, and it costs one `probe_mhz` call
    // (a file read or a `wmic` process, both cheap) per iteration rather than
    // per token.
    let mut mhz_readings = Vec::with_capacity(spec.measure_iters.max(1));
    for i in 0..spec.measure_iters.max(1) {
        progress("measure", i, spec.measure_iters.max(1));
        let iter = adapter.run_once(spec)?;
        measurements.iterations.push(iter);
        if let Some(mhz) = crate::environment::cpu::probe_mhz() {
            mhz_readings.push(mhz);
        }
    }
    measurements.energy_joules = meter.and_then(EnergyMeter::stop);
    measurements.cold = cold;

    // 4b. End-of-run CPU clock, for thermal-throttle detection. A second,
    //     cheap reading (not a full CpuInfo::probe) — the start reading was
    //     already taken by EnvironmentSnapshot::probe in step 1.
    let end_mhz = crate::environment::cpu::probe_mhz();
    environment.hardware = environment
        .hardware
        .with_end_mhz(end_mhz)
        .with_avg_mhz(crate::environment::thermal::average_mhz(&mhz_readings));

    // 5. Fill in facts known only after the run: the model footprint decode
    //    streams. Prefer the file size the environment probe already captured.
    measurements.model_bytes = environment.hardware.storage.model_file_bytes;

    // 6. Assemble the session, then run the derived passes.
    let label = default_label(spec);
    let mut session = BenchmarkSession::new(
        SessionMetadata::new(label),
        environment,
        engine_meta,
        spec.clone(),
        measurements,
    );

    // 7. Pull the engine's own view of the last run. Taken after the measured
    //    iterations (not the warmups) so the stage timings describe the same
    //    work the reported tok/s came from.
    session.telemetry = adapter.telemetry();

    // 8. Behavioral signals, from one EXTRA traced run.
    //
    //    Deliberately not one of the measured iterations: tracing costs an
    //    O(vocab) sweep per token, so including it would tax the throughput
    //    being reported. The model emits the same tokens either way — only the
    //    clock differs — so the timing facts above stay untainted while the
    //    behavior facts below come from a run that actually recorded them.
    //
    //    A failure here is not fatal: the benchmark's numbers are already
    //    collected and valid. Behavior is additive, so a backend that cannot
    //    trace simply reports no behavior rather than failing the whole run.
    progress("behavior", 0, 1);
    if let Ok((tokens, traces)) = adapter.run_traced(spec) {
        let mut report = BehaviorReport::compute(&tokens, &traces);
        // The CoT-aware read of the entropy level. Only when the thinking
        // capability is actually known (probed or overridden): assessing
        // against a guessed model kind would manufacture false anomalies.
        if let (Some(e), Some(thinking)) = (&report.entropy, session.engine.thinking_capable) {
            report.cot = Some(crate::behavior::cot::CotAssessment::assess(e, thinking));
        }
        if !report.is_empty() {
            session.behavior = Some(report);
        }
    }

    session.analysis = Some(summary::analyze(&session));
    let mut validation = integrity::validate(&session);

    // 9. Automatic token-stream cross-check against an oracle engine, if the
    //    workload asked for one. Opt-in (see WorkloadSpec::verify_against's
    //    docs on why): it loads and runs a second full engine, so this is not
    //    part of the default `run` cost. Skipped — not merely reported as
    //    trivial — when the oracle IS the candidate: glproc vs glproc always
    //    matches and would report a check that verified nothing.
    if let Some(oracle) = &spec.verify_against {
        if oracle == &spec.engine {
            validation.push(
                crate::validation::integrity::Severity::Info,
                "parity",
                format!(
                    "--verify-against '{oracle}' is the same as --engine; skipped (comparing an \
                     engine to itself proves nothing)"
                ),
            );
        } else {
            progress("verify", 0, 1);
            match crate::validation::parity::validate_against_oracle_capped(
                spec,
                oracle,
                &spec.engine,
                crate::validation::parity::AUTO_VERIFY_TOKEN_CAP,
            ) {
                Ok(report) => {
                    let severity = if report.passed() {
                        crate::validation::integrity::Severity::Info
                    } else {
                        crate::validation::integrity::Severity::Error
                    };
                    validation.push(
                        severity,
                        "parity",
                        format!(
                            "{}/{} tokens match oracle '{oracle}' ({:.0}% agreement)",
                            report.check.matching_prefix,
                            report.check.compared,
                            report.check.agreement() * 100.0,
                        ),
                    );
                }
                Err(e) => {
                    // Not fatal: the benchmark's own numbers are already
                    // collected and valid. An oracle that fails to load
                    // (e.g. this platform has no glcuda device) is reported,
                    // not treated as a reason to discard the whole run.
                    validation.push(
                        crate::validation::integrity::Severity::Warning,
                        "parity",
                        format!("could not cross-check against oracle '{oracle}': {e}"),
                    );
                }
            }
        }
    }
    session.validation = Some(validation);
    Ok(session)
}

/// A default session label: `<engine>-<model-stem>`.
fn default_label(spec: &WorkloadSpec) -> String {
    let stem = std::path::Path::new(&spec.model_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model");
    format!("{}-{}", spec.engine, stem)
}
