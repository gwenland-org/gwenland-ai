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

    // 3. Warmup — kept out of the statistics, to pay JIT/cold-cache costs
    //    before measuring. The FIRST warmup pass is still timed and recorded
    //    separately as the cold figure: it is the one iteration that shows
    //    page-in and cache-fill costs, and discarding it entirely (as pre-v2
    //    glbench did) threw away the number a deployment's first request pays.
    let mut cold = None;
    for i in 0..spec.warmup_iters {
        progress("warmup", i, spec.warmup_iters);
        let iter = adapter.run_once(spec)?;
        if i == 0 {
            cold = Some(iter);
        }
    }

    // 4. Measured iterations, bracketed by the energy meter so Joules cover
    //    exactly the work the tok/s figures describe. RAPL is package-level
    //    and Linux-only; where unreadable the meter is None and the report
    //    simply carries no energy figure (never a TDP estimate).
    let meter = EnergyMeter::start();
    let mut measurements = MeasurementSet::default();
    for i in 0..spec.measure_iters.max(1) {
        progress("measure", i, spec.measure_iters.max(1));
        let iter = adapter.run_once(spec)?;
        measurements.iterations.push(iter);
    }
    measurements.energy_joules = meter.and_then(EnergyMeter::stop);
    measurements.cold = cold;

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
    session.validation = Some(integrity::validate(&session));
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
