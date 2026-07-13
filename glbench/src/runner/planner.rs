//! The run planner — turns a [`WorkloadSpec`] into an ordered plan of phases
//! and drives it, producing a finished [`BenchmarkSession`].
//!
//! This is the top-level orchestration glbench's `run` command calls. It owns
//! no timing math (that is [`crate::measurement`]) and no inference (that is the
//! [`crate::engine::adapter`]); it sequences warmup → measured iterations →
//! snapshot assembly, then attaches the analysis and validation passes.

use glcore::GlError;

use crate::analysis::summary;
use crate::core::metrics::MeasurementSet;
use crate::core::result::SessionMetadata;
use crate::core::session::BenchmarkSession;
use crate::core::workload::WorkloadSpec;
use crate::engine::adapter::EngineAdapter;
use crate::environment::hardware::EnvironmentSnapshot;
use crate::validation::integrity;

/// A progress callback: `(phase, iteration, total)` — lets the CLI print a
/// heartbeat without the runner knowing about output.
pub type Progress<'a> = &'a dyn Fn(&str, usize, usize);

/// Execute a full benchmark for `spec` and return the finished session with
/// analysis + validation attached. `progress` is invoked before each phase/iter.
pub fn run(spec: &WorkloadSpec, progress: Progress<'_>) -> Result<BenchmarkSession, GlError> {
    // 1. Environment snapshot (before load, so memory reflects the idle baseline).
    let mut environment = EnvironmentSnapshot::probe(&spec.model_path);

    // 2. Load the engine + model.
    progress("load", 0, 1);
    let adapter = EngineAdapter::load(spec)?;

    // Attach the engine's GPU facts to the hardware snapshot.
    environment.hardware = environment.hardware.clone().with_gpu(adapter.gpu().clone());

    // 3. Warmup — untimed, to pay JIT/cold-cache costs before measuring.
    for i in 0..spec.warmup_iters {
        progress("warmup", i, spec.warmup_iters);
        adapter.run_once(spec)?;
    }

    // 4. Measured iterations.
    let mut measurements = MeasurementSet::default();
    for i in 0..spec.measure_iters.max(1) {
        progress("measure", i, spec.measure_iters.max(1));
        let iter = adapter.run_once(spec)?;
        measurements.iterations.push(iter);
    }

    // 5. Fill in facts known only after the run: the model footprint decode
    //    streams. Prefer the file size the environment probe already captured.
    measurements.model_bytes = environment.hardware.storage.model_file_bytes;

    // 6. Assemble the session, then run the derived passes.
    let label = default_label(spec);
    let mut session = BenchmarkSession::new(
        SessionMetadata::new(label),
        environment,
        adapter.metadata().clone(),
        spec.clone(),
        measurements,
    );

    // 7. Pull the engine's own view of the last run. Taken after the measured
    //    iterations (not the warmups) so the stage timings describe the same
    //    work the reported tok/s came from.
    session.telemetry = adapter.telemetry();

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
