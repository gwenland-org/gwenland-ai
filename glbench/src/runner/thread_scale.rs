//! A thread-count scaling sweep: run the identical workload at several
//! `glproc` thread-pool sizes and classify how decode throughput scales.
//!
//! [Vetted per `RESEARCH_REQUIREMENTS.md`'s 8 mandatory questions]
//! 1. What problem: `analysis::bottleneck` says whether a run is
//!    memory-bound or compute-bound at whatever thread count it happened to
//!    run at, but never shows whether *more threads* would actually help —
//!    the standard scalability question performance engineering always asks.
//! 2. Who benefits: anyone tuning `GLPROC_THREADS`, or deciding whether a
//!    smaller/larger-core machine is worth deploying to.
//! 3. Production/research use: parallel-scaling sweeps (Amdahl's-law-style
//!    efficiency curves) are standard practice; nothing novel about the
//!    technique.
//! 4. How calculated: reuses `analysis::scaling::classify` — already generic
//!    over "batch size, context length, **core count**" per its own module
//!    doc — with the axis set to thread count and one full `planner::run` per
//!    point, exactly the same shape as the existing `--sweep` token-budget
//!    command in [`super::scale`].
//! 5. Reproducible: yes, same guarantees as every other sweep in this crate.
//! 6. Actionable: directly answers "should I set GLPROC_THREADS higher" with
//!    a measured efficiency number, not a guess.
//! 7. Lightweight: no new measurement primitive — sequences existing
//!    `planner::run` calls, same as `scale`/`ab`.
//! 8. Philosophy: read-only. Setting `GLPROC_THREADS` for the duration of the
//!    sweep is a *runtime parallelism* parameter, not a change to model
//!    weights, quantization, or numerical execution order — the same
//!    distinction the Read-Only Rule draws elsewhere (measuring under
//!    different conditions is not the same as changing what's measured).
//!    The previous value (or absence) is restored when the sweep ends,
//!    including on error, so this leaves no trace on the process.
//!
//! # Honest scoping
//!
//! **`glproc` only.** `GLPROC_THREADS` is glproc's own env override
//! (`glproc::runner::n_threads`); other engines have no equivalent knob wired
//! through here, so sweeping them would silently produce a flat "no scaling"
//! result that looks like a real finding but is actually just an unused
//! parameter. The caller (see `main.rs::cmd_thread_scale`) rejects any
//! `--engine` other than `glproc` up front rather than let that happen.

use glcore::GlError;

use crate::analysis::scaling::{classify, ScalePoint, Scaling};
use crate::comparison::statistics::Stats;
use crate::core::session::BenchmarkSession;
use crate::core::workload::WorkloadSpec;
use crate::runner::planner::{self, Progress};

const GLPROC_THREADS_ENV: &str = "GLPROC_THREADS";

/// One completed sweep point.
pub struct ThreadScalePoint {
    pub threads: usize,
    pub session: BenchmarkSession,
}

/// The full sweep result: every point plus the scaling verdict.
pub struct ThreadScaleReport {
    pub points: Vec<ThreadScalePoint>,
    pub scaling: Scaling,
}

/// Run `spec` once per thread count in `thread_counts`, sequentially — same
/// "concurrent runs would contend for bandwidth" reasoning as `ab`/`scale`,
/// doubly true here since every point deliberately competes for the same
/// cores.
///
/// `thread_counts` need not be sorted; the result is sorted ascending before
/// classification. Whatever `GLPROC_THREADS` was set to (or unset) before
/// this call is restored afterward, on every exit path including an error.
pub fn run_thread_sweep(
    spec: &WorkloadSpec,
    thread_counts: &[usize],
    progress: Progress<'_>,
) -> Result<ThreadScaleReport, GlError> {
    let previous = std::env::var_os(GLPROC_THREADS_ENV);
    let result = sweep(spec, thread_counts, progress);
    // SAFETY: glbench is a single-threaded CLI at the point this runs (the
    // whole sweep is sequential by design, see the module docs); no other
    // thread can be reading/writing the environment concurrently.
    unsafe {
        match &previous {
            Some(v) => std::env::set_var(GLPROC_THREADS_ENV, v),
            None => std::env::remove_var(GLPROC_THREADS_ENV),
        }
    }
    result
}

fn sweep(spec: &WorkloadSpec, thread_counts: &[usize], progress: Progress<'_>) -> Result<ThreadScaleReport, GlError> {
    let mut points = Vec::with_capacity(thread_counts.len());
    for (n, &threads) in thread_counts.iter().enumerate() {
        progress("thread-scale", n, thread_counts.len());
        // SAFETY: see run_thread_sweep — sequential, single-threaded caller.
        unsafe {
            std::env::set_var(GLPROC_THREADS_ENV, threads.to_string());
        }
        let session = planner::run(spec, progress)?;
        points.push(ThreadScalePoint { threads, session });
    }
    points.sort_by_key(|p| p.threads);

    let scale_points: Vec<ScalePoint> = points
        .iter()
        .map(|p| ScalePoint {
            axis: p.threads as f64,
            throughput: Stats::from_samples(&p.session.measurements.decode_tps_samples()).mean,
        })
        .collect();
    let scaling = classify(&scale_points);

    Ok(ThreadScaleReport { points, scaling })
}
