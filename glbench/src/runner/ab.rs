//! Counterbalanced A/B: the same models, run forward and then in reverse.
//!
//! A machine drifts while it is being measured, about 24% between sessions on
//! the reference laptop and 8% on a Kaggle T4. Run A then B once and the drift
//! lands entirely on B; a sequential A/B on the laptop once reported an
//! impossible 17% speedup. Running A, B, B, A gives each model the same average
//! position, and running both orders exposes what no single order can: if the
//! second pass reverses the first, the difference is the machine's, not the
//! model's. Every retention decision in the glcuda waves was made this way, by
//! hand in a notebook; this makes it what `glbench ab` does.
//!
//! Each model loads, runs and unloads before the next, so only one is ever
//! resident. Sequential on purpose: parallel decodes would contend for the
//! memory bus and corrupt every number.

use glcore::GlError;

use crate::analysis::summary;
use crate::comparison::statistics::Stats;
use crate::core::session::BenchmarkSession;
use crate::core::workload::WorkloadSpec;
use crate::runner::planner::{self, Progress};

/// Forward, then reverse.
pub const PASSES: usize = 2;

/// The sessions of one `ab` run.
#[derive(Debug, Clone)]
pub struct VLAbRun {
    /// `passes[pass][model]`, with models in the order given on the command
    /// line whatever order the pass ran them in.
    pub passes: Vec<Vec<BenchmarkSession>>,
    /// One session per model with every pass's iterations pooled.
    pub pooled: Vec<BenchmarkSession>,
}

/// One candidate's change against the baseline, pass by pass.
#[derive(Debug, Clone, PartialEq)]
pub struct VLOrderCheck {
    /// The candidate's session label.
    pub label: String,
    /// Relative decode change per pass.
    pub decode: Vec<f64>,
    /// Relative prefill change per pass.
    pub prefill: Vec<f64>,
}

impl VLOrderCheck {
    /// Whether every pass moved `changes` the same way. When they do not, the
    /// order decided the sign, so the difference is not the model's.
    pub fn agrees(changes: &[f64]) -> bool {
        changes.iter().all(|&c| c > 0.0) || changes.iter().all(|&c| c < 0.0)
    }
}

/// The model indices of `pass`, in execution order: forward on even passes,
/// reversed on odd ones.
pub fn pass_order(models: usize, pass: usize) -> Vec<usize> {
    if pass % 2 == 0 {
        (0..models).collect()
    } else {
        (0..models).rev().collect()
    }
}

/// Run every model through every pass under the identical `spec`.
pub fn run(
    spec: &WorkloadSpec,
    models: &[String],
    progress: Progress<'_>,
) -> Result<VLAbRun, GlError> {
    let mut passes = Vec::with_capacity(PASSES);
    for pass in 0..PASSES {
        let mut sessions: Vec<Option<BenchmarkSession>> = vec![None; models.len()];
        for (step, index) in pass_order(models.len(), pass).into_iter().enumerate() {
            eprintln!(
                "=== pass {}/{PASSES}, model {}/{}: {}",
                pass + 1,
                step + 1,
                models.len(),
                models[index]
            );
            let mut spec = spec.clone();
            spec.model_path = models[index].clone();
            sessions[index] = Some(planner::run(&spec, progress)?);
        }
        passes.push(sessions.into_iter().flatten().collect::<Vec<_>>());
    }
    let pooled = (0..models.len())
        .map(|index| {
            let sessions: Vec<&BenchmarkSession> = passes.iter().map(|p: &Vec<_>| &p[index]).collect();
            pool(&sessions)
        })
        .collect();
    Ok(VLAbRun { passes, pooled })
}

/// Merge one model's passes into a session whose statistics cover them all.
///
/// Telemetry and validation stay those of the first pass: they describe the
/// engine and the conditions, which the passes share.
pub fn pool(sessions: &[&BenchmarkSession]) -> BenchmarkSession {
    let mut pooled = sessions[0].clone();
    for session in &sessions[1..] {
        let (m, s) = (&mut pooled.measurements, &session.measurements);
        m.iterations.extend(s.iterations.iter().cloned());
        m.cold.extend(s.cold.iter().cloned());
        m.energy_joules = m.energy_joules.zip(s.energy_joules).map(|(a, b)| a + b);
        m.peak_memory_bytes = m.peak_memory_bytes.max(s.peak_memory_bytes);
        pooled.workload.measure_iters += session.workload.measure_iters;
        pooled.workload.cold_iters += session.workload.cold_iters;
    }
    pooled.analysis = Some(summary::analyze(&pooled));
    pooled
}

/// Each candidate's per-pass change against the first model.
pub fn order_checks(run: &VLAbRun) -> Vec<VLOrderCheck> {
    let median = |samples: Vec<f64>| Stats::from_samples(&samples).median;
    let change = |baseline: f64, candidate: f64| {
        if baseline > 0.0 {
            candidate / baseline - 1.0
        } else {
            0.0
        }
    };
    (1..run.pooled.len())
        .map(|index| {
            let per_pass = |samples: fn(&BenchmarkSession) -> Vec<f64>| -> Vec<f64> {
                run.passes
                    .iter()
                    .map(|pass| change(median(samples(&pass[0])), median(samples(&pass[index]))))
                    .collect()
            };
            VLOrderCheck {
                label: run.pooled[index].metadata.label.clone(),
                decode: per_pass(|s| s.measurements.decode_tps_samples()),
                prefill: per_pass(|s| s.measurements.prefill_tps_samples()),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::metrics::{IterationMetrics, MeasurementSet};
    use crate::core::result::SessionMetadata;
    use crate::engine::metadata::EngineMetadata;
    use crate::environment::hardware::EnvironmentSnapshot;

    fn session(label: &str, decode_tps: f64) -> BenchmarkSession {
        let mut m = MeasurementSet::default();
        m.iterations.push(IterationMetrics {
            prompt_tokens: 100,
            generated_tokens: 100,
            prefill_ms: 100.0,
            decode_ms: 100.0 / decode_tps * 1000.0,
            total_ms: 0.0,
            ttft_ms: None,
        });
        BenchmarkSession::new(
            SessionMetadata::new(label),
            EnvironmentSnapshot::probe(""),
            EngineMetadata {
                name: "glproc".into(),
                backend: "cpu".into(),
                available: true,
                model_arch: None,
                quantization: None,
                thinking_capable: None,
            },
            WorkloadSpec { measure_iters: 1, cold_iters: 0, ..WorkloadSpec::default() },
            m,
        )
    }

    #[test]
    fn each_pass_reverses_the_one_before() {
        assert_eq!(pass_order(2, 0), [0, 1]);
        assert_eq!(pass_order(2, 1), [1, 0]);
        assert_eq!(pass_order(3, 1), [2, 1, 0]);
    }

    #[test]
    fn pooling_keeps_every_iteration_and_says_so_in_the_workload() {
        let (a, b) = (session("a", 100.0), session("a", 200.0));
        let pooled = pool(&[&a, &b]);
        assert_eq!(pooled.measurements.iterations.len(), 2);
        assert_eq!(pooled.workload.measure_iters, 2);
        let mean = pooled.analysis.as_ref().unwrap().decode_tps.mean;
        assert!((mean - 150.0).abs() < 1e-6, "analysis must cover the pool: {mean}");
    }

    fn ab(pass_one: (f64, f64), pass_two: (f64, f64)) -> VLAbRun {
        let passes = vec![
            vec![session("a", pass_one.0), session("b", pass_one.1)],
            vec![session("a", pass_two.0), session("b", pass_two.1)],
        ];
        let pooled = vec![
            pool(&[&passes[0][0], &passes[1][0]]),
            pool(&[&passes[0][1], &passes[1][1]]),
        ];
        VLAbRun { passes, pooled }
    }

    #[test]
    fn a_real_difference_survives_the_reversal() {
        let checks = order_checks(&ab((100.0, 110.0), (90.0, 99.0)));
        assert_eq!(checks.len(), 1);
        assert!(VLOrderCheck::agrees(&checks[0].decode), "{checks:?}");
    }

    /// Whichever model ran second looked faster: a machine warming up, not a
    /// model. Exactly what a single sequential order would have reported as a
    /// win.
    #[test]
    fn drift_is_caught_when_the_reversal_flips_the_sign() {
        let checks = order_checks(&ab((100.0, 110.0), (110.0, 100.0)));
        assert!(!VLOrderCheck::agrees(&checks[0].decode), "{checks:?}");
    }

    #[test]
    fn no_change_at_all_does_not_count_as_agreeing() {
        assert!(!VLOrderCheck::agrees(&[0.0, 0.0]));
    }
}
