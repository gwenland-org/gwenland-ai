//! Stummañ Deskiñ: anomaly codes, thresholds, and the detectors that fire them.
//!
//! # Why detection is separated from measurement
//!
//! Every function here is pure: it reads already-collected numbers and returns
//! anomalies. Nothing in this file touches a clock, the filesystem, `/proc`, or
//! a Windows API. That is what makes the thresholds testable — `MEMORY_LEAK`
//! can be exercised by handing it three fabricated snapshots on a machine that
//! never leaked a byte, and `THERMAL_THROTTLE` on a machine with no thermal
//! sensor at all. A detector that had to observe a real overheating CPU to be
//! tested would never be tested.
//!
//! [`super::probe`] does the measuring and returns `Option`s. This file decides
//! what the numbers mean.

use super::axes::{
    VLDynamics, VLHardwareSnapshot, VLMemorySnapshot, VLNumericalHealth, VLParamHealth,
    VLRuntimeHealth, VLSystemSnapshot, VLTiming,
};

// ── Thresholds ───────────────────────────────────────────────────────────
//
// Absolute where the spec calls for absolute. A relative explosion threshold
// would rise along with a run that is already diverging, and stop firing
// exactly when it matters.

/// Gradient norm above which the step counts as an explosion.
pub const GRAD_EXPLOSION: f32 = 10.0;
/// Gradient norm below which a trainable parameter counts as vanishing.
pub const GRAD_VANISHING: f32 = 1e-7;
/// Multiple of the running mean that counts as a spike.
///
/// The comparison is `>=`, so a value at exactly this multiple fires. The
/// boundary is pinned by a test rather than left to whoever reads the `>`.
pub const SPIKE_MULTIPLE: f32 = 3.0;
/// Window for every running mean in this module.
pub const RUNNING_WINDOW: usize = 5;
/// `|delta_loss|` below which a step counts as flat.
pub const PLATEAU_EPSILON: f32 = 1e-6;
/// Consecutive flat steps that make a plateau.
pub const PLATEAU_STEPS: usize = 5;
/// Multiple of the first loss that counts as divergence.
pub const DIVERGENCE_MULTIPLE: f32 = 2.0;
/// Divergence is not judged before this step: early training is allowed to
/// get worse before it gets better.
pub const DIVERGENCE_MIN_STEP: usize = 10;
/// Supervised fraction below which the batch is mostly noise.
pub const LOW_SUPERVISION: f32 = 0.3;
/// Multiple of the step-1 weight norm that counts as growth.
pub const WEIGHT_GROWTH_MULTIPLE: f32 = 3.0;
/// Multiple of the median step time that counts as a timing spike.
pub const TIMING_SPIKE_MULTIPLE: f64 = 2.0;
/// Backward percentage above which backward is said to dominate.
pub const BACKWARD_DOMINATES_PCT: f32 = 80.0;
/// Per-step RSS growth that counts toward a leak.
pub const LEAK_BYTES: i64 = 50 * 1024 * 1024;
/// Consecutive growing steps that make a leak.
pub const LEAK_STEPS: usize = 3;
/// Fraction of available RAM above which memory is under pressure.
pub const MEMORY_PRESSURE_FRACTION: f64 = 0.9;
/// Fraction of baseline frequency below which the CPU is throttling.
pub const THROTTLE_FRACTION: f32 = 0.90;
/// Temperature above which the CPU is running hot.
pub const HIGH_TEMP_C: f32 = 90.0;
/// Voluntary context switches in one step that count as a spike.
pub const CTX_SWITCH_SPIKE: u64 = 1000;

// ── Axis names, as they appear in reports ────────────────────────────────

/// Axis 1.
pub const AXIS_NUMERICAL: &str = "NUMERICAL";
/// Axis 2.
pub const AXIS_DYNAMICS: &str = "DYNAMICS";
/// Axis 3.
pub const AXIS_PARAMETERS: &str = "PARAMETER_HEALTH";
/// Axis 4.
pub const AXIS_TIMING: &str = "TIMING";
/// Axis 5.
pub const AXIS_MEMORY: &str = "MEMORY";
/// Axis 6.
pub const AXIS_HARDWARE: &str = "HARDWARE";
/// Axis 7.
pub const AXIS_SYSTEM: &str = "SYSTEM";
/// Axis 8.
pub const AXIS_RUNTIME: &str = "RUNTIME";

/// How bad a step is.
///
/// `EN` because a closed set of variants is this type's whole job. Ordered, so
/// a step's level is `max` over its anomalies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum ENHealthLevel {
    /// Nothing fired.
    #[default]
    Ok,
    /// Something fired that a run can survive: a spike, a plateau, a throttle.
    Warning,
    /// Something fired that invalidates the run: a NaN, an explosion, a
    /// corrupt checkpoint, a leaked tape.
    Critical,
}

impl ENHealthLevel {
    /// Display name.
    pub fn as_str(&self) -> &'static str {
        match self {
            ENHealthLevel::Ok => "Ok",
            ENHealthLevel::Warning => "Warning",
            ENHealthLevel::Critical => "Critical",
        }
    }

    /// The badge used in the live line.
    pub fn badge(&self) -> &'static str {
        match self {
            ENHealthLevel::Ok => "OK",
            ENHealthLevel::Warning => "WARN",
            ENHealthLevel::Critical => "CRIT",
        }
    }
}

/// The run's overall outcome.
///
/// `EN` for the same reason as [`ENHealthLevel`]. Kept separate because they
/// answer different questions: a level describes one step, a verdict describes
/// a whole run and is what a CI job would key on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ENVerdict {
    /// Zero anomalies across every step.
    #[default]
    Pass,
    /// Warnings but nothing critical.
    Warn,
    /// At least one critical anomaly.
    Fail,
}

impl ENVerdict {
    /// Display name, as it appears in the report.
    pub fn as_str(&self) -> &'static str {
        match self {
            ENVerdict::Pass => "PASS",
            ENVerdict::Warn => "WARN",
            ENVerdict::Fail => "FAIL",
        }
    }
}

/// One thing that went wrong, with enough context to act on it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VLAnomaly {
    /// Stable machine-readable code, e.g. `"GRAD_SPIKE"`.
    pub code: &'static str,
    /// Which axis raised it.
    pub axis: &'static str,
    /// The parameter it concerns, when it concerns one.
    pub param: Option<String>,
    /// A sentence a human can act on, carrying the numbers that fired it.
    pub message: String,
    /// How bad.
    pub severity: ENHealthLevel,
}

impl VLAnomaly {
    /// A warning-level anomaly.
    pub fn warning(code: &'static str, axis: &'static str, message: impl Into<String>) -> Self {
        Self {
            code,
            axis,
            param: None,
            message: message.into(),
            severity: ENHealthLevel::Warning,
        }
    }

    /// A critical anomaly.
    pub fn critical(code: &'static str, axis: &'static str, message: impl Into<String>) -> Self {
        Self {
            code,
            axis,
            param: None,
            message: message.into(),
            severity: ENHealthLevel::Critical,
        }
    }

    /// Attach the parameter this anomaly is about.
    pub fn about(mut self, param: impl Into<String>) -> Self {
        self.param = Some(param.into());
        self
    }
}

/// Everything that fired on one step, and how bad the worst of it was.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct VLStepHealth {
    /// The maximum severity across `anomalies`.
    pub level: ENHealthLevel,
    /// Every anomaly, in detection order.
    pub anomalies: Vec<VLAnomaly>,
}

impl VLStepHealth {
    /// Build from a list, taking the level from the worst member.
    pub fn from_anomalies(anomalies: Vec<VLAnomaly>) -> Self {
        let level = anomalies
            .iter()
            .map(|a| a.severity)
            .max()
            .unwrap_or(ENHealthLevel::Ok);
        Self { level, anomalies }
    }

    /// Whether nothing fired.
    pub fn is_ok(&self) -> bool {
        self.level == ENHealthLevel::Ok
    }
}

// ── History the detectors need ───────────────────────────────────────────

/// The rolling context a detector needs to judge the current step.
///
/// Held separately from the step record so the detectors stay pure functions
/// of (current, history) and can be driven straight from a test.
#[derive(Debug, Clone, Default)]
pub struct VLDetectorHistory {
    /// Loss at step 1.
    pub first_loss: Option<f32>,
    /// `|delta_loss|` for the previous steps, oldest first.
    pub abs_delta_history: Vec<f32>,
    /// How many consecutive flat steps have been seen, this one excluded.
    pub flat_streak: usize,
    /// Per-parameter weight norm at step 1, by name.
    pub first_weight_norm: Vec<(String, f32)>,
    /// Median step time so far.
    pub step_time_median_ms: f64,
    /// How many consecutive steps have grown RSS past the leak threshold.
    pub leak_streak: usize,
}

// ── Detectors ────────────────────────────────────────────────────────────

/// Axis 1. NaN, Inf and parameters that did not move.
pub fn detect_numerical(n: &VLNumericalHealth, global_grad_norm: f32) -> Vec<VLAnomaly> {
    let mut out = Vec::new();

    if n.any_nan() {
        let mut sources = Vec::new();
        if n.has_nan_loss {
            sources.push("loss");
        }
        if n.has_nan_grad {
            sources.push("gradient");
        }
        if n.has_nan_param {
            sources.push("parameter");
        }
        if n.has_nan_optimizer_state {
            sources.push("optimizer state");
        }
        out.push(VLAnomaly::critical(
            "NAN_DETECTED",
            AXIS_NUMERICAL,
            format!("NaN in {}", sources.join(", ")),
        ));
    }
    if n.any_inf() {
        let mut sources = Vec::new();
        if n.has_inf_loss {
            sources.push("loss");
        }
        if n.has_inf_grad {
            sources.push("gradient");
        }
        if n.has_inf_param {
            sources.push("parameter");
        }
        out.push(VLAnomaly::critical(
            "INF_DETECTED",
            AXIS_NUMERICAL,
            format!("Inf in {}", sources.join(", ")),
        ));
    }
    // Judged on the global norm, which is the quantity a clipper would act on.
    // A NaN norm is already reported above and must not also read as > 10.
    if global_grad_norm.is_finite() && global_grad_norm > GRAD_EXPLOSION {
        out.push(VLAnomaly::critical(
            "GRADIENT_EXPLOSION",
            AXIS_NUMERICAL,
            format!("global grad norm {global_grad_norm:.4} exceeds {GRAD_EXPLOSION}"),
        ));
    }
    for name in &n.dead_params {
        out.push(
            VLAnomaly::warning(
                "DEAD_PARAMETER",
                AXIS_NUMERICAL,
                format!("{name} did not move at all: update norm is exactly 0"),
            )
            .about(name),
        );
    }
    out
}

/// Axis 2. Spikes, plateaus, divergence and a batch with little to learn from.
pub fn detect_dynamics(step: usize, d: &VLDynamics, history: &VLDetectorHistory) -> Vec<VLAnomaly> {
    let mut out = Vec::new();
    let abs_delta = d.delta_loss.abs();

    // A spike is judged against the mean of *previous* steps. Including the
    // current one would let a large value inflate the bar it is measured
    // against and hide itself.
    if !history.abs_delta_history.is_empty() {
        let window: Vec<f32> = history
            .abs_delta_history
            .iter()
            .rev()
            .take(RUNNING_WINDOW)
            .copied()
            .collect();
        let mean = window.iter().sum::<f32>() / window.len() as f32;
        if mean > 0.0 && abs_delta >= SPIKE_MULTIPLE * mean {
            out.push(VLAnomaly::warning(
                "LOSS_SPIKE",
                AXIS_DYNAMICS,
                format!(
                    "|delta loss| {abs_delta:.6} is {:.1}x the mean of the last {} ({mean:.6})",
                    abs_delta / mean,
                    window.len()
                ),
            ));
        }
    }

    // The streak counts the current step too, so the threshold is reached on
    // the fifth flat step rather than the sixth.
    if abs_delta < PLATEAU_EPSILON && history.flat_streak + 1 >= PLATEAU_STEPS {
        out.push(VLAnomaly::warning(
            "LOSS_PLATEAU",
            AXIS_DYNAMICS,
            format!(
                "loss flat for {} consecutive steps (|delta| < {PLATEAU_EPSILON:e})",
                history.flat_streak + 1
            ),
        ));
    }

    if step >= DIVERGENCE_MIN_STEP {
        if let Some(first) = history.first_loss {
            if first > 0.0 && d.loss > first * DIVERGENCE_MULTIPLE {
                out.push(VLAnomaly::critical(
                    "LOSS_DIVERGENCE",
                    AXIS_DYNAMICS,
                    format!(
                        "loss {:.6} is {:.1}x the step-1 loss {first:.6}",
                        d.loss,
                        d.loss / first
                    ),
                ));
            }
        }
    }

    if d.total_token_count > 0 && d.supervised_ratio < LOW_SUPERVISION {
        out.push(VLAnomaly::warning(
            "LOW_SUPERVISION",
            AXIS_DYNAMICS,
            format!(
                "only {:.1}% of {} tokens are supervised",
                d.supervised_ratio * 100.0,
                d.total_token_count
            ),
        ));
    }
    out
}

/// Axis 3. Per-parameter gradient and update health.
pub fn detect_parameters(params: &[VLParamHealth], history: &VLDetectorHistory) -> Vec<VLAnomaly> {
    let mut out = Vec::new();

    for p in params {
        if p.trainable
            && p.grad_norm_running_mean_5 > 0.0
            && p.grad_norm >= SPIKE_MULTIPLE * p.grad_norm_running_mean_5
        {
            out.push(
                VLAnomaly::warning(
                    "GRAD_SPIKE",
                    AXIS_PARAMETERS,
                    format!(
                        "{}: grad_norm {:.6} ({:.1}x mean {:.6})",
                        p.name,
                        p.grad_norm,
                        p.grad_norm / p.grad_norm_running_mean_5,
                        p.grad_norm_running_mean_5
                    ),
                )
                .about(&p.name),
            );
        }

        // A vanishing gradient is only meaningful for a parameter that is
        // supposed to be learning. A frozen one has no gradient by design.
        if p.trainable && p.grad_norm < GRAD_VANISHING {
            out.push(
                VLAnomaly::warning(
                    "GRADIENT_VANISHING",
                    AXIS_PARAMETERS,
                    format!(
                        "{}: grad_norm {:.3e} below {GRAD_VANISHING:e}",
                        p.name, p.grad_norm
                    ),
                )
                .about(&p.name),
            );
        }

        // The update spiked while the gradient did not. Each quantity is
        // judged against **its own** history: comparing the update norm to the
        // gradient norm instead looks like a bug and is not one, because AdamW
        // normalizes every update to roughly `lr` per element regardless of
        // gradient magnitude. That ratio is ~40x on a perfectly healthy step,
        // and an earlier version of this check fired on all ten steps of a
        // converging run because of it.
        if p.trainable
            && p.grad_norm_running_mean_5 > 0.0
            && p.update_norm_running_mean_5 > 0.0
            && p.update_norm >= SPIKE_MULTIPLE * p.update_norm_running_mean_5
            && p.grad_norm < SPIKE_MULTIPLE * p.grad_norm_running_mean_5
        {
            out.push(
                VLAnomaly::warning(
                    "UPDATE_WITHOUT_GRAD_SPIKE",
                    AXIS_PARAMETERS,
                    format!(
                        "{}: update norm {:.6} is {:.1}x its own mean {:.6} while the gradient held steady at {:.1}x its mean — suspect optimizer state",
                        p.name,
                        p.update_norm,
                        p.update_norm / p.update_norm_running_mean_5,
                        p.update_norm_running_mean_5,
                        p.grad_norm / p.grad_norm_running_mean_5
                    ),
                )
                .about(&p.name),
            );
        }

        if let Some((_, first)) = history.first_weight_norm.iter().find(|(n, _)| *n == p.name) {
            if *first > 0.0 && p.weight_norm > *first * WEIGHT_GROWTH_MULTIPLE {
                out.push(
                    VLAnomaly::warning(
                        "WEIGHT_NORM_GROWTH",
                        AXIS_PARAMETERS,
                        format!(
                            "{}: weight norm {:.4} is {:.1}x its step-1 value {:.4}",
                            p.name,
                            p.weight_norm,
                            p.weight_norm / first,
                            first
                        ),
                    )
                    .about(&p.name),
                );
            }
        }

        if !p.trainable && p.grad_norm > 0.0 {
            out.push(
                VLAnomaly::warning(
                    "FROZEN_PARAM_WITH_GRAD",
                    AXIS_PARAMETERS,
                    format!(
                        "{} is frozen but carries a gradient of norm {:.6}",
                        p.name, p.grad_norm
                    ),
                )
                .about(&p.name),
            );
        }
    }
    out
}

/// Axis 4. Timing spikes and phase imbalance.
pub fn detect_timing(t: &VLTiming, history: &VLDetectorHistory) -> Vec<VLAnomaly> {
    let mut out = Vec::new();

    if history.step_time_median_ms > 0.0
        && t.total_step_ms > TIMING_SPIKE_MULTIPLE * history.step_time_median_ms
    {
        out.push(VLAnomaly::warning(
            "TIMING_SPIKE",
            AXIS_TIMING,
            format!(
                "step took {:.1}ms, {:.1}x the median {:.1}ms",
                t.total_step_ms,
                t.total_step_ms / history.step_time_median_ms,
                history.step_time_median_ms
            ),
        ));
    }
    if t.backward_pct > BACKWARD_DOMINATES_PCT {
        out.push(VLAnomaly::warning(
            "BACKWARD_DOMINATES",
            AXIS_TIMING,
            format!("backward is {:.1}% of the step", t.backward_pct),
        ));
    }
    if t.data_load_ms > t.forward_ms && t.forward_ms > 0.0 {
        out.push(VLAnomaly::warning(
            "DATA_BOTTLENECK",
            AXIS_TIMING,
            format!(
                "data load {:.1}ms exceeds forward {:.1}ms",
                t.data_load_ms, t.forward_ms
            ),
        ));
    }
    out
}

/// Axis 5. Leaks and pressure. Silent when the platform gave no reading.
pub fn detect_memory(m: &VLMemorySnapshot, history: &VLDetectorHistory) -> Vec<VLAnomaly> {
    let mut out = Vec::new();

    if let Some(delta) = m.heap_delta_bytes {
        // The streak counts this step, so the threshold is met on the third
        // consecutive growing step rather than the fourth.
        if delta > LEAK_BYTES && history.leak_streak + 1 >= LEAK_STEPS {
            out.push(VLAnomaly::warning(
                "MEMORY_LEAK",
                AXIS_MEMORY,
                format!(
                    "RSS grew by more than {}MB for {} consecutive steps (latest +{:.1}MB)",
                    LEAK_BYTES / (1024 * 1024),
                    history.leak_streak + 1,
                    delta as f64 / (1024.0 * 1024.0)
                ),
            ));
        }
    }

    if let (Some(rss), Some(avail)) = (m.heap_rss_bytes, m.available_ram_bytes) {
        if avail > 0 && rss as f64 > MEMORY_PRESSURE_FRACTION * avail as f64 {
            out.push(VLAnomaly::warning(
                "MEMORY_PRESSURE",
                AXIS_MEMORY,
                format!(
                    "RSS {:.1}MB is {:.0}% of the {:.1}MB available",
                    rss as f64 / 1048576.0,
                    100.0 * rss as f64 / avail as f64,
                    avail as f64 / 1048576.0
                ),
            ));
        }
    }
    out
}

/// Axis 6. Throttling and heat. Silent when the sensors are absent.
pub fn detect_hardware(h: &VLHardwareSnapshot) -> Vec<VLAnomaly> {
    let mut out = Vec::new();

    if h.throttle_detected {
        let detail = match (h.cpu_freq_mhz, h.cpu_freq_baseline_mhz) {
            (Some(now), Some(base)) if base > 0.0 => format!(
                "CPU at {:.0}MHz against a {:.0}MHz baseline ({:+.1}%)",
                now,
                base,
                100.0 * (now / base - 1.0)
            ),
            _ => "CPU frequency below baseline".to_string(),
        };
        out.push(VLAnomaly::warning(
            "THERMAL_THROTTLE",
            AXIS_HARDWARE,
            detail,
        ));
    }
    if let Some(temp) = h.cpu_temp_celsius {
        if temp > HIGH_TEMP_C {
            out.push(VLAnomaly::warning(
                "HIGH_TEMP",
                AXIS_HARDWARE,
                format!("CPU at {temp:.1}C, above {HIGH_TEMP_C}C"),
            ));
        }
    }
    out
}

/// Axis 7. Scheduler contention and swap.
pub fn detect_system(s: &VLSystemSnapshot) -> Vec<VLAnomaly> {
    let mut out = Vec::new();

    if let Some(cs) = s.voluntary_ctx_switches {
        if cs > CTX_SWITCH_SPIKE {
            out.push(VLAnomaly::warning(
                "CTX_SWITCH_SPIKE",
                AXIS_SYSTEM,
                format!("{cs} voluntary context switches in one step"),
            ));
        }
    }
    if let Some(major) = s.page_faults_major {
        if major > 0 {
            out.push(VLAnomaly::critical(
                "MAJOR_PAGE_FAULT",
                AXIS_SYSTEM,
                format!("{major} major page faults — the process is hitting swap"),
            ));
        }
    }
    out
}

/// Axis 8. Structural failures of the run itself.
pub fn detect_runtime(r: &VLRuntimeHealth) -> Vec<VLAnomaly> {
    let mut out = Vec::new();

    if r.checkpoint_verified == Some(false) {
        out.push(VLAnomaly::critical(
            "CHECKPOINT_CORRUPTION",
            AXIS_RUNTIME,
            "a checkpoint written this step did not reload bit-for-bit",
        ));
    }
    if r.tape_nodes_after_backward > 0 {
        out.push(VLAnomaly::critical(
            "TAPE_LEAK",
            AXIS_RUNTIME,
            format!(
                "{} nodes left on the tape after finish_step; KL-006 requires 0",
                r.tape_nodes_after_backward
            ),
        ));
    }
    if !r.optimizer_stepped {
        out.push(VLAnomaly::warning(
            "OPTIMIZER_NOT_STEPPING",
            AXIS_RUNTIME,
            format!(
                "optimizer step counter did not advance (still {})",
                r.optimizer_state_step
            ),
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn param(name: &str, grad_norm: f32, mean: f32) -> VLParamHealth {
        VLParamHealth {
            name: name.to_string(),
            trainable: true,
            grad_norm,
            grad_norm_running_mean_5: mean,
            ..Default::default()
        }
    }

    fn codes(a: &[VLAnomaly]) -> Vec<&str> {
        a.iter().map(|x| x.code).collect()
    }

    // ── Boundaries ───────────────────────────────────────────────────────

    /// The spec's boundary, pinned exactly: 3.0x fires, 2.9x does not. Left to
    /// a bare `>` this would be off by one ulp in the direction nobody wants.
    #[test]
    fn grad_spike_fires_at_exactly_three_times_and_not_below() {
        let at = detect_parameters(&[param("w", 0.3, 0.1)], &VLDetectorHistory::default());
        assert_eq!(codes(&at), ["GRAD_SPIKE"], "3.0x must fire");

        let below = detect_parameters(&[param("w", 0.29, 0.1)], &VLDetectorHistory::default());
        assert!(!codes(&below).contains(&"GRAD_SPIKE"), "2.9x must not fire");
    }

    #[test]
    fn gradient_explosion_is_absolute_and_critical() {
        let n = VLNumericalHealth::default();
        assert!(
            detect_numerical(&n, 10.0).is_empty(),
            "at the threshold, not over"
        );
        let over = detect_numerical(&n, 10.001);
        assert_eq!(codes(&over), ["GRADIENT_EXPLOSION"]);
        assert_eq!(over[0].severity, ENHealthLevel::Critical);
    }

    /// A NaN norm must be reported as a NaN, never silently compared against
    /// the explosion threshold (every comparison with NaN is false anyway, but
    /// relying on that is how the check quietly stops working).
    #[test]
    fn a_nan_gradient_reports_nan_and_not_an_explosion() {
        let n = VLNumericalHealth {
            has_nan_grad: true,
            ..Default::default()
        };
        let out = detect_numerical(&n, f32::NAN);
        assert_eq!(codes(&out), ["NAN_DETECTED"]);
        assert_eq!(out[0].severity, ENHealthLevel::Critical);
    }

    #[test]
    fn nan_and_inf_are_reported_separately_with_their_sources_named() {
        let n = VLNumericalHealth {
            has_nan_loss: true,
            has_inf_grad: true,
            ..Default::default()
        };
        let out = detect_numerical(&n, 0.5);
        assert_eq!(codes(&out), ["NAN_DETECTED", "INF_DETECTED"]);
        assert!(out[0].message.contains("loss"));
        assert!(out[1].message.contains("gradient"));
    }

    #[test]
    fn a_dead_parameter_is_named_in_its_anomaly() {
        let n = VLNumericalHealth {
            dead_params: vec!["embed.weight".into()],
            ..Default::default()
        };
        let out = detect_numerical(&n, 0.5);
        assert_eq!(codes(&out), ["DEAD_PARAMETER"]);
        assert_eq!(out[0].param.as_deref(), Some("embed.weight"));
    }

    // ── Dynamics ─────────────────────────────────────────────────────────

    #[test]
    fn loss_spike_fires_against_the_mean_of_previous_steps() {
        let d = VLDynamics {
            delta_loss: -0.30,
            ..Default::default()
        };
        let history = VLDetectorHistory {
            abs_delta_history: vec![0.1, 0.1, 0.1],
            ..Default::default()
        };
        assert!(codes(&detect_dynamics(2, &d, &history)).contains(&"LOSS_SPIKE"));

        let small = VLDynamics {
            delta_loss: -0.29,
            ..Default::default()
        };
        assert!(!codes(&detect_dynamics(2, &small, &history)).contains(&"LOSS_SPIKE"));
    }

    /// The first step has no history to compare against, so it can never spike.
    #[test]
    fn the_first_step_cannot_spike() {
        let d = VLDynamics {
            delta_loss: -99.0,
            ..Default::default()
        };
        assert!(detect_dynamics(1, &d, &VLDetectorHistory::default()).is_empty());
    }

    #[test]
    fn loss_plateau_fires_on_the_fifth_flat_step_not_the_fourth() {
        let flat = VLDynamics {
            delta_loss: 1e-9,
            ..Default::default()
        };
        let four = VLDetectorHistory {
            flat_streak: 3,
            ..Default::default()
        };
        assert!(!codes(&detect_dynamics(9, &flat, &four)).contains(&"LOSS_PLATEAU"));

        let five = VLDetectorHistory {
            flat_streak: 4,
            ..Default::default()
        };
        assert!(codes(&detect_dynamics(10, &flat, &five)).contains(&"LOSS_PLATEAU"));
    }

    /// Early training is allowed to get worse. Divergence is only judged once
    /// the run has had a chance to settle.
    #[test]
    fn loss_divergence_is_critical_and_not_judged_before_step_ten() {
        let d = VLDynamics {
            loss: 12.0,
            ..Default::default()
        };
        let history = VLDetectorHistory {
            first_loss: Some(5.0),
            ..Default::default()
        };
        assert!(
            detect_dynamics(9, &d, &history).is_empty(),
            "too early to judge"
        );

        let out = detect_dynamics(10, &d, &history);
        assert_eq!(codes(&out), ["LOSS_DIVERGENCE"]);
        assert_eq!(out[0].severity, ENHealthLevel::Critical);
    }

    #[test]
    fn low_supervision_fires_below_thirty_percent() {
        let d = VLDynamics {
            total_token_count: 100,
            supervised_token_count: 29,
            supervised_ratio: 0.29,
            ..Default::default()
        };
        assert!(
            codes(&detect_dynamics(1, &d, &VLDetectorHistory::default()))
                .contains(&"LOW_SUPERVISION")
        );

        let fine = VLDynamics {
            supervised_ratio: 0.31,
            ..d.clone()
        };
        assert!(
            !codes(&detect_dynamics(1, &fine, &VLDetectorHistory::default()))
                .contains(&"LOW_SUPERVISION")
        );
    }

    // ── Parameters ───────────────────────────────────────────────────────

    #[test]
    fn a_frozen_parameter_carrying_a_gradient_is_reported() {
        let p = VLParamHealth {
            name: "base.weight".into(),
            trainable: false,
            grad_norm: 0.01,
            ..Default::default()
        };
        let out = detect_parameters(&[p], &VLDetectorHistory::default());
        assert_eq!(codes(&out), ["FROZEN_PARAM_WITH_GRAD"]);
    }

    /// A frozen parameter has no gradient by design, so it must not also be
    /// reported as vanishing — that would make every LoRA run noisy.
    #[test]
    fn a_frozen_parameter_is_not_reported_as_vanishing() {
        let p = VLParamHealth {
            name: "base.weight".into(),
            trainable: false,
            grad_norm: 0.0,
            ..Default::default()
        };
        assert!(detect_parameters(&[p], &VLDetectorHistory::default()).is_empty());
    }

    /// The regression this detector was rewritten for. AdamW normalizes every
    /// update to roughly `lr` per element regardless of gradient magnitude, so
    /// `update_norm / grad_norm` is ~40x on a perfectly healthy step. An
    /// earlier version compared those two directly and fired on all ten steps
    /// of a converging run.
    #[test]
    fn a_large_update_to_gradient_ratio_alone_is_not_an_anomaly() {
        let healthy = VLParamHealth {
            name: "w".into(),
            trainable: true,
            grad_norm: 0.05,
            grad_norm_running_mean_5: 0.05,
            update_norm: 2.0,
            update_norm_running_mean_5: 2.0,
            ..Default::default()
        };
        assert!(
            detect_parameters(&[healthy], &VLDetectorHistory::default()).is_empty(),
            "a steady AdamW step must be silent however large update/grad is"
        );
    }

    /// What the detector is actually for: the update jumped against its own
    /// history while the gradient did not against its own.
    #[test]
    fn update_without_grad_spike_fires_when_only_the_update_jumps() {
        let p = VLParamHealth {
            name: "w".into(),
            trainable: true,
            grad_norm: 0.05,
            grad_norm_running_mean_5: 0.05,
            update_norm: 6.0,
            update_norm_running_mean_5: 2.0,
            ..Default::default()
        };
        assert_eq!(
            codes(&detect_parameters(&[p], &VLDetectorHistory::default())),
            ["UPDATE_WITHOUT_GRAD_SPIKE"]
        );
    }

    /// When the gradient spiked too, the update following it is explained.
    /// Reporting both would double-count one event.
    #[test]
    fn a_matching_gradient_spike_suppresses_the_update_anomaly() {
        let p = VLParamHealth {
            name: "w".into(),
            trainable: true,
            grad_norm: 0.30,
            grad_norm_running_mean_5: 0.05,
            update_norm: 6.0,
            update_norm_running_mean_5: 2.0,
            ..Default::default()
        };
        let fired = detect_parameters(&[p], &VLDetectorHistory::default());
        assert_eq!(
            codes(&fired),
            ["GRAD_SPIKE"],
            "the gradient spike explains the update"
        );
    }

    #[test]
    fn weight_norm_growth_fires_past_three_times_the_first_step() {
        let p = VLParamHealth {
            name: "w".into(),
            trainable: true,
            grad_norm: 0.5,
            grad_norm_running_mean_5: 0.5,
            weight_norm: 3.1,
            ..Default::default()
        };
        let history = VLDetectorHistory {
            first_weight_norm: vec![("w".to_string(), 1.0)],
            ..Default::default()
        };
        assert!(codes(&detect_parameters(&[p], &history)).contains(&"WEIGHT_NORM_GROWTH"));
    }

    // ── Timing, memory, hardware, system, runtime ────────────────────────

    #[test]
    fn timing_spike_fires_past_twice_the_median() {
        let t = VLTiming {
            total_step_ms: 210.0,
            ..Default::default()
        };
        let history = VLDetectorHistory {
            step_time_median_ms: 100.0,
            ..Default::default()
        };
        assert!(codes(&detect_timing(&t, &history)).contains(&"TIMING_SPIKE"));
    }

    #[test]
    fn memory_leak_fires_on_the_third_consecutive_growing_step() {
        let big = VLMemorySnapshot {
            heap_delta_bytes: Some(LEAK_BYTES + 1),
            ..Default::default()
        };
        let two = VLDetectorHistory {
            leak_streak: 1,
            ..Default::default()
        };
        assert!(
            detect_memory(&big, &two).is_empty(),
            "two steps is not a leak"
        );

        let three = VLDetectorHistory {
            leak_streak: 2,
            ..Default::default()
        };
        assert!(codes(&detect_memory(&big, &three)).contains(&"MEMORY_LEAK"));
    }

    /// A platform that reports nothing must produce no anomalies. Absence of
    /// measurement is not evidence of health, but it is not evidence of
    /// sickness either, and a log full of warnings on every Windows machine
    /// would train the reader to ignore the whole axis.
    #[test]
    fn unmeasured_axes_produce_no_anomalies() {
        assert!(
            detect_memory(&VLMemorySnapshot::default(), &VLDetectorHistory::default()).is_empty()
        );
        assert!(detect_hardware(&VLHardwareSnapshot::default()).is_empty());
        assert!(detect_system(&VLSystemSnapshot::default()).is_empty());
    }

    #[test]
    fn thermal_throttle_fires_below_ninety_percent_of_baseline() {
        // The probe sets the flag; the detector reports it with the numbers.
        let h = VLHardwareSnapshot {
            cpu_freq_mhz: Some(2600.0),
            cpu_freq_baseline_mhz: Some(2900.0),
            throttle_detected: true,
            ..Default::default()
        };
        let out = detect_hardware(&h);
        assert_eq!(codes(&out), ["THERMAL_THROTTLE"]);
        assert!(out[0].message.contains("2600"), "{}", out[0].message);
    }

    #[test]
    fn high_temperature_is_reported_above_ninety_celsius() {
        let h = VLHardwareSnapshot {
            cpu_temp_celsius: Some(91.0),
            ..Default::default()
        };
        assert_eq!(codes(&detect_hardware(&h)), ["HIGH_TEMP"]);
    }

    #[test]
    fn a_major_page_fault_is_critical_but_a_context_switch_spike_is_not() {
        let swap = VLSystemSnapshot {
            page_faults_major: Some(1),
            ..Default::default()
        };
        let out = detect_system(&swap);
        assert_eq!(out[0].severity, ENHealthLevel::Critical);

        let churn = VLSystemSnapshot {
            voluntary_ctx_switches: Some(CTX_SWITCH_SPIKE + 1),
            ..Default::default()
        };
        assert_eq!(detect_system(&churn)[0].severity, ENHealthLevel::Warning);
    }

    #[test]
    fn tape_leak_and_checkpoint_corruption_are_critical() {
        let leak = VLRuntimeHealth {
            tape_nodes_after_backward: 3,
            optimizer_stepped: true,
            ..Default::default()
        };
        let out = detect_runtime(&leak);
        assert_eq!(codes(&out), ["TAPE_LEAK"]);
        assert_eq!(out[0].severity, ENHealthLevel::Critical);

        let bad = VLRuntimeHealth {
            checkpoint_verified: Some(false),
            optimizer_stepped: true,
            ..Default::default()
        };
        assert_eq!(codes(&detect_runtime(&bad)), ["CHECKPOINT_CORRUPTION"]);
    }

    #[test]
    fn an_optimizer_that_did_not_step_is_reported() {
        let stalled = VLRuntimeHealth {
            optimizer_state_step: 4,
            optimizer_stepped: false,
            ..Default::default()
        };
        assert_eq!(codes(&detect_runtime(&stalled)), ["OPTIMIZER_NOT_STEPPING"]);
    }

    // ── Synthesis ────────────────────────────────────────────────────────

    #[test]
    fn step_health_takes_its_level_from_the_worst_anomaly() {
        assert!(VLStepHealth::from_anomalies(vec![]).is_ok());

        let warn = VLStepHealth::from_anomalies(vec![VLAnomaly::warning("A", "X", "m")]);
        assert_eq!(warn.level, ENHealthLevel::Warning);

        let mixed = VLStepHealth::from_anomalies(vec![
            VLAnomaly::warning("A", "X", "m"),
            VLAnomaly::critical("B", "X", "m"),
            VLAnomaly::warning("C", "X", "m"),
        ]);
        assert_eq!(mixed.level, ENHealthLevel::Critical);
        assert_eq!(
            mixed.anomalies.len(),
            3,
            "every anomaly is kept, not just the worst"
        );
    }

    #[test]
    fn health_levels_order_ok_below_warning_below_critical() {
        assert!(ENHealthLevel::Ok < ENHealthLevel::Warning);
        assert!(ENHealthLevel::Warning < ENHealthLevel::Critical);
    }
}
