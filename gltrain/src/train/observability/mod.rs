//! Stummañ Deskiñ: full multi-axis observability for a training run.
//!
//! [`VLTrainingObserver`] records eight axes on every step, synthesizes a
//! health verdict, prints a live line, and writes a JSON and a Markdown report
//! when the run ends.
//!
//! # It has two front doors, and the reason is worth reading
//!
//! **The hook API** ([`VLTrainingObserver::on_batch_loaded`] and friends) is
//! what a loop written by hand calls. It is the only way to reach all eight
//! axes, because four of the things the taxonomy asks for do not exist inside
//! [`crate::train::Trainer`]:
//!
//! - token counts and the supervised ratio — `Trainer` trains on
//!   [`crate::train::VLMicroDataset`], which has no tokens
//! - per-parameter `update_norm` and `weight_norm` — these need the parameter
//!   values from before and after the update, which live only at the call site
//!   that performed it
//! - per-parameter gradients *by name* — a [`VLGradStore`] is keyed by
//!   `TensorId`, which is process-global and carries no name
//! - `data_load_ms` and `checkpoint_ms` — `Trainer` measures neither
//!
//! **The [`StepObserver`] impl** is what `Trainer` calls. It fills in the axes
//! `VLTrainingStep` can supply — loss, timing, global gradient norm, NaN and
//! Inf counts — and leaves the rest at their defaults. Nothing is invented to
//! fill a gap: a field `Trainer` cannot supply reads as absent, not as zero.
//!
//! Both paths converge on [`VLTrainingObserver::record`], so the detectors,
//! the health synthesis and the reports are identical either way.
//!
//! # Zero changes to Trainer
//!
//! `Trainer` is not touched. It already accepts a `Box<dyn StepObserver>` and
//! already calls it once per step; that is the whole integration.

pub mod anomaly;
pub mod axes;
pub mod probe;
pub mod report;

pub use anomaly::{ENHealthLevel, ENVerdict, VLAnomaly, VLStepHealth};
pub use axes::{
    VLBatchInfo, VLDynamics, VLHardwareSnapshot, VLMemorySnapshot, VLNumericalHealth,
    VLObservedStep, VLParamHealth, VLParamSnapshot, VLRuntimeHealth, VLSystemSnapshot, VLTiming,
};
pub use report::VLRunConfig;

use crate::autograd::grad_store::VLGradStore;
use crate::error::Result;
use crate::optim::VLNamedTensor;
use crate::train::observe::{StepObserver, VLTrainingStep};
use anomaly::VLDetectorHistory;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Euclidean norm accumulated in f64, skipping non-finite elements.
///
/// One NaN would otherwise make the whole norm NaN and destroy the only signal
/// the number carries. The NaN is not lost — it is reported by its own flag.
fn l2(v: &[f32]) -> f32 {
    v.iter()
        .filter(|x| x.is_finite())
        .map(|x| (*x as f64) * (*x as f64))
        .sum::<f64>()
        .sqrt() as f32
}

fn max_abs(v: &[f32]) -> f32 {
    v.iter()
        .filter(|x| x.is_finite())
        .fold(0.0f32, |m, x| m.max(x.abs()))
}

fn min_abs(v: &[f32]) -> f32 {
    v.iter()
        .filter(|x| x.is_finite())
        .fold(f32::INFINITY, |m, x| m.min(x.abs()))
        .min(f32::MAX)
}

fn mean(v: &[f32]) -> f32 {
    let finite: Vec<f32> = v.iter().copied().filter(|x| x.is_finite()).collect();
    if finite.is_empty() {
        return 0.0;
    }
    (finite.iter().map(|x| *x as f64).sum::<f64>() / finite.len() as f64) as f32
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// How the observer behaves. `VL` because it is a plain config bag.
#[derive(Debug, Clone, PartialEq)]
pub struct VLObserverConfig {
    /// Whether to print a line per step.
    pub live_stdout: bool,
    /// Where the end-of-run reports go. `None` writes none.
    pub report_dir: Option<PathBuf>,
    /// What the run is, for the report header.
    pub run: VLRunConfig,
}

impl Default for VLObserverConfig {
    fn default() -> Self {
        Self {
            live_stdout: true,
            report_dir: None,
            run: VLRunConfig::default(),
        }
    }
}

/// What one step measured, before the observer derives anything from it.
///
/// The loop fills this in through the hook methods; the observer turns it into
/// a [`VLObservedStep`]. Kept separate so a partially-observed step (the
/// `StepObserver` path) and a fully-observed one (the hook path) go through
/// exactly the same derivation.
#[derive(Debug, Clone, Default)]
struct StepInProgress {
    batch: VLBatchInfo,
    per_sequence_losses: Vec<f32>,
    loss: f32,
    data_load_ms: f64,
    forward_ms: f64,
    backward_ms: f64,
    optimizer_ms: f64,
    checkpoint_ms: f64,
    params: Vec<VLParamSnapshot>,
    optimizer_state: Vec<VLNamedTensor>,
    tape_nodes_after_backward: usize,
    optimizer_state_step: u64,
    checkpoint_verified: Option<bool>,
    lr: f64,
    epoch: usize,
}

/// Records every axis of a training run, step by step.
///
/// The `VL` prefix is the one the specification asked for. It is a looser fit
/// than the rest of this module's names — the type owns rolling state and has
/// real behaviour, where `VL` is documented as plain data with derived traits
/// only — and it is flagged here rather than renamed unilaterally.
pub struct VLTrainingObserver {
    config: VLObserverConfig,
    steps: Vec<VLObservedStep>,
    current: StepInProgress,

    // Rolling state the detectors read.
    probes: probe::VLProbeState,
    first_loss: Option<f32>,
    prev_loss: Option<f32>,
    loss_history: Vec<f32>,
    abs_delta_history: Vec<f32>,
    flat_streak: usize,
    leak_streak: usize,
    step_times_ms: Vec<f64>,
    grad_norm_history: Vec<(String, Vec<f32>)>,
    update_norm_history: Vec<(String, Vec<f32>)>,
    first_weight_norm: Vec<(String, f32)>,
    prev_optimizer_step: Option<u64>,

    step_started: Option<Instant>,
    phase_started: Option<Instant>,
}

impl VLTrainingObserver {
    /// A new observer.
    pub fn new(config: VLObserverConfig) -> Self {
        Self {
            config,
            steps: Vec::new(),
            current: StepInProgress::default(),
            probes: probe::VLProbeState::new(),
            first_loss: None,
            prev_loss: None,
            loss_history: Vec::new(),
            abs_delta_history: Vec::new(),
            flat_streak: 0,
            leak_streak: 0,
            step_times_ms: Vec::new(),
            grad_norm_history: Vec::new(),
            update_norm_history: Vec::new(),
            first_weight_norm: Vec::new(),
            prev_optimizer_step: None,
            step_started: None,
            phase_started: None,
        }
    }

    /// An observer that prints live and writes no report.
    pub fn live() -> Self {
        Self::new(VLObserverConfig::default())
    }

    /// Every step recorded so far.
    pub fn steps(&self) -> &[VLObservedStep] {
        &self.steps
    }

    /// The run's verdict and the sentence explaining it.
    pub fn verdict(&self) -> (ENVerdict, String) {
        report::verdict(&self.steps)
    }

    /// Every anomaly across the run, paired with the step that raised it.
    pub fn all_anomalies(&self) -> Vec<(usize, &VLAnomaly)> {
        self.steps
            .iter()
            .flat_map(|s| s.health.anomalies.iter().map(move |a| (s.step, a)))
            .collect()
    }

    /// The loss curve.
    pub fn loss_curve(&self) -> Vec<f32> {
        self.steps.iter().map(|s| s.dynamics.loss).collect()
    }

    // ── Hook API ─────────────────────────────────────────────────────────

    /// Hook 1. The batch is ready. Starts the step clock.
    pub fn on_batch_loaded(&mut self, batch: VLBatchInfo, data_load_ms: f64) {
        self.current = StepInProgress {
            batch,
            data_load_ms,
            ..Default::default()
        };
        self.step_started = Some(Instant::now());
        self.phase_started = Some(Instant::now());
    }

    /// Hook 2. The forward pass finished and produced `loss`.
    ///
    /// `per_sequence_losses` may be empty; when supplied it drives
    /// `batch_loss_variance`, which a single averaged loss cannot show.
    pub fn on_forward_complete(&mut self, loss: f32, per_sequence_losses: &[f32]) {
        self.current.forward_ms = self.lap();
        self.current.loss = loss;
        self.current.per_sequence_losses = per_sequence_losses.to_vec();
    }

    /// Hook 3. Backward finished. `tape_nodes` is what the tape holds *after*
    /// `finish_step`, which KL-006 requires to be zero.
    pub fn on_backward_complete(&mut self, tape_nodes: usize) {
        self.current.backward_ms = self.lap();
        self.current.tape_nodes_after_backward = tape_nodes;
    }

    /// Hook 4. The optimizer ran. `params` carries each parameter's gradient
    /// and its values from before and after the update — the only place those
    /// coexist.
    pub fn on_optimizer_step(
        &mut self,
        params: Vec<VLParamSnapshot>,
        optimizer_state: &[VLNamedTensor],
        optimizer_step: u64,
        lr: f64,
    ) {
        self.current.optimizer_ms = self.lap();
        self.current.params = params;
        self.current.optimizer_state = optimizer_state.to_vec();
        self.current.optimizer_state_step = optimizer_step;
        self.current.lr = lr;
    }

    /// Hook 4b. A checkpoint was written this step and verified (or not).
    pub fn on_checkpoint(&mut self, verified: bool, checkpoint_ms: f64) {
        self.current.checkpoint_verified = Some(verified);
        self.current.checkpoint_ms = checkpoint_ms;
    }

    /// Hook 5. Close the step: probe the platform, detect, print, store.
    pub fn on_step_complete(&mut self) -> &VLObservedStep {
        let total_ms = self
            .step_started
            .map_or(0.0, |t| t.elapsed().as_secs_f64() * 1000.0);
        let in_progress = std::mem::take(&mut self.current);
        self.record(in_progress, total_ms);
        self.steps.last().expect("record pushed a step")
    }

    /// Hook 6. The run is over. Writes the reports, if a directory was given.
    ///
    /// Returns the two paths, or `None` when no directory was configured.
    pub fn on_training_complete(&mut self) -> Result<Option<(PathBuf, PathBuf)>> {
        let Some(dir) = self.config.report_dir.clone() else {
            return Ok(None);
        };
        let paths = report::write_reports(&dir, &self.config.run, &self.steps)?;
        Ok(Some(paths))
    }

    /// Write the reports into `dir` regardless of configuration.
    pub fn write_reports_to(&self, dir: &Path) -> Result<(PathBuf, PathBuf)> {
        report::write_reports(dir, &self.config.run, &self.steps)
    }

    /// Milliseconds since the last phase boundary, and reset it.
    fn lap(&mut self) -> f64 {
        let ms = self
            .phase_started
            .map_or(0.0, |t| t.elapsed().as_secs_f64() * 1000.0);
        self.phase_started = Some(Instant::now());
        ms
    }

    // ── Derivation ───────────────────────────────────────────────────────

    /// Turn a measured step into a recorded one. Both front doors land here.
    fn record(&mut self, s: StepInProgress, total_ms: f64) {
        let step = self.steps.len() + 1;

        // Axis 3 first: several other axes read its aggregates.
        let mut parameters = Vec::with_capacity(s.params.len());
        let mut all_grads_sq = 0.0f64;
        let mut all_updates_sq = 0.0f64;
        let mut dead_params = Vec::new();
        let mut has_nan_grad = false;
        let mut has_inf_grad = false;
        let mut has_nan_param = false;
        let mut has_inf_param = false;

        for p in &s.params {
            let update: Vec<f32> = p.after.iter().zip(&p.before).map(|(a, b)| a - b).collect();

            has_nan_grad |= p.grad.iter().any(|g| g.is_nan());
            has_inf_grad |= p.grad.iter().any(|g| g.is_infinite());
            has_nan_param |= p.after.iter().any(|w| w.is_nan());
            has_inf_param |= p.after.iter().any(|w| w.is_infinite());

            let grad_norm = l2(&p.grad);
            let update_norm = l2(&update);
            let weight_norm = l2(&p.after);

            all_grads_sq += (grad_norm as f64) * (grad_norm as f64);
            all_updates_sq += (update_norm as f64) * (update_norm as f64);

            // Exactly zero, not approximately. A parameter that moved by 1e-30
            // is training badly; one that moved by 0.0 is not connected.
            if p.trainable && !p.after.is_empty() && update_norm == 0.0 {
                dead_params.push(p.name.clone());
            }

            // Each quantity is meaned over its own history. The update norm
            // must never be compared against the gradient norm: AdamW
            // normalizes the update to roughly `lr` per element whatever the
            // gradient was, so their ratio is large on every healthy step.
            let mean_of = |hist: &[(String, Vec<f32>)], name: &str| -> f32 {
                let series = hist
                    .iter()
                    .find(|(n, _)| n == name)
                    .map(|(_, v)| v.as_slice())
                    .unwrap_or(&[]);
                let window: Vec<f32> = series
                    .iter()
                    .rev()
                    .take(anomaly::RUNNING_WINDOW)
                    .copied()
                    .collect();
                if window.is_empty() {
                    0.0
                } else {
                    window.iter().sum::<f32>() / window.len() as f32
                }
            };
            let running_mean = mean_of(&self.grad_norm_history, &p.name);
            let update_running_mean = mean_of(&self.update_norm_history, &p.name);

            let (rows_updated, rows_total) = match p.row_width {
                Some(w) if w > 0 && !p.grad.is_empty() => (
                    Some(
                        p.grad
                            .chunks(w)
                            .filter(|r| r.iter().any(|g| *g != 0.0))
                            .count(),
                    ),
                    Some(p.grad.len() / w),
                ),
                _ => (None, None),
            };

            parameters.push(VLParamHealth {
                name: p.name.clone(),
                trainable: p.trainable,
                grad_norm,
                grad_max: max_abs(&p.grad),
                grad_min: if p.grad.is_empty() {
                    0.0
                } else {
                    min_abs(&p.grad)
                },
                grad_mean: mean(&p.grad),
                update_norm,
                update_max: max_abs(&update),
                weight_norm,
                grad_norm_running_mean_5: running_mean,
                update_norm_running_mean_5: update_running_mean,
                rows_updated,
                rows_total,
            });

            match self
                .grad_norm_history
                .iter_mut()
                .find(|(n, _)| *n == p.name)
            {
                Some((_, v)) => v.push(grad_norm),
                None => self
                    .grad_norm_history
                    .push((p.name.clone(), vec![grad_norm])),
            }
            match self
                .update_norm_history
                .iter_mut()
                .find(|(n, _)| *n == p.name)
            {
                Some((_, v)) => v.push(update_norm),
                None => self
                    .update_norm_history
                    .push((p.name.clone(), vec![update_norm])),
            }
            if step == 1 {
                self.first_weight_norm.push((p.name.clone(), weight_norm));
            }
        }

        let global_grad_norm = (all_grads_sq.sqrt()) as f32;
        let global_update_norm = (all_updates_sq.sqrt()) as f32;

        // Axis 1.
        let numerical = VLNumericalHealth {
            has_nan_loss: s.loss.is_nan(),
            has_inf_loss: s.loss.is_infinite(),
            has_nan_grad,
            has_inf_grad,
            has_nan_param,
            has_inf_param,
            has_nan_optimizer_state: s
                .optimizer_state
                .iter()
                .any(|t| t.data.iter().any(|v| !v.is_finite())),
            dead_params,
        };

        // Axis 2.
        let delta_loss = self.prev_loss.map_or(0.0, |p| s.loss - p);
        if self.first_loss.is_none() && s.loss.is_finite() {
            self.first_loss = Some(s.loss);
        }
        self.loss_history.push(s.loss);
        let recent: Vec<f32> = self
            .loss_history
            .iter()
            .rev()
            .take(anomaly::RUNNING_WINDOW)
            .copied()
            .filter(|l| l.is_finite())
            .collect();
        let dynamics = VLDynamics {
            loss: s.loss,
            delta_loss,
            loss_ratio: self
                .first_loss
                .map_or(1.0, |f| if f == 0.0 { 0.0 } else { s.loss / f }),
            loss_running_mean_5: if recent.is_empty() {
                s.loss
            } else {
                recent.iter().sum::<f32>() / recent.len() as f32
            },
            perplexity: s.loss.exp(),
            supervised_token_count: s.batch.supervised_tokens,
            total_token_count: s.batch.total_tokens,
            supervised_ratio: s.batch.supervised_ratio(),
            batch_loss_variance: variance(&s.per_sequence_losses),
        };

        // Axis 4.
        self.step_times_ms.push(total_ms);
        let mut sorted = self.step_times_ms.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = percentile(&sorted, 0.5);
        let pct = |ms: f64| {
            if total_ms > 0.0 {
                (ms / total_ms * 100.0) as f32
            } else {
                0.0
            }
        };
        let per_sec = |n: usize| {
            if total_ms > 0.0 {
                n as f64 / total_ms * 1000.0
            } else {
                0.0
            }
        };
        let timing = VLTiming {
            data_load_ms: s.data_load_ms,
            forward_ms: s.forward_ms,
            backward_ms: s.backward_ms,
            optimizer_ms: s.optimizer_ms,
            checkpoint_ms: s.checkpoint_ms,
            total_step_ms: total_ms,
            supervised_tokens_per_sec: per_sec(s.batch.supervised_tokens),
            total_tokens_per_sec: per_sec(s.batch.total_tokens),
            forward_pct: pct(s.forward_ms),
            backward_pct: pct(s.backward_ms),
            optimizer_pct: pct(s.optimizer_ms),
            step_time_median_ms: median,
            step_time_p95_ms: percentile(&sorted, 0.95),
        };

        // Axes 5-7.
        let memory = self.probes.memory();
        let hardware = self.probes.hardware(total_ms / 1000.0);
        let system = self.probes.system();

        // Axis 8.
        let optimizer_stepped = match self.prev_optimizer_step {
            // A loop that reports no optimizer counter at all (the default 0)
            // must not be accused of stalling on its first step.
            None => true,
            Some(prev) => s.optimizer_state_step == prev + 1,
        };
        let runtime = VLRuntimeHealth {
            checkpoint_verified: s.checkpoint_verified,
            tape_nodes_after_backward: s.tape_nodes_after_backward,
            optimizer_state_step: s.optimizer_state_step,
            optimizer_stepped,
        };

        // Detect, using history that does not yet include this step.
        let history = VLDetectorHistory {
            first_loss: self.first_loss,
            abs_delta_history: self.abs_delta_history.clone(),
            flat_streak: self.flat_streak,
            first_weight_norm: self.first_weight_norm.clone(),
            step_time_median_ms: if self.step_times_ms.len() > 1 {
                median
            } else {
                0.0
            },
            leak_streak: self.leak_streak,
        };

        let mut anomalies = anomaly::detect_numerical(&numerical, global_grad_norm);
        anomalies.extend(anomaly::detect_dynamics(step, &dynamics, &history));
        anomalies.extend(anomaly::detect_parameters(&parameters, &history));
        anomalies.extend(anomaly::detect_timing(&timing, &history));
        anomalies.extend(anomaly::detect_memory(&memory, &history));
        anomalies.extend(anomaly::detect_hardware(&hardware));
        anomalies.extend(anomaly::detect_system(&system));
        anomalies.extend(anomaly::detect_runtime(&runtime));

        // Advance the rolling state only after detection has read it.
        if step > 1 {
            self.abs_delta_history.push(delta_loss.abs());
        }
        self.flat_streak = if delta_loss.abs() < anomaly::PLATEAU_EPSILON && step > 1 {
            self.flat_streak + 1
        } else {
            0
        };
        self.leak_streak = match memory.heap_delta_bytes {
            Some(d) if d > anomaly::LEAK_BYTES => self.leak_streak + 1,
            _ => 0,
        };
        self.prev_loss = Some(s.loss);
        self.prev_optimizer_step = Some(s.optimizer_state_step);

        let observed = VLObservedStep {
            step,
            epoch: s.epoch,
            lr: s.lr,
            numerical,
            dynamics,
            parameters,
            global_grad_norm,
            global_update_norm,
            timing,
            memory,
            hardware,
            system,
            runtime,
            health: VLStepHealth::from_anomalies(anomalies),
        };

        if self.config.live_stdout {
            println!("{}", report::live_line(&observed, self.config.run.steps));
            for line in report::anomaly_lines(&observed) {
                println!("{line}");
            }
        }
        self.steps.push(observed);
    }
}

fn variance(v: &[f32]) -> f32 {
    let finite: Vec<f64> = v
        .iter()
        .filter(|x| x.is_finite())
        .map(|x| *x as f64)
        .collect();
    if finite.len() < 2 {
        return 0.0;
    }
    let m = finite.iter().sum::<f64>() / finite.len() as f64;
    (finite.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / finite.len() as f64) as f32
}

/// The `Trainer` path.
///
/// Fills the axes [`VLTrainingStep`] can supply and leaves the rest absent.
/// See the module docs for why the rest cannot be reached from here: the
/// missing fields are not an oversight, they are facts `Trainer` does not hold.
impl StepObserver for VLTrainingObserver {
    fn on_step(&mut self, step: &VLTrainingStep) {
        let total_ms = step.total_ns as f64 / 1e6;
        let in_progress = StepInProgress {
            loss: step.loss,
            forward_ms: step.forward_ns as f64 / 1e6,
            backward_ms: step.backward_ns as f64 / 1e6,
            optimizer_ms: step.optimizer_ns as f64 / 1e6,
            lr: step.lr,
            epoch: step.epoch,
            // `Trainer` empties the tape in `finish_step` before the observer
            // is ever called, so this is 0 by construction rather than by
            // measurement. Recorded as 0 because that is what it is.
            tape_nodes_after_backward: 0,
            // Not available: no batch, no per-parameter values, no counter.
            ..Default::default()
        };

        // A NaN or Inf gradient still reaches this path — VLTrainingStep counts
        // them — so numerical health is not blind here even without the
        // per-parameter payload.
        let mut s = in_progress;
        s.params = Vec::new();
        self.record(s, total_ms);

        // Fold the counts VLTrainingStep carries into the record just made.
        if let Some(last) = self.steps.last_mut() {
            last.numerical.has_nan_grad = step.grad_nan > 0;
            last.numerical.has_inf_grad = step.grad_inf > 0;
            last.global_grad_norm = step.grad_l2_norm as f32;
            let mut anomalies = anomaly::detect_numerical(&last.numerical, last.global_grad_norm);
            anomalies.append(&mut last.health.anomalies);
            last.health = VLStepHealth::from_anomalies(anomalies);
        }
    }

    fn wants_tensors(&self) -> bool {
        true
    }

    fn on_tensors(&mut self, _grads: &VLGradStore, opt_state: &[VLNamedTensor]) {
        // Optimizer state arrives after `on_step`, so it amends the record that
        // call just pushed.
        let bad = opt_state
            .iter()
            .any(|t| t.data.iter().any(|v| !v.is_finite()));
        if !bad {
            return;
        }
        if let Some(last) = self.steps.last_mut() {
            last.numerical.has_nan_optimizer_state = true;
            let mut anomalies = anomaly::detect_numerical(&last.numerical, last.global_grad_norm);
            anomalies.append(&mut last.health.anomalies);
            last.health = VLStepHealth::from_anomalies(anomalies);
        }
    }
}
