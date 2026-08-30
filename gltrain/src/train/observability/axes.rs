//! Stummañ Deskiñ: the eight axes a step is recorded along.
//!
//! Every type here is `VL`: plain data, derived traits, no behaviour. The
//! detectors in [`super::anomaly`] read them, the writers in [`super::report`]
//! serialize them, and neither owns them.
//!
//! # Why every platform field is an `Option`
//!
//! Axes 5 to 7 read the operating system, and what a given OS exposes is not a
//! property this crate controls. A CPU temperature exists on a Linux laptop
//! with a `thermal_zone0`, does not exist in a container with `/sys` masked,
//! and does not exist on Windows at all. Encoding that as `f32` with a
//! sentinel would make "unavailable" indistinguishable from "0 °C", which is a
//! real reading. `None` says the thing was not measurable; it never says the
//! machine is healthy.

/// Token counts for the batch a step consumed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct VLBatchInfo {
    /// Positions in the batch, padding included.
    pub total_tokens: usize,
    /// Positions whose label is not `IGNORE_INDEX`.
    pub supervised_tokens: usize,
    /// Sequences in the batch.
    pub sequences: usize,
}

impl VLBatchInfo {
    /// Fraction of the batch that carries training signal, 0.0 when empty.
    pub fn supervised_ratio(&self) -> f32 {
        if self.total_tokens == 0 {
            return 0.0;
        }
        self.supervised_tokens as f32 / self.total_tokens as f32
    }
}

/// One parameter as the loop hands it over, before any statistic is derived.
///
/// The caller supplies the values because the observer cannot reach them: a
/// gradient store is keyed by `TensorId`, which is process-global and carries
/// no name, and parameter values before and after an update exist only at the
/// call site that performed it.
#[derive(Debug, Clone, PartialEq)]
pub struct VLParamSnapshot {
    /// Stable parameter name, the key everything downstream joins on.
    pub name: String,
    /// Gradient for this step, flat. Empty when the parameter received none.
    pub grad: Vec<f32>,
    /// Values immediately before the optimizer wrote.
    pub before: Vec<f32>,
    /// Values immediately after.
    pub after: Vec<f32>,
    /// Whether the optimizer is allowed to update it.
    pub trainable: bool,
    /// For an embedding table, the width of one row — used to report how many
    /// rows received a gradient. `None` for a parameter with no row structure.
    pub row_width: Option<usize>,
}

/// Axis 1 — numerical anomalies. The silent killers.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct VLNumericalHealth {
    /// The loss is NaN.
    pub has_nan_loss: bool,
    /// The loss is infinite.
    pub has_inf_loss: bool,
    /// Some gradient element is NaN.
    pub has_nan_grad: bool,
    /// Some gradient element is infinite.
    pub has_inf_grad: bool,
    /// Some post-update parameter element is NaN.
    pub has_nan_param: bool,
    /// Some post-update parameter element is infinite.
    pub has_inf_param: bool,
    /// Some AdamW moment is non-finite.
    pub has_nan_optimizer_state: bool,
    /// Trainable parameters whose update norm was exactly zero.
    ///
    /// Exactly, not approximately. A parameter that moved by 1e-30 is training
    /// badly; one that moved by 0.0 is not connected to the graph, and those
    /// are different bugs with different fixes.
    pub dead_params: Vec<String>,
}

impl VLNumericalHealth {
    /// Whether any NaN was seen anywhere.
    pub fn any_nan(&self) -> bool {
        self.has_nan_loss || self.has_nan_grad || self.has_nan_param || self.has_nan_optimizer_state
    }

    /// Whether any infinity was seen anywhere.
    pub fn any_inf(&self) -> bool {
        self.has_inf_loss || self.has_inf_grad || self.has_inf_param
    }
}

/// Axis 2 — training dynamics. Is the model actually learning?
#[derive(Debug, Clone, PartialEq, Default)]
pub struct VLDynamics {
    /// This step's loss.
    pub loss: f32,
    /// `loss - previous_loss`. Negative is progress. Zero on the first step.
    pub delta_loss: f32,
    /// `loss / loss_at_step_1`. Trending to zero means learning.
    pub loss_ratio: f32,
    /// Mean of the last five losses, this one included.
    pub loss_running_mean_5: f32,
    /// `exp(loss)`. Infinite for a large loss, which is honest rather than clamped.
    pub perplexity: f32,
    /// Positions that contributed to the loss.
    pub supervised_token_count: usize,
    /// Positions in the batch, padding included.
    pub total_token_count: usize,
    /// `supervised / total`.
    pub supervised_ratio: f32,
    /// Variance of the per-sequence losses in this batch.
    ///
    /// Zero when the caller did not supply per-sequence losses. High variance
    /// means the batch mixes easy and hard sequences, which is a source of
    /// step-to-step instability that a single averaged loss hides completely.
    pub batch_loss_variance: f32,
}

/// Axis 3 — one parameter's health this step.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct VLParamHealth {
    /// Parameter name.
    pub name: String,
    /// Whether the optimizer may write it.
    pub trainable: bool,
    /// Euclidean norm of the gradient.
    pub grad_norm: f32,
    /// Largest absolute gradient element.
    pub grad_max: f32,
    /// Smallest absolute gradient element.
    pub grad_min: f32,
    /// Mean gradient element. Near zero is what a stable step looks like.
    pub grad_mean: f32,
    /// Euclidean norm of the actual parameter change.
    pub update_norm: f32,
    /// Largest absolute parameter change.
    pub update_max: f32,
    /// Euclidean norm of the parameter after the update.
    pub weight_norm: f32,
    /// Mean of the previous five steps' `grad_norm`, for spike detection.
    /// Zero when there is no history yet.
    pub grad_norm_running_mean_5: f32,
    /// Mean of the previous five steps' `update_norm`.
    ///
    /// Tracked separately from the gradient mean because an update must be
    /// judged against its own history, never against the gradient that
    /// produced it: AdamW normalizes the update to roughly `lr` per element
    /// whatever the gradient was, so the ratio between them is large on every
    /// healthy step and carries no signal at all.
    pub update_norm_running_mean_5: f32,
    /// For a row-structured parameter, how many rows received a non-zero
    /// gradient. `None` when the parameter has no row structure.
    pub rows_updated: Option<usize>,
    /// Rows in total, when `rows_updated` is populated.
    pub rows_total: Option<usize>,
}

/// Axis 4 — where the time went.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct VLTiming {
    /// Fetching and collating the batch.
    pub data_load_ms: f64,
    /// The forward pass.
    pub forward_ms: f64,
    /// Backward, including `finish_step`.
    pub backward_ms: f64,
    /// The optimizer update.
    pub optimizer_ms: f64,
    /// Writing a checkpoint. Zero when none was written.
    pub checkpoint_ms: f64,
    /// The whole step.
    pub total_step_ms: f64,
    /// Supervised positions per second.
    pub supervised_tokens_per_sec: f64,
    /// All positions per second.
    pub total_tokens_per_sec: f64,
    /// Forward as a percentage of the step.
    pub forward_pct: f32,
    /// Backward as a percentage of the step.
    pub backward_pct: f32,
    /// Optimizer as a percentage of the step.
    pub optimizer_pct: f32,
    /// Median step time over the run so far.
    pub step_time_median_ms: f64,
    /// 95th percentile step time over the run so far.
    pub step_time_p95_ms: f64,
}

/// Axis 5 — process memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct VLMemorySnapshot {
    /// Resident set size. `None` when the platform did not answer.
    pub heap_rss_bytes: Option<u64>,
    /// Change since the previous step. Negative means memory was returned.
    pub heap_delta_bytes: Option<i64>,
    /// Largest RSS seen so far in this run.
    pub peak_rss_bytes: Option<u64>,
    /// System memory believed available, for the pressure check.
    pub available_ram_bytes: Option<u64>,
    /// Allocation count delta, when a counting allocator is installed.
    ///
    /// This crate installs none, so it is `None` unless the binary embedding
    /// the observer sets one up and feeds it in.
    pub alloc_count_delta: Option<i64>,
}

impl VLMemorySnapshot {
    /// RSS in megabytes, for display.
    pub fn heap_rss_mb(&self) -> Option<f32> {
        self.heap_rss_bytes.map(|b| b as f32 / (1024.0 * 1024.0))
    }
}

/// Axis 6 — hardware, best effort.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct VLHardwareSnapshot {
    /// Package temperature in Celsius.
    pub cpu_temp_celsius: Option<f32>,
    /// Current core frequency in MHz.
    pub cpu_freq_mhz: Option<f32>,
    /// The frequency observed on step 1, the throttle baseline.
    pub cpu_freq_baseline_mhz: Option<f32>,
    /// Whether the current frequency is below 90% of the baseline.
    ///
    /// `false` when either reading is missing. An unmeasurable machine is not
    /// a throttling one, and claiming otherwise would fill the log with
    /// warnings on every platform that does not expose `cpufreq`.
    pub throttle_detected: bool,
    /// Process CPU utilisation since the previous step, as a percentage of one
    /// core-second per wall-second. Can exceed 100 on a threaded backend.
    pub cpu_usage_pct: Option<f32>,
}

/// Axis 7 — operating-system signals.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct VLSystemSnapshot {
    /// Threads in this process.
    pub thread_count: Option<u32>,
    /// Voluntary context switches since the previous step.
    pub voluntary_ctx_switches: Option<u64>,
    /// Major faults since the previous step. Any is a disk hit.
    pub page_faults_major: Option<u64>,
    /// Minor faults since the previous step.
    pub page_faults_minor: Option<u64>,
}

/// Axis 8 — structural health of the run itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct VLRuntimeHealth {
    /// Whether a checkpoint written this step reloaded bit-for-bit.
    /// `None` when no checkpoint was written.
    pub checkpoint_verified: Option<bool>,
    /// Nodes left on the tape after `finish_step`. Must be zero (KL-006).
    pub tape_nodes_after_backward: usize,
    /// The optimizer's internal step counter after the update.
    pub optimizer_state_step: u64,
    /// Whether that counter advanced by exactly one this step.
    pub optimizer_stepped: bool,
}

/// One fully observed training step: all eight axes plus the verdict.
#[derive(Debug, Clone, PartialEq)]
pub struct VLObservedStep {
    /// One-based step index, as a human counts them.
    pub step: usize,
    /// Epoch, zero when the loop has no epoch structure.
    pub epoch: usize,
    /// Learning rate in force for this step.
    pub lr: f64,
    /// Axis 1.
    pub numerical: VLNumericalHealth,
    /// Axis 2.
    pub dynamics: VLDynamics,
    /// Axis 3, one entry per parameter, in the order the caller supplied them.
    pub parameters: Vec<VLParamHealth>,
    /// Euclidean norm over every parameter gradient concatenated. The number
    /// `clip_grad_norm_` would act on.
    pub global_grad_norm: f32,
    /// Euclidean norm over every parameter update concatenated.
    pub global_update_norm: f32,
    /// Axis 4.
    pub timing: VLTiming,
    /// Axis 5.
    pub memory: VLMemorySnapshot,
    /// Axis 6.
    pub hardware: VLHardwareSnapshot,
    /// Axis 7.
    pub system: VLSystemSnapshot,
    /// Axis 8.
    pub runtime: VLRuntimeHealth,
    /// The synthesized verdict for this step.
    pub health: super::anomaly::VLStepHealth,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn supervised_ratio_is_zero_for_an_empty_batch_rather_than_nan() {
        assert_eq!(VLBatchInfo::default().supervised_ratio(), 0.0);
        let b = VLBatchInfo {
            total_tokens: 10,
            supervised_tokens: 4,
            sequences: 1,
        };
        assert!((b.supervised_ratio() - 0.4).abs() < 1e-6);
    }

    #[test]
    fn any_nan_and_any_inf_cover_every_source_separately() {
        let mut n = VLNumericalHealth::default();
        assert!(!n.any_nan() && !n.any_inf());

        n.has_nan_optimizer_state = true;
        assert!(n.any_nan(), "optimizer state must count as a NaN source");
        assert!(!n.any_inf(), "a NaN is not an Inf");

        let inf = VLNumericalHealth {
            has_inf_grad: true,
            ..Default::default()
        };
        assert!(inf.any_inf() && !inf.any_nan());
    }

    /// `None` must stay distinguishable from a real zero reading.
    #[test]
    fn an_unmeasured_memory_snapshot_reports_none_not_zero() {
        assert_eq!(VLMemorySnapshot::default().heap_rss_mb(), None);
        let m = VLMemorySnapshot {
            heap_rss_bytes: Some(0),
            ..Default::default()
        };
        assert_eq!(
            m.heap_rss_mb(),
            Some(0.0),
            "0 bytes is a reading, not absence"
        );
    }
}
