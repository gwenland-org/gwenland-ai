//! Stummañ Deskiñ: `VLTrainingObserver` driven end to end.
//!
//! The detectors have unit tests next to them in `anomaly.rs`, where each
//! threshold is exercised against fabricated numbers. These tests are the
//! other half: they drive the observer through its hook API the way a training
//! loop does, and check that a step actually reaches every axis, that the
//! rolling state advances between steps, and that the reports come out
//! readable.

use gltrain::autograd::grad_store::VLGradStore;
use gltrain::checkpoint::safetensors;
use gltrain::optim::VLNamedTensor;
use gltrain::train::observability::{
    ENHealthLevel, ENVerdict, VLBatchInfo, VLObserverConfig, VLParamSnapshot, VLRunConfig,
    VLTrainingObserver,
};
use gltrain::train::{StepObserver, Trainer, VLMicroDataset, VLTrainerConfig, VLTrainingStep};
use gltrain::GlProc;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// Silent by default: a test suite that prints ten observability lines per
/// test buries the failures that matter.
fn observer(steps: usize) -> VLTrainingObserver {
    VLTrainingObserver::new(VLObserverConfig {
        live_stdout: false,
        report_dir: None,
        run: VLRunConfig {
            model: "ABEmbedding[64,16] + ABLinear[16,64]".into(),
            dataset: "synthetic".into(),
            batch_size: 2,
            lr: 0.02,
            steps,
            timestamp: "1700000000".into(),
        },
    })
}

fn batch() -> VLBatchInfo {
    VLBatchInfo {
        total_tokens: 100,
        supervised_tokens: 80,
        sequences: 2,
    }
}

/// A parameter that moved: gradient `g` everywhere, weights shifted by `d`.
fn param(name: &str, g: f32, d: f32, trainable: bool) -> VLParamSnapshot {
    VLParamSnapshot {
        name: name.to_string(),
        grad: vec![g; 8],
        before: vec![1.0; 8],
        after: vec![1.0 + d; 8],
        trainable,
        row_width: Some(4),
    }
}

/// Drive one whole step through every hook.
#[allow(clippy::too_many_arguments)]
fn drive(
    obs: &mut VLTrainingObserver,
    loss: f32,
    params: Vec<VLParamSnapshot>,
    opt_step: u64,
    tape_nodes: usize,
) {
    // 0.0: a mock forward does no work, so any non-zero data-load time would
    // dominate it and fire DATA_BOTTLENECK on every step. That code has its
    // own test below, where the number is the point.
    obs.on_batch_loaded(batch(), 0.0);
    obs.on_forward_complete(loss, &[loss - 0.1, loss + 0.1]);
    obs.on_backward_complete(tape_nodes);
    obs.on_optimizer_step(params, &[], opt_step, 0.02);
    obs.on_step_complete();
}

fn codes(obs: &VLTrainingObserver) -> Vec<&str> {
    obs.all_anomalies()
        .into_iter()
        .map(|(_, a)| a.code)
        .collect()
}

// ── The whole taxonomy on a short run ────────────────────────────────────

/// Every axis must be populated on every step. A field left at its default
/// because nobody wired it is the failure mode this whole module exists to
/// prevent, so it is asserted directly rather than eyeballed in a report.
#[test]
fn a_three_step_loop_populates_all_eight_axes() {
    let mut obs = observer(3);
    for (i, loss) in [5.0f32, 4.5, 4.2].iter().enumerate() {
        drive(
            &mut obs,
            *loss,
            vec![
                param("embed.weight", 0.05, 0.01, true),
                param("lm_head.weight", 0.04, 0.01, true),
            ],
            i as u64 + 1,
            0,
        );
    }

    assert_eq!(obs.steps().len(), 3);
    for (i, s) in obs.steps().iter().enumerate() {
        assert_eq!(s.step, i + 1, "steps are numbered from one");

        // Axis 1 — every flag reachable, all clean on this run.
        assert!(!s.numerical.any_nan() && !s.numerical.any_inf());
        assert!(s.numerical.dead_params.is_empty());

        // Axis 2.
        assert!(s.dynamics.loss > 0.0);
        assert_eq!(s.dynamics.supervised_token_count, 80);
        assert_eq!(s.dynamics.total_token_count, 100);
        assert!((s.dynamics.supervised_ratio - 0.8).abs() < 1e-6);
        assert!(s.dynamics.perplexity.is_finite());
        assert!(
            s.dynamics.batch_loss_variance > 0.0,
            "per-sequence losses were supplied, so variance must be real"
        );

        // Axis 3 — per parameter, not aggregated away.
        assert_eq!(s.parameters.len(), 2);
        for p in &s.parameters {
            assert!(p.grad_norm > 0.0, "{} has no gradient", p.name);
            assert!(p.update_norm > 0.0, "{} did not move", p.name);
            assert!(p.weight_norm > 0.0);
            assert_eq!(p.rows_updated, Some(2), "both rows carry a gradient");
            assert_eq!(p.rows_total, Some(2));
        }
        assert!(s.global_grad_norm > 0.0);
        assert!(s.global_update_norm > 0.0);

        // Axis 4.
        assert!(s.timing.total_step_ms > 0.0);
        assert_eq!(s.timing.data_load_ms, 0.0);
        assert!(s.timing.supervised_tokens_per_sec > 0.0);
        assert!(s.timing.step_time_median_ms > 0.0);

        // Axes 5-7 are platform-dependent: the requirement is that the field
        // exists and never panics, not that this machine can answer it.
        let _ = s.memory.heap_rss_bytes;
        let _ = s.hardware.cpu_temp_celsius;
        let _ = s.system.thread_count;

        // Axis 8.
        assert_eq!(s.runtime.tape_nodes_after_backward, 0);
        assert!(
            s.runtime.optimizer_stepped,
            "step {} did not advance",
            s.step
        );
    }

    // Rolling state actually rolled.
    assert_eq!(obs.loss_curve(), vec![5.0, 4.5, 4.2]);
    assert_eq!(
        obs.steps()[0].dynamics.delta_loss,
        0.0,
        "step 1 has no previous"
    );
    assert!(obs.steps()[1].dynamics.delta_loss < 0.0);
    assert!(obs.steps()[2].parameters[0].grad_norm_running_mean_5 > 0.0);
}

#[test]
fn a_clean_run_returns_a_pass_verdict_with_no_anomalies() {
    let mut obs = observer(3);
    for (i, loss) in [5.0f32, 4.5, 4.2].iter().enumerate() {
        drive(
            &mut obs,
            *loss,
            vec![param("w", 0.05, 0.01, true)],
            i as u64 + 1,
            0,
        );
    }
    assert!(obs.all_anomalies().is_empty(), "{:?}", codes(&obs));
    let (v, reason) = obs.verdict();
    assert_eq!(v, ENVerdict::Pass);
    assert!(reason.contains("no anomalies"), "{reason}");
}

// ── Axis 1 ───────────────────────────────────────────────────────────────

#[test]
fn an_injected_nan_loss_fires_nan_detected_and_fails_the_run() {
    let mut obs = observer(2);
    drive(&mut obs, 5.0, vec![param("w", 0.05, 0.01, true)], 1, 0);
    drive(&mut obs, f32::NAN, vec![param("w", 0.05, 0.01, true)], 2, 0);

    assert!(obs.steps()[1].numerical.has_nan_loss);
    assert!(codes(&obs).contains(&"NAN_DETECTED"));
    assert_eq!(obs.steps()[1].health.level, ENHealthLevel::Critical);
    assert_eq!(obs.verdict().0, ENVerdict::Fail);
}

#[test]
fn a_nan_gradient_is_caught_even_when_the_loss_is_finite() {
    let mut obs = observer(1);
    let mut p = param("w", 0.05, 0.01, true);
    p.grad[3] = f32::NAN;
    drive(&mut obs, 5.0, vec![p], 1, 0);

    let s = &obs.steps()[0];
    assert!(s.numerical.has_nan_grad);
    assert!(!s.numerical.has_nan_loss, "the loss itself was finite");
    // The norm must survive one NaN element, or the only health signal in the
    // record is destroyed by the thing it is supposed to report.
    assert!(
        s.global_grad_norm.is_finite(),
        "one NaN poisoned the whole norm"
    );
}

#[test]
fn an_infinite_parameter_is_reported_separately_from_a_nan() {
    let mut obs = observer(1);
    let mut p = param("w", 0.05, 0.01, true);
    p.after[0] = f32::INFINITY;
    drive(&mut obs, 5.0, vec![p], 1, 0);

    assert!(obs.steps()[0].numerical.has_inf_param);
    assert!(!obs.steps()[0].numerical.any_nan());
    assert!(codes(&obs).contains(&"INF_DETECTED"));
}

/// A trainable parameter whose update norm is exactly zero is not training.
#[test]
fn a_parameter_that_does_not_move_is_reported_dead() {
    let mut obs = observer(1);
    drive(&mut obs, 5.0, vec![param("stuck", 0.05, 0.0, true)], 1, 0);

    assert_eq!(
        obs.steps()[0].numerical.dead_params,
        vec!["stuck".to_string()]
    );
    assert!(codes(&obs).contains(&"DEAD_PARAMETER"));
}

#[test]
fn a_frozen_parameter_carrying_a_gradient_is_reported() {
    let mut obs = observer(1);
    drive(&mut obs, 5.0, vec![param("base", 0.05, 0.0, false)], 1, 0);

    assert!(codes(&obs).contains(&"FROZEN_PARAM_WITH_GRAD"));
    assert!(
        obs.steps()[0].numerical.dead_params.is_empty(),
        "a frozen parameter is not dead, it is frozen"
    );
}

// ── Axis 2 ───────────────────────────────────────────────────────────────

#[test]
fn a_loss_spike_fires_against_the_established_trend() {
    let mut obs = observer(6);
    // Four steady steps of -0.1, then one of -1.0.
    for (i, loss) in [5.0f32, 4.9, 4.8, 4.7, 4.6].iter().enumerate() {
        drive(
            &mut obs,
            *loss,
            vec![param("w", 0.05, 0.01, true)],
            i as u64 + 1,
            0,
        );
    }
    assert!(
        !codes(&obs).contains(&"LOSS_SPIKE"),
        "the steady run must be quiet"
    );

    drive(&mut obs, 3.6, vec![param("w", 0.05, 0.01, true)], 6, 0);
    assert!(codes(&obs).contains(&"LOSS_SPIKE"));
    assert_eq!(
        obs.verdict().0,
        ENVerdict::Warn,
        "a spike warns, it does not fail"
    );
}

#[test]
fn a_flat_loss_fires_a_plateau_on_the_fifth_consecutive_step() {
    let mut obs = observer(7);
    drive(&mut obs, 5.0, vec![param("w", 0.05, 0.01, true)], 1, 0);
    // Steps 2-5 are flat: four flat deltas, one short of the threshold.
    for i in 2..=5u64 {
        drive(&mut obs, 5.0, vec![param("w", 0.05, 0.01, true)], i, 0);
    }
    assert!(
        !codes(&obs).contains(&"LOSS_PLATEAU"),
        "four flat steps is not yet a plateau: {:?}",
        codes(&obs)
    );

    drive(&mut obs, 5.0, vec![param("w", 0.05, 0.01, true)], 6, 0);
    assert!(codes(&obs).contains(&"LOSS_PLATEAU"));
}

#[test]
fn a_batch_with_little_supervision_is_reported() {
    let mut obs = observer(1);
    obs.on_batch_loaded(
        VLBatchInfo {
            total_tokens: 100,
            supervised_tokens: 20,
            sequences: 2,
        },
        0.5,
    );
    obs.on_forward_complete(5.0, &[]);
    obs.on_backward_complete(0);
    obs.on_optimizer_step(vec![param("w", 0.05, 0.01, true)], &[], 1, 0.02);
    obs.on_step_complete();

    assert!(codes(&obs).contains(&"LOW_SUPERVISION"));
}

/// A data pipeline slower than the compute it feeds is a real finding, and it
/// is the one anomaly a mock loop raises by construction: a forward pass that
/// does no work is trivially outrun. Asserted deliberately here rather than
/// tolerated everywhere.
#[test]
fn a_data_pipeline_slower_than_the_forward_pass_is_reported() {
    let mut obs = observer(1);
    obs.on_batch_loaded(batch(), 50.0);
    obs.on_forward_complete(5.0, &[]);
    obs.on_backward_complete(0);
    obs.on_optimizer_step(vec![param("w", 0.05, 0.01, true)], &[], 1, 0.02);
    obs.on_step_complete();

    assert_eq!(obs.steps()[0].timing.data_load_ms, 50.0);
    assert!(codes(&obs).contains(&"DATA_BOTTLENECK"));
    assert_eq!(obs.verdict().0, ENVerdict::Warn);
}

// ── Axis 3 ───────────────────────────────────────────────────────────────

/// The boundary the spec pins: exactly 3x fires. Driven through the observer
/// here rather than the detector, so the running mean it compares against is
/// one the observer actually accumulated.
#[test]
fn a_gradient_spike_fires_at_exactly_three_times_the_running_mean() {
    let mut obs = observer(4);
    for i in 1..=3u64 {
        drive(
            &mut obs,
            5.0 - i as f32 * 0.1,
            vec![param("w", 0.1, 0.01, true)],
            i,
            0,
        );
    }
    // The mean of the last three is 0.1 per element over 8 elements. Tripling
    // the elements triples the norm.
    drive(&mut obs, 4.6, vec![param("w", 0.3, 0.01, true)], 4, 0);

    let spikes: Vec<_> = obs
        .all_anomalies()
        .into_iter()
        .filter(|(_, a)| a.code == "GRAD_SPIKE")
        .collect();
    assert_eq!(spikes.len(), 1, "exactly one spike, on step 4");
    assert_eq!(spikes[0].0, 4);
    assert_eq!(spikes[0].1.param.as_deref(), Some("w"));
}

#[test]
fn a_vanishing_gradient_is_reported_for_a_trainable_parameter_only() {
    let mut obs = observer(1);
    drive(
        &mut obs,
        5.0,
        vec![
            param("trainable", 0.0, 0.0, true),
            param("frozen", 0.0, 0.0, false),
        ],
        1,
        0,
    );
    let vanishing: Vec<_> = obs
        .all_anomalies()
        .into_iter()
        .filter(|(_, a)| a.code == "GRADIENT_VANISHING")
        .collect();
    assert_eq!(vanishing.len(), 1);
    assert_eq!(vanishing[0].1.param.as_deref(), Some("trainable"));
}

/// Row coverage is what tells a sparse embedding update from a dense one.
#[test]
fn row_coverage_counts_only_rows_that_received_a_gradient() {
    let mut obs = observer(1);
    let p = VLParamSnapshot {
        name: "embed.weight".into(),
        // Four rows of width 2; rows 0 and 2 have gradient, 1 and 3 do not.
        grad: vec![0.1, 0.1, 0.0, 0.0, 0.2, 0.2, 0.0, 0.0],
        before: vec![1.0; 8],
        after: vec![1.01; 8],
        trainable: true,
        row_width: Some(2),
    };
    obs.on_batch_loaded(batch(), 0.1);
    obs.on_forward_complete(5.0, &[]);
    obs.on_backward_complete(0);
    obs.on_optimizer_step(vec![p], &[], 1, 0.02);
    obs.on_step_complete();

    let health = &obs.steps()[0].parameters[0];
    assert_eq!(health.rows_updated, Some(2));
    assert_eq!(health.rows_total, Some(4));
}

// ── Axis 8 ───────────────────────────────────────────────────────────────

/// KL-006 requires the tape to be empty before any weight is written. A
/// non-zero count means a backward closure could still observe the update.
#[test]
fn a_leftover_tape_node_fires_tape_leak_as_critical() {
    let mut obs = observer(1);
    drive(&mut obs, 5.0, vec![param("w", 0.05, 0.01, true)], 1, 7);

    assert_eq!(obs.steps()[0].runtime.tape_nodes_after_backward, 7);
    assert!(codes(&obs).contains(&"TAPE_LEAK"));
    assert_eq!(obs.steps()[0].health.level, ENHealthLevel::Critical);
    assert_eq!(obs.verdict().0, ENVerdict::Fail);
}

#[test]
fn an_optimizer_counter_that_does_not_advance_is_reported() {
    let mut obs = observer(2);
    drive(&mut obs, 5.0, vec![param("w", 0.05, 0.01, true)], 1, 0);
    // The optimizer did not step: the counter is unchanged.
    drive(&mut obs, 4.9, vec![param("w", 0.05, 0.01, true)], 1, 0);

    assert!(codes(&obs).contains(&"OPTIMIZER_NOT_STEPPING"));
    assert!(
        !obs.steps()[0]
            .health
            .anomalies
            .iter()
            .any(|a| a.code == "OPTIMIZER_NOT_STEPPING"),
        "the first step has no previous counter and must not be accused"
    );
}

#[test]
fn a_non_finite_optimizer_moment_is_caught() {
    let mut obs = observer(1);
    obs.on_batch_loaded(batch(), 0.1);
    obs.on_forward_complete(5.0, &[]);
    obs.on_backward_complete(0);
    obs.on_optimizer_step(
        vec![param("w", 0.05, 0.01, true)],
        &[VLNamedTensor::new("w.v", vec![1.0, f32::NAN], vec![2])],
        1,
        0.02,
    );
    obs.on_step_complete();

    assert!(obs.steps()[0].numerical.has_nan_optimizer_state);
    assert!(codes(&obs).contains(&"NAN_DETECTED"));
}

/// A checkpoint that does not survive its own reload is corrupt. Tampered for
/// real here — a byte is flipped in the data section and the reload compared —
/// rather than by passing `false` to the hook.
#[test]
fn a_tampered_checkpoint_fires_checkpoint_corruption() {
    let path: PathBuf = std::env::temp_dir().join("gltrain_obs_tamper.safetensors");
    let original = vec![VLNamedTensor::new(
        "w",
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
    )];
    safetensors::write(&path, &original, &BTreeMap::new()).expect("write");

    // Flip the last byte, which is inside the tensor payload.
    let mut bytes = std::fs::read(&path).expect("read back");
    let last = bytes.len() - 1;
    bytes[last] ^= 0xFF;
    std::fs::write(&path, &bytes).expect("tamper");

    let reloaded = safetensors::read(&path).expect("a tampered file still parses");
    let verified = reloaded[0].data == original[0].data;
    assert!(!verified, "the tamper must actually change the values");

    let mut obs = observer(1);
    obs.on_batch_loaded(batch(), 0.1);
    obs.on_forward_complete(5.0, &[]);
    obs.on_backward_complete(0);
    obs.on_optimizer_step(vec![param("w", 0.05, 0.01, true)], &[], 1, 0.02);
    obs.on_checkpoint(verified, 1.5);
    obs.on_step_complete();

    assert_eq!(obs.steps()[0].runtime.checkpoint_verified, Some(false));
    assert_eq!(obs.steps()[0].timing.checkpoint_ms, 1.5);
    assert!(codes(&obs).contains(&"CHECKPOINT_CORRUPTION"));
    assert_eq!(obs.verdict().0, ENVerdict::Fail);

    let _ = std::fs::remove_file(&path);
}

// ── Platform tolerance ───────────────────────────────────────────────────

/// The observer must run on a machine that answers none of axes 5 to 7. On
/// Linux some of these are readable and on Windows most are not; either way
/// the run completes and the absent fields stay `None` rather than becoming a
/// zero that reads as a healthy measurement.
#[test]
fn unavailable_hardware_degrades_to_none_without_panicking() {
    let mut obs = observer(3);
    for i in 1..=3u64 {
        drive(
            &mut obs,
            5.0 - i as f32 * 0.1,
            vec![param("w", 0.05, 0.01, true)],
            i,
            0,
        );
    }

    for s in obs.steps() {
        if s.hardware.cpu_temp_celsius.is_none() {
            assert!(
                !s.health.anomalies.iter().any(|a| a.code == "HIGH_TEMP"),
                "an unmeasured temperature must not raise a heat anomaly"
            );
        }
        if s.hardware.cpu_freq_mhz.is_none() {
            assert!(
                !s.hardware.throttle_detected,
                "a machine that cannot report frequency is not throttling"
            );
        }
        if s.memory.heap_rss_bytes.is_none() {
            assert!(!s.health.anomalies.iter().any(|a| a.code == "MEMORY_LEAK"));
        }
    }
    // The first step never has a previous reading to difference against.
    assert_eq!(obs.steps()[0].memory.heap_delta_bytes, None);
}

// ── Reports ──────────────────────────────────────────────────────────────

#[test]
fn the_reports_are_written_readable_and_carry_the_verdict() {
    let dir = std::env::temp_dir().join("gltrain_obs_reports");
    let mut obs = VLTrainingObserver::new(VLObserverConfig {
        live_stdout: false,
        report_dir: Some(dir.clone()),
        run: VLRunConfig {
            model: "m".into(),
            dataset: "d".into(),
            batch_size: 2,
            lr: 0.02,
            steps: 2,
            timestamp: "1700000042".into(),
        },
    });
    drive(&mut obs, 5.0, vec![param("w", 0.05, 0.01, true)], 1, 0);
    // A tape leak on step 2, so the report has a critical anomaly to render.
    drive(&mut obs, 4.5, vec![param("w", 0.05, 0.01, true)], 2, 3);

    let (json_path, md_path) = obs
        .on_training_complete()
        .expect("writing must succeed")
        .expect("a report directory was configured");

    let doc = std::fs::read_to_string(&json_path).expect("json readable");
    let parsed = gltrain::checkpoint::json::parse(&doc).expect("the report must be valid JSON");
    let summary = parsed.get("summary").expect("summary");
    assert_eq!(summary.get("verdict").unwrap().as_str(), Some("FAIL"));
    assert_eq!(
        parsed.get("steps").unwrap().as_arr().unwrap().len(),
        2,
        "every step is in the report"
    );
    assert_eq!(
        summary
            .get("per_axis")
            .unwrap()
            .get("runtime")
            .unwrap()
            .get("tape_leak_steps")
            .unwrap()
            .as_arr()
            .unwrap()
            .len(),
        1
    );

    let md = std::fs::read_to_string(&md_path).expect("markdown readable");
    assert!(md.contains("**FAIL**"));
    assert!(md.contains("TAPE_LEAK"));
    assert!(md.contains("## 4. Anomaly log"));

    let _ = std::fs::remove_file(&json_path);
    let _ = std::fs::remove_file(&md_path);
}

#[test]
fn no_report_directory_means_no_files_and_no_error() {
    let mut obs = observer(1);
    drive(&mut obs, 5.0, vec![param("w", 0.05, 0.01, true)], 1, 0);
    assert!(obs
        .on_training_complete()
        .expect("must not error")
        .is_none());
}

// ── The Trainer path ─────────────────────────────────────────────────────

/// `Trainer` owns its observer as a `Box<dyn StepObserver>` and never gives it
/// back typed, so the test shares one through an `Arc<Mutex<_>>` and installs a
/// thin delegate. That is the same shape any real caller would use to read an
/// observer's records after a run.
struct Shared(Arc<Mutex<VLTrainingObserver>>);

impl StepObserver for Shared {
    fn on_step(&mut self, step: &VLTrainingStep) {
        self.0.lock().expect("not poisoned").on_step(step);
    }
    fn wants_tensors(&self) -> bool {
        true
    }
    fn on_tensors(&mut self, grads: &VLGradStore, opt_state: &[VLNamedTensor]) {
        self.0
            .lock()
            .expect("not poisoned")
            .on_tensors(grads, opt_state);
    }
}

/// The integration the specification asked for: `VLTrainingObserver` implements
/// the existing `StepObserver` trait and attaches to `Trainer` with zero
/// changes to `Trainer` itself.
///
/// What it records through this path is a subset, and that is a property of
/// `Trainer` rather than of the observer: `Trainer` trains a LoRA adapter on
/// `VLMicroDataset` with an MSE loss, so there are no tokens to count and no
/// before/after parameter values to difference. The fields it cannot supply
/// stay empty instead of being filled with zeros that would read as
/// measurements.
#[test]
fn the_observer_attaches_to_trainer_through_the_existing_step_observer_trait() {
    let shared = Arc::new(Mutex::new(observer(6)));
    let (dataset, _) = VLMicroDataset::synthetic_regression(3, 4, 4, 7).expect("dataset");
    let mut trainer =
        Trainer::<GlProc>::new(VLTrainerConfig::new(4, 4, 2, 0.01, 5)).expect("trainer");
    trainer.set_observer(Box::new(Shared(Arc::clone(&shared))));

    // `train` returns one mean loss per *epoch*; the observer is called per
    // step, so the two counts differ and both are asserted.
    let epoch_losses = trainer.train(&dataset, 2).expect("training runs");
    assert_eq!(epoch_losses.len(), 2, "one entry per epoch");

    let obs = shared.lock().expect("not poisoned");
    assert_eq!(
        obs.steps().len(),
        6,
        "every Trainer step reached the observer"
    );

    for s in obs.steps() {
        // What this path can supply.
        assert!(s.dynamics.loss.is_finite());
        assert!(s.timing.total_step_ms >= 0.0);
        assert!(
            s.lr > 0.0,
            "the lr comes from the optimizer, not the config"
        );
        assert!(!s.numerical.any_nan(), "a healthy LoRA step has no NaN");
        // KL-006: `finish_step` empties the tape before the observer is called.
        assert_eq!(s.runtime.tape_nodes_after_backward, 0);
        // Axes 5 to 7 are the observer's own work and are reached either way.
        let _ = s.memory.heap_rss_bytes;

        // What it cannot, and does not pretend to.
        assert_eq!(s.dynamics.total_token_count, 0, "Trainer has no tokens");
        assert!(
            s.parameters.is_empty(),
            "per-parameter values are not reachable from VLTrainingStep"
        );
    }
    assert_eq!(obs.loss_curve().len(), 6);
}
