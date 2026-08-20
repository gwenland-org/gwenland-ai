//! Drives a stumman training run under observation (D-05).
//!
//! # Why there is no `--dataset <path>`
//!
//! `architecture/glbench-v3/DESIGN.md` §8 writes the command as
//! `glbench train --model <path> --dataset <path>`. Neither flag has a subject
//! at stumman M2, and inventing one would be worse than saying so:
//!
//! - **No dataset loader exists.** `VLMicroDataset` is built in memory by
//!   `new`/`push` or by `synthetic_regression`; nothing reads a file. A
//!   `--dataset <path>` flag would need a file format glbench invented, which
//!   is a format nobody writes.
//! - **No model is loaded.** `Trainer::new` generates the frozen base weight
//!   from a seed. There is no `.gguf` or `.gllm` in the loop at all (design
//!   F-05).
//!
//! So the workload is described by the parameters that genuinely determine it —
//! layer shape, rank, sample count, epochs, seeds — and every one of them is
//! archived, which makes the run reproducible in a way a path never would be.
//! When stumman grows a real dataset loader, `--dataset` becomes meaningful and
//! this comment becomes the record of why it was not.

use std::path::PathBuf;
use std::time::Instant;

use stumman::backend::GlProc;
use stumman::{Trainer, VLMicroDataset, VLTrainerConfig};

use crate::core::availability::{self, VLAvailabilityMap};
use crate::core::inference::VLInferenceSession;
use crate::core::metrics::{IterationMetrics, MeasurementSet};
use crate::core::mode::{ENInferenceRole, ENSessionMode};
use crate::core::result::SessionMetadata;
use crate::core::session::BenchmarkSession;
use crate::core::workload::WorkloadSpec;
use crate::engine::metadata::EngineMetadata;
use crate::environment::hardware::EnvironmentSnapshot;
use crate::numerical::scope::ENBitScope;
use crate::training::adapter::VLAdapterObservation;
use crate::training::collector::{self, VLStepCollector};
use crate::training::memory::VLTrainingMemory;
use crate::training::session::VLTrainingSession;
use crate::training::{attribution, convergence};

/// Everything `glbench train` and `glbench unified` need to run.
#[derive(Debug, Clone)]
pub struct TrainArgs {
    /// Human label for the session.
    pub label: Option<String>,
    /// Input dimension of the adapted layer.
    pub d_in: usize,
    /// Output dimension.
    pub d_out: usize,
    /// LoRA rank.
    pub rank: usize,
    /// Samples in the synthetic dataset.
    pub samples: usize,
    /// Epochs to run.
    pub epochs: usize,
    /// Base learning rate.
    pub lr: f64,
    /// Seed for the adapter and the frozen base weight.
    pub seed: u64,
    /// Seed for the dataset.
    pub dataset_seed: u64,
    /// D-19 sampling. 1 archives every step.
    pub step_sample_n: usize,
    /// Convergence target. No default — see [`convergence`].
    pub target_loss: Option<f32>,
    /// Which tensor families to bit-profile.
    pub bit_scopes: Vec<ENBitScope>,
    /// Where to archive, if anywhere.
    pub out_path: Option<PathBuf>,
    /// `TrainingOnly` or `Unified`.
    pub mode: ENSessionMode,
}

impl Default for TrainArgs {
    fn default() -> TrainArgs {
        TrainArgs {
            label: None,
            d_in: 64,
            d_out: 64,
            rank: 4,
            samples: 32,
            epochs: 4,
            lr: 1e-2,
            seed: 42,
            dataset_seed: 11,
            step_sample_n: 1,
            target_loss: None,
            bit_scopes: Vec::new(),
            out_path: None,
            mode: ENSessionMode::TrainingOnly,
        }
    }
}

/// Run a training session under observation and return the finished session.
pub fn run(args: &TrainArgs) -> Result<BenchmarkSession, String> {
    if args.mode == ENSessionMode::InferenceOnly {
        return Err("training runner cannot produce an inference_only session".to_string());
    }
    if args.epochs == 0 || args.samples == 0 {
        return Err("training needs at least one epoch and one sample".to_string());
    }

    let config = VLTrainerConfig::new(args.d_in, args.d_out, args.rank, args.lr, args.seed);
    let mut trainer = Trainer::<GlProc>::new(config)
        .map_err(|e| format!("building the trainer: {e}"))?;
    let (dataset, _true_w) =
        VLMicroDataset::synthetic_regression(args.samples, args.d_in, args.d_out, args.dataset_seed)
            .map_err(|e| format!("building the dataset: {e}"))?;

    let rss_before = crate::measurement::memory::peak_rss_bytes();

    let (observer, collected) =
        VLStepCollector::new(args.step_sample_n, args.bit_scopes.clone());
    trainer.set_observer(Box::new(observer));

    let started = Instant::now();
    let epoch_losses = trainer
        .train(&dataset, args.epochs)
        .map_err(|e| format!("training: {e}"))?;
    let wall = started.elapsed();

    // The collector holds the final step back until it knows no more are
    // coming; nothing tells an observer a run has ended, so the caller flushes.
    // `clear_observer` hands back a trait object we deliberately drop — every
    // result reachable through the handle, never through the box.
    let _ = trainer
        .clear_observer()
        .ok_or("the observer vanished during the run")?;
    collector::finish(&collected);

    let rss_after = crate::measurement::memory::peak_rss_bytes();
    let out = collected.borrow();

    let adapter = VLAdapterObservation::lora(args.d_in, args.d_out, args.rank, args.rank as f32);
    let training = VLTrainingSession {
        steps_archived: out.steps.len(),
        steps: out.steps.clone(),
        steps_observed: out.steps_observed,
        step_sample_n: args.step_sample_n.max(1),
        epochs: args.epochs,
        epoch_losses: epoch_losses.clone(),
        optimizer: "adamw".to_string(),
        attribution: attribution::analyze(&out.steps),
        convergence: convergence::analyze(
            &out.steps,
            args.target_loss,
            convergence::DEFAULT_EMA_ALPHA,
            convergence::DEFAULT_PLATEAU_WINDOW,
            convergence::DEFAULT_PLATEAU_THRESHOLD,
        ),
        memory: Some(VLTrainingMemory::new(
            rss_before,
            rss_after,
            rss_after,
            adapter.trainable_parameters,
            out.optimizer_state_elements,
        )),
        adapter: Some(adapter),
        bit_profiles: out.bit_profiles.clone(),
        post_eval: match args.mode {
            // A Unified session evaluates after training. stumman M2 has no
            // generation loop, so the slot is present and carries its role
            // while its own fields stay empty — the same compatibility shape
            // `VLInferenceSession::standalone` uses.
            ENSessionMode::Unified => {
                Some(VLInferenceSession::nested(ENInferenceRole::PostTraining))
            }
            _ => None,
        },
    };

    let label = args
        .label
        .clone()
        .unwrap_or_else(|| format!("lora-r{}-{}x{}", args.rank, args.d_in, args.d_out));

    let mut metadata = SessionMetadata::new(label);
    metadata.session_mode = args.mode;
    metadata.collection_profile = Some(collection_profile(args));

    // The workload spec describes the training run in the fields it has. There
    // is no model path, so it stays empty rather than being given a fake one.
    let workload = WorkloadSpec {
        engine: "stumman".to_string(),
        seed: args.seed,
        measure_iters: args.epochs,
        ..WorkloadSpec::default()
    };

    // One "iteration" per epoch, timed. Training has no prefill/decode split,
    // so those stay zero rather than being invented — `measurements` is not
    // where a training run's numbers live; `training` is.
    let mut measurements = MeasurementSet::default();
    let per_epoch_ms = wall.as_secs_f64() * 1000.0 / args.epochs as f64;
    for _ in 0..args.epochs {
        measurements.iterations.push(IterationMetrics {
            prompt_tokens: 0,
            generated_tokens: 0,
            prefill_ms: 0.0,
            decode_ms: per_epoch_ms,
            total_ms: per_epoch_ms,
        });
    }

    let mut session = BenchmarkSession::new(
        metadata,
        EnvironmentSnapshot::probe(""),
        EngineMetadata {
            name: "stumman".into(),
            backend: "glproc".into(),
            available: true,
            model_arch: Some("lora-linear".into()),
            quantization: Some("f32".into()),
            thinking_capable: None,
        },
        workload,
        measurements,
    );

    // The §6.2 mode table: a training session's inference envelope is absent
    // for TrainingOnly and pre-training for Unified.
    session.inference = match args.mode {
        ENSessionMode::TrainingOnly => None,
        ENSessionMode::Unified => Some(VLInferenceSession::nested(ENInferenceRole::PreTraining)),
        ENSessionMode::InferenceOnly => unreachable!("rejected above"),
    };
    declare_availability(&mut session.availability, &training, args.mode)?;
    session.training = Some(training);

    Ok(session)
}

/// The `collection_profile` string, naming what was collected beyond the
/// defaults.
fn collection_profile(args: &TrainArgs) -> String {
    let mut parts = vec!["training".to_string()];
    for scope in &args.bit_scopes {
        parts.push(format!("bits+{}", scope.as_str()));
    }
    if args.step_sample_n > 1 {
        parts.push(format!("sample{}", args.step_sample_n));
    }
    parts.join(",")
}

/// Declare every `null` this session will emit (D-10).
fn declare_availability(
    map: &mut VLAvailabilityMap,
    training: &VLTrainingSession,
    mode: ENSessionMode,
) -> Result<(), String> {
    for (path, status) in training.null_paths() {
        if status.requires_note() {
            return Err(format!("training declared '{path}' as {status:?} with no note"));
        }
        availability::set(map, &path, status)?;
    }
    // The mode table's own requirement (§6.2).
    if mode == ENSessionMode::TrainingOnly {
        availability::set(map, "inference", crate::core::availability::ENAvailability::NotApplicable)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small() -> TrainArgs {
        TrainArgs {
            d_in: 8,
            d_out: 8,
            rank: 2,
            samples: 4,
            epochs: 2,
            ..TrainArgs::default()
        }
    }

    /// Gate 4's core claim: a real LoRA run on stumman produces a session.
    #[test]
    fn a_real_training_run_produces_a_training_session() {
        let session = run(&small()).expect("training runs");

        let training = session.training.as_ref().expect("training block present");
        assert_eq!(training.steps_observed, 8, "2 epochs x 4 samples");
        assert_eq!(training.steps_archived, 8, "sample_n 1 archives everything");
        assert_eq!(training.epoch_losses.len(), 2);
        assert_eq!(session.metadata.session_mode, ENSessionMode::TrainingOnly);
        assert!(session.inference.is_none(), "training_only has no inference block");
    }

    #[test]
    fn every_sub_report_a_real_run_can_fill_is_filled() {
        let session = run(&small()).expect("training runs");
        let t = session.training.as_ref().unwrap();

        assert!(t.attribution.is_some(), "steps were observed, so time is attributable");
        assert!(t.convergence.is_some());
        assert!(t.memory.is_some());
        let adapter = t.adapter.as_ref().expect("adapter is known from the config");
        assert_eq!(adapter.rank, 2);
        assert_eq!(adapter.trainable_parameters, 2 * (8 + 8));
    }

    /// D-19 end to end, through a real run.
    #[test]
    fn step_sampling_thins_the_archive_but_keeps_the_endpoints() {
        let mut args = small();
        args.samples = 5;
        args.epochs = 2;
        args.step_sample_n = 4;

        let session = run(&args).expect("training runs");
        let t = session.training.as_ref().unwrap();

        assert_eq!(t.steps_observed, 10);
        assert_eq!(t.step_sample_n, 4);
        let indices: Vec<usize> = t.steps.iter().map(|s| s.index).collect();
        assert_eq!(indices, vec![0, 4, 8, 9], "endpoints survive thinning");
        assert!(t.steps_archived < t.steps_observed);
        // The epoch losses come from the trainer and are never thinned.
        assert_eq!(t.epoch_losses.len(), 2);
    }

    #[test]
    fn a_unified_run_carries_both_inference_roles() {
        let mut args = small();
        args.mode = ENSessionMode::Unified;

        let session = run(&args).expect("training runs");
        assert_eq!(
            session.inference.as_ref().map(|i| i.role),
            Some(ENInferenceRole::PreTraining),
            "the outer envelope is the baseline"
        );
        assert_eq!(
            session.training.as_ref().unwrap().post_eval.as_ref().map(|p| p.role),
            Some(ENInferenceRole::PostTraining),
        );
    }

    #[test]
    fn an_inference_only_mode_is_refused_rather_than_silently_retyped() {
        let mut args = small();
        args.mode = ENSessionMode::InferenceOnly;
        let err = run(&args).unwrap_err();
        assert!(err.contains("inference_only"), "got {err}");
    }

    #[test]
    fn a_zero_epoch_or_zero_sample_run_is_refused() {
        for (epochs, samples) in [(0, 4), (2, 0)] {
            let mut args = small();
            args.epochs = epochs;
            args.samples = samples;
            assert!(run(&args).is_err(), "epochs={epochs} samples={samples}");
        }
    }

    #[test]
    fn a_target_loss_is_carried_into_the_convergence_report() {
        let mut args = small();
        args.target_loss = Some(1e9); // trivially reached
        let session = run(&args).expect("training runs");
        let c = session.training.as_ref().unwrap().convergence.as_ref().unwrap();
        assert_eq!(c.target_loss, Some(1e9));
        assert_eq!(c.steps_to_target, Some(0), "the first step already beats it");
    }

    #[test]
    fn the_collection_profile_names_what_was_collected() {
        let mut args = small();
        args.bit_scopes = vec![ENBitScope::Gradients];
        args.step_sample_n = 3;
        let session = run(&args).expect("training runs");

        let profile = session.metadata.collection_profile.as_deref().unwrap();
        assert!(profile.contains("training"), "got {profile}");
        assert!(profile.contains("bits+gradients"), "got {profile}");
        assert!(profile.contains("sample3"), "got {profile}");
    }

    #[test]
    fn a_gradient_bit_scope_produces_profiles_from_the_real_run() {
        let mut args = small();
        args.bit_scopes = vec![ENBitScope::Gradients];
        let session = run(&args).expect("training runs");
        let t = session.training.as_ref().unwrap();

        assert!(!t.bit_profiles.is_empty(), "gradients must have been profiled");
        assert!(t.bit_profiles.iter().all(|b| b.scope.scope == ENBitScope::Gradients));
        assert!(
            t.bit_profiles.iter().any(|b| b.scope.profile.count > 0),
            "a profile must describe real elements"
        );
    }

    #[test]
    fn an_optimizer_bit_scope_profiles_the_state_tensors() {
        let mut args = small();
        args.bit_scopes = vec![ENBitScope::Optimizer];
        let session = run(&args).expect("training runs");
        let t = session.training.as_ref().unwrap();

        assert!(t.bit_profiles.iter().any(|b| b.scope.scope == ENBitScope::Optimizer));
        assert!(
            t.memory.as_ref().unwrap().optimizer_state_bytes.is_some(),
            "requesting the payload also makes the state footprint known"
        );
    }

    /// Without a bit scope the O(n) payload is never requested, so the
    /// optimizer footprint is honestly unavailable rather than zero.
    #[test]
    fn no_bit_scope_leaves_the_optimizer_footprint_unavailable_not_zero() {
        let session = run(&small()).expect("training runs");
        let memory = session.training.as_ref().unwrap().memory.as_ref().unwrap();
        assert_eq!(memory.optimizer_state_bytes, None);
        assert!(memory.parameter_bytes > 0, "the derived figure is still exact");
    }
}
