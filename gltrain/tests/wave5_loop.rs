//! Stummañ Deskiñ + Pik: Wave 5's mini loop and its checkpoint.
//!
//! Two properties, tested separately because they fail for different reasons:
//! that ten steps of the loop actually lower the loss, and that the weights
//! survive a write/read round trip well enough to train on afterwards.
//!
//! # CPFull stays deferred
//!
//! The checkpoint here goes through [`gltrain::checkpoint::safetensors`]
//! directly. `CPLora` saves a LoRA adapter and this model has none; `CPFull` is
//! the layout that would fit and is a registered stub. `full.rs`'s own tests
//! already assert it stays one and name its blocker, so nothing is re-asserted
//! here — the point is only that this wave did not quietly implement it.

use gltrain::checkpoint::safetensors;
use gltrain::{
    ABEmbedding, ABLinear, GlProc, Module, OPAdamW, Optimizer, TPParameter, Tape, Tensor,
    VLAdamWConfig, VLNamedTensor, IGNORE_INDEX,
};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

const VOCAB: usize = 64;
const D_MODEL: usize = 16;
const N_TOKENS: usize = 128;
const STEPS: usize = 10;
const SEED: u64 = 20260823;
const LR: f64 = 0.02;

/// A deterministic synthetic batch. Small on purpose: this asserts the *loop*
/// converges, and the real corpus is already carried through the chain by
/// `wave3_forward` and `wave4_backward`. A 128-token batch keeps a ten-step
/// debug-build loop from dominating the suite's runtime.
///
/// Every third position is masked, so the mask is exercised too.
fn synthetic_batch() -> (Vec<u32>, Vec<i32>) {
    let ids: Vec<u32> = (0..N_TOKENS).map(|i| (i * 7 % VOCAB) as u32).collect();
    let labels: Vec<i32> = (0..N_TOKENS)
        .map(|i| {
            if i % 3 == 0 {
                IGNORE_INDEX
            } else {
                ((i * 11 + 3) % VOCAB) as i32
            }
        })
        .collect();
    (ids, labels)
}

fn model() -> (ABEmbedding<GlProc>, ABLinear<GlProc>) {
    (
        ABEmbedding::randn("embed", VOCAB, D_MODEL, 0.02, SEED).expect("embed"),
        ABLinear::randn("lm_head", D_MODEL, VOCAB, 0.02, SEED + 1).expect("head"),
    )
}

fn loss_of(
    embed: &ABEmbedding<GlProc>,
    head: &ABLinear<GlProc>,
    ids: &[u32],
    labels: &[i32],
) -> f32 {
    let tape = Arc::new(Mutex::new(Tape::new()));
    let hidden = embed.forward_ids(ids, &tape).expect("gather");
    let logits = head.forward(&hidden, &tape).expect("projection");
    logits
        .log_softmax()
        .expect("log_softmax")
        .masked_cross_entropy(labels)
        .expect("loss")
        .item()
        .expect("scalar")
}

/// Run `STEPS` updates, returning the per-step losses recorded *before* each
/// step's update — which is what a training log reports.
fn train(
    embed: &mut ABEmbedding<GlProc>,
    head: &mut ABLinear<GlProc>,
    ids: &[u32],
    labels: &[i32],
) -> Vec<f32> {
    let tape = Arc::new(Mutex::new(Tape::new()));
    let mut opt = OPAdamW::<GlProc>::new(VLAdamWConfig {
        lr: LR,
        weight_decay: 0.0,
        ..Default::default()
    });
    let mut curve = Vec::with_capacity(STEPS);

    for _ in 0..STEPS {
        let hidden = embed.forward_ids(ids, &tape).expect("gather");
        let logits = head.forward(&hidden, &tape).expect("projection");
        let loss = logits
            .log_softmax()
            .expect("log_softmax")
            .masked_cross_entropy(labels)
            .expect("loss");
        curve.push(loss.item().expect("scalar"));

        let grads = {
            let mut guard = Tape::lock(&tape);
            guard.backward().expect("backward");
            guard.finish_step()
        };
        let mut params: Vec<&mut TPParameter<GlProc>> = Vec::new();
        params.extend(embed.parameters_mut());
        params.extend(head.parameters_mut());
        opt.step(&mut params, &grads).expect("step");
    }
    curve
}

/// A unique path per test, so the suite's threads cannot collide on one file.
fn scratch(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("gltrain_wave5_{name}.safetensors"))
}

// ── The loop ─────────────────────────────────────────────────────────────

#[test]
fn ten_steps_lower_the_loss_monotonically() {
    let (ids, labels) = synthetic_batch();
    let (mut embed, mut head) = model();
    let curve = train(&mut embed, &mut head, &ids, &labels);

    assert_eq!(curve.len(), STEPS);
    assert!(curve.iter().all(|v| v.is_finite()), "curve: {curve:?}");
    assert!(curve.iter().all(|v| *v > 0.0), "curve: {curve:?}");

    // The batch is fixed across all ten steps, so there is no batch-to-batch
    // variance to excuse a rise. Every step must improve on the last.
    for (i, w) in curve.windows(2).enumerate() {
        assert!(
            w[1] < w[0],
            "step {} rose: {} -> {} (full curve {curve:?})",
            i + 2,
            w[0],
            w[1]
        );
    }
    assert!(
        curve[0] - curve[STEPS - 1] > 0.1,
        "ten steps barely moved the loss: {curve:?}"
    );
}

/// The loop must be reproducible from its seed, or a surprising curve cannot
/// be debugged.
#[test]
fn the_loop_is_deterministic_in_its_seed() {
    let (ids, labels) = synthetic_batch();
    let (mut e1, mut h1) = model();
    let (mut e2, mut h2) = model();
    assert_eq!(
        train(&mut e1, &mut h1, &ids, &labels),
        train(&mut e2, &mut h2, &ids, &labels)
    );
}

// ── The checkpoint ───────────────────────────────────────────────────────

/// safetensors stores f32 verbatim — no dtype change, no quantization — so
/// "allclose" here is exact equality, and asserting anything looser would hide
/// a writer that silently narrowed a value.
#[test]
fn the_checkpoint_round_trips_every_tensor_exactly() {
    let (ids, labels) = synthetic_batch();
    let (mut embed, mut head) = model();
    train(&mut embed, &mut head, &ids, &labels);

    let tensors = vec![
        VLNamedTensor::new(
            "embed.weight",
            embed.weight().to_vec().expect("host"),
            embed.weight().shape().to_vec(),
        ),
        VLNamedTensor::new(
            "lm_head.weight",
            head.weight().to_vec().expect("host"),
            head.weight().shape().to_vec(),
        ),
        // Untrained: ABLinear's bias is [1, d_out] and add() is exact-shape, so
        // it cannot compose above batch 1 until M3 broadcasting. Written anyway
        // so the schema is complete and a third shape is covered.
        VLNamedTensor::new("lm_head.bias", vec![0.0; VOCAB], vec![1, VOCAB]),
    ];

    let path = scratch("roundtrip");
    safetensors::write(&path, &tensors, &BTreeMap::new()).expect("write");
    let loaded = safetensors::read(&path).expect("read");

    assert_eq!(loaded.len(), tensors.len());
    for original in &tensors {
        let back = loaded
            .iter()
            .find(|t| t.name == original.name)
            .unwrap_or_else(|| panic!("{} missing after reload", original.name));
        assert_eq!(
            back.shape, original.shape,
            "{} changed shape",
            original.name
        );
        assert_eq!(
            back.data, original.data,
            "{} did not round-trip bit-for-bit",
            original.name
        );
    }
    let _ = std::fs::remove_file(&path);
}

/// A checkpoint that reads back cleanly but cannot be trained from is not a
/// checkpoint. This loads the weights into a fresh model and requires the
/// identical loss — the strongest statement available without a second run.
#[test]
fn a_reloaded_checkpoint_reproduces_the_loss_of_the_weights_it_holds() {
    let (ids, labels) = synthetic_batch();
    let (mut embed, mut head) = model();
    train(&mut embed, &mut head, &ids, &labels);

    // The loss of the weights as they stand *after* the last update, which is
    // what gets written — one step further along than the curve's last entry.
    let saved_loss = loss_of(&embed, &head, &ids, &labels);

    let tensors = vec![
        VLNamedTensor::new(
            "embed.weight",
            embed.weight().to_vec().expect("host"),
            embed.weight().shape().to_vec(),
        ),
        VLNamedTensor::new(
            "lm_head.weight",
            head.weight().to_vec().expect("host"),
            head.weight().shape().to_vec(),
        ),
    ];
    let path = scratch("resume");
    safetensors::write(&path, &tensors, &BTreeMap::new()).expect("write");
    let loaded = safetensors::read(&path).expect("read");

    // By name, not by index: `read` sorts by name, so positional access breaks
    // the day a tensor is added or renamed.
    let by_name = |n: &str| {
        let t = loaded
            .iter()
            .find(|t| t.name == n)
            .unwrap_or_else(|| panic!("{n} missing after reload"));
        Tensor::<GlProc>::from_vec(t.data.clone(), &t.shape).expect("tensor")
    };
    let restored_embed = ABEmbedding::new(TPParameter::trainable(
        "embed.weight",
        by_name("embed.weight"),
    ))
    .expect("embed");
    let restored_head = ABLinear::new(
        TPParameter::trainable("lm_head.weight", by_name("lm_head.weight")),
        None,
    )
    .expect("head");

    assert_eq!(
        loss_of(&restored_embed, &restored_head, &ids, &labels),
        saved_loss,
        "the reloaded weights must give the loss they were saved at"
    );
    let _ = std::fs::remove_file(&path);
}

/// The run's provenance travels with the file. A checkpoint whose step and loss
/// are only in someone's terminal scrollback cannot be placed later.
#[test]
fn the_metadata_survives_the_round_trip() {
    let tensors = vec![VLNamedTensor::new("w", vec![1.0, 2.0], vec![1, 2])];
    let mut metadata = BTreeMap::new();
    metadata.insert("step".to_string(), "10".to_string());
    metadata.insert("loss".to_string(), "3.104216".to_string());
    metadata.insert(
        "lm_head.bias".to_string(),
        "untrained zeros: cannot compose above batch 1 until M3".to_string(),
    );

    let path = scratch("metadata");
    safetensors::write(&path, &tensors, &metadata).expect("write");
    assert_eq!(
        safetensors::read_metadata(&path).expect("read_metadata"),
        metadata
    );
    let _ = std::fs::remove_file(&path);
}

/// Two tensors under one key make the file's meaning depend on which entry the
/// reader's map kept. The writer must refuse rather than pick.
#[test]
fn the_writer_refuses_a_duplicate_tensor_name() {
    let tensors = vec![
        VLNamedTensor::new("embed.weight", vec![1.0], vec![1]),
        VLNamedTensor::new("embed.weight", vec![2.0], vec![1]),
    ];
    let path = scratch("duplicate");
    assert!(safetensors::write(&path, &tensors, &BTreeMap::new()).is_err());
    let _ = std::fs::remove_file(&path);
}
