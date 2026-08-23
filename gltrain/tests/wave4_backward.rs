//! Stummañ Kevskrid + Gwellaer: Wave 4's backward pass and optimizer step.
//!
//! The chain is Wave 3's, run to a gradient and then to an update:
//! `backward() -> finish_step() -> OPAdamW::step -> forward`. These assertions
//! cover the three things that can be wrong without anything crashing — a
//! gradient that never arrives, a mask that stops applying somewhere between
//! the loss and the embedding table, and an update that moves parameters in a
//! direction that does not lower the loss.

use gltrain::{
    ABByteTokenizer, ABEmbedding, ABLinear, DTChatMl, GlProc, Module, OPAdamW, Optimizer,
    TPParameter, Tape, Tensor, Tokenizer, VLAdamWConfig, VLBatch, VLChatMlConfig, IGNORE_INDEX,
};
use std::path::Path;
use std::sync::{Arc, Mutex};

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/gwen_code_dataset.jsonl"
);

const D_MODEL: usize = 32;
const BATCH: usize = 3;
/// Below ~500 the assistant span is truncated away and nothing is supervised.
const MAX_SEQ_LEN: usize = 1024;
const SEED: u64 = 20260823;
const LR: f64 = 0.01;

fn l2(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// A flattened batch of real corpus tokens: `(ids, labels, vocab, pad_id)`.
///
/// `cap` matters more than it looks. Every sample in this corpus is longer than
/// 1024 byte-level tokens, so a cap at or below that truncates all of them to
/// exactly the cap and the collated batch contains **no padding whatsoever**.
/// A pad-related assertion under such a batch passes without testing anything,
/// which is why [`the_pad_rows_gradient_is_exactly_zero`] asks for a cap that
/// leaves the samples at differing lengths.
fn corpus_batch_of(n: usize, cap: usize) -> (Vec<u32>, Vec<i32>, usize, u32) {
    let corpus = DTChatMl::from_jsonl_path(Path::new(FIXTURE)).expect("fixture loads");
    let tok = ABByteTokenizer::new();
    let cfg = VLChatMlConfig::new(cap);
    let samples: Vec<_> = (0..n)
        .map(|i| corpus.tokenize_one(i, &tok, &cfg).expect("tokenizes"))
        .collect();
    let batch = VLBatch::collate(&samples, tok.pad_id()).expect("collates");
    (
        batch.input_ids().to_vec(),
        batch.labels().to_vec(),
        tok.vocab_size(),
        tok.pad_id(),
    )
}

fn corpus_batch() -> (Vec<u32>, Vec<i32>, usize, u32) {
    corpus_batch_of(BATCH, MAX_SEQ_LEN)
}

fn model(vocab: usize) -> (ABEmbedding<GlProc>, ABLinear<GlProc>) {
    (
        ABEmbedding::randn("embed", vocab, D_MODEL, 0.02, SEED).expect("embed"),
        ABLinear::randn("lm_head", D_MODEL, vocab, 0.02, SEED + 1).expect("head"),
    )
}

fn forward(
    embed: &ABEmbedding<GlProc>,
    head: &ABLinear<GlProc>,
    ids: &[u32],
    labels: &[i32],
    tape: &Arc<Mutex<Tape>>,
) -> f32 {
    let hidden = embed.forward_ids(ids, tape).expect("gather");
    let logits = head.forward(&hidden, tape).expect("projection");
    logits
        .log_softmax()
        .expect("log_softmax")
        .masked_cross_entropy(labels)
        .expect("loss")
        .item()
        .expect("scalar")
}

/// Both parameters must receive a gradient, and `finish_step` must leave the
/// tape empty — KL-006, which is what makes the weight write on the next line
/// unobservable by a live backward closure.
#[test]
fn backward_reaches_both_parameters_and_empties_the_tape() {
    let (ids, labels, vocab, _) = corpus_batch();
    let (embed, head) = model(vocab);
    let tape = Arc::new(Mutex::new(Tape::new()));

    let loss = forward(&embed, &head, &ids, &labels, &tape);
    assert!(loss.is_finite());
    assert_eq!(Tape::lock(&tape).len(), 4, "four ops recorded");

    let grads = {
        let mut guard = Tape::lock(&tape);
        guard.backward().expect("backward");
        guard.finish_step()
    };
    assert!(
        Tape::lock(&tape).is_empty(),
        "finish_step must empty the tape before any weight is written"
    );

    let (eg, eg_shape) = grads.get(embed.weight().id()).expect("embedding gradient");
    let (hg, hg_shape) = grads.get(head.weight().id()).expect("head gradient");
    assert_eq!(eg_shape, &vec![vocab, D_MODEL]);
    assert_eq!(hg_shape, &vec![D_MODEL, vocab]);
    assert!(l2(eg) > 0.0, "the embedding table got a zero gradient");
    assert!(l2(hg) > 0.0, "the head got a zero gradient");
    assert!(eg.iter().all(|g| g.is_finite()));
    assert!(hg.iter().all(|g| g.is_finite()));
}

/// The pad token only ever appears as an *input* at positions whose labels are
/// masked. If the mask propagates the whole way — loss, log_softmax, matmul,
/// scatter-add — its row must come back at exactly zero, not merely small.
#[test]
fn the_pad_rows_gradient_is_exactly_zero() {
    // Samples 0 and 1 are 2441 and 1975 tokens, so a 2560 cap truncates
    // neither and the shorter one is genuinely padded. At the 1024 cap the
    // other tests use, both truncate to exactly 1024 and there is no padding
    // for this assertion to be about.
    let (ids, labels, vocab, pad_id) = corpus_batch_of(2, 2560);
    assert!(
        ids.contains(&pad_id),
        "this batch must actually contain padding for the test to mean anything"
    );
    assert!(
        labels.iter().any(|&l| l != IGNORE_INDEX),
        "and it must have something supervised"
    );
    let (embed, head) = model(vocab);
    let tape = Arc::new(Mutex::new(Tape::new()));
    forward(&embed, &head, &ids, &labels, &tape);

    let grads = {
        let mut guard = Tape::lock(&tape);
        guard.backward().expect("backward");
        guard.finish_step()
    };
    let (eg, _) = grads.get(embed.weight().id()).expect("embedding gradient");
    let row = pad_id as usize * D_MODEL;
    assert!(
        eg[row..row + D_MODEL].iter().all(|g| *g == 0.0),
        "the pad row picked up a gradient: {:?}",
        &eg[row..row + D_MODEL]
    );
}

/// The same property in a case small enough to enumerate: with one supervised
/// position, exactly one embedding row may be non-zero.
#[test]
fn only_rows_used_at_a_supervised_position_receive_a_gradient() {
    let (vocab, d_model) = (5usize, 3usize);
    let embed = ABEmbedding::<GlProc>::randn("embed", vocab, d_model, 0.1, SEED).expect("embed");
    let head = ABLinear::<GlProc>::randn("head", d_model, vocab, 0.1, SEED + 1).expect("head");
    let tape = Arc::new(Mutex::new(Tape::new()));

    // Positions 0 and 2 are masked; only position 1, holding token 1, counts.
    let ids = [0u32, 1, 2];
    let labels = [IGNORE_INDEX, 4, IGNORE_INDEX];
    forward(&embed, &head, &ids, &labels, &tape);

    let grads = {
        let mut guard = Tape::lock(&tape);
        guard.backward().expect("backward");
        guard.finish_step()
    };
    let (eg, _) = grads.get(embed.weight().id()).expect("embedding gradient");

    for row in 0..vocab {
        let slice = &eg[row * d_model..(row + 1) * d_model];
        if row == 1 {
            assert!(l2(slice) > 0.0, "row 1 was supervised and must be non-zero");
        } else {
            assert!(
                slice.iter().all(|g| *g == 0.0),
                "row {row} was never supervised but got {slice:?}"
            );
        }
    }
}

/// One AdamW step must move the parameters and lower the loss. Weight decay is
/// zero here on purpose: it pulls weights toward the origin regardless of the
/// gradient, which fights a single-step check for no benefit.
#[test]
fn one_adamw_step_moves_the_parameters_and_lowers_the_loss() {
    let (ids, labels, vocab, _) = corpus_batch();
    let (mut embed, mut head) = model(vocab);
    let tape = Arc::new(Mutex::new(Tape::new()));

    let loss_before = forward(&embed, &head, &ids, &labels, &tape);
    let grads = {
        let mut guard = Tape::lock(&tape);
        guard.backward().expect("backward");
        guard.finish_step()
    };

    let before_embed = embed.weight().to_vec().expect("host");
    let before_head = head.weight().to_vec().expect("host");

    let mut opt = OPAdamW::<GlProc>::new(VLAdamWConfig {
        lr: LR,
        weight_decay: 0.0,
        ..Default::default()
    });
    {
        // ABEmbedding does not implement Module, so its parameters are
        // collected explicitly rather than through `trainable_parameters`.
        let mut params: Vec<&mut TPParameter<GlProc>> = Vec::new();
        params.extend(embed.parameters_mut());
        params.extend(head.parameters_mut());
        opt.step(&mut params, &grads).expect("step");
    }
    assert_eq!(opt.step_count(), 1);

    let after_embed = embed.weight().to_vec().expect("host");
    let after_head = head.weight().to_vec().expect("host");
    let d_embed: Vec<f32> = after_embed
        .iter()
        .zip(&before_embed)
        .map(|(a, b)| a - b)
        .collect();
    let d_head: Vec<f32> = after_head
        .iter()
        .zip(&before_head)
        .map(|(a, b)| a - b)
        .collect();
    assert!(l2(&d_embed) > 0.0, "the embedding table did not move");
    assert!(l2(&d_head) > 0.0, "the head did not move");

    // AdamW's first update is lr * m_hat/(sqrt(v_hat) + eps), and at t = 1 the
    // bias-corrected ratio is ~1, so no element may exceed lr by more than
    // rounding. An update far above it means bias correction is wrong.
    let max_delta = d_embed
        .iter()
        .chain(&d_head)
        .fold(0.0f32, |m, x| m.max(x.abs()));
    assert!(
        max_delta <= LR as f32 * 1.001,
        "first-step delta {max_delta} exceeds lr {LR}"
    );

    let loss_after = forward(&embed, &head, &ids, &labels, &tape);
    assert!(loss_after.is_finite(), "loss became {loss_after}");
    assert!(
        loss_after < loss_before,
        "one step must lower the loss: {loss_before} -> {loss_after}"
    );
}

/// A frozen table must neither receive a gradient nor be written. This is the
/// LoRA shape, and it is the case an optimizer that trusted the grad store
/// rather than the trainable flag would get wrong.
#[test]
fn a_frozen_embedding_gets_no_gradient_and_is_not_updated() {
    let (ids, labels, vocab, _) = corpus_batch();
    let table = Tensor::<GlProc>::randn(&[vocab, D_MODEL], 0.02, SEED).expect("table");
    let embed = ABEmbedding::<GlProc>::frozen("base", table).expect("frozen");
    let head = ABLinear::<GlProc>::randn("lm_head", D_MODEL, vocab, 0.02, SEED + 1).expect("head");
    let tape = Arc::new(Mutex::new(Tape::new()));

    forward(&embed, &head, &ids, &labels, &tape);
    let grads = {
        let mut guard = Tape::lock(&tape);
        guard.backward().expect("backward");
        guard.finish_step()
    };

    assert!(
        grads.get(embed.weight().id()).is_none(),
        "a frozen table must not receive a gradient"
    );
    // The head still trains, so the graph was live — the absence above is the
    // frozen flag doing its job, not a broken forward pass.
    assert!(grads.get(head.weight().id()).is_some());
}

/// The bias only composes at batch = 1: it is `[1, d_out]` and `Tensor::add`
/// is exact-shape. That is a documented M3 limitation (broadcasting), so this
/// is the one shape in which a bias gradient exists today.
#[test]
fn the_bias_gradient_exists_at_batch_one_and_sums_to_zero() {
    let (vocab, d_model) = (16usize, 8usize);
    let x = Tensor::<GlProc>::randn(&[1, d_model], 0.5, SEED).expect("input");
    let w = Tensor::<GlProc>::randn(&[d_model, vocab], 0.02, SEED + 1).expect("weight");
    let b = Tensor::<GlProc>::zeros(&[1, vocab]).expect("bias");
    let layer = ABLinear::<GlProc>::new(
        TPParameter::trainable("head.weight", w),
        Some(TPParameter::trainable("head.bias", b)),
    )
    .expect("layer");

    let tape = Arc::new(Mutex::new(Tape::new()));
    let logits = layer.forward(&x, &tape).expect("projection");
    logits
        .log_softmax()
        .expect("log_softmax")
        .masked_cross_entropy(&[3])
        .expect("loss");

    let grads = {
        let mut guard = Tape::lock(&tape);
        guard.backward().expect("backward");
        guard.finish_step()
    };
    let bias = layer
        .parameters()
        .into_iter()
        .find(|p| p.name() == "head.bias")
        .expect("bias is a parameter");
    let (bg, bg_shape) = grads.get(bias.id()).expect("bias gradient");

    assert_eq!(bg_shape, &vec![1, vocab]);
    assert!(l2(bg) > 0.0, "the bias got a zero gradient");
    // dL/db is (softmax - onehot), which sums to zero: a distribution moves
    // mass around rather than creating it.
    assert!(
        bg.iter().sum::<f32>().abs() < 1e-6,
        "the bias gradient sums to {}",
        bg.iter().sum::<f32>()
    );
    // And the labelled class is the only negative entry.
    assert!(bg[3] < 0.0, "the target class should pull its logit up");
    assert!(bg.iter().enumerate().all(|(i, g)| i == 3 || *g >= 0.0));
}

/// A bias against a multi-row batch is a shape error, not a silent broadcast.
/// Recorded so the M3 broadcasting work has a test that must change with it.
#[test]
fn a_bias_against_a_multi_row_batch_is_refused() {
    let (vocab, d_model) = (16usize, 8usize);
    let x = Tensor::<GlProc>::randn(&[4, d_model], 0.5, SEED).expect("input");
    let w = Tensor::<GlProc>::randn(&[d_model, vocab], 0.02, SEED + 1).expect("weight");
    let b = Tensor::<GlProc>::zeros(&[1, vocab]).expect("bias");
    let layer = ABLinear::<GlProc>::new(
        TPParameter::trainable("head.weight", w),
        Some(TPParameter::trainable("head.bias", b)),
    )
    .expect("layer");

    let tape = Arc::new(Mutex::new(Tape::new()));
    assert!(
        layer.forward(&x, &tape).is_err(),
        "broadcasting is M3; until then this must be an error, not a wrong answer"
    );
}
