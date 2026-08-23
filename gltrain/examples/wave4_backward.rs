//! Stummañ Kevskrid + Gwellaer: Wave 4's backward pass and optimizer step.
//!
//! ```text
//! forward -> loss -> backward() -> finish_step() -> OPAdamW::step -> forward
//! ```
//!
//! Wave 3 proved the forward chain lands on `ln(vocab)`. This proves the other
//! half: that the gradient reaches every parameter, that the update moves them,
//! and that the loss goes down as a result.
//!
//! # KL-006, in the order it has to happen
//!
//! `finish_step()` returns the gradients **and** empties the tape in one call,
//! so the in-place weight write on the next line cannot be observed by a live
//! backward closure. There is no ordering here for a caller to get wrong,
//! because there is no way to hold a `VLGradStore` and a populated tape at the
//! same time.
//!
//! # The bias is measured on a separate one-token forward
//!
//! `ABLinear`'s bias is `[1, d_out]` and `Tensor::add` requires exactly equal
//! shapes, so a bias only composes when the batch is one row. That is a
//! documented M3 limitation (broadcasting), not something this wave introduces
//! or works around. The 4096-token run below therefore has no bias, and the
//! bias gradient is reported from a `[1, d_model]` forward at the end — the
//! only shape in which it exists today.
//!
//! Run: cargo run --release --example wave4_backward

use gltrain::{
    ABByteTokenizer, ABEmbedding, ABLinear, DTChatMl, GlProc, Module, OPAdamW, Optimizer,
    TPParameter, Tape, Tensor, Tokenizer, VLAdamWConfig, VLBatch, VLChatMlConfig,
};
use std::path::Path;
use std::sync::{Arc, Mutex};

const D_MODEL: usize = 64;
const BATCH: usize = 4;
/// Above the length of samples 0-2 (2441, 1975, 3908) so they are not all
/// truncated to the same size. That matters for the pad-row check below: at a
/// cap every sample exceeds, the batch is a perfect rectangle with no padding
/// in it, and "the pad row has no gradient" would hold trivially because the
/// pad token never appears.
const MAX_SEQ_LEN: usize = 4096;
const INIT_STD: f32 = 0.02;
const SEED: u64 = 20260823;

/// Large enough that one step moves the loss visibly. AdamW's first update is
/// close to `lr` per parameter, so this is also the expected per-element delta.
const LR: f64 = 0.01;

/// Euclidean norm — the number `clip_grad_norm_` means by "gradient norm".
fn l2(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// `(l2, max_abs, non_zero_count)` of a buffer.
fn stats(v: &[f32]) -> (f32, f32, usize) {
    (
        l2(v),
        v.iter().fold(0.0f32, |m, x| m.max(x.abs())),
        v.iter().filter(|x| **x != 0.0).count(),
    )
}

/// One forward pass over the whole chain, returning the loss.
fn forward(
    embed: &ABEmbedding<GlProc>,
    head: &ABLinear<GlProc>,
    ids: &[u32],
    labels: &[i32],
    tape: &Arc<Mutex<Tape>>,
) -> anyhow::Result<Tensor<GlProc>> {
    let hidden = embed.forward_ids(ids, tape)?;
    let logits = head.forward(&hidden, tape)?;
    Ok(logits.log_softmax()?.masked_cross_entropy(labels)?)
}

fn main() -> anyhow::Result<()> {
    let path = Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/gwen_code_dataset.jsonl"
    ));

    // ── Data ─────────────────────────────────────────────────────────────
    let corpus = DTChatMl::from_jsonl_path(path)?;
    let tok = ABByteTokenizer::new();
    let vocab = tok.vocab_size();
    let cfg = VLChatMlConfig::new(MAX_SEQ_LEN);

    let samples: Vec<_> = (0..BATCH)
        .map(|i| corpus.tokenize_one(i, &tok, &cfg))
        .collect::<Result<_, _>>()?;
    let batch = VLBatch::collate(&samples, tok.pad_id())?;
    let ids = batch.input_ids().to_vec();
    let labels = batch.labels().to_vec();

    let mut embed = ABEmbedding::<GlProc>::randn("embed", vocab, D_MODEL, INIT_STD, SEED)?;
    let mut head = ABLinear::<GlProc>::randn("lm_head", D_MODEL, vocab, INIT_STD, SEED + 1)?;

    println!("── setup ─────────────────────────────────────────────");
    println!("  batch        [{}, {}]", batch.batch(), batch.seq_len());
    println!("  tokens       {}", ids.len());
    println!("  supervised   {}", batch.supervised_len());
    println!("  vocab        {vocab}");
    println!("  lr           {LR}   weight_decay 0 (it fights a one-step check)");

    // ── Forward ──────────────────────────────────────────────────────────
    let tape = Arc::new(Mutex::new(Tape::new()));
    let loss_before = forward(&embed, &head, &ids, &labels, &tape)?.item()?;
    println!("\n── forward #1 ───────────────────────────────────────");
    println!("  loss         {loss_before:.6}");
    println!("  tape nodes   {}", Tape::lock(&tape).len());

    // ── Backward (KL-006) ────────────────────────────────────────────────
    //
    // The gradients and the emptied tape arrive together, so the weight write
    // below cannot be seen by a live closure.
    let grads = {
        let mut guard = Tape::lock(&tape);
        guard.backward()?;
        guard.finish_step()
    };
    println!("\n── backward ─────────────────────────────────────────");
    println!(
        "  tape nodes   {} after finish_step (KL-006: emptied before any write)",
        Tape::lock(&tape).len()
    );
    println!(
        "  grad entries {} (parameters plus every intermediate)",
        grads.len()
    );

    let embed_id = embed.weight().id();
    let head_id = head.weight().id();
    let (eg, eg_shape) = grads
        .get(embed_id)
        .ok_or_else(|| anyhow::anyhow!("no gradient reached the embedding table"))?;
    let (hg, hg_shape) = grads
        .get(head_id)
        .ok_or_else(|| anyhow::anyhow!("no gradient reached the head"))?;

    let (e_l2, e_max, e_nz) = stats(eg);
    let (h_l2, h_max, h_nz) = stats(hg);

    println!("\n  parameter          shape         ‖grad‖₂     max|g|      non-zero");
    println!(
        "  embed.weight       {:<12}  {:<10.6}  {:<10.6}  {} / {}",
        format!("{eg_shape:?}"),
        e_l2,
        e_max,
        e_nz,
        eg.len()
    );
    println!(
        "  lm_head.weight     {:<12}  {:<10.6}  {:<10.6}  {} / {}",
        format!("{hg_shape:?}"),
        h_l2,
        h_max,
        h_nz,
        hg.len()
    );

    // The pad row is only ever an *input* at positions whose labels are masked,
    // so the mask must carry all the way back and leave it at exactly zero.
    let pad_row = tok.pad_id() as usize * D_MODEL;
    let pad_grad = l2(&eg[pad_row..pad_row + D_MODEL]);
    let pad_count = ids.iter().filter(|&&t| t == tok.pad_id()).count();
    println!(
        "\n  pad row {} gradient  {pad_grad:.9}  over {pad_count} padded positions",
        tok.pad_id()
    );
    assert!(
        pad_count > 0,
        "no padding in this batch, so a pad-row check would prove nothing"
    );

    // ── Optimizer step (Gwellaer) ────────────────────────────────────────
    //
    // ABEmbedding does not implement Module (its input is ids, not a tensor),
    // so its parameters are collected explicitly rather than through
    // `trainable_parameters`. This is the call site that note was about.
    let before_embed = embed.weight().to_vec()?;
    let before_head = head.weight().to_vec()?;

    let mut opt = OPAdamW::<GlProc>::new(VLAdamWConfig {
        lr: LR,
        weight_decay: 0.0,
        ..Default::default()
    });
    {
        let mut params: Vec<&mut TPParameter<GlProc>> = Vec::new();
        params.extend(embed.parameters_mut());
        params.extend(head.parameters_mut());
        opt.step(&mut params, &grads)?;
    }

    let after_embed = embed.weight().to_vec()?;
    let after_head = head.weight().to_vec()?;
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
    let (de_l2, de_max, de_nz) = stats(&d_embed);
    let (dh_l2, dh_max, dh_nz) = stats(&d_head);

    println!("\n── AdamW step ───────────────────────────────────────");
    println!("  step count   {}", opt.step_count());
    println!("\n  parameter          ‖Δθ‖₂       max|Δθ|     moved");
    println!(
        "  embed.weight       {de_l2:<11.6} {de_max:<11.6} {de_nz} / {}",
        d_embed.len()
    );
    println!(
        "  lm_head.weight     {dh_l2:<11.6} {dh_max:<11.6} {dh_nz} / {}",
        d_head.len()
    );

    // ── Forward #2 ───────────────────────────────────────────────────────
    let loss_after = forward(&embed, &head, &ids, &labels, &tape)?.item()?;

    println!("\n── forward #2 ───────────────────────────────────────");
    println!("  loss before  {loss_before:.6}");
    println!("  loss after   {loss_after:.6}");
    println!("  change       {:+.6}", loss_after - loss_before);
    println!(
        "  verdict      {}",
        if loss_after < loss_before {
            "decreased ✓"
        } else {
            "DID NOT DECREASE ✗"
        }
    );

    assert!(
        loss_before.is_finite() && loss_after.is_finite(),
        "NaN loss"
    );
    assert!(
        loss_after < loss_before,
        "one AdamW step must lower the loss: {loss_before} -> {loss_after}"
    );
    assert!(e_l2 > 0.0 && h_l2 > 0.0, "a parameter received no gradient");
    assert_eq!(
        pad_grad, 0.0,
        "the pad row must receive exactly no gradient"
    );

    // ── The bias, on the only shape it composes in ───────────────────────
    //
    // See the module docs: `ABLinear`'s bias is [1, d_out] and `add` is
    // exact-shape, so this is a one-token forward. Broadcasting is M3.
    println!("\n── bias gradient (separate 1-token forward) ─────────");
    let bias_tape = Arc::new(Mutex::new(Tape::new()));
    let w = Tensor::<GlProc>::randn(&[D_MODEL, vocab], INIT_STD, SEED + 2)?;
    let b = Tensor::<GlProc>::zeros(&[1, vocab])?;
    let biased = ABLinear::<GlProc>::new(
        TPParameter::trainable("lm_head.weight", w),
        Some(TPParameter::trainable("lm_head.bias", b)),
    )?;
    let one = embed.forward_ids(&ids[..1], &bias_tape)?;
    let logits = biased.forward(&one, &bias_tape)?;
    let bias_loss = logits
        .log_softmax()?
        .masked_cross_entropy(&labels[..1].iter().map(|_| 0i32).collect::<Vec<_>>())?;

    let bias_loss_value = bias_loss.item()?;
    let bias_grads = {
        let mut guard = Tape::lock(&bias_tape);
        guard.backward()?;
        guard.finish_step()
    };
    let bias_param = biased
        .parameters()
        .into_iter()
        .find(|p| p.name() == "lm_head.bias")
        .expect("the bias is a parameter of this layer");
    let (bg, bg_shape) = bias_grads
        .get(bias_param.id())
        .ok_or_else(|| anyhow::anyhow!("no gradient reached the bias"))?;
    let (b_l2, b_max, b_nz) = stats(bg);

    println!("  input        [1, {D_MODEL}]  (batch = 1, the only shape a bias fits)");
    println!("  loss         {bias_loss_value:.6}");
    println!(
        "  lm_head.bias       {:<12}  ‖grad‖₂ {:<10.6}  max|g| {:<10.6}  non-zero {} / {}",
        format!("{bg_shape:?}"),
        b_l2,
        b_max,
        b_nz,
        bg.len()
    );
    // A cross-entropy gradient sums to zero across the row: probabilities move
    // mass around rather than creating it.
    println!(
        "  Σg           {:+.9}  <- a CE gradient must sum to zero",
        bg.iter().sum::<f32>()
    );
    assert!(b_l2 > 0.0, "the bias received no gradient");

    println!("\n  gradients non-zero, parameters moved, loss decreased ✓");
    Ok(())
}
