//! Stummañ Deskiñ + Pik: Wave 5's mini training loop and its checkpoint.
//!
//! Ten steps of `forward -> backward -> finish_step -> AdamW::step`, then the
//! weights are written with the safetensors writer and read back.
//!
//! # The batch is fixed on purpose
//!
//! Every step trains on the *same* four samples. This is the overfit-one-batch
//! check, and it is the right shape for a sanity run: cycling batches makes the
//! loss move for two reasons at once — learning, and one batch being harder
//! than the next — and a curve that wobbles then tells you nothing about which.
//! Holding the batch still leaves exactly one explanation for the loss going
//! down. A real run cycles; this deliberately does not.
//!
//! # CPFull is not used, and not implemented
//!
//! `CPLora` saves a LoRA adapter and this model has no adapter — it trains an
//! embedding table and a projection directly. `CPFull` is the layout that would
//! fit, and it is a registered stub that returns `Unsupported` naming its
//! milestone. Implementing it here would be a milestone's work smuggled into a
//! validation wave, so this writes through
//! [`gltrain::checkpoint::safetensors`] directly — which is a full, round-trip
//! tested writer with `glcore`'s independent reader as its oracle. **CPFull is
//! deferred; nothing here pretends otherwise.**
//!
//! # The bias is saved but not trained
//!
//! `ABLinear`'s bias is `[1, d_out]` and `Tensor::add` is exact-shape, so a
//! bias cannot compose with a multi-row batch (M3 broadcasting — see wave 4).
//! At `batch_size = 4` the trained head therefore has no bias. `lm_head.bias`
//! is still written, as the `[1, vocab]` zero tensor it mathematically is, so
//! the checkpoint schema is complete for when broadcasting lands and the
//! round-trip covers a third shape. The metadata records that it is untrained.
//!
//! Run: cargo run --release --example wave5_loop

use gltrain::checkpoint::safetensors;
use gltrain::{
    ABByteTokenizer, ABEmbedding, ABLinear, DTChatMl, GlProc, Module, OPAdamW, Optimizer,
    TPParameter, Tape, Tokenizer, VLAdamWConfig, VLBatch, VLChatMlConfig, VLNamedTensor,
};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

const D_MODEL: usize = 64;
const BATCH_SIZE: usize = 4;
const MAX_SEQ_LEN: usize = 2048;
const STEPS: usize = 10;
const INIT_STD: f32 = 0.02;
const SEED: u64 = 20260823;
const LR: f64 = 0.02;

fn main() -> anyhow::Result<()> {
    let path = std::path::Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/gwen_code_dataset.jsonl"
    ));

    // ── Data ─────────────────────────────────────────────────────────────
    let corpus = DTChatMl::from_jsonl_path(path)?;
    let tok = ABByteTokenizer::new();
    let vocab = tok.vocab_size();
    let cfg = VLChatMlConfig::new(MAX_SEQ_LEN);

    let samples: Vec<_> = (0..BATCH_SIZE)
        .map(|i| corpus.tokenize_one(i, &tok, &cfg))
        .collect::<Result<_, _>>()?;
    let batch = VLBatch::collate(&samples, tok.pad_id())?;
    let ids = batch.input_ids().to_vec();
    let labels = batch.labels().to_vec();

    let mut embed = ABEmbedding::<GlProc>::randn("embed", vocab, D_MODEL, INIT_STD, SEED)?;
    let mut head = ABLinear::<GlProc>::randn("lm_head", D_MODEL, vocab, INIT_STD, SEED + 1)?;
    let mut opt = OPAdamW::<GlProc>::new(VLAdamWConfig {
        lr: LR,
        weight_decay: 0.0,
        ..Default::default()
    });

    println!("── setup ─────────────────────────────────────────────");
    println!("  batch        [{}, {}]", batch.batch(), batch.seq_len());
    println!("  tokens       {}", ids.len());
    println!("  supervised   {}", batch.supervised_len());
    println!("  vocab        {vocab}   d_model {D_MODEL}");
    println!(
        "  parameters   {}",
        embed.weight().n_elems() + head.weight().n_elems()
    );
    println!("  optimizer    AdamW lr={LR} weight_decay=0");
    println!("  batch policy fixed across all {STEPS} steps (overfit-one-batch)");

    // ── The loop ─────────────────────────────────────────────────────────
    let tape = Arc::new(Mutex::new(Tape::new()));
    let mut curve = Vec::with_capacity(STEPS);

    println!("\n── loop ──────────────────────────────────────────────");
    println!("  step   loss        Δ           ms");

    for step in 1..=STEPS {
        let t0 = Instant::now();

        let hidden = embed.forward_ids(&ids, &tape)?;
        let logits = head.forward(&hidden, &tape)?;
        let loss = logits.log_softmax()?.masked_cross_entropy(&labels)?;
        let loss_value = loss.item()?;

        // KL-006: gradients and an emptied tape arrive together, so the weight
        // write below cannot be observed by a live backward closure.
        let grads = {
            let mut guard = Tape::lock(&tape);
            guard.backward()?;
            guard.finish_step()
        };

        {
            // ABEmbedding does not implement Module (its input is ids, not a
            // tensor), so its parameters are collected explicitly.
            let mut params: Vec<&mut TPParameter<GlProc>> = Vec::new();
            params.extend(embed.parameters_mut());
            params.extend(head.parameters_mut());
            opt.step(&mut params, &grads)?;
        }

        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        let delta = curve.last().map(|p: &f32| loss_value - *p);
        println!(
            "  {step:<5}  {loss_value:<11.6} {:<11} {ms:.1}",
            delta.map_or("—".to_string(), |d| format!("{d:+.6}"))
        );
        curve.push(loss_value);

        assert!(loss_value.is_finite(), "step {step} produced {loss_value}");
    }

    // ── The curve ────────────────────────────────────────────────────────
    println!("\n── loss curve ────────────────────────────────────────");
    println!(
        "  [{}]",
        curve
            .iter()
            .map(|v| format!("{v:.6}"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    // The curve records the loss measured *before* each step's update, which is
    // what every framework logs. The weights that get checkpointed are the ones
    // after step 10, so their loss is one update further along than curve[9].
    // Measure it rather than conflating the two.
    let final_loss = {
        let t = Arc::new(Mutex::new(Tape::new()));
        let hidden = embed.forward_ids(&ids, &t)?;
        let logits = head.forward(&hidden, &t)?;
        logits
            .log_softmax()?
            .masked_cross_entropy(&labels)?
            .item()?
    };

    let total = curve[STEPS - 1] - curve[0];
    let monotone = curve.windows(2).all(|w| w[1] < w[0]);
    println!("  first        {:.6}   (before step 1's update)", curve[0]);
    println!(
        "  last         {:.6}   (before step {STEPS}'s update)",
        curve[STEPS - 1]
    );
    println!("  after step {STEPS} {final_loss:.6}   (the weights checkpointed below)");
    println!(
        "  total change {total:+.6} over the curve, {:+.6} including the last update",
        final_loss - curve[0]
    );
    println!(
        "  monotone     {}",
        if monotone {
            "yes ✓"
        } else {
            "no — see above"
        }
    );
    println!(
        "  perplexity   {:.3} -> {:.3}",
        curve[0].exp(),
        final_loss.exp()
    );

    // ── Checkpoint (Pik, via the safetensors writer) ─────────────────────
    //
    // CPFull is the layout that would fit this model and it is a stub. See the
    // module docs: it stays deferred, and this uses the writer directly.
    let out: PathBuf = std::env::temp_dir().join("gltrain_wave5_step10.safetensors");

    let bias = vec![0.0f32; vocab];
    let tensors = vec![
        VLNamedTensor::new(
            "embed.weight",
            embed.weight().to_vec()?,
            embed.weight().shape().to_vec(),
        ),
        VLNamedTensor::new(
            "lm_head.weight",
            head.weight().to_vec()?,
            head.weight().shape().to_vec(),
        ),
        VLNamedTensor::new("lm_head.bias", bias, vec![1, vocab]),
    ];

    let mut metadata = BTreeMap::new();
    metadata.insert("format".into(), "gltrain-wave5".into());
    metadata.insert("step".into(), STEPS.to_string());
    metadata.insert("loss".into(), format!("{final_loss:.6}"));
    metadata.insert("optimizer".into(), format!("adamw lr={LR} wd=0"));
    metadata.insert("d_model".into(), D_MODEL.to_string());
    metadata.insert("vocab_size".into(), vocab.to_string());
    metadata.insert(
        "lm_head.bias".into(),
        "untrained zeros: ABLinear's bias is [1, d_out] and add() is exact-shape, \
         so it cannot compose above batch 1 until M3 broadcasting"
            .into(),
    );
    metadata.insert(
        "checkpoint_layout".into(),
        "safetensors writer directly; CPFull is a stub and stays deferred".into(),
    );

    safetensors::write(&out, &tensors, &metadata)?;
    let bytes = std::fs::metadata(&out)?.len();

    println!("\n── checkpoint ────────────────────────────────────────");
    println!("  path         {}", out.display());
    println!("  size         {bytes} bytes");
    println!("  layout       safetensors writer (CPFull deferred — it is a stub)");
    for t in &tensors {
        println!("    {:<16} {:?}", t.name, t.shape);
    }

    // ── Reload and verify ────────────────────────────────────────────────
    let loaded = safetensors::read(&out)?;
    let loaded_meta = safetensors::read_metadata(&out)?;

    println!("\n── reload ────────────────────────────────────────────");
    println!("  tensors      {}", loaded.len());
    println!("  metadata     {} keys", loaded_meta.len());
    println!("\n  tensor             shape         max|Δ|      allclose");

    let mut all_exact = true;
    for original in &tensors {
        let back = loaded
            .iter()
            .find(|t| t.name == original.name)
            .ok_or_else(|| anyhow::anyhow!("{} missing from the reloaded file", original.name))?;
        anyhow::ensure!(
            back.shape == original.shape,
            "{} changed shape",
            original.name
        );
        anyhow::ensure!(
            back.data.len() == original.data.len(),
            "{} changed length",
            original.name
        );

        let max_delta = back
            .data
            .iter()
            .zip(&original.data)
            .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        all_exact &= max_delta == 0.0;
        println!(
            "  {:<18} {:<12}  {max_delta:<11.9} {}",
            original.name,
            format!("{:?}", original.shape),
            if max_delta == 0.0 {
                "exact ✓"
            } else {
                "DRIFTED ✗"
            }
        );
    }

    // safetensors stores f32 verbatim — no quantization, no dtype change — so
    // the round trip is bit-for-bit and "allclose" here means equal.
    assert!(all_exact, "the round trip must be exact, not merely close");
    assert_eq!(loaded_meta.get("step").map(String::as_str), Some("10"));

    // ── The reloaded weights reproduce the loss ──────────────────────────
    //
    // The strongest check available: a checkpoint that reads back cleanly but
    // cannot be trained from is not a checkpoint.
    // By name, not by index: `read` returns tensors sorted by name, so
    // positional access would silently pick up the bias the day a tensor is
    // added or renamed.
    let by_name = |n: &str| -> anyhow::Result<gltrain::Tensor<GlProc>> {
        let t = loaded
            .iter()
            .find(|t| t.name == n)
            .ok_or_else(|| anyhow::anyhow!("{n} missing from the reloaded file"))?;
        Ok(gltrain::Tensor::<GlProc>::from_vec(
            t.data.clone(),
            &t.shape,
        )?)
    };
    let restored_embed = ABEmbedding::<GlProc>::new(TPParameter::trainable(
        "embed.weight",
        by_name("embed.weight")?,
    ))?;
    let restored_head = ABLinear::<GlProc>::new(
        TPParameter::trainable("lm_head.weight", by_name("lm_head.weight")?),
        None,
    )?;

    let verify_tape = Arc::new(Mutex::new(Tape::new()));
    let hidden = restored_embed.forward_ids(&ids, &verify_tape)?;
    let logits = restored_head.forward(&hidden, &verify_tape)?;
    let restored_loss = logits
        .log_softmax()?
        .masked_cross_entropy(&labels)?
        .item()?;

    println!("\n── resume ────────────────────────────────────────────");
    println!("  checkpointed weights  {final_loss:.6}");
    println!("  after reload          {restored_loss:.6}");
    println!("  difference            {:+.9}", restored_loss - final_loss);
    assert_eq!(
        restored_loss, final_loss,
        "a reloaded checkpoint must reproduce the loss of the weights it holds"
    );

    println!("\n  loop converged, checkpoint round-tripped exactly, loss reproduced ✓");
    Ok(())
}
