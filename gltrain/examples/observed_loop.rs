//! Stummañ Deskiñ: Wave 5's loop with [`VLTrainingObserver`] attached.
//!
//! The same ten steps, the same corpus, the same checkpoint — with all eight
//! axes recorded on every step, a live line as it runs, and a JSON and
//! Markdown report at the end.
//!
//! # What this demonstrates that a mock cannot
//!
//! That the hook API composes with a real loop without the loop having to be
//! rearranged for it. The training code below is Wave 5's, with the observer
//! calls interleaved and nothing else moved. In particular the parameter
//! snapshots are taken where they are cheap — around the optimizer call, which
//! is the one place values from before and after the update both exist.
//!
//! Run: cargo run --release --example observed_loop

use gltrain::checkpoint::safetensors;
use gltrain::train::observability::{
    VLBatchInfo, VLObserverConfig, VLParamSnapshot, VLRunConfig, VLTrainingObserver,
};
use gltrain::{
    ABByteTokenizer, ABEmbedding, ABLinear, DTChatMl, GlProc, Module, OPAdamW, Optimizer,
    TPParameter, Tape, Tokenizer, VLAdamWConfig, VLBatch, VLChatMlConfig, VLNamedTensor,
};
use std::collections::BTreeMap;
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
    let fixture = std::path::Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/gwen_code_dataset.jsonl"
    ));

    // ── Data ─────────────────────────────────────────────────────────────
    let t_data = Instant::now();
    let corpus = DTChatMl::from_jsonl_path(fixture)?;
    let tok = ABByteTokenizer::new();
    let vocab = tok.vocab_size();
    let cfg = VLChatMlConfig::new(MAX_SEQ_LEN);

    let samples: Vec<_> = (0..BATCH_SIZE)
        .map(|i| corpus.tokenize_one(i, &tok, &cfg))
        .collect::<Result<_, _>>()?;
    let batch = VLBatch::collate(&samples, tok.pad_id())?;
    let ids = batch.input_ids().to_vec();
    let labels = batch.labels().to_vec();
    let data_load_ms = t_data.elapsed().as_secs_f64() * 1000.0;

    let batch_info = VLBatchInfo {
        total_tokens: batch.batch() * batch.seq_len(),
        supervised_tokens: batch.supervised_len(),
        sequences: batch.batch(),
    };

    // ── Model ────────────────────────────────────────────────────────────
    let mut embed = ABEmbedding::<GlProc>::randn("embed", vocab, D_MODEL, INIT_STD, SEED)?;
    let mut head = ABLinear::<GlProc>::randn("lm_head", D_MODEL, vocab, INIT_STD, SEED + 1)?;
    let mut opt = OPAdamW::<GlProc>::new(VLAdamWConfig {
        lr: LR,
        weight_decay: 0.0,
        ..Default::default()
    });

    let report_dir = std::env::temp_dir().join("gltrain_observability");
    let mut obs = VLTrainingObserver::new(VLObserverConfig {
        live_stdout: true,
        report_dir: Some(report_dir.clone()),
        run: VLRunConfig {
            model: format!("ABEmbedding[{vocab},{D_MODEL}] + ABLinear[{D_MODEL},{vocab}]"),
            dataset: "tests/fixtures/gwen_code_dataset.jsonl".into(),
            batch_size: BATCH_SIZE,
            lr: LR,
            steps: STEPS,
            timestamp: VLRunConfig::now_stamp(),
        },
    });

    println!("── run ───────────────────────────────────────────────");
    println!("  batch        [{}, {}]", batch.batch(), batch.seq_len());
    println!(
        "  supervised   {} / {}",
        batch_info.supervised_tokens, batch_info.total_tokens
    );
    println!("  vocab        {vocab}   d_model {D_MODEL}   lr {LR}");
    println!("  reports      {}", report_dir.display());
    println!();

    let tape = Arc::new(Mutex::new(Tape::new()));

    for step in 1..=STEPS {
        // Hook 1. The batch is already collated, so only the first step pays
        // for it; the rest report zero, which is the truth for a fixed batch.
        obs.on_batch_loaded(batch_info, if step == 1 { data_load_ms } else { 0.0 });

        // Hook 2 — forward.
        let hidden = embed.forward_ids(&ids, &tape)?;
        let logits = head.forward(&hidden, &tape)?;
        let loss = logits.log_softmax()?.masked_cross_entropy(&labels)?;
        let loss_value = loss.item()?;
        obs.on_forward_complete(loss_value, &[]);

        // Hook 3 — backward. KL-006: the gradients and the emptied tape arrive
        // together, and the observer records what the tape holds afterwards so
        // a leak would be caught rather than assumed absent.
        let grads = {
            let mut guard = Tape::lock(&tape);
            guard.backward()?;
            guard.finish_step()
        };
        let tape_nodes = Tape::lock(&tape).len();
        obs.on_backward_complete(tape_nodes);

        // Hook 4 — the optimizer. Parameter values are captured on both sides
        // of the update: this is the only point at which both exist.
        let before_embed = embed.weight().to_vec()?;
        let before_head = head.weight().to_vec()?;
        let grad_of = |id| grads.get(id).map(|(g, _)| g.clone()).unwrap_or_default();
        let embed_grad = grad_of(embed.weight().id());
        let head_grad = grad_of(head.weight().id());

        {
            // ABEmbedding does not implement Module (its input is ids, not a
            // tensor), so its parameters are collected explicitly.
            let mut params: Vec<&mut TPParameter<GlProc>> = Vec::new();
            params.extend(embed.parameters_mut());
            params.extend(head.parameters_mut());
            opt.step(&mut params, &grads)?;
        }

        let opt_state: Vec<VLNamedTensor> = {
            let params: Vec<&TPParameter<GlProc>> = vec![embed.weight(), head.weight()];
            opt.state_tensors(&params)?
        };

        obs.on_optimizer_step(
            vec![
                VLParamSnapshot {
                    name: "embed.weight".into(),
                    grad: embed_grad,
                    before: before_embed,
                    after: embed.weight().to_vec()?,
                    trainable: embed.weight().is_trainable(),
                    // One row per token, so row coverage is the sparse-update
                    // story this parameter actually has.
                    row_width: Some(D_MODEL),
                },
                VLParamSnapshot {
                    name: "lm_head.weight".into(),
                    grad: head_grad,
                    before: before_head,
                    after: head.weight().to_vec()?,
                    trainable: head.weight().is_trainable(),
                    row_width: None,
                },
            ],
            &opt_state,
            opt.step_count() as u64,
            opt.effective_lr("embed.weight"),
        );

        // Hook 4b — checkpoint at the last step, verified by reloading it.
        if step == STEPS {
            let t_ckpt = Instant::now();
            let path = report_dir.join("observed_step10.safetensors");
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
            ];
            safetensors::write(&path, &tensors, &BTreeMap::new())?;
            let reloaded = safetensors::read(&path)?;
            let verified = tensors.iter().all(|t| {
                reloaded
                    .iter()
                    .find(|r| r.name == t.name)
                    .is_some_and(|r| r.data == t.data && r.shape == t.shape)
            });
            obs.on_checkpoint(verified, t_ckpt.elapsed().as_secs_f64() * 1000.0);
        }

        // Hook 5 — close the step.
        obs.on_step_complete();
    }

    // ── Hook 6 — the reports ─────────────────────────────────────────────
    let (verdict, reason) = obs.verdict();
    let curve = obs.loss_curve();

    println!("\n── loss curve ────────────────────────────────────────");
    println!(
        "  [{}]",
        curve
            .iter()
            .map(|v| format!("{v:.6}"))
            .collect::<Vec<_>>()
            .join(", ")
    );

    println!("\n── anomalies ─────────────────────────────────────────");
    let anomalies = obs.all_anomalies();
    if anomalies.is_empty() {
        println!("  none");
    } else {
        for (step, a) in &anomalies {
            println!(
                "  step {step:<3} [{}/{}] {} — {}",
                a.axis,
                a.code,
                a.severity.as_str(),
                a.message
            );
        }
    }

    println!("\n── verdict ───────────────────────────────────────────");
    println!("  {}  {}", verdict.as_str(), reason);

    if let Some((json_path, md_path)) = obs.on_training_complete()? {
        println!("\n── reports ───────────────────────────────────────────");
        println!("  json  {}", json_path.display());
        println!("  md    {}", md_path.display());
    }

    Ok(())
}
