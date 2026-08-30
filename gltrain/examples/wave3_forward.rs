//! Stummañ Deskiñ: Wave 3's forward pass, end to end on the real corpus.
//!
//! ```text
//! ids ──> ABEmbedding ──> ABLinear ──> log_softmax ──> masked_cross_entropy
//!         [n, d_model]    [n, vocab]   [n, vocab]      scalar
//! ```
//!
//! This is not a language model. There is no attention, no norm, no position
//! information, and every token is embedded and projected independently, so
//! position `i`'s logits depend only on token `i`. What it *is* is the whole
//! data-to-loss path with nothing mocked: the corpus, the mask, the gather, the
//! projection, and a masked cross-entropy that Wave 4 can call `backward()` on.
//!
//! # The number to look at
//!
//! At initialization the head is near zero, so every logit is near zero, so the
//! softmax is near uniform and the loss should land on `ln(vocab_size)`. That
//! is the standard check that a language-model head is wired correctly: a loss
//! far below it means labels are leaking into the input, and far above it means
//! the head or the mask is wrong. It is printed alongside the measured value.
//!
//! Run: cargo run --release --example wave3_forward

use gltrain::{
    ABByteTokenizer, ABEmbedding, ABLinear, DTChatMl, GlProc, Module, Tape, Tokenizer, VLBatch,
    VLChatMlConfig, IGNORE_INDEX,
};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Width of the embedding. Small — this validates the path, not a model.
const D_MODEL: usize = 64;
/// How many sequences in the batch.
const BATCH: usize = 4;
/// Truncation cap. Byte-level ids run ~4x longer than BPE would.
const MAX_SEQ_LEN: usize = 1024;
/// Initialization scale, small enough that the head starts near-uniform.
const INIT_STD: f32 = 0.02;
/// Fixed so a surprising loss reproduces exactly.
const SEED: u64 = 20260823;

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

    // The batch is [batch, seq_len]; this crate has no 3-D storage, so it is
    // flattened to one long row of tokens. Nothing is lost: without attention
    // there is no interaction between positions to preserve.
    let ids: Vec<u32> = batch.input_ids().to_vec();
    let labels: Vec<i32> = batch.labels().to_vec();
    let n_tokens = ids.len();

    println!("── data ──────────────────────────────────────────────");
    println!("  corpus       {} samples", corpus.len());
    println!("  batch        [{}, {}]", batch.batch(), batch.seq_len());
    println!("  flattened    {n_tokens} tokens");
    println!(
        "  supervised   {} ({:.1}%)",
        batch.supervised_len(),
        100.0 * batch.supervised_len() as f64 / n_tokens as f64
    );
    println!("  vocab        {vocab}");

    // ── Model ────────────────────────────────────────────────────────────
    let tape = Arc::new(Mutex::new(Tape::new()));
    let embed = ABEmbedding::<GlProc>::randn("embed", vocab, D_MODEL, INIT_STD, SEED)?;
    let head = ABLinear::<GlProc>::randn("lm_head", D_MODEL, vocab, INIT_STD, SEED + 1)?;

    let params = embed.weight().n_elems() + head.weight().n_elems();
    println!("\n── model ─────────────────────────────────────────────");
    println!("  ABEmbedding  [{vocab}, {D_MODEL}]");
    println!("  ABLinear     [{D_MODEL}, {vocab}]");
    println!("  parameters   {params}");

    // ── Forward ──────────────────────────────────────────────────────────
    let t0 = Instant::now();

    let hidden = embed.forward_ids(&ids, &tape)?;
    let t_embed = t0.elapsed();

    let logits = head.forward(&hidden, &tape)?;
    let t_head = t0.elapsed() - t_embed;

    let log_probs = logits.log_softmax()?;
    let t_softmax = t0.elapsed() - t_embed - t_head;

    let loss = log_probs.masked_cross_entropy(&labels)?;
    let elapsed = t0.elapsed();

    let loss_value = loss.item()?;

    println!("\n── forward ───────────────────────────────────────────");
    println!("  hidden       {:?}", hidden.shape());
    println!("  logits       {:?}", logits.shape());
    println!("  log_probs    {:?}", log_probs.shape());
    println!("  loss         {:?}", loss.shape());
    println!("  tape nodes   {}", Tape::lock(&tape).len());
    println!(
        "  timings      embed {:?}  head {:?}  log_softmax {:?}  total {:?}",
        t_embed, t_head, t_softmax, elapsed
    );

    let uniform = (vocab as f32).ln();
    println!("\n── loss ──────────────────────────────────────────────");
    println!("  loss         {loss_value:.6}");
    println!("  ln(vocab)    {uniform:.6}   <- where a correct near-zero init lands");
    println!("  difference   {:+.6}", loss_value - uniform);
    println!("  perplexity   {:.3}", loss_value.exp());

    // ── Checks the run performs on itself ────────────────────────────────
    assert!(loss_value.is_finite(), "loss is {loss_value}");
    assert!(loss_value > 0.0, "cross-entropy cannot be negative");
    assert_eq!(logits.shape(), &[n_tokens, vocab]);
    assert!(
        log_probs
            .to_vec()?
            .iter()
            .all(|v| v.is_finite() && *v <= 0.0),
        "a log-probability must be finite and at most 0"
    );
    // Padding and prompt positions must not have reached the loss.
    let supervised = labels.iter().filter(|&&l| l != IGNORE_INDEX).count();
    assert_eq!(supervised, batch.supervised_len());
    println!("\n  finite, positive, correctly shaped ✓");

    Ok(())
}
