//! Stummañ Deskiñ: inspect the ChatML data pipeline.
//!
//! Wave 2's exit criterion, as a runnable program. For the first 5 samples of
//! `tests/fixtures/gwen_code_dataset.jsonl` it prints:
//!   1. the parsed roles and their content lengths
//!   2. the tokenized shape, and how much of it is supervised
//!   3. the decoded assistant label slice — the text the model is asked to
//!      produce, recovered from `labels` alone
//!   4. the collated batch: `[batch, seq_len]`, padding, and the mask
//!
//! Point 3 is the one that matters. It is reconstructed by dropping every
//! `IGNORE_INDEX` from `labels` and decoding what is left, so it can only come
//! out right if the mask landed on exactly the assistant span. A prompt that
//! leaked in, or a response that was masked out, is visible immediately.
//!
//! Run: cargo run --example chatml_inspect

use gltrain::{
    ABByteTokenizer, DTChatMl, Tokenizer, VLBatch, VLChatMlConfig, VLTokenizedSample, IGNORE_INDEX,
};
use std::path::Path;

/// Byte-level ids run ~4x longer than Qwen2.5's BPE would, so the cap is
/// generous. Nothing in this corpus reaches it.
const MAX_SEQ_LEN: usize = 16_384;

/// How many samples to show in full.
const SHOW: usize = 5;

/// One line, clipped, with newlines made visible.
fn preview(s: &str, width: usize) -> String {
    let flat: String = s.chars().map(|c| if c == '\n' { '⏎' } else { c }).collect();
    let taken: String = flat.chars().take(width).collect();
    if flat.chars().count() > width {
        format!("{taken}…")
    } else {
        taken
    }
}

fn main() -> anyhow::Result<()> {
    let path = Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/gwen_code_dataset.jsonl"
    ));

    let corpus = DTChatMl::from_jsonl_path(path)?;
    let tok = ABByteTokenizer::new();
    let cfg = VLChatMlConfig::new(MAX_SEQ_LEN);

    println!("corpus     {}", path.display());
    println!("samples    {}", corpus.len());
    println!("tokenizer  ABByteTokenizer (vocab {})", tok.vocab_size());
    println!(
        "config     max_seq_len={} supervise_im_end={}",
        cfg.max_seq_len, cfg.supervise_im_end
    );
    println!("ignore     IGNORE_INDEX = {IGNORE_INDEX}");

    let mut shown: Vec<VLTokenizedSample> = Vec::with_capacity(SHOW);

    for i in 0..SHOW.min(corpus.len()) {
        let sample = corpus.get(i).expect("i < corpus.len()");
        let t = corpus.tokenize_one(i, &tok, &cfg)?;

        println!("\n─── sample {i} ───────────────────────────────────────────");
        for turn in &sample.turns {
            println!(
                "  {:<9} {:>6} chars  {}",
                turn.role.as_str(),
                turn.content.len(),
                preview(&turn.content, 58)
            );
        }

        let supervised = t.supervised_len();
        println!(
            "  shape     input_ids [{}]  labels [{}]  truncated={}",
            t.len(),
            t.labels().len(),
            t.is_truncated()
        );
        println!(
            "  masked    {} of {} positions supervised ({:.1}%)",
            supervised,
            t.len(),
            100.0 * supervised as f64 / t.len() as f64
        );

        // The check: recover the target text from `labels` alone.
        let decoded = tok.decode(&t.supervised_ids());
        println!("  labels →  {}", preview(&decoded, 100));
        println!(
            "  ends in   {}",
            if decoded.ends_with("<|im_end|>") {
                "<|im_end|>  ✓ the stop token is supervised"
            } else {
                "??          ✗ the model would never learn to stop"
            }
        );

        shown.push(t);
    }

    // ── Collation ────────────────────────────────────────────────────────
    let batch = VLBatch::collate(&shown, tok.pad_id())?;
    println!("\n─── collated batch ──────────────────────────────────────");
    println!("  shape     [{}, {}]", batch.batch(), batch.seq_len());
    println!(
        "  payloads  input_ids {}  labels {}  attention_mask {}",
        batch.input_ids().len(),
        batch.labels().len(),
        batch.attention_mask().len()
    );
    println!("  pad_id    {}", tok.pad_id());
    println!(
        "  supervised {} of {} positions ({:.1}%)",
        batch.supervised_len(),
        batch.batch() * batch.seq_len(),
        100.0 * batch.supervised_len() as f64 / (batch.batch() * batch.seq_len()) as f64
    );

    println!("\n  row   real   pad    supervised");
    for b in 0..batch.batch() {
        let mask = &batch.attention_mask()[b * batch.seq_len()..(b + 1) * batch.seq_len()];
        let real = mask.iter().filter(|&&m| m == 1).count();
        let sup = batch
            .labels_row(b)
            .expect("b < batch")
            .iter()
            .filter(|&&l| l != IGNORE_INDEX)
            .count();
        println!("  {b:<5} {real:<6} {:<6} {sup}", batch.seq_len() - real);
    }

    // Padding must contribute nothing. Asserted here as well as in the unit
    // tests, so running the example is itself a check.
    let expected: usize = shown.iter().map(VLTokenizedSample::supervised_len).sum();
    assert_eq!(
        batch.supervised_len(),
        expected,
        "padding leaked into the supervised count"
    );
    println!("\n  padding adds no training signal ✓");

    // ── Whole-corpus summary ─────────────────────────────────────────────
    let all = corpus.tokenize(&tok, &cfg)?;
    let total: usize = all.iter().map(VLTokenizedSample::len).sum();
    let supervised: usize = all.iter().map(VLTokenizedSample::supervised_len).sum();
    let truncated = all.iter().filter(|s| s.is_truncated()).count();
    let longest = all.iter().map(VLTokenizedSample::len).max().unwrap_or(0);
    let unsupervised = all.iter().filter(|s| s.supervised_len() == 0).count();

    println!("\n─── whole corpus ────────────────────────────────────────");
    println!("  samples      {}", all.len());
    println!("  tokens       {total}");
    println!(
        "  supervised   {supervised} ({:.1}%)",
        100.0 * supervised as f64 / total as f64
    );
    println!("  longest      {longest} tokens");
    println!(
        "  mean         {:.0} tokens/sample",
        total as f64 / all.len() as f64
    );
    println!("  truncated    {truncated}");
    println!("  no signal    {unsupervised}");

    Ok(())
}
