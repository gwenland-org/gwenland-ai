//! GWEN-220 Wave 4 — real-attention loss-trend validation on a real GGUF.
//!
//! Wave 4 asks that, with the mean-pool surrogate replaced by a real attention
//! + MLP forward, the training loss is a *faithful proxy* — it must behave like
//! a real language-model loss on real data, not collapse or diverge.
//!
//! IMPORTANT — why this test trains on the real dataset, not a synthetic one:
//! an earlier draft fed a *single repeated token sequence* as a fast overfit
//! smoke test. That drove the loss to 0.0000 in ~3 steps — a memorization
//! artifact of the degenerate dataset (one example, 8.7M LoRA params), *not* a
//! property of the attention code. A faithful trend must use varied real text,
//! so this test runs the exact CLI dry-run path (`run_native_local`) over the
//! `sample_100.jsonl` fixture for several optimiser steps and checks the loss
//! stays finite and in a sane language-model band without diverging.
//!
//! Env-gated so CI without the model skips it:
//!
//! ```text
//! GWEN_DRYRUN_GGUF="C:/Users/me/Downloads/Qwen3-1.7B-Q8_0.gguf" \
//!   cargo test -p gwenland-core --test gwen220_wave4 -- --nocapture
//! ```
//!
//! Each optimiser step re-dequantizes all 28 Q8_0 layers twice (~200 s/step on
//! an i3), so keep `GWEN220_STEPS` small. Default 8.

use std::path::PathBuf;
use std::sync::mpsc;

use gwenland_core::train::config::{LoraConfig, NewTrainConfig};
use gwenland_core::train::native_runner::run_native_local;

/// All seven transformer projections — full GWEN-219 multi-tensor routing.
fn all_projections() -> Vec<String> {
    ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        .iter()
        .map(|s| s.to_string())
        .collect()
}

/// Extract every `"loss":<f32>` from the loop's `{"event":"step",…}` JSON lines.
fn losses_from_events(rx: &mpsc::Receiver<String>) -> Vec<f32> {
    let mut out = Vec::new();
    while let Ok(line) = rx.try_recv() {
        if !line.contains("\"event\":\"step\"") {
            continue;
        }
        if let Some(pos) = line.find("\"loss\":") {
            let rest = &line[pos + 7..];
            let end = rest
                .find(|c: char| c != '-' && c != '.' && !c.is_ascii_digit())
                .unwrap_or(rest.len());
            if let Ok(v) = rest[..end].parse::<f32>() {
                out.push(v);
            }
        }
    }
    out
}

#[test]
fn gwen220_wave4_real_attention_loss_trend_is_faithful() {
    let Ok(gguf) = std::env::var("GWEN_DRYRUN_GGUF") else {
        eprintln!("[gwen220_wave4] SKIP — set GWEN_DRYRUN_GGUF to a local .gguf to run");
        return;
    };
    let gguf = PathBuf::from(gguf);
    assert!(gguf.exists(), "GWEN_DRYRUN_GGUF does not exist: {}", gguf.display());

    let steps: usize = std::env::var("GWEN220_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);

    // Default to the bundled fixture; override with GWEN220_DATASET to point at
    // a larger local dataset (e.g. a curated 200-entry subset) without baking it
    // into the repo.
    let dataset = std::env::var("GWEN220_DATASET")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/sample_100.jsonl")
        });
    assert!(dataset.exists(), "dataset does not exist: {}", dataset.display());
    let output = std::env::temp_dir().join("gwen220_wave4_out");
    std::fs::create_dir_all(&output).unwrap();

    // `max_steps = Some(steps)` forces grad_accum = 1 (one optimiser step per
    // batch) and emits the `[dry-run]` RSS report — the exact CLI dry-run path,
    // just run for several steps so the loss trend is observable on real data.
    let config = NewTrainConfig {
        dataset_path: dataset,
        output_path: output,
        epochs: 1,
        batch_size: 1,
        grad_accum: 1,
        lr: 1e-4,
        dry_run: true,
        max_steps: Some(steps),
        lora: LoraConfig {
            r: 8,
            alpha: 16.0,
            dropout: 0.05,
            target_modules: all_projections(),
        },
        ..NewTrainConfig::default()
    };

    let (tx, rx) = mpsc::channel::<String>();
    eprintln!("[gwen220_wave4] training {steps} steps on real dataset / {} …", gguf.display());
    let result = run_native_local(&config, &gguf, None, Some(tx))
        .expect("real-dataset dry-run training failed");

    let losses = losses_from_events(&rx);
    eprintln!("[gwen220_wave4] per-step loss: {:?}", losses);
    assert!(!losses.is_empty(), "no per-step loss events captured");

    // 1. Every step is finite — no NaN/Inf from the attention/MLP forward.
    assert!(
        losses.iter().all(|v| v.is_finite()) && result.final_loss.is_finite(),
        "non-finite loss: trend={losses:?} final={}",
        result.final_loss
    );

    // 2. Loss sits in a sane language-model band. With real pretrained weights
    //    and real text the frozen base already predicts well (≈2.8 at step 1),
    //    far below the mean-pool baseline's ~ln(vocab)=9.0. It must never exceed
    //    ~ln(vocab); a value pinned at 0 would signal the degenerate-overfit bug
    //    this test was rewritten to avoid.
    let vocab_ln_ceiling = 9.5_f32; // ln(8192)=9.01 + margin
    for (i, &l) in losses.iter().enumerate() {
        assert!(
            l > 0.01 && l < vocab_ln_ceiling,
            "step {i} loss {l} outside sane band (0.01, {vocab_ln_ceiling}) — \
             pinned-0 = overfit bug, > ceiling = divergence"
        );
    }

    // 3. Real attention must not be *worse* than the mean-pool baseline: over a
    //    short run the loss should not run away upward. We allow step-to-step
    //    noise (real fine-tuning is noisy) but the last-third mean must not be
    //    meaningfully above the first-third mean.
    let third = (losses.len() / 3).max(1);
    let mean = |s: &[f32]| s.iter().sum::<f32>() / s.len() as f32;
    let first = mean(&losses[..third]);
    let last = mean(&losses[losses.len() - third..]);
    eprintln!(
        "[gwen220_wave4] first-third mean={first:.4}  last-third mean={last:.4}  final={:.4}",
        result.final_loss
    );
    assert!(
        last <= first + 0.5,
        "loss trending upward (real-attention regression?): first={first:.4} last={last:.4}"
    );

    eprintln!("[gwen220_wave4] ✓ loss finite, in sane LM band, not diverging — faithful trend");
}
