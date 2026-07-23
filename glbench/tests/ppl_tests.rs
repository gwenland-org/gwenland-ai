//! Integration tests for `glbench ppl` (feature-gated: `gllm-bench` only).
//!
//! Per the brief, these are pure math/struct tests — none load a real model.
//! `GllmEngine::score_sequence` (the teacher-forced logits path this command
//! drives) has its own real-package tests in glictus-caliburni itself
//! (`runtime::gllm_engine::tests`).

#![cfg(feature = "gllm-bench")]

use glbench::export::json::parse as parse_json;
use glbench::ppl::{PplSession, WIKITEXT2_SAMPLE};

/// Mirrors `ppl.rs`'s private `log_softmax` — kept in sync deliberately (a
/// second implementation of three lines of math is cheaper than making the
/// function `pub` just for a test to reach it) and checked against a naive
/// reference softmax below.
fn log_softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f64 = logits.iter().map(|&l| ((l - max) as f64).exp()).sum();
    let log_sum_exp = sum_exp.ln();
    logits.iter().map(|&l| l - max - log_sum_exp as f32).collect()
}

#[test]
fn test_ppl_log_softmax_correctness() {
    let logits = [2.0f32, 1.0, 0.1, -1.5];
    let lp = log_softmax(&logits);

    // Naive reference: softmax then ln, no max-shift trick.
    let sum: f64 = logits.iter().map(|&l| (l as f64).exp()).sum();
    for (i, &l) in logits.iter().enumerate() {
        let reference = ((l as f64).exp() / sum).ln();
        assert!(
            (lp[i] as f64 - reference).abs() < 1e-5,
            "index {i}: got {} expected {reference}",
            lp[i]
        );
    }
}

#[test]
fn test_ppl_sliding_window_token_count() {
    // Fake a 100-token sequence, context=20, stride=10: replicate the
    // window-count/evaluated-token bookkeeping `sliding_window_log_probs`
    // uses internally, since that function needs a real GllmEngine to call
    // and this test is pure bookkeeping math.
    let total_tokens = 100usize;
    let context = 20usize;
    let stride = 10usize;

    let mut window_start = 0usize;
    let mut window_count = 0usize;
    let mut evaluated_tokens = 0usize;
    while window_start + context <= total_tokens {
        let new_start = if window_start == 0 { 0 } else { context - stride };
        let mut this_window_evaluated = 0usize;
        for i in new_start..context {
            let target_abs = window_start + i + 1;
            if target_abs >= total_tokens {
                break;
            }
            this_window_evaluated += 1;
        }
        evaluated_tokens += this_window_evaluated;
        window_count += 1;
        window_start += stride;
    }

    // Windows at starts 0,10,...,80 (80+20=100 fits) => 9 windows.
    assert_eq!(window_count, 9);
    // First window evaluates all 20 new-token targets (0..20, all < 100).
    // Every later window evaluates only the last `stride` (10) positions,
    // except never double-counting the overlap.
    // window 0: new_start=0..20 -> 20 targets (all valid, since target_abs <= 20)
    // windows 1..8 (8 windows): new_start=10..20 -> 10 targets each = 80
    // total = 20 + 80 = 100, but the very last target (abs 100) is out of range and skipped.
    assert_eq!(evaluated_tokens, 99, "no token double-counted, last out-of-range target dropped");
}

#[test]
fn test_ppl_cross_entropy_to_perplexity() {
    let ce = 3.0f64;
    let ppl = ce.exp();
    assert!((ppl - 20.0855369).abs() < 1e-4, "got {ppl}");
}

#[test]
fn test_ppl_session_serializes() {
    let session = PplSession {
        model_path: "qwen-gq2a-cpp.gllm".into(),
        context_len: 512,
        stride: 256,
        dataset: "wikitext2-sample-embedded",
        total_tokens: 2041,
        evaluated_tokens: 1785,
        perplexity: 26.74,
        cross_entropy_mean: 3.2847,
        per_window_ce: vec![3.1, 3.2, 3.4, 3.3],
        timestamp: "2026-07-23T12:00:00Z".into(),
    };

    let text = session.to_json().to_pretty();
    let json = parse_json(&text).unwrap();
    let back = PplSession::from_json(&json).unwrap();

    assert_eq!(session, back);
}

#[test]
fn test_ppl_wikitext_sample_nonempty() {
    assert!(WIKITEXT2_SAMPLE.len() > 1000, "len={}", WIKITEXT2_SAMPLE.len());
    assert!(WIKITEXT2_SAMPLE.contains("Valkyria"));
}
