//! Colab-only MoE verification at Qwen3-30B-A3B scale, on Q8_0 experts.
//! Too slow for the default suite (128 experts x 2048x768); run explicitly.

use glproc::kernels::bridge::QuantFormat;
use glproc::kernels::dequant::q8_0;
use glproc::model::{GateUp, WeightMatrix};
use glproc::moe::{ExpertWeights, MoEConfig, MoELayer};
use glproc::simd_strategy::SimdStrategy;
use glproc::threading::ThreadPool;

/// Deterministic pseudo-random f32 in [-1, 1). No rand dep (crate has none).
fn prng(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*seed >> 33) as f32 / (1u64 << 31) as f32) - 1.0
}

fn rand_vec(n: usize, seed: &mut u64) -> Vec<f32> {
    (0..n).map(|_| prng(seed)).collect()
}

/// Build an MoE layer whose experts are Q8_0, in the row-interleaved
/// GateUp::FusedQuant layout the loader produces — i.e. the real decode path.
/// Returns the layer plus the f32 weights it was quantized FROM, so the
/// reference can be computed without re-dequantizing.
fn build_q8_layer(
    ne: usize,
    k: usize,
    h: usize,
    f: usize,
    seed: &mut u64,
) -> (MoELayer, Vec<(Vec<f32>, Vec<f32>, Vec<f32>)>) {
    let router = rand_vec(ne * h, seed);
    let mut experts = Vec::with_capacity(ne);
    let mut f32_ref = Vec::with_capacity(ne);

    for _ in 0..ne {
        let gate = rand_vec(f * h, seed);
        let up = rand_vec(f * h, seed);
        let down = rand_vec(h * f, seed);

        // Quantize, then dequantize back to f32 — the reference must be
        // computed from the ROUNDED weights, else we'd be measuring
        // quantization error and calling it a routing bug.
        let gq = q8_0::scalar::quantize(&gate);
        let uq = q8_0::scalar::quantize(&up);
        let dq = q8_0::scalar::quantize(&down);

        // Row-interleave gate/up exactly as loader::fuse_gate_up does:
        // [gate row 0][up row 0][gate row 1]...
        let row_bytes = gq.len() / f;
        let mut packed = Vec::with_capacity(gq.len() + uq.len());
        for o in 0..f {
            packed.extend_from_slice(&gq[o * row_bytes..(o + 1) * row_bytes]);
            packed.extend_from_slice(&uq[o * row_bytes..(o + 1) * row_bytes]);
        }

        f32_ref.push((
            q8_0::scalar::run(&gq),
            q8_0::scalar::run(&uq),
            q8_0::scalar::run(&dq),
        ));
        experts.push(ExpertWeights {
            gate_up: GateUp::FusedQuant(QuantFormat::Q8_0, packed),
            w_down: WeightMatrix::Quant(QuantFormat::Q8_0, dq),
        });
    }

    let layer = MoELayer {
        config: MoEConfig {
            num_experts: ne,
            num_experts_per_tok: k,
            expert_ffn_size: f,
            hidden_size: h,
            norm_topk_prob: true,
        },
        router,
        experts,
    };
    (layer, f32_ref)
}

/// Naive MoE, straight from the definition. Uses f32::exp deliberately —
/// sharing fast_exp with the code under test would make them agree by
/// construction and stop testing the approximation.
fn reference(
    layer: &MoELayer,
    w: &[(Vec<f32>, Vec<f32>, Vec<f32>)],
    input: &[f32],
    n_tokens: usize,
) -> Vec<f32> {
    let c = &layer.config;
    let (h, f, ne, k) = (
        c.hidden_size,
        c.expert_ffn_size,
        c.num_experts,
        c.num_experts_per_tok,
    );
    let mut out = vec![0f32; n_tokens * h];

    for t in 0..n_tokens {
        let x = &input[t * h..(t + 1) * h];

        // Router -> softmax over ALL experts -> top-k -> renormalize.
        // This order is Qwen3's. Selecting before the softmax gives
        // different weights, so it is part of what we are verifying.
        let scores: Vec<f32> = (0..ne)
            .map(|e| (0..h).map(|i| layer.router[e * h + i] * x[i]).sum())
            .collect();
        let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let probs: Vec<f32> = exps.iter().map(|e| e / sum).collect();

        let mut idx: Vec<usize> = (0..ne).collect();
        idx.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap().then(a.cmp(&b)));
        let top = &idx[..k];
        let mass: f32 = top.iter().map(|&e| probs[e]).sum();

        for &e in top {
            let wt = probs[e] / mass;
            let (gw, uw, dw) = &w[e];
            let mut act = vec![0f32; f];
            for o in 0..f {
                let g: f32 = (0..h).map(|i| gw[o * h + i] * x[i]).sum();
                let u: f32 = (0..h).map(|i| uw[o * h + i] * x[i]).sum();
                act[o] = g / (1.0 + (-g).exp()) * u;
            }
            for o in 0..h {
                let d: f32 = (0..f).map(|i| dw[o * f + i] * act[i]).sum();
                out[t * h + o] += wt * d;
            }
        }
    }
    out
}

fn check(name: &str, ne: usize, k: usize, h: usize, f: usize, n_tokens: usize) {
    let mut seed = 0xC0FFEEu64;
    let (layer, wref) = build_q8_layer(ne, k, h, f, &mut seed);
    let input = rand_vec(n_tokens * h, &mut seed);
    let pool = ThreadPool::new(num_cpus::get());
    let strategy = SimdStrategy::detect();

    let mut got = vec![0f32; n_tokens * h];
    let routing = layer.forward(&input, &mut got, n_tokens, &pool, strategy);
    let want = reference(&layer, &wref, &input, n_tokens);

    // Relative tolerance: forward() and the reference sum the same products
    // in different orders (threaded/batched vs sequential), and the SwiGLU
    // goes through fast_exp on one side and libm exp on the other.
    let mut worst = 0f32;
    for (g, w) in got.iter().zip(&want) {
        let rel = (g - w).abs() / w.abs().max(1.0);
        worst = worst.max(rel);
    }

    let load = &routing.expert_load;
    let active = load.iter().filter(|&&c| c > 0).count();
    let (mn, mx) = (
        load.iter().copied().min().unwrap(),
        load.iter().copied().max().unwrap(),
    );
    println!(
        "{name:<26} tokens={n_tokens:<4} worst_rel_err={worst:.2e}  \
         experts_touched={active}/{ne}  load min/max={mn}/{mx}"
    );

    assert_eq!(
        load.iter().sum::<usize>(),
        n_tokens * k,
        "{name}: every token must be routed to exactly top_k experts"
    );
    // 2e-3 relative: Q8_0 is ~7 bits of mantissa, and error compounds across
    // three quantized projections plus the fast_exp approximation. A routing
    // or dispatch BUG shows up as 1e-1+, not 1e-3 — this bound separates them.
    assert!(worst < 2e-3, "{name}: worst relative error {worst:.3e} — too large");
}

#[test]
fn moe_q8_qwen3_30b_a3b_shape() {
    // The real thing: Qwen3-30B-A3B's per-layer MoE block.
    println!();
    check("qwen3-30b-a3b decode", 128, 8, 2048, 768, 1);
    check("qwen3-30b-a3b prefill", 128, 8, 2048, 768, 32);
}

#[test]
fn moe_q8_ffn_wider_than_hidden() {
    // The shape Qwen3 does NOT have: expert_ffn > hidden. This is the case
    // where a shared-acts buffer sized to `hidden` writes out of bounds —
    // silently, in release, since quantize() only debug_asserts the fit.
    // If the scratch sizing regresses to min instead of max, this corrupts.
    println!();
    check("ffn>hidden (OOB guard)", 16, 4, 256, 1024, 8);
}

#[test]
fn moe_top_k_is_not_hardcoded() {
    // Mixtral (top-2 of 8) and Qwen3 (top-8 of 128) must both work. A
    // hardcoded top_k passes one and fails the other.
    println!();
    check("mixtral-ish top-2/8", 8, 2, 512, 256, 16);
    check("qwen3-ish top-8/128", 128, 8, 512, 256, 16);
}