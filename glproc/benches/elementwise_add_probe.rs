//! Bias-add / residual-add: currently plain `for (a, b) in x.iter_mut().zip(y)
//! { *a += b }` loops in `runner.rs` (unlike `rms_norm_into`/`silu_mul`,
//! which already dispatch through explicit AVX2 kernels). Tests whether
//! LLVM already auto-vectorizes this pattern at release opt level, or
//! whether an explicit AVX2 kernel buys anything real.
//!
//! Scale context (Qwen2.5-0.5B, 24 layers): per layer, up to 5 of these
//! loops (bq 896, bk 128, bv 128, residual x2 @ 896) = up to 2944
//! elements/layer x 24 layers = ~70,656 f32 adds/token — much smaller in
//! raw op count than RoPE's redundant sin_cos() calls were, so expectations
//! here should be modest going in.
//!
//! Run: cargo bench -p glproc --bench elementwise_add_probe

use std::arch::x86_64::*;
use std::time::Instant;

#[path = "common/mod.rs"]
mod common;
use common::prng_f32;

/// Exact copy of the current pattern in `runner.rs` (bias add / residual add).
fn add_scalar(a: &mut [f32], b: &[f32]) {
    for (ai, bi) in a.iter_mut().zip(b) {
        *ai += bi;
    }
}

/// Explicit AVX2 version, for comparison.
#[target_feature(enable = "avx2")]
unsafe fn add_avx2(a: &mut [f32], b: &[f32]) {
    let n = a.len();
    let mut i = 0;
    while i + 8 <= n {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        _mm256_storeu_ps(a.as_mut_ptr().add(i), _mm256_add_ps(va, vb));
        i += 8;
    }
    while i < n {
        a[i] += b[i];
        i += 1;
    }
}

fn main() {
    const N_LAYERS: usize = 24;
    const DIM: usize = 896;
    const Q_DIM: usize = 896; // 14 heads x 64
    const KV_DIM: usize = 128; // 2 heads x 64

    let mut seed = 0xADD1u64;
    let mut q_s = vec![0.1f32; Q_DIM];
    let mut q_v = vec![0.1f32; Q_DIM];
    let bq: Vec<f32> = (0..Q_DIM).map(|_| prng_f32(&mut seed)).collect();
    let mut k_s = vec![0.1f32; KV_DIM];
    let mut k_v = vec![0.1f32; KV_DIM];
    let bk: Vec<f32> = (0..KV_DIM).map(|_| prng_f32(&mut seed)).collect();
    let mut x_s = vec![0.1f32; DIM];
    let mut x_v = vec![0.1f32; DIM];
    let proj: Vec<f32> = (0..DIM).map(|_| prng_f32(&mut seed)).collect();

    let iters = 5000;

    // --- scalar (current) ---
    let t = Instant::now();
    for _ in 0..iters {
        for _layer in 0..N_LAYERS {
            add_scalar(&mut q_s, &bq); // bq
            add_scalar(&mut k_s, &bk); // bk
            add_scalar(&mut k_s, &bk); // bv (same size as bk, reuse buffer)
            add_scalar(&mut x_s, &proj); // residual 1
            add_scalar(&mut x_s, &proj); // residual 2
        }
        std::hint::black_box((&q_s, &k_s, &x_s));
    }
    let scalar_ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;

    // --- explicit AVX2 ---
    let t = Instant::now();
    for _ in 0..iters {
        for _layer in 0..N_LAYERS {
            unsafe {
                add_avx2(&mut q_v, &bq);
                add_avx2(&mut k_v, &bk);
                add_avx2(&mut k_v, &bk);
                add_avx2(&mut x_v, &proj);
                add_avx2(&mut x_v, &proj);
            }
        }
        std::hint::black_box((&q_v, &k_v, &x_v));
    }
    let avx2_ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;

    println!("Qwen2.5-0.5B shape: {N_LAYERS} layers x (bq {Q_DIM} + bk/bv {KV_DIM}x2 + residual {DIM}x2)");
    println!("Total adds/token: {}\n", N_LAYERS * (Q_DIM + 2 * KV_DIM + 2 * DIM));
    println!("{:<10} {:>12}", "impl", "us/token");
    println!("{:<10} {:>12.3}", "scalar", scalar_ms * 1e3);
    println!("{:<10} {:>12.3}", "avx2", avx2_ms * 1e3);
    println!("\nspeedup: {:.2}x", scalar_ms / avx2_ms);
    println!("\nContext: at ~37 tok/s (27ms/token decode), this cost is");
    println!("scalar={:.3}% / avx2={:.3}% of one decode step's wall clock.",
        scalar_ms / 27.0 * 100.0, avx2_ms / 27.0 * 100.0);
}
