//! Current per-head RoPE (`runner.rs::rope`, one `sin_cos()` call per pair,
//! called once per head) vs precompute-once-reuse — testing a fragmentation
//! hypothesis, not a SIMD-width one: `rope_freq_base`/`rope_style`/`head_dim`
//! are shared `Config` fields, not per-layer, so the (cos, sin) values for a
//! given token position are IDENTICAL across every Q/K head and every layer.
//! On Qwen2.5-0.5B (24 layers, 14 Q heads, 2 KV heads, head_dim=64), the
//! current code calls `sin_cos()` 32 (head_dim/2) x 16 (heads) x 24 (layers)
//! = 12,288 times per token where 32 would mathematically suffice — a 384x
//! redundancy factor.
//!
//! `rope()` itself is `fn`-private to `runner.rs` (not `pub`), so this probe
//! replicates its exact logic rather than touching production code — this
//! is intentionally a pure measurement, zero production risk, per this
//! session's established probe-first pattern.
//!
//! Run: cargo bench -p glproc --bench rope_probe

use std::time::Instant;

#[derive(Clone, Copy, PartialEq)]
enum RopeStyle {
    Norm,
    Neox,
}

/// Exact copy of `runner.rs::rope` — the current per-head implementation.
fn rope_current(x: &mut [f32], pos: usize, head_dim: usize, freq_base: f32, style: RopeStyle) {
    let half = head_dim / 2;
    for i in 0..half {
        let freq = 1.0 / freq_base.powf(2.0 * i as f32 / head_dim as f32);
        let theta = pos as f32 * freq;
        let (sin, cos) = theta.sin_cos();
        let (a, b) = match style {
            RopeStyle::Norm => (2 * i, 2 * i + 1),
            RopeStyle::Neox => (i, i + half),
        };
        let x0 = x[a];
        let x1 = x[b];
        x[a] = x0 * cos - x1 * sin;
        x[b] = x0 * sin + x1 * cos;
    }
}

/// Precompute the (cos, sin) table once per token position — `half` pairs,
/// shared by every head and every layer since `freq_base`/`style`/`head_dim`
/// are constant `Config` fields, not per-layer.
fn rope_table(pos: usize, head_dim: usize, freq_base: f32) -> Vec<(f32, f32)> {
    let half = head_dim / 2;
    (0..half)
        .map(|i| {
            let freq = 1.0 / freq_base.powf(2.0 * i as f32 / head_dim as f32);
            let theta = pos as f32 * freq;
            theta.sin_cos()
        })
        .collect()
}

/// Apply a precomputed table to one head — no `sin_cos()` call, no `powf`.
fn rope_apply(x: &mut [f32], table: &[(f32, f32)], head_dim: usize, style: RopeStyle) {
    let half = head_dim / 2;
    for (i, &(sin, cos)) in table.iter().enumerate().take(half) {
        let (a, b) = match style {
            RopeStyle::Norm => (2 * i, 2 * i + 1),
            RopeStyle::Neox => (i, i + half),
        };
        let x0 = x[a];
        let x1 = x[b];
        x[a] = x0 * cos - x1 * sin;
        x[b] = x0 * sin + x1 * cos;
    }
}

fn main() {
    // Qwen2.5-0.5B shape.
    const N_LAYERS: usize = 24;
    const N_HEADS: usize = 14;
    const N_KV_HEADS: usize = 2;
    const HEAD_DIM: usize = 64;
    const FREQ_BASE: f32 = 1_000_000.0; // Qwen2.5's actual rope_freq_base
    let style = RopeStyle::Neox;

    let pos = 512usize; // mid-context, representative

    // One Q vector (n_heads * head_dim) and one K vector (n_kv_heads *
    // head_dim) per layer, matching the real per-layer call pattern.
    let mut q = vec![0.37f32; N_HEADS * HEAD_DIM];
    let mut k = vec![0.61f32; N_KV_HEADS * HEAD_DIM];

    let iters = 200;

    // --- current: sin_cos() called fresh in every rope() invocation ---
    let t = Instant::now();
    for _ in 0..iters {
        for _layer in 0..N_LAYERS {
            for h in 0..N_HEADS {
                rope_current(&mut q[h * HEAD_DIM..(h + 1) * HEAD_DIM], pos, HEAD_DIM, FREQ_BASE, style);
            }
            for h in 0..N_KV_HEADS {
                rope_current(&mut k[h * HEAD_DIM..(h + 1) * HEAD_DIM], pos, HEAD_DIM, FREQ_BASE, style);
            }
        }
        std::hint::black_box((&q, &k));
    }
    let current_ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;

    // --- proposed: table computed once per token, reused across every
    // head and every layer (valid because freq_base/style/head_dim are
    // Config-level, not per-layer) ---
    let t = Instant::now();
    for _ in 0..iters {
        let table = rope_table(pos, HEAD_DIM, FREQ_BASE);
        for _layer in 0..N_LAYERS {
            for h in 0..N_HEADS {
                rope_apply(&mut q[h * HEAD_DIM..(h + 1) * HEAD_DIM], &table, HEAD_DIM, style);
            }
            for h in 0..N_KV_HEADS {
                rope_apply(&mut k[h * HEAD_DIM..(h + 1) * HEAD_DIM], &table, HEAD_DIM, style);
            }
        }
        std::hint::black_box((&q, &k, &table));
    }
    let table_ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;

    // --- correctness: current vs table-based must match exactly (same
    // trig values, same rotation math, only WHERE sin_cos is computed
    // differs) ---
    let mut q_current = vec![0.37f32; N_HEADS * HEAD_DIM];
    let mut q_table = q_current.clone();
    for h in 0..N_HEADS {
        rope_current(&mut q_current[h * HEAD_DIM..(h + 1) * HEAD_DIM], pos, HEAD_DIM, FREQ_BASE, style);
    }
    let table = rope_table(pos, HEAD_DIM, FREQ_BASE);
    for h in 0..N_HEADS {
        rope_apply(&mut q_table[h * HEAD_DIM..(h + 1) * HEAD_DIM], &table, HEAD_DIM, style);
    }
    let max_diff = q_current.iter().zip(&q_table).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);

    println!("Qwen2.5-0.5B shape: {N_LAYERS} layers x ({N_HEADS} Q heads + {N_KV_HEADS} KV heads), head_dim={HEAD_DIM}");
    println!("sin_cos() calls per token: current = {}, table = {}\n", (N_HEADS + N_KV_HEADS) * (HEAD_DIM / 2) * N_LAYERS, HEAD_DIM / 2);
    println!("{:<10} {:>12} {:>16}", "impl", "ms/token", "sin_cos calls");
    println!("{:<10} {:>12.4} {:>16}", "current", current_ms, (N_HEADS + N_KV_HEADS) * (HEAD_DIM / 2) * N_LAYERS);
    println!("{:<10} {:>12.4} {:>16}", "table", table_ms, HEAD_DIM / 2);
    println!("\nspeedup: {:.2}x | max correctness diff: {:.2e}", current_ms / table_ms, max_diff);
    println!("\nContext: at ~37 tok/s (27ms/token decode), this RoPE cost is");
    println!("current={:.2}% / table={:.2}% of one decode step's wall clock.",
        current_ms / 27.0 * 100.0, table_ms / 27.0 * 100.0);
}
