//! Does Q4_K's byte-saving EVER materialize on Tiger Lake, or does nibble
//! unpack always cancel the gain?
//!
//! The single-thread decode probe answered "no" — Q4_K runs compute-bound at
//! ~3.7 GB/s, far under the 28.7 GB/s ceiling, so reading half the bytes buys
//! nothing. But that is one regime. Byte-saving only pays when bandwidth is the
//! binding constraint, and two things move the operating point toward that:
//!
//!   1. THREADING — 4 cores stream ~4x the bytes/sec. If the ceiling binds
//!      first, Q4_K (half the bytes) wins; if per-core compute binds, it does
//!      not, and threading changes nothing.
//!   2. BATCHING (prefill) — weights stream once per batch, MACs scale with
//!      batch. Unpack amortizes across the batch, so a compute-bound kernel can
//!      turn bandwidth-bound as batch grows.
//!
//! This measures GMAC/s AND GB/s for Q4_K vs Q8_0 across thread count and batch
//! size, so the crossover (if any) is a number, not a guess.
//!
//! Run: cargo bench -p glproc --bench q4k_regime

use std::time::Instant;

use glproc::kernels::qdot::q8_k::Q8KActivation;
use glproc::kernels::qdot::QuantizedActivation;
use glproc::simd_strategy::SimdStrategy;
use glproc::threading::{par_matmul_q4k, par_matmul_qdot, par_matvec_q4k, par_matvec_qdot, ThreadPool};

fn prng(seed: &mut u64) -> u8 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (*seed >> 33) as u8
}
fn prng_f32(seed: &mut u64) -> f32 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*seed >> 33) as f32 / (1u64 << 31) as f32) - 1.0
}
fn half_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
    let mant = ((bits >> 13) & 0x3FF) as u16;
    if exp <= 0 { return sign; }
    sign | ((exp as u16) << 10) | mant
}
fn q4k_rows(out_dim: usize, in_dim: usize, seed: &mut u64) -> Vec<u8> {
    let nb = in_dim / 256;
    let mut v = Vec::with_capacity(out_dim * nb * 144);
    for _ in 0..out_dim * nb {
        v.extend_from_slice(&half_bits(0.02).to_le_bytes());
        v.extend_from_slice(&half_bits(0.01).to_le_bytes());
        for _ in 0..12 { v.push(prng(seed)); }
        for _ in 0..128 { v.push(prng(seed)); }
    }
    v
}
fn q8_rows(out_dim: usize, in_dim: usize, seed: &mut u64) -> Vec<u8> {
    let nb = in_dim / 32;
    let mut v = Vec::with_capacity(out_dim * nb * 34);
    for _ in 0..out_dim * nb {
        v.extend_from_slice(&half_bits(0.02).to_le_bytes());
        for _ in 0..32 { v.push(prng(seed) as i8 as u8); }
    }
    v
}

fn main() {
    let strategy = SimdStrategy::detect();
    println!("simd {strategy:?} | vnni256 {}\n", glproc::kernels::qdot::has_vnni_256());

    // gate_up shape. Rotate 8 weight copies so weights stream DRAM-cold like a
    // real chunk (a single 5-13 MB matrix would go L3-warm and hide the point).
    let (out_dim, in_dim) = (8960usize, 1536usize);
    let mut seed = 0x4E61u64;
    let copies4: Vec<Vec<u8>> = (0..8).map(|_| q4k_rows(out_dim, in_dim, &mut seed)).collect();
    let copies8: Vec<Vec<u8>> = (0..8).map(|_| q8_rows(out_dim, in_dim, &mut seed)).collect();
    let rb4 = in_dim / 256 * 144;
    let rb8 = in_dim / 32 * 34;
    let bytes4 = (out_dim * rb4) as f64;
    let bytes8 = (out_dim * rb8) as f64;

    // ================= REGIME 1: threading (decode, batch 1) =================
    println!("=== decode (batch 1) vs thread count ===");
    println!("{:<8}{:>22}{:>22}{:>10}", "threads", "Q4_K (GMAC/s, GB/s)", "Q8_0 (GMAC/s, GB/s)", "Q4K/Q8");
    for &nt in &[1usize, 2, 3, 4] {
        let pool = ThreadPool::new(nt);
        let x: Vec<f32> = (0..in_dim).map(|_| prng_f32(&mut seed)).collect();
        let mut ak = Q8KActivation::with_capacity(in_dim);
        ak.quantize(&x);
        let mut a8 = QuantizedActivation::with_capacity(in_dim);
        a8.quantize(&x);
        let mut y = vec![0f32; out_dim];

        let macs = (out_dim * in_dim) as f64;
        let time = |f: &mut dyn FnMut(usize)| {
            for i in 0..4 { f(i); }
            let it = 24;
            let t = Instant::now();
            for i in 0..it { f(i % 8); }
            t.elapsed().as_secs_f64() / it as f64
        };
        let e4 = time(&mut |i| par_matvec_q4k(&pool, &copies4[i], &x, &mut y, out_dim, in_dim, strategy));
        let e8 = time(&mut |i| par_matvec_qdot(&pool, glproc::kernels::bridge::QuantFormat::Q8_0, &copies8[i], &a8, &mut y, out_dim, in_dim, strategy));
        println!(
            "{nt:<8}{:>13.1}, {:>5.1}{:>13.1}, {:>5.1}{:>9.2}x",
            macs/e4/1e9, bytes4/e4/1e9, macs/e8/1e9, bytes8/e8/1e9, e8/e4
        );
    }

    // ================= REGIME 2: batching (prefill), 4 threads ==============
    println!("\n=== prefill (4 threads) vs batch size ===");
    println!("{:<8}{:>22}{:>22}{:>10}", "batch", "Q4_K (GMAC/s, GB/s)", "Q8_0 (GMAC/s, GB/s)", "Q4K/Q8");
    let pool = ThreadPool::new(4);
    for &batch in &[1usize, 4, 16, 32] {
        let xb: Vec<f32> = (0..batch * in_dim).map(|_| prng_f32(&mut seed)).collect();
        let a8: Vec<QuantizedActivation> = (0..batch).map(|b| {
            let mut a = QuantizedActivation::with_capacity(in_dim);
            a.quantize(&xb[b*in_dim..(b+1)*in_dim]);
            a
        }).collect();
        let mut y = vec![0f32; batch * out_dim];

        let macs = (out_dim * in_dim * batch) as f64;
        let time = |f: &mut dyn FnMut(usize)| {
            for i in 0..2 { f(i); }
            let it = 12;
            let t = Instant::now();
            for i in 0..it { f(i % 8); }
            t.elapsed().as_secs_f64() / it as f64
        };
        let e4 = time(&mut |i| par_matmul_q4k(&pool, &copies4[i], &xb, in_dim, &mut y, out_dim, 0, out_dim, in_dim, batch, strategy));
        let e8 = time(&mut |i| par_matmul_qdot(&pool, glproc::kernels::bridge::QuantFormat::Q8_0, &copies8[i], &a8, &mut y, out_dim, 0, out_dim, in_dim, strategy));
        println!(
            "{batch:<8}{:>13.1}, {:>5.1}{:>13.1}, {:>5.1}{:>9.2}x",
            macs/e4/1e9, bytes4/e4/1e9, macs/e8/1e9, bytes8/e8/1e9, e8/e4
        );
    }

    println!("\nReading:");
    println!("  Q4K/Q8 > 1.0 means Q4_K is FASTER — byte-saving materialized.");
    println!("  Watch Q4_K's GB/s: if it climbs toward the ~29 GB/s ceiling as threads/batch");
    println!("  rise, it is becoming bandwidth-bound and the half-byte read starts to pay.");
    println!("  If GB/s stays flat and low, unpack compute is the wall regardless of regime.");
}
