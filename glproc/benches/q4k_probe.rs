//! Q4_K vs Q8_0 integer-dot: where does the time actually go?
//!
//! Production A/B says the native Q4_K path is 1.5x SLOWER end-to-end than
//! repacking to Q8_0 (9.5 vs 14.1 tok/s), despite reading 1.89x fewer bytes.
//! Two "obviously right" kernel fixes (vector-domain accumulation, hoisted
//! scale_min) recovered only ~9%. So the cost is somewhere else.
//!
//! This isolates the kernels on identical work and reports GMAC/s and GB/s, so
//! the answer comes from a measurement rather than from reading the code.
//!
//! Run: cargo bench -p glproc --bench q4k_probe

use std::time::Instant;

use glproc::kernels::bridge::QuantFormat;
use glproc::kernels::qdot::{row_dot_q8, QuantizedActivation};
use glproc::simd_strategy::SimdStrategy;

fn prng(seed: &mut u64) -> u8 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*seed >> 33) as u8
}

fn prng_f32(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*seed >> 33) as f32 / (1u64 << 31) as f32) - 1.0
}

fn half_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
    let mant = ((bits >> 13) & 0x3FF) as u16;
    if exp <= 0 {
        return sign;
    }
    sign | ((exp as u16) << 10) | mant
}

/// `out_dim` rows of Q4_K, `in_dim` weights each.
fn q4k_rows(out_dim: usize, in_dim: usize, seed: &mut u64) -> Vec<u8> {
    let nb = in_dim / 256;
    let mut v = Vec::with_capacity(out_dim * nb * 144);
    for _ in 0..out_dim * nb {
        v.extend_from_slice(&half_bits(0.02).to_le_bytes());
        v.extend_from_slice(&half_bits(0.01).to_le_bytes());
        for _ in 0..12 {
            v.push(prng(seed));
        }
        for _ in 0..128 {
            v.push(prng(seed));
        }
    }
    v
}

/// `out_dim` rows of Q8_0, `in_dim` weights each.
fn q8_rows(out_dim: usize, in_dim: usize, seed: &mut u64) -> Vec<u8> {
    let nb = in_dim / 32;
    let mut v = Vec::with_capacity(out_dim * nb * 34);
    for _ in 0..out_dim * nb {
        v.extend_from_slice(&half_bits(0.02).to_le_bytes());
        for _ in 0..32 {
            v.push(prng(seed));
        }
    }
    v
}

fn main() {
    let strategy = SimdStrategy::detect();
    println!(
        "simd {strategy:?} | vnni256 {}",
        glproc::kernels::qdot::has_vnni_256()
    );
    println!("\nQwen2.5-1.5B gate_up shape: out_dim x in_dim per matvec\n");

    // The real FFN gate_up shape, and a small L2-resident one to separate the
    // memory effect from the kernel effect.
    for (out_dim, in_dim, label) in [
        (8960usize, 1536usize, "gate_up (DRAM-cold)"),
        (256, 1536, "small (L2-warm)"),
    ] {
        let mut seed = 0x4B4Bu64;

        let w4 = q4k_rows(out_dim, in_dim, &mut seed);
        let w8 = q8_rows(out_dim, in_dim, &mut seed);
        let x: Vec<f32> = (0..in_dim).map(|_| prng_f32(&mut seed)).collect();
        let mut act = QuantizedActivation::with_capacity(in_dim);
        act.quantize(&x);

        let rb4 = in_dim / 256 * 144;
        let rb8 = in_dim / 32 * 34;

        println!("--- {label}: {out_dim} x {in_dim} ---");
        println!(
            "{:<10} {:>10} {:>10} {:>10} {:>10}",
            "format", "MB", "ms", "GMAC/s", "GB/s"
        );

        let iters = if out_dim > 1000 { 40 } else { 2000 };
        for (name, w, rb, fmt) in [
            ("Q4_K", &w4, rb4, QuantFormat::Q4K),
            ("Q8_0", &w8, rb8, QuantFormat::Q8_0),
        ] {
            let run = || {
                let mut s = 0f32;
                for o in 0..out_dim {
                    s += row_dot_q8(fmt, &w[o * rb..(o + 1) * rb], &act, strategy);
                }
                s
            };
            let mut sink = 0f32;
            for _ in 0..3 {
                sink += run();
            }
            let t = Instant::now();
            for _ in 0..iters {
                sink += run();
            }
            let el = t.elapsed().as_secs_f64() / iters as f64;
            std::hint::black_box(sink);

            let bytes = (out_dim * rb) as f64;
            let macs = (out_dim * in_dim) as f64;
            println!(
                "{name:<10} {:>10.1} {:>10.3} {:>10.1} {:>10.1}",
                bytes / 1e6,
                el * 1e3,
                macs / el / 1e9,
                bytes / el / 1e9
            );
        }
        println!();
    }

    println!("Reading:");
    println!("  If Q4_K's GB/s is far BELOW Q8_0's, the kernel is compute-bound —");
    println!("  the unpack costs more than the bytes it saves.");
    println!("  If Q4_K's GMAC/s matches Q8_0's, the kernel is fine and the");
    println!("  regression is elsewhere (e.g. prefill, or a non-FFN path).");
}
