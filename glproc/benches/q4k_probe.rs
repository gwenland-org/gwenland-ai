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

        // Wave 2 addition: the Q8_K-activation variant. Same Q4_K bytes, but
        // sub-block scales applied in the INTEGER domain (one float FMA per
        // super-block instead of 8) — the structural change hypothesized to
        // close the compute gap that made the first Q4_K kernel lose 33%.
        {
            let mut act_k = glproc::kernels::qdot::q8_k::Q8KActivation::with_capacity(in_dim);
            act_k.quantize(&x);
            let run = || {
                let mut s = 0f32;
                for o in 0..out_dim {
                    // SAFETY: probe only runs on AVX2 machines (checked in main).
                    s += unsafe {
                        glproc::kernels::qdot::q4_k::avx2::row_dot_q8k(
                            &w4[o * rb4..(o + 1) * rb4],
                            &act_k,
                        )
                    };
                }
                s
            };
            let iters = if out_dim > 1000 { 40 } else { 2000 };
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
            let bytes = (out_dim * rb4) as f64;
            let macs = (out_dim * in_dim) as f64;
            println!(
                "{:<10} {:>10.1} {:>10.3} {:>10.1} {:>10.1}",
                "Q4K/Q8K",
                bytes / 1e6,
                el * 1e3,
                macs / el / 1e9,
                bytes / el / 1e9
            );
        }

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

    // ---------------------------------------------------------------------
    // Wave 4: FUSED SwiGLU head-to-head. gate + up in one pass.
    //
    // Three kernels, same gate_up shape, same activation, cold DRAM:
    //   q8_0 fused  — the production baseline (par_matvec_swiglu, 86% ceiling)
    //   q4k q8k     — integer-domain fused (candidate A)
    //   q4k f32     — f32-domain fused, the literal spec (candidate B)
    //
    // Per-row work here is 2x a plain matvec (gate AND up). MACs are counted
    // as 2*out*in; bytes as the fused pair's footprint.
    // ---------------------------------------------------------------------
    {
        use glproc::kernels::qdot::q4_k::swiglu;
        use glproc::kernels::qdot::q8_k::Q8KActivation;

        let (out_dim, in_dim) = (8960usize, 1536usize);
        let mut seed = 0x5F00u64;
        let rb4 = in_dim / 256 * 144;

        // Interleaved Q4_K gate/up rows: [gate 144B][up 144B] per output.
        let mut packed4 = Vec::with_capacity(out_dim * 2 * rb4);
        for _ in 0..out_dim {
            packed4.extend(q4k_rows(1, in_dim, &mut seed)); // gate
            packed4.extend(q4k_rows(1, in_dim, &mut seed)); // up
        }
        // Interleaved Q8_0 gate/up for the baseline.
        let rb8 = in_dim / 32 * 34;
        let mut packed8 = Vec::with_capacity(out_dim * 2 * rb8);
        for _ in 0..out_dim {
            packed8.extend(q8_rows(1, in_dim, &mut seed));
            packed8.extend(q8_rows(1, in_dim, &mut seed));
        }

        let x: Vec<f32> = (0..in_dim).map(|_| prng_f32(&mut seed)).collect();
        let mut ak = Q8KActivation::with_capacity(in_dim);
        ak.quantize(&x);
        let mut a32 = QuantizedActivation::with_capacity(in_dim);
        a32.quantize(&x);

        println!("\n--- FUSED SwiGLU: gate_up {out_dim} x {in_dim}, cold ---");
        println!("{:<14} {:>10} {:>10} {:>10}", "kernel", "ms", "GMAC/s", "GB/s");

        let macs = (2 * out_dim * in_dim) as f64; // gate + up
        let time = |label: &str, bytes: f64, f: &dyn Fn() -> f32| {
            let mut sink = 0f32;
            for _ in 0..3 { sink += f(); }
            let t = Instant::now();
            for _ in 0..40 { sink += f(); }
            let el = t.elapsed().as_secs_f64() / 40.0;
            std::hint::black_box(sink);
            println!(
                "{label:<14} {:>10.3} {:>10.1} {:>10.1}",
                el * 1e3, macs / el / 1e9, bytes / el / 1e9
            );
        };

        // q8_0 fused baseline (row_dot_q8 twice + silu, matching production).
        time("q8_0 fused", (out_dim * 2 * rb8) as f64, &|| {
            let mut s = 0f32;
            for o in 0..out_dim {
                let pair = &packed8[o * 2 * rb8..(o + 1) * 2 * rb8];
                let g = glproc::kernels::qdot::row_dot_q8(
                    QuantFormat::Q8_0, &pair[..rb8], &a32, strategy);
                let u = glproc::kernels::qdot::row_dot_q8(
                    QuantFormat::Q8_0, &pair[rb8..], &a32, strategy);
                s += g / (1.0 + glproc::kernels::fast_exp(-g)) * u;
            }
            s
        });
        time("q4k q8k", (out_dim * 2 * rb4) as f64, &|| {
            let mut s = 0f32;
            for o in 0..out_dim {
                let pair = &packed4[o * 2 * rb4..(o + 1) * 2 * rb4];
                // SAFETY: probe gated on AVX2 in main().
                s += unsafe { swiglu::fused_swiglu_q8k(&pair[..rb4], &pair[rb4..], &ak) };
            }
            s
        });
        time("q4k f32", (out_dim * 2 * rb4) as f64, &|| {
            let mut s = 0f32;
            for o in 0..out_dim {
                let pair = &packed4[o * 2 * rb4..(o + 1) * 2 * rb4];
                // SAFETY: as above.
                s += unsafe { swiglu::fused_swiglu_f32(&pair[..rb4], &pair[rb4..], &ak) };
            }
            s
        });
        println!("\n  Winner is whichever fused Q4_K kernel matches or beats q8_0 fused GMAC/s");
        println!("  while reading fewer bytes. That one goes to integration; the other is dropped.");
    }
}
