//! VNNI-256 (production) vs VNNI-512 (probe-only) on the Q8_0 qdot kernel —
//! the format the whole FFN hot path runs through (post-GATE, Q8_0 repack
//! always wins on this machine; see `benchmarks/post-gate-benchmark-report.md`
//! and `full-bottleneck-e2e.json`, which classify decode as `compute_bound`
//! at 57% of the bandwidth ceiling and prefill's FFN buckets as
//! `not-bandwidth-bound` at only 5-6% of ceiling despite 44-67 GMAC/s).
//!
//! `gl-agent-skills/cpu-skills/rejected-optimizations.md` entry 3 closed
//! "at least use AVX-512VNNI-512" for thermal/downclock reasons, without a
//! kernel-level A/B to back that verdict — the entry documents a policy
//! decision (`ArchGLLM_X5.md`'s "AVX2 only, NO AVX-512F" stability rule),
//! not a measured comparison of the two VNNI widths against each other.
//! This probe is that measurement, run under JinXSuper's explicit override
//! to revisit the entry (see this session's conversation).
//!
//! Kept SHORT deliberately: the rejected-optimizations entry cites *thermal*
//! risk on this machine's 28W AIO envelope, not just a throughput question.
//! This probe runs a few hundred ms of AVX-512 execution total, not a
//! sustained stress test — enough to reach steady-state GMAC/s (any
//! frequency-license transition settles in low-single-digit milliseconds),
//! not enough to be a thermal endurance test. If the numbers below justify
//! further investigation, a longer thermal-monitored run is a separate,
//! separately-approved step.
//!
//! Run: cargo bench -p glproc --bench vnni512_probe

#[path = "common/mod.rs"]
mod common;
use common::{default_iters, prng_f32, print_table_header, q8_rows, time_kernel};

use glproc::kernels::qdot::q8_0::{scalar, vnni, vnni512};
use glproc::kernels::qdot::QuantizedActivation;

fn main() {
    let has_vnni512 = std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512bw")
        && std::arch::is_x86_feature_detected!("avx512vnni");
    let has_vnni256 = glproc::kernels::qdot::has_vnni_256();
    println!("vnni256 available: {has_vnni256} | vnni512 available: {has_vnni512}");
    if !has_vnni512 {
        println!("This CPU lacks AVX-512VNNI — nothing to probe. Exiting.");
        return;
    }

    // Real FFN gate_up/down shapes from this session's own benchmark
    // (Qwen2.5-0.5B: in_dim=896), plus a small L2-resident shape to isolate
    // the kernel effect from any memory effect.
    for (out_dim, in_dim, label) in [
        (4864usize, 896usize, "ffn gate_up shape (DRAM-cold)"),
        (256, 896, "small (L2-warm)"),
    ] {
        let mut seed = 0x51DEu64;
        let w = q8_rows(out_dim, in_dim, &mut seed);
        let x: Vec<f32> = (0..in_dim).map(|_| prng_f32(&mut seed)).collect();
        let mut act = QuantizedActivation::with_capacity(in_dim);
        act.quantize(&x);
        let rb = in_dim / 32 * 34;

        println!("\n--- {label}: {out_dim} x {in_dim} ---");
        print_table_header();

        let macs = (out_dim * in_dim) as f64;
        let bytes = (out_dim * rb) as f64;
        let iters = default_iters(out_dim);

        time_kernel(iters, macs, bytes, &|| {
            let mut s = 0f32;
            for o in 0..out_dim {
                s += scalar::row_dot(&w[o * rb..(o + 1) * rb], &act);
            }
            s
        })
        .print_row("scalar");

        time_kernel(iters, macs, bytes, &|| {
            let mut s = 0f32;
            for o in 0..out_dim {
                // SAFETY: has_vnni_256() checked in main().
                s += unsafe { vnni::row_dot(&w[o * rb..(o + 1) * rb], &act) };
            }
            s
        })
        .print_row("vnni256");

        time_kernel(iters, macs, bytes, &|| {
            let mut s = 0f32;
            for o in 0..out_dim {
                // SAFETY: has_vnni512 checked in main().
                s += unsafe { vnni512::row_dot(&w[o * rb..(o + 1) * rb], &act) };
            }
            s
        })
        .print_row("vnni512");
    }

    println!("\nReading:");
    println!("  If vnni512's GMAC/s is meaningfully above vnni256's (not just noise),");
    println!("  the width buys real throughput on this CPU and rejected-optimizations.md");
    println!("  entry 3 is due a documented update. If it's flat or worse, entry 3 stands");
    println!("  confirmed at the kernel level, not just by policy.");
}
