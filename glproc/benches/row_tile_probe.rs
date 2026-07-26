//! Sequential row_dot (current decode dispatch) vs row-tiled dot (8 rows,
//! shared activation, 8 independent accumulator chains) — testing the axis
//! `architecture/percival/CPU/ARTX02-IceLake.md` Finding F05 says llama.cpp
//! actually wins on (many independent chains across OUTPUT ROWS), as
//! opposed to `vnni512_probe.rs`'s axis (wider instruction, one row at a
//! time), which came up neutral in production — see
//! `gl-agent-skills/cpu-skills/rejected-optimizations.md` entry 3.
//!
//! Decode-representative: ONE shared activation (single token), out_dim
//! rows — decode's `row_dot_q8` dispatch has zero cross-row ILP today,
//! unlike prefill's `row_dot_q8_packed8` (which already tiles on the
//! activation axis). If row-tiling wins here, it targets exactly the gap
//! prefill's packed8 doesn't cover.
//!
//! Run: cargo bench -p glproc --bench row_tile_probe

#[path = "common/mod.rs"]
mod common;
use common::{default_iters, prng_f32, print_table_header, q8_rows, time_kernel};

use glproc::kernels::qdot::q8_0::{row_tile, scalar, vnni};
use glproc::kernels::qdot::QuantizedActivation;

const R: usize = 8;

fn main() {
    let has_vnni256 = glproc::kernels::qdot::has_vnni_256();
    println!("vnni256 available: {has_vnni256}");
    if !has_vnni256 {
        println!("This CPU lacks AVX-512VNNI (256-bit) — nothing to probe. Exiting.");
        return;
    }

    for (out_dim, in_dim, label) in [
        (4864usize, 896usize, "ffn gate_up shape, single token (DRAM-cold)"),
        (256, 896, "small (L2-warm)"),
    ] {
        let mut seed = 0x717Eu64;
        let w = q8_rows(out_dim, in_dim, &mut seed);
        let x: Vec<f32> = (0..in_dim).map(|_| prng_f32(&mut seed)).collect();
        let mut act = QuantizedActivation::with_capacity(in_dim);
        act.quantize(&x);
        let rb = in_dim / 32 * 34;

        println!("\n--- {label}: {out_dim} x {in_dim}, single activation ---");
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
                // SAFETY: has_vnni_256 checked in main().
                s += unsafe { vnni::row_dot(&w[o * rb..(o + 1) * rb], &act) };
            }
            s
        })
        .print_row("vnni256 (current)");

        time_kernel(iters, macs, bytes, &|| {
            let mut s = 0f32;
            let mut o = 0;
            while o + R <= out_dim {
                let rows: [&[u8]; R] = std::array::from_fn(|r| &w[(o + r) * rb..(o + r + 1) * rb]);
                // SAFETY: has_vnni_256 checked in main() (same feature set
                // row_tile_dot requires).
                let dots = unsafe { row_tile::row_tile_dot::<R>(rows, &act) };
                s += dots.iter().sum::<f32>();
                o += R;
            }
            // Tail (out_dim not a multiple of R) via the sequential kernel —
            // fine for a probe; a real integration would need this too.
            while o < out_dim {
                s += unsafe { vnni::row_dot(&w[o * rb..(o + 1) * rb], &act) };
                o += 1;
            }
            s
        })
        .print_row("row_tile R=8 (new)");
    }

    println!("\nReading:");
    println!("  If row_tile R=8's GMAC/s meaningfully beats vnni256 (current decode dispatch),");
    println!("  cross-row ILP is a real lever for decode and worth production-A/B'ing next.");
    println!("  If it's flat, decode's single-activation path is bottlenecked by something");
    println!("  other than dependency-chain latency too (memory, per-call overhead, etc.).");
}
