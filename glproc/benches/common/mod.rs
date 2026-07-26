//! Shared harness for isolated kernel probes (`benches/*_probe.rs`).
//!
//! Every probe in this project needs the same three things: a deterministic
//! PRNG (no external `rand` dep — glproc is zero-dep), synthetic Q8_0/f32
//! data generators shaped like real model tensors, and a warmup-then-time
//! loop that reports GMAC/s and GB/s consistently. Before this module, each
//! probe (`vnni512_probe.rs`, `row_tile_probe.rs`, `rope_probe.rs`,
//! `elementwise_add_probe.rs`) redefined all of it from scratch — four
//! near-identical copies of `prng`/`prng_f32`/`half_bits`/the timing loop.
//! Include via `#[path = "common/mod.rs"] mod common;` at the top of a new
//! probe; `benches/*` binaries can't share code through the crate's public
//! API, so this is the standard Cargo pattern for cross-bench-binary sharing.
//!
//! This module does not change any probe's *methodology* — it only removes
//! the copy-paste. Every probe still stands alone as its own binary and
//! still requires the full production A/B before its result is trusted (see
//! `gl-agent-skills/bench-skills/measurement-discipline.md`).

use std::time::Instant;

/// Deterministic PCG-ish PRNG — one `u8` per call. No external `rand` dep
/// (glproc is zero-dep); this needs to be reproducible across runs, not
/// cryptographically random.
pub fn prng(seed: &mut u64) -> u8 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (*seed >> 33) as u8
}

/// Same generator, mapped to `[-1.0, 1.0)` `f32` — for synthetic activations.
pub fn prng_f32(seed: &mut u64) -> f32 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*seed >> 33) as f32 / (1u64 << 31) as f32) - 1.0
}

/// `f32` -> IEEE754 half-precision bits, for building Q8_0 block headers
/// (each block's 2-byte scale) without pulling in a half-float crate.
pub fn half_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
    let mant = ((bits >> 13) & 0x3FF) as u16;
    if exp <= 0 {
        return sign;
    }
    sign | ((exp as u16) << 10) | mant
}

/// `out_dim` rows of Q8_0 (`in_dim / 32` blocks of 34 bytes each: 2-byte f16
/// scale + 32 int8 quants), pseudo-random weights. Matches the on-disk
/// layout `kernels::qdot::q8_0`'s kernels expect.
pub fn q8_rows(out_dim: usize, in_dim: usize, seed: &mut u64) -> Vec<u8> {
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

/// One Q8_0 row (see [`q8_rows`] for the layout), `n_blocks` blocks.
pub fn q8_row(n_blocks: usize, seed: &mut u64) -> Vec<u8> {
    q8_rows(1, n_blocks * 32, seed)
}

/// Warmup 3 iterations (unmeasured), then time `iters` more, reporting
/// mean ms/call alongside derived GMAC/s and GB/s for a `macs`-MAC,
/// `bytes`-byte-per-call workload. `f` must return something to
/// `black_box` — every probe's inner closure accumulates a dummy sum so
/// the optimizer cannot elide the call entirely.
///
/// This is the one piece of methodology this module *does* encode — every
/// probe in this project used this exact warmup-then-measure shape by hand;
/// factoring it here means a new probe gets it right by construction instead
/// of by copying an existing file carefully.
pub fn time_kernel(iters: usize, macs: f64, bytes: f64, f: &dyn Fn() -> f32) -> KernelTiming {
    let mut sink = 0f32;
    for _ in 0..3 {
        sink += f();
    }
    let t = Instant::now();
    for _ in 0..iters {
        sink += f();
    }
    let el = t.elapsed().as_secs_f64() / iters as f64;
    std::hint::black_box(sink);
    KernelTiming { ms: el * 1e3, gmac_per_s: macs / el / 1e9, gb_per_s: bytes / el / 1e9 }
}

pub struct KernelTiming {
    pub ms: f64,
    pub gmac_per_s: f64,
    pub gb_per_s: f64,
}

impl KernelTiming {
    /// One row of the standard probe results table: `name  ms  GMAC/s  GB/s`.
    pub fn print_row(&self, name: &str) {
        println!("{name:<18} {:>10.4} {:>10.1} {:>10.1}", self.ms, self.gmac_per_s, self.gb_per_s);
    }
}

/// Print the standard probe results table header — pair with
/// [`KernelTiming::print_row`] for every kernel under comparison.
pub fn print_table_header() {
    println!("{:<18} {:>10} {:>10} {:>10}", "kernel", "ms", "GMAC/s", "GB/s");
}

/// Iteration count heuristic used by every probe so far: fewer iterations
/// for large/DRAM-cold shapes (each call is already slow enough to time
/// accurately), more for small/L2-warm ones (each call is fast, needs more
/// samples to average out timer granularity).
pub fn default_iters(out_dim: usize) -> usize {
    if out_dim > 1000 {
        60
    } else {
        2000
    }
}
