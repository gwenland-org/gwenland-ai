//! Measured cost of one GLBitProf pass (Gate 2 deliverable, D-13).
//!
//! Run with:
//!
//! ```text
//! cargo run --release -p glbench --example bench_bitprof
//! ```
//!
//! An example rather than a `benches/` entry because glbench takes **zero**
//! external dependencies (`glbench/DESIGN.md` §9) and `criterion` is not in
//! `[dev-dependencies]`. `std::time::Instant` with an explicit warmup and a
//! repeat count is what is left, and it is sufficient for a per-element cost
//! that differs by orders of magnitude between the two paths being separated.
//!
//! # What is being separated, and why not one number
//!
//! D-13 asks for **two** numbers, not an average:
//!
//! - **Tier 1+3** — the exponent histogram and the 32 per-position bit counts.
//!   Every tensor pays this, always.
//! - **Tier 2** — the sparse mantissa map, paid only by tensors at or under
//!   [`MANTISSA_SPARSE_CAP`] elements.
//!
//! Averaging them would hide which one a given model actually incurs, and the
//! answer differs per tensor: a 4096×4096 weight matrix never pays Tier 2,
//! while a LoRA `B` matrix always does.
//!
//! Both data shapes are measured at every size, because they are not the same
//! workload. Random f32 has near-uniform mantissa bits — the worst case, and
//! the one the entropy literature says real FP32 weights approach. Clustered
//! (normal-ish) data has far fewer distinct mantissa patterns, so its hash map
//! stays small and cache-resident.
//!
//! ## MEASURED — 2026-08-20, i3-1115G4 (2p/4l, 3.00 GHz), Windows, AC power
//! ## release profile, 3 warmup + 10 measured repeats per cell, median
//!
//! ```text
//! case               elements   tier2?    random ns    ns/elem  clustered ns   ns/elem
//! LoRA B matrix          1024      yes        63900     62.402         63800    62.305
//! LoRA A matrix         65536      yes      6566700    100.200       6775900   103.392
//! weight matrix       1048576       no     28265600     26.956      29618200    28.246
//! 4096x4096 full     16777216       no    446656100     26.623     453309700    27.019
//!
//! tier separation at the cap boundary (same data shape, +/-1 element):
//!   tier 1+3 only (cap+1): 3516900 ns = 26.832 ns/elem
//!   tier 1+2+3   (cap)   : 16158100 ns = 123.277 ns/elem   -> tier 2 costs 4.59x
//! ```
//!
//! **The two numbers, per D-13.** Tier 1+3 — what every tensor pays — is
//! **~27 ns/element**, flat from 1M to 16M elements. Tier 2, paid only under
//! the cap, multiplies that by **4.6x** to ~123 ns/element. Averaging them
//! would have produced ~75 ns/element, a figure no tensor actually incurs.
//!
//! **Extrapolation, labelled as such.** At 27 ns/element a full 0.5B-parameter
//! model profiles in ~13 s and a 7B model in ~3 min. Neither was run; this is
//! the per-element number multiplied out, not a measurement.
//!
//! **The 65,536 cell paid for itself.** It first measured *worse* per element
//! (149.5 ns) than a 16M-element tensor, which made no sense until the cause
//! was obvious: the mantissa map started at 4096 entries and rehashed its way
//! up. Pre-sizing it to the element count took that cell to 100.2 ns/elem and
//! the cap-boundary Tier 2 cost from 174.4 to 123.3 ns/elem. A benchmark that
//! only reported the two extremes would have missed it entirely.
//!
//! **D-12's premise, now measured rather than cited.** 131,072 near-uniform
//! elements produce 128,132 distinct mantissa patterns against a birthday-bound
//! prediction of 130,053 — ratio 0.985. The occupancy formula the cap is
//! derived from holds on this data to within 1.5%.
//!
//! Per `bench-skills/measurement-discipline.md` rule 1, these are probe numbers
//! describing this function in isolation. They bound the cost of a profiling
//! pass; they are not a claim about any end-to-end `glbench run`.

use std::time::Instant;

use glbench::numerical::bitprof::{profile, MANTISSA_SPARSE_CAP};

/// Untimed passes before measurement, to settle caches and the branch
/// predictor.
const WARMUP: usize = 3;

/// Timed passes per cell. The work is deterministic and allocation-dominated
/// at small sizes, so a modest count is enough; the median is reported.
const REPEATS: usize = 10;

/// Deterministic PRNG (xorshift64*) — a fixed seed keeps the generated data
/// identical across runs, so two runs of this example are comparable.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in [0, 1).
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32
    }
}

/// Near-uniform mantissa bits: the worst case for the sparse map, and what the
/// compression literature says real FP32 weights approach.
fn random_data(n: usize) -> Vec<f32> {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect()
}

/// Clustered data: a sum of uniforms approximating a normal, which is roughly
/// how a trained weight tensor is distributed.
fn clustered_data(n: usize) -> Vec<f32> {
    let mut rng = Rng(0x243F_6A88_85A3_08D3);
    (0..n)
        .map(|_| {
            let s: f32 = (0..6).map(|_| rng.next_f32()).sum();
            (s - 3.0) * 0.1
        })
        .collect()
}

/// Median wall-clock nanoseconds for one `profile()` call over `values`.
fn median_ns(values: &[f32]) -> u128 {
    for _ in 0..WARMUP {
        std::hint::black_box(profile(std::hint::black_box(values)));
    }
    let mut samples = Vec::with_capacity(REPEATS);
    for _ in 0..REPEATS {
        let t0 = Instant::now();
        let p = profile(std::hint::black_box(values));
        let elapsed = t0.elapsed().as_nanos();
        std::hint::black_box(p);
        samples.push(elapsed);
    }
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn main() {
    let cap = MANTISSA_SPARSE_CAP as usize;

    // The four sizes D-13 asks about, each named for the real case it stands
    // for. The cap sits between the second and third, which is the point.
    let cases: [(&str, usize); 4] = [
        ("LoRA B matrix", 1_024),
        ("LoRA A matrix", 65_536),
        ("weight matrix", 1_048_576),
        ("4096x4096 full", 4096 * 4096),
    ];

    println!("GLBitProf cost — {WARMUP} warmup + {REPEATS} measured, median reported");
    println!("mantissa sparse cap: {cap} elements\n");
    println!(
        "{:<16} {:>10} {:>8} {:>12} {:>10} {:>12} {:>10}",
        "case", "elements", "tier2?", "random ns", "ns/elem", "clustered ns", "ns/elem"
    );
    println!("{}", "-".repeat(84));

    for (name, n) in cases {
        let tier2 = if n <= cap { "yes" } else { "no" };

        let random = random_data(n);
        let random_ns = median_ns(&random);
        drop(random);

        let clustered = clustered_data(n);
        let clustered_ns = median_ns(&clustered);
        drop(clustered);

        println!(
            "{:<16} {:>10} {:>8} {:>12} {:>10.3} {:>12} {:>10.3}",
            name,
            n,
            tier2,
            random_ns,
            random_ns as f64 / n as f64,
            clustered_ns,
            clustered_ns as f64 / n as f64
        );
    }

    // The two numbers D-13 wants separated, measured at one size on both sides
    // of the cap so the difference is the tier and not the length.
    println!("\ntier separation at the cap boundary (same data shape, ±1 element):");
    let at_cap = random_data(cap);
    let over_cap = random_data(cap + 1);
    let with_tier2 = median_ns(&at_cap);
    let without_tier2 = median_ns(&over_cap);
    println!(
        "  tier 1+3 only (cap+1 elements): {:>10} ns  = {:.3} ns/elem",
        without_tier2,
        without_tier2 as f64 / (cap + 1) as f64
    );
    println!(
        "  tier 1+2+3    (cap   elements): {:>10} ns  = {:.3} ns/elem",
        with_tier2,
        with_tier2 as f64 / cap as f64
    );
    let ratio = with_tier2 as f64 / without_tier2.max(1) as f64;
    println!("  tier 2 multiplies the pass by {ratio:.2}x");

    // A sanity check on the cap's own premise (D-12): how full does the map
    // actually get on near-uniform data? If real data is far less uniform than
    // the literature implies, the threshold is worth revisiting with this
    // number in hand.
    let p = profile(&at_cap);
    if let Some(map) = &p.mantissa_sparse {
        let distinct = map.len();
        let predicted = 8_388_608.0 * (1.0 - (-(cap as f64) / 8_388_608.0).exp());
        println!(
            "\nmantissa occupancy at {cap} random elements: {distinct} distinct patterns \
             (birthday-bound prediction {predicted:.0}, ratio {:.3})",
            distinct as f64 / predicted
        );
    }
}
