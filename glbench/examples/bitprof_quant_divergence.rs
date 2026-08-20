//! Does bit-profile divergence tell a quantisation scheme apart from a bug?
//!
//! Run with:
//!
//! ```text
//! cargo run --release -p glbench --features gllm-bench --example bitprof_quant_divergence
//! ```
//!
//! # What this is, and what it is not
//!
//! Gate 2 asks for the divergence to be run against the **GQ4A vs Q6_K** case
//! on real archives. It is not run here, and cannot be on this machine: no
//! `.gllm` package and no `.gguf` file exists in the tree (`*.gguf` is
//! gitignored, per `rust-skills/testing-standards.md` rule 7), and there is no
//! Q6_K *encoder* anywhere in the workspace — Q6_K data only ever arrives from
//! GGUF files produced elsewhere. Grep confirms a Q6_K decoder and no encoder.
//!
//! So this runs the closest thing that is actually available, and says so:
//!
//! - **Real encoder, real kernel.** `glictus_caliburni::gquant::encode_gq4a_tensor`
//!   and `glproc::kernels::gquant::dequant_gq4a` are the production paths, not
//!   reimplementations. Only the input tensor is synthetic.
//! - **GQ4A and GQ2A** give a genuine 4-bit vs 2-bit contrast — two real
//!   schemes of different aggressiveness over the same weights.
//! - **The bug arm is faithful to its class.** `GQ4ABlock::weights` packs two
//!   4-bit codes per byte, low nibble = even index. Swapping them is exactly
//!   the defect that corrupted every `ffn_down.weight` in
//!   `notes/issues/gllm-e2e-garbage-output.md`, reproduced on the real block
//!   layout rather than imitated.
//!
//! The question being tested is the one from the research document §12 Case 2:
//! two paths both show residual against the f32 original — can the bit profile
//! say which residual is *the scheme working* and which is *a bug*?
//!
//! # ⛔ MEASURED 2026-08-20 — the answer is NO for a permutation-class bug
//!
//! i3-1115G4, Windows, release, 4096-weight synthetic tensor.
//!
//! ```text
//! baseline f32          count 4096, sign 50.5%, exponent 108..123, entropy 11.9966 b
//!
//! vs f32 original          MAE        exp L1   entropy Δ   max bit Δ
//!   GQ4A correct        2.809e-3      0.4028   -4.2886 b   -0.2793 @0
//!   GQ2A correct        9.681e-3      0.1367   -3.8327 b   -0.4497 @0
//!   GQ4A nibble-swap    4.010e-2      0.4028   -4.2886 b   -0.2793 @0
//!   GQ4A rotated        4.122e-2      0.4409   -4.2930 b   -0.2710 @0
//!
//! scheme held constant (correct GQ4A as the baseline)
//!   vs nibble-swap      4.004e-2      0.0000   +0.0000 b   +0.0000 @0   <-- INVISIBLE
//!   vs rotated          4.112e-2      0.0488   -0.0044 b   -0.0278 @22  <-- registers
//! ```
//!
//! **The nibble-swap arm is identical to the correct decode on every axis** —
//! not "small", exactly zero — while its MAE is 14x the scheme's own residual.
//! The design predicted a structured per-position anomaly; there is none.
//!
//! The cause is structural. Swapping the codes inside a byte exchanges weights
//! 2i and 2i+1, which share a 32-weight sub-block and therefore a scale, so the
//! decoded multiset is unchanged — and every statistic in a `VLBitProfile` is
//! permutation-invariant by construction.
//!
//! The **sub-block rotation arm is the control**: it moves codes across a scale
//! boundary, is not a pure permutation, and does register (exp L1 0.049). That
//! separates "the profiler is broken" from "the profiler cannot see this class
//! of defect", and it is the second one.
//!
//! Consequence, carried into `numerical::compare`'s own docs and pinned by
//! `compare::tests::permutation_invariance_is_a_known_blind_spot`: GLBitProf
//! detects defects that change the **distribution** of values and is blind to
//! defects that only change their **order**. Catching a permutation needs an
//! element-wise check against an oracle decode — `validation::parity`'s job.

use glbench::numerical::bitprof::profile;
use glbench::numerical::compare::{compare, VLBitDivergence};
use glictus_caliburni::gquant::encoder::{encode_gq2a_tensor, encode_gq4a_tensor};

/// Deterministic PRNG (xorshift64*), so two runs of this example agree.
struct Rng(u64);

impl Rng {
    fn next_f32(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        let v = x.wrapping_mul(0x2545_F491_4F6C_DD1D);
        (v >> 40) as f32 / (1u32 << 24) as f32
    }
}

/// A plausible weight tensor: roughly normal, which is what a trained matrix
/// looks like and what both schemes are tuned for.
fn weight_tensor(n: usize) -> Vec<f32> {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            let s: f32 = (0..6).map(|_| rng.next_f32()).sum();
            (s - 3.0) * 0.05
        })
        .collect()
}

/// Encode a tensor to the GQ4A byte stream a `.gllm` layer file would hold.
///
/// The byte stream, not the struct, because `GQ4ABlock` is declared
/// independently in both `glictus-caliburni` and `glproc` — the two do not
/// share a type. Bytes are the actual interchange format between them, and
/// `dequant_gq4a_stream` is the same entry point `tensor_stats`'s decode path
/// calls on a real package.
fn gq4a_bytes(weights: &[f32]) -> Vec<u8> {
    let blocks = encode_gq4a_tensor(weights).expect("tensor length is a multiple of 256");
    blocks.iter().flat_map(|b| b.as_bytes().to_vec()).collect()
}

/// GQ4A round trip through the production encoder and dequant kernel.
fn gq4a_round_trip(weights: &[f32]) -> Vec<f32> {
    glproc::kernels::gquant::dequant_gq4a_stream(&gq4a_bytes(weights))
}

/// The same round trip with the two 4-bit codes in each byte swapped before
/// decoding — the Q6_K-class nibble-order defect, on GQ4A's real layout.
///
/// `GQ4ABlock` is `#[repr(C)]` as `u16` super_scale, `[i8; 8]` scale deltas,
/// then the 128 packed weight bytes, so the codes start at byte 10 of each
/// 138-byte superblock. Only those are swapped; the scales are left alone,
/// which is what the real defect did.
fn gq4a_round_trip_wrong_nibble_order(weights: &[f32]) -> Vec<f32> {
    const WEIGHTS_OFFSET: usize = 10;
    let mut bytes = gq4a_bytes(weights);
    for block in bytes.chunks_exact_mut(glictus_caliburni::gquant::GQ4ABlock::BYTES) {
        for byte in &mut block[WEIGHTS_OFFSET..] {
            *byte = byte.rotate_left(4);
        }
    }
    glproc::kernels::gquant::dequant_gq4a_stream(&bytes)
}

/// A defect that is *not* a pure permutation: rotate the packed code bytes by
/// one sub-block, so codes are decoded against a different sub-block's scale.
///
/// The within-byte nibble swap above turns out to be permutation-*preserving*
/// (see this example's output), which is a genuinely adversarial case for any
/// distributional statistic. The real Q6_K defect moved codes across sub-block
/// boundaries, where the scales differ — this arm reproduces that instead, so
/// the two can be told apart.
fn gq4a_round_trip_rotated_subblock(weights: &[f32]) -> Vec<f32> {
    const WEIGHTS_OFFSET: usize = 10;
    /// 32 weights per sub-block, 2 codes per byte.
    const SUBBLOCK_BYTES: usize = 16;
    let mut bytes = gq4a_bytes(weights);
    for block in bytes.chunks_exact_mut(glictus_caliburni::gquant::GQ4ABlock::BYTES) {
        block[WEIGHTS_OFFSET..].rotate_left(SUBBLOCK_BYTES);
    }
    glproc::kernels::gquant::dequant_gq4a_stream(&bytes)
}

/// GQ2A round trip — the same weights at 2 bits instead of 4.
fn gq2a_round_trip(weights: &[f32]) -> Vec<f32> {
    let blocks = encode_gq2a_tensor(weights).expect("tensor length is a multiple of 256");
    let bytes: Vec<u8> = blocks.iter().flat_map(|b| b.as_bytes().to_vec()).collect();
    glproc::kernels::gquant::dequant_gq2a_stream(&bytes)
}

/// Mean absolute error against the original — the "both show residual" premise.
fn mae(original: &[f32], candidate: &[f32]) -> f64 {
    let n = original.len().min(candidate.len());
    if n == 0 {
        return 0.0;
    }
    original
        .iter()
        .zip(candidate)
        .take(n)
        .map(|(a, b)| (*a as f64 - *b as f64).abs())
        .sum::<f64>()
        / n as f64
}

fn report(label: &str, d: &VLBitDivergence, mae: f64) {
    println!("\n{label}");
    println!("  MAE vs f32 original    : {mae:.6e}");
    println!("  exponent L1 (0..2)     : {:.4}", d.exponent_l1);
    match d.mantissa_entropy_delta {
        Some(delta) => println!("  mantissa entropy delta : {delta:+.4} bits"),
        None => println!("  mantissa entropy delta : n/a (a side declined its map)"),
    }
    println!(
        "  largest bit delta      : {:+.4} at bit {}",
        d.max_bit_delta, d.max_bit_position
    );

    // The three regions of an IEEE-754 binary32, summarised — a structured
    // defect concentrates somewhere, a scheme spreads out.
    let mantissa: f64 = d.bit_fraction_delta[0..23].iter().map(|x| x.abs()).sum();
    let exponent: f64 = d.bit_fraction_delta[23..31].iter().map(|x| x.abs()).sum();
    let sign = d.bit_fraction_delta[31].abs();
    println!("  |delta| by region      : mantissa {mantissa:.4}, exponent {exponent:.4}, sign {sign:.4}");
}

fn main() {
    // 4096 weights = 16 superblocks of 256. Small enough that both sides stay
    // under the mantissa cap, so the entropy delta is available.
    let original = weight_tensor(4096);
    let base = profile(&original);
    assert!(
        !base.mantissa_sparse_skipped,
        "the tensor must stay under the cap for the entropy axis to exist"
    );

    println!("GLBitProf divergence — synthetic 4096-weight tensor, real encoders and kernels");
    println!("NOT the GQ4A/Q6_K archive comparison Gate 2 asks for: see this file's docs.");
    println!("\nbaseline (f32 original)");
    println!("  count {}, sign {:.1}%, exponent {}..{}, mantissa entropy {:.4} bits",
        base.count,
        base.sign_set_ratio * 100.0,
        base.exponent_min,
        base.exponent_max,
        base.mantissa_entropy_bits.unwrap()
    );

    let gq4a = gq4a_round_trip(&original);
    let gq2a = gq2a_round_trip(&original);
    let broken = gq4a_round_trip_wrong_nibble_order(&original);

    report("GQ4A (4-bit, correct)", &compare(&base, &profile(&gq4a)), mae(&original, &gq4a));
    report("GQ2A (2-bit, correct)", &compare(&base, &profile(&gq2a)), mae(&original, &gq2a));
    report(
        "GQ4A with swapped nibble order (the bug class)",
        &compare(&base, &profile(&broken)),
        mae(&original, &broken),
    );

    let rotated = gq4a_round_trip_rotated_subblock(&original);
    report(
        "GQ4A with a sub-block rotation (codes meet the wrong scale)",
        &compare(&base, &profile(&rotated)),
        mae(&original, &rotated),
    );

    // The comparisons that matter: same scheme, same bit width, same encoder —
    // the only difference is the defect. Anything showing up here is the bug
    // signature with the scheme's own contribution divided out.
    report(
        "correct GQ4A vs nibble-swapped GQ4A (scheme held constant)",
        &compare(&profile(&gq4a), &profile(&broken)),
        mae(&gq4a, &broken),
    );
    report(
        "correct GQ4A vs sub-block-rotated GQ4A (scheme held constant)",
        &compare(&profile(&gq4a), &profile(&rotated)),
        mae(&gq4a, &rotated),
    );

    println!("\n{}", "-".repeat(78));
    println!("MEASURED READING - and it contradicts the premise this was built on.");
    println!();
    println!("The design expected a nibble-order defect to show as a structured");
    println!("per-position anomaly. It does not. Against the correct GQ4A decode the");
    println!("nibble-swapped one scores EXACTLY zero on every axis - exponent L1,");
    println!("per-position bit deltas, mantissa entropy - while its MAE against the");
    println!("original is 14x the scheme's own residual.");
    println!();
    println!("The reason is structural, not a defect in the profiler: swapping the two");
    println!("codes inside a byte exchanges weights 2i and 2i+1, which sit in the same");
    println!("32-weight sub-block and therefore share a scale. The decoded multiset is");
    println!("unchanged. Every statistic in VLBitProfile is permutation-invariant by");
    println!("construction, so none of them can see a permutation.");
    println!();
    println!("The sub-block rotation arm is the control: it moves codes across a scale");
    println!("boundary, is NOT a pure permutation, and does register. So the honest");
    println!("scope is: GLBitProf detects defects that change the DISTRIBUTION of");
    println!("values and is blind to defects that only change their ORDER. A");
    println!("permutation-class bug needs a positional check - element-wise comparison");
    println!("against an oracle decode - which is validation::parity's job, not this");
    println!("module's.");
}
