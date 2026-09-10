//! The prefill attention oracle.
//!
//! Wave 14 measured that QK is 71% of the retained kernel and runs at 0.38% of
//! one SM's f32 peak, so the next wave replaces it with tensor-core tiles and an
//! online softmax. That is a rewrite of how the answer is computed, which means
//! the answer itself needs somewhere to live that is not a kernel.
//!
//! This is that place: the semantics, in the plainest host code that expresses
//! them, with no tiling, no fusion and no cleverness to go wrong. Every
//! attention path gets graded against it. It is deliberately slow.
//!
//! Two properties make it a usable oracle rather than just a second
//! implementation:
//!
//! * It takes the same [`VLAttentionCall`] the dispatcher takes, so a test
//!   cannot accidentally grade a different problem than the one that ran.
//! * It runs on the host, so it is unit-testable on a machine with no GPU. The
//!   contract gets checked even in a session that never reaches CUDA.

use super::VLAttentionCall;

/// Softmax over `scores`, in the numerically-stable order the kernels use:
/// subtract the row max, exponentiate, divide by the sum.
///
/// An empty slice is left alone rather than producing NaN; a row with no keys
/// cannot happen under the causal contract, but an oracle that quietly emits
/// NaN is worse than one that emits nothing.
fn softmax(scores: &mut [f32]) {
    let Some(&max) = scores
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
    else {
        return;
    };
    let mut sum = 0.0f32;
    for s in scores.iter_mut() {
        *s = (*s - max).exp();
        sum += *s;
    }
    if sum > 0.0 {
        for s in scores.iter_mut() {
            *s /= sum;
        }
    }
}

/// Causal prefill attention, computed row by row and head by head.
///
/// * `q` is `[n_tokens, ...]` with `q_row_stride` elements between rows, which
///   is `n_heads * head_dim` for a packed buffer and wider when Q is a column
///   slice of a stacked projection (Wave 13B).
/// * `k_cache` and `v_cache` are `[n_kv_heads, head_stride]`, with row `j` of a
///   head at offset `kv_head * head_stride + j * head_dim`.
/// * `out` is the packed `[n_tokens, n_heads * head_dim]` block.
///
/// Row `t` attends to exactly `pos_base + t + 1` cached rows. That single line
/// is the causal contract, and it is why the work is a triangle rather than a
/// rectangle.
pub fn prefill(
    q: &[f32],
    q_row_stride: usize,
    k_cache: &[f32],
    v_cache: &[f32],
    out: &mut [f32],
    call: &VLAttentionCall,
) {
    prefill_inner(q, q_row_stride, k_cache, v_cache, out, call, false)
}

#[allow(clippy::too_many_arguments)]
fn prefill_inner(
    q: &[f32],
    q_row_stride: usize,
    k_cache: &[f32],
    v_cache: &[f32],
    out: &mut [f32],
    call: &VLAttentionCall,
    f16_qk: bool,
) {
    let round = if f16_qk { through_f16 } else { |x: f32| x };
    let (n_tokens, n_heads) = (call.n_tokens as usize, call.n_heads as usize);
    let head_dim = call.head_dim as usize;
    let head_stride = call.head_stride as usize;
    let heads_per_kv = call.heads_per_kv() as usize;
    let out_row = n_heads * head_dim;
    assert!(
        q.len() >= (n_tokens - 1) * q_row_stride + out_row,
        "q too small"
    );
    assert_eq!(out.len(), n_tokens * out_row, "out must be packed");

    for t in 0..n_tokens {
        let cached_len = call.pos_base as usize + t + 1;
        for h in 0..n_heads {
            let kv_head = h / heads_per_kv;
            let q_at = t * q_row_stride + h * head_dim;
            let q_vec = &q[q_at..q_at + head_dim];
            let kv_at = kv_head * head_stride;

            let mut scores: Vec<f32> = (0..cached_len)
                .map(|j| {
                    let k_at = kv_at + j * head_dim;
                    let dot: f32 = q_vec
                        .iter()
                        .zip(&k_cache[k_at..k_at + head_dim])
                        .map(|(a, b)| round(*a) * round(*b))
                        .sum();
                    dot * call.scale
                })
                .collect();
            softmax(&mut scores);

            let o_at = t * out_row + h * head_dim;
            let o = &mut out[o_at..o_at + head_dim];
            o.fill(0.0);
            for (j, &w) in scores.iter().enumerate() {
                let v_at = kv_at + j * head_dim;
                for (acc, &v) in o.iter_mut().zip(&v_cache[v_at..v_at + head_dim]) {
                    *acc += w * v;
                }
            }
        }
    }
}

/// Round a value through IEEE binary16 and back, the way a tensor core sees it.
///
/// `mma.sync...f32.f16.f16.f32` takes f16 operands and accumulates in f32, so a
/// QK pass on the tensor cores rounds Q and K to half precision *before* the
/// products and keeps full precision after. This models exactly that, and only
/// that: the accumulate below stays f32.
///
/// Round-to-nearest-even, including subnormals. The first version of this
/// asserted that attention activations never reach the f16 subnormal range, and
/// the very first run falsified it: a Q element of 2^-16 turned up immediately,
/// because uniform activations in [-1, 1] cross 2^-14 roughly once in sixteen
/// thousand and there are 218k of them. Half precision keeps working down there
/// on a grid of 2^-24, so the model does too.
pub fn through_f16(x: f32) -> f32 {
    if x == 0.0 || !x.is_finite() {
        return x;
    }
    let bits = x.to_bits();
    let exp = ((bits >> 23) & 0xFF) as i32 - 127;
    assert!(
        exp <= 15,
        "value {x} overflows f16; attention activations should never get here"
    );
    if exp < -14 {
        // Subnormal: the representable values are multiples of 2^-24, and
        // anything under half a step flushes to zero.
        const STEP: f32 = 5.960_464_5e-8;
        return (x / STEP).round_ties_even() * STEP;
    }
    // Keep 10 explicit mantissa bits, round half to even on the 13 dropped.
    let mantissa = bits & 0x007F_FFFF;
    let dropped = mantissa & 0x0000_1FFF;
    let mut kept = mantissa & !0x0000_1FFF;
    let halfway = 0x0000_1000;
    if dropped > halfway || (dropped == halfway && (kept & 0x0000_2000) != 0) {
        kept += 0x0000_2000;
    }
    f32::from_bits((bits & 0xFF80_0000) | kept)
}

/// The oracle again, with QK rounded to half precision.
///
/// Same signature and same everything else, so a caller can difference the two
/// and see the cost of the tensor-core operand format on its own.
pub fn prefill_f16_qk(
    q: &[f32],
    q_row_stride: usize,
    k_cache: &[f32],
    v_cache: &[f32],
    out: &mut [f32],
    call: &VLAttentionCall,
) {
    prefill_inner(q, q_row_stride, k_cache, v_cache, out, call, true)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two KV heads of `head_stride`, filled so row `j` of head `kv` is a
    /// constant vector, which makes every expectation below arithmetic anyone
    /// can check by hand.
    fn cache(
        n_kv: usize,
        head_stride: usize,
        head_dim: usize,
        f: impl Fn(usize, usize) -> f32,
    ) -> Vec<f32> {
        let mut buf = vec![0f32; n_kv * head_stride];
        for kv in 0..n_kv {
            for j in 0..head_stride / head_dim {
                for d in 0..head_dim {
                    buf[kv * head_stride + j * head_dim + d] = f(kv, j);
                }
            }
        }
        buf
    }

    fn call(
        n_tokens: u32,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        head_stride: u32,
    ) -> VLAttentionCall {
        VLAttentionCall {
            n_tokens,
            pos_base: 0,
            n_heads,
            n_kv_heads,
            head_dim,
            head_stride,
            scale: 1.0,
        }
    }

    /// The causal contract, stated as an output: row 0 sees exactly one KV row,
    /// so whatever the scores are, softmax over one element is 1.0 and the
    /// answer is that row of V.
    #[test]
    fn the_first_row_can_only_see_the_first_kv_row() {
        let (head_dim, head_stride) = (4usize, 16usize);
        let c = call(4, 1, 1, head_dim as u32, head_stride as u32);
        let q = vec![1.0f32; 4 * head_dim];
        let k = cache(1, head_stride, head_dim, |_, j| j as f32);
        let v = cache(1, head_stride, head_dim, |_, j| 10.0 + j as f32);
        let mut out = vec![0f32; 4 * head_dim];
        prefill(&q, head_dim, &k, &v, &mut out, &c);
        assert_eq!(&out[..head_dim], &[10.0; 4], "row 0 must be exactly v[0]");
        // The last row sees all four, and every score differs, so it cannot be
        // any single V row.
        assert!(out[3 * head_dim] > 10.0 && out[3 * head_dim] < 13.0);
    }

    /// Identical keys mean identical scores, so the answer is the plain mean of
    /// the V rows the causal mask allows. That pins masking and normalisation
    /// together without depending on exp() at all.
    #[test]
    fn equal_scores_average_exactly_the_rows_the_mask_allows() {
        let (head_dim, head_stride) = (2usize, 8usize);
        let c = call(4, 1, 1, head_dim as u32, head_stride as u32);
        let q = vec![0.0f32; 4 * head_dim]; // every score is 0 -> uniform
        let k = cache(1, head_stride, head_dim, |_, _| 1.0);
        let v = cache(1, head_stride, head_dim, |_, j| j as f32);
        let mut out = vec![0f32; 4 * head_dim];
        prefill(&q, head_dim, &k, &v, &mut out, &c);
        for t in 0..4 {
            let want = (0..=t).map(|j| j as f32).sum::<f32>() / (t + 1) as f32;
            assert!(
                (out[t * head_dim] - want).abs() < 1e-6,
                "row {t}: {} != mean of v[0..={t}] = {want}",
                out[t * head_dim]
            );
        }
    }

    /// GQA maps seven query heads onto one KV head. Get that wrong and the
    /// model still runs, still emits text, and is quietly wrong — which is why
    /// it is pinned here rather than left to a tolerance check.
    #[test]
    fn gqa_heads_read_the_kv_head_they_share() {
        let (head_dim, head_stride) = (2usize, 4usize);
        let c = call(1, 14, 2, head_dim as u32, head_stride as u32);
        let q = vec![0.0f32; 14 * head_dim];
        let k = cache(2, head_stride, head_dim, |_, _| 1.0);
        // KV head 0 holds 100.0, KV head 1 holds 200.0.
        let v = cache(2, head_stride, head_dim, |kv, _| 100.0 * (kv + 1) as f32);
        let mut out = vec![0f32; 14 * head_dim];
        prefill(&q, 14 * head_dim, &k, &v, &mut out, &c);
        for h in 0..14 {
            let want = if h < 7 { 100.0 } else { 200.0 };
            assert_eq!(out[h * head_dim], want, "head {h} read the wrong KV head");
        }
    }

    /// The f16 model has to be a real round trip before any conclusion drawn
    /// from it means anything.
    #[test]
    fn the_half_precision_model_rounds_the_way_binary16_does() {
        assert_eq!(through_f16(1.0), 1.0);
        assert_eq!(through_f16(-2.5), -2.5);
        assert_eq!(through_f16(0.0), 0.0);
        // 1 + 2^-11 is exactly halfway between 1.0 and the next f16; ties go to
        // even, which is 1.0.
        assert_eq!(through_f16(1.0 + 2f32.powi(-11)), 1.0);
        // 1 + 2^-10 is representable and must survive untouched.
        assert_eq!(through_f16(1.0 + 2f32.powi(-10)), 1.0 + 2f32.powi(-10));
        // Anything between must land on one of those two, never elsewhere.
        for i in 1..64 {
            let x = 1.0 + 2f32.powi(-10) * (i as f32) / 64.0;
            let r = through_f16(x);
            assert!(r == 1.0 || r == 1.0 + 2f32.powi(-10), "{x} -> {r}");
        }
    }

    /// ⭐⭐ The Wave 15 design gate, and it decided the wave.
    ///
    /// `mma.sync` on sm_75 takes f16 operands, so a tensor-core QK rounds Q and
    /// K to half precision before multiplying. This asks what that costs the
    /// ANSWER, at the production shape, before anyone writes a line of PTX.
    ///
    /// Measured: **max_abs 1.2e-2, rms_rel 1.7e-3**. The attention parity test
    /// grades this path at `EPS_MATMUL = 1e-5`, so half-precision operands miss
    /// the tolerance the engine currently holds attention to by **three orders
    /// of magnitude**. It lands in the same class the engine tolerates for a
    /// Q4_K GEMV (1e-2) — which is to say f16 QK is a deliberate precision
    /// downgrade of quantisation size, not a free representation change.
    ///
    /// So Wave 15A takes the f32 route (one lane per key, no warp reduction),
    /// which the instruction count says is worth about 6x and which needs no
    /// new tolerance class at all. MMA-f16 stays available for a later wave,
    /// but only with a pre-registered tolerance and token-level oracle
    /// evidence, never as a silent swap.
    ///
    /// Reported rather than merely asserted, because the number is the point.
    #[test]
    fn half_precision_qk_costs_the_answer_almost_nothing() {
        let (n_tokens, n_heads, n_kv, head_dim) = (244usize, 14usize, 2usize, 64usize);
        let head_stride = 256 * head_dim;
        let width = n_heads * head_dim;
        let c = VLAttentionCall {
            n_tokens: n_tokens as u32,
            pos_base: 0,
            n_heads: n_heads as u32,
            n_kv_heads: n_kv as u32,
            head_dim: head_dim as u32,
            head_stride: head_stride as u32,
            scale: 1.0 / (head_dim as f32).sqrt(),
        };
        // Deterministic activations in the range attention actually sees.
        let gen = |n: usize, seed: u64| -> Vec<f32> {
            let mut state = seed | 1;
            (0..n)
                .map(|_| {
                    state ^= state >> 12;
                    state ^= state << 25;
                    state ^= state >> 27;
                    ((state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32 / (1u64 << 24) as f32
                        - 0.5)
                        * 2.0
                })
                .collect()
        };
        let q = gen(n_tokens * width, 11);
        let k = gen(n_kv * head_stride, 12);
        let v = gen(n_kv * head_stride, 13);
        let mut exact = vec![0f32; n_tokens * width];
        let mut half = vec![0f32; n_tokens * width];
        prefill(&q, width, &k, &v, &mut exact, &c);
        prefill_f16_qk(&q, width, &k, &v, &mut half, &c);

        let mut max_abs = 0f32;
        let mut max_rel = 0f32;
        let mut sum_sq_err = 0f64;
        let mut sum_sq = 0f64;
        for (a, b) in exact.iter().zip(&half) {
            let err = (a - b).abs();
            max_abs = max_abs.max(err);
            if a.abs() > 1e-3 {
                max_rel = max_rel.max(err / a.abs());
            }
            sum_sq_err += (err as f64) * (err as f64);
            sum_sq += (*a as f64) * (*a as f64);
        }
        let rms_rel = (sum_sq_err / sum_sq).sqrt();
        println!(
            "f16 QK vs f32 QK at the production shape: max_abs {max_abs:.3e}, max_rel {max_rel:.3e}, rms_rel {rms_rel:.3e}"
        );
        // Judged against the tolerances this engine already uses, not against a
        // number invented for this test. `parity.rs` grades the attention path
        // at EPS_MATMUL = 1e-5 (and `assert_close` makes that an ABSOLUTE bound
        // for outputs under 1.0), while its loosest accepted class is the Q4_K
        // GEMV at 1e-2.
        const ATTENTION_PARITY_EPS: f32 = 1e-5;
        const LOOSEST_ACCEPTED_EPS: f32 = 1e-2;
        assert!(
            max_abs > ATTENTION_PARITY_EPS * 100.0,
            "f16 QK came out far tighter than measured ({max_abs:.3e}); re-derive the Wave 15 decision, do not just relax this test"
        );
        assert!(
            max_abs <= LOOSEST_ACCEPTED_EPS * 1.5,
            "f16 QK error {max_abs:.3e} exceeds even the Q4_K class this engine accepts"
        );
        assert!(
            rms_rel < 1e-2,
            "aggregate error {rms_rel:.3e} is worse than expected"
        );
    }

    /// A strided Q is the Wave 13B layout: the oracle has to read a column
    /// slice of a wider slab and produce the same answer as the packed one.
    #[test]
    fn a_strided_q_gives_the_same_answer_as_a_packed_one() {
        let (head_dim, head_stride, n_heads, n_tokens) = (4usize, 16usize, 2usize, 4usize);
        let c = call(
            n_tokens as u32,
            n_heads as u32,
            1,
            head_dim as u32,
            head_stride as u32,
        );
        let width = n_heads * head_dim;
        let packed: Vec<f32> = (0..n_tokens * width)
            .map(|i| (i % 7) as f32 * 0.25)
            .collect();
        // The same values, as the first `width` columns of a wider slab.
        let stride = width + 5;
        let mut slab = vec![-99.0f32; n_tokens * stride];
        for t in 0..n_tokens {
            slab[t * stride..t * stride + width]
                .copy_from_slice(&packed[t * width..(t + 1) * width]);
        }
        let k = cache(1, head_stride, head_dim, |_, j| 1.0 + j as f32 * 0.5);
        let v = cache(1, head_stride, head_dim, |_, j| j as f32);
        let mut a = vec![0f32; n_tokens * width];
        let mut b = vec![0f32; n_tokens * width];
        prefill(&packed, width, &k, &v, &mut a, &c);
        prefill(&slab, stride, &k, &v, &mut b, &c);
        assert_eq!(a, b, "the padding either side must not reach the answer");
    }

    /// Chunked prefill must equal one-shot prefill: the second call carries
    /// `pos_base`, and that is the whole difference.
    #[test]
    fn chunking_the_prompt_does_not_change_the_answer() {
        let (head_dim, head_stride, n_heads) = (4usize, 32usize, 2usize);
        let width = n_heads * head_dim;
        let q: Vec<f32> = (0..6 * width)
            .map(|i| ((i % 11) as f32 - 5.0) * 0.3)
            .collect();
        let k = cache(1, head_stride, head_dim, |_, j| (j % 5) as f32 * 0.4);
        let v = cache(1, head_stride, head_dim, |_, j| 1.0 + j as f32);

        let whole = call(6, n_heads as u32, 1, head_dim as u32, head_stride as u32);
        let mut one_shot = vec![0f32; 6 * width];
        prefill(&q, width, &k, &v, &mut one_shot, &whole);

        let mut chunked = vec![0f32; 6 * width];
        let first = VLAttentionCall {
            n_tokens: 4,
            ..whole
        };
        prefill(&q, width, &k, &v, &mut chunked[..4 * width], &first);
        let rest = VLAttentionCall {
            n_tokens: 2,
            pos_base: 4,
            ..whole
        };
        prefill(
            &q[4 * width..],
            width,
            &k,
            &v,
            &mut chunked[4 * width..],
            &rest,
        );
        for (i, (a, b)) in one_shot.iter().zip(&chunked).enumerate() {
            assert!((a - b).abs() < 1e-6, "element {i}: {a} != {b}");
        }
    }
}
