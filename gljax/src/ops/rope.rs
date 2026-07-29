//! Rotary position embeddings, NeoX variant (ARTX03 §3).
//!
//! # ⛔ Which pairing is "NeoX"
//!
//! The series disagrees with itself, and the sprint brief disagrees with both:
//!
//! | Source | Claims NeoX pairs |
//! |---|---|
//! | ARTX03 §3 prose + its table builder + its `rope_neox` | `(2i, 2i+1)` — adjacent |
//! | ARTX01 §7.2's MLIR | `(i, i+D/2)` — half-split |
//! | Sprint brief, first line | "even-odd, NOT consecutive pairs" |
//! | Sprint brief, its own test comment | "the two halves" |
//!
//! **glproc settles it.** `glproc/src/runner.rs:161` has a `RopeStyle` that was
//! validated end-to-end against Qwen2.5-0.5B:
//!
//! ```text
//! RopeStyle::Norm => (2 * i, 2 * i + 1),      // llama / GPT-J
//! RopeStyle::Neox => (i, i + half),           // qwen2, phi, gemma, ...
//! ```
//!
//! gljax implements the **half-split**, matching the engine that measurably
//! produces correct Qwen2 output. Adjacent pairing is what `RopeStyle::Norm`
//! means and is a different model family.
//!
//! This is P4 in its purest form: both pairings have identical shapes, both
//! produce fluent-looking text, and only one is right.

use crate::ops::util::dense_const_f32;
use crate::precision;
use crate::stablehlo::types::{DType, Shape};
use crate::tensor::Tensor;
use crate::GlError;

/// The classic RoPE base frequency (Llama, Mistral, GPT-NeoX).
///
/// ⛔ **This is NOT Qwen2's.** `Qwen/Qwen2-0.5B`'s `config.json` sets
/// `"rope_theta": 1000000.0` — a hundred times larger. Using 1e4 for a
/// 1e6-trained model rotates every position by the wrong angle, and the
/// failure is P4-shaped: shapes match, no error, output degrades into
/// plausible-looking nonsense that gets worse further into the sequence.
///
/// Always read `rope_theta` from the checkpoint's config
/// ([`crate::model::Qwen2Config::from_hf_config_json`]). This constant exists
/// for tests and for models that genuinely use it.
pub const DEFAULT_ROPE_BASE: f32 = 10_000.0;

/// `Qwen/Qwen2-0.5B`'s `rope_theta`, verified against the published
/// `config.json` at revision `91d2aff3f957f99e4c74c962f2f408dcc88a18d8`.
pub const QWEN2_ROPE_BASE: f32 = 1_000_000.0;

/// Builds the cos/sin tables as `[max_seq_len, head_dim]` F32 data.
///
/// ```text
/// θ_i = base^(-2i / head_dim)     for i in 0..head_dim/2
/// ```
///
/// Each value is written at **both** `i` and `i + half`, because the half-split
/// rotation reads the same angle for the two elements of a pair. ARTX03's
/// builder writes them at `2i` and `2i+1`, which is the layout the adjacent
/// pairing needs — the two are not interchangeable.
///
/// Computed in Rust and emitted as a constant rather than as
/// `stablehlo.sine`/`cosine`, whose backend support varies (ARTX03 §3's design
/// decision, which does hold).
pub fn rope_tables(max_seq_len: usize, head_dim: usize, base: f32) -> (Vec<f32>, Vec<f32>) {
    assert!(
        head_dim.is_multiple_of(2),
        "rope: head_dim must be even, got {head_dim}"
    );
    let half = head_dim / 2;
    let mut cos = vec![0.0f32; max_seq_len * head_dim];
    let mut sin = vec![0.0f32; max_seq_len * head_dim];

    for pos in 0..max_seq_len {
        for i in 0..half {
            // Matches glproc's `fill_rope_table`: 1 / base^(2i/head_dim).
            let freq = 1.0f32 / base.powf(2.0 * i as f32 / head_dim as f32);
            let (s, c) = (pos as f32 * freq).sin_cos();
            cos[pos * head_dim + i] = c;
            cos[pos * head_dim + i + half] = c;
            sin[pos * head_dim + i] = s;
            sin[pos * head_dim + i + half] = s;
        }
    }
    (cos, sin)
}

/// Emits the cos/sin tables into a trace as dense constants.
///
/// # Errors
/// If the table exceeds the dense-constant cap — see
/// [`crate::stablehlo::ops::MAX_DENSE_CONSTANT_ELEMS`]. At `head_dim = 64` that
/// is a `max_seq_len` of 16384, so every ARTX05 bucket fits.
pub fn emit_rope_tables(
    like: &Tensor,
    max_seq_len: usize,
    head_dim: usize,
    base: f32,
) -> Result<(Tensor, Tensor), GlError> {
    let (cos, sin) = rope_tables(max_seq_len, head_dim, base);
    let shape = Shape::new([max_seq_len, head_dim], DType::F32);
    let cos_t = dense_const_f32(like, &cos, shape.clone())?;
    let sin_t = dense_const_f32(like, &sin, shape)?;
    Ok((cos_t, sin_t))
}

/// Applies NeoX RoPE to a `[B, H, S, head_dim]` tensor.
///
/// ```text
/// out[..i]      = x[..i]      · cos_i − x[..i+half] · sin_i
/// out[..i+half] = x[..i+half] · cos_i + x[..i]      · sin_i
/// ```
///
/// which is `x · cos + rotate_half(x) · sin` with
/// `rotate_half(x) = concat(−x[half:], x[:half])`. Expanding it that way keeps
/// it to two slices, a negate, a concatenate and three elementwise ops — no
/// rank-5 reshape gymnastics.
///
/// `seq_offset` is the position of the first token, which is how a decode step
/// at position `p` reads row `p` of the table.
pub fn rope_neox(
    x: &Tensor,
    cos_table: &Tensor,
    sin_table: &Tensor,
    seq_offset: usize,
) -> Tensor {
    let [b, h, s, head_dim] = crate::ops::util::expect_rank4(x, "rope_neox");
    assert!(
        head_dim.is_multiple_of(2),
        "rope_neox: head_dim must be even, got {head_dim}"
    );
    let half = head_dim / 2;

    let table_dims = cos_table.shape().dims.clone();
    assert_eq!(
        table_dims.len(),
        2,
        "rope_neox: cos table must be [max_seq_len, head_dim], got {table_dims:?}"
    );
    assert_eq!(
        table_dims[1], head_dim,
        "rope_neox: table head_dim {} does not match x's {head_dim}",
        table_dims[1]
    );
    assert!(
        seq_offset + s <= table_dims[0],
        "rope_neox: positions {seq_offset}..{} exceed the table's {} rows — \
         the RoPE table must cover max_seq_len",
        seq_offset + s,
        table_dims[0]
    );

    let acc = precision::current().rope;

    // Slice the table to this window, then broadcast over batch and heads.
    let take = |t: &Tensor| {
        t.slice(
            vec![seq_offset, 0],
            vec![seq_offset + s, head_dim],
            vec![1, 1],
        )
        .broadcast_to(vec![2, 3], vec![b, h, s, head_dim])
        .to_dtype(acc)
    };
    let cos = take(cos_table);
    let sin = take(sin_table);

    let x_acc = x.to_dtype(acc);

    // rotate_half(x) = concat(-x[..., half:], x[..., :half])
    let first = x_acc.slice(
        vec![0, 0, 0, 0],
        vec![b, h, s, half],
        vec![1, 1, 1, 1],
    );
    let second = x_acc.slice(
        vec![0, 0, 0, half],
        vec![b, h, s, head_dim],
        vec![1, 1, 1, 1],
    );
    let rotated_half = Tensor::concat(&[&second.neg(), &first], 3);

    let out = &x_acc.mul(&cos) + &rotated_half.mul(&sin);
    out.to_dtype(x.dtype())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::TraceCx;

    /// Scalar reference, transcribed from `glproc/src/runner.rs:161`
    /// (`RopeStyle::Neox`) — the implementation validated against Qwen2.5-0.5B.
    fn glproc_rope_neox(x: &mut [f32], cos: &[f32], sin: &[f32], head_dim: usize) {
        let half = head_dim / 2;
        for i in 0..half {
            let (a, b) = (i, i + half);
            let (x0, x1) = (x[a], x[b]);
            x[a] = x0 * cos[i] - x1 * sin[i];
            x[b] = x0 * sin[i] + x1 * cos[i];
        }
    }

    /// The `x·cos + rotate_half(x)·sin` form gljax emits, evaluated on the host.
    fn gljax_form(x: &[f32], cos_row: &[f32], sin_row: &[f32], head_dim: usize) -> Vec<f32> {
        let half = head_dim / 2;
        let rot: Vec<f32> = (0..head_dim)
            .map(|i| if i < half { -x[i + half] } else { x[i - half] })
            .collect();
        (0..head_dim)
            .map(|i| x[i] * cos_row[i] + rot[i] * sin_row[i])
            .collect()
    }

    /// ⭐ The two formulations must agree elementwise. This is what pins the
    /// half-split against the adjacent-pair reading of "NeoX".
    #[test]
    fn rope_neox_half_split_matches_glproc_reference() {
        // f32 trig plus three multiply-adds; a few ulps is the whole budget.
        const TOL_ROPE: f32 = 1e-5;
        let head_dim = 8;
        let half = head_dim / 2;
        let (cos_tab, sin_tab) = rope_tables(4, head_dim, DEFAULT_ROPE_BASE);

        for pos in 0..4 {
            let row = pos * head_dim;
            let cos_row = &cos_tab[row..row + head_dim];
            let sin_row = &sin_tab[row..row + head_dim];

            let x: Vec<f32> = (0..head_dim).map(|i| 0.5 + i as f32 * 0.25).collect();

            let mut expected = x.clone();
            glproc_rope_neox(&mut expected, cos_row, sin_row, head_dim);

            let got = gljax_form(&x, cos_row, sin_row, head_dim);

            for i in 0..head_dim {
                assert!(
                    (expected[i] - got[i]).abs() <= TOL_ROPE,
                    "pos {pos} lane {i}: glproc {} vs gljax {}",
                    expected[i],
                    got[i]
                );
            }
            // And confirm the table really does repeat the angle across the
            // half boundary, which is what makes the two forms equivalent.
            for i in 0..half {
                assert_eq!(cos_row[i], cos_row[i + half]);
                assert_eq!(sin_row[i], sin_row[i + half]);
            }
        }
    }

    /// The adjacent-pair reading is a *different function*. If this ever stops
    /// failing, the two conventions have been conflated somewhere.
    #[test]
    fn adjacent_pair_rotation_gives_a_different_answer() {
        let head_dim = 8;
        let (cos_tab, sin_tab) = rope_tables(4, head_dim, DEFAULT_ROPE_BASE);
        let row = 2 * head_dim; // pos = 2, where the angles are non-trivial
        let cos_row = &cos_tab[row..row + head_dim];
        let sin_row = &sin_tab[row..row + head_dim];
        let x: Vec<f32> = (0..head_dim).map(|i| 0.5 + i as f32 * 0.25).collect();

        let half_split = gljax_form(&x, cos_row, sin_row, head_dim);

        let mut adjacent = x.clone();
        for i in 0..head_dim / 2 {
            let (a, b) = (2 * i, 2 * i + 1);
            let (x0, x1) = (adjacent[a], adjacent[b]);
            adjacent[a] = x0 * cos_row[i] - x1 * sin_row[i];
            adjacent[b] = x0 * sin_row[i] + x1 * cos_row[i];
        }

        let max_diff = half_split
            .iter()
            .zip(&adjacent)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff > 1e-3,
            "the two pairings must not coincide, else this test proves nothing"
        );
    }

    #[test]
    fn rope_table_is_identity_at_position_zero() {
        // pos = 0 → angle 0 → cos 1, sin 0, so RoPE is the identity there.
        let (cos, sin) = rope_tables(3, 4, DEFAULT_ROPE_BASE);
        for i in 0..4 {
            assert_eq!(cos[i], 1.0);
            assert_eq!(sin[i], 0.0);
        }
        // pos = 1, i = 0 → θ_0 = 1.0 → cos(1), sin(1).
        assert!((cos[4] - 1.0f32.cos()).abs() < 1e-6);
        assert!((sin[4] - 1.0f32.sin()).abs() < 1e-6);
    }

    #[test]
    fn rope_emits_slice_negate_concat_and_preserves_shape() {
        let mut cx = TraceCx::new("main", "rope");
        let q = cx.input("q", Shape::new([1, 14, 8, 64], DType::F32));
        let (cos, sin) = emit_rope_tables(&q, 128, 64, DEFAULT_ROPE_BASE).expect("tables");
        let out = rope_neox(&q, &cos, &sin, 0);
        assert_eq!(out.shape().dims, vec![1, 14, 8, 64]);

        let mlir = cx.finish(&[&out]).mlir;
        assert!(mlir.contains(r#""stablehlo.negate""#), "{mlir}");
        assert!(mlir.contains(r#""stablehlo.concatenate""#), "{mlir}");
        // The halves are contiguous slices, stride 1 — not stride 2.
        assert!(
            mlir.contains("start_indices = array<i64: 0, 0, 0, 32>"),
            "the second half must start at head_dim/2:\n{mlir}"
        );
        assert!(
            !mlir.contains("strides = array<i64: 1, 1, 1, 2>"),
            "stride-2 slicing is the adjacent-pair convention:\n{mlir}"
        );
    }

    #[test]
    fn rope_slices_the_table_at_the_decode_position() {
        let mut cx = TraceCx::new("main", "rope");
        let q = cx.input("q", Shape::new([1, 2, 1, 8], DType::F32));
        let (cos, sin) = emit_rope_tables(&q, 16, 8, DEFAULT_ROPE_BASE).expect("tables");
        let out = rope_neox(&q, &cos, &sin, 5);
        let mlir = cx.finish(&[&out]).mlir;
        assert!(
            mlir.contains("start_indices = array<i64: 5, 0>, limit_indices = array<i64: 6, 8>"),
            "a decode step at position 5 must read table row 5:\n{mlir}"
        );
    }

    #[test]
    #[should_panic(expected = "exceed the table's")]
    fn rope_refuses_positions_past_the_end_of_the_table() {
        let mut cx = TraceCx::new("main", "rope");
        let q = cx.input("q", Shape::new([1, 2, 4, 8], DType::F32));
        let (cos, sin) = emit_rope_tables(&q, 8, 8, DEFAULT_ROPE_BASE).expect("tables");
        let _ = rope_neox(&q, &cos, &sin, 6); // 6 + 4 > 8
    }
}
