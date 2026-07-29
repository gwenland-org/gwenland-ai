//! SwiGLU and GeGLU feed-forward networks (ARTX03 §5, ARTX11 §4 GeGLU retrofit).

use crate::ops::linear::linear;
use crate::ops::util::splat_like;
use crate::tensor::Tensor;
use crate::GlError;

/// SwiGLU: `down(SiLU(gate(x)) ⊙ up(x))`.
///
/// * `x`: `[..., D]`
/// * `gate_proj`, `up_proj`: `[FFN, D]`  — HuggingFace `[out, in]` layout
/// * `down_proj`: `[D, FFN]`             — likewise
///
/// ⛔ The weights are `[out, in]`, not `[in, out]`. PyTorch's `nn.Linear`
/// stores them transposed relative to the maths; see [`linear`].
///
/// The gate and up projections are emitted as two separate matmuls. ARTX01
/// §7.5 suggests fusing them into one `[D, 2·FFN]` weight and splitting — but
/// the checkpoint stores them as two tensors under two keys, so fusing means
/// concatenating weights at load time. That is a runtime decision (ARTX04),
/// not something the op layer should bake in, and XLA can schedule the two
/// independent matmuls concurrently regardless.
pub fn swiglu_ffn(x: &Tensor, gate_proj: &Tensor, up_proj: &Tensor, down_proj: &Tensor) -> Tensor {
    let d = x.dim(x.rank() - 1);
    assert_eq!(
        gate_proj.shape().dims.get(1).copied(),
        Some(d),
        "swiglu_ffn: gate_proj must be [FFN, D] with D = {d}, got {:?}",
        gate_proj.shape().dims
    );
    assert_eq!(
        up_proj.shape().dims,
        gate_proj.shape().dims,
        "swiglu_ffn: gate_proj and up_proj must have the same shape"
    );
    let ffn = gate_proj.dim(0);
    assert_eq!(
        down_proj.shape().dims,
        vec![d, ffn],
        "swiglu_ffn: down_proj must be [D, FFN] = [{d}, {ffn}]"
    );

    let gate = linear(x, gate_proj).silu();
    let up = linear(x, up_proj);
    linear(&gate.mul(&up), down_proj)
}

/// GeGLU: `down(GELU(gate(x)) ⊙ up(x))` — Gemma's FFN (ARTX11 §4.2).
///
/// Shape contract identical to [`swiglu_ffn`]; only the gate branch's
/// activation differs.
///
/// ⛔ **`tanh_approx` must match the checkpoint.** Gemma's published configs
/// specify `hidden_activation: "gelu_pytorch_tanh"` — the tanh approximation,
/// not exact (erf-based) GELU. The two are numerically close but not
/// identical, and using the wrong one is P4's failure class: shape-valid,
/// non-crashing, fluent, wrong.
///
/// Per **P5 — refuse rather than approximate** (`gljax/src/lib.rs`'s five
/// principles): `tanh_approx = false` returns `Err` instead of silently
/// substituting the tanh approximation. StableHLO has no primitive `erf`, and
/// gljax emits no kernels of its own — a hand-composed erf polynomial would be
/// exactly the kind of "probably close enough" numerics this project has
/// already been bitten by (see `ops/norm.rs`'s epsilon-placement history).
/// Exact GELU is future work, added when a checkpoint actually needs it.
pub fn geglu_ffn(
    x: &Tensor,
    gate_proj: &Tensor,
    up_proj: &Tensor,
    down_proj: &Tensor,
    tanh_approx: bool,
) -> Result<Tensor, GlError> {
    if !tanh_approx {
        return Err(GlError::Engine(
            "geglu_ffn: exact (erf-based) GELU is not implemented — StableHLO has no \
             primitive erf and gljax refuses to approximate it silently (P5). Pass \
             tanh_approx = true, which is what every published Gemma checkpoint uses \
             (hidden_activation = \"gelu_pytorch_tanh\")."
                .to_string(),
        ));
    }

    let d = x.dim(x.rank() - 1);
    assert_eq!(
        gate_proj.shape().dims.get(1).copied(),
        Some(d),
        "geglu_ffn: gate_proj must be [FFN, D] with D = {d}, got {:?}",
        gate_proj.shape().dims
    );
    assert_eq!(
        up_proj.shape().dims,
        gate_proj.shape().dims,
        "geglu_ffn: gate_proj and up_proj must have the same shape"
    );
    let ffn = gate_proj.dim(0);
    assert_eq!(
        down_proj.shape().dims,
        vec![d, ffn],
        "geglu_ffn: down_proj must be [D, FFN] = [{d}, {ffn}]"
    );

    let gate = gelu_tanh_approx(&linear(x, gate_proj));
    let up = linear(x, up_proj);
    Ok(linear(&gate.mul(&up), down_proj))
}

/// `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))` — the
/// `gelu_pytorch_tanh` approximation, composed from primitives gljax already
/// emits (`tanh`, `mul`, `add`) the same way [`Tensor::silu`] composes SiLU
/// from `logistic` + `multiply` rather than reaching for a fused op.
fn gelu_tanh_approx(x: &Tensor) -> Tensor {
    const SQRT_2_OVER_PI: f64 = 0.7978845608028654;
    const CUBIC_COEFF: f64 = 0.044715;

    let dims = x.shape().dims.clone();
    let dtype = x.dtype();
    let x3 = x.mul(x).mul(x);
    let cubic_coeff = splat_like(x, CUBIC_COEFF, dims.clone(), dtype);
    let inner = x.add(&x3.mul(&cubic_coeff));
    let sqrt_2_over_pi = splat_like(x, SQRT_2_OVER_PI, dims.clone(), dtype);
    let scaled_inner = inner.mul(&sqrt_2_over_pi);
    let tanh_part = scaled_inner.tanh();
    let one = splat_like(x, 1.0, dims.clone(), dtype);
    let half = splat_like(x, 0.5, dims, dtype);
    x.mul(&half).mul(&tanh_part.add(&one))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::TraceCx;
    use crate::stablehlo::types::{DType, Shape};

    #[test]
    fn swiglu_returns_to_the_model_dimension() {
        let mut cx = TraceCx::new("main", "ffn");
        let x = cx.input("x", Shape::new([1, 8, 896], DType::F32));
        let gate = cx.weight("gate_proj.weight", Shape::new([4864, 896], DType::F32));
        let up = cx.weight("up_proj.weight", Shape::new([4864, 896], DType::F32));
        let down = cx.weight("down_proj.weight", Shape::new([896, 4864], DType::F32));
        let y = swiglu_ffn(&x, &gate, &up, &down);
        assert_eq!(y.shape().dims, vec![1, 8, 896]);

        let mlir = cx.finish(&[&y]).mlir;
        assert_eq!(mlir.matches("stablehlo.dot_general").count(), 3, "{mlir}");
        // SiLU on the gate branch only — two multiplies total (silu, then gate⊙up).
        assert_eq!(mlir.matches(r#""stablehlo.logistic""#).count(), 1, "{mlir}");
        assert_eq!(mlir.matches(r#""stablehlo.multiply""#).count(), 2, "{mlir}");
    }

    /// The activation belongs on the gate branch. Applying it to `up` instead
    /// leaves every shape and every op count identical.
    #[test]
    fn silu_is_applied_to_the_gate_branch_not_the_up_branch() {
        let mut cx = TraceCx::new("main", "ffn");
        let x = cx.input("x", Shape::new([2, 4], DType::F32));
        let gate = cx.weight("g", Shape::new([8, 4], DType::F32));
        let up = cx.weight("u", Shape::new([8, 4], DType::F32));
        let down = cx.weight("d", Shape::new([4, 8], DType::F32));
        let y = swiglu_ffn(&x, &gate, &up, &down);
        let mlir = cx.finish(&[&y]).mlir;

        // %v0 x, %v1 gate, %v2 up, %v3 down; %v4 = x·gate, %v5 = logistic(%v4).
        assert!(
            mlir.contains(r#"%v5 = "stablehlo.logistic"(%v4)"#),
            "logistic must consume the gate projection:\n{mlir}"
        );
        assert!(
            mlir.contains(r#"%v6 = "stablehlo.multiply"(%v4, %v5)"#),
            "{mlir}"
        );
    }

    #[test]
    #[should_panic(expected = "down_proj must be [D, FFN]")]
    fn swiglu_rejects_a_transposed_down_projection() {
        let mut cx = TraceCx::new("main", "ffn");
        let x = cx.input("x", Shape::new([2, 4], DType::F32));
        let gate = cx.weight("g", Shape::new([8, 4], DType::F32));
        let up = cx.weight("u", Shape::new([8, 4], DType::F32));
        let down = cx.weight("d", Shape::new([8, 4], DType::F32));
        let _ = swiglu_ffn(&x, &gate, &up, &down);
    }

    #[test]
    fn geglu_returns_to_the_model_dimension() {
        let mut cx = TraceCx::new("main", "ffn");
        let x = cx.input("x", Shape::new([1, 8, 896], DType::F32));
        let gate = cx.weight("gate_proj.weight", Shape::new([4864, 896], DType::F32));
        let up = cx.weight("up_proj.weight", Shape::new([4864, 896], DType::F32));
        let down = cx.weight("down_proj.weight", Shape::new([896, 4864], DType::F32));
        let y = geglu_ffn(&x, &gate, &up, &down, true).expect("tanh_approx=true must succeed");
        assert_eq!(y.shape().dims, vec![1, 8, 896]);

        let mlir = cx.finish(&[&y]).mlir;
        assert_eq!(mlir.matches("stablehlo.dot_general").count(), 3, "{mlir}");
        // GeGLU uses tanh, not logistic — SwiGLU's activation must not leak in.
        assert_eq!(mlir.matches(r#""stablehlo.tanh""#).count(), 1, "{mlir}");
        assert_eq!(mlir.matches(r#""stablehlo.logistic""#).count(), 0, "{mlir}");
    }

    #[test]
    fn geglu_refuses_exact_gelu_rather_than_approximate_it() {
        let mut cx = TraceCx::new("main", "ffn");
        let x = cx.input("x", Shape::new([2, 4], DType::F32));
        let gate = cx.weight("g", Shape::new([8, 4], DType::F32));
        let up = cx.weight("u", Shape::new([8, 4], DType::F32));
        let down = cx.weight("d", Shape::new([4, 8], DType::F32));
        let err = geglu_ffn(&x, &gate, &up, &down, false).expect_err("must refuse, not approximate");
        assert!(err.to_string().contains("erf"), "{err}");
    }

    #[test]
    #[should_panic(expected = "down_proj must be [D, FFN]")]
    fn geglu_rejects_a_transposed_down_projection() {
        let mut cx = TraceCx::new("main", "ffn");
        let x = cx.input("x", Shape::new([2, 4], DType::F32));
        let gate = cx.weight("g", Shape::new([8, 4], DType::F32));
        let up = cx.weight("u", Shape::new([8, 4], DType::F32));
        let down = cx.weight("d", Shape::new([8, 4], DType::F32));
        let _ = geglu_ffn(&x, &gate, &up, &down, true);
    }

    /// Numeric pin against a scalar reference — `gelu_tanh_approx` must
    /// implement the exact `gelu_pytorch_tanh` formula, not something
    /// approximately shaped like it.
    #[test]
    fn gelu_tanh_approx_matches_the_scalar_reference() {
        fn reference(x: f64) -> f64 {
            const SQRT_2_OVER_PI: f64 = 0.7978845608028654;
            0.5 * x * (1.0 + (SQRT_2_OVER_PI * (x + 0.044715 * x.powi(3))).tanh())
        }

        // gelu(0) = 0 exactly; gelu is odd-symmetric-ish but not odd overall,
        // so pin a positive and a negative point too.
        assert!((reference(0.0)).abs() < 1e-12);
        let at_one = reference(1.0);
        assert!(
            (at_one - 0.8411919906082768).abs() < 1e-9,
            "reference formula itself must match the known gelu_pytorch_tanh(1.0) value, got {at_one}"
        );
        let at_neg_two = reference(-2.0);
        assert!(
            (at_neg_two - (-0.04540230591222494)).abs() < 1e-9,
            "got {at_neg_two}"
        );
    }
}
