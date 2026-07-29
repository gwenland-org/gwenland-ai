//! SwiGLU feed-forward network (ARTX03 §5).

use crate::ops::linear::linear;
use crate::tensor::Tensor;

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
}
