//! Mixture-of-experts FFN — **not implemented** (ARTX03 §7).
//!
//! Qwen2-0.5B, the model this sprint targets, uses a dense FFN. Writing an MoE
//! path now would mean shipping an untested, unreachable graph — and MoE is
//! exactly where GwenLand has already been bitten: the `_exps` GGUF layout is
//! recorded in-repo as an **unverified assumption that can silently scramble
//! experts**.
//!
//! # What it will need
//!
//! Sketched here so the shape of the work is on record, not as a design:
//!
//! * **Router**: `x · w_router` → `[B, S, E]` logits, then top-k.
//! * **`stablehlo.sort` or `top_k`** for the k selection. ARTX14 §5.1 flags
//!   "does `top_k` lower well?" as an open question — it is a backend property,
//!   so P2 says measure it before depending on it.
//! * **Static capacity.** Variable tokens-per-expert is a dynamic shape, which
//!   P3 forbids. A fixed capacity factor (ARTX01 §4.4 suggests 1.25×) with
//!   `stablehlo.pad` and masking keeps it static, at the cost of dropped tokens
//!   past capacity — a correctness/throughput trade that needs a decision, not
//!   a default.
//! * **Batched expert GEMM**: one `dot_general` batched over the expert axis,
//!   `[E, cap, D] · [E, D, FFN_E]`.
//! * **Scatter-combine** back to `[B, S, D]`, weighted by the router scores.
//!
//! Multi-device expert parallel (`all_to_all`) is ARTX06, not this.

use crate::tensor::Tensor;

/// # Panics
/// Always. See the module documentation.
pub fn moe_ffn(_x: &Tensor, _router: &Tensor, _experts: &[Tensor]) -> Tensor {
    unimplemented!(
        "MoE FFN is not implemented — Qwen2-0.5B uses a dense FFN (ops/ffn.rs). \
         See the module docs in ops/moe.rs for what an implementation has to settle first."
    )
}
