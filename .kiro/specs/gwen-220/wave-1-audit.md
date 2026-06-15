# GWEN-220 Wave 1 Audit and Extraction Plan

Date: 2026-06-14

## Gate Scope

Wave 1 is audit-only. No Rust source was changed. The current dirty worktree is
treated as the GWEN-219 baseline and must not be reverted.

The checked-out branch is `feature/gwen-217-loss-fix`, not the branch named in
the spec. Because the worktree contains uncommitted GWEN-219 changes, this wave
does not switch branches.

## Audit Findings

### Inference reference

- `engine/inference/candle_ggqr/forward.rs` contains usable Candle formulas for
  RMSNorm, Q/K/V projection layout, GQA head expansion, scaled dot-product
  attention, SwiGLU, and residual ordering.
- The current reference attention does **not** implement RoPE or a causal mask.
  `rope_theta` is parsed but unused. RoPE and causal masking must therefore be
  implemented and tested for GWEN-220 rather than copied from this file.
- The inference path is 2-D (`[seq, hidden]`) and inference-error-specific.
  Training needs a batched `[batch, seq, hidden]` wrapper and `anyhow`/Candle
  errors.
- Qwen3 layer tensors include optional `attn_q_norm` and `attn_k_norm` weights.
  A faithful Qwen3 path should apply these per-head Q/K RMS norms when present,
  while remaining compatible with Llama-style layers where they are absent.
- `candle_ggqr` is feature-gated behind `candle-backend`; training is compiled
  without that feature. Training cannot directly depend on its `ModelConfig`.
- The existing candle GGUF metadata reader checks the wrong real-file magic
  value and only whitelists `llama`, `qwen2`, and `phi3`. It is not a safe
  metadata dependency for the Qwen3 regression model in its current form.

### Layered training and LoRA routing

- GWEN-219 creates persistent adapters under:
  `l{N}.attn_q`, `attn_k`, `attn_v`, `attn_o`, `ffn_gate`, `ffn_up`, and
  `ffn_down`.
- Each adapter has native projection dimensions and scaling `alpha / rank`.
  These namespaces and dimensions are already understood by adapter export and
  merge and must remain unchanged.
- The current forward embeds tokens, mean-pools the sequence, adds a synthetic
  layer signature, averages all projection LoRA deltas, and applies a random
  trainable head. The replacement boundary is the layer-signature construction
  plus `LayeredTrainingLoop::forward`.
- In the real block, each LoRA delta must be added only at its matching linear:
  `base(x) + scale * B(A(x))`.
- The current loop is layer-major. A real transformer forward must become
  batch-major so hidden states flow through layer 0 through layer N before the
  LM head and loss.

### Model metadata

The required metadata exists in GGUF under architecture-prefixed keys:

- `block_count`
- `embedding_length`
- `attention.head_count`
- `attention.head_count_kv`
- `feed_forward_length`
- `attention.layer_norm_rms_epsilon`
- `rope.freq_base`

`head_dim` is not stored in the current `ModelConfig`; derive it as
`hidden_size / n_heads` and validate divisibility. Also validate
`n_heads % n_kv_heads == 0`.

The general `convert::gguf_parser` currently skips KV values and eagerly reads
all tensor bytes before `LayerLoader` drops the parsed file. GWEN-220 needs a
header-only metadata/index path so opening the layered loader does not
temporarily materialize the full quantized model.

### Autograd and memory

A normal Candle graph cannot simply chain all dequantized layers. Candle
`Op::Matmul` retains both operands and needs the weight to compute the input
gradient, so all F32 base weights would remain live until the final backward.
That violates the one-layer-in-RAM requirement.

Use exact reverse recomputation:

1. Forward through layers one at a time with detached boundary activations.
2. Save only layer-boundary hidden states, not dequantized weights or full
   layer graphs.
3. Compute the LM-head loss and its gradient at the final boundary.
4. Walk layers in reverse. Reload one layer, recreate its input as a temporary
   `Var`, recompute that layer with its persistent LoRA Vars, and backpropagate
   the scalar vector-Jacobian product `(layer_output * upstream_grad).sum_all()`.
5. Accumulate the resulting LoRA gradients and pass the temporary input
   gradient to the preceding layer.
6. Apply one AdamW step only after all layers for the accumulation window have
   contributed.

This preserves the full transformer gradient while bounding resident base
weights to one layer.

## Extraction and Implementation Plan

### Shared, unconditional model metadata

Extend the general GGUF parser with a header-only parse result that retains
typed KV metadata and tensor descriptors without reading tensor payloads.
Build an unconditional `TransformerConfig` from it. `LayerLoader` stores this
config and exposes it to training.

Do not modify `GgqrCandleBackend` or rewire its forward path in this ticket.

### Shared transformer math

Add an unconditional `engine/transformer_ops.rs` containing batched,
autograd-compatible primitives:

- RMSNorm over the last dimension
- optional per-head Q/K RMSNorm
- RoPE table construction and Q/K rotation with absolute positions
- GQA K/V expansion with divisibility checks
- causal mask construction
- scaled dot-product attention
- SwiGLU

Keep these functions independent of tensor-name lookup, layer loading, LoRA,
and inference-specific errors. The inference backend can adopt them later
without changing their contracts, but remains read-only during GWEN-220.

### Training-specific layer wrapper

Add `train/transformer_layer.rs` with:

- a typed set of one layer's dequantized weights and optional Q/K norms
- tensor-name resolution for both `blk.N.*` and HF-style names
- a linear helper that applies the matching GWEN-219 LoRA adapter
- `attention_forward`
- `mlp_forward`
- `transformer_layer_forward`

Wave 2 implements and tests the attention half. Wave 3 adds MLP, full layer
assembly, final norm/head handling, and the reverse-recomputation scheduler.

### Objective and fixed model tensors

Preserve `VOCAB_CAP` because lifting it is out of scope. Replace synthetic
layer signatures immediately. Prefer the model's real embedding, final norm,
and capped output-head rows through selective tensor loading; do not export or
train synthetic `tok_embed`/`lm_head` parameters as part of the adapter.

The first implementation may keep the existing final-token objective per
sample, taking the last valid sequence position from final hidden states. A
padding mask or valid lengths must prevent padded tokens from affecting
attention.

## Wave Gates

### Wave 2

- Attention output is `[batch, seq, hidden]`.
- RoPE changes positions while preserving shape.
- Causal masking prevents future-token influence.
- GQA test covers `n_heads > n_kv_heads`.
- All outputs and gradients are finite.
- Existing tests and both default/candle-feature builds pass.

### Wave 3

- Full layer output is `[batch, seq, hidden]` and finite.
- Every one of the seven LoRA adapters affects only its intended projection.
- Layer-to-layer hidden propagation replaces mean-pooling and layer signatures.
- Reverse recomputation matches a small ordinary full-graph reference within
  tolerance.
- At most one loaded/dequantized base layer is live at a time.

### Wave 4

- Run the env-gated Qwen3-1.7B Q8_0 dry-run.
- Record loss and peak RSS including loader construction.
- Run a short train, adapter export, merge, and GGQR-Candle load/inference
  round trip.
- Run the full existing test suite with the relevant feature sets.

## Known Risks to Resolve Before Wave 4

- Qwen3 Q/K norm behavior must be confirmed against the model tensor shapes.
- Selective loading of capped embedding/output rows must respect GGUF dimension
  order and quantization block boundaries.
- CPU attention is quadratic in sequence length. The existing 1024-token
  default may be too slow or memory-heavy on the target i3; validation should
  start with a short sequence and then measure the configured limit.
