# GWEN-220 Wave 2 Attention Block

Date: 2026-06-14

## Scope

Wave 2 implements the autograd-compatible attention sub-block only. It does not
replace the mean-pool path in `LayeredTrainingLoop`; full layer assembly and
integration remain Wave 3 work.

The pre-existing GWEN-219 worktree changes were preserved.

## Implementation

Added `engine/transformer_ops.rs` with shared Candle tensor operations:

- RMSNorm over the final dimension
- absolute-position RoPE tables and Q/K rotation
- grouped-query K/V head expansion
- causal masking
- scaled dot-product attention

The attention softmax uses `candle_nn::ops::softmax` on the final dimension.
The convenience `softmax_last_dim` custom op was rejected because it does not
provide a backward implementation in the current Candle version.

Added `train/transformer_layer.rs` with:

- validated attention architecture configuration
- frozen attention weight references, including optional Qwen3 Q/K norms
- projection-matched Q/K/V/O LoRA adapters
- a batched linear-plus-LoRA helper
- `attention_forward` for:
  RMSNorm -> Q/K/V -> Q/K norm -> RoPE -> GQA -> causal attention ->
  output projection -> residual

The function accepts and returns `[batch, seq_len, hidden_size]` and supports an
absolute position offset.

## Tests

Six focused tests cover:

- RoPE shape, position-dependent values, and finite gradients
- GQA expansion from two K/V heads to four query heads
- causal masking against future-token influence
- a known LoRA projection delta
- attention output shape `[2, 3, 8]`
- finite output and gradients through input, Q LoRA, and O LoRA
- invalid GQA configuration rejection

## Gate Results

- `cargo test -p gwenland-core transformer --lib`: 6 passed
- `cargo build`: passed
- `cargo build -p gwenland-core --features candle-backend`: passed
- `cargo test -p gwenland-core --features candle-backend --lib`:
  306 passed, 0 failed

The default-feature core suite produced 247 passes and the same three
selector failures caused by running backend-selection tests without a compiled
backend:

- `empty_stop_sequences_ok`
- `relative_gguf_ok`
- `tilde_expand`

Those tests pass in the backend-enabled 306-test gate above. Existing compiler
warnings remain unchanged.

## Gate Outcome

Wave 2 is complete. Stop here before Wave 3.
