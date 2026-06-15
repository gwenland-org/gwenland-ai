# GWEN-220 Wave 3 Audit

Date: 2026-06-14

## Scope

Wave 3 implements the MLP sub-block, assembles the full transformer layer, and
replaces the synthetic mean-pool/layer-signature training path for every loaded
layer. It also completes the bounded-memory integration identified by the Wave 1
audit: real GGUF model tensors, layer-to-layer hidden propagation, and reverse
recomputation.

The pre-existing GWEN-219 worktree changes were preserved. Wave 4 model
validation, RAM measurement, and adapter merge/inference round trip were not
started.

## Implementation

`train/transformer_layer.rs` now provides:

- MLP weights and projection-matched gate/up/down LoRA references
- RMSNorm -> gate/up projections -> SiLU-gated product -> down projection
- MLP residual connection
- full attention-plus-MLP transformer layer assembly

`convert/gguf_parser.rs` and `train/layer_loader.rs` now provide:

- header-only GGUF parsing with retained scalar architecture metadata
- runtime transformer configuration for hidden size, heads, KV heads,
  intermediate size, vocab size, RMS epsilon, and RoPE base
- mmap-backed descriptors for layer and non-layer tensors
- selective loading of capped embedding and output-head rows
- GGUF-to-Candle matrix dimension reversal

`train/layered_training_loop.rs` now:

- uses the model's frozen embedding, final RMSNorm, and output head
- propagates `[batch, seq, hidden]` states through every transformer layer
- applies each of the seven GWEN-219 LoRA adapters only at its matching linear
- processes samples independently so padding cannot affect causal attention
- computes the final-token language-model objective
- stores detached layer-boundary activations
- walks layers in reverse and recomputes one layer at a time for exact gradients
- keeps only LoRA tensors trainable and checkpointed

The GWEN-216 integration fixture was upgraded from a synthetic two-tensor GGUF
to a tiny complete transformer GGUF. Its loaded-layer lifetime assertion now
runs on Windows as well as Unix.

## Wave 3 Tests

Focused coverage verifies:

- full-layer output shape `[2, 3, 8]`, changed values, and finite values
- finite gradients for input and all seven LoRA A/B adapter pairs
- exact projection-kind discovery and adapter routing
- metadata-derived transformer configuration
- hidden-state propagation across all layers
- reverse-recomputation loss and gradients against an ordinary full graph
  within `1e-5`
- finite end-to-end layered-training loss
- exactly one `LoadedLayer` object live at a time

## Gate Results

- `cargo test -p gwenland-core train::transformer_layer::tests --lib`:
  4 passed
- `cargo test -p gwenland-core train::layer_loader::tests --lib`:
  15 passed
- `cargo test -p gwenland-core train::layered_training_loop::tests --lib`:
  16 passed
- `cargo test -p gwenland-core --features test-utils --test gwen216_integration`:
  3 passed
- `cargo build`: passed
- `cargo build -p gwenland-core --features candle-backend`: passed
- `cargo test -p gwenland-core --features candle-backend --lib`:
  310 passed, 0 failed
- `git diff --check`: passed

The default-feature core suite produced 251 passes and the same three selector
failures documented in Wave 2 because no inference backend is compiled:

- `empty_stop_sequences_ok`
- `relative_gguf_ok`
- `tilde_expand`

All three pass in the backend-enabled 310-test gate. Existing compiler warnings
remain.

## Gate Outcome

Wave 3 is complete. Stop here before Wave 4.
