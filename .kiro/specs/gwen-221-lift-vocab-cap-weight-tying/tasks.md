# Implementation Plan: GWEN-221 — Lift VOCAB_CAP via Weight Tying

## Overview

Four-wave implementation in Rust. Wave 1 adds `tie_word_embeddings: bool` to `TransformerConfig` and resolves it from explicit GGUF metadata or the standard no-output-head structure. Wave 2 removes `VOCAB_CAP` and adds an `Err` guard in `LayeredTrainingLoop::new`. Wave 3 wires `lm_head = model_embedding.t()` with no LoRA on embeddings. Wave 4 updates the dry-run reporting line and adds validation coverage.

## Tasks

- [x] 1. Wave 1 — Add `tie_word_embeddings` to `TransformerConfig` (`layer_loader.rs`)
  - [x] 1.1 Add `tie_word_embeddings: bool` field to `TransformerConfig` struct
    - Add field after `rope_theta` with doc comment referencing GWEN-221 Wave 1
    - _Requirements: 1.1_

  - [x] 1.2 Resolve `tie_word_embeddings` in `build_transformer_config`
    - Use explicit `MetadataValue::Bool` when present
    - When the key is absent, infer tying only if no standalone `output.weight`, `lm_head.weight`, or `model.lm_head.weight` tensor exists
    - Treat explicit false and non-Bool metadata as untied
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

  - [x] 1.3 Update `write_transformer_gguf_pub` test helper to write `tie_word_embeddings = true` KV
    - Increment the KV count in the GGUF header from `9` to `10`
    - Append the KV entry: key `"test.tie_word_embeddings"`, type `7` (u32 LE), value `0x01` (u8)
    - Apply the same change to the private `write_minimal_gguf` helper if it is used by Wave-1 tests
    - _Requirements: 1.1, 1.2_ (enables all existing tests to pass through `LayeredTrainingLoop::new` without hitting the Wave-2 guard)

  - [x]* 1.4 Write unit tests for `tie_word_embeddings` resolution (Property 1, Property 2)
    - **Property 1: `tie_word_embeddings` parsing round-trip** — build a GGUF with `MetadataValue::Bool(true)` at `test.tie_word_embeddings`; assert `TransformerConfig::tie_word_embeddings == true`
    - **Property 2: Structural fallback is conservative** — absent metadata with no output head infers true; absent metadata with a separate head, explicit false, and `U64(1)` produce false
    - **Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5**

- [x] 2. Wave 2 — Remove `VOCAB_CAP` and add `tie_word_embeddings` guard (`layered_training_loop.rs`)
  - [x] 2.1 Delete `const VOCAB_CAP: usize = 8192` and its doc comment
    - Remove the constant declaration and all references to `VOCAB_CAP` in the module
    - _Requirements: 2.1_

  - [x] 2.2 Add `tie_word_embeddings` guard in `LayeredTrainingLoop::new` and update `vocab` computation
    - Immediately after `let model_config = layer_loader.transformer_config().clone()`, insert: if `!model_config.tie_word_embeddings` return `Err(anyhow!(...))` with a message containing `"sampled-softmax"`
    - Change `let vocab = model_config.vocab_size.min(VOCAB_CAP).max(2)` to `let vocab = model_config.vocab_size.max(2)`
    - Update the `load_capped_matrix` call for `model_embedding` to pass `vocab` (now the full `vocab_size`) as the `rows` argument with context `"load full model embedding"`
    - _Requirements: 1.5, 2.2, 2.3, 4.6_

  - [x]* 2.3 Write unit tests for Wave-2 guard and full-vocab usage (Property 3, Property 4)
    - **Property 3: `new()` rejects all untied configs** — pass a GGUF with `tie_word_embeddings=false`; assert `Err` with message containing `"sampled-softmax"`
    - **Property 4: Full `vocab_size` used as effective vocabulary** — construct `LayeredTrainingLoop` with a tied GGUF of `vocab_size=16`; assert `ltl.vocab == 16`; assert token-ID modulo in `forward_backward_sample` uses `self.vocab`
    - **Validates: Requirements 1.5, 2.2, 2.3, 2.4, 4.6**

- [x] 3. Checkpoint — ensure all tests pass after Waves 1–2
  - Ensure all tests pass, ask the user if questions arise.

- [x] 4. Wave 3 — Tie `lm_head = model_embedding.t()` and freeze embeddings (`layered_training_loop.rs`)
  - [x] 4.1 Replace the conditional `lm_head` load block with an unconditional transpose
    - Remove the `if layer_loader.find_tensor(…).is_some() { load_capped_matrix(…) } else { model_embedding.clone() }` block
    - Replace with `let lm_head = model_embedding.t().context("transpose model_embedding for tied lm_head")?;`
    - Add a comment that any `output.weight` / `lm_head.weight` tensor in the GGUF is intentionally ignored
    - Change logits to `last_hidden.matmul(&self.lm_head)` because `lm_head` now has shape `[hidden, vocab]`
    - _Requirements: 3.1, 3.2, 2.5_

  - [x]* 4.2 Write unit tests for tied `lm_head` shape and no-LoRA invariant (Property 5, Property 6, Property 7)
    - **Property 5: `lm_head` is the transposed embedding — no separate buffer** — after `new()`, assert `lm_head.dims() == [hidden, vocab]` and numerically verify `lm_head` is the transpose of `model_embedding` using a small known-value GGUF
    - **Property 6: VarMap never contains tok_embeddings or lm_head adapters** — assert VarMap contains no keys starting with `"tok_embed"` or `"lm_head"`
    - **Property 7: Logits shape is `[1, vocab_size]`** — run one `forward_backward_sample` step with a tiny tied GGUF; assert the logits tensor has shape `[1, vocab_size]`
    - **Validates: Requirements 3.1, 3.2, 3.3, 3.5, 2.5**

- [x] 5. Wave 4 — Update dry-run reporting and add validation tests (`layered_training_loop.rs`)
  - [x] 5.1 Change dry-run `eprintln!` from `vocab(capped)=` to `vocab(full)=`
    - In the `if max_steps.is_some()` dry-run block, replace the `"[dry-run] vocab(capped)={} hidden={} layers={}"` format string with `"[dry-run] vocab(full)={} hidden={} layers={}"`
    - _Requirements: 4.2, 5.1, 5.3_

  - [x]* 5.2 Write unit tests for dry-run stderr output (Property 8, Property 9)
    - **Property 8: Dry-run reports `vocab(full)=<vocab_size>`** — run with `max_steps=Some(1)` against a tiny tied GGUF; capture stderr; assert contains `"vocab(full)="` and does NOT contain `"vocab(capped)="`
    - **Property 9: Dry-run stderr contains all mandatory fields** — assert same stderr output also contains `"hidden="`, `"layers="`, `"trainable params="`, `"RSS"`, and `"loss="`
    - **Validates: Requirements 4.2, 5.1, 5.3**

  - [x]* 5.3 Write unit test for loss validity invariant (Property 10)
    - **Property 10: Loss remains finite and non-negative for the first 50 steps** — run 50 optimizer steps with a freshly initialized LoRA adapter against the tiny tied GGUF and assert every CE value is finite and at least zero
    - **Validates: Requirements 4.5**

- [ ] 6. Final checkpoint — ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.
  - The official Qwen3-1.7B Q8_0 GGUF omits both the metadata key and a separate
    output-head tensor. Structural tying support now allows this standard GGUF;
    the final harness must pass with the corrected pretrained-loss and RSS gates.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP; the core correctness is enforced by the non-optional tasks
- Task 1.3 (updating the test GGUF helper) is non-optional because fixtures with a separate output head must explicitly opt into tying to pass the guard
- The `load_capped_matrix` helper itself does NOT change — only the `rows` argument changes from `min(vocab_size, VOCAB_CAP)` to `vocab_size`
- The tied logit computation is `last_hidden.matmul(&self.lm_head)`: `model_embedding` is `[vocab, hidden]`, so `lm_head = model_embedding.t()` is `[hidden, vocab]`
- For stderr capture in tests, use `gag` or redirect stderr in a child thread; alternatively, expose a `dry_run_report()` method that returns a `String` to avoid raw stderr capture
- The `write_transformer_gguf_pub` KV count change (9 → 10) is a single `u64` write at a fixed offset in the helper; update both the `pub` variant and the private `write_transformer_gguf` copy if it exists separately

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2", "1.3"] },
    { "id": 2, "tasks": ["1.4", "2.1"] },
    { "id": 3, "tasks": ["2.2"] },
    { "id": 4, "tasks": ["2.3", "4.1"] },
    { "id": 5, "tasks": ["4.2", "5.1"] },
    { "id": 6, "tasks": ["5.2", "5.3"] }
  ]
}
```
