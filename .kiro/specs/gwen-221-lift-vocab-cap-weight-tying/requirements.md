# Requirements Document

## Introduction

GWEN-221 lifts the `VOCAB_CAP=8192` guard in `layered_training_loop.rs` by exploiting Qwen3's tied token embedding and output projection. The full `[vocab=151936, hidden=2048]` f32 embedding costs about 1.16 GiB, but weight tying avoids allocating a second matrix of the same size for `lm_head`. The output head is a transposed view of the resident embedding, so lifting the cap adds the full embedding cost but no additional output-head buffer.

The implementation proceeds in four waves:

1. **Wave 1** — Resolve weight tying from explicit GGUF metadata or the standard structural form where no standalone output head exists.
2. **Wave 2** — Remove `VOCAB_CAP=8192`; read the full `vocab_size` from the same KV path used for `n_heads`, `hidden_size`, etc. (GWEN-220 pattern).
3. **Wave 3** — Wire the tied `lm_head` as a transposed view of `tok_embeddings` (no new buffer, no LoRA adapter on the embedding).
4. **Wave 4** — Validate with a dry-run and a short real run; confirm peak RSS stays within budget and CE loss remains finite, non-negative, and stable.

Fallback: explicit `false`, malformed metadata, or an absent key combined with a standalone output head parks the spec and directs the unsupported case to sampled softmax.

---

## Glossary

- **VOCAB_CAP**: The compile-time constant `8192` in `layered_training_loop.rs` that caps the vocabulary dimension used for embeddings and the output head.
- **LayeredTrainingLoop**: The struct in `train/layered_training_loop.rs` that orchestrates bounded-memory LoRA training one transformer layer at a time.
- **LayerLoader**: The struct in `train/layer_loader.rs` that opens the GGUF with `LoadMode::Lazy` and provides per-layer zero-copy tensor slices.
- **TransformerConfig**: The struct in `train/layer_loader.rs` populated from GGUF KV metadata; contains `vocab_size`, `hidden_size`, `n_heads`, etc.
- **GgufHeader**: The struct in `convert/gguf_parser.rs` holding parsed GGUF scalar KV metadata as a `HashMap<String, MetadataValue>`.
- **MetadataValue**: The enum in `convert/gguf_parser.rs` representing scalar GGUF KV values (`U64`, `Bool`, `String`, `F64`, etc.).
- **tie_word_embeddings**: The effective weight-tying decision. Explicit Boolean GGUF metadata takes precedence; when the key is absent, no standalone `output.weight` / `lm_head.weight` tensor is the standard structural signal for tied embeddings.
- **tok_embeddings**: The model's token embedding matrix; loaded from `token_embd.weight` (llama.cpp) or `model.embed_tokens.weight` (HF) in the GGUF data section.
- **lm_head**: The output projection matrix used to compute next-token logits. When `tie_word_embeddings=true`, this is a transposed view of `tok_embeddings`, not an independent tensor.
- **weight tying**: The technique of sharing a single weight buffer between `tok_embeddings` and `lm_head` (as its transpose), eliminating the ~1.16 GiB cost of a standalone output head.
- **load_matrix_rows**: A helper in `layered_training_loop.rs` that loads a requested number of matrix rows before building the Candle `Tensor`.
- **dry-run**: A single-step training run (`max_steps=1`) used to verify memory usage and loss without committing to a full training epoch.
- **CE loss**: Cross-entropy loss over the full vocabulary. Because LoRA B starts at zero and the embedding/head are frozen pretrained weights, step 1 measures the pretrained model rather than a uniform random-logit baseline.
- **peak RSS**: Peak resident set size in megabytes, measured via `/proc/self/status` on Linux or equivalent. Bounded to the i3 8 GB budget.

---

## Requirements

### Requirement 1: Wave 1 — Resolve Weight Tying from GGUF Metadata or Structure

**User Story:** As a training engineer, I want the system to recognize both explicit weight-tying metadata and the standard GGUF omission of a separate output head, so valid converted models can use the full vocabulary without weakening the GWEN-216 memory-safety invariant.

#### Acceptance Criteria

1. THE `TransformerConfig` SHALL include a `tie_word_embeddings: bool` field resolved by `build_transformer_config` from GGUF metadata and tensor descriptors.

2. WHEN the GGUF KV entry `<arch>.tie_word_embeddings` is present and its `MetadataValue::Bool` equals `true`, THE `LayerLoader` SHALL set `TransformerConfig::tie_word_embeddings` to `true`.

3. WHEN the GGUF KV entry `<arch>.tie_word_embeddings` is absent and no standalone `output.weight`, `lm_head.weight`, or `model.lm_head.weight` tensor exists, THE `LayerLoader` SHALL infer tied embeddings and set `TransformerConfig::tie_word_embeddings` to `true`.

4. WHEN the GGUF KV entry is absent but a standalone output-head tensor exists, or the entry is present with a non-Boolean value, THE `LayerLoader` SHALL set `TransformerConfig::tie_word_embeddings` to `false`.

5. WHEN the GGUF explicitly declares `tie_word_embeddings=false`, THE explicit value SHALL override structural inference.

6. WHEN `LayeredTrainingLoop::new` is called with a `LayerLoader` whose effective `TransformerConfig::tie_word_embeddings` is `false`, THE `LayeredTrainingLoop` SHALL return `Err` describing both accepted tying signals and directing the caller to a sampled-softmax follow-up.

---

### Requirement 2: Wave 2 — Remove VOCAB_CAP and Read Full `vocab_size`

**User Story:** As a training engineer, I want the vocabulary dimension to reflect the model's actual `vocab_size` from the GGUF rather than the hardcoded 8192 cap, so that training operates on the complete token distribution.

#### Acceptance Criteria

1. THE `layered_training_loop.rs` module SHALL NOT contain the constant `const VOCAB_CAP: usize = 8192`.

2. WHEN `LayeredTrainingLoop::new` is called and `TransformerConfig::tie_word_embeddings` is `true`, THE `LayeredTrainingLoop` SHALL use `TransformerConfig::vocab_size` as the effective vocabulary dimension without truncation.

3. THE `LayeredTrainingLoop` SHALL set `self.vocab` to `TransformerConfig::vocab_size.max(2)` when `tie_word_embeddings` is `true`.

4. WHEN `forward_backward_sample` maps token IDs to embedding rows, THE `LayeredTrainingLoop` SHALL use `self.vocab` as the modulo guard, reflecting the full vocabulary size.

5. THE `LayeredTrainingLoop` SHALL NOT allocate a standalone output head tensor of shape `(vocab_size, hidden_size)` in addition to the embedding tensor when `tie_word_embeddings` is `true`.

---

### Requirement 3: Wave 3 — Tied `lm_head` as Transposed View

**User Story:** As a training engineer, I want the output head to be a transposed view of the already-loaded token embedding matrix when weight tying is active, so that the full vocabulary does not require a second 1.16 GiB output buffer.

#### Acceptance Criteria

1. WHEN `TransformerConfig::tie_word_embeddings` is `true`, THE `LayeredTrainingLoop` SHALL derive `lm_head` by calling `.t()` on the `model_embedding` tensor rather than loading a separate output weight tensor from the GGUF.

2. WHEN `TransformerConfig::tie_word_embeddings` is `true` and the GGUF also contains an `output.weight` or `lm_head.weight` tensor, THE `LayeredTrainingLoop` SHALL ignore that tensor and use the transposed `model_embedding` as `lm_head`.

3. THE `LayeredTrainingLoop` SHALL NOT register a LoRA adapter on `tok_embeddings` or `lm_head`; both SHALL remain frozen base tensors throughout training.

4. WHEN `TransformerConfig::tie_word_embeddings` is `false`, THE `LayeredTrainingLoop` SHALL return `Err` before loading an output head.

5. THE `LayeredTrainingLoop` SHALL produce logits of shape `(1, vocab_size)` from the tied `lm_head`, where `vocab_size` equals `TransformerConfig::vocab_size`.

---

### Requirement 4: Wave 4 — Validation Invariants

**User Story:** As a training engineer, I want the dry-run and short real-run to confirm that memory and loss behave as expected after lifting the cap, so that I can be confident no regression was introduced.

#### Acceptance Criteria

1. WHEN a dry-run (`max_steps=1`) is executed against the structurally tied Qwen3-1.7B Q8_0 GGUF, THE `LayeredTrainingLoop` SHALL complete without an out-of-memory error on a system with 8 GB of physical RAM.

2. WHEN a dry-run completes, THE `LayeredTrainingLoop` SHALL emit a `[dry-run]` stderr line reporting `vocab(full)=<vocab_size>` (not the capped value) and peak RSS in MB.

3. WHEN a dry-run completes with Qwen3-1.7B Q8_0, THE reported RSS baseline SHALL be sampled before the full embedding is loaded, and peak RSS SHALL remain below 2.5 GB on the reference CPU run.

4. WHEN a single-step real run begins with a freshly initialized LoRA adapter against the full Qwen3-1.7B vocabulary, THE `LayeredTrainingLoop` SHALL report a finite, non-negative CE loss. THE validation SHALL NOT require `ln(vocab)` because zero-initialized LoRA B preserves the pretrained model's non-uniform logits at step 1.

5. WHEN a short real run completes multiple optimizer steps, THE `LayeredTrainingLoop` SHALL keep CE finite and non-negative, SHALL reject numerical explosion above `100.0`, and SHALL NOT show a last-third mean more than `0.5` above the first-third mean.

6. IF neither explicit true metadata nor the absence of a standalone output head establishes weight tying, THEN THE `LayeredTrainingLoop` SHALL return `Err` and validation SHALL halt rather than silently proceeding with a capped vocabulary.

---

### Requirement 5: Dry-Run Reporting Update

**User Story:** As a training engineer, I want the dry-run output to clearly distinguish between the capped and full vocabulary modes, so that I can confirm at a glance which code path was exercised.

#### Acceptance Criteria

1. WHEN `max_steps` is `Some(_)` and `tie_word_embeddings` is `true`, THE `LayeredTrainingLoop` SHALL emit a `[dry-run]` stderr line containing `vocab(full)=<vocab_size>` in place of the previous `vocab(capped)=<value>` line.

2. WHEN `max_steps` is `Some(_)` and `tie_word_embeddings` is `false` (fallback path, should be unreachable post-GWEN-221 guard), THE `LayeredTrainingLoop` SHALL have already returned `Err` before reaching the dry-run reporting block.

3. THE `LayeredTrainingLoop` dry-run stderr block SHALL continue to report `hidden=<hidden_size>`, `layers=<num_layers>`, trainable param count, peak RSS, and step 1 loss in the same format as the pre-GWEN-221 baseline.
