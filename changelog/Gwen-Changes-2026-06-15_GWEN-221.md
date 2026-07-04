# GwenLand — GWEN-221: Lift `VOCAB_CAP` via Weight Tying

**Date:** 2026-06-15 (WIB)
**Scope:** `train/layer_loader.rs`, `train/layered_training_loop.rs`, `tests/gwen220_wave4.rs`, `.kiro/specs/gwen-221-lift-vocab-cap-weight-tying/tasks.md`
**Type:** Correctness change delivered in four gated waves: parse the GGUF weight-tying contract, remove the 8,192-token training cap, derive the LM head from the full embedding matrix, and strengthen dry-run/loss validation.
**Status:** ✅ Waves 1–4 implemented and automated tests green. ⚠️ Final real-model RSS/loss measurement remains open because the available Qwen3-1.7B Q8_0 GGUF does not declare `tie_word_embeddings=true`; the new guard correctly rejects it before training.

---

## Executive Summary

Before GWEN-221, `LayeredTrainingLoop` restricted the effective vocabulary to
`8,192` entries:

```rust
const VOCAB_CAP: usize = 8192;
let vocab = model_config.vocab_size.min(VOCAB_CAP).max(2);
```

That cap reduced resident embedding/head memory, but it also changed the
language-model objective. Token IDs were reduced modulo the capped vocabulary,
so different full-vocabulary tokens could alias to the same class. The output
head also operated over only 8,192 logits instead of the model's actual
vocabulary. For Qwen3-1.7B, whose vocabulary is approximately 151,936 entries,
the resulting cross-entropy was not measuring the intended full-vocabulary
next-token task.

This became especially important after GWEN-220. The real-attention forward
produced a surprisingly low step-1 loss of `2.7747`; rather than accepting that
number as proof of correctness, GWEN-221 traced the objective and identified the
remaining vocabulary cap and modulo aliasing as a correctness risk.

GWEN-221 removes the cap only when the GGUF explicitly declares that token
embeddings and the output projection are tied:

```text
<architecture>.tie_word_embeddings = true
```

For eligible models:

- the complete embedding matrix is loaded;
- `self.vocab` is the full GGUF vocabulary size;
- token modulo uses the full vocabulary rather than 8,192;
- `lm_head` is a transposed view of `model_embedding`;
- any separate `output.weight` / `lm_head.weight` tensor is deliberately ignored;
- embedding and LM-head weights stay frozen and receive no LoRA adapters;
- logits have shape `[1, full_vocab]`.

For absent, malformed, false, or non-Boolean metadata, construction fails with
an explicit diagnostic directing the unsupported case to a sampled-softmax
follow-up. The code does not silently return to the capped objective.

---

## Why This Change Was Necessary

The old cap was useful as an early memory boundary, but it imposed three
semantic changes:

1. **Token aliasing.** Input and target IDs were reduced with `% self.vocab`.
   When `self.vocab == 8192`, every full-vocabulary token was mapped into one of
   only 8,192 classes.
2. **Capped logits.** Cross-entropy saw `[1, 8192]` logits rather than
   `[1, model_vocab_size]`.
3. **Potentially separate output weights.** The loop loaded `output.weight` when
   present, otherwise cloned the embedding. That was not a strict weight-tying
   contract and could allocate another large dequantized matrix.

Weight tying solves the third problem while making the first two removable:
the full embedding matrix serves both input lookup and output projection, so
there is one underlying frozen weight buffer rather than two full-vocabulary
buffers.

The safety condition is explicit metadata. Inferring tying from tensor names or
from the absence of `output.weight` would be ambiguous, so GWEN-221 treats only
the Boolean value `true` as authorization to use the tied full-vocabulary path.

---

## What Changed

### Wave 1 — Parse the GGUF weight-tying contract

`TransformerConfig` in `train/layer_loader.rs` now contains:

```rust
pub tie_word_embeddings: bool,
```

`build_transformer_config` reads
`<architecture>.tie_word_embeddings` with strict semantics:

- `MetadataValue::Bool(true)` → `true`;
- `MetadataValue::Bool(false)` → `false`;
- absent key → `false`;
- non-Boolean values such as `U64(1)` → `false`.

This is intentionally conservative. A malformed or missing metadata value does
not gain full-vocabulary behavior by accident.

The public transformer GGUF fixture was updated to emit
`test.tie_word_embeddings = true`, including the correct KV count. A configurable
fixture helper supports true, false, absent, and non-Boolean cases.

Wave-1 tests cover:

- Boolean true/false round-trip;
- absent metadata defaulting to false;
- non-Boolean metadata defaulting to false.

### Wave 2 — Remove `VOCAB_CAP` and guard unsupported models

`VOCAB_CAP` and every reference to it were removed from
`train/layered_training_loop.rs`.

`LayeredTrainingLoop::new` now checks the parsed contract before allocating the
full embedding:

```rust
if !model_config.tie_word_embeddings {
    return Err(anyhow!(
        "GGUF '{}' does not declare tie_word_embeddings=true; \
         full-vocabulary training without weight tying is unsupported. \
         Open a sampled-softmax follow-up.",
        gguf_path.display()
    ));
}
```

For an eligible GGUF:

```rust
let vocab = model_config.vocab_size.max(2);
```

The embedding loader receives this full row count. The helper was renamed from
`load_capped_matrix` to `load_matrix_rows` because the old name described a
policy that no longer exists. Its errors and comments were also rewritten to
describe tensor rows rather than capped tensors.

Wave-2 tests prove:

- untied GGUFs return `Err`;
- the error contains both `tie_word_embeddings=true` and `sampled-softmax`;
- a tied fixture uses all 16 vocabulary rows;
- token IDs separated by exactly `self.vocab` map identically, proving modulo is
  based on the full effective vocabulary rather than the deleted 8,192 cap.

### Wave 3 — Tie the LM head to the embedding

The conditional output-head loader was removed. `lm_head` is now always derived
from the frozen embedding:

```rust
let lm_head = model_embedding
    .t()
    .context("transpose model_embedding for tied lm_head")?;
```

The shape contract is:

```text
model_embedding: [vocab, hidden]
lm_head:         [hidden, vocab]
last_hidden:     [1, hidden]
logits:          [1, vocab]
```

Therefore the correct projection is:

```rust
last_hidden.matmul(&self.lm_head)
```

During implementation, this exposed an important orientation bug: keeping the
old `.matmul(&self.lm_head.t()?)` would transpose the already-transposed head
back to `[vocab, hidden]` and make the tied path mathematically wrong. The
projection was extracted into `logits_from_last_hidden` so its shape contract is
directly testable.

Any GGUF `output.weight` or `lm_head.weight` tensor is intentionally ignored once
the metadata contract has approved weight tying. This prevents a separate output
matrix from silently replacing or duplicating the tied head.

Wave-3 tests verify:

- `lm_head.dims() == [hidden, vocab]`;
- every head value equals the corresponding transposed embedding value;
- embedding and head share the same Candle storage;
- a deliberately different fixture `output.weight` is ignored;
- no `tok_embed*` or `lm_head*` LoRA keys enter the `VarMap`;
- tied logits have shape `[1, vocab_size]`.

### Wave 4 — Reporting and validation invariants

The dry-run report now says:

```text
[dry-run] vocab(full)=... hidden=... layers=...
```

instead of:

```text
[dry-run] vocab(capped)=...
```

Rather than capture process-wide stderr with a global redirect, the reporting
boundary was made deterministic and testable:

```rust
struct DryRunReport { ... }
impl fmt::Display for DryRunReport { ... }
```

`LayeredTrainingLoop::run` builds the report and prints it, while unit tests can
render the same value to a string. This preserves operator-facing output without
adding a fragile test-only dependency or global stderr contention.

The report retains all required fields:

- `vocab(full)`;
- hidden size;
- layer count;
- trainable parameter count;
- RSS start, peak, and delta;
- step-1 loss;
- elapsed time;
- clean one-step completion message.

A real 50-optimizer-step unit test was also added. Each iteration performs
forward/backward, gradient clipping, and AdamW update against the tiny tied GGUF,
then asserts:

```text
loss >= 1.0
```

for every one of the first 50 steps. This catches the tied-head collapse failure
mode without requiring a multi-hour real-model CPU run in the normal test suite.

---

## Clean-Code Pass

The final implementation received a focused readability pass rather than only
the minimum mechanical edits:

- `load_capped_matrix` became `load_matrix_rows`;
- capped-vocabulary comments and error messages were removed;
- `TransformerConfig::tie_word_embeddings` documents the strict false default;
- the tied head field documents that it is a transposed view with no independent
  weight buffer;
- the constructor explains the full-vocabulary and frozen-weight invariants;
- the deliberate ignoring of separate output tensors is documented beside the
  decision;
- logits projection has a named helper and contextual error;
- dry-run formatting is separated from execution via `DryRunReport`;
- comments explain decisions and invariants rather than narrating obvious
  assignments.

The result keeps the implementation local to the loader/training boundaries and
does not introduce a new abstraction outside the behavior GWEN-221 requires.

---

## Property Coverage

All ten properties from the GWEN-221 plan are represented:

| Property | Invariant | Result |
|---|---|---|
| 1 | Boolean weight-tying metadata round-trips | ✅ |
| 2 | Missing/non-Boolean metadata defaults false | ✅ |
| 3 | Untied configurations are rejected | ✅ |
| 4 | Effective vocabulary equals full GGUF vocabulary | ✅ |
| 5 | LM head is the embedding transpose with shared storage | ✅ |
| 6 | Embedding/head never receive LoRA adapters | ✅ |
| 7 | Logits are `[1, vocab_size]` | ✅ |
| 8 | Dry-run reports `vocab(full)` and not `vocab(capped)` | ✅ |
| 9 | Dry-run preserves all mandatory fields | ✅ |
| 10 | Loss does not fall below 1.0 in the first 50 steps | ✅ |

---

## Real-Model Harness Update

The existing env-gated `tests/gwen220_wave4.rs` still encoded the capped
objective:

```text
loss < 9.5       # ln(8192) + margin
```

That assertion would reject the expected full-vocabulary initialization near:

```text
ln(151936) ≈ 11.93
```

The harness now checks:

- first-step loss is within `[10.0, 14.0]`;
- every observed early loss stays within `[1.0, 14.0]`;
- all losses remain finite;
- the short-run trend does not diverge upward.

This removes the old `ln(8192)` assumption and turns the harness into a direct
GWEN-221 regression gate.

---

## Validation

### Automated suites

The following verification completed successfully:

```text
cargo test -p gwenland-core --features candle-backend --lib
→ 334 passed, 0 failed

cargo test -p gwenland-core train::layered_training_loop::tests --lib
→ 26 passed, 0 failed

cargo test -p gwenland-core --features test-utils --test gwen216_integration
→ 3 passed, 0 failed

cargo test -p gwenland-core --features test-utils --test gwen222_e2e
→ 4 passed, 0 failed

cargo test -p gwenland-core --test gwen220_wave4
→ 1 passed (env-gated skip without a model)

git diff --check
→ clean
```

The first parallel integration attempt encountered Cargo artifact-directory lock
contention and timed out while commands waited on the shared target directory.
The affected suites were rerun sequentially with a warm build and passed. This
was build orchestration contention, not a test failure.

### Real Qwen3-1.7B Q8_0 attempt

The real-model command was run with:

```text
GWEN_DRYRUN_GGUF=C:\Users\reyha\Downloads\Qwen3-1.7B-Q8_0.gguf
GWEN220_STEPS=1
cargo test -p gwenland-core --test gwen220_wave4 -- --nocapture
```

The first sandboxed attempt stopped while resolving the Hugging Face tokenizer.
After network access was allowed, the harness loaded 98 real dataset samples,
opened the GGUF, and reached the GWEN-221 constructor guard.

It halted with the intended diagnostic:

```text
GGUF '...\Qwen3-1.7B-Q8_0.gguf' does not declare
tie_word_embeddings=true; full-vocabulary training without weight tying is
unsupported. Open a sampled-softmax follow-up.
```

This is the required behavior for Requirement 4.6. The model was not silently
trained with a capped vocabulary and was not treated as tied based on inference.

Because construction correctly stopped before allocating/running the
full-vocabulary path, this GGUF cannot provide the remaining acceptance
measurements:

- full-vocabulary step-1 loss in `[10.0, 14.0]`;
- peak RSS;
- peak RSS delta relative to the pre-GWEN-221 baseline;
- real-model no-OOM completion.

Those measurements require a GGUF whose architecture metadata explicitly
contains Boolean `tie_word_embeddings=true`.

---

## Deliberate Deviations and Important Decisions

- **Strict Boolean parsing.** Numeric `1`, strings, absent keys, and malformed
  values are false. This matches the safety requirement and prevents accidental
  opt-in.
- **Report object instead of stderr interception.** The task allowed exposing a
  report method as an alternative to `gag`/global redirection. `DryRunReport`
  gives deterministic tests and cleaner production code.
- **Loader helper renamed.** The plan note said the helper itself did not need to
  change, but retaining `load_capped_matrix` after deleting the cap would leave
  misleading vocabulary in production code. It became `load_matrix_rows`
  without changing its row-loading responsibility.
- **Storage identity is tested.** Numerical transpose equality alone would not
  prove that weight tying avoided a second buffer. The test additionally checks
  Candle storage identity.
- **Separate output tensors are tested, not merely commented.** The fixture
  contains a distinct `output.weight`; the test confirms the runtime head still
  comes from the embedding.
- **Suspicious historical loss was treated as a clue.** The prior `2.7747`
  result was not carried forward as a success criterion because it was measured
  under the capped/aliased objective.
- **Final checkpoint remains open.** Automated implementation is complete, but
  the changelog does not claim real-model memory/loss acceptance without an
  eligible GGUF.

---

## Files Changed

- **`packages/core/src/train/layer_loader.rs`**
  - parsed `tie_word_embeddings`;
  - extended `TransformerConfig`;
  - updated GGUF fixtures;
  - added strict metadata tests.
- **`packages/core/src/train/layered_training_loop.rs`**
  - removed `VOCAB_CAP`;
  - added the untied-model guard;
  - loaded the full embedding;
  - tied `lm_head` to `model_embedding.t()`;
  - corrected logits orientation;
  - renamed the matrix-row helper;
  - added deterministic dry-run reporting;
  - added full-vocabulary, storage, no-LoRA, report, and 50-step tests.
- **`packages/core/tests/gwen220_wave4.rs`**
  - replaced the capped `ln(8192)` loss ceiling with GWEN-221 full-vocabulary
    assertions.
- **`.kiro/specs/gwen-221-lift-vocab-cap-weight-tying/tasks.md`**
  - marked Waves 1–4 complete;
  - recorded the metadata-blocked final real-model checkpoint.

Source diff for the three tracked Rust files:

```text
396 insertions, 66 deletions
```

---

## Remaining Acceptance Item

Obtain or produce a Qwen3-1.7B Q8_0 GGUF that explicitly contains:

```text
qwen3.tie_word_embeddings = true
```

Then rerun the one-step env-gated harness and record:

1. `vocab(full)=151936` (or the exact GGUF vocabulary);
2. initial CE loss;
3. RSS start/peak/delta;
4. no-OOM completion on the 8 GB target;
5. RSS delta versus the pre-GWEN-221 baseline.

Until that model artifact exists, the implementation is complete and
unit/integration-tested, but Task 6 remains intentionally unchecked.

---

## Follow-up: GGUF Training-Readiness Auto-Detector in `gwen doctor`

**Date:** 2026-06-15 (follow-up conversation, same session day)
**Scope:** `diagnostics/doctor.rs`, `tui/src/commands/doctor.rs`, `tui/src/commands/setup.rs`
**Type:** Diagnostic tooling — foundational pre-training guard

### Background

After the GWEN-221 real-model attempt was blocked by the guard, an audit
identified the gap: there was no fast way to determine *why* the Qwen3-1.7B
Q8_0 GGUF was rejected — whether it was a **metadata gap** (GGUF conversion
omitted the KV, but the model is actually tied) or **genuinely untied** (the
architecture truly uses a separate output head). The two cases have different
remedies:

| Case | Remedy |
|---|---|
| Metadata gap | Re-convert GGUF, or source an alternate build that writes `qwen3.tie_word_embeddings=true` |
| Genuinely untied | Implement sampled-softmax (option b), or switch to a confirmed-tied model |

The cheapest diagnostic is the GGUF tensor list itself: if `output.weight` /
`lm_head.weight` exists as a standalone tensor, the model is structurally
untied regardless of metadata. If those tensors are absent and the KV is also
absent, the model is structurally tied — the metadata was just omitted during
conversion.

### What Was Implemented

#### `probe_gguf_training_readiness` — 4-way resolution

Added to `diagnostics/doctor.rs`. For each GGUF it calls `parse_header`
(header-only, no tensor data read into RAM), then classifies into exactly one
of four states:

| Condition | `value` output | Status |
|---|---|---|
| KV `{arch}.tie_word_embeddings = true` | `tied — metadata=true` | Pass |
| KV `{arch}.tie_word_embeddings = false` | `untied — metadata=false` | Fail |
| KV absent + no `output.weight` tensor | `tied — structural (no separate output head)` | Pass |
| KV absent + `output.weight` present | `untied — structural (output.weight / lm_head.weight present; metadata key absent)` | Fail |

The distinction between case 2 and case 4 is what the audit asked for:
`metadata=false` is definitively untied; the structural case with an absent KV
is a likely metadata gap and gets a different suggestion.

Suggestions surfaced on Fail:
- `metadata=false` → `"model is genuinely untied; use sampled-softmax (option b) or switch to a tied model (e.g. Qwen3-0.6B)"`
- structural untied → `"output.weight exists but tie_word_embeddings KV is absent — possible metadata gap in GGUF conversion; try re-converting or check config.json on HuggingFace"`

#### Default scan + `--model` override

`check_gguf_training_readiness(model_paths: Vec<PathBuf>)`:
- Empty vec → scans all `.gguf` files in `GwenPaths::models_dir()` automatically
- Non-empty → probes exactly those paths

One `CheckResult` is emitted per model, named `gguf-train:<stem>`. If the
models directory is empty or does not exist, the check emits nothing (silent
skip — the existing `models` check already surfaces this).

#### `run_all_checks` signature

```rust
// before
pub async fn run_all_checks(safe: bool, force: bool) -> Vec<CheckResult>

// after
pub async fn run_all_checks(safe: bool, force: bool, model_paths: Vec<PathBuf>) -> Vec<CheckResult>
```

GGUF results are appended after the existing infrastructure checks.

#### `--model` flag in `gwen doctor`

```
gwen doctor                               # scan all GGUFs in models dir
gwen doctor --model path/to/Model.gguf   # probe a specific file
gwen doctor --model a.gguf --model b.gguf
```

`setup.rs` call site updated to pass `vec![]` (scan all).

### Usage for the GWEN-221 Remaining Acceptance Item

Run against the blocked Qwen3-1.7B file:

```text
gwen doctor --model C:\Users\reyha\Downloads\Qwen3-1.7B-Q8_0.gguf
```

Expected output given the guard result:

```
  gguf-train:Qwen3-1.7B-Q8_0   untied — ...   ✗  → <suggestion>
```

If `value` contains `structural (output.weight / lm_head.weight present; metadata key absent)` → metadata gap; try re-converting.
If `value` contains `metadata=false` → genuinely untied; sampled-softmax or model swap required.

### Files Changed

- **`packages/core/src/diagnostics/doctor.rs`**
  - Added `use crate::convert::gguf_parser::{self, MetadataValue}` and `use crate::storage::paths::GwenPaths`
  - Updated `run_all_checks` signature (new `model_paths` param)
  - Added `check_gguf_training_readiness`, `collect_gguf_paths_from_models_dir`, `probe_gguf_training_readiness`
- **`packages/tui/src/commands/doctor.rs`**
  - Added `--model` flag to `DoctorArgs`
  - Updated `run_all_checks` call to pass `args.model`
- **`packages/tui/src/commands/setup.rs`**
  - Updated `run_all_checks` call to pass `vec![]`

Build: `cargo build -p gwenland-core -p gwenland-tui` → clean, 0 errors.

---

**End of Gwen-Changes-2026-06-15_GWEN-221.md**
