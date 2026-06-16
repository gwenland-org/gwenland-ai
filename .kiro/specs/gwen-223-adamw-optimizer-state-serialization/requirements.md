# Requirements Document

## Introduction

GWEN-223 extends the checkpoint-resume capability introduced in GWEN-222 to include the full
AdamW optimizer state — first moments (`m1`), second moments (`m2`), and the global step
counter (`t`). On GwenLand's target hardware (i3, no GPU, 200–320 s/step), restarting the
optimizer from cold moments causes 10–20 minutes of degraded training per resume event. This
feature eliminates that degradation by persisting the complete optimizer state to a companion
safetensors file written alongside every weight checkpoint. On resume the state is injected
back into the parallel `MomentStore` maintained inside `LayeredTrainingLoop`. All failures
are non-fatal and degrade gracefully to GWEN-222 behaviour.

## Glossary

- **Checkpoint_Writer**: The component responsible for saving LoRA weight and AdamW state
  files to disk. Corresponds to `save_checkpoint_and_adamw_state` and `save_adamw_state` in
  `layered_training_loop.rs` / `adamw_state.rs`.
- **Checkpoint_Loader**: The component responsible for reading an `_adamw` safetensors file
  back into the training loop. Corresponds to `load_adamw_state` in
  `layered_training_loop.rs` / `adamw_state.rs`.
- **Training_Loop**: The `LayeredTrainingLoop` struct in
  `packages/core/src/train/layered_training_loop.rs`.
- **MomentStore**: A `HashMap<String, (Tensor, Tensor)>` keyed by VarMap key string, mapping
  each trainable LoRA adapter to its `(m1, m2)` moment tensors.
- **Key_Translator**: The pair of pure functions `varmap_key_to_adamw_prefix` and
  `adamw_prefix_to_varmap_key` in `adamw_state.rs` that convert between VarMap key space and
  AdamW state file key space.
- **AdamW_State_File**: The file `checkpoint_{step:06}_adamw.safetensors` written alongside
  the weight checkpoint `checkpoint_{step:06}.safetensors`.
- **VarMap_Key**: A string of the form `l{n}.{proj}.lora_{a|b}` (multi-projection path) or
  `l{n}.lora_{a|b}` (fallback path) that identifies a trainable LoRA adapter in the VarMap.
- **step_t**: The global AdamW step counter, separate from the checkpoint naming counter.
  Used in bias correction. Initialized from `initial_step` and incremented after each
  optimizer step.
- **GWEN-222 fallback**: The resume behaviour inherited from GWEN-222 where LoRA weights are
  restored but AdamW optimizer state starts fresh from zero moments.

---

## Requirements

### Requirement 1: AdamW State Serialization

**User Story:** As a GwenLand training operator, I want the full AdamW optimizer state saved
alongside each weight checkpoint, so that a resumed training run does not waste 10–20 minutes
re-warming optimizer momentum on low-end CPU hardware.

#### Acceptance Criteria

1. WHEN `save_checkpoint_and_adamw_state` is called at step `S`, THE Checkpoint_Writer SHALL
   create or overwrite a file named `checkpoint_{S:06}_adamw.safetensors` in the same
   directory as `checkpoint_{S:06}.safetensors`.

2. WHEN `save_adamw_state` is called with a non-empty `MomentStore` of size `N`, THE
   Checkpoint_Writer SHALL write exactly `2 × N` moment tensors (one `m1` and one `m2` per
   key) to the AdamW_State_File alongside the `"step"` key, for a total of `2N + 1` tensor
   entries in the file.

3. WHEN `save_adamw_state` is called, THE Checkpoint_Writer SHALL write the global step
   counter `step_t` to the AdamW_State_File as a U64 tensor stored under the key `"step"`
   with shape `[1]`.

4. WHEN `save_adamw_state` is called with an empty `MomentStore`, THE Checkpoint_Writer SHALL
   still create the AdamW_State_File containing only the `"step"` tensor, without error.

5. IF a write error occurs during `save_adamw_state`, THEN THE Checkpoint_Writer SHALL emit
   a warning-severity log message to stderr identifying the failed checkpoint step, SHALL NOT
   propagate the error to the training loop, and the weight checkpoint file SHALL remain
   intact on disk and the training run SHALL continue uninterrupted.

---

### Requirement 2: AdamW State Deserialization on Resume

**User Story:** As a GwenLand training operator, I want the AdamW optimizer state restored on
`--resume`, so that training continues from exactly the momentum state that was checkpointed
rather than re-warming from zero.

#### Acceptance Criteria

1. WHEN `--resume` is specified and a matching AdamW_State_File exists, THE Checkpoint_Loader
   SHALL restore into `MomentStore` exactly one `(m1, m2)` pair for each key present in the
   file that also has a corresponding Var in the current VarMap.

2. WHEN `--resume` is specified and a matching AdamW_State_File exists, THE Checkpoint_Loader
   SHALL restore `step_t` to the step value stored in the `"step"` key of the
   AdamW_State_File.

3. WHEN `--resume` is specified and the AdamW_State_File is absent, THE Checkpoint_Loader
   SHALL emit a warning-severity log message to stderr identifying the missing file path and
   proceed with fresh AdamW state (GWEN-222 fallback behaviour) without aborting training.

4. WHEN `--resume` is specified and the AdamW_State_File exists but any read or parse
   operation raises an error, THE Checkpoint_Loader SHALL emit a warning-severity log message
   to stderr identifying the file path and nature of failure, clear `MomentStore`, and
   proceed with fresh AdamW state (GWEN-222 fallback behaviour) without aborting training.

5. WHEN loading the AdamW_State_File, IF the dimension list of a loaded moment tensor is not
   element-wise equal to the dimension list of the corresponding VarMap Var's tensor, THEN
   THE Checkpoint_Loader SHALL drop that `MomentStore` entry, emit a warning-severity log
   message to stderr identifying the key and the mismatched shapes, and continue loading
   remaining entries.

6. WHEN loading the AdamW_State_File and one or more entries are dropped due to shape
   mismatch, THE Checkpoint_Loader SHALL restore `step_t` and all non-mismatched `(m1, m2)`
   pairs normally, so that training proceeds with partial moment state rather than full
   fallback.

---

### Requirement 3: In-Memory Moment Bookkeeping

**User Story:** As a GwenLand training operator, I want the training loop to maintain
accurate AdamW first and second moment vectors in memory, so that the serialized state
faithfully represents the optimizer's internal state at checkpoint time.

#### Acceptance Criteria

1. THE Training_Loop SHALL maintain a `MomentStore` whose keys are exactly the VarMap key
   strings of all active trainable LoRA adapter Vars.

2. THE Training_Loop SHALL maintain a `step_t` counter initialized to `initial_step` when
   constructed and incremented by exactly 1 after each call to `update_moments`.

3. WHEN `update_moments` is called with gradient `g` for a Var with current moments
   `(m1_prev, m2_prev)`, THE Training_Loop SHALL compute the updated first moment as
   `m1' = 0.9 · m1_prev + 0.1 · g`.

4. WHEN `update_moments` is called with gradient `g` for a Var with current moments
   `(m1_prev, m2_prev)`, THE Training_Loop SHALL compute the updated second moment as
   `m2' = 0.999 · m2_prev + 0.001 · (g ⊙ g)` where `⊙` denotes element-wise
   multiplication.

5. THE Training_Loop SHALL maintain the invariant that for every VarMap key present in
   `MomentStore`, the shapes of the stored `m1` and `m2` tensors SHALL be equal to the shape
   of the corresponding trainable LoRA Var tensor at all times after their first
   `update_moments` call.

6. WHEN `update_moments` is called for the first time for a given Var, THE Training_Loop
   SHALL initialize `m1_prev` and `m2_prev` to zero tensors of the same shape and dtype as
   the Var's gradient tensor before applying the update formula.

7. WHEN `update_moments` is called and a VarMap key cannot be resolved from the Var's tensor
   id, THE Training_Loop SHALL skip that Var without error and continue processing remaining
   Vars.

8. WHEN `update_moments` is called and the gradient `g` shape is not element-wise equal to
   the stored moment shape for a given key, THE Training_Loop SHALL emit a warning-severity
   log message to stderr and skip the update for that key, leaving the existing moment tensors
   unchanged.

---

### Requirement 4: AdamW State File Path and Key Naming Convention

**User Story:** As a GwenLand developer, I want a deterministic, human-readable naming
convention for AdamW state files and their internal keys, so that files can be located and
inspected without additional tooling.

#### Acceptance Criteria

1. WHEN `varmap_key_to_adamw_prefix` is called with a VarMap key matching the pattern
   `l{n}.{proj}.lora_{a|b}` where `{n}` is a non-negative integer, `{proj}` is a non-empty
   alphanumeric string, and `{a|b}` is literally `a` or `b`, THE Key_Translator SHALL return
   `Some("layer_{n}.{proj}.lora_{a|b}")`.

2. WHEN `varmap_key_to_adamw_prefix` is called with a fallback VarMap key matching the
   pattern `l{n}.lora_{a|b}` where `{n}` is a non-negative integer and `{a|b}` is literally
   `a` or `b`, THE Key_Translator SHALL return `Some("layer_{n}.lora_{a|b}")`.

3. WHEN `varmap_key_to_adamw_prefix` is called with a string that matches neither the
   multi-projection nor the fallback pattern, THE Key_Translator SHALL return `None`.

4. THE `adamw_state_path` function SHALL return a path whose filename is equal to the weight
   checkpoint filename with `_adamw` inserted immediately before the `.safetensors` extension.

5. IF `adamw_state_path` is called with a path whose filename does not end in
   `.safetensors`, THE function SHALL still return a path in the same directory with
   `_adamw.safetensors` appended to the stem.

6. THE `adamw_state_path` function SHALL return a path in the same parent directory as the
   weight checkpoint path.

7. WHEN `adamw_prefix_to_varmap_key` is called with a string that was produced by
   `varmap_key_to_adamw_prefix` for a valid VarMap key `k`, THE Key_Translator SHALL return
   `Some(k)`, satisfying the round-trip property
   `adamw_prefix_to_varmap_key(varmap_key_to_adamw_prefix(k)?) == Some(k)`.

---

### Requirement 5: moment_store Entry Count Invariant

**User Story:** As a GwenLand developer, I want a verifiable invariant on the number of
moment entries in the store, so that correctness of the bookkeeping can be checked in tests
and assertions.

#### Acceptance Criteria

1. IF the Training_Loop is operating in non-fallback mode with `N` layers and `P` projections
   per layer (where the standard architecture has `P = 7`), THEN WHEN one or more optimizer
   steps complete, THE Training_Loop SHALL maintain exactly `N × P × 2` entries in
   `MomentStore`, where `P` is determined by the projection count discovered from layer 0 of
   the loaded GGUF.

2. IF the Training_Loop is operating in fallback mode (single-tensor fixture) with `N`
   layers, THEN WHEN one or more optimizer steps complete, THE Training_Loop SHALL maintain
   exactly `N × 2` entries in `MomentStore`.

---

### Requirement 6: Serialization Round-Trip Fidelity

**User Story:** As a GwenLand training operator, I want moment tensors to survive a
save-then-load cycle with no meaningful loss of precision, so that resumed training is
numerically equivalent to uninterrupted training.

#### Acceptance Criteria

1. WHEN `save_adamw_state` is called followed by `load_adamw_state` for the same checkpoint
   step, THE Checkpoint_Loader SHALL restore every F32 moment tensor such that the absolute
   difference between each restored element and its original value is less than `1e-7`, and
   the shape of every restored tensor SHALL be element-wise equal to the shape of the
   original tensor.

2. WHEN `save_adamw_state` is called with step value `S` followed by `load_adamw_state` for
   the same checkpoint, THE Checkpoint_Loader SHALL restore `step_t` to a value exactly
   equal to `S`.

3. WHEN `load_adamw_state` is called and the AdamW_State_File does not exist, THE
   Checkpoint_Loader SHALL return `Ok(None)` without panicking, without emitting an error,
   and without modifying `MomentStore` or `step_t`.

---

### Requirement 7: Backward Compatibility and Additive-Only Change

**User Story:** As a GwenLand developer, I want GWEN-223 to be purely additive, so that
existing training runs, tests, and CLI behaviour are unaffected.

#### Acceptance Criteria

1. WHEN `--resume` is not specified, THE Training_Loop SHALL start with an empty
   `MomentStore`, SHALL initialize `step_t` to `initial_step` (zero for a fresh run), and
   SHALL NOT read or derive the path to any AdamW_State_File.

2. IF the AdamW_State_File is absent at resume time, THEN THE Training_Loop SHALL start with
   an empty `MomentStore`, SHALL initialize `step_t` to `initial_step`, and SHALL NOT surface
   any error to the caller, proceeding as if no AdamW state file existed.

3. THE Checkpoint_Writer SHALL write the weight checkpoint file and receive a successful write
   confirmation before attempting to write the AdamW_State_File, so that a failed AdamW state
   write cannot prevent a valid weight checkpoint from being persisted.

4. WHEN the AdamW_State_File is absent, THE Checkpoint_Loader SHALL NOT surface any error to
   the caller and the training run SHALL proceed with the same observable behavior as a fresh
   run with no AdamW state.
