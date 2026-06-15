# Requirements Document

## Introduction

GWEN-222 adds two safety-critical capabilities to the GwenLand native training
pipeline: checkpoint resume and adapter export validation.

**Checkpoint resume** lets `gwen train` restart from the last persisted
`checkpoint_{step:06}.safetensors` file produced by `save_checkpoint()` in
`layered_training_loop.rs`. LoRA adapter weights are restored from the file
into the `VarMap` before the training loop starts; optimizer (AdamW) moment
state is intentionally not serialized. The `--resume` flag accepts an optional
explicit path; without a value it auto-discovers the lexicographically latest
`checkpoint_*.safetensors` file under `config.output_path`.

**Adapter export validation** adds a pre-write shape-check inside
`export_adapter()` in `lora_cli.rs`. Before `LoraExporter::export_safetensors`
is called, each extracted `lora_a` / `lora_b` tensor pair is cross-checked
against the corresponding base GGUF tensor shape using the GGUF-parsing
machinery already present in `lora_merger.rs`. A dimension mismatch produces a
clean error and prevents any partial output file from being written.

The four-wave delivery plan (Audit → Resume → Validation → E2E) is the
implementation schedule; these requirements govern observable, externally
testable behavior.

---

## Glossary

- **CheckpointResumer**: The subsystem inside `LayeredTrainingLoop` and
  `run_native_local()` responsible for discovering and loading an existing
  checkpoint file into a `VarMap` before training begins.
- **Checkpoint file**: A `.safetensors` file written by `save_checkpoint()`,
  named `checkpoint_{step:06}.safetensors`, located under
  `NewTrainConfig::output_path`.
- **Explicit checkpoint path**: A file path provided directly as the value of
  `--resume <path>`.
- **Auto-discovery**: The process of selecting the lexicographically last
  `checkpoint_*.safetensors` file under `config.output_path` when `--resume`
  is supplied without a value.
- **Resume step**: The optimizer step count encoded in the checkpoint filename,
  parsed from the `{step:06}` segment. The training loop advances the global
  step counter from this value so progress logging and checkpoint intervals are
  correct.
- **AdapterShapeValidator**: The subsystem in `lora_cli.rs` (and underlying
  `lora_bridge` / `lora_merger` helpers) that compares each extracted LoRA
  tensor pair's dimensions against the corresponding base GGUF tensor before
  writing the adapter file.
- **lora_a tensor**: The A matrix of a LoRA adapter with expected shape
  `(rank, d_in)`.
- **lora_b tensor**: The B matrix of a LoRA adapter with expected shape
  `(d_out, rank)`.
- **Base GGUF tensor shape**: The `(d_out, d_in)` shape recovered from the
  corresponding weight tensor in the base GGUF file via `parse_gguf_key` and
  `shape_to_2d` machinery in `lora_merger.rs`.
- **Partial output file**: A file created on disk before validation completes,
  leaving corrupt or incomplete data if the process is interrupted.
- **GwenTrainCLI**: The `gwen train` command implemented in
  `packages/tui/src/commands/train.rs`.
- **NewTrainConfig**: The configuration struct in
  `packages/core/src/train/config.rs` that carries all runtime training
  parameters, including the new `resume_checkpoint` field.
- **LayeredTrainingLoop**: The bounded-memory LoRA training orchestrator in
  `packages/core/src/train/layered_training_loop.rs`.
- **export_adapter()**: The public function in
  `packages/core/src/train/lora_cli.rs` that loads a checkpoint, extracts
  adapters, validates shapes, and writes the output adapter file.
- **LoraExporter**: The struct in `lora_bridge.rs` that performs adapter
  extraction and SafeTensors serialization.

---

## Requirements

### Requirement 1 — --resume CLI Flag

**User Story:** As a developer fine-tuning a local GGUF model, I want to pass
`--resume` to `gwen train` so that training continues from the last checkpoint
without restarting from scratch.

#### Acceptance Criteria

1. THE **GwenTrainCLI** SHALL accept a `--resume` flag that takes an optional
   file-path value on `TrainArgs`.

2. WHEN `--resume` is supplied without a value, THE **GwenTrainCLI** SHALL
   set `NewTrainConfig::resume_checkpoint` to `ResumeMode::Auto`.

3. WHEN `--resume <path>` is supplied with an explicit path, THE
   **GwenTrainCLI** SHALL set `NewTrainConfig::resume_checkpoint` to
   `ResumeMode::Explicit(path)`.

4. WHEN neither `--resume` nor an explicit path is supplied, THE
   **GwenTrainCLI** SHALL set `NewTrainConfig::resume_checkpoint` to
   `ResumeMode::None`, leaving training to start from step 0.

5. IF `--resume <path>` is supplied and `path` does not resolve to an
   existing file, THEN THE **GwenTrainCLI** SHALL print a descriptive error
   message and exit with a non-zero status code before training begins.

---

### Requirement 2 — Checkpoint Auto-Discovery

**User Story:** As a developer, I want `--resume` without a path to
automatically find and load the most recent checkpoint so that I do not have
to look up the exact filename.

#### Acceptance Criteria

1. WHEN `ResumeMode::Auto` is set and `config.output_path` contains one or
   more files matching the glob `checkpoint_*.safetensors`, THE
   **CheckpointResumer** SHALL select the file with the lexicographically
   greatest name.

2. WHEN `ResumeMode::Auto` is set and no `checkpoint_*.safetensors` file
   exists under `config.output_path`, THE **CheckpointResumer** SHALL log a
   warning message to stderr and continue training from step 0 without error.

3. THE **CheckpointResumer** SHALL perform checkpoint discovery before
   `LayeredTrainingLoop::new()` constructs the VarMap, so the loaded weights
   are present before the first optimizer step.

4. WHEN auto-discovery selects a checkpoint, THE **CheckpointResumer** SHALL
   log the selected file path to stderr before loading begins.

---

### Requirement 3 — VarMap Loading and Step Counter Restore

**User Story:** As a developer resuming training, I want the LoRA adapter
weights and the global step counter to be restored from the checkpoint so that
training continues coherently from where it left off.

#### Acceptance Criteria

1. WHEN a checkpoint path is resolved (either explicit or auto-discovered),
   THE **CheckpointResumer** SHALL call `VarMap::load()` with that path to
   populate the adapter weights before the training loop's first forward pass.

2. WHEN `VarMap::load()` returns an error, THE **CheckpointResumer** SHALL
   return that error to the caller and abort training initialization.

3. WHEN the checkpoint filename matches the pattern
   `checkpoint_{step:06}.safetensors`, THE **CheckpointResumer** SHALL parse
   the six-digit decimal `step` field and initialize the
   `LayeredTrainingLoop`'s global step counter to that value.

4. IF the checkpoint filename does not match the `checkpoint_{step:06}`
   pattern (e.g. an explicit path with an arbitrary name), THEN THE
   **CheckpointResumer** SHALL initialize the global step counter to 0 and
   log a warning to stderr indicating that step counting will restart from 0.

5. THE **LayeredTrainingLoop** SHALL emit checkpoint files at every 500
   optimizer steps counting from the restored step value, not from 0.

6. WHEN training completes after a resume, THE **LayeredTrainingLoop** SHALL
   report `TrainResult::total_steps` as the total number of steps executed
   in the current run (not the cumulative count including the resumed portion).

---

### Requirement 4 — No AdamW Moment State Serialization

**User Story:** As a developer, I want a clearly documented constraint that
optimizer state is not restored on resume so that I can account for the
warm-up period in my training schedule.

#### Acceptance Criteria

1. THE **CheckpointResumer** SHALL restore LoRA adapter weights only; AdamW
   moment vectors (first and second moment estimates) SHALL NOT be serialized
   to or deserialized from any checkpoint file.

2. WHEN a checkpoint is loaded, THE **LayeredTrainingLoop** SHALL construct a
   fresh `AdamW` optimizer instance with zeroed moment state, as if training
   were starting for the first time.

3. THE **GwenTrainCLI** `--resume` help text SHALL state that optimizer state
   is not restored and a momentum warm-up period will occur after resuming.

---

### Requirement 5 — Adapter Shape Validation Before Export

**User Story:** As a developer exporting a trained adapter, I want shape
validation against the base GGUF to happen before the output file is written
so that I never receive a silently corrupt adapter.

#### Acceptance Criteria

1. WHEN `export_adapter()` is called with a non-`None` `base_gguf_path`
   argument, THE **AdapterShapeValidator** SHALL parse the base GGUF file
   and retrieve each tensor's `(d_out, d_in)` shape before invoking
   `LoraExporter::export_safetensors`.

2. FOR EACH extracted adapter pair keyed `lora_{a|b}_layer_{N}_{proj}_proj`,
   THE **AdapterShapeValidator** SHALL verify:
   - `lora_a` has shape `(rank, d_in)` where `d_in` matches the base
     GGUF tensor's inner dimension for the corresponding projection.
   - `lora_b` has shape `(d_out, rank)` where `d_out` matches the base
     GGUF tensor's outer dimension for the corresponding projection.

3. IF any adapter tensor's `d_in` or `d_out` does not match the
   corresponding base GGUF tensor dimension, THEN THE
   **AdapterShapeValidator** SHALL return a `GwenError::ShapeMismatch` error
   containing the adapter name, the expected `(d_out, d_in)`, and the actual
   tensor shape.

4. WHEN a shape mismatch error is returned, THE **export_adapter()** function
   SHALL NOT write any bytes to `output_path`, leaving no partial output file
   on disk.

5. IF `export_adapter()` is called without a `base_gguf_path` (i.e. `None`),
   THE **AdapterShapeValidator** SHALL skip GGUF shape validation and proceed
   with extraction and write as before, preserving backward compatibility with
   the existing `--dry-run`-only workflow.

6. WHEN shape validation succeeds for all adapter pairs, THE
   **export_adapter()** function SHALL proceed to call
   `LoraExporter::export_safetensors` exactly once.

---

### Requirement 6 — Clean Error Reporting on Shape Mismatch

**User Story:** As a developer, I want a clear, actionable error message when
an adapter's dimensions don't match the base model so that I can identify the
mismatched projection without inspecting binary files.

#### Acceptance Criteria

1. WHEN a `GwenError::ShapeMismatch` is surfaced by `export_adapter()`, THE
   **GwenTrainCLI** `export-adapter` handler SHALL print a message to stderr
   that includes: the adapter key name, the expected shape derived from the
   base GGUF, and the actual adapter tensor shape.

2. WHEN a shape mismatch occurs, THE **GwenTrainCLI** `export-adapter` handler
   SHALL exit with a non-zero status code.

3. WHEN a shape mismatch occurs, THE **GwenTrainCLI** `export-adapter` handler
   SHALL NOT print any success message or indicate partial completion.

4. WHEN `export_adapter()` returns any error, THE **export-adapter** subcommand
   SHALL ensure that no file exists at `output_path` that was not present
   before the command was invoked.

---

### Requirement 7 — ExportAdapterArgs Base GGUF Flag

**User Story:** As a developer, I want to pass `--base-gguf <path>` to
`gwen train export-adapter` so that shape validation runs automatically.

#### Acceptance Criteria

1. THE **GwenTrainCLI** `export-adapter` subcommand SHALL accept an optional
   `--base-gguf <path>` argument on `ExportAdapterArgs`.

2. WHEN `--base-gguf <path>` is provided and the file does not exist, THE
   **GwenTrainCLI** SHALL return an error before loading the checkpoint.

3. WHEN `--base-gguf` is omitted, THE **GwenTrainCLI** SHALL pass `None` as
   `base_gguf_path` to `export_adapter()`, which SHALL skip GGUF shape
   validation (per Requirement 5, criterion 5), AND SHALL print a warning
   message to stderr stating that shape validation was skipped and the
   exported adapter has not been verified against a base model.

4. WHEN `--dry-run` and `--base-gguf` are both supplied, THE **export_adapter()**
   function SHALL perform shape validation but SHALL NOT write any output file.

---

### Requirement 8 — GGUF Shape Reuse from lora_merger Machinery

**User Story:** As a maintainer, I want shape validation to reuse
`parse_gguf_key` and `GgufDtype` from `lora_merger.rs` so that there is a
single canonical GGUF-parsing path.

#### Acceptance Criteria

1. THE **AdapterShapeValidator** SHALL use `gguf_parser::parse()` (already used
   by `LoraMerger::merge_into_gguf`) to read tensor shapes from the base GGUF.

2. THE **AdapterShapeValidator** SHALL use `parse_gguf_key()` from
   `lora_merger.rs` (or an extracted shared helper) to map GGUF tensor names
   to candle LoRA key names when matching adapter pairs to base tensors.

3. THE **AdapterShapeValidator** SHALL use `shape_to_2d()` (or equivalent
   logic) to interpret GGUF `[d_in, d_out]` dimension ordering into the
   `(d_out, d_in)` convention used by candle tensors.

4. THE **AdapterShapeValidator** SHALL NOT duplicate GGUF magic-number parsing,
   KV-metadata skipping, or tensor-info iteration logic that already exists in
   `gguf_parser.rs` or `lora_merger.rs`.
