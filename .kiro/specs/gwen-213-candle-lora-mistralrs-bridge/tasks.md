# Implementation Plan: GWEN-213 — candle LoRA Training Compatibility with mistral.rs Model Weights

## Overview

This implementation plan creates a SafeTensors-based bridge that enables candle-trained LoRA adapters to be applied to GGUF base models and served via mistral.rs inference engine. The system implements a dequant-merge-requant pipeline that merges LoRA weights with quantized GGUF base weights, producing a unified model consumable by mistral.rs without requiring runtime adapter loading.

The implementation adds two new modules (`lora_bridge.rs`, `lora_merger.rs`) totaling ~450 lines of Rust code, staying within 8GB RAM constraints through streaming weight merging, maintaining compatibility with candle 0.10.2 tensor format, and introducing zero breaking changes to existing 178 tests.

## Tasks

- [ ] 1. Set up core LoRA adapter infrastructure
  - [ ] 1.1 Create `lora_bridge.rs` module with `LoraAdapter` struct
    - Create file `packages/core/src/train/lora_bridge.rs`
    - Define `LoraAdapter` struct with fields: `layer_name`, `lora_a`, `lora_b`, `rank`, `alpha`
    - Implement `compute_delta()` method: Δ = (α/r) × B × A using candle matmul
    - Implement `validate_shapes()` to verify lora_a is (rank, d_in) and lora_b is (d_out, rank)
    - Ensure both tensors reside on the same device
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.7_

  - [ ]* 1.2 Write property test for LoRA weight consistency
    - **Property 1: LoRA Weight Consistency**
    - **Validates: Requirements 1.2, 2.5, 2.7**
    - Generate random LoraAdapter instances with various ranks (1-128) and dimensions
    - Verify lora_a.shape[0] == rank and lora_b.shape[1] == rank
    - Verify both tensors are on same device
    - Use QuickCheck for property-based testing

  - [ ] 1.3 Implement `LoraExporter` struct for SafeTensors export
    - Add `LoraExporter` struct with `config: LoraConfig` field
    - Implement `new(config: LoraConfig) -> Self` constructor
    - Implement `extract_adapters(&self, varmap: &VarMap) -> Result<Vec<LoraAdapter>>`
    - Parse VarMap variable names with pattern: `lora_{a|b}_layer_{N}_{proj_type}`
    - Pair lora_a with lora_b for each layer index
    - Return `MissingLoraPair` error if lora_a exists without corresponding lora_b
    - _Requirements: 1.1, 1.4, 12.1_

  - [ ] 1.4 Implement SafeTensors serialization for adapters
    - Implement `export_safetensors(&self, varmap: &VarMap, output_path: &Path) -> Result<()>`
    - Extract all adapters using `extract_adapters()`
    - Construct SafeTensors header with tensor metadata (shape, dtype, offsets)
    - Write 8-byte little-endian header size
    - Write JSON metadata mapping tensor names to `TensorMetadata`
    - Write tensor data blobs contiguously after header
    - Validate file is readable by safetensors library
    - _Requirements: 1.5, 1.6, 1.7, 7.1, 7.2, 7.3, 7.4, 7.5_

  - [ ]* 1.5 Write unit tests for LoraAdapter and LoraExporter
    - Test `compute_delta()` with known A/B matrices, verify output shape and values
    - Test `validate_shapes()` with valid and invalid rank configurations
    - Test SafeTensors export/parse roundtrip
    - Test VarMap extraction with missing lora_b pair (expect MissingLoraPair error)
    - Test extraction with invalid variable names (expect descriptive errors)
    - Target: 90% line coverage for lora_bridge.rs

- [ ] 2. Checkpoint: Verify adapter export functionality
  - Ensure all lora_bridge.rs tests pass
  - Manually export a small adapter and verify with Python safetensors library
  - Ask the user if questions arise about adapter structure or naming conventions

- [ ] 3. Implement tensor key mapping between candle and GGUF formats
  - [ ] 3.1 Create `KeyMapper` struct in `lora_merger.rs`
    - Create file `packages/core/src/train/lora_merger.rs`
    - Implement `KeyMapper::candle_to_gguf(candle_key: &str) -> Result<String>`
    - Parse candle format: `lora_layer_{N}_{proj}_proj`
    - Generate GGUF format: `model.layers.{N}.self_attn.{proj}_proj.weight` for q/k/v/o
    - Generate GGUF format: `model.layers.{N}.mlp.{proj}_proj.weight` for gate/up/down
    - Return descriptive error for malformed keys
    - _Requirements: 5.1, 5.2, 5.3, 5.6, 5.7_

  - [ ] 3.2 Implement reverse key mapping (GGUF to candle)
    - Implement `KeyMapper::gguf_to_candle(gguf_key: &str) -> Result<String>`
    - Parse GGUF format: `model.layers.{N}.{module}.{proj}_proj.weight`
    - Generate candle format: `lora_layer_{N}_{proj}_proj`
    - Support both self_attn and mlp modules
    - _Requirements: 5.4, 5.5_

  - [ ]* 3.3 Write property test for key mapping bijectivity
    - **Property 4: Key Mapping Bijectivity**
    - **Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5**
    - Generate random layer indices (0-100) and projection types
    - Verify `gguf_to_candle(candle_to_gguf(key)) == key` for all inputs
    - Verify mapping is one-to-one (no collisions)
    - Use QuickCheck with custom ProjectionType generator

- [ ] 4. Implement Q8_0 quantization and dequantization
  - [ ] 4.1 Implement Q8_0 quantization function
    - Create `quantize_q8_0(weights: &[f32]) -> Result<Vec<u8>>` in lora_merger.rs
    - Validate weight length is multiple of 32 (Q8_0 block size)
    - For each 32-element block: compute scale = max_abs / 127
    - Handle all-zero blocks with scale = 1.0 to avoid division by zero
    - Write scale as f16 (2 bytes) followed by 32 i8 quantized values
    - Quantize each element: q[i] = round(w[i] / scale) clamped to [-127, 127]
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.7_

  - [ ] 4.2 Implement Q8_0 dequantization function
    - Create `dequantize_q8_0(bytes: &[u8]) -> Result<Vec<f32>>` in lora_merger.rs
    - Read scale as f16 and 32 i8 values per block
    - Compute w[i] = scale × q[i] for each element
    - Return f32 vector with original dimensions
    - _Requirements: 6.5, 6.6_

  - [ ]* 4.3 Write property test for quantization round-trip error bounds
    - **Property 6: Quantization Round-Trip Error Bound**
    - **Validates: Requirements 6.2, 6.4, 6.6**
    - Generate random f32 weight vectors (multiples of 32 elements)
    - Quantize then dequantize each vector
    - Verify |original - roundtrip| ≤ scale/127 for all elements
    - Test edge cases: all zeros, all same value, alternating signs
    - Use QuickCheck with normalized finite f32 generators

- [ ] 5. Implement core LoRA merging pipeline
  - [ ] 5.1 Implement `LoraMerger` struct with memory budget tracking
    - Define `LoraMerger` struct with `memory_budget: usize` field (default 2GB)
    - Implement `new() -> Self` constructor with default 2GB budget
    - Add `with_memory_budget(budget: usize) -> Self` constructor for custom budgets
    - _Requirements: 4.1_

  - [ ] 5.2 Implement GGUF parsing and adapter loading
    - Implement `merge_into_gguf(&self, base_path, adapter_path, output_path) -> Result<()>`
    - Parse base GGUF file metadata using existing gguf_parser
    - Validate GGUF magic bytes and version
    - Load adapter SafeTensors file and extract LoraAdapter list
    - Validate adapter file size ≤ 10GB
    - _Requirements: 3.1, 3.2, 15.1, 15.2_

  - [ ] 5.3 Implement streaming layer-by-layer merge loop
    - Iterate over base model tensors one at a time
    - For each tensor, check if corresponding LoRA adapter exists using KeyMapper
    - Use memmap2 for zero-copy GGUF tensor reads to minimize memory usage
    - Track memory usage with sysinfo crate
    - Return `MemoryBudgetExceeded` error if memory exceeds budget
    - _Requirements: 3.3, 4.2, 4.3, 4.4, 4.5, 12.4_

  - [ ] 5.4 Implement dequant-merge-requant for layers with adapters
    - For tensors with adapters: dequantize Q8_0 bytes to f32 array
    - Compute adapter delta using `adapter.compute_delta()`
    - Verify base and delta shapes match (return ShapeMismatch error if not)
    - Merge weights: W_merged = W_base + Δ
    - Validate all merged values are finite (no NaN or Inf)
    - Return `InvalidMergedWeights` error with layer name if non-finite values detected
    - Requantize f32 array to Q8_0 bytes
    - _Requirements: 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 12.2, 12.6_

  - [ ] 5.5 Implement tensor writing and output GGUF generation
    - For tensors without adapters: copy base tensor bytes verbatim to output
    - Write each processed tensor immediately to output file (no buffering)
    - Preserve original GGUF metadata (architecture, version, tensor names)
    - Write final GGUF header with updated tensor index
    - Log number of layers successfully merged to stderr
    - _Requirements: 3.10, 3.11, 3.12_

  - [ ]* 5.6 Write property test for merge weight finiteness
    - **Property 3: Merge Weight Finiteness**
    - **Validates: Requirements 3.7, 3.8**
    - Generate random base weights and LoRA adapters
    - Perform merge operation for each combination
    - Verify all merged weights are finite (no NaN, no Inf)
    - Test edge cases: very large deltas, near-zero base weights, extreme alpha values
    - Use QuickCheck with normalized finite f32 generators

  - [ ]* 5.7 Write unit tests for LoraMerger
    - Test key mapping regex for all projection types (q/k/v/o/gate/up/down)
    - Test merge operation with synthetic 1-layer adapter and base model
    - Test memory budget enforcement with mock large tensors (expect error)
    - Test unsupported quantization format handling (expect UnsupportedQuantization error)
    - Test shape mismatch between adapter and base tensor (expect ShapeMismatch error)
    - Target: 90% line coverage for lora_merger.rs

- [ ] 6. Checkpoint: Verify merge functionality
  - Create a synthetic 1-layer GGUF base model with Q8_0 weights
  - Create a synthetic adapter with known delta values
  - Run merge operation and verify output GGUF file is valid
  - Manually inspect merged weights to confirm delta was applied
  - Ensure all tests pass, ask the user if questions arise

- [ ] 7. Integrate export functionality into training loop
  - [ ] 7.1 Add auto-export configuration to TrainingLoop
    - Open `packages/core/src/train/training_loop.rs`
    - Add `auto_export_adapter: bool` field to training configuration
    - Add `adapter_output_path: Option<PathBuf>` field to configuration
    - Update training loop to check auto_export flag after completion
    - _Requirements: 10.1, 10.2_

  - [ ] 7.2 Implement post-training adapter export
    - After training completes successfully, check if auto_export is enabled
    - Construct output path: `{output_dir}/adapter.safetensors`
    - Call `LoraExporter::export_safetensors()` with final VarMap
    - Log export success to stderr with file path and size
    - On export failure, log warning but do not fail training
    - _Requirements: 10.3, 10.4, 17.1, 17.2_

- [ ] 8. Add CLI commands for export and merge
  - [ ] 8.1 Implement `export-adapter` subcommand
    - Open `packages/tui/src/commands/train.rs`
    - Add `ExportAdapter` variant to TrainCommand enum
    - Accept `checkpoint_path` and `output_path` arguments
    - Load checkpoint VarMap from safetensors file
    - Call `LoraExporter::export_safetensors()`
    - Print output adapter path to stderr on success
    - Print descriptive error and exit non-zero on failure
    - Support `--dry-run` flag for validation without writing
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

  - [ ] 8.2 Implement `merge-adapter` subcommand
    - Add `MergeAdapter` variant to TrainCommand enum
    - Accept `--base`, `--adapter`, and `--output` path arguments
    - Support `--memory-budget` flag to override 2GB default
    - Create LoraMerger with specified memory budget
    - Display progress bar showing N/M layers processed using indicatif crate
    - Call `LoraMerger::merge_into_gguf()`
    - Print output model path to stderr on success
    - Print descriptive error and exit non-zero on failure
    - Support `--dry-run` flag for validation without executing merge
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7_

  - [ ] 8.3 Implement `--auto-merge` flag for one-step workflow
    - Add `auto_merge: bool` field to train command configuration
    - Add `base_model: Option<PathBuf>` field for auto-merge workflow
    - When `--auto-merge` is specified, require `--base-model` argument
    - After training completes: export adapter then merge into base model sequentially
    - Print final merged model path to stderr
    - On any step failure, print which step failed and exit non-zero
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5_

  - [ ] 8.4 Update CLI help text and documentation
    - Add detailed help text for `export-adapter` subcommand with examples
    - Add detailed help text for `merge-adapter` subcommand with examples
    - Document `--auto-merge` flag and required `--base-model` argument
    - Update main README with complete workflow example: train → export → merge → serve
    - _Requirements: 18.1, 18.2, 18.5_

- [ ] 9. Implement comprehensive error handling
  - [ ] 9.1 Define custom error variants
    - Create `GwenError` variants in existing error module
    - Add `InvalidLoraShape { expected: Shape, actual: Shape }`
    - Add `MissingLoraPair { layer_idx: usize }`
    - Add `ShapeMismatch { adapter: Shape, base: Shape }`
    - Add `UnsupportedQuantization { format: String }`
    - Add `MemoryBudgetExceeded { required: usize, available: usize }`
    - Add `InvalidMergedWeights { layer_name: String, index: usize }`
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.6_

  - [ ] 9.2 Implement validation and security checks
    - Validate tensor shapes: all dimensions non-zero and ≤ 100,000
    - Validate LoRA rank > 0 and ≤ 128
    - Validate LoRA alpha > 0.0 and < 1e6
    - Reject file paths containing ".." directory traversal sequences
    - Validate SafeTensors tensor offsets are within file bounds
    - Use checked arithmetic for tensor size computations (prevent overflow)
    - Canonicalize all file paths using `Path::canonicalize()`
    - _Requirements: 15.3, 15.4, 15.5, 15.6, 15.7, 20.1, 20.2, 20.3, 20.4, 20.5_

  - [ ] 9.3 Add warning logging for non-fatal errors
    - When adapter layer name doesn't match any GGUF tensor, log warning and skip layer
    - Log all validation warnings to stderr without failing operation
    - _Requirements: 12.5, 17.6_

  - [ ] 9.4 Implement progress and diagnostic logging
    - Log merge progress: layer index and name for each processed layer
    - Log merge completion: total layers merged and output file path
    - Log export completion: output file path and size
    - Log adapter count found in VarMap at export start
    - _Requirements: 17.3, 17.4, 17.5_

- [ ] 10. Write integration tests and verify backward compatibility
  - [ ]* 10.1 Create end-to-end integration test
    - Create file `packages/core/tests/integration_lora_bridge.rs`
    - Train tiny model: 2 layers, rank=4, 10-sample dataset
    - Export adapter to SafeTensors using LoraExporter
    - Create synthetic GGUF base with identity weights
    - Merge adapter into base using LoraMerger
    - Load merged model with mistral.rs backend (if feature enabled)
    - Verify inference produces expected output (golden test)
    - Ensure test completes in < 30 seconds
    - _Requirements: 19.5, 19.6_

  - [ ]* 10.2 Write property test for SafeTensors header integrity
    - **Property 2: SafeTensors Header Integrity**
    - **Validates: Requirements 7.6, 7.7**
    - Generate random SafeTensors files with various tensor counts and shapes
    - For each tensor, verify data_offsets[1] - data_offsets[0] == prod(shape) × sizeof(dtype)
    - Verify all offsets are within file bounds
    - Verify header size ≤ 100MB
    - Use QuickCheck with custom SafeTensors generator

  - [ ]* 10.3 Write property test for memory budget compliance
    - **Property 5: Memory Budget Compliance**
    - **Validates: Requirements 4.2, 4.3**
    - Generate random merge scenarios with varying layer sizes
    - For each scenario, set tight memory budget and run merge
    - Verify memory usage never exceeds budget during operation
    - Use QuickCheck with custom memory tracking

  - [ ]* 10.4 Run full regression test suite
    - Execute `cargo test --all` to run all 178 existing tests
    - Verify zero new test failures introduced
    - Verify all 13 existing LoRA training tests still pass
    - _Requirements: 13.1, 13.2, 13.3, 13.4_

- [ ] 11. Final checkpoint and documentation
  - Execute full test suite and verify all tests pass (178 existing + new tests)
  - Run benchmarks for export and merge operations, record metrics
  - Update changelog with Milestone 3 completion entry
  - Verify merged model works with existing `gwen serve` and `gwen chat` commands
  - Ensure all correctness properties are validated by property-based tests
  - Ask the user if questions arise before marking implementation complete

## Notes

- Tasks marked with `*` are optional test-related sub-tasks and can be skipped for faster MVP
- Each task references specific requirements for traceability (see requirements.md)
- Implementation uses Rust as specified in the design document
- Checkpoints ensure incremental validation at reasonable breaks
- Memory budget constraint (2GB default) critical for 8GB RAM machines
- Property tests validate universal correctness properties using QuickCheck
- Integration tests ensure end-to-end workflow functions correctly
- Zero breaking changes required: all 178 existing tests must pass

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "9.1"] },
    { "id": 1, "tasks": ["1.2", "1.3", "3.1", "4.1"] },
    { "id": 2, "tasks": ["1.4", "1.5", "3.2", "4.2"] },
    { "id": 3, "tasks": ["3.3", "4.3", "5.1"] },
    { "id": 4, "tasks": ["5.2", "9.2"] },
    { "id": 5, "tasks": ["5.3"] },
    { "id": 6, "tasks": ["5.4"] },
    { "id": 7, "tasks": ["5.5", "5.6", "5.7"] },
    { "id": 8, "tasks": ["7.1", "9.3"] },
    { "id": 9, "tasks": ["7.2", "8.1", "9.4"] },
    { "id": 10, "tasks": ["8.2"] },
    { "id": 11, "tasks": ["8.3"] },
    { "id": 12, "tasks": ["8.4", "10.1", "10.2", "10.3"] },
    { "id": 13, "tasks": ["10.4"] }
  ]
}
```
