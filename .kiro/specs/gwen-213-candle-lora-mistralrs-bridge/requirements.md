# Requirements Document: GWEN-213 — candle LoRA Training Compatibility with mistral.rs Model Weights

## Introduction

This document specifies the functional and non-functional requirements for enabling candle-trained LoRA adapters to be applied to GGUF base models and served via mistral.rs inference engine in GwenLand. The system implements a SafeTensors-based bridge that exports candle LoRA adapters, merges them with quantized GGUF base weights through dequant-merge-requant pipeline, and produces a unified model consumable by mistral.rs without requiring runtime adapter loading.

The solution addresses three core constraints: (1) 8GB RAM limit requiring streaming weight merging, (2) candle 0.10.2 tensor format compatibility, (3) zero breaking changes to existing 178 tests including 13 passing LoRA training tests.

## Glossary

- **LoRA_Adapter**: A trained low-rank adaptation consisting of two matrices (A and B) that modify a base model layer
- **SafeTensors_File**: A binary file format for storing tensors with metadata header
- **GGUF_File**: GPU-Generalized Unified Format for storing quantized language models
- **Q8_0_Quantization**: 8-bit quantization format with per-block scaling factors
- **VarMap**: Candle's variable map structure containing trained model parameters
- **Dequantization**: Converting quantized integer weights to floating-point format
- **Requantization**: Converting floating-point weights back to quantized integer format
- **Base_Model**: The pre-trained GGUF model to which LoRA adapters will be applied
- **Merged_Model**: The output GGUF model with LoRA weights merged into base weights
- **KeyMapper**: Component that maps tensor names between candle and mistral.rs formats
- **Memory_Budget**: Maximum RAM allocation for merge operations (default 2GB)
- **Training_Loop**: The candle-based training pipeline that produces LoRA weights
- **MistralRs_Backend**: The inference engine that loads and serves GGUF models
- **Layer_Index**: Zero-based integer identifying transformer layers (0 to N-1)
- **Projection_Type**: Type of linear projection (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj)

## Requirements

### Requirement 1: LoRA Adapter Export

**User Story:** As a developer, I want to export trained LoRA adapters from candle VarMap to SafeTensors format, so that I can persistently store adapter weights for later merging.

#### Acceptance Criteria

1. WHEN a VarMap contains lora_a and lora_b variables, THE LoRA_Exporter SHALL extract all adapter pairs into LoraAdapter structures
2. WHEN extracting adapters, THE LoRA_Exporter SHALL validate that lora_a shape is (rank, d_in) and lora_b shape is (d_out, rank)
3. WHEN adapter shapes are invalid, THE LoRA_Exporter SHALL return an InvalidLoraShape error with expected versus actual dimensions
4. WHEN a lora_a variable exists without corresponding lora_b, THE LoRA_Exporter SHALL return a MissingLoraPair error with the layer index
5. WHEN all adapters are extracted, THE LoRA_Exporter SHALL serialize tensors to SafeTensors binary format at the specified output path
6. WHEN writing SafeTensors file, THE LoRA_Exporter SHALL create a header containing tensor metadata including shape, dtype, and byte offsets
7. THE SafeTensors_File SHALL be readable by standard SafeTensors parsers including Python safetensors library

### Requirement 2: LoRA Adapter Structure

**User Story:** As a developer, I want a structured representation of LoRA adapters, so that I can manipulate and validate adapter weights programmatically.

#### Acceptance Criteria

1. THE LoraAdapter SHALL contain fields for layer_name, lora_a tensor, lora_b tensor, rank, and alpha
2. WHEN computing delta weights, THE LoraAdapter SHALL calculate Δ = (α/r) × B × A where α is alpha, r is rank, B is lora_b, and A is lora_a
3. WHEN computing delta weights, THE LoraAdapter SHALL return a tensor with shape (d_out, d_in)
4. WHEN validating shapes, THE LoraAdapter SHALL verify lora_a and lora_b are 2D tensors
5. WHEN validating shapes, THE LoraAdapter SHALL verify lora_a first dimension equals rank and lora_b second dimension equals rank
6. WHEN shape validation fails, THE LoraAdapter SHALL return a descriptive error including the mismatched dimensions
7. THE LoraAdapter SHALL ensure lora_a and lora_b tensors reside on the same device

### Requirement 3: LoRA Adapter Merging

**User Story:** As a developer, I want to merge LoRA adapters into GGUF base models, so that I can create a single unified model for inference without runtime adapter loading.

#### Acceptance Criteria

1. WHEN merging adapters, THE LoraMerger SHALL parse the base GGUF file metadata and tensor list
2. WHEN merging adapters, THE LoraMerger SHALL load adapter SafeTensors file and extract LoraAdapter structures
3. WHEN processing tensors, THE LoraMerger SHALL iterate over base model tensors and identify those with corresponding LoRA adapters
4. WHEN a tensor has a corresponding adapter, THE LoraMerger SHALL dequantize Q8_0 bytes to f32 array
5. WHEN a tensor has a corresponding adapter, THE LoraMerger SHALL compute adapter delta weights
6. WHEN a tensor has a corresponding adapter, THE LoraMerger SHALL add delta weights to dequantized base weights to produce merged weights
7. WHEN merged weights are computed, THE LoraMerger SHALL validate all values are finite (no NaN or Inf)
8. WHEN merged weights contain non-finite values, THE LoraMerger SHALL return an InvalidMergedWeights error with layer name and index
9. WHEN merged weights are validated, THE LoraMerger SHALL requantize f32 array to Q8_0 bytes
10. WHEN a tensor has no corresponding adapter, THE LoraMerger SHALL copy base tensor bytes verbatim to output
11. WHEN all tensors are processed, THE LoraMerger SHALL write output GGUF file with original metadata and merged tensors
12. THE LoraMerger SHALL log the number of layers successfully merged to stderr

### Requirement 4: Memory Budget Compliance

**User Story:** As a developer working on an 8GB RAM machine, I want the merge operation to stay within memory constraints, so that the system does not crash or swap to disk.

#### Acceptance Criteria

1. THE LoraMerger SHALL maintain a configurable memory_budget field with default value 2GB
2. WHEN processing each layer, THE LoraMerger SHALL process one layer at a time to minimize memory footprint
3. WHEN memory usage exceeds memory_budget, THE LoraMerger SHALL return a MemoryBudgetExceeded error with required versus available memory
4. THE LoraMerger SHALL write merged tensors immediately after processing each layer without buffering multiple layers
5. WHEN loading base model tensors, THE LoraMerger SHALL use memory-mapped file I/O for zero-copy reads

### Requirement 5: Tensor Key Mapping

**User Story:** As a developer, I want bidirectional mapping between candle layer names and GGUF tensor keys, so that adapters can be correctly matched to base model tensors.

#### Acceptance Criteria

1. WHEN mapping candle to GGUF, THE KeyMapper SHALL parse candle key format "lora_layer_{N}_{proj}_proj"
2. WHEN mapping candle to GGUF for attention projections, THE KeyMapper SHALL generate "model.layers.{N}.self_attn.{proj}_proj.weight"
3. WHEN mapping candle to GGUF for MLP projections, THE KeyMapper SHALL generate "model.layers.{N}.mlp.{proj}_proj.weight"
4. WHEN mapping GGUF to candle, THE KeyMapper SHALL parse GGUF key format "model.layers.{N}.{module}.{proj}_proj.weight"
5. WHEN mapping GGUF to candle, THE KeyMapper SHALL generate "lora_layer_{N}_{proj}_proj"
6. WHEN key format is invalid, THE KeyMapper SHALL return a descriptive error indicating the malformed key
7. THE KeyMapper SHALL support projection types: q, k, v, o, gate, up, down

### Requirement 6: Q8_0 Quantization

**User Story:** As a developer, I want to quantize and dequantize weights in Q8_0 format, so that I can work with GGUF models while maintaining acceptable precision.

#### Acceptance Criteria

1. WHEN quantizing f32 weights, THE Quantizer SHALL verify weight length is a multiple of 32 (Q8_0 block size)
2. WHEN quantizing a block of 32 f32 values, THE Quantizer SHALL compute scale as max_abs / 127
3. WHEN quantizing a block, THE Quantizer SHALL write scale as f16 (2 bytes) followed by 32 i8 quantized values
4. WHEN quantizing each element, THE Quantizer SHALL compute q[i] = round(w[i] / scale) clamped to [-127, 127]
5. WHEN dequantizing Q8_0 bytes, THE Dequantizer SHALL read scale as f16 and 32 i8 values per block
6. WHEN dequantizing, THE Dequantizer SHALL compute w[i] = scale × q[i] for each element
7. WHEN quantizing all-zero blocks, THE Quantizer SHALL use scale = 1.0 to avoid division by zero

### Requirement 7: SafeTensors File Format

**User Story:** As a developer, I want SafeTensors files to conform to the standard format, so that they are interoperable with other tools and libraries.

#### Acceptance Criteria

1. THE SafeTensors_File SHALL begin with an 8-byte little-endian integer specifying header size
2. THE SafeTensors_File SHALL contain a JSON header mapping tensor names to metadata
3. WHEN writing tensor metadata, THE System SHALL include dtype, shape, and data_offsets fields
4. THE dtype field SHALL be one of: F32, F16, BF16, I32, I64
5. WHEN writing tensor data, THE System SHALL place all tensor bytes contiguously after the header
6. WHEN reading SafeTensors files, THE System SHALL validate data_offsets[1] - data_offsets[0] equals shape product × dtype size
7. WHEN reading SafeTensors files, THE System SHALL validate total header size is ≤ 100MB to guard against malformed files

### Requirement 8: CLI Integration for Export

**User Story:** As a user, I want a CLI command to export adapters, so that I can save trained LoRA weights without writing custom code.

#### Acceptance Criteria

1. THE CLI SHALL provide a "gwen train export-adapter" subcommand
2. WHEN invoking export-adapter, THE CLI SHALL accept a checkpoint path argument and an output path argument
3. WHEN export completes successfully, THE CLI SHALL print the output adapter path to stderr
4. WHEN export fails, THE CLI SHALL print a descriptive error message to stderr and exit with non-zero status
5. THE CLI SHALL support a --dry-run flag that validates inputs without writing files

### Requirement 9: CLI Integration for Merge

**User Story:** As a user, I want a CLI command to merge adapters into base models, so that I can create inference-ready models without writing custom code.

#### Acceptance Criteria

1. THE CLI SHALL provide a "gwen train merge-adapter" subcommand
2. WHEN invoking merge-adapter, THE CLI SHALL accept --base, --adapter, and --output path arguments
3. WHEN merge is in progress, THE CLI SHALL display a progress bar showing N/M layers processed
4. WHEN merge completes successfully, THE CLI SHALL print the output model path to stderr
5. WHEN merge fails, THE CLI SHALL print a descriptive error message to stderr and exit with non-zero status
6. THE CLI SHALL support a --dry-run flag that validates inputs without executing merge
7. THE CLI SHALL support a --memory-budget flag to override default 2GB memory limit

### Requirement 10: Automatic Export After Training

**User Story:** As a user, I want LoRA adapters to be automatically exported after training completes, so that I do not need to run a separate export command.

#### Acceptance Criteria

1. WHEN training completes successfully, THE Training_Loop SHALL optionally export adapters to SafeTensors format
2. THE Training_Loop SHALL determine export behavior based on configuration or command-line flag
3. WHEN auto-export is enabled, THE Training_Loop SHALL write adapter to "{output_dir}/adapter.safetensors"
4. WHEN auto-export fails, THE Training_Loop SHALL log a warning but not fail the overall training run

### Requirement 11: One-Step Workflow

**User Story:** As a user, I want a single command that trains, exports, and merges, so that I can minimize manual steps in my workflow.

#### Acceptance Criteria

1. THE CLI SHALL support a --auto-merge flag on the "gwen train" command
2. WHEN --auto-merge is specified, THE CLI SHALL execute training, export adapter, and merge into base model sequentially
3. WHEN --auto-merge is specified, THE CLI SHALL require --base-model path argument
4. WHEN --auto-merge workflow completes, THE CLI SHALL print the final merged model path to stderr
5. WHEN any step in --auto-merge workflow fails, THE CLI SHALL print which step failed and exit with non-zero status

### Requirement 12: Error Handling and Recovery

**User Story:** As a developer, I want comprehensive error handling, so that I can diagnose and recover from failures easily.

#### Acceptance Criteria

1. WHEN VarMap is missing lora_b for a lora_a variable, THE System SHALL return a MissingLoraPair error with layer index
2. WHEN adapter and base model tensor shapes are incompatible, THE System SHALL return a ShapeMismatch error with both shapes
3. WHEN base model uses unsupported quantization format, THE System SHALL return an UnsupportedQuantization error with detected format name
4. WHEN memory budget is exceeded, THE System SHALL return a MemoryBudgetExceeded error with required versus available memory
5. WHEN adapter layer name does not match any GGUF tensor, THE System SHALL log a warning and skip the layer without failing the entire merge
6. WHEN file I/O operations fail, THE System SHALL propagate the underlying error with additional context about which file and operation
7. WHEN merged weights contain NaN or Inf, THE System SHALL return an InvalidMergedWeights error with layer name and first invalid index

### Requirement 13: Backward Compatibility

**User Story:** As a developer, I want the new functionality to not break existing tests, so that I can maintain confidence in the codebase.

#### Acceptance Criteria

1. WHEN running the full test suite, THE System SHALL pass all 178 existing tests without new failures
2. WHEN running LoRA training tests, THE System SHALL pass all 13 existing LoRA training tests
3. THE new modules SHALL not modify existing public APIs in train::training_loop or convert::dequant
4. WHEN new CLI commands are added, THE existing CLI commands SHALL continue to work with identical behavior

### Requirement 14: Performance Requirements

**User Story:** As a user, I want fast export and merge operations, so that I can iterate quickly during development.

#### Acceptance Criteria

1. WHEN exporting a Qwen3-1.7B adapter with rank=8 and 16 layers, THE System SHALL complete in less than 100 milliseconds
2. WHEN merging a single Q8_0 layer, THE System SHALL complete in less than 50 milliseconds per layer
3. WHEN merging a complete Qwen3-1.7B Q8_0 model, THE System SHALL complete in less than 5 seconds end-to-end
4. WHEN serving merged model via mistral.rs, THE inference latency SHALL not increase by more than 5% compared to base model

### Requirement 15: Validation and Data Integrity

**User Story:** As a developer, I want comprehensive validation of all inputs, so that I can fail fast with clear error messages rather than producing corrupted outputs.

#### Acceptance Criteria

1. WHEN loading adapter files, THE System SHALL validate file size is ≤ 10GB
2. WHEN loading GGUF files, THE System SHALL validate magic bytes match GGUF format specification
3. WHEN processing tensor shapes, THE System SHALL validate all dimensions are non-zero and ≤ 100,000
4. WHEN loading LoRA configuration, THE System SHALL validate rank is > 0 and ≤ 128
5. WHEN loading LoRA configuration, THE System SHALL validate alpha is > 0.0 and < 1e6
6. WHEN resolving file paths, THE System SHALL canonicalize paths to prevent directory traversal attacks
7. WHEN computing tensor byte sizes, THE System SHALL use checked arithmetic to prevent integer overflow

### Requirement 16: Inference Integration

**User Story:** As a user, I want merged models to work seamlessly with mistral.rs inference backend, so that I can serve fine-tuned models without runtime adapter loading.

#### Acceptance Criteria

1. WHEN loading merged GGUF file, THE MistralRs_Backend SHALL successfully parse the file via GgufModelBuilder
2. WHEN serving merged model, THE MistralRs_Backend SHALL support stream_chat_native API with identical behavior to base models
3. WHEN merged model is loaded, THE System SHALL not require separate adapter file or runtime adapter loading
4. THE merged model SHALL be usable with existing "gwen serve" and "gwen chat" commands without modifications

### Requirement 17: Logging and Observability

**User Story:** As a user, I want informative logging during export and merge operations, so that I can monitor progress and diagnose issues.

#### Acceptance Criteria

1. WHEN export begins, THE System SHALL log the number of adapters found in VarMap
2. WHEN export completes, THE System SHALL log the output file path and file size
3. WHEN merge begins, THE System SHALL log the base model path, adapter path, and detected layer count
4. WHEN processing each layer during merge, THE System SHALL log layer index and name
5. WHEN merge completes, THE System SHALL log the total number of layers merged and output file path
6. WHEN validation warnings occur, THE System SHALL log warnings to stderr without failing the operation

### Requirement 18: Documentation Requirements

**User Story:** As a user, I want comprehensive documentation, so that I can understand how to use the new functionality.

#### Acceptance Criteria

1. THE README SHALL include a complete example workflow showing train → export → merge → serve
2. THE CLI help text SHALL document all new subcommands with argument descriptions and examples
3. THE changelog SHALL include an entry describing the new SafeTensors bridge functionality
4. THE API documentation SHALL include docstrings for all public functions with preconditions and postconditions
5. THE documentation SHALL explain the memory budget constraint and how to adjust it

### Requirement 19: Testing Coverage

**User Story:** As a developer, I want comprehensive test coverage, so that I can refactor with confidence.

#### Acceptance Criteria

1. THE new modules SHALL achieve ≥ 90% line coverage in unit tests
2. THE test suite SHALL include tests for all error paths including shape mismatch, missing pairs, and memory budget exceeded
3. THE test suite SHALL include property-based tests for key mapping invertibility
4. THE test suite SHALL include property-based tests for quantization round-trip error bounds
5. THE test suite SHALL include an end-to-end integration test covering train → export → merge → inference
6. THE integration test SHALL complete in less than 30 seconds on CI runners

### Requirement 20: Security Requirements

**User Story:** As a developer, I want protection against malicious inputs, so that the system is robust against attack.

#### Acceptance Criteria

1. WHEN processing file paths, THE System SHALL reject paths containing ".." directory traversal sequences
2. WHEN reading SafeTensors headers, THE System SHALL validate tensor offsets are within file bounds before reading data
3. WHEN processing LoRA rank, THE System SHALL enforce maximum rank of 128 to prevent memory exhaustion
4. WHEN computing tensor sizes, THE System SHALL use checked arithmetic to prevent integer overflow attacks
5. WHEN allocating memory, THE System SHALL enforce memory_budget limit to prevent denial-of-service via OOM
