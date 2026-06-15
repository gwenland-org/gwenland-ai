# Requirements Document

## Introduction

This document specifies the requirements for GWEN-212: a zero-copy handoff mechanism that wires GGQR (GwenLand's custom dequantization engine) output directly into Candle tensors for CPU-only inference. This creates an alternative inference path that bypasses mistralrs' internal dequantization pipeline, giving GwenLand full control over dequantization quality while maintaining performance on resource-constrained systems (8GB RAM, CPU-only).

The system introduces GgqrCandleBackend, a new InferenceBackend implementation that:
1. Loads GGUF files via gguf_parser
2. Dequantizes tensors using GGQR-CF-mmap (proven at 9.7 GiB/s throughput)
3. Converts Vec<f32> → Arc<[f32]> → candle_core::Tensor with minimal copying
4. Implements a minimal autoregressive forward pass for LLaMA-family models
5. Streams tokens to the Tauri frontend via SSE-style events

## Glossary

- **GGQR**: GwenLand's custom GGUF quantization/dequantization engine located in convert/dequant.rs
- **GGQR-CF-mmap**: Continuous-flow memory-mapped dequantization mode in GGQR with AVX2 acceleration
- **GgqrCandleBackend**: New InferenceBackend implementation that combines GGQR dequant with Candle inference
- **Candle**: Pure Rust ML framework (candle-core, candle-nn, candle-transformers)
- **Zero-Copy Handoff**: Strategy to minimize allocations by consuming Vec<f32> into Arc<[f32]> then wrapping as Tensor
- **InferenceBackend**: Trait defining common interface for all inference engines (from GWEN-211)
- **MistralRsBackend**: Existing backend using mistralrs' internal dequantization (from GWEN-211)
- **GGUF**: Binary format for storing quantized language models
- **LLaMA-family**: Model architectures compatible with LLaMA (Qwen3, LLaMA3, Phi-3)
- **Autoregressive Forward Pass**: Sequential token generation loop without KV cache
- **SSE**: Server-Sent Events streaming protocol for token delivery
- **Tauri EventSource**: Tauri's event system for frontend-backend communication
- **Q4_K, Q6_K**: GGUF quantization formats supported by GGQR (2-bit through 6-bit)
- **DequantMode**: GGQR enum controlling dequantization strategy (Standard, Continuous, Quadratic)

## Requirements

### Requirement 1: GGUF File Loading and Parsing

**User Story:** As a developer, I want to load GGUF files independently of mistralrs, so that I can control the entire weight loading pipeline.

#### Acceptance Criteria

1. WHEN load_model is called with a GGUF file path, THE GgqrCandleBackend SHALL parse the file using gguf_parser::parse_gguf
2. THE GgqrCandleBackend SHALL extract model architecture from GGUF metadata key "general.architecture"
3. THE GgqrCandleBackend SHALL validate the architecture is one of ["llama", "qwen2", "phi3"]
4. WHEN architecture is unsupported, THE GgqrCandleBackend SHALL return UnsupportedArchitecture error
5. THE GgqrCandleBackend SHALL extract all tensor metadata (name, shape, quantization type) from GGUF headers
6. WHEN GGUF parsing fails, THE GgqrCandleBackend SHALL return ModelLoad error with descriptive message

### Requirement 2: GGQR Dequantization Integration

**User Story:** As a developer, I want to use GGQR's high-accuracy dequantization, so that model weights maintain quality during loading.

#### Acceptance Criteria

1. FOR ALL quantized tensors in the GGUF file, THE GgqrCandleBackend SHALL call dequant::dequantize with DequantMode::Standard
2. THE dequantize function SHALL return Vec<f32> for each tensor
3. THE GgqrCandleBackend SHALL support Q2_K, Q3_K, Q4_K, Q5_K, and Q6_K quantization formats
4. WHEN a tensor uses F32 or F16 format, THE GgqrCandleBackend SHALL convert directly without quantization
5. WHEN dequantization fails for a tensor, THE GgqrCandleBackend SHALL return error with tensor name and format
6. THE dequantization throughput SHALL achieve at least 8.0 GiB/s on systems with AVX2 support

### Requirement 3: Zero-Copy Tensor Conversion

**User Story:** As a performance-conscious developer, I want minimal memory copying during tensor creation, so that loading remains fast on 8GB RAM systems.

#### Acceptance Criteria

1. WHEN converting Vec<f32> to Tensor, THE GgqrCandleBackend SHALL consume the Vec (not clone) into Arc<[f32]>
2. THE Arc<[f32]> SHALL be wrapped as candle_core::Tensor using Tensor::from_slice or equivalent zero-copy constructor
3. THE conversion SHALL NOT allocate additional Vec<f32> or perform intermediate copies
4. FOR ALL tensors, THE total memory overhead SHALL NOT exceed 5% of the original Vec<f32> size
5. THE GgqrCandleBackend SHALL store tensors in HashMap<String, candle_core::Tensor> for lookup by name

### Requirement 4: Model Weight Organization

**User Story:** As a developer, I want structured weight storage, so that the forward pass can efficiently retrieve tensors by name.

#### Acceptance Criteria

1. THE GgqrCandleBackend SHALL store all model tensors in HashMap<String, candle_core::Tensor>
2. THE HashMap keys SHALL match GGUF tensor names exactly (e.g., "model.layers.0.self_attn.q_proj.weight")
3. WHEN load_model completes, THE HashMap SHALL contain all tensors required for inference
4. THE GgqrCandleBackend SHALL validate that required tensors exist (embedding, attention, MLP, output layers)
5. WHEN a required tensor is missing, THE GgqrCandleBackend SHALL return ModelLoad error listing missing tensor names

### Requirement 5: LLaMA-Family Forward Pass Implementation

**User Story:** As a user, I want autoregressive text generation for LLaMA-family models, so that I can generate responses without relying on mistralrs.

#### Acceptance Criteria

1. THE GgqrCandleBackend SHALL implement forward pass for LLaMA-family architectures (LLaMA3, Qwen3, Phi-3)
2. THE forward pass SHALL accept input token IDs and current position as parameters
3. THE forward pass SHALL compute embeddings, multi-head attention, MLP blocks, and output logits sequentially
4. THE forward pass SHALL apply RMSNorm normalization where required by the architecture
5. THE forward pass SHALL NOT implement KV cache (deferred to GWEN-215)
6. WHEN forward pass computation fails, THE GgqrCandleBackend SHALL return InferenceError with operation and layer details

### Requirement 6: Token Sampling

**User Story:** As a user, I want configurable sampling strategies, so that I can control generation randomness and diversity.

#### Acceptance Criteria

1. THE GgqrCandleBackend SHALL implement greedy sampling (select argmax token)
2. THE GgqrCandleBackend SHALL implement top-p (nucleus) sampling with configurable p threshold
3. WHEN temperature is provided in InferParams, THE GgqrCandleBackend SHALL apply temperature scaling to logits
4. WHEN top_p is provided in InferParams, THE GgqrCandleBackend SHALL filter tokens by cumulative probability
5. WHEN top_k is provided in InferParams, THE GgqrCandleBackend SHALL limit sampling to top-k tokens
6. THE sampling implementation SHALL use rand::thread_rng for random selection
7. THE default sampling strategy SHALL be greedy (temperature=0.0, top_p=1.0)

### Requirement 7: Autoregressive Generation Loop

**User Story:** As a user, I want sequential token generation, so that the model produces coherent text outputs.

#### Acceptance Criteria

1. WHEN stream_infer is called, THE GgqrCandleBackend SHALL tokenize the input prompt using tokenizers crate
2. THE GgqrCandleBackend SHALL initialize position counter to 0
3. FOR EACH generation step, THE GgqrCandleBackend SHALL run forward pass with current token IDs and position
4. FOR EACH generation step, THE GgqrCandleBackend SHALL sample next token from output logits
5. FOR EACH generated token, THE GgqrCandleBackend SHALL decode token ID to string and yield via async stream
6. WHEN max_tokens limit is reached, THE GgqrCandleBackend SHALL stop generation
7. WHEN EOS token is generated, THE GgqrCandleBackend SHALL stop generation
8. WHEN a stop sequence is encountered, THE GgqrCandleBackend SHALL stop generation

### Requirement 8: Streaming Token Output

**User Story:** As a user, I want real-time token streaming, so that I can see responses as they are generated.

#### Acceptance Criteria

1. THE GgqrCandleBackend stream_infer method SHALL return Pin<Box<dyn Stream<Item = String> + Send>>
2. FOR EACH generated token, THE stream SHALL yield a String containing the decoded token
3. THE stream SHALL yield tokens incrementally without buffering
4. WHEN generation completes, THE stream SHALL terminate naturally
5. WHEN an error occurs during generation, THE stream SHALL yield error and terminate
6. THE stream implementation SHALL use async_stream::stream! macro for clarity

### Requirement 9: Tauri Event Integration

**User Story:** As a frontend developer, I want token events delivered to the React app, so that I can render streaming responses in the UI.

#### Acceptance Criteria

1. WHEN stream_infer yields a token, THE System SHALL emit a Tauri event with label "gwen://token"
2. THE event payload SHALL include token string, position, and timestamp
3. WHEN generation completes, THE System SHALL emit a Tauri event with label "gwen://done"
4. WHEN an error occurs, THE System SHALL emit a Tauri event with label "gwen://error" with error message
5. THE Tauri event emitter SHALL be accessible via tauri::AppHandle
6. THE events SHALL be consumable via EventSource API in the React frontend

### Requirement 10: Synchronous Inference

**User Story:** As a developer, I want synchronous inference for batch processing, so that I can process prompts without streaming complexity.

#### Acceptance Criteria

1. THE GgqrCandleBackend infer method SHALL collect all tokens from stream_infer into a single String
2. THE infer method SHALL block until generation completes
3. THE infer output SHALL be equivalent to concatenating all tokens from stream_infer
4. WHEN generation fails, THE infer method SHALL return the same error as stream_infer

### Requirement 11: Backend Registration

**User Story:** As a user, I want GgqrCandleBackend available for selection, so that I can switch to it via configuration.

#### Acceptance Criteria

1. WHEN the candle-backend feature is enabled, THE BackendRegistry SHALL register GgqrCandleBackend
2. THE GgqrCandleBackend name method SHALL return "candle-ggqr"
3. WHEN backend is set to "candle-ggqr" in config, THE System SHALL select GgqrCandleBackend
4. WHEN candle-backend feature is disabled, THE BackendRegistry SHALL NOT include GgqrCandleBackend
5. THE GgqrCandleBackend SHALL coexist with MistralRsBackend and CandleBackend in the registry

### Requirement 12: Memory Constraints

**User Story:** As a user with 8GB RAM, I want models to load successfully, so that I can run inference on resource-constrained hardware.

#### Acceptance Criteria

1. WHEN loading Qwen3-1.7B Q4_K model, THE GgqrCandleBackend SHALL consume less than 2.5 GB peak RAM
2. WHEN unload is called, THE GgqrCandleBackend SHALL free all model tensors and reduce memory to baseline
3. THE GgqrCandleBackend SHALL NOT leak memory across multiple load/unload cycles
4. WHEN system memory is below 1GB free, THE load_model method SHALL return InsufficientMemory error before allocation
5. THE memory usage SHALL be measured using sysinfo crate during load_model

### Requirement 13: Benchmark Integration

**User Story:** As a developer, I want performance metrics, so that I can compare GGQR+Candle path against Ollama baseline.

#### Acceptance Criteria

1. THE System SHALL provide a benchmark command that measures tokens/sec for Qwen3-1.7B Q4_K
2. THE benchmark SHALL run inference with GgqrCandleBackend (Path 1) and collect timing data
3. THE benchmark SHALL run inference with Ollama REST API (Path 2) and collect timing data
4. THE benchmark output SHALL be formatted as JSON with fields: backend, model, tokens_per_sec, latency_ms
5. THE benchmark SHALL write results to benchmark/gwen-benchmark-{timestamp}.json
6. THE benchmark SHALL use a fixed prompt of 50 tokens for consistency
7. THE tokens/sec metric SHALL exclude model loading time

### Requirement 14: Error Handling and Diagnostics

**User Story:** As a user, I want clear error messages, so that I can troubleshoot issues effectively.

#### Acceptance Criteria

1. WHEN load_model fails, THE GgqrCandleBackend SHALL log the GGUF file path and failure reason
2. WHEN tensor dequantization fails, THE error message SHALL include tensor name, shape, and quantization type
3. WHEN forward pass fails, THE error message SHALL include layer name and operation that failed
4. WHEN insufficient memory is detected, THE error message SHALL include required vs available memory
5. ALL errors SHALL be wrapped in GwenError enum with appropriate variant
6. THE GgqrCandleBackend SHALL log to console using log crate at appropriate levels (error, warn, info)

### Requirement 15: Feature Gate Isolation

**User Story:** As a developer, I want feature-gated compilation, so that binaries without candle-backend remain small.

#### Acceptance Criteria

1. ALL GgqrCandleBackend code SHALL be behind #[cfg(feature = "candle-backend")] attribute
2. WHEN building without --features candle-backend, THE binary SHALL NOT link candle-core, candle-nn, or candle-transformers
3. WHEN building without --features candle-backend, THE binary size SHALL NOT increase
4. WHEN candle-backend feature is enabled, THE BackendRegistry SHALL register GgqrCandleBackend
5. THE System SHALL compile successfully with any combination of backend features

### Requirement 16: Tokenizer Integration

**User Story:** As a user, I want automatic tokenizer selection, so that I don't need to manually specify tokenizer files.

#### Acceptance Criteria

1. WHEN load_model is called, THE GgqrCandleBackend SHALL extract tokenizer metadata from GGUF file
2. WHEN tokenizer metadata is available, THE GgqrCandleBackend SHALL use it to construct a tokenizers::Tokenizer
3. WHEN tokenizer metadata is not available, THE GgqrCandleBackend SHALL fall back to loading tokenizer.json from model directory
4. WHEN tokenizer loading fails, THE GgqrCandleBackend SHALL return ModelLoad error with tokenizer path
5. THE tokenizer SHALL support BPE, WordPiece, and Unigram tokenization algorithms
6. THE tokenizer decode method SHALL handle UTF-8 conversion errors gracefully

### Requirement 17: No Panic in Production Code

**User Story:** As a reliability-conscious developer, I want production code to be panic-free, so that errors are handled gracefully.

#### Acceptance Criteria

1. THE GgqrCandleBackend SHALL NOT use unwrap(), expect(), panic!, or unreachable! in production code paths
2. ALL potentially failing operations SHALL return Result<T, GwenError>
3. ALL array indexing SHALL use checked indexing (get, get_mut) or iterators
4. WHEN an invariant violation is detected, THE code SHALL return an InvariantViolation error variant
5. THE code MAY use unwrap() in test code or unreachable!() after exhaustive enum matches

### Requirement 18: Test Coverage

**User Story:** As a developer, I want comprehensive tests, so that I can verify correctness and prevent regressions.

#### Acceptance Criteria

1. THE System SHALL include unit tests for Vec<f32> → Arc<[f32]> → Tensor conversion
2. THE System SHALL include unit tests for token sampling (greedy, top-p, top-k)
3. THE System SHALL include integration test for loading Qwen3-1.7B Q4_K model
4. THE System SHALL include integration test for generating 10 tokens from prompt
5. THE System SHALL include test for memory cleanup after unload
6. THE System SHALL include test for streaming token output
7. THE System SHALL include test for Tauri event emission
8. ALL tests SHALL pass with cargo test --features candle-backend

### Requirement 19: Parser and Serializer Round-Trip (Not Applicable)

**User Story:** N/A - This feature does not implement parsers or serializers for external formats.

#### Acceptance Criteria

(No criteria - this requirement is not applicable to GWEN-212)

### Requirement 20: CLI Integration (Deferred to GWEN-213)

**User Story:** As a user, I want CLI flags to select backends, so that I can switch inference paths without editing config files.

#### Acceptance Criteria

(Deferred to GWEN-213 - this requirement specifies future work not included in GWEN-212)

### Requirement 21: Backward Compatibility

**User Story:** As an existing user, I want the update to be seamless, so that my workflows continue working without changes.

#### Acceptance Criteria

1. WHEN candle-backend feature is disabled, THE System SHALL behave identically to GWEN-211
2. WHEN backend is not specified in config, THE System SHALL default to "mistralrs" for backward compatibility
3. THE existing MistralRsBackend path SHALL continue to work unchanged
4. ALL existing gwen-tui commands SHALL work without modification
5. THE InferenceBackend trait SHALL maintain the same method signatures from GWEN-211

### Requirement 22: Security and Validation

**User Story:** As a security-conscious user, I want input validation, so that malicious GGUF files cannot exploit the system.

#### Acceptance Criteria

1. WHEN parsing GGUF files, THE System SHALL validate magic bytes match "GGUF"
2. WHEN parsing GGUF metadata, THE System SHALL reject files with tensor counts exceeding 10,000
3. WHEN parsing GGUF metadata, THE System SHALL reject files with tensor dimensions exceeding 100,000 per axis
4. WHEN loading tensors, THE System SHALL validate that buffer sizes match expected tensor byte counts
5. THE System SHALL use only safe Rust code (no unsafe blocks) in GgqrCandleBackend unless required for FFI
6. WHEN unsafe code is required, THE code SHALL document safety invariants and justify usage
