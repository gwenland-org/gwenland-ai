# Implementation Plan: GGQR-Candle Zero-Copy Handoff

## Overview

This implementation plan creates a new inference backend (GgqrCandleBackend) that integrates GGQR's dequantization engine with Candle's ML framework. The implementation follows a layered approach: foundational types → model loading → tensor conversion → forward pass → sampling → streaming → integration.

Key technical achievements:
- Zero-copy tensor conversion (Vec<f32> → Arc<[f32]> → Tensor)
- CPU-only inference for LLaMA-family models (Qwen3, LLaMA3, Phi-3)
- Memory-efficient loading (Qwen3-1.7B Q4_K in <2.5GB RAM)
- Streaming token generation via async streams
- Feature-gated compilation behind `candle-backend`

## Tasks

- [ ] 1. Set up feature gate and core dependencies
  - Add `candle-backend` feature to Cargo.toml with candle-core, candle-nn, candle-transformers
  - Add tokenizers, async-stream, and sysinfo dependencies
  - Create `packages/core/src/backend/candle_ggqr/mod.rs` module structure
  - Add `#[cfg(feature = "candle-backend")]` attributes to module declarations
  - _Requirements: 15.1, 15.2, 15.3, 15.4, 15.5_

- [ ] 2. Implement core data structures and error types
  - [ ] 2.1 Create ModelConfig struct for GGUF metadata
    - Define struct with architecture, n_layers, hidden_size, n_heads, n_kv_heads, intermediate_size, vocab_size, rms_norm_eps, rope_theta
    - _Requirements: 1.2, 1.3, 1.5_
  
  - [ ] 2.2 Create InferParams struct for generation parameters
    - Define struct with max_tokens, temperature, top_p, top_k, stop_sequences, seed
    - Implement Default trait with temperature=0.0, top_p=1.0, top_k=50, max_tokens=512
    - _Requirements: 6.7, 10.1_
  
  - [ ] 2.3 Extend GwenError enum with backend-specific variants
    - Add Dequantization { tensor_name: String, error: String }
    - Add InferenceError { layer: String, operation: String, error: String }
    - Add InsufficientMemory { required_gb: f32, available_gb: f32 }
    - Add UnsupportedArchitecture(String)
    - Add CandleError wrapper variant
    - _Requirements: 1.4, 2.5, 5.6, 12.4, 14.5_

- [ ] 3. Implement GGUF parsing and validation
  - [ ] 3.1 Create GGUF validation function
    - Validate magic bytes match "GGUF"
    - Check tensor count does not exceed 10,000
    - Validate tensor dimensions do not exceed 100,000 per axis
    - Validate buffer sizes match expected tensor byte counts
    - Return ModelLoad error with descriptive messages on validation failures
    - _Requirements: 1.1, 1.6, 22.1, 22.2, 22.3, 22.4_
  
  - [ ] 3.2 Create architecture extraction function
    - Extract "general.architecture" from GGUF metadata
    - Validate architecture is one of ["llama", "qwen2", "phi3"]
    - Return UnsupportedArchitecture error for unsupported architectures
    - _Requirements: 1.2, 1.3, 1.4_
  
  - [ ] 3.3 Create ModelConfig builder from GGUF metadata
    - Parse n_layers, hidden_size, n_heads, n_kv_heads from metadata
    - Parse intermediate_size, vocab_size, rms_norm_eps, rope_theta
    - Handle missing fields with appropriate defaults or errors
    - _Requirements: 1.2, 1.5_

- [ ] 4. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 5. Implement GGQR dequantization integration
  - [ ] 5.1 Create tensor dequantization wrapper function
    - Call dequant::dequantize with DequantMode::Standard for each quantized tensor
    - Support Q2_K, Q3_K, Q4_K, Q5_K, Q6_K formats
    - Handle F32/F16 formats with direct conversion (no quantization)
    - Return Dequantization error with tensor name and format on failure
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_
  
  - [ ]* 5.2 Write unit tests for dequantization integration
    - Test Q4_K tensor dequantization produces Vec<f32>
    - Test F32 tensor returns data unchanged
    - Test unsupported format returns appropriate error
    - _Requirements: 2.3, 18.1_

- [ ] 6. Implement zero-copy tensor conversion
  - [ ] 6.1 Create vec_to_tensor conversion function
    - Consume Vec<f32> into Arc<[f32]> using Arc::from
    - Wrap Arc<[f32]> as candle_core::Tensor with specified shape and device
    - Verify no additional Vec allocations occur (memory overhead <5%)
    - Return CandleError on tensor construction failures
    - _Requirements: 3.1, 3.2, 3.3, 3.4_
  
  - [ ]* 6.2 Write unit tests for zero-copy conversion
    - Test Vec<f32> → Arc<[f32]> → Tensor produces correct shape and data
    - Test Arc reference count increments correctly
    - Test no additional allocations occur (measure memory before/after)
    - _Requirements: 18.1, 3.4_

- [ ] 7. Implement model loading and weight organization
  - [ ] 7.1 Create GgqrCandleBackend struct
    - Define struct with tensors: HashMap<String, Tensor>, architecture: String, tokenizer, config: ModelConfig, device: Device
    - _Requirements: 4.1, 4.2, 4.3_
  
  - [ ] 7.2 Implement load_model method
    - Check available memory using sysinfo before loading (require 1GB free)
    - Parse GGUF file and validate using functions from task 3
    - Dequantize all tensors using GGQR integration from task 5
    - Convert Vec<f32> to Tensors using zero-copy conversion from task 6
    - Store tensors in HashMap with GGUF names as keys
    - Validate required tensors exist (embedding, attention weights, MLP weights, output layer)
    - Load tokenizer from GGUF metadata or tokenizer.json fallback
    - Log model info (architecture, layers, memory usage)
    - _Requirements: 1.1, 1.5, 4.1, 4.3, 4.4, 4.5, 12.1, 12.4, 12.5, 14.1, 16.1, 16.2, 16.3, 16.4_
  
  - [ ] 7.3 Implement unload method
    - Clear tensors HashMap
    - Drop tokenizer and config
    - Log memory released
    - _Requirements: 12.2_
  
  - [ ]* 7.4 Write integration test for Qwen3-1.7B Q4_K loading
    - Test load_model with valid GGUF file succeeds
    - Verify tensors HashMap contains expected keys
    - Verify peak memory usage is under 2.5GB
    - _Requirements: 18.3, 12.1_
  
  - [ ]* 7.5 Write unit test for memory cleanup
    - Test unload() clears all tensors
    - Test multiple load/unload cycles don't leak memory
    - _Requirements: 18.5, 12.3_

- [ ] 8. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 9. Implement RMSNorm normalization
  - [ ] 9.1 Create rms_norm helper function
    - Compute root mean square: sqrt(mean(x^2) + eps)
    - Normalize: x / rms
    - Multiply by weight tensor
    - Return normalized tensor
    - _Requirements: 5.4_
  
  - [ ]* 9.2 Write unit tests for RMSNorm
    - Test normalization produces correct statistics (variance near 1)
    - Test epsilon prevents division by zero
    - _Requirements: 18.2_

- [ ] 10. Implement multi-head attention (without KV cache)
  - [ ] 10.1 Create attention function
    - Retrieve Q, K, V projection weights for specified layer from tensors HashMap
    - Apply linear projections: Q = x @ W_q, K = x @ W_k, V = x @ W_v
    - Reshape Q, K, V to [batch, n_heads, seq_len, head_dim]
    - Compute attention scores: scores = Q @ K^T / sqrt(head_dim)
    - Apply softmax to scores
    - Compute attention output: output = softmax(scores) @ V
    - Concatenate heads and apply output projection
    - Return attention output tensor
    - _Requirements: 5.2, 5.3_
  
  - [ ]* 10.2 Write unit tests for attention mechanism
    - Test attention output shape matches input shape
    - Test attention weights sum to 1 after softmax
    - _Requirements: 18.2_

- [ ] 11. Implement MLP block with SwiGLU activation
  - [ ] 11.1 Create mlp function
    - Retrieve gate, up, down projection weights for specified layer
    - Compute gate projection: gate = x @ W_gate
    - Compute up projection: up = x @ W_up
    - Apply SwiGLU activation: output = (gate * silu(up)) @ W_down
    - Return MLP output tensor
    - _Requirements: 5.3_
  
  - [ ]* 11.2 Write unit tests for MLP block
    - Test MLP output shape matches expected dimensions
    - Test SwiGLU activation produces non-zero gradients
    - _Requirements: 18.2_

- [ ] 12. Implement forward pass for LLaMA-family models
  - [ ] 12.1 Create forward method
    - Retrieve embedding weights and convert input_ids to embeddings
    - FOR each transformer layer (0 to n_layers):
      - Apply RMSNorm to hidden states
      - Compute attention output using attention function
      - Add residual connection: hidden_states = hidden_states + attention_output
      - Apply RMSNorm to hidden states
      - Compute MLP output using mlp function
      - Add residual connection: hidden_states = hidden_states + mlp_output
    - Apply final RMSNorm
    - Retrieve output projection weights (lm_head)
    - Compute logits: logits = normalized_hidden @ W_output
    - Return logits tensor for current position
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.6_
  
  - [ ]* 12.2 Write integration test for forward pass
    - Test forward pass with single token input produces logits
    - Test logits shape matches [1, vocab_size]
    - Test forward pass does not panic with valid input
    - _Requirements: 18.4_

- [ ] 13. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 14. Implement token sampling strategies
  - [ ] 14.1 Create greedy_sample function
    - Find argmax token from logits tensor
    - Return token ID with highest probability
    - _Requirements: 6.1_
  
  - [ ] 14.2 Create top_p_sample function
    - Sort logits by probability descending
    - Compute cumulative probability
    - Filter tokens until cumulative probability >= p
    - Sample from filtered distribution using rand::thread_rng
    - Return sampled token ID
    - _Requirements: 6.2, 6.6_
  
  - [ ] 14.3 Create sample_token dispatcher function
    - WHEN temperature = 0.0, call greedy_sample
    - WHEN temperature > 0.0, apply temperature scaling to logits
    - WHEN top_p < 1.0, call top_p_sample
    - WHEN top_k is set, limit to top-k tokens before sampling
    - Return sampled token ID
    - _Requirements: 6.3, 6.4, 6.5, 6.7_
  
  - [ ]* 14.4 Write unit tests for sampling strategies
    - Test greedy sampling returns argmax token
    - Test top-p filtering produces valid probability distribution
    - Test top-k limits candidates correctly
    - Test temperature scaling affects distribution
    - _Requirements: 18.2_

- [ ] 15. Implement autoregressive generation loop
  - [ ] 15.1 Create generate_stream method
    - Tokenize input prompt using tokenizer
    - Initialize position counter to 0
    - Initialize token_ids vector with prompt tokens
    - Use async_stream::stream! macro to create async stream
    - LOOP until max_tokens or EOS:
      - Call forward method with current token_ids and position
      - Call sample_token with logits and InferParams
      - Decode token ID to string using tokenizer
      - Yield decoded token string
      - Check for EOS token or stop sequences
      - Append token to token_ids
      - Increment position
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8_
  
  - [ ]* 15.2 Write integration test for 10-token generation
    - Test generate_stream produces at most 10 tokens
    - Test each yielded token is non-empty string
    - Test stream terminates after max_tokens
    - _Requirements: 18.4_

- [ ] 16. Implement streaming infrastructure
  - [ ] 16.1 Implement stream_infer method for InferenceBackend trait
    - Accept prompt and InferParams as parameters
    - Return Pin<Box<dyn Stream<Item = String> + Send>>
    - Call generate_stream and box the stream
    - Handle errors by yielding error string and terminating
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6_
  
  - [ ]* 16.2 Write integration test for streaming output
    - Test stream_infer yields tokens incrementally
    - Test stream terminates naturally after generation completes
    - Test errors are yielded correctly
    - _Requirements: 18.6_

- [ ] 17. Implement synchronous inference
  - [ ] 17.1 Implement infer method for InferenceBackend trait
    - Call stream_infer to get token stream
    - Collect all tokens into a single String using async runtime
    - Block until generation completes
    - Return concatenated string
    - _Requirements: 10.1, 10.2, 10.3, 10.4_

- [ ] 18. Integrate with Tauri event system
  - [ ] 18.1 Add AppHandle field to GgqrCandleBackend struct
    - Store tauri::AppHandle for event emission
    - _Requirements: 9.5_
  
  - [ ] 18.2 Emit Tauri events during generation
    - WHEN token is yielded, emit "gwen://token" event with token, position, timestamp
    - WHEN generation completes, emit "gwen://done" event
    - WHEN error occurs, emit "gwen://error" event with error message
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.6_
  
  - [ ]* 18.3 Write integration test for Tauri event emission
    - Mock AppHandle for testing
    - Test "gwen://token" events emitted for each token
    - Test "gwen://done" event emitted on completion
    - Test "gwen://error" event emitted on failure
    - _Requirements: 18.7_

- [ ] 19. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 20. Implement InferenceBackend trait
  - [ ] 20.1 Implement name method
    - Return "candle-ggqr"
    - _Requirements: 11.2_
  
  - [ ] 20.2 Wire all trait methods
    - Implement load_model by calling method from task 7.2
    - Implement unload by calling method from task 7.3
    - Implement infer by calling method from task 17.1
    - Implement stream_infer by calling method from task 16.1
    - _Requirements: 21.5_

- [ ] 21. Integrate with BackendRegistry
  - [ ] 21.1 Add GgqrCandleBackend registration to BackendRegistry
    - Create register_candle_backend method with #[cfg(feature = "candle-backend")]
    - Add GgqrCandleBackend to backend selection logic in select_backend function
    - Return Box<dyn InferenceBackend> when backend name is "candle-ggqr"
    - _Requirements: 11.1, 11.3, 11.4, 11.5_
  
  - [ ] 21.2 Preserve backward compatibility
    - Keep MistralRsBackend as default when backend unspecified
    - Ensure existing CLI commands work unchanged
    - _Requirements: 21.1, 21.2, 21.3, 21.4_

- [ ] 22. Implement error handling and diagnostics
  - [ ] 22.1 Add logging throughout GgqrCandleBackend
    - Log model loading progress (file path, architecture, layers, memory usage)
    - Log inference steps (forward pass position, sampled tokens, logits)
    - Log errors with full context (tensor names, layer names, operations)
    - Use appropriate log levels (error, warn, info, debug)
    - _Requirements: 14.1, 14.2, 14.3, 14.4, 14.6_
  
  - [ ] 22.2 Add context enrichment for all errors
    - Wrap dequantization errors with tensor name and format
    - Wrap forward pass errors with layer and operation details
    - Wrap memory errors with required vs available memory
    - _Requirements: 14.2, 14.3, 14.4_

- [ ] 23. Add no-panic safety checks
  - [ ] 23.1 Replace all unwrap/expect with proper error handling
    - Review all code for unwrap(), expect(), panic!, unreachable!()
    - Replace with ? operator or match statements returning Result
    - Use checked indexing (.get(), .get_mut()) for all array access
    - _Requirements: 17.1, 17.2, 17.3, 17.4_
  
  - [ ]* 23.2 Write unit tests for error conditions
    - Test invalid GGUF file returns ModelLoad error
    - Test missing required tensor returns ModelLoad error
    - Test unsupported architecture returns UnsupportedArchitecture error
    - Test insufficient memory returns InsufficientMemory error
    - _Requirements: 18.2_

- [ ] 24. Implement benchmark integration
  - [ ] 24.1 Create benchmark command for GgqrCandleBackend
    - Measure tokens/sec for Qwen3-1.7B Q4_K with fixed 50-token prompt
    - Collect timing data excluding model loading time
    - Format output as JSON with backend, model, tokens_per_sec, latency_ms fields
    - _Requirements: 13.1, 13.2, 13.6, 13.7_
  
  - [ ] 24.2 Add benchmark comparison with Ollama
    - Run inference with Ollama REST API using same prompt
    - Collect timing data for Ollama path
    - Write results to benchmark/gwen-benchmark-{timestamp}.json
    - _Requirements: 13.3, 13.4, 13.5_

- [ ] 25. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation at major milestones
- The design uses Rust as the implementation language throughout
- All code must be feature-gated behind `#[cfg(feature = "candle-backend")]`
- Zero-copy conversion is critical for memory efficiency (Requirement 3)
- Forward pass does NOT implement KV cache (deferred to GWEN-215)
- All production code must be panic-free (Requirement 17)
- Minimum 80% test coverage for GgqrCandleBackend module

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "2.1", "2.2", "2.3"] },
    { "id": 1, "tasks": ["3.1", "3.2", "3.3"] },
    { "id": 2, "tasks": ["5.1", "5.2"] },
    { "id": 3, "tasks": ["6.1", "6.2"] },
    { "id": 4, "tasks": ["7.1", "7.2", "7.3"] },
    { "id": 5, "tasks": ["7.4", "7.5", "9.1"] },
    { "id": 6, "tasks": ["9.2", "10.1", "11.1"] },
    { "id": 7, "tasks": ["10.2", "11.2", "12.1"] },
    { "id": 8, "tasks": ["12.2", "14.1", "14.2", "14.3"] },
    { "id": 9, "tasks": ["14.4", "15.1"] },
    { "id": 10, "tasks": ["15.2", "16.1"] },
    { "id": 11, "tasks": ["16.2", "17.1", "18.1"] },
    { "id": 12, "tasks": ["18.2", "18.3"] },
    { "id": 13, "tasks": ["20.1", "20.2"] },
    { "id": 14, "tasks": ["21.1", "21.2", "22.1", "22.2"] },
    { "id": 15, "tasks": ["23.1", "23.2"] },
    { "id": 16, "tasks": ["24.1", "24.2"] }
  ]
}
```
