# Requirements Document

## Introduction

This document specifies the requirements for integrating mistral.rs as a native Rust inference backend for GwenLand. The integration provides an alternative to the existing candle-transformers implementation through a trait-based architecture that enables runtime backend selection while maintaining full backward compatibility with existing chat interfaces and TUI components.

The system must satisfy three critical constraints:
1. Zero breaking changes to gwen-tui or chat interface
2. Pure Rust build with no C compiler dependency
3. Feature-gated compilation to minimize binary size when not needed

## Glossary

- **InferenceBackend**: Trait defining the common interface for all inference engine implementations
- **MistralRsBackend**: Implementation of InferenceBackend that wraps the mistral.rs inference engine
- **CandleBackend**: Existing implementation of InferenceBackend using candle-transformers
- **BackendRegistry**: Central registry for managing and selecting available inference backends
- **InferParams**: Configuration structure containing generation parameters (temperature, top_p, max_tokens, etc.)
- **ChatEvent**: Enum representing streaming events (Token, Done, Error) in the chat interface
- **Pipeline**: mistral.rs abstraction for loaded model and inference state
- **GGUF**: Binary format for storing quantized language models
- **SSE**: Server-Sent Events streaming protocol used for token delivery

## Requirements

### Requirement 1: Inference Backend Trait Definition

**User Story:** As a developer, I want a unified interface for all inference backends, so that I can add new backends without modifying client code.

#### Acceptance Criteria

1. THE InferenceBackend trait SHALL define load_model, infer, stream_infer, unload, and name methods
2. THE InferenceBackend trait SHALL be Send + Sync to support concurrent usage in async runtime
3. THE load_model method SHALL accept a Path parameter and return Result<()>
4. THE infer method SHALL accept a prompt string and InferParams and return Result<String>
5. THE stream_infer method SHALL accept a prompt string and InferParams and return Result<Pin<Box<dyn Stream<Item = String> + Send>>>
6. THE unload method SHALL free allocated GPU/CPU resources and return Result<()>
7. THE name method SHALL return a static string identifier for the backend

### Requirement 2: Inference Parameter Configuration

**User Story:** As a user, I want to configure generation parameters, so that I can control inference behavior.

#### Acceptance Criteria

1. THE InferParams struct SHALL include max_tokens, temperature, top_p, top_k, repetition_penalty, and stop_sequences fields
2. WHEN temperature is provided, THE System SHALL validate it is in range (0.0, 2.0]
3. WHEN top_p is provided, THE System SHALL validate it is in range (0.0, 1.0]
4. WHEN max_tokens is provided, THE System SHALL validate it is at least 1
5. THE InferParams struct SHALL provide a Default implementation with sensible values
6. THE default temperature SHALL be 0.7
7. THE default top_p SHALL be 0.9
8. THE default max_tokens SHALL be 512

### Requirement 3: MistralRs Backend Implementation

**User Story:** As a user, I want to use mistral.rs as an inference engine, so that I can benefit from its native GGUF optimizations.

#### Acceptance Criteria

1. THE MistralRsBackend struct SHALL implement the InferenceBackend trait
2. WHEN load_model is called, THE MistralRsBackend SHALL detect model architecture from GGUF metadata
3. WHEN load_model is called, THE MistralRsBackend SHALL initialize a mistralrs Pipeline with the detected architecture
4. THE MistralRsBackend SHALL support llama, qwen2, phi3, mistral, and gemma architectures
5. WHEN an unsupported architecture is detected, THE MistralRsBackend SHALL return UnsupportedArchitecture error
6. WHEN stream_infer is called, THE MistralRsBackend SHALL yield tokens incrementally via async stream
7. WHEN unload is called, THE MistralRsBackend SHALL drop the Pipeline and free model memory
8. THE MistralRsBackend name method SHALL return "mistralrs"

### Requirement 4: Backend Registry and Selection

**User Story:** As a user, I want to select inference backends via configuration, so that I can switch engines without recompiling.

#### Acceptance Criteria

1. THE BackendRegistry SHALL maintain a HashMap of registered backends
2. THE BackendRegistry SHALL register candle backend when feature "candle" is enabled
3. THE BackendRegistry SHALL register mistralrs backend when feature "mistralrs-backend" is enabled
4. WHEN get is called with a backend name, THE BackendRegistry SHALL return the corresponding backend if registered
5. WHEN list_available is called, THE BackendRegistry SHALL return all registered backend names
6. WHEN backend selection is "auto", THE System SHALL select mistralrs if available, otherwise candle
7. WHEN a requested backend is not registered, THE System SHALL return BackendNotAvailable error

### Requirement 5: Architecture Detection

**User Story:** As a developer, I want automatic architecture detection, so that users don't need to manually specify model types.

#### Acceptance Criteria

1. WHEN detect_architecture is called with a GGUF file path, THE System SHALL parse the GGUF header
2. WHEN the GGUF metadata contains "general.architecture", THE System SHALL extract its value
3. WHEN the architecture is "llama", THE System SHALL return "llama"
4. WHEN the architecture is "qwen2" or "qwen", THE System SHALL return "qwen2"
5. WHEN the architecture is "phi3", THE System SHALL return "phi3"
6. WHEN the architecture is "mistral", THE System SHALL return "mistral"
7. WHEN the architecture is "gemma", THE System SHALL return "gemma"
8. WHEN the architecture is not supported, THE System SHALL return an error with the unsupported architecture name

### Requirement 6: Streaming Inference

**User Story:** As a user, I want real-time token streaming, so that I can see responses as they are generated.

#### Acceptance Criteria

1. WHEN stream_infer is called, THE Backend SHALL return an async Stream of token strings
2. FOR ALL emitted tokens, THE Backend SHALL ensure they are valid UTF-8 strings
3. WHEN generation completes, THE Stream SHALL terminate naturally
4. WHEN generation fails mid-stream, THE Stream SHALL emit an error and terminate
5. WHEN max_tokens limit is reached, THE Stream SHALL stop yielding tokens
6. WHEN a stop sequence is encountered, THE Stream SHALL stop yielding tokens
7. THE Stream SHALL be Send to support cross-thread usage

### Requirement 7: Synchronous Inference

**User Story:** As a developer, I want synchronous inference for batch processing, so that I can process multiple prompts without streaming overhead.

#### Acceptance Criteria

1. WHEN infer is called, THE Backend SHALL return the complete generated text as a String
2. THE infer method SHALL respect the same InferParams as stream_infer
3. WHEN generation fails, THE Backend SHALL return an appropriate error
4. THE synchronous inference output SHALL be equivalent to concatenating all tokens from stream_infer

### Requirement 8: Configuration Management

**User Story:** As a user, I want to configure inference settings in a file, so that my preferences persist across sessions.

#### Acceptance Criteria

1. THE InferenceConfig struct SHALL include backend, model, model_path, params, and tokenizer_id fields
2. THE default backend SHALL be "candle" for backward compatibility
3. WHEN backend is set to "auto", THE System SHALL select the first available backend in priority order
4. WHEN model_path is not provided, THE System SHALL default to ~/.config/gwen/models/
5. THE InferenceConfig SHALL be serializable to and deserializable from JSON
6. WHEN an invalid backend name is provided, THE System SHALL validate against known backends
7. WHEN model path does not exist, THE System SHALL return a configuration error

### Requirement 9: Error Handling

**User Story:** As a user, I want clear error messages, so that I can troubleshoot issues effectively.

#### Acceptance Criteria

1. THE GwenError enum SHALL include InferenceBackend, BackendNotAvailable, UnsupportedArchitecture, and ModelLoad variants
2. WHEN a backend is requested but not compiled in, THE System SHALL return BackendNotAvailable with the backend name
3. WHEN a model architecture is unsupported, THE System SHALL return UnsupportedArchitecture with backend and architecture names
4. WHEN model loading fails, THE System SHALL wrap the underlying error in ModelLoad variant
5. WHEN inference parameters are invalid, THE System SHALL return a descriptive validation error
6. ALL errors SHALL be logged with appropriate context for debugging

### Requirement 10: Chat Integration

**User Story:** As a user, I want seamless chat integration, so that I can use the new backend without interface changes.

#### Acceptance Criteria

1. WHEN stream_chat is called, THE System SHALL select the configured backend from config
2. WHEN backend selection succeeds, THE System SHALL stream tokens to the ChatEvent channel
3. THE ChatEvent channel SHALL emit Token events for each generated token
4. WHEN generation completes, THE System SHALL emit ChatEvent::Done
5. WHEN generation fails, THE System SHALL emit ChatEvent::Error with error message
6. THE chat interface SHALL maintain backward compatibility with existing TUI
7. WHEN backend="candle", THE System SHALL produce identical output to pre-mistralrs implementation

### Requirement 11: Memory Management

**User Story:** As a system operator, I want efficient memory management, so that resources are freed when no longer needed.

#### Acceptance Criteria

1. WHEN load_model is called on a backend with an already loaded model, THE System SHALL free the previous model before loading the new one
2. WHEN unload is called, THE Backend SHALL release all GPU/CPU memory allocated for the model
3. THE System SHALL not leak memory across multiple load/unload cycles
4. WHEN a Pipeline is dropped, THE mistralrs library SHALL free its associated resources
5. WHEN the InferenceBackend reference count reaches zero, THE System SHALL automatically clean up resources

### Requirement 12: Feature Flag Isolation

**User Story:** As a developer, I want feature-gated compilation, so that users don't pay for backends they don't use.

#### Acceptance Criteria

1. WHEN building without --features mistralrs-backend, THE System SHALL not link mistralrs crate
2. WHEN building without --features mistralrs-backend, THE binary size SHALL not increase
3. WHEN mistralrs-backend feature is enabled, THE BackendRegistry SHALL register MistralRsBackend
4. WHEN mistralrs-backend feature is disabled, THE BackendRegistry SHALL not include MistralRsBackend
5. THE System SHALL compile successfully with any combination of backend features

### Requirement 13: Backward Compatibility

**User Story:** As an existing user, I want the update to be seamless, so that my workflows continue working without changes.

#### Acceptance Criteria

1. WHEN backend is not specified in config, THE System SHALL default to "candle"
2. WHEN using the candle backend, THE System SHALL produce identical outputs to pre-integration versions
3. ALL existing gwen-tui commands SHALL work without modification
4. THE chat interface SHALL maintain the same SSE format
5. THE TUI SHALL render responses identically regardless of backend
6. WHEN config files lack new fields, THE System SHALL apply sensible defaults

### Requirement 14: Performance Targets

**User Story:** As a user, I want fast inference, so that I can have responsive interactions.

#### Acceptance Criteria

1. WHEN load_model is called, THE Backend SHALL complete loading in under 15 milliseconds for Q8 models
2. THE mistralrs backend SHALL achieve at least 90 tokens per second on RTX 3090 for Qwen3-1.7B Q8_0
3. THE binary size increase with mistralrs-backend feature SHALL be less than 2 megabytes
4. THE cold start latency SHALL be measured with the existing benchmark::cold_start module
5. THE inference throughput SHALL be measured with the existing benchmark::inference module

### Requirement 15: Model Path Resolution

**User Story:** As a user, I want flexible model path specification, so that I can organize models my way.

#### Acceptance Criteria

1. WHEN model path starts with "/" or "./", THE System SHALL treat it as an explicit absolute/relative path
2. WHEN model path does not start with "/" or "./", THE System SHALL search in config.model_path directory
3. WHEN model file does not exist at resolved path, THE System SHALL return ModelLoad error with the full path
4. THE System SHALL validate that model_path directory exists before attempting to resolve models
5. THE model_path directory SHALL default to ~/.config/gwen/models/ on first use

### Requirement 16: Security and Validation

**User Story:** As a security-conscious user, I want input validation, so that malicious configurations cannot exploit the system.

#### Acceptance Criteria

1. WHEN backend name is provided in config, THE System SHALL validate it against a whitelist of ["candle", "mistralrs", "auto"]
2. WHEN parsing GGUF files, THE System SHALL validate magic bytes match "GGUF"
3. WHEN parsing GGUF metadata, THE System SHALL reject files with tensor counts exceeding 10,000
4. THE System SHALL never execute backend names as shell commands or file paths
5. THE System SHALL use only safe Rust code in the InferenceBackend trait and implementations
6. ALL configuration fields SHALL be validated before use in inference operations
