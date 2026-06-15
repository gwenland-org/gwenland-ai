# Implementation Plan: mistralrs Inference Backend

## Overview

Integrate mistral.rs as an optional, feature-gated inference backend for gwenland-core. The implementation introduces a `InferenceBackend` trait as the shared abstraction, implements `MistralRsBackend` behind the `mistralrs-backend` feature flag, adds a `BackendRegistry` for runtime selection, extends `GwenConfig` with an `InferenceConfig` section, and updates `chat.rs` to drive streaming inference through the trait — leaving all public signatures unchanged.

The implementation language is **Rust**, consistent with the existing codebase.

## Tasks

- [x] 1. Add `mistralrs-backend` feature flag and dependency to Cargo.toml
  - [x] 1.1 Add optional mistralrs dependency and feature flag in Cargo.toml
    - Add `mistralrs = { version = "0.3", optional = true, default-features = false, features = ["gguf"] }` under `[dependencies]`
    - Add `mistralrs-backend = ["dep:mistralrs"]` under `[features]`
    - Add `quickcheck` and `quickcheck_macros` under `[dev-dependencies]` (property tests)
    - Verify `cargo check` still compiles without the feature enabled
    - _Requirements: 12.1, 12.2, 12.5_

- [x] 2. Define `GwenError` extensions for inference
  - [x] 2.1 Add new error variants to the existing error type
    - Locate the existing error enum (search for `GwenError` or `pub enum`) — if it doesn't exist yet, create `src/error.rs` and wire it into `lib.rs`
    - Add variants: `InferenceBackend(String)`, `BackendNotAvailable { backend: String }`, `UnsupportedArchitecture { backend: String, arch: String }`, `ModelLoad(#[from] anyhow::Error)`
    - Derive or implement `thiserror::Error` and `std::fmt::Display` for each variant
    - Re-export from `crate::error` where appropriate
    - _Requirements: 9.1, 9.2, 9.3, 9.4_

  - [ ]* 2.2 Write unit tests for GwenError variants
    - Verify each variant formats its `Display` string correctly (contains backend/arch names)
    - Verify `BackendNotAvailable` and `UnsupportedArchitecture` both include the identifying name in the message
    - _Requirements: 9.1, 9.2, 9.3_

- [x] 3. Define `InferParams` struct with validation
  - [x] 3.1 Create `src/engine/inference/params.rs` with `InferParams` and its validation
    - Implement the struct with fields: `max_tokens: usize`, `temperature: f32`, `top_p: f32`, `top_k: Option<usize>`, `repetition_penalty: Option<f32>`, `stop_sequences: Vec<String>`
    - Implement `Default` with values: `max_tokens = 512`, `temperature = 0.7`, `top_p = 0.9`, `top_k = None`, `repetition_penalty = None`, `stop_sequences = vec![]`
    - Add `pub fn validate(&self) -> Result<(), anyhow::Error>` that checks temperature ∈ (0.0, 2.0], top_p ∈ (0.0, 1.0], max_tokens ≥ 1
    - Derive `Debug`, `Clone`, `Serialize`, `Deserialize`
    - Re-export `InferParams` from `engine/inference/mod.rs`
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8_

  - [ ]* 3.2 Write property test: temperature boundary acceptance (Property 2)
    - **Property 2: Temperature Validation** — for any `f32` in (0.0, 2.0], `validate()` returns `Ok`; for any value ≤ 0.0 or > 2.0, returns `Err`
    - Use `quickcheck` to generate random `f32` values and classify as valid/invalid
    - **Validates: Requirement 2.2**

  - [ ]* 3.3 Write property test: top_p boundary acceptance (Property 3)
    - **Property 3: Top-P Validation** — for any `f32` in (0.0, 1.0], `validate()` returns `Ok`; for values ≤ 0.0 or > 1.0, returns `Err`
    - **Validates: Requirement 2.3**

  - [ ]* 3.4 Write property test: max_tokens boundary acceptance (Property 4)
    - **Property 4: Max Tokens Validation** — for any `usize` ≥ 1, `validate()` returns `Ok`; for `max_tokens = 0`, returns `Err`
    - **Validates: Requirement 2.4**

- [x] 4. Define `InferenceBackend` trait
  - [x] 4.1 Create `src/engine/inference/backend.rs` with the trait definition
    - Define `pub trait InferenceBackend: Send + Sync` with methods: `load_model(&self, path: &Path) -> Result<()>`, `infer(&self, prompt: &str, params: &InferParams) -> Result<String>`, `stream_infer(&self, prompt: &str, params: &InferParams) -> Result<Pin<Box<dyn Stream<Item = String> + Send>>>`, `unload(&self) -> Result<()>`, `name(&self) -> &'static str`
    - Add necessary `use` imports: `std::path::Path`, `std::pin::Pin`, `futures_core::Stream`, `anyhow::Result`
    - Re-export `InferenceBackend` from `engine/inference/mod.rs`
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7_

- [x] 5. Implement `BackendRegistry`
  - [x] 5.1 Create `src/engine/inference/registry.rs` with `BackendRegistry`
    - Implement `BackendRegistry` struct with `backends: HashMap<String, Box<dyn InferenceBackend>>`
    - Implement `new()` that registers `CandleBackend` under `#[cfg(feature = "candle")]` and `MistralRsBackend` under `#[cfg(feature = "mistralrs-backend")]` (stubs — both backends are implemented in later tasks)
    - Implement `register(&mut self, name: &str, backend: Box<dyn InferenceBackend>)`
    - Implement `get(&self, name: &str) -> Option<&dyn InferenceBackend>`
    - Implement `list_available(&self) -> Vec<String>`
    - Re-export `BackendRegistry` from `engine/inference/mod.rs`
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

  - [ ]* 5.2 Write property test: registry lookup consistency (Property 8)
    - **Property 8: Registry Lookup Consistency** — for any backend registered with name N, `get(N)` returns `Some`, and `list_available()` contains N
    - Use `MockBackend` helper that implements `InferenceBackend` with no-op methods
    - **Validates: Requirements 4.4, 4.5**

  - [ ]* 5.3 Write property test: backend not available (Property 9)
    - **Property 9: Backend Not Available Error** — for any name not registered, `get(name)` returns `None`
    - Generate random strings and verify they are not found unless explicitly registered
    - **Validates: Requirements 4.7, 9.2**

- [x] 6. Implement architecture detection using existing GGUF parser
  - [x] 6.1 Create `src/engine/inference/arch_detect.rs` with `detect_architecture()`
    - Implement `pub fn detect_architecture(path: &Path) -> Result<&'static str>`
    - Use `crate::convert::gguf_parser` to parse GGUF metadata (reuse existing parser)
    - Add a separate GGUF metadata reader that reads the KV section (not just tensor data) — specifically extract the `general.architecture` string key
    - Validate magic bytes "GGUF" are present (delegate to existing parser's checks)
    - Reject files where tensor count exceeds 10,000 (security limit per Req 16.3)
    - Map: `"llama"` → `"llama"`, `"qwen2" | "qwen"` → `"qwen2"`, `"phi3"` → `"phi3"`, `"mistral"` → `"mistral"`, `"gemma"` → `"gemma"`
    - Return `Err` for unsupported architectures, including the arch name
    - Re-export from `engine/inference/mod.rs`
    - _Requirements: 3.2, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 16.2, 16.3_

  - [ ]* 6.2 Write property test: architecture detection correctness (Property 5)
    - **Property 5: Architecture Detection Correctness** — for each supported architecture string ("llama", "qwen", "qwen2", "phi3", "mistral", "gemma"), create a minimal GGUF file in a `tempfile` with the appropriate `general.architecture` KV entry and verify the correct mapping is returned
    - Test all five architecture mappings exhaustively
    - **Validates: Requirements 3.2, 5.3, 5.4, 5.5, 5.6, 5.7**

  - [ ]* 6.3 Write property test: unsupported architecture error (Property 6)
    - **Property 6: Unsupported Architecture Error** — for any architecture string not in the supported set, `detect_architecture` returns `Err` containing the architecture name
    - Generate arbitrary architecture strings (excluding the five supported ones) and verify error is returned
    - **Validates: Requirements 3.5, 5.8**

  - [ ]* 6.4 Write property test: GGUF magic bytes validation (Property 21)
    - **Property 21: GGUF Magic Bytes Validation** — for any file not starting with the 4-byte "GGUF" magic, parsing returns an error
    - Create temp files with random first 4 bytes ≠ GGUF magic and verify rejection
    - **Validates: Requirement 16.2**

  - [ ]* 6.5 Write property test: GGUF tensor count security limit (Property 22)
    - **Property 22: GGUF Tensor Count Security Limit** — a GGUF file claiming > 10,000 tensors is rejected
    - Craft a minimal GGUF header with tensor_count > 10,000 and verify the parser rejects it
    - **Validates: Requirement 16.3**

- [-] 7. Checkpoint — Ensure all unit and property tests pass so far
  - Run `cargo test -p gwenland-core` and verify all tests from tasks 1–6 pass. Ask the user if any issues arise before proceeding.

- [ ] 8. Implement `MistralRsBackend`
  - [~] 8.1 Create `src/engine/inference/mistralrs_backend.rs` gated on `#[cfg(feature = "mistralrs-backend")]`
    - Define `pub struct MistralRsBackend { inner: Arc<Mutex<MistralRsBackendInner>> }` with `MistralRsBackendInner { pipeline: Option<Arc<Pipeline>>, model_path: Option<PathBuf> }`
    - Implement `MistralRsBackend::new() -> Self`
    - Implement `load_model`: call `detect_architecture`, build `mistralrs::TextModelBuilder`, store resulting `Pipeline` in `inner.pipeline`, drop any existing pipeline first (satisfies Req 11.1)
    - Implement `unload`: set `inner.pipeline = None` to drop the Arc and free memory
    - Implement `name`: return `"mistralrs"`
    - All methods must hold lock for minimum time; release before async work
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.7, 3.8, 11.1, 11.2, 11.4_

  - [~] 8.2 Implement `stream_infer` on `MistralRsBackend`
    - Build `mistralrs::SamplingParams` from `InferParams`
    - Obtain `Arc<Pipeline>` clone from inner (drop lock before async work)
    - Return `Pin<Box<dyn Stream<Item = String> + Send>>` using `async_stream::stream!`
    - Stream terminates after EOS or `max_tokens` tokens
    - Stop on any configured `stop_sequences` match
    - Emit `eprintln!` on stream error and break loop
    - All yielded tokens must be valid UTF-8 (enforced by mistralrs tokenizer)
    - _Requirements: 3.6, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7_

  - [~] 8.3 Implement `infer` (synchronous wrapper) on `MistralRsBackend`
    - Collect all tokens from `stream_infer` and concatenate into a `String`
    - Respect same `InferParams` as streaming path
    - _Requirements: 7.1, 7.2, 7.3, 7.4_

  - [ ]* 8.4 Write property test: streaming token UTF-8 validity (Property 10)
    - **Property 10: UTF-8 Token Validity** — for any prompt and params, all tokens from `stream_infer()` on `MockBackend` are valid UTF-8
    - Use `MockBackend` that yields tokens from a pre-defined `Vec<String>` with arbitrary content
    - **Validates: Requirement 6.2**

  - [ ]* 8.5 Write property test: stream termination (Property 11)
    - **Property 11: Stream Termination** — for any `max_tokens` ≥ 1, the stream from `MockBackend::stream_infer` yields at most `max_tokens` tokens and then terminates
    - **Validates: Requirement 6.3**

  - [ ]* 8.6 Write property test: max tokens limit enforcement (Property 12)
    - **Property 12: Max Tokens Limit Enforcement** — the stream emits at most `max_tokens` items for any value of `max_tokens`
    - Count tokens emitted by `MockBackend` stream and assert ≤ `max_tokens`
    - **Validates: Requirement 6.5**

  - [ ]* 8.7 Write property test: synchronous equals streaming (Property 14)
    - **Property 14: Synchronous Equals Streaming** — `infer()` output equals concatenation of all `stream_infer()` tokens for identical prompt and params
    - Use `MockBackend` where both methods are driven from the same deterministic token list
    - **Validates: Requirements 7.2, 7.4**

  - [ ]* 8.8 Write property test: stop sequence handling (Property 13)
    - **Property 13: Stop Sequence Handling** — when a stop sequence is set and the backend generates it, no further tokens are yielded after the match
    - Configure `MockBackend` to emit a known token sequence; set stop_sequences to a token in the middle; verify stream halts at that point
    - **Validates: Requirement 6.6**

- [ ] 9. Implement `InferenceConfig` and extend `GwenConfig`
  - [~] 9.1 Create `src/engine/inference/config.rs` with `InferenceConfig`
    - Define `InferenceConfig` with fields: `backend: String`, `model: String`, `model_path: PathBuf`, `params: InferParams`, `tokenizer_id: Option<String>`
    - Implement `Default`: `backend = "candle"`, `model_path = dirs::config_dir().../gwen/models`
    - Implement `validate(&self) -> Result<()>`: whitelist check on backend name (["candle", "mistralrs", "auto"]), non-empty model, path existence
    - Derive `Serialize`, `Deserialize` with `#[serde(default)]` on all optional fields
    - Re-export from `engine/inference/mod.rs`
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 13.1, 13.6_

  - [~] 9.2 Extend `GwenConfig` in `src/storage/config.rs` with an `inference` section
    - Add `pub inference: InferenceConfig` field to `GwenConfig` with `#[serde(default)]`
    - Add `"inference.backend"`, `"inference.model"`, `"inference.params.temperature"`, `"inference.params.top_p"`, `"inference.params.max_tokens"` to the `get` and `set` match arms
    - Ensure existing configs without the `inference` key deserialize cleanly using `Default`
    - Add `pub fn load_inference_config() -> InferenceConfig` convenience function
    - _Requirements: 8.1, 8.4, 13.1, 13.6, 16.1_

  - [ ]* 9.3 Write property test: config serialization round-trip (Property 15)
    - **Property 15: Config Serialization Round-Trip** — for any valid `InferenceConfig`, `serde_json::to_string` then `serde_json::from_str` produces an equivalent config
    - Use `quickcheck` to generate configs with random (valid) field values and assert round-trip equality
    - **Validates: Requirement 8.5**

  - [ ]* 9.4 Write property test: backend name validation (Property 16)
    - **Property 16: Backend Name Validation** — any backend name not in {"candle", "mistralrs", "auto"} causes `validate()` to return `Err`
    - Generate arbitrary strings; only pass for the three valid values
    - **Validates: Requirements 8.6, 16.1_

  - [ ]* 9.5 Write property test: model path validation (Property 17)
    - **Property 17: Model Path Validation** — any non-existent path causes `validate()` to return a config/ModelLoad error
    - Generate random path strings that do not exist on the filesystem and verify error
    - **Validates: Requirements 8.7, 15.3, 15.4**

  - [ ]* 9.6 Write property test: config defaults for missing fields (Property 20)
    - **Property 20: Config Default Application** — `InferenceConfig` deserialized from `{}` applies all `Default` values without error
    - Deserialize minimal JSON fragments and assert all defaulted fields match `InferenceConfig::default()`
    - **Validates: Requirement 13.6**

- [ ] 10. Implement `select_backend` path resolution logic
  - [~] 10.1 Create `src/engine/inference/selector.rs` with `select_backend`
    - Implement `pub fn select_backend(config: &InferenceConfig, registry: &BackendRegistry, eager_load: bool) -> Result<Arc<dyn InferenceBackend>>`
    - Step 1: Resolve "auto" by iterating ["mistralrs", "candle"] and picking first available in registry
    - Step 2: Look up by name; return `GwenError::BackendNotAvailable` if not found
    - Step 3: Resolve model path — paths starting with "/" or "./" are taken as-is; others are joined to `config.model_path`
    - Step 4: If `eager_load`, call `backend.load_model(&resolved_path)`
    - Return `Arc::new(backend)` wrapped appropriately (note: registry gives `&dyn`, need to store `Arc<Box<dyn InferenceBackend>>` pattern or redesign registry to return `Arc`)
    - Log debug info for backend name selected
    - Re-export from `engine/inference/mod.rs`
    - _Requirements: 4.6, 4.7, 8.2, 8.3, 15.1, 15.2, 15.3_

  - [ ]* 10.2 Write property test: path resolution for absolute paths (Property 18)
    - **Property 18: Path Resolution for Absolute Paths** — any model path starting with "/" or "./" is used as-is and NOT joined to `model_path`
    - Generate absolute/relative path strings and verify they are returned unchanged
    - **Validates: Requirement 15.1**

  - [ ]* 10.3 Write property test: path resolution for model names (Property 19)
    - **Property 19: Path Resolution for Model Names** — any path not starting with "/" or "./" is resolved by joining to `config.model_path`
    - Verify the result equals `config.model_path.join(model_name)`
    - **Validates: Requirement 15.2**

  - [ ]* 10.4 Write property test: backend selection determinism (Property 23)
    - **Property 23: Backend Selection Determinism** — for identical `InferenceConfig` (non-"auto") and registry, `select_backend` returns the same backend name on every call
    - Call `select_backend` twice with identical inputs and compare `backend.name()` results
    - **Validates: Requirement 4.6**

- [~] 11. Checkpoint — Ensure all tests pass through task 10
  - Run `cargo test -p gwenland-core` (without feature flag) and `cargo test -p gwenland-core --features mistralrs-backend`. Ask the user if any failures arise.

- [ ] 12. Update `engine/inference/mod.rs` and wire registry into runtime
  - [~] 12.1 Update `src/engine/inference/mod.rs` to declare all new modules and re-export public API
    - Add `pub mod backend;`, `pub mod params;`, `pub mod registry;`, `pub mod arch_detect;`, `pub mod config;`, `pub mod selector;`
    - Add `#[cfg(feature = "mistralrs-backend")] pub mod mistralrs_backend;`
    - Re-export: `InferenceBackend`, `InferParams`, `BackendRegistry`, `InferenceConfig`, `select_backend`, `detect_architecture`
    - Ensure existing `pub mod loader; pub mod model_dispatch; pub mod runner; pub mod sampler;` remain untouched
    - _Requirements: 1.1, 12.5, 13.3_

  - [ ]* 12.2 Write property test: feature flag isolation (Property 24)
    - **Property 24: Feature Flag Isolation** — when `mistralrs-backend` feature is disabled, `BackendRegistry::new()` does NOT contain "mistralrs"; when enabled, it DOES
    - Write a `#[cfg(not(feature = "mistralrs-backend"))]` test asserting absence and a `#[cfg(feature = "mistralrs-backend")]` test asserting presence
    - **Validates: Requirements 12.1, 12.2, 12.3, 12.4**

- [ ] 13. Update `chat.rs` to support native backend streaming
  - [~] 13.1 Refactor `stream_chat` in `src/engine/chat.rs` to optionally drive the `InferenceBackend` trait
    - Add a new internal function `stream_inference_to_chat(backend: Arc<dyn InferenceBackend>, prompt: String, params: InferParams, tx: UnboundedSender<ChatEvent>) -> Result<()>` that streams via the trait
    - Update `stream_chat` to check config: if `inference.backend` is set (non-default or when native flag active), build `BackendRegistry`, call `select_backend`, and hand off to `stream_inference_to_chat` instead of the HTTP SSE path
    - When using the native path: build prompt from `session.build_request_messages()` into a single string, stream tokens, send `ChatEvent::Token` per token, send `ChatEvent::Done` on completion, send `ChatEvent::Error` on failure
    - Keep the existing HTTP SSE code path unchanged as the default (backward compat)
    - All public signatures (`stream_chat`, `ChatEvent`, `ChatSession`, etc.) remain identical
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 13.1, 13.2, 13.3, 13.4, 13.5_

  - [ ]* 13.2 Write unit tests for chat integration
    - Test that `ChatEvent::Token` events flow correctly when `MockBackend` yields tokens
    - Test that `ChatEvent::Done` is always sent after stream completion
    - Test that `ChatEvent::Error` is sent when the backend returns an error
    - Test that `ChatSession` history is updated after a successful turn
    - _Requirements: 10.2, 10.3, 10.4, 10.5_

- [x] 14. Add `MockBackend` test helper
  - [x] 14.1 Create `src/engine/inference/mock_backend.rs` (test-only via `#[cfg(test)]` or dev module)
    - Implement `pub struct MockBackend { tokens: Vec<String>, fail_on_load: bool }` with `Arc<Mutex<...>>` tracking of call counts
    - Implement `InferenceBackend` for `MockBackend`: `load_model` optionally fails, `stream_infer` yields configured tokens, `infer` concatenates them, `unload` clears state, `name` returns `"mock"`
    - This helper is used by property tests in tasks 5, 8, 10, 12, and 13
    - _Requirements: (test infrastructure)_

- [ ] 15. Write remaining property tests
  - [ ]* 15.1 Write property test: resource cleanup (Property 1)
    - **Property 1: Resource Cleanup** — calling `load_model` N times followed by `unload` should not accumulate more memory than a single load cycle (use `MockBackend` with a counter to verify `unload` is called and pipeline is cleared)
    - **Validates: Requirements 1.6, 11.1, 11.3**

  - [ ]* 15.2 Write property test: streaming token delivery (Property 7)
    - **Property 7: Streaming Token Delivery** — the stream from `stream_infer` yields tokens one-at-a-time rather than all at once (verify by collecting with `StreamExt::collect` and checking count matches token sequence length)
    - **Validates: Requirement 3.6**

- [~] 16. Final checkpoint — Full test suite
  - Run `cargo test -p gwenland-core` and `cargo test -p gwenland-core --features mistralrs-backend`. Run `cargo clippy` and `cargo check --all-features`. Ask the user if any issues arise before declaring done.

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP
- Each task references specific requirements for traceability
- The `MockBackend` (task 14) must be created before property tests in tasks 8, 10, 12, 13, and 15 can be implemented — the dependency graph reflects this
- `chat.rs` public API is intentionally unchanged; the native inference path is additive only
- The existing `engine/inference/runner.rs` candle pipeline is left untouched — it remains the "candle" backend path
- Property tests use `quickcheck`; add `quickcheck = "1"` and `quickcheck_macros = "1"` to `[dev-dependencies]`
- `detect_architecture` (task 6) requires reading GGUF KV metadata — the existing `gguf_parser.rs` currently skips KV entries; `arch_detect.rs` will need its own KV-reading pass that extracts string values rather than skipping

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["2.1", "3.1"] },
    { "id": 2, "tasks": ["2.2", "3.2", "3.3", "3.4", "4.1"] },
    { "id": 3, "tasks": ["5.1", "6.1", "14.1"] },
    { "id": 4, "tasks": ["5.2", "5.3", "6.2", "6.3", "6.4", "6.5", "8.1", "9.1"] },
    { "id": 5, "tasks": ["8.2", "8.3", "9.2", "10.1"] },
    { "id": 6, "tasks": ["8.4", "8.5", "8.6", "8.7", "8.8", "9.3", "9.4", "9.5", "9.6", "10.2", "10.3", "10.4"] },
    { "id": 7, "tasks": ["12.1"] },
    { "id": 8, "tasks": ["12.2", "13.1", "15.1", "15.2"] },
    { "id": 9, "tasks": ["13.2"] }
  ]
}
```
