//! # glcore
//!
//! Shared foundation for the GwenLand AI inference engine: tensor types,
//! error handling, model file parsers (GGUF, safetensors), a from-scratch
//! BPE tokenizer, the [`engine_trait::GlEngine`] contract every backend
//! implements, the [`runtime::Runtime`] that front-ends drive, and the
//! [`gate`] protocol boilerplate (see `architecture/GATE/README.md`).
//!
//! Zero external ML dependencies — everything is built from scratch.

pub mod engine_trait;
pub mod error;
pub mod format;
pub mod gate;
/// SHA-256, the workspace's single hash primitive. Lives here because both
/// `glictus-caliburni` (`.gllm` checksums) and `glbench` (archive content
/// digests) need it and neither depends on the other.
pub mod hash;
pub mod runtime;
pub mod stopping;
pub mod telemetry;
pub mod tensor;
/// The tokenizer: SentencePiece and byte-level BPE, 14 GGUF vocabulary
/// families verified exact against reference vectors.
///
/// ⚠️ This module *name* previously held a different implementation, which
/// scored 65.2%–97.8% per vocabulary with none fully correct. That code was
/// deleted in step 4; nothing here descends from it.
pub mod tokenizer;
pub mod trace;

pub use engine_trait::{EngineSpec, GlEngine, InferInput, InferOutput};
pub use stopping::StoppingCriteria;
pub use telemetry::{
    BackendTelemetry, EngineTelemetry, MemoryTelemetry, MoeTelemetry, PhaseProfile, StageTiming,
};
pub use trace::{TokenTrace, TraceConfig};
pub use error::GlError;
pub use runtime::Runtime;
pub use tensor::{DType, Tensor};
pub use tokenizer::{GllmTokenizer, TokError};
