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
pub mod runtime;
pub mod stopping;
pub mod telemetry;
pub mod tensor;
/// ⛔ **Superseded by the `gltokenizer` crate.** Zero production callers
/// remain; it is kept only so `tests/tokenizer_before_after.rs` can still
/// measure what it replaced.
///
/// Measured against llama.cpp's reference vectors it scored 65.2%–97.8% per
/// vocabulary — *no* vocabulary was fully correct. Do not build on it.
#[deprecated(
    since = "0.1.164",
    note = "use the `gltokenizer` crate; this scored 65-98% against reference vectors"
)]
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
pub use tokenizer::Tokenizer;
