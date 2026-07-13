//! # glcore
//!
//! Shared foundation for the GwenLand AI inference engine: tensor types,
//! error handling, model file parsers (GGUF, safetensors), a from-scratch
//! BPE tokenizer, the [`engine_trait::GlEngine`] contract every backend
//! implements, and the [`runtime::Runtime`] that front-ends drive.
//!
//! Zero external ML dependencies — everything is built from scratch.

pub mod engine_trait;
pub mod error;
pub mod format;
pub mod runtime;
pub mod telemetry;
pub mod tensor;
pub mod tokenizer;
pub mod trace;

pub use engine_trait::{EngineSpec, GlEngine, InferInput, InferOutput};
pub use telemetry::{
    BackendTelemetry, EngineTelemetry, MemoryTelemetry, MoeTelemetry, PhaseProfile, StageTiming,
};
pub use trace::{TokenTrace, TraceConfig};
pub use error::GlError;
pub use runtime::Runtime;
pub use tensor::{DType, Tensor};
pub use tokenizer::Tokenizer;
