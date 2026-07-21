//! GLLM runtime — layer-sequential execution (ARTX05 + ARTX06).
//!
//! The runtime maps one layer at a time, hands it to a backend, and unmaps it,
//! keeping the working set at roughly one layer plus shared components.
//!
//! ## Scope
//!
//! This module **orchestrates**; it does not compute. All tensor math is
//! delegated to an [`ExecutionBackend`](backend::ExecutionBackend)
//! implementation — in practice an adapter over glproc or glcuda. No GEMM,
//! attention, or activation kernel lives here, and none should: those belong
//! to the engines, which are tuned and validated per hardware model.
//!
//! ## Relationship to [`GllmInference`](crate::traits::inference::GllmInference)
//!
//! [`GllmRuntime`] is the *layer-level* API: one forward pass, no notion of
//! tokens. `GllmInference` is the token-level API (tokens in, logits out). A
//! backend will eventually implement the latter on top of the former, once
//! tokenizer packaging (ARTX1 OQ3) is settled.
//!
//! ## Status
//!
//! CPU path only. The GPU (ARTX05 §"GPU Runtime") and hybrid paths are
//! deliberately absent until they can follow glcuda's runtime dynamic-loading
//! pattern rather than a build-time CUDA dependency.

pub mod backend;
pub mod cpu;
pub mod device;
pub mod distributed;
#[cfg(feature = "glproc-backend")]
pub mod gllm_engine;
#[cfg(feature = "glproc-backend")]
pub mod glproc_backend;
pub mod kv_cache;
pub mod logger;
pub mod mmap;
#[allow(clippy::module_inception)]
pub mod runtime;
pub mod scheduler;
pub mod types;

pub use backend::{ExecutionBackend, NullBackend, KV_ELEMENT_SIZE_F16, KV_ELEMENT_SIZE_F32};
pub use cpu::CpuRuntime;
#[cfg(feature = "glproc-backend")]
pub use gllm_engine::GllmEngine;
#[cfg(feature = "glproc-backend")]
pub use glproc_backend::GlprocBackend;
pub use device::{
    DeviceMapConfig, DeviceMapResolver, DeviceSource, ResolvedDevice, parse_range,
};
pub use distributed::{RankAssignment, RankTopology};
pub use kv_cache::{KvCache, KvCacheConfig, KvCacheSlot};
pub use logger::{LogEntry, RuntimeLogger};
pub use mmap::LayerMapping;
pub use runtime::GllmRuntime;
pub use scheduler::{AdaptivePrefetcher, LayerScheduleState, Scheduler};
pub use types::{
    ActivationBuffer, ExecutionState, ExecutionStats, RuntimeConfig, RuntimeLogLevel,
};
