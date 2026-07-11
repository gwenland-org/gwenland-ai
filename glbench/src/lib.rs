//! # glbench — Mensura Veritatis
//!
//! A standalone benchmark execution and performance-analysis framework for
//! GwenLand AI. glbench measures the truth about engine performance:
//!
//! ```text
//! Execute → Measure → Analyze → Compare → Validate → Report
//! ```
//!
//! **glbench is not an optimizer.** It observes performance; engine developers
//! optimize it. glbench never touches a kernel, a model file, or a hardware
//! setting — it runs inference through the existing [`glcore::engine_trait::GlEngine`]
//! contract and reports what the hardware did.
//!
//! ## Architecture
//!
//! The [`core::session::BenchmarkSession`] is the single source of truth — a
//! pure data model every subsystem reads or fills. The pipeline stages are:
//!
//! - [`environment`] — snapshot the machine (CPU/GPU/memory/storage/runtime).
//! - [`engine`] — the single boundary to the engines; runs inference through
//!   glcore's `Runtime`, never duplicating inference logic.
//! - [`runner`] — orchestrate a run: warmup, measured iterations, phases.
//! - [`measurement`] — store raw facts (latency, tok/s, bytes), never verdicts.
//! - [`analysis`] — turn facts into insight (health, bottleneck, ceiling),
//!   always as recommendations, never actions.
//! - [`comparison`] — run/engine/quantization/hardware deltas, regression, trend.
//! - [`validation`] — is the benchmark trustworthy? (integrity, determinism,
//!   numerical parity vs the glproc oracle.)
//! - [`export`] / [`render`] / [`storage`] — JSON/Markdown/CSV, terminal output,
//!   user-managed archive files (no database).
//!
//! ## Dependency rule
//!
//! Zero new external dependencies: the standard library and existing GwenLand
//! workspace crates only. The JSON/CSV/Markdown writers are hand-rolled.

pub mod analysis;
pub mod comparison;
pub mod core;
pub mod engine;
pub mod environment;
pub mod export;
pub mod measurement;
pub mod render;
pub mod runner;
pub mod storage;
pub mod validation;

pub use crate::core::session::BenchmarkSession;
pub use crate::core::workload::{WorkloadKind, WorkloadSpec};
