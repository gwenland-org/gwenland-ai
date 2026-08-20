//! Training observation (D-05, D-07). Gated behind `train-bench`.
//!
//! # glbench observes; stumman drives
//!
//! Everything here is downstream of a run stumman controls. [`runner`]
//! constructs a `Trainer`, installs a [`collector::VLStepCollector`], and calls
//! `Trainer::train`. It never calls `train_step` in a loop of its own, never
//! touches optimizer state, and never writes a parameter — the boundary
//! `glbench/DESIGN.md` §1 draws between measuring and doing, applied to
//! training.
//!
//! # The boundary is non-generic
//!
//! stumman's `Backend` is not dyn-compatible (KL-001), so a naive observer API
//! would have to be generic over `B` and monomorphise into glbench. It does not
//! have to be: everything that crosses is a scalar, a count, or `&[f32]` plus a
//! shape (design F-04). No type in this module carries a backend parameter.
//!
//! # What has no subject at M2
//!
//! stumman M2 trains one linear layer: no tokenizer, no batching beyond one
//! sample, no data parallelism, f32 only. Every token-denominated field, every
//! synchronisation field and every mixed-precision field therefore has nothing
//! to measure. They are still in the schema, carrying `not_applicable` rather
//! than being omitted or zeroed — D-04, and the null-semantics vocabulary
//! earning its place.

pub mod adapter;
pub mod attribution;
pub mod collector;
pub mod convergence;
pub mod memory;
pub mod runner;
pub mod session;
pub mod step;

pub use session::VLTrainingSession;
pub use step::VLTrainingStep;
