//! Core data model: the [`session::BenchmarkSession`] single source of truth and
//! its component types. Pure data — no business logic lives in `core`.

pub mod availability;
pub mod inference;
pub mod metrics;
pub mod mode;
pub mod result;
pub mod schema;
pub mod session;
pub mod workload;
