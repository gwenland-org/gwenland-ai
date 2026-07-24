//! Benchmark execution. [`planner`] orchestrates a full run (load → warmup →
//! measured iterations → analysis + validation); [`warmup`], [`prefill`],
//! [`decode`], and [`stress`] hold the per-phase policy and stability helpers.

pub mod decode;
pub mod planner;
pub mod prefill;
pub mod scale;
pub mod stress;
pub mod thread_scale;
pub mod warmup;
