//! Benchmark execution. [`planner`] orchestrates a full run (load → warmup →
//! measured iterations → analysis + validation); [`warmup`], [`prefill`],
//! [`decode`], and [`stress`] hold the per-phase policy and stability helpers.

pub mod decode;
pub mod planner;
pub mod prefill;
pub mod stress;
pub mod warmup;
