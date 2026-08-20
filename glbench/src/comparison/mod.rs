//! Comparison: a first-class subsystem for run-vs-run, engine-vs-engine,
//! quantization, hardware, statistics, regression, and trend. All are views of
//! the same [`runs::compare`] delta along a particular axis. glbench compares;
//! it never routes between engines.

pub mod accuracy;
pub mod engine;
pub mod hardware;
pub mod quantization;
pub mod regression;
pub mod runs;
pub mod statistics;
/// Training-configuration comparison (Wave 4). Gated with the training tree.
#[cfg(feature = "train-bench")]
pub mod training;
pub mod trend;
