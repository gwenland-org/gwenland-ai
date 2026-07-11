//! Comparison: a first-class subsystem for run-vs-run, engine-vs-engine,
//! quantization, hardware, statistics, regression, and trend. All are views of
//! the same [`runs::compare`] delta along a particular axis. glbench compares;
//! it never routes between engines.

pub mod engine;
pub mod hardware;
pub mod quantization;
pub mod regression;
pub mod runs;
pub mod statistics;
pub mod trend;
