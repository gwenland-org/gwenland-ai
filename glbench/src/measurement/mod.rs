//! Measurement: raw facts and the conversions between them. Stores numbers,
//! never conclusions (that is [`crate::analysis`]). [`raw`] is the single seam
//! that turns an engine's output into glbench's iteration metrics.

pub mod bandwidth;
pub mod memory;
pub mod raw;
pub mod throughput;
pub mod timeline;
pub mod timing;
