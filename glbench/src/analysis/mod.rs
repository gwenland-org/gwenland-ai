//! Analysis: turn measurements into insight. Every conclusion is a
//! recommendation phrased as an observation — glbench observes, it never
//! optimizes. [`summary::analyze`] is the entry point; the other modules are the
//! individual analyzers it runs.

pub mod bottleneck;
pub mod ceiling;
pub mod efficiency;
pub mod health;
pub mod roofline;
pub mod scaling;
pub mod summary;
