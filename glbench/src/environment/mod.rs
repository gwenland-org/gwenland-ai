//! Environment probes: what machine, what build. [`hardware`] aggregates the
//! per-component probes ([`cpu`], [`gpu`], [`memory`], [`storage`]) and
//! [`runtime`] into the snapshots a session carries.

pub mod bandwidth;
pub mod cpu;
pub mod gpu;
pub mod hardware;
pub mod memory;
pub mod runtime;
pub mod storage;
