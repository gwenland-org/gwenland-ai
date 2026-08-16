//! Stummañ Kevskrid — Autograd engine.
//!
//! Sub-modules:
//! - [`node`][]: ComputationNode, TensorId, NodeId
//! - [`tape`][]: Tape (forward-pass recorder)
//! - `ops/`: per-op backward functions (Wave 3)
//! - `check`: numerical gradient checker (Wave 4)
//!
//! Wave 2 records the forward pass and nothing else. No node's backward
//! function is ever invoked here — see [`node::BackwardFn`].

pub mod node;
pub mod tape;

pub use node::{BackwardFn, ComputationNode, NodeId, TensorId};
pub use tape::{Tape, TensorMeta};
