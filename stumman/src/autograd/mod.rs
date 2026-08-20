//! Stummañ Kevskrid: autograd engine.
//!
//! Sub-modules:
//! - [`node`][]: ComputationNode, TensorId, NodeId, BackwardFn
//! - [`tape`][]: Tape, the forward recorder and backward driver
//! - [`grad_store`][]: VLGradStore, gradient accumulator
//! - [`ops`][]: Oberour, pure f32 helpers the backward closures run on
//! - `check`: numerical gradient checker (Wave 4)
//!
//! Nothing here names a backend type. Gradients travel as `Vec<f32>` so
//! `Tape` stays non-generic and can span a mixed-backend graph in M4.

pub mod grad_store;
pub mod node;
pub mod ops;
pub mod tape;

pub use grad_store::VLGradStore;
pub use node::{BackwardFn, ComputationNode, InputGrad, NodeId, TensorId};
pub use ops::{matmul_f32, transpose_2d};
pub use tape::{Tape, TensorMeta};
