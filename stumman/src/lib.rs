//! # Stummañ — GwenLand Training Framework
//!
//! Codename: Stummañ (Breton: "to train, to form")
//! Version: M1 Wave 1 — Core Tensor Abstraction
//!
//! Sub-systems:
//! - Kevrin  (tensor):   [`tensor`] module
//! - Karg    (backend):  [`backend`] module
//! - Kevskrid (autograd): [`autograd`] module — records only, no replay yet
//! - Gwellaer (optimizer): M2+

pub mod autograd;
pub mod backend;
pub mod error;
pub mod tensor;

// Convenient top-level re-exports
pub use autograd::{NodeId, Tape, TensorId};
pub use backend::{GlProc, SisdBackend};
pub use error::{GlTrainError, Result};
pub use tensor::{Backend, Tensor};
