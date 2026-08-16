//! # Stummañ — GwenLand Training Framework
//!
//! Codename: Stummañ (Breton: "to train, to form")
//! Version: M1 Wave 1 — Core Tensor Abstraction
//!
//! Sub-systems:
//! - Kevrin  (tensor):   [`tensor`] module
//! - Karg    (backend):  [`backend`] module
//! - Kevskrid (autograd): Wave 2+
//! - Gwellaer (optimizer): Wave 2+

pub mod backend;
pub mod error;
pub mod tensor;

// Convenient top-level re-exports
pub use backend::{GlProc, SisdBackend};
pub use error::{GlTrainError, Result};
pub use tensor::{Backend, Tensor};
