//! Stummañ Kevrin — the tensor sub-system.
//!
//! [`backend`] declares the compute contract, [`tensor`] is the user-facing
//! data structure, [`ops`] is reserved for ops that outgrow `Tensor` methods.

pub mod backend;
pub mod ops;
// `tensor::tensor` is module inception, which clippy flags by default. The
// path is fixed by the Stummañ plan (Part 3.3) and matches the sibling
// `autograd/`, `nn/`, `optim/` layout landing in later waves; `Tensor` is
// re-exported below so callers never spell the inner module.
#[allow(clippy::module_inception)]
pub mod tensor;

pub use backend::Backend;
pub use tensor::Tensor;
