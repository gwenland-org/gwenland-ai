//! Graph tracing: SSA values, shape inference, and the tracing context
//! (ARTX02 §4–§6).
//!
//! The split is deliberate and is what keeps shape bugs findable:
//!
//! | Layer | Knows about | Does not know about |
//! |---|---|---|
//! | [`crate::stablehlo::ops`] | MLIR syntax | shapes, validity |
//! | [`FuncBuilder`] | shapes, dtypes, validity | scope names, checkpoints |
//! | [`TraceCx`] | scopes, parameter names | MLIR syntax |
//! | [`crate::tensor::Tensor`] | nothing — it is a handle | everything above |

pub mod builder;
pub mod trace;
pub mod value;

pub use builder::{BuiltFunc, FuncBuilder, Signature};
pub use trace::TraceCx;
pub use value::SsaValue;
