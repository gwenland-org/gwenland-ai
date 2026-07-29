//! High-level LLM ops (ARTX03).
//!
//! Each op takes and returns [`Tensor`](crate::tensor::Tensor) and reads the
//! ambient [`PrecisionPolicy`](crate::PrecisionPolicy) where numerics demand
//! it. Nothing here emits MLIR directly except through
//! [`crate::stablehlo::ops`], and nothing here knows what a model is — that is
//! `model/`.
//!
//! # Where the corrections live
//!
//! Three of these ops deviate from ARTX03's sketches on points that decide
//! whether the output is right rather than merely well-shaped. Each carries the
//! reasoning at its definition:
//!
//! * [`rope::rope_neox`] — NeoX is the **half-split** `(i, i+D/2)`, settled
//!   against glproc's validated implementation, not ARTX03's adjacent pairing.
//! * [`norm::rms_norm`] — ε on the mean, inside the sqrt.
//! * [`attention::gqa_attention`] — KV heads repeat consecutively, so query
//!   head `h` reads KV head `h / repeat`.

pub mod attention;
pub mod embedding;
pub mod ffn;
pub mod kv_cache;
pub mod linear;
pub mod moe;
pub mod norm;
pub mod rope;
pub mod softmax;
pub(crate) mod util;

pub use attention::{causal_mask, gqa_attention};
pub use embedding::gather_embed;
pub use ffn::swiglu_ffn;
pub use linear::linear;
pub use norm::rms_norm;
pub use rope::{emit_rope_tables, rope_neox, rope_tables, DEFAULT_ROPE_BASE};
pub use softmax::softmax;
