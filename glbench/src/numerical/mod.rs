//! GLBitProf: bit-level observation of tensor values.
//!
//! Three modules, split along the line D-11 draws:
//!
//! - [`bitprof`] — the math. Pure `&[f32] -> VLBitProfile`, std-only, **no
//!   feature gate**. Bit-profiling a gradient has nothing to do with `.gllm`
//!   packages, so gating the math on the package reader would be wrong.
//! - [`compare`] — divergence between two profiles, also ungated.
//! - [`scope`] — where the values come from. This is where the gates live,
//!   one per source.
//!
//! The design note worth keeping in view: `tensor_stats.rs` was the obvious
//! place to grow this, and it is gated behind `gllm-bench`. Extending it in
//! place would have made every gradient and optimizer profile a
//! `gllm-bench`-only feature by accident.

pub mod bitprof;
pub mod compare;
pub mod scope;
