//! Fused SwiGLU gating: `gate[i] = silu(gate[i]) * up[i]`.
//!
//! Runs `hidden_dim` times per layer per token between the up and down
//! projections — worth its own vector kernel because the scalar loop pays
//! a `fast_exp` call per element.

pub mod avx2;
pub mod scalar;
