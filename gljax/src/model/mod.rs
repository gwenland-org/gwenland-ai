//! Model definitions — architectures composed from [`crate::ops`].
//!
//! ⚠️ Qwen2-shaped, deliberately. ARTX11 A11.0 records that ARTX03's op layer
//! "is Qwen2-shaped in 7 ways" and that a generic `Architecture` descriptor is
//! a retrofit that blocks multi-architecture support. That retrofit is not this
//! sprint's job; keeping the Qwen2 assumptions in one named module — rather
//! than spread through `ops/` — is what makes it a retrofit rather than a
//! rewrite.

pub mod hf_config;
pub mod qwen2;

pub use hf_config::eos_token_id;
pub use qwen2::{trace_forward, Qwen2Config};
