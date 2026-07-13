//! # glproc
//!
//! The GwenLand AI CPU inference engine — pure Rust, zero external ML
//! dependencies, correctness first. This crate is the source of truth the
//! GPU backends (glcuda, glvulkan, glmetal) are validated against.
//!
//! Modules mirror the inference pipeline: [`loader`] turns a GGUF file into
//! a [`model::GlprocModel`]; [`runner`] drives the transformer forward pass
//! using [`matmul`], [`attention`] and [`kv_cache`]; [`sampler`] picks the
//! next token; [`engine`] wraps it all behind [`glcore::GlEngine`].

pub mod attention;
pub mod engine;
pub mod kernels;
pub mod kv_cache;
pub mod loader;
pub mod memory;
pub mod moe;
pub mod model;
pub mod runner;
pub mod sampler;
pub mod simd_strategy;
pub mod threading;
pub mod topology;

pub use engine::{GlprocConfig, GlprocEngine};
