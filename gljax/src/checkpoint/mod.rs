//! Checkpoint loading — binding real weights to a traced signature.
//!
//! ⛔ **The sprint brief names APIs that do not exist.** It says to use
//! `glcore::GllmCheckpoint` and `SafetensorsCheckpoint` and "do NOT
//! re-implement checkpoint loading — it's already verified lossless". What
//! `glcore` actually exports is:
//!
//! | brief | reality |
//! |---|---|
//! | `glcore::GllmCheckpoint` | does not exist; `glcore::format::gllm` is one function, `decode_tensor` |
//! | `glcore::SafetensorsCheckpoint` | does not exist; the type is `glcore::format::SafetensorsFile` |
//!
//! `SafetensorsFile` is real, mmap-backed and already parses the header — so
//! nothing is re-implemented here. This module is the *binding* layer: given a
//! traced [`Signature`](crate::graph::Signature) and a checkpoint, produce the
//! tensors in call order, or refuse.
//!
//! The `.gllm` path goes through `glictus-caliburni`, which is a heavier
//! dependency than gljax's budget allows today — see [`GLLM_STATUS`].

pub mod safetensors;

pub use safetensors::{bind_safetensors, WeightSource};

/// Why `.gllm` is not wired up yet.
///
/// Reading a `.gllm` package means depending on `glictus-caliburni` (the
/// format's owner — a directory layout with `GLLMShared.gllm` and
/// `GLLMTensorLayer-NNNN.gllm` files, a manifest, and checksum verification).
/// That is a real dependency decision, not an oversight, and ARTX01 §5.4 caps
/// gljax's dependency list at three crates.
///
/// The binding layer in [`safetensors`] is deliberately written against
/// `&dyn WeightSource` so adding a `.gllm` source later is a new impl, not a
/// rewrite.
pub const GLLM_STATUS: &str =
    "gljax reads safetensors only. `.gllm` needs a glictus-caliburni dependency \
     (ARTX01 §5.4 caps the list at glcore/libloading/log) — implement WeightSource for it.";
