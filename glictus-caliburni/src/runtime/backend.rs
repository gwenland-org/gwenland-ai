//! Execution backend trait (ARTX05 AD-02).
//!
//! The runtime orchestrates; a backend computes. Everything the runtime knows
//! about tensor math arrives through this trait, which is implemented outside
//! this crate — in practice by an adapter over glproc or glcuda.
//!
//! Nothing here computes anything. Adding a kernel to this module would make
//! `glictus-caliburni` a second engine, which is exactly what the backend
//! boundary exists to prevent.

use std::sync::{Arc, Mutex};

use crate::error::GllmResult;
use crate::manifest::LayerManifest;
use crate::runtime::kv_cache::KvCacheSlot;
use crate::runtime::mmap::LayerMapping;
use crate::runtime::types::ActivationBuffer;

/// Bytes per KV element for a backend that keeps its cache in `f32`.
pub const KV_ELEMENT_SIZE_F32: usize = 4;

/// Bytes per KV element for a backend that keeps its cache in `f16`/`bf16`.
pub const KV_ELEMENT_SIZE_F16: usize = 2;

/// Computes one layer.
///
/// Implementors own their threading, their numerics, and their KV element
/// format. The runtime supplies a mapped layer, the manifest describing it,
/// and buffers to read from and write into.
pub trait ExecutionBackend: Send + Sync {
    /// Short identifier for logs (e.g. `"glproc-cpu"`, `"null"`).
    fn name(&self) -> &'static str;

    /// Whether this backend can run on the current machine.
    ///
    /// Checked at runtime, never at build time — a missing GPU makes a
    /// backend unavailable, not a compile error.
    fn is_available(&self) -> bool;

    /// Bytes per element this backend stores in the KV cache.
    ///
    /// The backend writes the cache, so it alone decides the format. The
    /// runtime reads this once at initialization to size the allocation and
    /// never interprets the bytes themselves — see
    /// [`KvCacheConfig`](crate::runtime::kv_cache::KvCacheConfig).
    ///
    /// Defaults to `f32`, the safe over-estimate: a backend that actually
    /// uses `f16` and forgets to override this over-allocates, whereas the
    /// reverse would corrupt the cache.
    fn kv_element_size(&self) -> usize {
        KV_ELEMENT_SIZE_F32
    }

    /// Execute one layer, writing into `output`.
    fn execute_layer(
        &self,
        layer: &LayerMapping,
        manifest: &LayerManifest,
        kv_cache: &mut KvCacheSlot,
        input: &ActivationBuffer,
        output: &mut ActivationBuffer,
    ) -> GllmResult<()>;
}

/// A backend that records calls and computes nothing.
///
/// Lets the orchestration be tested — scheduling, mapping, unmapping, KV
/// lifecycle — without depending on an engine. Its outputs are meaningless by
/// design: any test asserting on activation *values* is testing this stub,
/// not the runtime.
#[derive(Debug, Default)]
pub struct NullBackend {
    calls: Arc<Mutex<Vec<u32>>>,
    kv_element_size: Option<usize>,
}

impl NullBackend {
    /// A null backend reporting the default (`f32`) KV element size.
    pub fn new() -> Self {
        Self::default()
    }

    /// A null backend reporting a specific KV element size — used to prove
    /// the runtime takes its sizing from the backend rather than a constant.
    pub fn with_kv_element_size(size: usize) -> Self {
        NullBackend {
            calls: Arc::new(Mutex::new(Vec::new())),
            kv_element_size: Some(size),
        }
    }

    /// Layer indices passed to [`execute_layer`](Self::execute_layer), in
    /// call order.
    pub fn calls(&self) -> Vec<u32> {
        match self.calls.lock() {
            Ok(g) => g.clone(),
            Err(poisoned) => poisoned.into_inner().clone(),
        }
    }

    /// How many times the backend was invoked.
    pub fn call_count(&self) -> usize {
        self.calls().len()
    }
}

impl ExecutionBackend for NullBackend {
    fn name(&self) -> &'static str {
        "null"
    }

    fn is_available(&self) -> bool {
        true
    }

    fn kv_element_size(&self) -> usize {
        self.kv_element_size.unwrap_or(KV_ELEMENT_SIZE_F32)
    }

    fn execute_layer(
        &self,
        layer: &LayerMapping,
        _manifest: &LayerManifest,
        _kv_cache: &mut KvCacheSlot,
        input: &ActivationBuffer,
        output: &mut ActivationBuffer,
    ) -> GllmResult<()> {
        match self.calls.lock() {
            Ok(mut g) => g.push(layer.layer_index.unwrap_or(u32::MAX)),
            Err(poisoned) => poisoned.into_inner().push(layer.layer_index.unwrap_or(u32::MAX)),
        }
        // Pass the activation through unchanged: shape stays valid for the
        // next layer without inventing numbers that could be mistaken for a
        // real result.
        output.data.clear();
        output.data.extend_from_slice(&input.data);
        output.shape.clone_from(&input.shape);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn null_backend_is_available_and_named() {
        let b = NullBackend::new();
        assert_eq!(b.name(), "null");
        assert!(b.is_available());
    }

    #[test]
    fn default_kv_element_size_is_f32() {
        assert_eq!(NullBackend::new().kv_element_size(), KV_ELEMENT_SIZE_F32);
        assert_eq!(KV_ELEMENT_SIZE_F32, 4);
        assert_eq!(KV_ELEMENT_SIZE_F16, 2);
    }

    #[test]
    fn kv_element_size_is_overridable_per_backend() {
        let b = NullBackend::with_kv_element_size(KV_ELEMENT_SIZE_F16);
        assert_eq!(b.kv_element_size(), 2);
    }

    #[test]
    fn trait_default_applies_to_a_backend_that_does_not_override() {
        // A minimal implementor that overrides nothing optional must still
        // report a usable KV element size.
        struct Bare;
        impl ExecutionBackend for Bare {
            fn name(&self) -> &'static str {
                "bare"
            }
            fn is_available(&self) -> bool {
                true
            }
            fn execute_layer(
                &self,
                _l: &LayerMapping,
                _m: &LayerManifest,
                _k: &mut KvCacheSlot,
                _i: &ActivationBuffer,
                _o: &mut ActivationBuffer,
            ) -> GllmResult<()> {
                Ok(())
            }
        }
        assert_eq!(Bare.kv_element_size(), KV_ELEMENT_SIZE_F32);
    }

    #[test]
    fn backend_is_object_safe() {
        // Held as Arc<dyn ExecutionBackend> by the runtime; this must compile.
        let b: Arc<dyn ExecutionBackend> = Arc::new(NullBackend::new());
        assert_eq!(b.name(), "null");
        assert_eq!(b.kv_element_size(), 4);
    }
}
