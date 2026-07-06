//! # glcuda
//!
//! NVIDIA CUDA backend for GwenLand AI. **Stub for M1** — every method
//! reports "not yet implemented" so the runtime fallback chain can be wired
//! up in M2 without conditional compilation.

use glcore::engine_trait::{EngineSpec, GlEngine, InferInput, InferOutput};
use glcore::GlError;

/// Placeholder CUDA engine. Compiles everywhere; runs nowhere yet.
#[derive(Default)]
pub struct GlcudaEngine;

impl GlcudaEngine {
    /// Create the stub engine.
    pub fn new() -> Self {
        Self
    }
}

fn not_implemented() -> GlError {
    GlError::Engine("glcuda not yet implemented".into())
}

impl GlEngine for GlcudaEngine {
    fn init(&mut self) -> Result<(), GlError> {
        Err(not_implemented())
    }

    fn load_model(&mut self, _path: &str) -> Result<(), GlError> {
        Err(not_implemented())
    }

    fn infer(&self, _input: InferInput) -> Result<InferOutput, GlError> {
        Err(not_implemented())
    }

    fn shutdown(&mut self) {}

    fn capabilities(&self) -> EngineSpec {
        EngineSpec {
            name: "glcuda",
            backend: "cuda",
            available: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stub_reports_unavailable() {
        let mut e = GlcudaEngine::new();
        assert!(!e.capabilities().available);
        assert!(e.init().is_err());
        assert!(e.load_model("x.gguf").is_err());
    }
}
