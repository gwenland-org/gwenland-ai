//! # glmetal
//!
//! Apple Silicon Metal backend for GwenLand AI. **Stub for M1** — every
//! method reports "not yet implemented" so the runtime fallback chain can be
//! wired up in M2 without conditional compilation.

use glcore::engine_trait::{EngineSpec, GlEngine, InferInput, InferOutput};
use glcore::GlError;

/// Placeholder Metal engine. Compiles everywhere; runs nowhere yet.
#[derive(Default)]
pub struct GlmetalEngine;

impl GlmetalEngine {
    /// Create the stub engine.
    pub fn new() -> Self {
        Self
    }
}

fn not_implemented() -> GlError {
    GlError::Engine("glmetal not yet implemented".into())
}

impl GlEngine for GlmetalEngine {
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
            name: "glmetal",
            backend: "metal",
            available: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stub_reports_unavailable() {
        let mut e = GlmetalEngine::new();
        assert!(!e.capabilities().available);
        assert!(e.init().is_err());
    }
}
