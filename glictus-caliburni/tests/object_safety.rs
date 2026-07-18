//! Wave 3 gate: the three public traits must stay object-safe.
//!
//! `GllmRuntime` is held as `Box<dyn GllmRuntime>` by the runtime, and
//! `LayerPlugin` lives in a `Box<dyn LayerPlugin>` registry. If a future
//! change adds a generic method or `impl Trait` in argument/return position,
//! this file stops compiling — which is the point.

use std::path::Path;

use glictus_caliburni::error::GllmResult;
use glictus_caliburni::types::execution::Device;
use glictus_caliburni::types::package::GllmPackage;
use glictus_caliburni::types::tensor::{DType, TensorEntry};
use glictus_caliburni::{GllmConverter, GllmRuntime, LayerPlugin};

/// Minimal runtime that returns fixed logits — exercises the default `stream`.
struct StubRuntime {
    logits: Vec<f32>,
    loaded: bool,
    devices: Vec<Device>,
}

impl GllmRuntime for StubRuntime {
    fn load_package(&mut self, _package: &GllmPackage) -> GllmResult<()> {
        self.loaded = true;
        Ok(())
    }

    fn infer(&mut self, _input_ids: &[u32]) -> GllmResult<Vec<f32>> {
        Ok(self.logits.clone())
    }

    fn unload(&mut self) {
        self.loaded = false;
    }

    fn supported_devices(&self) -> &[Device] {
        &self.devices
    }

    fn name(&self) -> &str {
        "stub-runtime"
    }
}

struct StubPlugin;

impl LayerPlugin for StubPlugin {
    fn parse_tensors(&self, index: &[TensorEntry]) -> GllmResult<Vec<TensorEntry>> {
        Ok(index.to_vec())
    }

    fn memory_requirement_bytes(&self, tensors: &[TensorEntry], _device: &Device) -> u64 {
        tensors.iter().map(|t| t.size).sum()
    }

    fn execute(
        &self,
        inputs: &[f32],
        outputs: &mut Vec<f32>,
        _tensors: &[TensorEntry],
    ) -> GllmResult<()> {
        outputs.clear();
        outputs.extend_from_slice(inputs);
        Ok(())
    }

    fn supports_dtype(&self, dtype: DType) -> bool {
        matches!(dtype, DType::F32)
    }

    fn uri(&self) -> &str {
        "gllm:transformer/standard@v1"
    }
}

struct StubConverter;

impl GllmConverter for StubConverter {
    fn source_format(&self) -> &str {
        "stub"
    }

    fn convert(&self, _source: &Path, _output_dir: &Path) -> GllmResult<GllmPackage> {
        Err(glictus_caliburni::GllmError::ValidationError(
            "stub converter does not convert".into(),
        ))
    }

    fn validate(&self, _package: &GllmPackage) -> GllmResult<()> {
        Ok(())
    }
}

#[test]
fn traits_are_object_safe() {
    // The coercions themselves are the assertion: a non-object-safe trait
    // fails to compile here.
    let runtime: Box<dyn GllmRuntime> = Box::new(StubRuntime {
        logits: vec![0.0],
        loaded: false,
        devices: vec![Device::Cpu],
    });
    let plugin: Box<dyn LayerPlugin> = Box::new(StubPlugin);
    let converter: Box<dyn GllmConverter> = Box::new(StubConverter);

    assert_eq!(runtime.name(), "stub-runtime");
    assert_eq!(plugin.uri(), "gllm:transformer/standard@v1");
    assert_eq!(converter.source_format(), "stub");
}

#[test]
fn default_stream_emits_argmax_token_through_dyn() {
    let mut runtime: Box<dyn GllmRuntime> = Box::new(StubRuntime {
        // argmax is index 3
        logits: vec![0.1, -2.0, 0.9, 4.2, 1.5],
        loaded: false,
        devices: vec![Device::Cpu],
    });

    let mut seen = Vec::new();
    runtime
        .stream(&[1, 2, 3], &mut |token| {
            seen.push(token);
            true
        })
        .expect("stream succeeds");

    assert_eq!(seen, vec![3], "default stream should emit the argmax token");
}

#[test]
fn plugin_reports_dtype_support() {
    let plugin: Box<dyn LayerPlugin> = Box::new(StubPlugin);
    assert!(plugin.supports_dtype(DType::F32));
    assert!(!plugin.supports_dtype(DType::Q4K));
}
