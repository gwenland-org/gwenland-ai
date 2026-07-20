//! # glictus-caliburni
//!
//! GLLM (GwenLand Language Model Format) — Ictus Caliburni
//!
//! A correctness-first, layer-native binary format for LLM inference.
//! Part of the GwenLand AI ecosystem.
//!
//! ## Design Philosophy
//!
//! - **Storage Follows Execution** — physical layout mirrors execution flow
//! - **Metadata Is Executable** — manifest is a machine-readable contract
//! - **Fail Fast, Fail Loud** — checksum per file, detected at load time
//! - **Extensibility Over Generality** — narrow focus, plugin for the rest

pub mod checksum;
pub mod constants;
#[cfg(feature = "converter")]
pub mod converter;
pub mod error;
pub mod execution_unit;
pub mod layer_io;
pub mod manifest;
pub mod package;
pub mod runtime;
pub mod shared;
pub mod traits;
pub mod types;

#[cfg(test)]
pub(crate) mod test_helpers;

// Re-exports untuk convenience
pub use checksum::{ChecksumEntry, ChecksumVerifier, sha256_bytes, sha256_file};
pub use error::{GllmError, GllmResult};
pub use execution_unit::{
    ExecutionUnit, ExecutionUnitHeader, GLLM_CURRENT_VERSION, GLLM_HEADER_SIZE, GLLM_MAGIC,
};
pub use layer_io::{cross_check_manifest, read_entry, write_entry, write_unit_file};
pub use manifest::{
    CustomMetadata, DType, DevicePlacement, ExtensionUri, FormatVersion, GllmManifest,
    KnownDevicePlacement, LayerManifest, ManifestValidator, ModelMetadata, ProjectorManifest,
    RUNTIME_FORMAT_VERSION, RopeScaling, SharedManifest, TensorEntry, ValidationResult,
    VersionCompatibility,
};
pub use package::{GllmPackage, LayerPath, PackageFormat, PackageLayout};
pub use runtime::{
    ActivationBuffer, AdaptivePrefetcher, CpuRuntime, DeviceMapConfig, DeviceMapResolver,
    DeviceSource, ExecutionBackend, ExecutionState, ExecutionStats, GllmRuntime, KvCache,
    KvCacheConfig, KvCacheSlot, LayerMapping, LayerScheduleState, LogEntry, NullBackend,
    ResolvedDevice, RuntimeConfig, RuntimeLogLevel, RuntimeLogger, Scheduler,
};
pub use shared::SharedComponents;
pub use types::{
    execution::{Device, DeviceMap, ExecutionUnitMeta},
    layer::LayerFile,
    package::GllmPackageMeta,
};
pub use traits::{
    converter::GllmConverter,
    plugin::LayerPlugin,
    inference::GllmInference,
};

/// GLLM format version string
pub const FORMAT_VERSION: &str = "1.0.0";

/// Codename
pub const CODENAME: &str = "Ictus Caliburni";
