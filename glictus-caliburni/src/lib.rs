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
pub mod error;
pub mod execution_unit;
pub mod package;
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
pub use package::{GllmPackage, LayerPath, PackageFormat, PackageLayout};
pub use shared::SharedComponents;
pub use types::{
    execution::{Device, DeviceMap, ExecutionUnitMeta},
    extension::ExtensionUri,
    layer::LayerFile,
    manifest::GllmManifest,
    package::GllmPackageMeta,
    tensor::{DType, Shape, TensorEntry},
};
pub use traits::{
    converter::GllmConverter,
    plugin::LayerPlugin,
    runtime::GllmRuntime,
};

/// GLLM format version string
pub const FORMAT_VERSION: &str = "1.0.0";

/// Codename
pub const CODENAME: &str = "Ictus Caliburni";
