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

pub mod constants;
pub mod error;
pub mod package;
pub mod traits;
pub mod types;

// Re-exports untuk convenience
pub use error::{GllmError, GllmResult};
pub use package::{LayerPath, PackageFormat, PackageLayout};
pub use types::{
    execution::{Device, DeviceMap, ExecutionUnit},
    extension::ExtensionUri,
    layer::LayerFile,
    manifest::GllmManifest,
    package::GllmPackage,
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
