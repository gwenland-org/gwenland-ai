use std::path::PathBuf;
use crate::error::{GllmError, GllmResult};
use crate::types::manifest::GllmManifest;

/// Represents a loaded GLLM package (directory or ZIP)
#[derive(Debug, Clone)]
pub struct GllmPackage {
    /// Root directory atau ZIP path
    pub root: PathBuf,
    /// Parsed manifest
    pub manifest: GllmManifest,
    /// Package format
    pub format: PackageFormat,
}

// PackageFormat moved to the package-level module in ARTX02 (Wave 1);
// re-exported here so the ARTX01 path keeps working.
pub use crate::package::PackageFormat;

impl GllmPackage {
    pub fn manifest_path(&self) -> PathBuf {
        self.root.join("gllm.json")
    }

    pub fn shared_path(&self) -> PathBuf {
        self.root.join(&self.manifest.shared.file)
    }

    pub fn layer_path(&self, index: usize) -> GllmResult<PathBuf> {
        let entry = self.manifest.layer_file(index)
            .ok_or(GllmError::LayerOutOfBounds {
                index,
                max: self.manifest.num_layers().saturating_sub(1),
            })?;
        Ok(self.root.join(&entry.file))
    }

    pub fn num_layers(&self) -> usize {
        self.manifest.num_layers()
    }

    /// Expected layer filename untuk index tertentu
    pub fn expected_layer_filename(index: usize) -> String {
        format!("layer_{index:03}.gllm")
    }

    /// Total estimated size in bytes
    pub fn estimated_size_bytes(&self) -> u64 {
        self.manifest.estimated_memory_bytes()
    }
}
