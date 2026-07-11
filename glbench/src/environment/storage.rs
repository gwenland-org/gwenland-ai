//! Storage facts relevant to a benchmark: the size of the model file being
//! loaded (which decode must stream from memory, and load must read from disk).

use std::path::Path;

/// Observed storage facts for the workload's model file.
#[derive(Debug, Clone, Default)]
pub struct StorageInfo {
    /// Size of the model file on disk in bytes, if it exists.
    pub model_file_bytes: Option<u64>,
}

impl StorageInfo {
    /// Probe the size of `model_path`.
    pub fn probe(model_path: &str) -> StorageInfo {
        let model_file_bytes = Path::new(model_path)
            .metadata()
            .ok()
            .map(|m| m.len())
            .filter(|&n| n > 0);
        StorageInfo { model_file_bytes }
    }
}
