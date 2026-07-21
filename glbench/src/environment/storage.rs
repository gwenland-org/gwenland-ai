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
    ///
    /// A single file (GGUF, safetensors) reports its own size. A directory
    /// (a GLLM package root: manifest + shared + layer files) reports the
    /// sum of the regular files directly inside it — `Path::metadata` on a
    /// directory returns the directory *inode's* size, a small,
    /// platform-dependent number with no relation to how much the format
    /// actually occupies, and reporting that would silently understate a
    /// GLLM package's footprint rather than fail loud.
    pub fn probe(model_path: &str) -> StorageInfo {
        let path = Path::new(model_path);
        let model_file_bytes = match path.metadata() {
            Ok(m) if m.is_dir() => Self::dir_total_bytes(path),
            Ok(m) => Some(m.len()).filter(|&n| n > 0),
            Err(_) => None,
        };
        StorageInfo { model_file_bytes }
    }

    /// Sum the byte size of every regular file directly inside `dir` (not
    /// recursive — a GLLM package is a flat directory of execution units).
    /// `None` if the directory could not be read at all.
    fn dir_total_bytes(dir: &Path) -> Option<u64> {
        let entries = std::fs::read_dir(dir).ok()?;
        let total: u64 = entries
            .filter_map(|e| e.ok())
            .filter_map(|e| e.metadata().ok())
            .filter(|m| m.is_file())
            .map(|m| m.len())
            .sum();
        Some(total).filter(|&n| n > 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probes_a_single_file_size() {
        let dir = std::env::temp_dir().join(format!("glbench-storage-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("model.gguf");
        std::fs::write(&file, vec![0u8; 1024]).unwrap();

        let info = StorageInfo::probe(file.to_str().unwrap());
        assert_eq!(info.model_file_bytes, Some(1024));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn probes_a_package_directory_as_the_sum_of_its_files() {
        let dir = std::env::temp_dir().join(format!("glbench-storage-test-dir-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("gllm.json"), vec![0u8; 100]).unwrap();
        std::fs::write(dir.join("GLLMShared.gllm"), vec![0u8; 4000]).unwrap();

        let info = StorageInfo::probe(dir.to_str().unwrap());
        assert_eq!(info.model_file_bytes, Some(4100));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_missing_path_probes_to_none() {
        let info = StorageInfo::probe("definitely/not/a/real/path");
        assert_eq!(info.model_file_bytes, None);
    }
}
