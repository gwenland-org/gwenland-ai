//! Package discovery & layout (ARTX02 Wave 1).
//!
//! Resolves a package root (directory or uncompressed ZIP) into concrete
//! paths for every execution unit, without reading any file contents.

use std::path::{Path, PathBuf};

use crate::constants::{
    CHECKSUMS_FILENAME, LAYER_FILE_EXTENSION, LAYER_FILE_PREFIX, MANIFEST_FILENAME,
    SHARED_FILENAME,
};
use crate::error::GllmError;

/// How the package is stored on disk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PackageFormat {
    /// Extracted directory (runtime-preferred).
    Directory,
    /// Uncompressed ZIP archive (.gllm extension).
    /// Stored files inside the ZIP are directly mmap-able.
    ZipArchive,
}

/// A resolved layer file with its index.
#[derive(Debug, Clone)]
pub struct LayerPath {
    /// Layer index parsed from the `layer_NNN.gllm` filename.
    pub index: u32,
    /// Absolute (or root-relative) path to the layer file.
    pub path: PathBuf,
}

/// Resolved paths to all execution units within a GLLM package.
#[derive(Debug, Clone)]
pub struct PackageLayout {
    /// Root path (directory or ZIP file path).
    pub root: PathBuf,
    /// Storage format detected at the root.
    pub format: PackageFormat,
    /// Path to `gllm.json`.
    pub manifest_path: PathBuf,
    /// Path to `shared.gllm`.
    pub shared_path: PathBuf,
    /// Paths to `layer_NNN.gllm`, sorted by layer index ascending.
    pub layer_paths: Vec<LayerPath>,
    /// Optional `projector.gllm`.
    pub projector_path: Option<PathBuf>,
    /// Optional `checksums.sha256`.
    pub checksums_path: Option<PathBuf>,
}

/// Filename for the optional multimodal projector unit.
pub const PROJECTOR_FILENAME: &str = "projector.gllm";

impl PackageLayout {
    /// Discover package layout from a root path.
    ///
    /// Detects format automatically: a directory is a
    /// [`PackageFormat::Directory`] package; a file with the `.gllm`
    /// extension is a [`PackageFormat::ZipArchive`] package.
    ///
    /// Errors with [`GllmError::MissingManifest`] /
    /// [`GllmError::MissingSharedComponent`] when the required files are
    /// absent, and [`GllmError::InvalidPackageFormat`] for anything that is
    /// neither a directory nor a `.gllm` archive.
    pub fn discover(root: &Path) -> Result<Self, GllmError> {
        if root.is_dir() {
            Self::discover_directory(root)
        } else if root.is_file() {
            match root.extension().and_then(|e| e.to_str()) {
                Some("gllm") => Err(GllmError::InvalidPackageFormat(format!(
                    "{}: ZIP-archive packages are detected but not yet readable \
                     (ZIP central-directory parsing lands with the mmap work, ARTX06)",
                    root.display()
                ))),
                _ => Err(GllmError::InvalidPackageFormat(format!(
                    "{}: not a directory and not a .gllm archive",
                    root.display()
                ))),
            }
        } else {
            Err(GllmError::InvalidPackageFormat(format!(
                "{}: path does not exist",
                root.display()
            )))
        }
    }

    fn discover_directory(root: &Path) -> Result<Self, GllmError> {
        let manifest_path = root.join(MANIFEST_FILENAME);
        if !manifest_path.is_file() {
            return Err(GllmError::MissingManifest);
        }
        let shared_path = root.join(SHARED_FILENAME);
        if !shared_path.is_file() {
            return Err(GllmError::MissingSharedComponent);
        }

        let mut layer_paths: Vec<LayerPath> = Vec::new();
        for entry in std::fs::read_dir(root)? {
            let entry = entry?;
            if !entry.file_type()?.is_file() {
                continue;
            }
            let file_name = entry.file_name();
            let Some(name) = file_name.to_str() else {
                continue;
            };
            if let Some(index) = parse_layer_index(name)? {
                if layer_paths.iter().any(|lp| lp.index == index) {
                    return Err(GllmError::InvalidPackageFormat(format!(
                        "duplicate layer index {index} in {}",
                        root.display()
                    )));
                }
                layer_paths.push(LayerPath {
                    index,
                    path: entry.path(),
                });
            }
        }
        layer_paths.sort_by_key(|lp| lp.index);

        let projector_path = optional_file(root.join(PROJECTOR_FILENAME));
        let checksums_path = optional_file(root.join(CHECKSUMS_FILENAME));

        Ok(Self {
            root: root.to_path_buf(),
            format: PackageFormat::Directory,
            manifest_path,
            shared_path,
            layer_paths,
            projector_path,
            checksums_path,
        })
    }

    /// Returns total number of layers discovered.
    pub fn layer_count(&self) -> usize {
        self.layer_paths.len()
    }

    /// Returns the layer entry for the given layer index (not the position
    /// in the vec — indices need not be contiguous). `None` if absent.
    pub fn layer_path(&self, index: u32) -> Option<&LayerPath> {
        self.layer_paths.iter().find(|lp| lp.index == index)
    }
}

/// Parse `layer_NNN.gllm` → `Some(NNN)`; non-layer filenames → `None`.
///
/// A filename that matches the `layer_*.gllm` pattern but whose middle part
/// is not a number is an error, not a skip — Fail Fast, Fail Loud.
fn parse_layer_index(name: &str) -> Result<Option<u32>, GllmError> {
    let Some(rest) = name.strip_prefix(LAYER_FILE_PREFIX) else {
        return Ok(None);
    };
    let Some(digits) = rest.strip_suffix(LAYER_FILE_EXTENSION) else {
        return Ok(None);
    };
    digits.parse::<u32>().map(Some).map_err(|_| {
        GllmError::InvalidPackageFormat(format!(
            "layer filename {name:?} has a non-numeric index {digits:?}"
        ))
    })
}

fn optional_file(path: PathBuf) -> Option<PathBuf> {
    path.is_file().then_some(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// Creates a minimal valid package dir: manifest + shared + N layers.
    fn make_package_dir(dir: &Path, layer_indices: &[u32]) {
        fs::write(dir.join(MANIFEST_FILENAME), b"{}").unwrap();
        fs::write(dir.join(SHARED_FILENAME), b"dummy").unwrap();
        for &i in layer_indices {
            fs::write(dir.join(format!("layer_{i:03}.gllm")), b"dummy").unwrap();
        }
    }

    #[test]
    fn test_discover_directory_layout() {
        let tmp = tempfile::tempdir().unwrap();
        make_package_dir(tmp.path(), &[0, 1, 2]);

        let layout = PackageLayout::discover(tmp.path()).unwrap();
        assert_eq!(layout.format, PackageFormat::Directory);
        assert_eq!(layout.layer_count(), 3);
        assert!(layout.manifest_path.ends_with(MANIFEST_FILENAME));
        assert!(layout.shared_path.ends_with(SHARED_FILENAME));
        assert!(layout.projector_path.is_none());
        assert!(layout.checksums_path.is_none());
    }

    #[test]
    fn test_discover_missing_manifest() {
        let tmp = tempfile::tempdir().unwrap();
        fs::write(tmp.path().join(SHARED_FILENAME), b"dummy").unwrap();

        let err = PackageLayout::discover(tmp.path()).unwrap_err();
        assert!(matches!(err, GllmError::MissingManifest), "got {err:?}");
    }

    #[test]
    fn test_discover_missing_shared() {
        let tmp = tempfile::tempdir().unwrap();
        fs::write(tmp.path().join(MANIFEST_FILENAME), b"{}").unwrap();

        let err = PackageLayout::discover(tmp.path()).unwrap_err();
        assert!(
            matches!(err, GllmError::MissingSharedComponent),
            "got {err:?}"
        );
    }

    #[test]
    fn test_layer_path_ordering() {
        // Directory listing order is not numeric order; layout must be.
        let tmp = tempfile::tempdir().unwrap();
        make_package_dir(tmp.path(), &[10, 0, 1]);

        let layout = PackageLayout::discover(tmp.path()).unwrap();
        let indices: Vec<u32> = layout.layer_paths.iter().map(|lp| lp.index).collect();
        assert_eq!(indices, vec![0, 1, 10]);
    }

    #[test]
    fn test_layer_path_by_index() {
        let tmp = tempfile::tempdir().unwrap();
        make_package_dir(tmp.path(), &[0, 1, 2]);

        let layout = PackageLayout::discover(tmp.path()).unwrap();
        let lp = layout.layer_path(1).unwrap();
        assert_eq!(lp.index, 1);
        assert!(lp.path.ends_with("layer_001.gllm"));
        assert!(layout.layer_path(99).is_none());
    }

    #[test]
    fn test_discover_with_projector() {
        let tmp = tempfile::tempdir().unwrap();
        make_package_dir(tmp.path(), &[0]);
        fs::write(tmp.path().join(PROJECTOR_FILENAME), b"dummy").unwrap();
        fs::write(tmp.path().join(CHECKSUMS_FILENAME), b"").unwrap();

        let layout = PackageLayout::discover(tmp.path()).unwrap();
        assert!(layout.projector_path.is_some());
        assert!(layout.checksums_path.is_some());
    }

    #[test]
    fn test_discover_no_layers_ok() {
        // Shared-only model (e.g. embeddings-only) is a valid package.
        let tmp = tempfile::tempdir().unwrap();
        make_package_dir(tmp.path(), &[]);

        let layout = PackageLayout::discover(tmp.path()).unwrap();
        assert_eq!(layout.layer_count(), 0);
        assert!(layout.layer_path(0).is_none());
    }

    #[test]
    fn test_discover_zip_archive_not_yet_supported() {
        let tmp = tempfile::tempdir().unwrap();
        let zip = tmp.path().join("model.gllm");
        fs::write(&zip, b"PK\x03\x04").unwrap();

        let err = PackageLayout::discover(&zip).unwrap_err();
        assert!(
            matches!(err, GllmError::InvalidPackageFormat(_)),
            "got {err:?}"
        );
    }

    #[test]
    fn test_discover_rejects_non_numeric_layer_name() {
        let tmp = tempfile::tempdir().unwrap();
        make_package_dir(tmp.path(), &[0]);
        fs::write(tmp.path().join("layer_abc.gllm"), b"dummy").unwrap();

        let err = PackageLayout::discover(tmp.path()).unwrap_err();
        assert!(
            matches!(err, GllmError::InvalidPackageFormat(_)),
            "got {err:?}"
        );
    }
}
