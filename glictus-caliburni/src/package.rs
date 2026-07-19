//! Package discovery & layout (ARTX02 Wave 1).
//!
//! Resolves a package root (directory or uncompressed ZIP) into concrete
//! paths for every execution unit, without reading any file contents.

use std::path::{Path, PathBuf};

use crate::checksum::ChecksumVerifier;
use crate::constants::{
    CHECKSUMS_FILENAME, LAYER_FILE_EXTENSION, LAYER_FILE_PREFIX, MANIFEST_FILENAME,
    SHARED_FILENAME,
};
use crate::error::GllmError;
use crate::execution_unit::ExecutionUnit;
use crate::shared::SharedComponents;

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

/// Top-level handle to an opened GLLM package (ARTX02 Wave 5).
///
/// Provides access to the resolved layout, the validated shared
/// components, and lazily opened per-layer execution units.
#[derive(Debug)]
pub struct GllmPackage {
    /// Resolved paths for every execution unit.
    pub layout: PackageLayout,
    /// Header-validated (and, when a checksum was available, verified)
    /// `shared.gllm`.
    pub shared: SharedComponents,
    /// Lazily opened layer units, parallel to `layout.layer_paths`.
    /// `None` = layer not yet opened.
    layer_units: Vec<Option<ExecutionUnit>>,
    /// Loaded from `checksums.sha256` when present.
    pub checksum_verifier: Option<ChecksumVerifier>,
}

impl GllmPackage {
    /// Open a GLLM package at `root`.
    ///
    /// Discovers the layout, loads `checksums.sha256` when present, and
    /// opens `shared.gllm` — verifying its checksum if the checksum file
    /// carries an entry for it. Layer files are NOT opened here (lazy).
    pub fn open(root: &Path) -> Result<Self, GllmError> {
        let layout = PackageLayout::discover(root)?;

        let checksum_verifier = match &layout.checksums_path {
            Some(path) => Some(ChecksumVerifier::from_file(path)?),
            None => None,
        };

        let shared_checksum = checksum_verifier
            .as_ref()
            .and_then(|v| v.expected_for(SHARED_FILENAME));
        let shared = SharedComponents::open(&layout.shared_path, shared_checksum)?;

        let layer_units = (0..layout.layer_count()).map(|_| None).collect();
        Ok(Self {
            layout,
            shared,
            layer_units,
            checksum_verifier,
        })
    }

    /// Open and header-validate the layer with the given index.
    ///
    /// The result is cached — subsequent calls return the cached unit
    /// without touching the filesystem.
    pub fn open_layer(&mut self, index: u32) -> Result<&ExecutionUnit, GllmError> {
        let pos = self
            .layout
            .layer_paths
            .iter()
            .position(|lp| lp.index == index)
            .ok_or(GllmError::LayerOutOfBounds {
                index: index as usize,
                max: self.layout.layer_count().saturating_sub(1),
            })?;

        let unit = match self.layer_units[pos].take() {
            Some(cached) => cached,
            None => ExecutionUnit::open(&self.layout.layer_paths[pos].path)?,
        };
        Ok(self.layer_units[pos].insert(unit))
    }

    /// Verify every entry of the checksum file against the package root.
    ///
    /// Returns all `(filename, error)` failures without short-circuiting;
    /// empty when everything matches — or when no verifier was loaded
    /// (check [`has_checksum_file`](Self::has_checksum_file) to tell the
    /// two apart).
    pub fn verify_integrity(&self) -> Vec<(String, GllmError)> {
        match &self.checksum_verifier {
            Some(verifier) => verifier.verify_all(&self.layout.root),
            None => Vec::new(),
        }
    }

    /// Returns the number of layers in the package.
    pub fn layer_count(&self) -> usize {
        self.layout.layer_count()
    }

    /// Returns true if `checksums.sha256` was found and loaded.
    pub fn has_checksum_file(&self) -> bool {
        self.checksum_verifier.is_some()
    }
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

    /// Like `make_package_dir`, but every `.gllm` file carries a valid
    /// GLLM header so `GllmPackage::open` accepts it.
    fn make_valid_package_dir(dir: &Path, layer_indices: &[u32]) {
        use crate::test_helpers::make_test_gllm_file;
        fs::write(dir.join(MANIFEST_FILENAME), b"{}").unwrap();
        make_test_gllm_file(&dir.join(SHARED_FILENAME));
        for &i in layer_indices {
            make_test_gllm_file(&dir.join(format!("layer_{i:03}.gllm")));
        }
    }

    /// Write a real checksums.sha256 covering shared + all layer files.
    fn write_checksums_file(dir: &Path) {
        use crate::checksum::sha256_file;
        let mut lines = String::new();
        for entry in fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            let name = path.file_name().unwrap().to_str().unwrap().to_string();
            if name.ends_with(LAYER_FILE_EXTENSION) {
                let hash = sha256_file(&path).unwrap();
                lines.push_str(&format!("{hash}  {name}\n"));
            }
        }
        fs::write(dir.join(CHECKSUMS_FILENAME), lines).unwrap();
    }

    #[test]
    fn test_package_open_minimal() {
        let tmp = tempfile::tempdir().unwrap();
        make_valid_package_dir(tmp.path(), &[]);

        let pkg = GllmPackage::open(tmp.path()).unwrap();
        assert_eq!(pkg.layer_count(), 0);
        assert!(!pkg.has_checksum_file());
        assert!(!pkg.shared.is_verified());
    }

    #[test]
    fn test_package_open_with_layers() {
        let tmp = tempfile::tempdir().unwrap();
        make_valid_package_dir(tmp.path(), &[0, 1, 2]);

        let pkg = GllmPackage::open(tmp.path()).unwrap();
        assert_eq!(pkg.layer_count(), 3);
    }

    #[test]
    fn test_package_open_layer_lazy() {
        let tmp = tempfile::tempdir().unwrap();
        make_valid_package_dir(tmp.path(), &[0, 1]);

        let mut pkg = GllmPackage::open(tmp.path()).unwrap();
        let unit = pkg.open_layer(0).unwrap();
        assert_eq!(unit.header.version, crate::execution_unit::GLLM_CURRENT_VERSION);
        assert!(unit.path.ends_with("layer_000.gllm"));

        // Second call must serve the cache even if the file disappears.
        fs::remove_file(tmp.path().join("layer_000.gllm")).unwrap();
        assert!(pkg.open_layer(0).is_ok());
    }

    #[test]
    fn test_package_open_layer_out_of_range() {
        let tmp = tempfile::tempdir().unwrap();
        make_valid_package_dir(tmp.path(), &[0]);

        let mut pkg = GllmPackage::open(tmp.path()).unwrap();
        let err = pkg.open_layer(7).unwrap_err();
        assert!(
            matches!(err, GllmError::LayerOutOfBounds { index: 7, .. }),
            "got {err:?}"
        );
    }

    #[test]
    fn test_package_verify_integrity_no_checksum_file() {
        let tmp = tempfile::tempdir().unwrap();
        make_valid_package_dir(tmp.path(), &[0]);

        let pkg = GllmPackage::open(tmp.path()).unwrap();
        assert!(!pkg.has_checksum_file());
        assert!(pkg.verify_integrity().is_empty());
    }

    #[test]
    fn test_package_verify_integrity_with_checksums() {
        let tmp = tempfile::tempdir().unwrap();
        make_valid_package_dir(tmp.path(), &[0, 1]);
        write_checksums_file(tmp.path());

        let pkg = GllmPackage::open(tmp.path()).unwrap();
        assert!(pkg.has_checksum_file());
        // shared.gllm's entry was found, so open() verified it.
        assert!(pkg.shared.is_verified());
        assert!(pkg.verify_integrity().is_empty());

        // Corrupt one layer: verify_integrity must report exactly it.
        fs::write(tmp.path().join("layer_001.gllm"), b"CORRUPT").unwrap();
        let failures = pkg.verify_integrity();
        assert_eq!(failures.len(), 1);
        assert_eq!(failures[0].0, "layer_001.gllm");
    }

    #[test]
    fn test_package_open_rejects_corrupt_shared_checksum() {
        let tmp = tempfile::tempdir().unwrap();
        make_valid_package_dir(tmp.path(), &[0]);
        write_checksums_file(tmp.path());
        // Corrupt shared.gllm after checksums were recorded (keep a valid
        // header so only the checksum check can catch it).
        let mut bytes = fs::read(tmp.path().join(SHARED_FILENAME)).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xFF;
        fs::write(tmp.path().join(SHARED_FILENAME), bytes).unwrap();

        let err = GllmPackage::open(tmp.path()).unwrap_err();
        assert!(
            matches!(err, GllmError::ChecksumMismatch { .. }),
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
