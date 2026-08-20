//! Package discovery & layout (ARTX02 Wave 1).
//!
//! Resolves a package root (directory or uncompressed ZIP) into concrete
//! paths for every execution unit, without reading any file contents.

use std::path::{Path, PathBuf};

use crate::checksum::{ChecksumEntry, ChecksumVerifier};
use crate::constants::{
    CHECKSUMS_FILENAME, LAYER_FILE_EXTENSION, LAYER_FILE_PREFIX, MANIFEST_FILENAME,
    PROJECTOR_FILENAME, SHARED_FILENAME,
};
use crate::error::GllmError;
use crate::execution_unit::ExecutionUnit;
use crate::manifest::{GllmManifest, LayerManifest, ManifestValidator, ValidationResult};
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
    /// Layer index parsed from the `GLLMTensorLayer-NNNN.gllm` filename.
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
    /// Path to `GLLMShared.gllm`.
    pub shared_path: PathBuf,
    /// Paths to `GLLMTensorLayer-NNNN.gllm`, sorted by layer index ascending.
    pub layer_paths: Vec<LayerPath>,
    /// Optional `GLLMProj.gllm`.
    pub projector_path: Option<PathBuf>,
    /// Optional `checksums.sha256`.
    pub checksums_path: Option<PathBuf>,
}

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

/// Parse `GLLMTensorLayer-NNNN.gllm` → `Some(NNNN)`; other filenames → `None`.
///
/// Leading zeros are accepted and normalised away, so `-0007` and `-7` both
/// yield 7 — discovery is tolerant of padding width even though
/// [`format_layer_filename`](crate::manifest::format_layer_filename) always
/// writes four digits.
///
/// A filename that matches the prefix/extension pattern but whose middle part
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
    /// Parsed and semantically validated `gllm.json`.
    pub manifest: GllmManifest,
    /// Header-validated, checksum-verified `GLLMShared.gllm`.
    pub shared: SharedComponents,
    /// Lazily opened layer units, parallel to `layout.layer_paths`.
    /// `None` = layer not yet opened.
    layer_units: Vec<Option<ExecutionUnit>>,
    /// Loaded from `checksums.sha256` when present.
    pub checksum_verifier: Option<ChecksumVerifier>,
    /// Validation outcome from [`Self::open`] — errors are always empty
    /// (fatal errors abort `open`), warnings are preserved.
    pub validation: ValidationResult,
}

impl GllmPackage {
    /// Open a GLLM package at `root`.
    ///
    /// Steps: discover layout → parse `gllm.json` → run the semantic
    /// validator (fatal findings abort with
    /// [`GllmError::ManifestValidationError`]) → open `GLLMShared.gllm`
    /// verifying it against the manifest's checksum → load
    /// `checksums.sha256` when present. Layer files stay unopened (lazy).
    pub fn open(root: &Path) -> Result<Self, GllmError> {
        let layout = PackageLayout::discover(root)?;

        let manifest = GllmManifest::from_file(&layout.manifest_path)?;
        let validation = ManifestValidator::new(&manifest).validate();
        if !validation.is_ok() {
            return Err(GllmError::ManifestValidationError(
                validation.errors.join("; "),
            ));
        }

        let shared_hex = manifest.shared.checksum_hex()?;
        let shared = SharedComponents::open(&layout.shared_path, Some(&shared_hex))?;

        let checksum_verifier = match &layout.checksums_path {
            Some(path) => Some(ChecksumVerifier::from_file(path)?),
            None => None,
        };

        let layer_units = (0..layout.layer_count()).map(|_| None).collect();
        Ok(Self {
            layout,
            manifest,
            shared,
            layer_units,
            checksum_verifier,
            validation,
        })
    }

    /// The parsed manifest.
    pub fn manifest(&self) -> &GllmManifest {
        &self.manifest
    }

    /// Non-fatal validation warnings collected during [`Self::open`].
    pub fn warnings(&self) -> &[String] {
        &self.validation.warnings
    }

    /// True when validation produced warnings.
    pub fn has_warnings(&self) -> bool {
        self.validation.has_warnings()
    }

    /// Manifest entry for a layer by index (valid manifests have
    /// position == index, enforced by validator rule V07).
    pub fn layer_manifest(&self, index: u32) -> Option<&LayerManifest> {
        self.manifest.layers.get(index as usize)
    }

    /// Build a [`ChecksumVerifier`] from the manifest's per-file
    /// checksums — the alternative when `checksums.sha256` is absent.
    pub fn verifier_from_manifest(&self) -> Result<ChecksumVerifier, GllmError> {
        let mut entries = vec![ChecksumEntry {
            filename: self.manifest.shared.file.clone(),
            expected_sha256: self.manifest.shared.checksum_hex()?,
        }];
        for layer in &self.manifest.layers {
            entries.push(ChecksumEntry {
                filename: layer.file.clone(),
                expected_sha256: layer.checksum_hex()?,
            });
        }
        if let Some(projector) = &self.manifest.projector {
            entries.push(ChecksumEntry {
                filename: projector.file.clone(),
                expected_sha256: projector.checksum_hex()?,
            });
        }
        Ok(ChecksumVerifier::from_entries(entries))
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

    /// Read and parse a layer's full binary tensor index (ARTX04).
    /// Unlike [`Self::open_layer`] this parses the whole index, not just
    /// the header; the result is not cached.
    pub fn read_layer_file(&self, index: u32) -> Result<crate::types::layer::LayerFile, GllmError> {
        let lp = self
            .layout
            .layer_path(index)
            .ok_or(GllmError::LayerOutOfBounds {
                index: index as usize,
                max: self.layout.layer_count().saturating_sub(1),
            })?;
        crate::types::layer::LayerFile::read(&lp.path)
    }

    /// Cross-check a layer's binary tensor index against its manifest
    /// entry. Empty result = consistent. Errors if the layer index is
    /// unknown to layout or manifest.
    pub fn cross_check_layer(&self, index: u32) -> Result<Vec<String>, GllmError> {
        let layer = self.read_layer_file(index)?;
        let manifest = self.layer_manifest(index).ok_or(GllmError::LayerOutOfBounds {
            index: index as usize,
            max: self.manifest.num_layers().saturating_sub(1),
        })?;
        Ok(crate::layer_io::cross_check_manifest(&layer, manifest))
    }

    /// Check every layer's tensor index against the plugin for its declared
    /// type (ARTX8).
    ///
    /// Returns all `(unit, problem)` findings without short-circuiting; empty
    /// means every layer is structurally valid for the type it claims to be.
    /// Findings come in two kinds: a layer type no plugin implements, and a
    /// layer whose tensors do not match its type's layout.
    ///
    /// This reads each layer's index from disk but never its tensor data, so
    /// cost is independent of model size. It is deliberately **not** part of
    /// [`open`](Self::open): resolving layer types needs a registry the
    /// caller owns, and a package whose exotic layer types this build cannot
    /// describe is still worth opening for inspection.
    pub fn validate_layer_types(
        &self,
        registry: &crate::plugin::PluginRegistry,
    ) -> Vec<(String, String)> {
        let mut findings = Vec::new();

        for layer in &self.manifest.layers {
            let Some(plugin) = registry.resolve(&layer.layer_type) else {
                findings.push((
                    layer.file.clone(),
                    format!("no plugin registered for layer type {}", layer.layer_type),
                ));
                continue;
            };

            // Prefer the binary index — it is what the runtime will actually
            // map. Fall back to the manifest's copy when the file cannot be
            // read, so a missing layer is reported as one finding rather than
            // aborting the whole sweep.
            let index = match self.read_layer_file(layer.index) {
                Ok(file) => file.tensor_index,
                Err(e) => {
                    findings.push((layer.file.clone(), format!("unreadable: {e}")));
                    continue;
                }
            };

            if let Err(e) = plugin.validate_layout(&index) {
                findings.push((layer.file.clone(), e.to_string()));
            }
        }

        if let Some(projector) = &self.manifest.projector {
            match registry.resolve(&projector.projector_type) {
                Some(plugin) => {
                    if let Err(e) = plugin.validate_layout(&projector.tensors) {
                        findings.push((projector.file.clone(), e.to_string()));
                    }
                }
                None => findings.push((
                    projector.file.clone(),
                    format!(
                        "no plugin registered for projector type {}",
                        projector.projector_type
                    ),
                )),
            }
        }

        findings
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
            fs::write(dir.join(crate::manifest::format_layer_filename(i)), b"dummy").unwrap();
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
        assert!(lp.path.ends_with("GLLMTensorLayer-0001.gllm"));
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

    /// Full on-disk package with a valid manifest whose checksums match
    /// the written files (ARTX03: `open` now parses + validates it).
    fn make_valid_package_dir(dir: &Path, num_layers: u32) {
        crate::test_helpers::fixtures::write_manifest_package(dir, num_layers);
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
        make_valid_package_dir(tmp.path(), 0);

        let pkg = GllmPackage::open(tmp.path()).unwrap();
        assert_eq!(pkg.layer_count(), 0);
        assert!(!pkg.has_checksum_file());
        // ARTX03: shared is always verified against the manifest checksum.
        assert!(pkg.shared.is_verified());
    }

    #[test]
    fn test_package_open_with_layers() {
        let tmp = tempfile::tempdir().unwrap();
        make_valid_package_dir(tmp.path(), 3);

        let pkg = GllmPackage::open(tmp.path()).unwrap();
        assert_eq!(pkg.layer_count(), 3);
    }

    #[test]
    fn test_package_open_layer_lazy() {
        let tmp = tempfile::tempdir().unwrap();
        make_valid_package_dir(tmp.path(), 2);

        let mut pkg = GllmPackage::open(tmp.path()).unwrap();
        let unit = pkg.open_layer(0).unwrap();
        assert_eq!(unit.header.version, crate::execution_unit::GLLM_CURRENT_VERSION);
        assert!(unit.path.ends_with("GLLMTensorLayer-0000.gllm"));

        // Second call must serve the cache even if the file disappears.
        fs::remove_file(tmp.path().join("GLLMTensorLayer-0000.gllm")).unwrap();
        assert!(pkg.open_layer(0).is_ok());
    }

    #[test]
    fn test_package_open_layer_out_of_range() {
        let tmp = tempfile::tempdir().unwrap();
        make_valid_package_dir(tmp.path(), 1);

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
        make_valid_package_dir(tmp.path(), 1);

        let pkg = GllmPackage::open(tmp.path()).unwrap();
        assert!(!pkg.has_checksum_file());
        assert!(pkg.verify_integrity().is_empty());
    }

    #[test]
    fn test_package_verify_integrity_with_checksums() {
        let tmp = tempfile::tempdir().unwrap();
        make_valid_package_dir(tmp.path(), 2);
        write_checksums_file(tmp.path());

        let pkg = GllmPackage::open(tmp.path()).unwrap();
        assert!(pkg.has_checksum_file());
        // GLLMShared.gllm's entry was found, so open() verified it.
        assert!(pkg.shared.is_verified());
        assert!(pkg.verify_integrity().is_empty());

        // Corrupt one layer: verify_integrity must report exactly it.
        fs::write(tmp.path().join("GLLMTensorLayer-0001.gllm"), b"CORRUPT").unwrap();
        let failures = pkg.verify_integrity();
        assert_eq!(failures.len(), 1);
        assert_eq!(failures[0].0, "GLLMTensorLayer-0001.gllm");
    }

    #[test]
    fn test_package_open_rejects_corrupt_shared_checksum() {
        let tmp = tempfile::tempdir().unwrap();
        make_valid_package_dir(tmp.path(), 1);
        // Corrupt GLLMShared.gllm after the manifest recorded its checksum
        // (keep a valid header so only the checksum check can catch it).
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
    fn test_package_open_with_manifest() {
        let tmp = tempfile::tempdir().unwrap();
        make_valid_package_dir(tmp.path(), 2);

        let pkg = GllmPackage::open(tmp.path()).unwrap();
        assert_eq!(pkg.manifest().model_id, "org.gwenland.test-model");
        assert_eq!(pkg.manifest().num_layers(), 2);
        assert!(!pkg.has_warnings());
    }

    #[test]
    fn test_package_open_manifest_parse_error() {
        let tmp = tempfile::tempdir().unwrap();
        make_valid_package_dir(tmp.path(), 1);
        fs::write(tmp.path().join(MANIFEST_FILENAME), b"{ not json !").unwrap();

        let err = GllmPackage::open(tmp.path()).unwrap_err();
        assert!(
            matches!(err, GllmError::ManifestParseError(_)),
            "got {err:?}"
        );
    }

    #[test]
    fn test_package_open_manifest_validation_error() {
        let tmp = tempfile::tempdir().unwrap();
        make_valid_package_dir(tmp.path(), 2);
        // Valid JSON, inconsistent content: metadata claims 5 layers.
        let manifest = fs::read_to_string(tmp.path().join(MANIFEST_FILENAME)).unwrap();
        fs::write(
            tmp.path().join(MANIFEST_FILENAME),
            manifest.replace("\"num_layers\":2", "\"num_layers\":5"),
        )
        .unwrap();

        let err = GllmPackage::open(tmp.path()).unwrap_err();
        assert!(
            matches!(err, GllmError::ManifestValidationError(_)),
            "got {err:?}"
        );
    }

    #[test]
    fn test_package_warnings_accessible() {
        let tmp = tempfile::tempdir().unwrap();
        make_valid_package_dir(tmp.path(), 1);
        // Same major, newer minor → loadable with a V02 warning.
        let manifest = fs::read_to_string(tmp.path().join(MANIFEST_FILENAME)).unwrap();
        fs::write(
            tmp.path().join(MANIFEST_FILENAME),
            manifest.replace("\"1.0.0\"", "\"1.1.0\""),
        )
        .unwrap();

        let pkg = GllmPackage::open(tmp.path()).unwrap();
        assert!(pkg.has_warnings());
        assert!(pkg.warnings()[0].contains("V02"), "{:?}", pkg.warnings());
    }

    #[test]
    fn test_package_layer_manifest_lookup() {
        let tmp = tempfile::tempdir().unwrap();
        make_valid_package_dir(tmp.path(), 3);

        let pkg = GllmPackage::open(tmp.path()).unwrap();
        let lm = pkg.layer_manifest(1).unwrap();
        assert_eq!(lm.index, 1);
        assert_eq!(lm.file, "GLLMTensorLayer-0001.gllm");
        assert!(pkg.layer_manifest(3).is_none());
    }

    #[test]
    fn test_package_verifier_from_manifest() {
        let tmp = tempfile::tempdir().unwrap();
        make_valid_package_dir(tmp.path(), 2);

        let pkg = GllmPackage::open(tmp.path()).unwrap();
        assert!(!pkg.has_checksum_file(), "no checksums.sha256 was written");
        let verifier = pkg.verifier_from_manifest().unwrap();
        assert_eq!(verifier.entries.len(), 3); // shared + 2 layers
        let failures = verifier.verify_all(tmp.path());
        assert!(failures.is_empty(), "failures: {failures:?}");
    }

    #[test]
    fn test_package_read_layer_file_empty_index() {
        // Fixture layers are pre-ARTX04 (tensor_count = 0): the full
        // binary read must succeed with an empty index.
        let tmp = tempfile::tempdir().unwrap();
        make_valid_package_dir(tmp.path(), 1);

        let pkg = GllmPackage::open(tmp.path()).unwrap();
        let layer = pkg.read_layer_file(0).unwrap();
        assert!(layer.tensor_index.is_empty());
        // Manifest fixture lists no layer tensors either → consistent.
        assert!(pkg.cross_check_layer(0).unwrap().is_empty());

        let err = pkg.read_layer_file(9).unwrap_err();
        assert!(matches!(err, GllmError::LayerOutOfBounds { .. }), "got {err:?}");
    }

    #[test]
    fn test_package_open_manifest_validates_shared_checksum() {
        let tmp = tempfile::tempdir().unwrap();
        make_valid_package_dir(tmp.path(), 1);
        // Swap the manifest's shared checksum for a wrong (but
        // well-formed) digest — the file itself is untouched.
        let manifest = fs::read_to_string(tmp.path().join(MANIFEST_FILENAME)).unwrap();
        let shared_hex = crate::checksum::sha256_file(&tmp.path().join(SHARED_FILENAME)).unwrap();
        let wrong = crate::checksum::sha256_bytes(b"wrong");
        fs::write(
            tmp.path().join(MANIFEST_FILENAME),
            manifest.replace(&shared_hex, &wrong),
        )
        .unwrap();

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
        fs::write(tmp.path().join("GLLMTensorLayer-abc.gllm"), b"dummy").unwrap();

        let err = PackageLayout::discover(tmp.path()).unwrap_err();
        assert!(
            matches!(err, GllmError::InvalidPackageFormat(_)),
            "got {err:?}"
        );
    }

    #[test]
    fn validate_layer_types_flags_a_layer_that_lies_about_its_type() {
        use crate::plugin::PluginRegistry;
        use crate::test_helpers::write_runtime_package;

        // The fixture's layers declare gllm:transformer/standard@v1 but carry
        // only attn_q/attn_k — exactly the shape of a converter that dropped
        // a family of tensors.
        let dir = write_runtime_package(2);
        let pkg = GllmPackage::open(dir.path()).unwrap();

        let findings = pkg.validate_layer_types(&PluginRegistry::with_builtins());
        assert_eq!(findings.len(), 2, "one finding per layer: {findings:?}");
        for (file, problem) in &findings {
            assert!(file.starts_with("GLLMTensorLayer-"), "{file}");
            assert!(
                problem.contains("ffn_down.weight") && problem.contains("attn_v.weight"),
                "every missing tensor should be listed: {problem}"
            );
        }
    }

    #[test]
    fn validate_layer_types_reports_an_unregistered_type() {
        use crate::plugin::PluginRegistry;
        use crate::test_helpers::write_runtime_package;

        let dir = write_runtime_package(1);
        let pkg = GllmPackage::open(dir.path()).unwrap();

        // An empty registry knows no types at all.
        let findings = pkg.validate_layer_types(&PluginRegistry::new());
        assert_eq!(findings.len(), 1);
        assert!(
            findings[0].1.contains("no plugin registered"),
            "{:?}",
            findings[0]
        );
    }

    #[test]
    fn parse_layer_index_round_trips_with_format_layer_filename() {
        // The writer and the reader must agree, or a converted package
        // becomes undiscoverable. This is the test that would have caught
        // the naming change breaking discovery.
        for index in [0u32, 1, 9, 10, 99, 100, 999, 1000, 9999, 10_000] {
            let name = crate::manifest::format_layer_filename(index);
            assert_eq!(
                parse_layer_index(&name).unwrap(),
                Some(index),
                "round-trip failed for {name}"
            );
        }
    }

    #[test]
    fn parse_layer_index_ignores_foreign_filenames() {
        for name in [
            "gllm.json",
            "GLLMShared.gllm",
            "GLLMProj.gllm",
            "checksums.sha256",
            "layer_000.gllm", // the previous convention is no longer a layer
            "GLLMTensorLayer-0000.zip",
        ] {
            assert_eq!(parse_layer_index(name).unwrap(), None, "{name}");
        }
    }

    /// Writes a complete on-disk package like
    /// [`crate::test_helpers::fixtures::write_manifest_package`], but with
    /// `gllm_version` overridden — for exercising ARTX09 version negotiation
    /// through the real `GllmPackage::open` path (not just the validator
    /// unit tests in `manifest::metadata`).
    fn write_package_with_version(dir: &Path, version: &str) {
        use crate::checksum::sha256_file;
        use crate::test_helpers::make_test_gllm_file;

        make_test_gllm_file(&dir.join(SHARED_FILENAME));
        let shared_hex = sha256_file(&dir.join(SHARED_FILENAME)).unwrap();

        let manifest = serde_json::json!({
            "gllm_version": version,
            "model_id": "org.gwenland.test-model",
            "architecture": "transformer",
            "metadata": {
                "vocab_size": 1000,
                "context_length": 2048,
                "embedding_length": 64,
                "num_layers": 0,
                "num_heads": 8,
                "head_count_kv": 8
            },
            "shared": {
                "file": SHARED_FILENAME,
                "checksum": format!("sha256:{shared_hex}"),
                "tensors": [
                    { "name": "token_embeddings", "shape": [1000, 64],
                      "dtype": "F32", "offset": 0, "size": 256000 }
                ]
            },
            "layers": [],
            "extensions": []
        });
        fs::write(dir.join(MANIFEST_FILENAME), manifest.to_string()).unwrap();
    }

    #[test]
    fn open_rejects_major_version_mismatch() {
        use crate::manifest::{FormatVersion, RUNTIME_FORMAT_VERSION};

        let tmp = tempfile::tempdir().unwrap();
        let (major, minor, patch) = FormatVersion(RUNTIME_FORMAT_VERSION.to_string())
            .parts()
            .unwrap();
        write_package_with_version(tmp.path(), &format!("{}.{minor}.{patch}", major + 1));

        let err = GllmPackage::open(tmp.path()).unwrap_err();
        assert!(
            matches!(err, GllmError::ManifestValidationError(_)),
            "got {err:?}"
        );
        let msg = err.to_string();
        assert!(msg.contains("Format version mismatch"), "{msg}");
    }

    #[test]
    fn open_warns_on_minor_version_mismatch() {
        use crate::manifest::{FormatVersion, RUNTIME_FORMAT_VERSION};

        let tmp = tempfile::tempdir().unwrap();
        let (major, minor, patch) = FormatVersion(RUNTIME_FORMAT_VERSION.to_string())
            .parts()
            .unwrap();
        write_package_with_version(tmp.path(), &format!("{major}.{}.{patch}", minor + 1));

        let pkg = GllmPackage::open(tmp.path()).expect("minor mismatch must still load");
        assert!(pkg.validation.has_warnings());
        assert!(
            pkg.validation
                .warnings
                .iter()
                .any(|w| w.contains("minor mismatch")),
            "{:?}",
            pkg.validation.warnings
        );
    }

    #[test]
    fn open_is_silent_on_patch_only_difference() {
        use crate::manifest::{FormatVersion, RUNTIME_FORMAT_VERSION};

        let tmp = tempfile::tempdir().unwrap();
        let (major, minor, patch) = FormatVersion(RUNTIME_FORMAT_VERSION.to_string())
            .parts()
            .unwrap();
        write_package_with_version(tmp.path(), &format!("{major}.{minor}.{}", patch + 1));

        let pkg = GllmPackage::open(tmp.path()).expect("patch difference must load");
        assert!(
            !pkg.validation.has_warnings(),
            "patch-only difference must not warn: {:?}",
            pkg.validation.warnings
        );
    }
}
