//! Compile cache (ARTX01 §6.3, ARTX04).
//!
//! XLA compilation is the slowest thing in the system — ARTX16 §4.2 records
//! 20–30 minutes cold on TPU — and P3 makes every distinct shape a separate
//! compilation. Caching is not an optimisation here, it is what makes bucketing
//! affordable at all.
//!
//! # What is cached, and what is not
//!
//! ⚠️ A serialized `PJRT_Executable` is **plugin- and version-specific**
//! (ARTX01 §6.1): an artifact compiled by CUDA plugin 0.4.1 may not load in
//! 0.5.0, and there is no cross-plugin portability at all. So this is a cache,
//! never a distribution format, and the plugin's own version string is part of
//! the key.
//!
//! ⭐ **Weight *values* are not part of the key.** ARTX01 §8.2's compile/weight
//! separation: the executable is shape-parameterised and value-agnostic, so
//! new weights load into the same artifact. Weight *shapes* are in the key —
//! via the MLIR text, which spells every parameter type.

use std::path::{Path, PathBuf};

use crate::runtime::digest::{hex, sha256};
use crate::GlError;

/// What a cached artifact is keyed on.
///
/// Everything that can change the compiled machine code, and nothing that
/// cannot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompileKey {
    /// `(major, minor)` of the plugin's PJRT API.
    pub plugin_version: (i32, i32),
    /// The plugin's own version string (`PJRT_Client_PlatformVersion`), which
    /// distinguishes two builds reporting the same API version.
    pub platform_version: String,
    /// Platform name — `cpu`, `cuda`, `tpu`.
    pub platform_name: String,
    /// SHA-256 of the StableHLO text. Covers the ops, the shapes, the dtypes
    /// and the parameter order in one value.
    pub mlir_sha256: [u8; 32],
}

impl CompileKey {
    pub fn new(
        plugin_version: (i32, i32),
        platform_name: impl Into<String>,
        platform_version: impl Into<String>,
        mlir: &str,
    ) -> Self {
        CompileKey {
            plugin_version,
            platform_name: platform_name.into(),
            platform_version: platform_version.into(),
            mlir_sha256: sha256(mlir.as_bytes()),
        }
    }

    /// The cache filename stem: a digest over every field, so two keys that
    /// differ anywhere land in different files.
    pub fn digest(&self) -> String {
        let mut buf = Vec::with_capacity(256);
        buf.extend_from_slice(&self.plugin_version.0.to_le_bytes());
        buf.extend_from_slice(&self.plugin_version.1.to_le_bytes());
        // Length-prefixed, so ("ab", "c") and ("a", "bc") do not collide.
        for s in [&self.platform_name, &self.platform_version] {
            buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
            buf.extend_from_slice(s.as_bytes());
        }
        buf.extend_from_slice(&self.mlir_sha256);
        hex(&sha256(&buf))
    }

    /// Hex of the MLIR digest alone — useful in logs.
    pub fn mlir_hex(&self) -> String {
        hex(&self.mlir_sha256)
    }
}

/// A directory of serialized executables.
///
/// Each entry is two files: `<digest>.pjrt` (the artifact) and `<digest>.meta`
/// (a plain-text sidecar recording what produced it). The sidecar is not
/// consulted when loading — the filename already is the key — it exists so a
/// human can tell what is in a cache directory without a tool.
///
/// ⚠️ Plain text, not JSON: gljax has no serde dependency, and a cache sidecar
/// is not worth acquiring one.
pub struct CompileCache {
    dir: PathBuf,
}

impl CompileCache {
    /// Opens (creating if needed) a cache directory.
    pub fn open(dir: impl AsRef<Path>) -> Result<Self, GlError> {
        let dir = dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&dir)?;
        Ok(CompileCache { dir })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn artifact_path(&self, key: &CompileKey) -> PathBuf {
        self.dir.join(format!("{}.pjrt", key.digest()))
    }

    fn meta_path(&self, key: &CompileKey) -> PathBuf {
        self.dir.join(format!("{}.meta", key.digest()))
    }

    /// Reads a cached artifact, or `None` on a miss.
    ///
    /// A miss is not an error — it is the normal first-run path.
    pub fn get(&self, key: &CompileKey) -> Result<Option<Vec<u8>>, GlError> {
        let path = self.artifact_path(key);
        match std::fs::read(&path) {
            Ok(bytes) => {
                log::debug!("compile cache hit: {}", path.display());
                Ok(Some(bytes))
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                log::debug!("compile cache miss: {}", path.display());
                Ok(None)
            }
            Err(e) => Err(GlError::Io(e)),
        }
    }

    /// Stores an artifact and its sidecar.
    ///
    /// Written to a temporary file and renamed, so a crash mid-write cannot
    /// leave a truncated artifact that later loads as garbage.
    pub fn put(&self, key: &CompileKey, artifact: &[u8]) -> Result<(), GlError> {
        let final_path = self.artifact_path(key);
        let tmp_path = final_path.with_extension("pjrt.tmp");
        std::fs::write(&tmp_path, artifact)?;
        std::fs::rename(&tmp_path, &final_path)?;

        let meta = format!(
            "platform      = {}\n\
             platform_ver  = {}\n\
             pjrt_api      = {}.{}\n\
             mlir_sha256   = {}\n\
             artifact_size = {}\n",
            key.platform_name,
            key.platform_version,
            key.plugin_version.0,
            key.plugin_version.1,
            key.mlir_hex(),
            artifact.len(),
        );
        std::fs::write(self.meta_path(key), meta)?;
        log::debug!("compile cache store: {}", final_path.display());
        Ok(())
    }

    /// Number of cached artifacts. Diagnostics only.
    pub fn len(&self) -> Result<usize, GlError> {
        let mut n = 0;
        for entry in std::fs::read_dir(&self.dir)? {
            let entry = entry?;
            if entry.path().extension().is_some_and(|e| e == "pjrt") {
                n += 1;
            }
        }
        Ok(n)
    }

    pub fn is_empty(&self) -> Result<bool, GlError> {
        Ok(self.len()? == 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_dir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("gljax_cache_test_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    fn key(mlir: &str) -> CompileKey {
        CompileKey::new((0, 114), "cpu", "cpu-test", mlir)
    }

    #[test]
    fn compile_cache_roundtrip() {
        let dir = tmp_dir("roundtrip");
        let cache = CompileCache::open(&dir).expect("open");
        let k = key("module @m { }");

        assert!(cache.get(&k).expect("get").is_none(), "must start empty");
        cache.put(&k, b"fake-artifact-bytes").expect("put");
        assert_eq!(
            cache.get(&k).expect("get").as_deref(),
            Some(&b"fake-artifact-bytes"[..])
        );
        assert_eq!(cache.len().expect("len"), 1);

        // The sidecar is human-readable and names the digest.
        let meta = std::fs::read_to_string(cache.meta_path(&k)).expect("meta");
        assert!(meta.contains(&k.mlir_hex()), "{meta}");
        assert!(meta.contains("pjrt_api      = 0.114"), "{meta}");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn different_mlir_gets_a_different_slot() {
        let dir = tmp_dir("distinct");
        let cache = CompileCache::open(&dir).expect("open");
        let a = key("module @a { }");
        let b = key("module @b { }");
        assert_ne!(a.digest(), b.digest());

        cache.put(&a, b"A").expect("put");
        assert!(cache.get(&b).expect("get").is_none(), "b must still miss");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// ⛔ A serialized executable is plugin-specific (ARTX01 §6.1). Reusing a
    /// CUDA artifact for a CPU client — or across plugin builds — is exactly
    /// the failure the key exists to prevent.
    #[test]
    fn the_key_separates_plugins_and_plugin_versions() {
        let mlir = "module @m { }";
        let base = CompileKey::new((0, 114), "cpu", "cpu-1.0", mlir);
        let other_platform = CompileKey::new((0, 114), "cuda", "cpu-1.0", mlir);
        let other_build = CompileKey::new((0, 114), "cpu", "cpu-2.0", mlir);
        let other_api = CompileKey::new((0, 113), "cpu", "cpu-1.0", mlir);

        let digests = vec![
            base.digest(),
            other_platform.digest(),
            other_build.digest(),
            other_api.digest(),
        ];
        let mut unique = digests.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), 4, "every field must affect the key: {digests:?}");
    }

    /// Length-prefixing the strings: without it, ("ab","c") and ("a","bc")
    /// would hash the same bytes.
    #[test]
    fn adjacent_string_fields_cannot_be_confused() {
        let mlir = "m";
        let a = CompileKey::new((0, 1), "ab", "c", mlir);
        let b = CompileKey::new((0, 1), "a", "bc", mlir);
        assert_ne!(a.digest(), b.digest());
    }

    #[test]
    fn a_partial_write_never_becomes_a_visible_artifact() {
        // The temp-then-rename path leaves no `.pjrt` behind on the tmp name.
        let dir = tmp_dir("atomic");
        let cache = CompileCache::open(&dir).expect("open");
        let k = key("module @m { }");
        cache.put(&k, b"bytes").expect("put");
        let stray: Vec<_> = std::fs::read_dir(&dir)
            .expect("read_dir")
            .filter_map(|e| e.ok())
            .filter(|e| e.path().to_string_lossy().ends_with(".tmp"))
            .collect();
        assert!(stray.is_empty(), "temporary file left behind");
        std::fs::remove_dir_all(&dir).ok();
    }
}
