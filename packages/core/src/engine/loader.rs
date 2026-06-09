use std::fs::File;
use std::path::Path;

use sysinfo::System;

// ── LoadMode ──────────────────────────────────────────────────────────────────

/// Describes how the OS should handle the mmap'd pages.
///
/// Auto-detected via [`LoadMode::detect`] based on available system RAM vs
/// file size. Can also be forced via [`MmapLoader::open_with_mode`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadMode {
    /// RAM > file size: hint OS to prefetch all pages immediately.
    ///
    /// Applies `MADV_SEQUENTIAL` + `MADV_WILLNEED`. The kernel starts reading
    /// the file into page cache before the pages are accessed, maximising NVMe
    /// throughput when the model fits in available RAM.
    Eager,
    /// RAM ≤ file size: let OS page in on demand via SSD.
    ///
    /// Applies `MADV_SEQUENTIAL` only. Safe on 8 GB RAM machines loading a
    /// 1.9 GB model when most RAM is already occupied by other processes.
    Lazy,
}

impl LoadMode {
    /// Auto-detect the appropriate load mode based on available RAM vs file size.
    ///
    /// Uses [`sysinfo::System`] to query available physical memory. Falls back
    /// to [`LoadMode::Lazy`] if detection fails — the safe, memory-conservative
    /// choice.
    ///
    /// # Arguments
    ///
    /// * `file_size_bytes` — size of the file about to be mmap'd, in bytes.
    pub fn detect(file_size_bytes: u64) -> Self {
        let mut sys = System::new();
        sys.refresh_memory();
        let available = sys.available_memory(); // bytes; 0 on failure
        if available > 0 && available > file_size_bytes {
            LoadMode::Eager
        } else {
            LoadMode::Lazy
        }
    }
}

// ── MmapLoader ────────────────────────────────────────────────────────────────

/// The four magic bytes that identify a GGUF file.
const GGUF_MAGIC: &[u8; 4] = b"GGUF";

/// Memory-mapped GGUF file loader.
///
/// Opens a GGUF file, maps it read-only into virtual address space, validates
/// the GGUF magic bytes, and applies madvise hints based on [`LoadMode`]:
///
/// - [`LoadMode::Eager`] — `MADV_SEQUENTIAL` + `MADV_WILLNEED`: kernel
///   prefetches all pages into RAM before access. Best when model fits in
///   available RAM.
/// - [`LoadMode::Lazy`] — `MADV_SEQUENTIAL` only: OS pages in on demand. Safe
///   on memory-constrained machines.
///
/// On Windows and other non-Unix targets all madvise calls are no-ops; the
/// mmap itself still provides zero-copy access.
pub struct MmapLoader {
    /// The memory-mapped byte slice (entire file).
    data: memmap2::Mmap,
    /// The load mode selected (auto-detected or explicit).
    pub mode: LoadMode,
}

impl MmapLoader {
    /// Open and mmap a GGUF file at `path`, auto-detecting [`LoadMode`].
    ///
    /// Equivalent to calling [`open_with_mode`][Self::open_with_mode] with
    /// `LoadMode::detect(file_size)`.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the file is not found, not readable, too small to
    /// contain a magic header, or the magic bytes do not match `b"GGUF"`.
    pub fn open(path: &Path) -> Result<Self, String> {
        // Stat the file first to get size for LoadMode detection, without
        // opening the full file descriptor yet.
        let file_size = std::fs::metadata(path)
            .map(|m| m.len())
            .unwrap_or(0);
        let mode = LoadMode::detect(file_size);
        Self::open_with_mode(path, mode)
    }

    /// Open and mmap a GGUF file at `path` with an explicit [`LoadMode`].
    ///
    /// Useful for benchmarking or when the caller has already determined which
    /// mode is appropriate.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the file is not found, not readable, too small to
    /// contain a magic header, or the magic bytes do not match `b"GGUF"`.
    pub fn open_with_mode(path: &Path, mode: LoadMode) -> Result<Self, String> {
        // 1. Open file
        let file = File::open(path)
            .map_err(|e| format!("cannot open '{}': {}", path.display(), e))?;

        // 2. Safety: we open the file read-only. The mmap is read-only and we
        //    never write through it. No other code in this crate aliases the
        //    same mapping mutably. The file handle outlives the MmapOptions
        //    call; the OS retains the mapping after the handle closes.
        let mmap = unsafe {
            memmap2::MmapOptions::new()
                .map(&file)
                .map_err(|e| format!("mmap failed for '{}': {}", path.display(), e))?
        };

        // 3. Verify magic: first 4 bytes must be b"GGUF"
        if mmap.len() < GGUF_MAGIC.len() {
            return Err(format!(
                "'{}' is too small to be a GGUF file ({} bytes)",
                path.display(),
                mmap.len()
            ));
        }
        if &mmap[..4] != GGUF_MAGIC.as_slice() {
            return Err(format!(
                "'{}' has invalid GGUF magic bytes (expected {:?}, got {:?})",
                path.display(),
                GGUF_MAGIC,
                &mmap[..4]
            ));
        }

        // 4. Apply madvise hints based on mode. No-op on non-Unix platforms.
        apply_madvise(&mmap, mode);

        // 5. Return
        Ok(Self { data: mmap, mode })
    }

    /// Return the full file contents as a byte slice.
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// File size in bytes.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` if the mapped file is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

// ── madvise helpers ───────────────────────────────────────────────────────────

/// Apply madvise hints appropriate for `mode`.
///
/// - `Lazy`  → `MADV_SEQUENTIAL` only: OS reads ahead in order, pages in on demand.
/// - `Eager` → `MADV_SEQUENTIAL` + `MADV_WILLNEED`: kernel prefetches entire
///   mapping into page cache immediately.
///
/// madvise is purely advisory; ignoring return values is intentional.
#[cfg(unix)]
fn apply_madvise(mmap: &memmap2::Mmap, mode: LoadMode) {
    // Always advise sequential access — benefits both modes.
    let _ = mmap.advise(memmap2::Advice::Sequential);

    // Eager: additionally request that the kernel fault in all pages now.
    if mode == LoadMode::Eager {
        let _ = mmap.advise(memmap2::Advice::WillNeed);
    }
}

/// No-op stub on non-Unix platforms (e.g. Windows).
#[cfg(not(unix))]
fn apply_madvise(_mmap: &memmap2::Mmap, _mode: LoadMode) {}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    // ── Existing tests ────────────────────────────────────────────────────────

    #[test]
    fn test_mmap_loader_invalid_magic() {
        let mut f = NamedTempFile::new().expect("tempfile");
        f.write_all(b"NOTGGUF_GARBAGE_PADDING").expect("write");
        f.flush().expect("flush");

        let result = MmapLoader::open(f.path());
        assert!(result.is_err(), "expected Err for invalid magic");
        let msg = result.err().unwrap();
        assert!(
            msg.to_lowercase().contains("magic"),
            "error message should mention 'magic', got: {msg}"
        );
    }

    #[test]
    fn test_mmap_loader_valid_magic() {
        let mut f = NamedTempFile::new().expect("tempfile");
        let mut contents = b"GGUF".to_vec();
        contents.extend_from_slice(&[0u8; 64]);
        f.write_all(&contents).expect("write");
        f.flush().expect("flush");

        let result = MmapLoader::open(f.path());
        assert!(result.is_ok(), "expected Ok for valid magic, got: {:?}", result.err());
        let loader = result.unwrap();
        assert_eq!(&loader.as_bytes()[..4], b"GGUF");
    }

    #[test]
    fn test_mmap_loader_empty_file() {
        let f = NamedTempFile::new().expect("tempfile");
        let result = MmapLoader::open(f.path());
        assert!(result.is_err(), "expected Err for empty file");
    }

    // ── New LoadMode tests ────────────────────────────────────────────────────

    #[test]
    fn test_load_mode_detect_eager() {
        // A 100-byte file will always fit in available RAM on any development
        // machine. detect() should return Eager.
        let mode = LoadMode::detect(100);
        assert_eq!(mode, LoadMode::Eager, "100-byte file should select Eager mode");
    }

    #[test]
    fn test_load_mode_detect_lazy() {
        // u64::MAX bytes is larger than any physical RAM; detect() must return Lazy.
        let mode = LoadMode::detect(u64::MAX);
        assert_eq!(mode, LoadMode::Lazy, "u64::MAX file size should select Lazy mode");
    }

    #[test]
    fn test_mmap_loader_exposes_mode() {
        // Open a valid GGUF temp file and confirm .mode field is accessible.
        let mut f = NamedTempFile::new().expect("tempfile");
        let mut contents = b"GGUF".to_vec();
        contents.extend_from_slice(&[0u8; 64]);
        f.write_all(&contents).expect("write");
        f.flush().expect("flush");

        let loader = MmapLoader::open(f.path()).expect("open should succeed");
        // The mode must be one of the two valid variants — not a third state.
        let mode = loader.mode;
        assert!(
            mode == LoadMode::Eager || mode == LoadMode::Lazy,
            "mode must be Eager or Lazy, got: {:?}",
            mode
        );
    }

    #[test]
    fn test_mmap_loader_open_with_mode_lazy() {
        let mut f = NamedTempFile::new().expect("tempfile");
        let mut contents = b"GGUF".to_vec();
        contents.extend_from_slice(&[0u8; 64]);
        f.write_all(&contents).expect("write");
        f.flush().expect("flush");

        let result = MmapLoader::open_with_mode(f.path(), LoadMode::Lazy);
        assert!(result.is_ok(), "open_with_mode Lazy should succeed");
        assert_eq!(result.unwrap().mode, LoadMode::Lazy);
    }

    #[test]
    fn test_mmap_loader_open_with_mode_eager() {
        let mut f = NamedTempFile::new().expect("tempfile");
        let mut contents = b"GGUF".to_vec();
        contents.extend_from_slice(&[0u8; 64]);
        f.write_all(&contents).expect("write");
        f.flush().expect("flush");

        let result = MmapLoader::open_with_mode(f.path(), LoadMode::Eager);
        assert!(result.is_ok(), "open_with_mode Eager should succeed");
        assert_eq!(result.unwrap().mode, LoadMode::Eager);
    }
}
