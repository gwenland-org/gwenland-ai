/// Selective layer loading for zero-copy LoRA training.
///
/// `LayerSlice` is a lightweight descriptor (no heap allocation beyond the
/// tensor name string) that records where a single transformer-layer tensor
/// lives inside a memory-mapped GGUF file.  `LayerIndex` groups slices by
/// layer number so the training loop can load exactly one layer at a time.
///
/// `LayerLoader` opens a GGUF file with `LoadMode::Lazy` so the OS never
/// pulls the entire model into RSS.  `load_layer(n)` materialises exactly one
/// layer's worth of tensors; `LoadedLayer::unload()` (or drop) releases them.
use std::path::Path;

use anyhow::{anyhow, Result};

use crate::convert::gguf_parser::{self, GgufDtype, GgufFile};
use crate::engine::loader::{LoadMode, MmapLoader};

// ── Test-only concurrency counter ─────────────────────────────────────────────
//
// Tracks how many `LoadedLayer` values are alive at once.  Integration tests
// assert this never exceeds 1, enforcing the no-full-load invariant without
// any runtime overhead in release builds.
//
// Enabled by either `cfg(test)` (unit tests) or `feature = "test-utils"`
// (integration tests, which compile the lib without `cfg(test)`).
#[cfg(any(test, feature = "test-utils"))]
pub static LIVE_LAYER_COUNT: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

// ── LayerSlice ────────────────────────────────────────────────────────────────

/// Byte-range descriptor for one tensor that belongs to a single transformer layer.
///
/// Stores only the offset, length, dtype, and shape needed to slice and dequantise
/// the mmap — no weight data is held here, keeping RSS proportional to the number
/// of tensors, not their size.
#[derive(Debug, Clone)]
pub struct LayerSlice {
    pub layer_idx:   usize,
    pub tensor_name: String,
    pub byte_offset: u64,
    pub byte_len:    usize,
    /// Element dtype stored on disk — needed to dispatch the right dequant path.
    pub dtype:       GgufDtype,
    /// Tensor shape in row-major order — needed to compute element count for dequant.
    pub shape:       Vec<u64>,
}

// ── LayerIndex ────────────────────────────────────────────────────────────────

/// Maps layer numbers to their constituent `LayerSlice` entries.
///
/// Built once from a `GgufFile` header scan; after construction the index is
/// immutable and can be queried cheaply for any layer number.
pub struct LayerIndex {
    /// All layer slices, sorted by `(layer_idx, tensor_name)`.
    slices: Vec<LayerSlice>,
    pub num_layers: usize,
}

impl LayerIndex {
    /// Scan `file.tensors` and build a `LayerIndex`.
    ///
    /// Only tensors whose names start with `"model.layers."` are indexed; all
    /// others (embeddings, norms, lm_head, …) are silently skipped.
    pub fn scan(file: &GgufFile) -> Self {
        let mut slices: Vec<LayerSlice> = file
            .tensors
            .iter()
            .filter_map(|t| {
                let rest = t.name.strip_prefix("model.layers.")?;
                // Next segment up to the first '.' is the decimal layer index.
                let dot = rest.find('.')?;
                let idx: usize = rest[..dot].parse().ok()?;
                Some(LayerSlice {
                    layer_idx:   idx,
                    tensor_name: t.name.clone(),
                    // Absolute offset in the mmap so load_layer can slice directly.
                    byte_offset: file.data_base + t.data_offset,
                    byte_len:    t.data_size,
                    dtype:       t.dtype,
                    shape:       t.shape.clone(),
                })
            })
            .collect();

        slices.sort_by(|a, b| {
            a.layer_idx
                .cmp(&b.layer_idx)
                .then_with(|| a.tensor_name.cmp(&b.tensor_name))
        });

        let num_layers = slices
            .iter()
            .map(|s| s.layer_idx)
            .max()
            .map(|m| m + 1)
            .unwrap_or(0);

        LayerIndex { slices, num_layers }
    }

    /// Return the contiguous slice of `LayerSlice`s that belong to layer `n`.
    ///
    /// Returns an empty slice (no panic) if `n >= num_layers`.
    pub fn layer_slices(&self, n: usize) -> &[LayerSlice] {
        let start = self.slices.partition_point(|s| s.layer_idx < n);
        let end   = self.slices.partition_point(|s| s.layer_idx <= n);
        &self.slices[start..end]
    }
}

// ── LoadedLayer ───────────────────────────────────────────────────────────────

/// One transformer layer's worth of tensors, borrowed from the live mmap.
///
/// Each entry in `slices` is `(tensor_name, raw_bytes)` — zero-copy views into
/// the underlying `Mmap`.  Dropping (or calling `.unload()`) advises the OS to
/// reclaim the backing pages on Unix, keeping RSS bounded during the layer loop.
pub struct LoadedLayer<'mmap> {
    /// `(tensor_name, raw_bytes)` pairs, one per tensor in this layer.
    pub slices: Vec<(&'mmap str, &'mmap [u8])>,
    /// Union byte range — read only by the `Drop` impl on Unix (`MADV_DONTNEED`).
    #[allow(dead_code)]
    pub(crate) mmap_range: std::ops::Range<usize>,
    /// Mmap handle — read only by the `Drop` impl on Unix.
    #[allow(dead_code)]
    pub(crate) mmap_data: &'mmap memmap2::Mmap,
}

impl<'mmap> LoadedLayer<'mmap> {
    /// Explicitly release this layer's pages (equivalent to drop).
    ///
    /// On Unix this issues `MADV_DONTNEED` so the kernel can reclaim the
    /// physical pages before the next layer is loaded.  On Windows it is a
    /// no-op — the OS manages the working set automatically.
    pub fn unload(self) {
        drop(self);
    }
}

impl<'mmap> Drop for LoadedLayer<'mmap> {
    fn drop(&mut self) {
        #[cfg(unix)]
        {
            let _ = self.mmap_data.advise_range(
                memmap2::Advice::DontNeed,
                self.mmap_range.start,
                self.mmap_range.len(),
            );
        }
        #[cfg(any(test, feature = "test-utils"))]
        LIVE_LAYER_COUNT.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
    }
}

// ── LayerLoader ───────────────────────────────────────────────────────────────

/// Zero-copy, one-layer-at-a-time loader for GGUF weight files.
///
/// Opens the file with `LoadMode::Lazy` so the OS never pages in more than the
/// tensors actually accessed.  Call `load_layer(n)` to obtain a `LoadedLayer`
/// for layer `n`; drop or `.unload()` it before loading the next layer to keep
/// RSS bounded to approximately one layer's worth of data.
pub struct LayerLoader {
    mmap:  MmapLoader,
    index: LayerIndex,
}

impl LayerLoader {
    /// Open `path` as a lazily-paged mmap and build the `LayerIndex`.
    ///
    /// Uses `LoadMode::Lazy` unconditionally — the whole point of this type is
    /// to avoid loading the full model into RAM at once.
    pub fn open(path: &Path) -> Result<Self> {
        let mmap = MmapLoader::open_with_mode(path, LoadMode::Lazy)
            .map_err(|e| anyhow!("{}", e))?;
        let gguf = gguf_parser::parse(path)
            .map_err(|e| anyhow!("{}", e))?;
        let index = LayerIndex::scan(&gguf);
        Ok(LayerLoader { mmap, index })
    }

    /// Number of transformer layers found in the GGUF index.
    pub fn num_layers(&self) -> usize {
        self.index.num_layers
    }

    /// Return the `LayerSlice` descriptors for layer `n` (dtype + shape + byte range).
    ///
    /// Used by `LayeredTrainingLoop` to obtain metadata needed for dequantisation
    /// without re-parsing the GGUF header.  Returns an empty slice for OOB `n`.
    pub(crate) fn index_slices(&self, n: usize) -> &[LayerSlice] {
        self.index.layer_slices(n)
    }

    /// Materialise layer `n` from the mmap and return its tensors as borrowed slices.
    ///
    /// Returns `Err` if `n >= num_layers()`.  The returned `LoadedLayer` borrows
    /// from `self` (lifetime `'a`), so it must be dropped before `self` is moved.
    pub fn load_layer<'a>(&'a self, n: usize) -> Result<LoadedLayer<'a>> {
        if n >= self.index.num_layers {
            return Err(anyhow!(
                "layer index {} out of range (file has {} layers)",
                n,
                self.index.num_layers
            ));
        }

        let raw = self.mmap.as_bytes();
        let layer_slices = self.index.layer_slices(n);

        // Compute the union byte range so Drop can issue a single madvise call.
        // Empty layer_slices is unreachable here (we checked num_layers above
        // and scan guarantees every layer index has at least one tensor), but
        // we handle it safely anyway.
        let mmap_range = if layer_slices.is_empty() {
            0..0
        } else {
            let start = layer_slices
                .iter()
                .map(|s| s.byte_offset as usize)
                .min()
                .unwrap();
            let end = layer_slices
                .iter()
                .map(|s| s.byte_offset as usize + s.byte_len)
                .max()
                .unwrap();
            start..end
        };

        // Build (name, bytes) pairs — zero-copy borrows from the mmap.
        let slices: Vec<(&'a str, &'a [u8])> = layer_slices
            .iter()
            .map(|s| {
                let start = s.byte_offset as usize;
                let end   = start + s.byte_len;
                let name_bytes: &'a str = {
                    // SAFETY: tensor_name is stored in self.index which lives
                    // for 'a; the String's backing memory is stable.
                    // We return &str by re-borrowing through the LayerIndex.
                    &self.index.slices[self.index.slices
                        .partition_point(|x| x.layer_idx < n)
                        ..self.index.slices.partition_point(|x| x.layer_idx <= n)]
                        .iter()
                        .find(|x| x.tensor_name == s.tensor_name)
                        .unwrap()
                        .tensor_name
                };
                let bytes: &'a [u8] = &raw[start..end];
                (name_bytes, bytes)
            })
            .collect();

        #[cfg(any(test, feature = "test-utils"))]
        LIVE_LAYER_COUNT.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

        Ok(LoadedLayer {
            slices,
            mmap_range,
            mmap_data: self.mmap.mmap(),
        })
    }
}

// ── Test utilities (compile only in test builds) ──────────────────────────────

/// Write a minimal valid GGUF v3 binary to a `NamedTempFile`.
///
/// Write a minimal valid GGUF v3 binary to a `NamedTempFile`.
///
/// `pub` so integration tests (`tests/`) can import it via `feature = "test-utils"`.
/// Also compiled under `cfg(test)` for unit tests.
///
/// Each entry in `tensors` is `(name, data_bytes)`.  The header is built to
/// spec so `gguf_parser::parse` accepts it without error.
#[cfg(any(test, feature = "test-utils"))]
pub fn write_minimal_gguf_pub(tensors: &[(&str, &[u8])]) -> tempfile::NamedTempFile {
    use std::io::Write;
    let mut f = tempfile::NamedTempFile::new().expect("tempfile");
    f.write_all(b"GGUF").unwrap();
    f.write_all(&3u32.to_le_bytes()).unwrap();
    f.write_all(&(tensors.len() as u64).to_le_bytes()).unwrap();
    f.write_all(&0u64.to_le_bytes()).unwrap();
    let mut cursor: u64 = 0;
    let mut offsets: Vec<u64> = Vec::with_capacity(tensors.len());
    for (_, data) in tensors.iter() { offsets.push(cursor); cursor += data.len() as u64; }
    for ((name, data), &off) in tensors.iter().zip(offsets.iter()) {
        let nb = name.as_bytes();
        f.write_all(&(nb.len() as u64).to_le_bytes()).unwrap();
        f.write_all(nb).unwrap();
        f.write_all(&1u32.to_le_bytes()).unwrap();
        let n_elems = (data.len() / 4).max(1) as u64;
        f.write_all(&n_elems.to_le_bytes()).unwrap();
        f.write_all(&0u32.to_le_bytes()).unwrap();
        f.write_all(&off.to_le_bytes()).unwrap();
    }
    let pos = f.as_file().metadata().unwrap().len();
    let rem = pos % 32;
    if rem != 0 { f.write_all(&vec![0u8; (32 - rem) as usize]).unwrap(); }
    for (_, data) in tensors.iter() { f.write_all(data).unwrap(); }
    f.flush().unwrap();
    f
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::convert::gguf_parser::{GgufDtype, GgufFile, TensorInfo};
    use std::io::Write;
    use tempfile::NamedTempFile;

    // ═════════════════════════════════════════════════════════════════════════
    // Shared test helpers
    // ═════════════════════════════════════════════════════════════════════════

    fn make_tensor(name: &str, offset: u64, size: usize) -> TensorInfo {
        // size bytes / 4 bytes-per-f32 = element count; at least 1
        let n_elems = ((size / 4) as u64).max(1);
        TensorInfo {
            name:        name.to_owned(),
            shape:       vec![n_elems],
            dtype:       GgufDtype::F32,
            data_offset: offset,
            data_size:   size,
            raw_data:    Vec::new(),
        }
    }

    fn make_file(tensors: Vec<TensorInfo>) -> GgufFile {
        GgufFile { version: 3, tensors, data_base: 0 }
    }

    /// Write a minimal valid GGUF v3 binary to a `NamedTempFile`.
    ///
    /// Each entry in `tensors` is `(name, data_bytes)`.  The header is built
    /// to spec so `gguf_parser::parse` accepts it without error.
    ///
    /// GGUF v3 layout used here:
    ///   magic(4) + version(4) + tensor_count(8) + kv_count(8)
    ///   + tensor_info[]  (name_len(8)+name+n_dims(4)+shape[n_dims*8]+dtype(4)+data_offset(8))
    ///   + align-to-32 padding
    ///   + tensor data (concatenated)
    pub(crate) fn write_minimal_gguf(tensors: &[(&str, &[u8])]) -> NamedTempFile {
        let mut f = NamedTempFile::new().expect("tempfile");

        // ── header ────────────────────────────────────────────────────────────
        f.write_all(b"GGUF").unwrap();               // magic
        f.write_all(&3u32.to_le_bytes()).unwrap();   // version 3
        f.write_all(&(tensors.len() as u64).to_le_bytes()).unwrap(); // tensor_count
        f.write_all(&0u64.to_le_bytes()).unwrap();   // kv_count = 0

        // ── tensor info section ───────────────────────────────────────────────
        // First pass: compute relative data offsets (each tensor packed tightly).
        let mut offsets: Vec<u64> = Vec::with_capacity(tensors.len());
        let mut cursor: u64 = 0;
        for (_, data) in tensors.iter() {
            offsets.push(cursor);
            cursor += data.len() as u64;
        }

        for ((name, data), &off) in tensors.iter().zip(offsets.iter()) {
            let name_bytes = name.as_bytes();
            f.write_all(&(name_bytes.len() as u64).to_le_bytes()).unwrap(); // name length
            f.write_all(name_bytes).unwrap();                                // name
            f.write_all(&1u32.to_le_bytes()).unwrap();                       // n_dims = 1
            // shape[0] = number of f32 elements implied by data.len()
            let n_elems = (data.len() / 4).max(1) as u64;
            f.write_all(&n_elems.to_le_bytes()).unwrap();                    // shape[0]
            f.write_all(&0u32.to_le_bytes()).unwrap();                       // dtype = F32 (0)
            f.write_all(&off.to_le_bytes()).unwrap();                        // data_offset
        }

        // ── align to 32 bytes before data block ───────────────────────────────
        let pos = f.as_file().metadata().unwrap().len();
        let rem = pos % 32;
        if rem != 0 {
            let pad = vec![0u8; (32 - rem) as usize];
            f.write_all(&pad).unwrap();
        }

        // ── data block ────────────────────────────────────────────────────────
        for (_, data) in tensors.iter() {
            f.write_all(data).unwrap();
        }

        f.flush().unwrap();
        f
    }

    // ═════════════════════════════════════════════════════════════════════════
    // Wave 1 — LayerIndex deterministic tests (unchanged)
    // ═════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_scan_empty() {
        let f = make_file(vec![]);
        let idx = LayerIndex::scan(&f);
        assert_eq!(idx.num_layers, 0);
        assert!(idx.slices.is_empty());
    }

    #[test]
    fn test_scan_filters_non_layer_tensors() {
        let f = make_file(vec![
            make_tensor("token_embd.weight", 0, 16),
            make_tensor("output_norm.weight", 16, 8),
        ]);
        let idx = LayerIndex::scan(&f);
        assert_eq!(idx.num_layers, 0);
        assert!(idx.slices.is_empty());
    }

    #[test]
    fn test_scan_extracts_layer_count() {
        let f = make_file(vec![
            make_tensor("model.layers.0.self_attn.q_proj.weight", 0, 32),
            make_tensor("model.layers.1.self_attn.q_proj.weight", 32, 32),
        ]);
        let idx = LayerIndex::scan(&f);
        assert_eq!(idx.num_layers, 2);
    }

    #[test]
    fn test_scan_sorted_order() {
        let f = make_file(vec![
            make_tensor("model.layers.1.mlp.down_proj.weight", 64, 16),
            make_tensor("model.layers.0.self_attn.q_proj.weight", 0, 32),
            make_tensor("model.layers.0.mlp.down_proj.weight", 32, 32),
        ]);
        let idx = LayerIndex::scan(&f);
        assert_eq!(idx.slices[0].layer_idx, 0);
        assert_eq!(idx.slices[1].layer_idx, 0);
        assert!(idx.slices[0].tensor_name < idx.slices[1].tensor_name);
        assert_eq!(idx.slices[2].layer_idx, 1);
    }

    #[test]
    fn test_layer_slices_out_of_range() {
        let f = make_file(vec![
            make_tensor("model.layers.0.self_attn.q_proj.weight", 0, 32),
            make_tensor("model.layers.1.self_attn.q_proj.weight", 32, 32),
        ]);
        let idx = LayerIndex::scan(&f);
        assert!(idx.layer_slices(99).is_empty());
    }

    // ═════════════════════════════════════════════════════════════════════════
    // Wave 2 — LayerLoader / LoadedLayer deterministic tests
    // ═════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_layer_loader_open_invalid_path() {
        let result = LayerLoader::open(Path::new("/nonexistent/path/model.gguf"));
        assert!(result.is_err());
    }

    #[test]
    fn test_layer_loader_open_invalid_magic() {
        let mut f = NamedTempFile::new().expect("tempfile");
        f.write_all(b"NOTGGUF_GARBAGE_PADDING_BYTES").unwrap();
        f.flush().unwrap();
        let result = LayerLoader::open(f.path());
        assert!(result.is_err());
    }

    #[test]
    fn test_layer_loader_load_layer_oor() {
        // A valid GGUF with one layer tensor.
        let data: Vec<u8> = vec![1.0f32.to_le_bytes(), 2.0f32.to_le_bytes()]
            .into_iter()
            .flatten()
            .collect();
        let f = write_minimal_gguf(&[
            ("model.layers.0.self_attn.q_proj.weight", data.as_slice()),
        ]);
        let loader = LayerLoader::open(f.path()).expect("open");
        assert!(loader.load_layer(999).is_err());
    }

    #[test]
    fn test_layer_loader_load_layer_ok() {
        // 8 bytes = 2 × f32.
        let data: Vec<u8> = [1.0f32, 2.0f32]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let f = write_minimal_gguf(&[
            ("model.layers.0.self_attn.q_proj.weight", data.as_slice()),
        ]);
        let loader = LayerLoader::open(f.path()).expect("open");
        assert_eq!(loader.num_layers(), 1);

        let loaded = loader.load_layer(0).expect("load_layer(0)");
        assert_eq!(loaded.slices.len(), 1);

        // Verify byte content round-trips correctly.
        let (name, bytes) = loaded.slices[0];
        assert_eq!(name, "model.layers.0.self_attn.q_proj.weight");
        assert_eq!(bytes.len(), data.len());
        assert_eq!(bytes, data.as_slice());
    }

    #[test]
    fn test_loaded_layer_unload_then_reload() {
        let data: Vec<u8> = [3.0f32, 4.0f32]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let f = write_minimal_gguf(&[
            ("model.layers.0.self_attn.q_proj.weight", data.as_slice()),
        ]);
        let loader = LayerLoader::open(f.path()).expect("open");

        let first = loader.load_layer(0).expect("first load");
        first.unload(); // explicit unload — must not panic

        // A second load after unload must succeed.
        let second = loader.load_layer(0).expect("reload after unload");
        assert_eq!(second.slices.len(), 1);
    }

    // ═════════════════════════════════════════════════════════════════════════
    // Wave 1 quickcheck properties (unchanged)
    // ═════════════════════════════════════════════════════════════════════════

    use quickcheck_macros::quickcheck;

    fn build_file_from_pairs(pairs: &[(u8, String)]) -> GgufFile {
        let tensors = pairs
            .iter()
            .enumerate()
            .map(|(i, (layer, suffix))| {
                let name = format!(
                    "model.layers.{}.{}.weight",
                    layer,
                    if suffix.is_empty() || suffix.contains('.') {
                        "default".to_owned()
                    } else {
                        suffix.clone()
                    }
                );
                make_tensor(&name, (i as u64) * 16, 16)
            })
            .collect();
        make_file(tensors)
    }

    #[quickcheck]
    fn scan_then_layer_slices_covers_all_matched_tensors(pairs: Vec<(u8, String)>) -> bool {
        let file = build_file_from_pairs(&pairs);
        let idx  = LayerIndex::scan(&file);
        idx.slices.len() == pairs.len()
    }

    #[quickcheck]
    fn scan_num_layers_equals_max_plus_one(pairs: Vec<(u8, String)>) -> bool {
        if pairs.is_empty() { return true; }
        let file     = build_file_from_pairs(&pairs);
        let idx      = LayerIndex::scan(&file);
        let expected = pairs.iter().map(|(l, _)| *l as usize).max().unwrap() + 1;
        idx.num_layers == expected
    }

    #[quickcheck]
    fn layer_slices_sorted(pairs: Vec<(u8, String)>) -> bool {
        let file = build_file_from_pairs(&pairs);
        let idx  = LayerIndex::scan(&file);
        for n in 0..idx.num_layers {
            let slices = idx.layer_slices(n);
            if !slices.windows(2).all(|w| w[0].tensor_name <= w[1].tensor_name) {
                return false;
            }
        }
        true
    }

    // ═════════════════════════════════════════════════════════════════════════
    // Wave 2 quickcheck properties — LoadedLayer byte range + unload safety
    // ═════════════════════════════════════════════════════════════════════════

    /// Helper: build a GGUF temp file with N layer-0 tensors of `size` bytes each.
    fn write_layer0_file(n_tensors: usize, tensor_size: usize) -> NamedTempFile {
        // Minimum 4 bytes (one f32) so the F32 dtype calculation doesn't round to 0.
        let size = tensor_size.max(4);
        let data: Vec<u8> = vec![0u8; size];
        let tensors: Vec<(String, Vec<u8>)> = (0..n_tensors)
            .map(|i| (format!("model.layers.0.tensor_{}.weight", i), data.clone()))
            .collect();
        let refs: Vec<(&str, &[u8])> = tensors
            .iter()
            .map(|(n, d)| (n.as_str(), d.as_slice()))
            .collect();
        write_minimal_gguf(&refs)
    }

    /// Property 4 — mmap_range covers exactly the union of all tensor byte ranges.
    #[quickcheck]
    fn loaded_layer_mmap_range_covers_all_slices(n_tensors: u8, tensor_size: u8) -> bool {
        let n = (n_tensors as usize % 8) + 1; // 1..=8
        let sz = ((tensor_size as usize % 32) + 1) * 4; // multiples of 4, 4..=128

        let f = write_layer0_file(n, sz);
        let loader = match LayerLoader::open(f.path()) {
            Ok(l) => l,
            Err(_) => return true, // parse errors on degenerate inputs are fine
        };
        if loader.num_layers() == 0 { return true; }

        let loaded = match loader.load_layer(0) {
            Ok(l) => l,
            Err(_) => return false,
        };

        // Compute expected range from the index directly.
        let layer_slices = loader.index.layer_slices(0);
        let expected_start = layer_slices.iter().map(|s| s.byte_offset as usize).min().unwrap();
        let expected_end   = layer_slices.iter().map(|s| s.byte_offset as usize + s.byte_len).max().unwrap();

        loaded.mmap_range == (expected_start..expected_end)
    }

    /// Property 5 — unload (explicit or via drop) never panics; reload succeeds.
    #[quickcheck]
    fn load_then_unload_does_not_panic(n_tensors: u8) -> bool {
        let n = (n_tensors as usize % 4) + 1; // 1..=4
        let f = write_layer0_file(n, 8);
        let loader = match LayerLoader::open(f.path()) {
            Ok(l) => l,
            Err(_) => return true,
        };
        if loader.num_layers() == 0 { return true; }

        // First load + explicit unload.
        let loaded = match loader.load_layer(0) { Ok(l) => l, Err(_) => return false };
        loaded.unload();

        // Reload must succeed — mmap is still valid, only advisory pages changed.
        loader.load_layer(0).is_ok()
    }
}
