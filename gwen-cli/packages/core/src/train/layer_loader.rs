/// Selective layer loading for zero-copy LoRA training.
///
/// `LayerSlice` is a lightweight descriptor (no heap allocation beyond the
/// tensor name string) that records where a single transformer-layer tensor
/// lives inside a memory-mapped GGUF file.  `LayerIndex` groups slices by
/// layer number so the training loop can load exactly one layer at a time.
///
/// `LayerLoader` parses only the GGUF header, then opens the file with
/// `LoadMode::Lazy` so tensor payloads remain mmap-backed. `load_layer(n)`
/// materialises exactly one layer's raw slices; `LoadedLayer::unload()` (or
/// drop) releases them.
use std::path::Path;

use anyhow::{Result, anyhow};

use crate::convert::gguf_parser::{
    self, GgufDtype, GgufFile, GgufHeader, MetadataValue, TensorInfo,
};
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
    pub layer_idx: usize,
    pub tensor_name: String,
    pub byte_offset: u64,
    pub byte_len: usize,
    /// Element dtype stored on disk — needed to dispatch the right dequant path.
    pub dtype: GgufDtype,
    /// Tensor shape in row-major order — needed to compute element count for dequant.
    pub shape: Vec<u64>,
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
    /// Handles both naming conventions used in real GGUF files:
    ///   - `model.layers.{N}.*`  — HuggingFace-style exports
    ///   - `blk.{N}.*`           — llama.cpp-style (Qwen, Llama, Mistral, etc.)
    /// All other tensors (embeddings, norms, lm_head, …) are silently skipped.
    pub fn scan(file: &GgufFile) -> Self {
        Self::scan_tensors(&file.tensors, file.data_base)
    }

    fn scan_tensors(tensors: &[TensorInfo], data_base: u64) -> Self {
        let mut slices: Vec<LayerSlice> = tensors
            .iter()
            .filter_map(|t| {
                // Try both prefixes; extract the decimal layer index from the first segment.
                let rest = t
                    .name
                    .strip_prefix("model.layers.")
                    .or_else(|| t.name.strip_prefix("blk."))?;
                let dot = rest.find('.')?;
                let idx: usize = rest[..dot].parse().ok()?;
                Some(LayerSlice {
                    layer_idx: idx,
                    tensor_name: t.name.clone(),
                    // Absolute offset in the mmap so load_layer can slice directly.
                    byte_offset: data_base + t.data_offset,
                    byte_len: t.data_size,
                    dtype: t.dtype,
                    shape: t.shape.clone(),
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
        let end = self.slices.partition_point(|s| s.layer_idx <= n);
        &self.slices[start..end]
    }
}

/// Architecture values required by the training transformer forward.
#[derive(Debug, Clone)]
pub struct TransformerConfig {
    pub architecture: String,
    pub n_layers: usize,
    pub hidden_size: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    /// Whether the output projection shares the token-embedding weights.
    ///
    /// Explicit Boolean metadata takes precedence. When the key is absent,
    /// standard GGUF structure is used: no standalone output head means the
    /// embedding is the output projection.
    pub tie_word_embeddings: bool,
}

/// Descriptor for any GGUF tensor, including non-layer tensors.
#[derive(Debug, Clone)]
pub struct TensorSlice {
    pub tensor_name: String,
    pub byte_offset: u64,
    pub byte_len: usize,
    pub dtype: GgufDtype,
    /// GGUF dimension order. Reverse this when constructing a row-major Candle
    /// tensor.
    pub shape: Vec<u64>,
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

/// Find the token-embedding tensor's shape, model-agnostically.
///
/// Different exporters name the embedding differently:
///   - llama.cpp:     `token_embd.weight`
///   - HuggingFace:   `model.embed_tokens.weight`
///   - misc/generic:  any tensor whose name contains `embed`
///
/// We try the well-known names first, then fall back to a substring match so
/// new architectures work without code changes. Returns the GGUF dimension
/// order (`[hidden, vocab]`) or `None` if no embedding tensor is present.
fn find_embedding_shape(tensors: &[TensorInfo]) -> Option<Vec<u64>> {
    const KNOWN: [&str; 2] = ["token_embd.weight", "model.embed_tokens.weight"];
    for name in KNOWN {
        if let Some(t) = tensors.iter().find(|t| t.name == name) {
            return Some(t.shape.clone());
        }
    }
    tensors
        .iter()
        .find(|t| t.name.to_lowercase().contains("embed"))
        .map(|t| t.shape.clone())
}

fn projection_dims(tensors: &[TensorInfo], needles: &[&str]) -> Option<(usize, usize)> {
    let tensor = tensors
        .iter()
        .find(|tensor| needles.iter().any(|needle| tensor.name.contains(needle)))?;
    match tensor.shape.as_slice() {
        // GGUF stores the input dimension first and output dimension second.
        [d_in, d_out] => Some((*d_out as usize, *d_in as usize)),
        _ => None,
    }
}

fn has_separate_output_head(tensors: &[TensorInfo]) -> bool {
    const OUTPUT_HEAD_NAMES: [&str; 3] =
        ["output.weight", "lm_head.weight", "model.lm_head.weight"];

    tensors
        .iter()
        .any(|tensor| OUTPUT_HEAD_NAMES.contains(&tensor.name.as_str()))
}

fn resolves_to_tied_embeddings(header: &GgufHeader, architecture: &str) -> bool {
    let key = format!("{architecture}.tie_word_embeddings");
    match header.metadata.get(&key) {
        Some(MetadataValue::Bool(value)) => *value,
        Some(_) => false,
        None => !has_separate_output_head(&header.tensors),
    }
}

fn build_transformer_config(
    header: &GgufHeader,
    index: &LayerIndex,
    embed_shape: Option<&[u64]>,
) -> Result<TransformerConfig> {
    let architecture = header
        .metadata
        .get("general.architecture")
        .and_then(|value| value.as_str())
        .unwrap_or("unknown")
        .to_string();
    let prefixed = |suffix: &str| format!("{}.{}", architecture, suffix);
    let get_usize = |suffix: &str| {
        header
            .metadata
            .get(&prefixed(suffix))
            .and_then(|value| value.as_usize())
    };
    let get_f32 = |suffix: &str| {
        header
            .metadata
            .get(&prefixed(suffix))
            .and_then(|value| value.as_f32())
    };
    let inferred_hidden = embed_shape
        .and_then(|shape| match shape {
            [a, b] => Some((*a as usize).min(*b as usize)),
            [only] => Some(*only as usize),
            _ => None,
        })
        .or_else(|| {
            projection_dims(&header.tensors, &["attn_q.weight", "q_proj.weight"])
                .map(|(_, d_in)| d_in)
        });
    let hidden_size = get_usize("embedding_length")
        .or(inferred_hidden)
        .unwrap_or(1);
    let n_layers = get_usize("block_count").unwrap_or(index.num_layers);
    let n_heads = get_usize("attention.head_count").unwrap_or(1);
    let n_kv_heads = get_usize("attention.head_count_kv").unwrap_or(n_heads);
    let intermediate_size = get_usize("feed_forward_length")
        .or_else(|| {
            projection_dims(&header.tensors, &["ffn_gate.weight", "gate_proj.weight"])
                .map(|(d_out, _)| d_out)
        })
        .unwrap_or(hidden_size.saturating_mul(4));
    let vocab_size = get_usize("vocab_size")
        .or_else(|| {
            embed_shape.and_then(|shape| match shape {
                [a, b] => Some((*a as usize).max(*b as usize)),
                _ => None,
            })
        })
        .unwrap_or(hidden_size.max(2));
    let rms_norm_eps = get_f32("attention.layer_norm_rms_epsilon").unwrap_or(1e-5);
    let rope_theta = get_f32("rope.freq_base").unwrap_or(10_000.0);
    let tie_word_embeddings = resolves_to_tied_embeddings(header, &architecture);

    if n_layers != index.num_layers {
        return Err(anyhow!(
            "GGUF metadata reports {n_layers} layers but tensor index contains {}",
            index.num_layers
        ));
    }
    if hidden_size == 0
        || n_heads == 0
        || n_kv_heads == 0
        || hidden_size % n_heads != 0
        || n_heads % n_kv_heads != 0
    {
        return Err(anyhow!(
            "invalid transformer dimensions: hidden={hidden_size}, heads={n_heads}, kv_heads={n_kv_heads}"
        ));
    }

    Ok(TransformerConfig {
        architecture,
        n_layers,
        hidden_size,
        n_heads,
        n_kv_heads,
        intermediate_size,
        vocab_size,
        rms_norm_eps,
        rope_theta,
        tie_word_embeddings,
    })
}

// ── LayerLoader ───────────────────────────────────────────────────────────────

/// Zero-copy, one-layer-at-a-time loader for GGUF weight files.
///
/// Opens the file with `LoadMode::Lazy` so the OS never pages in more than the
/// tensors actually accessed.  Call `load_layer(n)` to obtain a `LoadedLayer`
/// for layer `n`; drop or `.unload()` it before loading the next layer to keep
/// RSS bounded to approximately one layer's worth of data.
pub struct LayerLoader {
    mmap: MmapLoader,
    index: LayerIndex,
    tensors: Vec<TensorSlice>,
    transformer_config: TransformerConfig,
    /// Shape of the token-embedding tensor, if present. Read generically from
    /// the GGUF at open time so callers can derive vocab/hidden dims at runtime
    /// without hardcoding per-architecture values. `[vocab, hidden]` row-major.
    embed_shape: Option<Vec<u64>>,
}

impl LayerLoader {
    /// Open `path` as a lazily-paged mmap and build the `LayerIndex`.
    ///
    /// Uses `LoadMode::Lazy` unconditionally — the whole point of this type is
    /// to avoid loading the full model into RAM at once.
    pub fn open(path: &Path) -> Result<Self> {
        let mmap =
            MmapLoader::open_with_mode(path, LoadMode::Lazy).map_err(|e| anyhow!("{}", e))?;
        let header = gguf_parser::parse_header(path).map_err(|e| anyhow!("{}", e))?;
        let index = LayerIndex::scan_tensors(&header.tensors, header.data_base);
        let embed_shape = find_embedding_shape(&header.tensors);
        let transformer_config = build_transformer_config(&header, &index, embed_shape.as_deref())?;
        let tensors = header
            .tensors
            .iter()
            .map(|tensor| TensorSlice {
                tensor_name: tensor.name.clone(),
                byte_offset: header.data_base + tensor.data_offset,
                byte_len: tensor.data_size,
                dtype: tensor.dtype,
                shape: tensor.shape.clone(),
            })
            .collect();
        Ok(LayerLoader {
            mmap,
            index,
            tensors,
            transformer_config,
            embed_shape,
        })
    }

    /// Shape of the token-embedding tensor (`[vocab, hidden]`), if the GGUF has
    /// one under a recognised name. Used to derive vocab/hidden at runtime.
    pub fn embedding_shape(&self) -> Option<&[u64]> {
        self.embed_shape.as_deref()
    }

    pub fn transformer_config(&self) -> &TransformerConfig {
        &self.transformer_config
    }

    /// Find the first tensor matching one of the supplied exact names.
    pub(crate) fn find_tensor(&self, names: &[&str]) -> Option<&TensorSlice> {
        names.iter().find_map(|name| {
            self.tensors
                .iter()
                .find(|tensor| tensor.tensor_name == *name)
        })
    }

    /// Borrow a tensor's quantized bytes directly from the mmap.
    pub(crate) fn tensor_bytes<'a>(&'a self, tensor: &TensorSlice) -> Result<&'a [u8]> {
        let start = tensor.byte_offset as usize;
        let end = start
            .checked_add(tensor.byte_len)
            .ok_or_else(|| anyhow!("tensor '{}' byte range overflow", tensor.tensor_name))?;
        self.mmap
            .as_bytes()
            .get(start..end)
            .ok_or_else(|| anyhow!("tensor '{}' byte range is outside mmap", tensor.tensor_name))
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
                let end = start + s.byte_len;
                let name_bytes: &'a str = {
                    // SAFETY: tensor_name is stored in self.index which lives
                    // for 'a; the String's backing memory is stable.
                    // We return &str by re-borrowing through the LayerIndex.
                    &self.index.slices[self.index.slices.partition_point(|x| x.layer_idx < n)
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
    for (_, data) in tensors.iter() {
        offsets.push(cursor);
        cursor += data.len() as u64;
    }
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
    if rem != 0 {
        f.write_all(&vec![0u8; (32 - rem) as usize]).unwrap();
    }
    for (_, data) in tensors.iter() {
        f.write_all(data).unwrap();
    }
    f.flush().unwrap();
    f
}

/// Write a tiny, complete F32 transformer GGUF for unit and integration tests.
#[cfg(any(test, feature = "test-utils"))]
pub fn write_transformer_gguf_pub(n_layers: usize) -> tempfile::NamedTempFile {
    write_transformer_gguf_with_tie_word_embeddings(n_layers, Some(MetadataValue::Bool(true)))
}

#[cfg(any(test, feature = "test-utils"))]
fn write_transformer_gguf_with_tie_word_embeddings(
    n_layers: usize,
    tie_word_embeddings: Option<MetadataValue>,
) -> tempfile::NamedTempFile {
    write_transformer_gguf_fixture(n_layers, tie_word_embeddings, true)
}

#[cfg(any(test, feature = "test-utils"))]
fn write_transformer_gguf_fixture(
    n_layers: usize,
    tie_word_embeddings: Option<MetadataValue>,
    include_output_head: bool,
) -> tempfile::NamedTempFile {
    use std::io::Write;

    const HIDDEN: usize = 4;
    const INTERMEDIATE: usize = 8;
    const VOCAB: usize = 16;
    const KV_DIM: usize = 2;

    fn values(count: usize, seed: usize) -> Vec<u8> {
        (0..count)
            .flat_map(|i| {
                let value = ((i + seed) % 19) as f32 * 0.005 - 0.04;
                value.to_le_bytes()
            })
            .collect()
    }

    fn ones(count: usize) -> Vec<u8> {
        (0..count).flat_map(|_| 1.0f32.to_le_bytes()).collect()
    }

    fn write_key(file: &mut tempfile::NamedTempFile, key: &str) {
        file.write_all(&(key.len() as u64).to_le_bytes()).unwrap();
        file.write_all(key.as_bytes()).unwrap();
    }

    let mut tensors: Vec<(String, Vec<u64>, Vec<u8>)> = vec![
        (
            "token_embd.weight".into(),
            vec![HIDDEN as u64, VOCAB as u64],
            values(HIDDEN * VOCAB, 1),
        ),
        (
            "output_norm.weight".into(),
            vec![HIDDEN as u64],
            ones(HIDDEN),
        ),
    ];
    if include_output_head {
        tensors.push((
            "output.weight".into(),
            vec![HIDDEN as u64, VOCAB as u64],
            values(HIDDEN * VOCAB, 7),
        ));
    }

    for layer in 0..n_layers {
        let prefix = format!("blk.{layer}");
        tensors.extend([
            (
                format!("{prefix}.attn_norm.weight"),
                vec![HIDDEN as u64],
                ones(HIDDEN),
            ),
            (
                format!("{prefix}.attn_q.weight"),
                vec![HIDDEN as u64, HIDDEN as u64],
                values(HIDDEN * HIDDEN, layer + 2),
            ),
            (
                format!("{prefix}.attn_k.weight"),
                vec![HIDDEN as u64, KV_DIM as u64],
                values(HIDDEN * KV_DIM, layer + 3),
            ),
            (
                format!("{prefix}.attn_v.weight"),
                vec![HIDDEN as u64, KV_DIM as u64],
                values(HIDDEN * KV_DIM, layer + 4),
            ),
            (
                format!("{prefix}.attn_output.weight"),
                vec![HIDDEN as u64, HIDDEN as u64],
                values(HIDDEN * HIDDEN, layer + 5),
            ),
            (format!("{prefix}.attn_q_norm.weight"), vec![2], ones(2)),
            (format!("{prefix}.attn_k_norm.weight"), vec![2], ones(2)),
            (
                format!("{prefix}.ffn_norm.weight"),
                vec![HIDDEN as u64],
                ones(HIDDEN),
            ),
            (
                format!("{prefix}.ffn_gate.weight"),
                vec![HIDDEN as u64, INTERMEDIATE as u64],
                values(HIDDEN * INTERMEDIATE, layer + 6),
            ),
            (
                format!("{prefix}.ffn_up.weight"),
                vec![HIDDEN as u64, INTERMEDIATE as u64],
                values(HIDDEN * INTERMEDIATE, layer + 7),
            ),
            (
                format!("{prefix}.ffn_down.weight"),
                vec![INTERMEDIATE as u64, HIDDEN as u64],
                values(HIDDEN * INTERMEDIATE, layer + 8),
            ),
        ]);
    }

    let mut file = tempfile::NamedTempFile::new().unwrap();
    file.write_all(b"GGUF").unwrap();
    file.write_all(&3u32.to_le_bytes()).unwrap();
    file.write_all(&(tensors.len() as u64).to_le_bytes())
        .unwrap();
    let kv_count = 9 + tie_word_embeddings.is_some() as u64;
    file.write_all(&kv_count.to_le_bytes()).unwrap();

    write_key(&mut file, "general.architecture");
    file.write_all(&8u32.to_le_bytes()).unwrap();
    file.write_all(&4u64.to_le_bytes()).unwrap();
    file.write_all(b"test").unwrap();

    for (key, value) in [
        ("test.block_count", n_layers as u32),
        ("test.embedding_length", HIDDEN as u32),
        ("test.attention.head_count", 2),
        ("test.attention.head_count_kv", 1),
        ("test.feed_forward_length", INTERMEDIATE as u32),
        ("test.vocab_size", VOCAB as u32),
    ] {
        write_key(&mut file, key);
        file.write_all(&4u32.to_le_bytes()).unwrap();
        file.write_all(&value.to_le_bytes()).unwrap();
    }

    for (key, value) in [
        ("test.attention.layer_norm_rms_epsilon", 1e-5f32),
        ("test.rope.freq_base", 10_000.0f32),
    ] {
        write_key(&mut file, key);
        file.write_all(&6u32.to_le_bytes()).unwrap();
        file.write_all(&value.to_le_bytes()).unwrap();
    }

    if let Some(value) = tie_word_embeddings {
        write_key(&mut file, "test.tie_word_embeddings");
        match value {
            MetadataValue::Bool(value) => {
                file.write_all(&7u32.to_le_bytes()).unwrap();
                file.write_all(&[u8::from(value)]).unwrap();
            }
            MetadataValue::U64(value) => {
                file.write_all(&10u32.to_le_bytes()).unwrap();
                file.write_all(&value.to_le_bytes()).unwrap();
            }
            other => panic!("unsupported test metadata value: {other:?}"),
        }
    }

    let mut offset = 0u64;
    for (name, shape, data) in &tensors {
        file.write_all(&(name.len() as u64).to_le_bytes()).unwrap();
        file.write_all(name.as_bytes()).unwrap();
        file.write_all(&(shape.len() as u32).to_le_bytes()).unwrap();
        for dim in shape {
            file.write_all(&dim.to_le_bytes()).unwrap();
        }
        file.write_all(&0u32.to_le_bytes()).unwrap();
        file.write_all(&offset.to_le_bytes()).unwrap();
        offset += data.len() as u64;
    }

    let position = file.as_file().metadata().unwrap().len();
    let padding = (32 - position % 32) % 32;
    file.write_all(&vec![0; padding as usize]).unwrap();
    for (_, _, data) in &tensors {
        file.write_all(data).unwrap();
    }
    file.flush().unwrap();
    file
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
            name: name.to_owned(),
            shape: vec![n_elems],
            dtype: GgufDtype::F32,
            data_offset: offset,
            data_size: size,
            raw_data: Vec::new(),
        }
    }

    fn make_file(tensors: Vec<TensorInfo>) -> GgufFile {
        GgufFile {
            version: 3,
            tensors,
            data_base: 0,
        }
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
        f.write_all(b"GGUF").unwrap(); // magic
        f.write_all(&3u32.to_le_bytes()).unwrap(); // version 3
        f.write_all(&(tensors.len() as u64).to_le_bytes()).unwrap(); // tensor_count
        f.write_all(&0u64.to_le_bytes()).unwrap(); // kv_count = 0

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
            f.write_all(&(name_bytes.len() as u64).to_le_bytes())
                .unwrap(); // name length
            f.write_all(name_bytes).unwrap(); // name
            f.write_all(&1u32.to_le_bytes()).unwrap(); // n_dims = 1
            // shape[0] = number of f32 elements implied by data.len()
            let n_elems = (data.len() / 4).max(1) as u64;
            f.write_all(&n_elems.to_le_bytes()).unwrap(); // shape[0]
            f.write_all(&0u32.to_le_bytes()).unwrap(); // dtype = F32 (0)
            f.write_all(&off.to_le_bytes()).unwrap(); // data_offset
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
        let f = write_minimal_gguf(&[("model.layers.0.self_attn.q_proj.weight", data.as_slice())]);
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
        let f = write_minimal_gguf(&[("model.layers.0.self_attn.q_proj.weight", data.as_slice())]);
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
    fn tie_word_embeddings_bool_round_trip() {
        for expected in [true, false] {
            let file = write_transformer_gguf_with_tie_word_embeddings(
                1,
                Some(MetadataValue::Bool(expected)),
            );
            let loader = LayerLoader::open(file.path()).expect("open transformer GGUF");

            assert_eq!(loader.transformer_config().tie_word_embeddings, expected);
        }
    }

    #[test]
    fn tie_word_embeddings_absent_with_separate_head_defaults_false() {
        let file = write_transformer_gguf_with_tie_word_embeddings(1, None);
        let loader = LayerLoader::open(file.path()).expect("open transformer GGUF");

        assert!(!loader.transformer_config().tie_word_embeddings);
    }

    #[test]
    fn tie_word_embeddings_absent_without_separate_head_is_inferred_true() {
        let file = write_transformer_gguf_fixture(1, None, false);
        let loader = LayerLoader::open(file.path()).expect("open transformer GGUF");

        assert!(loader.transformer_config().tie_word_embeddings);
    }

    #[test]
    fn tie_word_embeddings_explicit_false_overrides_structural_inference() {
        let file = write_transformer_gguf_fixture(1, Some(MetadataValue::Bool(false)), false);
        let loader = LayerLoader::open(file.path()).expect("open transformer GGUF");

        assert!(!loader.transformer_config().tie_word_embeddings);
    }

    #[test]
    fn tie_word_embeddings_non_bool_disables_structural_inference() {
        let file = write_transformer_gguf_fixture(1, Some(MetadataValue::U64(1)), false);
        let loader = LayerLoader::open(file.path()).expect("open transformer GGUF");

        assert!(!loader.transformer_config().tie_word_embeddings);
    }

    #[test]
    fn test_loaded_layer_unload_then_reload() {
        let data: Vec<u8> = [3.0f32, 4.0f32]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let f = write_minimal_gguf(&[("model.layers.0.self_attn.q_proj.weight", data.as_slice())]);
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
        let idx = LayerIndex::scan(&file);
        idx.slices.len() == pairs.len()
    }

    #[quickcheck]
    fn scan_num_layers_equals_max_plus_one(pairs: Vec<(u8, String)>) -> bool {
        if pairs.is_empty() {
            return true;
        }
        let file = build_file_from_pairs(&pairs);
        let idx = LayerIndex::scan(&file);
        let expected = pairs.iter().map(|(l, _)| *l as usize).max().unwrap() + 1;
        idx.num_layers == expected
    }

    #[quickcheck]
    fn layer_slices_sorted(pairs: Vec<(u8, String)>) -> bool {
        let file = build_file_from_pairs(&pairs);
        let idx = LayerIndex::scan(&file);
        for n in 0..idx.num_layers {
            let slices = idx.layer_slices(n);
            if !slices
                .windows(2)
                .all(|w| w[0].tensor_name <= w[1].tensor_name)
            {
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
        if loader.num_layers() == 0 {
            return true;
        }

        let loaded = match loader.load_layer(0) {
            Ok(l) => l,
            Err(_) => return false,
        };

        // Compute expected range from the index directly.
        let layer_slices = loader.index.layer_slices(0);
        let expected_start = layer_slices
            .iter()
            .map(|s| s.byte_offset as usize)
            .min()
            .unwrap();
        let expected_end = layer_slices
            .iter()
            .map(|s| s.byte_offset as usize + s.byte_len)
            .max()
            .unwrap();

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
        if loader.num_layers() == 0 {
            return true;
        }

        // First load + explicit unload.
        let loaded = match loader.load_layer(0) {
            Ok(l) => l,
            Err(_) => return false,
        };
        loaded.unload();

        // Reload must succeed — mmap is still valid, only advisory pages changed.
        loader.load_layer(0).is_ok()
    }
}
