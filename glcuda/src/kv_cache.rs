//! Device-resident KV cache: one region of the backend buffer, cursor
//! advanced per token — the VRAM mirror of glproc's `KvCache`.
//!
//! Layout matches glproc exactly: `[layer][kv][head][seq][dim]` (kv: 0 =
//! keys, 1 = values), so attention reads one contiguous `[seq][dim]` sweep
//! per layer+head, and the semantic translation stays 1:1.
//!
//! Deviation from the ArchGLML_X2 §12 budget sketch: entries are **f32**,
//! not f16 — glproc's cache is f32 and M2's contract is numerical parity
//! with it. KV-f16 (and the 2x VRAM saving) is a §24.2 follow-up that
//! needs its own numerical characterization first.

use glcore::GlError;

use crate::buffer::DevSlice;
use crate::driver::Cuda;
use crate::ffi::CUdeviceptr;

/// Pre-allocated per-layer, per-head key/value cache in VRAM.
pub struct KvCacheDev {
    /// Flat storage, `n_layers * 2 * n_heads * max_context * head_dim` f32.
    data: DevSlice,
    /// Sequence position the *next* token will be written to.
    current_pos: usize,
    /// Number of transformer layers.
    pub n_layers: usize,
    /// Number of KV heads per layer.
    pub n_heads: usize,
    /// Dimension of each head.
    pub head_dim: usize,
    /// Maximum sequence length the region was sized for.
    pub max_context: usize,
}

impl KvCacheDev {
    /// Number of f32 elements the cache region must hold.
    pub fn numel(n_layers: usize, n_heads: usize, head_dim: usize, max_context: usize) -> usize {
        n_layers * 2 * n_heads * max_context * head_dim
    }

    /// Wrap a pre-allocated backend-buffer region (sized via [`Self::numel`]).
    pub fn new(
        data: DevSlice,
        n_layers: usize,
        n_heads: usize,
        head_dim: usize,
        max_context: usize,
    ) -> Self {
        debug_assert_eq!(
            data.len_f32(),
            Self::numel(n_layers, n_heads, head_dim, max_context)
        );
        KvCacheDev { data, current_pos: 0, n_layers, n_heads, head_dim, max_context }
    }

    /// Element offset of the `[seq][dim]` region for one layer+kv+head —
    /// identical arithmetic to glproc's `KvCache::region`.
    #[inline(always)]
    fn region(&self, layer: usize, kv: usize, head: usize) -> usize {
        (((layer * 2 + kv) * self.n_heads) + head) * self.max_context * self.head_dim
    }

    /// Device address `elems` f32 past the region start.
    #[inline(always)]
    fn addr(&self, elems: usize) -> CUdeviceptr {
        self.data.dptr + (elems * 4) as u64
    }

    /// Copy this token's K row (`head_dim` f32 at `src` on device) into the
    /// cursor position for `layer`/`head`.
    pub fn write_k(
        &mut self,
        cuda: &Cuda,
        layer: usize,
        head: usize,
        src: CUdeviceptr,
    ) -> Result<(), GlError> {
        debug_assert!(self.current_pos < self.max_context);
        let off = self.region(layer, 0, head) + self.current_pos * self.head_dim;
        cuda.dtod(self.addr(off), src, self.head_dim * 4)
    }

    /// Copy this token's V row into the cursor position (see [`Self::write_k`]).
    pub fn write_v(
        &mut self,
        cuda: &Cuda,
        layer: usize,
        head: usize,
        src: CUdeviceptr,
    ) -> Result<(), GlError> {
        debug_assert!(self.current_pos < self.max_context);
        let off = self.region(layer, 1, head) + self.current_pos * self.head_dim;
        cuda.dtod(self.addr(off), src, self.head_dim * 4)
    }

    /// Device address of all cached keys for `layer`/`head` — the caller
    /// reads `(current_pos + 1) * head_dim` values (the in-flight token
    /// included; call after `write_k`).
    pub fn read_k(&self, layer: usize, head: usize) -> CUdeviceptr {
        self.addr(self.region(layer, 0, head))
    }

    /// Device address of all cached values for `layer`/`head`.
    pub fn read_v(&self, layer: usize, head: usize) -> CUdeviceptr {
        self.addr(self.region(layer, 1, head))
    }

    /// Elements between consecutive KV heads' `[seq][dim]` regions within a
    /// layer — the stride the fused attention kernel adds per `kv_head`.
    /// `read_k(l, 0)` + `n * head_stride()` == `read_k(l, n)` (in elements).
    pub fn head_stride(&self) -> usize {
        self.max_context * self.head_dim
    }

    /// Advance the cursor after every layer/head committed the current
    /// token. Call exactly once per token.
    pub fn advance(&mut self) {
        self.current_pos += 1;
    }

    /// Number of fully committed tokens.
    pub fn current_pos(&self) -> usize {
        self.current_pos
    }

    /// True when the next write would fall outside the region.
    pub fn is_full(&self) -> bool {
        self.current_pos >= self.max_context
    }

    /// Reset for a new conversation. O(1): cursor to 0, no zeroing — stale
    /// rows are unreachable because reads stop at the cursor.
    pub fn reset(&mut self) {
        self.current_pos = 0;
    }
}
