//! Cursor-based KV cache: one flat pre-allocated buffer, zero realloc
//! inside the context window.
//!
//! Layout: `[layer][kv][head][seq][dim]` (kv: 0 = keys, 1 = values).
//! Why this order? Attention scans `seq` sequentially for one layer+head at
//! a time, so keeping `[seq][dim]` contiguous makes those reads a single
//! linear sweep — cache-friendly and prefetcher-friendly.
//!
//! The cursor (`current_pos`) advances once per decoded token. Reset is
//! `current_pos = 0` — no free, no zeroing, O(1).

/// Pre-allocated per-layer, per-head key/value cache.
pub struct KvCache {
    /// Flat storage, `n_layers * 2 * n_heads * max_context * head_dim` f32.
    data: Vec<f32>,
    /// Sequence position the *next* token will be written to.
    current_pos: usize,
    /// Number of transformer layers.
    pub n_layers: usize,
    /// Number of KV heads per layer (fewer than query heads under GQA).
    pub n_heads: usize,
    /// Dimension of each head.
    pub head_dim: usize,
    /// Maximum sequence length the buffer was sized for.
    pub max_context: usize,
}

impl KvCache {
    /// Allocate the full cache up front — the only allocation this type
    /// ever performs.
    pub fn new(n_layers: usize, n_heads: usize, head_dim: usize, max_context: usize) -> Self {
        KvCache {
            data: vec![0f32; n_layers * 2 * n_heads * max_context * head_dim],
            current_pos: 0,
            n_layers,
            n_heads,
            head_dim,
            max_context,
        }
    }

    /// Start offset of the `[seq][dim]` region for one layer+kv+head.
    #[inline(always)]
    fn region(&self, layer: usize, kv: usize, head: usize) -> usize {
        (((layer * 2 + kv) * self.n_heads) + head) * self.max_context * self.head_dim
    }

    /// Write this token's K row for `layer`/`head` at the cursor position.
    /// `data.len()` must equal `head_dim`. Caller must ensure the cursor is
    /// within `max_context` (see [`KvCache::is_full`]).
    #[inline]
    pub fn write_k(&mut self, layer: usize, head: usize, data: &[f32]) {
        debug_assert_eq!(data.len(), self.head_dim);
        debug_assert!(self.current_pos < self.max_context);
        let off = self.region(layer, 0, head) + self.current_pos * self.head_dim;
        self.data[off..off + self.head_dim].copy_from_slice(data);
    }

    /// Write this token's V row for `layer`/`head` at the cursor position.
    #[inline]
    pub fn write_v(&mut self, layer: usize, head: usize, data: &[f32]) {
        debug_assert_eq!(data.len(), self.head_dim);
        debug_assert!(self.current_pos < self.max_context);
        let off = self.region(layer, 1, head) + self.current_pos * self.head_dim;
        self.data[off..off + self.head_dim].copy_from_slice(data);
    }

    /// All cached keys for `layer`/`head` *including* the row written for the
    /// in-flight token: `(current_pos + 1) * head_dim` values, contiguous.
    /// Call only after `write_k` for the current position.
    #[inline]
    pub fn read_k(&self, layer: usize, head: usize) -> &[f32] {
        let off = self.region(layer, 0, head);
        &self.data[off..off + (self.current_pos + 1) * self.head_dim]
    }

    /// All cached values for `layer`/`head` including the in-flight token.
    #[inline]
    pub fn read_v(&self, layer: usize, head: usize) -> &[f32] {
        let off = self.region(layer, 1, head);
        &self.data[off..off + (self.current_pos + 1) * self.head_dim]
    }

    /// Advance the cursor after every layer/head has written the current
    /// token. Call exactly once per decoded token.
    #[inline]
    pub fn advance(&mut self) {
        self.current_pos += 1;
    }

    /// Number of fully committed (advanced-past) tokens.
    pub fn current_pos(&self) -> usize {
        self.current_pos
    }

    /// True when the next write would fall outside the buffer.
    pub fn is_full(&self) -> bool {
        self.current_pos >= self.max_context
    }

    /// Reset for a new conversation. O(1): cursor to 0, no free, no zeroing —
    /// stale rows are unreachable because reads stop at the cursor.
    pub fn reset(&mut self) {
        self.current_pos = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_read_roundtrip_with_cursor() {
        let mut cache = KvCache::new(2, 3, 4, 8);
        assert_eq!(cache.current_pos(), 0);

        for t in 0..5 {
            for layer in 0..2 {
                for head in 0..3 {
                    let k = [t as f32; 4];
                    let v = [t as f32 + 100.0; 4];
                    cache.write_k(layer, head, &k);
                    cache.write_v(layer, head, &v);
                }
            }
            // Reads include the in-flight token before advance.
            assert_eq!(cache.read_k(1, 2).len(), (t + 1) * 4);
            cache.advance();
        }
        assert_eq!(cache.current_pos(), 5);

        // Token t's row must hold value t, contiguous in [seq][dim] order.
        let k = &cache.data[cache.region(1, 0, 2)..];
        for t in 0..5 {
            for d in 0..4 {
                assert_eq!(k[t * 4 + d], t as f32);
            }
        }
        let v_off = cache.region(0, 1, 0);
        assert_eq!(cache.data[v_off + 3 * 4], 103.0);
    }

    #[test]
    fn reset_is_cursor_only() {
        let mut cache = KvCache::new(1, 1, 2, 4);
        cache.write_k(0, 0, &[1.0, 2.0]);
        cache.write_v(0, 0, &[3.0, 4.0]);
        cache.advance();
        assert_eq!(cache.current_pos(), 1);

        cache.reset();
        assert_eq!(cache.current_pos(), 0);
        assert!(!cache.is_full());

        // New writes land at position 0 again.
        cache.write_k(0, 0, &[9.0, 9.0]);
        assert_eq!(cache.read_k(0, 0), &[9.0, 9.0]);
    }

    #[test]
    fn full_detection() {
        let mut cache = KvCache::new(1, 1, 2, 2);
        assert!(!cache.is_full());
        cache.advance();
        cache.advance();
        assert!(cache.is_full());
    }
}
