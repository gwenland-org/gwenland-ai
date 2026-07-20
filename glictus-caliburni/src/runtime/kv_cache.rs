//! KV cache allocation (ARTX06 Â§"KV Cache").
//!
//! The cache is sized once, up front, for `max_seq_len` tokens and never
//! reallocated during inference â€” a realloc mid-pass would move every key and
//! value the backend holds pointers into.
//!
//! Storage is raw bytes, not typed elements: the cache holds whatever the
//! backend writes, and this module never interprets it. It does not know
//! whether those bytes are FP32, FP16, or BF16 â€” only how many bytes one
//! element takes, read from
//! [`RuntimeConfig::kv_element_size`](crate::runtime::RuntimeConfig::kv_element_size),
//! which the runtime fills in from the backend at initialization.
//!
//! Dependencies run one way â€” backend â†’ runtime â†’ KV cache â€” so the cache
//! never reaches back for a format it would then have to understand.
//!
//! ## Why `max_seq_len` defaults well below `context_length`
//!
//! Cache size is linear in sequence length, and the per-token cost is set by
//! the KV-head count â€” which GQA decouples from model size. Computed from the
//! three models this crate is verified against, at f32:
//!
//! | Model | KV heads | per token | @2048 | @full ctx |
//! |---|---|---|---|---|
//! | Qwen2.5-0.5B | 2 | 24 KiB | 48 MiB | 768 MiB (32k) |
//! | Qwen2.5-1.5B | 2 | 56 KiB | 112 MiB | 1.75 GiB (32k) |
//! | Qwen3-1.7B | 8 | 224 KiB | 448 MiB | **8.75 GiB** (41k) |
//!
//! Qwen3-1.7B is the case that matters: a 1.7B model whose full-context cache
//! alone exceeds the 8 GB reference machine's entire RAM, because it has 4x
//! the KV heads of the 1.5B despite being a similar size. Defaulting
//! `max_seq_len` to the manifest's `context_length` would therefore turn a
//! model that loads fine into an instant OOM â€” hence
//! [`RuntimeConfig::max_seq_len`](crate::runtime::RuntimeConfig) defaults to
//! 2048 and is clamped, never inferred, upward.

use crate::error::{GllmError, GllmResult};
use crate::manifest::ModelMetadata;
use crate::runtime::types::RuntimeConfig;

/// Shape and element size of a model's KV cache.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvCacheConfig {
    /// Number of transformer layers (one slot each).
    pub num_layers: u32,
    /// KV heads per layer. Under GQA this is smaller than the query-head
    /// count â€” sizing off `num_heads` would over-allocate several times over.
    pub num_kv_heads: u32,
    /// Elements per head.
    pub head_dim: u64,
    /// Tokens the cache is sized for.
    pub max_seq_len: u32,
    /// Bytes per stored element, as reported by the backend.
    ///
    /// Not a dtype: the cache never learns whether these bytes are FP32,
    /// FP16, or BF16. Populated from
    /// [`RuntimeConfig::kv_element_size`](crate::runtime::RuntimeConfig::kv_element_size).
    pub element_size: usize,
}

impl KvCacheConfig {
    /// Derive a config from manifest metadata and the runtime's config.
    ///
    /// Takes the element size from `runtime_config` â€” which the runtime filled
    /// in from the backend â€” so no caller has to know, or can misreport, the
    /// KV format.
    ///
    /// `max_seq_len` is clamped to the model's own `context_length`: sizing
    /// beyond it would allocate memory no position can ever address. Returns
    /// [`GllmError::MissingMetadata`] when `head_dim` is not derivable
    /// (`embedding_length` not divisible by `num_heads`).
    pub fn from_metadata(meta: &ModelMetadata, runtime_config: &RuntimeConfig) -> GllmResult<Self> {
        let head_dim = meta.head_dim().ok_or_else(|| {
            GllmError::MissingMetadata(format!(
                "head_dim: embedding_length {} is not divisible by num_heads {}",
                meta.embedding_length, meta.num_heads
            ))
        })?;
        let ctx = u32::try_from(meta.context_length).unwrap_or(u32::MAX);
        Ok(KvCacheConfig {
            num_layers: meta.num_layers,
            num_kv_heads: meta.head_count_kv,
            head_dim,
            max_seq_len: runtime_config.max_seq_len.min(ctx),
            element_size: runtime_config.kv_element_size(),
        })
    }

    /// Bytes for one layer's K **and** V:
    /// `2 * num_kv_heads * head_dim * max_seq_len * element_size`.
    ///
    /// Saturates instead of overflowing, so an absurd config yields a huge
    /// number that [`KvCache::allocate`] then refuses â€” never a wrapped small
    /// one that would under-allocate.
    pub fn slot_size_bytes(&self) -> u64 {
        2u64.saturating_mul(self.num_kv_heads as u64)
            .saturating_mul(self.head_dim)
            .saturating_mul(self.max_seq_len as u64)
            .saturating_mul(self.element_size as u64)
    }

    /// Bytes for the whole cache: `slot_size_bytes * num_layers`.
    pub fn total_size_bytes(&self) -> u64 {
        self.slot_size_bytes()
            .saturating_mul(self.num_layers as u64)
    }

    /// Bytes one token adds to one layer (both K and V).
    pub fn bytes_per_token_per_layer(&self) -> u64 {
        2u64.saturating_mul(self.num_kv_heads as u64)
            .saturating_mul(self.head_dim)
            .saturating_mul(self.element_size as u64)
    }
}

/// One layer's slice of the cache.
///
/// `key` and `value` are each sized for `max_seq_len` tokens and stay that
/// size for the slot's lifetime; `current_seq_len` tracks how much is live.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvCacheSlot {
    /// Layer this slot belongs to.
    pub layer_index: u32,
    /// Key bytes, laid out `[num_kv_heads][max_seq_len][head_dim]`.
    pub key: Vec<u8>,
    /// Value bytes, same layout as `key`.
    pub value: Vec<u8>,
    /// Tokens currently cached.
    pub current_seq_len: u32,
    max_seq_len: u32,
}

impl KvCacheSlot {
    /// Allocate a zeroed slot for one layer.
    ///
    /// The K and V halves are allocated separately, each half of
    /// [`slot_size_bytes`](KvCacheConfig::slot_size_bytes).
    pub fn new(layer_index: u32, config: &KvCacheConfig) -> GllmResult<Self> {
        let half = config.slot_size_bytes() / 2;
        let half = usize::try_from(half).map_err(|_| GllmError::KvCacheAllocFailed {
            requested: config.total_size_bytes(),
            num_layers: config.num_layers,
            per_layer: config.slot_size_bytes(),
        })?;
        Ok(KvCacheSlot {
            layer_index,
            key: vec![0u8; half],
            value: vec![0u8; half],
            current_seq_len: 0,
            max_seq_len: config.max_seq_len,
        })
    }

    /// Drop cached tokens without freeing memory, readying the slot for a new
    /// sequence. Capacity is deliberately retained â€” that is the point of
    /// pre-allocating.
    pub fn reset(&mut self) {
        self.current_seq_len = 0;
    }

    /// Whether the slot has no room for another token.
    pub fn is_full(&self) -> bool {
        self.current_seq_len >= self.max_seq_len
    }

    /// Tokens this slot can still take.
    pub fn remaining_capacity(&self) -> u32 {
        self.max_seq_len.saturating_sub(self.current_seq_len)
    }

    /// Tokens this slot was sized for.
    pub fn max_seq_len(&self) -> u32 {
        self.max_seq_len
    }

    /// Record that `n` more tokens were cached.
    ///
    /// Returns [`GllmError::ExecutionFailed`] if that would exceed capacity â€”
    /// a silent clamp here would corrupt attention by making the backend and
    /// the cache disagree about how many positions are live.
    pub fn advance(&mut self, n: u32) -> GllmResult<()> {
        let next = self.current_seq_len.saturating_add(n);
        if next > self.max_seq_len {
            return Err(GllmError::ExecutionFailed {
                layer: self.layer_index,
                reason: format!(
                    "KV cache overflow: {} + {n} tokens exceeds max_seq_len {}",
                    self.current_seq_len, self.max_seq_len
                ),
            });
        }
        self.current_seq_len = next;
        Ok(())
    }

    /// Bytes allocated by this slot (K + V).
    pub fn allocated_bytes(&self) -> u64 {
        (self.key.len() + self.value.len()) as u64
    }
}

/// The model's full KV cache: one [`KvCacheSlot`] per layer.
#[derive(Debug)]
pub struct KvCache {
    slots: Vec<KvCacheSlot>,
    config: KvCacheConfig,
}

impl KvCache {
    /// Allocate every slot up front.
    ///
    /// Refuses configurations whose total exceeds [`Self::MAX_TOTAL_BYTES`]
    /// rather than letting the allocator abort the process â€” the reference
    /// 8 GB machine must get a diagnosable error, not an OOM kill.
    pub fn allocate(config: KvCacheConfig) -> GllmResult<Self> {
        let total = config.total_size_bytes();
        if total > Self::MAX_TOTAL_BYTES {
            return Err(GllmError::KvCacheAllocFailed {
                requested: total,
                num_layers: config.num_layers,
                per_layer: config.slot_size_bytes(),
            });
        }
        let slots = (0..config.num_layers)
            .map(|i| KvCacheSlot::new(i, &config))
            .collect::<GllmResult<Vec<_>>>()?;
        Ok(KvCache { slots, config })
    }

    /// Refuse to allocate more than this in one cache (32 GiB).
    ///
    /// Not a hardware limit â€” a guard against a manifest whose metadata
    /// implies an absurd cache, which should fail loud rather than swap the
    /// machine to death.
    pub const MAX_TOTAL_BYTES: u64 = 32 * 1024 * 1024 * 1024;

    /// Immutable access to one layer's slot.
    pub fn slot(&self, layer_index: u32) -> Option<&KvCacheSlot> {
        self.slots.get(layer_index as usize)
    }

    /// Mutable access to one layer's slot â€” the handle passed to a backend.
    pub fn slot_mut(&mut self, layer_index: u32) -> Option<&mut KvCacheSlot> {
        self.slots.get_mut(layer_index as usize)
    }

    /// Reset every slot for a fresh sequence, keeping the allocation.
    pub fn reset_all(&mut self) {
        for slot in &mut self.slots {
            slot.reset();
        }
    }

    /// Bytes actually allocated, summed over slots.
    ///
    /// Measured from the live `Vec`s rather than recomputed from the config,
    /// so it reports what was allocated, not what was intended.
    pub fn total_allocated_bytes(&self) -> u64 {
        self.slots.iter().map(|s| s.allocated_bytes()).sum()
    }

    /// Number of slots (one per layer).
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    /// Whether the cache has no slots (a zero-layer model).
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// The config this cache was built from.
    pub fn config(&self) -> &KvCacheConfig {
        &self.config
    }

    /// Longest `current_seq_len` across slots â€” how far the sequence has run.
    pub fn max_current_seq_len(&self) -> u32 {
        self.slots
            .iter()
            .map(|s| s.current_seq_len)
            .max()
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A `RuntimeConfig` at the given sequence length, element size left at
    /// its `f32` default (no backend attached).
    fn runtime_config(max_seq_len: u32) -> RuntimeConfig {
        RuntimeConfig::with_max_seq_len(max_seq_len)
    }

    /// Qwen2.5-0.5B: 24 layers, 2 KV heads (GQA), head_dim 64.
    fn qwen05b_config(max_seq_len: u32) -> KvCacheConfig {
        KvCacheConfig {
            num_layers: 24,
            num_kv_heads: 2,
            head_dim: 64,
            max_seq_len,
            element_size: 4,
        }
    }

    #[test]
    fn slot_size_follows_the_artx06_formula() {
        let c = qwen05b_config(2048);
        // 2 * 2 heads * 64 dim * 2048 tokens * 4 bytes = 2,097,152
        assert_eq!(c.slot_size_bytes(), 2 * 2 * 64 * 2048 * 4);
        assert_eq!(c.slot_size_bytes(), 2_097_152);
    }

    #[test]
    fn total_size_is_slot_size_times_layers() {
        let c = qwen05b_config(2048);
        assert_eq!(c.total_size_bytes(), c.slot_size_bytes() * 24);
        assert_eq!(c.total_size_bytes(), 50_331_648); // ~48 MiB
    }

    #[test]
    fn bytes_per_token_is_independent_of_seq_len() {
        // Per-token cost must not change with how long the cache is sized for.
        let short = qwen05b_config(128);
        let long = qwen05b_config(32768);
        assert_eq!(
            short.bytes_per_token_per_layer(),
            long.bytes_per_token_per_layer()
        );
        assert_eq!(short.bytes_per_token_per_layer(), 2 * 2 * 64 * 4);
    }

    #[test]
    fn gqa_sizing_uses_kv_heads_not_query_heads() {
        // Qwen2.5-0.5B has 14 query heads but only 2 KV heads. Sizing off
        // query heads would over-allocate 7x.
        let gqa = qwen05b_config(2048);
        let as_if_mha = KvCacheConfig {
            num_kv_heads: 14,
            ..gqa.clone()
        };
        assert_eq!(as_if_mha.total_size_bytes() / gqa.total_size_bytes(), 7);
    }

    #[test]
    fn slot_allocates_the_configured_size() {
        let c = qwen05b_config(512);
        let s = KvCacheSlot::new(3, &c).unwrap();
        assert_eq!(s.layer_index, 3);
        assert_eq!(s.allocated_bytes(), c.slot_size_bytes());
        assert_eq!(s.key.len(), s.value.len(), "K and V are symmetric");
        assert_eq!(s.current_seq_len, 0);
        assert!(s.key.iter().all(|&b| b == 0), "must start zeroed");
    }

    #[test]
    fn advance_tracks_and_reset_rewinds_without_freeing() {
        let c = qwen05b_config(16);
        let mut s = KvCacheSlot::new(0, &c).unwrap();
        let capacity = s.allocated_bytes();

        s.advance(10).unwrap();
        assert_eq!(s.current_seq_len, 10);
        assert_eq!(s.remaining_capacity(), 6);
        assert!(!s.is_full());

        s.reset();
        assert_eq!(s.current_seq_len, 0);
        assert_eq!(
            s.allocated_bytes(),
            capacity,
            "reset must not free the allocation"
        );
    }

    #[test]
    fn slot_reports_full_at_max_seq_len() {
        let c = qwen05b_config(8);
        let mut s = KvCacheSlot::new(0, &c).unwrap();
        s.advance(8).unwrap();
        assert!(s.is_full());
        assert_eq!(s.remaining_capacity(), 0);
    }

    #[test]
    fn advancing_past_capacity_errors_instead_of_clamping() {
        let c = qwen05b_config(4);
        let mut s = KvCacheSlot::new(7, &c).unwrap();
        s.advance(3).unwrap();

        let err = s.advance(2).unwrap_err();
        match err {
            GllmError::ExecutionFailed { layer, reason } => {
                assert_eq!(layer, 7);
                assert!(reason.contains("overflow"), "got {reason}");
            }
            other => panic!("expected ExecutionFailed, got {other:?}"),
        }
        assert_eq!(s.current_seq_len, 3, "failed advance must not mutate");
    }

    #[test]
    fn allocate_creates_one_slot_per_layer() {
        let c = qwen05b_config(256);
        let cache = KvCache::allocate(c.clone()).unwrap();
        assert_eq!(cache.len(), 24);
        assert!(!cache.is_empty());
        assert_eq!(cache.total_allocated_bytes(), c.total_size_bytes());
    }

    #[test]
    fn slots_are_indexed_by_layer() {
        let cache = KvCache::allocate(qwen05b_config(64)).unwrap();
        assert_eq!(cache.slot(0).unwrap().layer_index, 0);
        assert_eq!(cache.slot(23).unwrap().layer_index, 23);
        assert!(cache.slot(24).is_none(), "out of range");
    }

    #[test]
    fn reset_all_rewinds_every_slot() {
        let mut cache = KvCache::allocate(qwen05b_config(32)).unwrap();
        cache.slot_mut(0).unwrap().advance(10).unwrap();
        cache.slot_mut(5).unwrap().advance(20).unwrap();
        assert_eq!(cache.max_current_seq_len(), 20);

        cache.reset_all();
        assert_eq!(cache.max_current_seq_len(), 0);
        assert!(cache.slots.iter().all(|s| s.current_seq_len == 0));
    }

    #[test]
    fn zero_layer_model_allocates_an_empty_cache() {
        let c = KvCacheConfig {
            num_layers: 0,
            ..qwen05b_config(128)
        };
        let cache = KvCache::allocate(c).unwrap();
        assert!(cache.is_empty());
        assert_eq!(cache.total_allocated_bytes(), 0);
        assert_eq!(cache.max_current_seq_len(), 0);
    }

    #[test]
    fn absurd_config_is_refused_rather_than_oom() {
        // 128 layers x 64 KV heads x 128 dim x 1M tokens x 4B â€” far past the
        // guard. Must return an error, not attempt the allocation.
        let c = KvCacheConfig {
            num_layers: 128,
            num_kv_heads: 64,
            head_dim: 128,
            max_seq_len: 1_048_576,
            element_size: 4,
        };
        let err = KvCache::allocate(c).unwrap_err();
        assert!(
            matches!(err, GllmError::KvCacheAllocFailed { .. }),
            "got {err:?}"
        );
    }

    #[test]
    fn from_metadata_derives_head_dim_and_clamps_to_context() {
        let meta = crate::test_helpers::qwen05b_metadata();

        let c = KvCacheConfig::from_metadata(&meta, &runtime_config(2048)).unwrap();
        assert_eq!(c.head_dim, 64, "896 / 14");
        assert_eq!(c.num_kv_heads, 2, "GQA: not the 14 query heads");
        assert_eq!(c.max_seq_len, 2048);
        assert_eq!(c.element_size, 4, "f32 default when no backend reported");

        // Asking beyond the model's context is capped, not honoured.
        let capped = KvCacheConfig::from_metadata(&meta, &runtime_config(999_999)).unwrap();
        assert_eq!(capped.max_seq_len, 32768);
    }

    #[test]
    fn element_size_comes_from_the_backend_not_the_caller() {
        use crate::runtime::backend::{ExecutionBackend, NullBackend, KV_ELEMENT_SIZE_F16};

        let meta = crate::test_helpers::qwen05b_metadata();
        let backend = NullBackend::with_kv_element_size(KV_ELEMENT_SIZE_F16);

        // This is what the runtime does at init: ask the backend, store it.
        let mut cfg = runtime_config(2048);
        cfg.set_kv_element_size(backend.kv_element_size());

        let c = KvCacheConfig::from_metadata(&meta, &cfg).unwrap();
        assert_eq!(c.element_size, 2, "f16 backend halves the element size");

        // An f16 backend must halve the whole allocation versus f32.
        let f32_cfg = KvCacheConfig::from_metadata(&meta, &runtime_config(2048)).unwrap();
        assert_eq!(f32_cfg.total_size_bytes(), c.total_size_bytes() * 2);
    }

    #[test]
    fn a_zero_element_size_is_ignored() {
        // A backend reporting 0 would size the entire cache to nothing; the
        // previous value must survive instead.
        let mut cfg = runtime_config(2048);
        assert_eq!(cfg.kv_element_size(), 4);
        cfg.set_kv_element_size(0);
        assert_eq!(cfg.kv_element_size(), 4, "zero must not take effect");
    }

    #[test]
    fn qwen3_full_context_cache_exceeds_the_reference_machine() {
        // Qwen3-1.7B: 28 layers, 8 KV heads, head_dim 128, ctx 40960. A 1.7B
        // model whose full-context f32 cache is ~8.75 GiB â€” more RAM than the
        // reference i3 has in total. This is why max_seq_len is clamped down
        // by config rather than inferred from context_length.
        let full = KvCacheConfig {
            num_layers: 28,
            num_kv_heads: 8,
            head_dim: 128,
            max_seq_len: 40_960,
            element_size: 4,
        };
        let gib = full.total_size_bytes() as f64 / (1024.0 * 1024.0 * 1024.0);
        assert!(
            (8.7..8.8).contains(&gib),
            "expected ~8.75 GiB, computed {gib:.2} GiB"
        );

        // The 2048-token default keeps the same model at a workable 448 MiB.
        let default = KvCacheConfig {
            max_seq_len: 2048,
            ..full
        };
        assert_eq!(default.total_size_bytes(), 448 * 1024 * 1024);
    }

    #[test]
    fn from_metadata_rejects_indivisible_head_dim() {
        let mut meta = crate::test_helpers::qwen05b_metadata();
        meta.embedding_length = 100;
        meta.num_heads = 7; // 100 / 7 is not exact
        let err = KvCacheConfig::from_metadata(&meta, &runtime_config(128)).unwrap_err();
        assert!(
            matches!(err, GllmError::MissingMetadata(_)),
            "got {err:?}"
        );
    }
}
