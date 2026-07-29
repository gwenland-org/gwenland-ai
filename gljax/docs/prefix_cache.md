# Prefix cache — design only, ARTX09 Wave A9.4

Status: design note, transcribed and grounded against the current codebase.
**No implementation in this sprint.** This exists so the KV cache's actual
shape stays compatible with the design below, checked explicitly at the end
of this document — not because the design itself is built.

## The standard design doesn't fit gljax, and why

Prefix caching normally reuses a shared prompt prefix's KV across requests by
sharing the *physical* KV blocks between sequences — block-table addressing,
i.e. PagedAttention. gljax's continuous-batching design (ARTX07) explicitly
declines PagedAttention (its own "Why PagedAttention Is Not Used" section).
ARTX07's `StaticKvSlab` design gives each slot a private, contiguous region —
two slots cannot point at the same KV bytes. Physical sharing is
architecturally unavailable under that design.

⚠️ **ARTX07 itself does not exist in this codebase yet.** `StaticKvSlab`,
`KvSlotManager`, `SlotId`, and `max_slots` are all ARTX07 vocabulary this
sprint did not build (continuous batching / slot accounting is explicitly
listed in the sprint's build order and was not reached). What exists today is
`gljax::runtime::CachedSession`, which holds exactly one request's KV in a
single `[n_layers, max_seq_len, n_kv_heads, head_dim]` buffer (see
`ops::kv_cache`'s module docs) — there is no multi-slot slab to share
prefixes *between* at all right now. Everything below describes the design
ARTX09 specifies for when ARTX07 lands, not something gljax can do today.

## The reframing that makes it work anyway

Prefix caching bundles two separable wins:

| Win | Mechanism | Available under a static (non-paged) slab? |
|---|---|---|
| **Memory dedup** — N requests share one physical copy of the prefix KV | block tables / paging | ❌ needs the PagedAttention design ARTX07 rejected |
| **Compute reuse** — skip re-running prefill over the shared prefix | cached KV copied *into* a private slot | ✅ yes |

**gljax's prefix cache targets compute reuse only, and explicitly forgoes
memory dedup.** The justification is ARTX08's phase analysis: prefill is
compute-bound, so skipping prefill compute for a shared prefix is the
expensive half of the saving. Memory dedup helps capacity, which ARTX10 §4's
(parked) quantized KV cache attacks by a different route that *does* fit a
static slab — no paging needed for that half.

```text
Request arrives with prompt = [shared_system_prompt ++ user_turn]
   │
   ├─ Look up shared_system_prompt in the prefix store
   │     hit  → copy its cached KV into the slot at positions 0..P
   │            set the position counter to P
   │            prefill ONLY the user_turn (positions P..)
   │
   └─ miss → ordinary chunked prefill; optionally insert the
             resulting prefix KV into the store
```

The copy costs `P × H_kv × D × 2 (K,V) × n_layers × dtype_bytes` of
device-to-device movement — for `P = 512` on Qwen2.5-0.5B in bf16, **6.3 MB**
— against skipping 512 tokens of prefill compute across 24 layers. Strongly
favorable, but a *trade*, not a free win, and it must be measured at small
`P` where the copy may dominate. This is why the design includes a
`min_cacheable_prefix` tunable below which the cache is bypassed entirely,
**defaulted from measurement, not guessed** — no such measurement exists yet
either.

## Where the prefix KV would live

```rust
// gljax/src/attn/prefix.rs — NOT BUILT

/// A device-resident KV region OUTSIDE the ARTX07 slab, holding prefix KV
/// for reuse. Not addressable by attention directly — it is a copy source.
pub struct PrefixStore {
    /// Per-layer K/V for cached prefixes, packed back-to-back.
    kv_k: Vec<PjRtBuffer>,   // [prefix_capacity, H_kv, D] per layer
    kv_v: Vec<PjRtBuffer>,
    /// Radix index over token sequences -> (offset, length). ARTX09 Wave A9.5.
    index: RadixIndex,
    capacity_tokens: usize,
}

impl PrefixStore {
    /// Longest cached prefix of `tokens`. Returns None below min_cacheable_prefix.
    pub fn longest_match(&self, tokens: &[u32]) -> Option<PrefixMatch>;
    /// Copy a matched prefix's KV into a slab slot at positions 0..len.
    pub fn hydrate(&self, m: &PrefixMatch, slab: &mut StaticKvSlab, slot: SlotId)
        -> Result<usize, AttnError>;
    /// Insert after a miss. May evict (LRU over the radix tree).
    pub fn insert(&mut self, tokens: &[u32], slab: &StaticKvSlab, slot: SlotId, len: usize);
}
```

**Design decision (ARTX09's, kept as-is): `PrefixStore` is separate from
`StaticKvSlab` and does not participate in attention. It is a copy source
only.** This preserves ARTX07's ownership/storage split: `KvSlotManager`
knows nothing about buffers, `StaticKvSlab` knows nothing about requests or
prefixes, and `PrefixStore` would be a third module with a third concern
rather than a widened second one — once ARTX07 exists for it to sit next to.

## Compatibility check against the KV cache gljax actually has today

The one thing this sprint *can* verify without building any of the above:
does today's `ops::kv_cache` layout rule out this design, or leave room for
it?

`kv_k`/`kv_v` in `cache_shape(n_layers, max_seq_len, n_kv_heads, head_dim)` —
`[n_layers, max_seq_len, n_kv_heads, head_dim]` — is sequence-major within a
layer, written by `dynamic_update_slice` (`write_at`/`write_range`) and read
by `dynamic_slice`. A `hydrate()` copying `P` positions of prefix KV into a
fresh session's cache is exactly a `write_range` call at `start=0`,
`len=P` — the same primitive prefill's own bulk cache fill already uses (see
`ops::kv_cache::write_range`'s docs). Nothing about the current single-slot
layout blocks that; the only real ARTX07 dependency is *which slot* the copy
targets when more than one request is in flight, which requires the slab
this sprint did not build.

## What would need to exist before this moves from design to implementation

1. ARTX07 (continuous batching / `StaticKvSlab` / `KvSlotManager`) — the
   dependency this document's every reference to "slot" assumes.
2. ARTX09 Wave A9.5 (RadixAttention research) for the `RadixIndex` prefix
   lookup structure — not researched in this sprint either.
3. A `min_cacheable_prefix` measurement on real hardware, since the
   copy-vs-recompute trade is only favorable above some prefix length that
   has never been measured for gljax's targets.
