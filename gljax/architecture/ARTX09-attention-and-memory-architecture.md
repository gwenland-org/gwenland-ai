# ARTX09 — Attention & Memory Architecture

**Series:** gljax (Sanctum Visibilia) Architecture Research
**Status:** Draft — research-grounded
**Depends on:** ARTX03 (`gqa_attention`), ARTX05 (static KV cache + bucketing), ARTX07 (StaticKvSlab, chunked prefill), ARTX08 (matrix compute — **binding**)
**Introduces:** `gljax/src/attn/`
**Next:** [ARTX10 — Quantized Runtime Architecture](ARTX10-quantized-runtime-architecture.md)
**Research grounded:** 2026-07-27 (sources at end)

---

# 0. Scope — this document is **additive**

⚠️ **DESIGN DECISION — ARTX09 does not re-specify the KV cache or the static slab.**

Those are settled and remain authoritative where they are:

| Already specified | Where | ARTX09's relationship |
|---|---|---|
| KV buffer layout `[B, H_kv, S, D]`, `dynamic_update_slice` at a runtime position, buffer donation | **ARTX05 §1–§2** | Referenced, never restated |
| Sequence bucketing (128/256/512/1024/2048) | **ARTX05 §3** | Referenced |
| `StaticKvSlab` — slot-dimension addressing, `clear_slot`, ownership-vs-storage split | **ARTX07 §Static KV Slab** | Referenced |
| Chunked prefill scheduling | **ARTX07 §Chunked Prefill** | ⭐ Re-analyzed in §2.3 — it turns out to be load-bearing for attention memory |
| KV sharding under TP / PP | **ARTX16 §3** | Referenced |
| Quantized KV cache (FP8/INT8) | **ARTX10 §4** | Referenced — the *format*; this document owns the *access pattern* |

ARTX09 covers only new ground:

1. **§1–§2** — FlashAttention research, and the memory cost gljax actually pays without it
2. **§3** — how gljax can reach a flash kernel *without owning one*
3. **§4** — KV access optimization (layout, head grouping, read amplification)
4. **§5–§6** — prefix caching and RadixAttention, and what is reachable under a static slab

---

# 1. Wave A9.1 — FlashAttention Research

## 1.1 The three generations

| | Contribution | Reported result |
|---|---|---|
| **FlashAttention-1** | IO-aware tiling + **online softmax** so the `S×S` score matrix is never materialized; recomputation in the backward pass instead of storage | Attention memory O(B·S) instead of O(B·S²) |
| **FlashAttention-2** | Better work partitioning and parallelism across the sequence dimension; fewer non-matmul FLOPs | ~2× over FA-1 |
| **FlashAttention-3** | Hopper asynchrony: **warp specialization** (producer warps drive TMA/`cp.async`, consumer warps run WGMMA + softmax concurrently), interleaved block-wise matmul and softmax, and FP8 via incoherent processing | **1.5–2.0× over FA-2** at FP16, up to **740 TFLOPS ≈ 75%** of H100 peak; FP8 near **1.2 PFLOPS** with 2.6× smaller error than baseline FP8 attention |

## 1.2 ⛔ The finding that determines everything below

> **XLA can fuse operations and schedule instructions, but it cannot rewrite your algorithm. It
> cannot infer the streaming/online-softmax reformulation — the insight that you never need to
> materialize the full attention matrix.**

This is a categorically different situation from ARTX08's GEMM analysis. There, gljax emits
`dot_general` and XLA picks among cuBLASLt/Triton/MXU — the *algorithm* is the same, only the kernel
differs. Here, FlashAttention **is a different algorithm**: a running max and running sum with
rescaling, tiled over both Q and KV. Fusion does not produce it. No amount of emitting a "better"
StableHLO graph derives it.

⚠️ So the honest baseline is: **gljax emitting the natural `softmax(QKᵀ/√d)·V` graph materializes the
score tensor.** XLA will fuse the elementwise chain around the dots and avoid some HBM round-trips —
but the softmax reduction runs over the full key dimension, which is a fusion boundary, and the
scores must exist across it.

⚠️ How much XLA fuses *around* that boundary is **unmeasured for gljax's targets** and must not be
assumed in either direction. §7's wave gate is a measurement, not a claim.

---

# 2. The Memory Cost gljax Actually Pays

## 2.1 The score tensor

```text
scores : [B, H, S_q, S_kv]     bf16
```

For Qwen2.5-0.5B (H=14, head_dim=64, 24 layers) at ARTX07's `max_slots = 8`:

| Phase | `S_q` | `S_kv` | Score tensor | Share of TPU v5e's 16 GB |
|---|---|---|---|---|
| **Decode** | 1 | 2048 | **458 KB** | negligible |
| Prefill, **unchunked** | 2048 | 2048 | **939 MB** | 5.9% |
| Prefill, **unchunked**, S=4096 | 4096 | 4096 | **3.76 GB** | 23% |
| Prefill, **unchunked**, S=8192 | 8192 | 8192 | **15.0 GB** | ⛔ does not fit |

⭐ **First conclusion, and it is the one that reorders the priorities: attention memory is a
*prefill* problem, not a decode problem.** During decode `S_q = 1`, so the score tensor is a thin
`[B, H, 1, S_kv]` slice — under half a megabyte. FlashAttention buys gljax essentially nothing on the
decode path.

⚠️ This composes with ARTX08's finding rather than contradicting it. ARTX08 established decode is
bandwidth-bound on *weights*; ARTX09 establishes prefill is memory-pressured on *scores*. Different
phase, different resource, different fix.

## 2.2 ⭐ ARTX07's chunked prefill is already a partial FlashAttention

The table above assumes unchunked prefill. **gljax does not do unchunked prefill.** ARTX07 adopted
Sarathi-Serve-style chunking, which caps the query dimension at the chunk size:

```text
scores : [B, H, chunk, S_kv]        NOT  [B, H, S, S]
```

| Config | Score tensor |
|---|---|
| chunk=512, S_kv=2048 | **235 MB** (vs 939 MB unchunked — 4× less) |
| chunk=512, S_kv=8192 | **939 MB** (vs 15.0 GB unchunked — **16× less**) |

⚠️ **The complexity class changes: O(S²) becomes O(chunk · S).** Chunked prefill tiles the *query*
direction at the scheduler level, which is one of FlashAttention's two tiling axes — obtained for
free, as a side effect of a scheduling decision ARTX07 made for latency reasons.

**What chunking does not give:**

* **No KV-direction tiling.** `S_kv` is still the full context, so memory grows linearly with context
  rather than being bounded.
* **No online softmax.** The chunk's scores are fully materialized before the softmax reduction.
* **No HBM-traffic elimination.** The score tile still round-trips through HBM between the two matmuls
  unless XLA fuses across it.

⚠️ **DESIGN DECISION — chunk size is now an attention-memory parameter, not only a latency
parameter.** ARTX07 chose the chunk size to keep decode stall-free. ARTX09 adds a second constraint:
chunk size linearly controls peak attention memory. Those two pressures point in opposite directions
(latency wants small chunks; prefill throughput wants large ones), and the choice must be made
knowing both. This should be recorded in ARTX07's policy module, not rediscovered.

## 2.3 Revised priority

```text
1. Chunked prefill (ARTX07)         ← already shipped, removes the quadratic blowup
2. Reach a flash kernel (§3)        ← removes the remaining linear term + HBM traffic
3. Everything else
```

---

# 3. Wave A9.2 — Reaching a Flash Kernel Without Owning One

## 3.1 The ARTX08 tension, stated plainly

ARTX08's holding is *"gljax does not own a GEMM kernel, and must not acquire one,"* and its rejected
alternative #2 declined `custom_call` because *"a custom_call must be registered per backend, so one
kernel becomes N kernels."*

FlashAttention is a kernel. So is gljax simply excluded from it?

**No — and the reason is a genuine distinction, not a loophole.** ARTX08's objection was to gljax
*authoring and maintaining* per-backend kernels. If a kernel **already exists in the backend** and
gljax merely causes it to be selected, gljax's maintenance burden is zero. The question is therefore
not "may gljax use a kernel" but "**can gljax reach an existing one without authoring it?**"

## 3.2 ⭐ GPU: yes, and gljax does not even emit a custom call

XLA:GPU ships **`CudnnFusedMHARewriter`**, enabled by `--xla_gpu_enable_cudnn_fmha=true`. It
**pattern-matches** the bmm1 → softmax → bmm2 shape in the HLO graph and rewrites it to the
`__cudnn$fmha` custom-call target — the compiled cuDNN fused multi-head attention kernel.

```text
gljax emits:     dot_general(Q,Kᵀ) → scale → mask → softmax → dot_general(P,V)
                                    │
                 XLA:GPU CudnnFusedMHARewriter pattern-matches
                                    ▼
                            __cudnn$fmha   (cuDNN FMHA kernel)
```

⚠️ **DESIGN DECISION — gljax emits the natural attention graph and lets the rewriter fire. It never
emits `custom_call` itself.**

This preserves ARTX08 completely: the emitted module is standard StableHLO, portable, with no
backend-specific branch. gljax's responsibilities become:

1. **Emit a pattern the rewriter recognizes** — which constrains op ordering and shape, not
   semantics.
2. **Verify the rewrite fired** — because a near-miss silently falls back to the materializing path
   with no error.

⭐ **This is structurally identical to ARTX10 §5's quantization capability probe**: emit standard IR,
let the backend decide, then *measure whether the intended lowering happened*. Two independent
subsystems converging on the same pattern is a signal it should be a named principle of the series
(§8).

## 3.3 ⛔ TPU: no, and the asymmetry must not be papered over

TPU's flash-attention implementation is **Splash Attention** — JAX's FlashAttention for TPU, written
in **Pallas** and lowered through Mosaic. It is the default attention kernel in MaxText for DeepSeek,
Gemma, and Llama. Its advantages come from things Pallas exposes and StableHLO does not: explicit
double-buffered DMA against VMEM, a 2-D `(num_q_blocks, num_kv_blocks)` grid that lets Mosaic skip
fully-masked causal tiles entirely, and MXU-matched tile sizes.

**Pallas is a kernel language. Reaching Splash Attention means authoring a Pallas kernel — which is
exactly what ARTX08 declined.**

⚠️ So gljax's attention story is **backend-asymmetric**, and that must be stated in the capability
matrix rather than discovered:

| Backend | Flash path | gljax reachable? |
|---|---|---|
| **CUDA** | `CudnnFusedMHARewriter` → `__cudnn$fmha` | ✅ via pattern match, no custom call |
| **TPU** | Splash Attention (Pallas/Mosaic) | ❌ requires authoring a Pallas kernel |
| **CPU** | — | ❌ (oracle backend only, per ARTX16 scope) |

⚠️ This is a **real capability gap on gljax's flagship target**, not a footnote. TPU v5e was named in
ARTX01 as a primary device. The honest position: on TPU, gljax's prefill attention memory is bounded
by ARTX07's chunk size (§2.2) and nothing more, until either (a) XLA:TPU gains its own attention
rewriter, or (b) gljax accepts one Pallas kernel as a scoped exception. **(b) is a decision for a
future wave and must be taken explicitly, the same way ARTX07 deferred PagedAttention.**

## 3.4 What "emit a recognizable pattern" constrains

The rewriter matches a shape, so gljax must not perturb it:

```rust
// gljax/src/attn/pattern.rs
//
// ⚠️ The op ORDER below is load-bearing. Folding the scale into Q before the
// first dot, or applying the mask after the softmax, are both mathematically
// equivalent and both may defeat the rewriter's pattern match.

pub fn attention_flash_friendly(q: &Tensor, k: &Tensor, v: &Tensor, cfg: &AttnConfig) -> Tensor {
    let scores = q.dot_general(k, dnums_qk());          // bmm1
    let scores = scores.mul_scalar(cfg.query_scale);    // scale AFTER bmm1
    let scores = apply_mask(&scores, &cfg.mask);        // mask BEFORE softmax
    let probs  = ops::softmax(&scores, /*dim=*/ -1);
    probs.dot_general(v, dnums_pv())                    // bmm2
}
```

⚠️ **DESIGN DECISION — flash-friendly emission is the default; any deviation must be justified
against a measurement.** ARTX11 §4's `Architecture` descriptor introduces variants that *do* perturb
this shape — Gemma's query pre-attention scalar and QK-normalization both insert ops between the
projections and bmm1. Whether the rewriter still fires for those is **unknown and must be measured
per architecture**, and it is a real risk that Gemma silently loses the flash path.

---

# 4. Wave A9.3 — KV Access Optimization

## 4.1 Read amplification under GQA

ARTX03 records that GQA shares each KV head across `G = n_heads / n_kv_heads` query heads. The
question ARTX09 adds: **is the shared KV head read once or `G` times?**

```text
Logical:   G query heads each attend against the SAME kv head
Naive:     expand kv to [B, n_heads, S, D] first  → reads G× the bytes
Ideal:     read the kv head once, reuse across G queries in registers/SRAM
```

⚠️ ARTX03's `gqa_attention` performs the expansion (ARTX12 Part B §7 tests it for correctness — the
interleaved-vs-block grouping bug). Correctness is covered; **bandwidth is not**. A materialized
expansion multiplies decode's KV read by `G`, and decode is the bandwidth-bound phase (ARTX08).

For Qwen2.5-0.5B, `G = 14/2 = 7`. A materialized expand means **7× the KV bytes** on the phase that
can least afford it.

⚠️ **DESIGN DECISION — express the GQA expansion as a broadcast, never as a materialized copy, and
verify it stays a broadcast.**

```text
✅  broadcast_in_dim  kv[B, H_kv, S, D] → [B, H_kv, G, S, D]  then reshape the batch dims
    → XLA can keep this a view; the dot reads each kv element once

❌  concatenate / repeat_interleave into [B, H_q, S, D]
    → materializes G× the bytes in HBM
```

⚠️ Whether XLA actually elides the broadcast is **plugin-dependent and unmeasured**. This is the same
verify-don't-assume posture as §3.2, and it belongs in the same probe.

## 4.2 Layout

ARTX05 fixed the layout as `[B, H_kv, S, D]` per layer, and ARTX07 replaced `B` with `max_slots`.
ARTX09 does not change it — but records why it is the right default and where it strains:

```text
kv_k[layer] : [max_slots, H_kv, max_seq, D]
                   │        │       │     └─ contiguous: one head's vector at one position
                   │        │       └─ decode appends here (dynamic_update_slice)
                   │        └─ attention iterates here
                   └─ slot: never crossed within one request
```

✅ **Good:** a single slot's single head is contiguous along `(S, D)`, which is the access order
attention wants; and the slot dimension is outermost, so slots never interleave.

⚠️ **Strains at long context:** the stride between consecutive heads of one slot is
`max_seq × D × dtype_bytes`. For `max_seq = 2048, D = 64`, bf16 that is **256 KB per head**. Each
head therefore starts from a cold cache/TLB region. This is a *known* pattern — GwenLand's CPU engine
recorded the same shape (`KvCache` allocating for `max_context` leaves head regions megabytes apart,
so every head starts L2-cold) as an unfixed layout issue.

⚠️ ⛔ **Do not "fix" this speculatively.** That same repo's optimization history includes an
interleaved-row layout change that measured **−35%** and was reverted, because it broke the linear
streaming the hardware prefetcher was feeding on. The mechanism differs on an accelerator, but the
lesson holds: **layout changes here require a production measurement, not a cache-locality argument.**
Recorded as an open question (§7), not as a work item.

## 4.3 The decode KV read is the real budget

```text
decode KV bytes per token per slot = 2 (K,V) × n_layers × H_kv × S_kv × D × dtype_bytes

Qwen2.5-0.5B, S_kv=2048, bf16:  2 × 24 × 2 × 2048 × 64 × 2  =  25.2 MB per token per slot
```

⚠️ At 8 slots that is **201 MB read per decode iteration** — and on TPU v5e's 819 GB/s that alone is
**~246 µs per token** before any weight read. This is the number ARTX10 §4's quantized KV cache
attacks (halving it at FP8), and it is why §4.1's read-amplification question matters: a materialized
GQA expand would make it **1.4 GB** per iteration.

---

# 5. Wave A9.4 — Prefix Cache

## 5.1 ⛔ Why the obvious design does not fit gljax

Prefix caching reuses the KV of a shared prompt prefix across requests. The standard implementation
shares *physical* KV blocks between sequences — which requires block-table addressing, i.e.
PagedAttention, which **ARTX07 explicitly declined** (its §"Why PagedAttention Is Not Used").

ARTX07's `StaticKvSlab` gives each slot a private, contiguous region: `[slot, :, pos, :]`. Two slots
cannot point at the same KV bytes. **Physical sharing is architecturally unavailable.**

## 5.2 ⭐ The reframing that makes it work

Prefix caching delivers **two** distinct wins, and they are separable:

| Win | Mechanism | Available under a static slab? |
|---|---|---|
| **Memory dedup** — N requests share one physical copy of the prefix KV | block tables / paging | ❌ needs ARTX07's rejected design |
| **Compute reuse** — skip re-running prefill over the shared prefix | cached KV copied into the slot | ✅ **yes** |

⚠️ **DESIGN DECISION — gljax's prefix cache targets compute reuse, and explicitly forgoes memory
dedup.**

The justification is ARTX08's phase analysis: **prefill is compute-bound.** Skipping prefill compute
for a shared prefix is the expensive half of the saving. Memory dedup helps capacity, which ARTX10
§4's quantized KV attacks by a different route that *does* fit the static slab.

```text
Request arrives with prompt = [shared_system_prompt ++ user_turn]
   │
   ├─ Look up shared_system_prompt in the prefix store        (§6)
   │     hit → copy its cached KV into the slot at positions 0..P
   │            set position counter to P
   │            prefill ONLY the user_turn (positions P..)
   │
   └─ miss → ordinary chunked prefill; optionally insert the
             resulting prefix KV into the store
```

⚠️ The copy is `P × H_kv × D × 2 × n_layers × dtype_bytes` of device-to-device movement. For
`P = 512`, Qwen2.5-0.5B, bf16: **6.3 MB**. Against skipping 512 tokens of prefill compute across 24
layers, that is a strongly favourable trade — but it is a *trade*, not a free win, and it must be
measured at small `P` where the copy may dominate.

⚠️ **DESIGN DECISION — a minimum prefix length below which the cache is bypassed.** Recorded as a
tunable (`min_cacheable_prefix`), defaulted from measurement rather than guessed.

## 5.3 Where the prefix KV lives

```rust
// gljax/src/attn/prefix.rs

/// A device-resident KV region OUTSIDE the ARTX07 slab, holding prefix KV
/// for reuse. Not addressable by attention directly — it is a copy source.
pub struct PrefixStore {
    /// Per-layer K/V for cached prefixes, packed back-to-back.
    kv_k: Vec<PjRtBuffer>,   // [prefix_capacity, H_kv, D] per layer
    kv_v: Vec<PjRtBuffer>,
    /// Radix index over token sequences → (offset, length). §6
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

⚠️ **DESIGN DECISION — `PrefixStore` is separate from `StaticKvSlab` and does not participate in
attention.** It is a copy source only. This preserves ARTX07's ownership/storage split intact:
`KvSlotManager` still knows nothing about buffers, and `StaticKvSlab` still knows nothing about
requests or prefixes. A third module with a third concern, rather than a widened second one.

---

# 6. Wave A9.5 — RadixAttention Research

## 6.1 The data structure

SGLang's RadixAttention retains the KV cache for both prompts and generation results in a **radix
tree**, giving efficient prefix search, insertion, and eviction, and it operates **automatically at
runtime** with no manual prompt engineering or static configuration.

The radix tree is the right index for this because prompts share *variable-length* prefixes, and a
radix (compressed) trie matches the longest shared prefix in one descent while sharing storage for
common paths.

## 6.2 Measured hit rates — the number that decides whether to build it

| Workload | Cache hit rate |
|---|---|
| Agents sharing a fixed system prompt + tool definitions | **75–95%** |
| RAG with a fixed document corpus | **40–70%** |
| Multi-turn chat with persistent sessions | **20–40%** (mostly the system prompt) |

Reported throughput is **up to 5×** on workloads with heavy prefix sharing.

⚠️ **That 5× is explicitly an upper bound from synthetic shareable traces, not a deployment
guarantee**, and the source that reports it says so. It should never be quoted as gljax's expected
gain — the same attribution discipline ARTX08 applied to the Tensix fusion numbers.

⭐ **The agent row is why this is worth building.** A 75–95% hit rate on a workload where every
request carries the same long system prompt and tool schema means most prefill compute is redundant.
That is precisely the compute-reuse win §5.2 targets, and it is a workload profile that is becoming
the common case rather than an exotic one.

## 6.3 What gljax adopts and what it does not

| RadixAttention property | gljax |
|---|---|
| Radix tree over token sequences for longest-prefix match | ✅ **adopt** — pure host-side data structure, zero PJRT |
| Automatic, no manual annotation | ✅ adopt |
| LRU eviction over tree nodes | ✅ adopt |
| Physical KV sharing between sequences | ❌ **reject** — needs paging (§5.1) |
| Cache generation results, not only prompts | ⚠️ **defer** — interacts with ARTX11's variable accepted length |

⚠️ The last row is a real interaction, not caution for its own sake: ARTX11's speculative decoding
advances a slot's position by a variable amount per iteration, and rejected tokens leave KV written
but excluded. Inserting generated KV into a prefix store while speculation is active risks caching
KV for tokens that were rejected. **Deferred until ARTX11 lands, with an explicit note in ARTX11's
scheduler.**

```rust
// gljax/src/attn/radix.rs — host-side only, no device types.
pub struct RadixIndex { root: NodeId, nodes: Vec<Node>, lru: LruOrder }

struct Node {
    /// Compressed edge label — the token run on the edge INTO this node.
    tokens: Vec<u32>,
    children: SmallVec<[(u32, NodeId); 4]>,   // first-token → child
    /// Set only on nodes whose path is materialized in the PrefixStore.
    payload: Option<PrefixSlice>,
    last_used: u64,
}

impl RadixIndex {
    /// Descend matching `tokens`; return the deepest node with a payload.
    pub fn longest_match(&self, tokens: &[u32]) -> Option<(NodeId, usize)>;
    pub fn insert(&mut self, tokens: &[u32], slice: PrefixSlice);
    /// Evict least-recently-used payloads until `need` tokens are free.
    pub fn evict(&mut self, need: usize) -> Vec<PrefixSlice>;
}
```

⚠️ **DESIGN DECISION — the radix index is pure host-side Rust with no PJRT types**, mirroring
ARTX07's `KvSlotManager`. It is exhaustively unit-testable without a device, which puts it in
ARTX12's T0 tier.

---

# 7. Module Layout + Wave Plan

```text
gljax/src/attn/
├── mod.rs
├── pattern.rs      §3.4  flash-friendly emission order
├── probe.rs        §3.2  did CudnnFusedMHARewriter fire? did the GQA broadcast stay a view?
├── gqa.rs          §4.1  broadcast-not-materialize expansion
├── prefix.rs       §5.3  PrefixStore, hydrate, insert
└── radix.rs        §6.3  RadixIndex — host-side, no PJRT
```

| Wave | Scope | Gate |
|---|---|---|
| **A9.1** | `pattern.rs` + `probe.rs` — flash-friendly emission; detect whether the rewrite fired | ⭐ Post-optimization HLO contains `__cudnn$fmha` on CUDA — **or** the doc records that it does not |
| **A9.2** | `gqa.rs` — broadcast expansion; measure decode KV read | Measured KV bytes/token match §4.3's formula, **not** `G×` it |
| **A9.3** | `radix.rs` — host-side index | ARTX12 T0: longest-match and eviction property-tested, no device |
| **A9.4** | `prefix.rs` — store, hydrate, insert; wire into ARTX07 admission | TTFT drops on a repeated-prefix workload; **bit-identical output** vs cold prefill |
| **A9.5** | Chunk-size retuning with attention memory as a second constraint (§2.2) | Peak attention memory measured against the §2.2 model |

⚠️ **A9.1's gate is deliberately two-sided.** "The rewrite did not fire on our pattern" is a
legitimate, publishable outcome — it tells gljax its prefill attention memory is bounded only by
chunk size, which is actionable information. Treating a negative result as failure is how a doc ends
up asserting a fusion it never verified.

## 7.1 Open questions — recorded, not answered

1. **Does `CudnnFusedMHARewriter` fire for GQA?** The rewriter matches bmm1→softmax→bmm2; whether a
   broadcast GQA expansion between the projection and bmm1 breaks the match is unknown.
2. **Does it fire for Gemma?** ARTX11 §4's query pre-attention scalar and QK-norm insert ops into the
   matched region (§3.4).
3. **Does XLA:TPU have any attention rewriter?** §3.3 found Splash Attention behind Pallas; whether
   XLA:TPU pattern-matches anything without Pallas is unresolved.
4. **Is the ARTX05 head stride (§4.2) actually costing anything on an accelerator?** ⛔ Requires a
   production measurement. Do not act on the cache-locality argument alone.
5. **What is the right `min_cacheable_prefix`?** (§5.2) — the copy/recompute crossover.

---

# 8. ⭐ An Emerging Series Principle

ARTX09 §3.2 and ARTX10 §5 arrived at the same architecture independently:

```text
1. Emit standard, portable IR — no custom call, no backend branch
2. Let the backend decide whether to apply its optimized path
3. MEASURE whether it did, because a near-miss is silent
4. Keep the unoptimized path always reachable as the floor
```

Quantization calls step 3 a *capability probe*; attention calls it *rewrite verification*. They are
the same mechanism, and both exist because gljax's chosen position — a portable producer of
StableHLO — means the backend's decisions are invisible unless deliberately observed.

⚠️ **Recommendation: name this and hoist it.** A shared `gljax/src/probe/` with one result type, one
disk cache, and one versioning scheme would serve both, instead of two parallel implementations
drifting apart. This should be settled when ARTX10's `quant/probe.rs` is written, not after.

---

# Appendix A — Design Decision Summary

| # | Decision | Rationale | Reversible? |
|---|---|---|---|
| D1 | ARTX09 is additive; ARTX05/ARTX07 keep the slab and bucketing | Avoids duplicating settled specs | N/A |
| D2 | Attention memory is a **prefill** problem | Decode's `S_q = 1` makes the score tensor negligible (§2.1) | N/A — measurement |
| D3 | ARTX07's chunked prefill already removes the **O(S²)** term | Chunking tiles the query axis, one of FlashAttention's two axes (§2.2) | N/A — analysis |
| D4 | Chunk size is now an **attention-memory** parameter too | Latency and memory pull in opposite directions; the trade must be explicit | Trivial |
| D5 | **Emit the natural attention graph; never emit `custom_call`** | `CudnnFusedMHARewriter` pattern-matches it; keeps ARTX08 intact and the module portable | Hard |
| D6 | Verify the rewrite fired | A near-miss silently falls back with no error | Trivial |
| D7 | Flash-friendly op order is the default (scale after bmm1, mask before softmax) | Mathematically-equivalent reorderings may defeat the pattern match | Trivial |
| D8 | **TPU flash gap is stated, not papered over** | Splash Attention is Pallas; reaching it means authoring a kernel ARTX08 declined | Medium |
| D9 | GQA expansion is a **broadcast**, never a materialized copy | A materialized expand multiplies decode KV reads by `G` (7× on Qwen2.5-0.5B) | Trivial |
| D10 | ARTX05's KV layout unchanged | Contiguous along `(S, D)` is the access order attention wants | Medium |
| D11 | ⛔ Do not "fix" the head-stride locality speculatively | A structurally similar layout change measured −35% elsewhere in this repo | N/A |
| D12 | Prefix cache targets **compute reuse**, forgoing memory dedup | Physical sharing needs paging, which ARTX07 declined; prefill is the compute-bound phase | Hard |
| D13 | `PrefixStore` is a copy source, separate from `StaticKvSlab` | Preserves ARTX07's ownership/storage split rather than widening it | Medium |
| D14 | `min_cacheable_prefix` gate | At small `P` the KV copy can outweigh the skipped prefill | Trivial |
| D15 | Adopt the radix tree; reject physical KV sharing | The index is free (host-side); the sharing is not (§6.3) | Medium |
| D16 | `RadixIndex` is pure host-side Rust, no PJRT | Exhaustively testable without a device (ARTX12 T0) | Trivial |
| D17 | Defer caching *generated* KV | Interacts with ARTX11's variable accepted length and rejected-token KV | Medium |
| D18 | A9.1's gate accepts a negative result | "The rewrite does not fire" is actionable information, not failure | N/A |
| D19 | Hoist the probe pattern shared with ARTX10 §5 | Two subsystems converged on it independently (§8) | Trivial |

---

# Sources

- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/pdf/2307.08691) — tiling, online softmax, recomputation, sequence-dimension parallelism.
- [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://tridao.me/publications/flash3/flash3.pdf) and [the PyTorch write-up](https://pytorch.org/blog/flashattention-3/) — warp specialization (producer/consumer), interleaved matmul/softmax, FP8; 1.5–2.0× over FA-2, 740 TFLOPS ≈ 75% of H100 peak, ~1.2 PFLOPS at FP8 with 2.6× smaller error.
- [When XLA Isn't Enough: From Pallas to VLIW with Splash Attention on TPU](https://patricktoulme.substack.com/p/when-xla-isnt-enough-from-pallas) — ⭐ *"XLA can fuse ops and schedule instructions, but it can't rewrite your algorithm — it can't infer the streaming/online-softmax reformulation"*; Splash Attention's double-buffered DMA, 2-D Pallas grid skipping fully-masked causal tiles, VMEM residency.
- [JAX Splash Attention kernel](https://github.com/jax-ml/jax/blob/main/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py) and [JAX TPU flash attention](https://github.com/jax-ml/jax/blob/main/jax/experimental/pallas/ops/tpu/flash_attention.py) — the TPU implementations, in Pallas.
- [All XLA Options/Flags](https://guides.lw1.at/all-xla-options/) — `--xla_gpu_enable_cudnn_fmha`, `CudnnFusedMHARewriter`, the `__cudnn$fmha` custom-call target.
- [Accelerating Transformers with NVIDIA cuDNN 9](https://developer.nvidia.com/blog/accelerating-transformers-with-nvidia-cudnn-9/) — XLA's path to cuDNN SDPA, via the JAX SDPA API or by compiler lowering.
- [FlashInfer: Efficient and Customizable Attention Engine for LLM Serving](https://arxiv.org/pdf/2501.01005) and [the NVIDIA overview](https://developer.nvidia.com/blog/run-high-performance-llm-inference-kernels-from-nvidia-using-flashinfer/) — block-sparse composable KV formats, JIT-compiled attention templates, load-balanced scheduling; powers SGLang, vLLM, TensorRT-LLM, TGI, MLC-LLM.
- [Fast and Expressive LLM Inference with RadixAttention and SGLang](https://www.lmsys.org/blog/2024-01-17-sglang/) and [SGLang Prefix Caching](https://sgl-project-sglang-93.mintlify.app/concepts/prefix-caching) — radix tree over prompt and generation KV, automatic runtime operation, LRU eviction.
- [Benchmarking SGLang's RadixAttention for multi-turn chat](https://n4n.ai/blog/benchmarking-sglangs-radixattention-for-multi-turn-chat/) — hit rates: 40–70% RAG, 20–40% multi-turn, 75–95% agent workloads; the 5× figure noted as a synthetic upper bound.
- [Prefill Is Compute-Bound. Decode Is Memory-Bound.](https://towardsdatascience.com/prefill-is-compute-bound-decode-is-memory-bound-why-your-gpu-shouldnt-do-both/) — prefill's full `S×S` matrix per head per layer; decode's `1×(S+t)` dot reading the whole KV cache.

**Repo-internal:** `ARTX03` (`gqa_attention`, GQA repeat factor); `ARTX05 §1–§3` (KV layout,
`dynamic_update_slice`, buffer donation, bucketing); `ARTX07` (StaticKvSlab, chunked prefill,
PagedAttention rejection, ownership/storage split); `ARTX08` (no-kernel rule, `custom_call`
rejection, prefill/decode phase analysis); `ARTX10 §4–§5` (quantized KV, capability probe);
`ARTX11 §4` (`Architecture` descriptor, Gemma query scale and QK-norm); `ARTX12 Part B §7` (GQA
grouping correctness tests); `gl-agent-skills/cpu-skills/rejected-optimizations.md` entry 2
(interleaved-row layout, −35%, cited as *method* only — different hardware tier per ARTX08's Scope).
