# ARTX7 — Continuous Batching and Dynamic Sequence Multiplexing

**Series:** ARTX7
**Status:** Draft — research-grounded
**Depends On:** ARTX5 — Static KV Cache + Bucketing Strategy
**Related:** ARTX6 — Multi-Device Tensor Parallel + MoE Expert Sharding
**Next:** [ARTX08 — Matrix Compute Architecture](ARTX08-matrix-compute-architecture.md)
**Research grounded:** 2026-07-27 (sources listed at the end of the Research Summary)

---

# Overview

ARTX7 introduces a host-driven inference runtime for gljax that serves many concurrent requests while preserving the static-shape execution model established in ARTX4/ARTX5.

This is deliberately **not** a vLLM clone, and it is **not** "PagedAttention for PJRT." Both of those systems solve the multi-request serving problem by making the KV cache dynamically addressable. ARTX7 solves the same problem from the other direction:

* Batch-dimension bucketing (2D: slot count × sequence length)
* Static, pre-allocated KV cache slabs (ARTX5's `[B, H_kv, S, D]` buffer, with `B` reinterpreted as a slot dimension)
* Continuous batching at the request-admission level
* Iteration-level scheduling at the execution level
* Chunked prefill so long prompts never block ready decodes
* Dynamic sequence multiplexing — slots are reused the moment a request finishes, without waiting for the whole batch to drain

All scheduling decisions happen in Rust, on the host. PJRT remains a pure execution backend: it compiles and runs fixed-shape programs and knows nothing about requests, queues, or scheduling policy.

---

# Motivation

Single-request execution (ARTX4/ARTX5) leaves the accelerator idle between requests and wastes the parallelism the hardware has to offer. The serving-systems literature converges on the same fix from several independent directions — Orca's iteration-level scheduling, the continuous-batching implementations in HuggingFace TGI/vLLM, DeepSpeed-FastGen's Dynamic SplitFuse — all interleave many requests at sub-batch granularity instead of running one request (or one static batch) to completion before starting the next.

The dominant implementation of that idea, vLLM's PagedAttention, gets its memory efficiency from a KV cache that is **not** a fixed tensor: it's a set of non-contiguous physical blocks addressed through a per-request block table, allocated and freed like OS virtual memory pages. That is a natural fit for a CUDA/HIP kernel that can do arbitrary pointer-chasing gathers. It is a poor fit for a StableHLO program compiled once by XLA and executed unchanged thousands of times — dynamic, per-token block-table indirection is exactly the kind of shape-dependent, data-dependent control flow that forces recompilation or falls back to slow dynamic-shape code paths.

ARTX7 chooses a different trade:

> Compile once, execute many.

Requests adapt to compiled artifacts (by bucket selection and padding). Compiled artifacts never adapt to requests. This is the same bet ARTX5 already made for sequence length; ARTX7 extends it to the batch/slot dimension and to request admission over time.

---

# Design Principles

## 1. Host-Driven Scheduling

Scheduling lives entirely in Rust. PJRT performs execution only.

---

## 2. Static-Shape Execution

All execution uses precompiled bucket shapes. No runtime graph modification, no shape-polymorphic kernels.

---

## 3. Separation of Ownership and Storage

`KvSlotManager` and `StaticKVSlab` are **two independent modules** that share exactly one type: `SlotId`.

| | KvSlotManager (A7.1) | StaticKVSlab (A7.3) |
|---|---|---|
| **Concern** | Ownership + lifecycle | Buffer layout + addressing |
| **Knows** | Which slots are free/occupied/reserved, which request owns which slot | PJRT buffer shapes, byte offsets, slab dimensions, how to index K/V per layer per step |
| **Does NOT know** | Buffer pointers, tensor shapes, PJRT, memory layout, head_dim, n_layers | Request IDs, scheduling, why a slot was allocated, lifecycle transitions |
| **Input** | Request arrives / finishes | SlotId + position + new K/V data |
| **Output** | SlotId (opaque integer) | Updated PJRT buffer (in-place via donation) |

```text
Scheduler                KvSlotManager           StaticKVSlab
   │                          │                       │
   │── allocate(req_id) ─────►│                       │
   │◄── Ok(SlotId=2) ────────│                       │
   │                          │                       │
   │── (pass SlotId=2 to executor) ──────────────────►│
   │                          │       write_kv(slot=2, layer, pos, k, v)
   │                          │                       │
   │── free(SlotId=2) ───────►│                       │
   │                          │── (slot 2 now free)   │
   │                          │                       │
   │                          │       (slab does NOT know slot 2 was freed;
   │                          │        stale data is harmless — attention
   │                          │        mask excludes it)
```

This separation allows replacing `StaticKVSlab` (e.g., with a `PagedKVSlab` in ARTX16+) without touching `KvSlotManager` or the scheduler.

---

## 4. Zero Scheduler Knowledge of PJRT

The scheduler does not know: buffers, tensors, memory layout, StableHLO.

The scheduler only knows: `Request`, `Batch`, `Slot`, `Bucket`.

---

## 5. Compile Once, Execute Many

Compiled artifacts are cached by shape. No recompilation during serving as long as traffic stays within the configured buckets.

---

## 6. Work-Conserving Scheduling

If executable work exists — a decode step or a prefill chunk — the accelerator stays busy. The scheduler never idles the device while any request has runnable work, which is the same principle Sarathi-Serve calls "stall-free" scheduling (see Research Summary).

---

# Research Summary

## Continuous Batching

"Continuous batching" (also called dynamic batching, or rolling batching) is the industry term — popularized by Anyscale's 2023 benchmark writeup — for admitting new requests into an in-flight batch as soon as a slot frees up, instead of waiting for the entire batch to finish before starting the next one. Anyscale measured up to **23x throughput improvement** over static/naive batching on real workloads, with a lower p50 latency, because the accelerator is never left idle waiting for the slowest sequence in a batch to finish.

ARTX7's `KvSlotManager` + `BatchFormer` are the direct mechanism for this: a slot is freed and reallocated to a new request within the same scheduler tick a finished request retires (see [Dynamic Sequence Multiplexing](#dynamic-sequence-multiplexing) below), with no batch-level barrier.

## Iteration-Level Scheduling

Continuous batching's academic origin is Orca (Yu et al., OSDI 2022), which introduced two techniques together:

1. **Iteration-level scheduling** — the scheduler is invoked once per iteration (i.e., once per generated token across the whole batch), not once per request. Every iteration it decides which requests run next.
2. **Selective batching** — because requests in the same iteration can be at different sequence positions, only the operations that tolerate it (QKV/FFN projections, layernorm) are executed as a single batched matmul; attention itself is computed per-request with its own KV context, then the results are re-batched for the next matmul.

Orca reported **36.9x throughput** over NVIDIA FasterTransformer at equivalent latency on GPT-3 175B. ARTX7's `Scheduler::schedule_decode()` / `schedule_prefill_chunks()` tick, invoked once per loop iteration in [Scheduler Loop](#scheduler-loop), is gljax's iteration-level scheduling point. Selective batching itself is naturally satisfied by ARTX5's per-slot KV addressing: the batched projections run over the full `[max_slots, ...]` tensor, while attention reads are scoped to one slot's slice.

## Why PagedAttention Is Not Used

vLLM's PagedAttention (Kwon et al., SOSP 2023) gets its memory efficiency by making the KV cache **non-contiguous**: each request's cache lives in fixed-size physical blocks (16 tokens each, by default) addressed through a per-request logical→physical block table, the same trick as OS virtual-memory paging. This eliminates the internal fragmentation of pre-allocating each request's cache for its worst-case length.

This is a poor fit for ARTX7, but not because it is *impossible* on static-shape backends — Google's own **Ragged Paged Attention** kernel (2026) proves the opposite: it is a hand-written Pallas/Mosaic TPU kernel that implements ragged, block-table-addressed attention and measures a 5x speedup over padded multi-query attention. The real reason ARTX7 skips it is a **scope and portability trade specific to gljax**, not a hard technical wall:

* Block-table addressing is data-dependent gather/scatter. Expressing it in portable StableHLO (no custom kernel) reintroduces the dynamic-shape, per-token indirection that forces XLA into slow dynamic-shape code paths or recompilation — the exact problem ARTX5's static slab + bucketing was built to avoid.
* Expressing it *efficiently* requires a custom kernel (Pallas/Mosaic on TPU, a CUDA/cuDNN kernel on GPU) per backend. ARTX1 committed gljax to being a pure-Rust, plugin-only PJRT client — compiling standard StableHLO through whatever PJRT plugin is loaded (TPU v5e, A100, H100, CPU) — specifically to avoid owning a per-backend custom-kernel toolchain. Taking on PagedAttention now means taking on that toolchain now.

ARTX7 pays a different, bounded cost instead:

```text
Bucketing
+
Padding
+
Static KV Slabs
```

Because `StaticKVSlab` is isolated behind the same interface a `PagedKVSlab` would implement (Design Principle #3), this is a deferral, not a rejection — ARTX16+ can adopt a custom paged/ragged kernel later, on one backend at a time, without touching the scheduler. See [Future Work](#future-work--artx9).

## Static Shape Constraints

XLA/PJRT compiles a StableHLO program for one fixed set of tensor shapes; any shape it hasn't seen either recompiles or falls back to a slower path. vLLM's own TPU backend documents this cost directly: a first-run warmup that precompiles every configured bucket shape takes on the order of 20–30 minutes, dropping to ~5 minutes once the compiled XLA graphs are cached on disk. This is the same cost ARTX5 already pays per sequence-length bucket (10 compiled `.pjrt` artifacts for 5 buckets × {prefill, decode}); ARTX7 multiplies it by the number of batch/slot buckets it configures.

The one place a *runtime* value is allowed inside an otherwise static program is `stablehlo.dynamic_update_slice`/`slice` with scalar dynamic start-indices (ARTX5 §2) — the position/slot index changes every call, but the tensor shapes it operates on never do. Bucketed execution and chunked prefill (below) are both just applications of this one primitive at a coarser granularity.

## Bucketed Execution

vLLM's TPU backend pads every request to the nearest configured bucket before compiling/dispatching, and exposes exactly the tradeoff ARTX7 inherits: exponential padding (powers of two — few buckets, more padding waste) versus linear/gap-based padding (more buckets, less waste, more compiled artifacts and warmup time to manage). ARTX5 already picked linear sequence-length buckets (128/256/512/1024/2048) for this reason.

ARTX7 extends bucketing to a second axis — slot count — because a batch of 1 active request and a batch of 8 active requests are different compiled programs:

```text
(1,128)   (2,128)   (4,128)   (8,128)
(1,512)   (2,512)   (4,512)   (8,512)
```

The same tradeoff applies on both axes: more buckets means less padding waste per request but a larger [Compile Cache](#compile-cache) and longer warmup. This is a tuning knob (`bucket.rs`), not a fixed constant.

## Chunked Prefill

A naive scheduler that always runs a full prefill before any decode step will stall every in-flight decode for the duration of a long prompt. Sarathi-Serve (Agrawal et al., OSDI 2024) fixes this with **chunked prefill**: a long prompt is split into fixed-size chunks (e.g., 512 tokens) processed over several iterations, interleaved with ongoing decodes rather than blocking them — a "stall-free" schedule. Sarathi-Serve measured 2.6x (Mistral-7B, 1 GPU) to 6.9x (Falcon-180B, 8 GPUs) throughput improvement over Orca and vLLM within the same latency SLO, and chunked prefill has since become the default scheduling strategy in both vLLM and SGLang.

DeepSpeed-FastGen's Dynamic SplitFuse (2024) arrived at a closely related design independently: it composes each forward pass out of a fixed *token budget*, filled by combining prefill chunks from long prompts with decode tokens from other requests, rather than scheduling prefill and decode as separate phases at all. Two independently-developed systems converging on "never run an unbounded-length prefill in one shot" is a strong signal this is a load-bearing technique, not a one-off trick.

ARTX7's chunked-prefill policy (`policy.rs`) follows Sarathi-Serve's ordering directly:

```text
Decode First
↓
Remaining Budget
↓
Prefill Chunk
```

```text
Prompt = 4096, Chunk Size = 512  →  8 prefill executions,
each one scheduled only after decode requests for that iteration are satisfied.
```

## Sources

- [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu) — Yu et al., OSDI 2022. Iteration-level scheduling + selective batching, 36.9x vs FasterTransformer.
- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/pdf/2309.06180) — Kwon et al., SOSP 2023. Block-table-addressed, non-contiguous KV cache.
- [How continuous batching enables 23x throughput in LLM inference while reducing p50 latency](https://bestofai.com/article/how-continuous-batching-enables-23x-throughput-in-llm-inference-while-reducing-p50-latency-anyscale) — Anyscale, 2023.
- [Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve](https://arxiv.org/abs/2403.02310) — Agrawal et al., OSDI 2024. Chunked prefill + stall-free batching.
- [DeepSpeed-FastGen: High-throughput Text Generation for LLMs via MII and DeepSpeed-Inference](https://arxiv.org/pdf/2401.08671) — Dynamic SplitFuse.
- [Ragged Paged Attention: A High-Performance and Flexible LLM Inference Kernel for TPU](https://arxiv.org/abs/2604.15464) — 2026. Proof that paged/ragged attention is achievable on TPU via a custom Pallas/Mosaic kernel (5x vs padded MQPA) — the option ARTX7 explicitly defers, not rules out.
- [vLLM TPU Optimization Tips](https://docs.vllm.ai/en/v0.11.0/configuration/tpu.html) — bucket vs exponential padding, warmup/recompilation costs on XLA/TPU.
- [Google Cloud: JetStream](https://docs.cloud.google.com/kubernetes-engine/docs/tutorials/serve-gemma-tpu-jetstream) — continuous batching + KV cache optimization already in production on JAX/TPU serving, evidence the approach transfers to gljax's target backends.

---

# Goals

* Multi-request execution
* Continuous batching
* Static execution
* High utilization
* PJRT compatibility

---

# Non Goals

* PagedAttention
* Prefix cache
* Speculative decoding
* PD disaggregation
* Remote KV
* Multi-host scheduling
* KV compression
* Async runtime

These belong to ARTX16+ (or, for async runtime, are ruled out entirely — the scheduler loop in ARTX7 is single-threaded and synchronous by design; see [Scheduler Loop](#scheduler-loop)).

---

# Runtime Architecture

```text
RequestQueue
        │
        ▼
Scheduler
        │
        ▼
BatchFormer
        │
        ▼
KvSlotManager
        │
        ▼
StaticKVSlab
        │
        ▼
Executor
        │
        ▼
PJRT
```

---

# Wave A7.1 — Request Runtime

## Modules

```text
request.rs
queue.rs
batch.rs
batch_slot.rs
kv_slot_manager.rs
```

## Responsibilities

```text
Request           owns request metadata
  ↓
RequestQueue       owns pending requests
  ↓
Batch              owns active requests
  ↓
BatchSlot          logical slot id
  ↓
KvSlotManager      ownership, allocation, reservation, lifecycle
```

NO PJRT. NO Tensor. NO Buffer.

Orca's design separates request management from model execution entirely, keeping the scheduler lightweight and backend-independent. A7.1 is that separation: every module here can be unit-tested without a PJRT client at all.

Struct definitions and the `KvSlotManager` API live in [Request Lifecycle](#request-lifecycle) and [Batch Lifecycle](#batch-lifecycle) below.

---

# Wave A7.2 — Host Scheduler

## Modules

```text
scheduler.rs
batch_former.rs
policy.rs
```

## Responsibilities

* Iteration scheduling
* Continuous batching
* Chunked prefill scheduling
* Batch selection / priority
* Admission control

The scheduler produces an `ExecutionBatch` — never a tensor. Orca demonstrated that scheduling decisions must be made every iteration, not once per batch at admission time; vLLM adopted the same iteration-level model for continuous batching. See [Scheduler Loop](#scheduler-loop) for the concrete loop and [Pseudocode](#pseudocode) for `BatchFormer`.

---

# Wave A7.3 — Static Execution

## Modules

```text
static_kv_slab.rs
bucket.rs
compile_cache.rs
executor.rs
```

## Responsibilities

* Static KV slab storage (physical layout + addressing)
* Batch bucket selection
* Compiled-program caching
* PJRT execution
* Dynamic sequence multiplexing (slot reuse across requests)

PagedAttention is well-suited to dynamic KV allocation, but it depends on data-dependent block tables that fight XLA's static-shape compilation model. For a PJRT/StableHLO backend, bucketing + padding is the better-aligned trade (full rationale in [Why PagedAttention Is Not Used](#why-pagedattention-is-not-used)). Full mechanics live in [Static KV Slab](#static-kv-slab) below.

---

# Request Lifecycle

```text
Pending
  ↓
Queued
  ↓
Scheduled
  ↓
Prefill
  ↓
Decode
  ↓
Finished  (or Cancelled, from any state)
```

## Request

```rust
pub struct Request {
    id: RequestId,
    prompt_tokens: Vec<Token>,
    generated_tokens: Vec<Token>,
    max_tokens: usize,
    state: RequestState,
    slot: Option<SlotId>,
    bucket: BucketId,
}
```

## RequestState

```rust
pub enum RequestState {
    Pending,
    Queued,
    Scheduled,
    Prefill,
    Decode,
    Finished,
    Cancelled,
}
```

## RequestQueue

```rust
pub struct RequestQueue {
    pending: VecDeque<Request>,
}
```

Methods: `push()`, `pop()`, `peek()`, `cancel()`, `len()`, `is_empty()`.

---

# Batch Lifecycle

```text
Created
  ↓
Running
  ↓
Refilled   ◄──┐   (a finished request's slot is reused by a new one —
  ↓           │    see Dynamic Sequence Multiplexing)
Running ──────┘
  ↓
Shrunk     (a request finishes and no replacement is admitted yet)
  ↓
Destroyed  (all slots empty, batch retired)
```

A batch cycles through `Running ⇄ Refilled ⇄ Shrunk` for as long as any slot is occupied — it only reaches `Destroyed` when every slot has drained and the scheduler chooses not to keep the (now-empty) batch/bucket resident. This is what makes continuous batching *continuous*: `Refilled` and `Shrunk` are not terminal, and neither one waits for the others.

## Batch

```rust
pub struct Batch {
    requests: Vec<RequestId>,
    slots: Vec<SlotId>,
    bucket: BucketId,
}
```

## BatchSlot

```rust
pub struct BatchSlot {
    id: SlotId,
    request: Option<RequestId>,
}
```

## KV Slot Lifecycle

```text
Free ──allocate(req)──► Occupied ("Allocated") ──free()──► Free ("Released")
  ▲                        │
  │                     (also reachable via)
  │
Free ──reserve()──► Reserved ──release()──► Free
                       │
                   allocate(req)
                       │
                       ▼
                   Occupied
```

**Concern:** logical ownership and lifecycle only — no PJRT, no tensor shapes, no byte offsets. That is entirely [Static KV Slab](#static-kv-slab)'s job.

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotState {
    Free,
    Occupied { request: RequestId },
    Reserved,
}

pub struct KvSlotManager {
    /// Fixed-size array: index = SlotId, value = state.
    /// Length = max_concurrent_requests (configured at init).
    slots: Vec<SlotState>,
}

impl KvSlotManager {
    pub fn new(max_slots: usize) -> Self;

    /// Find a Free slot, transition it to Occupied, return its SlotId.
    /// Returns None if all slots are occupied/reserved.
    pub fn allocate(&mut self, request: RequestId) -> Option<SlotId>;

    /// Transition an Occupied slot back to Free. Panics if not Occupied.
    pub fn free(&mut self, slot: SlotId);

    /// Pre-claim a Free slot for an upcoming prefill.
    pub fn reserve(&mut self) -> Option<SlotId>;

    /// Transition a Reserved slot back to Free (prefill was cancelled).
    pub fn release(&mut self, slot: SlotId);

    /// Which request owns a slot (None if Free/Reserved).
    pub fn lookup(&self, slot: SlotId) -> Option<RequestId>;

    /// Number of Free slots available.
    pub fn available(&self) -> usize;

    /// Iterate all Occupied slots (for batch formation).
    pub fn occupied(&self) -> impl Iterator<Item = (SlotId, RequestId)> + '_;
}
```

```text
Slot0 → Occupied(RequestA)      Slot0 → Occupied(RequestA)
Slot1 → Occupied(RequestB)      Slot1 → Occupied(RequestB)
Slot2 → Free                →   Slot2 → Occupied(RequestC)   ← was Free
Slot3 → Reserved                Slot3 → Reserved

available() = 1                 allocate(RequestC) → Some(SlotId(2))
                                 available() = 0
                                 allocate(RequestD) → None   ← no slots left
```

> **Key invariant:** `KvSlotManager` never touches PJRT. It produces `SlotId` values the executor passes to `StaticKVSlab`. It does not need to know what happens to the KV data — that is entirely A7.3's concern.

---

# Scheduler Loop

```text
Collect
  ↓
Retire
  ↓
Form Batch
  ↓
Schedule Decode
  ↓
Schedule Prefill
  ↓
Execute
```

```rust
loop {
    scheduler.collect_new_requests();
    scheduler.retire_finished();
    scheduler.form_batch();
    scheduler.schedule_decode();
    scheduler.schedule_prefill_chunks();
    executor.execute();
}
```

This loop is deliberately synchronous and single-threaded (see [Non Goals](#non-goals) — no async runtime). Orca and Sarathi-Serve both make their scheduling decision once per iteration of this same shape; the loop body above *is* ARTX7's iteration-level scheduling point, and `schedule_decode()` running before `schedule_prefill_chunks()` every tick *is* the "decode first, prefill fills the remaining budget" chunked-prefill policy from the Research Summary.

---

# Static KV Slab

**Concern:** physical KV buffer layout and addressing.

**Knows:** PJRT buffer shapes, how to index into a slab given `(SlotId, layer, position)`, byte offsets, head dimensions, bucket sizes.

**Does NOT know:** request IDs, request lifecycle, scheduling policy, why a slot was allocated or freed.

> The slab is a **dumb storage array.** It receives a `SlotId` (opaque integer from `KvSlotManager`) and a position, and writes/reads KV data at the correct offset. It never asks "who owns this slot?" — that is entirely A7.1's concern.

## PJRT Buffer Shape

The slab pre-allocates **one contiguous PJRT buffer per layer, per K/V** holding KV data for **all slots simultaneously** — this extends ARTX5's single-request `[B, H_kv, S, D]` buffer by replacing `B=1` with `max_slots`:

```text
kv_k[layer]: [max_slots, n_kv_heads, max_seq_len, head_dim]    (bf16)
kv_v[layer]: [max_slots, n_kv_heads, max_seq_len, head_dim]    (bf16)
```

Example for Qwen2-0.5B, 8 slots, bucket=2048:

```text
kv_k[layer]: [8, 2, 2048, 64]  = 8 × 2 × 2048 × 64 × 2 bytes = 4MB per layer
kv_v[layer]: [8, 2, 2048, 64]  = 4MB per layer
total_kv    = 24 layers × 2 × 4MB = 192MB
```

## Addressing

Given a `SlotId`, `layer_idx`, and decode `position`, using the same `dynamic_update_slice`/`slice` primitive as ARTX5 (§2), with `slot_id` as an added static-shape, dynamic-index dimension:

```text
Write new K at decode step:
  dynamic_update_slice(
    kv_k[layer_idx],           // [max_slots, H_kv, max_seq, D]
    new_k,                     // [1, H_kv, 1, D]
    start_indices = [slot_id, 0, position, 0]
  )

Read full K for attention:
  slice(
    kv_k[layer_idx],           // [max_slots, H_kv, max_seq, D]
    start = [slot_id, 0, 0, 0],
    limit = [slot_id+1, H_kv, max_seq, D]
  )                            // → [1, H_kv, max_seq, D]
```

Padding positions are handled by the attention mask, same as ARTX5 §3.

## Slab State + API

```rust
pub struct StaticKvSlab {
    kv_k: Vec<PjRtBuffer>,   // kv_k[layer] : [max_slots, n_kv_heads, max_seq_len, head_dim]
    kv_v: Vec<PjRtBuffer>,
    max_slots: usize,
    n_kv_heads: usize,
    max_seq_len: usize,      // = bucket size
    head_dim: usize,
    n_layers: usize,
}

impl StaticKvSlab {
    /// Allocate slab buffers on device (all zeros). Called once at runtime init.
    pub fn new(
        client: &PjRtClient,
        max_slots: usize,
        n_kv_heads: usize,
        max_seq_len: usize,
        head_dim: usize,
        n_layers: usize,
    ) -> Result<Self, SlabError>;

    /// K/V buffer references for a layer (passed to PJRT execute as donated inputs).
    pub fn layer_buffers(&self, layer: usize) -> (&PjRtBuffer, &PjRtBuffer);

    /// Mutable K/V buffer references (receiving donated outputs from PJRT).
    pub fn layer_buffers_mut(&mut self, layer: usize) -> (&mut PjRtBuffer, &mut PjRtBuffer);

    /// Zero-fill one slot's KV region across all layers. Optional: stale data
    /// is harmless (attention mask excludes freed positions) but zeroing
    /// prevents information leakage between requests that reuse a slot.
    pub fn clear_slot(&mut self, slot: SlotId) -> Result<(), SlabError>;

    /// Zero-fill all slots across all layers (shutdown / full reset).
    pub fn clear_all(&mut self) -> Result<(), SlabError>;

    pub fn max_slots(&self) -> usize;
}
```

> **Key invariant:** `StaticKvSlab` never checks whether a slot is logically free or occupied. Correctness depends entirely on the scheduler (via `KvSlotManager`) only issuing writes to slots that are logically allocated.

## Ownership vs Storage — How They Interact

```text
1. Request arrives
   └─► KvSlotManager.allocate(req_id) → SlotId(2)          [A7.1: Free → Occupied]

2. Prefill execution
   └─► Executor passes SlotId(2) to compiled prefill function
       └─► StaticKvSlab: PJRT writes KV data at slot_id=2   [A7.3: dynamic_update_slice, [2,:,0..prompt_len,:]]

3. Decode iterations
   └─► Executor passes SlotId(2) + position to compiled decode function
       └─► StaticKvSlab writes new K/V at [2,:,pos,:]        [A7.3: donated in-place update]

4. Request finishes
   └─► KvSlotManager.free(SlotId(2))                        [A7.1: Occupied → Free]
   └─► (optional) StaticKvSlab.clear_slot(SlotId(2))         [A7.3: zero-fill]

5. New request reuses slot
   └─► KvSlotManager.allocate(new_req_id) → SlotId(2) again  [A7.1: Free → Occupied]
       (A7.3: old KV data is overwritten by new prefill — no ambiguity)
```

This separation allows future replacement of the storage backend without touching the scheduler or `KvSlotManager`:

```text
StaticKvSlab         (ARTX7 — contiguous per-slot slab)
     ↓
PagedKvSlab          (ARTX16+ — block-table addressing, e.g. a Ragged Paged Attention kernel)
     ↓
CompressedKvSlab     (ARTX16+ — quantized KV storage)
     ↓
RemoteKvSlab         (ARTX16+ — disaggregated KV over network)
```

All backends implement the same `layer_buffers()` / `clear_slot()` interface.

---

# Dynamic Sequence Multiplexing

```text
Slot0 → RequestA
Slot1 → RequestB
Slot2 → RequestC

RequestB finishes
  ↓
Slot1 reused
  ↓
RequestD
```

Batch execution continues — no full batch restart, no waiting for RequestA/RequestC to also finish. This is the mechanism, at the slot level, behind the "23x throughput" continuous-batching result cited in the Research Summary.

---

# Compile Cache

Key:

```rust
(batch_size,       // slot-count bucket
 sequence_bucket,  // ARTX5 sequence-length bucket
 dtype,
 device)
```

Value:

```rust
CompiledProgram
```

No recompilation occurs while traffic stays within the configured `(batch_size, sequence_bucket)` grid (see [Bucketed Execution](#bucketed-execution)) — this is the direct payoff of Design Principle #5.

---

# Pseudocode

Reference appendix — the same algorithms shown in prose above, collected for quick scanning.

## BatchFormer

```rust
for request in queue {
    bucket = choose_bucket(request);
    slot = kv.allocate(request);
    batch.push(slot);
}
```

## Slot allocate/free (KvSlotManager)

```rust
let slot = kv_slot_manager.allocate(request.id)
    .ok_or(SchedulerError::NoFreeSlots)?;
request.slot = Some(slot);

// ... on completion:
kv_slot_manager.free(slot);
static_kv_slab.clear_slot(slot)?;   // optional, prevents leakage into next occupant
request.slot = None;
```

## KV addressing (StaticKVSlab)

```rust
// decode step, one token
dynamic_update_slice(kv_k[layer], new_k, [slot_id, 0, position, 0]);
dynamic_update_slice(kv_v[layer], new_v, [slot_id, 0, position, 0]);

// attention read
let k = slice(kv_k[layer], [slot_id, 0, 0, 0], [slot_id+1, h_kv, max_seq, d]);
let v = slice(kv_v[layer], [slot_id, 0, 0, 0], [slot_id+1, h_kv, max_seq, d]);
```

## Chunked prefill split

```rust
let chunks = prompt_tokens.chunks(chunk_size);   // e.g. 512
for chunk in chunks {
    scheduler.schedule_decode();        // decode requests always go first this tick
    scheduler.schedule_prefill_chunk(chunk, remaining_budget);
    executor.execute();
}
```

---

# Folder Layout

```text
runtime/
    request.rs
    queue.rs
    batch.rs
    batch_slot.rs

    kv_slot_manager.rs

    scheduler.rs
    batch_former.rs
    policy.rs

    static_kv_slab.rs
    bucket.rs
    compile_cache.rs
    executor.rs

    session.rs
```

---

# Benchmarks

**Status: not yet measured.** ARTX7 is a design document — Waves A7.1–A7.3 are not implemented yet, so there are no numbers to report here. This section defines the methodology and metrics that Wave A7 must report against once it lands, following this repo's rule that every number in an architecture doc is a measurement, not a projection (see `.agents/skills/read-architecture-first/SKILL.md`).

## Metrics

| Metric | What it validates |
|---|---|
| Tokens/sec (aggregate, across all active slots) | Continuous batching is actually improving utilization vs. ARTX5 single-request baseline |
| Time-to-first-token (TTFT), p50/p99 | Chunked prefill is not starving decode, and vice versa |
| Time-per-output-token (TPOT), p50/p99 | Iteration-level scheduling overhead is not eating the throughput win |
| Padding waste % (tokens computed vs. tokens real) | Whether the chosen `(batch_size, sequence_bucket)` grid is too coarse |
| Compile cache hit rate | Whether real traffic actually stays inside the configured bucket grid |
| Slot utilization (occupied / max_slots over time) | Whether `max_slots` is sized correctly for the workload |

## Planned workloads

1. **Concurrency sweep** — fixed prompt/output length, vary concurrent requests from 1 to `max_slots`, to reproduce (on gljax's own backend/hardware) the shape of Anyscale's and Orca's throughput-vs-concurrency curves cited above.
2. **Mixed prefill/decode** — a steady stream of short decode-heavy requests plus periodic long prompts, to measure whether chunked prefill holds decode TPOT stable (this is the exact scenario Sarathi-Serve's "stall-free" claim addresses).
3. **Bucket-grid stress** — request lengths deliberately chosen to fall just above/below each bucket boundary, to measure real padding waste against the estimate in [Bucketed Execution](#bucketed-execution).

## Baseline

The comparison baseline is gljax's own ARTX5 single-request runtime (`B=1`, no scheduler), not vLLM or Orca — cross-framework/cross-hardware throughput numbers are not comparable and will not be cited as if they were. Measure on gljax's own PJRT backend, on the same device, before and after ARTX7 lands.

## Testing Strategy

* **Unit:** Request, Queue, Batch, KvSlotManager (state transitions), bucket selection, KV addressing math, compile cache key derivation.
* **Integration:** continuous batching (slot reuse mid-batch), chunked prefill (long prompt interleaved with decode), multi-request execution end-to-end against a real PJRT plugin.
* **Stress:** queue overflow, slot exhaustion (`allocate()` returns `None`), large prompt workloads, adversarial mixed decode/prefill timing.

---

# Future Work — ARTX16

* Prefix cache (candidate approach: SGLang's RadixAttention-style prefix sharing)
* Speculative decoding
* PD (prefill/decode) disaggregation
* Remote KV
* Multi-host scheduling
* Distributed serving
* Revisit `PagedKvSlab` as a custom per-backend kernel (e.g. a Ragged Paged Attention–style Pallas/Mosaic kernel on TPU) now that Design Principle #3 means the scheduler and `KvSlotManager` would not need to change to adopt it

---

# Summary

ARTX7 transforms gljax from a **single-request engine** into a **host-driven multi-request runtime**, while preserving static shapes, PJRT compatibility, compile-cache reuse, and deterministic execution.

ARTX7 is not a clone of vLLM, and it is not PagedAttention retrofitted onto PJRT. It is a compiler-oriented serving runtime, purpose-built for static-shape execution environments — trading the flexibility of a dynamically-addressed KV cache for the ability to compile once and execute unchanged, many times over.
