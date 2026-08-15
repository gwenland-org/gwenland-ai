# ARTX21 — Memory Allocator & Backend Buffer System

**Audited commit:** `555881ebc8b0fc0402b30e09258a32a7bfd13c52`
**Last updated:** 2026-07-25
**Auditor:** Percival-aux (ARTX21)
**Target GwenLand modules:** `glproc`, `glcuda`, `glmetal`, `glvulkan`, `GATE` (shared memory subsystem)

---

## 1. Executive Summary

The ggml memory subsystem is a two-layer abstraction sitting *between* the graph scheduler (ARTX22) and the per-backend kernels. It owns no computation and no dtype knowledge; it owns only the contract that says "every `ggml_tensor` has a `buffer` of some `buffer_type`, and the data lives at `tensor->data` inside that buffer."

The two layers are:

1. **`ggml-alloc.c`** — a backend-agnostic allocator with two faces: `ggml_tallocr` (Section 5.1), a single-buffer **bump allocator** for pre-sized contexts (e.g., model weights); and `ggml_gallocr` (Section 5.2), a **graph allocator** that lays out every tensor in a compute graph, reusing memory across nodes via a best-fit free-list per buffer type and a per-tensor `n_children`/`n_views` lifetime tracker. It supports up to 16 buffer types in one graph (multi-GPU, mixed CPU+GPU).
2. **`ggml-backend.cpp`** — the buffer-type and buffer interfaces (`ggml_backend_buffer_type_i`, `ggml_backend_buffer_i`, `ggml_backend_i`) plus cross-backend tensor copy, async copy, events, and scheduler-side copy machinery. It also defines the canonical CPU buffer type (`ggml_backend_cpu_buffer_type()`) and the `buffer_from_ptr` mmap-style path used by all backends that need to import external host memory.

For GwenLand, the architectural decisions worth **ADOPT**ing are: the two-layer split (allocator vs. buffer-type), the `n_children`/`n_views` lifetime model, the `is_host` fast-path for cross-backend copies, and the CUDA VMM pool with 128-byte alignment and reverse-order free discipline. The decisions worth **REJECT**ing are: the 32-byte `TENSOR_ALIGNMENT` (too small for AVX-512 / AMX), the per-graph linear best-fit scan (`O(N²)` in tensor count), and the synchronous fallback inside `ggml_backend_tensor_copy_async` when the dst backend lacks `cpy_tensor_async`.

---

## 2. Purpose

Provide a backend-agnostic memory layer that: allocates every tensor in a compute graph into the right backend buffer, respecting per-buffer alignment and max-size constraints; reuses memory across nodes whose lifetimes do not overlap (in-place reuse for `ggml_op_can_inplace` ops, free-list reuse for everything else); supports multi-backend graphs (CPU + CUDA, multi-GPU) by allowing each node to be tagged with a buffer-type id and by creating per-buffer-type sub-allocators; exposes a stable C ABI (`ggml_backend_buffer_type_t`, `ggml_backend_buffer_t`) so that backends compiled as separate `.so` files can register their own buffer types without recompiling ggml; handles host-mapped (mmap) tensors and host-pinned (cudaMallocHost) tensors uniformly through the same `is_host` flag.

It is **not** responsible for: kernel dispatch (delegated to backends), graph splitting (delegated to the scheduler in ARTX22), or quantization format choice (delegated to type-traits in ARTX01).

---

## 3. Source Files

| File                                       | Lines  | Role                                                                          |
| ------------------------------------------ | ------ | ----------------------------------------------------------------------------- |
| `ggml/src/ggml-alloc.c`                    | 1248   | `ggml_tallocr` bump allocator, `ggml_dyn_tallocr` free-list, `ggml_gallocr` graph allocator |
| `ggml/src/ggml-backend.cpp`                | 2372   | Buffer-type / buffer / backend interfaces, async copy, events, CPU buffer type, scheduler-side copy machinery |
| `ggml/src/ggml-backend-impl.h`             | 275    | Internal vtable definitions: `ggml_backend_buffer_type_i`, `ggml_backend_buffer_i`, `ggml_backend_i`, `ggml_backend_device_i` |
| `ggml/src/ggml-impl.h`                     | 783    | `TENSOR_ALIGNMENT`, `ggml_are_same_layout`, `ggml_impl_is_view`, hash set, bitset |
| `ggml/src/ggml.c`                          | 8023   | `ggml_aligned_malloc`/`_free`, `ggml_nbytes_pad`, `ggml_tensor_overhead`, context arena |
| `ggml/include/ggml.h`                      | 2931   | `struct ggml_tensor` (with `view_src`, `view_offs`, `extra`, `padding`), `GGML_PAD`, `GGML_MEM_ALIGN` |
| `ggml/include/ggml-alloc.h`                | 86     | Public API for `ggml_tallocr`, `ggml_gallocr`, `ggml_backend_alloc_ctx_tensors_from_buft` |
| `ggml/include/ggml-backend.h`              | 436    | Public API for buffer type / buffer / backend / device |
| `ggml/src/ggml-cuda/ggml-cuda.cu`          | 5426   | Reference backend: `ggml_backend_cuda_buffer_type_get_alignment` (128), `ggml_cuda_pool_leg` (legacy free-list), `ggml_cuda_pool_vmm` (VMM), host pinned buffer type |
| `ggml/src/ggml-cuda/common.cuh`            | 1662   | `ggml_cuda_pool` abstract base, `ggml_cuda_pool_alloc<T>` RAII wrapper, `ggml_tensor_extra_gpu` struct (declared, currently unused by CUDA path) |

> Note: ARTX21 is a Shared-layer audit. Where the CUDA or SYCL backend is
> referenced, it is to illustrate how the shared contract is *consumed* by
> a real backend. Per-backend ARTX documents (ARTX08–14) cover the
> backend-internal memory in depth.

---

## 4. Architecture Overview

```
                      ┌─────────────────────────────────────────────┐
   user / scheduler   │  ggml_backend_sched (ARTX22)                │
                      │   - assigns node_buffer_ids[]               │
                      │   - calls ggml_gallocr_reserve_n / alloc    │
                      └────────────────────┬────────────────────────┘
                                           │
                                           ▼
                ┌──────────────────────────────────────────────────┐
                │  ggml-alloc.c                                    │
                │  ├─ ggml_tallocr       (single-buffer bump)      │
                │  ├─ ggml_dyn_tallocr   (best-fit free-list,      │
                │  │                      GGML_VBUFFER_MAX_CHUNKS) │
                │  └─ ggml_gallocr       (graph allocator:         │
                │                          n_children/n_views      │
                │                          lifetime tracking)      │
                └────────────────────┬─────────────────────────────┘
                                     │ uses
                                     ▼
                ┌──────────────────────────────────────────────────┐
                │  ggml-backend.cpp                                │
                │  ├─ ggml_backend_buffer_type_i (6 methods,       │
                │  │   3 optional)                                 │
                │  ├─ ggml_backend_buffer_i      (11 methods,      │
                │  │   7 optional)                                 │
                │  ├─ ggml_backend_cpu_buffer_type()  (singleton)  │
                │  ├─ ggml_backend_cpu_buffer_from_ptr()  (mmap)    │
                │  ├─ multi_buffer  (a buffer that wraps N buffers) │
                │  └─ async copy / events / synchronize             │
                └────────────────────┬─────────────────────────────┘
                                     │ implements
            ┌────────────────────────┼────────────────────────┐
            ▼                        ▼                        ▼
   ┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐
   │ CPU buffer type │     │ CUDA buffer type │     │ Metal/Vulkan/... │
   │ TENSOR_ALIGN=32 │     │ align=128        │     │ backend-specific │
   │ is_host=true    │     │ pool_vmm / leg   │     │ host_buffer_type │
   │ from_ptr (mmap) │     │ host pinned      │     │ pinned memory    │
   └─────────────────┘     └──────────────────┘     └──────────────────┘
```

Key design points:

* **No polymorphism in C — vtables in structs.** Every buffer type is a `struct ggml_backend_buffer_type { iface; device; context; }` where `iface` is a struct of function pointers. Same pattern for buffers, backends, devices, registries.
* **Buffer type is a singleton per backend.** `ggml_backend_cpu_buffer_type()` returns a pointer to a `static` struct; CUDA returns one of N per-device statics. Identity is by pointer equality (e.g., `ggml_backend_buft_is_cuda_host` checks `buft->iface.get_name == ggml_backend_cuda_host_buffer_type_name`).
* **Two-tier allocation.** The *graph* allocator uses a *dynamic* sub-allocator (`ggml_dyn_tallocr`) to plan offsets within a virtual address range, then asks the buffer type to allocate one or more backend buffers (`vbuffer` chunks) sized to the planned maximum. This is why graph allocation is a two-phase `reserve` → `alloc` operation (Section 5.3).
* **`is_host` is the cross-backend contract.** When true, the buffer's data is readable/writable directly from the CPU; the copy fast-path becomes `memcpy`. The CUDA host-pinned buffer type also reports `is_host=true`, so the same fast path applies to pinned host memory.
* **No mmap of the model file in the allocator itself.** mmap happens in the loader; the result is wrapped via `ggml_backend_cpu_buffer_from_ptr`, which produces a buffer with `free_buffer = NULL` (does not own the mapping).

---

## 5. Execution Flow

### 5.1 Single-tensor allocation: `ggml_tallocr`

`ggml_tallocr_new(buffer)` (`ggml-alloc.c:60`) returns a 40-byte struct holding `{ buffer, base, alignment, offset }`. `ggml_tallocr_alloc` (`ggml-alloc.c:75`): (1) `size = ggml_backend_buffer_get_alloc_size(buffer, tensor)` (defaults to `ggml_nbytes`; CUDA overrides for `FLASH_ATTN_EXT`); (2) `size = GGML_PAD(size, alignment)`; (3) bump `offset` forward, assert in-buffer; (4) `addr = base + offset`; (5) `ggml_backend_tensor_alloc(buffer, tensor, addr)` sets `tensor->buffer`, `tensor->data`, and invokes `init_tensor` if present.

There is **no free**. The bump allocator is monotonic; it is used for model weights and other long-lived contexts where the entire buffer is freed at once.

### 5.2 Graph allocation: `ggml_gallocr`

`ggml_gallocr_new_n(bufts, n_bufs)` (`ggml-alloc.c:497`) creates a graph allocator with one `ggml_dyn_tallocr` per **distinct** buffer type — if the same `buft` pointer appears multiple times in `bufts[]`, the same dynamic allocator is reused (line 515–520). This is the mechanism for "two CUDA streams sharing one pool".

The graph allocator tracks per-tensor state in a `hash_node`:

```c
struct hash_node {
    int n_children;       // number of graph nodes that consume this tensor
    int n_views;          // number of views into this tensor
    int buffer_id;        // which buft the tensor lives in
    struct buffer_address addr; // { chunk, offset }
    bool allocated;       // is this hash entry currently owning memory?
};
```

`ggml_gallocr_alloc_graph_impl` (`ggml-alloc.c:717`) is the heart: (1) allocate leafs and `GGML_TENSOR_FLAG_INPUT` nodes first; (2) for each graph node, count `n_children` and `n_views` of every source; (3) walk nodes in execution order, allocate each, then decrement parents' `n_children` — when both `n_children == 0` and `n_views == 0`, free the parent via `ggml_dyn_tallocr_free_bytes`; (4) `ggml_gallocr_allocate_node` (`ggml-alloc.c:622`) tries in-place reuse first: if `ggml_op_can_inplace(node->op)` and a parent has `n_children == 1 && n_views == 0`, the node aliases the parent's address; (5) after the walk, `hash_node` entries are written into `node_allocs[i]` so the second phase can re-init tensors without re-planning.

### 5.3 Reserve vs. alloc

`ggml_gallocr_reserve_n` (`ggml-alloc.c:961`) calls `ggml_gallocr_alloc_graph_impl` to *plan* offsets, then asks each buffer type to `alloc_buffer` the planned `max_size` per chunk. It creates a `vbuffer` (a 16-element array of backend buffer pointers) per buffer type. `ggml_gallocr_alloc_graph` (`ggml-alloc.c:1051`) skips planning if `!ggml_gallocr_needs_realloc`; it just calls `ggml_gallocr_init_tensor` for every leaf and node, setting `tensor->buffer` and `tensor->data` from the stored `addr`. This two-phase split lets `ggml_backend_sched` reserve once with a worst-case graph and run many small batches without re-allocation (see ARTX22).

### 5.4 Cross-backend tensor copy

`ggml_backend_tensor_copy` (`ggml-backend.cpp:477`) decides the copy path at runtime:

```
if (src->buffer is host)  tensor_set(dst, src->data, 0, nbytes)
else if (dst->buffer is host)  tensor_get(src, dst->data, 0, nbytes)
else if (dst->buffer->iface.cpy_tensor(src, dst))  return  // backend did it
else  malloc staging; tensor_get(src, staging); tensor_set(dst, staging); free
```

The first two branches bypass the staging buffer when one side is host-visible. The third branch lets a backend handle its own cross-device copy (CUDA uses `cudaMemcpyAsync` between devices). The fourth is the slow fallback. `ggml_backend_tensor_copy_async` (`ggml-backend.cpp:500`) tries the dst backend's `cpy_tensor_async` hook first; if absent or it returns false, it **synchronizes both backends and does a blocking copy**. This is a contract-level limitation: true async cross-backend copy requires the dst backend to implement the hook (CUDA does; CPU does not).

### 5.5 mmap path

`ggml_backend_cpu_buffer_from_ptr(ptr, size)` (`ggml-backend.cpp:2368`) asserts that `ptr` is `TENSOR_ALIGNMENT`-aligned (32 bytes — see Finding F06), then wraps it in a buffer using the `ggml_backend_cpu_buffer_from_ptr_i` vtable, whose `free_buffer` is **NULL** (the caller owns the mapping). The buffer reports `is_host=true`, so cross-backend copies from it take the memcpy fast path. This is the path used by mmap'd model weights and by Vulkan's pinned-memory staging.

---

## 6. Data Layout

### 6.1 The `ggml_tensor` descriptor

`struct ggml_tensor` (`ggml.h:673`) is 192 bytes on a 64-bit system:

```
type                  (4B)  + 4B padding
buffer                (8B)  — pointer to ggml_backend_buffer
ne[GGML_MAX_DIMS]    (32B)  — element counts
nb[GGML_MAX_DIMS]    (32B)  — byte strides
op                    (4B)  + 4B padding
op_params[16]        (64B)  — 64 bytes of int32/float params
flags                 (4B)  + 4B padding
src[GGML_MAX_SRC]   (80B)  — up to 10 source tensors
view_src              (8B)  — non-NULL if this tensor is a view
view_offs             (8B)  — byte offset into view_src->data
data                  (8B)  — current pointer to payload
name[GGML_MAX_NAME] (64B)  — null-terminated name
extra                 (8B)  — backend-specific pointer
padding               (8B)  — reserved
```

Two fields drive the allocator:

* `buffer` — the `ggml_backend_buffer_t` that owns `data`. NULL until
  `ggml_backend_tensor_alloc` or `ggml_backend_view_init` sets it.
* `view_src` / `view_offs` — if `view_src != NULL`, this tensor is a
  view; its `data` is `view_src->data + view_offs` and its `buffer` is
  `view_src->buffer`. The allocator does **not** allocate storage for
  views; it only sets `view_src->buffer` on the view via
  `ggml_backend_view_init` (`ggml-backend.cpp:1980`).

### 6.2 Tensor views

Views are pure stride/offset descriptors — no separate allocation.
`ggml_view_src->n_views` is bumped in the planner so that the source's
memory is not freed while any view is live. `ggml_are_same_layout`
(`ggml-impl.h:75`) is the predicate for in-place reuse: it compares
`type`, all `ne[]`, and all `nb[]` exactly. Strides are *not* normalized;
a transposed tensor and its non-transposed original have different
layouts and cannot share storage.

### 6.3 The `extra` field

`extra` is a backend-specific `void *` on every tensor. At this commit: **CUDA** declares `struct ggml_tensor_extra_gpu { void * data_device[N]; cudaEvent_t events[N][M]; }` (`common.cuh:1213`) but **does not populate it** for any tensor (grep for `extra =` in `ggml-cuda/` returns no assignment sites). The field is dead in the CUDA path; CUDA tracks split tensors via `ggml_backend_cuda_split_buffer_context::tensor_extras` instead. **SYCL** still uses `extra` for `ggml_tensor_extra_gpu` (see `ggml-sycl.cpp:562, 1113`). **OpenVINO / OpenCL** use `extra` for vendor IR handles and per-quant-format metadata. This is a real cross-backend inconsistency — see Finding F12.

---

## 7. Memory Layout

### 7.1 The `vbuffer`: virtual buffer of multiple chunks

`struct vbuffer` (`ggml-alloc.c:397`) is an array of up to `GGML_VBUFFER_MAX_CHUNKS = 16` `ggml_backend_buffer_t` pointers. It represents a single contiguous virtual address range split across multiple physical allocations. The dynamic allocator plans offsets within this virtual range; `ggml_vbuffer_alloc` then asks the buffer type for one real buffer per planned chunk (line 423). This lets a single graph allocate more memory than a single backend buffer can hold. CUDA's default max_size is `SIZE_MAX` (no chunking needed); a backend with a per-buffer cap can have the planner split the graph across multiple chunks. The fallback when `n_chunks == GGML_VBUFFER_MAX_CHUNKS - 1` is to set the last chunk's `free_blocks[0].size = SIZE_MAX/2` (line 172), effectively giving the last chunk "infinite" size to avoid OOM during planning.

### 7.2 Free-list layout: `tallocr_chunk`

Each chunk has a `struct tallocr_chunk`:

```c
struct free_block { size_t offset; size_t size; };
struct tallocr_chunk {
    struct free_block free_blocks[MAX_FREE_BLOCKS];  // 256 * 16B = 4 KB
    int n_free_blocks;
    size_t max_size;  // high-water mark of used bytes
};
```

`MAX_FREE_BLOCKS = 256` is a hard cap. If a chunk hits 256 free blocks, the next free attempt asserts and aborts. This is rarely hit because the planner coalesces adjacent blocks aggressively in `ggml_dyn_tallocr_free_bytes` (`ggml-alloc.c:311`).

### 7.3 Per-tensor metadata

The planner stores per-tensor allocation results in `node_allocs[]` and `leaf_allocs[]`:

```c
struct tensor_alloc {
    int buffer_id;
    struct buffer_address addr;  // { chunk, offset }
    size_t size_max;             // 0 if pre-allocated / unused / view
};
```

This is what survives between `reserve` and `alloc`: the planner can skip the layout pass on subsequent calls if `!needs_realloc` (line 1008).

### 7.4 Alignment constants

There are **five** distinct alignment constants in the codebase:

| Constant             | Value | Defined at                | Used by                                  |
| -------------------- | ----- | ------------------------- | ---------------------------------------- |
| `TENSOR_ALIGNMENT`   | 32    | `ggml-impl.h:44`          | CPU buffer type; mmap buffer alignment   |
| `GGML_MEM_ALIGN`     | 4/8/16| `ggml.h:236-243`          | Context arena (`ggml_new_object`)        |
| `ggml_aligned_malloc`| 64    | `ggml.c:331` (256 on s390x) | Host-side heap allocations (CPU buffer, work buffer) |
| (CUDA buft)          | 128   | `ggml-cuda.cu:901`        | CUDA buffer type                         |
| (CUDA VMM)           | 128   | `ggml-cuda.cu:571`        | CUDA VMM pool                            |

The CPU buffer type advertises **32 bytes** of alignment to the graph allocator; the underlying host allocation returns 64-byte-aligned memory. The 32-byte advertised alignment is therefore the binding constraint on tensor placement — see Finding F06.

---

## 8. Parallelism Strategy

The allocator itself is **single-threaded by design**. The graph planner runs on the inference thread; no atomics, no locks. This is intentional: the planner is O(N²) in tensor count (Section 12.1), but N is bounded by graph size (typically < 10 000 nodes), and the planner runs *once per graph topology change*, not per token.

Parallelism happens *after* allocation: the scheduler (ARTX22) splits the graph into per-backend subgraphs and computes them in parallel via `ggml_backend_graph_compute_async`; within a backend, parallelism is per-op (CPU) or per-stream (CUDA); the `ggml_backend_sched` events array (`events[MAX_BACKENDS][MAX_COPIES]`, `ggml-backend.cpp:807`) supports up to `GGML_SCHED_MAX_COPIES = 4` pipeline copies for parallel decoding, allowing the scheduler to run two consecutive batches back-to-back without synchronizing the GPU.

The copy phase in `ggml_backend_sched_compute_splits`
(`ggml-backend.cpp:1541`) is the only place where the memory layer
interacts with parallelism: it must synchronize the dst backend's
event before overwriting an input copy, and it must record an event
on the src backend after the split completes so the next split can
wait on it.

---

## 9. SIMD / GPU Strategy

The allocator is SIMD-agnostic, but alignment directly affects SIMD: 32-byte alignment (`TENSOR_ALIGNMENT`) is sufficient for AVX2 (256-bit loads) but **not** for AVX-512 (64-byte) or AMX (128-byte tile loads). Backends that need stronger alignment override `get_alignment` (CUDA: 128). The CUDA VMM pool (`ggml_cuda_pool_vmm`, `ggml-cuda.cu:536`) enforces 128-byte alignment on every allocation (line 572). The CUDA legacy pool (`ggml_cuda_pool_leg`, `ggml-cuda.cu:419`) does best-fit over a 256-slot free list and rounds up to 256 bytes for look-ahead (line 493), relying on `cudaMalloc`'s native alignment (≥256 bytes).

The `extra` field could carry per-tensor SIMD hints (e.g., "this tensor is 128-byte aligned, prefer AVX-512"), but no backend populates it for that purpose at this commit.

---

## 10. Quantization Strategy

The allocator is quantization-agnostic. The only place quant enters is `ggml_backend_buft_get_alloc_size`, which defaults to `ggml_nbytes(tensor)` (`ggml-backend.cpp:61`). CUDA overrides for `FLASH_ATTN_EXT` (`ggml-cuda.cu:909`) to reserve extra workspace for the KV cache transpose; CPU does not override — quantized tensors are allocated exactly `ggml_nbytes` (block-aligned by construction). The allocator never needs to know block sizes, scale bytes, or zero-point layout. The contract is purely "give me N bytes aligned to A". A new quant format requires zero allocator changes — the same property ARTX01-F03 noted for the type-traits table.

---

## 11. Correctness Analysis

### 11.1 Lifetime tracking via `n_children` / `n_views`

The planner frees a parent when its `n_children` reaches 0 *and* its `n_views` is 0 (`ggml-alloc.c:804`). Correct under the assumption that: every consumer of a tensor appears in some node's `src[]`; every view of a tensor is itself a graph node (and bumps `view_src->n_views` at `ggml-alloc.c:740`); no consumer reads a tensor after its last consumer has executed (guaranteed by the topological order of `cgraph->nodes`). The in-place reuse path (`ggml-alloc.c:631-680`) adds the constraint that the parent must have exactly one consumer (`p_hn->n_children == 1`) and no other views (`p_hn->n_views == 0`). This prevents aliasing when an op is in-place but its output is later consumed by two downstream nodes.

### 11.2 View aliasing

`ggml_backend_view_init` (`ggml-backend.cpp:1980`) sets `tensor->buffer = view_src->buffer` and `tensor->data = view_src->data + view_offs`. The view does not own its buffer; freeing the view is a no-op. The planner ensures `view_src` is not freed while any view is live via the `n_views` counter (line 808). Correct as long as `view_offs` does not exceed `view_src->buffer`'s size — asserted by `ggml_backend_tensor_alloc`'s bounds check (line 1997).

### 11.3 In-place op correctness

`ggml_op_can_inplace` (`ggml-alloc.c:22`) lists 22 ops. The comment at line 21: "ops that return true for this function must not use restrict pointers for their backend implementations." This is the correctness invariant: if a backend kernel marks its dst `restrict`, in-place reuse would be UB. CPU honors this; CUDA honors it for the listed ops. A new op added to the in-place list must be audited per-backend.

### 11.4 Alignment of views

A view's `data` is `view_src->data + view_offs`. The allocator does **not** verify that `view_offs` is `TENSOR_ALIGNMENT`-aligned. If a backend creates a view with an unaligned offset, downstream SIMD loads may fault. Mitigated by the convention that `view_offs` is always a multiple of the parent's `nb[i]` for some `i`, and `nb[0] = type_size` which is power-of-2 for every type. But there is no assertion.

### 11.5 `ggml_aligned_malloc` and zero size

`ggml_aligned_malloc(0)` logs a warning and returns `NULL` (`ggml.c:341-344`). Downstream code that dereferences the pointer will crash. The bump allocator's `ggml_tallocr_alloc` does not check for 0-size tensors before computing the padded size, but a 0-size tensor padded to 32 bytes is 32, not 0, so this is not hit in practice.

### 11.6 VMM pool stack discipline

`ggml_cuda_pool_vmm::free` (`ggml-cuda.cu:672`) asserts `ptr == (char*)pool_addr + pool_used - size`. Allocations must be freed in **strict reverse order**. The RAII wrapper `ggml_cuda_pool_alloc<T>` (`common.cuh:1167`) guarantees this via destructor ordering. Any backend code that holds a raw `void*` from the pool and frees it out of order will trigger the assertion.

---

## 12. Optimization Analysis

### 12.1 Identified optimizations

| Optimization                                   | Where                                       | Notes                                                            |
| ---------------------------------------------- | ------------------------------------------- | ---------------------------------------------------------------- |
| Two-phase reserve → alloc                      | `ggml-alloc.c:824-948, 1051-1097`          | Reserve once on worst-case graph; alloc is O(N) tensor init only.|
| Best-fit free-list with adjacent-block coalescing | `ggml-alloc.c:201-349`                    | `O(N²)` scan for best fit, but N (free blocks) is small (<256).  |
| In-place reuse for 22 in-place ops             | `ggml-alloc.c:622-688`                      | Avoids alloc + free for `ADD`, `MUL`, `ROPE`, `RMS_NORM`, etc.   |
| Per-buffer-type sub-allocator sharing          | `ggml-alloc.c:515-520`                      | Two backends with the same `buft` pointer share one free-list.   |
| Multi-chunk vbuffer for >max_size graphs       | `ggml-alloc.c:159-177`                      | Splits a virtual range across up to 16 backend buffers.          |
| `is_host` fast path for cross-backend copy     | `ggml-backend.cpp:484-487`                  | `memcpy` directly when either side is host-visible.              |
| `cpy_tensor` backend hook                      | `ggml-backend.cpp:207-210`                  | Backend can do its own DMA (e.g., CUDA `cudaMemcpyAsync` P2P).   |
| MoE expert-only copy                            | `ggml-backend.cpp:1576-1660`                | Copies only the experts used by `MUL_MAT_ID`, not the full weight. |
| Pipeline parallelism via `n_copies=4` events   | `ggml-backend.cpp:807, 1541-1722`           | Two consecutive batches overlap on different event slots.        |
| CUDA VMM pool with 32 GB VA reserve            | `ggml-cuda.cu:537, 593`                     | Avoids `cudaMalloc` per tensor; `cuMemMap` granularity-aligned.  |
| `ggml_tallocr` bump for weights                | `ggml-alloc.c:75`                           | O(1) per-tensor; no free-list overhead for one-shot loads.       |

### 12.2 Optimizations *not* present

* **No thread-parallel planning.** The planner is single-threaded. For a 10 000-node graph, the `O(N²)` best-fit scan dominates planning time. A parallel allocator (one thread per buffer type) is not implemented.
* **No size-class segregation.** The free-list is a single sorted array per chunk; there are no per-size-class pools (unlike jemalloc/tcmalloc). Allocations of wildly different sizes coexist, increasing fragmentation.
* **No persistent pool across graph runs.** The dynamic allocator is reset on every `reserve_n_impl` call (line 843). The `vbuffer` persists, but the free-list is rebuilt from scratch. A persistent free-list that survives topology-identical re-runs would skip the `O(N²)` scan.
* **No alignment-aware best-fit.** The free-list search finds the smallest block ≥ size but does not check whether the block's offset satisfies a per-tensor alignment override. All tensors in one buffer type share the same alignment.
* **No NUMA-aware allocation for the CPU buffer type.** `ggml_aligned_malloc` uses `posix_memalign`, which is not NUMA-aware. Buffer placement is incidental to whichever thread calls it.

---

## 13. Architectural Strengths

1. **Two-layer split is clean.** The allocator knows nothing about backends; the backend interface knows nothing about graphs. A new backend plugs in by implementing six functions in `ggml_backend_buffer_type_i` and eleven in `ggml_backend_buffer_i`. Same property that makes the type-traits table (ARTX01-F03) good.

2. **The `n_children` / `n_views` lifetime model is correct and minimal.** Two counters per tensor decide free / reuse. The in-place path adds one extra check (`n_children == 1 && n_views == 0`) and reuses storage without a round-trip through the free-list.

3. **The `is_host` flag is the right cross-backend abstraction.** One boolean instead of a per-backend "is this readable from CPU?" hook. The copy fast-path is three lines in `ggml_backend_tensor_copy` and benefits every backend that reports `is_host=true` (CPU, CUDA host pinned, SYCL host, Vulkan host).

4. **The vbuffer / chunk mechanism handles buffer-size limits gracefully.** A backend with a per-buffer cap can split a graph across up to 16 chunks. The planner transparently manages offsets within the virtual range; the backend only sees per-chunk `alloc_buffer` calls.

5. **The CUDA VMM pool is well-engineered.** 32 GB VA reserve, granularity-aligned `cuMemMap`, peer-access setup for multi-GPU, and reverse-order free discipline (asserted). Right design for a long-running inference server amortizing `cudaMalloc` cost.

6. **`ggml_backend_cpu_buffer_from_ptr` is the right mmap hook.** The buffer does not own the pointer (`free_buffer = NULL`); the loader can mmap a model file, wrap each tensor in a buffer, and unmap the whole file at teardown. No reference counting, no double-free.

7. **Per-buffer-type sub-allocator sharing is a nice trick.** When the scheduler registers the same `buft` pointer twice (e.g., two CUDA streams), the planner uses one free-list for both (`ggml-alloc.c:515-520`). Invisible to the scheduler; avoids duplicate pools.

---

## 14. Architectural Weaknesses

### W1 — 32-byte `TENSOR_ALIGNMENT` is too small for modern SIMD

**Evidence:** `ggml-impl.h:44` `#define TENSOR_ALIGNMENT 32`; CPU buffer type returns this value (`ggml-backend.cpp:2316-2320`).

**Impact:** AVX-512 needs 64-byte alignment for aligned loads; AMX tile loads need 128. Tensors in CPU buffers can be placed at 32-byte offsets even though the underlying `ggml_aligned_malloc` returns 64-byte-aligned memory, forcing backends to use unaligned load variants. GGUF's 32-byte mmap guarantee is the binding constraint; the fix is per-buffer-type alignment (CUDA already does 128) rather than a single global constant.

### W2 — `O(N²)` best-fit scan in `ggml_dyn_tallocr_alloc`

**Evidence:** `ggml-alloc.c:211-223` — for each allocation, iterate over all chunks × all free blocks (minus the last). `MAX_FREE_BLOCKS = 256`, so worst case is 4096 comparisons per allocation.

**Impact:** Planning a 10 000-node graph can take tens of milliseconds. Painful for interactive re-planning. Fix: size-class segregated free-list (like jemalloc) or a binary search tree keyed by block size.

### W3 — `MAX_FREE_BLOCKS = 256` is a hard abort

**Evidence:** `ggml-alloc.c:135` `GGML_ASSERT(chunk->n_free_blocks < MAX_FREE_BLOCKS && "out of free blocks")`.

**Impact:** Pathological graphs with many small non-adjacent free blocks can hit this limit and abort. Coalescing mitigates it in practice; there is no graceful fallback.

### W4 — Synchronous fallback in `ggml_backend_tensor_copy_async`

**Evidence:** `ggml-backend.cpp:514-518` — if dst backend lacks `cpy_tensor_async`, synchronizes **both** backends and does a blocking copy.

**Impact:** The CPU backend has no `cpy_tensor_async` (ARTX01-F01). Any cross-backend copy involving the CPU stalls the GPU. For hybrid CPU+GPU workloads, this serializes the pipeline. See Finding F10.

### W5 — `ggml_tallocr` has no free

**Evidence:** `ggml-alloc.c:60-91` — only `offset += size`; no `ggml_tallocr_free` exists.

**Impact:** Long-lived contexts that want to free individual tensors cannot use `ggml_tallocr`; they must use the heavier `ggml_gallocr`. The workaround (allocate a new buffer and copy live tensors) is not implemented.

### W6 — `extra` field is under-used and inconsistent across backends

**Evidence:** `ggml.h:702` comment "extra things e.g. for ggml-cuda.cu"; CUDA declares `ggml_tensor_extra_gpu` (`common.cuh:1213`) but does not populate it; SYCL does (`ggml-sycl.cpp:562`); OpenVINO and OpenCL use it for vendor handles.

**Impact:** Cross-backend code cannot rely on `extra` carrying any specific structure. The misleading comment suggests CUDA uses it, which is no longer true. See Finding F12.

### W7 — No NUMA-aware CPU buffer allocation

**Evidence:** `ggml_aligned_malloc` (`ggml.c:331`) calls `posix_memalign`, which allocates on the local NUMA node of the calling thread. The CPU backend does not bind the allocation thread.

**Impact:** On multi-socket systems, a buffer allocated by the inference thread may end up on a different node than the worker threads that read it, causing cross-socket traffic. ARTX01-F09's NUMA-aware chunking mitigates this for matmul, but the buffer placement is still incidental.

### W8 — `aligned_offset` NULL-relative math is misleading

**Evidence:** `ggml-alloc.c:52` — the function takes a `const void * buffer` but is called with `NULL` in `ggml_dyn_tallocr_alloc` (line 202, 312). The math still works because `(uintptr_t)NULL + offset == offset`.

**Impact:** No correctness bug, but the API is confusing — the function name suggests buffer-relative alignment, but in the dyn-tallocr path it is offset-relative.

### W9 — CPU buffer type singleton has `device = NULL`

**Evidence:** `ggml-backend.cpp:2338` `/* .device = */ NULL, // FIXME ...`.

**Impact:** Code that traverses buffer → buft → device to find the owning backend will get NULL for CPU buffers. Works today because the scheduler tracks backends separately, but it is a leaky abstraction. See Finding F08.

### W10 — `GGML_VBUFFER_MAX_CHUNKS = 16` is a compile-time cap

**Evidence:** `ggml-alloc.c:95`. A graph that needs more than 16 chunks in a single buffer type silently gets `SIZE_MAX/2` for the last chunk (line 172), which is likely to fail at `alloc_buffer` time.

**Impact:** Unlikely to be hit in practice (16 × 4 GB = 64 GB per buffer type), but the failure mode is a runtime OOM, not a clear error message. See Unknown U6.

---

## 15. GwenLand Mapping

| GwenLand module | ADOPT / ADAPT / REJECT / MONITOR / DEFER | What | Reasoning |
| --------------- | ---------------------------------------- | ---- | --------- |
| `glproc`        | **ADOPT** | Two-layer split (allocator vs. buffer type) | Clean separation; lets new backends plug in with six functions. |
| `glproc`        | **ADOPT** | `n_children` / `n_views` lifetime model | Minimal, correct, easy to reason about. |
| `glproc`        | **ADOPT** | `ggml_backend_cpu_buffer_from_ptr` mmap pattern | Right way to wrap external memory; no ownership. |
| `glproc`        | **REJECT**| `TENSOR_ALIGNMENT = 32` | Use 64 minimum; 128 if AVX-512 / AMX targets exist. |
| `glproc`        | **ADAPT** | `ggml_tallocr` bump allocator | Keep it but add an optional `free` for compaction. |
| `glcuda`        | **ADOPT** | CUDA VMM pool with 32 GB VA reserve, 128-byte alignment | Best-in-class design; reverse-order free is a real constraint but worth it. |
| `glcuda`        | **ADOPT** | `ggml_cuda_pool_alloc<T>` RAII wrapper | Guarantees stack discipline; prevents VMM assert. |
| `glcuda`        | **ADOPT** | `ggml_backend_cuda_host_buffer_type` with fallback to CPU | Graceful degradation when `cudaMallocHost` fails. |
| `glmetal`/`glvulkan` | **ADAPT** | `is_host` fast-path for cross-backend copy | Metal shared memory and Vulkan pinned staging should report `is_host=true`. |
| `GATE`          | **ADOPT** | Two-phase `reserve` → `alloc`; `n_copies=4` event slots | Lets scheduler amortize planning across batches; enables back-to-back batch overlap. |
| `GATE`          | **REJECT**| Synchronous fallback in `tensor_copy_async` | Implement a real event-based async copy; never stall both backends. |
| `GATE`          | **ADAPT** | Per-buffer-type sub-allocator sharing | Keep the dedup; extend it to allow intentional pool sharing between siblings. |
| multiple        | **ADAPT** | `extra` field | Keep the field but document a per-backend struct layout convention; deprecate the misleading comment. |
| multiple        | **MONITOR**| `MAX_FREE_BLOCKS = 256` | Watch for aborts in production; raise to 1024 if hit. |

---

## 16. Recommendations

### R1 — ADOPT the two-layer allocator / buffer-type split
**Priority:** Critical | **Difficulty:** M | **Dependencies:** none
GwenLand's memory subsystem should mirror ggml's split: a backend-agnostic allocator (`gl_gallocr`) that consumes a `gl_buffer_type` vtable, and per-backend implementations of that vtable. Same six-method buffer-type interface; same eleven-method buffer interface.

### R2 — ADOPT the `n_children` / `n_views` lifetime model
**Priority:** Critical | **Difficulty:** S | **Dependencies:** R1
Two-counter per-tensor lifetime tracker is the simplest correct design for graph tensor reuse. Count consumers in pass 1, decrement in execution-order pass 2, free when both counters hit zero. Add the in-place reuse path for the same 22 ops.

### R3 — REJECT `TENSOR_ALIGNMENT = 32`; use 64 minimum, 128 where possible
**Priority:** High | **Difficulty:** S | **Dependencies:** R1
GwenLand's `gl_tensor_alignment` should be 64 by default (covers AVX-512) and 128 for AMX targets. The mmap path can still accept 32-byte-aligned pointers but should re-align on view-init by inserting padding.

### R4 — ADOPT the CUDA VMM pool design
**Priority:** High | **Difficulty:** L | **Dependencies:** R1
`glcuda` should implement a VMM pool identical to `ggml_cuda_pool_vmm`: 32 GB VA reserve, `cuMemCreate` + `cuMemMap`, 128-byte alignment, peer-access setup. Use the RAII wrapper pattern (`gl_cuda_pool_alloc<T>`) to enforce reverse-order free. Fall back to a legacy best-fit pool when VMM is unavailable.

### R5 — REJECT the synchronous fallback in `tensor_copy_async`
**Priority:** High | **Difficulty:** M | **Dependencies:** GATE event system
`gl_tensor_copy_async` should never synchronize both backends. If the dst backend lacks an async copy hook, the call should enqueue a synchronous copy on the dst backend's stream (naturally ordered against subsequent ops) and return immediately.

### R6 — ADAPT the `is_host` fast-path
**Priority:** High | **Difficulty:** S | **Dependencies:** R1
Keep the boolean `is_host` on buffer types. Cross-backend copy uses the same three-branch fast path: if src is host, `memcpy` into dst; if dst is host, `memcpy` out of src; otherwise ask dst's `cpy_tensor` hook. Metal shared memory and Vulkan host-visible memory should report `is_host=true`.

### R7 — ADOPT two-phase `reserve` → `alloc`
**Priority:** High | **Difficulty:** M | **Dependencies:** R2
The scheduler should call `reserve` once with a worst-case graph and then `alloc` per batch. This amortizes the O(N²) planning cost. The `needs_realloc` check (graph topology + per-tensor size comparison) skips re-planning entirely for identical topologies.

### R8 — ADOPT pipeline parallelism via `n_copies` events
**Priority:** Medium | **Difficulty:** M | **Dependencies:** GATE event system, R5
GATE should support up to 4 pipeline copies per backend, with per-(backend, copy) event slots. The scheduler rotates `cur_copy` and `next_copy` per batch, allowing batch N+1's input copies to overlap with batch N's compute.

### R9 — ADAPT the `extra` field with a documented convention
**Priority:** Medium | **Difficulty:** S | **Dependencies:** R1
Keep the `void *extra` field on tensors but require each backend to document its layout in a header. Deprecate the "extra things e.g. for ggml-cuda.cu" comment. Consider adding an `extra_kind` enum tag for type safety.

### R10 — ADAPT the `ggml_backend_cpu_buffer_from_ptr` mmap pattern
**Priority:** Medium | **Difficulty:** S | **Dependencies:** R1
`gl_buffer_from_ptr(ptr, size, alignment)` should assert pointer alignment, wrap the pointer in a buffer with `free_buffer = NULL` (caller owns the mapping), and report `is_host = true`. This is the right hook for mmap'd model weights, imported CUDA host memory, and Vulkan pinned staging.

---

## 17. Findings

### Finding ARTX21-F01

```
Finding ID:           ARTX21-F01
Category:             MEMORY_PATTERN
Engine:               Shared (cross-backend)
Component:            Tensor allocator (bump)
Source File:          ggml/src/ggml-alloc.c
Function:             ggml_tallocr_new, ggml_tallocr_alloc
Lines:                60-91
Summary:              ggml_tallocr is a single-buffer monotonic bump allocator
                      with no per-tensor free.
Observation:          The struct holds {buffer, base, alignment, offset}. Each
                      alloc pads size to alignment, asserts the offset stays in
                      the buffer, advances offset, and calls
                      ggml_backend_tensor_alloc. There is no free function;
                      the only way to reclaim memory is to free the entire
                      buffer. This is appropriate for one-shot loads (model
                      weights) but forces long-lived contexts to either over-
                      allocate or switch to the graph allocator.
Evidence:             ggml-alloc.c:60-91 (struct + alloc);
                      ggml-alloc.c:75 (size = GGML_PAD(size, talloc->alignment));
                      ggml-alloc.c:90 (ggml_backend_tensor_alloc call).
Architectural Impact: Simple, O(1) per-tensor, no fragmentation. Cannot reclaim
                      individual tensors, so memory usage is monotonic for the
                      life of the buffer. The graph allocator (F03) is the
                      escape hatch for graphs that need reuse.
Correctness Impact:   None. Monotonic allocation is correct by construction.
Optimization Type:    None (bump pointer).
GwenLand Target:      glproc, GATE
Recommendation:       ADOPT for weights; ADAPT by adding an optional free for
                      compaction. Long-lived contexts in GwenLand may want to
                      free individual tensors without tearing down the buffer.
Priority:             Medium
Difficulty:           S
Dependencies:         none
Confidence:           High
```

### Finding ARTX21-F02

```
Finding ID:           ARTX21-F02
Category:             MEMORY_PATTERN
Engine:               Shared (cross-backend)
Component:            Dynamic free-list allocator
Source File:          ggml/src/ggml-alloc.c
Function:             ggml_dyn_tallocr_alloc
Lines:                201-308
Summary:              Best-fit free-list with O(N) scan per allocation and
                      MAX_FREE_BLOCKS=256 hard cap.
Observation:          For each allocation, the allocator iterates over all
                      chunks (up to GGML_VBUFFER_MAX_CHUNKS=16) and all free
                      blocks (up to MAX_FREE_BLOCKS=256) excluding the last
                      block, finding the smallest block ≥ size. If no fit, it
                      tries the last block of each chunk (which may grow the
                      chunk), then creates a new chunk. Free coalesces with
                      adjacent blocks. The O(N) scan is per-allocation, making
                      the planner O(N²) in tensor count for dense graphs.
Evidence:             ggml-alloc.c:211-223 (best-fit scan);
                      ggml-alloc.c:225-246 (last-block reuse);
                      ggml-alloc.c:248-252 (new chunk creation);
                      ggml-alloc.c:135 (MAX_FREE_BLOCKS assert).
Architectural Impact: Planning a 10 000-node graph can take 40 M comparisons.
                      Acceptable for batch inference, painful for interactive
                      re-planning. No size-class segregation means internal
                      fragmentation is uncontrolled.
Correctness Impact:   None. The free-list is correctly maintained and coalesced.
Optimization Type:    Best-fit free-list with adjacent-block coalescing.
GwenLand Target:      GATE
Recommendation:       ADAPT. Keep the design but add size-class buckets (e.g.,
                      8 size classes) to reduce scan to O(1) average. Raise
                      MAX_FREE_BLOCKS to 1024 to avoid the abort hazard.
Priority:             Medium
Difficulty:           M
Dependencies:         R1
Confidence:           High
```

### Finding ARTX21-F03

```
Finding ID:           ARTX21-F03
Category:             BACKEND_DESIGN
Engine:               Shared (cross-backend)
Component:            Graph allocator (lifetime tracking)
Source File:          ggml/src/ggml-alloc.c
Function:             ggml_gallocr_alloc_graph_impl, ggml_gallocr_allocate_node
Lines:                622-688, 717-822
Summary:              Per-tensor n_children / n_views counters drive free and
                      in-place reuse decisions.
Observation:          Pass 1 counts n_children (every consumer in src[]) and
                      n_views (every view_src bump). Pass 2 walks nodes in
                      execution order, allocates each, then decrements
                      parents' n_children. When both n_children and n_views hit
                      zero, the parent is freed. In-place reuse triggers when
                      ggml_op_can_inplace(op) and a parent has
                      n_children==1 && n_views==0; the node aliases the parent's
                      address and the parent's allocated flag is cleared.
Evidence:             ggml-alloc.c:731-760 (counting pass);
                      ggml-alloc.c:763-821 (allocation + free pass);
                      ggml-alloc.c:631-680 (in-place reuse);
                      ggml-alloc.c:22-50 (ggml_op_can_inplace).
Architectural Impact: Correct, minimal lifetime tracking with two counters per
                      tensor. In-place reuse avoids alloc + free for 22 ops
                      (ADD, MUL, ROPE, RMS_NORM, SOFT_MAX, ...). The model
                      extends naturally to multi-backend graphs via per-buffer-
                      id sub-allocators.
Correctness Impact:   None. The counters are exact; the free condition is both
                      necessary and sufficient.
Optimization Type:    Lifetime-based memory reuse + in-place aliasing.
GwenLand Target:      GATE, glproc
Recommendation:       ADOPT. Replicate the two-counter model verbatim. Add
                      per-op audit when extending ggml_op_can_inplace to ensure
                      backends honor the no-restrict-pointer invariant.
Priority:             Critical
Difficulty:           S
Dependencies:         R1, R2
Confidence:           High
```

### Finding ARTX21-F04

```
Finding ID:           ARTX21-F04
Category:             BACKEND_DESIGN
Engine:               Shared (cross-backend)
Component:            Backend buffer-type interface
Source File:          ggml/src/ggml-backend-impl.h
Function:             struct ggml_backend_buffer_type_i
Lines:                17-29
Summary:              Six-method vtable: get_name, alloc_buffer, get_alignment,
                      get_max_size, get_alloc_size, is_host. Three are optional
                      with defaults.
Observation:          The interface is the minimal contract for a backend to
                      provide storage. get_max_size defaults to SIZE_MAX;
                      get_alloc_size defaults to ggml_nbytes; is_host defaults
                      to false. alloc_buffer and get_alignment are required.
                      Backends install the vtable via a static struct and return
                      a pointer to it from ggml_backend_dev_buffer_type.
                      Identity is by pointer (singletons) or by comparing
                      get_name function pointers (e.g., CUDA host check at
                      ggml-cuda.cu:1266).
Evidence:             ggml-backend-impl.h:17-29 (struct definition);
                      ggml-backend.cpp:47-78 (default dispatch with NULL checks);
                      ggml-backend.cpp:2329-2343 (CPU singleton);
                      ggml-cuda.cu:1306-1321 (CUDA host singleton).
Architectural Impact: Clean ABI; new backends implement six functions. The
                      optional methods with defaults reduce boilerplate. The
                      singleton-per-backend pattern makes pointer-equality
                      checks valid for "is this the CPU buffer type?".
Correctness Impact:   None. The interface is a vtable; correctness depends on
                      implementations.
Optimization Type:    Indirect call via vtable (branch-predictor friendly due
                      to singletons).
GwenLand Target:      glproc, glcuda, glmetal, glvulkan
Recommendation:       ADOPT. Same six-method interface in GwenLand.
Priority:             Critical
Difficulty:           S
Dependencies:         none
Confidence:           High
```

### Finding ARTX21-F05

```
Finding ID:           ARTX21-F05
Category:             BACKEND_DESIGN
Engine:               Shared (cross-backend)
Component:            Backend buffer interface
Source File:          ggml/src/ggml-backend-impl.h
Function:             struct ggml_backend_buffer_i
Lines:                41-62
Summary:              Eleven-method vtable including 2D variants, cpy_tensor
                      (cross-backend), memset_tensor, init_tensor, reset.
Observation:          The buffer interface extends the buffer-type interface
                      with per-tensor operations: init_tensor (called on alloc
                      and view_init), set/get/set_2d/get_2d/memset_tensor (data
                      access), cpy_tensor (cross-backend copy hook), clear
                      (whole-buffer memset), reset (clear internal state on
                      graph re-run). The 2D variants optimize strided copies
                      (e.g., batched expert copies in MoE). Multi-buffer is a
                      composite that forwards clear to all children but
                      delegates everything else to NULL (line 693-705).
Evidence:             ggml-backend-impl.h:41-62 (struct definition);
                      ggml-backend.cpp:205-211 (cpy_tensor dispatch);
                      ggml-backend.cpp:354-396 (2D variants with fallback to 1D);
                      ggml-backend.cpp:693-705 (multi_buffer vtable).
Architectural Impact: The 2D variants let backends with strided DMA (CUDA
                      cudaMemcpy2DAsync) avoid per-row dispatch overhead. The
                      cpy_tensor hook lets a backend handle its own cross-
                      device copy without going through staging memory. The
                      reset hook lets the graph allocator re-init tensors on
                      re-run without re-allocating buffers.
Correctness Impact:   None. The interface is a vtable.
Optimization Type:    2D strided copy fast path; backend-owned cross-device copy.
GwenLand Target:      glcuda, glvulkan
Recommendation:       ADOPT. Implement all 11 methods; default the 2D variants
                      to loop over 1D if the backend lacks strided DMA.
Priority:             High
Difficulty:           M
Dependencies:         R1
Confidence:           High
```

### Finding ARTX21-F06

```
Finding ID:           ARTX21-F06
Category:             LAYOUT_SUBOPTIMAL
Engine:               Shared (cross-backend)
Component:            Tensor alignment
Source File:          ggml/src/ggml-impl.h, ggml/src/ggml-backend.cpp
Function:             TENSOR_ALIGNMENT, ggml_backend_cpu_buffer_type_get_alignment
Lines:                ggml-impl.h:44; ggml-backend.cpp:2316-2320
Summary:              CPU buffer type advertises 32-byte alignment, insufficient
                      for AVX-512 (64) or AMX (128).
Observation:          TENSOR_ALIGNMENT is 32, with the comment "required for
                      mmap as gguf only guarantees 32-byte alignment". The CPU
                      buffer type's get_alignment returns this value, so the
                      graph allocator places tensors at 32-byte boundaries.
                      The underlying ggml_aligned_malloc returns 64-byte-aligned
                      memory, so the buffer *base* is 64-aligned but individual
                      tensors within it may be only 32-aligned. AVX-512
                      aligned-load instructions require 64-byte alignment;
                      using them on a 32-aligned tensor faults. AMX tile loads
                      require 128-byte alignment.
Evidence:             ggml-impl.h:44 (#define TENSOR_ALIGNMENT 32);
                      ggml-backend.cpp:2316-2320 (CPU get_alignment);
                      ggml.c:331-336 (ggml_aligned_malloc uses 64);
                      ggml-cuda.cu:901 (CUDA uses 128).
Architectural Impact: Backends that need stronger alignment must either
                      override get_alignment (CUDA does: 128) or use unaligned
                      load variants. The CPU backend's choice of 32 forces
                      AVX-512 kernels to use unaligned loads, which are slower
                      on some microarchitectures. AMX kernels cannot use the
                      CPU buffer type at all and must allocate via the AMX
                      extra-buffer-type mechanism (ARTX01-F04).
Correctness Impact:   None directly. Misaligned aligned-loads would fault, but
                      the CPU backend uses unaligned variants, so no fault.
                      The cost is performance, not correctness.
Optimization Type:    None (suboptimal alignment).
GwenLand Target:      glproc
Recommendation:       REJECT. GwenLand should default to 64-byte alignment for
                      the CPU buffer type, with an optional 128-byte mode for
                      AMX-enabled builds. The mmap path can re-align on view-
                      init by padding.
Priority:             High
Difficulty:           S
Dependencies:         R1
Confidence:           High
```

### Finding ARTX21-F07

```
Finding ID:           ARTX21-F07
Category:             MEMORY_PATTERN
Engine:               Shared (cross-backend)
Component:            Host heap allocation
Source File:          ggml/src/ggml.c
Function:             ggml_aligned_malloc, ggml_aligned_free
Lines:                331-402
Summary:              Platform-specialized posix_memalign with 64-byte alignment
                      (256 on s390x, vm_allocate on macOS, hbw_posix_memalign
                      with HBM).
Observation:          Dispatches on platform: _aligned_malloc on MSVC,
                      hbw_posix_memalign if GGML_USE_CPU_HBM, vm_allocate on
                      macOS (large pages), posix_memalign elsewhere. Alignment
                      is 64 bytes except 256 on s390x (matches that arch's cache
                      line). Zero-size allocations log a warning and return NULL.
Evidence:             ggml.c:331-385 (malloc); ggml.c:387-402 (free);
                      ggml.c:332-336 (s390x special case).
Architectural Impact: Correct per-platform allocation. The 64-byte default
                      matches x86 cache lines; s390x's 256-byte special case
                      matches that architecture's 256-byte cache line. The macOS
                      vm_allocate path gives the OS control over page placement.
Correctness Impact:   None. Each platform's deallocator matches its allocator.
Optimization Type:    Platform-specific aligned allocation (cache-line-aligned
                      by default).
GwenLand Target:      glproc
Recommendation:       ADOPT. Same platform dispatch in GwenLand. Add a runtime
                      query for cache line size instead of hardcoding 64.
Priority:             Medium
Difficulty:           XS
Dependencies:         none
Confidence:           High
```

### Finding ARTX21-F08

```
Finding ID:           ARTX21-F08
Category:             BACKEND_DESIGN
Engine:               Shared (cross-backend)
Component:            CPU buffer type (singleton)
Source File:          ggml/src/ggml-backend.cpp
Function:             ggml_backend_cpu_buffer_type, ggml_backend_cpu_buffer_from_ptr
Lines:                2295-2371
Summary:              CPU buffer type is a singleton with TENSOR_ALIGNMENT
                      alignment, is_host=true, no max_size cap, and a separate
                      from_ptr variant that does not own its memory.
Observation:          ggml_backend_cpu_buffer_type() returns a pointer to a
                      static struct (line 2329). Its alloc_buffer calls
                      ggml_aligned_malloc; set/get_tensor are plain memcpy;
                      cpy_tensor fast-paths when src is also host. The from_ptr
                      variant (line 2368) wraps an external pointer with
                      free_buffer=NULL (caller owns) and asserts TENSOR_ALIGNMENT
                      alignment. The device field is NULL with a FIXME comment
                      (line 2338).
Evidence:             ggml-backend.cpp:2299-2343 (CPU buffer type);
                      ggml-backend.cpp:2345-2371 (from_ptr type and function);
                      ggml-backend.cpp:2251-2260 (cpy_tensor host fast path);
                      ggml-backend.cpp:2338 (device=NULL FIXME).
Architectural Impact: Singleton pattern makes pointer equality a valid "is this
                      CPU?" test. The from_ptr variant is the mmap hook. The
                      NULL device prevents traversal from buffer to backend.
Correctness Impact:   None. NULL device is a leaky abstraction, not a bug.
Optimization Type:    Singleton pointer-equality dispatch; memcpy fast path for
                      host-visible buffers.
GwenLand Target:      glproc
Recommendation:       ADOPT the singleton + from_ptr pattern. Fix the device
                      field to point at the real CPU device in GwenLand.
Priority:             High
Difficulty:           S
Dependencies:         R1
Confidence:           High
```

### Finding ARTX21-F09

```
Finding ID:           ARTX21-F09
Category:             BACKEND_DESIGN
Engine:               Shared (cross-backend)
Component:            Host-pinned memory
Source File:          ggml/src/ggml-cuda/ggml-cuda.cu, ggml/src/ggml-vulkan/ggml-vulkan.cpp
Function:             ggml_backend_cuda_host_buffer_type, ggml_backend_vk_host_buffer_type
Lines:                ggml-cuda.cu:1257-1321; ggml-vulkan.cpp:15660-15715
Summary:              GPU backends expose a host buffer type backed by cudaMallocHost
                      (or equivalent), with graceful fallback to the CPU buffer type.
Observation:          CUDA's host buffer type calls ggml_cuda_host_malloc,
                      which uses cudaMallocHost and respects the GGML_CUDA_NO_PINNED env
                      var. On failure, it falls back to ggml_backend_cpu_buffer_type
                      (line 1296). The resulting buffer is wrapped via
                      ggml_backend_cpu_buffer_from_ptr, then its buft is repointed
                      to the CUDA host buft and its free_buffer is replaced with
                      cudaFreeHost (line 1300-1301). The CUDA host buffer inherits
                      the CPU buffer's vtable (memcpy-based) but with CUDA-managed
                      free. SYCL, CANN, and Vulkan have equivalent host buffer types.
Evidence:             ggml-cuda.cu:1273-1289 (ggml_cuda_host_malloc);
                      ggml-cuda.cu:1291-1304 (alloc_buffer with fallback);
                      ggml-cuda.cu:1306-1321 (singleton with CPU-inherited
                      get_alignment and is_host);
                      ggml-vulkan.cpp:15671-15715 (Vulkan equivalent).
Architectural Impact: Pinned host memory enables faster PCIe transfers and is
                      required for async copy overlap. The fallback to CPU buffer
                      type means graceful degradation when pinned memory is
                      exhausted. The repointing trick (CPU vtable + CUDA free)
                      reuses the CPU memcpy path without duplicating it.
Correctness Impact:   None. The buffer is correctly freed via cudaFreeHost.
Optimization Type:    Pinned host memory for faster host-device transfer.
GwenLand Target:      glcuda, glvulkan
Recommendation:       ADOPT. Same pattern: pinned alloc with fallback to CPU,
                      inherit CPU vtable, override free_buffer. Respect an
                      env var to disable pinned memory for debugging.
Priority:             High
Difficulty:           S
Dependencies:         R1
Confidence:           High
```

### Finding ARTX21-F10

```
Finding ID:           ARTX21-F10
Category:             MEMORY_PATTERN
Engine:               Shared (cross-backend)
Component:            Cross-backend tensor copy
Source File:          ggml/src/ggml-backend.cpp
Function:             ggml_backend_tensor_copy, ggml_backend_tensor_copy_async
Lines:                477-519
Summary:              Cross-backend copy has three fast paths (host-src, host-dst,
                      cpy_tensor hook) and one slow fallback (malloc staging).
                      Async copy synchronizes both backends if the dst lacks
                      cpy_tensor_async.
Observation:          The sync copy (line 477) checks is_host on src and dst
                      buffers; if either is host, it does a direct memcpy via
                      set_tensor or get_tensor. Otherwise it asks dst's cpy_tensor
                      hook; if that returns false, it mallocs a staging buffer,
                      copies src into it, then copies staging into dst. The async
                      copy (line 500) tries dst's cpy_tensor_async hook first; if
                      absent or it returns false, it synchronizes both backends
                      and does the sync copy. The comment acknowledges this is
                      suboptimal: "to simulate the same behavior, we need to
                      synchronize both backends first."
Evidence:             ggml-backend.cpp:484-487 (host fast paths);
                      ggml-backend.cpp:488-491 (cpy_tensor hook);
                      ggml-backend.cpp:492-497 (slow malloc staging);
                      ggml-backend.cpp:508-518 (async fallback to sync).
Architectural Impact: The host fast paths are critical for hybrid CPU+GPU
                      inference: a copy from a CUDA host-pinned buffer to a CPU
                      buffer is a single memcpy. The slow fallback is a real
                      performance cliff (two copies + malloc). The async fallback
                      that synchronizes both backends is a pipeline-parallelism
                      killer: it stalls the GPU while the CPU does the copy.
Correctness Impact:   None. All four paths produce correct results.
Optimization Type:    Host-visible fast path; backend-owned DMA; async copy
                      via dst-backend hook.
GwenLand Target:      GATE, glcuda
Recommendation:       ADAPT. Keep the three fast paths. REJECT the synchronous
                      fallback in async copy: if the dst backend lacks
                      cpy_tensor_async, enqueue a sync copy on the dst's stream
                      (naturally ordered) and return immediately. Never stall
                      both backends.
Priority:             High
Difficulty:           M
Dependencies:         R5
Confidence:           High
```

### Finding ARTX21-F11

```
Finding ID:           ARTX21-F11
Category:             MEMORY_PATTERN
Engine:               CUDA
Component:            CUDA memory pool (VMM)
Source File:          ggml/src/ggml-cuda/ggml-cuda.cu
Function:             ggml_cuda_pool_vmm, ggml_cuda_pool_leg
Lines:                419-682
Summary:              Two-tier CUDA pool: legacy best-fit free-list (256 slots)
                      and VMM pool with 32 GB VA reserve, cuMemMap, 128-byte
                      alignment, reverse-order free discipline.
Observation:          The legacy pool (ggml_cuda_pool_leg) keeps a 256-slot
                      array of {ptr, size} per device, does best-fit search,
                      over-allocates 5% for look-ahead, and flushes the pool on
                      OOM. The VMM pool (ggml_cuda_pool_vmm) reserves 32 GB of
                      virtual address space up front, allocates physical memory
                      in granularity-aligned chunks via cuMemCreate + cuMemMap,
                      and bumps a pool_used pointer. Free asserts
                      ptr == pool_addr + pool_used - size, enforcing strict
                      stack discipline. The RAII wrapper ggml_cuda_pool_alloc<T>
                      (common.cuh:1167) enforces this via destructor ordering.
Evidence:             ggml-cuda.cu:419-534 (legacy pool);
                      ggml-cuda.cu:536-682 (VMM pool);
                      ggml-cuda.cu:571-572 (128-byte alignment);
                      ggml-cuda.cu:680 (reverse-order assert);
                      ggml-cuda.cu:685-693 (pool selection).
Architectural Impact: The VMM pool amortizes cudaMalloc cost across thousands
                      of per-op workspace allocations. The 128-byte alignment
                      matches CUDA's buffer type alignment, so all tensors are
                      usable by every CUDA kernel. The reverse-order free
                      constraint is a real limitation on kernel authors.
Correctness Impact:   None. The VMM pool is correct; the legacy pool is correct.
                      The reverse-order assert is a debug aid, not a correctness
                      requirement (the pool would still be usable without it,
                      just with fragmentation).
Optimization Type:    Virtual memory management with physical memory mapping;
                      best-fit free-list with OOM flush.
GwenLand Target:      glcuda
Recommendation:       ADOPT the VMM pool design. Keep the legacy pool as
                      fallback. Use the RAII wrapper pattern for all
                      allocations. Document the reverse-order constraint in
                      the glcuda header.
Priority:             High
Difficulty:           L
Dependencies:         R4
Confidence:           High
```

### Finding ARTX21-F12

```
Finding ID:           ARTX21-F12
Category:             BACKEND_DESIGN
Engine:               Shared (cross-backend)
Component:            Tensor extra field
Source File:          ggml/include/ggml.h, ggml/src/ggml-cuda/common.cuh, ggml/src/ggml-sycl/ggml-sycl.cpp
Function:             ggml_tensor.extra, ggml_tensor_extra_gpu
Lines:                ggml.h:702; common.cuh:1213-1216; ggml-sycl.cpp:562, 1113
Summary:              The extra field is a backend-specific void* on every
                      tensor; its use is inconsistent across backends (SYCL
                      populates it; CUDA declares the struct but does not).
Observation:          ggml.h:702 declares `void * extra; // extra things e.g.
                      for ggml-cuda.cu`. The CUDA backend declares
                      ggml_tensor_extra_gpu (common.cuh:1213) with per-device
                      data pointers and per-stream events, but a grep for
                      `extra =` in ggml-cuda/ finds no assignment sites —
                      the struct is dead code in the CUDA path. CUDA tracks
                      split tensors via ggml_backend_cuda_split_buffer_context
                      instead. The SYCL backend still uses extra
                      (ggml-sycl.cpp:562, 1113). OpenVINO uses it for vendor
                      IR handles; OpenCL uses it for per-quant-format metadata.
                      The misleading comment "extra things e.g. for ggml-cuda.cu"
                      suggests CUDA uses it, which is no longer true.
Evidence:             ggml.h:702 (extra field);
                      common.cuh:1213-1216 (ggml_tensor_extra_gpu declaration);
                      ggml-sycl.cpp:562, 1113 (SYCL usage);
                      openvino (grep result);
                      opencl (grep result);
                      (no assignment in ggml-cuda/).
Architectural Impact: Cross-backend code cannot rely on extra carrying any
                      specific structure. A new backend author reading the
                      comment will be misled. The field is a backdoor for
                      backend-specific state that bypasses the buffer
                      interface, which is acceptable for vendor handles but
                      should be documented.
Correctness Impact:   None. The field is opaque; misuse is a backend bug, not
                      a shared-layer bug.
Optimization Type:    None.
GwenLand Target:      multiple
Recommendation:       ADAPT. Keep the extra field but require each backend to
                      document its layout in a header (e.g.,
                      glcuda_extra.h defines gl_tensor_extra_cuda). Fix the
                      misleading comment. Consider adding an extra_kind enum
                      tag for type safety.
Priority:             Medium
Difficulty:           S
Dependencies:         R1
Confidence:           High
```

### Finding ARTX21-F13

```
Finding ID:           ARTX21-F13
Category:             EXECUTION_GRAPH
Engine:               Shared (cross-backend)
Component:            Scheduler-side copy with pipeline events
Source File:          ggml/src/ggml-backend.cpp
Function:             ggml_backend_sched_compute_splits
Lines:                1541-1725
Summary:              Scheduler copies split inputs with per-(backend, copy)
                      event synchronization, enabling pipeline parallelism
                      across consecutive batches.
Observation:          The scheduler maintains events[GGML_SCHED_MAX_BACKENDS][GGML_SCHED_MAX_COPIES=4]
                      (line 807). For each split, it copies inputs: if the
                      input is a user FLAG_INPUT, it synchronizes the dst
                      event first then does a sync copy; otherwise it waits
                      on the dst event and tries async copy via
                      cpy_tensor_async, falling back to sync. After the split
                      computes, an event is recorded on the split backend.
                      The cur_copy counter rotates per batch, so batch N+1
                      uses a different copy slot than batch N, enabling
                      overlap. The MoE expert-only copy (line 1576-1660)
                      further reduces transfer by reading the MUL_MAT_ID ids
                      tensor and copying only used experts.
Evidence:             ggml-backend.cpp:807 (events array);
                      ggml-backend.cpp:1555-1575 (input copy with event sync);
                      ggml-backend.cpp:1664-1672 (async copy fallback);
                      ggml-backend.cpp:1717-1721 (event record after split);
                      ggml-backend.cpp:1576-1660 (MoE expert copy);
                      ggml-backend.cpp:1869-1870 (cur_copy rotation).
Architectural Impact: Pipeline parallelism across batches is the single biggest
                      throughput win for multi-backend inference. The event
                      discipline (record after split, wait before next split's
                      input copy) is correct and minimal. The MoE expert copy
                      is a domain-specific optimization that can save GBs of
                      transfer per batch for large MoE models.
Correctness Impact:   None. The event synchronization ensures correctness;
                      without it, batch N+1 could overwrite batch N's input
                      before the GPU finished reading it.
Optimization Type:    Asynchronous execution via per-backend event slots;
                      pipeline parallelism via copy rotation.
GwenLand Target:      GATE, glcuda
Recommendation:       ADOPT. Same event discipline; same copy rotation. The
                      MoE expert copy is worth adopting if GwenLand targets
                      MoE models.
Priority:             High
Difficulty:           M
Dependencies:         R5, R8
Confidence:           High
```

---

## 18. Unknowns

* **U1**. Whether the `O(N²)` best-fit scan in `ggml_dyn_tallocr_alloc` is a measurable bottleneck for typical llama.cpp workloads. The free-list is capped at 256 entries per chunk and coalescing keeps it small in practice, but no profiling data is in the source. Requires runtime measurement.

* **U2**. Whether `TENSOR_ALIGNMENT = 32` causes measurable performance loss for AVX-512 kernels. The CPU backend uses unaligned loads, documented as "almost as fast" on Skylake-X and later, but measurements vary by microarchitecture. Requires differential benchmarking.

* **U3**. Whether the `ggml_tensor_extra_gpu` struct in `ggml-cuda/common.cuh:1213` is truly dead code in the CUDA path or whether it is populated by a code path the static grep missed (e.g., via a macro or template instantiation). The SYCL backend uses it; CUDA may have removed it in a recent refactor without cleaning up the declaration.

* **U4**. Whether the `n_copies = 4` pipeline parallelism scheme actually overlaps batches in practice. The event discipline is correct, but overlap requires the GPU to have spare SM capacity while input copies run. On a fully-utilized GPU, the copies serialize. Requires profiling on the target GPU.

* **U5**. Whether the MoE expert-only copy optimization (`ggml-backend.cpp:1576-1660`) is a net win for small MoE models where the ids tensor readback cost may exceed the saved expert copies. The optimization is unconditional for `MUL_MAT_ID` with host-visible weights; there is no heuristic to skip it for small expert counts.

* **U6**. Whether the `vbuffer` chunking mechanism (`GGML_VBUFFER_MAX_CHUNKS=16`) is ever exercised in practice. CUDA's max_size is SIZE_MAX, so it never chunks. A backend with a per-buffer cap would chunk, but no such backend exists in the audited tree. The mechanism may be dead code.

---

## 19. References

| Reference | File                                       | Function / Symbol                                         | Lines         |
| --------- | ------------------------------------------ | --------------------------------------------------------- | ------------- |
| R01       | `ggml/src/ggml-alloc.c`                    | `ggml_op_can_inplace`                                     | 22–50         |
| R02       | `ggml/src/ggml-alloc.c`                    | `aligned_offset`, `ggml_tallocr_new/_alloc`               | 52–91         |
| R03       | `ggml/src/ggml-alloc.c`                    | `ggml_dyn_tallocr_alloc` (best-fit + coalesce)            | 201–349       |
| R04       | `ggml/src/ggml-alloc.c`                    | `struct vbuffer`, `ggml_vbuffer_alloc`                    | 397–449       |
| R05       | `ggml/src/ggml-alloc.c`                    | `struct hash_node`, `struct ggml_gallocr`, `ggml_gallocr_new_n` | 458–531  |
| R06       | `ggml/src/ggml-alloc.c`                    | `ggml_gallocr_allocate_node` (in-place reuse)             | 622–688       |
| R07       | `ggml/src/ggml-alloc.c`                    | `ggml_gallocr_alloc_graph_impl` (lifetime walk)           | 717–822       |
| R08       | `ggml/src/ggml-alloc.c`                    | `ggml_gallocr_reserve_n_impl`, `_alloc_graph` (two-phase) | 824–1097      |
| R09       | `ggml/src/ggml-alloc.c`                    | `ggml_backend_alloc_ctx_tensors_from_buft_impl`           | 1167–1229     |
| R10       | `ggml/src/ggml-backend-impl.h`             | `struct ggml_backend_buffer_type_i`                       | 17–29         |
| R11       | `ggml/src/ggml-backend-impl.h`             | `struct ggml_backend_buffer_i`                            | 41–62         |
| R12       | `ggml/src/ggml-backend-impl.h`             | `struct ggml_backend_i` (async + events), `ggml_backend_device_i` | 105–202 |
| R13       | `ggml/src/ggml-backend.cpp`                | `ggml_backend_buft_get_*`, `ggml_backend_tensor_set/_get/_2d` | 47–396   |
| R14       | `ggml/src/ggml-backend.cpp`                | `ggml_backend_tensor_copy` / `_async` (host fast paths + sync fallback) | 477–519 |
| R15       | `ggml/src/ggml-backend.cpp`                | `ggml_backend_multi_buffer_*`, `ggml_backend_sched_compute_splits` | 667–735, 1541–1725 |
| R16       | `ggml/src/ggml-backend.cpp`                | `ggml_backend_view_init`, `ggml_backend_tensor_alloc`     | 1980–2005     |
| R17       | `ggml/src/ggml-backend.cpp`                | `ggml_backend_cpu_buffer_i`/`_type`/`_from_ptr`           | 2267–2371     |
| R18       | `ggml/src/ggml-impl.h`                     | `TENSOR_ALIGNMENT = 32`, `ggml_are_same_layout`, `ggml_impl_is_view` | 44, 75–105 |
| R19       | `ggml/src/ggml-impl.h`                     | `struct ggml_hash_set`, `ggml_hash_find_or_insert`        | 226–308       |
| R20       | `ggml/src/ggml.c`                          | `ggml_aligned_malloc` / `_free`, `ggml_nbytes_pad`, `ggml_tensor_overhead`, `ggml_new_object` | 331–402, 1322–1324, 1465–1467, 1707–1759 |
| R21       | `ggml/include/ggml.h`                      | `struct ggml_tensor`, `GGML_PAD`, `GGML_MEM_ALIGN`        | 236–267, 673–705 |
| R22       | `ggml/include/ggml-alloc.h`                | `ggml_tallocr`, `ggml_gallocr` public API                 | 13–72         |
| R23       | `ggml/src/ggml-cuda/ggml-cuda.cu`          | `ggml_backend_cuda_buffer_type_get_alignment` (128), `ggml_cuda_pool_leg`, `ggml_cuda_pool_vmm`, `ggml_backend_cuda_host_buffer_type` | 419–534, 536–682, 900–904, 1257–1321 |
| R24       | `ggml/src/ggml-cuda/common.cuh`            | `ggml_cuda_pool` (abstract), `ggml_cuda_pool_alloc<T>` (RAII), `ggml_tensor_extra_gpu` | 1159–1216 |
| R25       | `ggml/src/ggml-sycl/ggml-sycl.cpp`         | `ggml_tensor_extra_gpu` usage (SYCL populates extra)      | 562, 1113     |
| R26       | `ggml/src/ggml-vulkan/ggml-vulkan.cpp`     | `ggml_backend_vk_host_buffer_type` (pinned)               | 15660–15715   |
| R27       | `ggml/src/ggml-backend-meta.cpp`           | `ggml_backend_meta_buffer_type_*` (multi-device wrapper)  | 276–369       |
