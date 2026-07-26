# ARTX15 — Metal Backend Core

**Audited commit:** `555881ebc8b0fc0402b30e09258a32a7bfd13c52`
**Last updated:** 2026-07-25
**Auditor:** Percival-aux (ARTX15)
**Target GwenLand module:** `glmetal` (Metal backend core), `GATE` (graph execution)

---

## 1. Executive Summary

The Metal backend of llama.cpp is a *single-process, multi-command-buffer* GPU backend that plugs into ggml's three-tier backend registry. Compared to the CPU backend (ARTX01), it is **async, event-aware, and able to share host memory zero-copy on Apple Silicon**. It owns no kernels itself; instead, it provides:

1. A **backend / device / registry vtable triad** advertising `async=true`, `events=true`, `buffer_from_host_ptr=true`, `host_buffer=false`.
2. A **single shared `MTLCommandQueue` per device** plus a pool of `MTLCommandBuffer`s per backend context (one per parallel encoder + one for the main thread + an unbounded list of "extra" buffers for async transfers).
3. A **graph executor** (`ggml_metal_graph_compute`) that splits the node list into a main-thread prefix and `n_cb` parallel encoder slices.
4. A **per-op dispatch switch** (`ggml_metal_op_encode_impl`) of ~50 cases, each selecting a Metal pipeline state from a lazily-compiled, name-keyed cache.
5. A **kernel-specialization system** built on `MTLFunctionConstantValues`: each (op, dtype, shape-summary) tuple gets its own compiled pipeline, cached under a string name.
6. A **plan-time graph optimizer** (`ggml_graph_optimize`) that fuses ADD/MUL/NORM chains and reorders nodes for concurrent dispatch.
7. A **memory-residency-set manager** (macOS 15+ / iOS 18+) that pins large weight buffers and keeps them wired via a 5 ms background heartbeat.
8. An **op-fusion engine** at encode time that fuses ADD chains (up to 8), NORM+MUL+ADD, and the 5-op Snake activation pattern.

For GwenLand, the decisions worth **ADOPT**ing are: the lazy per-shape pipeline cache keyed by string name, the `MTLFunctionConstantValues` specialization pattern, the `n_cb` parallel encoder scheme, the memory-range overlap tracker for concurrency, and the residency-set keep-alive pattern. The decisions worth **REJECT**ing are the `MTLCreateSystemDefaultDevice` choice (which silently ignores Mac Pro multi-GPU) and the hardcoded `ne11_mm_min = 8` GEMV-vs-GEMM cutoff.

---

## 2. Purpose

Provide a Metal execution backend for the ggml graph that:

* dispatches every supported `ggml_op` to a Metal compute kernel,
* parallelizes graph encoding (not execution) across `n_cb` host
  threads, each emitting into its own `MTLCommandBuffer`,
* defers to Metal's own GPU scheduling for in-flight parallelism,
* advertises `async=true`, `events=true`, and
  `buffer_from_host_ptr=true` so the ggml scheduler can treat Metal
  as a peer to CUDA/Vulkan,
* detects Apple-Silicon feature families (Apple7 / Metal3 / Metal4)
  and gates ops, dtypes, and the experimental tensor API on them,
* supports zero-copy host buffer mapping for unified-memory devices,
* supports discrete GPUs via `MTLResourceStorageModePrivate` buffers
  and `MTLBlitCommandEncoder` for host↔device transfers,
* exposes a `MTLSharedEvent`-based event system for cross-backend
  synchronization (`cpy_tensor_async` between two Metal contexts).

It is **not** responsible for: graph construction, graph allocation
(delegated to `ggml-alloc.c`), or backend selection across backends
(delegated to the scheduler — ARTX22). The Metal backend does own
graph *optimization* (reorder + fuse), invoked from
`ggml_metal_graph_optimize` and gated by the
`GGML_METAL_GRAPH_OPTIMIZE_DISABLE` env var.

---

## 3. Source Files

| File                                                     | Lines  | Role                                                              |
| -------------------------------------------------------- | ------ | ---------------------------------------------------------------- |
| `ggml/src/ggml-metal/ggml-metal.cpp`                     | 951    | Backend / device / registry vtables; buffer-type factories       |
| `ggml/src/ggml-metal/ggml-metal-context.h`               | 41     | Public C interface to `ggml_metal_t` (context)                   |
| `ggml/src/ggml-metal/ggml-metal-context.m`               | 740    | `ggml_metal_t` implementation: graph compute, async, events     |
| `ggml/src/ggml-metal/ggml-metal-device.h`                | 326    | Public C interface to `ggml_metal_device_t`, pipeline & encoder  |
| `ggml/src/ggml-metal/ggml-metal-device.m`                | 1917   | Objective-C: MTLDevice, MTLCommandQueue, MTLLibrary, buffers     |
| `ggml/src/ggml-metal/ggml-metal-device.cpp`              | 2161   | C++: pipeline-name → pipeline-state cache + ~60 pipeline selectors|
| `ggml/src/ggml-metal/ggml-metal-ops.h`                   | 100    | Per-op encoder function declarations + per-op extra-size helpers |
| `ggml/src/ggml-metal/ggml-metal-ops.cpp`                 | 4864   | Per-op encoder functions + ~50-case dispatch switch              |
| `ggml/src/ggml-metal/ggml-metal-common.cpp`              | 457    | `ggml_mem_ranges` overlap tracker + plan-time graph optimizer    |
| `ggml/src/ggml-metal/ggml-metal-impl.h`                  | 1222   | Function-constant offsets, per-op kernel-args structs, blck sizes|
| `ggml/src/ggml-metal/ggml-metal.metal`                   | 11218  | All Metal kernels (cross-reference only; audited in ARTX16/17)   |

> **Structural note (vs ARTX08 CUDA analog).** The Metal backend has
> been refactored across the audited commit into five `.cpp/.m` files
> plus a shared `.metal` source. The split follows a clean
> device/library/ops/common boundary, with Objective-C (`.m`) owning
> all `MTL*` object references and C++ (`.cpp`) owning pipeline-name
> selection and op encoding. The only `.h` exported to the rest of
> ggml is `ggml-metal.h` (in `ggml/src/`); the rest are internal.

---

## 4. Architecture Overview

```
       ┌─────────────────────────────────────────────────────────┐
       │  ggml-metal.cpp                                         │
       │  ├─ ggml_backend_metal_reg_i   (registry vtable)        │
       │  ├─ ggml_backend_metal_device_i (device vtable)         │
       │  └─ ggml_backend_metal_i       (backend vtable)         │
       │  (plugs Metal into ggml backend registry)               │
       └─────────────────────────────────────────────────────────┘
                              │
                              ▼
       ┌─────────────────────────────────────────────────────────┐
       │  ggml-metal-device.{m,cpp}                              │
       │  ├─ ggml_metal_device_t   (MTLDevice + MTLCommandQueue) │
       │  ├─ ggml_metal_library_t  (MTLLibrary + pipeline cache) │
       │  ├─ ggml_metal_buffer_t   (1..64 MTLBuffer wrappers)    │
       │  ├─ ggml_metal_event_t    (MTLSharedEvent + atomic)     │
       │  └─ ggml_metal_rsets_t    (residency sets + bg thread)  │
       └─────────────────────────────────────────────────────────┘
                              │
                              ▼
       ┌─────────────────────────────────────────────────────────┐
       │  ggml-metal-context.m                                   │
       │  ├─ struct ggml_metal (per-backend state)               │
       │  ├─ cmd_bufs[GGML_METAL_MAX_COMMAND_BUFFERS + 1]        │
       │  ├─ cmd_bufs_ext (NSMutableArray of async transfers)    │
       │  ├─ ggml_metal_graph_compute                            │
       │  └─ async cpy/get/set via BlitCommandEncoder            │
       └─────────────────────────────────────────────────────────┘
                              │
                              ▼
       ┌─────────────────────────────────────────────────────────┐
       │  ggml-metal-ops.cpp                                     │
       │  ├─ ggml_metal_op_encode_impl  (switch on node->op)     │
       │  ├─ ggml_metal_op_mul_mat     (GEMV/GEMM/ext heuristic) │
       │  ├─ ggml_metal_op_flash_attn_ext (vec vs half8x8)       │
       │  ├─ ggml_metal_op_bin / norm / rope / cpy / ...         │
       │  └─ ggml_metal_op_can_fuse_snake / bin×N / norm+mul+add │
       └─────────────────────────────────────────────────────────┘
                              │
                              ▼
       ┌─────────────────────────────────────────────────────────┐
       │  ggml-metal-common.cpp                                  │
       │  ├─ ggml_mem_ranges (per-encoder overlap tracker)       │
       │  └─ ggml_graph_optimize (plan-time fuse + reorder)      │
       └─────────────────────────────────────────────────────────┘
                              │
                              ▼
       ┌─────────────────────────────────────────────────────────┐
       │  ggml-metal.metal (compiled into MTLLibrary at init)    │
       │  ~120 kernel_* functions, function-constant specialized │
       └─────────────────────────────────────────────────────────┘
```

Key design points:

* **Pure C interface, Objective-C implementation**. All `MTL*` types
  are held as `void *` or `id` in opaque structs behind the C
  boundary; the C++ side never touches an Objective-C object
  directly. This is the inverse of the CUDA backend, where
  `cuda.h` is itself C.
* **Single shared `MTLCommandQueue` per device**
  (`ggml-metal-device.m:526`). All backends on the same device
  enqueue command buffers into this one queue. The comment at
  `ggml-metal-context.m:103-105` notes an unresolved TODO about
  whether to give each backend its own queue.
* **`MTLComputeCommandEncoder` with optional `MTLDispatchTypeConcurrent`**
  (`ggml-metal-device.m:468-472`). When `use_concurrency` is true,
  the encoder is created with the concurrent dispatch type, allowing
  Metal to overlap independent dispatches inside one command buffer.
  A memory-range overlap tracker (`ggml_mem_ranges`,
  `ggml-metal-common.cpp:8-185`) decides per-node whether a barrier
  is needed.
* **Lazy pipeline compilation with name-keyed cache**
  (`ggml-metal-device.m:349-453`). The first time a particular
  (op, dtype, shape-summary) tuple is needed, the library compiles
  the kernel with the appropriate `MTLFunctionConstantValues` and
  caches the resulting `MTLComputePipelineState` under a string
  name. Subsequent calls are a `std::unordered_map` lookup under
  `NSLock`.
* **Three buffer types, all on one device**
  (`ggml-metal.cpp:188-474`). Shared (`MTLResourceStorageModeShared`
  on unified memory), Private (`MTLResourceStorageModePrivate` on
  discrete), and Mapped (host pointer wrapped with
  `newBufferWithBytesNoCopy`). No split buffer type (single-GPU
  assumption). `get_host_buffer_type` is `NULL`.
* **`MTLSharedEvent` for cross-context sync** (`ggml-metal-device.m:989-1039`).
  Each context owns one `ev_cpy` shared event for `cpy_tensor_async`
  between two Metal backends. The signal/wait pair is encoded into
  the source and destination command buffers respectively.

---

## 5. Execution Flow

### 5.1 Top-level entry

`ggml_backend_metal_graph_compute` (`ggml-metal.cpp:534-538`) is a
one-line forwarder to `ggml_metal_graph_compute`
(`ggml-metal-context.m:438-615`).

### 5.2 Graph compute

`ggml_metal_graph_compute` (`ggml-metal-context.m:438-615`):

1. If `ctx->has_error` is set, return `GGML_STATUS_FAILED` — the only recovery is backend recreate.
2. Compute `n_main = MAX(64, 0.1 * gf->n_nodes)`; first `n_main` nodes go to the main thread, the rest split across `n_cb` workers as `n_nodes_per_cb = ceil(n_nodes_1 / n_cb)` chunks.
3. Reset the residency-set keep-alive countdown (`ggml-metal-device.m:981-987`).
4. Optionally start an `MTLCaptureScope` writing `/tmp/perf-metal-<pid>.gputrace`.
5. Allocate `n_cb + 1` `MTLCommandBuffer`s via `[queue commandBufferWithUnretainedReferences]` + manual `[retain]`. Main thread's buffer is at `cmd_bufs[n_cb]`; workers at `cmd_bufs[0..n_cb)`.
6. `[cmd_buf enqueue]` the main buffer and the first two worker buffers (places them in the queue without committing, so the GPU can start while encoding continues). The rest are committed only when needed (abort path).
7. Build the `encode_async` Objective-C block (`ggml-metal-context.m:676-721`), call it synchronously with `n_cb` (main thread), then `dispatch_apply(n_cb, ctx->d_queue, encode_async)` to fan out the workers on a concurrent GCD queue.
8. Each encoder commits its buffer at end of block (unless an abort callback is installed).
9. Unless capturing, the function returns immediately — `cmd_buf_last` is remembered for the next `synchronize`.

### 5.3 Per-encoder op dispatch

`ggml_metal_op_encode` (`ggml-metal-ops.cpp:507-523`) wraps `ggml_metal_op_encode_impl` with optional `pushDebugGroup`/`popDebugGroup` for capture. `ggml_metal_op_encode_impl` (lines 175–505):

1. If the node is `NONE`/`RESHAPE`/`VIEW`/`TRANSPOSE`/`PERMUTE`: no-op, return 1.
2. Validate `ggml_metal_device_supports_op` — aborts if unsupported.
3. Skip if `GGML_TENSOR_FLAG_COMPUTE` is clear.
4. **Concurrency check**: ask `ggml_mem_ranges_check` whether this node's src/dst ranges overlap with any prior range. If yes, `memoryBarrierWithScope:MTLBarrierScopeBuffers` + reset tracker.
5. **Big switch on `node->op`** dispatches to ~50 per-op encoders. Each returns `n_fuse` (>=1) so the caller skips the next `n_fuse - 1` nodes.
6. Add the (fused) node's src/dst ranges to the tracker.

### 5.4 Per-op encoder pattern (mul_mat as exemplar)

`ggml_metal_op_mul_mat` (`ggml-metal-ops.cpp:2043-2273`):

1. Validate shapes: `ne00 == ne10`, `ne12 % ne02 == 0`,
   `ne13 % ne03 == 0`. Compute `r2 = ne12/ne02`, `r3 = ne13/ne03`.
2. **Three-way heuristic**:
   * **`mul_mv_ext`** (small-batch mat-mv): if `src1->type == F32`,
     `ne00 % 128 == 0`, and `ne11` is in `[2, 8]` (or `[4, 8]` for
     K-quants). Pick `nsg=2`, `nxpsg` from `{16, 8, 4}` based on
     `ne00` alignment and `ne11`, `r1ptg` from a per-`ne11` table.
     Specialize via
     `ggml_metal_library_get_pipeline_mul_mv_ext(lib, op, nsg, nxpsg, r1ptg)`.
   * **`mul_mm`** (matrix-matrix): if `has_simdgroup_mm`,
     `ne00 >= 64`, `ne11 > ne11_mm_min (=8)`, and neither operand is
     transposed. Specialize via
     `ggml_metal_library_get_pipeline_mul_mm`. Set
     `threadgroup_memory_size` from `pipeline.smem`.
   * **`mul_mv`** (matrix-vector fallback): everything else. Pick
     `nsg` / `nr0` / `smem` from per-dtype constants in
     `ggml-metal-impl.h` (e.g. `N_SG_Q4_0 = 2`, `N_R0_Q4_0 = 4`).
3. Set pipeline, set bytes (args struct), set buffers (src0, src1,
   dst at indices 1, 2, 3), set threadgroup memory size, dispatch
   `((ne01 + nr0 - 1) / nr0, (ne11 + nr1 - 1) / nr1, ne12 * ne13)`
   threadgroups with `32 × nsg × 1` threads per threadgroup.

### 5.5 Async transfer path

`ggml_metal_set_tensor_async` (`ggml-metal-context.m:307-349`) allocates a temporary shared `MTLBuffer` from `data` via `newBufferWithBytes:` (a **copy**), looks up the destination `MTLBuffer` via `ggml_metal_buffer_get_id`, encodes a `MTLBlitCommandEncoder` copy, commits, and adds the cmd_buf to `ctx->cmd_bufs_ext`. Does not wait.

`ggml_metal_get_tensor_async` (`ggml-metal-context.m:351-393`) is the mirror: wraps `data` (the destination host pointer) with `newBufferWithBytesNoCopy` (zero-copy).

`ggml_metal_cpy_tensor_async` (`ggml-metal-context.m:395-436`) is the cross-context variant: encodes a Blit copy in the source queue, signals `ctx_src->ev_cpy` (a `MTLSharedEvent`), then encodes a wait on the same event in the destination queue. This is the only path that uses the event system from within the backend.

---

## 6. Data Layout

### 6.1 Tensor descriptor

Same `ggml_tensor` shape (`ne[]`, `nb[]`) as CPU/CUDA. Metal
requires, for the matmul path:

| Constraint                                            | Source                              |
| ----------------------------------------------------- | ----------------------------------- |
| `ne00 == ne10`                                        | `ggml-metal-ops.cpp:2058`           |
| `ne12 % ne02 == 0`, `ne13 % ne03 == 0`                | `ggml-metal-ops.cpp:2060-2061`      |
| `!ggml_is_transposed(src[0])` and `src[1]` for mul_mm | `ggml-metal-ops.cpp:2173-2174`      |
| `ne11 <= INT16_MAX`, `ne13 <= INT16_MAX`              | `ggml-metal-device.cpp:722-725`     |
| `nb01` aligned for Metal matrix types (asserted off)  | `ggml-metal-ops.cpp:2182-2187`      |

The matmul kernel `r2 = ne12/ne02` and `r3 = ne13/ne03` "broadcast
factors" are passed as `int16_t` function constants, hence the
`INT16_MAX` cap. Larger broadcasts would silently truncate.

### 6.2 Buffer view lookup

`ggml_metal_buffer_get_id` (`ggml-metal-device.m:1893-1916`)
implements a 1-to-N linear search over the buffer's wrappers:

```c
for (i = 0; i < buf->n_buffers; i++) {
    ioffs = (uintptr_t) t->data - (uintptr_t) buf->buffers[i].data;
    if (ioffs >= 0 && ioffs + tsize <= buf->buffers[i].size) {
        return { buf->buffers[i].metal, ioffs };
    }
}
```

For shared buffers `n_buffers == 1`. For mapped buffers from
`ggml_metal_buffer_map`, `n_buffers` can be up to
`GGML_METAL_MAX_BUFFERS = 64` (`ggml-metal-device.m:1386`) when the
host allocation exceeds `max_buffer_size` — the wrapper array is
overlapped so any tensor fits in exactly one view.

### 6.3 Kernel args structs

Every per-op encoder fills a `ggml_metal_kargs_*` struct
(declared in `ggml-metal-impl.h:158-1222`) and calls
`ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0)`. The
struct is laid out to match the kernel's `constant` buffer slot 0.
Element counts are `int32_t` (to save registers); strides are
`uint64_t`. This is the Metal equivalent of CPU's
`ggml_compute_params` but per-op rather than global.

---

## 7. Memory Layout

### 7.1 Buffer allocation (`ggml_metal_buffer_init`)

`ggml-metal-device.m:1515-1583`:

* Page-align `size` to `sysconf(_SC_PAGESIZE)`.
* If `shared` (and the device supports it): allocate host memory
  via `vm_allocate` (macOS) or `posix_memalign` (iOS), then wrap
  with `[device newBufferWithBytesNoCopy:length:options:MTLResourceStorageModeShared]`.
  This is **zero-copy on Apple Silicon**.
* Else (private): allocate a fake virtual host address
  (`atomic_fetch_add(&dev->addr_virt, size_aligned)`, starting at
  `0x400` — `ggml-metal-device.m:693`) for bookkeeping, and wrap
  with `[device newBufferWithLength:length:options:MTLResourceStorageModePrivate]`.
  The GPU owns the memory; host access requires a Blit copy.
* Optionally attach an `MTLResidencySet` (`ggml-metal-device.m:1443-1493`)
  and register it with the device's residency-set collection.

### 7.2 Buffer mapping (`ggml_metal_buffer_map`)

`ggml-metal-device.m:1585-1678`:

* Page-align `ptr` downward.
* If the size fits in one buffer: single
  `newBufferWithBytesNoCopy` with `MTLResourceStorageModeShared`.
* Else: split into overlapping views of `max_buffer_size` with
  `size_ovlp = ceil(max_tensor_size / page) + 2 pages` overlap, so
  any single tensor fits entirely in one view. Iterate in
  `size_step = max_buffer_size - size_ovlp` increments.

The overlap is the structural answer to Metal's per-buffer size cap
(typically 1 GiB on iOS, much larger on macOS but still finite).

### 7.3 Per-context command-buffer state

`struct ggml_metal` (`ggml-metal-context.m:26-82`) holds:

| Field                         | Purpose                                                      |
| ----------------------------- | ------------------------------------------------------------ |
| `cmd_bufs[GGML_METAL_MAX_COMMAND_BUFFERS + 1]` | 1 main + up to 8 worker command buffers      |
| `cmd_bufs_ext` (`NSMutableArray`) | Unbounded list of async transfer buffers                  |
| `cmd_buf_last`                | Last queued command buffer, used by `synchronize`            |
| `n_nodes_0`, `n_nodes_1`, `n_nodes_per_cb` | Per-encoder node-range split                    |
| `encode_async` (Objective-C block) | Per-`n_cb` encoder closure, rebuilt when `n_cb` changes |
| `ev_cpy`                      | `MTLSharedEvent` for cross-context `cpy_tensor_async`        |
| `d_queue` (`dispatch_queue_t`) | Concurrent GCD queue for `dispatch_apply` of encoders       |
| `pipelines_ext`               | Runtime-compiled pipelines (currently unused — placeholder)  |
| `has_error`                   | Sticky error flag set on command-buffer failure              |

### 7.4 Library pipeline cache

`struct ggml_metal_library` (`ggml-metal-device.m:97-104`) holds:

* `id<MTLLibrary> obj` — the compiled Metal library (from embedded
  source, `default.metallib`, or `ggml-metal.metal` source).
* `ggml_metal_pipelines_t pipelines` —
  `std::unordered_map<std::string, ggml_metal_pipeline_t>`
  (`ggml-metal-device.cpp:28-30`).
* `NSLock * lock` — serializes compile_pipeline calls.

The cache is per-library, and the library is per-device, so cache
hits are shared across all backends on the same device.

### 7.5 Residency-set collection

`struct ggml_metal_rsets` (`ggml-metal-device.m:542-558`):

* `NSMutableArray * data` — non-owning list of `MTLResidencySet`s
  (one per buffer).
* `keep_alive_s = 3*60` (default), `time_per_loop_ms = 5`,
  `loops_per_s = 200`.
* `atomic_int d_loop` — countdown, reset to
  `loops_per_s * keep_alive_s` on every `graph_compute`.
* `dispatch_group_t d_group` — background thread that calls
  `[rset requestResidency]` for every set in the list every 5 ms
  while `d_loop > 0`.

The collection is created at device init
(`ggml-metal-device.m:866-870`), gated by `props.use_residency_sets`
(macOS 15+ / iOS 18+).

---

## 8. Parallelism Strategy

### 8.1 Two-level parallelism

The Metal backend exploits two levels of parallelism:

1. **Host-side encoder parallelism** (n_cb). The main thread encodes
   the first `MAX(64, 0.1 * n_nodes)` nodes; up to
   `GGML_METAL_MAX_COMMAND_BUFFERS = 8` worker threads encode the
   rest in parallel, each into its own command buffer. The default
   `n_cb` is 1 (`ggml-metal.cpp:611`); the code warns that
   `n_cb > 2` "is not recommended and can degrade the performance
   in some cases" (`ggml-metal-context.m:667-669`).
2. **GPU-side dispatch parallelism** (concurrent encoder). Within
   each command buffer, the `MTLComputeCommandEncoder` is created
   with `MTLDispatchTypeConcurrent` (when `use_concurrency` is
   true), letting Metal overlap independent dispatches. A
   `memoryBarrierWithScope:MTLBarrierScopeBuffers` is inserted
   only when the per-node overlap tracker detects a conflict
   (`ggml-metal-ops.cpp:147-173`).

### 8.2 Concurrency overlap tracker

`ggml_mem_ranges` (`ggml-metal-common.cpp:19-23`) is a per-encoder
`std::vector<ggml_mem_range>` of `(buffer_ptr, [p0, p1), src|dst)`
tuples. For each new node:

* Add all `src[i]` ranges as `MEM_RANGE_TYPE_SRC`.
* Add the `dst` range as `MEM_RANGE_TYPE_DST`.
* Check before adding: if any prior range overlaps, return false →
  caller inserts a memory barrier and resets the tracker.

The check (`ggml_mem_ranges_check`, lines 124-153) is O(N) per node,
so total overhead is O(N²) per graph. For typical llama.cpp graphs
(N ≈ 1000–3000), this is 10⁶ comparisons — negligible on the host
but a non-zero CPU cost.

### 8.3 Plan-time graph reorder

`ggml_graph_optimize` (`ggml-metal-common.cpp:375-457`) runs at
graph-plan time (called from `ggml_metal_graph_optimize`):

1. Pack fusable ADD/MUL/NORM chains into `node_info` structs (so
   reorder doesn't break fusion).
2. For each node, look forward up to `N_FORWARD = 64` nodes for any
   node that can run concurrently with the current concurrent set
   (and with all unprocessed nodes in between). Reorder it to run
   earlier.
3. Only a whitelist of ops (`h_safe`, lines 259-291) is eligible
   for reorder: `MUL_MAT`, `MUL_MAT_ID`, `ROPE`, `NORM`,
   `RMS_NORM`, `GROUP_NORM`, `L2_NORM`, `SUM_ROWS`, `SSM_CONV`,
   `SSM_SCAN`, `CLAMP`, `TRI`, `DIAG`, `MUL`, `ADD`, `SUB`, `DIV`,
   `GLU`, `SCALE`, `UNARY`, `GET_ROWS`, `SET_ROWS`, `SET`, `CPY`,
   `CONT`, `REPEAT`. Everything else blocks reorder.
4. Unfuse and write back the reordered node list.

This is a **plan-time** counterpart to the **encode-time**
overlap tracker. Together they maximize concurrent dispatch
opportunities.

### 8.4 Per-op threadgroup sizing

Each encoder picks its own threadgroup size: elementwise ops use `nth = min(256, ne0)` with `nrptg = 256 / nth` rows per threadgroup; reductions use `nth = 32` (one simdgroup), `smem = 32 * sizeof(float)`; `mul_mm` uses `32 × 4 × 1`; `mul_mv` uses `32 × nsg × 1` (nsg from per-dtype table, 2–4); flash attention uses `32 × nsg × 1` with `nsg = ne00 >= 512 ? 8 : 4`. There is no autotuner.

### 8.5 Async & events

The backend advertises `async=true`, `events=true`
(`ggml-metal.cpp:679-684`). The async APIs are:

| API                        | Implementation                            | Wait?                    |
| -------------------------- | ----------------------------------------- | ------------------------ |
| `set_tensor_async`         | Blit copy, host→shared MTLBuffer→dst      | No (deferred to sync)    |
| `get_tensor_async`         | Blit copy, src→NoCopy host MTLBuffer      | No                       |
| `cpy_tensor_async`         | Blit copy + MTLSharedEvent signal/wait    | No (event-encoded)       |
| `synchronize`              | `[cmd_buf_last waitUntilCompleted]` + scan| Yes                      |
| `event_record`             | Signal `MTLSharedEvent` from a new cmd_buf| No                       |
| `event_wait`               | Encode wait on `MTLSharedEvent`           | No                       |

`ggml_metal_synchronize` (`ggml-metal-context.m:239-295`) waits on
`cmd_buf_last`, then iterates every `cmd_bufs[cb_idx]` and every
`cmd_bufs_ext` entry checking `MTLCommandBufferStatus`. Any non-
`Completed` status sets `has_error = true` and returns; the backend
must be recreated.

---

## 9. SIMD / GPU Strategy

### 9.1 Simdgroup matrix multiply

The Metal backend relies heavily on `simdgroup_matrix` from the
Metal Shading Language (`metal_simdgroup_matrix` from
`<metal_simdgroup>`). The `has_simdgroup_mm` capability
(`ggml-metal-device.m:699`, gated on `MTLGPUFamilyApple7`) is
required for:

* `MUL_MAT` matrix-matrix path (`mul_mm` kernel,
  `ggml-metal-ops.cpp:2177`).
* `MUL_MAT_ID` matrix-matrix path (`ggml-metal-ops.cpp:2332`).
* `FLASH_ATTN_EXT` non-vec path (`ggml-metal-device.m:1267`).

The kernel-block parameters are hardcoded in `ggml-metal-impl.h:8-15`:

```c
#define SZ_SIMDGROUP 16
#define N_MM_NK 2
#define N_MM_NK_TOTAL (SZ_SIMDGROUP * N_MM_NK)  // 32
#define N_MM_BLOCK_X 4
#define N_MM_BLOCK_Y 2
#define N_MM_SIMD_GROUP_X 2
#define N_MM_SIMD_GROUP_Y 2
```

So one threadgroup of `mul_mm` produces an `NRA × NRB` =
`(16 * 2 * 2) × (16 * 4 * 2) = 64 × 128` output tile, using 4
simdgroups (`N_MM_SIMD_GROUP_X * N_MM_SIMD_GROUP_Y`).

### 9.2 Simdgroup reduction

`has_simdgroup_reduction` (also `MTLGPUFamilyApple7`,
`ggml-metal-device.m:696-697`) gates elementwise reductions
(`SUM`, `SUM_ROWS`, `NORM`, `RMS_NORM`, `L2_NORM`, `GROUP_NORM`,
`SOFT_MAX`, `ARGMAX`, `OPT_STEP_ADAMW`, `SSM_CONV`, `SSM_SCAN`).
Without it, the backend reports these ops as unsupported.

### 9.3 Tensor API (experimental, Metal4)

`has_tensor` (`ggml-metal-device.m:708-725`, gated on
`MTLGPUFamilyMetal4_GGML = 5002`) enables the experimental
`metal_tensor` / `MetalPerformancePrimitives` path. At device init,
two dummy kernels (one f16, one bf16) are compiled to verify the
tensor API works in the current environment; if compilation fails,
`has_tensor` is silently downgraded to false. Even when
compilation succeeds, `has_tensor` is **disabled for pre-M5 / pre-A19
devices** by name match (`ggml-metal-device.m:718-725`) because the
current implementation is ~5% slower than simdgroup on M2 Ultra and
no better on M4.

The `has_tensor` flag changes the `mul_mm` tile size:
`ggml-metal-device.cpp:716-720` — `bc_out` is computed against
`(NRA, NRB) = (16 * 2 * 2, 16 * 4 * 2) = (64, 128)` if `has_tensor`
else `(64, 32)`. The smem allocation differs correspondingly
(`ggml-metal-device.cpp:748-759`).

### 9.4 Bfloat16 gating

`has_bfloat` (`ggml-metal-device.m:702-706`, gated on
`MTLGPUFamilyMetal3_GGML` or `MTLGPUFamilyApple6`) gates every op
that touches `GGML_TYPE_BF16`. `supports_op` rejects BF16 inputs
upfront (`ggml-metal-device.m:1056-1066`).

### 9.5 Heuristic kernel selection

The matmul path (`ggml-metal-ops.cpp:2043-2273`) is the canonical
example:

| Branch         | Condition                                            | Pipeline                |
| -------------- | ---------------------------------------------------- | ----------------------- |
| `mul_mv_ext`   | `src1==F32`, `ne00%128==0`, `ne11∈[2,8]` (or `[4,8]` for K-quants) | specialized ext kernel  |
| `mul_mm`       | `has_simdgroup_mm`, `ne00≥64`, `ne11>8`, no transpose | simdgroup matmul        |
| `mul_mv`       | else                                                  | per-dtype GEMV kernel   |

The `ne11_mm_min = 8` cutoff (`ggml-metal-ops.cpp:2068`) is a
hardcoded constant with no per-device or per-shape tuning. The
comment at line 2101-2107 admits "I still don't know why we should
not always use the maximum available threads" — i.e. the `nsg`
choice for `mul_mv_ext` is also untuned.

For FlashAttention (`ggml-metal-ops.cpp:2526-2534`):

```c
bool ggml_metal_op_flash_attn_ext_use_vec(op) {
    return (ne01 < 20) && (ne00 % 32 == 0);
}
```

`ne01` is the query batch size; `ne00` is head size. Below 20
queries, use the vec kernel (single-query optimization); above,
use the half8x8 matrix kernel.

---

## 10. Quantization Strategy

### 10.1 Per-dtype pipeline selectors

The Metal backend has no type-traits table like CPU's
`type_traits_cpu[]`. Instead, each op encoder picks a pipeline name
by `switch (tsrc)` on the source dtype, then dispatches to a
dtype-specialized Metal kernel.

For `mul_mv`, the per-dtype table is in
`ggml-metal-device.cpp:766-955`: ~25 cases covering F32, F16, BF16,
Q1_0, Q2_0, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, MXFP4, Q2_K, Q3_K, Q4_K,
Q5_K, Q6_K, IQ2_XXS, IQ2_XS, IQ3_XXS, IQ3_S, IQ2_S, IQ1_S, IQ1_M,
IQ4_NL, IQ4_XS. Each sets `nsg`, `nr0`, `smem` from per-dtype
constants in `ggml-metal-impl.h:24-88`.

For `mul_mm`, dtype dispatch is via Metal template specialization:
`kernel_mul_mm<half, half4x4, simdgroup_half8x8, ...>` is
instantiated for each (tsrc, tdst) pair at compile time (see
`ggml-metal.metal:10684` for the f32_f32 instantiation).

### 10.2 Quantized weight dequantization

Quantized weight rows are dequantized inside the kernel via
`dequantize_*` functions (defined per-quant in `ggml-metal.metal`,
audited in ARTX16). The `mul_mm` template takes a `dequantize_func`
template parameter so the same matrix-multiply skeleton serves all
quants.

### 10.3 Activation conversion

Unlike CPU (which converts src1 to `vec_dot_type` upfront in
`wdata`), Metal kernels accept src1 in its native dtype and convert
inside the kernel. There is no `wdata` allocation. The trade-off:
no host-side conversion cost, but each kernel must include its own
F32→X conversion logic.

### 10.4 Supported quant formats

`supports_op` (`ggml-metal-device.m:1280-1341`) enumerates the
dtype matrix for `CPY`/`DUP`/`CONT` (the conversion ops), which is
a proxy for "which quant conversions Metal can do":

* F32 → F32, F16, BF16, Q8_0, Q1_0, Q2_0, Q4_0, Q4_1, Q5_0, Q5_1,
  IQ4_NL, I32.
* F16 → F32, F16.
* BF16 → F32, BF16.
* Q1_0/Q2_0/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 → F32, F16.
* I32 → F32, I32.

Notable absences: no `F32 → Q4_K`, no `Q4_K → F16`, no `BF16 → Q*`.
K-quants are dequantized inside matmul kernels but not exposed via
`CPY`. `NVFP4` is explicitly excluded from `MUL_MAT`, `MUL_MAT_ID`,
and `GET_ROWS` (`ggml-metal-device.m:1279, 1341`).

---

## 11. Correctness Analysis

### 11.1 Floating-point reassociation

* **Simdgroup matrix multiply** (`mul_mm`) accumulates into `simdgroup_half8x8` (f16 intermediate) or `simdgroup_float8x8` (f32 intermediate) per template instantiation. Reduction order is set by Metal's simdgroup-matrix implementation; reassociates vs. scalar at ULP level.
* **Per-row GEMV** (`mul_mv`) accumulates into f32 across the row, one simdgroup at a time, then horizontally reduces. Standard SIMD reassociation.
* **Multi-encoder parallelism**: when `n_cb > 1`, different command buffers encode disjoint node ranges. The result is *not* bit-identical to `n_cb == 1` only if a fusion decision differs between splits — explicitly forbidden by the assertion at `ggml-metal-ops.cpp:513-516` ("fusion error: nodes spanning multiple encoders have been fused").

### 11.2 Approximate math

* GELU / SILU / etc. are computed inline in the kernel, not via LUT (contrast with CPU's 128 KB f16 LUT, ARTX01-F07). No precision reduction beyond the storage format.
* `logit_softcap` in FlashAttention (`ggml-metal-ops.cpp:2690-2692`): `scale /= logit_softcap` is performed on the host before dispatching, so the kernel sees a pre-divided scale. No precision loss.

### 11.3 Precision reduction

* Quantized matmul is a Q×F32 (or Q×Q8) inner product, same as CPU. The activation (src1) is *not* pre-quantized to `vec_dot_type` — each kernel handles the conversion internally.
* FlashAttention vec kernel uses f32 accumulators; the half8x8 kernel uses f16 for K/V cache loads but f32 for attention scores. Standard FlashAttention precision trade-offs.

### 11.4 Non-deterministic reductions

* **Concurrent dispatch within a command buffer**: when `use_concurrency` is true, Metal may overlap dispatches. If two dispatches write to the same buffer (which the overlap tracker should prevent), the result is undefined. The tracker (`ggml_mem_ranges`) is the correctness guard.
* **`n_cb > 1`**: each encoder produces its own command buffer, submitted to the same queue. Metal guarantees in-order execution of command buffers within a queue, so cross-encoder dependencies are honored automatically. No cross-encoder fusion is allowed.
* **Graph reorder**: `ggml_graph_optimize` reorders nodes only when they are concurrent (no overlapping src/dst ranges), so the observable result is the same.

### 11.5 Atomic accumulation

None in the matmul path — each threadgroup writes to disjoint output tiles. `OPT_STEP_ADAMW`, `OPT_STEP_SGD`, `ARGMAX`, `SUM`, `COUNT_EQUAL` use simdgroup-level reductions (no cross-threadgroup atomics in the hot path).

### 11.6 Architecture-specific assumptions

* `MTLGPUFamilyApple7` is the minimum for simdgroup reduction and matrix multiply. Pre-Apple7 devices (A10, A11) cannot run matmul kernels.
* `MTLGPUFamilyMetal3` for bfloat; `MTLGPUFamilyMetal4` for the experimental tensor API.
* macOS 15+ / iOS 18+ for residency sets. Older OSes silently disable the feature.
* Unified memory assumed for `use_shared_buffers=true`. On discrete GPUs, `use_shared_buffers` is forced true for eGPUs but defaults false otherwise, requiring Blit copies for host access.

---

## 12. Optimization Analysis

### 12.1 Identified optimizations

| Optimization                         | Where                                              | Notes                                                              |
| ------------------------------------ | -------------------------------------------------- | ------------------------------------------------------------------ |
| Lazy pipeline cache (name-keyed)     | `ggml-metal-device.m:349-453`                      | First-call compiles, subsequent calls are unordered_map lookups    |
| `MTLFunctionConstantValues` specialization | `ggml-metal-device.m:40-67`, `ggml-metal-device.cpp:671-702` | Per-(shape, flags) pipeline; dead-code eliminated by Metal compiler |
| `n_cb` parallel encoders             | `ggml-metal-context.m:438-615`                     | Main thread + up to 8 workers; warning at `n_cb > 2`               |
| `MTLDispatchTypeConcurrent` encoder  | `ggml-metal-device.m:468-472`                      | Overlaps independent dispatches inside one command buffer          |
| Memory-range overlap tracker         | `ggml-metal-common.cpp:8-185`                      | O(N²) per graph but cheap; avoids unnecessary barriers             |
| Plan-time graph reorder              | `ggml-metal-common.cpp:209-373`                    | Forward search N=64, whitelist-limited ops                         |
| Plan-time fuse (ADD/MUL/NORM chains) | `ggml-metal-common.cpp:375-457`                    | Up to MAX_FUSE=16 chained ops                                      |
| Encode-time fuse (bin×N, norm+mul+add, snake) | `ggml-metal-ops.cpp:3127-3288, 3081-3125, 3409-3500` | Encode-time decision, propagated via function constant             |
| `newBufferWithBytesNoCopy` (shared)  | `ggml-metal-device.m:1552-1557`                    | Zero-copy host memory on unified memory                            |
| Multi-view buffer mapping            | `ggml-metal-device.m:1585-1678`                    | Overlapping views to bypass per-buffer size cap                    |
| `MTLResidencySet` keep-alive thread  | `ggml-metal-device.m:542-633`                      | 5 ms heartbeat; 3-min cooldown; pins weights in GPU memory         |
| `MTLSharedEvent` cross-context sync  | `ggml-metal-device.m:989-1039`, `ggml-metal-context.m:395-436` | Non-blocking cpy_tensor_async between two Metal backends   |
| `commandBufferWithUnretainedReferences` | `ggml-metal-context.m:512, 531`, `ggml-metal-device.m:1719, 1760, 1809, 1848, 1876` | Avoids Metal's internal NSArray retain churn for buffers   |
| Dispatch-semaphore completion wait   | `ggml-metal-device.m:1758-1784`                    | Used in `set_tensor` (sync path) as alternative to `waitUntilCompleted` |
| Per-op kernel-args struct (packed)   | `ggml-metal-impl.h:158-1222`                       | `int32_t` counts + `uint64_t` strides, matches kernel `constant` slot 0 |
| `cmd_bufs_ext` retained list         | `ggml-metal-context.m:69-70, 269-294`              | Async transfers live until next `synchronize`                      |
| `AGX_RELAX_CDM_CTXSTORE_TIMEOUT=1` env | `ggml-metal.cpp:926`                            | Workaround for macOS issue #20141 (CDM ctxstore timeout)           |

### 12.2 Optimizations *not* present (worth noting)

* **No `MTLParallelRenderCommandEncoder`**. The backend is
  compute-only; no render path.
* **No `MTLPipelineCache`** (Metal does not expose one). Pipeline
  state is cached in ggml's `std::unordered_map` per-process; it
  does not persist across runs.
* **No persistent kernel binary archive** (Metal 3's
  `MTLBinaryArchiver` is not used). Every process recompiles every
  specialized pipeline from source on first use.
* **No autotuner**. Threadgroup sizes, `nsg`, `nr0`, `nr1` are
  hardcoded per-dtype in `ggml-metal-impl.h`. The comment at
  `ggml-metal-ops.cpp:2101-2107` admits `nsg` for `mul_mv_ext` is
  untuned.
* **No per-shape kernel selection beyond the GEMV/GEMM/ext split**.
  Two matmuls with the same dtype but different `ne00` get the same
  pipeline (modulo `nsg` and alignment-based suffix).
* **No graph-level plan caching**. The plan-time reorder + fuse runs
  on every `graph_compute` call (when
  `GGML_METAL_GRAPH_OPTIMIZE_DISABLE` is not set).
* **No multi-GPU exploitation**. `MTLCreateSystemDefaultDevice` is
  used (`ggml-metal-device.m:685`); `MTLCopyAllDevices` is only
  called for debug logging on macOS (`ggml-metal-context.m:87-93`).
  Multi-GPU Mac Pro configurations are silently single-GPU.
* **No peer-to-peer**. `cpy_tensor_async` between two Metal contexts
  uses a `MTLSharedEvent` and a Blit copy through shared host
  memory (if both contexts are on the same device) — there is no
  direct GPU↔GPU copy.

---

## 13. Architectural Strengths

1. **Clean device/context split**. `ggml_metal_device_t` owns `MTLDevice`, `MTLCommandQueue`, `MTLLibrary`, and the pipeline cache; `ggml_metal_t` (context) owns command buffers and per-backend state. Multiple backends can share a device without duplicating the library. Structurally cleaner than CPU (no device) and on par with CUDA.

2. **Lazy, name-keyed pipeline cache with function-constant specialization**. The single best design decision in the Metal backend. Each (op, dtype, shape-summary, flags) tuple gets its own compiled pipeline, with dead-code elimination by the Metal compiler (e.g., `has_mask=false` removes the mask-load branch at compile time). Process-local, `NSLock`-protected; first-call cost paid once per unique shape.

3. **Memory-range overlap tracker + concurrent dispatch**. The `ggml_mem_ranges` mechanism is a simple, correct, cheap way to decide when a `memoryBarrierWithScope:Buffers` is needed. Encode-time analog of the plan-time graph reorder; both maximize the concurrent-dispatch window inside a command buffer.

4. **Three buffer types for unified vs discrete memory**. Shared (Apple Silicon), Private (discrete GPU), Mapped (host pointer) — with `supports_buft` accepting all three — lets the same backend serve both unified-memory Apple hardware and discrete-GPU Macs without per-op branching.

5. **Residency-set keep-alive thread**. A 5 ms heartbeat that calls `[rset requestResidency]` for every pinned buffer for 3 minutes after the last `graph_compute` keeps weights wired without explicit user intervention. Env-tunable via `GGML_METAL_RESIDENCY_KEEP_ALIVE_S`.

6. **Plan-time graph optimizer**. Reorder + fuse runs at plan time (not encode time), so the per-`graph_compute` cost is paid once. The whitelist of reorderable ops is conservative but extensible.

7. **`MTLSharedEvent`-based cross-context async copy**. `cpy_tensor_async` encodes a signal in the source context and a wait in the destination context, all without blocking the host. Correct Metal idiom for cross-queue synchronization; structurally cleaner than CPU's "no async at all" (ARTX01-F01).

8. **Per-op kernel-args structs**. Packing all kernel arguments into a single `constant` buffer (slot 0) with a typed struct (`ggml_metal_kargs_*`) is more cache-friendly and type-safe than per-arg `setBytes` calls. CPU has no equivalent.

9. **Sticky error state + structural op fusion**. `has_error` on the context prevents silent corruption: subsequent `graph_compute` calls fail fast until recreate. The Snake 5-op pattern (`ggml_metal_op_can_fuse_snake`) is detected by structural pattern matching, not by op hints — more flexible than CPU's single `RMS_NORM+MUL` fusion.

---

## 14. Architectural Weaknesses

### W1 — `MTLCreateSystemDefaultDevice` ignores multi-GPU Macs

**Evidence**: `ggml-metal-device.m:685`. `MTLCopyAllDevices()` is called only for debug logging at `ggml-metal-context.m:87-93`. The `GGML_METAL_DEVICES` env var can synthesize virtual device slots but they all map to the same physical `MTLDevice`.

**Impact**: Mac Pro with multiple MPX modules reports as one device. Multi-GPU tensor parallelism is impossible without code changes. The per-device buffer-type infrastructure (`ggml-metal.cpp:281-321`) exists but is unused.

### W2 — `n_cb > 2` is "not recommended" but the cap is 8

**Evidence**: `ggml-metal-context.m:665-669`. The warning admits degradation; the cap `GGML_METAL_MAX_COMMAND_BUFFERS = 8` is arbitrary. Default `n_cb = 1` (`ggml-metal.cpp:611`).

**Impact**: Users tuning `n_cb` have no data to go on.

### W3 — No persistent kernel binary archive

**Evidence**: `ggml-metal-device.m:106-260` compiles the library on every process start. The pipeline cache is per-process and lost on exit. `MTLBinaryArchiver` (Metal 3) is not used.

**Impact**: Cold-start latency. First `graph_compute` pays ~10s–60s of pipeline compilation per unique shape.

### W4 — Hardcoded `ne11_mm_min = 8` GEMV-vs-GEMM cutoff

**Evidence**: `ggml-metal-ops.cpp:2068`. The comment at line 2101-2107 admits `nsg` is untuned. The `mul_mv_ext` path (lines 2072-2170) has its own hardcoded `nsg=2`, `nxpsg` table, and per-`ne11` `r1ptg` table — none tuned per-device.

**Impact**: Suboptimal for shapes near the boundary. Apple Silicon generations differ in simdgroup throughput, so the cutoff should be per-device.

### W5 — `supports_op` does not consult the pipeline cache

**Evidence**: `ggml-metal-device.m:1051-1375`. Pure shape/type check; never tries to compile a pipeline. The FlashAttention head-size whitelist (lines 1231-1244) is hardcoded.

**Impact**: Adding a new head size requires editing the whitelist and recompiling.

### W6 — Memory-range overlap tracker is O(N²) per graph

**Evidence**: `ggml-metal-common.cpp:124-153`. The check loops over all prior ranges for every new node. For N ≈ 3000, ~9M comparisons per graph_compute. Tracker resets on every barrier, so steady-state N is smaller, but worst-case is O(N²).

**Impact**: Negligible for small graphs; measurable for 10K+ node graphs. An interval-tree would reduce to O(N log N).

### W7 — `addr_virt` fake virtual address for private buffers

**Evidence**: `ggml-metal-device.m:693, 1537`. Private buffers store `all_data = (void *) atomic_fetch_add(&dev->addr_virt, size)`. Not a real pointer — dereferencing would segfault.

**Impact**: Correctness risk if any code path accidentally dereferences `all_data` for a private buffer.

### W8 — `use_graph_optimize` runs on every `graph_compute`

**Evidence**: `ggml-metal-context.m:617-625`. No caching of the optimized graph; same `cgraph` submitted repeatedly pays the reorder + fuse cost every time.

**Impact**: O(N) per `graph_compute`. For N=3000 at 100 tokens/sec, ~300K node-comparisons/sec of CPU work.

### W9 — `set_tensor_async` allocates a new MTLBuffer per call

**Evidence**: `ggml-metal-context.m:311-313`. `[device newBufferWithBytes:data length:size options:Shared]` allocates a fresh MTLBuffer and copies the source bytes. No buffer pool.

**Impact**: Per-call allocation + copy cost. A ring-buffer of pre-allocated shared MTLBuffers would be cheaper.

### W10 — Tensor API disabled by name match

**Evidence**: `ggml-metal-device.m:718-725`. Check is `![[name] containsString:@"M5"] && ![@"M6"] && ![@"A19"] && ![@"A20"]`. New chip names require code changes; "M5 Pro" matches but a hypothetical "M6X" would not.

**Impact**: Brittle. A capability-bit check would be more robust.

---

## 15. GwenLand Mapping

| GwenLand module | ADOPT / ADAPT / REJECT / MONITOR / DEFER | What | Reasoning |
| --------------- | ---------------------------------------- | ---- | --------- |
| `glmetal`       | **ADOPT** | Lazy name-keyed pipeline cache with `MTLFunctionConstantValues` specialization | Best design in the backend; enables per-shape dead-code elimination |
| `glmetal`       | **ADOPT** | `n_cb` parallel encoder scheme (main + workers) | Cheap host-side parallelism; default `n_cb=1` is safe |
| `glmetal`       | **ADOPT** | `MTLDispatchTypeConcurrent` encoder + `ggml_mem_ranges` overlap tracker | Maximizes in-flight dispatches without barriers |
| `glmetal`       | **ADOPT** | Three buffer types (Shared / Private / Mapped) | Unified vs discrete memory handled cleanly |
| `glmetal`       | **ADOPT** | `MTLResidencySet` keep-alive heartbeat thread | Pins weights without user intervention; tunable via env |
| `glmetal`       | **ADOPT** | `MTLSharedEvent`-based cross-context async copy | Correct Metal idiom; non-blocking |
| `glmetal`       | **ADOPT** | Per-op kernel-args structs packed into `constant` slot 0 | Cache-friendly, type-safe |
| `glmetal`       | **ADAPT** | Plan-time graph reorder (forward N=64) | Keep the idea; make N and the whitelist configurable |
| `glmetal`       | **ADAPT** | Encode-time op fusion (bin×N, norm+mul+add, snake) | Keep the patterns; move the bin×N cap from 8 to a constant |
| `glmetal`       | **ADAPT** | `has_error` sticky flag | Keep; add a "reset error" API so the backend can be recovered without recreate |
| `glmetal`       | **REJECT**| `MTLCreateSystemDefaultDevice` for multi-GPU Macs | Use `MTLCopyAllDevices` and register one `ggml_backend_dev_t` per physical GPU |
| `glmetal`       | **REJECT**| Hardcoded `ne11_mm_min = 8` cutoff | Use per-device benchmarked cutoffs or a runtime microbenchmark |
| `glmetal`       | **REJECT**| `addr_virt` fake pointer for private buffers | Use `std::optional<MTLBuffer>` or a tagged union; never store a fake pointer |
| `glmetal`       | **REJECT**| Tensor-API enable/disable by chip name | Use capability bits + a one-time benchmark |
| `glmetal`       | **MONITOR**| `MTLBinaryArchiver` (Metal 3) for persistent pipeline cache | Could eliminate cold-start compile cost; not yet used upstream |
| `glmetal`       | **MONITOR**| `n_cb > 2` performance | The upstream warning suggests this is unresolved; revisit on M4 Ultra |
| `glmetal`       | **DEFER** | Experimental `metal_tensor` / MPP path | Currently slower than simdgroup on most chips; revisit when M5/A19 hardware is available |
| `GATE`          | **ADOPT** | Plan-time graph reorder + fuse as a separate pass | Run once per graph, not per `graph_compute` |
| `GATE`          | **ADOPT** | Memory-range overlap tracker for barrier insertion | Generalizes to any backend with explicit barriers (CUDA, Vulkan) |
| `GATE`          | **ADAPT** | `cmd_bufs_ext` retained-list pattern for async transfers | Keep; add a ring-buffer pool to avoid per-call allocation |

---

## 16. Recommendations

### R1 — ADOPT lazy name-keyed pipeline cache with function-constant specialization
**Priority:** Critical | **Difficulty:** M | **Dependencies:** none
GwenLand's `glmetal` should define a `gl_metal_pipeline_cache` keyed by `std::string` (or `uint64_t` hash). Each entry is a compiled `MTLComputePipelineState` plus the `nr0`/`nr1`/`nsg`/`smem` parameters needed for dispatch. Use `std::shared_mutex` for read-mostly locking. First call compiles via `newFunctionWithName:constantValues:`.

### R2 — ADOPT `n_cb` parallel encoder scheme
**Priority:** High | **Difficulty:** M | **Dependencies:** R1
Split the node list as `n_main = MAX(64, 0.1 * n_nodes)` for the main thread, remainder split evenly across `n_cb` workers via `dispatch_apply`. Default `n_cb = 1`. Cap at 8. Warn at `n_cb > 2` but allow it.

### R3 — ADOPT memory-range overlap tracker + concurrent dispatch
**Priority:** High | **Difficulty:** M | **Dependencies:** R2
For each encoder, maintain a `std::vector<(buffer_id, [p0,p1), src|dst)>`. Before encoding a node, check for overlap; if found, emit `memoryBarrierWithScope:Buffers` and clear. Use `MTLDispatchTypeConcurrent` on the compute encoder.

### R4 — REJECT `MTLCreateSystemDefaultDevice` for multi-GPU
**Priority:** High | **Difficulty:** M | **Dependencies:** R1
On macOS, call `MTLCopyAllDevices()` and register one `ggml_backend_dev_t` per physical `MTLDevice`. Keep `MTLCreateSystemDefaultDevice` as the iOS fallback. Unblocks Mac Pro tensor parallelism.

### R5 — ADOPT `MTLResidencySet` keep-alive heartbeat
**Priority:** Medium | **Difficulty:** M | **Dependencies:** R1
On macOS 15+ / iOS 18+, create an `MTLResidencySet` per large buffer. Background `dispatch_group` thread calls `[rset requestResidency]` every 5 ms while a countdown is positive; reset to `keep_alive_s * loops_per_s` on every `graph_compute`. Make `keep_alive_s` env-tunable.

### R6 — ADOPT `MTLSharedEvent`-based async ops
**Priority:** High | **Difficulty:** S | **Dependencies:** R1
Implement `set_tensor_async` / `get_tensor_async` / `cpy_tensor_async` via `MTLBlitCommandEncoder` + `MTLSharedEvent`. Retain command buffers in a per-context list until the next `synchronize`.

### R7 — ADAPT plan-time graph reorder + fuse
**Priority:** High | **Difficulty:** L | **Dependencies:** GATE design
Move `ggml_graph_optimize` into GATE as a plan-time pass with result caching. Make the forward-search depth N and the `h_safe` whitelist configurable. Combine with CPU's plan-time fusion goal (ARTX01 R5).

### R8 — REJECT hardcoded `ne11_mm_min` and per-shape `nsg` tables
**Priority:** Medium | **Difficulty:** L | **Dependencies:** R1
Replace with a runtime microbenchmark at device-init: time a small `mul_mv` and `mul_mm` at several `ne11` values, fit a crossover curve, store per-device. Similarly for `nsg` in `mul_mv_ext`.

### R9 — ADOPT per-op kernel-args structs
**Priority:** Medium | **Difficulty:** S | **Dependencies:** R1
Define `gl_metal_kargs_*` structs matching each kernel's `constant` slot 0. Use `int32_t` for element counts (register pressure), `uint64_t` for strides.

### R10 — MONITOR `MTLBinaryArchiver` for persistent pipeline cache
**Priority:** Low | **Difficulty:** M | **Dependencies:** R1
Metal 3's `MTLBinaryArchiver` can serialize compiled pipeline states to disk. If adopted, cold-start compile cost drops from seconds to milliseconds. Not yet used upstream; revisit when stable.

### R11 — REJECT `addr_virt` fake pointer
**Priority:** Low | **Difficulty:** XS | **Dependencies:** R1
Use `std::optional<MTLBuffer>` or a tagged union for buffer wrappers. Never store a fake pointer that could be dereferenced.

### R12 — ADAPT `has_error` sticky flag with reset API
**Priority:** Medium | **Difficulty:** S | **Dependencies:** R1
Keep the sticky error flag, but add a `glmetal_reset_error(backend)` that drains all command buffers and clears the flag, so the backend can be recovered without full recreate.

---

## 17. Findings

### Finding ARTX15-F01

```
Finding ID:           ARTX15-F01
Category:             BACKEND_DESIGN
Engine:               Metal
Component:            Backend / device / registry vtables
Source File:          ggml/src/ggml-metal/ggml-metal.cpp
Function:             ggml_backend_metal_i, ggml_backend_metal_device_i, ggml_backend_metal_reg_i
Lines:                568-813, 881-948
Summary:              Metal exposes a true three-tier vtable (reg / device / backend), unlike
                      CPU's two-tier (reg / device fold into the backend).
Observation:          The registry vtable has 4 entries (get_name, get_device_count,
                      get_device, get_proc_address). The device vtable has 13 entries
                      (get_name, get_description, get_memory, get_type, get_props,
                      init_backend, get_buffer_type, get_host_buffer_type=NULL,
                      buffer_from_host_ptr, supports_op, supports_buft, offload_op,
                      event_new, event_free, event_synchronize). The backend vtable has
                      14 entries including async (set/get/cpy_tensor_async), events
                      (event_record/wait), graph_compute, graph_optimize. Caps advertise
                      async=true, host_buffer=false, buffer_from_host_ptr=true, events=true.
                      This is the most feature-complete backend vtable in ggml.
Evidence:             ggml-metal.cpp:568-585 (backend_i), 797-813 (device_i), 881-886
                      (reg_i), 672-685 (get_props caps).
Architectural Impact: The three-tier split mirrors CUDA's and allows multiple physical
                      devices per registry. The `buffer_from_host_ptr` + `events` combo
                      enables hybrid scheduling with the CPU backend.
Correctness Impact:   None. The vtable is a contract layer.
Optimization Type:    None (architectural).
GwenLand Target:      glmetal, GATE
Recommendation:       ADOPT. Replicate the three-tier vtable in glmetal with the same
                      cap flags. GATE should treat Metal as a peer to CUDA.
Priority:             Critical
Difficulty:           M
Dependencies:         none
Confidence:           High
```

### Finding ARTX15-F02

```
Finding ID:           ARTX15-F02
Category:             BACKEND_DESIGN
Engine:               Metal
Component:            MTLDevice discovery
Source File:          ggml/src/ggml-metal/ggml-metal-device.m
Function:             ggml_metal_device_init
Lines:                679-918
Summary:              Device discovery uses MTLCreateSystemDefaultDevice, silently
                      ignoring multi-GPU Mac Pro configurations.
Observation:          The function calls MTLCreateSystemDefaultDevice() at line 685,
                      which returns the system's preferred device. On macOS, this is
                      always the integrated or discrete GPU chosen by the OS — never
                      both. MTLCopyAllDevices() is called only inside a debug-only
                      logging block at ggml-metal-context.m:87-93. The
                      GGML_METAL_DEVICES env var (ggml-metal.cpp:916-919) can synthesize
                      N virtual device slots, but they all map to the same physical
                      MTLDevice via ggml_metal_device_get(device) which calls
                      ggml_metal_device_init(device) for each index — and
                      ggml_metal_device_init ignores the index, always creating the
                      default device. This is a known gap: PR #15906 (cited at
                      ggml-metal-device.m:525) added discrete-GPU support but stopped
                      short of multi-GPU.
Evidence:             ggml-metal-device.m:685 (MTLCreateSystemDefaultDevice),
                      ggml-metal-device.m:20-26 (device_get singleton),
                      ggml-metal-context.m:87-93 (debug-only MTLCopyAllDevices),
                      ggml-metal.cpp:916-919 (env override).
Architectural Impact: Mac Pro with multiple MPX modules, or any multi-GPU Mac, is
                      effectively single-GPU. Tensor parallelism across Metal devices
                      is impossible without code changes. The per-device buffer-type
                      infrastructure (ggml-metal.cpp:281-321) exists but is unused.
Correctness Impact:   None. Single-device is correct.
Optimization Type:    None (absence of optimization).
GwenLand Target:      glmetal
Recommendation:       REJECT this design. Use MTLCopyAllDevices() on macOS and register
                      one ggml_backend_dev_t per physical device. Keep
                      MTLCreateSystemDefaultDevice as the iOS fallback.
Priority:             High
Difficulty:           M
Dependencies:         R1
Confidence:           High
```

### Finding ARTX15-F03

```
Finding ID:           ARTX15-F03
Category:             BACKEND_DESIGN
Engine:               Metal
Component:            MTLCommandQueue ownership
Source File:          ggml/src/ggml-metal/ggml-metal-device.m, ggml/src/ggml-metal/ggml-metal-context.m
Function:             ggml_metal_device_init (queue creation), ggml_metal_init (queue fetch)
Lines:                device.m:526, 688; context.m:103-110
Summary:              A single MTLCommandQueue is created per device and shared by all
                      backend contexts on that device.
Observation:          ggml_metal_device_init creates the queue once at line 688
                      ([dev->mtl_device newCommandQueue]). Backends fetch it via
                      ggml_metal_device_get_queue(dev) — ggml-metal-device.m:945-947.
                      The context does not own a queue (commented-out at
                      ggml-metal-context.m:105 //[ctx->queue release]). The TODO at
                      context.m:103-105 asks "would it be better to have one queue
                      for the backend and one queue for the device?". All command
                      buffers (graph compute, async transfers, events) enqueue into
                      this shared queue, so they execute in FIFO order across all
                      backends on the device.
Evidence:             ggml-metal-device.m:526 (queue field), 688 (creation), 945-947
                      (getter); ggml-metal-context.m:103-110 (TODO + fetch).
Architectural Impact: FIFO ordering across backends is correct but may serialize
                      unrelated work. A per-backend queue would allow concurrent
                      execution of independent graphs on the same device, at the
                      cost of explicit cross-queue synchronization.
Correctness Impact:   None. FIFO is correct.
Optimization Type:    None (architectural choice).
GwenLand Target:      glmetal
Recommendation:       MONITOR. Keep the shared-queue design for v1; revisit if
                      multi-backend-per-device workloads become common.
Priority:             Low
Difficulty:           S
Dependencies:         R1
Confidence:           High
```

### Finding ARTX15-F04

```
Finding ID:           ARTX15-F04
Category:             EXECUTION_GRAPH
Engine:               Metal
Component:            Per-graph command buffer pool
Source File:          ggml/src/ggml-metal/ggml-metal-context.m
Function:             ggml_metal_graph_compute
Lines:                438-615
Summary:              Graph compute allocates n_cb+1 MTLCommandBuffers per graph,
                      splits nodes as main(0.1*n_nodes) + n_cb workers, dispatches
                      workers via dispatch_apply on a concurrent GCD queue.
Observation:          The main thread encodes the first n_main = MAX(64, 0.1*n_nodes)
                      nodes into cmd_bufs[n_cb]; the remaining n_nodes_1 nodes are
                      split into n_cb chunks of n_nodes_per_cb each, encoded in
                      parallel by dispatch_apply(n_cb, ctx->d_queue, encode_async).
                      Command buffers are created with
                      commandBufferWithUnretainedReferences (lower overhead) and
                      manually retained. The first two are pre-enqueued so the GPU
                      can start while encoding continues; the rest are committed at
                      the end of each encoder block. GGML_METAL_MAX_COMMAND_BUFFERS
                      is 8 (line 20). The default n_cb is 1, set at
                      ggml-metal.cpp:611. n_cb > 2 triggers a warning (line 668).
Evidence:             ggml-metal-context.m:20 (MAX_COMMAND_BUFFERS), 445 (n_main),
                      463-466 (split), 510-548 (cmd_buf alloc + enqueue), 550
                      (dispatch_apply), 676-721 (encode_async block).
Architectural Impact: Host-side encoding parallelism hides encoder overhead behind
                      GPU execution. The 0.1 ratio is empirical (comment line 458
                      cites M1 Pro / M2 Ultra LLaMA tests). The n_cb=2 cap-in-practice
                      leaves performance on the table for very large graphs.
Correctness Impact:   None. Each command buffer encodes disjoint node ranges; FIFO
                      queue ordering ensures cross-buffer dependencies.
Optimization Type:    Asynchronous execution (host encoder parallelism).
GwenLand Target:      glmetal, GATE
Recommendation:       ADOPT. Replicate the n_cb scheme with default n_cb=1, cap 8.
                      Make the 0.1 split ratio configurable.
Priority:             High
Difficulty:           M
Dependencies:         R2
Confidence:           High
```

### Finding ARTX15-F05

```
Finding ID:           ARTX15-F05
Category:             BACKEND_DESIGN
Engine:               Metal
Component:            Buffer types
Source File:          ggml/src/ggml-metal/ggml-metal.cpp
Function:             ggml_backend_metal_buffer_type_shared, _private, _mapped
Lines:                188-474
Summary:              Three buffer types (Shared, Private, Mapped) are registered
                      per device. There is no host_buffer_type and no split buffer type.
Observation:          Shared buffer type allocates via
                      newBufferWithBytesNoCopy+MTLResourceStorageModeShared (line 1552
                      of device.m) when use_shared_buffers is true (Apple Silicon or
                      eGPU). Private allocates via
                      newBufferWithLength+MTLResourceStorageModePrivate (line 1557) for
                      discrete GPUs. Mapped wraps a host pointer via
                      ggml_metal_buffer_map (device.m:1585). All three return
                      is_host=false (lines 275, 351, 427). The device's
                      get_host_buffer_type is NULL (ggml-metal.cpp:805).
                      supports_buft accepts all three by name (ggml-metal.cpp:736-743).
                      There is no split buffer type (single-GPU assumption).
Evidence:             ggml-metal.cpp:188-474 (three buffer types), 805 (NULL host),
                      736-743 (supports_buft); ggml-metal-device.m:1515-1583 (init),
                      1585-1678 (map).
Architectural Impact: The three-type split cleanly handles unified vs discrete
                      memory. The absence of a host buffer type means the ggml
                      scheduler must use buffer_from_host_ptr for host tensors.
Correctness Impact:   None.
Optimization Type:    None (architectural).
GwenLand Target:      glmetal
Recommendation:       ADOPT. Replicate the three-type split. Document that
                      is_host=false is intentional (Metal does not expose a separate
                      host-only memory pool).
Priority:             High
Difficulty:           M
Dependencies:         R1
Confidence:           High
```

### Finding ARTX15-F06

```
Finding ID:           ARTX15-F06
Category:             MEMORY_PATTERN
Engine:               Metal
Component:            Zero-copy host buffer mapping
Source File:          ggml/src/ggml-metal/ggml-metal-device.m
Function:             ggml_metal_buffer_init, ggml_metal_buffer_map
Lines:                1495-1678
Summary:              On Apple Silicon (unified memory), host allocations via
                      vm_allocate are wrapped as MTLResourceStorageModeShared
                      MTLBuffer with newBufferWithBytesNoCopy — true zero-copy.
Observation:          For shared buffers, host memory is allocated via vm_allocate
                      on macOS (line 1499) or posix_memalign on iOS (line 1505),
                      page-aligned. The MTLBuffer is created with
                      newBufferWithBytesNoCopy:length:options:Shared deallocator:nil
                      (line 1552-1555). The host pointer and the MTLBuffer's
                      contents() pointer are identical — writes by either CPU or
                      GPU are visible to both. For mapped buffers (user-supplied
                      pointer), the same mechanism is used (line 1621). When the
                      requested size exceeds max_buffer_size, the buffer is split
                      into multiple overlapping views (lines 1633-1665) so any
                      single tensor fits in one view.
Evidence:             ggml-metal-device.m:1495-1513 (host malloc), 1552-1557 (shared
                      wrap), 1621 (mapped wrap), 1633-1665 (multi-view split).
Architectural Impact: Zero-copy is the foundation of Metal's performance on Apple
                      Silicon. The multi-view split is a structural workaround for
                      Metal's per-buffer size cap (typically 1 GiB on iOS).
Correctness Impact:   None. NoCopy requires page-aligned memory; vm_allocate and
                      posix_memalign guarantee this.
Optimization Type:    None (memory pattern).
GwenLand Target:      glmetal
Recommendation:       ADOPT. Always use vm_allocate for shared buffer host memory
                      on macOS to ensure page alignment. Replicate the multi-view
                      overlap split for >max_buffer_size allocations.
Priority:             High
Difficulty:           S
Dependencies:         R1
Confidence:           High
```

### Finding ARTX15-F07

```
Finding ID:           ARTX15-F07
Category:             BACKEND_DESIGN
Engine:               Metal
Component:            Pipeline state cache
Source File:          ggml/src/ggml-metal/ggml-metal-device.m, ggml/src/ggml-metal/ggml-metal-device.cpp
Function:             ggml_metal_library_compile_pipeline, ggml_metal_pipelines_*
Lines:                device.m:349-453; device.cpp:28-60
Summary:              Compiled MTLComputePipelineState objects are cached in a
                      per-library std::unordered_map<string, pipeline>, keyed by
                      an op+shape+flags string. First call compiles; subsequent
                      calls are O(1) lookups under NSLock.
Observation:          ggml_metal_library_compile_pipeline first checks the cache by
                      name (line 382-387); on miss, it calls [lib->obj
                      newFunctionWithName:base_func constantValues:cv] then [device
                      newComputePipelineStateWithFunction:]. The resulting
                      MTLComputePipelineState is wrapped in a ggml_metal_pipeline
                      and inserted into the map (line 447). The cache is per-library
                      (one per device), shared across all backends. The lock is
                      NSLock (not a read-write lock), so concurrent lookups
                      serialize. Names are e.g.
                      "kernel_mul_mv_q4_0_f32_nsg=2_ne12=128_r2=1_r3=1".
Evidence:             ggml-metal-device.m:349-453 (compile + cache), 97-104 (library
                      struct); ggml-metal-device.cpp:28-60 (unordered_map).
Architectural Impact: Per-shape specialization with caching means the first graph
                      pays compile cost; subsequent graphs (same shapes) are fast.
                      The cache is lost on process exit — no persistent binary
                      archive (see W3).
Correctness Impact:   None. Cache is keyed by exact string; no aliasing.
Optimization Type:    None (architectural).
GwenLand Target:      glmetal
Recommendation:       ADOPT. Use std::shared_mutex instead of NSLock for read-mostly
                      access. Consider a hash-based key instead of string for faster
                      lookup.
Priority:             Critical
Difficulty:           M
Dependencies:         R1
Confidence:           High
```

### Finding ARTX15-F08

```
Finding ID:           ARTX15-F08
Category:             GPU_KERNEL
Engine:               Metal
Component:            Function-constant specialization
Source File:          ggml/src/ggml-metal/ggml-metal-device.m, ggml/src/ggml-metal/ggml-metal-device.cpp, ggml/src/ggml-metal/ggml-metal-impl.h
Function:             ggml_metal_cv_set_*, ggml_metal_library_compile_pipeline (cv path)
Lines:                device.m:40-67, 389-401; device.cpp:671-702, 704-764, 1395-1458; impl.h:91-106
Summary:              MTLFunctionConstantValues are used to specialize kernels per
                      (shape, flags) tuple, enabling compile-time dead-code elimination.
Observation:          Each op that supports specialization defines a function-constant
                      offset (FC_MUL_MV=600, FC_MUL_MM=700, FC_FLASH_ATTN_EXT=300,
                      FC_ROPE=800, FC_BIN=1300, etc. — impl.h:91-106). The per-op
                      pipeline selector (e.g. ggml_metal_library_get_pipeline_mul_mv_ext,
                      device.cpp:671-702) creates a ggml_metal_cv_t, sets int16/bool
                      values at the offset indices, and passes it to
                      compile_pipeline. Metal's compiler then emits a specialized
                      kernel where e.g. has_mask=false removes the mask-load branch
                      entirely. The specialized pipeline is cached under a name
                      that encodes all the constant values.
Evidence:             ggml-metal-device.m:40-67 (cv wrapper), 389-401 (newFunction
                      with constantValues); ggml-metal-device.cpp:671-702 (mul_mv_ext),
                      704-764 (mul_mm), 1395-1458 (flash_attn_ext);
                      ggml-metal-impl.h:91-106 (FC offsets).
Architectural Impact: This is the Metal equivalent of CUDA's template specialization.
                      Per-shape specialization removes branches and enables constant
                      folding in the kernel. The cost is compile time per unique
                      (shape, flags) tuple.
Correctness Impact:   None. Specialization is purely a performance optimization.
Optimization Type:    Kernel specialization via function constants.
GwenLand Target:      glmetal
Recommendation:       ADOPT. Replicate the FC offset scheme. Document that each
                      FC_* offset is a base, with per-op sub-indices.
Priority:             Critical
Difficulty:           M
Dependencies:         R1, R7
Confidence:           High
```

### Finding ARTX15-F09

```
Finding ID:           ARTX15-F09
Category:             SIMD_STRATEGY
Engine:               Metal
Component:            Matmul kernel selection heuristic
Source File:          ggml/src/ggml-metal/ggml-metal-ops.cpp
Function:             ggml_metal_op_mul_mat
Lines:                2043-2273
Summary:              MUL_MAT selects among three kernel families (mul_mv_ext,
                      mul_mm, mul_mv) via a hardcoded heuristic with ne11_mm_min=8
                      and per-ne11 r1ptg/nxpsg tables — no per-device tuning.
Observation:          The three-way branch is:
                      (1) mul_mv_ext: src1==F32, ne00%128==0, ne11∈[2,8] (or [4,8]
                          for K-quants). nsg=2 hardcoded. nxpsg∈{16,8,4} based on
                          ne00 alignment. r1ptg from a per-ne11 switch
                          (lines 2126-2140).
                      (2) mul_mm: has_simdgroup_mm, ne00>=64, ne11>8, no transpose.
                          Uses simdgroup_matrix.
                      (3) mul_mv: else. Per-dtype nsg/nr0/smem from impl.h.
                      The comment at line 2101-2107 explicitly says "I still don't
                      know why we should not always use the maximum available
                      threads" — i.e. the nsg choice for mul_mv_ext is untuned.
                      The crossover at ne11=8 is not benchmarked per-device.
Evidence:             ggml-metal-ops.cpp:2068 (ne11_mm_min), 2072-2170 (mul_mv_ext
                      branch with hardcoded nsg=2, nxpsg table, r1ptg table),
                      2172-2222 (mul_mm branch), 2223-2270 (mul_mv fallback).
Architectural Impact: Suboptimal for shapes near the boundary. Apple Silicon
                      generations differ in simdgroup throughput, so the cutoff
                      should be per-device. The mul_mv_ext parameters are unlikely
                      to be optimal across M1/M2/M3/M4.
Correctness Impact:   None. All three paths produce correct results.
Optimization Type:    None (heuristic, not tuned).
GwenLand Target:      glmetal
Recommendation:       REJECT the hardcoded cutoff. ADOPT a runtime microbenchmark
                      at device-init time to fit a crossover curve. Keep the
                      three-way branch structure.
Priority:             Medium
Difficulty:           L
Dependencies:         R8
Confidence:           High
```

### Finding ARTX15-F10

```
Finding ID:           ARTX15-F10
Category:             EXECUTION_GRAPH
Engine:               Metal
Component:            Concurrent compute encoder + memory-range overlap tracker
Source File:          ggml/src/ggml-metal/ggml-metal-device.m, ggml/src/ggml-metal/ggml-metal-common.cpp, ggml/src/ggml-metal/ggml-metal-ops.cpp
Function:             ggml_metal_encoder_init, ggml_mem_ranges_*, ggml_metal_op_concurrency_*
Lines:                device.m:463-477; common.cpp:8-185; ops.cpp:147-173, 220-263, 497-502
Summary:              The MTLComputeCommandEncoder is created with
                      MTLDispatchTypeConcurrent; a per-encoder ggml_mem_ranges
                      tracker decides whether to insert a memoryBarrierWithScope:Buffers.
Observation:          When use_concurrency is true (default; disabled by
                      GGML_METAL_CONCURRENCY_DISABLE env), the encoder is created
                      with MTLDispatchTypeConcurrent (device.m:469). For each node,
                      ggml_metal_op_concurrency_check (ops.cpp:159-165) asks
                      ggml_mem_ranges_check (common.cpp:124-153) whether the node's
                      src/dst ranges overlap with any prior range. If yes, the
                      encoder emits memoryBarrierWithScope:MTLBarrierScopeBuffers
                      (ops.cpp:152) and resets the tracker. If no, the dispatch can
                      run concurrently with prior dispatches in the same command
                      buffer. The tracker is O(N) per node, O(N²) per graph worst
                      case.
Evidence:             ggml-metal-device.m:463-477 (concurrent encoder);
                      ggml-metal-common.cpp:8-185 (range tracking);
                      ggml-metal-ops.cpp:147-173 (reset + check + add).
Architectural Impact: Maximizes in-flight dispatches without unnecessary barriers.
                      The O(N²) worst case is acceptable for typical graphs (N≈1000).
Correctness Impact:   None. The tracker is conservative: it only allows concurrency
                      when ranges are provably disjoint.
Optimization Type:    Asynchronous execution (concurrent dispatch within a command
                      buffer).
GwenLand Target:      glmetal, GATE
Recommendation:       ADOPT. Replicate the concurrent encoder + range tracker. For
                      very large graphs, consider an interval-tree to reduce to
                      O(N log N).
Priority:             High
Difficulty:           M
Dependencies:         R3
Confidence:           High
```

### Finding ARTX15-F11

```
Finding ID:           ARTX15-F11
Category:             EXECUTION_GRAPH
Engine:               Metal
Component:            Plan-time graph reorder
Source File:          ggml/src/ggml-metal/ggml-metal-common.cpp
Function:             ggml_metal_graph_optimize_reorder, ggml_graph_optimize
Lines:                209-457
Summary:              At plan time, the graph optimizer fuses ADD/MUL/NORM chains and
                      reorders nodes (forward search N=64, whitelist-limited) to
                      increase the concurrent-dispatch window.
Observation:          ggml_graph_optimize (line 375) first packs fusable ADD/MUL/NORM
                      chains into node_info structs (so reorder doesn't break fusion).
                      Then ggml_metal_graph_optimize_reorder (line 209) iterates
                      nodes; when a node is not concurrent with the current set, it
                      looks forward up to N_FORWARD=64 nodes for any node that is
                      concurrent with both the current set (mrs0) and all
                      unprocessed nodes in between (mrs1). Eligible ops are limited
                      by h_safe (lines 259-291): MUL_MAT, ROPE, NORM, RMS_NORM, ADD,
                      MUL, etc. — ~24 op types. The reordered + fused node list is
                      written back into gf->nodes[]. This runs on every
                      graph_compute call (unless GGML_METAL_GRAPH_OPTIMIZE_DISABLE
                      is set).
Evidence:             ggml-metal-common.cpp:209-373 (reorder), 375-457 (fuse +
                      reorder + unfuse), 326 (N_FORWARD=64), 259-291 (h_safe
                      whitelist).
Architectural Impact: Plan-time reorder increases the concurrent-dispatch window
                      for the encode-time tracker (F10). Running every graph_compute
                      is wasteful (W10); a plan-cache would help.
Correctness Impact:   None. Reorder only happens for provably-concurrent nodes.
Optimization Type:    Execution-graph reordering for concurrency.
GwenLand Target:      glmetal, GATE
Recommendation:       ADAPT. Move the optimizer into GATE as a plan-time pass with
                      result caching. Make N_FORWARD and the whitelist configurable.
Priority:             High
Difficulty:           L
Dependencies:         R7
Confidence:           High
```

### Finding ARTX15-F12

```
Finding ID:           ARTX15-F12
Category:             MEMORY_PATTERN
Engine:               Metal
Component:            Residency-set keep-alive
Source File:          ggml/src/ggml-metal/ggml-metal-device.m
Function:             ggml_metal_rsets_init, ggml_metal_buffer_rset_init, ggml_metal_device_rsets_keep_alive
Lines:                542-633, 1422-1493, 981-987
Summary:              On macOS 15+/iOS 18+, MTLResidencySet pins large Metal buffers
                      in GPU memory; a 5 ms background heartbeat thread requests
                      residency for 3 minutes after the last graph_compute.
Observation:          At device init (line 866-870), a ggml_metal_rsets collection
                      is created. ggml_metal_rsets_init (line 560) spawns a background
                      dispatch_group thread that loops every 5 ms; while
                      atomic d_loop > 0, it locks the set list and calls
                      [rset requestResidency] for every set, then decrements d_loop.
                      The countdown is reset to loops_per_s * keep_alive_s (= 200 *
                      180 = 36000) on every graph_compute via
                      ggml_metal_device_rsets_keep_alive (line 981). Each buffer
                      (shared or private) creates its own MTLResidencySet at alloc
                      time (line 1443-1478) and adds its MTLBuffer allocations to
                      it. The whole mechanism is gated by props.use_residency_sets
                      (env: GGML_METAL_NO_RESIDENCY).
Evidence:             ggml-metal-device.m:542-633 (rsets init + bg thread),
                      1422-1493 (per-buffer rset init), 866-870 (device-level
                      collection), 981-987 (keep_alive reset), 569-572
                      (keep_alive_s env).
Architectural Impact: Pins weights in GPU memory, avoiding eviction during
                      inference. The 3-minute default is generous; the env var
                      allows tuning. The 5 ms heartbeat is conservative.
Correctness Impact:   None. Residency is a performance hint, not a correctness
                      requirement.
Optimization Type:    None (memory pattern).
GwenLand Target:      glmetal
Recommendation:       ADOPT. Replicate the heartbeat thread + countdown design.
                      Consider a shorter default keep_alive_s (e.g. 60s) for
                      interactive workloads.
Priority:             Medium
Difficulty:           M
Dependencies:         R5
Confidence:           High
```

### Finding ARTX15-F13

```
Finding ID:           ARTX15-F13
Category:             BACKEND_DESIGN
Engine:               Metal
Component:            Async transfers and events
Source File:          ggml/src/ggml-metal/ggml-metal-context.m, ggml/src/ggml-metal/ggml-metal-device.m
Function:             ggml_metal_set_tensor_async, ggml_metal_get_tensor_async, ggml_metal_cpy_tensor_async, ggml_metal_event_record, ggml_metal_event_wait
Lines:                context.m:307-436, 627-661; device.m:989-1039
Summary:              Async transfers use MTLBlitCommandEncoder; cross-context copies
                      use MTLSharedEvent signal/wait; all async command buffers are
                      retained in cmd_bufs_ext until the next synchronize.
Observation:          set_tensor_async (context.m:307) allocates a temporary shared
                      MTLBuffer from the source data (line 311 — a copy), encodes a
                      Blit copy, commits, and adds the cmd_buf to cmd_bufs_ext.
                      get_tensor_async (line 351) wraps the destination host pointer
                      with newBufferWithBytesNoCopy (zero-copy). cpy_tensor_async
                      (line 395) encodes a Blit copy in the source context, signals
                      ctx_src->ev_cpy (a MTLSharedEvent created at device init,
                      device.m:1011-1020), then encodes a wait on the same event in
                      the destination context (context.m:432). event_record (line 627)
                      and event_wait (line 643) encode signal/wait on a
                      user-supplied event. All paths retain the cmd_buf in
                      cmd_bufs_ext and update cmd_buf_last.
Evidence:             ggml-metal-context.m:307-349 (set_async), 351-393 (get_async),
                      395-436 (cpy_async), 627-661 (event_record/wait);
                      ggml-metal-device.m:989-1039 (event encode signal/wait),
                      1011-1020 (event init).
Architectural Impact: The MTLSharedEvent idiom is the correct Metal way to
                      synchronize across command queues. The cmd_bufs_ext retention
                      is simple but leaks memory if synchronize is never called —
                      acceptable because synchronize is called at graph_compute
                      boundaries.
Correctness Impact:   None. Events and Blit copies are correct by construction.
Optimization Type:    Asynchronous execution (Blit + event-encoded sync).
GwenLand Target:      glmetal
Recommendation:       ADOPT. Add a ring-buffer pool for the temporary shared
                      MTLBuffer in set_tensor_async (W11) to avoid per-call
                      allocation.
Priority:             High
Difficulty:           S
Dependencies:         R6
Confidence:           High
```

### Finding ARTX15-F14

```
Finding ID:           ARTX15-F14
Category:             BACKEND_DESIGN
Engine:               Metal
Component:            Op-fusion engine
Source File:          ggml/src/ggml-metal/ggml-metal-ops.cpp
Function:             ggml_metal_op_bin, ggml_metal_op_norm, ggml_metal_op_can_fuse_snake, ggml_metal_op_snake_fused
Lines:                3081-3125, 3127-3288, 3409-3546
Summary:              Encode-time op fusion supports three patterns: bin×N (up to 8
                      chained ADDs), norm+mul+add (3-op), and snake (5-op
                      MUL+SIN+SQR+MUL+ADD). Fusion decisions use ggml_can_fuse_ext
                      plus per-pattern shape/type/contiguity checks.
Observation:          bin×N: at encode time, the encoder peeks up to 7 nodes ahead;
                      if each is ADD with matching src1 layout and same Metal buffer,
                      it fuses up to 8 ADDs into a single dispatch. The n_fuse count
                      is passed as a function constant to the kernel
                      (FC_BIN+1, device.cpp:1582). norm+mul+add: fuses
                      NORM/RMS_NORM + MUL + ADD into a single kernel
                      (kernel_rms_norm_mul_add_f32). snake: structural 5-op pattern
                      MUL→SIN→SQR→MUL→ADD detected by ggml_metal_op_can_fuse_snake
                      (line 3081) with strict shape/type/contiguity checks. Each
                      fusion returns n_fuse > 1, and the caller skips the fused
                      nodes.
Evidence:             ggml-metal-ops.cpp:3127-3288 (bin×N), 3409-3546 (norm+mul+add),
                      3081-3125 (snake can_fuse), 4040+ (snake_fused encode).
Architectural Impact: Fusion at encode time (not plan time) means the decision is
                      re-evaluated every graph_compute. The bin×N cap of 8 is
                      arbitrary. The snake pattern is structural (no op hints).
Correctness Impact:   None. Each fused kernel is verified against the unfused path.
Optimization Type:    Kernel fusion.
GwenLand Target:      glmetal, GATE
Recommendation:       ADAPT. Move fusion detection to plan time (cache the result).
                      Keep the three patterns; make the bin×N cap configurable.
Priority:             High
Difficulty:           M
Dependencies:         R7
Confidence:           High
```

---

## 18. Unknowns

* **U1**. Whether `n_cb == 2` is faster than `n_cb == 1` on M3/M4 for
  typical llama.cpp workloads. The warning at
  `ggml-metal-context.m:667-669` cites M1 Pro / M2 Ultra tests; M3/M4
  Ultra may differ. Requires runtime profiling.

* **U2**. Whether the `ne11_mm_min = 8` cutoff is optimal for any
  Apple Silicon generation. The comment at
  `ggml-metal-ops.cpp:2101-2107` admits the `nsg` for `mul_mv_ext`
  is untuned. Requires per-device microbenchmarking.

* **U3**. The actual hit rate of the pipeline cache after the first
  graph_compute. For a fixed model, every subsequent graph_compute
  should hit the cache (assuming the same shapes). For dynamic
  shapes (e.g. variable batch), cache misses occur. Requires PMU
  analysis of the `std::unordered_map` lookup.

* **U4**. Whether `MTLBinaryArchiver` (Metal 3) can persist the
  pipeline cache across runs. Not used upstream; static analysis
  cannot determine whether it would actually eliminate cold-start
  compile cost. Requires experimentation.

* **U5**. The real-world cost of the O(N²) memory-range overlap
  tracker for very large graphs (N > 10000). The tracker is reset
  on every barrier, so steady-state N is smaller — but worst-case
  is still O(N²). Requires profiling on large-graph workloads.

* **U6**. Whether the `MTLSharedEvent`-based `cpy_tensor_async`
  actually overlaps with graph compute. The signal is encoded in
  the source context's queue; the wait is encoded in the
  destination context's queue. If both contexts share the same
  device queue (F01-F03), the wait blocks the destination's
  graph_compute. Requires multi-device testing.

* **U7**. Whether the experimental tensor API (`has_tensor`) ever
  outperforms simdgroup on M5 / A19. The code disables it by name
  match for pre-M5 chips (`ggml-metal-device.m:718-725`); static
  analysis cannot predict M5 performance. Requires hardware that
  is not yet available.

* **U8**. The retention behavior of `cmd_bufs_ext` if
  `synchronize` is never called. The NSMutableArray grows
  unbounded; each entry is a retained `MTLCommandBuffer`. Whether
  Metal's internal pool would GC them under memory pressure is
  unclear. Requires memory profiling.

* **U9**. Whether the `AGX_RELAX_CDM_CTXSTORE_TIMEOUT=1` env var
  workaround (`ggml-metal.cpp:926`) is still needed on current
  macOS versions. The comment cites issue #20141; static analysis
  cannot determine if the underlying bug is fixed. Requires
  testing on macOS 15+.

---

## 19. References

| Reference | File                                                | Function / Symbol                                | Lines                |
| --------- | --------------------------------------------------- | ------------------------------------------------ | -------------------- |
| R01       | `ggml/src/ggml-metal/ggml-metal.cpp`                | `ggml_backend_metal_i` (backend vtable)          | 568–585              |
| R02       | `ggml/src/ggml-metal/ggml-metal.cpp`                | `ggml_backend_metal_device_i` (device vtable)    | 797–813              |
| R03       | `ggml/src/ggml-metal/ggml-metal.cpp`                | `ggml_backend_metal_reg_i` (registry vtable)     | 881–886              |
| R04       | `ggml/src/ggml-metal/ggml-metal.cpp`                | `ggml_backend_metal_reg` (registry init)         | 908–948              |
| R05       | `ggml/src/ggml-metal/ggml-metal.cpp`                | `ggml_backend_metal_buffer_type_shared/private/mapped` | 188–474         |
| R06       | `ggml/src/ggml-metal/ggml-metal.cpp`                | `ggml_backend_metal_device_get_props` (caps)     | 672–685              |
| R07       | `ggml/src/ggml-metal/ggml-metal-context.h`          | `ggml_metal_t` public interface                  | 1–41                 |
| R08       | `ggml/src/ggml-metal/ggml-metal-context.m`          | `struct ggml_metal` (context)                    | 26–82                |
| R09       | `ggml/src/ggml-metal/ggml-metal-context.m`          | `ggml_metal_init`                                | 84–187               |
| R10       | `ggml/src/ggml-metal/ggml-metal-context.m`          | `ggml_metal_graph_compute`                       | 438–615              |
| R11       | `ggml/src/ggml-metal/ggml-metal-context.m`          | `ggml_metal_set/get_tensor_async`                | 307–393              |
| R12       | `ggml/src/ggml-metal/ggml-metal-context.m`          | `ggml_metal_cpy_tensor_async`                    | 395–436              |
| R13       | `ggml/src/ggml-metal/ggml-metal-context.m`          | `ggml_metal_event_record/wait`                   | 627–661              |
| R14       | `ggml/src/ggml-metal/ggml-metal-context.m`          | `ggml_metal_set_n_cb` (encode_async block)       | 663–722              |
| R15       | `ggml/src/ggml-metal/ggml-metal-device.h`           | `ggml_metal_device_t`, pipeline/encoder/buffer   | 1–326                |
| R16       | `ggml/src/ggml-metal/ggml-metal-device.m`           | `struct ggml_metal_device`                       | 520–536              |
| R17       | `ggml/src/ggml-metal/ggml-metal-device.m`           | `ggml_metal_device_init`                         | 679–918              |
| R18       | `ggml/src/ggml-metal/ggml-metal-device.m`           | `ggml_metal_library_init` (library load)         | 106–260              |
| R19       | `ggml/src/ggml-metal/ggml-metal-device.m`           | `ggml_metal_library_compile_pipeline`            | 369–453              |
| R20       | `ggml/src/ggml-metal/ggml-metal-device.m`           | `ggml_metal_encoder_init` (concurrent dispatch)  | 463–477              |
| R21       | `ggml/src/ggml-metal/ggml-metal-device.m`           | `ggml_metal_rsets_init` (keep-alive thread)      | 560–633              |
| R22       | `ggml/src/ggml-metal/ggml-metal-device.m`           | `ggml_metal_buffer_init` (shared vs private)     | 1515–1583            |
| R23       | `ggml/src/ggml-metal/ggml-metal-device.m`           | `ggml_metal_buffer_map` (multi-view)             | 1585–1678            |
| R24       | `ggml/src/ggml-metal/ggml-metal-device.m`           | `ggml_metal_buffer_get_id` (view lookup)         | 1893–1916            |
| R25       | `ggml/src/ggml-metal/ggml-metal-device.m`           | `ggml_metal_event_*` (MTLSharedEvent)            | 989–1039             |
| R26       | `ggml/src/ggml-metal/ggml-metal-device.m`           | `ggml_metal_device_supports_op`                  | 1051–1375            |
| R27       | `ggml/src/ggml-metal/ggml-metal-device.cpp`         | `ggml_metal_pipelines` (unordered_map cache)     | 28–60                |
| R28       | `ggml/src/ggml-metal/ggml-metal-device.cpp`         | `ggml_metal_library_get_pipeline_mul_mv`         | 766–955              |
| R29       | `ggml/src/ggml-metal/ggml-metal-device.cpp`         | `ggml_metal_library_get_pipeline_mul_mm`         | 704–764              |
| R30       | `ggml/src/ggml-metal/ggml-metal-device.cpp`         | `ggml_metal_library_get_pipeline_mul_mv_ext`     | 671–702              |
| R31       | `ggml/src/ggml-metal/ggml-metal-device.cpp`         | `ggml_metal_library_get_pipeline_flash_attn_ext` | 1395–1458            |
| R32       | `ggml/src/ggml-metal/ggml-metal-ops.h`              | per-op encoder declarations                      | 1–100                |
| R33       | `ggml/src/ggml-metal/ggml-metal-ops.cpp`            | `struct ggml_metal_op` (per-encoder state)       | 28–111               |
| R34       | `ggml/src/ggml-metal/ggml-metal-ops.cpp`            | `ggml_metal_op_encode_impl` (dispatch switch)    | 175–505              |
| R35       | `ggml/src/ggml-metal/ggml-metal-ops.cpp`            | `ggml_metal_op_mul_mat` (3-way heuristic)        | 2043–2273            |
| R36       | `ggml/src/ggml-metal/ggml-metal-ops.cpp`            | `ggml_metal_op_mul_mat_id`                       | 2292–2478            |
| R37       | `ggml/src/ggml-metal/ggml-metal-ops.cpp`            | `ggml_metal_op_flash_attn_ext`                   | 2650–3126            |
| R38       | `ggml/src/ggml-metal/ggml-metal-ops.cpp`            | `ggml_metal_op_flash_attn_ext_use_vec`           | 2526–2534            |
| R39       | `ggml/src/ggml-metal/ggml-metal-ops.cpp`            | `ggml_metal_op_bin` (bin×N fusion)               | 3127–3288            |
| R40       | `ggml/src/ggml-metal/ggml-metal-ops.cpp`            | `ggml_metal_op_norm` (norm+mul+add fusion)       | 3409–3546            |
| R41       | `ggml/src/ggml-metal/ggml-metal-ops.cpp`            | `ggml_metal_op_can_fuse_snake`                   | 3081–3125            |
| R42       | `ggml/src/ggml-metal/ggml-metal-common.cpp`         | `ggml_mem_ranges` (overlap tracker)              | 8–185                |
| R43       | `ggml/src/ggml-metal/ggml-metal-common.cpp`         | `ggml_metal_graph_optimize_reorder`              | 209–373              |
| R44       | `ggml/src/ggml-metal/ggml-metal-common.cpp`         | `ggml_graph_optimize` (fuse + reorder + unfuse)  | 375–457              |
| R45       | `ggml/src/ggml-metal/ggml-metal-impl.h`             | Function-constant offsets (FC_*)                 | 91–106               |
| R46       | `ggml/src/ggml-metal/ggml-metal-impl.h`             | Per-dtype N_SG/N_R0 constants                    | 24–88                |
| R47       | `ggml/src/ggml-metal/ggml-metal-impl.h`             | `ggml_metal_kargs_*` structs                     | 158–1222             |
| R48       | `ggml/src/ggml-metal/ggml-metal.metal`              | `kernel_mul_mv_*`, `kernel_mul_mm_*`             | 3687+, 10374+, 10684 |
| R49       | `ggml/src/ggml-metal/ggml-metal.metal`              | `kernel_flash_attn_ext*`                         | 6180+, 6991+, 7218+  |
| R50       | `ggml/src/ggml-metal/ggml-metal.metal`              | `kernel_bin_fuse_impl`, `kernel_rms_norm_fuse_impl` | 1265+, 3112+      |
