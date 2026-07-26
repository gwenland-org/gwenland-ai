# ARTX08 — CUDA Backend Core

**Audited commit:** `555881ebc8b0fc0402b30e09258a32a7bfd13c52`
**Last updated:** 2026-07-25
**Auditor:** Percival-aux
**Target GwenLand module:** `glcuda` (CUDA backend), `GATE` (graph execution, multi-stream)

---

## 1. Executive Summary

The CUDA backend of llama.cpp is the *contract layer* between the ggml
graph scheduler and the per-op CUDA kernels (cuBLAS, custom MMQ/MMVF/MMVQ,
fattn). Unlike the CPU backend (ARTX01), it is **inherently asynchronous**:
every backend exposes a stream pool, async tensor copy, CUDA events, and a
graph compute path that can capture the cgraph into a `cudaGraphExec`.

The CUDA backend provides:

1. A **backend interface** (`ggml_backend_cuda_interface` /
   `_device_interface` / `_reg_interface`) with `async=true`, `events=true`,
   `buffer_from_host_ptr=false`. All async APIs and the event system are
   implemented, in contrast to the CPU backend which sets them all to `NULL`.
2. A **stream pool** sized `streams[GGML_CUDA_MAX_DEVICES][GGML_CUDA_MAX_STREAMS]`
   (= 16 × 8), with lazy `cudaStreamCreateWithFlags(cudaStreamNonBlocking)`.
   The pool is sized for the global device array but each backend context
   owns exactly one device, so most slots are unused (Finding ARTX08-F02).
3. **Two allocator backends** for scratch buffers: a legacy best-fit 256-slot
   pool (`ggml_cuda_pool_leg`) and a VMM pool (`ggml_cuda_pool_vmm`)
   guarded by `GGML_USE_VMM`.
4. A **graph compute path** (`ggml_backend_cuda_graph_compute`) that captures
   the cgraph into a `cudaGraph_t` on the second stable execution, then
   reuses it via `cudaGraphExecUpdate` / `cudaGraphLaunch`. A warmup counter
   gates reuse; any property change resets warmup.
5. An **op dispatch** in `ggml_cuda_compute_forward` as a ~150-case
   `switch (dst->op)`, *not* a function-pointer table (contrast ARTX01-F03).
6. A **fusion engine** (`ggml_cuda_try_fuse`) with ~12 patterns including
   FFN-style `MUL_MAT+ADD+MUL_MAT+ADD+GLU`, `RMS_NORM+MUL[+ADD]`,
   `SSM_CONV+ADD+SILU`, `ROPE+VIEW+SET_ROWS`, `topk-moe`, `snake`,
   `gated_delta_net+cpy`, plus a QKV multi-stream concurrency pass.
7. A **communication context** for multi-GPU AllReduce via NCCL (Linux
   default), an internal 2-GPU AR pipeline, or a meta-backend butterfly
   fallback.
8. A **pinned host buffer type** using `cudaMallocHost`, plus opt-in
   `cudaHostRegister` for externally allocated buffers.

For GwenLand, the decisions worth **ADOPT**ing are the lazy stream pool,
the VMM pool with `cuMemSetAccess` peer grants, the CUDA graph warmup +
`cudaGraphExecUpdate` path, the CC-encoding scheme, and PDL on Hopper. The
decisions worth **REJECT**ing are the over-allocated per-context arrays,
the dead `ggml_tensor_extra_gpu` and `default_tensor_split` remnants of
the removed split-buffer mechanism, and the execution-time fusion
detection (same anti-pattern as ARTX01-F08).

---

## 2. Purpose

Provide a CUDA execution backend for the ggml graph that:

* dispatches every supported `ggml_op` to a CUDA kernel or cuBLAS call,
* exposes a fully asynchronous backend interface (streams, events, async
  copy) so the scheduler can overlap compute with transfer,
* captures a stable cgraph into a `cudaGraphExec` for replay,
* supports multi-GPU tensor-parallel inference via NCCL AllReduce (or an
  internal pipeline for the 2-GPU case),
* supports pinned host memory and peer-to-peer device copies,
* auto-selects kernels based on runtime compute-capability detection
  (Pascal, Volta, Turing, Ampere, Ada, Hopper, Blackwell; AMD GCN/CDNA/
  RDNA; Moore Threads QY/PH),
* exposes a standard `ggml_backend` interface to the rest of ggml.

It is **not** responsible for: graph construction, graph optimization
beyond fusion and QKV reordering (delegated to GATE), memory allocation
policy (delegated to `ggml-alloc.c`), or cross-backend op selection
(delegated to the scheduler — ARTX22).

---

## 3. Source Files

| File                                          | Lines | Role                                                                              |
| --------------------------------------------- | ----- | --------------------------------------------------------------------------------- |
| `ggml/src/ggml-cuda/ggml-cuda.cu`             | 5425  | Backend / device / registry interface, op dispatch, graph compute, buffer types,  |
|                                               |       | async APIs, events, peer-to-peer, AllReduce context, pinned memory, fusion,       |
|                                               |       | cuBLAS matmul wrapper, MMQ/MMV/MMVQ routing, op support table.                    |
| `ggml/src/ggml-cuda/common.cuh`               | 1661  | CC capability constants and detection helpers, `CUDA_CHECK` / `CUBLAS_CHECK` /    |
|                                               |       | `NCCL_CHECK` / `CU_CHECK` macros, `ggml_cuda_device_info` struct, pool interface, |
|                                               |       | `ggml_cuda_pool_alloc<T>` RAII helper, `ggml_tensor_extra_gpu` (dead),            |
|                                               |       | `ggml_cuda_graph` struct, `ggml_cuda_concurrent_event`, `ggml_backend_cuda_context`,|
|                                               |       | `ggml_cuda_type_traits<ggml_type>` template specializations, PDL launch helper.   |
| `ggml/include/ggml-cuda.h`                    | 47    | Public API: `ggml_backend_cuda_init`, `ggml_backend_cuda_buffer_type`,            |
|                                               |       | `ggml_backend_cuda_host_buffer_type`, `GGML_CUDA_MAX_DEVICES = 16`.               |
| `ggml/src/ggml-cuda/vendors/{cuda,hip,musa}.h`| ~150  | Vendor-alias macros that map `cu*` / `cuda*` to `hip*` / `musa*` equivalents.     |

> Note: the audit prompt's reference to a `ggml_backend_cuda_split_buffer_type()`
> function reflects an **older** state of the codebase. At the audited commit
> the CUDA backend exposes **no** split buffer type; multi-GPU tensor
> parallelism is implemented via the NCCL/internal-AR AllReduce path. The
> SYCL backend still has `ggml_backend_sycl_split_buffer_type()` — see
> Finding ARTX08-F04.

---

## 4. Architecture Overview

```
                ┌────────────────────────────────────────────────┐
                │  ggml-cuda.cu : ggml_backend_cuda_reg /         │
                │                 _interface / _device_interface /│
                │                 _reg_interface                  │
                │  (plugs CUDA into ggml backend registry)        │
                └────────────────────────────────────────────────┘
                              │
                              ▼
                ┌────────────────────────────────────────────────┐
                │  ggml_backend_cuda_context  (one per backend)   │
                │  ├─ device: int                                 │
                │  ├─ streams[16][8]   (lazy cudaStreamNonBlocking)│
                │  ├─ cublas_handles[16] (lazy, TF32 math mode)   │
                │  ├─ pools[16][8]     (legacy best-fit or VMM)   │
                │  ├─ copy_event       (lazy, cudaEventDisableTiming)│
                │  ├─ cuda_graphs map  (per first-node-ptr)       │
                │  └─ concurrent_stream_context (QKV fork/join)   │
                └────────────────────────────────────────────────┘
                              │
              ┌───────────────┼────────────────┐
              ▼               ▼                ▼
   ┌──────────────────┐  ┌──────────────────┐ ┌────────────────────┐
   │ buffer types     │  │  op dispatch     │ │ graph compute      │
   │  cuda_buffer_type│  │  switch(op) →    │ │  cudaStreamBegin-  │
   │  host_buffer_type│  │  ggml_cuda_<op>  │ │  Capture → graph   │
   │  (no split type) │  │  ~150 cases      │ │  → cudaGraphLaunch │
   └──────────────────┘  └──────────────────┘ └────────────────────┘
                              │
                              ▼
                ┌────────────────────────────────────────────────┐
                │  kernel routers                                │
                │  ├─ ggml_cuda_mul_mat  → mmvf / mmf / mmvq /    │
                │  │                       mmq / cublas           │
                │  ├─ ggml_cuda_mul_mat_id                       │
                │  ├─ ggml_cuda_flash_attn_ext                   │
                │  └─ ggml_cuda_try_fuse (~12 fusion patterns)   │
                └────────────────────────────────────────────────┘
                              │
                              ▼
                ┌────────────────────────────────────────────────┐
                │  common.cuh : CC detection, error macros,       │
                │  type_traits<ggml_type>, pool_alloc<T>, PDL     │
                │  ggml_cuda_kernel_launch (PDL-aware launcher)   │
                └────────────────────────────────────────────────┘
                              │
                              ▼
                ┌────────────────────────────────────────────────┐
                │  per-op .cu files (mmq.cu, fattn*.cu, norm.cu,  │
                │  rope.cu, ...) + template-instances/            │
                │  (per-dtype MMQ/MMF/fattn-vec instantiations)   │
                └────────────────────────────────────────────────┘
```

Key design points:

* **C++ throughout**. Unlike the CPU backend (which keeps its dispatch in C
  and only uses C++ for the `extra_buffer_type` plugin), the CUDA backend is
  C++ end-to-end: `ggml_backend_cuda_context`, `ggml_cuda_pool_alloc<T>`,
  `std::unique_ptr`, `std::unordered_map`, `std::vector`, `std::mutex`,
  `std::condition_variable`.
* **Async by default**. The backend's `ggml_backend_i` vtable at
  `ggml-cuda.cu:4428-4445` implements every async hook
  (`set_tensor_async`, `get_tensor_async`, `set_tensor_2d_async`,
  `get_tensor_2d_async`, `cpy_tensor_async`, `synchronize`,
  `event_record`, `event_wait`, `graph_optimize`). The CPU vtable (ARTX01-F01)
  sets all of these to `NULL`.
* **No split-buffer type**. The CUDA backend exposes
  `ggml_backend_cuda_buffer_type(device)` and
  `ggml_backend_cuda_host_buffer_type()` but no
  `ggml_backend_cuda_split_buffer_type`. Multi-GPU tensor parallelism goes
  through the comm-context AllReduce path (NCCL / internal / butterfly),
  not through a split weight tensor.
* **Per-backend stream pool, not per-device**. The context stores
  `streams[GGML_CUDA_MAX_DEVICES][GGML_CUDA_MAX_STREAMS]` (= 16 × 8 = 128
  stream pointers) even though each backend owns exactly one device. Only
  `streams[device][*]` is ever touched. See Finding ARTX08-F02.

---

## 5. Execution Flow

### 5.1 Top-level entry

`ggml_backend_cuda_graph_compute` (`ggml-cuda.cu:4100`)

1. `ggml_cuda_set_device(cuda_ctx->device)` — cheap; skips `cudaSetDevice`
   if the current device is already correct (`ggml-cuda.cu:118-130`).
2. If `USE_CUDA_GRAPH` is defined, fetch the per-cgraph `ggml_cuda_graph`
   keyed by `cgraph->nodes[0]` (the first node pointer — see
   `ggml_cuda_graph_get_key`, `ggml-cuda.cu:2531`).
3. Run `ggml_cuda_graph_check_compability(cgraph)` (`:2496`) — disables
   graphs if any `MUL_MAT_ID` node cannot use the
   `[TAG_MUL_MAT_ID_CUDA_GRAPHS]` fast path (i.e., it would synchronize
   the stream).
4. Run `ggml_cuda_graph_update_required(cuda_ctx, cgraph)` (`:2535`) —
   compares per-node `node_properties` (data pointers + ne/nb for all
   `GGML_MAX_SRC` sources) against the cached graph. If unchanged and the
   graph is already instantiated, reuse is signaled.
5. Apply the **warmup rule** (`:4120-4139`): the first time a graph key is
   seen, execute directly (no capture). On the second consecutive call
   with no property changes, mark `warmup_complete = true` and begin
   capture. Any later property change resets `warmup_complete = false`
   and reverts to direct execution.
6. If capture is needed:
   * Take `ggml_cuda_lock` and increment `ggml_cuda_lock_counter` —
     prevents other backends from destroying cuBLAS handles mid-capture
     (`:4147-4149`, `:695-700`).
   * `cudaStreamBeginCapture(cuda_ctx->stream(), cudaStreamCaptureModeRelaxed)`.
7. Call `ggml_cuda_graph_evaluate_and_capture(cuda_ctx, cgraph, ...)` (`:3869`).
8. After capture, end capture, instantiate (`cudaGraphInstantiate`) or
   update (`cudaGraphExecUpdate`) the executable graph, and launch via
   `cudaGraphLaunch(graph->instance, cuda_ctx->stream())` (`:4075`).

### 5.2 Per-node execution (inside capture or direct eval)

`ggml_cuda_graph_evaluate_and_capture` (`ggml-cuda.cu:3869`)

1. If concurrent events are scheduled, restore the original (non-interleaved)
   node order within each fork/join region so fusion within a stream can
   still see adjacent nodes (`:3913-3962`).
2. For each node `i` in `cgraph->nodes`:
   a. If we are inside a concurrent-event region, route the node to its
      assigned stream via `cuda_ctx->curr_stream_no =
      concurrent_event->stream_mapping[node]` (`:3984-3986`).
   b. If the previous node was fused, try to launch the next
      `try_launch_concurrent_event` (`:3988-3997`).
   c. Skip view/no-op nodes (`ggml_cuda_is_view_or_noop`, `:2490`).
   d. Skip nodes without `GGML_TENSOR_FLAG_COMPUTE` (`:4005`).
   e. **Try fusion**: `nodes_to_skip = ggml_cuda_try_fuse(cuda_ctx, cgraph, i)`
      (`:4009`). If non-zero, skip ahead.
   f. Otherwise call `ggml_cuda_compute_forward(*cuda_ctx, node)`.
   g. After the node, if not already in a concurrent region, try
      `try_launch_concurrent_event(node)` — forks a new stream group.
3. On the join node, record `join_events[i]` on each forked stream and
   `cudaStreamWaitEvent` on the main stream (`:3972-3981`).

### 5.3 Op dispatch

`ggml_cuda_compute_forward` (`ggml-cuda.cu:2011`)

```c++
switch (dst->op) {
    case GGML_OP_ARGMAX:        ggml_cuda_argmax(ctx, dst); break;
    case GGML_OP_MUL_MAT:       ggml_cuda_mul_mat(ctx, src0, src1, dst); break;
    case GGML_OP_MUL_MAT_ID:    ggml_cuda_mul_mat_id(ctx, dst); break;
    case GGML_OP_FLASH_ATTN_EXT: ggml_cuda_flash_attn_ext(ctx, dst); break;
    /* ... ~150 cases ... */
    default: return false;
}
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) { GGML_LOG_ERROR(...); CUDA_CHECK(err); }
return true;
```

There is no per-op function-pointer table; the dispatch is a literal
`switch`. After every op, `cudaGetLastError()` is polled so any kernel
launch failure surfaces as an abort (via `ggml_cuda_error`,
`ggml-cuda.cu:97-107`).

### 5.4 Matmul hot path

`ggml_cuda_mul_mat` (`ggml-cuda.cu:1812`)

1. If `GGML_HINT_SRC0_IS_HADAMARD` and `ggml_cuda_op_fwht` succeeds, return
   (Hadamard fast path, same op-hint mechanism as ARTX01).
2. If `bad_padding_clear || src1->type != F32 || dst->type != F32`, fall
   through to cuBLAS.
3. Read `cc = ggml_cuda_info().devices[ctx.device].cc` and `warp_size`.
4. Try in order, returning on the first hit:
   * `ggml_cuda_should_use_mmvf` → `ggml_cuda_mul_mat_vec_f` (custom
     F16/BF16/F32 vector kernel for thin src0 without tensor cores).
   * `ggml_cuda_should_use_mmf` → `ggml_cuda_mul_mat_f` (custom matmul,
     small-M tile-mm kernel).
   * `ggml_cuda_should_use_mmvq` → `ggml_cuda_mul_mat_vec_q` (vector
     kernel for quantized src0, `ne11 <= MMVQ_MAX_BATCH_SIZE`).
   * `ggml_cuda_should_use_mmq` → `ggml_cuda_mul_mat_q` (custom quantized
     matmul, the CUDA analog of the CPU `vec_dot` kernel).
   * else → `ggml_cuda_mul_mat_cublas`.
5. `ggml_cuda_mul_mat_cublas` (`:1619`) selects compute type from `cc`,
   op_params, and env `GGML_CUDA_CUBLAS_COMPUTE_TYPE`, then dispatches
   to `ggml_cuda_mul_mat_cublas_impl<F32|BF16|F16>` (`:1405`).
6. `ggml_cuda_mul_mat_cublas_impl` may use `cublasSgemm`, `cublasGemmEx`,
   `cublasGemmStridedBatchedEx`, or `cublasGemmBatchedEx` depending on
   shape (single matrix, batched without broadcast, or batched with
   broadcast — the last path launches `k_compute_batched_ptrs` to fill
   the per-batch pointer arrays).

### 5.5 MUL_MAT_ID hot path

`ggml_cuda_mul_mat_id` (`ggml-cuda.cu:1854`)

1. If the expert count and batch fit the MMVQ/MMVF fast path
   (`ne2 <= MMVQ_MAX_BATCH_SIZE`, `[TAG_MUL_MAT_ID_CUDA_GRAPHS]`), use
   `ggml_cuda_mul_mat_vec_q` or `ggml_cuda_mul_mat_vec_f` directly.
   This path is **graph-safe** (no stream synchronization).
2. Otherwise fall back to the **sorting path**: copy `ids` to host,
   `cudaStreamSynchronize`, sort tokens per expert on CPU, build
   `ids_to_sorted` / `ids_from_sorted` device buffers, run `get_rows_cuda`
   to gather src1 per-expert, then call `ggml_cuda_mul_mat` per expert
   slice, and finally `get_rows_cuda` to scatter back. **This path
   synchronizes the stream** and is therefore disabled under CUDA graph
   capture (`ggml_cuda_graph_check_compability`, `:2509-2521`).

---

## 6. Data Layout

### 6.1 Tensor descriptor

Same `ggml_tensor` (`ne[GGML_MAX_DIMS]`, `nb[GGML_MAX_DIMS]`) as CPU. The
CUDA backend requires, for the matmul path:

| Constraint                                            | Source                |
| ----------------------------------------------------- | --------------------- |
| `nb00 == ggml_type_size(src0->type)` (cuBLAS path)    | `ggml-cuda.cu:1422`   |
| `nb10 == ggml_type_size(src1->type)` (cuBLAS path)    | `ggml-cuda.cu:1428`   |
| `dst` contiguous (cuBLAS path)                        | `ggml-cuda.cu:1410`   |
| `nb0 == ggml_element_size(a)` (supports_op for MUL_MAT)| `ggml-cuda.cu:4782`  |
| `src0` row-contiguous in innermost dim (cuBLAS path)  | required by GemmEx    |

### 6.2 Quantized weight padding (`MATRIX_ROW_PADDING = 512`)

`common.cuh:176` defines `MATRIX_ROW_PADDING = 512`. For quantized
tensors, `ggml_backend_cuda_buffer_type_get_alloc_size` (`ggml-cuda.cu:906`)
adds up to `MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING` bytes of padding
to the last row. The padding is zeroed in `ggml_backend_cuda_buffer_init_tensor`
(`:761-769`) so that MMQ/MMVQ vecdot kernels can issue full-vector loads
past the end of the actual data without reading NaNs. This is the CUDA
analog of the CPU backend's implicit row padding but explicitly allocated.

### 6.3 `ggml_tensor_extra_gpu` — dead

`common.cuh:1213-1216` declares:

```cpp
struct ggml_tensor_extra_gpu {
    void * data_device[GGML_CUDA_MAX_DEVICES];                       // 16 ptrs
    cudaEvent_t events[GGML_CUDA_MAX_DEVICES][GGML_CUDA_MAX_STREAMS]; // 16×8 events
};
```

A `grep` for `ggml_tensor_extra_gpu` in `ggml/src/ggml-cuda/` finds only
the declaration site. **The struct is never instantiated or accessed.** It
is a remnant of the removed split-buffer mechanism. See Finding ARTX08-F03.

### 6.4 `default_tensor_split` — dead

`ggml_cuda_device_info::default_tensor_split[GGML_CUDA_MAX_DEVICES]`
(`common.cuh:1151`) is populated in `ggml_cuda_init` (`:302`, `:386`) but
never read again anywhere in `ggml-cuda.cu`. The model layer
(`llama-model.cpp:949-967`) calls `ggml_backend_reg_get_proc_address(reg,
"ggml_backend_split_buffer_type")` — but the CUDA backend's
`ggml_backend_cuda_reg_get_proc_address` (`:5317`) does **not** expose this
symbol, so the model falls back to layer-split mode. See Finding ARTX08-F04.

---

## 7. Memory Layout

### 7.1 Per-backend context (`ggml_backend_cuda_context`)

`common.cuh:1407-1518` defines:

```cpp
struct ggml_backend_cuda_context {
    int device;
    std::string name;
    cudaEvent_t copy_event = nullptr;
    cudaStream_t streams[16][8] = { { nullptr } };   // 128 ptrs = 1 KiB
    cublasHandle_t cublas_handles[16] = { nullptr }; //  16 ptrs = 128 B
    int curr_stream_no = 0;
    std::unordered_map<const void*, std::unique_ptr<ggml_cuda_graph>> cuda_graphs;
    int64_t last_graph_eviction_sweep = 0;
    ggml_cuda_stream_context concurrent_stream_context;
    std::unique_ptr<ggml_cuda_pool> pools[16][8];    // 128 unique_ptrs = 1 KiB
};
```

Total fixed-size state per backend context: **~2.3 KiB** before any stream
or pool is created. Most of it (the `streams`/`cublas_handles`/`pools`
arrays for 15 other devices) is never used because each backend owns
exactly one device. See Finding ARTX08-F02.

### 7.2 Pool types

Two scratch-buffer pools are provided, selected at construction time by
`ggml_backend_cuda_context::new_pool_for_device` (`ggml-cuda.cu:685-693`):

* `ggml_cuda_pool_leg` (`:419-532`): a 256-slot best-fit free-list with
  5 % size look-ahead and `cudaFree` fallback when the pool is full. On
  allocation failure, it `cudaDeviceSynchronize`s, clears the pool, and
  retries once (`:496-507`).
* `ggml_cuda_pool_vmm` (`:536-682`): a virtual-memory pool backed by
  `cuMemCreate` / `cuMemMap` / `cuMemAddressReserve`. Reserves a 32 GiB
  virtual range per pool (`CUDA_POOL_VMM_MAX_SIZE = 1ull << 35`) and
  grows it by `vmm_granularity` chunks. Frees must occur in reverse
  allocation order (`:680`). When `GGML_CUDA_P2P` is set or `GGML_USE_NCCL`
  is defined, `cuMemSetAccess` is called per peer device to grant
  read/write access to the VMM-backed allocation (`:612-641`).

Each pool is per-device per-stream (`pools[device][curr_stream_no]`),
lazily constructed on first `pool()` call.

### 7.3 RAII scratch allocations (`ggml_cuda_pool_alloc<T>`)

`common.cuh:1166-1208` defines a non-copyable, non-movable RAII wrapper
around `ggml_cuda_pool`. Per-op code uses it for temporary device buffers:

```cpp
ggml_cuda_pool_alloc<half> src0_alloc(ctx.pool());
src0_alloc.alloc(ggml_nelements(src0));
// ... use src0_alloc.get() ...
// ~ggml_cuda_pool_alloc returns the buffer to the pool
```

This is the CUDA analog of the CPU `cplan.work_data` scratch, but with
two key differences: (a) it's per-op (not pre-sized for the whole graph),
(b) buffers return to the pool rather than being freed.

### 7.4 CUDA graph node-properties cache

`ggml_cuda_graph::node_properties` (`common.cuh:1241-1246`) caches, per
node, the destination tensor's full `ggml_tensor` snapshot plus every
source's data pointer + `ne[4]` + `nb[4]`. This is 1 + 4*8 = 33 int64s
plus the tensor itself (~144 B) per node. For a 500-node graph that's
~80 KiB of cache, used to detect property changes for graph reuse.

### 7.5 Pinned host memory

`ggml_cuda_host_malloc` (`ggml-cuda.cu:1273-1289`) calls `cudaMallocHost`
unless `GGML_CUDA_NO_PINNED` is set, falling back to a NULL return on
failure (the buffer type then falls through to a regular CPU buffer,
`:1291-1304`). Separately, `ggml_backend_cuda_register_host_buffer`
(`:4494-4515`) calls `cudaHostRegister(... | cudaHostRegisterPortable |
cudaHostRegisterReadOnly)` on externally allocated buffers when the
`GGML_CUDA_REGISTER_HOST` env var is set.

---

## 8. Parallelism Strategy

### 8.1 Stream model

Every backend context owns a 2-D pool of CUDA streams indexed by
`(device, stream_no)`. Stream 0 is the "main" compute stream; streams
1..N are fork streams for QKV concurrency. Streams are created lazily with
`cudaStreamNonBlocking` (`common.cuh:1478-1484`), so they do not
synchronize with the NULL stream.

`curr_stream_no` is a mutable field on the context: the graph executor
flips it before launching a node (`ggml-cuda.cu:3985, 3994`). All
subsequent `cuda_ctx->stream()` calls within that op return the forked
stream. After the join node, `curr_stream_no` is reset to 0.

### 8.2 QKV multi-stream concurrency

`ggml_backend_cuda_graph_optimize` (`ggml-cuda.cu:4184-4426`) reorders the
cgraph to interleave Q/K/V branches and inserts a
`ggml_cuda_concurrent_event` into the context's
`concurrent_stream_context`. The optimization is gated by
`GGML_CUDA_GRAPH_OPT=1` and currently limited to:

* `min_fan_out == max_fan_out == 3` (exactly 3-way fork),
* the fork node's name contains `"attn_norm"`,
* single-device execution (`ggml_backend_cuda_get_device_count() == 1`),
* CUDA graphs are enabled,
* all branch nodes are contiguous between fork and join (no foreign nodes
  in the middle).

At execution time (`:3880-3897`), the fork records `fork_event` on the
main stream, then `cudaStreamWaitEvent` on each forked stream. The join
records `join_events[i]` on each forked stream and `cudaStreamWaitEvent`s
on the main stream. See Finding ARTX08-F11.

### 8.3 Multi-GPU model

The CUDA backend supports three multi-GPU AllReduce modes, selected at
`ggml_backend_cuda_comm_init` time (`ggml-cuda.cu:1208-1245`):

| Mode       | Trigger                                              | Implementation                                       |
| ---------- | ---------------------------------------------------- | ---------------------------------------------------- |
| `nccl`     | Linux default; `GGML_CUDA_ALLREDUCE=nccl`            | `ncclCommInitAll` + `ncclAllReduce` (FP32 or BF16)  |
| `internal` | non-Linux default; `GGML_CUDA_ALLREDUCE=internal`    | `ggml_cuda_ar_pipeline` (custom 2-GPU ring, F32/F16/BF16) |
| `none`     | `GGML_CUDA_ALLREDUCE=none`                           | returns `false`, meta-backend butterfly takes over  |

NCCL falls back to internal if `ncclCommInitAll` fails or if virtual
devices are in use (`:1176-1181`). Internal falls back to butterfly if
the device count is not 2 (`:1167-1170`).

NCCL AllReduce uses a heuristic to choose between FP32 and BF16 reduction
(`:1016`): for small tensors (`ne < 32768` on 2 GPUs, scaling up at 4×
per extra GPU), reduce as FP32; for larger tensors, compress to BF16
on-device, reduce as BF16, then decompress back to FP32.

### 8.4 Peer-to-peer

Peer access is enabled eagerly at startup if `GGML_CUDA_P2P` is set
(`ggml-cuda.cu:392-406`) — for every ordered pair of physical devices
where `cudaDeviceCanAccessPeer` returns 1, `cudaDeviceEnablePeerAccess`
is called. Otherwise, peer access is implicitly enabled by NCCL.

Cross-device `cpy_tensor_async` (`:2423-2480`) uses `cudaMemcpyPeerAsync`
when the backing physical devices differ, or `cudaMemcpyAsync(D2D)` when
they're the same physical device (handles the virtual-device case,
`:2453-2456`). A `copy_event` is recorded on the src stream and waited on
by the dst stream. See Finding ARTX08-F16.

### 8.5 Per-op work distribution

Unlike the CPU backend (which splits each op across N worker threads via
`ith/nth`), the CUDA backend launches a single kernel per op. Parallelism
is *inside* the kernel: each kernel chooses its own grid/block dimensions
based on tensor shape and CC. There is no per-node cross-stream
parallelism except via the QKV fork mechanism (§8.2).

---

## 9. GPU Strategy

### 9.1 Compute capability encoding

`common.cuh:50-108` defines a single integer encoding for every supported
GPU family:

| Vendor        | Encoding                                       | Examples                       |
| ------------- | ---------------------------------------------- | ------------------------------ |
| NVIDIA        | `100*major + 10*minor`                         | Pascal=600, Volta=700, Turing=750, Ampere=800, Ada=890, Hopper=900, Blackwell=1200, DGX Spark=1210, Rubin=1300 |
| AMD (HIP)     | `0x1000000 + major*0x100 + minor*0x10`         | GCN4=0x1000803, Vega=0x1000900, CDNA1=0x1000908, CDNA3=0x1000942, RDNA1=0x1001010, RDNA4=0x1001200 |
| Moore Threads | `0x0100000 + major*0x100 + minor*0x10`         | QY1=0x0100210, QY2=0x0100220, PH1=0x0100310 |

A set of `GGML_CUDA_CC_IS_*(cc)` and `*_mma_available(cc)` helpers
(`common.cuh:64-108, 298-363`) test the encoding. The encoding is queried
once at startup in `ggml_cuda_init` (`:347-372`) and cached in
`ggml_cuda_device_info::devices[id].cc`. Kernel routers like
`ggml_cuda_mul_mat` read `cc` once and pass it to `_should_use_*` helpers.

### 9.2 Per-CC kernel selection

The compilation unit is built once per target architecture via CMake's
`CMAKE_CUDA_ARCHITECTURES`. The `__CUDA_ARCH_LIST__` macro enumerates the
compiled targets; `ggml_cuda_highest_compiled_arch(cc)`
(`common.cuh:135-172`) returns the highest compiled arch ≤ `cc`. Helpers
like `ampere_mma_available(cc)` check
`ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_AMPERE` — this
gates whether the Ampere MMA (Tensor Core) code path is available at
runtime. Crucially, the runtime `cc` may be higher than the highest
compiled arch (forward compatibility), in which case the highest compiled
arch is used.

### 9.3 Matmul kernel routing

Five matmul backends are tried in order (`ggml-cuda.cu:1833-1851`):

| Kernel           | When                                                              | Implementation                |
| ---------------- | ----------------------------------------------------------------- | ----------------------------- |
| `mul_mat_vec_f`  | F32/F16/BF16 src0, very thin src0, no tensor cores preferred      | `mmvf.cu` + template instances |
| `mul_mat_f`      | Quantized or F16 src0, small-M tile-mm shape                      | `mmf.cu` + template instances |
| `mul_mat_vec_q`  | Quantized src0, `ne11 <= MMVQ_MAX_BATCH_SIZE`                     | `mmvq.cu` + template instances |
| `mul_mat_q`      | Quantized src0, larger batches (the CUDA analog of CPU `vec_dot`) | `mmq.cu` + template instances |
| `mul_mat_cublas` | Fallback                                                          | cuBLAS GemmEx / Sgemm / Batched |

Each `*_should_use_*` helper consults `cc` and shape to pick the best
kernel. The MMQ kernel itself is split into per-CC config files
(`mmq-config-pascal.cuh`, `mmq-config-ampere.cuh`, `mmq-config-cdna.cuh`,
`mmq-config-rdna2.cuh`, `mmq-config-rdna4.cuh`, `mmq-config-blackwell.cuh`)
that define tile sizes, pipeline depth, and `cp.async` usage.

### 9.4 PDL (Programmatic Dependent Launch)

`common.cuh:114-133, 1552-1660` defines PDL support for Hopper+ kernels.
`ggml_cuda_kernel_launch` (`:1641-1660`) checks a per-(device, kernel)
cache (`ggml_cuda_kernel_can_use_pdl`, `:1577-1630`) and, if the kernel's
PTX version is ≥ 90, uses `cudaLaunchKernelEx` with
`cudaLaunchAttributeProgrammaticStreamSerialization`. Device-side
`cudaTriggerProgrammaticLaunchCompletion()` and
`cudaGridDependencySynchronize()` (gated by `__CUDA_ARCH__ >= 900`) let
the next kernel begin executing before the previous one fully drains.
`__restrict__` is disabled on PDL kernels (`:1634-1639`) per CUDA
documentation.

### 9.5 cuBLAS math mode

`common.cuh:1494` sets `CUBLAS_TF32_TENSOR_OP_MATH` on every cuBLAS
handle. This means F32 cuBLAS matmuls use TF32 precision (10-bit mantissa)
by default on Ampere+. The user can force F32 via
`dst->op_params[0] = GGML_PREC_F32` or
`GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32`. See Finding ARTX08-F08.

### 9.6 Shared memory

`CUDA_SET_SHARED_MEMORY_LIMIT(kernel, nbytes)` (`common.cuh:230-245`)
raises `cudaFuncAttributeMaxDynamicSharedMemorySize` once per (kernel,
device) pair, guarded by a static `bool` array. This is required to use
the full `sharedMemPerBlockOptin` (e.g., 100 KiB on Ampere, 228 KiB on
Hopper) for fattn and MMQ kernels.

---

## 10. Quantization Strategy

### 10.1 Type traits

`common.cuh:964-1125` defines `ggml_cuda_type_traits<ggml_type>` with
three constexpr fields:

* `qk` — elements per quant block (e.g., 32 for Q4_0, 256 for Q*K),
* `qr` — quant ratio (size ratio vs. F32),
* `qi` — interactions per block (used by MMQ tiling).

There is **no** `vec_dot` / `from_float` / `vec_dot_type` field — the CUDA
backend has no equivalent of the CPU `type_traits_cpu[]` table. Instead,
per-dtype kernels are instantiated from templates in
`ggml-cuda/template-instances/` (e.g.,
`mmq-instance-q4_0.cu`, `fattn-vec-instance-q4_0-f16.cu`). Each
instantiation is a separate `.cu` file generated by
`template-instances/generate_cu_files.py`.

### 10.2 Supported quant formats

`ggml_backend_cuda_device_supports_op` for `GGML_OP_MUL_MAT`
(`ggml-cuda.cu:4801-4831`) accepts: F32, F16, BF16, Q1_0, Q4_0, Q4_1,
Q5_0, Q5_1, Q8_0, MXFP4, NVFP4, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K,
IQ1_M, IQ1_S, IQ2_S, IQ2_XS, IQ2_XXS, IQ3_S, IQ3_XXS, IQ4_NL, IQ4_XS.
Same set for `MUL_MAT_ID`.

### 10.3 FP4 paths (MXFP4, NVFP4)

* `GGML_TYPE_MXFP4` — uses E8M0 block scales. Converters
  `ggml_cuda_e8m0_to_fp32` and `ggml_cuda_ue4m3_to_fp32`
  (`common.cuh:821-866`) handle the scale formats. MMQ instance at
  `template-instances/mmq-instance-mxfp4.cu`.
* `GGML_TYPE_NVFP4` — uses UE4M3 block scales, requires Blackwell MMA
  for the native FP4 path. `ggml_cuda_fp32_to_ue4m3`
  (`common.cuh:868-878`) is gated by `BLACKWELL_MMA_AVAILABLE`. The
  feature flag `BLACKWELL_NATIVE_FP4` is exported via
  `ggml_backend_cuda_get_features` (`:5297-5301`) when any device
  supports it. NVFP4 fusion (mul_mat + scale) is detected by
  `ggml_cuda_try_fuse` (`:3338-3353`).

### 10.4 Activation conversion

`ggml_cuda_mul_mat_cublas_impl` (`:1444-1492`) converts src0 and src1 to
the compute type via `traits::convert(src_type)` /
`traits::convert_nc(src_type)` — these return function pointers from
`ggml_get_to_fp32_cuda` / `ggml_get_to_fp16_cuda` / etc. The conversion
happens once per matmul call (no caching across calls), into a
`ggml_cuda_pool_alloc<cuda_t>` scratch.

---

## 11. Correctness Analysis

Static analysis found the following correctness-relevant properties.
None are bugs; all are intentional design decisions that have correctness
consequences.

### 11.1 TF32 by default

`common.cuh:1494` sets `CUBLAS_TF32_TENSOR_OP_MATH` on every cuBLAS
handle. F32 matmuls on Ampere+ therefore accumulate in TF32 (10-bit
mantissa, ~3 decimal digits). The user can opt out via
`dst->op_params[0] = GGML_PREC_F32` (per-tensor) or
`GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32` (global env). The default is a
deliberate speed/accuracy tradeoff and matches the cuBLAS default since
CUDA 11.

### 11.2 FP16 / BF16 round-trip in cuBLAS

`ggml_cuda_mul_mat_cublas` may convert src0 from F16 to F32 if
`!fast_fp16_hardware_available(cc)` (e.g., on Pascal without FP16 MMA),
run the matmul in F32, and leave the output in F32 (`:1623-1625`). On
Volta+ with FP16 MMA, F16 inputs are kept in F16, the matmul runs as
`CUBLAS_COMPUTE_16F`, and the F16 output is converted back to F32 via
`ggml_get_to_fp32_cuda` (`:1614-1616`). This means the same F16 matmul
on different GPUs produces different ULP results, depending on whether
the cuBLAS path or the custom MMQ path is taken.

### 11.3 BF16 AllReduce precision

`ggml_backend_cuda_comm_allreduce_nccl` (`:1033-1067`) compresses to
BF16 for the reduction when the tensor is "large" (per the heuristic at
`:1016`). The reduction itself is therefore done in BF16 (8-bit mantissa),
then converted back to F32. For tensors near the threshold, the choice
between FP32 and BF16 reduction is non-deterministic w.r.t. tensor size,
which means two tensors of similar size may produce different precision
outputs.

### 11.4 Multi-row vecdot non-determinism (MMQ)

MMQ kernels accumulate per-warp and reduce via `warp_reduce_sum` /
`block_reduce`. The block reduction order depends on the block size
template and the warp ID, which depend on shape. Therefore the F32
output of `mul_mat_q` can differ at the ULP level from `mul_mat_cublas`
for the same inputs.

### 11.5 Pinned host memory fallback

`ggml_backend_cuda_host_buffer_type_alloc_buffer` (`:1291-1304`) falls
back to the regular CPU buffer type if `cudaMallocHost` fails. The
resulting buffer is not pinned, so async H2D copies on it will
implicitly stage through pinned internal buffers (slower). This is a
silent performance regression, not a correctness issue.

### 11.6 VMM pool free ordering

`ggml_cuda_pool_vmm::free` asserts `ptr == pool_addr + pool_used - size`
(`ggml-cuda.cu:680`) — frees must occur in strict reverse allocation
order. The `ggml_cuda_pool_alloc<T>` RAII wrapper enforces this for
single buffers, but if a kernel internally allocates from the pool out
of order (e.g., nested `pool_alloc`s), the assert fires. The codebase
carefully avoids this, but it is a fragile invariant.

### 11.7 QKV concurrency memory-range check

`ggml_cuda_check_fusion_memory_ranges` (`:2859-2920`) and
`ggml_cuda_concurrent_event::is_valid` (`common.cuh:1297-1385`) both
cast `tensor->data` to `int64_t` for pointer comparison
(`:2866-2870, 1304-1310`). This is technically implementation-defined
on platforms where pointers exceed 64 bits, but works on every CUDA
target supported by llama.cpp.

### 11.8 Cross-backend event wait

`ggml_backend_cuda_event_wait` (`:4165-4182`) handles two cases:

* If `ggml_backend_is_cuda(backend)`: `cudaStreamWaitEvent` on the
  backend's stream. Correct.
* Otherwise: the code path is `#if 0`'d out with a `GGML_ABORT("fatal
  error")`. **Cross-backend event wait is not implemented.** A CPU
  backend waiting on a CUDA event will abort. See Unknowns U2.

### 11.9 Graph capture concurrency lock

`ggml_cuda_lock` + `ggml_cuda_lock_counter` (`:698-700`) prevent cuBLAS
handle destruction during graph capture. The destructor
`ggml_backend_cuda_context::~ggml_backend_cuda_context` waits on
`ggml_cuda_lock_cv` until `ggml_cuda_lock_counter == 0` before destroying
handles (`:702-719`). This is a process-wide lock; capturing a graph on
one backend blocks the destruction of any other backend's context.

---

## 12. Optimization Analysis

### 12.1 Identified optimizations

| Optimization                          | Where                                  | Notes                                                                  |
| ------------------------------------- | -------------------------------------- | ---------------------------------------------------------------------- |
| Async stream pool (lazy)              | `common.cuh:1478-1484`                 | Streams created on first use; `cudaStreamNonBlocking` for overlap.    |
| Per-device cuBLAS handle (lazy)       | `common.cuh:1490-1497`                 | One handle per device; `CUBLAS_TF32_TENSOR_OP_MATH` by default.       |
| Best-fit pool with look-ahead         | `ggml-cuda.cu:453-516`                 | 5 % look-ahead, 256-slot free list, auto-flush on OOM.                |
| VMM pool with peer access             | `ggml-cuda.cu:536-682`                 | 32 GiB virtual reserve, `cuMemSetAccess` per peer device.             |
| `cudaStreamPerThread` for sync I/O    | `ggml-cuda.cu:778-815`                 | Buffer-level set/get/memset use the per-thread default stream + sync. |
| `cudaEventDisableTiming` for sync     | `:2468, 5199, 1278`                    | Events used purely for ordering — no timing overhead.                 |
| CUDA graph capture + warmup           | `:4100-4157`                           | 2-call warmup, `cudaGraphExecUpdate` for incremental changes.         |
| Concurrent-event QKV fork/join        | `:3880-3897, 4184-4426`                | 3-way fork on `attn_norm` nodes; stream assignment via map.           |
| ~12 fusion patterns                   | `:3144-3867`                           | FFN GLU, RMS_NORM+MUL[+ADD], SSM_CONV+ADD+SILU, ROPE+VIEW+SET_ROWS, topk-moe, snake, gated_delta_net+cpy. |
| Per-dtype MMQ/MMVF/MMF instances      | `template-instances/`                  | ~150 .cu files generated by `generate_cu_files.py`.                   |
| Hadamard fast path via op hint        | `:1816-1818`                           | `GGML_HINT_SRC0_IS_HADAMARD` dispatches to FWHT kernel.               |
| `ggml_cuda_set_device` cache          | `:118-130`                             | Skips `cudaSetDevice` if current device matches.                      |
| Quantized weight padding              | `common.cuh:176`, `:906-922`           | Pads last row to `MATRIX_ROW_PADDING=512` for OOB-safe vector loads.  |
| PDL launch on Hopper+                 | `common.cuh:1552-1660`                 | `cudaLaunchKernelEx` with programmatic stream serialization.          |
| Per-(kernel, device) PDL cache        | `:1577-1630`                           | Avoids repeated `cudaFuncGetAttributes` calls.                        |
| Per-(kernel, device) shared-mem raise | `common.cuh:230-245`                   | `cudaFuncSetAttribute` called once per (kernel, device).              |
| `memcmp` of node-props for graph reuse| `:2535-2575`                           | Skips `cudaGraphExecUpdate` if per-node properties unchanged.         |
| `fastdiv` for index arithmetic        | `common.cuh:902-945`                   | Replaces division by mulhi + shift.                                   |
| BF16 AllReduce for large tensors      | `:1033-1067`                           | Bandwidth-bound reduction compressed to BF16 on-device.               |
| `cudaDeviceScheduleSpin` for cc121    | `:367-370`                             | Workaround for DGX Spark iGPU sync latency.                           |

### 12.2 Optimizations *not* present (worth noting)

* **No plan-time fusion**. `ggml_cuda_try_fuse` is called per-node per
  graph execution (`:4009`), same anti-pattern as ARTX01-F08. Fusion
  decisions are not cached.
* **No split-buffer type**. Multi-GPU tensor parallelism goes through
  AllReduce only; the old `ggml_backend_cuda_split_buffer_type` was
  removed. The dead `default_tensor_split` array is the only remnant.
* **No cross-backend event wait**. `ggml_backend_cuda_event_wait` aborts
  if the backend is not CUDA (`:4180`).
* **No kernel autotuning**. Tile sizes and kernel selection are
  heuristic (based on `cc` and shape), not benchmarked at runtime.
* **No graph-level node reordering beyond QKV**. The only graph
  optimization is the QKV fork/join; no general topological reordering
  or memory pressure aware scheduling.
* **No persistent kernels**. Each op is a separate kernel launch; PDL
  is the only kernel-overlap mechanism.

---

## 13. Architectural Strengths

1. **Full async / event vtable**. Unlike the CPU backend, every async
   hook is implemented (`ggml-cuda.cu:4428-4445`). This makes the CUDA
   backend a true peer in a multi-backend scheduler.

2. **CUDA graph capture with warmup + `cudaGraphExecUpdate`**. The
   warmup rule (`:4120-4139`) avoids capturing unstable graphs; the
   incremental update path (`:2577-2603`) re-instantiates only when
   `cudaGraphExecUpdate` fails. This is the right design for repeated
   inference with occasional shape changes.

3. **Two-pool strategy (legacy + VMM)**. The legacy pool is simple and
   correct; the VMM pool enables peer access via `cuMemSetAccess` and
   avoids `cudaMalloc` overhead for frequent scratch allocations. The
   choice is per-device at construction time.

4. **Single CC encoding across NVIDIA / AMD / Moore Threads**. The
   integer encoding (`common.cuh:50-108`) lets `*_mma_available(cc)`
   helpers be vendor-agnostic. The same `ggml_cuda_mul_mat` router
   works for CUDA, HIP, and MUSA.

5. **PDL with per-(kernel, device) cache**. PDL is gated by PTX version
   ≥ 90 (which is checked via `cudaFuncGetAttributes`), and the result
   is cached per (kernel, device). This is the correct way to use PDL
   without paying the attribute-query cost on every launch.

6. **`ggml_cuda_pool_alloc<T>` RAII**. Clean, non-copyable, non-movable
   wrapper that returns buffers to the pool on destruction. Eliminates
   a class of memory leaks.

7. **Per-graph-key graph cache with eviction**. The
   `cuda_graphs` map (`common.cuh:1420`) keys graphs by first-node
   pointer, sweeps every 5 s, and evicts graphs unused for 10 s
   (`:1428-1437`). This handles the case where a process runs multiple
   distinct cgraphs (e.g., CPU+GPU split inference) without unbounded
   growth.

8. **Per-fork-stream mapping in QKV concurrency**. The
   `stream_mapping` map (`common.cuh:1261`) allows arbitrary node →
   stream assignment, generalizing beyond strict branch interleaving.

9. **Compute capability forward-compatibility**. `ggml_cuda_highest_compiled_arch`
   returns the highest compiled arch ≤ runtime `cc`, so a binary compiled
   for Ampere runs (with Ampere code paths) on Hopper.

10. **NCCL fallback chain**. NCCL → internal AR → butterfly, with
    per-step warnings (`:1154-1202`). The meta-backend's butterfly
    ensures correctness even when neither CUDA-side path can run.

---

## 14. Architectural Weaknesses

### W1 — Over-allocated per-context arrays

**Evidence**: `common.cuh:1412-1413, 1504` declare
`streams[GGML_CUDA_MAX_DEVICES][GGML_CUDA_MAX_STREAMS]` (= 16 × 8 = 128
pointers), `cublas_handles[GGML_CUDA_MAX_DEVICES]` (= 16 handles), and
`pools[GGML_CUDA_MAX_DEVICES][GGML_CUDA_MAX_STREAMS]` (= 128 unique_ptrs).
Each backend owns exactly one device.

**Impact**: ~2.3 KiB of mostly-unused state per backend context. For a
16-GPU system with 16 backends, that's ~37 KiB of dead array slots.
Negligible in absolute terms, but the design leaks the global
`GGML_CUDA_MAX_DEVICES` constant into every backend.

### W2 — Dead `ggml_tensor_extra_gpu` and `default_tensor_split`

**Evidence**: `common.cuh:1213-1216` (`ggml_tensor_extra_gpu`) is never
instantiated. `common.cuh:1151` (`default_tensor_split`) is written at
`ggml-cuda.cu:302, 386` but never read.

**Impact**: Confusing for future maintainers; suggests a multi-GPU
split-buffer mechanism that no longer exists. The model layer
(`llama-model.cpp:949-967`) still queries for
`ggml_backend_split_buffer_type` via `get_proc_address`, which returns
NULL for CUDA — so the model falls back to layer-split mode silently.

### W3 — Op dispatch via giant switch, not function-pointer table

**Evidence**: `ggml-cuda.cu:2011-2364` is a ~350-line switch with ~150
cases. There is no per-op function-pointer table analogous to
`type_traits_cpu[]` (ARTX01-F03).

**Impact**: Adding a new op means editing the switch. The compiler
generates a jump table from the switch, so performance is similar to a
function-pointer table, but extensibility is worse. There is no
plugin mechanism analogous to the CPU `extra_buffer_type` for
out-of-tree kernels.

### W4 — Fusion detected at execution time

**Evidence**: `ggml_cuda_try_fuse` (`:3144-3867`) is called per-node
per-graph-execution (`:4009`). The TODO at ARTX01-F08 also applies.

**Impact**: O(N) fusion checks per graph run, where N is node count.
The fusion logic itself is non-trivial (~700 lines of pattern matching),
so this is a measurable CPU-side cost on large graphs.

### W5 — TF32 default for cuBLAS

**Evidence**: `common.cuh:1494` sets `CUBLAS_TF32_TENSOR_OP_MATH`.
`ggml_cuda_mul_mat_cublas` (`:1619-1660`) lets the user opt out via
`GGML_PREC_F32` or `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32`, but the default
is TF32.

**Impact**: F32 cuBLAS matmuls lose ~13 bits of mantissa precision
silently. Acceptable for inference, surprising for validation.

### W6 — `op_offload_min_batch_size` is global, not per-op

**Evidence**: `ggml-cuda.cu:5357` reads `GGML_OP_OFFLOAD_MIN_BATCH` once
at registry init and assigns it to every device. `:5187` applies it as
`get_op_batch_size(op) >= dev_ctx->op_offload_min_batch_size`.

**Impact**: An op with batch 31 stays on CPU even if it would benefit
from GPU; an op with batch 32 goes to GPU even if CPU would be faster.
No per-op or per-shape policy.

### W7 — Cross-backend event wait is unimplemented

**Evidence**: `ggml-cuda.cu:4171-4181` `#if 0`'s out the
`cudaLaunchHostFunc` path and `GGML_ABORT`s.

**Impact**: A CPU backend cannot wait on a CUDA event. Mixed CPU+GPU
scheduling must use `synchronize()` (blocking) rather than event-based
dependencies. This limits GATE's ability to overlap CPU and GPU work.

### W8 — QKV concurrency is `attn_norm`-name-gated

**Evidence**: `ggml-cuda.cu:4274` checks
`strstr(root_node->name, "attn_norm")`. The TODO at `:4273` says
"make this more generic".

**Impact**: Any fork node not named `attn_norm*` (e.g., a custom
attention variant) is not parallelized. The optimization is gated by
naming convention rather than graph structure.

### W9 — Single `copy_event` per backend context

**Evidence**: `common.cuh:1410` declares a single `copy_event`. It's
created lazily in `cpy_tensor_async` (`:2466-2469`) and reused for every
subsequent cross-backend copy.

**Impact**: Multiple concurrent cross-backend copies share one event.
Each `cudaEventRecord` overwrites the previous recording, but the
`cudaStreamWaitEvent` on the dst stream still serializes correctly
because the event records form a queue. The semantic is slightly off
(if you wanted to wait on a specific copy, you can't), but correctness
holds. See Finding ARTX08-F16.

### W10 — `MUL_MAT_ID` fallback synchronizes the stream

**Evidence**: `ggml-cuda.cu:1895-1945` copies `ids` to host, sorts on
CPU, and `cudaStreamSynchronize(stream)`. This is disabled under CUDA
graph capture (`:2509-2521`), but outside graph capture it blocks the
stream.

**Impact**: MoE inference with expert counts above
`MMVQ_MAX_BATCH_SIZE` pays a stream-sync cost per `MUL_MAT_ID` node.

### W11 — Graph capture uses `cudaStreamCaptureModeRelaxed`

**Evidence**: `ggml-cuda.cu:4151`. Relaxed mode allows captures to
cross external stream boundaries but does not detect them.

**Impact**: A capture in progress can be violated by an external
operation on the same stream (e.g., a debug print kernel). The
`ggml_cuda_lock` mitigates this for cuBLAS handle destruction but not
for arbitrary external ops.

### W12 — `default_tensor_split` computed even when never used

**Evidence**: `ggml-cuda.cu:302, 386`. The split is computed from VRAM
proportions but the field is never read.

**Impact**: Wasted init work (~16 float divisions); dead code in the
hot init path.

---

## 15. GwenLand Mapping

| GwenLand module | ADOPT / ADAPT / REJECT / MONITOR / DEFER | What | Reasoning |
| --------------- | ---------------------------------------- | ---- | --------- |
| `glcuda`        | **ADOPT** | Full async/event vtable | CPU's NULL async hooks (ARTX01-F01) are wrong for GPU; CUDA's full impl is the right baseline. |
| `glcuda`        | **ADAPT** | Per-backend stream pool | Keep lazy `cudaStreamNonBlocking`, but size arrays for ONE device per backend (REJECT the `[16][8]` shape). |
| `glcuda`        | **ADOPT** | Per-device cuBLAS handle, lazy, TF32 default | Matches cuBLAS best practice; expose env override. |
| `glcuda`        | **ADOPT** | VMM pool with `cuMemSetAccess` peer grants | Better than `cudaMalloc` + implicit peer for multi-GPU. |
| `glcuda`        | **ADOPT** | Legacy best-fit pool as fallback | VMM may not be available on all platforms. |
| `glcuda`        | **ADOPT** | `ggml_cuda_pool_alloc<T>` RAII | Clean, leak-free scratch management. |
| `glcuda`        | **ADOPT** | CUDA graph warmup + `cudaGraphExecUpdate` | Correct handling of stable-vs-unstable graphs. |
| `glcuda`        | **ADAPT** | QKV multi-stream fork/join | Generalize beyond `attn_norm`-name gate; make fork-count configurable. |
| `glcuda`        | **REJECT**| Execution-time fusion detection | Move to plan-time (same recommendation as ARTX01-F08). |
| `glcuda`        | **ADAPT** | ~12 fusion patterns | Keep the patterns; lift detection into GATE's planner. |
| `glcuda`        | **REJECT**| Op dispatch via giant switch | Use a per-op function-pointer table (analogous to ARTX01-F03) for extensibility. |
| `glcuda`        | **ADOPT** | CC encoding scheme | Single-integer encoding across NVIDIA/AMD/Moore Threads is clean. |
| `glcuda`        | **ADOPT** | PDL on Hopper+ with per-(kernel,device) cache | Correct, modern GPU scheduling. |
| `glcuda`        | **ADAPT** | TF32 default for cuBLAS | Keep as default for inference; expose per-tensor override. |
| `glcuda`        | **ADOPT** | `MATRIX_ROW_PADDING=512` | Required for OOB-safe vectorized loads in quantized kernels. |
| `glcuda`        | **REJECT**| `op_offload_min_batch_size` global | Replace with per-op policy or scheduler-driven decision. |
| `glcuda`        | **ADOPT** | `copy_event` with `cudaEventDisableTiming` | Right primitive for cross-stream ordering. |
| `glcuda`        | **MONITOR**| Single `copy_event` per backend | Acceptable for now; revisit if multi-stream peer-copy becomes common. |
| `glcuda`        | **ADOPT** | NCCL → internal → butterfly fallback chain | Defense-in-depth for multi-GPU AllReduce. |
| `glcuda`        | **REJECT**| Dead `ggml_tensor_extra_gpu` and `default_tensor_split` | Don't carry the legacy split-buffer structures into GwenLand. |
| `GATE`          | **ADOPT** | Concurrent-event fork/join mechanism | Generalize beyond QKV; let the scheduler insert fork/join for any independent subgraph. |
| `GATE`          | **ADAPT** | Per-graph-key graph cache with eviction | Keep the design; consider LRU instead of time-based sweep. |
| `GATE`          | **ADOPT** | `cudaStreamCaptureModeRelaxed` for capture | Necessary for cross-stream capture; pair with explicit lock for handle destruction. |
| `GATE`          | **MONITOR**| `MUL_MAT_ID` host-sort fallback | Watch for upstream graph-safe MMID for large expert counts. |

---

## 16. Recommendations

### R1 — ADOPT full async/event vtable as glcuda baseline
**Priority:** Critical
**Difficulty:** M
**Dependencies:** none
GwenLand's `glcuda` should implement every async hook (`set_tensor_async`,
`get_tensor_async`, `cpy_tensor_async`, `synchronize`, `event_record`,
`event_wait`). Use `cudaStreamNonBlocking` streams, `cudaEventDisableTiming`
events, and `cudaStreamWaitEvent` for cross-stream ordering. Match the
CUDA backend's contract exactly.

### R2 — ADAPT per-backend stream pool, sized for ONE device
**Priority:** High
**Difficulty:** S
**Dependencies:** R1
Keep `streams[N_STREAMS]` (one-dimensional, per-backend), not
`streams[N_DEVICES][N_STREAMS]`. Each backend owns one device; cross-
device work goes through `cpy_tensor_async` (which uses peer copies).
Reduces per-context overhead from ~2 KiB to ~64 B.

### R3 — ADOPT VMM pool with `cuMemSetAccess` peer grants
**Priority:** High
**Difficulty:** L
**Dependencies:** R1
Use `cuMemCreate` / `cuMemMap` / `cuMemAddressReserve` for scratch
allocations. Call `cuMemSetAccess` per peer device when peer access is
enabled. Fall back to a best-fit free-list pool when VMM is unavailable.

### R4 — ADOPT CUDA graph warmup + `cudaGraphExecUpdate`
**Priority:** High
**Difficulty:** M
**Dependencies:** R1
Implement the 2-call warmup rule, the per-node-properties cache, and the
incremental `cudaGraphExecUpdate` path. Re-instantiate only when update
fails. This is the right design for repeated inference.

### R5 — REJECT execution-time fusion; move to plan-time
**Priority:** High
**Difficulty:** L
**Dependencies:** GATE design
Lift `ggml_cuda_try_fuse`'s ~12 patterns into GATE's graph planner.
Detect patterns once at plan time, mark fused nodes in the plan, execute
without re-checking. Same recommendation as ARTX01-R5.

### R6 — ADOPT QKV fork/join, generalize beyond `attn_norm`
**Priority:** Medium
**Difficulty:** M
**Dependencies:** R1, R4
Keep the concurrent-event mechanism, but drop the
`strstr(name, "attn_norm")` gate. Use a structural test: any node with
fan-out ≥ 2 where all branches rejoin at a single consumer is a
candidate. Make the fork count configurable (currently hard-coded to 3).

### R7 — ADOPT CC encoding scheme
**Priority:** High
**Difficulty:** S
**Dependencies:** none
Replicate the single-integer CC encoding
(`100*major + 10*minor` for NVIDIA, `0x1000000 + …` for AMD, `0x0100000 + …`
for Moore Threads) and the `*_mma_available(cc)` helper pattern. Single
source of truth for kernel selection.

### R8 — ADOPT PDL on Hopper+ with per-(kernel, device) cache
**Priority:** Medium
**Difficulty:** M
**Dependencies:** R7
Use `cudaLaunchKernelEx` with
`cudaLaunchAttributeProgrammaticStreamSerialization` for kernels whose
PTX version is ≥ 90. Cache the result of `cudaFuncGetAttributes` per
(kernel, device). Drop `__restrict__` on PDL kernels per CUDA docs.

### R9 — ADOPT `MATRIX_ROW_PADDING=512`
**Priority:** Medium
**Difficulty:** XS
**Dependencies:** none
Pad quantized weight tensors to a multiple of 512 bytes per row. Zero
the padding at `init_tensor` time. Required for OOB-safe vectorized
loads in MMQ/MMVQ.

### R10 — ADOPT NCCL → internal → butterfly fallback chain
**Priority:** Medium
**Difficulty:** L
**Dependencies:** R1
For multi-GPU AllReduce: try NCCL first (Linux default), fall back to a
custom 2-GPU ring, then to the meta-backend's butterfly. Each step
warns and falls through on init failure.

### R11 — REJECT `op_offload_min_batch_size` global
**Priority:** Low
**Difficulty:** M
**Dependencies:** GATE design
Replace the global threshold with a per-op policy: op type, shape, and
estimated cost should drive offload decisions, not a single batch-size
cutoff.

### R12 — REJECT dead split-buffer structures
**Priority:** Low
**Difficulty:** XS
**Dependencies:** none
Do not carry `ggml_tensor_extra_gpu` or `default_tensor_split` into
GwenLand. Multi-GPU tensor parallelism in glcuda goes through AllReduce,
not split weight tensors.

---

## 17. Findings

### Finding ARTX08-F01

```
Finding ID:           ARTX08-F01
Category:             BACKEND_DESIGN
Engine:               CUDA
Component:            Backend interface
Source File:          ggml/src/ggml-cuda/ggml-cuda.cu
Function:             ggml_backend_cuda_interface (vtable)
Lines:                4428-4445
Summary:              CUDA backend implements the full async/event/graph_optimize
                      vtable, in contrast to the CPU backend which sets every
                      async hook to NULL.
Observation:          The vtable assigns non-NULL function pointers to
                      set_tensor_async, get_tensor_async, set_tensor_2d_async,
                      get_tensor_2d_async, cpy_tensor_async, synchronize,
                      graph_compute, event_record, event_wait, and
                      graph_optimize. The graph_plan_* hooks are NULL (the
                      CUDA backend uses graph_compute directly, not the plan
                      API). This makes the CUDA backend a fully asynchronous
                      peer in a multi-backend scheduler.
Evidence:             ggml-cuda.cu:4428-4445 — vtable definition.
                      ggml-cuda.cu:2383-2488 — async API implementations.
                      ggml-cuda.cu:4159-4182 — event_record/event_wait.
Architectural Impact: glcuda can be treated as a peer to glproc and glvulkan
                      in GATE's scheduler, with stream-based overlap and
                      event-based dependencies. The CPU backend's NULL async
                      hooks (ARTX01-F01) are the exception, not the rule.
Correctness Impact:   None. Async execution is correct by construction; each
                      async op enqueues on the backend's stream and the caller
                      must synchronize or wait on an event before reading.
Optimization Type:    asynchronous execution.
GwenLand Target:      glcuda
Recommendation:       ADOPT. glcuda should implement the same vtable.
Priority:             Critical
Difficulty:           M
Dependencies:         none
Confidence:           High
```

### Finding ARTX08-F02

```
Finding ID:           ARTX08-F02
Category:             LAYOUT_SUBOPTIMAL
Engine:               CUDA
Component:            Backend context
Source File:          ggml/src/ggml-cuda/common.cuh
Function:             ggml_backend_cuda_context (struct)
Lines:                1407-1518
Summary:              Each backend context stores streams/cublas_handles/pools
                      arrays sized for GGML_CUDA_MAX_DEVICES (16), even though
                      each backend owns exactly one device.
Observation:          The struct declares
                      cudaStream_t streams[16][8] = 128 pointers (1 KiB),
                      cublasHandle_t cublas_handles[16] = 16 handles (128 B),
                      std::unique_ptr<ggml_cuda_pool> pools[16][8] = 128
                      unique_ptrs (1 KiB). Only the slots for ctx->device are
                      ever touched. The remaining 15 device slots are unused
                      but contribute ~2.3 KiB of zero-initialized state per
                      backend context. The design leaks the global
                      GGML_CUDA_MAX_DEVICES constant into every backend.
Evidence:             common.cuh:1412 (streams), 1413 (cublas_handles), 1504
                      (pools); common.cuh:1478-1484 stream() accessor only
                      ever touches streams[device][stream]; ggml-cuda.cu:5403
                      ggml_backend_cuda_init creates a context per device.
Architectural Impact: ~2.3 KiB × n_backends wasted memory; negligible in
                      absolute terms but architecturally leaky. For a 16-GPU
                      system, ~37 KiB of dead slots. More importantly, the
                      shape prevents the struct from being trivially copied
                      or moved (unique_ptr members), so the per-backend
                      overhead is paid in construction time too.
Correctness Impact:   None.
Optimization Type:    None (suboptimal layout).
GwenLand Target:      glcuda
Recommendation:       ADAPT. Keep the lazy stream/cublas/pool creation, but
                      size the arrays for one device: streams[N_STREAMS],
                      cublas_handles[1] (or just a single member), pools[N_STREAMS].
Priority:             Medium
Difficulty:           S
Dependencies:         R1, R2
Confidence:           High
```

### Finding ARTX08-F03

```
Finding ID:           ARTX08-F03
Category:             MISSING_FEATURE
Engine:               CUDA
Component:            Tensor extra (dead)
Source File:          ggml/src/ggml-cuda/common.cuh
Function:             ggml_tensor_extra_gpu (struct)
Lines:                1213-1216
Summary:              The ggml_tensor_extra_gpu struct is declared but never
                      instantiated or accessed anywhere in the CUDA backend.
Observation:          The struct contains data_device[16] and
                      events[16][8] fields clearly intended for split-tensor
                      multi-GPU support. A grep across ggml/src/ggml-cuda/
                      finds only the declaration; no file instantiates it,
                      no tensor's extra field is set to it. The struct is
                      dead code at the audited commit. The split-buffer
                      mechanism that would have used it was removed.
Evidence:             common.cuh:1213-1216 (declaration). Grep for
                      ggml_tensor_extra_gpu in ggml/src/ggml-cuda/ returns
                      only this one site.
Architectural Impact: Future maintainers may believe split-tensor support
                      exists when it does not. The struct's presence
                      suggests an API that was removed without cleanup.
Correctness Impact:   None (dead code).
Optimization Type:    None.
GwenLand Target:      glcuda
Recommendation:       REJECT. Do not carry this struct into glcuda. If
                      GwenLand needs per-tensor device pointers for
                      split-tensor parallelism, design it fresh.
Priority:             Low
Difficulty:           XS
Dependencies:         none
Confidence:           High
```

### Finding ARTX08-F04

```
Finding ID:           ARTX08-F04
Category:             BACKEND_DESIGN
Engine:               CUDA
Component:            Multi-GPU split buffer
Source File:          ggml/src/ggml-cuda/ggml-cuda.cu
Function:             ggml_cuda_init, ggml_backend_cuda_reg_get_proc_address
Lines:                302, 386, 5317-5338
Summary:              The CUDA backend exposes no split_buffer_type function;
                      default_tensor_split[] is computed in init but never
                      read; multi-GPU tensor parallelism goes through AllReduce.
Observation:          ggml_cuda_init writes default_tensor_split[id] at lines
                      302 and 386 based on per-device VRAM proportions. A
                      grep for default_tensor_split across ggml-cuda.cu finds
                      only these two write sites — no read. The SYCL backend
                      (ggml-sycl.cpp:1378 ggml_backend_sycl_split_buffer_type)
                      still implements split buffers and reads its
                      default_tensor_split. The CUDA backend's
                      ggml_backend_cuda_reg_get_proc_address (ggml-cuda.cu:5317)
                      does not expose "ggml_backend_split_buffer_type", so
                      the model layer (llama-model.cpp:955-967) gets NULL
                      and falls back to layer-split mode. Multi-GPU tensor
                      parallelism is implemented via the comm-context
                      AllReduce path (ggml-cuda.cu:965-1255).
Evidence:             ggml-cuda.cu:302, 386 (writes), ggml-cuda.cu:5317-5338
                      (no split_buffer_type in proc_address), ggml-cuda.cu:965-1255
                      (comm_context AllReduce), llama-model.cpp:955-967 (model
                      layer falls back when split_buffer_type is NULL).
Architectural Impact: The CUDA backend's multi-GPU strategy is now
                      AllReduce-centric, not split-buffer-centric. This is a
                      significant architectural shift from older versions of
                      llama.cpp. The dead default_tensor_split field is a
                      remnant that should be removed.
Correctness Impact:   None. The AllReduce path is correct; the model layer
                      correctly falls back to layer-split mode.
Optimization Type:    None.
GwenLand Target:      glcuda
Recommendation:       ADOPT the AllReduce-centric design. REJECT the dead
                      default_tensor_split field. glcuda should not expose
                      a split_buffer_type; multi-GPU goes through AllReduce.
Priority:             High
Difficulty:           S
Dependencies:         R10
Confidence:           High
```

### Finding ARTX08-F05

```
Finding ID:           ARTX08-F05
Category:             MEMORY_PATTERN
Engine:               CUDA
Component:            Scratch pool
Source File:          ggml/src/ggml-cuda/ggml-cuda.cu
Function:             ggml_cuda_pool_leg, ggml_cuda_pool_vmm, ggml_backend_cuda_context::new_pool_for_device
Lines:                419-532, 536-682, 685-693
Summary:              Two scratch-buffer pool implementations: a legacy
                      best-fit 256-slot free-list and a VMM-backed pool with
                      cuMemSetAccess peer grants.
Observation:          ggml_cuda_pool_leg (419-532) maintains a 256-entry
                      buffer pool with best-fit selection, 5% size look-ahead,
                      and auto-flush-on-OOM with retry. ggml_cuda_pool_vmm
                      (536-682) reserves a 32 GiB virtual address range per
                      pool (CUDA_POOL_VMM_MAX_SIZE = 1ull << 35), grows it by
                      vmm_granularity chunks via cuMemCreate/cuMemMap, and
                      uses cuMemSetAccess to grant peer R/W access when P2P
                      is enabled. new_pool_for_device (685-693) selects VMM
                      if the device supports it (info.devices[device].vmm),
                      otherwise legacy.
Evidence:             ggml-cuda.cu:419-532 (legacy), 536-682 (VMM), 685-693
                      (selection), 606-649 (VMM peer access grants).
Architectural Impact: Two-layer strategy: simple correct pool + optimized
                      VMM pool. The VMM pool's cuMemSetAccess per-peer-device
                      loop is required because VMM allocations do not inherit
                      peer access from cudaDeviceEnablePeerAccess.
Correctness Impact:   None. Both pools correctly track allocated/freed sizes.
                      VMM pool asserts frees occur in reverse allocation order
                      (ggml-cuda.cu:680), which is enforced by
                      ggml_cuda_pool_alloc<T> RAII.
Optimization Type:    blocking (pool reuse) + asynchronous execution (peer
                      access for cross-device reads).
GwenLand Target:      glcuda
Recommendation:       ADOPT. glcuda should provide both pool implementations
                      and select at construction time based on VMM support.
Priority:             High
Difficulty:           L
Dependencies:         R3
Confidence:           High
```

### Finding ARTX08-F06

```
Finding ID:           ARTX08-F06
Category:             EXECUTION_GRAPH
Engine:               CUDA
Component:            Graph compute
Source File:          ggml/src/ggml-cuda/ggml-cuda.cu
Function:             ggml_backend_cuda_graph_compute, ggml_cuda_graph_update_required, ggml_cuda_graph_update_executable
Lines:                4100-4157, 2535-2575, 2577-2603
Summary:              CUDA graph capture is gated by a 2-call warmup rule and
                      reuses the executable via cudaGraphExecUpdate on
                      subsequent calls; any property change resets warmup.
Observation:          On first call with a new graph key, warmup_complete is
                      false and use_cuda_graph stays false — the graph is
                      executed directly (no capture). On the second call with
                      no property changes (per ggml_cuda_graph_update_required,
                      which memcmps per-node tensor + src data pointers +
                      ne/nb), warmup_complete is set true and capture begins.
                      Subsequent calls reuse the captured graph: if properties
                      changed, warmup resets; else, cudaGraphExecUpdate is
                      called, with fallback to full re-instantiation on update
                      failure (2577-2603).
Evidence:             ggml-cuda.cu:4120-4139 (warmup rule), 2535-2575
                      (update_required — property cache memcmp), 2577-2603
                      (update_executable — cudaGraphExecUpdate with fallback),
                      4144-4152 (capture begin), 4066-4075 (instantiate/
                      update/launch).
Architectural Impact: Avoids capturing unstable graphs (which would waste
                      memory and trigger re-instantiation) and minimizes
                      capture overhead for stable graphs. The fallback to
                      full re-instantiation on cudaGraphExecUpdate failure
                      handles the case where the graph topology changes
                      (e.g., a kernel is removed).
Correctness Impact:   None. The captured graph is semantically identical to
                      direct execution; cudaStreamCaptureModeRelaxed allows
                      external stream operations during capture.
Optimization Type:    asynchronous execution (graph replay).
GwenLand Target:      glcuda, GATE
Recommendation:       ADOPT. glcuda should implement the same warmup rule
                      and incremental update path. Consider exposing the
                      warmup threshold as a parameter (currently hard-coded
                      to 2).
Priority:             High
Difficulty:           M
Dependencies:         R4
Confidence:           High
```

### Finding ARTX08-F07

```
Finding ID:           ARTX08-F07
Category:             BACKEND_DESIGN
Engine:               CUDA
Component:            Op dispatch
Source File:          ggml/src/ggml-cuda/ggml-cuda.cu
Function:             ggml_cuda_compute_forward
Lines:                2011-2364
Summary:              Op dispatch is a ~350-line switch statement with ~150
                      cases, not a per-op function-pointer table.
Observation:          The function is a single switch (dst->op) that calls
                      a per-op ggml_cuda_<op> function. After the switch,
                      cudaGetLastError() is polled and any error aborts.
                      There is no function-pointer table analogous to the
                      CPU type_traits_cpu[] (ARTX01-F03), and no plugin
                      mechanism analogous to the CPU extra_buffer_type
                      (ARTX01-F04). Adding an op requires editing this
                      switch AND the supports_op switch (4719-5161).
Evidence:             ggml-cuda.cu:2011-2364 (dispatch switch),
                      4719-5161 (supports_op switch).
Architectural Impact: Adding a new op is a multi-site edit; no out-of-tree
                      kernel registration. The compiler generates a jump
                      table from the switch, so dispatch performance is
                      comparable to a function-pointer table.
Correctness Impact:   None. Dispatch is deterministic.
Optimization Type:    None (architectural choice).
GwenLand Target:      glcuda
Recommendation:       ADAPT. glcuda should use a per-op function-pointer
                      table (similar to glproc's type-traits table) for
                      extensibility. The switch can be auto-generated from
                      the table at build time.
Priority:             Medium
Difficulty:           M
Dependencies:         none
Confidence:           High
```

### Finding ARTX08-F08

```
Finding ID:           ARTX08-F08
Category:             CORRECTNESS_SHORTCUT
Engine:               CUDA
Component:            cuBLAS math mode
Source File:          ggml/src/ggml-cuda/common.cuh
Function:             ggml_backend_cuda_context::cublas_handle
Lines:                1490-1497
Summary:              cuBLAS handles are created with CUBLAS_TF32_TENSOR_OP_MATH,
                      so F32 matmuls on Ampere+ use TF32 (10-bit mantissa)
                      by default.
Observation:          cublasSetMathMode is called once at handle creation
                      with CUBLAS_TF32_TENSOR_OP_MATH. This is the cuBLAS
                      default since CUDA 11 and matches NVIDIA's
                      recommendation for inference. The user can opt out
                      per-tensor (dst->op_params[0] = GGML_PREC_F32) or
                      globally (GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32). Without
                      the override, F32 cuBLAS matmuls lose ~13 bits of
                      mantissa precision silently.
Evidence:             common.cuh:1490-1497 (handle creation with TF32),
                      ggml-cuda.cu:1626-1628 (per-tensor opt-out),
                      1630-1645 (env override).
Architectural Impact: Inference-quality matmuls are faster; validation
                      against an F32 reference will mismatch at the ULP
                      level. The opt-out is per-tensor, not per-call, so
                      differential testing requires graph-level flag
                      setting.
Correctness Impact:   TF32 has 10-bit mantissa (vs F32's 23-bit). For
                      matmul accumulators, the relative error is ~1e-3
                      (vs ~1e-7 for F32). Acceptable for LLM inference,
                      surprising for validation.
Optimization Type:    SIMD (Tensor Core TF32 instructions).
GwenLand Target:      glcuda
Recommendation:       ADOPT as default. Expose the per-tensor override
                      and the env override. Document the precision
                      reduction clearly.
Priority:             Medium
Difficulty:           XS
Dependencies:         R7
Confidence:           High
```

### Finding ARTX08-F09

```
Finding ID:           ARTX08-F09
Category:             LAYOUT_SUBOPTIMAL
Engine:               CUDA
Component:            Quantized weight padding
Source File:          ggml/src/ggml-cuda/common.cuh, ggml/src/ggml-cuda/ggml-cuda.cu
Function:             MATRIX_ROW_PADDING, ggml_backend_cuda_buffer_type_get_alloc_size, ggml_backend_cuda_buffer_init_tensor
Lines:                common.cuh:176, ggml-cuda.cu:906-922, 761-769
Summary:              Quantized weight tensors are padded to a multiple of
                      MATRIX_ROW_PADDING (512) bytes per row; padding is
                      zeroed at init_tensor time.
Observation:          get_alloc_size (906-922) adds
                      ggml_row_size(type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING)
                      bytes to the allocation when ne0 % MATRIX_ROW_PADDING != 0.
                      init_tensor (761-769) cudaMemsets the padding region
                      to zero. This allows MMQ/MMVQ vecdot kernels to issue
                      full 16-byte vectorized loads past the actual end of
                      the row without risking NaN contamination.
Evidence:             common.cuh:176 (MATRIX_ROW_PADDING=512), ggml-cuda.cu:914-919
                      (alloc_size padding), 761-769 (init_tensor zero-fill).
Architectural Impact: Up to 511 bytes of padding per quantized tensor.
                      Negligible memory cost; required for kernel correctness
                      on vectorized loads.
Correctness Impact:   None (the padding is zeroed; vecdot of zero is zero).
Optimization Type:    vectorization (full 16-byte loads).
GwenLand Target:      glcuda
Recommendation:       ADOPT. glcuda should pad quantized weight rows to
                      the same boundary and zero the padding at init.
Priority:             Medium
Difficulty:           XS
Dependencies:         R9
Confidence:           High
```

### Finding ARTX08-F10

```
Finding ID:           ARTX08-F10
Category:             BACKEND_DESIGN
Engine:               CUDA
Component:            Peer-to-peer
Source File:          ggml/src/ggml-cuda/ggml-cuda.cu
Function:             ggml_cuda_init, ggml_backend_cuda_cpy_tensor_async
Lines:                392-406, 2423-2480
Summary:              Peer access is enabled eagerly at startup only if
                      GGML_CUDA_P2P is set or GGML_USE_NCCL is defined;
                      otherwise cross-device copies use cudaMemcpyPeerAsync
                      without explicit peer access.
Observation:          ggml_cuda_init (392-406) loops over all physical
                      device pairs, calls cudaDeviceCanAccessPeer, and if
                      true calls cudaDeviceEnablePeerAccess — but only
                      when getenv("GGML_CUDA_P2P") is non-NULL. The VMM
                      pool's cuMemSetAccess (606-649) is similarly gated
                      by GGML_CUDA_P2P or GGML_USE_NCCL. cpy_tensor_async
                      (2453-2463) uses cudaMemcpyPeerAsync unconditionally
                      for cross-physical-device copies, which works
                      correctly without peer access enabled (the driver
                      transparently stages through host memory) but is
                      slower.
Evidence:             ggml-cuda.cu:392-406 (P2P enable loop, env-gated),
                      606-649 (VMM cuMemSetAccess, env-or-NCCL gated),
                      2453-2463 (cpy_tensor_async peer copy).
Architectural Impact: Default single-GPU setups pay no peer-access cost.
                      Multi-GPU setups without NCCL pay the host-staging
                      cost silently. NCCL implicitly enables peer access
                      via its own init.
Correctness Impact:   None. cudaMemcpyPeerAsync works without peer access
                      enabled (driver-managed).
Optimization Type:    asynchronous execution (peer direct copy vs host staging).
GwenLand Target:      glcuda
Recommendation:       ADOPT the env-gate. glcuda should enable peer access
                      explicitly when multi-GPU is requested, not silently.
                      MONITOR the NCCL auto-enable (it can surprise users
                      who didn't request P2P).
Priority:             Medium
Difficulty:           S
Dependencies:         R10
Confidence:           High
```

### Finding ARTX08-F11

```
Finding ID:           ARTX08-F11
Category:             EXECUTION_GRAPH
Engine:               CUDA
Component:            QKV multi-stream concurrency
Source File:          ggml/src/ggml-cuda/ggml-cuda.cu
Function:             ggml_backend_cuda_graph_optimize, ggml_cuda_graph_evaluate_and_capture
Lines:                4184-4426, 3880-3897, 3967-4043
Summary:              Graph optimizer reorders cgraph nodes to interleave QKV
                      branches and assigns each branch to a forked stream via
                      cudaEventRecord/cudaStreamWaitEvent; limited to nodes
                      named "attn_norm" with exactly 3-way fan-out.
Observation:          graph_optimize (4184-4426) finds fork nodes with
                      fan_out in [3, 3] (min_fan_out == max_fan_out == 3)
                      whose name contains "attn_norm" (4274), finds the join
                      node where 2+ branches converge (4312-4325), creates a
                      ggml_cuda_concurrent_event with n_streams=3, interleaves
                      the branch nodes (4395-4422) to extend tensor lifetimes,
                      and stores the event in the context's stream_context.
                      At execution time (3880-3897), the fork records
                      fork_event on the main stream, then
                      cudaStreamWaitEvent on each of the 3 forked streams.
                      The join records join_events[i] on each forked stream
                      and cudaStreamWaitEvent on the main stream (3972-3981).
                      Between fork and join, each node is routed to its
                      assigned stream via curr_stream_no.
Evidence:             ggml-cuda.cu:4184-4426 (graph_optimize), 4262-4263
                      (min/max fan-out = 3), 4274 (attn_norm name check),
                      4347-4354 (concurrent_event creation), 3880-3897 (fork),
                      3972-3981 (join), common.cuh:1256-1397 (event struct).
Architectural Impact: Enables 3-way parallelism in attention computation,
                      which can overlap the Q, K, V matmuls. The interleave
                      step (4395-4422) extends tensor lifetimes so the
                      allocator doesn't recycle them mid-fork. The
                      attn_norm-name gate (4274) limits applicability.
Correctness Impact:   The is_valid() check (common.cuh:1297-1385) verifies
                      no two branches write to overlapping memory ranges
                      and no branch depends on another branch's output.
                      Without this check, the reordering would be unsafe.
Optimization Type:    asynchronous execution (multi-stream fork/join).
GwenLand Target:      glcuda, GATE
Recommendation:       ADOPT but GENERALIZE. Drop the attn_norm-name gate;
                      use a structural test (any 2+-way fork that rejoins).
                      Make the fork count configurable (currently hard-coded
                      to 3).
Priority:             Medium
Difficulty:           L
Dependencies:         R6
Confidence:           High
```

### Finding ARTX08-F12

```
Finding ID:           ARTX08-F12
Category:             EXECUTION_GRAPH
Engine:               CUDA
Component:            Op fusion
Source File:          ggml/src/ggml-cuda/ggml-cuda.cu
Function:             ggml_cuda_try_fuse, ggml_cuda_can_fuse
Lines:                3144-3867, 2923-3141
Summary:              ~12 fusion patterns are detected at execution time
                      (per-node per-graph-execution), same anti-pattern as
                      ARTX01-F08.
Observation:          ggml_cuda_try_fuse (3144-3867) is called from the
                      per-node loop in ggml_cuda_graph_evaluate_and_capture
                      (4009). It tries, in order: gated_delta_net+cpy,
                      topk-moe, ROPE+VIEW+SET_ROWS, snake, multi-(add/mul),
                      MUL_MAT+MUL[+ADD]+MUL_MAT+MUL[+ADD]+GLU (FFN),
                      MUL_MAT+scale[+bias], MUL_MAT+ADD[+bias],
                      RMS_NORM+MUL[+ADD], SSM_CONV+ADD+SILU, SSM_CONV+SILU,
                      UNARY+MUL (SILU/SIGMOID/SOFTPLUS gate), UNARY+SQR
                      (ReLU-squared), SCALE+UNARY(TANH)+SCALE (softcap).
                      Each pattern calls ggml_cuda_can_fuse (2923-3141)
                      which checks shape, type, and memory-range constraints
                      via ggml_cuda_check_fusion_memory_ranges (2859-2920).
                      The fusion decision is not cached.
Evidence:             ggml-cuda.cu:3144-3867 (try_fuse), 2923-3141 (can_fuse),
                      2859-2920 (memory-range check), 4009 (per-node call).
Architectural Impact: O(N) fusion checks per graph run. For repeated
                      inference, this work is wasted. The breadth of
                      patterns is much larger than the CPU backend's
                      single RMS_NORM+MUL pattern (ARTX01-F08).
Correctness Impact:   None. Unfused execution is correct; fused execution
                      is verified via memory-range checks.
Optimization Type:    kernel fusion.
GwenLand Target:      glcuda, GATE
Recommendation:       ADAPT. Keep the patterns; lift detection into GATE's
                      planner so it runs once per graph, not per execution.
Priority:             High
Difficulty:           L
Dependencies:         R5
Confidence:           High
```

### Finding ARTX08-F13

```
Finding ID:           ARTX08-F13
Category:             BACKEND_DESIGN
Engine:               CUDA
Component:            Op offload policy
Source File:          ggml/src/ggml-cuda/ggml-cuda.cu
Function:             ggml_backend_cuda_reg, ggml_backend_cuda_device_offload_op
Lines:                5357-5380, 5169-5188
Summary:              op_offload_min_batch_size is a single global threshold
                      (default 32) applied uniformly to every device and
                      every op type.
Observation:          At registry init (5357), getenv("GGML_OP_OFFLOAD_MIN_BATCH")
                      is read once (default 32) and assigned to every
                      device's op_offload_min_batch_size. device_offload_op
                      (5184-5188) returns get_op_batch_size(op) >=
                      dev_ctx->op_offload_min_batch_size. get_op_batch_size
                      (5169-5182) returns ne[1] for MUL_MAT, ne[2] for
                      MUL_MAT_ID/ROPE/ROPE_BACK, nrows for everything else.
                      The threshold applies regardless of op type, op cost,
                      or device capability.
Evidence:             ggml-cuda.cu:5357 (env read), 5380 (per-device assign),
                      5184-5188 (offload_op), 5169-5182 (get_op_batch_size).
Architectural Impact: An op with batch 31 stays on CPU even if it would
                      benefit from GPU. An op with batch 32 goes to GPU
                      even if CPU would be faster (e.g., for tiny matmuls
                      where launch overhead dominates). No per-op policy.
Correctness Impact:   None.
Optimization Type:    None (policy choice).
GwenLand Target:      glcuda, GATE
Recommendation:       REJECT the single-threshold policy. Replace with a
                      per-op policy that considers op type, shape, and
                      estimated cost. Let GATE's scheduler make the decision.
Priority:             Low
Difficulty:           M
Dependencies:         R11
Confidence:           High
```

### Finding ARTX08-F14

```
Finding ID:           ARTX08-F14
Category:             GPU_KERNEL
Engine:               CUDA
Component:            Compute capability detection
Source File:          ggml/src/ggml-cuda/common.cuh, ggml/src/ggml-cuda/ggml-cuda.cu
Function:             GGML_CUDA_CC_* macros, ggml_cuda_init, ggml_cuda_highest_compiled_arch
Lines:                common.cuh:50-108, 135-172, 298-363; ggml-cuda.cu:217-410
Summary:              A single-integer CC encoding unifies NVIDIA, AMD (HIP),
                      and Moore Threads (MUSA) compute capabilities; runtime
                      helpers select kernels based on encoding and compiled
                      arch list.
Observation:          common.cuh:50-108 defines encoding constants and IS_*
                      macros. NVIDIA uses 100*major+10*minor (e.g., Hopper
                      = 900). AMD uses 0x1000000 + major*0x100 + minor*0x10
                      (parsed from gcnArchName at ggml-cuda.cu:169-214).
                      Moore Threads uses 0x0100000 + major*0x100 + minor*0x10.
                      The encoding is queried once at startup in
                      ggml_cuda_init (217-410) via cudaGetDeviceProperties
                      and stored in ggml_cuda_device_info::devices[id].cc.
                      Kernel routers (e.g., ggml_cuda_mul_mat:1830) read cc
                      once and pass to _should_use_* helpers, which consult
                      *_mma_available(cc) predicates (common.cuh:298-363).
                      ggml_cuda_highest_compiled_arch (135-172) returns the
                      highest compiled arch ≤ runtime cc, enabling forward
                      compatibility (e.g., a binary compiled for Ampere
                      runs with Ampere paths on Hopper).
Evidence:             common.cuh:50-108 (CC constants), 64-108 (IS_* macros),
                      298-363 (availability helpers), 135-172 (highest_compiled_arch);
                      ggml-cuda.cu:217-410 (init), 347-372 (per-vendor cc parse),
                      1830 (mul_mat reads cc).
Architectural Impact: Single source of truth for kernel selection across
                      three GPU vendors. Adding a new vendor means adding a
                      new offset constant and IS_* macro, not touching
                      kernel code. Forward compatibility is handled by
                      highest_compiled_arch.
Correctness Impact:   None. The encoding is a dispatch key, not a value
                      used in computation.
Optimization Type:    None (architectural choice).
GwenLand Target:      glcuda
Recommendation:       ADOPT. glcuda should use the same single-integer
                      encoding and the same *_available(cc) helper pattern.
Priority:             High
Difficulty:           S
Dependencies:         R7
Confidence:           High
```

### Finding ARTX08-F15

```
Finding ID:           ARTX08-F15
Category:             GPU_KERNEL
Engine:               CUDA
Component:            PDL launch
Source File:          ggml/src/ggml-cuda/common.cuh
Function:             ggml_cuda_kernel_launch, ggml_cuda_kernel_can_use_pdl, ggml_cuda_pdl_config
Lines:                114-133, 1552-1660
Summary:              Programmatic Dependent Launch (PDL) is used on Hopper+
                      via cudaLaunchKernelEx with programmaticStreamSerialization,
                      gated by per-(kernel, device) PTX version check.
Observation:          When GGML_CUDA_USE_PDL is defined (CUDART >= 12.3, or
                      >= 11.8 on non-MSVC), ggml_cuda_kernel_launch (1641-1660)
                      checks the per-(kernel, device) cache (1577-1630) for
                      whether the kernel's PTX version is >= 90. If yes,
                      cudaLaunchKernelEx is called with a
                      cudaLaunchAttributeProgrammaticStreamSerialization
                      attribute, allowing the next kernel to begin execution
                      before the current one fully drains. The PTX check is
                      cached in a mutex-guarded unordered_map<cache_key, bool>
                      with a custom MurmurHash3-based hash (1587-1609).
                      Device-side ggml_cuda_pdl_sync/lc (123-133) emit
                      cudaGridDependencySynchronize and
                      cudaTriggerProgrammaticLaunchCompletion for __CUDA_ARCH__
                      >= 900. __restrict__ is disabled on PDL kernels (1634-1639)
                      per CUDA documentation.
Evidence:             common.cuh:114-121 (PDL define), 123-133 (device-side
                      helpers), 1552-1575 (pdl_config struct), 1577-1630
                      (can_use_pdl with cache), 1641-1660 (kernel_launch),
                      1634-1639 (RESTRICT define).
Architectural Impact: Overlaps consecutive kernel launches on Hopper+,
                      reducing pipeline bubbles. The cache avoids repeated
                      cudaFuncGetAttributes calls. The __restrict__ disable
                      is a documented CUDA requirement for PDL.
Correctness Impact:   None. PDL is a scheduling hint; the device-side
      cudaGridDependencySynchronize ensures correctness if the kernel
      actually depends on the prior kernel's outputs.
Optimization Type:    asynchronous execution (kernel overlap via PDL).
GwenLand Target:      glcuda
Recommendation:       ADOPT. glcuda should use PDL on Hopper+ with the
                      same per-(kernel, device) cache. Document the
                      __restrict__ disable.
Priority:             Medium
Difficulty:           M
Dependencies:         R8
Confidence:           High
```

### Finding ARTX08-F16

```
Finding ID:           ARTX08-F16
Category:             MEMORY_PATTERN
Engine:               CUDA
Component:            Cross-stream copy event
Source File:          ggml/src/ggml-cuda/ggml-cuda.cu, ggml/src/ggml-cuda/common.cuh
Function:             ggml_backend_cuda_cpy_tensor_async, ggml_backend_cuda_context
Lines:                2423-2480, common.cuh:1410
Summary:              A single copy_event per backend context serializes
                      the cudaEventRecord for every cross-backend async
                      copy; correctness is preserved by stream ordering.
Observation:          cpy_tensor_async (2423-2480) handles cross-backend
                      copies by: (1) cudaMemcpyPeerAsync on src stream,
                      (2) if copy_event is null, create it with
                      cudaEventDisableTiming (2466-2469), (3) cudaEventRecord
                      on src stream, (4) cudaStreamWaitEvent on dst stream.
                      The copy_event is a single member of the context
                      (common.cuh:1410). If multiple cross-backend copies
                      are in flight from the same src backend, they share
                      the event. Each cudaEventRecord overwrites the
                      previous recording, but the cudaStreamWaitEvent on
                      the dst stream still serializes correctly because
                      event records form a queue on the src stream — the
                      dst stream waits for the most recent record, which
                      by stream ordering includes all prior records.
Evidence:             ggml-cuda.cu:2423-2480 (cpy_tensor_async),
                      2466-2469 (lazy copy_event creation), common.cuh:1410
                      (single member).
Architectural Impact: Correct but semantically lossy: the caller cannot
                      wait on a specific copy, only on "all copies up to
                      now on the src stream". For the current usage (one
                      cross-backend copy per graph node), this is fine.
                      If glcuda ever needs pipelined cross-backend copies,
                      a per-pair event would be cleaner.
Correctness Impact:   None. Stream ordering ensures the dst stream sees
                      all src-side copies before any subsequent dst-side
                      op that depends on them.
Optimization Type:    asynchronous execution (event-based cross-stream sync).
GwenLand Target:      glcuda
Recommendation:       ADOPT the single-event design for the common case.
                      MONITOR: if pipelined cross-backend copies become
                      common, switch to a per-pair event pool.
Priority:             Low
Difficulty:           S
Dependencies:         R1
Confidence:           Medium
```

---

## 18. Unknowns

* **U1**. Whether the 2-call warmup rule for CUDA graph capture is
  optimal. The rule disables capture on the first call and on any
  property change, requiring two consecutive stable calls before
  capture begins. For workloads where the graph changes every call
  (e.g., variable-length prefill), graphs never engage. Static analysis
  cannot determine the typical workload stability.

* **U2**. Cross-backend event wait (`ggml_backend_cuda_event_wait` when
  the backend is not CUDA) is `#if 0`'d out and aborts. The commented
  code uses `cudaLaunchHostFunc` to schedule a host callback that calls
  `ggml_backend_event_synchronize`. Whether this path was disabled due
  to a correctness issue, performance issue, or simply lack of testing
  is not documented. Requires runtime testing to enable safely.

* **U3**. Whether the `attn_norm`-name gate for QKV concurrency
  reflects a structural limitation or a conservative initial
  implementation. The TODO at `ggml-cuda.cu:4273` says "make this more
  generic" but doesn't explain what would break. Requires runtime
  testing on non-attention fork/join subgraphs.

* **U4**. Whether `op_offload_min_batch_size = 32` is the right default
  for current GPUs. The value dates back to early CUDA backend
  development and may be suboptimal for Hopper/Blackwell. Requires
  benchmarking on representative workloads.

* **U5**. Whether the VMM pool's 32 GiB virtual address reserve
  (`CUDA_POOL_VMM_MAX_SIZE = 1ull << 35`) is sufficient for very large
  models. The reserve is virtual, not physical, so it should be fine,
  but `cuMemAddressReserve` may fail on systems with limited virtual
  address space (e.g., 32-bit processes, which are unsupported anyway).
  Requires testing on edge-case configurations.

* **U6**. Whether the NCCL/BF16 AllReduce threshold heuristic
  (`ggml-cuda.cu:1016`) is optimal for non-RTX-4090 hardware. The
  comment cites "RTX 4090s connected via 16x PCIe 4.0" as the
  calibration target. For data-center GPUs with NVLink, the threshold
  should likely be higher (BF16 beneficial only for larger tensors).
  Requires benchmarking.

* **U7**. Whether the per-(kernel, device) PDL cache
  (`common.cuh:1611-1612`) grows unbounded across a long-running
  process. The cache is a static `unordered_map` with no eviction. If
  the process loads many distinct kernels over its lifetime, the cache
  could grow. Static analysis cannot determine typical kernel counts.

* **U8**. Whether `cudaStreamCaptureModeRelaxed` is safe in the
  presence of external stream operations (e.g., user-installed
  callbacks). The `ggml_cuda_lock` mitigates cuBLAS handle destruction
  but not arbitrary external ops. Requires understanding of all
  external stream users.

* **U9**. The interaction between QKV concurrency and CUDA graph
  capture. The concurrent event mechanism is set up in
  `graph_optimize` (ggml-cuda.cu:4184-4426) and executed in
  `graph_evaluate_and_capture` (3869-4081). When the graph is captured,
  the cudaEventRecord/cudaStreamWaitEvent calls are captured into the
  graph. Whether the captured graph correctly replays the fork/join on
  subsequent launches is not documented. Requires runtime testing.

---

## 19. References

| Reference | File                                          | Function / Symbol                                | Lines                  |
| --------- | --------------------------------------------- | ------------------------------------------------ | ---------------------- |
| R01       | `ggml/src/ggml-cuda/ggml-cuda.cu`             | `ggml_cuda_error` (fatal error handler)          | 97-107                 |
| R02       | `ggml/src/ggml-cuda/ggml-cuda.cu`             | `ggml_cuda_set_device`                           | 118-130                |
| R03       | `ggml/src/ggml-cuda/ggml-cuda.cu`             | `ggml_cuda_device_malloc`                        | 138-166                |
| R04       | `ggml/src/ggml-cuda/ggml-cuda.cu`             | `ggml_cuda_init` (CC detection, P2P, VMM, virtual devices) | 217-409        |
| R05       | `ggml/src/ggml-cuda/ggml-cuda.cu`             | `ggml_cuda_info` (lazy singleton)                | 411-414                |
| R06       | `ggml/src/ggml-cuda/ggml-cuda.cu`             | `ggml_cuda_pool_leg` (legacy best-fit pool)      | 419-532                |
| R07       | `ggml/src/ggml-cuda/ggml-cuda.cu`             | `ggml_cuda_pool_vmm` (VMM pool)                  | 536-682                |
| R08       | `ggml/src/ggml-cuda/ggml-cuda.cu`             | `ggml_backend_cuda_context::new_pool_for_device` | 685-693                |
| R09       | `ggml/src/ggml-cuda/ggml-cuda.cu`             | `ggml_backend_cuda_context::~ggml_backend_cuda_context` | 702-719       |
| R10       | `ggml/src/ggml-cuda/ggml-cuda.cu`             | `ggml_backend_cuda_buffer_*` (free/get/init/set/get/cpy/clear) | 739-863 |
| R11       | `ggml/src/ggml-cuda/ggml-cuda.cu`             | `ggml_backend_cuda_buffer_type_*`                | 866-957                |
| R12       | `ggml/src/ggml-cuda/ggml-cuda.cu`             | `ggml_backend_cuda_comm_context` + AllReduce (NCCL/internal/butterfly) | 965-1255 |
| R13       | `ggml/src/ggml-cuda/ggml-cuda.cu`             | `ggml_backend_cuda_host_buffer_type` + `ggml_cuda_host_malloc` | 1257-1325 |
| R14       | `ggml/src/ggml-cuda/ggml-cuda.cu`             | `batched_mul_mat_traits`                         | 1359-1403              |
| R15       | `ggml/src/ggml-cuda/ggml-cuda.cu`             | `ggml_cuda_mul_mat_cublas_impl`                  | 1405-1617              |
| R16       | `ggml/src/ggml-cuda/ggml-cuda.cu`             | `ggml_cuda_mul_mat_cublas` (compute type select) | 1619-1660              |
| R17       | `ggml/src/ggml-cuda/ggml-cuda.cu`             | `ggml_cuda_mul_mat` (kernel router)              | 1812-1852              |
| R18       | `ggml/src/ggml-cuda/ggml-cuda.cu`             | `ggml_cuda_mul_mat_id`                           | 1854-2009              |
| R19       | `ggml/src/ggml-cuda/ggml-cuda.cu`             | `ggml_cuda_compute_forward` (op dispatch switch) | 2011-2364              |
| R20       | `ggml/src/ggml-cuda/ggml-cuda.cu`             | `ggml_backend_cuda_set_tensor_async` / `get_tensor_async` / `cpy_tensor_async` / `synchronize` | 2383-2488 |
| R21       | `ggml/src/ggml-cuda/ggml-cuda.cu`             | `ggml_cuda_graph_check_compability`              | 2496-2529              |
| R22       | `ggml/src/ggml-cuda/ggml-cuda.cu`             | `ggml_cuda_graph_update_required`                | 2535-2575              |
| R23       | `ggml/src/ggml-cuda/ggml-cuda.cu`             | `ggml_cuda_graph_update_executable`              | 2577-2603              |
| R24       | `ggml/src/ggml-cuda/ggml-cuda.cu`             | `ggml_cuda_can_fuse` / `ggml_cuda_try_fuse`      | 2923-3141, 3144-3867   |
| R25       | `ggml/src/ggml-cuda/ggml-cuda.cu`             | `ggml_cuda_graph_evaluate_and_capture`           | 3869-4081              |
| R26       | `ggml/src/ggml-cuda/ggml-cuda.cu`             | `ggml_backend_cuda_graph_compute`                | 4100-4157              |
| R27       | `ggml/src/ggml-cuda/ggml-cuda.cu`             | `ggml_backend_cuda_event_record` / `event_wait`  | 4159-4182              |
| R28       | `ggml/src/ggml-cuda/ggml-cuda.cu`             | `ggml_backend_cuda_graph_optimize` (QKV reordering) | 4184-4426           |
| R29       | `ggml/src/ggml-cuda/ggml-cuda.cu`             | `ggml_backend_cuda_interface` (vtable)           | 4428-4445              |
| R30       | `ggml/src/ggml-cuda/ggml-cuda.cu`             | `ggml_backend_cuda_register_host_buffer` / `unregister` | 4494-4527        |
| R31       | `ggml/src/ggml-cuda/ggml-cuda.cu`             | `ggml_backend_cuda_device_supports_op`           | 4719-5161              |
| R32       | `ggml/src/ggml-cuda/ggml-cuda.cu`             | `ggml_backend_cuda_device_event_*`               | 5190-5218              |
| R33       | `ggml/src/ggml-cuda/ggml-cuda.cu`             | `ggml_backend_cuda_device_interface`             | 5220-5236              |
| R34       | `ggml/src/ggml-cuda/ggml-cuda.cu`             | `ggml_backend_cuda_reg_get_proc_address`         | 5317-5338              |
| R35       | `ggml/src/ggml-cuda/ggml-cuda.cu`             | `ggml_backend_cuda_reg`                          | 5348-5401              |
| R36       | `ggml/src/ggml-cuda/ggml-cuda.cu`             | `ggml_backend_cuda_init`                         | 5403-5423              |
| R37       | `ggml/src/ggml-cuda/common.cuh`               | CC encoding constants + IS_* macros              | 50-108                 |
| R38       | `ggml/src/ggml-cuda/common.cuh`               | `ggml_cuda_highest_compiled_arch`                | 135-172                |
| R39       | `ggml/src/ggml-cuda/common.cuh`               | `MATRIX_ROW_PADDING`, `GGML_CUDA_MAX_STREAMS`    | 176-178                |
| R40       | `ggml/src/ggml-cuda/common.cuh`               | `CUDA_CHECK` / `CUBLAS_CHECK` / `NCCL_CHECK` / `CU_CHECK` | 180-228        |
| R41       | `ggml/src/ggml-cuda/common.cuh`               | `CUDA_SET_SHARED_MEMORY_LIMIT`                   | 230-245                |
| R42       | `ggml/src/ggml-cuda/common.cuh`               | `*_mma_available(cc)` helpers                    | 298-363                |
| R43       | `ggml/src/ggml-cuda/common.cuh`               | `ggml_cuda_type_traits<ggml_type>`               | 964-1125               |
| R44       | `ggml/src/ggml-cuda/common.cuh`               | `ggml_cuda_device_info`                          | 1129-1152              |
| R45       | `ggml/src/ggml-cuda/common.cuh`               | `ggml_cuda_pool` + `ggml_cuda_pool_alloc<T>`     | 1159-1208              |
| R46       | `ggml/src/ggml-cuda/common.cuh`               | `ggml_tensor_extra_gpu` (dead)                   | 1213-1216              |
| R47       | `ggml/src/ggml-cuda/common.cuh`               | `ggml_cuda_graph`                                | 1223-1254              |
| R48       | `ggml/src/ggml-cuda/common.cuh`               | `ggml_cuda_concurrent_event`                     | 1256-1397              |
| R49       | `ggml/src/ggml-cuda/common.cuh`               | `ggml_cuda_stream_context`                       | 1399-1405              |
| R50       | `ggml/src/ggml-cuda/common.cuh`               | `ggml_backend_cuda_context`                      | 1407-1518              |
| R51       | `ggml/src/ggml-cuda/common.cuh`               | PDL support (`ggml_cuda_kernel_launch`, `can_use_pdl`, `pdl_config`) | 114-133, 1552-1660 |
| R52       | `ggml/include/ggml-cuda.h`                    | `GGML_CUDA_MAX_DEVICES = 16`, public API         | 1-47                   |
| R53       | `ggml/src/ggml-cuda/vendors/{cuda,hip,musa}.h`| Vendor-alias macros                              | various                |
| R54       | `ggml/src/ggml-cuda/ggml-cuda.cu`             | `ggml_backend_cuda_device_offload_op` + `get_op_batch_size` | 5169-5188      |
| R55       | `ggml/src/ggml-cuda/ggml-cuda.cu`             | `ggml_cuda_check_fusion_memory_ranges`           | 2859-2920              |
