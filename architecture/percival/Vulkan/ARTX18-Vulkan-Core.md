# ARTX18 — Vulkan Backend Core

**Audited commit:** `555881ebc8b0fc0402b30e09258a32a7bfd13c52`
**Last updated:** 2026-07-25
**Auditor:** Percival-aux
**Target GwenLand module:** `glvulkan` (instance/device/queue, pipeline cache,
descriptor pool, buffer types, graph compute, async), `GATE` (graph traversal,
fusion, submit policy)

---

## 1. Executive Summary

The Vulkan backend of llama.cpp is the most feature-complete backend in the
tree. It is the *only* backend that simultaneously advertises `async=true`,
`events=true`, `host_buffer=true` and `graph_optimize` — making it the closest
existing analogue to what GwenLand wants `GATE`+`glvulkan` to be. The whole
backend lives in a single 19,146-line C++ translation unit
(`ggml/src/ggml-vulkan/ggml-vulkan.cpp`), with SPIR-V shaders compiled at build
time by a separate `vulkan-shaders-gen` tool driven from
`vulkan-shaders/CMakeLists.txt`.

Architecturally, the backend is built around six pillars:

1. A **process-wide Vulkan instance** (`vk_instance_t`) created lazily in
   `ggml_vk_instance_init`. Requires Vulkan 1.2. Layers are restricted to
   `VK_LAYER_KHRONOS_validation` and only when `GGML_VULKAN_VALIDATE` is set.
2. A **per-physical-device logical device** (`vk_device_struct`) created in
   `ggml_vk_get_device`. It enumerates device extensions, queries features
   through a long `pNext` chain, picks queue families, and installs a single
   shared descriptor set layout used by every pipeline.
3. A **pipeline-per-(shader × spec-constants × subgroup-size)** cache. There
   is *no* `VkPipelineCache` object — every pipeline is created with
   `VK_NULL_HANDLE` as the cache (`ggml-vulkan.cpp:2817`). Pipelines are
   compiled lazily on first dispatch via `ggml_vk_load_shaders` and protected
   by a `compile_mutex` + `compile_cv` pair.
4. A **descriptor pool-per-context**, growing geometrically in chunks of
   `VK_DEVICE_DESCRIPTOR_POOL_SIZE = 256` sets. The pool is *not* reset
   between graph computes; descriptor set indices reset only on
   `ggml_vk_graph_cleanup`.
5. A **graph executor** (`ggml_backend_vk_graph_compute`) that traverses the
   `ggml_cgraph`, applies a fairly elaborate plan-time fusion pass, builds
   one command buffer per "batch" of nodes, and submits batches to the
   compute queue, optionally overlapping host command-buffer recording with
   GPU execution.
6. An **event/async layer** built on `vk::Event` (for in-flight recording)
   and timeline semaphores (for cross-queue and cross-process
   synchronization). `ggml_backend_vk_event_record` records a `vkCmdSetEvent`
   *and* signals a timeline semaphore; `event_synchronize` blocks on the
   timeline semaphore with `waitSemaphores`.

The standout architectural decisions are: (a) the descriptor pool grown per
backend context, not per graph, (b) the absence of a `VkPipelineCache`, (c)
the dual-queue design (compute + transfer) on AMD dGPUs and via
`GGML_VK_ASYNC_USE_TRANSFER_QUEUE`, (d) the plan-time fusion pass with ~15
patterns including a 10-op `TOPK_MOE` subgraph, (e) the SPIR-V patching layer
that rewrites shader modules at runtime to insert float-control capabilities
and strip `SPV_NV_cooperative_matrix_decode_vector` when the driver exposes
only `VK_NV_cooperative_matrix2`.

For GwenLand, the decisions worth **ADOPT**ing are the descriptor-pool growth
strategy, the lazy pipeline compilation with mutex + condvar, the SPIR-V
patching layer for float controls, the dual-queue model with timeline
semaphores, and the `vk_event` dual-event-plus-timeline-semaphore design.
The decisions worth **REJECT**ing are the absence of `VkPipelineCache` and the
"all pipelines share one descriptor set layout" choice (which makes descriptor
sets cheap but forces every pipeline to declare `MAX_PARAMETER_COUNT = 12`
storage buffer bindings, raising memory pressure on descriptor memory).

---

## 2. Purpose

This document is the architectural record of the Vulkan backend's *core
machinery*: instance, device, queue, descriptor, pipeline, buffer, graph
compute, op dispatch, async, events. It is the Vulkan analogue of
ARTX08 (CUDA core) and ARTX15 (Metal core). It does **not** cover the
individual compute shaders (that is ARTX19, Vulkan shaders) nor the
cooperative-matrix matmul kernels (ARTX20).

Where static analysis cannot reach a conclusion, an **Unknowns** section
records the question explicitly.

---

## 3. Source Files

| Path                                                    | Lines  | Role |
| ------------------------------------------------------- | ------ | ---- |
| `ggml/src/ggml-vulkan/ggml-vulkan.cpp`                  | 19146  | Primary — the entire backend in one .cpp |
| `ggml/src/ggml-vulkan/CMakeLists.txt`                   | 248    | Build: shader extension tests, ExternalProject for `vulkan-shaders-gen` |
| `ggml/src/ggml-vulkan/vulkan-shaders/CMakeLists.txt`    | 43     | Build: `vulkan-shaders-gen` host tool |
| `ggml/src/ggml-vulkan/cmake/host-toolchain.cmake.in`    | —      | Cross-compile host toolchain template (out of scope) |

The `vulkan-shaders/` directory contains 90+ `.comp` GLSL compute shaders
plus 7 feature-test shaders used by `CMakeLists.txt` to probe `glslc`
extension support. Those shaders are out of scope here.

The cpp file is large; the audit targeted:

* Header (lines 1–700): vendor IDs, pipeline/buffer struct declarations,
  descriptor set layout, queue handle hierarchy.
* `vk_device_struct` (lines 720–1077): per-device state and lifetime.
* `vk_buffer_struct`, push-constant structs (lines 1095–1530).
* `vk_instance_t` and instance init (lines 2347–2364, 7078–7326).
* `ggml_vk_get_device` (lines 5906–6820): physical-device feature walk.
* `ggml_vk_create_pipeline_func` (lines 2627–2878): pipeline creation,
  SPIR-V patching, no pipeline cache.
* Descriptor pool allocation (lines 2889–2934).
* `ggml_vk_dispatch_pipeline` template (lines 7877–7907): the single
  dispatch site for all compute shaders.
* `ggml_vk_host_malloc` / `ggml_vk_host_get` / pinned memory
  (lines 7739–7800).
* `ggml_vk_create_buffer` and `ggml_vk_create_buffer_device`
  (lines 3194–3398): memory type selection, `VK_EXT_external_memory_host`.
* `ggml_vk_sync_buffers` and event helpers (lines 3416–3470).
* `ggml_vk_load_shaders` (lines 3938–5901): per-device pipeline
  instantiation, spec constants, async compile handshake.
* `ggml_vk_op_f32` (lines 11610–11910): generic op dispatch template.
* `ggml_vk_build_graph` (lines 14797–15315): per-node command buffer
  recording, overlap-aware sync insertion.
* `ggml_vk_compute_forward` (lines 15317–15363): submit hook.
* `ggml_vk_synchronize` (lines 15928–15987): fence + timeline sema sync.
* `ggml_backend_vk_graph_compute` (lines 16525–16938): top-level graph
  traversal, fusion detection, batched submit.
* `ggml_vk_graph_optimize` (lines 16941–17196): plan-time node reorder.
* Backend / device / reg vtables (lines 17255–17272, 18184–18200, 18246–18272).
* `ggml_backend_vk_event_record` / `event_wait` / `event_synchronize`
  (lines 17198–17252, 18107–18137).
* `ggml_backend_vk_device_buffer_from_host_ptr`
  (lines 18139–18182): pinned-host import.

---

## 4. Architecture Overview

```
                          ┌────────────────────────────────────┐
                          │  ggml_backend_vk_reg()  (18253)    │
                          │   reg_i.get_device → device list   │
                          └────────────────┬───────────────────┘
                                           │
              ┌────────────────────────────┴───────────────────────────┐
              ▼                                                         ▼
   ┌───────────────────────┐                            ┌────────────────────────┐
   │ ggml_backend_vk_device_i  │                       │ ggml_backend_vk_i        │
   │  (18184)                    │                      │  (17255)                 │
   │  - get_props (async=true)   │                      │  - set_tensor_async      │
   │  - get_buffer_type          │                      │  - get_tensor_async      │
   │  - get_host_buffer_type     │                      │  - cpy_tensor_async      │
   │  - buffer_from_host_ptr     │                      │  - synchronize           │
   │  - supports_op              │                      │  - graph_compute         │
   │  - event_new/free/sync      │                      │  - event_record/wait     │
   └────────────────┬───────────┘                      │  - graph_optimize        │
                    │                                    └──────────────┬──────────┘
                    ▼                                                   ▼
        ┌───────────────────────────┐                  ┌──────────────────────────────┐
        │ vk_instance_t (2347)      │                  │ ggml_backend_vk_context      │
        │  - vk::Instance           │                  │   (2164)                     │
        │  - device_indices[]       │                  │  - device (vk_device)        │
        │  - devices[GGML_VK_MAX…]  │                  │  - descriptor_pools[]        │
        └───────────┬───────────────┘                  │  - descriptor_sets[]         │
                    │                                    │  - compute/transfer_cmd_pool│
                    ▼                                    │  - compute_ctx, transfer_ctx│
        ┌───────────────────────────┐                  │  - prealloc_x/y/split_k     │
        │ vk_device_struct (720)    │                  │  - unsynced_nodes_{r,w}     │
        │  - physical_device        │                  └──────────────┬───────────────┘
        │  - device (VkDevice)      │                                 │
        │  - compute_queue          │                                 │
        │  - transfer_queue         │                                 ▼
        │  - dsl (shared layout)    │                  ┌──────────────────────────────┐
        │  - pipeline_* (per op)    │◀─────────────────│ ggml_vk_dispatch_pipeline    │
        │  - all_pipelines[]        │                  │  (7877)                      │
        │  - pinned_memory[]        │                  │  pushConstants + bindPipeline│
        │  - compile_mutex/cv       │                  │  + bindDescriptorSets        │
        └───────────┬───────────────┘                  │  + dispatch(wg0,wg1,wg2)     │
                    │                                    └──────────────┬───────────────┘
                    ▼                                                   │
        ┌───────────────────────────┐                                 │
        │ vk_queue (332)            │                                 ▼
        │  - handle (sync or unsync)│                   ┌──────────────────────────────┐
        │  - cmd_pool (deque<cb>)   │                   │ vk_command_buffer            │
        │  - stage_flags            │                   │  (280)                       │
        │  - transfer_only          │                   │  - buf, use_counter, in_use  │
        └───────────────────────────┘                   └──────────────────────────────┘
```

Key invariants:

* **One `vk::Instance` per process.** `vk_instance_initialized` is a process-
  wide static bool guarded by idempotent re-entry (`ggml-vulkan.cpp:7079`).
* **One `vk_device` per physical device, shared across backends.** A
  `vk_instance.devices[idx]` cache means multiple `ggml_backend_vk_init`
  calls on the same physical device share their device, queue, descriptor
  set layout, and pipeline cache.
* **One `dsl` per device, shared by every pipeline.** All pipelines are
  created with `device->dsl` (line 2750), which declares
  `MAX_PARAMETER_COUNT = 12` storage-buffer bindings.
* **One `vk_command_pool` per (context, queue) pair.** The context owns
  `compute_cmd_pool` and (if async + transfer queue) `transfer_cmd_pool`
  (lines 2211–2212, 7349–7358).
* **Backend device props advertise `async=true`, `events=true`,
  `host_buffer=true`, `buffer_from_host_ptr=false`** (lines 17445–17450).
  `buffer_from_host_ptr` is false because Vulkan's
  `VK_EXT_external_memory_host` requires page-aligned host pointers and is
  not a general `buffer_from_host_ptr` implementation.

---

## 5. Execution Flow

`ggml_backend_vk_graph_compute(backend, cgraph)` (line 16525) drives
execution. The flow is:

1. **Pre-allocate scratch buffers** (line 16593, `ggml_vk_preallocate_buffers`)
   — `prealloc_x`, `prealloc_y`, `prealloc_split_k`, and a fixed 1 KB
   `prealloc_add_rms_partials` for the ADD+RMS_NORM fusion.
2. **Submit any pending transfer-queue work** (`ggml_vk_submit_transfer_ctx`,
   line 16555) so that the compute context can wait on the transfer
   semaphore.
3. **Initialize RMS partials to zero** if the ADD+RMS_NORM fusion is
   enabled (lines 16596–16598).
4. **Compute `flops_per_submit`** (lines 16606–16618): a 200 GFLOP cap,
   scaled down on weak AMD GPUs, scaled to `last_total_flops / 40` so small
   graphs submit earlier. Doubles after each of the first 3 submits.
5. **For each node `i` in `cgraph->nodes`:**
   a. **Fusion detection** (lines 16640–16774): runs `ggml_vk_fuse_multi_add`
      and 14 other pattern matchers. Each match sets
      `ctx->num_additional_fused_ops` and a `fusion_string`. The fusion
      detection is entirely **plan-time** here, not execution-time.
   b. **Anti-aliasing check** (lines 16778–16836): if a fusion would
      overwrite a still-live src, fusion is rolled back. The exception is
      `TOPK_MOE` single-row, where all src values are loaded before any
      store.
   c. **Submit decision** (lines 16838–16843): submit if `>=100` nodes
      accumulated, `>= flops_per_submit` flops accumulated, this is the
      last node, or the "almost ready" fence hasn't been signaled yet.
   d. **`ggml_vk_build_graph`** (line 16845): records this node (and any
      fused successors) into the current compute context's command buffer.
   e. **`ggml_vk_compute_forward`** is called from inside `build_graph`
      when `submit` is true: it ends the command buffer, calls
      `ggml_vk_submit(subctx, fence-or-{})`, and sets `submit_pending`.
   f. **Skip ahead** past fused nodes (`i += ctx->num_additional_fused_ops`).
6. **Final sync** (line 16931): if `support_async` is false, force a full
   `ggml_vk_synchronize`.

### 5.1 `ggml_vk_build_graph` per node

Inside `ggml_vk_build_graph` (line 14797), each node:

1. Skips empty ops and tensors with `GGML_TENSOR_FLAG_COMPUTE` unset.
2. Inspects the next node — if it's `RMS_NORM` and the current is `ADD`
   with matching shape, sets up `do_add_rms_partials` to enable the
   fused ADD→RMS partials path (lines 14814–14827).
3. Acquires the compute context (`ggml_vk_get_compute_ctx`), waiting on
   the transfer semaphore if pending.
4. Runs the **unsynced-overlap analysis** (lines 14837–14917): if the
   current node or any of its srcs overlaps in `(buffer, offset, size)`
   with a node on `unsynced_nodes_written` / `_read` that hasn't been
   barriered yet, call `ggml_vk_sync_buffers` and clear the unsynced
   lists. This is the only cross-node memory-hazard detection.
5. **Dispatches the op** via the giant `switch (node->op)` at line 14934.
   Each case calls a `ggml_vk_<op>(ctx, compute_ctx, srcs..., node)`
   helper. The helper selects a pipeline, requests descriptor sets,
   computes push constants, and calls `ggml_vk_dispatch_pipeline`.
6. Updates `unsynced_nodes_written` / `_read` with this node and its srcs
   for the next iteration's overlap analysis.

### 5.2 `ggml_vk_dispatch_pipeline`

The single dispatch site (line 7877) is a template that takes a push-
constant struct. It:

1. Computes workgroup counts `wg_i = CEIL_DIV(elements[i], pipeline->wg_denoms[i])`.
2. Asserts `wg_i ≤ maxComputeWorkGroupCount[i]`.
3. Takes the next descriptor set from `ctx->descriptor_sets[ctx->descriptor_set_idx++]`.
4. Calls `vkUpdateDescriptorSets` with one `WriteDescriptorSet` covering
   all `pipeline->parameter_count` storage-buffer bindings.
5. Records `pushConstants`, `bindPipeline`, `bindDescriptorSets`, `dispatch`.

There is **no** `vkCmdDispatchBase` use; batch dispatch uses ordinary
`vkCmdDispatch` with `base_work_group_z` encoded into the push constants
(`vk_mat_mat_push_constants.base_work_group_z`, line 1158). This avoids
requiring `VK_KHR_device_group` and keeps the dispatch path simple.

### 5.3 `ggml_vk_synchronize`

`ggml_vk_synchronize` (line 15928) is the backend's hard sync. It:

1. Submits any pending transfer-queue work and signals the transfer
   timeline semaphore.
2. Ends the compute context's command buffer and submits it (no fence).
3. If the transfer semaphore was signaled, submits a *zero-command*
   `SubmitInfo` to the compute queue that waits on the transfer semaphore
   and signals `ctx->fence`.
4. Otherwise submits an empty `SubmitInfo` with `ctx->fence`.
5. Calls `ggml_vk_wait_for_fence` (line 2386), which:
   a. If `almost_ready_fence_pending` is set, `waitForFences` on it
      (allows the CPU to sleep).
   b. Otherwise, **spin-polls** `getFenceStatus` with 100 `YIELD()` calls
      per iteration (lines 2397–2413). This is a hot CPU loop, not a
      blocking wait.

The hot-spin poll is a deliberate latency optimization: by the time
`ggml_vk_synchronize` is called, the GPU should be nearly done (the
`almost_ready_fence` was signaled at < 20% remaining), so spinning is
cheaper than another `waitForFences` syscall.

---

## 6. Data Layout

### 6.1 Tensor descriptor

A `ggml_tensor` carries `ne[GGML_MAX_DIMS]` and `nb[GGML_MAX_DIMS]`.
Vulkan buffers are bound via `vk::DescriptorBufferInfo { buffer, offset,
range }`. The `vk_subbuffer` (line 1116) wraps a `vk_buffer` + offset +
size and converts implicitly to `vk::DescriptorBufferInfo`.

Tensor offsets into the backing `vk_buffer` are computed via
`vk_tensor_offset(tensor)` (line 2237), which subtracts a sentinel
`vk_ptr_base = 0x1000` from `tensor->data`. This works because
`tensor->data` is set to `vk_ptr_base + buffer_offset` at buffer alloc
time (not visible in this file; in `ggml-backend-vk.cpp`'s
`set_tensor` path the data pointer is synthesized from the buffer
context). The `0x1000` base avoids null-pointer representation pitfalls.

### 6.2 Misalignment handling

`get_misalign_bytes` (line 2244) returns the bytes by which a tensor's
offset violates `minStorageBufferOffsetAlignment`. Most ops require
zero misalignment (asserted in `init_pushconst_tensor_offsets`,
line 2257); the few that don't (mul_mat_vec_p021, mul_mat_vec_nc,
fwht) carry explicit `b_offset`/`d_offset`/`src_offset`/`dst_offset`
fields in their push constants so the shader can add the misalignment
back. This is the same pattern as ARTX01's `wdata` offset arithmetic,
expressed per-op.

### 6.3 Subbuffer range clamping

`ggml_vk_get_max_buffer_range` (line 2379) returns
`min(buf->size - offset, limits.maxStorageBufferRange)`. This is
critical because Vulkan storage-buffer descriptor ranges must not exceed
`maxStorageBufferRange` (commonly 2²⁷ bytes). Tensors backed by a large
`vk_buffer` whose total size exceeds the limit get their descriptor
ranges clamped here, so the shader never reads out-of-bounds *from the
descriptor's perspective* — though shader-side bounds checks on actual
tensor `ne[]` are still required.

### 6.4 Push constant structs

Push constants are limited to 128 bytes by default
(`static_assert(sizeof(vk_flash_attn_push_constants) <= 128)` at line 1273,
`vk_op_glu_push_constants <= 128` at line 1325, etc.). The exception is
`vk_op_multi_add_push_constants` (line 1484), which is allowed 256 bytes
because it carries `nb[MAX_PARAMETER_COUNT][4]` — 12 sources × 4 strides.
The 256-byte path is gated on `device->properties.limits.maxPushConstantsSize
>= sizeof(vk_op_multi_add_push_constants)` at line 6416.

---

## 7. Memory Layout

### 7.1 Buffer types

| Type                    | Backing memory                                       | Use |
| ----------------------- | ---------------------------------------------------- | --- |
| `vk_buffer_type`        | Device-local (prefers ReBAR, falls back to device-only) | Model weights, activations |
| `vk_host_buffer_type`   | Host-visible + coherent + cached (pinned)            | Staging, CPU↔GPU pinned transfers |

`ggml_vk_create_buffer_device` (line 3361) selects memory type with a
preference chain that depends on `device->uma`,
`device->prefer_host_memory`, `device->disable_host_visible_vidmem`, and
`device->allow_sysmem_fallback`:

* UMA: prefer `eDeviceLocal | eHostVisible | eHostCoherent`, fall back
  to `eDeviceLocal`, then `eHostVisible | eHostCoherent`.
* Discrete + ReBAR: prefer `eDeviceLocal | eHostVisible | eHostCoherent`,
  fall back to `eDeviceLocal`.
* Discrete without ReBAR: `eDeviceLocal` only (unless `allow_sysmem_fallback`).

### 7.2 Pinned memory registry

`vk_device_struct::pinned_memory` is a `vector<tuple<void*, size_t, vk_buffer>>`
(line 1034), protected by a `std::shared_mutex`. `ggml_vk_host_malloc`
(line 7739) allocates a host-visible buffer and registers its mapped
pointer; `ggml_vk_host_get` (line 7787) does a linear scan to find which
pinned buffer contains a given host pointer. This is the path that lets
`ggml_vk_buffer_write_2d_async` skip the staging copy and use
`vkCmdCopyBuffer` directly from a pinned source (line 8130).

The linear scan is O(N) in the number of pinned allocations. The code
comments don't mention this; for a workload with many small pinned
regions it could become hot.

### 7.3 Prealloc scratch buffers

The backend context pre-allocates three scratch buffers reused across
graphs (lines 2171–2172, 7340–7342):

* `prealloc_x` — converted src0 (e.g., dequantized weights for matmul).
* `prealloc_y` — converted src1 (e.g., Q8_1 activations, fp16 staging
  for coopmat2 decode-vector).
* `prealloc_split_k` — split-K reduction workspace for tall-K matmuls.
* `prealloc_add_rms_partials` — 1 KB fixed, accumulates squared norms
  for the ADD→RMS_NORM fusion.

A `prealloc_y_last_pipeline_used` cache (line 2184) skips re-conversion
if the previous graph already materialized the same tensor with the same
pipeline. This is a one-entry cache; there is no LRU.

### 7.4 Sync staging buffer

`vk_device_struct::sync_staging` (line 1037) is a host-visible buffer
grown on demand by `ggml_vk_ensure_sync_staging_buffer` (line 8006) to
hold tensors that need to round-trip through host memory for
synchronization reads. It is shared device-wide, not per-context.

---

## 8. Parallelism Strategy

### 8.1 Compute shader workgroup sizing

Every pipeline declares `wg_denoms[3]` — the workgroup "denominators"
used to compute dispatch dimensions. `ggml_vk_dispatch_pipeline` computes
`wg_i = CEIL_DIV(elements[i], pipeline->wg_denoms[i])`. The workgroup
size itself is baked into the SPIR-V via `layout(local_size_x_id = K,
...)` and selected at pipeline-create time via specialization constants
(e.g., `{512, 1, 1}` for elementwise ops,
`{subgroup_size * 4, 1, 1}` for matmul).

For elementwise ops, `ggml_vk_op_f32` (line 11884) picks a 512×512×Z
dispatch when `ne > 262144`, falling back to 512×Y×1 then ne×1×1. The
512×512×Z form keeps `maxComputeWorkGroupCount[0]` (commonly 65535)
from being the bottleneck on very large tensors.

### 8.2 Multi-queue (compute + transfer)

Device init (`ggml_vk_get_device`, lines 6198–6789) picks two queue
families:

* Compute family: required `VK_QUEUE_COMPUTE_BIT`, avoids
  `VK_QUEUE_GRAPHICS_BIT` unless `GGML_VK_ALLOW_GRAPHICS_QUEUE` is set
  (line 6202). The comment cites RADV performance gains from avoiding
  the graphics queue.
* Transfer family: required `VK_QUEUE_TRANSFER_BIT`, avoids
  `VK_QUEUE_COMPUTE_BIT` and graphics. Falls back to reusing the
  compute family if no dedicated transfer family exists.

If the families are the same but the family has ≥ 2 queues, two queues
are created from the same family (line 6393). If only one queue is
available, `device->single_queue = true` and the transfer queue is
"aliased" — it shares the compute queue's handle via
`ggml_vk_create_aliased_queue` (line 6786).

`async_use_transfer_queue` is enabled by default on AMD dGPUs (non-GCN,
non-UMA, no graphics queue) or via `GGML_VK_ASYNC_USE_TRANSFER_QUEUE`
(line 6784). When enabled, `set_tensor_async` and friends record into
the transfer context, and the next compute context waits on the
`transfer_semaphore` (a timeline semaphore, line 2201).

### 8.3 Internally synchronized queues

If the device supports `VK_KHR_internally_synchronized_queues`
(fallback definitions at lines 159–175), `vk_queue_handle_unsynchronized`
(line 324) is used — its `submit` calls `queue.submit` without holding
any host mutex, relying on driver-internal synchronization. Otherwise
`vk_queue_handle_synchronized` (line 314) wraps `submit` in a
`std::mutex`. The decision is made per-queue in `ggml_vk_create_queue`
(line 3083).

### 8.4 Pipeline compile parallelism

Pipeline compilation is parallelized across threads via a mutex +
condvar handshake (lines 728–729, 3977, 5881–5900). The pattern:

1. Thread A enters `ggml_vk_load_shaders` holding `compile_mutex`.
2. A walks the pipeline list. For the requested-but-uncompiled pipeline,
   A claims it (sets `compile_pending = true`) and remembers a
   `CompileTask` (line 3925).
3. A **drops** `compile_mutex` and runs `ggml_vk_create_pipeline_func`
   (the actual `vkCreateComputePipelines` call) without holding any
   lock.
4. Other threads can concurrently compile other pipelines.
5. At the end, `ggml_vk_create_pipeline_func` reacquires `compile_mutex`,
   flips `compiled = true`, appends to `all_pipelines`, and notifies
   `compile_cv`.

This is the cleanest async-compile handshake in the audited backends.
There is no thread pool — parallelism comes from concurrent dispatch
threads each running `ggml_vk_load_shaders`.

---

## 9. SIMD / GPU Strategy

Vulkan's "SIMD" story is subgroups + cooperative matrices. The backend
queries both at device init.

### 9.1 Subgroup properties

Subgroup support is read from `VkPhysicalDeviceVulkan11Properties`
(lines 6150–6174). The backend records:

* `subgroup_size` (from `subgroup_props.subgroupSize`, line 6135).
* `subgroup_basic`, `subgroup_arithmetic`, `subgroup_shuffle`,
  `subgroup_clustered`, `subgroup_ballot`, `subgroup_vote` — each gated
  on `subgroupSupportedStages & eCompute` and the corresponding
  `subgroupSupportedOperations` bit.

Subgroup operations are core Vulkan 1.1, not the `VK_KHR_shader_subgroup`
extension (which is folded into 1.1). The backend never references
`VK_KHR_shader_subgroup` by name.

### 9.2 Subgroup size control

`VK_EXT_subgroup_size_control` is queried at lines 5973–5974 and
6264–6273. The backend stores `subgroup_min_size`, `subgroup_max_size`,
`subgroup_require_full_support`. When supported, pipelines can request
a specific subgroup size via `vk::PipelineShaderStageRequiredSubgroupSizeCreateInfoEXT`
(line 2781). This is used for matmul pipelines on Intel (line 3946)
where `subgroup_min_size` is preferred, and for flash-attention
pipelines whose tuning params specify a subgroup size.

`subgroup_require_full_support` enables
`vk::PipelineShaderStageCreateFlagBits::eRequireFullSubgroupsEXT` (line
2771), which guarantees that every workgroup contains an integer number
of subgroups.

### 9.3 Cooperative matrix (Tensor Core analogue)

Two cooperative-matrix extensions are supported:

* `VK_KHR_cooperative_matrix` — Khronos cross-vendor. Used on AMD,
  Intel Xe2, NVIDIA. The backend queries all `VkCooperativeMatrixPropertiesKHR`
  shapes (lines 6558–6649) and picks a single `(M, N, K)` tuple that
  supports both f16×f16→f32 and f16×f16→f16 accumulation, with a
  preference for 16×16×16.
* `VK_NV_cooperative_matrix2` — NVIDIA-only, with workgroup-scope
  flexible dimensions, reductions, conversions, tensor addressing.
  Queried at lines 6287–6296 and 6440–6553. Requires
  `VK_KHR_cooperative_matrix` shapes *or* `bufferDeviceAddress`.

Three code paths in flash attention: `FA_SCALAR`, `FA_COOPMAT1`
(KHR_cooperative_matrix), `FA_COOPMAT2` (NV_cooperative_matrix2)
(enum at line 525).

### 9.4 Other GPU features

* `VK_KHR_shader_float16_int8` — required for fp16 compute (line 6555).
* `VK_KHR_16bit_storage` — required for fp16 storage.
* `VK_KHR_shader_bfloat16` — optional, enables bf16.
* `VK_KHR_shader_integer_dot_product` — optional, used for Q8_1 matmul
  fast path.
* `VK_VALVE_shader_mixed_float_dot_product` — optional, fp16-acc-fp32
  dot product for AMD.
* `VK_EXT_shader_float8` + `VK_EXT_shader_ocp_microscaling_types` —
  optional, enables MXFP4 / NVFP4 paths.
* `VK_EXT_shader_64bit_indexing` — optional, enables >2²⁷-byte
  storage buffers (lines 6019–6022, 2803–2814).

### 9.5 Float controls

`shaderRoundingModeRTEFloat16` and `shaderDenormPreserveFloat16` are
read from `VkPhysicalDeviceVulkan12Properties` (lines 6147–6148). When
either is supported, the backend **patches SPIR-V at runtime** in
`ggml_vk_create_pipeline_func` (lines 2642–2712) to inject the
corresponding `OpCapability` + `OpExecutionMode` instructions. This
avoids needing separate shader variants for fp16 denorm behavior.

This is a notable pattern: instead of compiling N shader variants, the
backend compiles one and rewrites the SPIR-V binary in-memory before
`vkCreateShaderModule`. The patcher is a simple opcode walker that
tracks insertion points respecting SPIR-V's required layout order.

### 9.6 Architecture detection

`get_device_architecture` (line 377) infers a coarse architecture enum
(AMD_GCN, AMD_RDNA1/2/3, INTEL_XE1/XE2, NVIDIA_PRE_TURING/TURING) from
vendor ID + subgroup size + shader core properties. This enum drives
matmul tile-size selection (line 4065) and subgroup-size overrides for
specific pipelines. The mapping is heuristic, not authoritative.

---

## 10. Quantization Strategy

The Vulkan backend supports all ggml quant formats via dedicated
dequant shaders (`dequant_q4_0.comp`, `dequant_q4_k.comp`, etc.).
The `pipeline_dequant[GGML_TYPE_COUNT]` table (line 855) holds one
dequant pipeline per dtype; `ggml_vk_get_to_fp16` (line 7372) returns
the right one.

For matmul, the backend has three pipeline families:

* `pipeline_dequant_mul_mat_mat[type]` — dequant weights, multiply by
  fp16/fp32 activation, accumulate in fp32 or fp16. Indexed by
  quant type and accumulator precision (`vk_matmul_pipeline2` with
  `f32acc` and `f16acc` variants, line 256).
* `pipeline_dequant_mul_mat_vec_f32_f16_f32[wg][type][cols]` — GEMV
  path with dequant on-the-fly. Two workgroup sizes (subgroup, large)
  and 8 column variants for batched GEMV.
* `pipeline_dequant_mul_mat_vec_q8_1_f32[wg][type][cols]` — same as
  above but with Q8_1-quantized activations (integer dot product path
  when `VK_KHR_shader_integer_dot_product` is supported).

The dispatch picks l/m/s tile variants based on `device->mul_mat_l/m/s[type]`
flags, which are pre-computed during `ggml_vk_load_shaders` based on
shared-memory availability (lines 4086–4134).

Quantized activations are produced on-the-fly inside the matmul shader
via `pipeline_quantize_q8_1_x4` (line 853) for the split-K reduction
path. There is no separate "convert src1 to Q8_1" pass — the
quantization is fused into the matmul dispatch.

The Q8_1 mmq path uses a *different* shared-memory layout from the float
matmul shaders, hence the separate `mul_mat_l_int/m_int/s_int` flag
arrays (line 823, comment at 823–827). This is a quant-format-specific
optimization.

---

## 11. Correctness Analysis

Static analysis found the following correctness-relevant properties.

### 11.1 Memory hazard detection between nodes

`ggml_vk_build_graph` (lines 14837–14917) maintains
`unsynced_nodes_written` and `unsynced_nodes_read` lists and inserts a
`pipelineBarrier` only when a new node overlaps an unsynced node. This
is a per-node, per-tensor overlap check, not a global barrier. The
overlap test (line 14859) is:

```
(o_base <= n_base && n_base < o_base + o_size) ||
(n_base <= o_base && o_base < n_base + n_size)
```

This is correct for half-open ranges but assumes no wrap-around (safe,
since `vk_ptr_base = 0x1000` and pointers are positive).

When in doubt, the code calls `ggml_vk_sync_buffers` (line 3416), which
emits a `vkCmdPipelineBarrier` with shader-read+shader-write+transfer-
read+transfer-write on both sides. This is conservative — it serializes
more than strictly necessary — but correct.

### 11.2 Fusion aliasing checks

The fusion pass (lines 16778–16836) checks that no fused op's
destination aliases a still-live source of an earlier fused op, with
two exceptions:

1. `is_topk_moe_single_row` — `TOPK_MOE` with `nrows(src0) == 1` is
   safe because the shader loads all src values into registers before
   any store.
2. `op_srcs_fused_elementwise[k] == true` — for elementwise fused ops,
   in-place writes are allowed because each thread reads its input
   before writing its output.

`ggml_vk_tensors_overlap` (not shown; called at line 16814) is the
underlying overlap test. The `op_srcs_fused_elementwise` mask is set
per-fusion-pattern at lines 16645–16736.

### 11.3 Pipeline barriers

`ggml_vk_sync_buffers` (line 3416) emits a single
`vkCmdPipelineBarrier` with `srcStageMask == dstStageMask ==
ctx->p->q->stage_flags` (compute+transfer for compute queue,
transfer-only for transfer queue). The barrier's
`BufferMemoryBarrier` array is empty (`{},{}`) — only a global memory
dependency is declared, no buffer-range-specific barrier.

This works because Vulkan memory dependencies are scoped by access
flags, not by buffer ranges. But it means *every* pending buffer write
is made visible, not just the overlapping ones. This is correct but
conservative.

### 11.4 Event + timeline semaphore race

`ggml_backend_vk_event_record` (line 17198) records `vkCmdSetEvent` and
signals a timeline semaphore on the same command buffer submit. The
event is used by `ggml_backend_vk_event_wait` (line 17236) on a
*different* command buffer in the same queue — the wait records
`vkCmdWaitEvents`. The timeline semaphore is used by
`event_synchronize` (line 18107) for host-side blocking.

The comment at line 1131 explains: "Polling on an event for
event_synchronize wouldn't be sufficient to wait for command buffers to
complete, and would lead to validation errors." The dual representation
(event for in-queue ordering, timeline semaphore for host wait) is
correct.

### 11.5 Pinned memory lookup race

`ggml_vk_host_get` (line 7787) holds a `shared_lock` on
`pinned_memory_mutex`. `ggml_vk_host_malloc` and `ggml_vk_host_free`
hold an exclusive lock. This is correct — readers don't see a torn
vector — but the linear scan is O(N).

### 11.6 Float-control SPIR-V patching

The patcher at lines 2642–2712 inserts `OpCapability` and
`OpExecutionMode` instructions into a SPIR-V module. It uses
`std::vector::insert` at tracked positions, working from latest to
earliest so earlier indices remain valid (line 2677). This is correct
because the patcher respects SPIR-V's required layout order
(header → capabilities → extensions → entry points → execution modes).

If the patcher fails to find an insertion point (e.g., a malformed
shader with no `OpEntryPoint`), `entry_point_id` stays 0 and the
execution-mode insert at `exec_insert_pos = pos` is at the start of
the file, which would be invalid. The code does not assert
`entry_point_id != 0` before using it.

### 11.7 Architecture-specific determinism

* `nrows = 2` ARM I8MM path: not applicable (CPU only).
* AMD GCN vs RDNA subgroup size: matmul pipelines pick different
  subgroup sizes per architecture (lines 4065–4077). Two runs on
  different architectures will produce different ULP-level results.
* coopmat1 vs coopmat2 vs scalar flash attention: three code paths,
  each with its own reduction order. Per-architecture determinism only.

---

## 12. Optimization Analysis

### 12.1 Identified optimizations

| Optimization                          | Where                                | Notes |
| ------------------------------------- | ------------------------------------ | ----- |
| Lazy pipeline compilation             | `ggml_vk_load_shaders:3938`          | Pipelines compiled on first dispatch; mutex+condvar handshake allows parallel compiles. |
| SPIR-V float-control patching         | `ggml_vk_create_pipeline_func:2642`  | One shader variant, runtime-patched for fp16 denorm/RTE. Avoids N× shader explosion. |
| SPIR-V decode-vector stripping        | `ggml_vk_strip_decode_vector:2428`  | Strips `SPV_NV_cooperative_matrix_decode_vector` ops when driver only supports coopmat2. |
| SPIR-V BK-loop unroll toggle          | `ggml_vk_roll_bk_loop:2556`          | Replaces Unroll with DontUnroll on Apple M1/M2 for mul_mm. |
| Descriptor pool geometric growth      | `ggml_pipeline_allocate_descriptor_sets:2898` | Grows by 50% (3/2) per allocation, in chunks of 256. |
| Per-context descriptor pool           | `ctx->descriptor_pools:2206`         | Pool survives across graph computes; only index resets. Avoids pool reset overhead. |
| Pinned-memory fast path               | `ggml_vk_host_malloc:7739`           | Pinned host allocations registered; `buffer_write_2d_async` skips staging when src is pinned. |
| Prealloc-y conversion cache           | `prealloc_y_last_pipeline_used:2184` | One-entry cache; skips re-dequant if same tensor+pipeline as previous graph. |
| Plan-time fusion                      | `ggml_backend_vk_graph_compute:16640` | ~15 patterns including a 10-op TOPK_MOE subgraph. Detection is plan-time, not execution-time. |
| Batched submit by flops               | `ggml_backend_vk_graph_compute:16606` | Submits every ~200 GFLOP (or 100 nodes) to overlap CPU recording with GPU execution. |
| Almost-ready fence                    | `ggml_vk_wait_for_fence:2386`        | Signals a fence at <20% remaining so `synchronize` can sleep instead of spin. |
| Dual-queue async transfer             | `ggml_vk_get_transfer_ctx:7948`      | Transfer-queue copies overlap with compute-queue execution via timeline semaphore. |
| Internally-synchronized queues        | `vk_queue_handle_unsynchronized:324` | `VK_KHR_internally_synchronized_queues` skips host mutex on `submit`. |
| Node-reorder graph optimizer          | `ggml_vk_graph_optimize:16941`       | Reorders nodes to expose fusion patterns; preserves fusion-pattern atomicity. |
| Per-op workgroup specialization       | `ggml_vk_load_shaders:3938`          | l/m/s tile variants per matmul shape; per-architecture subgroup size. |
| Async ADD→RMS partials fusion         | `do_add_rms_partials:14822`          | ADD writes squared sums into a 1 KB partials buffer; RMS_NORM consumes them, avoiding a full pass. |
| UMA host-visible buffer preference    | `ggml_vk_create_buffer_device:3367`  | On UMA, host-visible device-local memory enables zero-copy tensor borrowing. |

### 12.2 Optimizations *not* present

* **No `VkPipelineCache`.** `vkCreateComputePipelines` is called with
  `VK_NULL_HANDLE` as the cache (line 2817). Cold-start cost is paid
  every process launch.
* **No on-disk shader cache.** The `vulkan-shaders-gen` tool embeds
  SPIR-V into the binary at build time, but there is no runtime cache
  of compiled `VkPipeline` objects.
* **No `VK_KHR_push_descriptor`** use. Every dispatch calls
  `vkUpdateDescriptorSets` (line 7897), which is a host-side write.
  Push descriptors would skip the write and the descriptor-set
  allocation entirely.
* **No `vkCmdDispatchBase`** use. Batch dispatch is encoded via push
  constants (`base_work_group_z`), which means the GPU can't early-out
  a partially-filled trailing batch via `VK_KHR_device_group` semantics.
  This is fine for single-device but limits multi-GPU scaling.
* **No descriptor set reuse via `vkResetDescriptorPool`.** The pool
  grows monotonically per context; sets are taken in order and the
  index resets only on `graph_cleanup` (line 15401). No
  `VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT` flag.
* **No multi-queue parallelism for compute.** Only one compute queue
  is used. Independent subgraphs cannot run on separate queues.
* **Pinned memory lookup is O(N) linear scan.** No hash table.
* **Prealloc-y cache is one entry.** No LRU; a 2-element graph that
  alternates tensors thrashes the cache.

---

## 13. Architectural Strengths

1. **Lazy + parallel pipeline compilation.** The mutex + condvar
   handshake at lines 728–729, 3977, 5881–5900 lets multiple threads
   compile different pipelines concurrently without holding the lock
   during `vkCreateComputePipelines`. This is the cleanest async-compile
   design in the audited backends.

2. **SPIR-V patching layer.** Instead of shipping N shader variants per
   feature combination (float controls, decode-vector, BK-loop unroll),
   the backend patches SPIR-V at runtime. This keeps the shader library
   small and lets feature decisions be made per-device at runtime.
   GwenLand should adopt this pattern.

3. **Plan-time fusion with anti-aliasing rollback.** The fusion pass
   (lines 16640–16836) detects ~15 patterns *before* dispatching, sets
   `num_additional_fused_ops`, then validates that no fused op aliases
   a still-live source. If validation fails, fusion is rolled back to
   `num_additional_fused_ops = 0`. This is the most sophisticated
   fusion machinery in the audited backends.

4. **Dual-queue async with timeline semaphores.** The
   compute-queue + transfer-queue split, coordinated by a single
   timeline semaphore per context (line 2201), is a textbook Vulkan
   async design. The fallback to single-queue on devices without a
   dedicated transfer family is graceful.

5. **Internally-synchronized-queues fast path.** When
   `VK_KHR_internally_synchronized_queues` is available, the queue
   handle's `submit` skips the host mutex (line 324). This is a clean
   way to use vendor extensions without polluting the call sites.

6. **Per-context descriptor pool with geometric growth.** Growing by
   3/2 per allocation (line 2908) amortizes allocation cost. The pool
  survives across graphs, so steady-state descriptor allocation is
  just `descriptor_set_idx++`.

7. **`vk_event` dual representation.** Using `vk::Event` for in-queue
   ordering and a timeline semaphore for host-side wait (line 1135) is
   correct and avoids the validation errors that pure-event polling
   would cause.

8. **Architecture detection drives tile size.** `get_device_architecture`
   (line 377) infers a coarse enum that drives matmul tile selection
   (line 4065). This lets the backend tune for AMD GCN vs RDNA vs
   Intel Xe vs NVIDIA Turing without per-shader conditionals.

9. **Almost-ready fence.** Signaling a fence at <20% remaining so the
   final `synchronize` can `waitForFences` (sleep) instead of
   spin-polling (line 2386) is a clever latency optimization.

10. **`flops_per_submit` adaptive batching.** Submitting every ~200
    GFLOP, scaled to `last_total_flops / 40` so small graphs submit
    earlier, is a pragmatic way to overlap CPU recording with GPU
    execution without requiring a full graph executor.

---

## 14. Architectural Weaknesses

### W1 — No `VkPipelineCache`

**Evidence**: `ggml_vk_create_pipeline_func` line 2817:
`device->device.createComputePipeline(VK_NULL_HANDLE, ...)`.

**Impact**: Every process launch re-compiles every pipeline from
SPIR-V. For a model with 50+ unique pipeline variants, this is
seconds of cold-start time. `VkPipelineCache` is *the* standard
Vulkan mechanism for this — it's free, it's serializable to disk,
and it's transparent.

**Why it's hard to fix**: Pipeline cache serialization requires a
stable on-disk location and careful invalidation when driver/shader
versions change. The backend currently has no persistent state.

### W2 — All pipelines share one descriptor set layout

**Evidence**: Line 6756–6769, every pipeline is created with
`device->dsl`. The layout declares `MAX_PARAMETER_COUNT = 12`
storage-buffer bindings per set (line 6759).

**Impact**: An elementwise op that uses 2 buffers still allocates a
descriptor set with 12 bindings. The unused bindings cost descriptor
memory (10 unused `VkDescriptorBufferInfo` entries per set, though the
backend only writes `pipeline->parameter_count` of them). On devices
with limited descriptor memory this could force earlier pool growth.

The tradeoff is real: a shared layout means `vkCmdBindDescriptorSets`
can switch pipelines without re-binding descriptor sets, and the
backend can pre-allocate one descriptor pool that serves every
pipeline. The cost is descriptor memory overhead per set.

### W3 — Spin-poll fence wait

**Evidence**: `ggml_vk_wait_for_fence` lines 2397–2413 spin-polls
`getFenceStatus` with 100 `YIELD()` calls per iteration.

**Impact**: On the hot path (every `synchronize` call after the
almost-ready fence is signaled), the CPU burns cycles spinning. The
comment acknowledges this is intentional ("Hopefully the CPU can sleep
during this wait" — line 2388 — referring to the almost-ready fence,
not the final spin). For workloads where `synchronize` is called
frequently (e.g., interactive inference), this is wasted CPU.

### W4 — Pinned memory O(N) linear scan

**Evidence**: `ggml_vk_host_get` line 7791 loops over
`device->pinned_memory` with no index.

**Impact**: For a workload with many small pinned regions (e.g., a
model with many small KV caches pinned individually), every
`buffer_write_2d_async` call scans the whole list. A hash table or
interval tree would be O(1)/O(log N).

### W5 — Single-entry prealloc-y cache

**Evidence**: `prealloc_y_last_pipeline_used` (line 2184) is a single
`vk_pipeline_struct*` + `const ggml_tensor*`.

**Impact**: A 2-element graph that alternates between two source
tensors thrashes the cache, re-dequantizing on every iteration. An
LRU of size 4–8 would capture typical attention patterns.

### W6 — `ggml_vk_sync_buffers` emits global barriers

**Evidence**: Line 3425 emits a `vkCmdPipelineBarrier` with empty
`BufferMemoryBarrier` arrays. Only `srcAccessMask`/`dstAccessMask`
are set (lines 3430–3431).

**Impact**: Every pending buffer write is made visible, not just the
overlapping ones. The overlap analysis at lines 14837–14917 decides
*when* to call `sync_buffers`, but the barrier itself is broader than
necessary. Per-buffer barriers would be more precise but require
tracking every buffer touched since the last barrier.

### W7 — `supports_op` doesn't consult shader availability

**Evidence**: `ggml_backend_vk_device_supports_op` (line 17459) checks
tensor sizes, dtype support, and a few op-specific constraints, but
does not check whether a pipeline exists for the op. A
`ggml_vk_op_get_pipeline` returning `nullptr` causes `GGML_ABORT`
inside `ggml_vk_op_f32` (line 11649).

**Impact**: The scheduler may offload an op to Vulkan that Vulkan
can't actually run, causing a runtime abort instead of a graceful
fallback. This is a contract gap between `supports_op` and
`op_get_pipeline`.

### W8 — No multi-queue compute parallelism

**Evidence**: Only one compute queue is created (line 6680). The
transfer queue exists for async copies, but no second compute queue
is used for parallel subgraph execution.

**Impact**: Independent subgraphs run serially on the same queue.
Vulkan's queue-level parallelism is unused for compute. This is a
missed opportunity, though driver support for multiple compute queues
is uneven.

### W9 — `GGML_VK_MAX_DEVICES` is a compile-time constant

**Evidence**: `vk_instance.devices[GGML_VK_MAX_DEVICES]` (line 2360).
`GGML_VK_MAX_DEVICES` is typically 16.

**Impact**: A system with >16 Vulkan devices (e.g., a render farm
with many GPUs) silently ignores devices beyond the limit. A
`std::vector` would be more robust.

### W10 — Hardcoded `flops_per_submit` heuristic

**Evidence**: Lines 16606–16618. 200 GFLOP cap, scaled by
`last_total_flops / 40`, doubled after first 3 submits. The "40" and
"200 GFLOP" are magic numbers.

**Impact**: Suboptimal for shapes that don't fit the heuristic. An
autotuner or a feedback-based policy would help.

### W11 — `get_device_architecture` is heuristic

**Evidence**: Lines 377–495. Architecture is inferred from vendor ID +
subgroup size + shader core properties + integer dot product
acceleration. The mapping is documented but not authoritative.

**Impact**: A new GPU generation that doesn't match the heuristic
(e.g., AMD RDNA4) falls into `OTHER` and gets the default tile
sizes, which may be suboptimal.

### W12 — `event_record` always submits the command buffer

**Evidence**: Line 17229 `ggml_vk_submit(compute_ctx, {})` is called
inside `event_record`, ending the current command buffer and
submitting it.

**Impact**: Recording an event forces a command-buffer boundary. If a
caller records many events in quick succession (e.g., for fine-grained
profiling), each one pays a submit + fence round-trip. There is no
"record event without submitting" path.

---

## 15. GwenLand Mapping

| GwenLand module | ADOPT / ADAPT / REJECT / MONITOR / DEFER | What | Reasoning |
| --------------- | ---------------------------------------- | ---- | --------- |
| `glvulkan`      | **ADOPT** | Lazy + parallel pipeline compilation (mutex + condvar handshake) | Cleanest async-compile design audited. Lets multiple threads compile different pipelines without holding a lock during `vkCreateComputePipelines`. |
| `glvulkan`      | **ADOPT** | SPIR-V patching for float controls / decode-vector stripping | One shader variant, runtime-patched. Avoids shader explosion. |
| `glvulkan`      | **ADOPT** | Per-context descriptor pool with geometric growth | Amortizes allocation; survives across graphs. |
| `glvulkan`      | **ADOPT** | `vk_event` dual representation (event + timeline semaphore) | Correct; avoids validation errors. |
| `glvulkan`      | **ADOPT** | Dual-queue (compute + transfer) with timeline semaphore | Textbook Vulkan async design. |
| `glvulkan`      | **ADOPT** | Internally-synchronized-queues fast path | Clean vendor-extension use without polluting call sites. |
| `glvulkan`      | **ADOPT** | Architecture detection drives tile size | Lets one binary tune for AMD/Intel/NVIDIA without per-shader conditionals. |
| `glvulkan`      | **ADAPT** | Shared descriptor set layout (12 bindings) | Keep the shared-layout idea but consider a second layout for small-arity ops to save descriptor memory. |
| `glvulkan`      | **ADAPT** | `flops_per_submit` adaptive batching | Keep the idea, make the policy pluggable. |
| `glvulkan`      | **ADAPT** | Almost-ready fence + spin-poll | Keep the almost-ready fence; replace the spin-poll with `waitForFences` on systems where the syscall overhead is acceptable. |
| `glvulkan`      | **REJECT**| Absence of `VkPipelineCache` | GwenLand must use `VkPipelineCache` with on-disk persistence. Free cold-start win. |
| `glvulkan`      | **REJECT**| `GGML_VK_MAX_DEVICES` fixed array | Use `std::vector` for device list. |
| `glvulkan`      | **ADAPT** | Pinned memory O(N) scan | Replace with hash table keyed on (page-aligned) pointer. |
| `glvulkan`      | **ADAPT** | Single-entry prealloc-y cache | LRU of size 4–8. |
| `glvulkan`      | **ADAPT** | `ggml_vk_sync_buffers` global barriers | Add per-buffer `BufferMemoryBarrier` arrays when the unsynced list is small. |
| `glvulkan`      | **ADOPT** | `supports_op` consults shader availability | Make `supports_op` return false if no pipeline exists, so the scheduler falls back gracefully. |
| `GATE`          | **ADOPT** | Plan-time fusion with anti-aliasing rollback | The ~15-pattern fusion pass is the most sophisticated audited. |
| `GATE`          | **ADOPT** | Node-reorder graph optimizer | Reorders nodes to expose fusion patterns; preserves fusion atomicity. |
| `GATE`          | **ADOPT** | `flops_per_submit` adaptive batching | Overlap CPU recording with GPU execution. |
| `GATE`          | **MONITOR**| `TOPK_MOE` fusion patterns | Watch whether these remain competitive as MoE models evolve. |
| `GATE`          | **DEFER** | `event_record`-forces-submit | Defer until GwenLand has a use case for fine-grained event recording. |

---

## 16. Recommendations

### R1 — ADOPT lazy + parallel pipeline compilation
**Priority:** Critical
**Difficulty:** M
**Dependencies:** none
GwenLand's `glvulkan` should replicate the mutex + condvar handshake at lines 728–729, 3977, 5881–5900. The key insight is that the compile itself runs without holding the lock, so different pipelines compile in parallel. The lock is only held to claim a pipeline (set `compile_pending`) and to flip `compiled = true` at the end.

### R2 — ADOPT `VkPipelineCache` with on-disk persistence
**Priority:** Critical
**Difficulty:** S
**Dependencies:** R1
GwenLand must use `vkCreatePipelineCache` and serialize the cache to disk between runs. The cache should be invalidated when the driver version, shader SPIR-V hash, or pipeline-creation parameters change. This is a free cold-start win that llama.cpp leaves on the table.

### R3 — ADOPT SPIR-V patching layer
**Priority:** High
**Difficulty:** L
**Dependencies:** none
Replicate the SPIR-V walker at lines 2642–2712 and the decode-vector stripper at lines 2428–2546. This lets GwenLand ship one shader variant per logical pipeline and apply per-device feature decisions at runtime.

### R4 — ADOPT plan-time fusion with anti-aliasing rollback
**Priority:** High
**Difficulty:** L
**Dependencies:** GATE design
Replicate the fusion detection at lines 16640–16774 and the aliasing rollback at lines 16778–16836. GwenLand's GATE should detect fusion patterns once at plan time, mark fused nodes, and validate that no fused op aliases a still-live source before committing.

### R5 — ADOPT dual-queue async with timeline semaphores
**Priority:** High
**Difficulty:** M
**Dependencies:** R1
Replicate the compute-queue + transfer-queue split coordinated by a single timeline semaphore per context (lines 2201, 7940–7943, 15956–15972). The fallback to single-queue on devices without a dedicated transfer family should be graceful.

### R6 — ADOPT per-context descriptor pool with geometric growth
**Priority:** High
**Difficulty:** S
**Dependencies:** R1
Replicate the 3/2-growth descriptor pool at lines 2898–2934. The pool survives across graphs; only the index resets on `graph_cleanup`.

### R7 — ADOPT `vk_event` dual representation
**Priority:** Medium
**Difficulty:** S
**Dependencies:** R5
Replicate the `vk_event` struct (line 1135) with `vk::Event` for in-queue ordering and a timeline semaphore for host-side wait. The comment at line 1131 explains why pure-event polling is insufficient.

### R8 — ADOPT `flops_per_submit` adaptive batching
**Priority:** Medium
**Difficulty:** S
**Dependencies:** R4
Replicate the adaptive submit policy at lines 16606–16618. The key idea is to submit roughly every `last_total_flops / 40` flops (capped at 200 GFLOP), so the GPU starts executing while the CPU is still recording the next batch.

### R9 — ADAPT pinned memory lookup to hash table
**Priority:** Medium
**Difficulty:** S
**Dependencies:** R6
Replace the O(N) linear scan at `ggml_vk_host_get` (line 7787) with a hash table keyed on page-aligned host pointer. The current scan is fine for <16 pinned regions but doesn't scale.

### R10 — ADAPT prealloc-y cache to LRU
**Priority:** Medium
**Difficulty:** S
**Dependencies:** R6
Replace the single-entry cache at line 2184 with an LRU of size 4–8. Captures typical attention patterns that alternate between 2–4 source tensors.

### R11 — ADOPT architecture detection
**Priority:** Medium
**Difficulty:** M
**Dependencies:** R1
Replicate `get_device_architecture` (line 377). The coarse enum drives tile-size selection without per-shader conditionals. GwenLand should extend the enum as new GPU generations ship.

### R12 — ADOPT node-reorder graph optimizer
**Priority:** Medium
**Difficulty:** M
**Dependencies:** R4
Replicate `ggml_vk_graph_optimize` (line 16941). The reorder pass exposes fusion patterns by pulling independent nodes earlier, while preserving fusion-pattern atomicity via `keep_pattern`.

### R13 — ADAPT `supports_op` to consult shader availability
**Priority:** Medium
**Difficulty:** S
**Dependencies:** R1
Make `supports_op` return false if no pipeline exists for the op, so the scheduler falls back gracefully instead of `GGML_ABORT` at runtime (line 11649).

### R14 — REJECT `GGML_VK_MAX_DEVICES` fixed array
**Priority:** Low
**Difficulty:** XS
**Dependencies:** none
Use `std::vector<vk_device>` instead of `vk_device devices[GGML_VK_MAX_DEVICES]` (line 2360).

---

## 17. Findings

### Finding ARTX18-F01

```
Finding ID:           ARTX18-F01
Category:             BACKEND_DESIGN
Engine:               Vulkan
Component:            Instance creation
Source File:          ggml/src/ggml-vulkan/ggml-vulkan.cpp
Function:             ggml_vk_instance_init
Lines:                7078-7326
Summary:              Vulkan instance is created with VK_KHRONOS_validation layer
                      and VK_EXT_layer_settings only when GGML_VULKAN_VALIDATE is set;
                      debug_utils requires GGML_VK_DEBUG_MARKERS env var.
Observation:          The instance creation path (lines 7104-7137) conditionally
                      enables the validation layer with best-practices validation,
                      but only if `ggml_vk_instance_layer_settings_available()`
                      returns true (which requires GGML_VULKAN_VALIDATE to be set,
                      per line 18276). On macOS, VK_KHR_portability_enumeration is
                      enabled for MoltenVK. No other instance extensions are
                      requested (no VK_KHR_external_memory_capabilities, no
                      VK_KHR_get_physical_device_properties2 — both are core in 1.1+).
                      The minimum API version is 1.2 (line 7089).
Evidence:             ggml-vulkan.cpp:7078-7137 (instance creation),
                      18275-18291 (layer settings gate).
Architectural Impact: The instance is minimal: validation is opt-in, debug markers
                      are opt-in, no portability-related extensions outside macOS.
                      This is the right default for a production backend.
Correctness Impact:   None. Instance creation is correct.
Optimization Type:    None.
GwenLand Target:      glvulkan
Recommendation:       ADOPT. Same minimal-instance policy for GwenLand: validation
                      and debug markers opt-in via env var, Vulkan 1.2 minimum.
Priority:             Medium
Difficulty:           S
Dependencies:         none
Confidence:           High
```

### Finding ARTX18-F02

```
Finding ID:           ARTX18-F02
Category:             BACKEND_DESIGN
Engine:               Vulkan
Component:            Device selection
Source File:          ggml/src/ggml-vulkan/ggml-vulkan.cpp
Function:             ggml_vk_instance_init (device enumeration)
Lines:                7167-7307
Summary:              Device selection enumerates all physical devices, prefers
                      discrete/integrated GPUs, deduplicates by UUID, and picks
                      the higher-priority driver when the same GPU appears twice.
Observation:          Lines 7194-7202 keep only eDiscreteGpu and eIntegratedGpu
                      devices that pass `ggml_vk_device_is_supported` (which
                      requires `storageBuffer16BitAccess`, line 18335). When two
                      physical devices share the same deviceUUID (e.g., RADV +
                      AMDVLK on the same GPU), the code (lines 7209-7289) keeps
                      the one with the higher-priority driver based on a per-vendor
                      driver priority map (lines 7248-7269). Falls back to first
                      non-CPU device if no discrete/integrated GPUs found
                      (lines 7294-7301).
Evidence:             ggml-vulkan.cpp:7167-7307 (enumeration + dedup),
                      18324-18336 (is_supported gate).
Architectural Impact: Handles the common multi-driver-on-one-GPU case correctly.
                      The driver priority map is hardcoded per vendor.
Correctness Impact:   None.
Optimization Type:    Driver-priority-based device selection.
GwenLand Target:      glvulkan
Recommendation:       ADOPT. Same dedup + driver-priority logic for GwenLand.
Priority:             Medium
Difficulty:           M
Dependencies:         none
Confidence:           High
```

### Finding ARTX18-F03

```
Finding ID:           ARTX18-F03
Category:             BACKEND_DESIGN
Engine:               Vulkan
Component:            Queue family selection
Source File:          ggml/src/ggml-vulkan/ggml-vulkan.cpp
Function:             ggml_vk_find_queue_family_index, ggml_vk_get_device
Lines:                3024-3068, 6198-6208
Summary:              Compute queue family prefers non-graphics; transfer queue
                      family prefers non-compute non-graphics; falls back to
                      reusing compute family if no dedicated transfer family.
Observation:          `ggml_vk_find_queue_family_index` (line 3024) takes
                      `required`, `avoid`, and `compute_index` parameters and
                      tries four fallback strategies: (1) required + avoid,
                      (2) required only, (3) required (ignoring compute_index
                      exclusion), (4) required (ignoring min_num_queues), and
                      finally (5) reuse compute_index. The compute family
                      (line 6204) requires eCompute, avoids eGraphics unless
                      GGML_VK_ALLOW_GRAPHICS_QUEUE is set. The transfer family
                      (line 6205) requires eTransfer, avoids eCompute + eGraphics.
                      `single_queue` is set when both families are the same and
                      the family has only one queue (line 6208).
Evidence:             ggml-vulkan.cpp:3024-3068 (finder), 6204-6208 (selection).
Architectural Impact: Avoids the graphics queue on RADV (cited performance gain).
                      Graceful fallback to single-queue on minimal devices.
Correctness Impact:   None.
Optimization Type:    Queue-family-preference-based selection.
GwenLand Target:      glvulkan
Recommendation:       ADOPT. Same four-fallback strategy for GwenLand.
Priority:             High
Difficulty:           S
Dependencies:         none
Confidence:           High
```

### Finding ARTX18-F04

```
Finding ID:           ARTX18-F04
Category:             MISSING_FEATURE
Engine:               Vulkan
Component:            Pipeline cache
Source File:          ggml/src/ggml-vulkan/ggml-vulkan.cpp
Function:             ggml_vk_create_pipeline_func
Lines:                2816-2822
Summary:              No VkPipelineCache is used; every pipeline is created with
                      VK_NULL_HANDLE as the cache, forcing full recompilation on
                      every process launch.
Observation:          Line 2817 calls `device->device.createComputePipeline(
                      VK_NULL_HANDLE, compute_pipeline_create_info)`. There is no
                      `vkCreatePipelineCache` call anywhere in the file (grep for
                      `createPipelineCache|PipelineCacheCreateInfo` returns no
                      matches). Cold-start cost is paid every process launch:
                      for a model with 50+ unique pipeline variants, this is
                      seconds of latency.
Evidence:             ggml-vulkan.cpp:2817 (VK_NULL_HANDLE cache),
                      2880-2887 (destroy — no cache to destroy).
Architectural Impact: Significant cold-start regression vs. backends that cache
                      pipelines. VkPipelineCache is the standard Vulkan mechanism;
                      it's free, serializable, and transparent.
Correctness Impact:   None.
Optimization Type:    None (absence of optimization).
GwenLand Target:      glvulkan
Recommendation:       REJECT. GwenLand must use VkPipelineCache with on-disk
                      persistence, invalidated on driver/shader version change.
Priority:             Critical
Difficulty:           S
Dependencies:         R1, R2
Confidence:           High
```

### Finding ARTX18-F05

```
Finding ID:           ARTX18-F05
Category:             BACKEND_DESIGN
Engine:               Vulkan
Component:            Descriptor set layout
Source File:          ggml/src/ggml-vulkan/ggml-vulkan.cpp
Function:             ggml_vk_get_device (dsl creation)
Lines:                6756-6769
Summary:              All pipelines share a single descriptor set layout with
                      MAX_PARAMETER_COUNT=12 storage-buffer bindings.
Observation:          Lines 6758-6761 create `MAX_PARAMETER_COUNT` (12) bindings,
                      each `eStorageBuffer` with `eCompute` stage. The layout is
                      stored in `device->dsl` (line 6769) and reused by every
                      pipeline (line 2750). `MAX_PARAMETER_COUNT` is 12 (line 207),
                      driven by the multi-add fusion which needs up to 9 source
                      tensors + 1 dst + 2 partials.
Evidence:             ggml-vulkan.cpp:6756-6769 (layout creation), 2750 (pipeline
                      uses device->dsl), 207 (MAX_PARAMETER_COUNT), 884-885
                      (multi_add pipelines driving the count).
Architectural Impact: One layout = one descriptor pool design = simpler allocation.
                      But every descriptor set allocates 12 binding slots even for
                      2-buffer elementwise ops, wasting descriptor memory.
Correctness Impact:   None.
Optimization Type:    Shared-layout for descriptor set reuse across pipelines.
GwenLand Target:      glvulkan
Recommendation:       ADAPT. Keep the shared-layout idea but consider a second
                      layout for small-arity ops (2-3 bindings) to save descriptor
                      memory on constrained devices.
Priority:             Medium
Difficulty:           M
Dependencies:         R6
Confidence:           High
```

### Finding ARTX18-F06

```
Finding ID:           ARTX18-F06
Category:             BACKEND_DESIGN
Engine:               Vulkan
Component:            Descriptor pool
Source File:          ggml/src/ggml-vulkan/ggml-vulkan.cpp
Function:             ggml_pipeline_allocate_descriptor_sets
Lines:                2898-2934
Summary:              Per-context descriptor pool grows geometrically (3/2) in
                      chunks of VK_DEVICE_DESCRIPTOR_POOL_SIZE=256 sets; pool is
                      not reset between graph computes, only the index resets.
Observation:          Lines 2907-2933 grow the pool by `max(3 * size / 2,
                      required)` sets, allocating new `vk::DescriptorPool` objects
                      in chunks of 256. Each pool is created with 256 sets × 12
                      storage-buffer descriptors (line 2919). The pool is *not*
                      created with VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
                      so individual sets can't be freed — only the whole pool can
                      be reset (which never happens; the index just resets on
                      graph_cleanup at line 15401). `descriptor_set_idx` is
                      incremented per dispatch (line 7895) and reset on cleanup.
Evidence:             ggml-vulkan.cpp:2898-2934 (allocation), 7895 (per-dispatch
                      index bump), 15401 (reset on cleanup), 187 (pool size
                      constant).
Architectural Impact: Steady-state descriptor allocation is just an index bump.
                      No per-graph pool reset. Memory grows monotonically per
                      context until the context is freed.
Correctness Impact:   None.
Optimization Type:    Geometric-growth descriptor pool.
GwenLand Target:      glvulkan
Recommendation:       ADOPT. Same geometric-growth + per-context-pool design.
Priority:             High
Difficulty:           S
Dependencies:         R6
Confidence:           High
```

### Finding ARTX18-F07

```
Finding ID:           ARTX18-F07
Category:             BACKEND_DESIGN
Engine:               Vulkan
Component:            Pipeline lazy compilation
Source File:          ggml/src/ggml-vulkan/ggml-vulkan.cpp
Function:             ggml_vk_load_shaders, ggml_vk_create_pipeline_func
Lines:                3938-5901, 2871-2878
Summary:              Pipelines are compiled lazily on first dispatch via a
                      mutex+condvar handshake that allows parallel compiles
                      across threads without holding the lock during
                      vkCreateComputePipelines.
Observation:          `ggml_vk_load_shaders` (line 3938) walks the pipeline list
                      under `compile_mutex` (line 3977). For the requested-but-
                      uncompiled pipeline, it claims it (sets `compile_pending`)
                      and remembers a `CompileTask` (lines 4213-4223). It then
                      drops the lock (line 5882) and runs
                      `ggml_vk_create_pipeline_func` (the actual
                      `vkCreateComputePipelines` call) without holding any lock.
                      At the end, `create_pipeline_func` reacquires
                      `compile_mutex`, flips `compiled = true`, appends to
                      `all_pipelines`, and notifies `compile_cv` (lines 2872-2877).
                      If another thread needs the same pipeline, it waits on
                      `compile_cv` (lines 5895-5900).
Evidence:             ggml-vulkan.cpp:3977 (lock), 4213-4223 (claim),
                      5882 (unlock), 5886-5892 (compile without lock),
                      2872-2877 (flip + notify), 5895-5900 (wait).
Architectural Impact: Cleanest async-compile handshake in the audited backends.
                      No thread pool — parallelism comes from concurrent dispatch
                      threads each running `load_shaders`.
Correctness Impact:   None.
Optimization Type:    Asynchronous execution (parallel pipeline compilation).
GwenLand Target:      glvulkan
Recommendation:       ADOPT. Same mutex+condvar handshake for GwenLand.
Priority:             Critical
Difficulty:           M
Dependencies:         none
Confidence:           High
```

### Finding ARTX18-F08

```
Finding ID:           ARTX18-F08
Category:             GPU_KERNEL
Engine:               Vulkan
Component:            SPIR-V patching
Source File:          ggml/src/ggml-vulkan/ggml-vulkan.cpp
Function:             ggml_vk_create_pipeline_func (float controls),
                      ggml_vk_strip_decode_vector, ggml_vk_roll_bk_loop
Lines:                2642-2712, 2428-2546, 2556-2625
Summary:              SPIR-V shader modules are patched at runtime to inject
                      float-control capabilities/execution modes, strip
                      SPV_NV_cooperative_matrix_decode_vector ops, and toggle
                      BK-loop unrolling on Apple M1/M2.
Observation:          The float-controls patcher (lines 2642-2712) walks the
                      SPIR-V binary, tracks insertion points for OpCapability,
                      OpExtension, and OpExecutionMode respecting layout order,
                      and injects RoundingModeRTE+DenormPreserve for fp16 when
                      the device supports them. The decode-vector stripper
                      (lines 2428-2546) removes SPV_NV_cooperative_matrix_decode_vector
                      OpExtension, OpCapability, and the DecodeVectorFunc operand
                      from OpCooperativeMatrixLoadTensorNV instructions when the
                      driver only supports coopmat2. The BK-loop roller
                      (lines 2556-2625) replaces Unroll with DontUnroll on Asahi
                      Linux for matmul shaders.
Evidence:             ggml-vulkan.cpp:2642-2712 (float controls),
                      2428-2546 (decode vector strip), 2556-2625 (BK loop),
                      2742 (patch applied before createShaderModule).
Architectural Impact: One shader variant per logical pipeline, runtime-patched
                      for per-device feature decisions. Avoids shader explosion.
Correctness Impact:   The patchers assume well-formed SPIR-V. If a shader has no
                      OpEntryPoint, the float-controls patcher would insert at
                      position 5 (header end), which may be invalid. No assertion
                      guards `entry_point_id != 0` before use (line 2681).
Optimization Type:    None (binary patching, not a runtime optimization per se).
GwenLand Target:      glvulkan
Recommendation:       ADOPT. Same SPIR-V patching layer for GwenLand. Add an
                      assertion that `entry_point_id != 0` before using it.
Priority:             High
Difficulty:           L
Dependencies:         R3
Confidence:           High
```

### Finding ARTX18-F09

```
Finding ID:           ARTX18-F09
Category:             EXECUTION_GRAPH
Engine:               Vulkan
Component:            Graph compute
Source File:          ggml/src/ggml-vulkan/ggml-vulkan.cpp
Function:             ggml_backend_vk_graph_compute
Lines:                16525-16938
Summary:              Graph executor traverses cgraph nodes, applies plan-time
                      fusion (~15 patterns), builds one command buffer per batch,
                      and submits batches adaptively based on accumulated flops.
Observation:          The executor (line 16525) pre-allocates scratch buffers,
                      computes `flops_per_submit = min(200 GFLOP, last_total_flops/40)`
                      (lines 16606-16618), then for each node runs fusion
                      detection (lines 16640-16774), anti-aliasing validation
                      (lines 16778-16836), and `ggml_vk_build_graph` (line 16845).
                      Submit is triggered when `>=100` nodes accumulated, `>=
                      flops_per_submit` flops accumulated, last node reached, or
                      almost-ready fence not yet signaled (lines 16838-16843).
                      `flops_per_submit` doubles after each of the first 3 submits
                      (line 16877) to reduce submit overhead as the graph
                      progresses.
Evidence:             ggml-vulkan.cpp:16525-16938 (full function), 16640-16774
                      (fusion), 16778-16836 (anti-aliasing), 16838-16843 (submit
                      decision).
Architectural Impact: Plan-time fusion + adaptive batching = sophisticated graph
                      executor. The most advanced in the audited backends.
Correctness Impact:   None.
Optimization Type:    Kernel fusion + asynchronous execution (batched submit).
GwenLand Target:      GATE, glvulkan
Recommendation:       ADOPT. Same plan-time fusion + adaptive batching for GATE.
Priority:             Critical
Difficulty:           L
Dependencies:         R4, R8
Confidence:           High
```

### Finding ARTX18-F10

```
Finding ID:           ARTX18-F10
Category:             EXECUTION_GRAPH
Engine:               Vulkan
Component:            Graph optimizer (node reorder)
Source File:          ggml/src/ggml-vulkan/ggml-vulkan.cpp
Function:             ggml_vk_graph_optimize
Lines:                16941-17196
Summary:              Plan-time graph reorder pass pulls independent nodes
                      earlier to expose fusion patterns, while preserving
                      fusion-pattern atomicity via keep_pattern.
Observation:          The optimizer (line 16941) walks the graph, keeping
                      fusion patterns (topk_moe_*, snake) atomic via
                      `keep_pattern` (lines 16991-17004). For each unused node,
                      it scans the next 20 nodes (NUM_TO_CHECK, line 17031) for
                      nodes that don't depend on unrun predecessors, and pulls
                      them earlier. Special-case: when RMS_NORM+MUL is found,
                      the optimizer looks ahead 15 nodes for a ROPE that uses
                      the MUL's output (lines 17068-17083), pulling it adjacent
                      to enable RMS_NORM_MUL_ROPE fusion.
Evidence:             ggml-vulkan.cpp:16941-17196 (full function), 16991-17004
                      (keep_pattern), 17031 (NUM_TO_CHECK), 17068-17083 (ROPE
                      lookahead).
Architectural Impact: Exposes fusion patterns that the original graph order
                      wouldn't. The 20-node lookahead is O(N*20) per graph.
Correctness Impact:   None. The reorder preserves data dependencies.
Optimization Type:    Kernel fusion (enabling).
GwenLand Target:      GATE
Recommendation:       ADOPT. Same reorder + keep_pattern design for GATE.
Priority:             High
Difficulty:           M
Dependencies:         R4
Confidence:           High
```

### Finding ARTX18-F11

```
Finding ID:           ARTX18-F11
Category:             BACKEND_DESIGN
Engine:               Vulkan
Component:            Buffer types (device + host)
Source File:          ggml/src/ggml-vulkan/ggml-vulkan.cpp
Function:             ggml_backend_vk_buffer_type_alloc_buffer,
                      ggml_backend_vk_host_buffer_type_alloc_buffer,
                      ggml_vk_create_buffer_device
Lines:                15616-15726, 3361-3398
Summary:              Device buffer type selects device-local memory (ReBAR
                      preferred); host buffer type allocates pinned host-visible
                      memory and falls back to CPU buffer on failure.
Observation:          `ggml_vk_create_buffer_device` (line 3361) picks memory
                      type via a preference chain: UMA prefers host-visible
                      device-local; discrete+ReBAR prefers device-local
                      host-visible; discrete without ReBAR uses device-local
                      only (unless allow_sysmem_fallback). The host buffer type
                      (line 15671) calls `ggml_vk_host_malloc` (line 7739) which
                      allocates host-visible+coherent+cached memory, registers
                      the pointer in `device->pinned_memory`, and returns it.
                      On failure, falls back to `ggml_backend_cpu_buffer_type`
                      (line 15681). The host buffer type is **not device-specific**
                      — it always uses `vk_instance.devices[0]` (line 15677), with
                      a TODO comment at line 15705 acknowledging this.
Evidence:             ggml-vulkan.cpp:3361-3398 (device buffer), 15671-15691
                      (host buffer), 7739-7757 (host_malloc + registration),
                      15705 (TODO comment).
Architectural Impact: Pinned memory enables zero-copy host→device transfers via
                      vkCmdCopyBuffer (line 8130). The non-device-specific host
                      buffer type is a known limitation.
Correctness Impact:   None.
Optimization Type:    Pinned host memory for staging.
GwenLand Target:      glvulkan
Recommendation:       ADOPT device buffer type. ADAPT host buffer type to be
                      device-specific (fix the TODO at line 15705).
Priority:             High
Difficulty:           M
Dependencies:         R6
Confidence:           High
```

### Finding ARTX18-F12

```
Finding ID:           ARTX18-F12
Category:             BACKEND_DESIGN
Engine:               Vulkan
Component:            Events + async
Source File:          ggml/src/ggml-vulkan/ggml-vulkan.cpp
Function:             ggml_backend_vk_event_record, ggml_backend_vk_event_wait,
                      ggml_backend_vk_device_event_synchronize
Lines:                17198-17252, 18107-18137
Summary:              Events use dual representation: vk::Event for in-queue
                      ordering (vkCmdSetEvent/vkCmdWaitEvents) and a timeline
                      semaphore for host-side wait (waitSemaphores). This avoids
                      the validation errors that pure-event polling would cause.
Observation:          `vk_event` struct (line 1135) holds a `vk::Event` plus a
                      `vk_semaphore tl_semaphore`. `event_record` (line 17198)
                      records `vkCmdSetEvent` on the current command buffer and
                      signals the timeline semaphore on submit. `event_wait`
                      (line 17236) records `vkCmdWaitEvents` on the next command
                      buffer in the same queue. `event_synchronize` (line 18107)
                      calls `device.waitSemaphores(tl_semaphore, UINT64_MAX)` for
                      host-side blocking. The comment at line 1131 explains the
                      design: "Polling on an event for event_synchronize wouldn't
                      be sufficient to wait for command buffers to complete."
Evidence:             ggml-vulkan.cpp:1131-1144 (struct + comment), 17198-17234
                      (record), 17236-17252 (wait), 18107-18137 (synchronize).
Architectural Impact: Correct event semantics across host and device. The dual
                      representation is the standard Vulkan pattern.
Correctness Impact:   None.
Optimization Type:    Asynchronous execution (event + timeline semaphore).
GwenLand Target:      glvulkan
Recommendation:       ADOPT. Same dual-representation event design for GwenLand.
Priority:             High
Difficulty:           S
Dependencies:         R5, R7
Confidence:           High
```

### Finding ARTX18-F13

```
Finding ID:           ARTX18-F13
Category:             MEMORY_PATTERN
Engine:               Vulkan
Component:            Memory barriers
Source File:          ggml/src/ggml-vulkan/ggml-vulkan.cpp
Function:             ggml_vk_sync_buffers
Lines:                3416-3436
Summary:              Sync inserts a global vkCmdPipelineBarrier with empty
                      BufferMemoryBarrier arrays; only srcAccessMask/dstAccessMask
                      are set, making every pending buffer write visible.
Observation:          Line 3425 emits `vkCmdPipelineBarrier` with
                      `srcStageMask == dstStageMask == ctx->p->q->stage_flags`
                      and `BufferMemoryBarrier` arrays empty (`{},{}` at lines
                      3433-3434). Only `srcAccessMask`/`dstAccessMask` are set
                      (lines 3430-3431) to shader-read+shader-write+transfer-
                      read+transfer-write (or transfer-only for transfer queue).
                      This makes every pending buffer write visible, not just
                      the overlapping ones. The per-node overlap analysis at
                      lines 14837-14917 decides *when* to call sync_buffers,
                      but the barrier itself is broader than necessary.
Evidence:             ggml-vulkan.cpp:3416-3436 (sync_buffers), 14890-14896
                      (call site driven by overlap analysis).
Architectural Impact: Conservative but correct. Per-buffer barriers would be
                      more precise but require tracking every buffer touched
                      since the last barrier.
Correctness Impact:   None. Global barriers are correct.
Optimization Type:    None (conservative barrier).
GwenLand Target:      glvulkan
Recommendation:       ADAPT. Keep the global barrier as default; add per-buffer
                      BufferMemoryBarrier arrays when the unsynced list is small
                      (e.g., <8 buffers).
Priority:             Medium
Difficulty:           M
Dependencies:         R6
Confidence:           High
```

### Finding ARTX18-F14

```
Finding ID:           ARTX18-F14
Category:             GPU_KERNEL
Engine:               Vulkan
Component:            Cooperative matrix
Source File:          ggml/src/ggml-vulkan/ggml-vulkan.cpp
Function:             ggml_vk_get_device (coopmat query)
Lines:                5976-6026, 6275-6296, 6435-6649
Summary:              VK_KHR_cooperative_matrix and VK_NV_cooperative_matrix2 are
                      queried at device init; supported shapes are enumerated and
                      a single (M,N,K) tuple is selected. Three flash-attention
                      code paths (scalar, coopmat1, coopmat2) are selected per
                      device.
Observation:          For KHR_cooperative_matrix (lines 6558-6649), the backend
                      queries all `VkCooperativeMatrixPropertiesKHR` shapes and
                      picks a single (M,N,K) tuple that supports both f16×f16→f32
                      and f16×f16→f16 accumulation, with a preference for
                      16×16×16. For NV_cooperative_matrix2 (lines 6440-6553), it
                      queries `VkCooperativeMatrixFlexibleDimensionsPropertiesNV`
                      and requires workgroup-scope + flexible-dimensions +
                      reductions + conversions + tensor-addressing + block-loads
                      + bufferDeviceAddress. The `ggml_vk_khr_cooperative_matrix_support`
                      function (line 18338) applies vendor-specific gating:
                      Intel only allows Xe2/iGPU; AMD proprietary only allows
                      RDNA3. Three FA code paths (FA_SCALAR, FA_COOPMAT1,
                      FA_COOPMAT2) are selected per device (enum at line 525).
Evidence:             ggml-vulkan.cpp:5976-6026 (extension detection), 6558-6649
                      (KHR shape query), 6440-6553 (NV shape query), 18338-18353
                      (vendor gating), 525-529 (FA code paths).
Architectural Impact: Tensor-Core-like matmul on supported devices. Vendor
                      gating avoids known-bad configurations (e.g., AMD
                      proprietary on non-RDNA3).
Correctness Impact:   None.
Optimization Type:    SIMD (cooperative matrix for tensor-core-like matmul).
GwenLand Target:      glvulkan
Recommendation:       ADOPT. Same extension query + vendor gating + per-device
                      code-path selection for GwenLand.
Priority:             High
Difficulty:           L
Dependencies:         R1, R11
Confidence:           High
```

---

## 18. Unknowns

* **U1**. Whether the absence of `VkPipelineCache` is a measured tradeoff
  or an oversight. The code has no comment explaining it. Static analysis
  cannot determine whether the team tried it and rejected it, or simply
  never added it. Requires git-history investigation or asking the team.

* **U2**. Whether the spin-poll fence wait at `ggml_vk_wait_for_fence`
  (lines 2397–2413) is faster than a second `waitForFences` call in
  practice. The comment says "hopefully the CPU can sleep during this
  wait" (line 2388) referring to the almost-ready fence, but the final
  spin is unconditional once that fence is signaled. Requires profiling
  on real hardware to compare spin vs. `waitForFences`.

* **U3**. Whether the shared descriptor set layout (12 bindings per set)
  causes descriptor-memory pressure on constrained devices (e.g., mobile
  GPUs with limited `maxDescriptorSetStorageBuffers`). Static analysis
  shows the layout declares 12 bindings; whether any device rejects it
  is not visible. Requires running on a range of devices.

* **U4**. Whether the pinned-memory O(N) linear scan is a measurable
  bottleneck. For a workload with many small pinned regions (e.g.,
  per-layer KV cache pinning), the scan could be hot. For typical
  workloads with 1–4 pinned regions, it's negligible. Requires
  profiling with a pinned-region-heavy workload.

* **U5**. Whether the `flops_per_submit` heuristic (200 GFLOP cap,
  `last_total_flops / 40`) is optimal for the wide range of GPU
  performance levels in the wild. The doubling after the first 3
  submits (line 16877) suggests the team found early submits too
  frequent, but the magic numbers are undocumented. Requires
  benchmarking across GPU classes.

* **U6**. Whether the SPIR-V float-controls patcher correctly handles
  shaders with no `OpEntryPoint`. The code at line 2681 uses
  `entry_point_id` without asserting it's nonzero; if a shader had no
  entry point, the `OpExecutionMode` insert would be at the wrong
  position. Static analysis cannot determine whether any shader in the
  tree lacks an entry point.

* **U7**. Whether `ggml_vk_graph_optimize`'s 20-node lookahead
  (`NUM_TO_CHECK`, line 17031) is sufficient for all fusion patterns.
  For graphs with very long dependency chains, the lookahead may miss
  reorder opportunities. Static analysis shows the lookahead is fixed
  at 20; whether this is enough for real models requires runtime
  testing.

* **U8**. Whether `event_record`'s forced command-buffer submit (line
  17229) is a measurable overhead for workloads that record many
  events. The submit ends the current command buffer and starts a new
  one. For typical inference (1–2 events per graph), negligible; for
  fine-grained profiling, potentially significant. Requires profiling.

* **U9**. Whether `VK_EXT_external_memory_host` is actually used in
  practice. The code at line 18140 returns early if the extension
  isn't supported, and MoltenVK is explicitly disabled (line 6092).
  Whether any production workload calls `buffer_from_host_ptr` is not
  visible from static analysis.

---

## 19. References

| Reference | File                                                | Function / Symbol                              | Lines         |
| --------- | --------------------------------------------------- | ---------------------------------------------- | ------------- |
| R01       | `ggml/src/ggml-vulkan/ggml-vulkan.cpp`              | `vk_pipeline_struct`                           | 213–238       |
| R02       | `ggml/src/ggml-vulkan/ggml-vulkan.cpp`              | `vk_queue`, `vk_queue_handle*`                 | 306–341       |
| R03       | `ggml/src/ggml-vulkan/ggml-vulkan.cpp`              | `vk_command_pool`, `vk_command_buffer`         | 280–303       |
| R04       | `ggml/src/ggml-vulkan/ggml-vulkan.cpp`              | `vk_device_struct`                             | 720–1077      |
| R05       | `ggml/src/ggml-vulkan/ggml-vulkan.cpp`              | `vk_buffer_struct`, `vk_subbuffer`             | 1095–1124     |
| R06       | `ggml/src/ggml-vulkan/ggml-vulkan.cpp`              | `vk_event`, `vk_submission`, `vk_sequence`     | 1131–1152     |
| R07       | `ggml/src/ggml-vulkan/ggml-vulkan.cpp`              | `vk_instance_t`                                | 2347–2364     |
| R08       | `ggml/src/ggml-vulkan/ggml-vulkan.cpp`              | `ggml_backend_vk_context`                      | 2164–2233     |
| R09       | `ggml/src/ggml-vulkan/ggml-vulkan.cpp`              | `ggml_backend_vk_buffer_type_interface`        | 348–355       |
| R10       | `ggml/src/ggml-vulkan/ggml-vulkan.cpp`              | `ggml_vk_instance_init`                        | 7078–7326     |
| R11       | `ggml/src/ggml-vulkan/ggml-vulkan.cpp`              | `ggml_vk_init` (backend context init)          | 7328–7370     |
| R12       | `ggml/src/ggml-vulkan/ggml-vulkan.cpp`              | `ggml_vk_get_device`                           | 5906–6820     |
| R13       | `ggml/src/ggml-vulkan/ggml-vulkan.cpp`              | `ggml_vk_find_queue_family_index`              | 3024–3068     |
| R14       | `ggml/src/ggml-vulkan/ggml-vulkan.cpp`              | `ggml_vk_create_queue`                         | 3070–3097     |
| R15       | `ggml/src/ggml-vulkan/ggml-vulkan.cpp`              | `ggml_vk_create_buffer`                        | 3194–3349     |
| R16       | `ggml/src/ggml-vulkan/ggml-vulkan.cpp`              | `ggml_vk_create_buffer_device`                 | 3361–3398     |
| R17       | `ggml/src/ggml-vulkan/ggml-vulkan.cpp`              | `ggml_vk_host_malloc` / `ggml_vk_host_get`     | 7739–7800     |
| R18       | `ggml/src/ggml-vulkan/ggml-vulkan.cpp`              | `ggml_vk_strip_decode_vector`                  | 2428–2546     |
| R19       | `ggml/src/ggml-vulkan/ggml-vulkan.cpp`              | `ggml_vk_roll_bk_loop`                         | 2556–2625     |
| R20       | `ggml/src/ggml-vulkan/ggml-vulkan.cpp`              | `ggml_vk_create_pipeline_func`                 | 2627–2878     |
| R21       | `ggml/src/ggml-vulkan/ggml-vulkan.cpp`              | `ggml_pipeline_allocate_descriptor_sets`       | 2898–2934     |
| R22       | `ggml/src/ggml-vulkan/ggml-vulkan.cpp`              | `ggml_vk_submit`                               | 2947–3022     |
| R23       | `ggml/src/ggml-vulkan/ggml-vulkan.cpp`              | `ggml_vk_sync_buffers`                         | 3416–3436     |
| R24       | `ggml/src/ggml-vulkan/ggml-vulkan.cpp`              | `ggml_vk_set_event` / `ggml_vk_wait_events`    | 3438–3470     |
| R25       | `ggml/src/ggml-vulkan/ggml-vulkan.cpp`              | `ggml_vk_load_shaders`                         | 3938–5901     |
| R26       | `ggml/src/ggml-vulkan/ggml-vulkan.cpp`              | `ggml_vk_dispatch_pipeline` (template)         | 7877–7907     |
| R27       | `ggml/src/ggml-vulkan/ggml-vulkan.cpp`              | `ggml_vk_get_compute_ctx` / `get_transfer_ctx` | 7929–7960     |
| R28       | `ggml/src/ggml-vulkan/ggml-vulkan.cpp`              | `ggml_vk_submit_transfer_ctx`                  | 7965–7983     |
| R29       | `ggml/src/ggml-vulkan/ggml-vulkan.cpp`              | `ggml_vk_wait_for_fence`                       | 2386–2416     |
| R30       | `ggml/src/ggml-vulkan/ggml-vulkan.cpp`              | `ggml_vk_op_f32` (op dispatch template)        | 11610–11910   |
| R31       | `ggml/src/ggml-vulkan/ggml-vulkan.cpp`              | `ggml_vk_build_graph`                          | 14797–15315   |
| R32       | `ggml/src/ggml-vulkan/ggml-vulkan.cpp`              | `ggml_vk_compute_forward`                      | 15317–15363   |
| R33       | `ggml/src/ggml-vulkan/ggml-vulkan.cpp`              | `ggml_vk_graph_cleanup`                        | 15366–15402   |
| R34       | `ggml/src/ggml-vulkan/ggml-vulkan.cpp`              | `ggml_vk_synchronize`                          | 15928–15987   |
| R35       | `ggml/src/ggml-vulkan/ggml-vulkan.cpp`              | `ggml_backend_vk_graph_compute`                | 16525–16938   |
| R36       | `ggml/src/ggml-vulkan/ggml-vulkan.cpp`              | `ggml_vk_graph_optimize`                       | 16941–17196   |
| R37       | `ggml/src/ggml-vulkan/ggml-vulkan.cpp`              | `ggml_backend_vk_event_record` / `event_wait`  | 17198–17252   |
| R38       | `ggml/src/ggml-vulkan/ggml-vulkan.cpp`              | `ggml_backend_vk_i` (vtable)                   | 17255–17272   |
| R39       | `ggml/src/ggml-vulkan/ggml-vulkan.cpp`              | `ggml_backend_vk_init`                         | 17279–17297   |
| R40       | `ggml/src/ggml-vulkan/ggml-vulkan.cpp`              | `ggml_backend_vk_device_event_new/free/sync`   | 18064–18137   |
| R41       | `ggml/src/ggml-vulkan/ggml-vulkan.cpp`              | `ggml_backend_vk_device_buffer_from_host_ptr`  | 18164–18182   |
| R42       | `ggml/src/ggml-vulkan/ggml-vulkan.cpp`              | `ggml_backend_vk_device_i` (vtable)            | 18184–18200   |
| R43       | `ggml/src/ggml-vulkan/ggml-vulkan.cpp`              | `ggml_backend_vk_reg_i` (vtable)               | 18246–18251   |
| R44       | `ggml/src/ggml-vulkan/ggml-vulkan.cpp`              | `ggml_vk_khr_cooperative_matrix_support`       | 18338–18353   |
| R45       | `ggml/src/ggml-vulkan/ggml-vulkan.cpp`              | `get_device_architecture`                      | 377–495       |
| R46       | `ggml/src/ggml-vulkan/ggml-vulkan.cpp`              | `vk_matmul_pipeline_struct`, `vk_matmul_pipeline2` | 244–263   |
| R47       | `ggml/src/ggml-vulkan/ggml-vulkan.cpp`              | `ggml_vk_buffer_from_host_ptr`                 | 18139–18162   |
| R48       | `ggml/src/ggml-vulkan/CMakeLists.txt`               | `test_shader_extension_support`, ExternalProject | 38–57, 177–195 |
| R49       | `ggml/src/ggml-vulkan/vulkan-shaders/CMakeLists.txt`| `vulkan-shaders-gen` build                     | 1–43         |
| R50       | `ggml/src/ggml-vulkan/ggml-vulkan.cpp`              | `vk_op_multi_add_push_constants`               | 1484–1495     |
