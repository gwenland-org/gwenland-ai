# ARTX22 — Execution Graph, Scheduler, and Backend Dispatch

**Audited commit:** `555881ebc8b0fc0402b30e09258a32a7bfd13c52`
**Last updated:** 2026-07-25
**Auditor:** Percival-aux (ARTX22)
**Target GwenLand module:** `GATE` (execution graph / scheduler), with shared implications for `glproc`, `glcuda`, `glmetal`, `glvulkan`

---

## 1. Executive Summary

The ggml execution-graph stack is a three-layer system sitting *between* the
user's tensor-building code (which constructs a `ggml_cgraph` by recursive
post-order DFS over `tensor->src[]`) and the per-backend kernels. The three
layers are:

1. **The graph itself** (`ggml_cgraph` in `ggml-impl.h:329-347`) — a flat
   array of `ggml_tensor *` node pointers plus a flat array of leaf
   pointers, a `visited_hash_set` for membership / dedup, and a
   `use_counts[]` array indexed by hash slot. Built by
   `ggml_visit_parents_graph` (`ggml.c:7098-7164`); sliced into views by
   `ggml_graph_view` (`ggml.c:7384`).
2. **The CPU-side plan-and-compute interface**
   (`ggml_graph_plan` / `ggml_graph_compute` in `ggml-cpu.c:2781` and
   `ggml-cpu.c:3350`) — the only "graph plan" that actually exists. The
   shared backend interface declares `graph_plan_create` /
   `graph_plan_free` / `graph_plan_update` / `graph_plan_compute` vtable
   slots but explicitly comments them as "not used currently"
   (`ggml-backend-impl.h:120-127`). No backend implements them.
3. **The cross-backend scheduler** (`ggml_backend_sched` in
   `ggml-backend.cpp:774-828`) — assigns every node to one of N backends
   via a five-pass heuristic (`ggml_backend_sched_split_graph`,
   `ggml-backend.cpp:1014-1487`), partitions the graph into *splits*
   (contiguous node ranges executed on the same backend), inserts
   implicit cross-backend copies for inputs that live on a different
   backend, pre-allocates every tensor through `ggml_gallocr`, and runs
   the splits sequentially in graph-node order via
   `ggml_backend_sched_compute_splits` (`ggml-backend.cpp:1541-1725`).

For GwenLand, the architectural decisions worth **ADOPT**ing are: the flat
`ggml_cgraph` representation with a hash-set for membership/use-count; the
CPU-as-last-backend priority convention; the implicit cross-backend copy
insertion (with node-pointer rewriting) for splits; the pipeline-parallel
N-copies-plus-events mechanism for overlapping host↔device copies with
backend compute; the per-tensor `prev_*_backend_ids` swap-dance that
short-circuits reallocation when assignments are stable; and the `op_offload`
hook that lets a higher-prio backend claim a CPU-resident-weight op. The
decisions worth **REJECT**ing are: the *sequential* execution of splits
(independent splits never run concurrently), the per-sub-batch
synchronization forced by the eval callback (kills async), and the
synchronous-fallback taken when a backend lacks event support (which
serializes the pipeline). The decisions worth **ADAPT**ing are: moving
graph optimization from per-backend per-split time into the scheduler
itself (so optimization sees the whole graph and can fuse across splits),
and merging the dead `graph_plan_*` vtable into a real shared plan-time
pass.

---

## 2. Purpose

Provide a backend-agnostic execution layer that:

* lets the user build a graph once with `ggml_build_forward_expand` and
  have it run unchanged on a single CPU, on a single GPU, or on a hybrid
  CPU+GPU+GPU mix;
* assigns every node in the graph to one of N registered backends using
  only `supports_op` + `supports_buft` + buffer-usage hints (no
  backend-specific knowledge in the scheduler);
* partitions the graph into contiguous *splits* per backend, with
  automatic cross-backend copies of any input that lives on a different
  backend's buffer type;
* pre-allocates every intermediate tensor before execution begins, so
  that no per-node allocation happens on the hot path;
* optionally pipelines execution across N copies of every input tensor
  and intermediate, so that one batch's compute can overlap with the
  next batch's host→device copies;
* exposes a stable C ABI (`ggml_backend_sched_*`) so applications can
  manually override backend assignments, reserve max-size buffers,
  introspect the number of splits / copies, and observe per-node
  results via an eval callback.

It is **not** responsible for: kernel dispatch (delegated to each
backend's `graph_compute`), per-ISA SIMD selection (delegated to the CPU
backend, ARTX01–07), graph construction from a model (delegated to the
llama.cpp layer above ggml), or gradient computation (handled in
`ggml_build_backward_expand`, separate code path). It is also not
responsible for tensor memory layout decisions — those happen in the
allocator (ARTX21).

---

## 3. Source Files

| File                                       | Lines  | Role                                                                                  |
| ------------------------------------------ | ------ | ------------------------------------------------------------------------------------- |
| `ggml/src/ggml.c`                          | 8023   | `ggml_cgraph` construction (`ggml_visit_parents_graph`, `ggml_build_forward_impl`), graph view / cpy / dup / reset, `ggml_graph_dump_dot`, `ggml_can_fuse_subgraph_ext`. Note: `ggml_graph_plan` / `ggml_graph_compute` are *not* here. |
| `ggml/src/ggml-impl.h`                     | 783    | `struct ggml_cgraph` definition (lines 329-347); `struct ggml_hash_set` and inline `ggml_hash_find` / `ggml_hash_insert` / `ggml_hash_find_or_insert` (linear-probing, `ptr >> 4` hash); `ggml_node_get_use_count` / `ggml_can_fuse` / `ggml_check_edges` helpers. |
| `ggml/src/ggml-backend.cpp`                | 2372   | Primary file. `ggml_backend_sched` struct + entire scheduler: 5-pass `ggml_backend_sched_split_graph`, `ggml_backend_sched_alloc_splits`, `ggml_backend_sched_compute_splits`, async-copy + events, CPU buffer type. |
| `ggml/src/ggml-backend-impl.h`             | 275    | All vtable definitions: `ggml_backend_buffer_type_i`, `ggml_backend_buffer_i`, `ggml_backend_i` (with dead `graph_plan_*` slots), `ggml_backend_device_i` (with `supports_op`, `supports_buft`, `offload_op`), `ggml_backend_reg_i`. |
| `ggml/src/ggml-cpu/ggml-cpu.c`             | 3895   | `ggml_graph_plan` (2781) and `ggml_graph_compute` (3350) — the only "graph plan" implementation. Per-op `work_size` estimation in the plan. |
| `ggml/include/ggml-cpu.h`                  | 153    | `struct ggml_cplan` definition (`work_size`, `work_data`, `n_threads`, `threadpool`, `abort_callback`, `use_ref`). |
| `ggml/include/ggml-backend.h`              | 436    | Public scheduler API: `ggml_backend_sched_new` / `_reserve` / `_alloc_graph` / `_graph_compute` / `_synchronize` / `_reset` / `_set_eval_callback` / `_set_tensor_backend` / `_get_n_splits` / `_get_n_copies` / `_get_n_backends`. |
| `ggml/src/ggml-cuda/ggml-cuda.cu` (ref)    | 5426   | Reference backend: implements `event_record`, `event_wait`, `cpy_tensor_async`, `graph_optimize` (env-gated). CUDA graph capture path. |
| `ggml/src/ggml-cpu/ggml-cpu.cpp` (ref)     | 708    | Reference backend: CPU sets all async/event/graph_plan/graph_optimize vtable slots to NULL (ARTX01-F01). |

> Note: ARTX22 is a Shared-layer audit. Where the CUDA or CPU backend is
> referenced, it is to illustrate how the scheduler contract is *consumed*
> by a real backend. Per-backend ARTX documents (ARTX01, ARTX08, ARTX15,
> ARTX18) cover the backend-internal graph execution in depth.

---

## 4. Architecture Overview

```
                ┌───────────────────────────────────────────────────────────────┐
   user code    │  ggml_build_forward_expand(cgraph, root_tensor)               │
                │  → ggml_visit_parents_graph (post-order DFS, hash dedup)      │
                └────────────────────────────────┬──────────────────────────────┘
                                                 │
                                                 ▼
                ┌───────────────────────────────────────────────────────────────┐
   graph data   │  struct ggml_cgraph {                                         │
   structure    │    nodes[], leafs[], grads[], grad_accs[], use_counts[]       │
   (ggml-impl)  │    visited_hash_set (linear-probing, ptr>>4 hash)             │
                │    order = LEFT_TO_RIGHT | RIGHT_TO_LEFT                      │
                │    uid  (graph identity for cross-call re-allocation checks)  │
                │  }                                                            │
                └────────────────────────────────┬──────────────────────────────┘
                                                 │
                                                 ▼
                ┌───────────────────────────────────────────────────────────────┐
   scheduler    │  ggml_backend_sched (ggml-backend.cpp:774-828)                │
   (cross-      │   ├─ backends[16], bufts[16], galloc                           │
   backend)     │   ├─ hv_tensor_backend_ids[hash]  (per-tensor assignment)     │
                │   ├─ hv_tensor_copies[hash][16][4]  (per-copy duplicates)     │
                │   ├─ node_backend_ids[], leaf_backend_ids[]                   │
                │   ├─ prev_node_backend_ids[], prev_leaf_backend_ids[]         │
                │   ├─ splits[]  (contiguous node range + input list + graph)   │
                │   ├─ events[16][4]  (per-backend per-copy events)             │
                │   ├─ graph_inputs[] + n_graph_inputs (pipeline inputs)        │
                │   └─ callback_eval + user_data  (per-node observation hook)   │
                └────────────────────────────────┬──────────────────────────────┘
                                                 │
              ┌────────────────────────────────┼─────────────────────────────┐
              ▼                                ▼                              ▼
   ┌───────────────────┐         ┌────────────────────────┐         ┌──────────────────┐
   │ Pass 1: assign    │         │ Pass 2: expand GPU up  │         │ Pass 3: upgrade  │
   │ from pre-allocated│  →      │ & down + rest up & down│  →      │ to higher prio   │
   │ weights + inputs  │         │ (skip CPU as boundary) │         │ buft; pick best  │
   │ + view_src        │         │                        │         │ for unassigned   │
   └───────────────────┘         └────────────────────────┘         └──────────────────┘
                                                 │
              ┌────────────────────────────────┼─────────────────────────────┐
              ▼                                ▼                              ▼
   ┌───────────────────┐         ┌────────────────────────┐         ┌──────────────────┐
   │ Pass 4: assign    │         │ Pass 5: split graph    │         │ Per-split:       │
   │ remaining src     │  →      │ into contiguous        │  →      │ backend.graph_   │
   │ from dst + view   │         │ per-backend ranges;    │         │ optimize();      │
   │                   │         │ insert copies          │         │ gallocr_alloc    │
   └───────────────────┘         └────────────────────────┘         └──────────────────┘
                                                 │
                                                 ▼
                ┌───────────────────────────────────────────────────────────────┐
   execution    │  ggml_backend_sched_compute_splits (1541-1725)                │
                │   for (split_id in 0..n_splits):                              │
                │     1. for each split input: synchronize event or sync dst   │
                │     2. async-copy input → input_cpy (or sync fallback)       │
                │     3. (special MoE path: partial expert copy via bitset)    │
                │     4. backend.graph_compute_async(split.graph)              │
                │     5. event_record(events[split_backend][cur_copy])         │
                └────────────────────────────────┬──────────────────────────────┘
                                                 │
                                                 ▼
                ┌───────────────────────────────────────────────────────────────┐
   per-backend  │  backend.iface.graph_compute(backend, cgraph)                 │
   execution    │   CPU:   ggml_graph_compute (ggml-cpu.c:3350) — SPMD+bARRIER  │
                │   CUDA:  ggml_backend_cuda_graph_compute (ggml-cuda.cu:4100)  │
                │           optional CUDA-graph capture (USE_CUDA_GRAPH)        │
                │   Metal/Vulkan: similar per-backend graph_compute             │
                └───────────────────────────────────────────────────────────────┘
```

Key design points:

* **The graph is a flat array, not an adjacency list.** Splits are
  contiguous slices of `cgraph->nodes[]` (via `ggml_graph_view`). The
  scheduler never builds a separate DAG; it walks the same array five
  times. Topological order is implicit because `ggml_visit_parents_graph`
  emits in post-order.
* **Backend assignment is priority-ordered.** `backends[]` is supplied
  by the caller; lower index = higher priority.
  `ggml_backend_sched_new` asserts the last backend is CPU
  (`ggml-backend.cpp:1736`); CPU is the catch-all fallback. The 5-pass
  heuristic in pass 2 explicitly *skips* CPU when expanding GPU
  backends up/down (lines 1087, 1108), so GPU is preferred unless the
  op is unsupported or its inputs are incompatible.
* **Cross-backend copies are inserted silently.** Pass 5 rewrites
  `node->src[j]` in-place to point at a freshly-duplicated tensor
  (`ggml-backend.cpp:1370`). The original tensor is preserved in
  `split->inputs[]` so the executor can copy into the duplicate. This
  means the *user's* tensor pointers in the graph are mutated by the
  scheduler.
* **Pipeline parallelism is per-copy, not per-split.** When
  `parallel = true`, the scheduler allocates `GGML_SCHED_MAX_COPIES = 4`
  copies of every cross-backend input and every graph input. Each
  execution rotates `cur_copy`, recording events on the destination
  backend after each split. The next execution waits on the event for
  the *same copy id* before overwriting, achieving overlap.
* **Two levels of "plan" exist, but they don't compose.** The CPU
  `ggml_graph_plan` / `ggml_graph_compute` pair is a *per-backend*
  mechanism that precomputes `work_size` and threads. The scheduler
  has no equivalent — it calls `backend.graph_compute_async` directly
  per split. The shared `graph_plan_*` vtable methods in
  `ggml-backend-impl.h:120-127` are explicitly documented as unused.

---

## 5. Execution Flow

### 5.1 Graph construction: `ggml_build_forward_expand`

`ggml_build_forward_expand` (`ggml.c:7199`) calls
`ggml_build_forward_impl(cgraph, tensor, expand=true, compute=true)`
(`ggml.c:7166`), which calls
`ggml_visit_parents_graph(cgraph, tensor, compute=true)`
(`ggml.c:7098-7164`):

1. If `node->op != GGML_OP_NONE` and `compute`, set
   `node->flags |= GGML_TENSOR_FLAG_COMPUTE`.
2. Hash-find `node` in `cgraph->visited_hash_set`. If already visited,
   update child `COMPUTE` flags if necessary and return.
3. Otherwise, mark as visited (set bit + zero `use_counts[hash_pos]`).
4. Recurse into each `node->src[i]` (i=0..GGML_MAX_SRC-1=9). Source
   iteration order depends on `cgraph->order`: `LEFT_TO_RIGHT`
   (default) iterates `0..N`, `RIGHT_TO_LEFT` reverses. Each visited
   source's `use_counts[src_hash_pos]` is incremented.
5. Append to `nodes[]` or `leafs[]` based on whether the node is an
   `GGML_OP_NONE` non-PARAM tensor (leaf) or anything else (node).

The resulting `nodes[]` array is in topological (post-order) order:
every node appears after all its sources. The last node is the
user's root tensor. This is the order in which the scheduler and the
per-backend executor will walk the graph.

### 5.2 Graph slicing: `ggml_graph_view`

`ggml_graph_view(cgraph, i0, i1)` (`ggml.c:7384-7400`) returns a
*stack-allocated* `ggml_cgraph` whose `nodes` pointer is
`cgraph->nodes + i0` and whose `n_nodes = i1 - i0`. The view shares
the parent's `use_counts` and `visited_hash_set` (so hash lookups
work). Views have no leafs, no grads, and `size = 0`.

The scheduler creates one view per split:
`split->graph = ggml_graph_view(graph, split->i_start, split->i_end)`
(`ggml-backend.cpp:1413`). Each view is then handed to its backend's
`graph_compute_async`.

### 5.3 Scheduler entry: `ggml_backend_sched_graph_compute`

The synchronous entry `ggml_backend_sched_graph_compute`
(`ggml-backend.cpp:1883-1887`) is just a wrapper:

```
err = ggml_backend_sched_graph_compute_async(sched, graph);
ggml_backend_sched_synchronize(sched);
return err;
```

The async entry (`ggml-backend.cpp:1889-1902`) does the real work:

1. If `!sched->is_reset && !sched->is_alloc`, call
   `ggml_backend_sched_reset(sched)`.
2. If `!sched->is_alloc`, call
   `ggml_backend_sched_alloc_graph(sched, graph)`. If that fails,
   return `GGML_STATUS_ALLOC_FAILED`.
3. Call `ggml_backend_sched_compute_splits(sched)`.

### 5.4 Allocation: `ggml_backend_sched_alloc_graph`

`ggml_backend_sched_alloc_graph` (`ggml-backend.cpp:1864-1881`):

1. Rotate the pipeline copy: `cur_copy = next_copy;
   next_copy = (next_copy + 1) % n_copies`.
2. Call `ggml_backend_sched_split_graph(sched, graph)` — the 5-pass
   assignment + split + copy-insertion described in Section 5.5.
3. Call `ggml_backend_sched_alloc_splits(sched)` (Section 5.6) — the
   `ggml_gallocr`-based allocation.
4. Set `sched->is_alloc = true` so the next `_graph_compute_async`
   can skip re-allocation.

### 5.5 Assignment + split: `ggml_backend_sched_split_graph`

`ggml_backend_sched_split_graph` (`ggml-backend.cpp:1014-1487`) is
the heart of the scheduler. Five passes:

**Pass 1** (`ggml-backend.cpp:1036-1070`): assign backends to
pre-allocated tensors. For every leaf and every node, call
`ggml_backend_sched_backend_id_from_cur(sched, tensor)` (line 878),
which in turn:

* Tries the tensor's own buffer's backend via
  `ggml_backend_sched_backend_from_buffer` (line 845) — the first
  backend (lowest index) that both `supports_buft(buffer->buft)` and
  `supports_op(op)`.
* Falls back to `view_src->buffer`'s backend if the tensor is a view.
* If `tensor->flags & GGML_TENSOR_FLAG_INPUT`, assigns to
  `sched->n_backends - 1` (CPU, the last backend).
* If any source `src` is in a buffer with
  `usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS`, assigns to that
  source's backend (line 916). Special case: ROPE is skipped here
  because "the rope freqs tensor is too small to choose a backend
  based on it" (line 914).
* `op_offload` escape hatch (line 919): if the weights are on CPU
  (`n_backends - 1`) but `is_host`, and a higher-prio backend
  supports the op and `offload_op` returns true, assign to the
  higher-prio backend.

**Pass 2** (`ggml-backend.cpp:1078-1150`): four sub-passes that
*expand* existing assignments to neighboring nodes:

* **GPU down**: walk forward, propagate the current non-CPU backend
  to subsequent unassigned nodes that support it.
* **GPU up**: walk backward, same.
* **Rest down**: walk forward, propagate *any* backend (including
  CPU) to unassigned nodes.
* **Rest up**: walk backward, same.

The GPU-only sub-passes run first so that GPU assignment "stretches"
across CPU-compatible ops (which would otherwise fall back to CPU),
reducing cross-backend copies. View ops (`VIEW`, `RESHAPE`, `PERMUTE`,
`TRANSPOSE`) are skipped.

**Pass 3** (`ggml-backend.cpp:1160-1211`): for each node:

* If *still unassigned*, pick the backend with the most supported
  inputs (`3.best` cause).
* If *assigned*, try to *upgrade* to a higher-prio backend that has
  the same buffer type (`3.upg` cause). This handles the BLAS+CPU
  case where two backends share host memory.

**Pass 4** (`ggml-backend.cpp:1214-1243`): assign any remaining
unassigned sources from their consumer's backend, or from
`view_src`. Assert that every node ends up assigned.

**Pass 5** (`ggml-backend.cpp:1246-1376`): build the splits. Walk
`graph->nodes[]` in order; whenever the backend id changes (or a new
split is needed because the current split has hit
`GGML_SCHED_MAX_SPLIT_INPUTS = 30` inputs, or because a weight on
another backend is "in the way" — line 1282), close the current
split and open a new one. For each source on a different backend,
allocate a duplicate tensor (one per copy if `n_copies > 1`) in
`sched->ctx` and **rewrite `node->src[j]` to point at the duplicate**
(line 1370). Track the original in `split->inputs[]`.

After the splits are built, the scheduler constructs
`sched->graph` — a *copy* of the original graph with split inputs
injected as fake `GGML_OP_VIEW` dependency nodes at the start of
each split (line 1428-1435). This graph copy is what gets passed to
`ggml_gallocr`.

### 5.6 Allocation: `ggml_backend_sched_alloc_splits`

`ggml_backend_sched_alloc_splits` (`ggml-backend.cpp:1489-1539`):

1. **Diff check**: for each node in `sched->graph`, compare
   `node_backend_ids[i]` with `prev_node_backend_ids[i]`. If the
   *buffer type* (not the backend id) changed, mark
   `backend_ids_changed = true`. Same for leafs. This is the
   short-circuit: same backend assignments → no reallocation.
2. Try `ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)`.
3. If that fails (or `backend_ids_changed`), synchronize all
   backends, call `ggml_gallocr_reserve_n` (which re-plans offsets
   and re-allocates the underlying backend buffers), then try
   `ggml_gallocr_alloc_graph` again. If that fails, log error and
   return false.

The `prev_*` arrays are swapped with the current arrays at line
1383-1391 — a zero-copy dance that lets the next call compare
against this call's assignments.

### 5.7 Execution: `ggml_backend_sched_compute_splits`

`ggml_backend_sched_compute_splits` (`ggml-backend.cpp:1541-1725`)
runs splits **sequentially** in `split_id` order:

```
for (split_id in 0..n_splits):
    split_backend = sched->backends[split->backend_id]
    for each input in split->inputs[]:
        # wait for previous use of this (split_backend, cur_copy)
        if events[split_backend][cur_copy] != NULL:
            event_wait(split_backend, events[split_backend][cur_copy])
        else:
            synchronize(split_backend)

        # if input is GGML_TENSOR_FLAG_INPUT, copy synchronously
        # else if MUL_MAT_ID weights on host: copy only used experts
        # else: try cpy_tensor_async; fall back to sync copy
        ...

    if callback_eval == NULL:
        backend.graph_compute_async(split_backend, &split->graph)
    else:
        # sub-batch execution: walk nodes, call callback_eval,
        # synchronize per sub-batch, allow cancel
        ...

    if split->n_inputs > 0:
        if events[split_backend][cur_copy] != NULL:
            event_record(events[split_backend][cur_copy], split_backend)
```

Three notable special cases inside the per-input loop:

* **User-input copies** (`input->flags & GGML_TENSOR_FLAG_INPUT`,
  line 1560): synchronize the destination backend's event for the
  current copy *first*, then issue a *synchronous* `tensor_copy`.
  The comment explains: "inputs from the user must be copied
  immediately to prevent the user overwriting the data before the
  copy is done" (line 1561).
* **MoE expert partial copy** (lines 1576-1660): if the input is a
  weights tensor on a host buffer and the first node of the split is
  `GGML_OP_MUL_MAT_ID` whose `src[0]` is the input copy, the
  scheduler reads the `ids` tensor (`node->src[2]`), computes a
  bitset of used expert indices, and copies only the referenced
  experts. Consecutive expert ids are grouped into single
  `tensor_set_async` calls. The trailing `padding` (line 1627)
  ensures the last copied expert has clean bytes after it (the CUDA
  MMQ kernel reads a few extra bytes per expert).
* **Async-copy fallback** (lines 1661-1673): if
  `split_backend->iface.cpy_tensor_async` is NULL or returns false,
  synchronize `input_backend`, then synchronize the destination's
  event (or full backend), then issue a blocking `tensor_copy`.

### 5.8 Per-backend execution

Each backend's `graph_compute` vtable slot is invoked once per
split. Reference behaviors:

* **CPU** (`ggml-cpu.c:3350`): SPMD-with-barrier as documented in
  ARTX01 §5.2–5.3. Every thread runs every node in the split; a
  central barrier separates nodes.
* **CUDA** (`ggml-cuda.cu:4100`): optionally captures the split as
  a CUDA graph (after a 2-call warmup), else dispatches each node as
  a stream-queued kernel. CUDA graph capture is keyed by
  `ggml_cuda_graph_get_key(cgraph)` and gated by
  `ggml_cuda_graph_check_compability`.
* **Metal / Vulkan**: similar per-node dispatch on their respective
  command queues (see ARTX15–20).

---

## 6. Data Layout

### 6.1 The graph struct

`struct ggml_cgraph` (`ggml-impl.h:329-347`):

```c
struct ggml_cgraph {
    int size;             // capacity of nodes[]/leafs[]
    int n_nodes;          // current count
    int n_leafs;
    struct ggml_tensor ** nodes;     // ops to compute
    struct ggml_tensor ** grads;     // optional: gradients for nodes
    struct ggml_tensor ** grad_accs; // optional: gradient accumulators
    struct ggml_tensor ** leafs;     // constants / inputs (op == NONE)
    int32_t             * use_counts;// per-hash-slot use count
    struct ggml_hash_set visited_hash_set;
    enum ggml_cgraph_eval_order order; // LEFT_TO_RIGHT | RIGHT_TO_LEFT
    uint64_t uid;         // identity for cross-call re-alloc checks
};
```

The hash set doubles as the "visited" marker and the
`use_counts` index. `use_counts[hash_find(node)]` is the number of
downstream consumers — used by `ggml_can_fuse_subgraph_ext`
(`ggml.c:7618`) to determine if a node has external consumers.

### 6.2 The scheduler struct

`struct ggml_backend_sched` (`ggml-backend.cpp:774-828`) — 25+
fields, of which the hot ones are:

* `backends[16]`, `bufts[16]` — fixed-size arrays, max 16 backends.
* `hash_set` + `hv_tensor_backend_ids[hash_set.size]` — per-tensor
  assignment, indexed by `hash_id(tensor)`.
* `hv_tensor_copies[hash_set.size][n_backends][n_copies]` — flat
  3D array of tensor pointers, indexed by
  `tensor_id_copy(id, backend_id, copy_id)`.
* `node_backend_ids[graph_size]`, `leaf_backend_ids[graph_size]` —
  per-index assignment for the *graph copy*.
* `prev_node_backend_ids[]`, `prev_leaf_backend_ids[]` — for the
  reallocation short-circuit (Section 5.6).
* `splits[]` (dynamic array, capacity-doubling) — each split has
  `backend_id`, `i_start`, `i_end`, `inputs[30]`, `n_inputs`,
  `graph` (a `ggml_cgraph` view).
* `events[16][4]` — per-backend per-copy events. Allocated at
  scheduler creation if `parallel = true`.
* `graph_inputs[30]` + `n_graph_inputs` — pipeline-copied user
  inputs.

### 6.3 The split struct

`struct ggml_backend_sched_split` (`ggml-backend.cpp:764-772`):

```c
struct ggml_backend_sched_split {
    int backend_id;
    int i_start;
    int i_end;
    struct ggml_tensor * inputs[GGML_SCHED_MAX_SPLIT_INPUTS]; // 30
    int n_inputs;
    struct ggml_cgraph graph; // view into the user's graph
};
```

The `graph` field is a `ggml_graph_view` of the user's `cgraph`
(line 1413). Its `nodes` pointer aliases the user's array;
`n_nodes = i_end - i_start`.

### 6.4 The cplan struct (CPU only)

`struct ggml_cplan` (`ggml-cpu.h:12-25`):

```c
struct ggml_cplan {
    size_t    work_size; // computed by ggml_graph_plan
    uint8_t * work_data; // caller-allocated
    int n_threads;
    struct ggml_threadpool * threadpool;
    ggml_abort_callback abort_callback;
    void *              abort_callback_data;
    bool use_ref;
};
```

---

## 7. Memory Layout

### 7.1 Per-tensor assignment hash

`sched->hv_tensor_backend_ids` is a flat `int[]` of size
`hash_set.size`, initialized to `-1` at reset
(`ggml-backend.cpp:1826`). Looked up via the `tensor_backend_id(tensor)`
macro (`ggml-backend.cpp:831`):

```c
#define hash_id(tensor) ggml_hash_find_or_insert(&sched->hash_set, tensor)
#define tensor_backend_id(tensor) sched->hv_tensor_backend_ids[hash_id(tensor)]
```

`hash_id` *inserts* if not present — meaning every lookup has the side
effect of populating the hash set. This is intentional: it deduplicates
tensors across the leaf and node passes.

### 7.2 Per-tensor copies

`sched->hv_tensor_copies` is a flat `ggml_tensor **` of size
`hash_set.size * n_backends * n_copies`, indexed by the
`tensor_id_copy(id, backend_id, copy_id)` macro
(`ggml-backend.cpp:832`). Layout: `[hash_id][backend_id][copy_id]`.
Initialized to NULL at reset.

### 7.3 Split array

`sched->splits` is a `realloc`'d array starting at capacity 16,
doubling on overflow (`ggml-backend.cpp:1306-1311`). Maximum number
of splits is `graph_size` (one split per node, the degenerate case).
Each split is ~280 bytes (30 input pointers + cgraph + ints).

### 7.4 Graph copy

`sched->graph` is a separate `ggml_cgraph` whose `nodes[]` and
`leafs[]` arrays are `realloc`'d as needed (line 1399-1405). Size is
graph_size + `n_splits * GGML_SCHED_MAX_SPLIT_INPUTS * 2 * n_copies`
to accommodate the fake input-dependency nodes. The graph copy holds
the same node pointers as the user's graph (some of which now have
their `src[j]` rewritten to point at copies).

### 7.5 Context buffer for duplicate tensors

`sched->context_buffer` (`ggml-backend.cpp:816-817,
1769-1770`) is a flat `char[]` sized at
`max_splits * GGML_SCHED_MAX_SPLIT_INPUTS * 2 * sizeof(ggml_tensor)
+ ggml_graph_overhead_custom(graph_size, false)`. It backs
`sched->ctx`, the `ggml_context` that owns every duplicate tensor and
every fake input-dependency node. The context is torn down and
re-initialized at every `split_graph` call (line 1026-1031).

---

## 8. Parallelism Strategy

### 8.1 Sequential split execution

Splits are executed **strictly sequentially** in `split_id` order
(`ggml-backend.cpp:1549`). There is no mechanism for two independent
splits to run concurrently, even when their backends are different
(e.g., a CPU split could run in parallel with a CUDA split). The
scheduler treats the entire graph as a single serial chain of
splits.

This is a deliberate simplification: cross-split dependencies are
guaranteed by graph-node order, so sequential execution is trivially
correct. The cost is that hybrid CPU+GPU systems cannot exploit
concurrent execution of independent subgraphs.

### 8.2 Pipeline parallelism via N copies + events

When `parallel = true` at scheduler creation
(`ggml-backend.cpp:1732, 1751`), `n_copies =
GGML_SCHED_MAX_COPIES = 4`. For every cross-backend input and every
`GGML_TENSOR_FLAG_INPUT` graph input, the scheduler allocates 4
duplicates (one of which aliases the user's original as
`cur_copy`-0 initially).

Each execution rotates `cur_copy` (`ggml-backend.cpp:1869-1870`).
At the *end* of each split, an event is recorded on
`events[split_backend_id][cur_copy]` (line 1718). At the *start* of
the next execution's split, the scheduler waits on that event before
overwriting the input copies (line 1570-1575). This allows backend
compute on copy K to overlap with host→device copies on copy K+1.

The mechanism is per-(backend, copy), not per-tensor. One event per
backend per copy — total of `n_backends * n_copies` events (max
16*4 = 64).

### 8.3 Per-node parallelism (CPU only)

Inside the CPU backend's `graph_compute`, the SPMD-with-barrier
model (ARTX01-F02) parallelizes *within* a node across N threads.
This is the only intra-split parallelism in the entire stack. GPU
backends parallelize across warps / wavefronts inside each kernel,
which is opaque to the scheduler.

### 8.4 Per-backend parallelism is impossible

Because the scheduler is one C++ call stack with a single `for`
loop over splits, there is no way to dispatch split A to backend X
and split B to backend Y in parallel. To do so would require either
(a) two scheduler threads, or (b) the scheduler handing splits to
backend queues and waiting on futures. Neither exists in this code.

### 8.5 CUDA graph capture

For the CUDA backend, when conditions permit (single GPU, stable
graph topology, no incompatible ops), `ggml_backend_cuda_graph_compute`
(`ggml-cuda.cu:4100`) captures the entire split as a CUDA graph
after a 2-call warmup. Subsequent calls replay the captured graph
via `cudaGraphLaunch`. This is per-backend parallelism *within* a
split — but it's a CUDA-stream mechanism, invisible to the
scheduler.

---

## 9. SIMD / GPU Strategy

The scheduler itself contains **no SIMD and no GPU code**. All
kernel-level parallelism is delegated to the per-backend
`graph_compute`. The scheduler's only "GPU awareness" is:

* **Backend priority order**: lower index = higher prio. CPU is
  forced to be last. The user supplies GPU backends first.
* **`op_offload` hook** (`ggml-backend-impl.h:194-196`): a device
  method that returns true if the backend wants to claim an op even
  though its weights are on CPU. CUDA implements this; CPU does
  not.
* **`supports_op` + `supports_buft`**: the two-pronged capability
  check. `supports_op` says "can this backend run this op at all";
  `supports_buft` says "can this backend read tensors from this
  buffer type". Both must be true.
* **Event / async-copy vtable slots**: backends that support
  `cpy_tensor_async`, `event_record`, `event_wait` participate in
  pipeline parallelism. Backends that don't (CPU) serialize.

The CPU backend's `supports_op` (`ggml-cpu.cpp:423-475`) is the
reference: it returns true for view ops unconditionally, defers to
extra-buffer-types (AMX, KleidiAI) for tensors that live in those
buffers, and otherwise returns true for everything except a small
blocklist of unsupported (op, dtype) pairs.

---

## 10. Quantization Strategy

The scheduler has **no quantization-specific logic**. Quantization
matters to the scheduler only insofar as it affects:

* **`supports_op`** — backends advertise which dtypes they can
  compute. CPU's `MUL_MAT` requires `src1->type` to be F32 or the
  weight's `vec_dot_type` (`ggml-cpu.cpp:452-453`).
* **Buffer-type compatibility** — quantized weights live in backend-
  specific buffer types (e.g., CUDA's `ggml_backend_cuda_buffer_type`
  for device memory). `supports_buft` checks compatibility.
* **`vec_dot_type` conversion work buffer** — the CPU
  `ggml_graph_plan` estimates `work_size` for the activation
  conversion (F32 → Q8_0 etc.) per matmul node (`ggml-cpu.c:2848-2855`).
  This is the only place quantization enters the plan.
* **MoE expert partial copy** — the bitset-driven expert copy
  (Section 5.7) treats the weights tensor as an opaque byte range;
  it does not interpret the quantization format. The padding
  adjustment at line 1627 is "copy a bit extra to ensure there are
  no NaNs in the padding of the last expert — this is necessary for
  MMQ in the CUDA backend" (comment line 1633).

See ARTX06 for the per-quant block layouts and ARTX21 for how
quantized tensors are allocated into backend buffers.

---

## 11. Correctness Analysis

Static analysis found the following correctness-relevant properties.

### 11.1 Topological order is implicit but trusted

The scheduler does **not** re-verify topological order. It trusts
that `nodes[]` is in post-order (as `ggml_visit_parents_graph`
produces). Any code that constructs a `ggml_cgraph` by other means
(e.g., manually appending to `nodes[]`) must respect this. There is
no assertion that `node->src[j]` appears at a lower index than
`node` itself.

The `ggml_check_edges` helper (`ggml-impl.h:766`) exists for
debugging but is not called by the scheduler.

### 11.2 In-place `node->src[j]` rewriting

Pass 5 of `ggml_backend_sched_split_graph` rewrites
`node->src[j] = tensor_id_copy(...)` in-place
(`ggml-backend.cpp:1370`). This mutation is **persistent** — it
survives across calls. If the user reuses the same `ggml_cgraph`
across scheduler calls with different assignments, the rewritten
`src[j]` pointers will be re-rewritten on the next call. The
original tensor is preserved in `split->inputs[]` for the executor
to find, but the user's `tensor->src[j]` no longer points at the
user's original tensor.

This is correct in the normal use pattern (one `cgraph` per
inference, rebuilt from scratch each iteration), but it is a
correctness hazard for any code that holds onto a tensor pointer
and expects its `src[]` to remain stable across scheduler calls.

### 11.3 `hash_id` has insert side-effects

`#define hash_id(tensor) ggml_hash_find_or_insert(...)`
(`ggml-backend.cpp:830`) inserts if not present. This means a bare
`tensor_backend_id(tensor)` lookup will *create* a hash entry. The
scheduler relies on this for deduplication, but it means the hash
set grows monotonically within a `split_graph` call. The hash set
is reset only at `ggml_backend_sched_reset` (line 1825).

### 11.4 Pipeline parallelism requires events or fallback

When `parallel = true` but a backend lacks `event_new`
(`device->iface.event_new == NULL`, `ggml-backend.cpp:525`), the
scheduler's `events[b][c]` slots are NULL. The execution loop
checks for NULL and falls back to `ggml_backend_synchronize(split_backend)`
(line 1565, 1573, 1669). This is correct but serializes the
pipeline — the "pipeline parallelism" silently degrades to
"sequential execution with 4× memory use".

The CPU backend advertises no events (ARTX01-F01). Therefore, any
scheduler with `parallel = true` that includes a CPU backend will
serialize on CPU splits. This is undocumented.

### 11.5 Eval callback forces per-sub-batch synchronization

When `callback_eval` is set (`ggml-backend.cpp:1682-1713`), the
scheduler runs each split in sub-batches, calling
`ggml_backend_synchronize(split_backend)` after each sub-batch
(line 1706). This defeats CUDA graph capture and any backend-side
pipelining. The comment at line 1705 acknowledges: "TODO: pass
backend to the callback, then the user can decide if they want to
synchronize".

### 11.6 MoE expert copy reads `ids` tensor from the source backend

The MoE expert partial-copy path
(`ggml-backend.cpp:1576-1660`) reads `node->src[2]` (the ids
tensor) from the *source* backend, synchronously. It uses
`ggml_backend_tensor_get_async` followed by
`ggml_backend_synchronize(ids_backend)` (line 1606-1607) — a
blocking read. The bitset scan then determines which experts to
copy. This is correct but introduces a synchronous host→device
round-trip per split that contains a `MUL_MAT_ID` with
host-resident weights.

The `prev_ids_tensor` cache (line 1604) avoids re-reading the ids
if multiple inputs in the same split share the same ids tensor
(uncommon but possible).

### 11.7 `uid` is a 64-bit graph identity, not a hash

`cgraph->uid` is set to `ggml_graph_next_uid()` at the start of
`split_graph` (`ggml-backend.cpp:1033`) and per-split at line 1485.
It is a monotonically increasing counter, not a content hash.
Backends that cache compiled graphs (CUDA graph capture) key on
this uid. A graph rebuild that produces the same topology will get
a *different* uid, invalidating the cache. There is no mechanism to
re-use a uid for "same topology" graphs.

### 11.8 CPU `ggml_graph_plan` does not abort

The plan function (`ggml-cpu.c:2781-3019`) never returns an error.
It computes `work_size` and `n_threads` and returns. If a node is
unsupported, the plan still succeeds; the failure surfaces later
inside `ggml_graph_compute` when `ggml_compute_forward` dispatches
to a missing kernel (typically a `GGML_ABORT`). There is no
plan-time validation.

---

## 12. Optimization Analysis

### 12.1 Identified optimizations

| Optimization                                       | Where                                            | Notes                                                                |
| -------------------------------------------------- | ------------------------------------------------ | -------------------------------------------------------------------- |
| Five-pass backend assignment heuristic             | `ggml-backend.cpp:1014-1487`                     | Pass 1 anchors from pre-allocated weights; pass 2 stretches GPU; pass 3 upgrades by buft-equality; pass 4 backfills; pass 5 splits + inserts copies. |
| `prev_*_backend_ids` reallocation short-circuit    | `ggml-backend.cpp:1489-1506`                     | Skips `ggml_gallocr_reserve_n` entirely when assignments and buffer types are unchanged. |
| Pipeline parallelism (N copies + events)           | `ggml-backend.cpp:803-807, 1541-1725`            | Overlaps host↔device copies with backend compute. Up to 4× memory cost. |
| MoE expert partial copy                            | `ggml-backend.cpp:1576-1660`                     | Bitset scan over `ids` tensor; copies only used experts. Up to (n_experts/n_used)× bandwidth savings. |
| Hash-set dedup of tensor pointers                  | `ggml-impl.h:300-319`, `ggml-backend.cpp:830`    | Linear-probing with `ptr >> 4` hash; one indirection per assignment lookup. |
| `op_offload` hook for CPU→GPU offload              | `ggml-backend.cpp:919-926`, `ggml-backend-impl.h:194-196` | Lets a GPU backend claim an op whose weights are CPU-resident (host memory). |
| Per-backend `graph_optimize` hook                  | `ggml-backend.cpp:559-564, 1417`                 | Called per-split per-backend; lets each backend fuse / reorder its slice of the graph. |
| CUDA-graph capture (per-split)                     | `ggml-cuda.cu:4100-4157`                         | 2-call warmup; key on `cgraph->uid`. Captures entire split as a single `cudaGraphLaunch`. |
| Manual tensor→backend override API                 | `ggml-backend.cpp:1960-1967`                     | `ggml_backend_sched_set_tensor_backend` for expert users; persists across calls until `reset`. |
| Split-input capacity doubling                      | `ggml-backend.cpp:1306-1311`                     | `realloc` with 2× growth; avoids per-call allocation.                |
| Hash-set 25% margin                                | `ggml-alloc.c:826-828`                           | `min_hash_size += min_hash_size / 4` in `ggml_gallocr_reserve_n_impl` to reduce collisions. |

### 12.2 Optimizations *not* present (worth noting)

* **No cross-split parallelism.** Splits run sequentially; the
  scheduler cannot dispatch independent splits to different backends
  concurrently.
* **No graph-level optimization in the scheduler.** Constant
  folding, dead-code elimination, op fusion, and topological
  reordering are *not* done by `ggml_backend_sched`. They happen
  per-split per-backend inside `graph_optimize` if implemented.
* **No plan-time fusion across split boundaries.** Because
  `graph_optimize` is per-split, an op fusion that would cross a
  backend boundary (e.g., `MUL_MAT` on GPU followed by `ADD` on
  CPU) is impossible. The split boundary forces two separate
  `graph_compute` calls with a memory round-trip in between.
* **No critical-path scheduling.** The scheduler walks splits in
  node order; it does not prioritize splits whose outputs feed
  long critical chains.
* **No plan-time reuse of compiled graphs across scheduler calls.**
  Each `split_graph` call re-derives assignments. The `prev_*`
  short-circuit avoids re-allocation, but the assignment work
  itself is O(N·backends) per call.
* **No backend-aware graph construction.** The user builds the
  graph with no knowledge of which backend will run which op; the
  scheduler decides post-hoc. There is no way for the user to hint
  "build this op fused with the next one" at construction time.
* **No event pooling.** `events[16][4]` are allocated at scheduler
  creation and never freed until scheduler free, even if the graph
  never uses pipeline parallelism.

---

## 13. Architectural Strengths

1. **Clean three-tier vtable ABI.** The reg / device / backend /
   buffer-type / buffer vtable hierarchy
   (`ggml-backend-impl.h:17-230`) lets backends register as
   dynamically-loaded `.so` files and be discovered at runtime
   through `ggml_backend_init` + `ggml_backend_score`. The
   scheduler treats every backend uniformly through this ABI.

2. **CPU-as-last-backend convention.** The assert at
   `ggml-backend.cpp:1736` (`backends[n-1]` must be CPU type)
   guarantees a universal fallback. Every op the user can build is
   either run on a higher-prio GPU backend or falls through to CPU.
   This eliminates "no backend supports this op" failures for any
   op the CPU implements.

3. **Implicit cross-backend copy insertion.** The user does not
   need to know about backend boundaries. Pass 5 silently creates
   duplicate tensors and rewrites `node->src[j]` to point at them
   (`ggml-backend.cpp:1352-1370`). The split inputs are tracked
   separately for the executor to copy. This is a clean separation
   of concerns: the user builds one logical graph; the scheduler
   handles the physical realization.

4. **Five-pass assignment is locally optimal.** Pass 1 anchors on
   pre-allocated weights (so heavy matmuls stay with their
   weights); pass 2 stretches GPU assignments across CPU-compatible
   ops to minimize copies; pass 3 upgrades to higher-prio backends
   when buffer types match; pass 4 backfills; pass 5 splits. The
   result is that *most* graphs end up with the minimum number of
   cross-backend copies.

5. **Pipeline parallelism is opt-in and graceful.** `parallel =
   true` enables N copies + events. If a backend lacks event
   support, the mechanism gracefully degrades to synchronization
   (slower but correct). The user pays 4× memory for input copies
   only when they ask for it.

6. **Reallocation short-circuit.** The `prev_*_backend_ids`
   swap-dance (Section 5.6) avoids re-allocation when assignments
   are stable. For steady-state inference (same model, same batch
   size, same inputs), this means the second and subsequent calls
   skip `ggml_gallocr_reserve_n` entirely.

7. **Manual override API.** `ggml_backend_sched_set_tensor_backend`
   lets expert users pin a specific op to a specific backend,
   overriding the heuristic. The assignment persists across calls
   until `reset`. Useful for debugging and for hand-tuned hybrid
   execution.

8. **MoE expert partial copy.** The bitset-driven expert copy
   (Section 5.7) is a measurable bandwidth win for MoE models with
   sparse expert activation. The grouping of consecutive ids into
   single `tensor_set_async` calls (line 1624) reduces per-copy
   overhead.

9. **Per-split `graph_optimize` hook.** Each backend gets a chance
   to optimize its slice of the graph before execution. Vulkan
   uses this for plan-time fusion (ARTX18-F09); CUDA uses it for
   graph-capture setup (env-gated); CPU opts out.

10. **Hash-set dedup of tensor pointers.** The `visited_hash_set`
    in `ggml_cgraph` and the `hash_set` in `ggml_backend_sched`
    both use linear-probing with a `ptr >> 4` hash. O(1) membership
    and use-count updates. The 25% margin in `ggml_gallocr` keeps
    probe lengths short.

---

## 14. Architectural Weaknesses

### W1 — Sequential split execution

**Evidence:** `ggml-backend.cpp:1549` `for (int split_id = 0; split_id
< sched->n_splits; split_id++)` — single-threaded serial loop.

**Impact:** Independent splits cannot run concurrently. A hybrid
CPU+GPU system that has, say, a CPU split for tokenization followed
by a GPU split for matmul cannot overlap them. The CPU sits idle
while the GPU computes; the GPU sits idle while the CPU computes.

**Why it's hard to fix:** Cross-backend dependencies would need
explicit tracking (futures or per-tensor events). The scheduler
would need to dispatch splits to backend queues and wait. The
current single-call-stack design would have to become multi-threaded
or callback-driven.

### W2 — No graph-level optimization in the scheduler

**Evidence:** `ggml-backend.cpp:559-564` — `ggml_backend_graph_optimize`
is a one-line vtable dispatch. It is called *per-split* at line 1417,
*after* the split boundaries are determined. No fusion across
splits, no dead-code elimination, no constant folding at the
scheduler level.

**Impact:** Common patterns like `MUL_MAT (GPU) → ADD (CPU) →
RMS_NORM (GPU)` end up as three splits with two host↔device
round-trips. A scheduler-level fusion pass could either move the
ADD to the GPU (eliminating one split) or fuse `MUL_MAT + ADD` on
the GPU (eliminating one round-trip).

### W3 — Eval callback forces per-sub-batch synchronization

**Evidence:** `ggml-backend.cpp:1706` `ggml_backend_synchronize(split_backend)`
inside the eval-callback branch.

**Impact:** Any user of the eval callback (e.g., for inspecting
intermediate activations) loses all backend-side pipelining and
CUDA graph capture. The TODO at line 1705 acknowledges the issue.

### W4 — Synchronous fallback when events are unsupported

**Evidence:** `ggml-backend.cpp:1565, 1573, 1669` — NULL check on
`events[b][c]` followed by `ggml_backend_synchronize`.

**Impact:** The CPU backend has no events (ARTX01-F01). Any
scheduler with `parallel = true` that includes a CPU backend will
silently serialize on every CPU split, defeating the pipeline. The
4× memory cost of input copies is paid anyway. There is no warning
logged.

### W5 — In-place `node->src[j]` rewriting is a hidden mutation

**Evidence:** `ggml-backend.cpp:1370` `node->src[j] = tensor_id_copy(...)`.

**Impact:** The user's `ggml_cgraph` is mutated by the scheduler.
The original `src[j]` pointer is preserved in `split->inputs[]`,
but the user has no way to recover it from the graph itself. If
the user inspects `cgraph->nodes[i]->src[j]` after a scheduler
call, they see the *copy*, not the original. This is undocumented
behavior that breaks the principle of least surprise.

### W6 — Dead `graph_plan_*` vtable slots

**Evidence:** `ggml-backend-impl.h:120-127` —
`graph_plan_create`, `graph_plan_free`, `graph_plan_update`,
`graph_plan_compute` are declared with the comment "not used
currently". No backend implements them. The public API
`ggml_backend_graph_plan_create` (`ggml-backend.cpp:423-428`)
asserts `iface.graph_plan_create != NULL` — so any caller would
crash.

**Impact:** The shared interface promises a plan-and-compute
lifecycle that does not exist. The only real "plan" is the CPU's
`ggml_cplan` (`ggml-cpu.h:12-25`), which is CPU-specific and not
exposed through the vtable. This is a maintenance trap: future
contributors may try to use the vtable methods and find them
unimplemented.

### W7 — `GGML_SCHED_MAX_SPLIT_INPUTS = 30` is a hard cap

**Evidence:** `ggml-backend.cpp:757, 1291, 1367`.

**Impact:** A split with more than 30 cross-backend inputs forces
the scheduler to start a new split mid-backend
(`ggml-backend.cpp:1291-1299`). This can fragment a logical
backend's work into multiple splits, each paying its own
synchronization overhead. The cap is arbitrary and not tunable at
runtime.

### W8 — `GGML_SCHED_MAX_BACKENDS = 16` is a hard cap

**Evidence:** `ggml-backend.cpp:753`.

**Impact:** Systems with more than 16 backends (e.g., a large
multi-GPU cluster with CPU + 16 GPUs) would silently fail to
register all backends. The assert at `ggml-backend.cpp:1735` would
fire. Not a practical concern today but a fixed-size assumption.

### W9 — `op_offload` requires `is_host` weights

**Evidence:** `ggml-backend.cpp:919` — the condition
`src_backend_id == sched->n_backends - 1 && ggml_backend_buffer_is_host(src->buffer)`
restricts offload to host-resident weights.

**Impact:** A weight that lives in pinned host memory (CUDA host
buffer type, `is_host = true`) can be offloaded. A weight that
lives in regular device memory cannot (correctly — there's no
reason to "offload" what's already on the device). The restriction
is correct but limits the hook's applicability.

### W10 — Per-split `graph_optimize` cannot fuse across splits

**Evidence:** `ggml-backend.cpp:1417` calls
`ggml_backend_graph_optimize(sched->backends[split->backend_id],
&split->graph)` *after* the split boundaries are determined.

**Impact:** Fusion patterns that would cross a backend boundary
are invisible to `graph_optimize`. The scheduler's split decisions
constrain the optimizer's view. A scheduler-level optimizer that
runs *before* splitting could fuse cross-backend patterns by
choosing to run both ops on the same backend.

### W11 — `tensor_copy` macro has insert side-effect

**Evidence:** `ggml-backend.cpp:830` `#define hash_id(tensor)
ggml_hash_find_or_insert(...)` — *insert* semantics, not *find*.

**Impact:** Every `tensor_backend_id(tensor)` or
`tensor_copy(tensor, ...)` lookup mutates the hash set. Code that
reads `tensor_backend_id` for "is this assigned?" purposes will
*create* the entry, then see `-1` (uninitialized). The scheduler
relies on this side effect for dedup, but it makes the macros
unsafe to use outside `split_graph`.

### W12 — No backpressure on pipeline copies

**Evidence:** `ggml-backend.cpp:1869-1870` rotates `cur_copy`
unconditionally at each `alloc_graph` call.

**Impact:** If the user calls `alloc_graph` faster than backends
can consume copies, the rotation will wrap around and overwrite a
copy that's still in use. The event wait at line 1570 blocks
until the previous use completes, providing correctness, but
introduces a stall. There is no mechanism to skip a copy if it's
still in flight.

---

## 15. GwenLand Mapping

| GwenLand module | ADOPT / ADAPT / REJECT / MONITOR / DEFER | What | Reasoning |
| --------------- | ---------------------------------------- | ---- | --------- |
| `GATE`          | **ADOPT** | Flat `cgraph` with hash-set membership + use_counts | Simple, cache-friendly, O(1) membership. The `ptr >> 4` hash is cheap. |
| `GATE`          | **ADOPT** | Five-pass backend assignment heuristic | Locally optimal; minimizes cross-backend copies; anchors on weights. |
| `GATE`          | **ADOPT** | CPU-as-last-backend priority convention | Universal fallback; eliminates "no backend" failures. |
| `GATE`          | **ADOPT** | Implicit cross-backend copy insertion (with node rewriting) | Clean separation of logical graph from physical realization. |
| `GATE`          | **ADOPT** | Pipeline parallelism via N copies + per-(backend,copy) events | Overlap host↔device copies with compute. Graceful degradation when events absent. |
| `GATE`          | **ADOPT** | `prev_*_backend_ids` reallocation short-circuit | Avoids `reserve_n` on steady-state inference. |
| `GATE`          | **ADOPT** | `op_offload` hook for CPU→GPU offload | Lets GPU claim ops with host-resident weights without forcing a copy. |
| `GATE`          | **ADOPT** | MoE expert partial copy via bitset | Measurable bandwidth win for sparse MoE. |
| `GATE`          | **ADAPT** | Per-split `graph_optimize` hook | Keep the hook, but *also* add a scheduler-level optimizer that runs before splitting. |
| `GATE`          | **ADAPT** | Manual tensor→backend override API | Keep, but expose via a thread-safe API and document the persistence-until-reset semantics. |
| `GATE`          | **REJECT**| Sequential split execution | GwenLand should dispatch independent splits to different backends concurrently. |
| `GATE`          | **REJECT**| Per-sub-batch synchronization in eval callback | Pass the backend to the callback; let the user decide whether to sync. |
| `GATE`          | **REJECT**| Synchronous fallback when events are unsupported | Log a warning; consider disabling pipeline mode for that backend. |
| `GATE`          | **REJECT**| Dead `graph_plan_*` vtable slots | Either implement or remove. Don't ship dead API. |
| `GATE`          | **MONITOR**| `GGML_SCHED_MAX_SPLIT_INPUTS = 30` | Watch whether real workloads hit the cap; make tunable if so. |
| `GATE`          | **DEFER** | CUDA-graph-capture integration | Backend-specific; defer to glcuda. |
| `glproc`        | **ADOPT** | CPU `ggml_cplan` work-size estimation pattern | Per-op `work_size` computed once at plan time, allocated once, reused across runs. |
| `glcuda`        | **ADOPT** | CUDA-graph warmup + key-on-uid pattern | 2-call warmup avoids capturing unstable graphs; uid keying invalidates on topology change. |
| `glvulkan`      | **ADOPT** | Per-split `graph_optimize` for plan-time fusion | Vulkan's ~15-pattern fusion pass (ARTX18-F09) is the model to follow. |

---

## 16. Recommendations

### R1 — ADOPT flat `cgraph` with hash-set membership

**Priority:** Critical
**Difficulty:** S
**Dependencies:** none

GwenLand's GATE should define `struct gl_cgraph { nodes[], leafs[],
use_counts[], visited_hash_set, order, uid }`. Same layout, same
semantics: post-order DFS construction, hash-set dedup, `use_counts`
for fusion analysis. The `ptr >> 4` hash is fine for 16-byte-aligned
allocations; bump to `ptr >> 5` if GwenLand aligns to 32 bytes.

### R2 — ADOPT five-pass backend assignment

**Priority:** Critical
**Difficulty:** L
**Dependencies:** R1, backend interface design

Replicate the five-pass scheme: pass 1 anchors on pre-allocated
weights and inputs; pass 2 stretches GPU assignments up and down;
pass 3 upgrades to higher-prio backends with matching buffer types;
pass 4 backfills from `view_src` and `cur`; pass 5 splits and
inserts copies. The CPU-as-last-backend convention is essential.

### R3 — ADOPT implicit cross-backend copy insertion

**Priority:** High
**Difficulty:** M
**Dependencies:** R2

Keep the design where pass 5 rewrites `node->src[j]` to point at
duplicate tensors and tracks the originals in `split->inputs[]`.
Document the mutation prominently. Consider an alternative where
the rewrite happens in a *shadow* graph copy (the scheduler already
maintains one) so the user's original pointers are preserved.

### R4 — REJECT sequential split execution; ADOPT concurrent dispatch

**Priority:** High
**Difficulty:** XL
**Dependencies:** R2, R3

GwenLand's GATE should dispatch splits to backend queues and track
per-tensor readiness via events or futures. Two splits on different
backends with no data dependency should run concurrently. This is
the single biggest throughput win available for hybrid CPU+GPU
systems.

### R5 — ADOPT pipeline parallelism via N copies + events

**Priority:** High
**Difficulty:** L
**Dependencies:** R2, event system

Replicate the `n_copies = 4` design. Record events at split end;
wait on events at split start. Per-(backend, copy) event matrix.
Make the copy count tunable per scheduler.

### R6 — REJECT synchronous fallback when events are unsupported

**Priority:** Medium
**Difficulty:** S
**Dependencies:** R5

If a backend lacks event support and `parallel = true`, log a
warning and either (a) disable pipeline mode for that backend's
splits, or (b) refuse to create the scheduler. Silent degradation
to "4× memory with no benefit" is unacceptable.

### R7 — ADOPT scheduler-level graph optimizer

**Priority:** High
**Difficulty:** XL
**Dependencies:** R2

Add a `gl_graph_optimize(cgraph)` pass that runs *before* the
five-pass assignment. Implement at least: constant folding, dead-
code elimination, op fusion across potential split boundaries
(e.g., `MUL_MAT + ADD` → fuse on whichever backend supports both),
and topological reordering for critical-path scheduling.

### R8 — ADAPT per-split `graph_optimize` hook

**Priority:** Medium
**Difficulty:** M
**Dependencies:** R7

Keep the per-backend hook for backend-specific optimization (e.g.,
CUDA graph capture setup, Vulkan plan-time fusion). Run it after
the scheduler-level optimizer and after split boundaries are
determined.

### R9 — REJECT dead `graph_plan_*` vtable slots

**Priority:** Low
**Difficulty:** XS
**Dependencies:** none

Either remove `graph_plan_create` / `_free` / `_update` / `_compute`
from the vtable or implement them. The current "declared but unused"
state is a maintenance trap.

### R10 — ADOPT MoE expert partial copy

**Priority:** Medium
**Difficulty:** M
**Dependencies:** R3

Replicate the bitset-driven expert copy for MoE workloads. Read the
`ids` tensor, scan for used experts, group consecutive ids into
single `tensor_set_async` calls. Include the trailing-padding
adjustment for backends whose kernels read past expert boundaries.

### R11 — ADOPT `op_offload` hook

**Priority:** Medium
**Difficulty:** S
**Dependencies:** R2

Keep the `offload_op` device method. Lets a higher-prio backend
claim an op whose weights are host-resident, avoiding a forced
copy. Document the `is_host` precondition.

### R12 — ADOPT `prev_*_backend_ids` reallocation short-circuit

**Priority:** Medium
**Difficulty:** S
**Dependencies:** R2, allocator design

Replicate the swap-dance: keep two arrays (`current` and `prev`),
swap after each assignment pass, diff at the start of the next
allocation. Skip `reserve_n` when assignments and buffer types are
unchanged.

### R13 — ADOPT eval callback, but pass the backend

**Priority:** Low
**Difficulty:** S
**Dependencies:** R4

Keep the `eval_callback` API for inspecting intermediate
activations. Pass the backend to the callback so the user can
decide whether to synchronize. Document that per-sub-batch
synchronization is the user's choice, not the scheduler's default.

### R14 — DEFER CUDA-graph capture integration

**Priority:** Low
**Difficulty:** L
**Dependencies:** glcuda design

The CUDA-graph warmup + key-on-uid pattern is backend-specific.
Defer to glcuda's design pass. The scheduler's `cgraph->uid`
convention is the only contract GATE needs to provide.

---

## 17. Findings

### Finding ARTX22-F01

```
Finding ID:           ARTX22-F01
Category:             BACKEND_DESIGN
Engine:               Shared (scheduler)
Component:            Backend interface vtables
Source File:          ggml/src/ggml-backend-impl.h
Function:             struct ggml_backend_buffer_type_i / buffer_i / backend_i / device_i / reg_i
Lines:                17-230
Summary:              Five-tier vtable ABI (reg/device/backend + buffer_type/buffer) lets
                      backends register as dynamically-loaded .so files.
Observation:          Each tier is a struct of function pointers with explicit
                      "(optional)" markers in comments. The scheduler treats every
                      backend uniformly through device->supports_op / supports_buft /
                      offload_op and backend->graph_compute / event_record / event_wait /
                      cpy_tensor_async. Optional slots default to NULL; callers check
                      for NULL before invoking. The CPU backend sets most async/event
                      slots to NULL; CUDA implements all of them.
Evidence:             ggml-backend-impl.h:17-29 (buffer_type_i),
                      41-62 (buffer_i), 105-140 (backend_i),
                      160-202 (device_i), 214-224 (reg_i).
Architectural Impact: Clean separation of concerns. Adding a backend means
                      implementing 5 vtables; no scheduler changes needed. The ABI
                      is stable across backend versions (api_version field in reg).
Correctness Impact:   None. The vtable is a dispatch mechanism.
Optimization Type:    Indirect dispatch via function pointers (stable targets,
                      branch-predictor friendly).
GwenLand Target:      GATE
Recommendation:       ADOPT. Replicate the five-tier vtable hierarchy in GATE's
                      backend interface.
Priority:             Critical
Difficulty:           M
Dependencies:         none
Confidence:           High
```

### Finding ARTX22-F02

```
Finding ID:           ARTX22-F02
Category:             EXECUTION_GRAPH
Engine:               Shared
Component:            Graph data structure
Source File:          ggml/src/ggml-impl.h, ggml/src/ggml.c
Function:             struct ggml_cgraph, ggml_visit_parents_graph
Lines:                ggml-impl.h:329-347; ggml.c:7098-7164
Summary:              ggml_cgraph is a flat array of node pointers + hash-set
                      membership + use_counts; built by recursive post-order DFS.
Observation:          The graph struct holds nodes[], leafs[], optional grads[]/
                      grad_accs[], use_counts[] indexed by hash slot, and a
                      visited_hash_set. Construction via ggml_visit_parents_graph
                      performs post-order DFS over tensor->src[0..9], deduplicating
                      via the hash set, incrementing use_counts for each edge. The
                      resulting nodes[] array is in topological order. Views
                      (RESHAPE/PERMUTE/TRANSPOSE/VIEW) are treated as ops and
                      included in nodes[].
Evidence:             ggml-impl.h:329-347 (struct); ggml.c:7098-7164 (DFS);
                      ggml.c:7127-7140 (use_counts increment); ggml.c:7142-7161
                      (leaf vs node classification).
Architectural Impact: O(1) membership check, O(1) use-count lookup, cache-friendly
                      linear iteration. No adjacency list; edges are implicit via
                      tensor->src[].
Correctness Impact:   Topological order is *trusted*, not verified. Code that
                      manually constructs cgraph by appending to nodes[] must
                      respect post-order or the scheduler will execute ops before
                      their inputs are ready.
Optimization Type:    Flat array layout + hash-set dedup.
GwenLand Target:      GATE
Recommendation:       ADOPT. Equivalent struct in GATE with same construction
                      semantics.
Priority:             Critical
Difficulty:           S
Dependencies:         none
Confidence:           High
```

### Finding ARTX22-F03

```
Finding ID:           ARTX22-F03
Category:             EXECUTION_GRAPH
Engine:               Shared
Component:            Graph plan vtable
Source File:          ggml/src/ggml-backend-impl.h, ggml/src/ggml-cpu/ggml-cpu.c
Function:             ggml_backend_i.graph_plan_*, ggml_graph_plan
Lines:                ggml-backend-impl.h:120-127; ggml-cpu.c:2781-3019
Summary:              The shared backend vtable declares graph_plan_create / _free /
                      _update / _compute but explicitly comments them "not used
                      currently". The only real "graph plan" is the CPU-specific
                      ggml_cplan / ggml_graph_plan, which lives in the CPU backend
                      and is not exposed through the vtable.
Observation:          The vtable slots would, if implemented, allow a backend to
                      precompute a plan for a graph and reuse it across calls. No
                      backend implements them. The public API
                      ggml_backend_graph_plan_create (ggml-backend.cpp:423-428)
                      asserts the slot is non-NULL, so any caller would crash. The
                      CPU's ggml_cplan is a separate, CPU-only struct (ggml-cpu.h:12-25)
                      with work_size, work_data, n_threads, threadpool, abort_callback,
                      use_ref. It is computed by ggml_graph_plan, which walks every
                      node and sums per-op work_size estimates (the large switch at
                      ggml-cpu.c:2816-3003).
Evidence:             ggml-backend-impl.h:120-127 (declared, commented "not used
                      currently"); ggml-backend.cpp:423-442 (public wrappers, asserts
                      non-NULL); ggml-cpu.h:12-25 (cplan struct); ggml-cpu.c:2781-3019
                      (ggml_graph_plan implementation).
Architectural Impact: The shared interface promises a plan-and-compute lifecycle
                      that does not exist. The CPU plan is opaque to the scheduler.
                      The scheduler has no plan-time concept at all.
Correctness Impact:   None. The dead slots are simply never called.
Optimization Type:    None (absence of optimization).
GwenLand Target:      GATE
Recommendation:       REJECT the dead vtable slots; either remove or implement.
                      ADAPT the CPU's per-op work_size estimation pattern into a
                      shared plan-time pass.
Priority:             Medium
Difficulty:           S
Dependencies:         R7 (scheduler-level optimizer)
Confidence:           High
```

### Finding ARTX22-F04

```
Finding ID:           ARTX22-F04
Category:             EXECUTION_GRAPH
Engine:               Shared (scheduler)
Component:            Split execution
Source File:          ggml/src/ggml-backend.cpp
Function:             ggml_backend_sched_compute_splits
Lines:                1541-1725
Summary:              Splits are executed strictly sequentially in split_id order;
                      no two splits run concurrently even on different backends.
Observation:          The function is a single C++ for-loop over sched->n_splits.
                      Each iteration: copies inputs, calls
                      backend.graph_compute_async, records event. There is no
                      mechanism to dispatch split A to backend X and split B to
                      backend Y in parallel. Even when parallel=true (pipeline
                      mode), the parallelism is across *calls* (different copies
                      of the same input), not across *splits within a call*.
Evidence:             ggml-backend.cpp:1549 (single for-loop);
                      1677-1681 (graph_compute_async call);
                      1717-1721 (event record).
Architectural Impact: Hybrid CPU+GPU systems cannot overlap independent CPU and
                      GPU work. The CPU sits idle during GPU splits and vice
                      versa. This is the biggest throughput gap in the scheduler.
Correctness Impact:   None. Sequential execution is trivially correct.
Optimization Type:    None (absence of optimization).
GwenLand Target:      GATE
Recommendation:       REJECT. GwenLand's GATE should dispatch independent splits
                      to backend queues and track per-tensor readiness via events
                      or futures.
Priority:             High
Difficulty:           XL
Dependencies:         R2 (assignment), event system design
Confidence:           High
```

### Finding ARTX22-F05

```
Finding ID:           ARTX22-F05
Category:             EXECUTION_GRAPH
Engine:               Shared (scheduler)
Component:            Backend assignment heuristic
Source File:          ggml/src/ggml-backend.cpp
Function:             ggml_backend_sched_split_graph
Lines:                1014-1487
Summary:              Five-pass backend assignment: (1) anchor on pre-allocated
                      weights/inputs/view_src; (2) expand GPU up+down, then rest
                      up+down; (3) upgrade to higher-prio buft-compatible backend
                      or pick best for unassigned; (4) backfill src from dst/
                      view_src; (5) split graph + insert cross-backend copies.
Observation:          Pass 1 calls ggml_backend_sched_backend_id_from_cur (line 878)
                      which checks tensor's own buffer, then view_src's buffer, then
                      INPUT flag (forces CPU = n_backends-1), then weights-flagged
                      sources (with op_offload escape hatch). Pass 2 runs four
                      sub-passes that propagate the current backend to neighboring
                      unassigned nodes; GPU sub-passes skip CPU as a boundary so
                      GPU stretches across CPU-compatible ops. Pass 3 either picks
                      the backend with the most supported inputs (for unassigned)
                      or upgrades to a higher-prio backend with the same buft
                      (for assigned). Pass 4 backfills sources from their
                      consumer's backend or view_src. Pass 5 walks the node array,
                      starts a new split on backend-id change, and creates
                      duplicate tensors for cross-backend sources, rewriting
                      node->src[j] in-place.
Evidence:             ggml-backend.cpp:1036-1070 (pass 1); 1078-1150 (pass 2,
                      four sub-passes); 1160-1211 (pass 3); 1214-1243 (pass 4);
                      1246-1376 (pass 5, split + copy insertion).
Architectural Impact: Locally optimal: minimizes cross-backend copies by
                      anchoring on weights and stretching GPU assignments. The
                      CPU-as-last-backend convention guarantees a universal
                      fallback.
Correctness Impact:   The heuristic is correct but not globally optimal. Pass 3's
                      "upgrade" check (line 1190-1209) uses buft pointer equality,
                      not buft compatibility — two backends with different buft
                      pointers but compatible memory layouts will not upgrade.
                      This is documented at line 1156 as a "more strict requirement".
Optimization Type:    Multi-pass greedy assignment.
GwenLand Target:      GATE
Recommendation:       ADOPT. Replicate the five-pass scheme. Consider adding a
                      global pass 0 (scheduler-level optimizer) that fuses
                      cross-backend patterns before assignment.
Priority:             Critical
Difficulty:           L
Dependencies:         R1 (cgraph), R2 (assignment)
Confidence:           High
```

### Finding ARTX22-F06

```
Finding ID:           ARTX22-F06
Category:             BACKEND_DESIGN
Engine:               Shared (scheduler)
Component:            Backend priority convention
Source File:          ggml/src/ggml-backend.cpp
Function:             ggml_backend_sched_new
Lines:                1727-1794 (assert at 1736)
Summary:              The last backend in the backends[] array MUST be CPU type;
                      CPU is the universal fallback at the lowest priority.
Observation:          ggml_backend_sched_new asserts
                      ggml_backend_dev_type(backends[n_backends - 1]) ==
                      GGML_BACKEND_DEVICE_TYPE_CPU. The five-pass assignment
                      treats the last backend specially: pass 2 skips it as a
                      boundary when expanding GPU; pass 1's INPUT-flag branch
                      (line 902-906) forces INPUT tensors to the last backend
                      (assumed CPU). The op_offload hook (line 919) checks
                      src_backend_id == n_backends - 1 to detect "weights on CPU".
Evidence:             ggml-backend.cpp:1736 (assert); 902-906 (INPUT forces
                      last backend); 919 (op_offload checks for last backend);
                      1087, 1108 (pass 2 skips CPU as boundary).
Architectural Impact: Guarantees every op has a fallback. Eliminates "no backend
                      supports this op" failures for any op CPU implements. The
                      convention is hardcoded; non-CPU last backends are
                      rejected.
Correctness Impact:   None. The assert is a runtime check.
Optimization Type:    None.
GwenLand Target:      GATE
Recommendation:       ADOPT. Keep the CPU-as-last-backend convention.
Priority:             High
Difficulty:           XS
Dependencies:         R2
Confidence:           High
```

### Finding ARTX22-F07

```
Finding ID:           ARTX22-F07
Category:             EXECUTION_GRAPH
Engine:               Shared (scheduler)
Component:            Cross-backend copy insertion
Source File:          ggml/src/ggml-backend.cpp
Function:             ggml_backend_sched_split_graph (pass 5)
Lines:                1352-1371
Summary:              When a node's source lives on a different backend whose
                      buffer type is not supported by the current split's backend,
                      the scheduler creates a duplicate tensor and rewrites
                      node->src[j] in-place to point at the duplicate.
Observation:          Pass 5 walks every node's sources. For each source on a
                      different backend (and not buffer-supported), it calls
                      ggml_dup_tensor_layout to create a duplicate in sched->ctx,
                      records it in hv_tensor_copies via tensor_id_copy, appends
                      the original to split->inputs[], and rewrites node->src[j]
                      = tensor_id_copy(...). The rewrite is *persistent* — it
                      survives across calls. The duplicate is allocated into the
                      split's backend's buffer by the subsequent gallocr pass.
Evidence:             ggml-backend.cpp:1352-1371 (rewrite loop);
                      1354-1365 (duplicate creation);
                      1370 (in-place src rewrite).
Architectural Impact: User's graph is mutated by the scheduler. The original
                      tensor is preserved in split->inputs[] but not in the graph
                      itself. This is a hidden mutation that breaks the principle
                      of least surprise.
Correctness Impact:   Correct in the normal use pattern (rebuild graph each
                      iteration). A correctness hazard for code that holds tensor
                      pointers and expects src[] to remain stable across
                      scheduler calls.
Optimization Type:    Implicit copy insertion via pointer rewriting.
GwenLand Target:      GATE
Recommendation:       ADOPT the design but consider doing the rewrite in a shadow
                      graph copy (the scheduler already maintains one) so the
                      user's original pointers are preserved.
Priority:             High
Difficulty:           M
Dependencies:         R3
Confidence:           High
```

### Finding ARTX22-F08

```
Finding ID:           ARTX22-F08
Category:             EXECUTION_GRAPH
Engine:               Shared (scheduler)
Component:            Pipeline parallelism
Source File:          ggml/src/ggml-backend.cpp
Function:             ggml_backend_sched (struct), ggml_backend_sched_compute_splits,
                      ggml_backend_sched_alloc_graph
Lines:                803-807 (struct fields); 1541-1725 (execution);
                      1869-1870 (copy rotation)
Summary:              When parallel=true, the scheduler maintains n_copies=4
                      duplicates of every cross-backend input and graph input,
                      rotates cur_copy each call, and uses per-(backend, copy)
                      events to overlap host↔device copies with backend compute.
Observation:          The struct has events[16][4] (per-backend per-copy), n_copies
                      (1 or 4), cur_copy, next_copy, graph_inputs[30]. At each
                      alloc_graph call, cur_copy = next_copy; next_copy = (next_copy+1)
                      % n_copies. At split start, the executor waits on
                      events[split_backend][cur_copy] (or synchronizes if NULL). At
                      split end, if the split had inputs, it records
                      events[split_backend][cur_copy]. This allows backend compute
                      on copy K to overlap with host→device copies on copy K+1.
Evidence:             ggml-backend.cpp:803-807 (struct); 1751 (n_copies init);
                      1781-1785 (event creation); 1869-1870 (rotation);
                      1562-1575 (wait); 1717-1721 (record).
Architectural Impact: Overlaps copies with compute. 4× memory cost for input
                      copies. Graceful degradation when events absent (silently
                      serializes — see F13).
Correctness Impact:   The event wait at split start ensures the previous use of
                      the same copy has completed before overwrite. Correct as
                      long as backends honor event_wait semantics.
Optimization Type:    Asynchronous execution via double-buffering (quad-buffering).
GwenLand Target:      GATE
Recommendation:       ADOPT. Make n_copies tunable per scheduler. Log a warning
                      when a backend lacks event support and parallel=true.
Priority:             High
Difficulty:           L
Dependencies:         R5 (pipeline), event system
Confidence:           High
```

### Finding ARTX22-F09

```
Finding ID:           ARTX22-F09
Category:             MEMORY_PATTERN
Engine:               Shared (scheduler)
Component:            Allocation reuse
Source File:          ggml/src/ggml-backend.cpp
Function:             ggml_backend_sched_alloc_splits, ggml_backend_sched_split_graph
Lines:                1383-1391 (swap); 1489-1539 (alloc_splits)
Summary:              The scheduler maintains prev_node_backend_ids[] and
                      prev_leaf_backend_ids[] arrays, swapped with the current
                      arrays after each split_graph call. alloc_splits diffs
                      current vs prev; if no buffer-type changes, it skips
                      ggml_gallocr_reserve_n entirely.
Observation:          After split_graph, the current and prev array pointers are
                      swapped (lines 1383-1391). On the next alloc_splits call,
                      the diff loop (1489-1506) checks each node and leaf: if the
                      backend_id changed AND the buffer type at the new id differs
                      from the buffer type at the old id, mark
                      backend_ids_changed. If not changed, attempt
                      ggml_gallocr_alloc_graph directly; only on failure or
                      changed-ids does it fall back to reserve_n + alloc_graph.
Evidence:             ggml-backend.cpp:1383-1391 (swap); 1489-1506 (diff);
                      1509-1536 (alloc or reserve+alloc).
Architectural Impact: Steady-state inference (same model, same batch size) skips
                      the expensive reserve_n call. First call and any topology
                      change pay the full reserve+alloc cost.
Correctness Impact:   The diff checks buffer-type equality, not backend-id equality.
                      Two different backends with the same buffer type (e.g., two
                      CUDA devices sharing a buffer type) will not trigger
                      reallocation. This is correct: same buffer type means same
                      allocator, so offsets remain valid.
Optimization Type:    Diff-based short-circuit.
GwenLand Target:      GATE
Recommendation:       ADOPT. Replicate the swap-dance and diff check.
Priority:             Medium
Difficulty:           S
Dependencies:         R2, allocator design
Confidence:           High
```

### Finding ARTX22-F10

```
Finding ID:           ARTX22-F10
Category:             EXECUTION_GRAPH
Engine:               Shared (scheduler)
Component:            Graph optimization
Source File:          ggml/src/ggml-backend.cpp, ggml/src/ggml-cpu/ggml-cpu.cpp,
                      ggml/src/ggml-cuda/ggml-cuda.cu
Function:             ggml_backend_graph_optimize, ggml_backend_sched_split_graph
Lines:                ggml-backend.cpp:559-564, 1417; ggml-cpu.cpp:209;
                      ggml-cuda.cu:4184-4210
Summary:              The scheduler performs no graph optimization itself. The
                      per-backend graph_optimize hook is called per-split *after*
                      split boundaries are determined, so it cannot fuse across
                      splits. CPU opts out (NULL); CUDA gates behind env var;
                      Vulkan implements ~15 patterns.
Observation:          ggml_backend_graph_optimize (line 559) is a one-line vtable
                      dispatch. It is called at line 1417 inside split_graph, once
                      per split, on the split's graph view. The CPU vtable sets it
                      to NULL (ggml-cpu.cpp:209). CUDA sets it to
                      ggml_backend_cuda_graph_optimize which checks
                      getenv("GGML_CUDA_GRAPH_OPT") and returns early if not "1"
                      (ggml-cuda.cu:4196-4203). Vulkan implements a real fusion
                      pass (ARTX18-F09). No constant folding, dead-code
                      elimination, or topological reordering happens at the
                      scheduler level.
Evidence:             ggml-backend.cpp:559-564 (dispatch); 1417 (per-split call);
                      ggml-cpu.cpp:209 (NULL); ggml-cuda.cu:4184-4210 (env-gated).
Architectural Impact: Common patterns (MUL_MAT+ADD, ADD+ACT, ROPE+MV) that span
                      a split boundary cannot be fused. The scheduler's split
                      decisions constrain the optimizer's view.
Correctness Impact:   None. Unfused execution is correct.
Optimization Type:    None at scheduler level; per-backend at split level.
GwenLand Target:      GATE
Recommendation:       ADAPT. Keep the per-backend hook for backend-specific
                      optimization. Add a scheduler-level optimizer (R7) that runs
                      before splitting.
Priority:             High
Difficulty:           XL
Dependencies:         R7
Confidence:           High
```

### Finding ARTX22-F11

```
Finding ID:           ARTX22-F11
Category:             BACKEND_DESIGN
Engine:               Shared (scheduler)
Component:            Op offload hook
Source File:          ggml/src/ggml-backend-impl.h, ggml/src/ggml-backend.cpp
Function:             ggml_backend_device_i.offload_op,
                      ggml_backend_sched_backend_id_from_cur
Lines:                ggml-backend-impl.h:194-196; ggml-backend.cpp:919-926
Summary:              The op_offload device method lets a higher-prio backend
                      claim an op whose weights are CPU-resident (host memory),
                      avoiding a forced copy.
Observation:          offload_op is an optional device method that returns true
                      if the backend "wants to run an operation, even if the
                      weights are allocated in an incompatible buffer" (impl
                      comment line 195). The scheduler checks it in pass 1: if
                      the weights are on the last backend (CPU) and is_host, and
                      a higher-prio backend supports the op AND offload_op
                      returns true, assign to the higher-prio backend. The hook
                      is gated by sched->op_offload (set at scheduler creation
                      from the op_offload parameter). CPU does not implement
                      offload_op (returns false via the NULL check at line 635).
                      CUDA implements it for expensive ops like MUL_MAT.
Evidence:             ggml-backend-impl.h:194-196 (declaration);
                      ggml-backend.cpp:919-926 (call site);
                      633-640 (NULL-safe wrapper);
                      1789 (op_offload stored from constructor param).
Architectural Impact: Lets GPU backends claim CPU-resident-weight ops without
                      forcing a weight copy. The GPU reads the weights via
                      host-visible memory (e.g., CUDA unified memory or pinned
                      host). Useful for MoE where not all experts fit on GPU.
Correctness Impact:   None. The hook is a hint; the scheduler still checks
                      supports_op.
Optimization Type:    Hint-based op placement.
GwenLand Target:      GATE
Recommendation:       ADOPT. Keep the offload_op hook. Document the is_host
                      precondition.
Priority:             Medium
Difficulty:           S
Dependencies:         R2
Confidence:           High
```

### Finding ARTX22-F12

```
Finding ID:           ARTX22-F12
Category:             EXECUTION_GRAPH
Engine:               Shared (scheduler)
Component:            Eval callback
Source File:          ggml/src/ggml-backend.cpp
Function:             ggml_backend_sched_compute_splits (callback branch),
                      ggml_backend_sched_set_eval_callback
Lines:                1682-1713; 1917-1921
Summary:              When an eval callback is set, the scheduler runs each split
                      in sub-batches, calling the callback to ask whether the user
                      wants to observe each node, then synchronizing the backend
                      after each sub-batch.
Observation:          The callback has signature bool (*)(tensor *t, bool ask,
                      void *user_data). When ask=true, the scheduler wants to
                      know if the user wants to observe t; if true, the scheduler
                      extends the sub-batch to include t. When ask=false, the
                      scheduler passes t to the user for observation; if the user
                      returns false, the scheduler cancels the graph compute.
                      After each sub-batch's graph_compute_async, the scheduler
                      calls ggml_backend_synchronize(split_backend) (line 1706) —
                      this defeats CUDA graph capture and any backend-side
                      pipelining. The TODO at line 1705 acknowledges: "pass
                      backend to the callback, then the user can decide if they
                      want to synchronize".
Evidence:             ggml-backend.cpp:1682-1713 (sub-batch loop);
                      1706 (forced sync); 1705 (TODO);
                      1917-1921 (set_eval_callback API).
Architectural Impact: Any user of the eval callback loses all backend-side
                      pipelining. Useful for debugging and differential testing;
                      costly for production.
Correctness Impact:   None. The sync is conservative but correct.
Optimization Type:    None (the sync is the absence of an optimization).
GwenLand Target:      GATE
Recommendation:       ADAPT. Keep the callback API but pass the backend to the
                      user; let the user decide whether to sync. Document the
                      cost.
Priority:             Low
Difficulty:           S
Dependencies:         R4
Confidence:           High
```

### Finding ARTX22-F13

```
Finding ID:           ARTX22-F13
Category:             CORRECTNESS_SHORTCUT
Engine:               Shared (scheduler)
Component:            Cross-backend synchronization fallback
Source File:          ggml/src/ggml-backend.cpp
Function:             ggml_backend_sched_compute_splits
Lines:                1562-1575, 1664-1672
Summary:              When events[split_backend][cur_copy] is NULL (backend lacks
                      event support), the scheduler falls back to
                      ggml_backend_synchronize(split_backend), silently
                      serializing the pipeline.
Observation:          The execution loop checks events[b][c] != NULL at three
                      points: before user-input copy (line 1562), before
                      non-input copy (line 1570), and inside the async-copy
                      fallback (line 1666). In all three, the fallback is
                      ggml_backend_synchronize(split_backend). The CPU backend
                      advertises no events (ARTX01-F01), so any scheduler with
                      parallel=true that includes a CPU backend will serialize
                      on every CPU split. No warning is logged. The 4× memory
                      cost of input copies is paid regardless.
Evidence:             ggml-backend.cpp:1562-1566 (user-input sync fallback);
                      1570-1574 (non-input sync fallback); 1664-1672 (async-copy
                      sync fallback); 525 (event_new returns NULL if unsupported).
Architectural Impact: Pipeline parallelism silently degrades to sequential
                      execution with 4× memory use. Users who enable parallel
                      mode expecting speedup may instead see slowdown.
Correctness Impact:   None. Synchronization is correct but conservative.
Optimization Type:    None (degraded mode).
GwenLand Target:      GATE
Recommendation:       REJECT. Log a warning when a backend lacks event support
                      and parallel=true. Consider disabling pipeline mode for
                      that backend's splits.
Priority:             Medium
Difficulty:           S
Dependencies:         R5, R6
Confidence:           High
```

### Finding ARTX22-F14

```
Finding ID:           ARTX22-F14
Category:             EXECUTION_GRAPH
Engine:               Shared (scheduler)
Component:            MoE expert partial copy
Source File:          ggml/src/ggml-backend.cpp
Function:             ggml_backend_sched_compute_splits (MoE branch)
Lines:                1576-1660
Summary:              When a split input is a MUL_MAT_ID weights tensor on a host
                      buffer, the scheduler reads the ids tensor, computes a
                      bitset of used expert indices, and copies only the
                      referenced experts, grouping consecutive ids into single
                      tensor_set_async calls.
Observation:          The condition at line 1578-1583 checks: split->graph.n_nodes
                      > 0, input->buffer usage is WEIGHTS, input->buffer is_host,
                      node->op == GGML_OP_MUL_MAT_ID, node->src[0] == input_cpy.
                      If all true, the scheduler reads node->src[2] (the ids
                      tensor) via ggml_backend_tensor_get_async + synchronize
                      (line 1606-1607), builds a bitset of used expert indices
                      (line 1610-1618), then walks the bitset, grouping
                      consecutive used ids into ranges and calling copy_experts
                      for each range (line 1624-1660). copy_experts uses
                      ggml_backend_tensor_set_async with an offset and size
                      covering the range, plus a small padding at the end "to
                      ensure there are no NaNs in the padding of the last expert
                      — this is necessary for MMQ in the CUDA backend" (comment
                      line 1633). A prev_ids_tensor cache (line 1604) avoids
                      re-reading the ids if multiple inputs share it.
Evidence:             ggml-backend.cpp:1576-1583 (condition); 1585-1586
                      (n_expert, expert_size); 1604-1621 (ids read + bitset);
                      1624-1660 (grouped copy).
Architectural Impact: Up to (n_experts / n_used)× bandwidth savings for sparse
                      MoE. The bitset scan is O(n_experts + n_ids). The grouped
                      copy reduces per-call overhead.
Correctness Impact:   The padding read at the end (line 1627-1628) reads up to
                      512 bytes past the last expert. This is safe if the source
                      buffer has at least that much trailing space, which the
                      allocator should guarantee. The comment cites a CUDA MMQ
                      requirement.
Optimization Type:    Sparse data transfer with bitmap compaction.
GwenLand Target:      GATE
Recommendation:       ADOPT. Replicate for MoE workloads. Include the trailing
                      padding adjustment. Document the buffer-overrun assumption.
Priority:             Medium
Difficulty:           M
Dependencies:         R3
Confidence:           High
```

### Finding ARTX22-F15

```
Finding ID:           ARTX22-F15
Category:             BACKEND_DESIGN
Engine:               Shared (scheduler)
Component:            Manual backend override
Source File:          ggml/src/ggml-backend.cpp
Function:             ggml_backend_sched_set_tensor_backend,
                      ggml_backend_sched_get_tensor_backend
Lines:                1960-1976
Summary:              Expert users can manually pin a specific tensor to a
                      specific backend, overriding the five-pass heuristic. The
                      assignment persists across calls until reset.
Observation:          set_tensor_backend (line 1960) writes to
                      tensor_backend_id(node) (which uses hash_id, so it has the
                      insert side-effect). It sets sched->is_reset = false (line
                      1966), which prevents the next split_graph from clearing
                      the assignment. Pass 1 of split_graph (line 1040-1042,
                      1048-1050) checks "do not overwrite user assignments" by
                      skipping nodes whose tensor_backend_id is already != -1.
                      get_tensor_backend (line 1969) reads the assignment,
                      returning NULL if unassigned (-1).
Evidence:             ggml-backend.cpp:1960-1967 (set); 1969-1976 (get);
                      1040-1042 (pass 1 leaf check); 1048-1050 (pass 1 node
                      check).
Architectural Impact: Lets expert users hand-tune hybrid execution. Useful for
                      debugging (force an op to CPU to isolate a GPU bug) and for
                      workloads where the heuristic is suboptimal.
Correctness Impact:   The user is responsible for ensuring the assigned backend
                      supports the op. The scheduler does not re-verify. If the
                      backend lacks supports_op, execution will fail inside
                      graph_compute.
Optimization Type:    Manual placement override.
GwenLand Target:      GATE
Recommendation:       ADOPT. Keep the API. Document the persistence-until-reset
                      semantics and the lack of supports_op re-verification.
Priority:             Medium
Difficulty:           S
Dependencies:         R2
Confidence:           High
```

### Finding ARTX22-F16

```
Finding ID:           ARTX22-F16
Category:             OTHER
Engine:               Shared (scheduler)
Component:            Hash set sizing and hash function
Source File:          ggml/src/ggml-impl.h, ggml/src/ggml.c
Function:             ggml_hash_find, ggml_hash_insert, ggml_hash_find_or_insert,
                      ggml_new_graph_custom
Lines:                ggml-impl.h:254-319; ggml.c:7341
Summary:              The visited_hash_set is sized at 2× the graph capacity to
              hold both nodes and leafs. The hash function is ptr >> 4 (drops
              the low 4 alignment bits). Linear probing with wraparound.
Observation:          ggml_new_graph_custom (ggml.c:7335) computes hash_size =
              ggml_hash_size(size * 2) — doubled because both nodes and leafs
              share the same hash set. The hash function (ggml-impl.h:254-257) is
              (size_t)(uintptr_t)p >> 4, exploiting the fact that ggml_tensor
              pointers are at least 16-byte aligned. Collision resolution is
              linear probing with wraparound (line 264-265). ggml_hash_find_or_insert
              (line 300-319) inserts if not present, returns index if present,
              GGML_ABORT if table is full. The 25% margin in ggml_gallocr
              (ggml-alloc.c:826-828) keeps probe lengths short for the
              allocator's hash set, but the cgraph's visited_hash_set has no such
              margin — it relies on the 2× sizing.
Evidence:             ggml-impl.h:254-257 (hash function); 259-272 (find with
              linear probing); 300-319 (find_or_insert); ggml.c:7341 (2× sizing).
Architectural Impact: O(1) membership and use-count updates. The ptr >> 4 hash
              is cheap (one shift). Linear probing is cache-friendly. The 2×
              sizing keeps load factor below 0.5 in the worst case (n_nodes +
              n_leafs == size).
Correctness Impact:   ggml_hash_insert and ggml_hash_find_or_insert call
              GGML_ABORT if the table is full (line 297, 317). This is a hard
              failure: a graph with more unique tensors than 2×size will crash
              the process. The 2× sizing should prevent this for well-formed
              graphs, but a user who manually adds nodes beyond cgraph->size
              could hit it.
Optimization Type:    Linear-probing hash set with cheap hash function.
GwenLand Target:      GATE
Recommendation:       ADOPT. Keep the design. Consider growing the hash set
              dynamically if load factor exceeds 0.75 instead of aborting.
Priority:             Low
Difficulty:           S
Dependencies:         R1
Confidence:           High
```

---

## 18. Unknowns

* **U1**. Whether the five-pass assignment heuristic produces the
  globally optimal assignment for real llama.cpp workloads. The
  heuristic is locally optimal (pass 3's upgrade is greedy) but
  cannot undo earlier decisions. Requires running the scheduler
  with debug logging (`GGML_SCHED_DEBUG=1`) on real graphs and
  comparing to a brute-force optimum on small graphs.

* **U2**. How much throughput is lost to sequential split execution
  (W1 / F04) on hybrid CPU+GPU systems. Requires profiling on a
  system with both backends active, measuring GPU idle time during
  CPU splits and vice versa. Static analysis cannot determine this.

* **U3**. Whether the `prev_*_backend_ids` diff check (F09) correctly
  handles the case where two different backends share the same
  buffer type pointer (e.g., two CUDA devices with the same
  `ggml_backend_cuda_buffer_type` static). The diff checks `bufts[id]
  != bufts[prev_id]`, so same-pointer would skip reallocation. This
  is correct *if* the two backends share the same allocator, which
  is the intent. Requires verification on a multi-GPU system.

* **U4**. Whether the MoE expert partial copy (F14) correctly handles
  the case where the `ids` tensor contains duplicate expert indices.
  The bitset scan marks each expert as used at most once, but the
  `MUL_MAT_ID` op may rely on the order of ids. Static analysis
  suggests the copy is order-independent (only the set of used
  experts matters), but this needs verification against the
  `MUL_MAT_ID` kernel semantics.

* **U5**. Whether the in-place `node->src[j]` rewrite (F07 / W5)
  causes problems for any real llama.cpp code path that holds tensor
  pointers across scheduler calls. The normal pattern (rebuild graph
  each iteration) is safe, but caching code may break. Requires
  auditing llama.cpp's higher layers.

* **U6**. The actual benefit of pipeline parallelism (F08) on real
  workloads. The 4× memory cost is significant; the throughput
  benefit depends on the ratio of copy time to compute time per
  split. Requires benchmarking with `parallel=true` vs `false` on
  CPU+GPU and GPU+GPU systems.

* **U7**. Whether the `op_offload` hook (F11) is actually used by
  the CUDA backend in practice. The hook exists in the vtable, but
  the audit did not trace CUDA's `offload_op` implementation in
  detail. Requires reading `ggml-cuda.cu`'s device interface
  (out of scope for ARTX22).

* **U8**. Whether the eval callback's per-sub-batch synchronization
  (F12 / W3) is acceptable for production use. Some llama.cpp
  features (e.g., speculative decoding) may rely on the eval
  callback. Requires checking llama.cpp callers of
  `ggml_backend_sched_set_eval_callback`.

* **U9**. Whether the `graph_plan_*` vtable slots (F03 / W6) were
  ever implemented and removed, or were never implemented. Requires
  git archaeology. The comment "not used currently" suggests intent
  to implement later.

* **U10**. Whether the 2× hash-set sizing (F16) is sufficient for
  graphs with very high leaf-to-node ratios. The worst case is
  `n_nodes + n_leafs == size`, giving a 0.5 load factor. Linear
  probing degrades gracefully, but the GGML_ABORT on full table
  (ggml-impl.h:297, 317) is a hard failure. Requires stress-testing
  with adversarial graph topologies.

---

## 19. References

| Reference | File                                                | Function / Symbol                                          | Lines            |
| --------- | --------------------------------------------------- | ---------------------------------------------------------- | ---------------- |
| R01       | `ggml/src/ggml-impl.h`                              | `struct ggml_cgraph`                                       | 329-347          |
| R02       | `ggml/src/ggml-impl.h`                              | `struct ggml_hash_set`                                     | 226-230          |
| R03       | `ggml/src/ggml-impl.h`                              | `ggml_hash_find` / `_insert` / `_find_or_insert`           | 259-319          |
| R04       | `ggml/src/ggml-impl.h`                              | `ggml_hash` (ptr >> 4)                                     | 254-257          |
| R05       | `ggml/src/ggml-impl.h`                              | `ggml_node_get_use_count`, `ggml_can_fuse`, `ggml_check_edges` | 629-766       |
| R06       | `ggml/src/ggml.c`                                   | `ggml_visit_parents_graph`                                 | 7098-7164        |
| R07       | `ggml/src/ggml.c`                                   | `ggml_build_forward_impl` / `_expand` / `_select`          | 7166-7201        |
| R08       | `ggml/src/ggml.c`                                   | `ggml_build_backward_expand`                               | 7203-7300        |
| R09       | `ggml/src/ggml.c`                                   | `ggml_graph_nbytes` / `_overhead_custom`                   | 7309-7333        |
| R10       | `ggml/src/ggml.c`                                   | `ggml_new_graph_custom` / `_graph`                         | 7335-7382        |
| R11       | `ggml/src/ggml.c`                                   | `ggml_graph_view`                                          | 7384-7400        |
| R12       | `ggml/src/ggml.c`                                   | `ggml_graph_cpy` / `_dup` / `_reset` / `_clear`            | 7402-7508        |
| R13       | `ggml/src/ggml.c`                                   | `ggml_graph_print`                                         | 7568-7594        |
| R14       | `ggml/src/ggml.c`                                   | `ggml_can_fuse_subgraph_ext`                               | 7618-7675        |
| R15       | `ggml/src/ggml.c`                                   | `ggml_graph_dump_dot`                                      | 7723-7848        |
| R16       | `ggml/src/ggml.c`                                   | `ggml_set_input` / `_output` / `_param` / `_loss`          | 7852-7869        |
| R17       | `ggml/src/ggml-backend-impl.h`                      | `struct ggml_backend_buffer_type_i`                        | 17-29            |
| R18       | `ggml/src/ggml-backend-impl.h`                      | `struct ggml_backend_buffer_i`                             | 41-62            |
| R19       | `ggml/src/ggml-backend-impl.h`                      | `struct ggml_backend_i` (with dead `graph_plan_*`)         | 105-140          |
| R20       | `ggml/src/ggml-backend-impl.h`                      | `struct ggml_backend_device_i`                             | 160-202          |
| R21       | `ggml/src/ggml-backend-impl.h`                      | `struct ggml_backend_reg_i`                                | 214-224          |
| R22       | `ggml/src/ggml-backend.cpp`                         | `ggml_backend_tensor_copy` / `_copy_async`                 | 477-519          |
| R23       | `ggml/src/ggml-backend.cpp`                         | `ggml_backend_event_*`                                     | 523-557          |
| R24       | `ggml/src/ggml-backend.cpp`                         | `ggml_backend_graph_optimize` (dispatch)                   | 559-564          |
| R25       | `ggml/src/ggml-backend.cpp`                         | `struct ggml_backend_sched`                                | 774-828          |
| R26       | `ggml/src/ggml-backend.cpp`                         | `hash_id` / `tensor_backend_id` / `tensor_id_copy` macros  | 830-833          |
| R27       | `ggml/src/ggml-backend.cpp`                         | `ggml_backend_sched_backend_from_buffer`                  | 845-865          |
| R28       | `ggml/src/ggml-backend.cpp`                         | `ggml_backend_sched_backend_id_from_cur`                   | 878-933          |
| R29       | `ggml/src/ggml-backend.cpp`                         | `ggml_backend_sched_split_graph` (5-pass)                  | 1014-1487        |
| R30       | `ggml/src/ggml-backend.cpp`                         | `ggml_backend_sched_alloc_splits`                          | 1489-1539        |
| R31       | `ggml/src/ggml-backend.cpp`                         | `ggml_backend_sched_compute_splits`                        | 1541-1725        |
| R32       | `ggml/src/ggml-backend.cpp`                         | `ggml_backend_sched_new` (CPU-last assert)                 | 1727-1794        |
| R33       | `ggml/src/ggml-backend.cpp`                         | `ggml_backend_sched_reset` / `_reserve` / `_alloc_graph`   | 1821-1881        |
| R34       | `ggml/src/ggml-backend.cpp`                         | `ggml_backend_sched_graph_compute` / `_async`              | 1883-1902        |
| R35       | `ggml/src/ggml-backend.cpp`                         | `ggml_backend_sched_synchronize`                           | 1904-1915        |
| R36       | `ggml/src/ggml-backend.cpp`                         | `ggml_backend_sched_set_eval_callback`                     | 1917-1921        |
| R37       | `ggml/src/ggml-backend.cpp`                         | `ggml_backend_sched_get_n_splits` / `_n_copies` / `_n_backends` | 1923-1936   |
| R38       | `ggml/src/ggml-backend.cpp`                         | `ggml_backend_sched_set_tensor_backend` / `_get_tensor_backend` | 1960-1976 |
| R39       | `ggml/src/ggml-backend.cpp`                         | `ggml_backend_cpu_buffer_type` / `_from_ptr`               | 2211-2372        |
| R40       | `ggml/src/ggml-cpu/ggml-cpu.c`                      | `ggml_graph_plan`                                          | 2781-3019        |
| R41       | `ggml/src/ggml-cpu/ggml-cpu.c`                      | `ggml_cpu_try_fuse_ops`                                    | 3026-3058        |
| R42       | `ggml/src/ggml-cpu/ggml-cpu.c`                      | `ggml_graph_compute_thread`                                | 3060-3133        |
| R43       | `ggml/src/ggml-cpu/ggml-cpu.c`                      | `ggml_graph_compute`                                       | 3350-3425        |
| R44       | `ggml/src/ggml-cpu/ggml-cpu.c`                      | `ggml_graph_compute_with_ctx`                              | 3427-3433        |
| R45       | `ggml/include/ggml-cpu.h`                           | `struct ggml_cplan`                                        | 12-25            |
| R46       | `ggml/src/ggml-cpu/ggml-cpu.cpp`                    | `ggml_backend_cpu_device_supports_op`                      | 423-475          |
| R47       | `ggml/src/ggml-cpu/ggml-cpu.cpp`                    | `ggml_backend_cpu_device_supports_buft`                    | 477-480          |
| R48       | `ggml/src/ggml-cpu/ggml-cpu.cpp`                    | `ggml_backend_cpu_device_i` (vtable, graph_optimize=NULL)  | 482-495          |
| R49       | `ggml/src/ggml-cuda/ggml-cuda.cu`                   | `ggml_backend_cuda_graph_compute` (CUDA graph capture)     | 4100-4157        |
| R50       | `ggml/src/ggml-cuda/ggml-cuda.cu`                   | `ggml_backend_cuda_event_record` / `_wait`                 | 4159-4182        |
| R51       | `ggml/src/ggml-cuda/ggml-cuda.cu`                   | `ggml_backend_cuda_graph_optimize` (env-gated)             | 4184-4210        |
| R52       | `ggml/include/ggml-backend.h`                       | `ggml_backend_sched_*` public API                          | 305-352          |
| R53       | `ggml/src/ggml-alloc.c`                             | `ggml_gallocr_reserve_n_impl` (25% hash margin)            | 824-849          |
| R54       | `ggml/src/ggml-alloc.c`                             | `ggml_gallocr_alloc_graph_impl` (lifetime tracking)        | 717-822          |
