# ARTX01 — Generic CPU Core

**Audited commit:** `555881ebc8b0fc0402b30e09258a32a7bfd13c52`
**Last updated:** 2026-07-25
**Auditor:** Percival
**Target GwenLand module:** `glproc` (core dispatch), `GATE` (graph execution)

---

## 1. Executive Summary

The generic CPU backend of llama.cpp is the *contract layer* between the
graph scheduler (`ggml_graph_compute`) and the per-ISA kernels. It owns
no SIMD itself; instead, it provides:

1. A **type-traits table** (`type_traits_cpu[GGML_TYPE_COUNT]`) that maps
   every tensor dtype to its `from_float`, `vec_dot`, `vec_dot_type`, and
   `nrows` parameters.
2. A **backend interface** (`ggml_backend_cpu_i` / `_device_i` / `_reg_i`)
   that plugs the CPU into ggml's backend registry, with `async=false`,
   `events=false`, `buffer_from_host_ptr=true`.
3. A **graph executor** (`ggml_graph_compute`) that runs every node on
   every thread under a per-node barrier, with dynamic chunk stealing
   via an atomic counter for the heavy matmul path.
4. A **fusion stub** (`ggml_cpu_try_fuse_ops`) that today fuses exactly
   one pattern (`RMS_NORM + MUL`).
5. An **extension hook** (`extra_buffer_type`) that lets AMX, KleidiAI,
   SpacemiT, and Repack override `compute_forward` for ops they claim.

The backend is *intentionally synchronous*. There is no event system,
no async tensor copy, no graph optimizer hook in the CPU backend
itself. Optimization lives in the kernels (ARTX02–ARTX06) and in the
scheduler (ARTX22), not here.

For GwenLand, the architectural decisions worth **ADOPT**ing are the
type-traits table, the extra-buffer-type extension hook, and the
dynamic-chunk atomic-counter scheme. The decisions worth **REJECT**ing
are the per-node central barrier (limits short-op throughput) and the
"every thread runs every node" SPMD model (no per-node parallelism
control).

---

## 2. Purpose

Provide a CPU execution backend for the ggml graph that:

* dispatches every `ggml_op` to a CPU implementation,
* parallelizes each op across N worker threads,
* defers to per-ISA kernels via function pointers,
* allows optional accelerators (AMX, KleidiAI, SpacemiT, Repack) to
  override individual ops,
* exposes a standard `ggml_backend` interface to the rest of ggml.

It is **not** responsible for: graph construction, graph optimization,
memory allocation (delegated to `ggml-alloc.c`), or kernel selection
across backends (delegated to the scheduler — ARTX22).

---

## 3. Source Files

| File                                       | Lines  | Role                                                      |
| ------------------------------------------ | ------ | --------------------------------------------------------- |
| `ggml/src/ggml-cpu/ggml-cpu.c`             | 3895   | Type-traits table, threading, graph executor, op dispatch |
| `ggml/src/ggml-cpu/ggml-cpu.cpp`           | 707    | Backend / device / registry interface                     |
| `ggml/src/ggml-cpu/ggml-cpu-impl.h`        | 539    | `ggml_compute_params`, per-ISA intrinsic shims            |
| `ggml/src/ggml-cpu/traits.{cpp,h}`         | 38+38  | `tensor_traits` / `extra_buffer_type` C++ interfaces      |
| `ggml/src/ggml-cpu/common.h`               | 95     | Tile constants, fp16/bf16 converters, thread range helper |
| `ggml/src/ggml-cpu/vec.{cpp,h}`            | 613+1570 | Scalar + SIMD elementwise primitives, `vec_dot_*`        |
| `ggml/src/ggml-cpu/ops.{cpp,h}`            | 12004+124 | Reference + SIMD op implementations                      |
| `ggml/src/ggml-cpu/quants.{c,h}`           | 1339+106 | Generic quantize / dequantize / vecdot skeletons         |
| `ggml/src/ggml-threading.{cpp,h}`          | small  | `ggml_critical_section_*` (the only public threading API)|

> Note: the audit prompt's file names (`ggml-cpu-icelake.cpp`,
> `ggml-cpu-amd.cpp`, `ggml-cpu-arm.cpp`, `ggml-cpu-aarch64.cpp`,
> `ggml-cpu-quants.c`) no longer exist at this commit. The per-ISA
> code was factored into `arch/<isa>/{cpu-feats.cpp, quants.c,
> repack.cpp}`. See the README's Structural Drift Notice.

---

## 4. Architecture Overview

```
                ┌────────────────────────────────────────────────┐
                │   ggml-cpu.cpp : ggml_backend_cpu_reg / _i /    │
                │                  _device_i  /  _reg_i           │
                │   (plugs CPU into ggml backend registry)        │
                └────────────────────────────────────────────────┘
                              │
                              ▼
                ┌────────────────────────────────────────────────┐
                │   ggml-cpu.c : ggml_graph_compute              │
                │   ├─ ggml_threadpool (N workers + main)        │
                │   ├─ per-node barrier (cache-aligned atomic)   │
                │   └─ ggml_compute_forward(node)                │
                └────────────────────────────────────────────────┘
                              │
            ┌─────────────────┼──────────────────┐
            ▼                 ▼                  ▼
   ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
   │ extra_buffer_type│  │  type_traits_cpu │  │  op dispatch     │
   │  (AMX/KleidiAI/  │  │  [GGML_TYPE_COUNT│  │  switch(op)      │
   │   SpacemiT/      │  │  ]               │  │  → ggml_compute_ │
   │   Repack)        │  │  .vec_dot,       │  │    forward_*     │
   │                  │  │  .from_float,    │  │                  │
   │ supports_op()    │  │  .vec_dot_type,  │  │                  │
   │ compute_forward()│  │  .nrows          │  │                  │
   └──────────────────┘  └──────────────────┘  └──────────────────┘
                              │
                              ▼
                ┌────────────────────────────────────────────────┐
                │   arch/<isa>/quants.c                          │
                │   (x86, arm, loongarch, powerpc, riscv, s390,  │
                │    wasm) — actual SIMD kernels                 │
                └────────────────────────────────────────────────┘
```

Key design points:

* **No polymorphism in C**. The type-traits table is a flat array
  indexed by `enum ggml_type`. Dispatch is a single indexed load +
  indirect call. No virtual tables, no RTTI.
* **Extension via C++ virtuals only at the "extra buffer" boundary**.
  `tensor_traits` and `extra_buffer_type` (in `traits.h`) are C++
  abstract classes. The CPU backend downcasts `buft->context` to
  invoke their `supports_op` / `compute_forward`. This is the *only*
  C++ in the dispatch hot path.
* **Synchronous backend**. The `ggml_backend_i` vtable in
  `ggml-cpu.cpp:193` sets `set_tensor_async`, `get_tensor_async`,
  `cpy_tensor_async`, `synchronize`, `event_record`, `event_wait`
  all to `NULL`. The CPU is, by contract, blocking-only.

---

## 5. Execution Flow

### 5.1 Top-level entry

`ggml_backend_cpu_graph_compute` (`ggml-cpu.cpp:170`)

1. Build a `ggml_cplan` via `ggml_graph_plan(cgraph, n_threads, threadpool)`.
2. If the cached `work_data` is too small, reallocate.
3. Call `ggml_graph_compute(cgraph, &cplan)`.

### 5.2 Graph execution

`ggml_graph_compute` (`ggml-cpu.c:3350`)

1. If no threadpool was supplied, create a disposable one via
   `ggml_threadpool_new_impl`. Otherwise, reset `cgraph`, `cplan`,
   `current_chunk=0`, `abort=-1`, `ec=SUCCESS` on the existing pool.
2. Branch on threading backend:
   * **OpenMP path** (`GGML_USE_OPENMP`): spawn `#pragma omp parallel`,
     each thread calls `ggml_graph_compute_thread(&workers[ith])`.
   * **Custom pthread path**: call `ggml_graph_compute_kickoff` (bumps
     `n_graph` and wakes workers via `cond`), then the main thread
     runs `ggml_graph_compute_thread(&workers[0])` itself.
3. After all threads exit, clear NUMA affinity on main thread and
   return `threadpool->ec`.

### 5.3 Per-thread worker

`ggml_graph_compute_thread` (`ggml-cpu.c:3060`)

```
for node_n in [0, n_nodes):
    if abort == node_n: break
    if node has GGML_TENSOR_FLAG_COMPUTE clear: skip
    n_fused = ggml_cpu_try_fuse_ops(cgraph, node_n, params, cplan)
    if n_fused > 0:
        node_n += n_fused
    else:
        ggml_compute_forward(&params, node)
    if ith == 0 and abort_callback fires:
        abort = node_n + 1; ec = ABORTED
    if not last node:
        ggml_barrier(threadpool)
ggml_barrier(threadpool)  // final barrier
```

Every thread runs the **same loop**. Parallelism happens *inside*
`ggml_compute_forward`, not across nodes. This is the SPMD-with-barrier
model.

### 5.4 Op dispatch

`ggml_compute_forward` (`ggml-cpu.c:1711`)

1. If op is `NONE` or tensor is empty: return.
2. Ask every registered `extra_buffer_type` if it claims this op via
   `ggml_cpu_extra_compute_forward`. If yes, the extra's
   `tensor_traits->compute_forward` runs and the function returns.
3. Otherwise, a large `switch (tensor->op)` dispatches to
   `ggml_compute_forward_<op>(params, tensor)`.

### 5.5 Matmul hot path

`ggml_compute_forward_mul_mat` (`ggml-cpu.c:1254`)

1. If op carries hint `GGML_HINT_SRC0_IS_HADAMARD`, dispatch to
   `ggml_compute_forward_fwht` (fast Walsh-Hadamard transform) and
   return. This is the only op-hint fast path.
2. If `src1->type != vec_dot_type` (e.g., weights are Q4_0 → vec_dot
   expects Q8_0 activations), every thread cooperatively converts
   `src1` rows into `params->wdata` using `from_float`. Threads
   partition the work by blocks: `ne10_block_start = (ith * ne10/bs) / nth`.
3. **Barrier**. No thread may start chunked matmul until conversion
   finishes.
4. **Optional llamafile sgemm** (compiled in via `GGML_USE_LLAMAFILE`,
   disabled on `__ARM_FEATURE_MATMUL_INT8` builds). If it accepts the
   op, it runs and returns; otherwise falls through.
5. Compute chunk grid:
   * `chunk_size = 16` default, `64` if `nr0 == 1 || nr1 == 1`
     (GEMV-like).
   * `nchunk0 = ceil(nr0 / chunk_size)`, `nchunk1 = ceil(nr1 / chunk_size)`.
   * If `nchunk0 * nchunk1 < nth * 4` or system is NUMA: collapse to
     `nchunk0 = nth, nchunk1 = 1` (or vice versa) — one chunk per
     thread to keep memory local.
6. **Dynamic chunk stealing**: each thread starts at `current_chunk = ith`,
   processes its chunk, then `current_chunk = atomic_fetch_add(...)`
   to grab the next. Exits when `current_chunk >= nchunk0 * nchunk1`.
7. **Inside a chunk** (`ggml_compute_forward_mul_mat_one_chunk`,
   `ggml-cpu.c:1164`): tile-blocked by `blck_0 = 16, blck_1 = 16`. For
   each `(iir0, iir1)` tile, invoke `vec_dot` (function pointer from
   `type_traits_cpu[type]`) with `num_rows_per_vec_dot` rows at once
   (1 generally; 2 on ARM I8MM builds).

---

## 6. Data Layout

### 6.1 Tensor descriptor

A `ggml_tensor` carries `ne[GGML_MAX_DIMS]` (element counts) and
`nb[GGML_MAX_DIMS]` (byte strides). The CPU backend requires, for the
matmul path:

| Constraint                                   | Source                |
| -------------------------------------------- | --------------------- |
| `nb00 == ggml_type_size(src0->type)`         | `ggml-cpu.c:1282`     |
| `nb10 == ggml_type_size(src1->type)`         | `ggml-cpu.c:1283`     |
| `nb0 == sizeof(float)` (dst is F32)          | `ggml-cpu.c:1286`     |
| `nb0 <= nb1 <= nb2 <= nb3` (dst not permuted)| `ggml-cpu.c:1287`     |
| `ne0 == ne01`, `ne1 == ne11`, ...            | `ggml-cpu.c:1276-1279`|

In other words: src0 must be row-contiguous in its innermost dim,
src1 must be row-contiguous in its innermost dim, and dst must be
contiguous in the standard ggml sense. Transposed inputs must be
materialized via `GGML_OP_CONT`/`GGML_OP_CPY` first.

### 6.2 Activation conversion (`wdata`)

When `src1->type != vec_dot_type`, src1 is re-laid-out into
`params->wdata` as a contiguous `vec_dot_type` tensor with strides
`nbw0 = type_size(vec_dot_type)`, `nbw1 = row_size(vec_dot_type, ne10)`,
`nbw2 = nbw1 * ne11`, `nbw3 = nbw2 * ne12`. This is a fully contiguous
NCHW-style layout that lets the `vec_dot` kernel use unit-stride loads
inside a row.

The conversion happens inside `ggml_compute_forward_mul_mat` at
`ggml-cpu.c:1344-1355`. Threads partition along the `ne10` axis
(activation row length) in units of `ggml_blck_size(vec_dot_type)`.

### 6.3 Quantized weight layout

Source: `ggml/src/ggml-common.h` block definitions (referenced from
`quants.h`). Each quant format defines a fixed-size block (e.g.,
`QK4_0 = 32`, `QK8_0 = 32`, `QK_K = 256`). Within a block: scales,
zero points (where applicable), and packed weights. Blocks are
contiguous along the row. ARTX06 covers this in detail.

---

## 7. Memory Layout

### 7.1 Work buffer (`cplan.work_data`)

A single flat `uint8_t[]` allocated by the backend and resized lazily
(`ggml-cpu.cpp:175-184`). Size is computed up-front by
`ggml_graph_plan`. Inside, it stores converted activations (`wdata`)
for matmuls and any per-op scratch.

### 7.2 Threadpool

`struct ggml_threadpool` is allocated via `ggml_aligned_malloc(sizeof(...))`
(`ggml-cpu.c:3279`). The three hot atomics — `n_barrier`,
`n_barrier_passed`, `current_chunk` — are declared with
`GGML_CACHE_ALIGN` (64 bytes), so each sits in its own cache line.
This is the only explicit false-sharing mitigation in the structure.

### 7.3 Per-thread state

`struct ggml_compute_state` (`ggml-cpu.c:507`) holds `ith`,
`cpumask[GGML_MAX_N_THREADS]`, `threadpool` back-pointer, and (without
OpenMP) `thrd`, `last_graph`, `pending`. Workers are allocated as a
flat array `workers[n_threads]`.

### 7.4 Precomputed tables

`ggml-cpu.c:80-86` declares four global tables:
`ggml_table_f32_f16[1<<16]` (256 KB),
`ggml_table_f32_e8m0_half[1<<8]` (1 KB),
`ggml_table_f32_ue4m3[1<<8]` (1 KB),
`ggml_table_gelu_f16[1<<16]` (128 KB),
`ggml_table_gelu_quick_f16[1<<16]` (128 KB).

These are populated in `ggml_cpu_init` (`ggml-cpu.c:3835-3854`). They
exist because (a) GELU activations are computed by table lookup from
f16 input, and (b) the FP16 → FP32 conversion is faster as a 256 KB
LUT than via intrinsic on some ISAs.

### 7.5 Cache line constant

`#define GGML_CACHE_LINE 64` (`ggml-cpu.c:60`). Hardcoded — the source
comment notes the intent to use `std::hardware_destructive_interference_size`
once the code moves to C++.

---

## 8. Parallelism Strategy

### 8.1 Threading backend selection

Two mutually exclusive threading backends, selected at compile time:

| Backend              | When defined            | Workers              |
| -------------------- | ----------------------- | -------------------- |
| OpenMP               | `GGML_USE_OPENMP`       | `#pragma omp parallel`|
| Custom pthread       | otherwise               | `pthread_create`     |

OpenMP is simpler but offers less control over polling and CPU
affinity. The custom backend supports `poll`, `prio`, `cpumask`,
`paused` parameters via `ggml_threadpool_params`.

### 8.2 Worker spin loop (custom backend)

`ggml_graph_compute_secondary_thread` (`ggml-cpu.c:3201`):

1. Set NUMA affinity for `state->ith`.
2. Loop:
   a. `ggml_graph_compute_check_for_work(state)`:
      - Poll up to `n_rounds = 1024 * 128 * poll` iterations calling
        `ggml_thread_cpu_relax()` (yield/pause).
      - If work appears, set `state->pending = true`.
      - Otherwise, lock `threadpool->mutex`, wait on
        `threadpool->cond`.
   b. If pending: run `ggml_graph_compute_thread(state)`.
   c. Loop until `threadpool->stop`.

### 8.3 Per-node barrier

`ggml_barrier` (`ggml-cpu.c:575`):

* If `n_threads == 1`: no-op.
* OpenMP: `#pragma omp barrier`.
* Custom: central-counter barrier.
  - `n_barrier = atomic_fetch_add(&tp->n_barrier, 1, seq_cst)`
  - Last thread resets `n_barrier = 0` and bumps `n_barrier_passed`.
  - Others spin on `n_barrier_passed` with `ggml_thread_cpu_relax()`.

`seq_cst` is used on both entry and exit fences. The TSAN path uses
`atomic_fetch_add(..., 0, seq_cst)` because TSAN does not support
standalone fences (`ggml-cpu.c:3158-3163`).

### 8.4 Per-op work distribution

Inside `ggml_compute_forward_<op>`, each op picks its own scheme:

| Op family                  | Scheme                                                     |
| -------------------------- | ---------------------------------------------------------- |
| `MUL_MAT`, `MUL_MAT_ID`    | Dynamic chunk stealing via `current_chunk` atomic          |
| Elementwise (add, mul, …)  | `get_thread_range(params, src)` — static row split         |
| Reductions (`SUM_ROWS`, …) | Static row split                                           |
| Softmax / attention        | Static row split, with tile constants `GGML_FA_TILE_Q=64`  |

The split is per-op, decided inside the kernel, not by the scheduler.

### 8.5 NUMA

`ggml_numa_init` reads `/sys/devices/system/node/...` on Linux and
records CPU→node mapping. `set_numa_thread_affinity(ith)` pins worker
`ith` to a specific CPU. For matmul specifically, NUMA triggers a
different chunking strategy: one chunk per thread, no stealing
(`ggml-cpu.c:1413-1417`). The comment cites PR #6915, which measured
this to be faster on multi-socket systems even though it should be
equivalent in theory.

---

## 9. SIMD / GPU Strategy

This file deliberately contains **no SIMD**. All SIMD lives in:

* `vec.cpp` — elementwise primitives (`ggml_vec_add_f32`, etc.) with
  inline `#if defined(__AVX2__)` / `__ARM_NEON` / `__ARM_FEATURE_SVE`
  blocks. See ARTX04 / ARTX05.
* `arch/<isa>/quants.c` — per-ISA quantized vecdot. See ARTX02–05.
* `ops.cpp` — op implementations with inline SIMD. See ARTX06.

The only ISA-aware code in `ggml-cpu.c` is `ggml_thread_cpu_relax`
(`ggml-cpu.c:519-538`): `yield` on aarch64, `_mm_pause` on x86_64,
`pause` on riscv, no-op elsewhere.

The dispatch mechanism to per-ISA code is the type-traits table at
`ggml-cpu.c:214`. Each entry's `.vec_dot` is a function pointer set
*at link time* to whatever `arch/<isa>/quants.c` provided. There is
no runtime CPU detection inside `ggml-cpu.c`; the right .so is
selected by the backend registry using `ggml_backend_cpu_x86_score`
(`arch/x86/cpu-feats.cpp:263`) or its ARM/other-ISA equivalent.

---

## 10. Quantization Strategy

The type-traits table is the contract. For each `GGML_TYPE_*`:

```c
[GGML_TYPE_Q4_0] = {
    .from_float    = quantize_row_q4_0,
    .vec_dot       = ggml_vec_dot_q4_0_q8_0,
    .vec_dot_type  = GGML_TYPE_Q8_0,
    .nrows         = 1, // or 2 on ARM I8MM
},
```

Implications:

* `from_float` is the **quantizer** (F32 → quant).
* `vec_dot` is the **inner-product kernel** between a quantized
  weight row and a `vec_dot_type` activation row.
* `vec_dot_type` is the activation dtype the kernel expects. For
  almost all quants, this is `Q8_0` or `Q8_K` (a "quantized
  activation" form). The matmul path converts src1 to this type
  once up-front (Section 5.5 step 2).
* `nrows` lets a kernel advertise it can consume 2 weight rows × 2
  activation rows per call (used by ARM I8MM `vmmla`).

The full quant format list with block sizes, scales, and zero-point
handling is audited in ARTX06.

Notable: `IQ2_XXS`, `IQ2_XS`, `IQ1_S`, `IQ1_M` have `from_float = NULL`
(`ggml-cpu.c:337, 343, 368, 374`). These formats cannot be quantized
at runtime — they require an offline importance-sampling procedure
(`ggml_quantize_init`, not in scope). This is a deliberate **design
constraint**: these quants are *inference-only*.

---

## 11. Correctness Analysis

Static analysis found the following correctness-relevant properties.
None are bugs; all are intentional design decisions that have
correctness consequences.

### 11.1 Floating-point reassociation

* **Vecdot unrolling**. `vec.cpp:11-110` (`ggml_vec_dot_f32`)
  unrolls 8 SVE accumulators (or 16 AVX-512 accumulators, or 8 AVX2
  accumulators — depending on ISA) and sums them at the end. This
  reassociates the sum: result differs from a strict left-to-right
  scalar sum at the ULP level. Standard for any SIMD dot product.
  Documented implicitly via `GGML_VEC_DOT_UNROLL = 2` and the
  unroll factors inside `vec.cpp`.
* **Reductions inside quantized vecdot**. The Q4_0/Q8_0 kernel
  (audited in ARTX06) accumulates into multiple vector lanes and
  horizontally reduces at the end. Same reassociation pattern.
* **Multi-row vecdot on ARM I8MM**. `nrows = 2` for Q4_0/Q4_1/Q8_0/
  Q4_K/Q6_K on `__ARM_FEATURE_MATMUL_INT8` builds (`ggml-cpu.c:243,
  253, 275, 314, 330`). The kernel computes two output rows in
  parallel from a single activation row. The arithmetic per output
  is the same, but the reduction order within each row may differ
  from the `nrows=1` path due to lane interleaving.

### 11.2 Approximate math

* **GELU via 128 KB f16 LUT**. `ggml_table_gelu_f16[1<<16]` stores
  GELU output as f16, looked up by f16 input (`ggml-cpu.c:3842`).
  The result is therefore accurate only to f16 precision (11 bits).
  Same for `ggml_table_gelu_quick_f16`. This is a deliberate
  accuracy/speed tradeoff.
* **E8M0 / UE4M3 LUTs**. `ggml-cpu.c:3848, 3853` precompute 256-entry
  tables for these small float formats. Used by MXFP4 / NVFP4 paths.
* **FP16↔FP32 conversions**. The 256 KB `ggml_table_f32_f16` table
  replaces intrinsic-based conversion on some ISAs. Precision is
  identical to IEEE 754 half — no approximation beyond the format
  itself.

### 11.3 Precision reduction

* **Quantized activations**. By design, every quantized matmul
  converts src1 from F32 to `vec_dot_type` (e.g., Q8_0). This is a
  lossy conversion before the dot product. It is the whole point of
  quantized inference and not a bug, but it means the matmul is
  *not* an F32 matmul with quantized weights — it is a Q8×Q8 matmul
  with quantized weights.
* **bf16 / f16 paths**. `GGML_TYPE_BF16` and `GGML_TYPE_F16` have
  `vec_dot_type` equal to themselves, so no conversion happens; the
  kernel accumulates in F32 (per `vec.cpp:ggml_vec_dot_f16`). No
  precision reduction beyond the storage format.

### 11.4 Non-deterministic reductions

* **Dynamic chunk stealing**. `current_chunk = atomic_fetch_add(...)`
  (`ggml-cpu.c:1450`) means the chunk a thread processes is
  non-deterministic across runs. Combined with the per-chunk
  reassociation in `vec_dot`, the F32 result of a matmul can vary
  at the ULP level across runs with the same inputs.
* **Cooperative activation conversion**. The parallel `from_float`
  at `ggml-cpu.c:1344-1355` writes to disjoint regions of `wdata`,
  so the result is deterministic. But the *timing* of when each row
  becomes visible to other threads is non-deterministic (barrier
  enforces only correctness, not order).
* **Conclusion**: Matmul output is deterministic bit-for-bit only
  when `nth = 1`. With `nth > 1`, expect ULP-level variation. This
  is unavoidable for any parallel reduction; llama.cpp makes no
  attempt to enforce bit-reproducibility.

### 11.5 Atomic accumulation

* None in the matmul path. Output tiles are written by exactly one
  thread each (chunk stealing assigns disjoint chunks), so no atomics
  are needed on `dst->data`.
* `current_chunk` is the only atomic in the matmul hot path, and it
  is a counter, not an accumulator.

### 11.6 Architecture-specific assumptions

* `nrows = 2` is set only when `__ARM_FEATURE_MATMUL_INT8` is
  defined. On non-I8MM ARM, the same quants use `nrows = 1`. The
  two paths produce *slightly different* results due to lane
  interleaving. This is a per-arch determinism leak.
* `GGML_USE_LLAMAFILE` swaps in `llamafile_sgemm` for F16/F32
  matmuls when src1 is contiguous. The llamafile SGEMM has its own
  tiling and may produce different ULPs than the ggml path.
* `GGML_CACHE_LINE = 64` assumes 64-byte cache lines. True on every
  ISA llama.cpp currently targets, but not guaranteed in general.

---

## 12. Optimization Analysis

### 12.1 Identified optimizations

| Optimization                 | Where                                       | Notes                                                  |
| ---------------------------- | ------------------------------------------- | ------------------------------------------------------ |
| Type-traits function pointer | `ggml-cpu.c:214`                            | One indirect call per `vec_dot`; devirtualized? No, but the table is read-only after init so branch prediction is excellent. |
| Tile blocking in matmul      | `ggml-cpu.c:1202-1203` `blck_0=blck_1=16`   | Keeps a 16×16 tile of dst + 16 rows of src0 + 16 cols of src1 in L1. |
| Dynamic chunk stealing       | `ggml-cpu.c:1426-1451`                       | Load balancing for variable-cost chunks (e.g., when quants differ per row). |
| Cache-aligned atomics        | `ggml-cpu.c:489-491`                         | Prevents false sharing on `n_barrier` / `current_chunk`. |
| Spin-then-wait polling       | `ggml-cpu.c:3167-3180`                       | `poll * 128K` rounds of `_mm_pause`/`yield` before sleeping on cond var. |
| NUMA-aware chunking          | `ggml-cpu.c:1413-1417`                       | Switches to one-chunk-per-thread on NUMA to keep memory local. |
| GELU LUT                     | `ggml-cpu.c:3842`                            | 128 KB f16 LUT replaces transcendental in hot path.    |
| F16→F32 LUT                  | `ggml-cpu.c:3841`                            | 256 KB LUT; faster than intrinsic on some ISAs.        |
| Multi-row vecdot (ARM I8MM)  | `ggml-cpu.c:243, 253, 275, 314, 330`         | 2× throughput on I8MM-capable ARM cores.               |
| Pre-converted activations    | `ggml-cpu.c:1322-1357`                       | `from_float` runs once per matmul, not per chunk.      |
| `use_ref` toggle             | `ggml-cpu.cpp:280-285`                       | Forces reference (scalar) paths — used for validation. |
| Hadamard fast path           | `ggml-cpu.c:1262-1265`                       | Detects `GGML_HINT_SRC0_IS_HADAMARD` and dispatches to FWHT. |
| Disposable threadpool        | `ggml-cpu.c:3360-3367`                       | Avoids thread creation cost when caller has no pool.   |

### 12.2 Optimizations *not* present (worth noting)

* **No kernel fusion beyond `RMS_NORM+MUL`**. `ggml_cpu_try_fuse_ops`
  has exactly one fusion pattern (`ggml-cpu.c:3038-3055`). No
  `ADD+ACT`, no `MUL_MAT+ADD` (bias), no `ROPE+MV` fusion.
* **No software prefetching** in the matmul path. Tiling relies on
  hardware prefetchers.
* **No persistent threads**. Threads are created in
  `ggml_threadpool_new_impl` and destroyed in `ggml_threadpool_free`.
  The pool can be paused/resumed across graphs, but not across
  process boundaries.
* **No async execution**. The backend advertises `async=false`.
* **No graph-level parallelism**. All threads execute the same node
  at the same time; no two nodes run in parallel even if they are
  independent.

---

## 13. Architectural Strengths

1. **Type-traits table is a clean ABI**. Adding a new quant format
   means adding one entry to the table and providing three functions.
   No other code in `ggml-cpu.c` needs to change. This is the
   single best design decision in the file.

2. **Extra-buffer-type extension hook**. AMX, KleidiAI, SpacemiT, and
   Repack can each claim ops via `supports_op` and override
   `compute_forward` without touching the core dispatch. This is a
   clean plugin architecture for accelerators that share the CPU
   address space.

3. **Disposable vs. persistent threadpool**. Callers without a
   threadpool get one transparently; callers who care about latency
   can pre-create and reuse. The pause/resume mechanism lets a pool
   stay warm across inference calls.

4. **NUMA-aware chunking fallback**. The "one chunk per thread on
   NUMA" rule is a measured pragmatic win that the code documents
   with a PR reference. Good evidence-based engineering.

5. **Cache-aligned atomics**. Explicit `GGML_CACHE_ALIGN` on the
   three hot atomics. Small but correct.

6. **`use_ref` toggle**. Provides a reference path for validation.
   Critical for debugging quantization correctness issues without
   needing a separate build.

7. **Hadamard fast path via op hint**. The `GGML_HINT_SRC0_IS_HADAMARD`
   mechanism is a clean way to dispatch a structurally-special op
   without a separate op code. GwenLand should consider this pattern
   for any structurally-special matmul variants.

---

## 14. Architectural Weaknesses

### W1 — Per-node central barrier

**Evidence**: `ggml-cpu.c:3116` `ggml_barrier(state->threadpool);`
after every node; `ggml-cpu.c:575` `ggml_barrier` implementation
spins on a single atomic.

**Impact**: Every node — even a 1-microsecond elementwise add — pays
a full barrier. On a 16-thread machine, this is ~16 `_mm_pause`
rounds per node. For graphs with many small ops (e.g., attention
prefill with many small reductions), barrier overhead dominates.

**Why it's hard to fix**: The SPMD-with-barrier model means every
thread sees every node, so any per-node parallelism scheme must be
encoded inside `ggml_compute_forward_<op>`. The scheduler cannot
reorder or pipeline nodes because there is no scheduler-level
concurrency.

### W2 — No kernel fusion beyond one pattern

**Evidence**: `ggml-cpu.c:3026-3058` — only `RMS_NORM + MUL` is
fused. The TODO at `ggml-cpu.c:3100` says "move fused-op detection
into `ggml_graph_plan` so fusion decisions are made once at planning
time" — i.e., the team knows fusion is under-developed.

**Impact**: Every `MUL_MAT` followed by `ADD` (residual) is two
barriers and two memory round-trips. Every `MUL_MAT` followed by
`RMS_NORM` is two barriers. For transformer inference, this is the
majority of the graph.

### W3 — No async / event support

**Evidence**: `ggml-cpu.cpp:196-209` sets `set_tensor_async`,
`get_tensor_async`, `cpy_tensor_async`, `synchronize`, `event_record`,
`event_wait` all to `NULL`.

**Impact**: The CPU cannot overlap computation with host→device
transfers. For CPU+GPU hybrid execution, the CPU is a strict
synchronization point. This is acceptable for a pure-CPU backend
but means GwenLand's GATE cannot treat the CPU as a peer to GPU
backends.

### W4 — `use_ref` is per-backend, not per-op

**Evidence**: `ggml-cpu.cpp:280-285` sets `use_ref` on the backend
context; `ggml-cpu.c:3079` propagates it to every node's params;
`ggml-cpu.c:3032` disables *all* fusion when `use_ref` is set.

**Impact**: You cannot validate a single op (e.g., the Q4_K vecdot)
against its reference without forcing the entire graph to reference
mode. This makes differential testing expensive.

### W5 — Hardcoded `GGML_CACHE_LINE = 64`

**Evidence**: `ggml-cpu.c:60`. Comment acknowledges the issue:
"once we move threading into a separate C++ file will use
`std::hardware_destructive_interference_size` instead of hardcoding."

**Impact**: On a future ISA with 128-byte cache lines (some POWER
configurations), false-sharing mitigations would silently fail.

### W6 — Per-thread CPU mask is a `bool[GGML_MAX_N_THREADS]`

**Evidence**: `ggml-cpu.c:513` `bool cpumask[GGML_MAX_N_THREADS]`.
`GGML_MAX_N_THREADS` is typically 256 or higher.

**Impact**: Each worker state is 256+ bytes; the `workers[]` array
can span multiple cache lines. Iterating over workers to apply
affinity (e.g., in `ggml_threadpool_new_impl`) touches many lines.
A bitmap would be 32 bytes for 256 CPUs.

### W7 — Matmul chunk size is heuristic, not learned

**Evidence**: `ggml-cpu.c:1397-1417`. `chunk_size = 16` always,
`64` for GEMV-like shapes. No feedback from prior runs.

**Impact**: Suboptimal for shapes that don't fit the heuristic
(e.g., very wide short matmuls, or tall skinny ones). A
shape-adaptive chunk size or a small autotuner would help.

### W8 — Activation conversion writes to `wdata` even when src1 is already `vec_dot_type`

**Evidence**: `ggml-cpu.c:1322` `if (src1->type != vec_dot_type)`.
The branch is correct, but the `wdata` pointer is still computed
even when not needed (negligible, but a sign of code that grew
rather than was designed).

### W9 — `type_traits_cpu` is `static const` in `ggml-cpu.c`

**Evidence**: `ggml-cpu.c:214` `static const struct ggml_type_traits_cpu type_traits_cpu[GGML_TYPE_COUNT]`.

**Impact**: External code cannot override entries at runtime (e.g.,
to install a tuned kernel). Extension is only possible via the
extra-buffer-type mechanism, which requires a full buffer type
registration. There is no lightweight "swap one vecdot" hook.

### W10 — Fusion detection runs at execution time

**Evidence**: `ggml-cpu.c:3100-3102` — `ggml_cpu_try_fuse_ops` is
called per-node per-graph-execution. The TODO acknowledges this.

**Impact**: O(N) fusion checks per graph run, where N is node count.
For repeated inference, this work is wasted.

---

## 15. GwenLand Mapping

| GwenLand module | ADOPT / ADAPT / REJECT / MONITOR / DEFER | What | Reasoning |
| --------------- | ---------------------------------------- | ---- | --------- |
| `glproc`        | **ADOPT** | Type-traits table indexed by dtype | Clean ABI; one indirect call; trivial to extend. |
| `glproc`        | **ADOPT** | Extra-buffer-type extension hook | Plugin architecture for in-address-space accelerators. |
| `glproc`        | **ADOPT** | Dynamic chunk stealing via atomic counter | Load-balances variable-cost chunks; cheap. |
| `glproc`        | **ADOPT** | `use_ref` toggle | Essential for differential validation. |
| `glproc`        | **ADAPT** | Per-node barrier | Keep the barrier, but add a fast path for single-node graphs and a "no-barrier mode" for explicitly pipelined subgraphs. |
| `glproc`        | **REJECT**| SPMD-with-barrier as the *only* execution model | GwenLand should support per-node parallelism: independent nodes should run concurrently. |
| `glproc`        | **REJECT**| `chunk_size = 16` hardcoded | Use a shape-aware policy; consider a small offline autotuner. |
| `glproc`        | **ADAPT** | NUMA-aware chunking fallback | Keep the idea (one chunk per thread on NUMA), but make the policy pluggable. |
| `glproc`        | **MONITOR**| `GGML_USE_LLAMAFILE` integration | Watch whether llamafile's SGEMM remains competitive with hand-tuned kernels. |
| `glproc`        | **DEFER** | GELU/QuickGELU f16 LUTs | Only adopt if GwenLand uses f16 activations internally; otherwise irrelevant. |
| `GATE`          | **ADOPT** | Op-hint mechanism (`GGML_HINT_*`) | Clean way to dispatch structurally-special matmuls without new op codes. |
| `GATE`          | **REJECT**| Fusion decided at execution time | Plan fusion once at graph-plan time. |
| `GATE`          | **ADOPT** | Disposable vs. persistent threadpool | Both modes are useful for different workloads. |
| `GATE`          | **ADAPT** | `ggml_barrier` central-counter design | Keep the design, but consider a sense-reversing barrier to avoid cache-line traffic on the counter reset. |
| `GATE`          | **MONITOR**| TSAN fence workaround | Watch for TSAN upstream support for standalone fences; remove the dummy RMW when available. |

---

## 16. Recommendations

### R1 — ADOPT type-traits table as glproc's primary dispatch
**Priority:** Critical
**Difficulty:** S
**Dependencies:** none
GwenLand's `glproc` should define an equivalent `gl_type_traits[GL_TYPE_COUNT]` table. Each entry exposes `from_float`, `vec_dot`, `vec_dot_type`, `nrows`. Same ABI, same semantics.

### R2 — ADOPT extra-buffer-type plugin mechanism
**Priority:** High
**Difficulty:** M
**Dependencies:** R1
GwenLand will need to integrate AMX, KleidiAI-equivalents, and possibly vendor SDKs. The extra-buffer-type pattern lets these claim ops without touching core dispatch.

### R3 — REJECT per-node SPMD barrier as the only model
**Priority:** High
**Difficulty:** L
**Dependencies:** GATE design
GwenLand's GATE should support: (a) per-node parallelism for independent nodes, (b) per-op parallelism strategy hooks (op says "I want N threads" or "I am single-threaded"), (c) barrier only when an op actually needs cross-thread synchronization.

### R4 — ADOPT dynamic chunk stealing
**Priority:** High
**Difficulty:** S
**Dependencies:** R1
For matmul-style ops with variable chunk cost, an atomic-counter chunk-stealing scheme is simple and effective. Use `memory_order_relaxed` for the counter; correctness does not require `acq_rel`.

### R5 — ADAPT fusion: plan-time, not execution-time
**Priority:** High
**Difficulty:** M
**Dependencies:** R3
Move fusion detection into the graph planner. Detect patterns once, mark fused nodes in the plan, execute without re-checking. Add at least: `MUL_MAT + ADD` (bias), `MUL_MAT + RMS_NORM`, `ADD + ACT`, `ROPE + MV`.

### R6 — ADOPT `use_ref` toggle but make it per-op
**Priority:** Medium
**Difficulty:** S
**Dependencies:** R1
Allow `use_ref` to be set per-op (e.g., via a tensor flag), not just per-backend. This enables differential testing of one op without forcing the whole graph to reference mode.

### R7 — ADOPT op-hint mechanism
**Priority:** Medium
**Difficulty:** S
**Dependencies:** R1
Replicate `GGML_HINT_*` for structurally-special matmuls GwenLand may need (Hadamard, block-diagonal, sparse, etc.).

### R8 — DEFER GELU/F16 LUTs
**Priority:** Low
**Difficulty:** S
**Dependencies:** none
Only relevant if GwenLand uses f16 activations internally. Re-evaluate when activation dtype strategy is decided.

### R9 — ADOPT cache-aligned atomics, but verify line size
**Priority:** Medium
**Difficulty:** XS
**Dependencies:** none
Use `std::hardware_destructive_interference_size` where available, fallback to 64. Apply to every shared atomic in hot paths.

### R10 — ADAPT NUMA-aware chunking
**Priority:** Medium
**Difficulty:** M
**Dependencies:** R4
Keep the "one chunk per thread on NUMA" fallback, but make the policy a function pointer so GwenLand can swap in alternative policies (e.g., NUMA-aware chunk stealing that respects node locality).

---

## 17. Findings

### Finding ARTX01-F01

```
Finding ID:           ARTX01-F01
Category:             BACKEND_DESIGN
Engine:               CPU
Component:            Backend interface
Source File:          ggml/src/ggml-cpu/ggml-cpu.cpp
Function:             ggml_backend_cpu_i (vtable)
Lines:                193-210
Summary:              CPU backend advertises async=false and exposes no event APIs.
Observation:          The vtable sets set_tensor_async, get_tensor_async,
                      cpy_tensor_async, synchronize, event_record, event_wait
                      all to NULL. The CPU backend is purely synchronous.
Evidence:             ggml-cpu.cpp:193-210 — explicit NULL assignments.
Architectural Impact: The CPU cannot overlap computation with host↔device
                      transfers. In hybrid CPU+GPU execution, the CPU is a
                      strict synchronization point.
Correctness Impact:   None. Synchronous execution is correct by definition.
Optimization Type:    None (absence of optimization).
GwenLand Target:      GATE
Recommendation:       REJECT this constraint. GwenLand's glproc should
                      expose at least a trivial event system so the
                      scheduler can treat CPU as a peer to GPU backends.
Priority:             Medium
Difficulty:           M
Dependencies:         GATE design
Confidence:           High
```

### Finding ARTX01-F02

```
Finding ID:           ARTX01-F02
Category:             EXECUTION_GRAPH
Engine:               CPU
Component:            Graph executor
Source File:          ggml/src/ggml-cpu/ggml-cpu.c
Function:             ggml_graph_compute_thread
Lines:                3060-3133
Summary:              Every thread runs every node; barrier after every node.
Observation:          The worker loop iterates over cgraph->nodes[node_n]
                      and calls ggml_barrier() after each node. Parallelism
                      exists only *inside* a node (via ith/nth split), not
                      across nodes.
Evidence:             ggml-cpu.c:3088-3118 — loop with per-node barrier.
Architectural Impact: Independent nodes cannot run concurrently. Short
                      ops pay a full barrier cost (~16 _mm_pause rounds
                      on a 16-thread machine).
Correctness Impact:   None. The barrier is correct.
Optimization Type:    None (this is the absence of an optimization).
GwenLand Target:      GATE
Recommendation:       REJECT. GwenLand should support per-node parallelism
                      for independent subgraphs.
Priority:             High
Difficulty:           L
Dependencies:         GATE design
Confidence:           High
```

### Finding ARTX01-F03

```
Finding ID:           ARTX01-F03
Category:             SIMD_STRATEGY
Engine:               CPU
Component:            Type-traits table
Source File:          ggml/src/ggml-cpu/ggml-cpu.c
Function:             type_traits_cpu[]
Lines:                214-415
Summary:              Per-dtype dispatch table maps every quant format to
                      from_float / vec_dot / vec_dot_type / nrows.
Observation:          A static const array indexed by enum ggml_type
                      provides function pointers for every dtype. The
                      matmul path consults this table once per op.
Evidence:             ggml-cpu.c:1181-1182 (read), 214-415 (definition).
Architectural Impact: Adding a quant format = adding one entry. Clean ABI.
Correctness Impact:   None. Dispatch is indirect but deterministic.
Optimization Type:    Indirect call with stable target (branch predictor
                      friendly).
GwenLand Target:      glproc
Recommendation:       ADOPT. Equivalent table in glproc.
Priority:             Critical
Difficulty:           S
Dependencies:         none
Confidence:           High
```

### Finding ARTX01-F04

```
Finding ID:           ARTX01-F04
Category:             BACKEND_DESIGN
Engine:               CPU
Component:            Extra buffer types
Source File:          ggml/src/ggml-cpu/traits.h, ggml/src/ggml-cpu/ggml-cpu.cpp
Function:             ggml::cpu::extra_buffer_type, ggml_backend_cpu_get_extra_buffer_types
Lines:                traits.h:27-32; ggml-cpu.cpp:42-95
Summary:              Plugin architecture lets AMX/KleidiAI/SpacemiT/Repack
                      override supports_op and compute_forward for ops they claim.
Observation:          Each extra buffer type is a C++ abstract class with
                      supports_op() and get_tensor_traits(). The CPU backend
                      checks all registered extras before falling through to
                      the default dispatch.
Evidence:             traits.h:27-32; ggml-cpu.cpp:88-95; ggml-cpu.c:1719
                      (ggml_cpu_extra_compute_forward call site).
Architectural Impact: Accelerators that share the CPU address space can
                      claim specific ops without modifying core dispatch.
Correctness Impact:   None.
Optimization Type:    Plugin architecture for per-op kernel selection.
GwenLand Target:      glproc
Recommendation:       ADOPT. Essential for integrating vendor SDKs.
Priority:             High
Difficulty:           M
Dependencies:         R1 (type-traits table)
Confidence:           High
```

### Finding ARTX01-F05

```
Finding ID:           ARTX01-F05
Category:             CORRECTNESS_SHORTCUT
Engine:               CPU
Component:            vec_dot (F32)
Source File:          ggml/src/ggml-cpu/vec.cpp
Function:             ggml_vec_dot_f32
Lines:                11-110 (SVE path 21-90; AVX-512 path similar; AVX2 path similar)
Summary:              SIMD dot product uses 8 (SVE/AVX2) or 16 (AVX-512)
                      independent accumulators and horizontally reduces at end.
Observation:          The kernel unrolls the loop into N independent vector
                      accumulators (sum1..sum8 for SVE). Each accumulator is
                      an independent dependency chain. The final sum combines
                      them with vector adds, then a horizontal reduction.
Evidence:             vec.cpp:27-69 (8 accumulators), vec.cpp:72-90 (tail).
Architectural Impact: High throughput via ILP. Reassociates the sum
                      relative to scalar left-to-right.
Correctness Impact:   ULP-level difference vs. scalar reference. Deterministic
                      for a fixed ISA and thread count.
Optimization Type:    SIMD unrolling + independent accumulator chains.
GwenLand Target:      glproc
Recommendation:       ADOPT. Standard SIMD dot-product pattern.
Priority:             High
Difficulty:           S
Dependencies:         none
Confidence:           High
```

### Finding ARTX01-F06

```
Finding ID:           ARTX01-F06
Category:             CORRECTNESS_SHORTCUT
Engine:               CPU
Component:            Matmul chunk scheduler
Source File:          ggml/src/ggml-cpu/ggml-cpu.c
Function:             ggml_compute_forward_mul_mat
Lines:                1426-1451
Summary:              Dynamic chunk stealing via atomic_fetch_add on
                      threadpool->current_chunk.
Observation:          Each thread starts at current_chunk = ith, processes
                      its chunk, then current_chunk = atomic_fetch_add(1).
                      The chunk a thread processes is non-deterministic
                      across runs (depends on scheduling).
Evidence:             ggml-cpu.c:1424-1450.
Architectural Impact: Load balancing for variable-cost chunks. Combined
                      with per-chunk reassociation (F05), produces ULP-level
                      non-determinism across runs.
Correctness Impact:   Bit-exact reproducibility only when nth=1.
Optimization Type:    Dynamic load balancing via atomic counter.
GwenLand Target:      glproc
Recommendation:       ADOPT for general case. Provide a "deterministic mode"
                      (static chunk assignment) for testing.
Priority:             High
Difficulty:           S
Dependencies:         R4
Confidence:           High
```

### Finding ARTX01-F07

```
Finding ID:           ARTX01-F07
Category:             CORRECTNESS_SHORTCUT
Engine:               CPU
Component:            Activation LUTs
Source File:          ggml/src/ggml-cpu/ggml-cpu.c
Function:             ggml_cpu_init
Lines:                3835-3854
Summary:              GELU and QuickGELU are precomputed as 128 KB f16 LUTs.
Observation:          For each of 65536 f16 inputs, the table stores the
                      activation output as f16. Hot-path activation is a
                      single table lookup.
Evidence:             ggml-cpu.c:3842-3843.
Architectural Impact: Replaces transcendental with LUT lookup.
Correctness Impact:   Output is f16-precision (11 bits), not f32. For
                      models where GELU input is f32, this is a precision
                      reduction.
Optimization Type:    LUT-based activation.
GwenLand Target:      glproc
Recommendation:       DEFER. Only adopt if GwenLand uses f16 activations
                      for GELU input; otherwise compute GELU in f32.
Priority:             Low
Difficulty:           S
Dependencies:         none
Confidence:           High
```

### Finding ARTX01-F08

```
Finding ID:           ARTX01-F08
Category:             EXECUTION_GRAPH
Engine:               CPU
Component:            Op fusion
Source File:          ggml/src/ggml-cpu/ggml-cpu.c
Function:             ggml_cpu_try_fuse_ops
Lines:                3026-3058
Summary:              Only one fusion pattern exists (RMS_NORM + MUL).
                      Fusion is detected at execution time, not plan time.
Observation:          The function checks if the current node is RMS_NORM
                      and the next is MUL with matching shapes; if so,
                      runs ggml_compute_forward_rms_norm_mul_fused.
                      No other patterns are checked.
Evidence:             ggml-cpu.c:3038-3055. TODO at ggml-cpu.c:3100.
Architectural Impact: Common patterns (MUL_MAT+ADD, ADD+ACT, ROPE+MV)
                      are not fused. Each pays a barrier and a memory
                      round-trip.
Correctness Impact:   None. Unfused execution is correct.
Optimization Type:    Kernel fusion (limited).
GwenLand Target:      GATE, glproc
Recommendation:       ADAPT. Move fusion to plan time. Add at least
                      MUL_MAT+ADD (bias), MUL_MAT+RMS_NORM, ADD+ACT.
Priority:             High
Difficulty:           M
Dependencies:         R3, R5
Confidence:           High
```

### Finding ARTX01-F09

```
Finding ID:           ARTX01-F09
Category:             THREADING_MISMATCH
Engine:               CPU
Component:            NUMA chunking
Source File:          ggml/src/ggml-cpu/ggml-cpu.c
Function:             ggml_compute_forward_mul_mat
Lines:                1413-1417
Summary:              On NUMA systems, matmul switches from many-chunk
                      dynamic stealing to one-chunk-per-thread.
Observation:          The comment cites PR #6915, which measured this to
                      be faster on NUMA even though "in theory" chunking
                      should be equivalent. The fallback is triggered by
                      ggml_is_numa().
Evidence:             ggml-cpu.c:1413-1417.
Architectural Impact: Better locality on multi-socket systems. Worse
                      load balancing if chunks have variable cost.
Correctness Impact:   None.
Optimization Type:    NUMA-aware work distribution.
GwenLand Target:      glproc
Recommendation:       ADAPT. Keep the policy but make it pluggable so
                      GwenLand can experiment with NUMA-aware chunk
                      stealing.
Priority:             Medium
Difficulty:           M
Dependencies:         R4
Confidence:           High
```

### Finding ARTX01-F10

```
Finding ID:           ARTX01-F10
Category:             LAYOUT_SUBOPTIMAL
Engine:               CPU
Component:            Threadpool state
Source File:          ggml/src/ggml-cpu/ggml-cpu.c
Function:             struct ggml_compute_state
Lines:                507-516
Summary:              Per-thread cpumask is bool[GGML_MAX_N_THREADS].
Observation:          GGML_MAX_N_THREADS is typically 256+. Each worker
                      state is 256+ bytes. The workers[] array can span
                      many cache lines.
Evidence:             ggml-cpu.c:513.
Architectural Impact: Iterating workers touches many cache lines. A
                      bitmap (32 bytes for 256 CPUs) would be more
                      compact.
Correctness Impact:   None.
Optimization Type:    None (suboptimal layout).
GwenLand Target:      glproc
Recommendation:       REJECT. Use a bitmap for cpumask in GwenLand.
Priority:             Low
Difficulty:           XS
Dependencies:         none
Confidence:           Medium
```

### Finding ARTX01-F11

```
Finding ID:           ARTX01-F11
Category:             BACKEND_DESIGN
Engine:               CPU
Component:            Type-traits table
Source File:          ggml/src/ggml-cpu/ggml-cpu.c
Function:             type_traits_cpu
Lines:                214
Summary:              type_traits_cpu is static const — entries cannot
                      be overridden at runtime.
Observation:          The table is declared static const in ggml-cpu.c.
                      External code cannot swap an entry to install a
                      tuned kernel without going through the extra-
                      buffer-type mechanism.
Evidence:             ggml-cpu.c:214.
Architectural Impact: No lightweight "swap one vecdot" hook. Extension
                      requires a full buffer-type registration.
Correctness Impact:   None.
Optimization Type:    None.
GwenLand Target:      glproc
Recommendation:       ADAPT. Make the table mutable at runtime (with
                      atomic pointer swaps or RCU) so tuned kernels
                      can be installed per-dtype.
Priority:             Medium
Difficulty:           S
Dependencies:         R1
Confidence:           Medium
```

### Finding ARTX01-F12

```
Finding ID:           ARTX01-F12
Category:             OTHER
Engine:               CPU
Component:            Backend selection
Source File:          ggml/src/ggml-cpu/arch/x86/cpu-feats.cpp
Function:             ggml_backend_cpu_x86_score
Lines:                263-323
Summary:              CPU backend selection uses a compile-time-macro +
                      runtime-CPUID scoring scheme. Each .so variant is
                      compiled with specific GGML_* macros; the runtime
                      score returns 0 if the CPU lacks the features,
                      higher for more advanced features.
Observation:          This is "multi-binary dispatch": build N .so
                      files, one per ISA target; the loader picks the
                      best matching one. Alternative would be one
                      fat binary with runtime function-pointer dispatch.
Evidence:             arch/x86/cpu-feats.cpp:263-325.
Architectural Impact: Smaller code per binary; no indirect-call overhead
                      inside hot loops. But requires building/distributing
                      multiple binaries.
Correctness Impact:   None.
Optimization Type:    Multi-binary ISA dispatch.
GwenLand Target:      glproc
Recommendation:       MONITOR. Consider both schemes for GwenLand. Multi-
                      binary is simpler but heavier to ship; function-
                      pointer dispatch is more flexible.
Priority:             Low
Difficulty:           M
Dependencies:         none
Confidence:           Medium
```

---

## 18. Unknowns

* **U1**. Whether the SPMD-with-barrier model is a measurable
  bottleneck for typical llama.cpp workloads. Requires runtime
  profiling of barrier time vs. compute time per node. Static
  analysis cannot determine this.
* **U2**. Whether the `nrows = 2` ARM I8MM path produces
  bit-identical results to the `nrows = 1` path for the same input.
  Requires executing both paths on the same input. Static analysis
  shows the *arithmetic* is equivalent but the *reduction order*
  differs due to lane interleaving.
* **U3**. The actual branch-prediction hit rate on the type-traits
  indirect call. The table is read-only, so the call target is
  stable per (op, dtype) pair, but the predictor's behavior depends
  on the call-site history. Requires PMU analysis.
* **U4**. Whether `llamafile_sgemm` outperforms the ggml path on
  current x86 hardware. The codebase retains both paths but does
  not document which is preferred. Requires benchmarking.
* **U5**. Whether the central-counter barrier in `ggml_barrier`
  scales acceptably beyond 32 threads. The spin loop on
  `n_barrier_passed` is on a single cache line; this could become
  a bottleneck on high-core-count systems. Requires profiling on
  ≥32-core hardware.
* **U6**. Whether the `poll` parameter (default value not visible in
  this file) is tuned per-architecture or left at a sensible default.
  The factor `1024 * 128 * poll` suggests `poll = 1` gives ~128K
  rounds, which is roughly 100µs on a modern CPU. Confirm by
  inspecting `ggml_threadpool_params_default` (not in this file).

---

## 19. References

| Reference | File                                                | Function / Symbol                              | Lines         |
| --------- | --------------------------------------------------- | ---------------------------------------------- | ------------- |
| R01       | `ggml/src/ggml-cpu/ggml-cpu.cpp`                    | `ggml_backend_cpu_i` (vtable)                  | 193–210       |
| R02       | `ggml/src/ggml-cpu/ggml-cpu.cpp`                    | `ggml_backend_cpu_graph_compute`               | 170–191       |
| R03       | `ggml/src/ggml-cpu/ggml-cpu.cpp`                    | `ggml_backend_cpu_device_supports_op`          | 423–475       |
| R04       | `ggml/src/ggml-cpu/ggml-cpu.cpp`                    | `ggml_backend_cpu_get_extra_buffer_types`      | 42–74         |
| R05       | `ggml/src/ggml-cpu/ggml-cpu.cpp`                    | `ggml_backend_cpu_init`                        | 217–247       |
| R06       | `ggml/src/ggml-cpu/ggml-cpu.c`                      | `type_traits_cpu[]`                            | 214–415       |
| R07       | `ggml/src/ggml-cpu/ggml-cpu.c`                      | `ggml_compute_params` (struct, in impl.h)      | impl.h:18–30  |
| R08       | `ggml/src/ggml-cpu/ggml-cpu.c`                      | `struct ggml_threadpool`                       | 480–504       |
| R09       | `ggml/src/ggml-cpu/ggml-cpu.c`                      | `struct ggml_compute_state`                    | 507–516       |
| R10       | `ggml/src/ggml-cpu/ggml-cpu.c`                      | `ggml_barrier`                                 | 575–612       |
| R11       | `ggml/src/ggml-cpu/ggml-cpu.c`                      | `ggml_thread_cpu_relax`                        | 519–538       |
| R12       | `ggml/src/ggml-cpu/ggml-cpu.c`                      | `ggml_compute_forward_mul_mat`                 | 1254–1452     |
| R13       | `ggml/src/ggml-cpu/ggml-cpu.c`                      | `ggml_compute_forward_mul_mat_one_chunk`       | 1164–1252     |
| R14       | `ggml/src/ggml-cpu/ggml-cpu.c`                      | `ggml_compute_forward` (op dispatch)           | 1711–1810+    |
| R15       | `ggml/src/ggml-cpu/ggml-cpu.c`                      | `ggml_graph_compute`                           | 3350–3425     |
| R16       | `ggml/src/ggml-cpu/ggml-cpu.c`                      | `ggml_graph_compute_thread`                    | 3060–3133     |
| R17       | `ggml/src/ggml-cpu/ggml-cpu.c`                      | `ggml_graph_compute_secondary_thread`          | 3201–3237     |
| R18       | `ggml/src/ggml-cpu/ggml-cpu.c`                      | `ggml_graph_compute_kickoff`                   | 3239–3272     |
| R19       | `ggml/src/ggml-cpu/ggml-cpu.c`                      | `ggml_threadpool_new_impl`                     | 3273–3344     |
| R20       | `ggml/src/ggml-cpu/ggml-cpu.c`                      | `ggml_cpu_try_fuse_ops`                        | 3026–3058     |
| R21       | `ggml/src/ggml-cpu/ggml-cpu.c`                      | `ggml_cpu_init`                                | 3818–3895     |
| R22       | `ggml/src/ggml-cpu/ggml-cpu.c`                      | `ggml_cpu_extra_compute_forward` (in traits.cpp)| traits.cpp:12–23 |
| R23       | `ggml/src/ggml-cpu/traits.h`                        | `class tensor_traits`, `class extra_buffer_type`| 20–32        |
| R24       | `ggml/src/ggml-cpu/vec.cpp`                         | `ggml_vec_dot_f32` (SVE path)                  | 11–110        |
| R25       | `ggml/src/ggml-cpu/arch/x86/cpu-feats.cpp`          | `ggml_backend_cpu_x86_score`                   | 263–325       |
| R26       | `ggml/src/ggml-cpu/ggml-cpu-impl.h`                 | `ggml_compute_params`, ISA intrinsic shims     | 18–539        |
| R27       | `ggml/src/ggml-cpu/common.h`                        | `GGML_FA_TILE_Q/KV`, type conversion helpers  | 9–95          |
