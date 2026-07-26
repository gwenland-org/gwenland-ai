# IMPLEMENTATION-PLAN — GwenLand Work Items

**Audited commit:** `555881ebc8b0fc0402b30e09258a32a7bfd13c52`

This file lists GwenLand implementation work items. An item appears
here **only when** at least one ARTX document provides direct source
evidence for it. Items without evidence live in `GAP-MAP.md` instead.

## Work item format

```
[ID] Title
  Source:   ARTX##  (file:function:lines)
  Target:   glproc | glcuda | glmetal | glvulkan | GATE
  Action:   ADOPT | ADAPT | REJECT | MONITOR | DEFER
  Priority: Critical | High | Medium | Low
  Difficulty: XS | S | M | L | XL
  Dependencies: <IDs or "none">
  Notes: <one paragraph>
```

## Items

### [IP-01] Type-traits dispatch table for glproc
  Source:   ARTX01-F03 (ggml-cpu.c:type_traits_cpu[]:214-415)
  Target:   glproc
  Action:   ADOPT
  Priority: Critical
  Difficulty: S
  Dependencies: none
  Notes: Per-dtype table mapping from_float/vec_dot/vec_dot_type/nrows.
    One indirect call per op; branch-predictor-friendly. Adding a
    quant format = adding one entry.

### [IP-02] Extra-buffer-type plugin mechanism for glproc
  Source:   ARTX01-F04 (traits.h:27-32; ggml-cpu.cpp:42-95)
  Target:   glproc
  Action:   ADOPT
  Priority: High
  Difficulty: M
  Dependencies: IP-01
  Notes: C++ abstract base with supports_op() / get_tensor_traits().
    Lets accelerators (AMX, KleidiAI-equivalents, vendor SDKs) claim
    ops without touching core dispatch.

### [IP-03] Per-node parallelism in GATE
  Source:   ARTX01-F02 (ggml-cpu.c:ggml_graph_compute_thread:3060-3133)
  Target:   GATE
  Action:   REJECT llama.cpp's SPMD-with-barrier as the *only* model
  Priority: High
  Difficulty: L
  Dependencies: GATE design
  Notes: GATE must support: (a) per-node parallelism for independent
    nodes, (b) per-op parallelism-strategy hooks, (c) barrier only
    when an op actually needs cross-thread synchronization.

### [IP-04] Dynamic chunk stealing for matmul-style ops
  Source:   ARTX01-F06 (ggml-cpu.c:1426-1451)
  Target:   glproc
  Action:   ADOPT
  Priority: High
  Difficulty: S
  Dependencies: IP-01
  Notes: Atomic-fetch-add on a shared counter; relaxed memory order
    suffices. Provide a "deterministic mode" (static chunk assignment)
    for testing.

### [IP-05] Plan-time op fusion
  Source:   ARTX01-F08 (ggml-cpu.c:ggml_cpu_try_fuse_ops:3026-3058)
  Target:   GATE, glproc
  Action:   ADAPT
  Priority: High
  Difficulty: M
  Dependencies: IP-03
  Notes: Move fusion detection to graph-plan time. Add at least:
    MUL_MAT+ADD (bias), MUL_MAT+RMS_NORM, ADD+ACT, ROPE+MV.

### [IP-06] Per-op use_ref toggle
  Source:   ARTX01 (ggml-cpu.cpp:280-285)
  Target:   glproc
  Action:   ADAPT
  Priority: Medium
  Difficulty: S
  Dependencies: IP-01
  Notes: Allow use_ref to be set per-op via a tensor flag, not just
    per-backend. Enables differential testing of one op.

### [IP-07] Op-hint mechanism (GGML_HINT_*)
  Source:   ARTX01 (ggml-cpu.c:1262-1265)
  Target:   glproc, GATE
  Action:   ADOPT
  Priority: Medium
  Difficulty: S
  Dependencies: IP-01
  Notes: Replicate for structurally-special matmuls GwenLand may need
    (Hadamard, block-diagonal, sparse, etc.).

### [IP-08] Cache-aligned atomics, verified line size
  Source:   ARTX01 (ggml-cpu.c:60, 489-491)
  Target:   glproc, GATE
  Action:   ADOPT
  Priority: Medium
  Difficulty: XS
  Dependencies: none
  Notes: Use std::hardware_destructive_interference_size where
    available, fallback to 64. Apply to every shared atomic in hot
    paths.

### [IP-09] Pluggable NUMA policy
  Source:   ARTX01-F09 (ggml-cpu.c:1413-1417)
  Target:   glproc
  Action:   ADAPT
  Priority: Medium
  Difficulty: M
  Dependencies: IP-04
  Notes: Keep one-chunk-per-thread fallback for NUMA. Make the policy
    a function pointer so alternatives can be tried.

### [IP-10] Runtime-mutable type-traits table
  Source:   ARTX01-F11 (ggml-cpu.c:214)
  Target:   glproc
  Action:   ADAPT
  Priority: Medium
  Difficulty: S
  Dependencies: IP-01
  Notes: Make the table mutable at runtime (atomic pointer swap or
    RCU) so tuned kernels can be installed per-dtype without a full
    buffer-type registration.

### [IP-11] Event API for CPU backend
  Source:   ARTX01-F01 (ggml-cpu.cpp:193-210)
  Target:   GATE
  Action:   REJECT llama.cpp's no-event design
  Priority: Medium
  Difficulty: M
  Dependencies: IP-03
  Notes: Expose at least a trivial event system so GATE can treat
    CPU as a peer to GPU backends. Even a no-op event API unblocks
    the scheduler.

### [IP-12] Native 512-bit AVX-512 vecdot kernels
  Source:   ARTX02-F01 (arch/x86/quants.c — entire file)
  Target:   glproc
  Action:   REJECT the absence
  Priority: High
  Difficulty: L
  Dependencies: IP-01
  Notes: Provide __m512 variants for Q4_0/Q4_K/Q6_K/IQ4_XS on IceLake+
    with 8 independent accumulators. Currently only 256-bit VNNI is
    used in arch/x86/quants.c; the 8×8 batched GEMM in repack.cpp is
    the only 512-bit path.

### [IP-13] 256-bit batched GEMM for AMD Zen4/Zen5
  Source:   ARTX03-F03 (arch/x86/repack.cpp — 8×8 batched GEMM)
  Target:   glproc
  Action:   REJECT the absence
  Priority: High
  Difficulty: L
  Dependencies: IP-12
  Notes: Zen4/Zen5 has 256-bit data paths; 512-bit instructions split
    into 2 uops. Provide a 256-bit variant of the 8×8 batched GEMM.
    Use the tinyBLAS_Q0_AVX pattern (ARTX03-F06) as the template.

### [IP-14] Baseline-NEON batched GEMV/GEMM
  Source:   ARTX04-F03 (arch/arm/quants.c — no batched paths)
  Target:   glproc
  Action:   REJECT the absence
  Priority: High
  Difficulty: L
  Dependencies: IP-01
  Notes: Baseline ARM (Cortex-A53/A55, Raspberry Pi) runs scalar
    prefill because no batched GEMV/GEMM exists for non-I8MM cores.
    Provide at least a 4×4 baseline NEON path for Q4_0/Q4_K/IQ4_NL.

### [IP-15] Fix SVE VL assert(false) landmine
  Source:   ARTX05-F02 (arch/arm/quants.c — SVE VL switch)
  Target:   glproc
  Action:   ADAPT
  Priority: High
  Difficulty: S
  Dependencies: IP-01
  Notes: Replace `default: assert(false)` in SVE vector-length switch
    with NEON-baseline fallback. Non-{128,256,512} VLs (e.g., 384 on
    some emulators) currently crash.

### [IP-16] Adopt static_assert-enforced block-layout ABI
  Source:   ARTX06-F01 (ggml-common.h — block_q4_0, block_q8_0, etc.)
  Target:   glproc, glcuda, glmetal, glvulkan
  Action:   ADOPT
  Priority: Critical
  Difficulty: S
  Dependencies: none
  Notes: Use llama.cpp's block-layout structs verbatim as GwenLand's
    on-disk weight ABI. static_assert enforces byte offsets. This
    guarantees weight file compatibility across all GwenLand backends.

### [IP-17] Adopt vec_dot_type indirection
  Source:   ARTX06-F02 (type_traits_cpu[].vec_dot_type)
  Target:   glproc, glcuda, glmetal, glvulkan
  Action:   ADOPT
  Priority: Critical
  Difficulty: S
  Dependencies: IP-01, IP-16
  Notes: Each weight format declares its expected activation format;
    the matmul path pre-converts src1 once. Q4_0→Q8_0, Q4_K→Q8_K, etc.

### [IP-18] Adopt five-pass backend assignment in GATE
  Source:   ARTX22-F05 (ggml-backend.cpp — backend scheduler)
  Target:   GATE
  Action:   ADOPT
  Priority: Critical
  Difficulty: L
  Dependencies: IP-03
  Notes: Pass 1 anchor-on-weights → pass 2 GPU stretch → pass 3
    buft-equal upgrade → pass 4 backfill → pass 5 split+copy.
    CPU-as-last-backend convention (ARTX22-F06) is essential.

### [IP-19] Reject sequential split execution
  Source:   ARTX22-F04 (ggml-backend.cpp — split execution)
  Target:   GATE
  Action:   REJECT
  Priority: High
  Difficulty: XL
  Dependencies: IP-18
  Notes: Independent splits on different backends must run concurrently.
    GATE must dispatch to backend queues with event/future tracking.

### [IP-20] Scheduler-level graph optimizer (pre-splitting)
  Source:   ARTX22-F10 (ggml-backend.cpp — per-split graph_optimize)
  Target:   GATE
  Action:   ADOPT
  Priority: High
  Difficulty: XL
  Dependencies: IP-18
  Notes: Per-split graph_optimize cannot fuse across split boundaries.
    Add a pre-splitting optimizer for constant folding, DCE, op fusion,
    topological reordering. Mirror Vulkan's anti-aliasing rollback
    pattern (ARTX18-F09).

### [IP-21] Plan-time fusion (cross-backend)
  Source:   ARTX01-F08, ARTX08-F13, ARTX15-F14, ARTX18-F09
  Target:   GATE
  Action:   ADAPT (Vulkan's pattern is the model)
  Priority: High
  Difficulty: XL
  Dependencies: IP-20
  Notes: Vulkan has the most mature fusion machinery: ~15 patterns,
    anti-aliasing rollback, plan-time detection. CPU (1 pattern) and
    CUDA (~12 patterns, execution-time) are weaker. Unify on Vulkan's
    design. Cover at minimum: MUL_MAT+ADD, MUL_MAT+RMS_NORM, ADD+ACT,
    ROPE+MV, RMS_NORM+MUL, SOFTMAX+DIAGMASK, MUL_MAT_ID+ADD.

### [IP-22] Persistent VkPipelineCache with on-disk persistence
  Source:   ARTX18-F04 (ggml-vulkan.cpp — pipeline creation)
  Target:   glvulkan
  Action:   REJECT the absence
  Priority: High
  Difficulty: M
  Dependencies: none
  Notes: llama.cpp does not use VkPipelineCache at all — pipelines are
    recompiled every run. GwenLand must add a serializable cache to
    eliminate cold-start compilation latency.

### [IP-23] Fix Vk subgroup_size=32 hardcoding in mul_mm_cm2.comp
  Source:   ARTX20-F06 (vulkan-shaders/mul_mm_cm2.comp:115)
  Target:   glvulkan
  Action:   REJECT the hardcoding
  Priority: High
  Difficulty: S
  Dependencies: none
  Notes: mul_mm_cm2.comp hardcodes subgroup_size=32 for shared-memory
    sizing. If the host mis-specializes (or the device has a different
    subgroup size), this is a silent correctness bug. Use
    gl_SubgroupSize + specialization constants.

### [IP-24] Adopt MLA V_is_K_view detection cross-backend
  Source:   ARTX11-F15, ARTX24 (FA MLA support)
  Target:   glcuda, glmetal, glvulkan
  Action:   ADOPT
  Priority: Medium
  Difficulty: M
  Dependencies: IP-18
  Notes: For MLA models (DeepSeek-V3/R1, MiMo, Mistral Small 4),
    alias V to K to halve KV bandwidth. Currently CUDA and Metal
    support this; Vulkan does not.

### [IP-25] Adopt FA op_params 4-slot layout as cross-backend contract
  Source:   ARTX24-F01 (FA_EXT op_params)
  Target:   glproc, glcuda, glmetal, glvulkan, GATE
  Action:   ADOPT
  Priority: Critical
  Difficulty: S
  Dependencies: IP-18
  Notes: scale, max_bias, logit_softcap, prec — 4-slot layout is the
    de-facto cross-backend attention contract. Formalize it in GATE.

### [IP-26] Reject CPU TENSOR_ALIGNMENT = 32
  Source:   ARTX21-F03 (CPU buffer type alignment)
  Target:   glproc
  Action:   REJECT
  Priority: High
  Difficulty: XS
  Dependencies: none
  Notes: CPU buffer type advertises 32-byte alignment — insufficient
    for AVX-512 (64) and AMX (128). Default to 64 (or 128 on AMX-
    capable CPUs).

### [IP-27] Reject sync fallback in tensor_copy_async
  Source:   ARTX21-F05 (ggml-backend.cpp — tensor_copy_async)
  Target:   GATE
  Action:   REJECT
  Priority: High
  Difficulty: M
  Dependencies: IP-19
  Notes: ggml stalls both backends when dst lacks cpy_tensor_async.
    GwenLand should require async from all backends (including CPU) and
    eliminate the sync fallback.

### [IP-28] Add FP8 KV cache path
  Source:   ARTX11-F15, ARTX17-F04
  Target:   glcuda, glmetal, glvulkan
  Action:   REJECT the absence
  Priority: High
  Difficulty: L
  Dependencies: IP-25
  Notes: No FP8 KV cache path on CUDA, Metal, or Vulkan. FP8 KV cuts
    memory bandwidth in half for long-context inference. Implement
    GGML_TYPE_F8_E4M3 KV cache support in all three GPU backends.

### [IP-29] Hopper-native FA and MMQ (wgmma + TMA)
  Source:   ARTX10-F09, ARTX11-F12
  Target:   glcuda
  Action:   REJECT the absence
  Priority: High
  Difficulty: XL
  Dependencies: IP-22
  Notes: Both MMQ and FA use Ampere-style mma.sync even on Hopper,
    leaving ~4× Tensor Core throughput on the table. Implement
    wgmma.mma_async + TMA paths for Hopper (and tc_gen5.mma for
    Blackwell when available).

### [IP-30] Adopt get_proc_address extension hook
  Source:   ARTX23-F02 (ggml-backend-impl.h — reg_i.get_proc_address)
  Target:   GATE
  Action:   ADOPT
  Priority: Medium
  Difficulty: S
  Dependencies: IP-18
  Notes: String-keyed function lookup lets backends expose optional
    APIs (set_n_threads, set_abort_callback, etc.) without bumping
    the interface version.

### [IP-31] Adopt score-based multi-binary dispatch (generalize)
  Source:   ARTX01-F12, ARTX23-F03 (CPU x86 score pattern)
  Target:   glproc, glcuda, glmetal, glvulkan
  Action:   ADOPT
  Priority: Medium
  Difficulty: M
  Dependencies: IP-18
  Notes: Generalize the CPU's per-ISA score function pattern to all
    backends. Lets the loader pick the best .so variant per device
    without runtime dispatch inside hot loops.

## Sequencing

The order of items below reflects dependency order, not priority. An
item that unblocks several others is sequenced early regardless of its
own priority.

1. IP-01 (type-traits table) — unblocks IP-02, IP-04, IP-06, IP-07, IP-10, IP-12, IP-13, IP-14, IP-15, IP-17
2. IP-08, IP-16, IP-22, IP-23, IP-26 (no-dep items) — do early
3. IP-03 (per-node parallelism) — unblocks IP-05, IP-11, IP-18
4. IP-02, IP-04, IP-12, IP-13, IP-14, IP-15, IP-17 (depend on IP-01)
5. IP-18 (five-pass scheduler) — unblocks IP-19, IP-20, IP-24, IP-25, IP-27, IP-30, IP-31
6. IP-05, IP-20, IP-21 (fusion — depend on scheduler)
7. IP-19, IP-27 (async split execution — depend on scheduler)
8. IP-28, IP-29 (GPU deep work — depend on IP-25/IP-22)
9. IP-06, IP-07, IP-10, IP-11, IP-30, IP-31 — independent refinements
