# ARTX19 — Vulkan Compute Shaders

**Audited commit:** `555881ebc8b0fc0402b30e09258a32a7bfd13c52`
**Last updated:** 2026-07-25
**Auditor:** Percival-aux
**Target GwenLand module:** `glvulkan` (shader library, kernel selection),
`GATE` (per-op kernel specialization, fusion flags)

---

## 1. Executive Summary

The Vulkan backend's kernel side is a **163-file GLSL compute-shader library**
under `ggml/src/ggml-vulkan/vulkan-shaders/`. ARTX18 covered the host-side
pipeline/descriptor/buffer machinery that compiles and dispatches these
shaders. ARTX19 covers what the shaders *do*: the GLSL kernel architecture,
the dequantize function library, the GEMV/GEMM/FA/RoPE/softmax/quantize
kernels, and the three matmul flavors (scalar, cooperative-matrix-1,
cooperative-matrix-2).

The shader library is built around six architectural decisions:

1. **Macro-dispatched dequantize functions** in `dequant_funcs.glsl`. Every
   `DATA_A_*` macro selects exactly one `vec2 dequantize(...)` and one
   `vec4 dequantize4(...)` overload via `#if defined(...)`. There is no
   runtime dispatch — the SPIR-V contains exactly one path after
   preprocessing.
2. **A types header** (`types.glsl`, 1914 lines) that declares every block
   struct (`block_q4_0`, `block_q4_k`, …) and every storage alias
   (`A_TYPE`, `A_TYPE_PACKED16`, `A_TYPE_PACKED32`, `A_TYPEV4`). Storage
   aliases let the same kernel read 1-, 2-, 4-, or 8-byte chunks of the
   same buffer depending on what's fastest on the host architecture.
3. **Three matmul flavors**: scalar (`mul_mm.comp` with no COOPMAT),
  cooperative-matrix-1 (`mul_mm.comp` with `COOPMAT`), and
  cooperative-matrix-2 (`mul_mm_cm2.comp`, NV-only). Plus a fourth path
  for integer-dot-product quantized GEMM (`mul_mmq.comp`).
4. **Spec-constant-driven tile sizes**. `BM`, `BN`, `BK`, `WM`, `WN`,
   `TM`, `TN`, `WARP`, `BLOCK_SIZE` are all `layout(constant_id = N)`
   so the same SPIR-V can be specialized per architecture and per shape.
5. **Vectorized 4-element loads** everywhere: `vec4`, `f16vec4`,
   `data_b_v4[]`, `data_a_v4[]`. Almost every hot path loads 4 floats
   per memory transaction.
6. **Two reduction strategies co-existing**: shared-memory tree reductions
   (default) and subgroup reductions (gated on `USE_SUBGROUP_ADD` /
   `USE_SUBGROUPS` / `SubGroupSize` spec constants). ARTX20 covers the
   subgroup strategy in depth.

For GwenLand, the architectural decisions worth **ADOPT**ing are: the
macro-dispatched dequantize library, the multi-view storage aliases
(`A_TYPE` / `A_TYPE_PACKED16` / `A_TYPE_PACKED32`), the spec-constant
tile sizes, and the four-flavor matmul taxonomy. The decisions worth
**REJECT**ing are the pure shared-memory tree reduction in `soft_max.comp`
and `argmax.comp` (subgroup reductions are simpler and faster on
subgroup-capable hardware), and the lack of `subgroupBroadcastFirst`-
based mask broadcast in `flash_attn.comp`'s `tmpsh` path.

---

## 2. Purpose

Provide the GLSL kernel side of the Vulkan backend:

* Implement every `ggml_op` the Vulkan backend supports as a compute
  shader (or a family of compute shaders).
* Decode every ggml quant format (Q4_0 through NVFP4, including all
  IQ-* formats) directly inside the shader that consumes them — no
  separate dequantize pass for the matmul hot path.
* Provide three matmul flavors so the host can pick the best one per
  device (scalar for old GPUs, coopmat1 for KHR-supporting GPUs,
  coopmat2 for NVIDIA Hopper/Blackwell).
* Carry tile-size and subgroup-size tuning as spec constants so the
  same compiled SPIR-V specializes per architecture at pipeline-create
  time.

It is **not** responsible for: shader compilation (`vulkan-shaders-gen`
+ `glslang`), pipeline creation (ARTX18), descriptor binding (ARTX18),
or graph scheduling (ARTX18).

---

## 3. Source Files

| File                                              | Lines  | Role                                                              |
| ------------------------------------------------- | ------ | ---------------------------------------------------------------- |
| `vulkan-shaders/types.glsl`                       | 1914   | Block struct defs, QUANT_K/R, A_TYPE / D_TYPE / FLOAT_TYPE macros |
| `vulkan-shaders/dequant_funcs.glsl`               | 727    | Macro-dispatched dequantize library for scalar/vector paths      |
| `vulkan-shaders/dequant_funcs_cm2.glsl`           | 1425   | Coopmat2-aware dequant callbacks for `coopMatLoadTensorNV`       |
| `vulkan-shaders/dequant_head.glsl`                | 13     | Header for `dequant_*.comp` shaders (push constant + types)     |
| `vulkan-shaders/generic_head.glsl`                | 11     | Header for generic elementwise shaders (push constant + types)   |
| `vulkan-shaders/utils.glsl`                       | 25     | `fastmod`, `fastdiv`, `get_indices` helpers                      |
| `vulkan-shaders/mul_mat_vec.comp`                 | 264    | GEMV main path (F32/F16/BF16 + most quants)                      |
| `vulkan-shaders/mul_mat_vec_base.glsl`            | 230    | GEMV shared infrastructure: push constants, `reduce_result`      |
| `vulkan-shaders/mul_mat_vec_iface.glsl`           | 35     | GEMV SSBO bindings + fusion flag macros                          |
| `vulkan-shaders/mul_mat_vec_q4_k.comp`            | 134    | Q4_K-specialized GEMV (super-block layout)                       |
| `vulkan-shaders/mul_mm.comp`                      | 466    | GEMM (scalar + coopmat1 paths in one shader)                     |
| `vulkan-shaders/mul_mm_funcs.glsl`                | 644    | Per-quant `load_a_to_shmem` / `load_b_to_shmem` for GEMM         |
| `vulkan-shaders/mul_mmq.comp`                     | 311    | Integer-dot-product quantized GEMM                               |
| `vulkan-shaders/mul_mmq_funcs.glsl`               | 488    | Per-quant `block_a_to_shmem` / `mmq_dot_product` for MMQ         |
| `vulkan-shaders/mul_mmq_shmem_types.glsl`         | ~100   | Per-quant `block_a_cache` / `block_b_cache` shmem types          |
| `vulkan-shaders/mul_mm_cm2.comp`                  | 658    | Cooperative-matrix-2 GEMM (NV-only)                              |
| `vulkan-shaders/mul_mm_id_funcs.glsl`             | 74     | MoE row-id compaction via `subgroupBallot`                       |
| `vulkan-shaders/dot_product_funcs.glsl`           | 27     | `dot_product` for f16vec4 (VALVE mixed-float dot) and F32         |
| `vulkan-shaders/flash_attn.comp`                  | 758    | Flash attention (scalar + MMQ paths)                              |
| `vulkan-shaders/flash_attn_base.glsl`             | 265    | FA push constants, spec constants, index init, slope/sink helpers |
| `vulkan-shaders/flash_attn_dequant.glsl`          | 132    | Aliased SSBO views + macro-expanded per-quant decode for FA      |
| `vulkan-shaders/flash_attn_cm1.comp`              | 646    | FA via KHR_cooperative_matrix                                     |
| `vulkan-shaders/flash_attn_cm2.comp`              | 481    | FA via NV_cooperative_matrix2                                     |
| `vulkan-shaders/rope_head.glsl` / `rope_funcs.glsl` | 19+210| RoPE shared infra + norm/neox/multi/vision variants              |
| `vulkan-shaders/rope_norm.comp` / `rope_neox.comp`| 17+17  | Thin RoPE entry points                                            |
| `vulkan-shaders/soft_max.comp`                    | 195    | Single-pass softmax (workgroup fits in BLOCK_SIZE shmem)         |
| `vulkan-shaders/soft_max_large_common.glsl`       | 53     | Multi-pass softmax shared infra                                   |
| `vulkan-shaders/quantize_q8_1.comp`               | 127    | F32→Q8_1 quantizer (subgroup + shmem paths)                      |
| `vulkan-shaders/argmax.comp`                      | 60     | Argmax via shmem tree reduction                                   |
| `vulkan-shaders/sum_rows.comp`                    | 47     | Row-sum via shmem tree reduction                                  |

> Note: The audit prompt's file names (`mul_mat_vec_q8_1.comp`,
> `mul_mat_q4_0.comp`, `flash_attn_vec_f16.comp`, `rope_norm_f32.comp`,
> `soft_max_f32.comp`) **do not exist** at this commit. The actual
> shaders use one shared `mul_mat_vec.comp` (parameterized by `DATA_A_*`
> macros) plus a few format-specific overrides (e.g. `mul_mat_vec_q4_k.comp`
> for Q4_K's super-block layout). GEMM uses `mul_mm.comp` /
> `mul_mmq.comp` / `mul_mm_cm2.comp`. Softmax uses `soft_max.comp` +
> `soft_max_large1/2/3.comp` for multi-pass. RoPE uses
> `rope_norm.comp` / `rope_neox.comp` / `rope_multi.comp` /
> `rope_vision.comp` — all thin wrappers around `rope_funcs.glsl`.

---

## 4. Architecture Overview

```
                  ┌──────────────────────────────────────────────┐
                  │  dequant_head.glsl / generic_head.glsl       │
                  │   push constant, #include types.glsl         │
                  └──────────────────────────────────────────────┘
                                    │
                                    ▼
                  ┌──────────────────────────────────────────────┐
                  │  types.glsl (1914 lines)                     │
                  │  ├─ block_q4_0..block_nvfp4 structs          │
                  │  ├─ QUANT_K / QUANT_R constants              │
                  │  ├─ A_TYPE / A_TYPE_PACKED16 / PACKED32      │
                  │  ├─ FLOAT_TYPE / D_TYPE / ACC_TYPE macros    │
                  │  └─ BLOCK_SIZE / LOAD_VEC_A spec constants   │
                  └──────────────────────────────────────────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              ▼                     ▼                     ▼
   ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐
   │ dequant_funcs.glsl│  │dequant_funcs_cm2  │  │ dequant_head.glsl │
   │ (vec2/vec4 macros)│  │ .glsl (callbacks) │  │ + dequant_q4_0    │
   │ for scalar GEMV/  │  │ for coopmat2 path │  │ .comp (standalone)│
   │ GEMM/FA hot paths │  │                   │  │                   │
   └───────────────────┘  └───────────────────┘  └───────────────────┘
              │                     │
              ▼                     ▼
   ┌───────────────────────────────────────────────────────────────┐
   │ Per-op shaders (.comp)                                        │
   │ ├─ mul_mat_vec.comp     (GEMV)                                │
   │ ├─ mul_mm.comp          (GEMM: scalar + coopmat1)             │
   │ ├─ mul_mmq.comp         (GEMM: integer dot product)           │
   │ ├─ mul_mm_cm2.comp      (GEMM: coopmat2 NV-only)              │
   │ ├─ flash_attn.comp      (FA: scalar + MMQ)                    │
   │ ├─ flash_attn_cm1.comp  (FA: coopmat1)                        │
   │ ├─ flash_attn_cm2.comp  (FA: coopmat2)                        │
   │ ├─ rope_{norm,neox,multi,vision}.comp (RoPE variants)         │
   │ ├─ soft_max.comp        (softmax)                             │
   │ ├─ quantize_q8_1.comp   (F32→Q8_1)                            │
   │ └─ …                                                         │
   └───────────────────────────────────────────────────────────────┘
```

Key design points:

* **No polymorphism in GLSL**. Quant format selection is by `#if
  defined(DATA_A_Q4_0)` / `#if defined(DATA_A_Q4_K)` etc. The
  `vulkan-shaders-gen` tool (see ARTX18) compiles each shader N times
  with different `-DDATA_A_*` defines, producing one SPIR-V per
  (shader, format) pair.
* **No `switch` on type**. Even where a `switch (FaTypeK)` appears
  (e.g. `flash_attn_cm2.comp:42-51`), the switch is on a spec
  constant, so the driver folds it away after specialization. This is
  documented explicitly: "After spec-constant specialization the driver
  folds away every path except the one matching the K/V type for this
  pipeline." (`flash_attn_dequant.glsl:8`).
* **Multiple views of the same buffer**. `mul_mat_vec_iface.glsl:8-25`
  declares up to four bindings for buffer A (binding 0, repeated as
  `data_a[]`, `data_a_v4[]`, `data_a_packed16[]`, `data_a_packed32[]`).
  The host assigns the same VkBuffer to all four; the shader picks the
  view that gives the widest load for the access pattern.

---

## 5. Execution Flow

### 5.1 GEMV entry (`mul_mat_vec.comp`)

`mul_mat_vec.comp:248 main()`:

1. Compute `first_row = NUM_ROWS * (gl_WorkGroupID.x + gl_NumWorkGroups.x * gl_WorkGroupID.z)`
   (`mul_mat_vec.comp:249`). Each workgroup produces `NUM_ROWS` output
   rows.
2. Bounds-check `first_row + NUM_ROWS <= p.stride_d`; if partial, run
   with smaller `num_rows` (`mul_mat_vec.comp:256-263`).
3. Call `compute_outputs(first_row, num_rows)`.

### 5.2 GEMV compute (`mul_mat_vec.comp:141`)

1. `get_offsets(a_offset, b_offset, d_offset)` — reads batch/expert
   indices from workgroup IDs (`mul_mat_vec_base.glsl:47-87`).
2. Detect `is_aligned_nonquant` — `batch_stride_b % 4 == 0 && b_offset
   % 4 == 0 && ncols % 4 == 0 && BLOCK_SIZE % 4 == 0 && K_PER_ITER ==
   4` (`mul_mat_vec.comp:145-148`).
3. Zero `temp[NUM_COLS][NUM_ROWS]` accumulator.
4. Loop with **4x manual unroll** in the body, then 2x, then 1x. The
   unroll is hand-rolled because `[[unroll]]` hints alone weren't
   sufficient on all drivers (`mul_mat_vec.comp:164-243`).
5. Each iteration calls `iter(temp, first_row, num_rows, tid, i*K_PER_ITER, lastiter)`.
6. `reduce_result(temp, d_offset, first_row, num_rows, tid)` —
   reduces partial sums across the workgroup. Three paths:
   * `USE_SUBGROUP_ADD_NO_SHMEM`: pure `subgroupAdd`, write from tid 0.
   * `USE_SUBGROUP_ADD`: `subgroupAdd` per subgroup, then shmem across
     subgroups (`tmpsh[j][n][gl_SubgroupID]`).
   * Default: shmem tree reduction `for (s = BLOCK_SIZE/2; s > 0; s >>= 1)`.

### 5.3 GEMV iteration body (`mul_mat_vec.comp:45 iter()`)

For each of `NUM_COLS` columns and `num_rows` weight rows:

1. Compute `col = i*BLOCK_SIZE + K_PER_ITER*tid`.
2. Compute `iqs = (col%QUANT_K)/QUANT_R`, `iybs = col - col%QUANT_K`.
3. Load `K_PER_ITER` elements of B as a `vec4` (or two `vec4`s for
   K_PER_ITER==8).
4. For each weight row, call `dequantize4(ib, iqs, a_offset)` to get
   a `vec4` of weight elements, then `dot(v, b)` accumulate.
5. For symmetric quants (`get_dm(ib, a_offset).y == 0`), scale by `dm.x`
   at the end; for asymmetric (Q4_1/Q5_1/Q4_K/Q5_K), apply `v = v *
   dm.x + dm.y` per element before the dot.

### 5.4 GEMM entry (`mul_mm.comp:139 main()`)

1. Decode `ir`, `ik`, `ic`, `batch_idx` from workgroup IDs.
2. For non-MUL_MAT_ID: `start_k = ik * p.k_split`, `end_k = min(p.K,
   (ik+1) * p.k_split)` — split-K is achieved by mapping the K-axis
   into `gl_WorkGroupID.x` above the M-axis.
3. For MUL_MAT_ID: load expert row indices into shared memory via
   `load_row_ids` (subgroup-ballot compaction).
4. Allocate accumulator registers: `coopmat<> sums[cms_per_row *
   cms_per_col]` for coopmat path, `ACC_TYPEV2 sums[WMITER*TM*WNITER*TN/2]`
   for scalar path (`mul_mm.comp:264, 270`).
5. K-loop: `for (block = start_k; block < end_k; block += BK)`:
   * Cooperative load to `buf_a[BM * SHMEM_STRIDE]` and
     `buf_b[BN * SHMEM_STRIDE]`.
   * `barrier()`.
   * Either `coopMatMulAdd` (coopmat path) or hand-unrolled
     `dot_product` (scalar path).
   * `barrier()`.
6. Store accumulator to `data_d[]` with bounds checks; coopmat path
   uses `coopMatStore` for aligned in-bounds case, otherwise stages
   through `coopmat_stage[]` shmem.

### 5.5 GEMM MMQ entry (`mul_mmq.comp:109 main()`)

Same overall structure as `mul_mm.comp` scalar path, but:

* B is always `block_q8_1_x4_packed128` — quantized activations.
* A's per-format `block_a_to_shmem` repacks quant blocks into
  `block_a_cache` (per-format shmem type defined in
  `mul_mmq_shmem_types.glsl`).
* `mmq_dot_product` uses `dotPacked4x8EXT` (integer SDP) to compute
  the inner product of packed 4-bit/8-bit values against packed
  8-bit Q8_1 values.
* `BK_STEP = 4` (default): processes 4 BK-tiles per K-loop iteration
  to amortize barrier cost (`mul_mmq.comp:88-92`).

### 5.6 Flash attention entry (`flash_attn.comp:81 main()`)

1. `init_indices()` — computes `i`, `iq2`, `iq3`, `start_j`, `end_j`
   from workgroup IDs and GQA/split-K parameters (`flash_attn_base.glsl:191`).
2. Load Q tile into `Qf[Br * qf_stride]` shared memory. For the MMQ
   path, also quantize Q to Q8_0/Q4_* on-the-fly using
   `subgroupClusteredMax(8)` / `subgroupClusteredAdd(8)` for the
   per-block scale (`flash_attn.comp:126, 141`).
3. Initialize `Of[r][d] = 0`, `Lf[r] = 0`, `Mf[r] = -FLT_MAX/2`.
4. **Online softmax loop** `for (j = start_j; j < end_j; ++j)`:
   a. Optional mask block skip via `data_mask_opt` (2 bits per block).
   b. Load K block into `kvsh[]` (when `SHMEM_STAGING != 0`) or read
      directly.
   c. Compute `Sf[r][c] = dot(Q, K)` per (row, col).
   d. Reduce `Sf` across `D_split` lanes via `subgroupShuffleXor`.
   e. Optional logit softcap `tanh`.
   f. Optional ALiBi mask add.
   g. Update `Mf[r] = max(rowmaxf, Moldf[r])`, `eMf[r] = exp(Moldf -
      Mf[r])`, scale `Lf[r]` and `Of[r]` by `eMf[r]`.
   h. Load V block.
   i. `Pf[r] = exp(Sf[r][c] - Mf[r])`, `Lf[r] += Pf[r]`,
      `Of[r][d] += Pf[r] * V[d]`.
5. After the K-loop: reduce `Mf`, `Lf`, `Of` across `D_split` and
   `row_split` dimensions (subgroup shuffle then shmem).
6. If split-K (`p.k_num > 1`): store `O`, `L`, `M` to separate
   buffers; the final division by `L` happens in
   `flash_attn_split_k_reduce.comp`. Otherwise: divide `Of` by `Lf`
   and store.

### 5.7 RoPE entry (`rope_neox.comp:6` / `rope_norm.comp:6`)

Per-pair-of-elements kernel:

1. `i0 = 2 * gl_GlobalInvocationID.y` — pair index.
2. `row = gl_GlobalInvocationID.x + 32768 * gl_GlobalInvocationID.z` —
   row packed across Y+Z axes (32768 = 2^15 rows per Y axis).
3. Decode `(i1, i2, i3)` from `row` via integer division.
4. Call `rope_neox(i0, i1, i2, i3, pc)` / `rope_norm(...)`.

No workgroup cooperation — workgroup size is `local_size_x = 1,
local_size_y = 256, local_size_z = 1` (`rope_head.glsl:7`). Every
invocation is fully independent.

### 5.8 Softmax entry (`soft_max.comp:174 main()`)

1. Dispatch with `num_blocks = ceil(KX / BLOCK_SIZE)` selected from
   `1, 2, 3, 4, 8, 16, 32` or `> 32` (variable). Each branch calls
   `soft_max(N)` with a constant N to enable unrolling
   (`soft_max.comp:177-194`).
2. `soft_max(N)`:
   a. Compute `slope` for ALiBi if `max_bias > 0`.
   b. **Pass 1 — max**: each thread reads `N` columns, computes `v =
      a*scale + slope*b`, caches in `data_cache[16]` (up to 16
      values), accumulates `max_val`.
   c. Tree-reduce `max_val` across `BLOCK_SIZE` via shmem.
   d. **Pass 2 — sum**: each thread reads `N` columns again, computes
      `exp(v - max_val)`, accumulates `sum`. If `idx < 16`, reuses
      cached value; otherwise recomputes from buffer.
   e. Tree-reduce `sum` across `BLOCK_SIZE` via shmem.
   f. Add sink contribution if `has_sinks != 0`.
   g. **Pass 3 — normalize**: each thread divides cached or
      buffer-resident `exp(v-max)` by `sum` and writes back.

### 5.9 Quantize Q8_1 entry (`quantize_q8_1.comp:121 main()`)

Persistent-threads loop:

1. `wgid = gl_WorkGroupID.x`.
2. `while (wgid < p.num_blocks) { quantize(wgid); wgid += gl_NumWorkGroups.x; }`
   — each workgroup processes multiple blocksizes.
3. `quantize(wgid)`:
   a. Each thread reads a `vec4` (4 floats = 4 of the 32 elements in
      a Q8_1 block). 8 threads per block.
   b. Per-thread max-of-abs via `max(max(abs_vals.x, abs_vals.y),
      max(abs_vals.z, abs_vals.w))`.
   c. **Block max** via either `subgroupClusteredMax(thread_max, 8)`
      (USE_SUBGROUPS path) or shared-memory tree reduction
      (`shmem[tid]`, 4-step reduce).
   d. `d = amax / 127.0`, `d_inv = 1/d`, `vals = round(vals * d_inv)`.
   e. Write quantized ints: `data_b[ib].qs[iqs] = pack32(i8vec4(round(vals)))`.
   f. Per-thread sum via `vals.x + vals.y + vals.z + vals.w`.
   g. **Block sum** via `subgroupClusteredAdd(thread_sum, 8)` or
      shmem tree reduction.
   h. Lane 0 writes `data_b[ib].ds = f16vec2(d, sum * d)`.

---

## 6. Data Layout

### 6.1 Block struct layout

Defined in `types.glsl:59-63` and following. Every quant format has a
matching `block_q*_` struct that mirrors the host-side `block_q*_`
in `ggml-common.h`. Examples:

```glsl
struct block_q4_0 {           // 18 bytes
    float16_t d;
    uint8_t   qs[16];         // 32 nibbles
};
struct block_q4_0_packed16 {  // 18 bytes, 16-bit aligned
    float16_t d;
    uint16_t  qs[16/2];
};
// (No packed32 variant for Q4_0: nibbles don't pack cleanly into uint32)
```

Each format defines up to three views: `block_q*_` (byte view),
`block_q*_packed16`, `block_q*_packed32`. The shader's `A_TYPE` /
`A_TYPE_PACKED16` / `A_TYPE_PACKED32` macros select which view to use
based on what gives the widest load (`types.glsl:74-77` for Q4_0).

### 6.2 Padded shared memory

For matmul: `SHMEM_STRIDE = BK/2 + 1` (scalar) or `BK/2 + 4` (coopmat)
(`mul_mm.comp:122-125`). The `+1`/`+4` padding avoids 4-way bank
conflicts on column-major access patterns.

For FA: `qf_stride = HSK/4 + 1`, `kvsh_stride = D/4 + 1`
(`flash_attn.comp:54, 67`). Same pattern.

### 6.3 Quantized activation layout (Q8_1 B matrix in MMQ)

`mul_mmq.comp:33` declares B as `block_q8_1_x4_packed128 data_b[]`.
`block_q8_1_x4` groups four Q8_1 blocks (128 elements) together so
that one `int32_t` load fetches 4 packed int8 values from the same
position across 4 blocks (`mul_mmq_shmem_types.glsl`). This is the
"Q8_1 × 4" layout, matching `GGML_TYPE_Q8_1_X4` on the host.

---

## 7. Memory Layout

### 7.1 Push constants

Every shader carries a push-constant block. Two distinct shapes:

* **Elementwise shaders** use `generic_head.glsl`'s 12-field block
  (`KX`, `KY`, `param1`..`param4`). Used by `add.comp`, `mul.comp`,
  `silu.comp`, etc.
* **Quantize/dequantize shaders** use `dequant_head.glsl`'s 5-field
  block (`M`, `K`, `stride_a`, `stride_b`, `nel`).
* **Matmul/FA/RoPE/softmax** each have their own push-constant
  struct, often conditionally laid out (MUL_MAT_ID swaps in
  `nei0/ne11/expert_i1/nbi1` for `base_work_group_y/ne02/ne12/
  broadcast2/broadcast3`, see `mul_mat_vec_base.glsl:28-40`).

### 7.2 Shared memory budget

The Vulkan spec minimum is 32 KB; the backend's pipeline creation
queries `maxComputeSharedMemorySize` (ARTX18). Hot shaders use:

* `mul_mm.comp` scalar: `buf_a[BM * SHMEM_STRIDE] + buf_b[BN *
  SHMEM_STRIDE]` = `(64 * 17) + (64 * 17) = 2176` FLOAT_TYPEV2 =
  ~17 KB at f32, ~9 KB at f16.
* `mul_mm.comp` coopmat: same + `coopmat_stage[TM * TN * NUM_WARPS]`
  = 4*2*8 = 64 floats = 256 B extra.
* `flash_attn.comp`: `Qf[Br * qf_stride] + kvsh[Bc * kvsh_stride] +
  masksh[Bc * (Br+1)] + tmpsh[]`. For `Br=Bc=32, HSK=HSV=128`: ~12 KB.
  With `LIMIT_OCCUPANCY_SHMEM > 0`, an extra `occupancy_limiter[]`
  array intentionally inflates shmem to reduce occupancy
  (`flash_attn.comp:75-104`) — a register-pressure tuning knob.

### 7.3 Storage aliases

`mul_mat_vec_iface.glsl:8-25` declares up to four `layout(binding = 0)`
aliases for buffer A:

```glsl
layout (binding = 0) readonly buffer A {A_TYPE data_a[];};
layout (binding = 0) readonly buffer AV4 {A_TYPEV4 data_a_v4[];};        // optional
layout (binding = 0) readonly buffer A_PACKED16 {A_TYPE_PACKED16 data_a_packed16[];};
layout (binding = 0) readonly buffer A_PACKED32 {A_TYPE_PACKED32 data_a_packed32[];};
```

The host binds the *same* VkBuffer to all aliases (the layout qualifier
only affects the shader's view of the memory). The shader picks the
view that gives the widest aligned load. This is the cleanest pattern
in the codebase for handling "I want to load 1, 2, 4, or 8 bytes at
once depending on alignment."

---

## 8. Parallelism Strategy

### 8.1 Workgroup size as spec constant

Every shader declares `layout(local_size_x_id = 0, local_size_y = 1,
local_size_z = 1) in;` (`mul_mat_vec.comp:8`, `mul_mm.comp:47`,
`mul_mmq.comp:24`, `flash_attn_base.glsl:2`, etc.). The actual
`WorkGroupSize` is a specialization constant supplied at pipeline
creation. The host picks `subgroup_size * 4` for most matmul pipelines
(ARTX18 §5).

### 8.2 Per-op parallelism scheme

| Op family                       | Scheme                                                                       |
| ------------------------------- | ---------------------------------------------------------------------------- |
| GEMV (`mul_mat_vec.comp`)       | 1 WG per `NUM_ROWS` output rows; all threads cooperate on K-axis reduction   |
| GEMM (`mul_mm.comp`)            | 1 WG per `(M_tile, N_tile, batch, k_split)`; tile-blocked, warp-tiled        |
| GEMM MMQ (`mul_mmq.comp`)       | Same as GEMM but BK_STEP=4 inner unroll                                      |
| GEMM coopmat2 (`mul_mm_cm2`)    | 1 WG per `(M_tile, N_tile, batch, k_split)`; cooperative matrix handles tiling |
| Flash attention (`flash_attn`)  | 1 WG per `(Q_row_block, head, batch)`; Br × Bc tile per WG                   |
| RoPE (`rope_*.comp`)            | 1 thread per pair of elements; no cooperation                                |
| Softmax (`soft_max.comp`)       | 1 WG per row; BLOCK_SIZE threads cooperate                                   |
| Softmax large (`soft_max_large`)| Multi-WG per row + reduction kernels                                         |
| Quantize Q8_1                   | Persistent WGs; each handles multiple blocks                                 |
| Argmax, sum_rows                | 1 WG per row; BLOCK_SIZE threads tree-reduce                                 |
| Cumsum                          | 1 WG per row; subgroup scan + cross-subgroup shmem                           |
| Top-K (nary_search)             | 1 WG per row; subgroup ballot + bucket-count radix                          |

### 8.3 Split-K

Both GEMM and FA support split-K: the K-axis is divided into `k_num`
chunks, each chunk produces a partial output, a separate reduce kernel
(`mul_mat_split_k_reduce.comp`, `flash_attn_split_k_reduce.comp`)
sums them.

`mul_mat_split_k_reduce.comp:30-32` is straightforward: each
invocation handles 4 consecutive elements (vec4 load), sums across
`p.k_num` partials, writes back. No inter-thread cooperation.

For FA, split-K stores per-row `L` and `M` alongside `O`, and the
reduction kernel (`flash_attn_split_k_reduce.comp`) computes the
final `O / L` after combining across splits.

---

## 9. SIMD / GPU Strategy

### 9.1 Three matmul flavors

| Flavor                            | Extension(s)                                              | Use case                                       |
| --------------------------------- | --------------------------------------------------------- | ---------------------------------------------- |
| Scalar (`mul_mm.comp` no COOPMAT) | None (just `GL_EXT_shader_*_storage`)                     | Old GPUs, Intel pre-Xe2, fallback              |
| Coopmat1 (`mul_mm.comp` COOPMAT)  | `GL_KHR_cooperative_matrix` + `GL_KHR_memory_scope_semantics` | AMD, Intel Xe2, NVIDIA non-Hopper              |
| Coopmat2 (`mul_mm_cm2.comp`)      | `GL_NV_cooperative_matrix2` + `GL_EXT_buffer_reference` + `GL_NV_cooperative_matrix_decode_vector` | NVIDIA Hopper/Blackwell |
| MMQ (`mul_mmq.comp`)              | `GL_EXT_integer_dot_product`                              | Quantized GEMM on integer-SDP-capable GPUs    |

### 9.2 Cooperative matrix shape selection

Coopmat1 uses `coopmat<FLOAT_TYPE, gl_ScopeSubgroup, TM, TK,
gl_MatrixUseA>` with `TM=4, TN=2, TK=1` (spec constants,
`mul_mm.comp:107-109`). Scope is `gl_ScopeSubgroup` — one cooperative
matrix lives in one subgroup.

Coopmat2 uses `coopmat<MAT_TYPE, gl_ScopeWorkgroup, BM, BK,
gl_MatrixUseA>` — scope is `gl_ScopeWorkgroup`, so the matrix is
distributed across the *whole* workgroup, not just one subgroup.
`coopMatLoadTensorNV` reads directly from the SSBO with a
`tensorLayoutNV` describing strides and block sizes, calling back
into `dequantFuncA` per element for quants (`mul_mm_cm2.comp:381`).

### 9.3 Vectorized loads

Almost every hot loop uses `vec4` loads:

* GEMV: `data_b_v4[(...)/4]` (`mul_mat_vec.comp:56-62`)
* GEMM shmem load: `FLOAT_TYPEV4` and `FLOAT_TYPEV8` when
  `LOAD_VEC_A == 4` or `== 8` (`mul_mm_funcs.glsl:5-23`)
* FA Q load: `data_qv4[]` (`flash_attn.comp:38`)
* FA K/V load: `data_kv4[]`, `data_vv4[]` (`flash_attn.comp:40-42`)
* Quantize Q8_1: `vec4 data_a[]` (`quantize_q8_1.comp:26`)

### 9.4 Integer dot product (MMQ)

`mul_mmq.comp:7` requires `GL_EXT_integer_dot_product`. The hot path
is `dotPacked4x8EXT(qs_a, qs_b)` which performs 4-way int8 SDP in a
single instruction. Per-format `mmq_dot_product` in
`mul_mmq_funcs.glsl` unpacks quant-specific bits into `int32_t qs[8]`
then chains 4-8 `dotPacked4x8EXT` calls.

For Q8_0 K in flash attention, the same pattern applies:
`flash_attn.comp:418` calls `dotPacked4x8EXT(Qf[qib].qs[qiqs + d],
k_quants[d])`.

### 9.5 VALVE mixed-float dot product

`dot_product_funcs.glsl:1-14` declares a `spirv_instruction` for
`SPV_VALVE_mixed_float_dot_product` (capability 6912, id 6916). This
is the `v_dot2_f32_f16` instruction that does a 2-way f16 dot with
f32 accumulate in one instruction — used by the FA scalar path when
`DOT2_F16` is defined. AMD/RDNA-specific optimization.

---

## 10. Quantization Strategy

The dequantize function library is the *contract* between the host
and the kernel side. It is structured as follows:

### 10.1 Per-format dispatch via `DATA_A_*` macros

`dequant_funcs.glsl:1-3` enables `int8` arithmetic types unless F32
or F16 is selected. Then for each format, a `#if defined(DATA_A_Q4_0)`
block defines:

```glsl
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    const uint vui = uint(data_a[a_offset + ib].qs[iqs]);
    return (vec2(vui & 0xF, vui >> 4) - 8.0f);
}
vec4 dequantize4(uint ib, uint iqs, uint a_offset) {
    const uint vui = uint(data_a_packed16[a_offset + ib].qs[iqs/2]);
    return (vec4(vui & 0xF, (vui >> 4) & 0xF, (vui >> 8) & 0xF, vui >> 12) - 8.0f);
}
```

The `vec2` overload returns 2 elements (paired by `iqs`); the `vec4`
overload returns 4 elements (paired by `iqs/2` from the packed16 view
— wider load). For F32/F16/BF16, a `dequantize1` (single) and
`dequantize4_2aligned` (4-wide aligned) overload also exist.

### 10.2 Scale/zero-point: `get_dm`

`dequant_funcs.glsl:546-591` defines `vec2 get_dm(uint ib, uint
a_offset)` returning `(d, m)` for each format:

* Symmetric quants (Q4_0, Q5_0, Q8_0, IQ2_*, IQ3_*, IQ4_*):
  `vec2(float(data_a[ib].d), 0)`.
* Asymmetric quants (Q4_1, Q5_1): `vec2(data_a_packed32[ib].dm)` —
  reads the packed (d, m) half2 directly.
* K-quants (Q2_K..Q6_K): `vec2(1, 0)` — the K-quant `dequantize`
  function bakes the scale into the output directly, so `get_dm`
  returns identity.
* MXFP4: `vec2(e8m0_to_fp32(data_a[ib].e), 0)` — block-scaled FP4.
* NVFP4: `vec2(1.0, 0.0)` — scale is per-sub-block, baked into
  `dequantize`.

The GEMV iteration body (`mul_mat_vec.comp:80-91`) uses `get_dm` to
decide whether to apply symmetric (`temp *= dm.x` at end) or
asymmetric (`v = v * dm.x + dm.y` per element) scaling. This is a
clever factoring: the iteration body is identical for symmetric and
asymmetric quants; only the scale application differs.

### 10.3 Lookup tables for IQ formats

IQ1_S, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS, IQ4_NL all
use lookup tables declared in `types.glsl`:

* `iq1s_grid[256]` — 12-bit codes
* `iq2xxs_grid[256]` — 4-element u8vec4 grid
* `iq2xs_grid[512]` — 4-element u8vec4 grid
* `iq2s_grid[1024]` — 4-element u8vec4 grid
* `iq3xxs_grid[256]` — uint32 grid
* `iq3s_grid[512]` — uint32 grid
* `kvalues_iq4nl[16]` — float LUT for IQ4_NL
* `kvalues_mxfp4[16]` — float LUT for MXFP4

These are populated by `vulkan-shaders-gen.cpp` (which emits them as
hardcoded GLSL `const` arrays — see ARTX18). The shader's
`dequantize4` for IQ4_NL is just `dl * vec4(kvalues_iq4nl[qs.x], …)`
(`dequant_funcs.glsl:480-489`) — a 16-entry LUT lookup.

### 10.4 Cooperative-matrix-aware dequant (`dequant_funcs_cm2.glsl`)

This 1425-line file defines per-format *callbacks* for
`coopMatLoadTensorNV`:

```glsl
float16_t dequantFuncQ4_0(const in decodeBufQ4_0 bl,
                          const in uint blockCoords[2],
                          const in uint coordInBlock[2]) {
    const float16_t d = bl.block.d;
    const uint idx = coordInBlock[1];
    ...
}
f16vec4 dequantFuncQ4_0_v(...) { ... } // V=4 vector variant
```

The `decodeBufQ4_0` is a `buffer_reference`-typed view
(`dequant_funcs_cm2.glsl:69-71`) that lets the coopmat2 hardware read
the block directly via `GL_EXT_buffer_reference`. The callback runs
per-element (or per-4-elements for `_v`) inside the cooperative
matrix load.

Lines 1350-1424 macro-dispatch `dequantFuncA` / `dequantFuncA_v` based
on `DATA_A_*`. The host passes these as the optional decode callback
to `coopMatLoadTensorNV` (`mul_mm_cm2.comp:381`).

For Q4_K and Q5_K, the file also defines `fetch_scalesQ4_K` /
`store_scalesQ4_K` — these precompute per-block scales into shared
memory so the per-element decode callback only does a shmem lookup,
not a global-memory scale fetch (`dequant_funcs_cm2.glsl:1380-1386`).
This is the K-quant "scale prefetch" optimization.

---

## 11. Correctness Analysis

Static analysis found the following correctness-relevant properties.

### 11.1 Floating-point reassociation

* **GEMV `dot(v, b)`** (`mul_mat_vec.comp:97, 110, 135`). Each `dot`
  is a 4-way FMA chain, then summed across `BLOCK_SIZE` threads.
  Reduction order differs from a scalar left-to-right sum.
* **GEMM scalar path** (`mul_mm.comp:341-342`). `dot_product` is a
  4-way FMA chain (`dot_product_funcs.glsl:18-21`). Per-warp sums
  then accumulate across BK tiles.
* **MMQ integer accumulator** (`mul_mmq_funcs.glsl:34-40`). Integer
  SDP accumulation in `int32_t`, converted to float only at the end:
  `ACC_TYPE(float(cache_a[ib_a].dm) * (float(q_sum) *
  float(cache_b.ds.x) - float(cache_b.ds.y)))`. Integer accumulation
  is exact; the precision loss is in the final float scale.
* **FA online softmax**. Standard flash-attention reassociation:
  `Of += Pf * V` after `Mf` update. Determinism depends on the order
  of K-block iteration, which is fixed by the workgroup ID. Cross-
  split-K reduction may reorder; the reduce kernel sums in index
  order, so split-K is deterministic for a fixed `k_num`.

### 11.2 Approximate math

* **`exp` in softmax and FA**. GLSL `exp` is the GPU's intrinsic
  transcendental, accurate to ~ULP. FA uses `exp(Mold - Mnew)` for
  the rescale factor — when `Mold - Mnew` is very negative (a much
  larger new max), this underflows to 0, which is correct (the old
  contributions vanish).
* **`tanh` for logit softcap** (`flash_attn.comp:442`).
  `p.logit_softcap * tanh(Sf[r][c])` — used by Gemma-2 and Grok.
  Precision is the GPU's `tanh` intrinsic.
* **`pow(p.theta_scale, i0/2.0f)` in RoPE** (`rope_funcs.glsl:60,
  97`). `theta_scale = 1.0 / (theta_base^(2/d))` is precomputed on
  the host and passed as a uniform. Per-thread `pow` could be replaced
  by an iterative multiply, but the GPU's `pow` is fast enough.

### 11.3 Precision reduction

* **Quantized activations**. MMQ and FA-MMQ convert Q (the
  activations) to Q8_0/Q4_* inside the shader before the integer dot
  product (`flash_attn.comp:122-131`). This is lossy and deliberate
  — it's the whole point of the MMQ path. The output is then
  rescaled by `Qf.ds.x * k_dm.x` to recover an approximate F32 result.
* **F16 accumulation in coopmat2**. `mul_mm_cm2.comp:368` declares
  `coopmat<ACC_TYPE, gl_ScopeWorkgroup, BM, BNover4,
  gl_MatrixUseAccumulator>`. `ACC_TYPE` defaults to `float` but can
  be `float16_t` for the FP16-accumulator variants (selected by the
  host via spec constant). FP16 accumulation can overflow on large
  K — the `ACC_TYPE_MAX` clamp at lines 402-404, 446-448, etc.
  mitigates this.
* **`FATTN_KQ_MAX_OFFSET = 3.0f*0.6931f`** (`flash_attn_base.glsl:257`).
  A bias added to the softmax max to keep `exp(S - M)` in fp16 range.
  Documented as based on `ggml-cuda issue #18606`.

### 11.4 Non-deterministic reductions

* **Split-K**. When `k_num > 1`, the partial outputs are summed in
  `mul_mat_split_k_reduce.comp:30-32`. The sum order is fixed by
  iteration order, so split-K is *deterministic for a fixed k_num*.
  Different `k_num` values give different roundoff.
* **Subgroup reductions**. `subgroupAdd`, `subgroupMin`, etc. have
  implementation-defined reduction order. The result is
  bit-reproducible per (vendor, driver, subgroup size) but not across
  vendors.
* **Atomic-free**. None of the audited shaders use `atomicAdd` on
  float outputs. The only atomic in the audit is in
  `topk_nary_search.comp:118` (`atomicAdd(counts[bucket], 1)` for
  bucket counting — integer, no precision concern).

### 11.5 Architecture-specific assumptions

* **`WARP = 32`** (`mul_mm.comp:111`, `mul_mmq.comp:80`). On a 64-wide
  subgroup GPU (e.g., some Intel Xe), the warp-tile math
  (`warp_i = gl_LocalInvocationID.x / WARP`) would be wrong. The host
  must specialize `WARP` to the actual subgroup size at pipeline
  creation.
* **`subgroupClusteredMax(8)` in quantize_q8_1.comp:77** assumes
  subgroup size ≥ 8. True on every Vulkan-conformant GPU.
* **`subgroup_size = 32`** constant in `mul_mm_cm2.comp:115` — used
  only for sizing the `ballots_sh[]` shmem array. Host must
  specialize.
* **`OLD_AMD_WINDOWS`** flag in `flash_attn.comp:26, 618-624`. A
  workaround for an AMD RDNA2 Windows driver bug where
  `subgroupShuffleXor` on `f16vec4` produces wrong results. The
  workaround shuffles `vec4` (F32) instead, paying a conversion cost.
  See llama.cpp issue #19881.
* **`subgroupClusteredAdd(partial, 2u)`** etc. in
  `gated_delta_net.comp:74-84`. `clusterSize` must be a compile-time
  constant in GLSL, so the code has a switch statement with one case
  per power of two. Comment at line 71 acknowledges this is a
  workaround for a GLSL spec limitation.

### 11.6 Out-of-bounds handling

GEMV has extensive OOB handling in `load_b` and `iter`
(`mul_mat_vec.comp:19-43, 95-111`): the `lastiter` flag triggers
per-element bounds checks against `p.ncols`, falling through 4-wide →
3-wide → 2-wide → 1-wide dot product. This is verbose but correct.

GEMM uses the `Clamp` spec constant / `tensorLayoutAClamp` path
(`mul_mm_cm2.comp:329, 633`) for OOB tiles — the coopmat2 hardware
handles clamping via the tensor layout's
`gl_CooperativeMatrixClampModeConstantNV`. The scalar path uses
explicit `if (dr_warp + 2*cr < p.M && dc_warp + cc < p.N)` bounds
checks (`mul_mm.comp:454-459`).

---

## 12. Optimization Analysis

### 12.1 Identified optimizations

| Optimization                                  | Where                                                   | Notes                                                                |
| --------------------------------------------- | ------------------------------------------------------- | -------------------------------------------------------------------- |
| Macro-dispatched dequantize functions         | `dequant_funcs.glsl:1-727`                              | Zero runtime cost; one path per SPIR-V.                              |
| 4-element vectorized loads                    | `mul_mat_vec.comp:56-62`, `mul_mm_funcs.glsl:5-23`      | 4× load throughput vs scalar loads.                                  |
| 8-element vectorized loads                    | `mul_mm_funcs.glsl:3-13` (`LOAD_VEC_A == 8`)            | 8× load throughput when alignment allows.                            |
| Multiple storage views of same buffer         | `mul_mat_vec_iface.glsl:8-25`                           | Lets the shader pick the widest aligned load.                        |
| Padded shared memory (`SHMEM_STRIDE = BK/2+1`)| `mul_mm.comp:122-125`                                   | Avoids bank conflicts on column-major access.                        |
| Manual 4x/2x loop unrolling                   | `mul_mat_vec.comp:164-243`                              | `[[unroll]]` alone insufficient on some drivers.                     |
| Spec-constant tile sizes                      | `mul_mm.comp:102-111`                                   | Same SPIR-V specializes per architecture.                            |
| Three matmul flavors                          | `mul_mm.comp` / `mul_mmq.comp` / `mul_mm_cm2.comp`      | Best path per device capability.                                     |
| Cooperative matrix multiply-accumulate        | `mul_mm.comp:301-313` (coopmat1), `mul_mm_cm2.comp:384` | Hardware matrix unit.                                                |
| Integer SDP for quantized GEMM                | `mul_mmq.comp:7`, `mul_mmq_funcs.glsl:36`               | `dotPacked4x8EXT` — 4-way int8 SDP per instruction.                  |
| VALVE mixed-float dot                         | `dot_product_funcs.glsl:4-9`                            | `v_dot2_f32_f16` — AMD-specific 2-way f16 dot with f32 acc.          |
| Online softmax in flash attention             | `flash_attn.comp:457-535`                               | O(N²) memory → O(N) memory.                                          |
| Mask block-skip via 2-bit summary             | `flash_attn.comp:199-208`                               | `data_mask_opt` summarizes Br×Bc mask blocks.                        |
| `subgroupAll` mask-skip                       | `flash_attn.comp:229`                                   | Skips K-block load if entire mask is -inf.                           |
| Persistent threads (quantize Q8_1)            | `quantize_q8_1.comp:123-126`                            | Avoids launch overhead for many-block inputs.                        |
| Subgroup-clustered reductions (Q8_1)          | `quantize_q8_1.comp:77, 106`                            | 8-lane cluster matches Q8_1 block size.                              |
| Split-K with separate reduce kernel           | `mul_mat_split_k_reduce.comp`, `flash_attn_split_k_reduce.comp` | Parallelizes K dimension; reduce is cheap.                  |
| On-the-fly Q quantization (FA MMQ)            | `flash_attn.comp:118-147`                               | Avoids separate Q quantization pass.                                 |
| `occupancy_limiter` shmem tuning              | `flash_attn.comp:75-104`                                | Intentionally inflates shmem to reduce occupancy when reg pressure is high. |
| Softmax template specialization by `num_blocks` | `soft_max.comp:177-194`                               | Unrolls `num_iters`-loop for small KX.                              |
| `data_cache[16]` in softmax                   | `soft_max.comp:75, 97-99`                               | Caches first 16 elements to avoid re-reading in pass 2/3.            |
| Scale prefetch for K-quants (coopmat2)        | `dequant_funcs_cm2.glsl:1380-1386`                      | Pre-fetches per-block scales into shmem.                             |
| RoPE+VIEW+SET_ROWS fusion                     | `rope_funcs.glsl:46-50, 83-87`                          | Fuses RoPE with output indexing for sparse MoE.                      |
| RoPE+RMS_NORM fusion                          | `rope_funcs.glsl:8-14` (`RMS_NORM_ROPE_FUSION`)         | Reads from shmem (RMS output) instead of global.                     |
| `MAT_VEC_FUSION_FLAGS_BIAS0/1/SCALE0/1`       | `mul_mat_vec_iface.glsl:3-6`, `mul_mat_vec_base.glsl:104-122` | MoE bias/scale fused into GEMV reduction.                    |

### 12.2 Optimizations not present

* **No `subgroupBroadcastFirst` in mask broadcast**. FA's
  `tmpsh[gl_SubgroupID]` mask broadcast (`flash_attn.comp:231-237`)
  uses shmem + barrier where a single `subgroupBroadcastFirst` +
  cross-subgroup shmem (only if `gl_NumSubgroups > 1`) would suffice.
* **No `subgroupBallot`-based compaction in argmax**. `argmax.comp`
  uses pure shmem tree reduction. A subgroup-ballot compaction
  (like `mul_mm_id_funcs.glsl:44`) could reduce register pressure.
* **No kernel fusion for `MUL_MAT + RMS_NORM`**. ARTX18 noted the
  host side has no fusion for this pattern; the kernel side has no
  fused shader either.
* **No software prefetching** in matmul. The K-loop relies on
  hardware prefetchers across the `barrier()`.
* **No persistent-threads for GEMM**. Each workgroup does one
  (M, N, K) tile and exits. Persistent threads would let the host
  amortize launch cost for many small matmuls.
* **No `GL_EXT_shader_atomic_int64`**. The shaders avoid atomics
  entirely (except `atomicAdd` on `counts[]` in topk_nary_search);
  split-K uses separate output buffers + reduce kernel instead of
  atomic accumulation.

---

## 13. Architectural Strengths

1. **Macro-dispatched dequantize library is a clean ABI**. Adding a
   new quant format = adding one `#if defined(DATA_A_NEW)` block to
   `dequant_funcs.glsl` and one set of types in `types.glsl`. No
   other shader code changes. This is the single best design
   decision in the shader library.

2. **Multiple storage views of the same buffer**. The
   `data_a` / `data_a_packed16` / `data_a_packed32` / `data_a_v4`
   aliasing pattern (`mul_mat_vec_iface.glsl:8-25`) is elegant: one
   VkBuffer, four shader views, no copying. Lets each access pattern
   pick its optimal load width.

3. **Spec-constant tile sizes**. The same `mul_mm.comp` SPIR-V
   specializes to BM=32 on small GPUs and BM=64 on big GPUs without
   recompilation. The host picks tile sizes per architecture at
   pipeline-create time.

4. **Three matmul flavors with shared infrastructure**.
   `mul_mm_funcs.glsl` is shared between scalar and coopmat1 paths;
   only the inner K-loop differs. `mul_mm_cm2.comp` is separate
   (different extensions, different memory model) but uses the same
   `dequant_funcs_cm2.glsl` callback pattern. The separation is
   principled.

5. **Online softmax in FA with split-K reduction**. The FA shader
   is the most complex in the codebase (758 lines) but cleanly
   separates: online softmax per K-block, then optional split-K
   reduce. The `L`/`M` side outputs (`flash_attn.comp:694-696`) make
   the reduce kernel trivial.

6. **Persistent threads in `quantize_q8_1.comp`**. Simple, correct,
   and amortizes launch overhead for the common case of many-block
   inputs.

7. **`OLD_AMD_WINDOWS` workaround flag**. Documented with issue
   link, gated by a spec constant, has a fallback path. Good
   evidence-based workaround engineering.

8. **`FATTN_KQ_MAX_OFFSET` bias**. Prevents fp16 overflow in FA
   softmax with a documented derivation.

---

## 14. Architectural Weaknesses

### W1 — `soft_max.comp` uses pure shared-memory tree reduction

**Evidence**: `soft_max.comp:103-110, 143-150`. Two passes (max,
sum), each doing `for (s = BLOCK_SIZE/2; s > 0; s >>= 1)`. No
`subgroupMax` / `subgroupAdd` anywhere in the file.

**Impact**: On subgroup-capable hardware, this is 5× slower than
necessary. The tree reduction takes `log2(BLOCK_SIZE)` barrier rounds;
a subgroup reduction takes one instruction. Compare with
`quantize_q8_1.comp:77` which uses `subgroupClusteredMax(8)` for the
same kind of per-block max.

**Why it's hard to fix**: The shader pre-dates widespread subgroup
support and would need a `USE_SUBGROUPS` spec-constant path (like
`quantize_q8_1.comp` does). Low-priority because softmax is rarely
the bottleneck.

### W2 — `argmax.comp` and `sum_rows.comp` use pure shmem tree reduction

**Evidence**: `argmax.comp:46-55`, `sum_rows.comp:36-42`. Same
pattern as W1. No subgroup operations.

**Impact**: Same as W1 — leaving subgroup performance on the table.
The `cumsum.comp` shader in the same family *does* use
`subgroupExclusiveAdd` (`cumsum.comp:54`), proving the codebase
knows how.

### W3 — FA's cross-subgroup `tmpsh` reduction uses shmem even when one subgroup would suffice

**Evidence**: `flash_attn.comp:231-237`. `subgroupAll` produces a
single bool; then `if (gl_SubgroupInvocationID == 0) tmpsh[gl_SubgroupID]
= ...; barrier(); for (s = 0; s < gl_NumSubgroups; ++s) max_mask =
max(max_mask, tmpsh[s]);`. The whole shmem round-trip is unnecessary
when `gl_NumSubgroups == 1` — a `subgroupBroadcastFirst(all_less)`
would suffice. The shader doesn't special-case this.

**Impact**: One unnecessary barrier per mask-skip check. For long
sequences with many K-blocks, this adds up.

### W4 — `mul_mat_vec.comp` has hand-rolled 4x/2x/1x unroll that's fragile

**Evidence**: `mul_mat_vec.comp:164-243`. Three separate while loops
with `[[unroll]] for (k = 0; k < unroll_count; ++k)` bodies. The
`unroll_count` is mutated mid-function (`= 4` then `= 2`), and the
loop bodies are duplicated for `is_aligned_nonquant` vs quant paths.

**Impact**: Hard to maintain; a change to the iteration body must be
made in up to 6 places. A `[[unroll]]`-hint-based loop with a single
body would be cleaner if drivers honored it.

### W5 — Per-format `mmq_dot_product` functions are duplicated boilerplate

**Evidence**: `mul_mmq_funcs.glsl:33-40, 71-90, 122-145, 383-397, 434-455`.
Each format has a near-identical `mmq_dot_product` body: zero
`q_sum`, unroll over `iqs`, `dotPacked4x8EXT`, final scale. The only
differences are the unpacking pattern and the final scale formula.

**Impact**: Adding a new quant format = copying ~30 lines and tweaking
2-3 lines. A template/macro approach would be cleaner.

### W6 — `mul_mm_cm2.comp` is 658 lines with massive code duplication

**Evidence**: `mul_mm_cm2.comp:367-497` — three near-identical
branches for `BNover4`, `BNover2`, `BN` cases. Each has the same
K-loop structure with different coopmat shapes. Lines 514-657 repeat
the same three-branch structure for the MUL_MAT_ID / clamped path.

**Impact**: Maintenance burden; the file is hard to read. A loop over
shape variants with the shape as a runtime constant would be cleaner
(at the cost of one extra branch).

### W7 — `dequant_head.glsl` includes `types.glsl` but `generic_head.glsl` does not

**Evidence**: `dequant_head.glsl:13` `#include "types.glsl"`;
`generic_head.glsl:11` ends without include. Each consumer of
`generic_head.glsl` must include `types.glsl` separately (e.g.
`argmax.comp:4`).

**Impact**: Inconsistency that surprises readers. Minor.

### W8 — No shader-level asserts / debug validation

**Evidence**: No use of `GL_KHR_shader_subgroup`'s
`subgroupElect`-based assertion pattern; no use of
`debugPrintfEXT` anywhere in the audited files.

**Impact**: Shader bugs manifest as silent wrong results. The host
must rely on `GGML_VULKAN_VALIDATE` (ARTX18) for validation. A
debug-mode shader with assertions would catch issues earlier.

### W9 — `rope_funcs.glsl` re-computes `theta_base` per pair

**Evidence**: `rope_funcs.glsl:60, 97, 142-162`. Each call to
`rope_norm` / `rope_neox` / `rope_multi` recomputes `pow(p.theta_scale,
i0/2.0f)` from scratch. The result depends only on `i0`, which is the
same for many invocations within a head.

**Impact**: Minor — GPU `pow` is fast. But a per-thread cache or
shared-memory precomputation would save instructions.

### W10 — FA `OLD_AMD_WINDOWS` workaround runs even on non-AMD hardware

**Evidence**: `flash_attn.comp:617-625`. The `if (!OLD_AMD_WINDOWS)`
branch is gated by a spec constant, so it *is* folded away at
specialization. But the host must set `OLD_AMD_WINDOWS` correctly per
device — if not, the workaround path runs and pays a conversion cost
(`vec4(Of[r][d])` instead of `Of[r][d]`).

**Impact**: Correct but suboptimal if the host's device detection is
wrong. The condition is in `flash_attn_base.glsl:26` as a flag bit.

---

## 15. GwenLand Mapping

| GwenLand module | ADOPT / ADAPT / REJECT / MONITOR / DEFER | What | Reasoning |
| --------------- | ---------------------------------------- | ---- | --------- |
| `glvulkan`      | **ADOPT** | Macro-dispatched dequantize library (`dequant_funcs.glsl` pattern) | Zero-cost dispatch; one SPIR-V per format; trivial to extend. |
| `glvulkan`      | **ADOPT** | Multiple storage views (`A_TYPE`/`PACKED16`/`PACKED32`/`V4`) of same buffer | One VkBuffer, four views, optimal load width per access. |
| `glvulkan`      | **ADOPT** | Spec-constant tile sizes (`BM`, `BN`, `BK`, `WM`, …) | Same SPIR-V specializes per architecture. |
| `glvulkan`      | **ADOPT** | Four-flavor matmul taxonomy (scalar / coopmat1 / coopmat2 / MMQ) | Clean separation by hardware capability. |
| `glvulkan`      | **ADOPT** | `MAT_VEC_FUSION_FLAGS_*` bias/scale fusion in GEMV reduction | MoE hot-path optimization. |
| `glvulkan`      | **ADOPT** | RoPE+VIEW+SET_ROWS fusion pattern | Sparse MoE output indexing. |
| `glvulkan`      | **ADAPT** | FA online softmax + split-K | Keep the structure; consider `subgroupBroadcastFirst` for mask broadcast. |
| `glvulkan`      | **ADAPT** | `quantize_q8_1.comp` dual-path (subgroup + shmem) | Keep the dual path; add a runtime fallback when subgroup_clustered is unavailable. |
| `glvulkan`      | **REJECT**| Pure shmem tree reduction in `soft_max.comp` / `argmax.comp` / `sum_rows.comp` | Use subgroup reductions; fall back to shmem only on non-subgroup hardware. |
| `glvulkan`      | **REJECT**| Hand-rolled 4x/2x/1x unroll in `mul_mat_vec.comp` | Use `[[unroll]]` hints + driver testing; collapse to one body. |
| `glvulkan`      | **MONITOR**| VALVE mixed-float dot product | AMD-specific; monitor for cross-vendor standardization. |
| `glvulkan`      | **MONITOR**| `OLD_AMD_WINDOWS` workaround | Monitor AMD driver fixes; remove when no longer needed. |
| `glvulkan`      | **DEFER** | `occupancy_limiter` shmem tuning | Adopt only if GwenLand's FA shader has the same occupancy issue. |
| `GATE`          | **ADOPT** | Push-constant `fusion_flags` bitfield | Clean way to fuse bias/scale into matmul without per-fusion shader variants. |
| `GATE`          | **ADOPT** | Per-(shader × spec-constants × subgroup-size) SPIR-V cache key | Already adopted on host side (ARTX18); kernel side provides the spec-constant axes. |

---

## 16. Recommendations

### R1 — ADOPT macro-dispatched dequantize library
**Priority:** Critical
**Difficulty:** M
**Dependencies:** none
GwenLand's `glvulkan` should define an equivalent `dequant_funcs.glsl`
with one `#if defined(GL_DATA_A_*)` block per quant format. Each block
defines `vec2 dequantize(...)`, `vec4 dequantize4(...)`, and
`vec2 get_dm(...)`. Same macro structure, same semantics.

### R2 — ADOPT multiple storage views of the same buffer
**Priority:** High
**Difficulty:** S
**Dependencies:** R1
GwenLand's `glvulkan` should declare `data_a`, `data_a_packed16`,
`data_a_packed32`, `data_a_v4` as aliased bindings (same `binding = N`,
different types). The host binds the same VkBuffer to all; the shader
picks the optimal view per access pattern.

### R3 — ADOPT spec-constant tile sizes
**Priority:** High
**Difficulty:** S
**Dependencies:** none
GwenLand's `glvulkan` matmul shaders should declare `BM`, `BN`, `BK`,
`WM`, `WN`, `TM`, `TN`, `WARP`, `BLOCK_SIZE` as `layout(constant_id)`.
The host specializes per architecture at pipeline-create time.

### R4 — REJECT pure shmem tree reduction in softmax/argmax/sum_rows
**Priority:** High
**Difficulty:** S
**Dependencies:** ARTX20-R1 (subgroup reduction strategy)
GwenLand's `glvulkan` softmax/argmax/sum_rows kernels should use
`subgroupMax`/`subgroupAdd`/`subgroupBallot` reductions. Provide a
shmem fallback path for non-subgroup hardware (the
`quantize_q8_1.comp` pattern).

### R5 — ADOPT FA online softmax with `subgroupBroadcastFirst` for mask broadcast
**Priority:** High
**Difficulty:** M
**Dependencies:** R3
Keep the FA structure but replace the
`if (gl_SubgroupInvocationID == 0) tmpsh[gl_SubgroupID] = ...; barrier();
for (s ...) max_mask = max(max_mask, tmpsh[s])` pattern with
`max_mask = subgroupBroadcastFirst(all_less ? NEG_FLT_MAX_OVER_2 :
0.0f)` when `gl_NumSubgroups == 1`, falling back to shmem only when
multi-subgroup.

### R6 — ADOPT `MAT_VEC_FUSION_FLAGS_*` bitfield
**Priority:** Medium
**Difficulty:** S
**Dependencies:** R1
GwenLand's `GATE` should define an equivalent bitfield for matmul
fusion: `BIAS0`, `BIAS1`, `SCALE0`, `SCALE1`. Pass via push constant.
Avoids per-fusion shader variants.

### R7 — ADAPT four-flavor matmul taxonomy
**Priority:** High
**Difficulty:** L
**Dependencies:** R1, R3
GwenLand's `glvulkan` should provide: scalar (fallback), coopmat1
(KHR_cooperative_matrix), coopmat2 (NV_cooperative_matrix2), MMQ
(GL_EXT_integer_dot_product). Same per-flavor structure as llama.cpp.

### R8 — ADAPT `dequant_funcs_cm2.glsl` callback pattern
**Priority:** Medium
**Difficulty:** M
**Dependencies:** R7
GwenLand's coopmat2 path should use `coopMatLoadTensorNV` with a
per-format decode callback (`dequantFuncA` / `dequantFuncA_v`). Same
pattern as llama.cpp — let the hardware do the matrix load and call
back into GLSL for per-element decode.

### R9 — DEFER `occupancy_limiter` shmem tuning
**Priority:** Low
**Difficulty:** S
**Dependencies:** none
Only relevant if GwenLand's FA shader has the same occupancy-vs-shmem
tradeoff. Re-evaluate when FA shader is implemented.

### R10 — MONITOR `OLD_AMD_WINDOWS` workaround
**Priority:** Low
**Difficulty:** S
**Dependencies:** none
Watch for AMD driver fixes for the `f16vec4 subgroupShuffleXor` bug
(llama.cpp issue #19881). Remove the workaround when fixed.

---

## 17. Findings

### Finding ARTX19-F01

```
Finding ID:           ARTX19-F01
Category:             QUANTIZATION
Engine:               Vulkan
Component:            Dequantize function library
Source File:          ggml/src/ggml-vulkan/vulkan-shaders/dequant_funcs.glsl
Function:             dequantize / dequantize4 (per-format overloads)
Lines:                1-727
Summary:              Per-quant-format dequantize functions are macro-dispatched
                      via #if defined(DATA_A_*), producing one SPIR-V per format
                      with zero runtime dispatch overhead.
Observation:          The file is structured as ~25 #if blocks, one per quant
                      format (F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q1_0,
                      Q2_0, IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S,
                      IQ4_XS, IQ4_NL, MXFP4, NVFP4, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K).
                      Each block defines a vec2 dequantize(...) returning 2 elements
                      and a vec4 dequantize4(...) returning 4 elements. The vec4
                      overload uses the wider A_TYPE_PACKED16 view for 2x load
                      throughput. F32/F16/BF16 also provide dequantize1 (single)
                      and dequantize4_2aligned (4-wide aligned) overloads. A
                      separate get_dm(ib, a_offset) function returns (d, m) for
                      symmetric/asymmetric scale application.
Evidence:             dequant_funcs.glsl:64-127 (Q4_0..Q8_0), :157-232 (IQ1_*),
                      :234-489 (IQ2_*, IQ3_*, IQ4_*), :491-544 (MXFP4, NVFP4),
                      :546-591 (get_dm), :593-727 (K-quants).
Architectural Impact: Adding a quant format = adding one #if block. Clean ABI,
                      zero runtime cost. The single best design decision in the
                      shader library.
Correctness Impact:   None. Macro dispatch is deterministic at compile time.
Optimization Type:    Vectorization (vec2/vec4 overloads), macro dispatch.
GwenLand Target:      glvulkan
Recommendation:       ADOPT. Equivalent macro-dispatched library in glvulkan.
Priority:             Critical
Difficulty:           M
Dependencies:         none
Confidence:           High
```

### Finding ARTX19-F02

```
Finding ID:           ARTX19-F02
Category:             LAYOUT_SUBOPTIMAL
Engine:               Vulkan
Component:            Storage buffer aliasing
Source File:          ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_iface.glsl
Function:             (top-level buffer declarations)
Lines:                8-25
Summary:              Up to four SSBO aliases (data_a, data_a_v4, data_a_packed16,
                      data_a_packed32) are declared for the same binding, letting
                      the shader pick the widest aligned load per access pattern.
Observation:          The interface declares A_TYPE data_a[], A_TYPEV4 data_a_v4[]
                      (when A_TYPEV4 is defined), A_TYPE_PACKED16 data_a_packed16[],
                      A_TYPE_PACKED32 data_a_packed32[]. All share binding=0. The
                      host binds the same VkBuffer to all; the layout qualifier
                      only affects the shader's view. dequantize4 for Q4_0 reads
                      data_a_packed16 (2-byte loads), while dequantize4_2aligned
                      for F16 reads data_a_packed32 (4-byte loads). This pattern
                      repeats in mul_mm.comp:49-62, flash_attn_dequant.glsl:17-37.
Evidence:             mul_mat_vec_iface.glsl:8-25; mul_mm.comp:49-62;
                      flash_attn_dequant.glsl:17-37.
Architectural Impact: One VkBuffer serves all access patterns. No copying, no
                      per-pattern buffer. The shader picks the optimal load width
                      without runtime checks.
Correctness Impact:   None. The views are type-aliases of the same memory.
Optimization Type:    Vectorization (variable-width loads).
GwenLand Target:      glvulkan
Recommendation:       ADOPT. Same aliasing pattern in glvulkan.
Priority:             High
Difficulty:           S
Dependencies:         ARTX19-F01
Confidence:           High
```

### Finding ARTX19-F03

```
Finding ID:           ARTX19-F03
Category:             GPU_KERNEL
Engine:               Vulkan
Component:            GEMV main path
Source File:          ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec.comp
Function:             compute_outputs / iter
Lines:                141-246
Summary:              GEMV uses per-row workgroup assignment (NUM_ROWS per WG),
                      K_PER_ITER=4 or 8 inner loop, manual 4x/2x/1x unrolling,
                      and a 3-way reduce_result (subgroup-only / subgroup+shmem /
                      shmem-only).
Observation:          Each workgroup produces NUM_ROWS output rows. The K-axis is
                      split across BLOCK_SIZE threads, each processing K_PER_ITER
                      elements per iter() call. The compute_outputs function has
                      three while loops with unroll_count=4, then =2, then =1,
                      each with [[unroll]] for (k=0; k<unroll_count; ++k) inside.
                      A separate is_aligned_nonquant path uses iter_aligned_nonquant
                      which loads vec4 directly without dequantize. The final
                      reduce_result has three paths selected by USE_SUBGROUP_ADD
                      and USE_SUBGROUP_ADD_NO_SHMEM spec constants.
Evidence:             mul_mat_vec.comp:248-263 (main), :141-246 (compute_outputs),
                      :45-115 (iter), mul_mat_vec_base.glsl:93-228 (reduce_result).
Architectural Impact: NUM_ROWS per WG lets the shader amortize B loads across
                      multiple output rows. The 3-way reduce_result path lets the
                      host pick the best reduction per device.
Correctness Impact:   ULP-level non-determinism from subgroup reduction order
                      (vendor-specific). Deterministic for fixed vendor + subgroup
                      size.
Optimization Type:    Tiling (NUM_ROWS rows per WG), kernel fusion (MAT_VEC_FUSION_FLAGS),
                      vectorization (vec4 loads).
GwenLand Target:      glvulkan
Recommendation:       ADOPT. Same NUM_ROWS-per-WG + 3-way reduce pattern.
Priority:             High
Difficulty:           M
Dependencies:         ARTX19-F01
Confidence:           High
```

### Finding ARTX19-F04

```
Finding ID:           ARTX19-F04
Category:             SIMD_STRATEGY
Engine:               Vulkan
Component:            GEMM scalar + coopmat1 paths
Source File:          ggml/src/ggml-vulkan/vulkan-shaders/mul_mm.comp
Function:             main
Lines:                1-466
Summary:              Single mul_mm.comp shader contains both scalar (no COOPMAT)
                      and coopmat1 (COOPMAT defined) paths, selected at compile
                      time. Spec-constant tile sizes (BM=BN=64, BK=32/16, WM=WN=32,
                      TM=4, TN=2, WARP=32) allow per-architecture specialization.
Observation:          The shader has #ifdef COOPMAT branches at lines 121-134
                      (shmem types), 172-190 (warp/thread index math), 261-282
                      (accumulator declaration), 301-313 (coopMatMulAdd inner loop),
                      354-430 (output store). The scalar path uses
                      ACC_TYPEV2 sums[WMITER*TM*WNITER*TN/2] with manual
                      dot_product calls; the coopmat path uses
                      coopmat<ACC_TYPE, gl_ScopeSubgroup, TM, TN, ...> with
                      coopMatMulAdd. SHMEM_STRIDE = BK/2+1 (scalar) or BK/2+4
                      (coopmat) to avoid bank conflicts. Shared buf_a[BM*SHMEM_STRIDE]
                      and buf_b[BN*SHMEM_STRIDE].
Evidence:             mul_mm.comp:102-134 (tile sizes + shmem), :261-313 (coopmat
                      inner loop), :314-348 (scalar inner loop), :354-430 (coopmat
                      store), :431-465 (scalar store).
Architectural Impact: One source file serves two flavors. Spec-constant tile sizes
                      let the same SPIR-V specialize per architecture (BM=32 on
                      small GPUs, BM=64 on big). Shared `dot_product_funcs.glsl`
                      inner primitive.
Correctness Impact:   None. Both paths produce identical results up to FMA
                      reassociation differences.
Optimization Type:    Tiling (BM×BN×BK), blocking (warp tile WM×WN), vectorization
                      (FLOAT_TYPEV4 / FLOAT_TYPEV8 loads), kernel fusion (MUL_MAT_ID).
GwenLand Target:      glvulkan
Recommendation:       ADOPT. Same scalar+coopmat1-in-one-shader pattern with spec
                      constants.
Priority:             High
Difficulty:           L
Dependencies:         ARTX19-F01
Confidence:           High
```

### Finding ARTX19-F05

```
Finding ID:           ARTX19-F05
Category:             SIMD_STRATEGY
Engine:               Vulkan
Component:            Quantized GEMM via integer SDP
Source File:          ggml/src/ggml-vulkan/vulkan-shaders/mul_mmq.comp, mul_mmq_funcs.glsl
Function:             main / mmq_dot_product (per-format)
Lines:                mul_mmq.comp:1-311; mul_mmq_funcs.glsl:1-489
Summary:              MMQ path uses GL_EXT_integer_dot_product (dotPacked4x8EXT) to
                      compute 4-way int8 SDP per instruction, with per-format
                      block_a_to_shmem and mmq_dot_product functions. BK_STEP=4
                      inner unroll amortizes barrier cost.
Observation:          B is always block_q8_1_x4_packed128 (4 Q8_1 blocks grouped
                      for 4-uint loads). A is repacked into per-format block_a_cache
                      shmem (e.g. int32_t qs[8] for Q4_0, qs[4]+qh+dm for Q5_0).
                      The mmq_dot_product body is per-format: zero q_sum, unroll
                      over iqs, dotPacked4x8EXT, final float scale
                      (dm.x*q_sum*ds.x - 8*ds.y for Q4_0, etc.). BK_STEP=4 processes
                      4 BK-tiles per K-loop iteration (mul_mmq.comp:213), each
                      loaded with a separate barrier-bound shmem fill.
Evidence:              mul_mmq.comp:7 (extension), :33 (B type), :82 (BK), :88-92
                      (BK_STEP), :213-275 (K-loop), mul_mmq_funcs.glsl:33-40
                      (Q2_0 mmq_dot_product), :71-90 (Q4_0/Q4_1), :383-397 (Q4_K/Q5_K).
Architectural Impact: 4-8x throughput vs scalar dequant-then-FMA. The
                      dotPacked4x8EXT instruction does 4 int8 multiplies + 3 adds
                      in one cycle on supporting hardware.
Correctness Impact:   Integer accumulation is exact. The float-scale conversion
                      at the end may lose 1 ULP relative to a pure-FMA path. Same
                      ULP behavior as CPU MMQ.
Optimization Type:    SIMD (integer SDP), tiling (BM×BN×BK with BK_STEP inner unroll).
GwenLand Target:      glvulkan
Recommendation:       ADOPT. Same MMQ path with GL_EXT_integer_dot_product.
Priority:             High
Difficulty:           L
Dependencies:         ARTX19-F01, ARTX19-F04
Confidence:           High
```

### Finding ARTX19-F06

```
Finding ID:           ARTX19-F06
Category:             SIMD_STRATEGY
Engine:               Vulkan
Component:            Cooperative matrix 2 GEMM
Source File:          ggml/src/ggml-vulkan/vulkan-shaders/mul_mm_cm2.comp
Function:             main
Lines:                1-658
Summary:              NV_cooperative_matrix2 path uses tensorLayoutNV + coopMatLoadTensorNV
                      with per-format decode callbacks (dequantFuncA / dequantFuncA_v).
                      Workgroup-scope coopmat (BM×BN) replaces per-subgroup coopmat.
Observation:          Requires GL_NV_cooperative_matrix2, GL_EXT_buffer_reference,
                      GL_KHR_shader_subgroup_ballot, GL_KHR_shader_subgroup_vote. The
                      coopmat scope is gl_ScopeWorkgroup (line 368: coopmat<ACC_TYPE,
                      gl_ScopeWorkgroup, BM, BN, ...>), so the matrix is distributed
                      across the entire workgroup, not just one subgroup. Tensor
                      layouts (tensorLayoutNV, setTensorLayoutDimensionNV,
                      setTensorLayoutStrideNV, sliceTensorLayoutNV) describe the
                      matrix layout; coopMatLoadTensorNV reads directly from the SSBO
                      via GL_EXT_buffer_reference, calling back into dequantFuncA for
                      per-element decode. The shader has three branches: BNover4,
                      BNover2, BN for tail-tile handling (enable_smaller_matrices
                      spec constant). The MUL_MAT_ID path adds row-id compaction via
                      subgroupBallot + subgroupBallotBitCount + subgroupBallotExclusiveBitCount.
Evidence:              mul_mm_cm2.comp:11-23 (extensions), :309-334 (tensor layouts),
                      :367-497 (fast path with 3 branches), :514-657 (clamped path),
                      :163-227 (load_row_ids ballot compaction).
Architectural Impact: Best matmul performance on NVIDIA Hopper/Blackwell. The
                      workgroup-scope coopmat handles tiling automatically; the
                      shader doesn't need manual warp/thread tile math. Per-element
                      decode callback lets the hardware do the heavy lifting.
Correctness Impact:   ACC_TYPE_MAX clamp (lines 402-404 etc.) prevents fp16 acc
                      overflow on large K. Otherwise none.
Optimization Type:    Hardware matrix unit (coopmat2), tiling (auto via tensor layout),
                      kernel fusion (MUL_MAT_ID row compaction).
GwenLand Target:      glvulkan
Recommendation:       ADOPT for NVIDIA targets. Same callback-based decode pattern.
Priority:             High
Difficulty:           XL
Dependencies:         ARTX19-F04, ARTX19-F08
Confidence:           High
```

### Finding ARTX19-F07

```
Finding ID:           ARTX19-F07
Category:             GPU_KERNEL
Engine:               Vulkan
Component:            Flash attention (scalar + MMQ)
Source File:          ggml/src/ggml-vulkan/vulkan-shaders/flash_attn.comp
Function:             main
Lines:                1-758
Summary:              Online-softmax flash attention with Br×Bc tiling, D_split
                      head-dim partitioning, optional MMQ path via dotPacked4x8EXT,
                      mask block-skip via 2-bit data_mask_opt, ALiBi, logit softcap,
                      sink attention, split-K reduction.
Observation:          Per workgroup: load Q tile into Qf shmem (quantize to Q8_0/Q4_*
                      on-the-fly for MMQ path via subgroupClusteredMax(8) +
                      subgroupClusteredAdd(8)). Initialize Of=0, Lf=0, Mf=-FLT_MAX/2.
                      K-loop: optional mask block-skip via 2-bit data_mask_opt
                      (MASK_OPT_ALL_NEG_INF, MASK_OPT_ALL_ZERO); load K block into
                      kvsh shmem or direct; compute Sf=dot(Q,K) with subgroupShuffleXor
                      cross-D_split reduction; update Mf=max, eMf=exp(Mold-Mnew),
                      rescale Of and Lf; load V block; Pf=exp(Sf-Mf), Lf+=Pf, Of+=Pf*V.
                      After K-loop: cross-D_split and cross-row_split reduction via
                      subgroupShuffleXor + shmem. If split-K (k_num>1): store O, L, M
                      to separate buffers for flash_attn_split_k_reduce.comp. Else:
                      divide Of by Lf, store. ALiBi slope via perElemOpComputeSlope,
                      logit softcap via tanh, sink attention via perElemOpGetSink.
                      OLD_AMD_WINDOWS flag works around f16vec4 subgroupShuffleXor bug.
Evidence:              flash_attn.comp:81-104 (init), :110-148 (Q load + on-the-fly
                      quant), :197-242 (mask block skip), :251-293 (K load),
                      :295-428 (Sf compute, MMQ branch), :430-437 (D_split reduce),
                      :457-535 (online softmax update + V accumulate), :546-651 (final
                      reduction), :657-700 (split-K store), :702-757 (final divide +
                      store).
Architectural Impact: One shader handles F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 K/V
                      types via FaTypeK/FaTypeV spec constants + uber-dequant macros.
                      Online softmax keeps shmem O(Br*HSV) instead of O(Br*Bc*HSV).
Correctness Impact:   FATTN_KQ_MAX_OFFSET bias (line 257) keeps exp(S-M) in fp16
                      range. Split-K is deterministic for fixed k_num.
Optimization Type:    Tiling (Br×Bc), online softmax (O(1) memory), kernel fusion
                      (RoPE-style on-the-fly Q quant, mask block skip), subgroup
                      shuffle reduction.
GwenLand Target:      glvulkan
Recommendation:       ADAPT. Keep the structure; replace tmpsh[gl_SubgroupID] mask
                      broadcast with subgroupBroadcastFirst when gl_NumSubgroups==1.
Priority:             High
Difficulty:           XL
Dependencies:         ARTX19-F01
Confidence:           High
```

### Finding ARTX19-F08

```
Finding ID:           ARTX19-F08
Category:             QUANTIZATION
Engine:               Vulkan
Component:            Coopmat2 dequantize callbacks
Source File:          ggml/src/ggml-vulkan/vulkan-shaders/dequant_funcs_cm2.glsl
Function:             dequantFunc* / dequantFunc*_v (per-format) / fetch_scalesQ4_K
Lines:                1-1425
Summary:              Per-format scalar (dequantFunc*) and vector (dequantFunc*_v)
                      callbacks for coopMatLoadTensorNV, using buffer_reference-typed
                      views. Macro-dispatched to dequantFuncA/dequantFuncA_v. Q4_K/Q5_K
                      have fetch_scales/store_scales for scale prefetching.
Observation:          Each format declares a layout(buffer_reference) buffer
                      decodeBufQ* { block_q* block; } and two functions: float16_t
                      dequantFuncQ*(decodeBufQ* bl, uint blockCoords[2], uint
                      coordInBlock[2]) returns 1 element; f16vec4 dequantFuncQ*_v(...)
                      returns 4 elements. The decode callback runs per-element (or
                      per-4) inside the cooperative matrix load — the hardware
                      iterates the matrix and calls back into GLSL. Lines 1350-1424
                      macro-dispatch dequantFuncA based on DATA_A_*. For Q4_K and
                      Q5_K, fetch_scalesQ4_K / store_scalesQ4_K precompute per-block
                      scales into shmem so the per-element callback only does a shmem
                      lookup. The _v variant is used when
                      GL_NV_cooperative_matrix_decode_vector is supported (the host
                      strips SPV_NV_cooperative_matrix_decode_vector ops if not — see
                      ARTX18).
Evidence:              dequant_funcs_cm2.glsl:12-14 (F32 layout), :24-47 (Q1_0),
                      :69-98 (Q4_0), :1350-1424 (macro dispatch), :1380-1386 (Q4_K
                      fetch_scales), mul_mm_cm2.comp:381 (coopMatLoadTensorNV call
                      with DECODEFUNCA callback).
Architectural Impact: Lets the hardware do the matrix load (with all its coalescing
                      and caching) while the GLSL callback handles per-element decode.
                      Scale prefetching for K-quants amortizes the scale fetch across
                      the whole BM×BK tile.
Correctness Impact:   None. The decode logic mirrors dequant_funcs.glsl's
                      dequantize4 but in a callback form.
Optimization Type:    Hardware matrix unit (coopmat2), kernel fusion (decode inside
                      matrix load), scale prefetching.
GwenLand Target:      glvulkan
Recommendation:       ADOPT. Same callback-based decode pattern for glvulkan's
                      coopmat2 path.
Priority:             High
Difficulty:           L
Dependencies:         ARTX19-F06
Confidence:           High
```

### Finding ARTX19-F09

```
Finding ID:           ARTX19-F09
Category:             GPU_KERNEL
Engine:               Vulkan
Component:            Q4_K-specialized GEMV
Source File:          ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_q4_k.comp
Function:             calc_superblock / compute_outputs
Lines:                1-134
Summary:              Q4_K GEMV has a dedicated shader that processes one 256-element
                      super-block with 16 threads, using unpack8/unpack16 for vectorized
                      bit extraction and FMA chains for accumulation.
Observation:          Each workgroup has it_size = WorkGroupSize/16 "iteration slots".
                      Threads within a slot are mapped (itid=0..15) to (il=0..3, ir=0..3)
                      covering the 4 sub-blocks × 4 lanes within a Q4_K block. The
                      calc_superblock function (line 11) reads 16 4-bit q4 values
                      (qs0_u32_lo4, qs0_u32_hi4, qs64_u32_lo4, qs64_u32_hi4 as vec4s),
                      8 scale values (sc0..sc7), 4 B vec4s (by10, by132, by20, by232),
                      and computes sx, sy, sz, sw via 4-deep FMA chains, then
                      smin via 16-deep FMA chain, then temp = fma(dm.x, sx*scales +
                      sy*scales + sz*scales + sw*scales, fma(-dm.y, smin, temp)).
                      This avoids the generic dequantize4 path entirely.
Evidence:              mul_mat_vec_q4_k.comp:11-85 (calc_superblock), :87-120
                      (compute_outputs), :93-108 (thread mapping).
Architectural Impact: ~3-4x faster than the generic mul_mat_vec.comp path for Q4_K
                      because the super-block layout is exploited directly. The
                      generic path would call dequantize4(ib, iqs, a_offset) per
                      4 elements, incurring per-call function-call overhead and
                      redundant scale unpacking.
Correctness Impact:   None. Same arithmetic as the generic path, just unrolled.
Optimization Type:    SIMD (FMA chains, unpack8/unpack16), kernel fusion (scale
                      unpacking baked into accumulator), tiling (16-thread super-block).
GwenLand Target:      glvulkan
Recommendation:       ADOPT for Q4_K (the most common format). Evaluate for Q6_K and
                      other K-quants if performance warrants.
Priority:             Medium
Difficulty:           M
Dependencies:         ARTX19-F03
Confidence:           High
```

### Finding ARTX19-F10

```
Finding ID:           ARTX19-F10
Category:             GPU_KERNEL
Engine:               Vulkan
Component:            RoPE kernels
Source File:          ggml/src/ggml-vulkan/vulkan-shaders/rope_norm.comp, rope_neox.comp, rope_funcs.glsl
Function:             main / rope_norm / rope_neox / rope_multi / rope_vision
Lines:                rope_norm.comp:1-17; rope_neox.comp:1-17; rope_funcs.glsl:1-210
Summary:              RoPE is a single-thread-per-element-pair kernel (local_size=(1,256,1))
                      with no workgroup cooperation. Four variants (norm, neox, multi,
                      vision) share rope_funcs.glsl. YaRN ramp, ext_factor, mscale,
                      RoPE+VIEW+SET_ROWS fusion, RoPE+RMS_NORM fusion all included.
Observation:          layout(local_size_x=1, local_size_y=256, local_size_z=1)
                      (rope_head.glsl:7). i0 = 2*gl_GlobalInvocationID.y (pair index).
                      row = gl_GlobalInvocationID.x + 32768 * gl_GlobalInvocationID.z
                      (row packed across Y+Z to avoid 2^32 dispatch limits). The
                      thread decodes (i1, i2, i3) from row via integer division, then
                      calls rope_norm/neox/multi/vision. theta_base = rope_data_pos[i2]
                      * pow(p.theta_scale, i0/2.0f). rope_yarn applies YaRN ramp +
                      mscale. rope_multi handles multi-section RoPE (sections[4]) for
                      Qwen3. rope_vision handles vision encoder RoPE (sections[2]).
                      RMS_NORM_ROPE_FUSION flag reads from shmem instead of global
                      (rope_a_coord, rope_funcs.glsl:8-14). set_rows_stride != 0
                      enables RoPE+VIEW+SET_ROWS fusion (rope_funcs.glsl:46-50) for
                      sparse MoE output indexing.
Evidence:              rope_norm.comp:6-17, rope_neox.comp:6-17, rope_funcs.glsl:7-15
                      (coord), :17-35 (rope_yarn), :37-72 (rope_norm), :74-109
                      (rope_neox), :112-175 (rope_multi), :177-209 (rope_vision),
                      :46-50 (set_rows fusion), :8-14 (RMS_NORM fusion).
Architectural Impact: One kernel handles all RoPE variants via shared infra. The
                      1-thread-per-pair design avoids any barrier overhead — useful
                      because RoPE is bandwidth-bound, not compute-bound. Row packing
                      across Y+Z axes supports up to 32768 * 2^32 rows.
Correctness Impact:   pow(p.theta_scale, i0/2.0f) re-computed per thread; GPU pow
                      is accurate to ~1 ULP. The (cos, sin) rotation is exact.
Optimization Type:    Kernel fusion (RoPE+VIEW+SET_ROWS, RoPE+RMS_NORM), bandwidth
                      optimization (1 thread per pair = max parallelism).
GwenLand Target:      glvulkan
Recommendation:       ADOPT. Same 1-thread-per-pair layout + shared funcs.
Priority:             Medium
Difficulty:           S
Dependencies:         none
Confidence:           High
```

### Finding ARTX19-F11

```
Finding ID:           ARTX19-F11
Category:             GPU_KERNEL
Engine:               Vulkan
Component:            Softmax kernel
Source File:          ggml/src/ggml-vulkan/vulkan-shaders/soft_max.comp
Function:             soft_max / main
Lines:                1-195
Summary:              Three-pass softmax (max, sum, normalize) using pure shared-
                      memory tree reduction. Template-specialized via num_blocks (1,
                      2, 3, 4, 8, 16, 32, or >32) to enable unrolling. 16-element
                      per-thread data cache avoids re-reading in pass 2/3.
Observation:          BLOCK_SIZE=32 (spec). Each WG processes one row. Pass 1 (max,
                      lines 78-100): each thread reads N columns, computes v=a*scale+
                      slope*b, caches in data_cache[16], reduces max_val across WG
                      via shmem tree (lines 103-110). Pass 2 (sum, lines 118-140):
                      re-reads or uses cached v, computes exp(v-max_val), accumulates
                      sum, reduces via shmem tree (lines 143-150). Pass 3 (normalize,
                      lines 159-171): divides cached or re-read exp(v-max) by sum.
                      ALiBi slope via head index. Sink support via data_c[i02].
                      Optional mask via data_b (KY>0). main() (lines 174-194) selects
                      num_blocks from 1,2,3,4,8,16,32,>32 and calls soft_max(N) to
                      enable loop unrolling for small KX.
Evidence:              soft_max.comp:28-36 (shmem + spec), :41-100 (pass 1 max),
                      :103-113 (max reduce), :115-140 (pass 2 sum), :143-155 (sum
                      reduce + sink), :157-171 (pass 3 normalize), :174-194 (main
                      template dispatch).
Architectural Impact: Functional but slow on subgroup-capable hardware — 5 barrier
                      rounds per pass where 1 subgroup instruction would suffice.
                      Compare with quantize_q8_1.comp:77 which uses
                      subgroupClusteredMax(8) for the same kind of reduction.
Correctness Impact:   None. Deterministic for fixed BLOCK_SIZE.
Optimization Type:    Tiling (BLOCK_SIZE per row), data_cache[16] cache, template
                      specialization by num_blocks.
GwenLand Target:      glvulkan
Recommendation:       REJECT the pure-shmem reduction. Use subgroupMax/subgroupAdd
                      (with shmem fallback for non-subgroup hardware). Keep the
                      data_cache[16] and template specialization.
Priority:             Medium
Difficulty:           S
Dependencies:         ARTX20 subgroup strategy
Confidence:           High
```

### Finding ARTX19-F12

```
Finding ID:           ARTX19-F12
Category:             GPU_KERNEL
Engine:               Vulkan
Component:            Q8_1 quantizer
Source File:          ggml/src/ggml-vulkan/vulkan-shaders/quantize_q8_1.comp
Function:             quantize / main
Lines:                1-127
Summary:              F32→Q8_1 quantizer with persistent threads, vec4 input loads,
                      dual-path per-block reduction (subgroupClustered(8) when
                      USE_SUBGROUPS, shmem tree otherwise), and optional x4 block
                      grouping (QBLOCK_X4) for the MMQ B-side layout.
Observation:          GROUP_SIZE=32 spec (line 23). Each thread handles one vec4
                      (4 of 32 elements in a Q8_1 block); 8 threads per block. The
                      persistent-threads loop (main, lines 121-127): while (wgid <
                      p.num_blocks) { quantize(wgid); wgid += gl_NumWorkGroups.x; }.
                      quantize(wgid): vec4 vals = data_a[a_idx]; thread_max = max-of-
                      abs; block_max via subgroupClusteredMax(8) (USE_SUBGROUPS path,
                      line 77) or shmem tree (lines 66-75); d = amax/127, d_inv = 1/d;
                      vals = round(vals * d_inv); data_b[ib].qs[iqs] = pack32(i8vec4(
                      round(vals))); thread_sum = vals.x+vals.y+vals.z+vals.w; block
                      sum via subgroupClusteredAdd(8) (line 106) or shmem tree (lines
                      97-104); lane 0 writes data_b[ib].ds = f16vec2(d, sum*d).
                      QBLOCK_X4 path (lines 48-56, 84-88, 113-117) groups 4 Q8_1
                      blocks for the MMQ B-side layout.
Evidence:              quantize_q8_1.comp:6-13 (extension + INVOCATION_ID macro),
                      :23 (GROUP_SIZE spec), :37-119 (quantize function), :121-127
                      (persistent main).
Architectural Impact: Persistent threads amortize launch cost. Dual-path reduction
                      handles both subgroup-capable and legacy hardware. QBLOCK_X4
                      output layout matches mul_mmq.comp's B-side input, avoiding a
                      separate repacking pass.
Correctness Impact:   None. round() is deterministic. The cluster size (8) matches
                      the Q8_1 block layout (32 elements / 4 per thread = 8 threads).
Optimization Type:    Persistent threads, SIMD (subgroupClustered), kernel fusion
                      (QBLOCK_X4 layout), vectorization (vec4 input).
GwenLand Target:      glvulkan
Recommendation:       ADOPT. Same persistent-threads + dual-path reduction pattern.
Priority:             High
Difficulty:           S
Dependencies:         ARTX20 subgroup strategy
Confidence:           High
```

---

## 18. Unknowns

* **U1**. Whether the 4x/2x/1x manual unrolling in `mul_mat_vec.comp:164-243`
  actually outperforms a single `[[unroll]]`-hinted loop on current
  drivers. The comment says `[[unroll]]` alone was insufficient, but
  no driver versions are cited. Requires per-driver benchmarking.

* **U2**. Whether the `occupancy_limiter` shmem in `flash_attn.comp:75-104`
  is ever non-zero in shipped configurations. The flag is
  `LIMIT_OCCUPANCY_SHMEM` spec constant; the host (ARTX18) picks the
  value. Static analysis can't determine which devices set it.

* **U3**. Whether `mul_mat_vec_q4_k.comp`'s dedicated super-block path
  outperforms the generic `mul_mat_vec.comp` path on every GPU, or
  only on GPUs where the generic path's `dequantize4` function call
  isn't inlined. Requires comparative benchmarking.

* **U4**. Whether the `OLD_AMD_WINDOWS` workaround in
  `flash_attn.comp:618-624` is still needed on current AMD Windows
  drivers. The cited issue (#19881) suggests it was a driver bug;
  status of the fix is not visible from static analysis.

* **U5**. Whether the `BK_STEP` tuning (default 4 for non-MUL_MAT_ID,
  1 for MUL_MAT_ID, see `mul_mmq.comp:88-92`) is optimal per
  architecture. The comment at `mul_mmq_shmem_types.glsl:44` notes
  "AMD likes 4, Intel likes 1, Nvidia likes 2" for Q8_0 — but the
  default is 4. Per-architecture tuning would require runtime
  selection.

* **U6**. Whether the FA `data_cache[16]` in `soft_max.comp:75` is
  the right size. The constant `DATA_CACHE_SIZE = 16` is hardcoded.
  For BLOCK_SIZE=32 and num_blocks=32, 16 elements per thread covers
  half the row — the other half is re-read in pass 2. A larger cache
  (e.g., 32) would trade register pressure for memory bandwidth.

* **U7**. Whether `dequant_funcs_cm2.glsl`'s scale prefetching
  (fetch_scalesQ4_K / store_scalesQ4_K) actually pipelines
  effectively on NVIDIA hardware. The pipelining relies on the
  cooperative matrix load not blocking on shmem writes — requires
  profiling on Hopper/Blackwell.

* **U8**. Whether the RoPE row packing (32768 rows per Y axis,
  `rope_norm.comp:8`) limits dispatch flexibility. For very large
  batches (>32768 rows), the Z axis takes over, but the integer
  division to decode (i1, i2, i3) may be slow. No alternative path
  is visible.

---

## 19. References

| Reference | File                                                | Function / Symbol                              | Lines         |
| --------- | --------------------------------------------------- | ---------------------------------------------- | ------------- |
| R01       | `vulkan-shaders/types.glsl`                         | `block_q4_0`, `block_q4_k`, etc. structs       | 56-1914       |
| R02       | `vulkan-shaders/dequant_funcs.glsl`                 | `dequantize` / `dequantize4` / `get_dm`        | 1-727         |
| R03       | `vulkan-shaders/dequant_funcs_cm2.glsl`             | `dequantFunc*` / `dequantFunc*_v` / `fetch_scales` | 1-1425      |
| R04       | `vulkan-shaders/dequant_head.glsl`                  | push constant + types include                  | 1-13          |
| R05       | `vulkan-shaders/generic_head.glsl`                  | push constant + types include                  | 1-11          |
| R06       | `vulkan-shaders/utils.glsl`                         | `fastmod`, `fastdiv`, `get_indices`            | 1-25          |
| R07       | `vulkan-shaders/mul_mat_vec.comp`                   | `main`, `compute_outputs`, `iter`              | 1-264         |
| R08       | `vulkan-shaders/mul_mat_vec_base.glsl`              | `reduce_result`, `get_offsets`, push constant  | 1-230         |
| R09       | `vulkan-shaders/mul_mat_vec_iface.glsl`             | SSBO bindings + fusion flag macros             | 1-35          |
| R10       | `vulkan-shaders/mul_mat_vec_q4_k.comp`              | `calc_superblock`, `compute_outputs`           | 1-134         |
| R11       | `vulkan-shaders/mul_mm.comp`                        | `main` (scalar + coopmat1)                     | 1-466         |
| R12       | `vulkan-shaders/mul_mm_funcs.glsl`                  | `load_a_to_shmem`, `load_b_to_shmem`           | 1-644         |
| R13       | `vulkan-shaders/mul_mmq.comp`                       | `main` (integer SDP MMQ)                       | 1-311         |
| R14       | `vulkan-shaders/mul_mmq_funcs.glsl`                 | `block_a_to_shmem`, `mmq_dot_product`          | 1-489         |
| R15       | `vulkan-shaders/mul_mmq_shmem_types.glsl`           | `block_a_cache`, `block_b_cache`               | 1-100         |
| R16       | `vulkan-shaders/mul_mm_cm2.comp`                    | `main` (coopmat2)                              | 1-658         |
| R17       | `vulkan-shaders/mul_mm_id_funcs.glsl`               | `load_row_ids` (subgroupBallot compaction)     | 1-74          |
| R18       | `vulkan-shaders/dot_product_funcs.glsl`             | `dot_product` (VALVE mixed-float + F32)        | 1-27          |
| R19       | `vulkan-shaders/flash_attn.comp`                    | `main` (FA scalar + MMQ)                       | 1-758         |
| R20       | `vulkan-shaders/flash_attn_base.glsl`               | push constant, spec constants, `init_indices`  | 1-265         |
| R21       | `vulkan-shaders/flash_attn_dequant.glsl`            | aliased SSBO views + per-quant decode macros   | 1-132         |
| R22       | `vulkan-shaders/flash_attn_cm1.comp`                | `main` (FA coopmat1)                           | 1-646         |
| R23       | `vulkan-shaders/flash_attn_cm2.comp`                | `main` (FA coopmat2)                           | 1-481         |
| R24       | `vulkan-shaders/flash_attn_mmq_funcs.glsl`          | `k_block_to_shmem`, `get_k_qs`, `k_dot_correction` | 1-…        |
| R25       | `vulkan-shaders/rope_head.glsl`                     | push constant + bindings                       | 1-19          |
| R26       | `vulkan-shaders/rope_funcs.glsl`                    | `rope_norm`, `rope_neox`, `rope_multi`, `rope_vision`, `rope_yarn` | 1-210 |
| R27       | `vulkan-shaders/rope_norm.comp` / `rope_neox.comp`  | `main`                                         | 1-17 / 1-17   |
| R28       | `vulkan-shaders/rope_params.glsl`                   | `struct rope_params`                           | 1-34          |
| R29       | `vulkan-shaders/soft_max.comp`                      | `soft_max`, `main`                             | 1-195         |
| R30       | `vulkan-shaders/soft_max_large_common.glsl`         | shared infra for multi-pass softmax            | 1-53          |
| R31       | `vulkan-shaders/quantize_q8_1.comp`                 | `quantize`, `main`                             | 1-127         |
| R32       | `vulkan-shaders/argmax.comp`                        | `main`                                         | 1-60          |
| R33       | `vulkan-shaders/sum_rows.comp` / `sum_rows.glsl`    | `main`, `fastdiv`                              | 1-47 / 1-25   |
| R34       | `vulkan-shaders/mul_mat_split_k_reduce.comp`        | `main` (split-K reduce)                        | 1-48          |
| R35       | `vulkan-shaders/flash_attn_split_k_reduce.comp`     | `main` (FA split-K reduce)                     | 1-…           |
| R36       | `vulkan-shaders/cumsum.comp`                        | `main` (subgroupExclusiveAdd scan)             | 1-83          |
| R37       | `vulkan-shaders/topk_nary_search.comp`              | `main` (subgroupBallot top-K)                  | 1-247         |
| R38       | `vulkan-shaders/fwht.comp`                          | `main` (subgroupShuffleXor butterfly)          | 1-115         |
| R39       | `vulkan-shaders/gated_delta_net.comp`               | `main` (subgroupClusteredAdd)                  | 1-190         |
| R40       | `vulkan-shaders/conv2d_mm.comp` / `conv3d_mm.comp`  | `main` (subgroupShuffle index broadcast)       | 1-481 / 1-…   |
