# ARTX20 — Vulkan Subgroup Operations Usage

**Audited commit:** `555881ebc8b0fc0402b30e09258a32a7bfd13c52`
**Last updated:** 2026-07-25
**Auditor:** Percival-aux
**Target GwenLand module:** `glvulkan` (subgroup reduction strategy,
per-architecture subgroup-size selection), `GATE` (per-op kernel
selection based on subgroup capability)

---

## 1. Executive Summary

The Vulkan shader library uses subgroup operations in **24 of 163
shaders** (15%). The pattern is consistent and disciplined: every
subgroup-using shader enables one or more `GL_KHR_shader_subgroup_*`
extensions, gates the subgroup path behind a spec constant, and
provides a shared-memory tree-reduction fallback. The host (ARTX18)
queries `VkPhysicalDeviceVulkan11Properties.subgroupSize` and
`subgroupSupportedOperations`, requests a specific subgroup size at
pipeline-create time via `VK_EXT_subgroup_size_control`, and
dispatches to the subgroup or non-subgroup SPIR-V variant.

Subgroup usage is concentrated in seven patterns:

1. **Whole-subgroup reductions** (`subgroupAdd`, `subgroupMax`,
   `subgroupMin`) for cross-lane sums/maxes inside a single subgroup.
   Used by `add.comp`, `multi_add.comp`, `rms_norm_partials.comp`,
   `ssm_scan.comp`, `cumsum_multipass2.comp`, `flash_attn_cm1.comp`,
   `topk_moe.comp`.
2. **Clustered reductions** (`subgroupClusteredAdd`, `subgroupClusteredMax`)
   with compile-time-constant cluster size — used to reduce across
   sub-clusters of a subgroup. Used by `quantize_q8_1.comp` (cluster=8),
   `flash_attn.comp` (cluster=8), `gated_delta_net.comp` (cluster=2..64).
3. **Inclusive/exclusive scans** (`subgroupInclusiveAdd`,
   `subgroupExclusiveAdd`) for prefix sums. Used by `cumsum.comp`,
   `cumsum_multipass1.comp`, `topk_nary_search.comp`.
4. **Shuffles** (`subgroupShuffleXor`, `subgroupShuffle`) for cross-
   lane value exchange without shared memory. Used by `flash_attn.comp`
   (D_split reduction), `fwht.comp` (butterfly), `topk_moe.comp`
   (argmax reduction), `conv2d_mm.comp` (index broadcast).
5. **Ballots** (`subgroupBallot`, `subgroupBallotBitCount`,
   `subgroupBallotExclusiveBitCount`, `subgroupBallotFindLSB`) for
   compaction of predicate-true lanes into a dense output. Used by
   `mul_mm_id_funcs.glsl` / `mul_mm_cm2.comp` (MoE row-id compaction),
   `topk_nary_search.comp` (top-K compaction).
6. **Votes** (`subgroupAll`) for "does any/all lane satisfy predicate"
   broadcast. Used by `flash_attn.comp` / `flash_attn_cm1.comp` for
   mask-skip.
7. **Built-in indexing** (`gl_SubgroupInvocationID`, `gl_SubgroupID`,
   `gl_NumSubgroups`, `gl_SubgroupSize`) for cross-subgroup
   coordination via shared memory. Used by virtually every subgroup
   shader that needs to reduce across multiple subgroups.

For GwenLand, the architectural decisions worth **ADOPT**ing are:
the spec-constant-gated subgroup/shmem dual path (from
`quantize_q8_1.comp`), the `subgroupBallot`-based MoE row-id
compaction (from `mul_mm_id_funcs.glsl`), the
`subgroupShuffleXor`-based D_split reduction (from `flash_attn.comp`),
and the clustered reduction pattern (from `quantize_q8_1.comp` /
`flash_attn.comp`). The decisions worth **REJECT**ing are the pure
shmem tree reductions in `soft_max.comp` / `argmax.comp` /
`sum_rows.comp` (which leave subgroup performance on the table) and
the `subgroup_size = 32` spec constant in `mul_mm_cm2.comp` (which
is sized for a `ballots_sh[]` array but assumes a fixed subgroup
size).

---

## 2. Purpose

Document the shader-side use of Vulkan subgroup operations so a
GwenLand engineer can:

* Reproduce the subgroup reduction patterns in `glvulkan`'s shaders.
* Know which subgroup extensions each kernel family requires.
* Understand the cooperative-matrix-vs-subgroup distinction (cooperative
  matrix is a separate extension family that *uses* subgroups
  internally but exposes a different API).
* Pick the right subgroup-size policy per architecture.

ARTX18 §9 covered the host side: how the backend queries subgroup
properties, requests specific subgroup sizes via
`VK_EXT_subgroup_size_control`, and uses
`vk::PipelineShaderStageRequiredSubgroupSizeCreateInfoEXT`. This
document covers the shader side: which extensions each shader
enables, which built-ins and operations it uses, and how it falls
back when subgroups aren't available.

---

## 3. Source Files

| File                                              | Lines  | Subgroup role                                                                 |
| ------------------------------------------------- | ------ | ----------------------------------------------------------------------------- |
| `vulkan-shaders/mul_mat_vec_base.glsl`            | 230    | 3-way `reduce_result`: subgroup-only / subgroup+shmem / shmem-only            |
| `vulkan-shaders/mul_mat_vec_p021.comp`            | 157    | `subgroupAdd` for GEMV reduce (gated on `USE_SUBGROUP_ADD`)                   |
| `vulkan-shaders/mul_mm.comp`                      | 466    | `gl_SubgroupID` / `gl_SubgroupInvocationID` for coopmat indexing              |
| `vulkan-shaders/mul_mmq.comp`                     | 311    | `GL_KHR_shader_subgroup_basic` + `ballot` (for MUL_MAT_ID_USE_SUBGROUPS)      |
| `vulkan-shaders/mul_mm_cm2.comp`                  | 658    | `subgroupBallot` MoE row-id compaction; `subgroup_size = 32` spec constant    |
| `vulkan-shaders/mul_mm_id_funcs.glsl`             | 74     | `subgroupBallot` + `BitCount` + `ExclusiveBitCount` MoE row-id compaction     |
| `vulkan-shaders/flash_attn.comp`                  | 758    | `subgroupClusteredMax/Add(8)`, `subgroupShuffleXor`, `subgroupAll`, built-ins |
| `vulkan-shaders/flash_attn_cm1.comp`              | 646    | `subgroupMax`, `subgroupAdd`, `subgroupAll`, built-ins                        |
| `vulkan-shaders/flash_attn_cm2.comp`              | 481    | `GL_KHR_shader_subgroup_ballot` + `vote` (enabled, lightly used)              |
| `vulkan-shaders/flash_attn_mask_opt.comp`         | 163    | `subgroupMin` / `subgroupMax` for block mask summary                          |
| `vulkan-shaders/quantize_q8_1.comp`               | 127    | `subgroupClusteredMax(8)` + `subgroupClusteredAdd(8)` (gated on `USE_SUBGROUPS`) |
| `vulkan-shaders/rms_norm_partials.comp`           | 66     | `subgroupAdd` for partial-sum reduction; uses `gl_SubgroupSize` as stride     |
| `vulkan-shaders/add.comp`                          | ~75    | `subgroupAdd` for sum-of-squares reduce (gated)                               |
| `vulkan-shaders/multi_add.comp`                    | ~200   | Same pattern as `add.comp`                                                    |
| `vulkan-shaders/ssm_scan.comp`                     | ~150   | `subgroupAdd` for state reduction; uses `gl_SubgroupID`/`InvocationID`        |
| `vulkan-shaders/cumsum.comp`                       | 83     | `subgroupExclusiveAdd` for per-subgroup scan + cross-subgroup shmem           |
| `vulkan-shaders/cumsum_multipass1.comp`            | ~60    | `subgroupInclusiveAdd` for per-subgroup inclusive scan                        |
| `vulkan-shaders/cumsum_multipass2.comp`            | ~60    | `subgroupAdd` for cross-subgroup combine                                      |
| `vulkan-shaders/topk_nary_search.comp`             | 247    | `subgroupBallot` + `BitCount` + `ExclusiveBitCount` + `InclusiveAdd` + `FindLSB` |
| `vulkan-shaders/topk_moe.comp`                     | ~215   | `subgroupShuffleXor` for argmax reduce, `subgroupAdd`/`subgroupMax`           |
| `vulkan-shaders/fwht.comp`                         | 115    | `subgroupShuffleXor` for in-block butterfly (vs `FWHT_SHMEM` path)            |
| `vulkan-shaders/gated_delta_net.comp`              | 190    | `subgroupClusteredAdd(2/4/8/16/32/64)` switch on `LANES_PER_COLUMN` spec const |
| `vulkan-shaders/conv2d_mm.comp`                    | 481    | `subgroupShuffle` for index broadcast (`USE_COLLECTIVES` path)                |
| `vulkan-shaders/conv3d_mm.comp`                    | ~350   | `gl_SubgroupID` for warp indexing                                             |

> **Note**: `soft_max.comp`, `argmax.comp`, `sum_rows.comp`,
> `rope_*.comp`, and most elementwise shaders do **not** use subgroup
> operations — they use pure shared-memory tree reductions or no
> cooperation at all. This is a deliberate gap (see Finding ARTX20-F09).

---

## 4. Architecture Overview

```
            ┌────────────────────────────────────────────────────────┐
            │  Host (ARTX18 §9):                                      │
            │  ├─ query subgroup_size, subgroupSupportedOperations    │
            │  ├─ request subgroup size via VK_EXT_subgroup_size_     │
            │  │   control + PipelineShaderStageRequiredSubgroup      │
            │  │   SizeCreateInfoEXT                                  │
            │  └─ pick USE_SUBGROUPS spec constant per pipeline       │
            └────────────────────────────────────────────────────────┘
                                    │
                                    ▼
            ┌────────────────────────────────────────────────────────┐
            │  Shader extensions (per shader, conditional):           │
            │  ├─ GL_KHR_shader_subgroup_basic    (built-ins)         │
            │  ├─ GL_KHR_shader_subgroup_arithmetic (Add/Max/Min)     │
            │  ├─ GL_KHR_shader_subgroup_clustered (ClusteredAdd/Max) │
            │  ├─ GL_KHR_shader_subgroup_shuffle  (Shuffle/ShuffleXor)│
            │  ├─ GL_KHR_shader_subgroup_ballot    (Ballot/BitCount)  │
            │  └─ GL_KHR_shader_subgroup_vote      (All/Any/AllEqual) │
            └────────────────────────────────────────────────────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              ▼                     ▼                     ▼
   ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐
   │ Reductions        │  │ Ballots + Shuffles│  │ Cooperative matrix│
   │ (subgroupAdd/Max/ │  │ (MoE row-id       │  │ (separate ext):   │
   │  Min, Clustered,  │  │  compaction, top-K│  │ GL_KHR_cooperative│
   │  Inclusive/Exclus)│  │  compaction, FA   │  │ _matrix (coopmat1)│
   │                   │  │  D_split, FWHT    │  │ GL_NV_cooperative │
   │ add.comp,         │  │                   │  │ _matrix2 (coopmat2│
   │ rms_norm_partials,│  │ mul_mm_id_funcs,  │  │ + tensor layout)  │
   │ quantize_q8_1,    │  │  mul_mm_cm2,      │  │                   │
   │ flash_attn,       │  │  topk_nary_search,│  │ mul_mm.comp       │
   │ gated_delta_net,  │  │  flash_attn,      │  │  (COOPMAT),       │
   │ cumsum, topk_moe  │  │  fwht, topk_moe,  │  │ mul_mm_cm2.comp,  │
   │                   │  │  conv2d_mm        │  │ flash_attn_cm1/2  │
   └───────────────────┘  └───────────────────┘  └───────────────────┘
```

Key design points:

* **Subgroup operations are core Vulkan 1.1**, not the
  `VK_KHR_shader_subgroup` extension. ARTX18 §9.1 confirms: the
  backend never references `VK_KHR_shader_subgroup` by name; it
  reads `VkPhysicalDeviceVulkan11Properties.subgroup*` directly.
  The shaders use `GL_KHR_shader_subgroup_*` extension names because
  that's the GLSL-side naming convention; the underlying SPIR-V
  capabilities (`SubgroupBallotKHR`, `GroupNonUniformArithmetic`,
  etc.) are core in Vulkan 1.1.
* **Cooperative matrix is a separate extension family**, not a
  subgroup operation. `GL_KHR_cooperative_matrix` (coopmat1) and
  `GL_NV_cooperative_matrix2` (coopmat2) expose `coopmat<>` types and
  `coopMatMulAdd`/`coopMatLoadTensorNV` operations. They use subgroups
  internally (coopmat1 scope is `gl_ScopeSubgroup`) but the shader
  code doesn't call subgroup operations directly — it calls coopmat
  operations.
* **Every subgroup-using shader has a fallback path**. The fallback
  is gated by a spec constant (`USE_SUBGROUPS`, `USE_SUBGROUP_ADD`,
  `USE_COLLECTIVES`, `FWHT_SHMEM`, etc.) and uses shared-memory tree
  reductions. This is the cleanest pattern in the codebase for
  handling heterogeneous subgroup support.

---

## 5. Execution Flow

### 5.1 Host-side subgroup setup (recap from ARTX18)

ARTX18 §9 documents:

1. `VkPhysicalDeviceVulkan11Properties` is queried at
   `ggml-vulkan.cpp:6135` to read `subgroupSize`,
   `subgroupSupportedStages`, `subgroupSupportedOperations`.
2. `VK_EXT_subgroup_size_control` is queried at lines 5973–5974 and
   6264–6273 to read `minSubgroupSize`, `maxSubgroupSize`,
   `requiredSubgroupSizeStages`, `fullSubgroupsSupportStages`.
3. Per-pipeline, the backend sets
   `vk::PipelineShaderStageRequiredSubgroupSizeCreateInfoEXT` with
   `requiredSubgroupSize = subgroup_min_size` (preferred) for matmul
   pipelines, and a per-pipeline-tuning-params size for FA pipelines.
4. `subgroup_require_full_support` enables
   `vk::PipelineShaderStageCreateFlagBits::eRequireFullSubgroupsEXT`
   for pipelines that need all subgroups to be full (no partial
   subgroups at the workgroup tail).

The shader-side effect: `gl_SubgroupSize` is a known constant at
compile time (per pipeline). Shaders can rely on this to size shmem
arrays and unroll loops.

### 5.2 Shader-side subgroup extension selection

Each subgroup-using shader enables the specific extensions it needs:

```glsl
// mul_mat_vec_base.glsl:5-8
#if USE_SUBGROUP_ADD || USE_SUBGROUP_ADD_NO_SHMEM
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#endif
```

```glsl
// flash_attn.comp:15, 20-21
#extension GL_KHR_shader_subgroup_clustered : require
#extension GL_KHR_shader_subgroup_shuffle : enable
#extension GL_KHR_shader_subgroup_vote : enable
```

```glsl
// topk_nary_search.comp:4-7
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_ballot : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable
#extension GL_KHR_shader_subgroup_shuffle : enable
```

The host (ARTX18) compiles each shader with the appropriate spec
constants; if a device lacks one of the extensions, the host picks a
different (non-subgroup) SPIR-V variant.

### 5.3 Subgroup reduction pattern (canonical)

The canonical pattern, seen in `add.comp`, `multi_add.comp`,
`mul_mat_vec_base.glsl`, `rms_norm_partials.comp`,
`ssm_scan.comp`, `flash_attn_cm1.comp`, etc.:

```glsl
// 1. Each thread computes a partial
FLOAT_TYPE partial = ...;

// 2. Reduce within subgroup
partial = subgroupAdd(partial);  // or subgroupMax, subgroupMin

// 3. Lane 0 of each subgroup writes to shmem
if (gl_SubgroupInvocationID == 0) {
    shmem[gl_SubgroupID] = partial;
}
barrier();

// 4. One thread (typically lane 0 of subgroup 0) reduces across subgroups
if (gl_SubgroupID == 0 && gl_SubgroupInvocationID == 0) {
    FLOAT_TYPE result = shmem[0];
    for (uint s = 1; s < gl_NumSubgroups; ++s) {
        result += shmem[s];
    }
    // write result
}
```

Variants:

* `add.comp:50-65` uses a tree reduction across subgroups in shmem
  (`for (s = NumSubgroups/2; s > 0; s >>= 1)`), not a single-thread
  sequential sum.
* `flash_attn.comp:235-237` uses a single-thread sequential scan
  across subgroups (`for (s = 0; s < gl_NumSubgroups; ++s)`), which
  is fine when `gl_NumSubgroups` is small (≤4).
* `mul_mat_vec_base.glsl:144-158` does subgroup reduce, then
  single-thread sequential scan across subgroups, then write from
  `tid == 0`.

### 5.4 Clustered reduction pattern

`quantize_q8_1.comp:77, 106` reduces across clusters of 8 lanes
within a subgroup:

```glsl
const float amax = subgroupClusteredMax(thread_max, 8);
// ...
const float sum = subgroupClusteredAdd(thread_sum, 8);
```

This works because each Q8_1 block (32 elements) is processed by 8
threads (each handles a `vec4`). The cluster size (8) matches the
block layout.

`flash_attn.comp:126, 141` uses the same pattern for Q8 quantization
of the Q tile (cluster=8). `gated_delta_net.comp:65-92` uses a
`switch` statement with one `case` per cluster size (2, 4, 8, 16,
32, 64) because GLSL requires the cluster size to be a compile-time
constant — the switch lets the host set `LANES_PER_COLUMN` as a spec
constant and have the right case fold away after specialization.

### 5.5 Shuffle-based reduction pattern

`flash_attn.comp:432-437` reduces `Sf[r][c]` across `D_split` lanes
using `subgroupShuffleXor`:

```glsl
for (uint s = D_split / 2; s > 0; s >>= 1) {
    for (uint r = 0; r < rows_per_thread; ++r) {
        Sf[r][c] += subgroupShuffleXor(Sf[r][c], s);
    }
}
```

This is a butterfly reduction: each lane XORs its index with `s`,
gets the partner lane's value, and adds. After `log2(D_split)` steps,
every lane has the sum. No barrier, no shmem. The same pattern
appears at lines 547-549 (max), 586-588 (sum), 617-625 (Of).

`fwht.comp:78-85` uses `subgroupShuffleXor(val, h)` for the in-block
butterfly of the fast Walsh-Hadamard transform. The block size must
be ≤ `gl_SubgroupSize` for this to work — the shader checks this
implicitly via `BLOCK_SIZE` spec constant.

`topk_moe.comp:169-171` uses `subgroupShuffleXor` for warp-level
argmax reduction (3 fields: max_val, max_val_s, max_expert shuffled
in lockstep).

### 5.6 Ballot-based compaction pattern

`mul_mm_id_funcs.glsl:44-65` compacts predicate-true lanes into a
dense output array:

```glsl
uvec4 ballot = subgroupBallot(in_range && id == expert_idx);

// Lane 0 of each subgroup stores its ballot in shmem
if (gl_SubgroupInvocationID == 0) {
    ballots_sh[gl_SubgroupID] = ballot;
}
barrier();

// Each subgroup computes its base offset by summing prior subgroups' bit counts
uint subgroup_base = 0;
uint total = 0;
for (uint k = 0; k < gl_NumSubgroups; ++k) {
    if (k == gl_SubgroupID) subgroup_base = total;
    total += subgroupBallotBitCount(ballots_sh[k]);
}
barrier();

// Each lane computes its output index = subgroup_base + popcount of bits before it
uint idx = subgroup_base + subgroupBallotExclusiveBitCount(ballot);
if (in_range && id == expert_idx) {
    row_ids[_ne1 + idx - ic * BN] = u16vec2(ii0, ii1);
}
_ne1 += total;
```

This is the standard stream-compaction pattern: each lane that
satisfies the predicate gets a unique dense index via
`subgroupBallotExclusiveBitCount`. Cross-subgroup coordination uses
shmem + barrier.

`topk_nary_search.comp:166-184` uses the same pattern for top-K
compaction: ballot the "is this element in the top K" predicate,
count bits per subgroup, compute base offsets, write to dense output.

### 5.7 Vote-based mask-skip pattern

`flash_attn.comp:229`:

```glsl
bool all_less = subgroupAll(max_mask <= NEG_FLT_MAX_OVER_2);
```

`subgroupAll` returns true iff the predicate is true in all lanes of
the subgroup. Used here to check "is the entire mask block -inf?" —
if so, skip the K block entirely. The result is broadcast via shmem
(lines 231-237) to coordinate the skip across subgroups.

`flash_attn_cm1.comp:211` uses the same pattern.

---

## 6. Data Layout

### 6.1 Subgroup-sized shmem arrays

Several shaders declare shmem arrays sized to `gl_NumSubgroups`:

```glsl
// flash_attn_mask_opt.comp:33-34
shared float minsh[NUM_SUBGROUPS];
shared float maxsh[NUM_SUBNUM_SUBGROUPS];
```

```glsl
// mul_mm_id_funcs.glsl:6
shared uvec4 ballots_sh[NUM_WARPS];
```

```glsl
// mul_mm_cm2.comp:116
shared uvec4 ballots_sh[BLOCK_SIZE / subgroup_size];
```

```glsl
// add.comp, multi_add.comp
shared ... sumsh[NumSubgroups];  // computed at runtime
```

`NUM_SUBGROUPS` and `subgroup_size` are spec constants. The host
must specialize them to match the actual subgroup size and
workgroup size; otherwise the shmem array is sized wrong.

### 6.2 `gl_SubgroupInvocationID`-indexed shmem

In `flash_attn.comp:553-557`:

```glsl
if (gl_SubgroupInvocationID == d_tid) {
    tmpsh[gl_SubgroupID * D_split + d_tid] = rowmaxf;
}
barrier();
rowmaxf = tmpsh[d_tid];
for (uint32_t s = 1; s < num_subgroups; ++s) {
    rowmaxf = max(rowmaxf, tmpsh[s * D_split + d_tid]);
}
```

Only one lane per subgroup (the one where `gl_SubgroupInvocationID
== d_tid`) writes to shmem. The shmem is sized
`num_subgroups * D_split` and indexed by `(subgroup_id, d_tid)`. The
sequential scan across subgroups is fine when `num_subgroups` is
small.

### 6.3 Subgroup scope for cooperative matrix

`mul_mm.comp:262-264` (coopmat1 path):

```glsl
coopmat<FLOAT_TYPE, gl_ScopeSubgroup, TM, TK, gl_MatrixUseA> cache_a;
coopmat<FLOAT_TYPE, gl_ScopeSubgroup, TK, TN, gl_MatrixUseB> cache_b;
coopmat<ACC_TYPE, gl_ScopeSubgroup, TM, TN, gl_MatrixUseAccumulator> sums[...];
```

Scope is `gl_ScopeSubgroup` — the matrix is distributed across one
subgroup. Each subgroup owns its own `sums[]` array; no cross-
subgroup coordination needed during the K-loop. Cross-subgroup
coordination happens only at the output store stage (via shmem
`coopmat_stage[]`).

`mul_mm_cm2.comp:368` (coopmat2 path):

```glsl
coopmat<ACC_TYPE, gl_ScopeWorkgroup, BM, BNover4, gl_MatrixUseAccumulator> sum;
```

Scope is `gl_ScopeWorkgroup` — the matrix is distributed across the
*entire* workgroup. The cooperative matrix hardware handles the
cross-subgroup coordination internally.

---

## 7. Memory Layout

### 7.1 Subgroup ballot storage

`mul_mm_id_funcs.glsl:6` declares `shared uvec4 ballots_sh[NUM_WARPS]`
where `NUM_WARPS = BLOCK_SIZE / WARP`. Each `uvec4` holds one
subgroup's ballot (128 bits, enough for subgroups up to 128 lanes).
The host must ensure `WARP` matches the actual subgroup size.

`mul_mm_cm2.comp:115-116`:

```glsl
layout (constant_id = 6) const uint subgroup_size = 32;
shared uvec4 ballots_sh[BLOCK_SIZE / subgroup_size];
```

This is a hardcoded `subgroup_size = 32` spec constant used only for
shmem sizing. The host must specialize it correctly; otherwise the
array is sized wrong.

### 7.2 Subgroup-ordered accumulator staging

`flash_attn.comp:47-48`:

```glsl
shared float tmpsh[tmpsh_size];           // tmpsh_size = num_subgroups * D_split (when row_split == 1)
shared FLOAT_TYPEV4 tmpshv4[tmpsh_size];
```

`tmpsh_size` is computed at compile time from spec constants
(`flash_attn.comp:46`):

```glsl
const uint32_t tmpsh_size = (SubGroupSize > 0) ?
    (row_split == 1 ? num_subgroups * D_split : num_subgroups) :
    WorkGroupSize;
```

When `SubGroupSize == 0` (subgroups disabled), the shmem falls back
to `WorkGroupSize` slots and the reduction uses pure shmem tree
reduction (`flash_attn.comp:563-573`).

### 7.3 Cooperative matrix staging shmem

`mul_mm.comp:133` (coopmat1):

```glsl
shared ACC_TYPE coopmat_stage[TM * TN * NUM_WARPS];
```

Each warp's coopmat accumulator is staged through this shmem before
being scattered to the output buffer by all lanes. Sized to one
coopmat-tile per warp.

---

## 8. Parallelism Strategy

### 8.1 Subgroup size as workgroup-axis unit

Most subgroup shaders treat the subgroup as the unit of workgroup-
level parallelism:

* `mul_mm.comp:184`: `warp_i = gl_LocalInvocationID.x / WARP` (or
  `gl_SubgroupID` in the coopmat path). Each "warp" (subgroup) owns
  one `WM × WN` tile of the output.
* `conv2d_mm.comp:235-236`: `warp_r = gl_SubgroupID / warps_N;
  warp_c = gl_SubgroupID % warps_N`. Subgroups are arranged in a 2D
  grid.
* `flash_attn_cm1.comp:473`: `o_offset = gl_SubgroupID * MatBr / 4`.
  Each subgroup owns one coopmat tile of the output.

### 8.2 `gl_SubgroupSize` as stride

`rms_norm_partials.comp:43`:

```glsl
for (uint32_t i = gl_SubgroupInvocationID; i < num_partials; i += gl_SubgroupSize) {
    sum += partial_sums[i];
}
sum = subgroupAdd(sum);
```

The stride-`gl_SubgroupSize` loop covers the input array
cooperatively across the subgroup, then `subgroupAdd` collapses the
per-lane partials into one value. This is the standard "subgroup as
warp" pattern.

### 8.3 Subgroup vs shmem: when each is used

| Shader                    | Subgroup path                   | Shmem path                       | When to use subgroup                |
| ------------------------- | ------------------------------- | -------------------------------- | ----------------------------------- |
| `quantize_q8_1.comp`      | `subgroupClusteredMax/Add(8)`   | 4-step tree on `shmem[8]`        | `USE_SUBGROUPS` spec constant       |
| `mul_mat_vec_base.glsl`   | `subgroupAdd`                   | tree on `tmpsh[BLOCK_SIZE]`      | `USE_SUBGROUP_ADD` or `USE_SUBGROUP_ADD_NO_SHMEM` |
| `mul_mat_vec_p021.comp`   | `subgroupAdd`                   | tree on `tmp[8][BLOCK_SIZE]`     | `USE_SUBGROUP_ADD`                  |
| `flash_attn.comp`         | `subgroupShuffleXor`            | tree on `tmpsh[]` / `tmpshv4[]`  | `SubGroupSize > 0` spec constant    |
| `fwht.comp`               | `subgroupShuffleXor` butterfly  | shmem butterfly on `shmem[4*N]`  | `!defined(FWHT_SHMEM)`              |
| `gated_delta_net.comp`    | `subgroupClusteredAdd` / `subgroupAdd` | tree on `temp[SUBGROUP_SIZE]` | `USE_SUBGROUP_ADD` / `USE_SUBGROUP_CLUSTERED` |
| `conv2d_mm.comp`          | `subgroupShuffle` for index broadcast | recomputes per-lane        | `USE_COLLECTIVES`                   |

The pattern is uniform: subgroup path is faster (no barrier), shmem
path is the fallback. The host picks based on device capability.

---

## 9. SIMD / GPU Strategy

### 9.1 Subgroup operations as "warp-level primitives"

In CUDA/Metal terms:

| Vulkan subgroup op             | CUDA equivalent              | Metal equivalent                |
| ------------------------------ | ---------------------------- | ------------------------------- |
| `subgroupAdd` / `Max` / `Min`  | `__reduce_add_sync` / etc.   | `simd_sum` / `simd_max` / `simd_min` |
| `subgroupClusteredAdd(partial, N)` | (no direct equivalent)   | `simd_sum` on subset            |
| `subgroupExclusiveAdd`         | `__shfl_up_sync` + manual    | `simd_prefix_sum_exclusive`     |
| `subgroupInclusiveAdd`         | `__shfl_up_sync` + manual    | `simd_prefix_sum_inclusive`     |
| `subgroupShuffleXor(val, m)`   | `__shfl_xor_sync`            | `simd_shuffle_xor`              |
| `subgroupShuffle(val, idx)`    | `__shfl_sync`                | `simd_shuffle`                  |
| `subgroupBallot(pred)`         | `__ballot_sync`              | `simd_ballot`                   |
| `subgroupBallotBitCount(b)`    | `__popc` on `b.x` (if ≤ 32)  | `popcount` on ballot            |
| `subgroupBallotExclusiveBitCount(b)` | `__clz` + manual       | `simd_prefix_exclusive_ballot`  |
| `subgroupAll(pred)`            | `__all_sync`                 | `simd_all`                      |
| `gl_SubgroupInvocationID`      | `laneid`                     | `thread_index_in_simdgroup`     |
| `gl_SubgroupID`                | `warpIdx` (manual)           | `simdgroup_index_in_threadgroup`|
| `gl_SubgroupSize`              | `warpSize` (32)              | `simd_length`                   |
| `gl_NumSubgroups`              | `blockDim.x / warpSize`      | `thread_execution_width / simd_length` |

### 9.2 Cooperative matrix as "tensor core"

`coopMatMulAdd` is the Vulkan equivalent of CUDA's `wmma::mma_sync`
or `__tensor_core_multiply_add`. The cooperative matrix type
(`coopmat<T, scope, M, N, use>`) is the Vulkan equivalent of
`wmma::fragment<T, M, N, K>`. The two flavors:

| Flavor    | Extension                     | Scope              | Hardware analogue                        |
| --------- | ----------------------------- | ------------------ | ---------------------------------------- |
| Coopmat1  | `GL_KHR_cooperative_matrix`   | `gl_ScopeSubgroup` | AMD MFMA, Intel XMX, NVIDIA (non-Hopper) |
| Coopmat2  | `GL_NV_cooperative_matrix2`   | `gl_ScopeWorkgroup`| NVIDIA Hopper/Blackwell tensor cores     |

Coopmat1's scope is the subgroup — one coopmat lives in one
subgroup. Coopmat2's scope is the workgroup — the coopmat is
distributed across all subgroups in the workgroup, and the hardware
handles cross-subgroup coordination. This is why coopmat2 shaders
don't need manual cross-subgroup reductions for the matmul itself
(only for MoE row-id compaction, which is a separate concern).

### 9.3 VALVE mixed-float dot product

`dot_product_funcs.glsl:4-9` declares a `spirv_instruction` for
`SPV_VALVE_mixed_float_dot_product` (capability 6912, id 6916). This
is an AMD/RDNA-specific extension that exposes a 2-way f16 dot with
f32 accumulate in one instruction (`v_dot2_f32_f16`). Used by the
FA scalar path when `DOT2_F16` is defined. This is not technically
a subgroup operation, but it's a subgroup-*adjacent* SIMD primitive
that complements subgroup reductions.

### 9.4 Integer dot product (MMQ)

`mul_mmq.comp:7` requires `GL_EXT_integer_dot_product`. The
`dotPacked4x8EXT(a, b)` function performs 4-way int8 signed dot
product with int32 accumulate in one instruction. This is also not
a subgroup operation, but it's the SIMD primitive that makes MMQ
competitive with cooperative matrix on integer-SDP-capable hardware.

---

## 10. Quantization Strategy

Subgroup operations interact with quantization in three places:

### 10.1 On-the-fly Q quantization in FA MMQ

`flash_attn.comp:122-147` quantizes Q (the activations) to Q8_0/Q4_*
inside the FA shader before the integer dot product. The per-block
scale computation uses:

```glsl
const FLOAT_TYPE thread_max = max(max(abs_vals.x, abs_vals.y), max(abs_vals.z, abs_vals.w));
const FLOAT_TYPE amax = subgroupClusteredMax(thread_max, 8);
const FLOAT_TYPE qd = amax / FLOAT_TYPE(127.0);
// ... quantize ...
const FLOAT_TYPE thread_sum = vals.x + vals.y + vals.z + vals.w;
const FLOAT_TYPE sum = subgroupClusteredAdd(thread_sum, 8);
```

The cluster size (8) matches the Q8_0 block layout (32 elements /
4 per thread = 8 threads per block). This is the same pattern as
`quantize_q8_1.comp:77, 106`.

### 10.2 Quantize Q8_1 kernel

`quantize_q8_1.comp:65-107` has dual paths:

* `USE_SUBGROUPS` defined: `subgroupClusteredMax(8)` /
  `subgroupClusteredAdd(8)` — no barrier, no shmem.
* `USE_SUBGROUPS` undefined: `shmem[tid] = thread_max; barrier();
  for (s = 4; s > 0; s >>= 1) { shmem[tid] = max(shmem[tid],
  shmem[tid + s]); barrier(); }` — 3 barriers, shmem traffic.

The subgroup path is strictly faster on subgroup-capable hardware.

### 10.3 Subgroup-ballot MoE row-id compaction

`mul_mm_id_funcs.glsl:44-65` (used by `mul_mm.comp` and
`mul_mmq.comp` when `MUL_MAT_ID_USE_SUBGROUPS` is defined) compacts
the MoE expert-id matching rows into a dense `row_ids[]` array via
`subgroupBallot` + `subgroupBallotBitCount` +
`subgroupBallotExclusiveBitCount`. This replaces a sequential scan
that would otherwise require a global atomic or a serial loop.

---

## 11. Correctness Analysis

### 11.1 Reduction order non-determinism

`subgroupAdd`, `subgroupMax`, `subgroupMin` have implementation-
defined reduction order across lanes. The result is bit-reproducible
per (vendor, driver, subgroup size) but not across vendors. This
affects:

* GEMV `reduce_result` (`mul_mat_vec_base.glsl:97, 139`) — ULP-level
  differences vs the shmem tree path.
* FA `subgroupShuffleXor` reductions (`flash_attn.comp:434, 548,
  587, 619`) — ULP-level differences across vendors.
* Quantize Q8_1 `subgroupClusteredMax/Add(8)` — integer, exact.
* `subgroupBallotBitCount` — integer, exact.

### 11.2 Subgroup size assumption

Most shaders are written to work with any subgroup size, but a few
have implicit assumptions:

* `mul_mm_cm2.comp:115` `subgroup_size = 32` is used to size
  `ballots_sh[]`. If the actual subgroup size is 64, the array is
  half the required size — ballot data would be lost. The host must
  specialize this constant correctly.
* `quantize_q8_1.comp:77` `subgroupClusteredMax(thread_max, 8)`
  requires subgroup size ≥ 8. True on every Vulkan-conformant GPU.
* `gated_delta_net.comp:74-84` has `case 64u: subgroupClusteredAdd(
  partial, 64u)` — requires subgroup size ≥ 64. Only true on some
  Intel Xe GPUs (subgroup size 8/16/32) — the host must not
  specialize `LANES_PER_COLUMN = 64` on sub-64 hardware.
* `flash_attn.comp:34` `num_subgroups = SubGroupSize == 0 ? 0 :
  WorkGroupSize / SubGroupSize`. When `SubGroupSize == 0` (subgroups
  disabled), `num_subgroups == 0` and the shmem path is used
  (`flash_attn.comp:46`). When `SubGroupSize > 0`, the host must
  ensure `WorkGroupSize % SubGroupSize == 0`.

### 11.3 Partial subgroups

`vk::PipelineShaderStageCreateFlagBits::eRequireFullSubgroupsEXT`
(ARTX18) ensures the last subgroup in a workgroup is full. Without
this flag, the last subgroup may have fewer than `subgroupSize`
active lanes, which breaks `subgroupAdd` / `subgroupBallot` (which
are defined on all lanes but produce undefined results on inactive
lanes for some operations).

The shaders don't explicitly check `eRequireFullSubgroupsEXT`; they
rely on the host to set it correctly. If the host forgets, the last
subgroup's reduction may be wrong.

### 11.4 `subgroupBallot` on inactive lanes

`subgroupBallot(pred)` returns a `uvec4` where bit `i` is set iff
lane `i` is active *and* `pred` is true. Inactive lanes contribute
0 to the ballot. `subgroupBallotBitCount(b)` returns the popcount
of the ballot, which equals the number of active lanes with
`pred == true`. This is well-defined even with partial subgroups.

`subgroupBallotExclusiveBitCount(b)` returns the popcount of bits
*before* the current lane. Used to compute the dense output index
for each predicate-true lane. Also well-defined with partial
subgroups.

### 11.5 `OLD_AMD_WINDOWS` workaround

`flash_attn.comp:26, 617-625` works around an AMD RDNA2 Windows
driver bug where `subgroupShuffleXor` on `f16vec4` produces wrong
results. The workaround shuffles `vec4` (F32) instead, paying a
conversion cost. Cited issue: llama.cpp #19881.

This is a per-driver bug, not a spec compliance issue. The
workaround is gated by the `OLD_AMD_WINDOWS` flag bit in the `Flags`
spec constant (`flash_attn_base.glsl:26`), which the host sets
based on device detection (ARTX18).

---

## 12. Optimization Analysis

### 12.1 Identified subgroup optimizations

| Optimization                                  | Where                                              | Notes                                                            |
| --------------------------------------------- | -------------------------------------------------- | ---------------------------------------------------------------- |
| 3-way `reduce_result` (subgroup-only / +shmem / shmem-only) | `mul_mat_vec_base.glsl:93-228`        | Host picks the best reduction per device.                        |
| `subgroupClusteredMax/Add(8)` for per-block reduce | `quantize_q8_1.comp:77, 106`, `flash_attn.comp:126, 141` | Cluster size matches block layout. No barrier.                   |
| `subgroupShuffleXor` for D_split reduction in FA | `flash_attn.comp:432-437, 547-549, 586-588, 617-625` | Butterfly reduce; no barrier, no shmem.                          |
| `subgroupBallot` MoE row-id compaction        | `mul_mm_id_funcs.glsl:44-65`, `mul_mm_cm2.comp:199-216` | Replaces serial scan; O(log N) vs O(N).                          |
| `subgroupBallot` top-K compaction             | `topk_nary_search.comp:166-184, 188-212`           | Same pattern for top-K output compaction.                        |
| `subgroupAll` mask-skip in FA                 | `flash_attn.comp:229`, `flash_attn_cm1.comp:211`   | Skip K-block load if entire mask is -inf.                        |
| `subgroupMin/Max` block summary               | `flash_attn_mask_opt.comp:68-69, 136-137`          | Per-block min/max for mask summary.                              |
| `subgroupExclusiveAdd` per-subgroup scan      | `cumsum.comp:54`                                   | Prefix sum in one instruction; cross-subgroup via shmem.         |
| `subgroupInclusiveAdd` per-subgroup scan      | `cumsum_multipass1.comp:42`, `topk_nary_search.comp:129` | Inclusive scan in one instruction.                               |
| `subgroupShuffle` index broadcast             | `conv2d_mm.comp:264-267, 318-321`                  | Broadcast computed indices to all lanes; avoids redundant compute. |
| `subgroupShuffleXor` butterfly in FWHT        | `fwht.comp:78-85`                                  | In-block butterfly without shmem (vs `FWHT_SHMEM` path).         |
| `subgroupShuffleXor` argmax in topk_moe       | `topk_moe.comp:169-171`                            | 3-field (val, val_s, expert) argmax reduction.                   |
| Clustered-reduce switch for variable cluster  | `gated_delta_net.comp:65-92`                       | Workaround for GLSL compile-time-constant cluster size requirement. |
| Subgroup + shmem dual path                    | `quantize_q8_1.comp`, `mul_mat_vec_base.glsl`, `flash_attn.comp`, `fwht.comp`, `gated_delta_net.comp`, `conv2d_mm.comp` | Universal pattern: subgroup path when supported, shmem fallback otherwise. |

### 12.2 Optimizations not present

* **No `subgroupBroadcastFirst`**. The mask-skip pattern in
  `flash_attn.comp:229-240` uses `subgroupAll` then writes the
  result to shmem and reads it back. A single
  `subgroupBroadcastFirst(all_less ? NEG : 0)` followed by a
  cross-subgroup shmem reduction (only if `gl_NumSubgroups > 1`)
  would be simpler and faster.
* **No `subgroupQuadSwap` / `subgroupQuadBroadcast`**. Quad-level
  operations (cluster size 4) are useful for 4-lane reductions
  (e.g., per-`vec4` reductions). The shaders use
  `subgroupClusteredAdd(4)` instead, which is more general but may
  be slower on hardware that has dedicated quad instructions.
* **No `subgroupClusteredMax` / `Min` with cluster > 32**. The
  cluster size is limited to `gl_SubgroupSize`. For workgroup-level
  reductions, the code always falls back to shmem.
* **No `subgroupClusteredBallot`**. The ballot is always full-
  subgroup; clustered ballots (e.g., per-quad) would let the code
  compact within clusters before cross-cluster coordination.

---

## 13. Architectural Strengths

1. **Universal dual-path pattern**. Every subgroup-using shader has
   a spec-constant-gated shmem fallback. This is the cleanest way to
   handle heterogeneous subgroup support: one source file, two
   SPIR-Vs, host picks based on device capability.

2. **Clustered reductions match block layout**. `quantize_q8_1.comp`
   and FA's on-the-fly Q quantization use `subgroupClusteredMax(8)`
   because the Q8_1 block has 32 elements / 4 per thread = 8 threads
   per block. The cluster size (8) is not arbitrary — it's dictated
   by the block layout. This is the right way to use clustered
   reductions.

3. **Ballot-based MoE row-id compaction**. `mul_mm_id_funcs.glsl`
   replaces a serial scan with `O(log N)` subgroup ballot
   operations. For MoE with many experts, this is a significant
   speedup.

4. **`subgroupShuffleXor` for D_split reduction in FA**. The
   butterfly reduction is the canonical pattern for cross-lane sum
   without shmem. The `OLD_AMD_WINDOWS` workaround shows the team
   understands the failure modes and has a documented fallback.

5. **Cluster-size switch in `gated_delta_net.comp`**. The GLSL spec
   requires `subgroupClusteredAdd`'s cluster size to be a compile-
   time constant. The switch with one case per power of two lets
   the host set `LANES_PER_COLUMN` as a spec constant and have the
   right case fold away after specialization. Documented workaround.

6. **Subgroup scope vs workgroup scope for cooperative matrix**.
   Coopmat1 uses `gl_ScopeSubgroup` (one coopmat per subgroup);
   coopmat2 uses `gl_ScopeWorkgroup` (one coopmat per workgroup).
   The choice is principled: coopmat1 hardware (AMD MFMA, Intel XMX)
   has per-subgroup matrix units; coopmat2 hardware (NVIDIA Hopper)
   has per-workgroup tensor cores.

7. **`subgroupAll` for mask-skip in FA**. A single instruction
   replaces what would otherwise be a shmem tree reduction. The
   result is broadcast via shmem only because the skip decision
   must be uniform across the workgroup.

---

## 14. Architectural Weaknesses

### W1 — `mul_mm_cm2.comp` hardcodes `subgroup_size = 32`

**Evidence**: `mul_mm_cm2.comp:115-116`:
```glsl
layout (constant_id = 6) const uint subgroup_size = 32;
shared uvec4 ballots_sh[BLOCK_SIZE / subgroup_size];
```

**Impact**: On a GPU with subgroup size 64 (some Intel Xe), the
`ballots_sh[]` array is half the required size — ballot data from
the upper 32 lanes of each subgroup would be lost. The host must
specialize `subgroup_size` correctly; if it doesn't, the MoE
row-id compaction is silently wrong.

**Fix**: Use `gl_SubgroupSize` instead — but GLSL doesn't allow
`gl_SubgroupSize` in array size declarations (it's a runtime
constant, not a compile-time constant). The correct fix is to
declare `ballots_sh[BLOCK_SIZE]` (one slot per lane, wasteful but
correct) or use a sufficiently large fixed size (`BLOCK_SIZE / 8`
covers subgroup sizes up to 64).

### W2 — `flash_attn.comp` mask broadcast uses shmem even when one subgroup would suffice

**Evidence**: `flash_attn.comp:229-240`:
```glsl
bool all_less = subgroupAll(max_mask <= NEG_FLT_MAX_OVER_2);
barrier();
if (gl_SubgroupInvocationID == 0) {
    tmpsh[gl_SubgroupID] = all_less ? NEG_FLT_MAX_OVER_2 : 0.0f;
}
barrier();
for (uint s = 0; s < gl_NumSubgroups; ++s) {
    max_mask = max(max_mask, tmpsh[s]);
}
```

**Impact**: When `gl_NumSubgroups == 1` (common for small
workgroups), the shmem round-trip is unnecessary — a single
`subgroupBroadcastFirst(all_less ? NEG : 0)` would suffice. The
shader doesn't special-case this.

**Fix**: Add a `if (gl_NumSubgroups == 1)` fast path that uses
`subgroupBroadcastFirst` instead of shmem.

### W3 — `soft_max.comp` / `argmax.comp` / `sum_rows.comp` don't use subgroups at all

**Evidence**: `soft_max.comp` has no `#extension GL_KHR_shader_subgroup_*`
directives; it uses pure shmem tree reduction (`soft_max.comp:103-110,
143-150`). Same for `argmax.comp:46-55` and `sum_rows.comp:36-42`.

**Impact**: On subgroup-capable hardware, these kernels are 5×
slower than necessary. Compare with `quantize_q8_1.comp:77` which
uses `subgroupClusteredMax(8)` for the same kind of per-block max.

**Why it's hard to fix**: The shaders pre-date widespread subgroup
support. Adding a `USE_SUBGROUPS` spec-constant path (like
`quantize_q8_1.comp` does) would require reworking the
template-specialization dispatch in `main()`.

### W4 — `gated_delta_net.comp` switch statement for cluster size is fragile

**Evidence**: `gated_delta_net.comp:65-92` has cases for 2, 4, 8,
16, 32, 64. If `LANES_PER_COLUMN` is set to a value not in this
list (e.g., 128), the `default` case falls through to
`subgroupAdd` (or `reduce_add_shmem`), which is correct but slower.

**Impact**: Adding new cluster sizes requires editing the switch.
A `subgroupClusteredAdd(partial, cluster_size)` that takes a
runtime cluster size would be cleaner, but GLSL forbids it.

**Fix**: None within GLSL. Document the limitation.

### W5 — `conv2d_mm.comp` `subgroupShuffle` path is commented out

**Evidence**: `conv2d_mm.comp:18`:
```glsl
//#    extension GL_KHR_shader_subgroup_shuffle : enable
```

The `subgroupShuffle` calls at lines 264-267 and 318-321 are
gated on `USE_COLLECTIVES` but the extension is commented out. The
host must enable the extension via a different mechanism (probably
a `-DUSE_COLLECTIVES` define that triggers an `#extension` elsewhere,
or the extension is enabled by `GL_KHR_cooperative_matrix` which is
already enabled).

**Impact**: Confusing. If `USE_COLLECTIVES` is defined but the
extension isn't enabled, the shader fails to compile.

**Fix**: Uncomment the extension directive, or document that it's
enabled implicitly by another extension.

### W6 — No `subgroupBroadcastFirst` anywhere

**Evidence**: `grep subgroupBroadcastFirst` returns no matches in
the shader directory.

**Impact**: The mask-skip pattern in FA (W2) and the MoE row-id
base-offset broadcast in `mul_mm_id_funcs.glsl:51-58` both use
shmem + barrier where `subgroupBroadcastFirst` would suffice for
the intra-subgroup broadcast.

**Fix**: Use `subgroupBroadcastFirst` for intra-subgroup
broadcasts; keep shmem only for cross-subgroup coordination.

### W7 — No `subgroupElect` usage

**Evidence**: `grep subgroupElect` returns no matches.

**Impact**: `subgroupElect` (returns true in exactly one lane per
subgroup) is the canonical way to do "lane 0 does work". The
shaders use `if (gl_SubgroupInvocationID == 0)` instead, which is
equivalent but less idiomatic.

**Fix**: Cosmetic. No functional difference.

### W8 — `flash_attn_cm1.comp` uses `subgroupMax` / `subgroupAdd` but no clustered variant

**Evidence**: `flash_attn_cm1.comp:359, 529`:
```glsl
rowmaxf = subgroupMax(rowmaxf);
Lf[r] = subgroupAdd(Lf[r]);
```

**Impact**: Whole-subgroup reduction is correct but may be slower
than a clustered reduction if the workgroup has multiple subgroups
and the reduction only needs to span one subgroup. The coopmat1
path's row_split parameter implies the reduction scope is one
subgroup, so whole-subgroup is correct here.

**Fix**: None. Correct as-is.

### W9 — Cross-subgroup reduction is always sequential scan

**Evidence**: `flash_attn.comp:235-237, 558-560, 596-598, 634-636`:
```glsl
for (uint s = 0; s < gl_NumSubgroups; ++s) {
    max_mask = max(max_mask, tmpsh[s]);
}
```

**Impact**: O(gl_NumSubgroups) sequential reads from shmem. For
workgroups with 8+ subgroups, this is a measurable cost. A tree
reduction across subgroups (`for (s = NumSubgroups/2; s > 0; s >>= 1)`)
would be O(log NumSubgroups).

**Fix**: Use a tree reduction across subgroups (like `add.comp:57-60`
does).

### W10 — `mul_mat_vec_base.glsl` has three `reduce_result` overloads but no documentation of which to pick

**Evidence**: `mul_mat_vec_base.glsl:93-228` defines three
`reduce_result` functions gated on `USE_SUBGROUP_ADD_NO_SHMEM`,
`USE_SUBGROUP_ADD`, and the default. The comment at line 133 says
"subgroupAdd is probably faster on devices that support it,
particularly when the workgroup has more than one subgroup" — but
doesn't say *when* to use `NO_SHMEM` vs `USE_SUBGROUP_ADD`.

**Impact**: The host must empirically pick between the two subgroup
paths. Unclear which is better when.

**Fix**: Document the trade-off: `NO_SHMEM` is faster when the
workgroup is one subgroup (no cross-subgroup coordination needed);
`USE_SUBGROUP_ADD` is needed when the workgroup spans multiple
subgroups.

---

## 15. GwenLand Mapping

| GwenLand module | ADOPT / ADAPT / REJECT / MONITOR / DEFER | What | Reasoning |
| --------------- | ---------------------------------------- | ---- | --------- |
| `glvulkan`      | **ADOPT** | Spec-constant-gated subgroup/shmem dual path | Universal pattern; one source, two SPIR-Vs; host picks per device. |
| `glvulkan`      | **ADOPT** | `subgroupClusteredMax/Add(N)` for per-block reductions where N matches block layout | No barrier; matches the natural data tile. |
| `glvulkan`      | **ADOPT** | `subgroupBallot` + `BitCount` + `ExclusiveBitCount` for MoE row-id compaction | Replaces serial scan; O(log N) vs O(N). |
| `glvulkan`      | **ADOPT** | `subgroupShuffleXor` butterfly for D_split / small reductions | No barrier, no shmem for intra-subgroup reduce. |
| `glvulkan`      | **ADOPT** | `subgroupAll` / `subgroupAny` for mask-skip | Single instruction replaces shmem tree. |
| `glvulkan`      | **ADOPT** | `subgroupExclusiveAdd` / `subgroupInclusiveAdd` for prefix sums | One instruction vs O(log N) shmem rounds. |
| `glvulkan`      | **ADAPT** | `subgroupBroadcastFirst` for intra-subgroup broadcast (NOT in llama.cpp) | Add this where llama.cpp uses shmem for intra-subgroup broadcast. |
| `glvulkan`      | **ADAPT** | Tree reduction across subgroups (instead of sequential scan) | O(log N) vs O(N) for `gl_NumSubgroups` ≥ 8. |
| `glvulkan`      | **REJECT**| `subgroup_size = 32` hardcoded spec constant for shmem sizing | Use `gl_SubgroupSize`-aware sizing or a sufficiently large fixed array. |
| `glvulkan`      | **REJECT**| Pure shmem tree reduction in softmax/argmax/sum_rows | Use subgroup reductions; shmem fallback for non-subgroup hardware. |
| `glvulkan`      | **MONITOR**| `OLD_AMD_WINDOWS` workaround for `f16vec4 subgroupShuffleXor` | Monitor AMD driver fixes; remove when fixed. |
| `glvulkan`      | **MONITOR**| `gated_delta_net.comp` cluster-size switch | Monitor for GLSL spec changes allowing runtime cluster size. |
| `glvulkan`      | **DEFER** | `subgroupElect` (use `gl_SubgroupInvocationID == 0` instead) | Cosmetic; no functional difference. |
| `glvulkan`      | **DEFER** | `subgroupQuadSwap` / `subgroupQuadBroadcast` | Only relevant if GwenLand targets hardware with dedicated quad instructions. |
| `GATE`          | **ADOPT** | Per-pipeline subgroup-size selection via `VK_EXT_subgroup_size_control` | Already adopted on host (ARTX18); shader side provides the spec-constant axes. |

---

## 16. Recommendations

### R1 — ADOPT spec-constant-gated subgroup/shmem dual path
**Priority:** Critical
**Difficulty:** M
**Dependencies:** none
Every subgroup-using shader in `glvulkan` should have a `USE_SUBGROUPS`
(or similar) spec constant that gates the subgroup path, with a shmem
tree-reduction fallback. The host picks the path per pipeline based
on device capability. Same pattern as `quantize_q8_1.comp` and
`mul_mat_vec_base.glsl`.

### R2 — ADOPT `subgroupClusteredMax/Add(N)` for per-block reductions
**Priority:** High
**Difficulty:** S
**Dependencies:** R1
Use clustered reductions where the cluster size matches the natural
data tile (e.g., 8 for Q8_1 blocks, 4 for vec4 reductions). The
cluster size must be a compile-time constant; use a switch statement
with one case per power of two if the cluster size is a spec constant
(same pattern as `gated_delta_net.comp:65-92`).

### R3 — ADOPT `subgroupBallot`-based MoE row-id compaction
**Priority:** High
**Difficulty:** M
**Dependencies:** R1
GwenLand's MoE matmul should use the ballot-based compaction pattern
from `mul_mm_id_funcs.glsl:44-65`: `subgroupBallot` to capture the
predicate-true lanes, `subgroupBallotBitCount` for per-subgroup
counts, `subgroupBallotExclusiveBitCount` for per-lane dense index.
Cross-subgroup coordination via shmem + barrier.

### R4 — ADOPT `subgroupShuffleXor` butterfly for small reductions
**Priority:** High
**Difficulty:** S
**Dependencies:** R1
For reductions that fit within one subgroup (e.g., FA's D_split
reduction across 16 lanes), use `subgroupShuffleXor` butterfly
instead of shmem tree. No barrier, no shmem. Fall back to shmem
when the reduction spans multiple subgroups.

### R5 — ADOPT `subgroupBroadcastFirst` for intra-subgroup broadcast
**Priority:** Medium
**Difficulty:** S
**Dependencies:** R1
Where llama.cpp uses `if (gl_SubgroupInvocationID == 0) shmem[...] =
val; barrier(); val = shmem[0]` for intra-subgroup broadcast, use
`subgroupBroadcastFirst(val)` instead. Single instruction, no
barrier. Keep shmem only for cross-subgroup coordination.

### R6 — REJECT `subgroup_size = 32` hardcoded spec constant
**Priority:** Medium
**Difficulty:** S
**Dependencies:** R1
Don't hardcode subgroup size in shmem array declarations. Either:
(a) use a sufficiently large fixed size (`BLOCK_SIZE / 8` covers
subgroup sizes up to 64), or (b) declare the array per-lane
(`[BLOCK_SIZE]`) and accept the waste, or (c) use a `gl_SubgroupSize`-
aware layout if the GLSL frontend supports it.

### R7 — ADAPT cross-subgroup tree reduction
**Priority:** Medium
**Difficulty:** S
**Dependencies:** R1
Where llama.cpp uses `for (s = 0; s < gl_NumSubgroups; ++s) val =
max(val, shmem[s])` (sequential scan across subgroups), use a tree
reduction: `for (s = NumSubgroups/2; s > 0; s >>= 1) { if
(gl_SubgroupID < s) shmem[gl_SubgroupID] = max(shmem[gl_SubgroupID],
shmem[gl_SubgroupID + s]); barrier(); }`. O(log N) vs O(N).

### R8 — REJECT pure shmem reduction in softmax/argmax/sum_rows
**Priority:** High
**Difficulty:** S
**Dependencies:** R1
GwenLand's softmax/argmax/sum_rows should use `subgroupMax`/
`subgroupAdd`/`subgroupBallot`-based reductions, with a shmem
fallback for non-subgroup hardware. Same dual-path pattern as
`quantize_q8_1.comp`.

### R9 — MONITOR `OLD_AMD_WINDOWS` workaround
**Priority:** Low
**Difficulty:** S
**Dependencies:** none
Watch for AMD driver fixes for the `f16vec4 subgroupShuffleXor` bug
(llama.cpp issue #19881). Remove the workaround when fixed.

### R10 — DEFER `subgroupElect` / `subgroupQuadSwap` / `subgroupQuadBroadcast`
**Priority:** Low
**Difficulty:** S
**Dependencies:** none
`subgroupElect` is cosmetic (use `gl_SubgroupInvocationID == 0`
instead). Quad operations are only relevant on hardware with
dedicated quad instructions; defer until GwenLand targets such
hardware.

---

## 17. Findings

### Finding ARTX20-F01

```
Finding ID:           ARTX20-F01
Category:             SIMD_STRATEGY
Engine:               Vulkan
Component:            Subgroup/shmem dual-path pattern
Source File:          ggml/src/ggml-vulkan/vulkan-shaders/quantize_q8_1.comp
                      (also mul_mat_vec_base.glsl, flash_attn.comp, fwht.comp,
                      gated_delta_net.comp, conv2d_mm.comp)
Function:             quantize / reduce_result / main / reduce_partial
Lines:                quantize_q8_1.comp:6-13, 65-107; mul_mat_vec_base.glsl:5-8, 93-228;
                      flash_attn.comp:34, 46, 546-650; gated_delta_net.comp:45-93
Summary:              Every subgroup-using shader gates the subgroup path behind a spec
                      constant (USE_SUBGROUPS, USE_SUBGROUP_ADD, SubGroupSize,
                      USE_COLLECTIVES, FWHT_SHMEM) and provides a shared-memory tree
                      reduction fallback.
Observation:          The pattern is uniform: (1) enable GL_KHR_shader_subgroup_*
                      extensions conditionally on the spec constant; (2) define
                      INVOCATION_ID as gl_SubgroupInvocationID.x (subgroup path) or
                      gl_LocalInvocationID.x (shmem path); (3) write the reduction two
                      ways — subgroupClusteredMax/Add for the subgroup path, shmem[tid]
                      + barrier + tree loop for the shmem path. The host picks the path
                      per pipeline based on device capability. quantize_q8_1.comp is
                      the cleanest example: USE_SUBGROUPS gates both the extension
                      (lines 6-13) and the reduction body (lines 65-78 for max, 90-107
                      for sum).
Evidence:              quantize_q8_1.comp:6-13 (extension gate), :65-78 (subgroup max
                      path), :90-107 (subgroup sum path); mul_mat_vec_base.glsl:5-8
                      (gate), :93-128 (NO_SHMEM path), :129-182 (USE_SUBGROUP_ADD
                      path), :183-228 (default shmem path); flash_attn.comp:34
                      (num_subgroups formula), :46 (tmpsh_size formula), :546-650
                      (SubGroupSize > 0 vs == 0 branches).
Architectural Impact: One source file serves both subgroup-capable and legacy hardware.
                      The host compiles two SPIR-Vs (with and without the spec
                      constant) and dispatches to the right one per device. No
                      runtime branching cost — the spec constant is folded at
                      pipeline-create.
Correctness Impact:   None. Both paths produce identical results up to FMA
                      reassociation differences.
Optimization Type:    SIMD (subgroup ops when supported), fallback (shmem tree).
GwenLand Target:      glvulkan
Recommendation:       ADOPT. Same dual-path pattern in every subgroup-using shader.
Priority:             Critical
Difficulty:           M
Dependencies:         none
Confidence:           High
```

### Finding ARTX20-F02

```
Finding ID:           ARTX20-F02
Category:             SIMD_STRATEGY
Engine:               Vulkan
Component:            Clustered reductions for per-block reductions
Source File:          ggml/src/ggml-vulkan/vulkan-shaders/quantize_q8_1.comp,
                      flash_attn.comp, gated_delta_net.comp
Function:             quantize / main (FA Q quant) / reduce_partial
Lines:                quantize_q8_1.comp:77, 106; flash_attn.comp:126, 141;
                      gated_delta_net.comp:65-92
Summary:              subgroupClusteredMax/Add(N) used for per-block reductions where N
                      matches the natural data tile (8 for Q8_1 blocks, configurable
                      for gated_delta_net).
Observation:          quantize_q8_1.comp: 8 threads per Q8_1 block (32 elements / 4
                      per thread); subgroupClusteredMax(thread_max, 8) gives the
                      per-block max in one instruction. flash_attn.comp:126 same
                      pattern for on-the-fly Q quantization. gated_delta_net.comp
                      uses a switch statement (lines 65-92) because GLSL requires the
                      cluster size to be a compile-time constant — one case per power
                      of two, the right case folds away after specialization. Comment
                      at line 71 documents the workaround.
Evidence:              quantize_q8_1.comp:77 (subgroupClusteredMax(8)), :106
                      (subgroupClusteredAdd(8)); flash_attn.comp:126, 141 (same);
                      gated_delta_net.comp:65-92 (switch with cases 2,4,8,16,32,64).
Architectural Impact: No barrier, no shmem for the per-block reduction. Cluster size
                      matches the data layout — the reduction is "free" (one
                      instruction). The switch-statement workaround handles the GLSL
                      spec limitation that cluster size must be a compile-time constant.
Correctness Impact:   None. Integer reductions are exact; float reductions have the
                      usual ULP-level non-determinism.
Optimization Type:    SIMD (clustered subgroup ops).
GwenLand Target:      glvulkan
Recommendation:       ADOPT. Same clustered-reduce pattern where cluster size matches
                      data tile.
Priority:             High
Difficulty:           S
Dependencies:         ARTX20-F01
Confidence:           High
```

### Finding ARTX20-F03

```
Finding ID:           ARTX20-F03
Category:             SIMD_STRATEGY
Engine:               Vulkan
Component:            Ballot-based MoE row-id compaction
Source File:          ggml/src/ggml-vulkan/vulkan-shaders/mul_mm_id_funcs.glsl
                      (also mul_mm_cm2.comp)
Function:             load_row_ids
Lines:                mul_mm_id_funcs.glsl:1-74; mul_mm_cm2.comp:163-227
Summary:              MoE row-id compaction via subgroupBallot + subgroupBallotBitCount
                      + subgroupBallotExclusiveBitCount, replacing a serial scan with
                      O(log N) subgroup operations.
Observation:          The shader needs to compact the predicate-true lanes (where
                      data_ids[i] == expert_idx) into a dense row_ids[] array. The
                      pattern: (1) subgroupBallot(in_range && id == expert_idx) —
                      uvec4 with one bit per lane; (2) lane 0 of each subgroup stores
                      its ballot in ballots_sh[gl_SubgroupID] shmem; (3) barrier; (4)
                      each subgroup computes its base offset by summing prior
                      subgroups' bit counts (subgroupBallotBitCount); (5) each lane
                      computes its output index = subgroup_base +
                      subgroupBallotExclusiveBitCount(ballot); (6) predicate-true
                      lanes write to row_ids[_ne1 + idx]. The non-subgroup fallback
                      (mul_mm.comp:221-231) uses a serial nested loop, which is O(N²).
Evidence:              mul_mm_id_funcs.glsl:44 (subgroupBallot), :46-48 (lane 0 writes
                      to shmem), :53-58 (cross-subgroup base offset), :61
                      (ExclusiveBitCount for per-lane index), :62-64 (predicate-true
                      write); mul_mm_cm2.comp:199-216 (same pattern).
Architectural Impact: O(log N) vs O(N²) for the non-subgroup path. For MoE with many
                      experts and many tokens, this is a significant speedup. The
                      ballot pattern is the canonical stream-compaction primitive.
Correctness Impact:   None. Ballot/BitCount/ExclusiveBitCount are well-defined on
                      partial subgroups (inactive lanes contribute 0).
Optimization Type:    SIMD (subgroup ballot compaction).
GwenLand Target:      glvulkan
Recommendation:       ADOPT. Same ballot-based compaction for MoE row-ids.
Priority:             High
Difficulty:           M
Dependencies:         ARTX20-F01
Confidence:           High
```

### Finding ARTX20-F04

```
Finding ID:           ARTX20-F04
Category:             SIMD_STRATEGY
Engine:               Vulkan
Component:            Subgroup shuffle butterfly for D_split reduction in FA
Source File:          ggml/src/ggml-vulkan/vulkan-shaders/flash_attn.comp
Function:             main
Lines:                432-437, 547-549, 586-588, 617-625
Summary:              D_split head-dim partitioning reduced via subgroupShuffleXor
                      butterfly — no barrier, no shmem for intra-subgroup reduction.
                      OLD_AMD_WINDOWS workaround shuffles vec4 (F32) instead of
                      f16vec4 due to a driver bug.
Observation:          FA partitions the head dimension (HSK) across D_split lanes.
                      After computing Sf[r][c] = dot(Q, K) per lane, the partial sums
                      must be reduced across D_split lanes. The reduction uses
                      subgroupShuffleXor(Sf[r][c], s) for s = D_split/2, D_split/4,
                      ..., 1 — a butterfly reduction in log2(D_split) steps. Same
                      pattern for rowmaxf (line 548), Lf[r] (line 587), and Of[r][d]
                      (line 619). The OLD_AMD_WINDOWS workaround (lines 617-625)
                      shuffles vec4(Of[r][d]) instead of Of[r][d] (which is
                      FLOAT_TYPEV4 = f16vec4 when FLOAT16 is defined) because
                      subgroupShuffleXor on f16vec4 produces wrong results on AMD
                      RDNA2 Windows drivers. Cited issue: llama.cpp #19881.
Evidence:              flash_attn.comp:432-437 (Sf reduce), :547-549 (rowmaxf reduce),
                      :586-588 (Lf reduce), :617-625 (Of reduce with OLD_AMD_WINDOWS
                      branch), :621 (comment citing issue #19881).
Architectural Impact: No barrier for the intra-subgroup reduction — log2(D_split)
                      shuffle instructions vs log2(D_split) shmem rounds + barriers.
                      The workaround pays a conversion cost (f16vec4 → vec4 → f16vec4)
                      but only on buggy hardware.
Correctness Impact:   ULP-level non-determinism across vendors (subgroupShuffleXor
                      reduction order is implementation-defined). The workaround
                      produces identical arithmetic to the non-workaround path.
Optimization Type:    SIMD (subgroup shuffle butterfly).
GwenLand Target:      glvulkan
Recommendation:       ADOPT. Same subgroupShuffleXor butterfly for small reductions.
                      Include the OLD_AMD_WINDOWS workaround gated by a flag.
Priority:             High
Difficulty:           M
Dependencies:         ARTX20-F01
Confidence:           High
```

### Finding ARTX20-F05

```
Finding ID:           ARTX20-F05
Category:             SIMD_STRATEGY
Engine:               Vulkan
Component:            Cooperative matrix vs subgroup operations
Source File:          ggml/src/ggml-vulkan/vulkan-shaders/mul_mm.comp, mul_mm_cm2.comp,
                      flash_attn_cm1.comp, flash_attn_cm2.comp
Function:             main
Lines:                mul_mm.comp:17-25, 121-134, 172-190, 261-313; mul_mm_cm2.comp:11-23,
                      368, 381; flash_attn_cm1.comp:13-17, 296-530; flash_attn_cm2.comp:15-23
Summary:              Cooperative matrix (GL_KHR_cooperative_matrix / GL_NV_cooperative_matrix2)
                      is a separate extension family from subgroup ops. Coopmat1 uses
                      gl_ScopeSubgroup; coopmat2 uses gl_ScopeWorkgroup. The shaders
                      call coopMat* operations, not subgroup ops, for the matmul itself.
                      Subgroup ops are still used for cross-subgroup coordination (MoE
                      row-ids, mask-skip).
Observation:          mul_mm.comp enables GL_KHR_cooperative_matrix (line 18) and
                      GL_KHR_memory_scope_semantics (line 19) for the COOPMAT path,
                      plus GL_KHR_shader_subgroup_basic + ballot (lines 22-25) for
                      MUL_MAT_ID_USE_SUBGROUPS. The coopmat path (lines 261-313) uses
                      coopmat<..., gl_ScopeSubgroup, ...> and coopMatMulAdd — no
                      subgroup ops in the inner K-loop. Subgroup ops are used only in
                      the MoE row-id compaction (mul_mm_id_funcs.glsl). mul_mm_cm2.comp
                      uses gl_ScopeWorkgroup (line 368) — the matrix is distributed
                      across the whole workgroup; coopMatLoadTensorNV handles
                      cross-subgroup coordination internally. flash_attn_cm1.comp
                      uses subgroupMax/subgroupAdd (lines 359, 529) for the online
                      softmax reduction — this is a subgroup op *outside* the coopmat
                      matmul itself.
Evidence:              mul_mm.comp:17-25 (extensions), :261-264 (coopmat decl with
                      gl_ScopeSubgroup), :301-313 (coopMatMulAdd loop); mul_mm_cm2.comp:11-23
                      (extensions), :368 (gl_ScopeWorkgroup), :381 (coopMatLoadTensorNV
                      with decode callback); flash_attn_cm1.comp:13-17 (extensions),
                      :359 (subgroupMax), :529 (subgroupAdd).
Architectural Impact: Cooperative matrix is the Vulkan analogue of CUDA tensor cores /
                      Metal simdgroup matrix. The two flavors (KHR cross-vendor, NV
                      Hopper+) have different scope models — coopmat1 is per-subgroup,
                      coopmat2 is per-workgroup. The shader code doesn't mix coopmat
                      and subgroup ops in the same reduction; coopmat handles the
                      matmul, subgroup ops handle the surrounding coordination.
Correctness Impact:   None. Cooperative matrix operations are well-defined per the
                      extension spec.
Optimization Type:    Hardware matrix unit (cooperative matrix), SIMD (subgroup ops
                      for coordination).
GwenLand Target:      glvulkan
Recommendation:       ADOPT. Same separation: coopmat for matmul, subgroup ops for
                      coordination.
Priority:             High
Difficulty:           L
Dependencies:         ARTX19-F04, ARTX19-F06
Confidence:           High
```

### Finding ARTX20-F06

```
Finding ID:           ARTX20-F06
Category:             CORRECTNESS_SHORTCUT
Engine:               Vulkan
Component:            Hardcoded subgroup_size spec constant
Source File:          ggml/src/ggml-vulkan/vulkan-shaders/mul_mm_cm2.comp
Function:             (top-level declarations)
Lines:                115-116
Summary:              subgroup_size = 32 spec constant used to size ballots_sh[] shmem
                      array. If the actual subgroup size is 64, the array is half the
                      required size — ballot data would be lost.
Observation:          The shader declares `layout (constant_id = 6) const uint
                      subgroup_size = 32;` and uses it to size `shared uvec4
                      ballots_sh[BLOCK_SIZE / subgroup_size];`. This is used only for
                      sizing the shmem array — the actual subgroup operations use
                      gl_SubgroupSize (a runtime constant). If the host specializes
                      subgroup_size = 32 but the actual subgroup size is 64 (e.g.,
                      on some Intel Xe), ballots_sh[] has half the required slots —
                      the upper 32 lanes of each subgroup's ballot would overwrite
                      the next subgroup's slot, or be lost entirely. The host (ARTX18)
                      must specialize subgroup_size to match the actual
                      PipelineShaderStageRequiredSubgroupSizeCreateInfoEXT value.
Evidence:              mul_mm_cm2.comp:115 (subgroup_size = 32), :116 (ballots_sh
                      decl), :199-216 (ballot usage in load_row_ids).
Architectural Impact: Fragile dependency between shader-side shmem sizing and host-
                      side subgroup size selection. A host bug here would silently
                      corrupt MoE row-id compaction.
Correctness Impact:   If subgroup_size != actual subgroup size, ballots_sh[] is sized
                      wrong — MoE row-id compaction produces wrong results. Silent
                      data corruption.
Optimization Type:    None (this is a correctness risk, not an optimization).
GwenLand Target:      glvulkan
Recommendation:       REJECT. Use a sufficiently large fixed size (e.g., BLOCK_SIZE /
                      8 covers subgroup sizes up to 64) or declare the array per-lane
                      ([BLOCK_SIZE]) and accept the waste.
Priority:             Medium
Difficulty:           S
Dependencies:         ARTX20-F01
Confidence:           High
```

### Finding ARTX20-F07

```
Finding ID:           ARTX20-F07
Category:             SIMD_STRATEGY
Engine:               Vulkan
Component:            Mask-skip via subgroupAll
Source File:          ggml/src/ggml-vulkan/vulkan-shaders/flash_attn.comp
                      (also flash_attn_cm1.comp)
Function:             main
Lines:                flash_attn.comp:229-240; flash_attn_cm1.comp:211-220
Summary:              subgroupAll(pred) used to check if entire mask block is -inf;
                      result broadcast via shmem + barrier for cross-subgroup
                      coordination. Skip K-block load if all lanes return true.
Observation:          The mask block (Br × Bc) is loaded into masksh[] shmem. Each
                      lane computes max_mask = max(max_mask, mask_value) for its
                      portion. subgroupAll(max_mask <= NEG_FLT_MAX_OVER_2) returns
                      true iff all lanes see -inf. The result is broadcast via shmem
                      (lane 0 of each subgroup writes to tmpsh[gl_SubgroupID];
                      barrier; all lanes scan tmpsh[0..gl_NumSubgroups-1]). If
                      max_mask <= NEG_FLT_MAX_OVER_2 after the scan, the K-block is
                      skipped entirely (continue to next j).
Evidence:              flash_attn.comp:229 (subgroupAll), :231-237 (shmem broadcast),
                      :238-240 (skip decision); flash_attn_cm1.comp:211 (subgroupAll),
                      :213-220 (shmem broadcast).
Architectural Impact: Saves a K-block load + dequant + matmul when the mask block is
                      entirely -inf. For causal attention with sparse masks, this can
                      skip significant work.
Correctness Impact:   None. subgroupAll is well-defined.
Optimization Type:    SIMD (subgroup vote for predicate-all check), kernel fusion
                      (mask-skip fused into FA inner loop).
GwenLand Target:      glvulkan
Recommendation:       ADAPT. Keep subgroupAll; replace the shmem broadcast with
                      subgroupBroadcastFirst when gl_NumSubgroups == 1, falling back
                      to shmem only for multi-subgroup.
Priority:             Medium
Difficulty:           S
Dependencies:         ARTX20-F01
Confidence:           High
```

### Finding ARTX20-F08

```
Finding ID:           ARTX20-F08
Category:             SIMD_STRATEGY
Engine:               Vulkan
Component:            Inclusive/exclusive scan via subgroup ops
Source File:          ggml/src/ggml-vulkan/vulkan-shaders/cumsum.comp,
                      cumsum_multipass1.comp, topk_nary_search.comp
Function:             main
Lines:                cumsum.comp:54; cumsum_multipass1.comp:42;
                      topk_nary_search.comp:129
Summary:              subgroupExclusiveAdd and subgroupInclusiveAdd used for prefix-sum
                      scans. Per-subgroup scan in one instruction; cross-subgroup
                      coordination via shmem.
Observation:          cumsum.comp:54: `thread_sum = subgroupExclusiveAdd(thread_sum);`
                      — exclusive prefix sum within the subgroup. Each lane gets the
                      sum of all lanes before it. Then shmem coordination (lines 60-75)
                      carries the per-subgroup total forward to the next subgroup.
                      cumsum_multipass1.comp:42: `v = subgroupInclusiveAdd(v);` —
                      inclusive prefix sum (includes the current lane). Used in a
                      multi-pass cumsum where each pass covers a wider range.
                      topk_nary_search.comp:129: `partial_sum = subgroupInclusiveAdd(
                      partial_sum) + total;` — inclusive scan over bucket counts to
                      find the pivot bucket in n-ary search top-K.
Evidence:              cumsum.comp:54 (subgroupExclusiveAdd), :60-75 (cross-subgroup
                      shmem); cumsum_multipass1.comp:42 (subgroupInclusiveAdd);
                      topk_nary_search.comp:129 (subgroupInclusiveAdd for bucket scan).
Architectural Impact: One instruction vs O(log N) shmem rounds for the per-subgroup
                      scan. The cross-subgroup part still uses shmem + barrier, but
                      it's O(gl_NumSubgroups) instead of O(BLOCK_SIZE).
Correctness Impact:   None. Scan operations are well-defined.
Optimization Type:    SIMD (subgroup scan).
GwenLand Target:      glvulkan
Recommendation:       ADOPT. Same subgroupExclusiveAdd/subgroupInclusiveAdd for
                      prefix sums.
Priority:             Medium
Difficulty:           S
Dependencies:         ARTX20-F01
Confidence:           High
```

### Finding ARTX20-F09

```
Finding ID:           ARTX20-F09
Category:             MISSING_FEATURE
Engine:               Vulkan
Component:            Softmax / argmax / sum_rows lack subgroup ops
Source File:          ggml/src/ggml-vulkan/vulkan-shaders/soft_max.comp, argmax.comp,
                      sum_rows.comp
Function:             soft_max / main
Lines:                soft_max.comp:1-195; argmax.comp:1-60; sum_rows.comp:1-47
Summary:              These three kernels use pure shared-memory tree reduction with
                      no subgroup operations, despite the codebase having a clean
                      subgroup/shmem dual-path pattern (quantize_q8_1.comp) that could
                      be applied.
Observation:          soft_max.comp has no #extension GL_KHR_shader_subgroup_*
                      directives. The max reduction (lines 103-110) and sum reduction
                      (lines 143-150) both use the pattern: `vals[tid] = max_val;
                      barrier(); for (s = BLOCK_SIZE/2; s > 0; s >>= 1) { if (tid < s)
                      vals[tid] = max(vals[tid], vals[tid + s]); barrier(); }`. This
                      is 5 barrier rounds per reduction (for BLOCK_SIZE=32). A
                      subgroupMax/subgroupAdd path would be 1 instruction. argmax.comp
                      (lines 46-55) and sum_rows.comp (lines 36-42) have the same
                      pattern. Compare with quantize_q8_1.comp:77 which uses
                      subgroupClusteredMax(8) for the same kind of per-block max.
Evidence:              soft_max.comp:103-110 (max reduce), :143-150 (sum reduce);
                      argmax.comp:46-55 (max+argmax reduce); sum_rows.comp:36-42 (sum
                      reduce); quantize_q8_1.comp:77 (subgroupClusteredMax(8) for
                      comparison).
Architectural Impact: On subgroup-capable hardware (every modern Vulkan GPU), these
                      kernels are 5× slower than necessary. Softmax is rarely the
                      bottleneck, but argmax (used in sampling) and sum_rows (used in
                      reductions) can be hot paths.
Correctness Impact:   None. Shmem tree reduction is correct.
Optimization Type:    None (this is the absence of an optimization).
GwenLand Target:      glvulkan
Recommendation:       REJECT. Use subgroupMax/subgroupAdd (with shmem fallback) in
                      glvulkan's softmax/argmax/sum_rows.
Priority:             High
Difficulty:           S
Dependencies:         ARTX20-F01
Confidence:           High
```

### Finding ARTX20-F10

```
Finding ID:           ARTX20-F10
Category:             LAYOUT_SUBOPTIMAL
Engine:               Vulkan
Component:            Cross-subgroup sequential scan
Source File:          ggml/src/ggml-vulkan/vulkan-shaders/flash_attn.comp
Function:             main
Lines:                235-237, 558-560, 596-598, 634-636
Summary:              Cross-subgroup reduction uses sequential scan (for s = 0; s <
                      gl_NumSubgroups; ++s) instead of tree reduction. O(N) vs O(log N).
Observation:          The pattern is: `if (gl_SubgroupInvocationID == 0) tmpsh[
                      gl_SubgroupID] = val; barrier(); for (s = 0; s < gl_NumSubgroups;
                      ++s) val = max(val, tmpsh[s]);`. This sequential scan reads
                      gl_NumSubgroups shmem slots per lane. For workgroups with 8+
                      subgroups (e.g., BLOCK_SIZE=256, SubGroupSize=32 → 8 subgroups),
                      this is 8 sequential shmem reads. A tree reduction (`for (s =
                      NumSubgroups/2; s > 0; s >>= 1)`) would be log2(8) = 3 rounds.
                      Compare with add.comp:57-60 which uses a tree reduction across
                      subgroups.
Evidence:              flash_attn.comp:235-237 (mask broadcast), :558-560 (rowmaxf
                      cross-subgroup), :596-598 (Lf cross-subgroup), :634-636 (Of
                      cross-subgroup); add.comp:57-60 (tree reduction for comparison).
Architectural Impact: For small gl_NumSubgroups (1-4), the sequential scan is fine.
                      For 8+ subgroups, the tree reduction is measurably faster.
Correctness Impact:   None. Sequential scan is correct.
Optimization Type:    None (this is a suboptimal pattern).
GwenLand Target:      glvulkan
Recommendation:       ADAPT. Use tree reduction across subgroups for gl_NumSubgroups
                      ≥ 8. Sequential scan is fine for small gl_NumSubgroups.
Priority:             Medium
Difficulty:           S
Dependencies:         ARTX20-F01
Confidence:           Medium
```

### Finding ARTX20-F11

```
Finding ID:           ARTX20-F11
Category:             MISSING_FEATURE
Engine:               Vulkan
Component:            No subgroupBroadcastFirst usage
Source File:          (all shaders — grep returns no matches)
Function:             N/A
Lines:                N/A
Summary:              No shader uses subgroupBroadcastFirst, despite several places
                      where it would replace shmem+barrier for intra-subgroup broadcast.
Observation:          grep for subgroupBroadcastFirst returns no matches in the
                      shader directory. The mask-skip pattern in flash_attn.comp:229-240
                      uses `if (gl_SubgroupInvocationID == 0) tmpsh[gl_SubgroupID] =
                      all_less ? NEG : 0; barrier(); ...` — this is a cross-subgroup
                      broadcast (lane 0 of each subgroup → all lanes of all subgroups),
                      which requires shmem. But the *intra-subgroup* broadcast (lane 0
                      → all lanes of the same subgroup) could use
                      subgroupBroadcastFirst. Similarly, mul_mm_id_funcs.glsl:51-58
                      computes subgroup_base via shmem scan —
                      subgroupBroadcastFirst(subgroup_base) would let each lane get
                      its subgroup's base without shmem. subgroupBroadcastFirst is
                      core Vulkan 1.1 (no extension needed beyond
                      GL_KHR_shader_subgroup_basic).
Evidence:              (absence — no subgroupBroadcastFirst calls anywhere);
                      flash_attn.comp:229-240 (mask-skip pattern that could use it);
                      mul_mm_id_funcs.glsl:51-58 (subgroup_base computation that could
                      use it).
Architectural Impact: Missing one-instruction broadcasts; replacing them with shmem
                      + barrier costs ~5 cycles per broadcast. Small but additive.
Correctness Impact:   None.
Optimization Type:    None (this is the absence of an optimization).
GwenLand Target:      glvulkan
Recommendation:       ADOPT subgroupBroadcastFirst for intra-subgroup broadcasts in
                      glvulkan. Keep shmem only for cross-subgroup coordination.
Priority:             Medium
Difficulty:           S
Dependencies:         ARTX20-F01
Confidence:           High
```

### Finding ARTX20-F12

```
Finding ID:           ARTX20-F12
Category:             OTHER
Engine:               Vulkan
Component:            Subgroup shuffle for index broadcast in conv2d_mm
Source File:          ggml/src/ggml-vulkan/vulkan-shaders/conv2d_mm.comp
                      (also conv3d_mm.comp)
Function:             main
Lines:                conv2d_mm.comp:264-267, 318-321; conv3d_mm.comp:243-244
Summary:              subgroupShuffle used to broadcast computed indices (CRS_idx,
                      Cin_idx, KH_idx, KW_idx) from one lane to all lanes in the
                      subgroup, avoiding redundant computation.
Observation:          The conv-as-GEMM shader computes per-lane indices (CRS_idx,
                      Cin_idx, KH_idx, KW_idx) from a base index. With USE_COLLECTIVES
                      defined, one lane per subgroup computes the indices and
                      broadcasts them via subgroupShuffle(cached_idx, Ac) — Ac is the
                      destination lane. Without USE_COLLECTIVES, every lane recomputes
                      the indices from scratch (lines 275-281). The subgroup-shuffle
                      path trades 4 shuffle instructions for ~10 integer
                      multiplies/divides per lane — a win when the index computation
                      is expensive. Note: the GL_KHR_shader_subgroup_shuffle extension
                      directive is commented out at line 18 (`//#    extension
                      GL_KHR_shader_subgroup_shuffle : enable`) — it must be enabled
      by another mechanism (possibly implicit via GL_KHR_cooperative_matrix).
Evidence:              conv2d_mm.comp:264-267 (subgroupShuffle for index broadcast),
                      :318-321 (same for B-side), :18 (commented-out extension
                      directive); conv3d_mm.comp:243-244 (gl_SubgroupID for warp
                      indexing).
Architectural Impact: Saves redundant index computation across the subgroup. The
                      conv-as-GEMM pattern benefits because the index decomposition
                      (CRS → C, R, S) involves integer division, which is expensive
                      on GPUs.
Correctness Impact:   None. subgroupShuffle is well-defined.
Optimization Type:    SIMD (subgroup shuffle for index broadcast).
GwenLand Target:      glvulkan
Recommendation:       ADOPT. Same subgroupShuffle pattern for index broadcast where
                      index computation is expensive.
Priority:             Low
Difficulty:           S
Dependencies:         ARTX20-F01
Confidence:           Medium
```

---

## 18. Unknowns

* **U1**. Whether `subgroupBroadcastFirst` would actually be faster
  than the shmem+barrier pattern in `flash_attn.comp:229-240` on
  real hardware. The shmem path is 1 barrier + 1 shmem read;
  `subgroupBroadcastFirst` is 1 instruction. On most hardware the
  instruction wins, but the barrier may be hidden by other work.
  Requires per-vendor profiling.

* **U2**. Whether the `OLD_AMD_WINDOWS` workaround in
  `flash_attn.comp:617-625` is still needed on current AMD Windows
  drivers. The cited issue (#19881) suggests it was a driver bug;
  status of the fix is not visible from static analysis.

* **U3**. Whether the `subgroup_size = 32` spec constant in
  `mul_mm_cm2.comp:115` is always specialized correctly by the host.
  The host (ARTX18) queries `subgroup_min_size` and uses it for
  matmul pipelines — but the spec constant default (32) is a silent
  fallback if the host forgets to specialize. Static analysis can't
  determine if the host always specializes.

* **U4**. Whether the cross-subgroup sequential scan in
  `flash_attn.comp:235-237` is a measurable bottleneck. For
  workgroups of 128 threads (4 subgroups), the scan is 4 shmem
  reads — probably negligible. For 256 threads (8 subgroups), it's
  8 reads — possibly measurable. Requires profiling on the
  target workgroup size.

* **U5**. Whether the `gated_delta_net.comp` cluster-size switch
  (cases 2, 4, 8, 16, 32, 64) covers all realistic configurations.
  If `LANES_PER_COLUMN` is set to a value not in the switch (e.g.,
  128), the default case falls back to `subgroupAdd` — correct but
  slower. Whether any device would actually use `LANES_PER_COLUMN =
  128` is unclear.

* **U6**. Whether `subgroupElect` would be cleaner than
  `if (gl_SubgroupInvocationID == 0)` anywhere. `subgroupElect`
  returns true in exactly one lane; the `if (InvocationID == 0)`
  pattern is equivalent. Cosmetic; no performance difference.

* **U7**. Whether `VK_EXT_subgroup_size_control`'s
  `requiredSubgroupSize` is always honored by the driver. The
  Vulkan spec allows drivers to round to a nearby supported size;
  if the driver picks a different size than requested, the
  shader's `subgroup_size = 32` spec constant (W6/F06) would be
  wrong. Requires per-driver validation.

* **U8**. Whether the `subgroup_size` constant in `mul_mm_cm2.comp`
  is the *only* place where subgroup size is hardcoded for shmem
  sizing. A grep for `subgroup_size` returns only this one match
  in shmem sizing context, but other shaders may have implicit
  assumptions (e.g., `WARP = 32` in `mul_mm.comp:111` for warp
  indexing).

---

## 19. References

| Reference | File                                                | Function / Symbol                              | Lines         |
| --------- | --------------------------------------------------- | ---------------------------------------------- | ------------- |
| R01       | `vulkan-shaders/mul_mat_vec_base.glsl`              | `reduce_result` (3-way: NO_SHMEM / +SHMEM / SHMEM-only) | 5-8, 93-228 |
| R02       | `vulkan-shaders/mul_mat_vec_p021.comp`              | `main` (subgroupAdd reduce)                    | 6, 110, 117   |
| R03       | `vulkan-shaders/mul_mm.comp`                        | `main` (coopmat1, gl_SubgroupID)               | 17-25, 172-190, 261-313 |
| R04       | `vulkan-shaders/mul_mmq.comp`                       | `main` (MMQ + ballot for MUL_MAT_ID)           | 13-16         |
| R05       | `vulkan-shaders/mul_mm_cm2.comp`                    | `main` (coopmat2 + ballot MoE row-ids)         | 11-23, 115-116, 163-227, 368, 381 |
| R06       | `vulkan-shaders/mul_mm_id_funcs.glsl`               | `load_row_ids` (ballot compaction)             | 1-74          |
| R07       | `vulkan-shaders/flash_attn.comp`                    | `main` (clustered, shuffle, vote, built-ins)   | 15, 20-21, 126, 141, 229-240, 432-437, 547-549, 586-588, 617-625 |
| R08       | `vulkan-shaders/flash_attn_cm1.comp`                | `main` (coopmat1 FA, subgroupMax/Add/All)      | 13-17, 211-220, 359, 529 |
| R09       | `vulkan-shaders/flash_attn_cm2.comp`                | `main` (coopmat2 FA, ballot+vote extensions)   | 15-23         |
| R10       | `vulkan-shaders/flash_attn_mask_opt.comp`           | `main` (subgroupMin/Max block summary)         | 5, 68-72, 136-140 |
| R11       | `vulkan-shaders/quantize_q8_1.comp`                 | `quantize` (clustered Max/Add(8))              | 6-13, 65-107  |
| R12       | `vulkan-shaders/rms_norm_partials.comp`             | `main` (subgroupAdd, gl_SubgroupSize stride)   | 7-8, 43-46    |
| R13       | `vulkan-shaders/add.comp`                            | (subgroupAdd + shmem tree cross-subgroup)      | 5-6, 50-65    |
| R14       | `vulkan-shaders/multi_add.comp`                      | (same pattern as add.comp)                     | 7-8, 175-189  |
| R15       | `vulkan-shaders/ssm_scan.comp`                       | `main` (subgroupAdd state reduce)              | 4-6, 51-53, 99 |
| R16       | `vulkan-shaders/cumsum.comp`                         | `main` (subgroupExclusiveAdd)                  | 7-8, 54, 60   |
| R17       | `vulkan-shaders/cumsum_multipass1.comp`              | `main` (subgroupInclusiveAdd)                  | 7-8, 42, 46   |
| R18       | `vulkan-shaders/cumsum_multipass2.comp`              | `main` (subgroupAdd cross-subgroup)            | 7-8, 51-53    |
| R19       | `vulkan-shaders/topk_nary_search.comp`               | `main` (ballot + InclusiveAdd + FindLSB)       | 4-7, 129-130, 166-184, 188-212 |
| R20       | `vulkan-shaders/topk_moe.comp`                       | `main` (subgroupShuffleXor argmax)             | 4-6, 60, 77, 92, 169-171, 192 |
| R21       | `vulkan-shaders/fwht.comp`                           | `main` (subgroupShuffleXor butterfly)          | 5-6, 37-38, 78-85 |
| R22       | `vulkan-shaders/gated_delta_net.comp`                | `main` (clustered Add switch)                  | 4-9, 50, 65-92, 98-99 |
| R23       | `vulkan-shaders/conv2d_mm.comp`                      | `main` (subgroupShuffle index broadcast)       | 12, 18, 235-236, 258-267, 318-321 |
| R24       | `vulkan-shaders/conv3d_mm.comp`                      | `main` (gl_SubgroupID warp indexing)           | 12, 243-244   |
| R25       | `vulkan-shaders/soft_max.comp`                       | `soft_max` (no subgroup ops — shmem only)      | 1-195         |
| R26       | `vulkan-shaders/argmax.comp`                         | `main` (no subgroup ops — shmem only)          | 1-60          |
| R27       | `vulkan-shaders/sum_rows.comp`                       | `main` (no subgroup ops — shmem only)          | 1-47          |
| R28       | `vulkan-shaders/dot_product_funcs.glsl`              | `dot_product` (VALVE mixed-float dot)          | 1-14          |
| R29       | (host side, ARTX18 §9)                              | `VkPhysicalDeviceVulkan11Properties.subgroup*` | ggml-vulkan.cpp:6135 |
| R30       | (host side, ARTX18 §9)                              | `VK_EXT_subgroup_size_control` query           | ggml-vulkan.cpp:5973-5974, 6264-6273 |
| R31       | (host side, ARTX18 §9)                              | `PipelineShaderStageRequiredSubgroupSizeCreateInfoEXT` | ggml-vulkan.cpp (per-pipeline) |
| R32       | (host side, ARTX18 §9)                              | `eRequireFullSubgroupsEXT` flag                | ggml-vulkan.cpp (per-pipeline) |
