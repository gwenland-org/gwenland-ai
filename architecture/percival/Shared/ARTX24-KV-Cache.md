# ARTX24 — KV Cache Architecture and Cross-Backend Attention Contract

**Audited commit:** `555881ebc8b0fc0402b30e09258a32a7bfd13c52`
**Last updated:** 2026-07-25
**Auditor:** Percival-aux (ARTX24)
**Target GwenLand module:** `GATE` (graph construction and op contracts), `glproc` / `glcuda` / `glmetal` / `glvulkan` (all backends must implement the FA contract)

---

## 1. Executive Summary

The ggml KV cache and attention subsystem is a *contract layer* between the
model code in `llama.cpp/src/` (which builds the graph) and the per-backend
kernels (which execute `GGML_OP_FLASH_ATTN_EXT`). ggml itself does not
allocate or own the KV cache — `llama_kv_cache` does, in
`src/llama-kv-cache.cpp`. ggml provides only:

1. **The KV append primitive** — `GGML_OP_SET_ROWS`, a scattered-row write
   indexed by an I32/I64 vector. Used to push each new token's K and V
   rows into the cache.
2. **The attention op** — `GGML_OP_FLASH_ATTN_EXT`, a fused online-softmax
   QKV attention with optional mask, ALiBi bias, logit softcap, and
   attention sinks. This is the *only* attention op ggml ships — the
   older `GGML_OP_FLASH_ATTN` is gone, and `GGML_OP_FLASH_ATTN_BACK` is
   `GGML_ABORT`'d as "TODO: adapt to ggml_flash_attn_ext() changes".
3. **The position-encoding op** — `GGML_OP_ROPE`, with five modes
   (NEOX, NORMAL, MROPE, VISION, IMROPE) and 15 i32 op_params covering
   YaRN, longrope, and sectioned mrope.
4. **The linear-attention / SSM family** — `GGML_OP_SSM_CONV`,
   `GGML_OP_SSM_SCAN`, `GGML_OP_RWKV_WKV6`, `GGML_OP_RWKV_WKV7`,
   `GGML_OP_GATED_LINEAR_ATTN`, `GGML_OP_GATED_DELTA_NET`,
   `GGML_OP_LIGHTNING_INDEXER`, and `GGML_OP_DSV4_HC_{PRE,COMB,POST}`.
   These coexist with FA as separate ops; none are fused.

The FA op's parameter layout — `float scale, float max_bias, float
logit_softcap, int32 prec` packed into `op_params[0..3]`, with `src[0..4]`
= {Q, K, V, mask, sinks} — is the **cross-backend attention contract**.
Every backend that implements `FLASH_ATTN_EXT` (CPU, CUDA, Metal, Vulkan,
SYCL, CANN) must agree on this layout, on the semantics of `max_bias`
(ALiBi slope), `logit_softcap` (Gemma-style `tanh` capping), and the
optional sinks tensor. Backends *need not* agree on KV dtype support:
CUDA supports 8 dtypes (F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0),
Metal ships a subset via per-dtype template specializations, Vulkan uses
spec constants, and the CPU tiled path supports only F16/F32 KV while
the scalar path supports any `vec_dot_type`.

For GwenLand, the architectural decisions worth **ADOPT**ing are the FA
op param layout, the `SET_ROWS`-based KV append pattern, the online
softmax with f32 max-and-sum (with optional f16 accumulator on the
scalar CPU path), the ALiBi slope formula (identical to `soft_max_ext`
— a rare cross-op consistency win), the logit softcap formulation
(`scale /= softcap; s = softcap * tanh(s*scale)`), the attention sinks
contract (applied only on the first KV chunk), the MLA `V_is_K_view`
optimization, and the RoPE 15-slot op_params layout. The decisions worth
**REJECT**ing are the broken `FLASH_ATTN_BACK` (gradient path is dead
code), the hardcoded `GGML_FA_TILE_Q=64` / `GGML_FA_TILE_KV=64` (too
coarse for varied head dims), and the hand-tuned per-DKQ × per-CC CUDA
kernel selection matrix (an autotuner would be cleaner).

This document complements ARTX23 (backend dispatch) and ARTX22
(execution graph). ARTX22 covers how the scheduler routes ops to
backends; ARTX23 covers how backends register and discover each other;
this document covers the *op-level contract* for attention and KV cache
updates that all backends must honor.

---

## 2. Purpose

Provide a uniform, cross-backend op-level contract for:

* appending per-token K and V rows into a cache (SET_ROWS),
* computing scaled-dot-product attention with optional mask, ALiBi bias,
  logit softcap, and attention sinks (FLASH_ATTN_EXT),
* applying rotary position embeddings with YaRN/NEoX/MRoPE/Vision modes
  (ROPE),
* supporting MLA (Multi-head Latent Attention, DeepSeek) without a
  separate op (via the `V_is_K_view` convention),
* supporting linear-attention / SSM variants (Mamba, RWKV, RetNet,
  GatedDeltaNet) as first-class ops (not fused with FA).

It is **not** responsible for: KV cache memory management (that's
`llama_kv_cache`), KV cache slot allocation / eviction (that's
`llama_kv_cells`), sliding-window mask construction (that's the model
code), or per-backend kernel selection heuristics (those live in each
backend's `supports_op`).

---

## 3. Source Files

| File                                              | Lines    | Role                                                              |
| ------------------------------------------------- | -------- | ----------------------------------------------------------------- |
| `ggml/src/ggml.c`                                 | 8024     | Op construction: `ggml_flash_attn_ext`, `ggml_set_rows`, `ggml_rope_impl`, `ggml_ssm_conv`, `ggml_ssm_scan`, `ggml_gated_delta_net`, `ggml_flash_attn_back` |
| `ggml/include/ggml.h`                             | 2931     | Op enum (`GGML_OP_FLASH_ATTN_EXT`, `_BACK`, `_SSM_CONV`, `_SSM_SCAN`, `_RWKV_WKV6`, `_RWKV_WKV7`, `_GATED_LINEAR_ATTN`, `_GATED_DELTA_NET`, `_LIGHTNING_INDEXER`, `_DSV4_HC_*`), `GGML_ROPE_TYPE_*` constants |
| `ggml/src/ggml-cpu/common.h`                      | 95       | `GGML_FA_TILE_Q = 64`, `GGML_FA_TILE_KV = 64`; `ggml_fa_tile_config` template |
| `ggml/src/ggml-cpu/ops.cpp`                       | 12005    | CPU FA: `_f16_one_chunk` (scalar, online softmax), `_tiled` (simd_gemm), `_reduce_partials` (split-KV decode), dispatch at `ggml_compute_forward_flash_attn_ext` |
| `ggml/src/ggml-cuda/fattn.cu`                     | 590      | CUDA FA dispatch: `ggml_cuda_get_best_fattn_kernel` (per-DKQ × per-CC if-else), `ggml_cuda_flash_attn_ext` entry |
| `ggml/src/ggml-cuda/fattn-common.cuh`             | large    | `V_is_K_view` detection, mask/sinks/scale/max_bias/logit_softcap plumbing |
| `ggml/src/ggml-cuda/fattn-mma-f16.cuh`            | large    | MMA f16 kernel; `constexpr bool V_is_K_view = DKQ == 576` |
| `ggml/src/ggml-metal/ggml-metal.metal`            | 11219    | `kernel_flash_attn_ext` template + per-(DK, DV, dtype) specializations |
| `ggml/src/ggml-vulkan/vulkan-shaders/flash_attn.comp` | 759  | Vulkan FA; spec constants `FaTypeK`, `FaTypeV`, `Br`, `Bc`, `HSK`, `HSV` |
| `ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_base.glsl` | 266 | Push-constant struct (scale, max_bias, logit_softcap, m0, m1, mask_n_head_log2); binding layout (Q/K/V/M/S/O) |
| `src/llama-kv-cache.cpp`                           | 2642     | KV cache tensor allocation: `ggml_new_tensor_3d(ctx, type_k, n_embd_k_gqa, kv_size, n_stream)`; `cpy_k`, `cpy_v` use `ggml_set_rows` |
| `src/llama-hparams.cpp`                            | 287      | `is_mla()` (K-only cache when MLA), `n_embd_head_k_mla`, `n_embd_head_v_mla` |

---

## 4. Architecture Overview

```
                ┌──────────────────────────────────────────────────────────┐
                │  llama_kv_cache (src/llama-kv-cache.cpp)                 │
                │  ├─ per-layer: k = ggml_new_tensor_3d(type_k,            │
                │  │                n_embd_k_gqa, kv_size, n_stream)        │
                │  │              v = ggml_new_tensor_3d(type_v,            │
                │  │                n_embd_v_gqa, kv_size, n_stream)        │
                │  │   shape: [head_dim * n_kv_heads, seq_len, n_stream]    │
                │  ├─ MLA: has_v = false (K-only cache; V is a view of K)   │
                │  └─ cpy_k / cpy_v: build ggml_set_rows(k, k_cur, k_idxs)  │
                └──────────────────────────────────────────────────────────┘
                                       │
                                       ▼
                ┌──────────────────────────────────────────────────────────┐
                │  GGML_OP_SET_ROWS  (ggml.c:3937)                         │
                │  result = view(a)                                       │
                │  src[0] = b (new rows, F32 or F16)                      │
                │  src[1] = c (i32 or i64 row indices)                    │
                │  src[2] = a (the cache, view tensor)                    │
                │  CPU impl: ops.cpp:5092, writes b rows at c[i] in a     │
                └──────────────────────────────────────────────────────────┘
                                       │
                                       ▼
                ┌──────────────────────────────────────────────────────────┐
                │  GGML_OP_ROPE  (ggml.c:4168)                            │
                │  src[0] = a (Q or K, any float dtype)                   │
                │  src[1] = b (i32 position ids)                          │
                │  src[2] = c (optional f32 freq factors, longrope)       │
                │  op_params[0..14]: n_past, n_dims, mode, n_ctx,         │
                │                   n_ctx_orig, freq_base, freq_scale,    │
                │                   ext_factor, attn_factor, beta_fast,   │
                │                   beta_slow, sections[4]                │
                │  modes: NORMAL(0), NEOX(2), MROPE(8), VISION(24),       │
                │         IMROPE(40)                                       │
                └──────────────────────────────────────────────────────────┘
                                       │
                                       ▼
                ┌──────────────────────────────────────────────────────────┐
                │  GGML_OP_FLASH_ATTN_EXT  (ggml.c:5402)                  │
                │  src[0] = q  [D, N, n_q_heads,  batch]                  │
                │  src[1] = k  [D, M, n_kv_heads, batch]                  │
                │  src[2] = v  [Dv, M, n_kv_heads, batch]                 │
                │  src[3] = mask (F16, contiguous, [M, N, 1, batch] or bcast)│
                │  src[4] = sinks (F32, [n_q_heads], optional)            │
                │  op_params[0] = scale (f32)                              │
                │  op_params[1] = max_bias (f32, 0 = no ALiBi)            │
                │  op_params[2] = logit_softcap (f32, 0 = no cap)         │
                │  op_params[3] = prec (i32, GGML_PREC_*)                 │
                │  result: F32 [Dv, n_q_heads, N, batch]  (permute 0,2,1,3)│
                └──────────────────────────────────────────────────────────┘
                                       │
                                       ▼
        ┌────────────────────┬───────────────────┬───────────────────┐
        ▼                    ▼                   ▼                   ▼
   ┌─────────┐         ┌──────────┐         ┌──────────┐       ┌──────────┐
   │  CPU    │         │  CUDA    │         │  Metal   │       │  Vulkan  │
   │ ops.cpp │         │ fattn.cu │         │  .metal  │       │  .comp   │
   │ 3 paths │         │ 4 kerns  │         │ templated│       │ spec-const│
   │         │         │ per-CC   │         │ per-DK,DV│       │ per-DK,DV│
   └─────────┘         └──────────┘         └──────────┘       └──────────┘
```

Key design points:

* **KV cache shape is `[head_dim * n_kv_heads, seq_len, n_stream]`** — a
  3D tensor allocated by `llama_kv_cache`. The first dim packs all KV
  heads together (GQA-aware); the second dim is the sequence axis; the
  third is the streaming axis (for parallel decoding). See Finding F02.
* **KV append is `SET_ROWS`** — a scattered-row write indexed by an I32
  or I64 vector. The cache is the destination; `k_cur` (the new token's
  K) is the source; `k_idxs` (where to write each row) is the index.
  See Finding F03.
* **FA op params are a fixed 4-slot packed array** — `scale`, `max_bias`,
  `logit_softcap`, `prec`. Sinks are *not* in op_params; they are an
  optional 5th source tensor. See Finding F01.
* **Mask is F16, contiguous, additive** — values are added to the QK
  score after scaling; `-INFINITY` masks out a position. ALiBi slopes
  are applied as `slope * mask[ic]` inside the kernel. See Finding F08.
* **MLA is not a separate op** — when `hparams.is_mla()`, `llama_kv_cache`
  allocates only K (no V); the model code makes V a view of K; CUDA's FA
  kernel detects this via `V->view_src == K` and reuses the K tile in
  shared memory. See Finding F10.
* **Linear-attention / SSM ops coexist with FA** — Mamba's `SSM_CONV` /
  `SSM_SCAN`, RWKV's `WKV6` / `WKV7`, RetNet's `GATED_LINEAR_ATTN`,
  GatedDeltaNet's `GATED_DELTA_NET`, and DeepSeek-V4's
  `DSV4_HC_{PRE,COMB,POST}` are all separate ops. None are fused with FA.

---

## 5. Execution Flow

### 5.1 KV cache allocation

`llama_kv_cache::llama_kv_cache` (`src/llama-kv-cache.cpp:160-248`) loops
over every layer. For each layer:

```
n_embd_k_gqa = hparams.n_embd_k_gqa(il)   // = n_embd_head_k * n_kv_heads
n_embd_v_gqa = v_trans ? n_embd_v_gqa_max : hparams.n_embd_v_gqa(il)
k = ggml_new_tensor_3d(ctx, type_k, n_embd_k_gqa, kv_size, n_stream)
if (!is_mla) {
    v = ggml_new_tensor_3d(ctx, type_v, n_embd_v_gqa, kv_size, n_stream)
} else {
    v = nullptr   // MLA: V is a view of K, allocated elsewhere
}
```

The cache shape is `[head_dim * n_kv_heads, kv_size, n_stream]`. The
innermost dim packs all KV heads together so that a single `SET_ROWS`
call can update every head of one token in one shot. `n_stream` is the
number of parallel decode streams (1 for unified cache, `n_seq_max` for
split-stream cache).

### 5.2 Per-token KV append

`llama_kv_cache::cpy_k` (`src/llama-kv-cache.cpp:1276-1328`):

```
k_cur = view_2d(k_cur, n_embd_gqa, n_tokens, k_cur->nb[2], 0)
if (n_stream > 1) {
    k = reshape_2d(k, n_embd_gqa, kv_size * n_stream)   // merge streams
}
return ggml_set_rows(ctx, k, k_cur, k_idxs)
```

`k_idxs` is an I64 1D tensor of length `n_tokens` holding the cache slot
each token should write to. The same pattern applies for `cpy_v` (with
the `v_trans` branch handling the transposed-V layout that non-FA paths
use).

### 5.3 Attention graph construction (model side)

A typical attention block (simplified, from any transformer model in
`src/models/`):

```
q = ggml_mul_mat(ctx, wq, x)              // [D, N, n_q_heads, batch]
k = ggml_mul_mat(ctx, wk, x)              // [D, M, n_kv_heads, batch]
v = ggml_mul_mat(ctx, wv, x)              // [Dv, M, n_kv_heads, batch]

q = ggml_rope_ext(ctx, q, pos_ids, NULL, n_dims, mode, ...)
k = ggml_rope_ext(ctx, k, pos_ids, NULL, n_dims, mode, ...)

k = cpy_k(ctx, k, k_idxs, ...)            // SET_ROWS into cache
v = cpy_v(ctx, v, v_idxs, ...)

mask = build_input_kq_mask(...)           // F16, contiguous, [M, N, 1, batch]

attn = ggml_flash_attn_ext(ctx, q, k_cache, v_cache, mask,
                           scale, max_bias, logit_softcap)
ggml_flash_attn_ext_set_prec(attn, GGML_PREC_DEFAULT)
if (use_sinks) {
    ggml_flash_attn_ext_add_sinks(attn, sinks_tensor)
}
```

The model code is responsible for: building the mask (including causal
and sliding-window), computing the ALiBi `max_bias` (passed as a scalar
— the kernel derives slopes), and supplying sinks (one F32 per query
head).

### 5.4 FA execution (CPU)

`ggml_compute_forward_flash_attn_ext` (`ops.cpp:9202-9217`) dispatches on
`op_params[3]` (prec). For `GGML_PREC_DEFAULT` / `GGML_PREC_F32`, it
calls `ggml_compute_forward_flash_attn_ext_f16` (`ops.cpp:9066-9200`),
which picks one of three paths:

1. **Split-KV decode path** (`use_split_kv_path`, line 9115): when
   `neq1 == 1 && neq3 == 1` (single-token decode), `kv` is F32 or F16,
   `k->type == v->type`, `q->type == F32`, and `nek1 >= 512`. Splits the
   KV dimension across threads; each thread computes partial `(M, S,
   VKQ)` for its KV chunk; a barrier; then `ggml_flash_attn_ext_reduce_partials`
   combines partials across chunks.
2. **Tiled prefill path** (`use_tiled`, line 9172): when `q->type == F32`,
   `kv_is_f32_or_f16`, `k->type == v->type`, `neq1 >= Q_TILE_SZ (64)`,
   and `DV % f32_epr == 0`. Packs Q into a `Q_TILE_SZ × DK` tile, K into
   a `KV_TILE_SZ × DK` tile (transposed), uses `simd_gemm` for both the
   QK and the softmax-V products.
3. **Scalar fallback** (`_f16_one_chunk`, line 9194): one query row at a
   time, online softmax with a single accumulator (M, S, VKQ), `vec_dot`
   for the QK product, `vec_mad` for the V accumulation. Supports any
   `vec_dot_type` (including quantized K).

### 5.5 FA execution (CUDA)

`ggml_cuda_flash_attn_ext` (`fattn.cu:570-585`) calls
`ggml_cuda_get_best_fattn_kernel(device, dst)` (line 545), which returns
one of `BEST_FATTN_KERNEL_NONE / _TILE / _VEC / _MMA_F16`. The dispatch
is a hand-tuned if-else chain over `Q->ne[0]` (DKQ), `Q->ne[1]`
(n_tokens), `gqa_ratio`, `cc` (compute capability), and `K->type`. See
Finding F11.

### 5.6 FA execution (Metal / Vulkan)

Metal ships a single `kernel_flash_attn_ext` template specialized per
`(DK, DV, K_type, V_type)` combination (line 7066+). The host code
picks the specialization by name. Vulkan uses spec constants
(`FaTypeK`, `FaTypeV`, `Br`, `Bc`, `HSK`, `HSV`) compiled into multiple
shader variants.

---

## 6. Data Layout

### 6.1 KV cache tensor

```c
k = ggml_new_tensor_3d(ctx, type_k, n_embd_k_gqa, kv_size, n_stream)
//                ne[0] = head_dim * n_kv_heads   (packed GQA)
//                ne[1] = kv_size                 (sequence axis)
//                ne[2] = n_stream                (parallel decode streams)
```

The innermost dim (`ne[0]`) packs all KV heads together so that one
row of the cache is one token's full K (or V) vector across all heads.
This layout lets `SET_ROWS` write a full token's K in one call.

### 6.2 FA source tensors

| src | name  | type            | shape                                  | constraints                          |
| --- | ----- | --------------- | -------------------------------------- | ------------------------------------ |
| 0   | q     | F32 (CPU/CUDA), F16/BF16 (CUDA) | `[D, N, n_q_heads, batch]` | `ne[0] == D`, `ne[3] == k->ne[3]` |
| 1   | k     | F16/F32/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/BF16 | `[D, M, n_kv_heads, batch]` | `ne[0] == q->ne[0]` (D matches)   |
| 2   | v     | F16/F32/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/BF16 | `[Dv, M, n_kv_heads, batch]` | `ne[1] == k->ne[1]` (M matches)   |
| 3   | mask  | F16 (mandatory) | `[M, N, 1, batch]` or bcast            | contiguous; `q->ne[2] % mask->ne[2] == 0` |
| 4   | sinks | F32 (optional)  | `[n_q_heads]`                          | added via `ggml_flash_attn_ext_add_sinks` |

### 6.3 FA op_params

```c
float    scale;          // op_params[0]  (e.g., 1/sqrt(D))
float    max_bias;       // op_params[1]  (0 = no ALiBi)
float    logit_softcap;  // op_params[2]  (0 = no cap; e.g., 50 for Gemma)
int32_t  prec;           // op_params[3]  (GGML_PREC_DEFAULT / _F32 / _F16)
```

Set by `ggml_flash_attn_ext` (ggml.c:5434-5435) and modified by
`ggml_flash_attn_ext_set_prec` (line 5447-5455). The 4-slot layout is
the **cross-backend contract** — every backend reads the same memory.

### 6.4 FA result

```c
// permute(0, 2, 1, 3) of [Dv, n_q_heads, N, batch]
int64_t ne[4] = { v->ne[0], q->ne[2], q->ne[1], q->ne[3] };
result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);
```

The result is always F32. The permutation `(0, 2, 1, 3)` puts the head
axis second and the query-token axis third, which is the layout the
output projection (`wO · attn`) expects.

### 6.5 Mask layout

The mask is F16, contiguous, with shape `[M, N, 1, batch]` (or
broadcastable). Values are **additive**: the kernel does
`KQ[i,j] += mask[i,j]` (after scaling), so a `-INFINITY` mask entry
zeros out that position in softmax. ALiBi biases are baked into the
mask on the model side (multiplied by the per-head slope — see Finding
F08). Sliding window is implemented by setting out-of-window mask
entries to `-INFINITY` in `set_input_kq_mask` (`llama-kv-cache.cpp:1531`).

---

## 7. Memory Layout

### 7.1 KV cache memory

The cache is allocated by `ggml_backend_alloc_ctx_tensors_from_buft`
(`llama-kv-cache.cpp:283`) on the device buffer type of the layer's
assigned backend. Each layer has its own K and V tensors; they are not
pooled. For multi-GPU inference, each layer's cache lives on its
assigned GPU's memory; cross-device attention requires a copy (handled
by the scheduler, ARTX22).

### 7.2 CPU FA per-thread scratch

The CPU FA kernel uses `params->wdata` for per-thread scratch. Layout
(from `ops.cpp:8564-8567` for the scalar path, `8815-8822` for the
tiled path):

```
per-thread scratch:
  scalar: [VKQ32: DV floats][V32: DV floats][VKQ16: DV f16][Q_q: DK f16]
          + CACHE_LINE_SIZE_F32 padding
  tiled:  [Q_q: Q_TILE_SZ * DK][KQ: Q_TILE_SZ * KV_TILE_SZ]
          [mask32: Q_TILE_SZ * KV_TILE_SZ][VKQ32: Q_TILE_SZ * DV]
          [V32: KV_TILE_SZ * DV][K_f32: KV_TILE_SZ * DK]
          + CACHE_LINE_SIZE_F32 padding
```

The split-KV decode path adds a partials buffer after all threads'
scratch:
```
partials: [n_q_heads][n_chunks][2 + DV]   // M, S, VKQ per chunk per head
```

### 7.3 CUDA FA workspace

`ggml_cuda_flash_attn_ext_get_alloc_size` (`fattn.cu:536-568`) computes
the extra allocation needed beyond `dst->data`. The `f16_extra` struct
holds intermediate F16 conversions of K and V (when the kernel needs
them) and fixup buffers for split-K reduction. The exact layout depends
on the selected kernel (`TILE` / `VEC` / `MMA_F16`).

---

## 8. Parallelism Strategy

### 8.1 CPU

* **Scalar path**: each thread takes a disjoint range of query rows
  (`ir0..ir1`); no cross-thread coordination within a chunk.
* **Tiled path**: same — each thread takes a disjoint range of Q tiles.
  Within a tile, the work is sequential (one thread does the whole
  `simd_gemm` for that tile).
* **Split-KV decode path**: threads *cooperate* on a single query row.
  Each thread takes a disjoint range of KV positions; writes `(M, S,
  VKQ)` partials to a shared buffer; barrier; then one thread per query
  head reduces the partials. This is the only CPU FA path with
  cross-thread coordination.

### 8.2 CUDA

Each CUDA FA kernel is a 2D grid: one block per `(query_head, batch)`
pair (or per `(query_head_tile, batch)` for the MMA kernel). Within a
block, warps cooperate on the KV dimension. The GQA optimization
collapses `gqa_ratio` query heads into one block when the ratio is
favorable.

### 8.3 Metal / Vulkan

Metal dispatches one threadgroup per `(query_head, batch)`. The
`FC_flash_attn_ext_nsg` switch (metal:7007-7013) picks the number of
simdgroups per threadgroup (4 or 8). Vulkan uses a 3D global size with
`row_split` and `D_split` constants to partition work within a
workgroup.

---

## 9. SIMD / GPU Strategy

### 9.1 CPU SIMD

The tiled CPU path uses `simd_gemm` (declared in `simd-gemm.h`) for
both the QK matmul and the softmax-V matmul. `simd_gemm` is a
hand-tuned per-ISA GEMM (audited in ARTX02-05). The scalar path uses
`vec_dot` and `vec_mad` from the type-traits table (ARTX01-F03).

### 9.2 CUDA tensor cores

The MMA_F16 kernel uses `mma.sync` (Turing+) or `wmma` (Volta+) for
both QK and softmax-V. K and V are pre-converted to F16 in workspace
if they arrived as F32 or quantized. The `V_is_K_view` template
parameter (Finding F10) lets the kernel skip the V load entirely when
V is a view of K.

### 9.3 Metal simdgroup matrix multiply

Metal uses `simdgroup_matrix` (8x8 half/float) for both matmuls. The
template parameters pick the dtype of K and V (F16, F32, BF16, Q4_0,
Q8_0, Q4_1, Q5_0, Q5_1) and the head dims (DK, DV).

### 9.4 Vulkan coopmat

Vulkan uses `GL_EXT_cooperative_matrix` (coopmat1) or
`GL_KHR_cooperative_matrix` (coopmat2) when available. Spec constants
`FaTypeK` / `FaTypeV` select the K/V dtype; the shader includes
`flash_attn_dequant.glsl` to handle quantized types.

---

## 10. Quantization Strategy

The FA op supports quantized KV cache. The set of supported dtypes
varies per backend:

| Backend | Supported K/V dtypes for FA                                                    | Source                              |
| ------- | ----------------------------------------------------------------------------- | ----------------------------------- |
| CPU     | F16, F32 (tiled path); any `vec_dot_type` (scalar path)                       | ops.cpp:8749 (`k->type == v->type`) |
| CUDA    | F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 (Q4_1/Q5_0/Q5_1 gated by `GGML_CUDA_FA_ALL_QUANTS`) | fattn.cu:338-355 |
| Metal   | F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 (per-template specializations)   | metal:7164+                         |
| Vulkan  | F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q1_0 (per spec constant)        | flash_attn_base.glsl:93-101         |

K and V may have different dtypes on CUDA (with `GGML_CUDA_FA_ALL_QUANTS`)
and Vulkan; CPU tiled path requires `k->type == v->type`; Metal requires
the same (one template parameter for both). The Q dtype is always F32 on
CPU, F16/BF16/F32 on CUDA, F16/F32/BF16 on Metal, F16/F32 on Vulkan.

FP8 KV is **not** supported in FA at this commit. The `IQ*` and `K*`
quants are not supported in FA (only in `MUL_MAT`).

---

## 11. Correctness Analysis

### 11.1 Online softmax reassociation

The online softmax algorithm (from the FlashAttention paper, cited at
`ops.cpp:8590`) maintains a running max `M` and sum `S` and rescales
`VKQ` when a new max is found. The order in which KV positions are
processed changes the intermediate rescaling factors and thus the ULP-
level result. The split-KV decode path explicitly reduces partials in
chunk order (`ops.cpp:9034-9054`); the tiled path processes KV in tile
order; the scalar path processes KV sequentially. All three produce
bit-identical results *for the same path*, but **switching paths
changes the ULPs**.

### 11.2 f16 vs f32 VKQ accumulator

The scalar path uses an F16 VKQ accumulator when `v->type == F16`
(`ops.cpp:8618-8632`), saving memory bandwidth and accumulator storage.
The tiled path always uses F32 VKQ (`ops.cpp:8824`). The two paths
produce slightly different results for F16 V — the scalar path's
accumulator is F16 (11 bits), the tiled path's is F32 (24 bits). This
is a per-path determinism leak. See Finding F06.

### 11.3 ALiBi slope derivation

The slope formula (`ops.cpp:8539-8540`):
```
n_head_log2 = 1 << floor(log2(n_head))
m0 = powf(2.0, -(max_bias       ) / n_head_log2)
m1 = powf(2.0, -(max_bias / 2.0) / n_head_log2)
slope(h) = h < n_head_log2 ? powf(m0, h + 1) : powf(m1, 2*(h - n_head_log2) + 1)
```
is identical between `flash_attn_ext` and `soft_max_ext` (the latter
is audited in the CPU soft_max code, same formula at `ops.cpp:5455+`).
This is a deliberate cross-op consistency. See Finding F07.

### 11.4 Logit softcap

When `logit_softcap != 0` (`ops.cpp:8532-8534, 8605-8607`):
```
scale /= logit_softcap           // pre-divide scale
s = s * scale                    // apply scale
s = logit_softcap * tanhf(s)     // cap
```
This is the Gemma-style logit softcap. The pre-division keeps the
scaled score in a reasonable range for `tanhf`. Cross-backend
consistency: CUDA and Metal apply the same formula.

### 11.5 Attention sinks

`ggml_flash_attn_ext_add_sinks` (`ggml.c:5466-5480`) sets `src[4]` to
a 1D F32 tensor of length `n_q_heads`. The CPU kernel applies sinks
**only on the first KV chunk** (`ops.cpp:8666` `if (sinks && ic_start == 0)`).
This is a contract: sinks represent an extra "always-attended" virtual
position with logit `sinks[h]`. The first-chunk-only rule means the
sink contributes to the initial softmax max; later chunks see the
sink's effect via the rescaled `M`. If a backend applied sinks on
every chunk, the result would be wrong (the sink would be counted N
times). See Finding F09.

### 11.6 MLA V_is_K_view correctness

When `V_is_K_view` is true (CUDA, `fattn-mma-f16.cuh:1915`), the
kernel reads V from the same shared-memory tile as K. This is correct
**only** because DeepSeek-MLA's V is literally a view of K (same data,
same strides). The detection (`fattn-common.cuh:63`) is:
```c
V_is_K_view = V->view_src && (V->view_src == K ||
             (V->view_src == K->view_src && V->view_offs == K->view_offs))
```
If a future model makes V a *different* view of the same buffer (same
`view_src`, different `view_offs`), the detection returns false and the
kernel does a separate V load. Correct.

### 11.7 Broken FLASH_ATTN_BACK

`ggml_flash_attn_back` (`ggml.c:5484-5551`) begins with
`GGML_ABORT("TODO: adapt to ggml_flash_attn_ext() changes")`. The
function body still references the old `FLASH_ATTN` semantics (a single
`masked` boolean instead of a mask tensor). Any caller that builds a
backward attention graph will abort. See Finding F04.

---

## 12. Optimization Analysis

### 12.1 Identified optimizations

| Optimization                          | Where                                  | Notes                                                                  |
| ------------------------------------- | -------------------------------------- | ---------------------------------------------------------------------- |
| Online softmax (FlashAttention)       | ops.cpp:8588-8657                      | O(N) memory instead of O(N²) for the KQ matrix.                       |
| Split-KV decode                       | ops.cpp:9115-9145                      | Single-token decode parallelizes across the KV dimension.             |
| Tiled prefill                         | ops.cpp:8706-8992                      | Q tile × KV tile, `simd_gemm` for both matmuls.                       |
| F16 VKQ accumulator (scalar path)     | ops.cpp:8618-8632                      | Halves accumulator memory and bandwidth when V is F16.                |
| Mask tile skip                        | ops.cpp:8853-8872                      | Skip KV tile if all mask entries are -INF.                             |
| ALiBi slope precomputation            | ops.cpp:8536-8540                      | `m0`, `m1` computed once per kernel, `slope(h)` per head.             |
| Logit softcap fused into FA           | ops.cpp:8605-8607                      | No separate `tanh` op.                                                |
| Sinks fused into first-chunk softmax  | ops.cpp:8666-8681                      | No separate sink op.                                                  |
| MLA V_is_K_view (CUDA)                | fattn-mma-f16.cuh:1915                 | Skip V global load; read V from K tile in smem.                       |
| GQA ratio collapsing (CUDA)           | fattn.cu:91-104                        | Collapse `gqa_ratio` query heads into one block when favorable.       |
| Per-CC kernel selection (CUDA)        | fattn.cu:460-533                       | Pick MMA / tile / vec based on compute capability.                    |
| Quantized KV (all backends)           | fattn.cu:338-355; metal:7164+          | Q4_0/Q8_0 KV reduces cache memory 4-8×.                               |

### 12.2 Optimizations *not* present (worth noting)

* **No persistent kernels.** Each FA launch is a separate kernel; no
  persistent thread blocks across multiple FA ops.
* **No cross-op fusion.** FA is not fused with the preceding `ROPE` or
  the following output projection. Each is a separate graph node.
* **No FP8 KV.** FP8 (E4M3, E5M2) is supported in `MUL_MAT` (CUDA) but
  not in FA at this commit.
* **No autotuner.** CUDA's per-DKQ × per-CC kernel selection is hand-
  tuned in source. Adding a new DKQ (e.g., 576 for DeepSeek) requires
  adding a new `case` in `ggml_cuda_flash_attn_ext_mma_f16` and a new
  set of template instantiations.
* **No KV cache compression beyond quantization.** No PagedAttention,
  no H2O sparsification, no streaming-LLM eviction at the op level
  (these are handled by `llama_kv_cache` slot management).

---

## 13. Architectural Strengths

1. **FA op param layout is a clean cross-backend contract.** 4 slots
   (scale, max_bias, logit_softcap, prec) + 5 optional sources. Every
   backend reads the same memory. Adding a new parameter (e.g., a
   future sliding-window scalar) would require bumping the layout and
   updating every backend, but the current set covers all current
   models.

2. **`SET_ROWS` is the right KV append primitive.** A scattered-row
   write indexed by I32/I64 is exactly what KV cache update needs: the
   cache slot for each token is determined by the slot allocator (not
   by a sequential offset), so a gather-write is necessary. Using one
   op for both K and V (and for any other per-token row update, like
   SSM state) is clean.

3. **Online softmax with optional F16 accumulator.** The scalar path's
   F16 VKQ accumulator halves bandwidth when V is F16; the tiled path's
   F32 accumulator preserves precision for the GEMM-based reduction.
   The choice is per-path, not per-config — sensible.

4. **ALiBi slope formula is identical between FA and soft_max.** This
   is a rare cross-op consistency win. A model that uses ALiBi with
   the legacy `KQ + soft_max_ext` path gets the same slopes as one that
   uses FA. Switching paths does not change ALiBi behavior.

5. **MLA via `V_is_K_view`, not a separate op.** DeepSeek's MLA is
   handled by the existing FA op with one extra template parameter.
   No new op, no new graph node, no new contract. The model code just
   makes V a view of K and the backend detects it.

6. **Linear-attention / SSM ops coexist as first-class ops.** Mamba,
   RWKV, RetNet, GatedDeltaNet each get their own op (`SSM_CONV`,
   `SSM_SCAN`, `WKV6`, `WKV7`, `GATED_LINEAR_ATTN`,
   `GATED_DELTA_NET`). This is cleaner than overloading FA.

7. **RoPE packs 15 parameters into one op.** All of NEOX, NORMAL,
   MROPE, VISION, IMROPE, YaRN, longrope, and sectioned mrope share
   one op via a 15-slot op_params array. No rope-neox, rope-vision,
   rope-yarn, rope-mrope op explosion.

---

## 14. Architectural Weaknesses

### W1 — `GGML_OP_FLASH_ATTN_BACK` is broken

**Evidence:** `ggml.c:5491` `GGML_ABORT("TODO: adapt to ggml_flash_attn_ext() changes")`.

**Impact:** Training or any gradient-based attention path through FA
is impossible. Anyone building a training graph that includes FA will
abort at graph construction. The function body below the abort is dead
code referencing the old `FLASH_ATTN` semantics.

### W2 — Tile constants hardcoded

**Evidence:** `ggml/src/ggml-cpu/common.h:9-10`:
```c
#define GGML_FA_TILE_Q  64
#define GGML_FA_TILE_KV 64
```

**Impact:** The CPU tiled path requires `neq1 >= 64` and `DV %
f32_epr == 0`. For models with `head_dim` not divisible by the SIMD
width (e.g., 40 on some phi models, or 72 on Gemma), the tiled path
falls back to scalar, losing 5-10× throughput. A per-DK tile size
would help.

### W3 — Three CPU FA paths with implicit selection

**Evidence:** `ops.cpp:9115` (split-KV), `9172` (tiled), `9194` (scalar).
Selection is by `use_split_kv_path`, `use_tiled` flags computed from
shape predicates.

**Impact:** Switching paths changes ULPs (Section 11.1, 11.2). The
flags are not configurable; the user cannot force a path for testing
(except via `use_ref` which disables both tiled and split-KV, falling
to scalar). Differential testing of tiled vs scalar is impossible
without code changes.

### W4 — CUDA kernel selection is a hand-tuned if-else chain

**Evidence:** `fattn.cu:460-533` — per-DKQ switch (64, 80, 96, 112, 128,
192, 256, 320, 512, 576), per-CC branches (Volta, Turing, Ada,
Blackwell, AMD MFMA, AMD WMMA), per-`gqa_ratio` branches.

**Impact:** Adding a new DKQ requires editing this file. The selection
is not autotuned — it's based on heuristics derived from offline
benchmarking. A new GPU CC could trigger a suboptimal kernel until
someone re-tunes.

### W5 — Cross-backend KV dtype support is non-uniform

**Evidence:** Section 10 table. CPU tiled path supports only F16/F32
KV; CUDA supports 8 dtypes; Metal and Vulkan ship per-template
specializations. The set is not documented in the public API; the
model code must query `ggml_backend_dev_supports_op` to know what's
allowed.

**Impact:** A model with Q8_0 KV cache will silently fall back to CPU
if the GPU backend doesn't support Q8_0 FA. The scheduler handles
this, but the user may not realize their quantized KV is running on
CPU.

### W6 — `logit_softcap` divides `scale` in place

**Evidence:** `ops.cpp:8532-8534`:
```c
if (logit_softcap != 0) { scale /= logit_softcap; }
```

**Impact:** The `scale` local variable is modified. If a future caller
reads `op_params[0]` expecting the original scale, they'd get the
pre-divided value. Not a bug today (no caller does this), but fragile.

### W7 — Sinks applied only on `ic_start == 0`

**Evidence:** `ops.cpp:8666` `if (sinks && ic_start == 0)`.

**Impact:** Correct only because the split-KV path guarantees chunk 0
is the first KV chunk. If a future path reorders chunks (e.g., for
flash-decoding with non-contiguous KV), the sink would be applied to
the wrong chunk. The contract is implicit, not enforced.

### W8 — RoPE `mode & 1 == 1` is asserted out

**Evidence:** `ggml.c:4184` `GGML_ASSERT((mode & 1) == 0 && "mode & 1 == 1 is no longer supported")`.

**Impact:** Old code that passed `mode = 1` (interleaved RoPE) will
abort. The deprecation is silent in the enum (no comment in `ggml.h`);
only the assertion message documents it. Migrating users may hit this
at runtime.

---

## 15. GwenLand Mapping

| GwenLand module | ADOPT / ADAPT / REJECT / MONITOR / DEFER | What | Reasoning |
| --------------- | ---------------------------------------- | ---- | --------- |
| `GATE`          | **ADOPT** | FA op param layout (scale, max_bias, logit_softcap, prec) | Cross-backend contract; clean. |
| `GATE`          | **ADOPT** | `SET_ROWS`-based KV append | Scattered-row write is the right primitive. |
| `GATE`          | **ADOPT** | Online softmax with optional F16 accumulator | Standard FA pattern. |
| `GATE`          | **ADOPT** | ALiBi slope formula (identical to soft_max) | Cross-op consistency. |
| `GATE`          | **ADOPT** | Logit softcap formulation | Gemma-compatible. |
| `GATE`          | **ADOPT** | Attention sinks via `src[4]`, first-chunk-only | Streaming-LLM compatible. |
| `GATE`          | **ADOPT** | MLA via `V_is_K_view` detection | No new op needed. |
| `GATE`          | **ADOPT** | RoPE 15-slot op_params + 5 modes | Covers all current models. |
| `GATE`          | **REJECT**| Broken `FLASH_ATTN_BACK` | Either implement it or remove the op. |
| `glproc`        | **ADAPT** | Tile constants `Q_TILE=64, KV_TILE=64` | Make per-DK; fall back to scalar when misaligned. |
| `glproc`        | **ADAPT** | Three FA paths with implicit selection | Make path selection configurable for testing. |
| `glcuda`        | **ADAPT** | Hand-tuned per-DKQ × per-CC kernel selection | Keep the heuristics, but add an autotuner for new CCs. |
| `glvulkan`      | **ADOPT** | Spec-constant `FaTypeK` / `FaTypeV` | Cleanest way to specialize shaders per dtype. |

---

## 16. Recommendations

### R1 — ADOPT FA op param layout as the GwenLand attention contract
**Priority:** Critical
**Difficulty:** S
**Dependencies:** none
GwenLand's `gl_flash_attn_ext` should use the same 4-slot op_params (`scale`, `max_bias`, `logit_softcap`, `prec`) and 5-source layout (`q`, `k`, `v`, `mask`, `sinks`). Every backend (`glproc`, `glcuda`, `glmetal`, `glvulkan`) must honor this layout. Adding a parameter (e.g., a future sliding-window scalar) bumps a `GL_ATTN_API_VERSION` integer.

### R2 — ADOPT `SET_ROWS` as the KV append primitive
**Priority:** Critical
**Difficulty:** S
**Dependencies:** none
GwenLand's `gl_set_rows(dst, src, idxs)` should support F32 and F16 sources, I32 and I64 indices, and call `from_float` to convert F32 to the cache's quantized dtype. Same semantics as ggml.

### R3 — ADOPT online softmax with optional F16 accumulator
**Priority:** High
**Difficulty:** M
**Dependencies:** R1
GwenLand's `glproc` scalar FA path should use an F16 VKQ accumulator when V is F16 (saves bandwidth); the tiled path should use F32. Document the ULP difference.

### R4 — ADOPT ALiBi slope formula
**Priority:** High
**Difficulty:** XS
**Dependencies:** R1
Same formula in `gl_flash_attn_ext` and `gl_soft_max_ext`: `m0 = pow(2, -max_bias/n_head_log2)`, `m1 = pow(2, -(max_bias/2)/n_head_log2)`, `slope(h) = h < n_head_log2 ? pow(m0, h+1) : pow(m1, 2*(h-n_head_log2)+1)`. Cross-op consistency is a feature.

### R5 — ADOPT logit softcap formulation
**Priority:** Medium
**Difficulty:** XS
**Dependencies:** R1
`if (logit_softcap != 0) { scale /= logit_softcap; }` then `s = logit_softcap * tanh(s * scale)`. Same as ggml.

### R6 — ADOPT attention sinks via `src[4]`, first-chunk-only
**Priority:** Medium
**Difficulty:** S
**Dependencies:** R1, R3
Sinks are an optional 5th source; one F32 per query head; applied only on the first KV chunk in split-KV decode. Document the contract.

### R7 — ADOPT MLA via `V_is_K_view` detection
**Priority:** High
**Difficulty:** M
**Dependencies:** R1
In `glcuda` and `glmetal`, detect `V->view_src == K` (or shared `view_src` + `view_offs`) and skip the V global load. Template parameter on the CUDA kernel.

### R8 — REJECT broken `FLASH_ATTN_BACK`
**Priority:** Medium
**Difficulty:** L
**Dependencies:** R1
Either implement gradient attention (substantial work) or remove the op entirely. Do not ship a `GGML_ABORT`'d stub.

### R9 — ADAPT tile constants per-DK
**Priority:** Medium
**Difficulty:** M
**Dependencies:** R1
Replace `GGML_FA_TILE_Q = 64` with a per-DK table: `tile_q = max(32, round_down_pow2(DK / 2))`. Fall back to scalar when `DV % f32_epr != 0`.

### R10 — ADAPT CPU FA path selection to be configurable
**Priority:** Low
**Difficulty:** S
**Dependencies:** R3, R9
Expose `use_split_kv`, `use_tiled`, `use_scalar` as runtime flags (per-backend context) so differential testing is possible without code changes.

### R11 — ADAPT CUDA FA kernel selection with autotuner fallback
**Priority:** Medium
**Difficulty:** L
**Dependencies:** R7
Keep the hand-tuned heuristics for known CCs, but if `cc` is unknown or `DKQ` is not in the switch, fall back to a runtime autotuner that benchmarks `TILE` vs `VEC` vs `MMA_F16` on the first call and caches the result.

---

## 17. Findings

### Finding ARTX24-F01

```
Finding ID:           ARTX24-F01
Category:             BACKEND_DESIGN
Engine:               Shared
Component:            FA op parameter layout (cross-backend contract)
Source File:          ggml/src/ggml.c
Function:             ggml_flash_attn_ext
Lines:                5402-5444
Summary:              FA op_params are a fixed 4-slot packed array (scale,
                      max_bias, logit_softcap, prec); sinks are an optional
                      5th source. This layout is the cross-backend attention
                      contract — every backend reads the same memory.
Observation:          ggml_flash_attn_ext packs 3 floats (scale, max_bias,
                      logit_softcap) into op_params[0..2] via ggml_set_op_params,
                      then ggml_flash_attn_ext_set_prec writes prec as i32 to
                      op_params[3]. The 5 source slots are q (src[0]), k (src[1]),
                      v (src[2]), mask (src[3], F16, contiguous), sinks (src[4],
                      F32, optional via ggml_flash_attn_ext_add_sinks). Every
                      backend (CPU, CUDA, Metal, Vulkan, SYCL, CANN) reads this
                      exact layout — see ops.cpp:8528-8530 (CPU reads 3 floats),
                      fattn.cu:46 (CUDA reads max_bias at offset 1), flash_attn_base.glsl:62-64
                      (Vulkan reads scale/max_bias/logit_softcap from push constant).
                      Adding a new parameter requires bumping the layout and
                      updating every backend; the current set covers all
                      current models (Llama, Gemma, Mistral, Qwen, DeepSeek,
                      GLM, Kimi).
Evidence:             ggml.c:5434-5435 (pack params), 5447-5455 (set_prec),
                      5466-5480 (add_sinks); ops.cpp:8528-8530 (CPU read);
                      fattn.cu:46 (CUDA read); flash_attn_base.glsl:62-64
                      (Vulkan read).
Architectural Impact: Clean cross-backend contract. One op, one layout,
                      every backend agrees. Adding parameters is a versioned
                      change.
Correctness Impact:   None. Layout is consistent across backends.
Optimization Type:    None.
GwenLand Target:      GATE, glproc, glcuda, glmetal, glvulkan
Recommendation:       ADOPT. Replicate the 4-slot op_params + 5-source layout
                      in GwenLand's gl_flash_attn_ext. Version the layout with
                      a GL_ATTN_API_VERSION integer.
Priority:             Critical
Difficulty:           S
Dependencies:         R1
Confidence:           High
```

### Finding ARTX24-F02

```
Finding ID:           ARTX24-F02
Category:             BACKEND_DESIGN
Engine:               Shared
Component:            KV cache tensor shape
Source File:          src/llama-kv-cache.cpp
Function:             llama_kv_cache::llama_kv_cache (ctor)
Lines:                198-247
Summary:              KV cache is allocated as a 3D tensor [head_dim * n_kv_heads,
                      kv_size, n_stream]; MLA models allocate K-only (has_v=false).
Observation:          The ctor loops over every layer and allocates:
                        k = ggml_new_tensor_3d(ctx, type_k, n_embd_k_gqa, kv_size, n_stream)
                        v = has_v ? ggml_new_tensor_3d(ctx, type_v, n_embd_v_gqa, kv_size, n_stream) : nullptr
                      where n_embd_k_gqa = n_embd_head_k * n_kv_heads (GQA-packed),
                      kv_size is the cache slot count, n_stream is the parallel
                      decode stream count (1 for unified cache). For MLA models
                      (hparams.is_mla()), has_v = false — V is a view of K,
                      allocated by the model code, not the cache. type_k and
                      type_v are configurable (typically F16; Q4_0/Q8_0/BF16
                      supported per backend).
Evidence:             llama-kv-cache.cpp:198-247; llama-hparams.cpp:244-249
                      (is_mla); llama-kv-cache.cpp:229 (has_v = !is_mla).
Architectural Impact: GQA-packed innermost dim lets one SET_ROWS call write a
                      full token's K across all heads. MLA's K-only cache saves
                      half the KV memory.
Correctness Impact:   None.
Optimization Type:    GQA-packed layout.
GwenLand Target:      GATE, glproc, glcuda, glmetal, glvulkan
Recommendation:       ADOPT. Same 3D shape for GwenLand's KV cache. Same MLA
                      K-only convention.
Priority:             High
Difficulty:           S
Dependencies:         R2
Confidence:           High
```

### Finding ARTX24-F03

```
Finding ID:           ARTX24-F03
Category:             BACKEND_DESIGN
Engine:               Shared
Component:            KV append primitive (SET_ROWS)
Source File:          ggml/src/ggml.c, ggml/src/ggml-cpu/ops.cpp, src/llama-kv-cache.cpp
Function:             ggml_set_rows, ggml_compute_forward_set_rows_impl, llama_kv_cache::cpy_k
Lines:                ggml.c:3937-3963; ops.cpp:5092-5160; llama-kv-cache.cpp:1276-1328
Summary:              Per-token KV append is a scattered-row write indexed by
                      an I32/I64 vector. The cache is the destination; the new
                      token's K/V is the source; the slot index is the index
                      tensor. Supports F32->quant via from_float.
Observation:          ggml_set_rows constructs an op with src[0]=b (new rows,
                      F32 or F16), src[1]=c (i32 or i64 row indices), src[2]=a
                      (the cache, as a view tensor). The CPU impl templated on
                      <src_t, idx_t> iterates rows, reads the index from src[1],
                      and writes the row (via from_float if dst is quantized)
                      to dst[i1]. The index tensor is built by the model code
                      (llama_kv_cache::build_input_k_idxs) from the slot
                      allocator's output. The same op is used for K, V, and
                      any other per-token row update (e.g., SSM state writes).
                      The order of src[] is "weird due to legacy reasons"
                      (comment at ggml.c:3960, citing PR #16063).
Evidence:             ggml.c:3937-3963 (construction); ops.cpp:5092-5160
                      (CPU impl); llama-kv-cache.cpp:1327 (cpy_k calls
                      ggml_set_rows).
Architectural Impact: One primitive for all per-token row updates. Cache slot
                      assignment is decoupled from the write. Works for any
                      cache dtype (F32, F16, Q4_0, Q8_0, BF16).
Correctness Impact:   None. Index bounds checked at ops.cpp:5134
                      (GGML_ASSERT(i1 >= 0 && i1 < ne1)).
Optimization Type:    Scattered-row write.
GwenLand Target:      GATE, glproc, glcuda, glmetal, glvulkan
Recommendation:       ADOPT. Replicate gl_set_rows in GwenLand with the same
                      semantics. Document the src[] order explicitly to avoid
                      the "legacy" confusion.
Priority:             Critical
Difficulty:           S
Dependencies:         R2
Confidence:           High
```

### Finding ARTX24-F04

```
Finding ID:           ARTX24-F04
Category:             MISSING_FEATURE
Engine:               Shared
Component:            Flash attention backward (gradient path)
Source File:          ggml/src/ggml.c
Function:             ggml_flash_attn_back
Lines:                5484-5551
Summary:              The old GGML_OP_FLASH_ATTN has been removed; only
                      FLASH_ATTN_EXT and FLASH_ATTN_BACK remain. FLASH_ATTN_BACK
                      is GGML_ABORT'd as "TODO: adapt to ggml_flash_attn_ext()
                      changes" — the gradient path is dead code.
Observation:          The op enum (ggml.h:560-561) has GGML_OP_FLASH_ATTN_EXT
                      and GGML_OP_FLASH_ATTN_BACK but no GGML_OP_FLASH_ATTN.
                      ggml_flash_attn_back (ggml.c:5484) begins with
                        GGML_ABORT("TODO: adapt to ggml_flash_attn_ext() changes");
                      The function body below the abort still references the old
                      FLASH_ATTN semantics (a single `masked` boolean instead
                      of a mask tensor, a `d` gradient tensor at src[3]). The
                      op_params layout (a single i32 `masked`) does not match
                      FLASH_ATTN_EXT's 4-slot layout. The body is unreachable.
                      Any training graph that includes FA backward will abort
                      at graph construction. The CPU impl
                      (ggml_compute_forward_flash_attn_back at ops.cpp:9219+)
                      still exists but is unreachable because the op cannot be
                      constructed without aborting.
Evidence:             ggml.c:5484-5551; ggml.h:560-561; ops.cpp:9219+
                      (CPU impl exists but unreachable).
Architectural Impact: Training or any gradient-based attention through FA is
                      impossible. Users must use a non-FA attention path
                      (mul_mat + soft_max_ext + mul_mat) for training.
Correctness Impact:   None for inference. Training users hit abort.
Optimization Type:    None.
GwenLand Target:      GATE
Recommendation:       REJECT this state. Either implement FA backward
                      (substantial work — needs gradient w.r.t. q, k, v, mask)
                      or remove GGML_OP_FLASH_ATTN_BACK and the unreachable
                      CPU impl. Do not ship a GGML_ABORT'd stub.
Priority:             Medium
Difficulty:           L
Dependencies:         R8
Confidence:           High
```

### Finding ARTX24-F05

```
Finding ID:           ARTX24-F05
Category:             LAYOUT_SUBOPTIMAL
Engine:               Shared
Component:            FA tile constants (CPU)
Source File:          ggml/src/ggml-cpu/common.h, ggml/src/ggml-cpu/ops.cpp
Function:             GGML_FA_TILE_Q, GGML_FA_TILE_KV, ggml_compute_forward_flash_attn_ext_f16
Lines:                common.h:9-10; ops.cpp:8780-8781, 9172-9184
Summary:              CPU FA tile sizes are hardcoded to Q=64, KV=64. The tiled
                      path requires neq1 >= 64 and DV % f32_epr == 0; otherwise
                      falls back to the scalar one_chunk path.
Observation:          common.h defines GGML_FA_TILE_Q = 64 and GGML_FA_TILE_KV = 64
                      as compile-time constants. The tiled path
                      (ggml_compute_forward_flash_attn_ext_tiled) uses these as
                      Q_TILE_SZ and KV_TILE_SZ. The selection predicate at
                      ops.cpp:9172-9184 is:
                        use_tiled = !use_ref && q->type == F32 && kv_is_f32_or_f16
                                    && k->type == v->type && neq1 >= Q_TILE_SZ
                                    && (DV % f32_epr == 0);
                      For models with head_dim not divisible by the SIMD width
                      (e.g., 40, 72, 96 on some phi/gemma variants), DV % f32_epr
                      != 0 and the tiled path is disabled — falling back to
                      scalar one_chunk, which is 5-10x slower for prefill. The
                      tile sizes are not per-DK; a model with DK=128 always
                      uses Q_TILE=64 even when Q_TILE=128 might fit in L1.
Evidence:             common.h:9-10; ops.cpp:8780-8781, 9172-9184.
Architectural Impact: Suboptimal prefill throughput for models with unusual
                      head dims. The fallback to scalar is silent.
Correctness Impact:   None. Both paths produce correct results (with ULP
                      differences — see F06).
Optimization Type:    Tiling (hardcoded).
GwenLand Target:      glproc
Recommendation:       ADAPT. Replace the compile-time constants with a per-DK
                      table: tile_q = max(32, round_down_pow2(DK / 2)); fall
                      back to scalar when DV % f32_epr != 0. Make tile size a
                      runtime parameter so autotuning is possible.
Priority:             Medium
Difficulty:           M
Dependencies:         R9
Confidence:           High
```

### Finding ARTX24-F06

```
Finding ID:           ARTX24-F06
Category:             CORRECTNESS_SHORTCUT
Engine:               Shared
Component:            CPU FA path selection and online softmax accumulator
Source File:          ggml/src/ggml-cpu/ops.cpp
Function:             ggml_compute_forward_flash_attn_ext_f16, _f16_one_chunk, _tiled, _reduce_partials
Lines:                9066-9200, 8468-8704, 8706-8992, 8996-9064
Summary:              CPU FA has three paths (split-KV decode, tiled prefill,
                      scalar fallback) with implicit shape-driven selection.
                      The scalar path uses an F16 VKQ accumulator when V is F16;
                      the tiled path always uses F32. Switching paths changes
                      ULPs.
Observation:          The dispatch in ggml_compute_forward_flash_attn_ext_f16
                      picks one of three paths based on shape predicates:
                        - use_split_kv_path: neq1 == 1 && neq3 == 1 (decode) &&
                          kv_is_f32_or_f16 && k->type == v->type && q->type == F32
                          && nek1 >= 512. Splits KV across threads, partials
                          reduced via ggml_flash_attn_ext_reduce_partials.
                        - use_tiled: q->type == F32 && kv_is_f32_or_f16 &&
                          k->type == v->type && neq1 >= Q_TILE_SZ && DV % f32_epr == 0.
                          Uses simd_gemm.
                        - else: scalar _f16_one_chunk. Online softmax with one
                          accumulator (M, S, VKQ).
                      The scalar path uses F16 VKQ16 accumulator when V is F16
                      (ops.cpp:8618-8632); the tiled path always uses F32 VKQ32
                      (ops.cpp:8824). The split-KV partials are always F32
                      (ops.cpp:8689: memcpy(partial + 2, VKQ32, DV * sizeof(float))).
                      Switching paths (e.g., from decode to prefill on the next
                      token) changes the accumulator dtype and the KV processing
                      order, producing ULP-level differences.
Evidence:             ops.cpp:9066-9200 (dispatch), 8468-8704 (scalar),
                      8706-8992 (tiled), 8996-9064 (reduce_partials).
Architectural Impact: Differential testing across paths is impossible without
                      code changes. The path selection is not configurable
                      (use_ref forces scalar, but no flag forces tiled or
                      split-KV).
Correctness Impact:   Bit-exact reproducibility only within one path. Switching
                      paths (decode -> prefill) produces ULP-level differences.
Optimization Type:    Online softmax + path-specific accumulator dtype.
GwenLand Target:      glproc
Recommendation:       ADAPT. Keep all three paths (they serve different shapes).
                      Add runtime flags (use_split_kv, use_tiled, use_scalar) so
                      differential testing is possible. Document the ULP
                      difference between paths.
Priority:             High
Difficulty:           M
Dependencies:         R3, R10
Confidence:           High
```

### Finding ARTX24-F07

```
Finding ID:           ARTX24-F07
Category:             ADOPT
Engine:               Shared
Component:            ALiBi slope formula (cross-op consistency)
Source File:          ggml/src/ggml-cpu/ops.cpp, ggml/src/ggml.c
Function:             ggml_compute_forward_flash_attn_ext_f16_one_chunk, ggml_compute_forward_soft_max_f32, ggml_soft_max_impl
Lines:                ops.cpp:8536-8540 (FA), ops.cpp:5455+ (soft_max); ggml.c:4047-4079
Summary:              The ALiBi slope formula is identical between
                      flash_attn_ext and soft_max_ext. A model using the legacy
                      KQ+soft_max path gets the same slopes as one using FA.
Observation:          The FA kernel derives per-head ALiBi slopes from the
                      scalar max_bias parameter:
                        n_head_log2 = 1 << floor(log2(n_head))
                        m0 = powf(2.0, -(max_bias       ) / n_head_log2)
                        m1 = powf(2.0, -(max_bias / 2.0) / n_head_log2)
                        slope(h) = h < n_head_log2 ? powf(m0, h+1) : powf(m1, 2*(h-n_head_log2)+1)
                      The slope is applied as slope * mask[ic] inside the
                      kernel (ops.cpp:8593). The soft_max_ext op uses the same
                      formula (audited in ops.cpp:5455+). The mask tensor
                      already contains the ALiBi bias values (set by the model
                      code in set_input_kq_mask); the kernel multiplies by the
                      per-head slope. This means a model can switch between FA
                      and KQ+soft_max_ext without changing ALiBi behavior.
                      max_bias == 0 disables ALiBi (slope = 1.0 for all heads).
                      max_bias > 0 requires a mask (ggml.c:5426-5428).
Evidence:             ops.cpp:8536-8540 (FA slope), 8593 (slope applied);
                      ops.cpp:5455+ (soft_max same formula); ggml.c:5426-5428
                      (max_bias > 0 requires mask).
Architectural Impact: Cross-op consistency. ALiBi behaves identically across
                      attention implementations.
Correctness Impact:   None. Formula is correct and consistent.
Optimization Type:    Precomputed slope constants per kernel invocation.
GwenLand Target:      GATE, glproc, glcuda, glmetal, glvulkan
Recommendation:       ADOPT. Same formula in gl_flash_attn_ext and
                      gl_soft_max_ext. Document the cross-op consistency as a
                      feature.
Priority:             High
Difficulty:           XS
Dependencies:         R4
Confidence:           High
```

### Finding ARTX24-F08

```
Finding ID:           ARTX24-F08
Category:             ADOPT
Engine:               Shared
Component:            Logit softcap (Gemma-style)
Source File:          ggml/src/ggml-cpu/ops.cpp, ggml/src/ggml-cuda/fattn.cu
Function:             ggml_compute_forward_flash_attn_ext_f16_one_chunk, ggml_compute_forward_flash_attn_ext_tiled
Lines:                ops.cpp:8526-8534, 8605-8607, 8903-8906; (CUDA applies same in fattn-mma-f16.cuh)
Summary:              Logit softcap is baked into FA op_params (slot 2). The
                      kernel pre-divides scale by softcap, then applies
                      softcap*tanh(s*scale). This is the Gemma-style logit
                      softcap; identical across CPU and CUDA.
Observation:          The FA op param logit_softcap (op_params[2]) is read as
                      a float. When nonzero:
                        scale /= logit_softcap          (ops.cpp:8532-8534)
                        s = s * scale                   (within the KQ loop)
                        s = logit_softcap * tanhf(s)    (ops.cpp:8605-8607)
                      The pre-division keeps the scaled score in a reasonable
                      range for tanhf. The tiled path applies the same formula
                      to the whole KQ tile at once via ggml_vec_tanh_f32 and
                      ggml_vec_scale_f32 (ops.cpp:8903-8906). CUDA's
                      fattn-mma-f16.cuh applies the same formula. logit_softcap
                      = 0 disables the cap. Gemma-2 uses logit_softcap = 50.0.
                      This is fused into FA — no separate tanh op in the graph.
Evidence:             ops.cpp:8526-8534 (read + pre-divide), 8605-8607 (scalar
                      apply), 8903-8906 (tiled apply); ggml.c:5434 (packed into
                      op_params).
Architectural Impact: Fused softcap saves a memory round-trip and a graph node.
                      Cross-backend: CPU and CUDA agree.
Correctness Impact:   None. Formula is the standard Gemma softcap.
Optimization Type:    Kernel fusion (softcap + softmax).
GwenLand Target:      GATE, glproc, glcuda, glmetal, glvulkan
Recommendation:       ADOPT. Same formula in gl_flash_attn_ext. Keep the
                      pre-division trick. Document that scale is modified in
                      place (so future callers reading op_params[0] get the
                      pre-divided value — see W6).
Priority:             Medium
Difficulty:           XS
Dependencies:         R5
Confidence:           High
```

### Finding ARTX24-F09

```
Finding ID:           ARTX24-F09
Category:             ADOPT
Engine:               Shared
Component:            Attention sinks (streaming-LLM)
Source File:          ggml/src/ggml.c, ggml/src/ggml-cpu/ops.cpp
Function:             ggml_flash_attn_ext_add_sinks, ggml_compute_forward_flash_attn_ext_f16_one_chunk, ggml_compute_forward_flash_attn_ext_tiled
Lines:                ggml.c:5466-5480; ops.cpp:8480 (read), 8666-8681 (scalar apply), 8958-8974 (tiled apply)
Summary:              Attention sinks are an optional 5th source tensor (F32,
                      one value per query head). The kernel applies them only
                      on the first KV chunk. This is the streaming-LLM "sink
                      token" mechanism.
Observation:          ggml_flash_attn_ext_add_sinks (ggml.c:5466-5480) sets
                      src[4] to a 1D F32 tensor of length n_q_heads. The kernel
                      reads sinks[h] (ops.cpp:8667) and applies it as an extra
                      "always-attended" virtual position with logit sinks[h]:
                        if (sinks && ic_start == 0) {
                            s = sinks[h]
                            if (s > M) { ms = expf(M - s); M = s; rescale VKQ }
                            else       { vs = expf(s - M); }
                            S = S*ms + vs
                            VKQ += (vs == 1 ? 0 : V_sink)  // V_sink is implicit (zero)
                        }
                      The first-chunk-only rule (ic_start == 0) is critical: the
                      sink contributes to the initial softmax max M; later
                      chunks see the sink's effect via the rescaled M. If a
                      backend applied sinks on every chunk, the sink would be
                      counted N times. The tiled path applies sinks after all
                      KV tiles are processed, to every row in the Q tile
                      (ops.cpp:8958-8974). The contract is implicit — there is
                      no assertion that ic_start == 0 in the split-KV path
                      (ops.cpp:8666 just checks ic_start == 0).
Evidence:             ggml.c:5466-5480 (add_sinks); ops.cpp:8480 (read src[4]),
                      8666-8681 (scalar apply with ic_start check), 8958-8974
                      (tiled apply, no ic_start check because tiled path does
                      not split KV).
Architectural Impact: Sinks fused into FA — no separate sink op. The
                      first-chunk-only contract is fragile (see W7).
Correctness Impact:   None today. A future path that reorders KV chunks could
                      apply sinks to the wrong chunk.
Optimization Type:    Kernel fusion (sinks + softmax).
GwenLand Target:      GATE, glproc, glcuda, glmetal, glvulkan
Recommendation:       ADOPT. Same src[4] sinks contract in GwenLand. Document
                      the first-chunk-only rule explicitly. Add an assertion in
                      the split-KV path that chunk 0 is always the first KV
                      chunk.
Priority:             Medium
Difficulty:           S
Dependencies:         R6
Confidence:           High
```

### Finding ARTX24-F10

```
Finding ID:           ARTX24-F10
Category:             ADOPT
Engine:               Shared
Component:            MLA (Multi-head Latent Attention) via V_is_K_view
Source File:          ggml/src/ggml-cuda/fattn-common.cuh, ggml/src/ggml-cuda/fattn-mma-f16.cuh, src/llama-kv-cache.cpp
Function:             (V_is_K_view detection), flash_attn_ext_f16 (template)
Lines:                fattn-common.cuh:63, 75; fattn-mma-f16.cuh:1915; llama-kv-cache.cpp:229 (has_v = !is_mla)
Summary:              MLA is handled by the existing FA op with no new op code.
                      When V is a view of K (V->view_src == K), the CUDA kernel
                      reuses the K tile in shared memory for V, skipping a
                      global load. The detection is by view_src pointer
                      comparison.
Observation:          llama_kv_cache allocates K-only (has_v = false) when
                      hparams.is_mla() is true (llama-kv-cache.cpp:229). The
                      model code (DeepSeek, GLM, Kimi) makes V a view of K via
                      ggml_view_2d. CUDA detects this in fattn-common.cuh:63:
                        V_is_K_view = V->view_src && (V->view_src == K ||
                                     (V->view_src == K->view_src && V->view_offs == K->view_offs))
                      When true, the MMA_F16 kernel (fattn-mma-f16.cuh:1915)
                      sets constexpr bool V_is_K_view = (DKQ == 576) — guaranteed
                      by the kernel selection logic in fattn.cu (case 576 at
                      line 182). The kernel then uses stride_tile_V = stride_tile_K
                      and reads V from the K tile in shared memory (lines 573,
                      970, 1173, 1788, 1821, 1867). The static_assert at line
                      586 disables multi-stage V loading when V_is_K_view
                      (because the K tile is already loaded). The CPU path
                      does not have this optimization — it always loads V
                      separately. Metal and Vulkan do not have it either
                      (audited at this commit).
Evidence:             fattn-common.cuh:63, 75; fattn-mma-f16.cuh:1915, 573,
                      586, 970, 1173, 1788, 1821, 1867; llama-kv-cache.cpp:229;
                      llama-hparams.cpp:244-249.
Architectural Impact: MLA runs at the same speed as standard attention on
                      CUDA (one global load instead of two). CPU/Metal/Vulkan
                      pay the V load cost — a gap.
Correctness Impact:   None. Detection is conservative (only true when V is
                      literally a view of K).
Optimization Type:    Shared-memory tile reuse.
GwenLand Target:      glcuda, glmetal, glvulkan
Recommendation:       ADOPT. glcuda and glmetal should implement V_is_K_view
                      detection. Add a static_assert that multi-stage V loading
                      is disabled when V_is_K_view. Consider extending to CPU
                      and Vulkan as a follow-up.
Priority:             High
Difficulty:           M
Dependencies:         R7
Confidence:           High
```

### Finding ARTX24-F11

```
Finding ID:           ARTX24-F11
Category:             GPU_KERNEL
Engine:               Shared
Component:            CUDA FA kernel selection
Source File:          ggml/src/ggml-cuda/fattn.cu
Function:             ggml_cuda_get_best_fattn_kernel, ggml_cuda_flash_attn_ext_mma_f16
Lines:                330-534, 113-242
Summary:              CUDA FA kernel selection is a hand-tuned per-DKQ × per-CC
                      if-else chain. Adding a new head dim requires editing the
                      source. There is no autotuner fallback for unknown CCs.
Observation:          ggml_cuda_get_best_fattn_kernel (fattn.cu:330-534) returns
                      one of BEST_FATTN_KERNEL_NONE / _TILE (200) / _VEC (100) /
                      _MMA_F16 (400). The decision tree considers:
                        - DKQ (Q->ne[0]): must be one of 40, 64, 72, 80, 96, 112,
                          128, 192, 256, 320, 512, 576 — any other value returns
                          NONE (unsupported).
                        - DV (V->ne[0]): usually must equal DKQ; exceptions are
                          192->128, 320->256, 576->512 (DeepSeek/GLM/MiMo).
                        - gqa_ratio (Q->ne[2] / K->ne[2]): determines ncols2 in
                          the MMA template (1, 2, 4, 8, 16, 32).
                        - cc (compute capability): Volta, Turing, Ada, Blackwell,
                          AMD MFMA, AMD WMMA each have different heuristics.
                        - mask, max_bias, K/V quantization, K->ne[1] alignment.
                      The MMA kernel itself (ggml_cuda_flash_attn_ext_mma_f16,
                      fattn.cu:113-242) is a per-DKQ switch that instantiates
                      the template with the right ncols2. Adding a new DKQ
                      requires adding a case here AND a new template instantiation
                      in fattn-mma-f16.cuh. There is no autotuner — selection
                      is based on heuristics derived from offline benchmarking.
                      A new GPU CC (e.g., a future Nvidia architecture) would
                      trigger the fallback path (line 520: "If there are no
                      tensor cores available, use the generic tile kernel"),
                      which may be suboptimal.
Evidence:             fattn.cu:330-534 (selection), 113-242 (MMA dispatch),
                      460-533 (per-CC branches).
Architectural Impact: Adding head dims requires source edits. New CCs may
                      trigger suboptimal kernels. Maintenance burden grows with
                      each new model.
Correctness Impact:   None. All kernel variants produce correct results (with
                      ULP differences).
Optimization Type:    Hand-tuned per-shape, per-CC dispatch.
GwenLand Target:      glcuda
Recommendation:       ADAPT. Keep the hand-tuned heuristics for known CCs and
                      DKQs. Add an autotuner fallback: if cc is unknown or DKQ
                      is not in the switch, benchmark TILE vs VEC vs MMA_F16 on
                      the first call and cache the result per (DKQ, DV, gqa_ratio,
                      cc).
Priority:             Medium
Difficulty:           L
Dependencies:         R11
Confidence:           High
```

### Finding ARTX24-F12

```
Finding ID:           ARTX24-F12
Category:             BACKEND_DESIGN
Engine:               Shared
Component:            RoPE op parameter layout
Source File:          ggml/src/ggml.c, ggml/include/ggml.h
Function:             ggml_rope_impl, GGML_ROPE_TYPE_* (defines)
Lines:                ggml.c:4168-4223; ggml.h:250-254
Summary:              RoPE packs 15 i32 slots into one op_params array,
                      covering 5 modes (NORMAL, NEOX, MROPE, VISION, IMROPE)
                      and all YaRN/longrope/sectioned parameters. The
                      deprecated mode & 1 == 1 is asserted out.
Observation:          ggml_rope_impl (ggml.c:4168-4223) packs 15 i32 params:
                        [0] n_past (unused, kept for ABI)
                        [1] n_dims
                        [2] mode (GGML_ROPE_TYPE_*)
                        [3] n_ctx (unused, kept for ABI)
                        [4] n_ctx_orig
                        [5] freq_base (float, reinterpreted as i32)
                        [6] freq_scale (float)
                        [7] ext_factor (float)
                        [8] attn_factor (float)
                        [9] beta_fast (float)
                        [10] beta_slow (float)
                        [11..14] sections[4] (mrope only)
                      Five modes (ggml.h:250-254):
                        GGML_ROPE_TYPE_NORMAL  = 0   (interleaved cos/sin)
                        GGML_ROPE_TYPE_NEOX    = 2   (split half cos/sin)
                        GGML_ROPE_TYPE_MROPE   = 8   (multi-modal, 4 pos ids per token)
                        GGML_ROPE_TYPE_VISION  = 24  (vision, single pos id)
                        GGML_ROPE_TYPE_IMROPE  = 40  (interleaved mrope)
                      The deprecated mode & 1 == 1 (old interleaved) is asserted
                      out at ggml.c:4184. The source operand a can be any float
                      dtype (F32, F16, BF16). The position ids b are I32. The
                      optional c (longrope frequency factors) is F32 with
                      c->ne[0] >= n_dims / 2. mrope requires a->ne[2] * 4 ==
                      b->ne[0] (4 position ids per token: t, t, h, w). All
                      RoPE variants (rope, rope_multi, rope_ext, rope_custom,
                      rope_inplace) route through ggml_rope_impl with different
                      parameter combinations.
Evidence:             ggml.c:4168-4223 (impl), 4225-4365 (public variants);
                      ggml.h:250-254 (mode defines); ggml.c:4184 (deprecated
                      mode assertion).
Architectural Impact: One op covers all RoPE variants. No rope-neox,
                      rope-vision, rope-yarn op explosion. Adding a new mode
                      requires a new GGML_ROPE_TYPE_* constant and a kernel
                      branch.
Correctness Impact:   None. The deprecated mode & 1 is correctly rejected.
Optimization Type:    Packed op_params.
GwenLand Target:      GATE, glproc, glcuda, glmetal, glvulkan
Recommendation:       ADOPT. Same 15-slot layout and 5 modes in GwenLand's
                      gl_rope. Keep the deprecated-mode assertion. Document
                      the [0] n_past and [3] n_ctx slots as unused-but-reserved
                      for ABI compatibility.
Priority:             High
Difficulty:           S
Dependencies:         R1
Confidence:           High
```

---

## 18. Unknowns

* **U1**. Whether the CPU scalar path's F16 VKQ accumulator produces
  bit-identical results to the tiled path's F32 VKQ accumulator when V
  is F16. Static analysis shows the arithmetic is mathematically
  equivalent but the reduction order and accumulator dtype differ.
  Requires executing both paths on the same input. See F06.

* **U2**. Whether any model in `src/models/` actually uses the
  `ggml_flash_attn_ext_add_sinks` API. Static analysis of the FA op
  shows sinks are supported, but the model-side usage is not in scope.
  If no model uses sinks, the sinks code path is dead.

* **U3**. Whether the `op_params[0]` (`scale`) modification by
  `logit_softcap` (Section 11.4, W6) is observable to any caller.
  The modification is to a local variable, not to the op_params array
  itself (the kernel reads `op_params[0]` into a local `scale` and
  modifies the local). The op_params array is untouched. So no caller
  can observe the modification. W6 is a non-issue.

* **U4**. Whether the MLA `V_is_K_view` optimization is correct for
  DeepSeek-V3's specific MLA variant (which down-projects K and V from
  a shared latent). Static analysis shows the detection is by view_src
  pointer equality, which is correct for *any* model that makes V a
  view of K. DeepSeek-V3's specific shape (DKQ=576, DV=512) is
  hardcoded in the kernel selection. Requires confirming against
  DeepSeek-V3 model code (in `src/models/deepseek*.cpp`).

* **U5**. Whether the `DSV4_HC_{PRE,COMB,POST}` ops (ggml.h:574-576)
  are part of a new MLA-like attention variant for DeepSeek-V4. Their
  names suggest "high-combine", "pre", "post" — possibly a 3-stage
  attention. Not audited here; out of scope.

* **U6**. Whether the `LIGHTNING_INDEXER` op (ggml.h:573) is related
  to FA or is a separate retrieval mechanism. Its op construction
  (ggml.c:6313) takes q, k, weights, mask and produces
  [k->ne[2], q->ne[2], 1, q->ne[3]] — looks like a learnable retrieval
  attention. Not audited here.

* **U7**. Whether the CPU FA tiled path's `simd_gemm` call (ops.cpp:8891,
  8954) is the same `simd_gemm` audited in ARTX02-05 for matmul, or a
  separate FA-specific implementation. The function name is the same;
  confirming same implementation requires reading simd-gemm.h, which is
  outside this audit's scope.

---

## 19. References

| Reference | File                                                | Function / Symbol                              | Lines         |
| --------- | --------------------------------------------------- | ---------------------------------------------- | ------------- |
| R01       | `ggml/src/ggml.c`                                   | `ggml_flash_attn_ext`                          | 5402-5444     |
| R02       | `ggml/src/ggml.c`                                   | `ggml_flash_attn_ext_set_prec`                 | 5447-5455     |
| R03       | `ggml/src/ggml.c`                                   | `ggml_flash_attn_ext_add_sinks`                | 5466-5480     |
| R04       | `ggml/src/ggml.c`                                   | `ggml_flash_attn_back` (broken)                | 5484-5551     |
| R05       | `ggml/src/ggml.c`                                   | `ggml_set_rows`                                | 3937-3963     |
| R06       | `ggml/src/ggml.c`                                   | `ggml_rope_impl`                               | 4168-4223     |
| R07       | `ggml/src/ggml.c`                                   | `ggml_ssm_conv`, `ggml_ssm_scan`               | 5555-5644     |
| R08       | `ggml/src/ggml.c`                                   | `ggml_gated_delta_net`                         | 6254-6309     |
| R09       | `ggml/src/ggml.c`                                   | `ggml_soft_max_impl` (ALiBi)                   | 4047-4079     |
| R10       | `ggml/include/ggml.h`                               | `GGML_OP_FLASH_ATTN_EXT`, `_BACK`, SSM ops     | 560-576       |
| R11       | `ggml/include/ggml.h`                               | `GGML_ROPE_TYPE_*`                             | 250-254       |
| R12       | `ggml/src/ggml-cpu/common.h`                        | `GGML_FA_TILE_Q`, `GGML_FA_TILE_KV`, `ggml_fa_tile_config` | 9-10, 90-93 |
| R13       | `ggml/src/ggml-cpu/ops.cpp`                         | `ggml_compute_forward_flash_attn_ext_f16_one_chunk` | 8468-8704 |
| R14       | `ggml/src/ggml-cpu/ops.cpp`                         | `ggml_compute_forward_flash_attn_ext_tiled`    | 8706-8992     |
| R15       | `ggml/src/ggml-cpu/ops.cpp`                         | `ggml_flash_attn_ext_reduce_partials`          | 8996-9064     |
| R16       | `ggml/src/ggml-cpu/ops.cpp`                         | `ggml_compute_forward_flash_attn_ext_f16` (dispatch) | 9066-9200 |
| R17       | `ggml/src/ggml-cpu/ops.cpp`                         | `ggml_compute_forward_flash_attn_ext`          | 9202-9217     |
| R18       | `ggml/src/ggml-cpu/ops.cpp`                         | `ggml_compute_forward_set_rows_impl`           | 5092-5160     |
| R19       | `ggml/src/ggml-cuda/fattn.cu`                       | `ggml_cuda_flash_attn_ext`                     | 570-585       |
| R20       | `ggml/src/ggml-cuda/fattn.cu`                       | `ggml_cuda_get_best_fattn_kernel`              | 330-534       |
| R21       | `ggml/src/ggml-cuda/fattn.cu`                       | `ggml_cuda_flash_attn_ext_mma_f16`             | 113-242       |
| R22       | `ggml/src/ggml-cuda/fattn.cu`                       | `ggml_cuda_fattn_kv_type_supported`            | 338-355       |
| R23       | `ggml/src/ggml-cuda/fattn-common.cuh`               | `V_is_K_view` detection                        | 63, 75, 983, 1051 |
| R24       | `ggml/src/ggml-cuda/fattn-mma-f16.cuh`              | `constexpr bool V_is_K_view = DKQ == 576`      | 1915          |
| R25       | `ggml/src/ggml-metal/ggml-metal.metal`              | `kernel_flash_attn_ext` (template + specializations) | 6991-7080+ |
| R26       | `ggml/src/ggml-vulkan/vulkan-shaders/flash_attn.comp` | FA kernel                                   | 1-759         |
| R27       | `ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_base.glsl` | Push constant struct, FaTypeK/FaTypeV constants | 1-120    |
| R28       | `src/llama-kv-cache.cpp`                            | `llama_kv_cache` ctor (KV tensor allocation)   | 198-247       |
| R29       | `src/llama-kv-cache.cpp`                            | `cpy_k`, `cpy_v` (SET_ROWS calls)              | 1276-1384     |
| R30       | `src/llama-kv-cache.cpp`                            | `set_input_kq_mask_impl` (SWA mask)            | 1531+         |
| R31       | `src/llama-hparams.cpp`                             | `is_mla`, `n_embd_head_k_mla`, `n_embd_head_v_mla` | 244-264   |
