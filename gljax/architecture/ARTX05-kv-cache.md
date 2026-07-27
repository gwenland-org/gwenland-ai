# ARTX5 — gljax Static KV Cache + Bucketing Strategy

**Series:** gljax (Sanctum Visibilia) Architecture Research  
**Depends on:** ARTX1–ARTX4 (PJRT FFI, IR, ops/, runtime/, checkpoint/)  
**Next:** ARTX6 — Multi-Device Tensor Parallel + MoE Expert Sharding

---

## Overview

ARTX4 delivers prefill-only inference: one compiled function, full sequence
recomputation per request. ARTX5 adds **incremental decode** via a static KV cache —
the mechanism that makes autoregressive generation efficient.

The core challenge: XLA/PJRT requires static shapes, but a KV cache grows by one
token each decode step. The solution used by all production TPU serving stacks
(MaxText, JetStream, SGLang-JAX) is:

1. **Pre-allocate** a fixed `[B, n_kv_heads, max_seq_len, head_dim]` buffer
2. **Write** new K/V at position `t` via `stablehlo.dynamic_update_slice`  
3. **Read** K/V for positions `0..bucket_size` via static `stablehlo.slice`  
4. **Mask** future positions with `-inf` in the attention mask
5. **Donate** the KV cache buffer at PJRT level for in-place mutation (zero copy)

The bucketing strategy handles variable sequence lengths: compile one function
per bucket size (128/256/512/1024/2048), pad to the next bucket, select at runtime.

### What changes from ARTX4

| | ARTX4 | ARTX5 |
|---|---|---|
| Functions | 1 (prefill only) | 2 per bucket (prefill + decode) = 10 total |
| KV cache | None | Static `[B, H_kv, max_seq, D]` per layer |
| Sequence length | Fixed at compile | Bucketed (5 sizes) |
| Decode loop | Host-driven, recompute full seq | Device-side, 1 token per step |
| Compiled artifacts | 1 `.pjrt` | 10 `.pjrt` files in CompileCache |

---

## 1. KV Cache Design

### Buffer layout

```
kv_cache: [n_layers, 2, B, n_kv_heads, max_seq_len, head_dim]
```

- `n_layers`: 24 for Qwen2-0.5B
- `2`: K and V (index 0 = K, index 1 = V)
- `B`: batch size
- `n_kv_heads`: 2 for Qwen2-0.5B, 8 for Qwen2-7B
- `max_seq_len`: bucket size (128/256/512/1024/2048)
- `head_dim`: 64 for Qwen2-0.5B

Alternative per-layer layout (simpler for tracing):

```
kv_k: [B, n_kv_heads, max_seq_len, head_dim]   # K cache per layer
kv_v: [B, n_kv_heads, max_seq_len, head_dim]   # V cache per layer
```

gljax v1 uses **per-layer separate K/V tensors** — simpler to trace through
the 24-layer loop without a 6D tensor.

### Memory budget (Qwen2-0.5B, BF16)

```
per_layer = 2 × B × n_kv_heads × max_seq_len × head_dim × 2 bytes
           = 2 × 1 × 2 × 2048 × 64 × 2
           = 2MB per layer (B=1, bucket=2048)

total_kv  = 24 layers × 2MB = 48MB (B=1, max context)
```

For Qwen2-7B (n_kv_heads=8, head_dim=128):
```
per_layer = 2 × 1 × 8 × 2048 × 128 × 2 = 8MB
total_kv  = 32 × 8MB = 256MB (B=1, max context)
```

Both fit comfortably on A100 (80GB) and TPU v5e (16GB HBM).

### KV cache layout on device

KV cache **stays resident on device** across all decode steps. It is:
- Allocated once in `Session::new()` alongside weights
- Passed as input to the decode function every step
- **Donated** (in-place aliased) so PJRT can update it without a copy
- Zeroed between conversations via `PjRtBuffer::zero_fill()`

⚠️ **DESIGN DECISION — Buffer donation for in-place KV update**  
From OpenXLA discuss (Sep 2024): the correct mechanism for efficient KV cache
update in XLA/PJRT is **buffer donation** — telling PJRT to alias the input KV
buffer with the output KV buffer. XLA's buffer assignment then uses `scatter`/
`dynamic_update_slice` in-place, avoiding a full tensor copy per decode step.
Without donation, each decode step would allocate a new `[B, H, S, D]` buffer
and copy the entire cache — catastrophic for latency.

In gljax, buffer donation is expressed via `PJRT_LoadedExecutable_Execute_Args`
by setting `input_layouts` with `donated_input_indices`. See §6.

---

## 2. StableHLO for KV Cache Update

### `stablehlo.dynamic_update_slice`

This is the op that writes new K/V into the pre-allocated cache at position `t`.

```
stablehlo.dynamic_update_slice(operand, update, start_indices...)
```

- `operand`: the full KV cache `[B, H, S, D]`
- `update`: new K or V for current step `[B, H, 1, D]`
- `start_indices`: `(0, 0, t, 0)` — position `t` in the seq dimension

The `start_indices` are **runtime scalars** (i32) — this is the one "dynamic"
part. XLA handles this correctly: the index `t` is a runtime value, but the
tensor shapes remain static.

### Annotated MLIR (decode step, Qwen2-0.5B layer, B=1, H=2, S=2048, D=64)

```mlir
// kv_k_cache:  tensor<1x2x2048x64xbf16>  (full K cache for this layer)
// new_k:       tensor<1x2x1x64xbf16>     (K for current token)
// pos:         tensor<i32>               (current decode position, e.g. 42)

// Write new K into cache at position `pos`
%zero_i32 = stablehlo.constant dense<0> : tensor<i32>

%updated_kv_k = stablehlo.dynamic_update_slice
    %kv_k_cache, %new_k,
    %zero_i32, %zero_i32, %pos, %zero_i32
    : (tensor<1x2x2048x64xbf16>, tensor<1x2x1x64xbf16>,
       tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>)
    -> tensor<1x2x2048x64xbf16>
```

### `stablehlo.slice` for reading cached K/V

During decode, attention reads K/V for positions `0..bucket_size`. The slice is
**static** — we always read the full bucket-sized window. Padding positions are
masked out in the causal mask.

```mlir
// Read full K cache for attention: [1, 2, 2048, 64]
// (same as the full cache — no slice needed for full-bucket attention)
// For smaller buckets: slice is always the full bucket, mask handles padding
%k_for_attn = %updated_kv_k  // just pass the updated cache directly
```

⚠️ **DESIGN DECISION — Always attend to full bucket, mask padding**  
We could slice the K/V cache to `[B, H, t+1, D]` (only attend to real tokens),
but that requires dynamic shapes (`t` varies). Instead, we always attend to the
full bucket `[B, H, bucket_size, D]` and rely on the position mask to zero out
attention to padding positions. XLA's softmax with `-inf` mask produces zero
attention weight for masked positions — numerically correct, slightly wasteful
in compute (attending to zero-weight positions), but static-shape friendly.

---

## 3. Causal + Position Mask for Decode

The decode attention mask must express:
1. Can attend to all real tokens in positions `0..t` (including current token t)
2. Cannot attend to positions `t+1..bucket_size-1` (future padding)

This is a **position-aware mask** that changes per decode step. For static
shapes, it is computed at trace time as a function of `pos` (runtime scalar).

### Approach: precompute all masks at trace time

Rather than computing the mask dynamically (which requires dynamic shapes in
the mask itself), precompute the mask for each position at compile time by
encoding it as a function of `pos` using `stablehlo.iota` + comparison ops:

```mlir
// Causal + position mask for decode at position `pos`
// Output: [1, 1, 1, bucket_size] — broadcast across batch and heads

// Create position indices: [0, 1, 2, ..., bucket_size-1]
%idx = stablehlo.iota dim = 0 : tensor<2048xi32>
%idx_bc = stablehlo.reshape %idx : (tensor<2048xi32>) -> tensor<1x1x1x2048xi32>

// Broadcast pos to [1, 1, 1, 2048]
%pos_bc = stablehlo.broadcast_in_dim %pos, dims = []
          : (tensor<i32>) -> tensor<1x1x1x2048xi32>

// mask[j] = (j <= pos) ? 0.0 : -inf
%valid = stablehlo.compare LE, %idx_bc, %pos_bc, SIGNED
         : (tensor<1x1x1x2048xi32>, tensor<1x1x1x2048xi32>) -> tensor<1x1x1x2048xi1>

%zeros   = stablehlo.constant dense<0.000000e+00> : tensor<1x1x1x2048xbf16>
%neg_inf = stablehlo.constant dense<0xFF80> : tensor<1x1x1x2048xbf16>  // bf16 -inf

%mask = stablehlo.select %valid, %zeros, %neg_inf
        : tensor<1x1x1x2048xi1>, tensor<1x1x1x2048xbf16>
```

This produces a runtime mask from the runtime scalar `pos` — static output shape
`[1, 1, 1, bucket_size]`, dynamic values. XLA compiles this efficiently.

---

## 4. Two Compiled Functions per Bucket

### Prefill function

```
prefill_<bucket>(
    token_ids: [B, S],          // padded to bucket S
    pos_ids:   [B, S],          // [0, 1, ..., S-1, pad, pad, ...]
    weights...,                 // 219 weight tensors
) -> (
    logits: [B, S, vocab],
    kv_caches: [[B, H, S, D] × n_layers × 2]  // K and V per layer
)
```

- Processes all S tokens in parallel (standard attention)
- Outputs the filled KV cache for subsequent decode steps
- Compiled once per bucket size

### Decode function

```
decode_<bucket>(
    token_id:  [B, 1],          // single new token
    pos:       [1],             // scalar i32, current position
    kv_caches: [[B, H, S, D] × n_layers × 2],  // donated input
    weights...,
) -> (
    logits: [B, 1, vocab],
    kv_caches: [[B, H, S, D] × n_layers × 2]   // updated (aliased, in-place)
)
```

- Processes 1 token, attends to full bucket (masked)
- Returns updated KV caches (same buffers, donated/aliased)
- Compiled once per bucket size

### Session changes (ARTX4 → ARTX5)

```rust
pub struct Session {
    client:     PjRtClient,

    // ARTX5: one executable per bucket per function type
    prefill_execs: HashMap<usize, PjRtLoadedExecutable>,  // bucket_size → executable
    decode_execs:  HashMap<usize, PjRtLoadedExecutable>,

    weights:    Vec<PjRtBuffer>,    // unchanged: loaded once

    // ARTX5: KV cache buffers (per layer, per K/V)
    kv_caches:  Vec<[PjRtBuffer; 2]>,  // [layer_idx][0=K, 1=V]

    plan: ExecutionPlan,
}
```

---

## 5. Bucketing Strategy

### Bucket sizes

```rust
pub const BUCKETS: &[usize] = &[128, 256, 512, 1024, 2048];
```

Standard choice validated by production systems (JetStream, SGLang-JAX, KV-RM).
These bucket sizes:
- Cover common prompt lengths efficiently (waste ≤ 2× at worst)
- Compile in reasonable time (10 total compilations)
- Fit in CompileCache (10 `.pjrt` files)

### Bucket selection

```rust
/// Select the smallest bucket >= seq_len.
/// Returns None if seq_len > max bucket (request rejected).
pub fn select_bucket(seq_len: usize) -> Option<usize> {
    BUCKETS.iter().copied().find(|&b| b >= seq_len)
}
```

### Padding

```rust
/// Pad token_ids to bucket_size with pad_token_id.
/// Pad positions get -inf in the attention mask.
pub fn pad_tokens(tokens: &[u32], bucket: usize, pad_id: u32) -> Vec<u32> {
    let mut padded = tokens.to_vec();
    padded.resize(bucket, pad_id);
    padded
}
```

Padding token: use the model's `pad_token_id` from config.json. For Qwen2: `151643`
(the EOS token, which is standard for Qwen2 family). Padding positions are masked
to `-inf` in the position mask, so the embedding value doesn't affect output.

### Compile-time cost

On first run, all 10 executables are compiled and cached:

| Bucket | Prefill compile | Decode compile | Total |
|---|---|---|---|
| 128   | ~5s  | ~5s  | ~10s  |
| 256   | ~7s  | ~5s  | ~12s  |
| 512   | ~10s | ~5s  | ~15s  |
| 1024  | ~15s | ~5s  | ~20s  |
| 2048  | ~25s | ~5s  | ~30s  |
| **Total** | | | **~87s** |

(Estimates for CPU plugin on i3-1115G4. GPU/TPU: faster compilation, larger models.)

After the first run, all 10 artifacts are loaded from `CompileCache` in <1s each.

---

## 6. Buffer Donation for In-Place KV Update

### Why donation matters

Without donation: each decode step allocates a new `[B, H, S, D]` buffer and copies
the old cache before writing the new K/V. For Qwen2-7B at bucket=2048:
- Copy cost: 256MB × 2 (K+V) × 32 layers = 16GB of memory traffic per decode step
- At 31 GB/s bandwidth: **516ms per step** — completely unusable

With donation: `dynamic_update_slice` runs in-place. Only the new K/V slice is
written. Memory traffic: `2 × B × H × D × 2` bytes = 512 bytes per layer = 16KB total.

### PJRT buffer donation API

Buffer donation is expressed in `PJRT_LoadedExecutable_Execute_Args` via
the `executable_output_lists` aliasing. The PJRT C API doesn't have a simple
"donate" flag — instead, aliasing is declared at **compile time** in the HLO
module via `input_output_alias`:

```mlir
// In the module-level attributes:
// Declare that output[0] (kv_k, layer 0) aliases input[2] (kv_k input, layer 0)
// This tells XLA to use the same buffer for both
"input_output_alias" = {0: (2, [], may-alias), 1: (3, [], may-alias), ...}
```

In gljax, this is emitted by `FuncBuilder::finish()` when `ParamKind::KvCache`
params are declared:

```rust
// src/graph/builder.rs addition for ARTX5

pub enum ParamKind {
    Input,          // runtime input (token_ids, pos)
    Weight,         // checkpoint weight (never aliased)
    KvCache,        // ARTX5: donated buffer (aliased input=output)
}
```

The `finish()` method, upon seeing `KvCache` params, emits the `input_output_alias`
attribute pairing each KV input with its corresponding output in the MLIR module.

⚠️ **DESIGN DECISION — Alias declaration at compile time, not execute time**  
PJRT buffer donation is declared in the compiled XLA program, not at execute time.
This means the `decode` compiled function has aliasing baked in — it always
expects donated KV cache buffers. The `prefill` function does NOT alias (it creates
fresh KV outputs). This asymmetry is intentional and correct.

---

## 7. RoPE + KV Cache Integration

### Pre-rotated K (Option A — recommended)

Apply RoPE to K **before writing to cache**. The cache stores pre-rotated K vectors.

```
decode step t:
  q_rot = rope_neox(q, cos_table, sin_table, seq_offset=t)
  k_new_rot = rope_neox(k_new, cos_table, sin_table, seq_offset=t)
  kv_k = dynamic_update_slice(kv_k, k_new_rot, pos=t)
  k_full = kv_k  // [B, H, S, D] — all pre-rotated K
  scores = q_rot @ k_full^T
```

This is the approach used by llama.cpp, vLLM, and MaxText. The RoPE table slice
at position `t` is:

```mlir
// Get cos/sin for position t only (decode: one position)
%cos_t = stablehlo.dynamic_slice %cos_table, %pos, %zero
         : (tensor<2048x64xf32>, tensor<i32>, tensor<i32>)
         -> tensor<1x64xf32>
%sin_t = stablehlo.dynamic_slice %sin_table, %pos, %zero
         : (tensor<2048x64xf32>, tensor<i32>, tensor<i32>)
         -> tensor<1x64xf32>
```

`stablehlo.dynamic_slice` (not `dynamic_update_slice`) — reads a static-size slice
at a dynamic offset. For decode: slice size is `[1, head_dim]` (one position), offset
is `[pos, 0]`. Shape remains static.

### Updated `rope_neox` signature for ARTX5

```rust
// src/ops/rope.rs — updated for decode path

pub fn rope_neox(
    x: &Tensor,
    cos_table: &Tensor,   // [max_seq_len, head_dim]
    sin_table: &Tensor,
    seq_offset: RopeOffset,  // ARTX5: enum, not usize
) -> Tensor {
    match seq_offset {
        RopeOffset::Static(offset) => {
            // Prefill: static slice (compile-time offset)
            // stablehlo.slice with static limits
            rope_neox_static(x, cos_table, sin_table, offset)
        }
        RopeOffset::Dynamic(pos_tensor) => {
            // Decode: dynamic slice at runtime position
            // stablehlo.dynamic_slice
            rope_neox_dynamic(x, cos_table, sin_table, pos_tensor)
        }
    }
}

pub enum RopeOffset {
    Static(usize),          // prefill: position known at trace time
    Dynamic(Tensor),        // decode: position is runtime i32 scalar
}
```

---

## 8. Decode Attention with KV Cache

Updated `gqa_attention` for decode path:

```rust
// src/ops/attention.rs — additions for ARTX5

/// GQA attention for decode: Q is [B, H, 1, D], KV cache is [B, H_kv, S, D].
/// Returns (output [B, H, 1, D], updated_kv_k [B, H_kv, S, D], updated_kv_v)
pub fn gqa_attention_decode(
    q:      &Tensor,       // [B, n_heads, 1, head_dim]
    k_new:  &Tensor,       // [B, n_kv_heads, 1, head_dim] — pre-rotated
    v_new:  &Tensor,       // [B, n_kv_heads, 1, head_dim]
    kv_k:   &Tensor,       // [B, n_kv_heads, S, head_dim] — donated cache
    kv_v:   &Tensor,
    pos:    &Tensor,       // scalar i32
    mask:   &Tensor,       // [1, 1, 1, S] — position mask
) -> (Tensor, Tensor, Tensor) {   // (output, updated_kv_k, updated_kv_v)
    let [b, n_kv, s, d] = match kv_k.shape().dims.as_slice() {
        &[b, h, s, d] => [b, h, s, d],
        _ => panic!("kv_k must be rank 4"),
    };
    let n_heads = q.dim(1);
    let repeat = n_heads / n_kv;

    // 1. Write new K/V into cache at position `pos`
    let updated_kv_k = dynamic_update_slice_seq(kv_k, k_new, pos);
    let updated_kv_v = dynamic_update_slice_seq(kv_v, v_new, pos);

    // 2. Expand KV heads for GQA (same as prefill path)
    let k_full = if repeat > 1 {
        expand_kv_heads(&updated_kv_k, repeat, b, n_kv, s, d)
    } else {
        updated_kv_k.clone_ref()
    };
    let v_full = if repeat > 1 {
        expand_kv_heads(&updated_kv_v, repeat, b, n_kv, s, d)
    } else {
        updated_kv_v.clone_ref()
    };

    // 3. Scaled dot product: Q [B,H,1,D] × K^T [B,H,D,S] → [B,H,1,S]
    let scale = 1.0 / (d as f64).sqrt();
    let mut b_ref = q.builder().borrow_mut();
    let scale_s = b_ref.constant_scalar(scale, q.dtype());
    drop(b_ref);
    let scale_bc = scale_s.broadcast_to(vec![], q.shape().dims.clone());
    let q_scaled = q.mul(&scale_bc);
    let k_t = k_full.transpose(vec![0, 1, 3, 2]);   // [B,H,D,S]

    let scores = q_scaled.dot_general(&k_t, DotDimensionNumbers {
        lhs_batching: vec![0, 1], rhs_batching: vec![0, 1],
        lhs_contracting: vec![3], rhs_contracting: vec![2],
    });  // [B, H, 1, S]

    // 4. Add position mask + softmax
    let scores_masked = scores.add(mask);
    let attn_weights = crate::ops::softmax::softmax(&scores_masked, 3);

    // 5. AV: [B,H,1,S] × [B,H,S,D] → [B,H,1,D]
    let output = attn_weights.dot_general(&v_full, DotDimensionNumbers {
        lhs_batching: vec![0, 1], rhs_batching: vec![0, 1],
        lhs_contracting: vec![3], rhs_contracting: vec![2],
    });

    (output, updated_kv_k, updated_kv_v)
}

/// Write update [B, H, 1, D] into cache [B, H, S, D] at seq position `pos`.
fn dynamic_update_slice_seq(cache: &Tensor, update: &Tensor, pos: &Tensor) -> Tensor {
    let mut b = cache.builder().borrow_mut();
    let zero = b.constant_scalar(0i32, crate::stablehlo::types::DType::I32);
    drop(b);
    // start_indices: (0, 0, pos, 0) — update at seq dim only
    cache.dynamic_update_slice(update, &[&zero, &zero, pos, &zero])
}
```

### New `stablehlo/ops.rs` emitter for ARTX5

```rust
// emit_dynamic_update_slice
pub fn emit_dynamic_update_slice(
    e: &mut MlirEmitter,
    operand: SsaName,
    update: SsaName,
    start_indices: &[SsaName],
    operand_shape: &Shape,
) -> SsaName {
    let out = e.fresh();
    let starts = start_indices.iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    e.line(format!(
        "{out} = stablehlo.dynamic_update_slice {operand}, {update}, {starts} \
         : ({}, {}, {}) -> {}",
        operand_shape.mlir_type(),
        // update shape
        // start indices types (all i32 scalars)
        start_indices.iter().map(|_| "tensor<i32>").collect::<Vec<_>>().join(", "),
        operand_shape.mlir_type(),
    ));
    out
}

// emit_dynamic_slice
pub fn emit_dynamic_slice(
    e: &mut MlirEmitter,
    operand: SsaName,
    start_indices: &[SsaName],
    slice_sizes: &[usize],
    operand_shape: &Shape,
) -> SsaName {
    let out = e.fresh();
    let starts = start_indices.iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let sizes = slice_sizes.iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    e.line(format!(
        "{out} = stablehlo.dynamic_slice {operand}, {starts}, sizes = [{sizes}] \
         : ({}, {}) -> {}",
        operand_shape.mlir_type(),
        start_indices.iter().map(|_| "tensor<i32>").collect::<Vec<_>>().join(", "),
        // output shape: same rank, slice_sizes as dims
        Shape::new(slice_sizes.to_vec(), operand_shape.dtype).mlir_type(),
    ));
    out
}
```

---

## 9. KV Cache Lifecycle in Session

```rust
// src/runtime/session.rs — ARTX5 additions

impl Session {
    /// Initialize KV cache buffers on device (all zeros).
    /// Called once in Session::new() after weights are loaded.
    fn init_kv_caches(
        client: &PjRtClient,
        config: &ModelConfig,
        bucket: usize,
    ) -> Result<Vec<[PjRtBuffer; 2]>, SessionError> {
        let mut caches = Vec::with_capacity(config.n_layers);

        let kv_shape = Shape::new(
            vec![config.batch_size, config.n_kv_heads, bucket, config.head_dim],
            DType::BF16,
        );
        let n_bytes = kv_shape.n_elements() * 2;  // BF16 = 2 bytes
        let zeros = vec![0u8; n_bytes];

        for _ in 0..config.n_layers {
            let k_buf = client.buffer_from_host(&zeros, &kv_shape, client.default_device()?)?;
            let v_buf = client.buffer_from_host(&zeros, &kv_shape, client.default_device()?)?;
            caches.push([k_buf, v_buf]);
        }

        Ok(caches)
    }

    /// Reset KV cache between conversations (zero-fill all cache buffers).
    pub fn reset_kv_cache(&mut self) -> Result<(), SessionError> {
        // Re-transfer zeros to all KV buffers
        // In production: use PJRT_Buffer_ZeroFill if available,
        // otherwise retransfer zeros from host
        for cache in &mut self.kv_caches {
            for buf in cache.iter_mut() {
                buf.zero_fill().map_err(|e| SessionError::Execute(e.to_string()))?;
            }
        }
        Ok(())
    }

    /// Run prefill: process prompt tokens, fill KV cache, return first token logits.
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<u32, SessionError> {
        let bucket = select_bucket(tokens.len())
            .ok_or_else(|| SessionError::Execute("sequence too long".into()))?;

        let exec = self.prefill_execs.get(&bucket)
            .ok_or_else(|| SessionError::Execute(format!("no prefill exec for bucket {bucket}")))?;

        let padded = pad_tokens(tokens, bucket, self.config.pad_token_id);
        let input = HostTensor::from_i32s(&padded.iter().map(|&x| x as i32).collect::<Vec<_>>(),
                                           Shape::new([self.config.batch_size, bucket], DType::I32));

        // Prefill outputs: logits + new KV caches for all layers
        let outputs = crate::runtime::execution::run_prefill(
            &self.client, exec, &self.weights, &[input],
            &mut self.kv_caches, &self.plan, bucket,
        )?;

        // Sample next token from last position logits
        let logits = bytemuck::cast_slice::<u8, f32>(&outputs[0].data);
        let last_pos_logits = &logits[(tokens.len() - 1) * self.config.vocab_size..
                                       tokens.len() * self.config.vocab_size];
        Ok(argmax_f32(last_pos_logits) as u32)
    }

    /// Run one decode step: process one token, update KV cache, return next token.
    pub fn decode_step(&mut self, token: u32, pos: usize) -> Result<u32, SessionError> {
        let bucket = self.current_bucket;  // set during prefill

        let exec = self.decode_execs.get(&bucket)
            .ok_or_else(|| SessionError::Execute(format!("no decode exec for bucket {bucket}")))?;

        let token_input = HostTensor::from_i32s(
            &[token as i32],
            Shape::new([self.config.batch_size, 1], DType::I32),
        );
        let pos_input = HostTensor::from_i32s(
            &[pos as i32],
            Shape::new([], DType::I32),
        );

        // Decode: KV caches are donated (in-place updated)
        let outputs = crate::runtime::execution::run_decode(
            &self.client, exec, &self.weights,
            &[token_input, pos_input],
            &mut self.kv_caches,  // mutable: updated in-place
            &self.plan, bucket,
        )?;

        let logits = bytemuck::cast_slice::<u8, f32>(&outputs[0].data);
        Ok(argmax_f32(logits) as u32)
    }

    /// Full autoregressive generation.
    pub fn generate(
        &mut self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        eos_token_id: u32,
    ) -> Result<Vec<u32>, SessionError> {
        self.reset_kv_cache()?;

        // Prefill
        let mut generated = Vec::with_capacity(max_new_tokens);
        let mut next_token = self.prefill(prompt_tokens)?;
        generated.push(next_token);

        // Decode loop
        let mut pos = prompt_tokens.len();
        for _ in 1..max_new_tokens {
            if next_token == eos_token_id { break; }
            next_token = self.decode_step(next_token, pos)?;
            generated.push(next_token);
            pos += 1;
        }

        Ok(generated)
    }
}

fn argmax_f32(logits: &[f32]) -> usize {
    logits.iter().enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}
```

---

## 10. Compile All Buckets at Session Init

```rust
// src/runtime/session.rs — compile_all_buckets

impl Session {
    /// Compile or load from cache all 10 executables (5 buckets × 2 functions).
    fn compile_all_buckets(
        client: &PjRtClient,
        plugin: &PjRtPlugin,
        config: &ModelConfig,
        cache: Option<&CompileCache>,
    ) -> Result<(HashMap<usize, PjRtLoadedExecutable>, HashMap<usize, PjRtLoadedExecutable>), SessionError> {
        let mut prefill_execs = HashMap::new();
        let mut decode_execs  = HashMap::new();

        for &bucket in BUCKETS {
            // Trace prefill for this bucket
            let built_prefill = precision::with_policy(PrecisionPolicy::bf16(), || {
                trace_model_prefill(config, bucket)  // traces all layers, prefill path
            });

            // Trace decode for this bucket
            let built_decode = precision::with_policy(PrecisionPolicy::bf16(), || {
                trace_model_decode(config, bucket)   // traces all layers, decode path
            });

            let prefill_exec = match cache {
                Some(c) => c.get_or_compile(client, plugin, &built_prefill.mlir)?,
                None => client.compile(&built_prefill.mlir, plugin)?,
            };
            let decode_exec = match cache {
                Some(c) => c.get_or_compile(client, plugin, &built_decode.mlir)?,
                None => client.compile(&built_decode.mlir, plugin)?,
            };

            prefill_execs.insert(bucket, prefill_exec);
            decode_execs.insert(bucket, decode_exec);

            println!("  compiled bucket {bucket}");
        }

        Ok((prefill_execs, decode_execs))
    }
}
```

---

## 11. `examples/bench.rs` — Decode Throughput

```rust
// examples/bench.rs
// Measures decode tok/s after ARTX5

use gljax::runtime::session::Session;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let session = Session::build_from_args()?;

    // Warm up: compile all buckets, load weights
    println!("Session ready on {:?}", session.device_kind());

    let prompt = "Explain quantum entanglement in simple terms.";
    let tokens = session.tokenize(prompt)?;

    // Benchmark: 5 warmup + 5 measurement runs
    let mut decode_tps_samples = Vec::new();

    for run in 0..10 {
        session.reset_kv_cache()?;
        let t0 = Instant::now();
        let generated = session.generate(&tokens, 128, 151643)?;
        let elapsed = t0.elapsed().as_secs_f64();
        let tps = generated.len() as f64 / elapsed;

        if run >= 5 {
            decode_tps_samples.push(tps);
            println!("run {}: {:.1} tok/s ({} tokens)", run, tps, generated.len());
        }
    }

    let mean_tps: f64 = decode_tps_samples.iter().sum::<f64>() / decode_tps_samples.len() as f64;
    println!("\nDecode throughput: {:.1} tok/s (mean of 5 runs)", mean_tps);
    println!("Target: > glproc ({:.1} tok/s on same hardware)", 38.7);

    Ok(())
}
```

---

## 12. Memory Layout — `[B, H, S, D]` vs `[B, S, H, D]`

Research finding: TPU MXU prefers the contracting dimension (D) to be the last
axis. For KV cache, attention computes `Q @ K^T` where K is `[B, H, S, D]` —
the contracting dim is D (last), which is TPU-optimal.

Layout recommendation: **`[B, n_kv_heads, max_seq_len, head_dim]`** — keeps D last,
matches glproc's layout for consistency in the precision cross-check.

---

## 13. What ARTX6 Should Cover

### ARTX6 — Multi-Device Tensor Parallel + MoE Expert Sharding

1. **Tensor parallel for dense models:**
   - Column-parallel: split `gate_proj`, `up_proj`, `q_proj`, `k_proj`, `v_proj` along output dim
   - Row-parallel: split `down_proj`, `o_proj` along input dim  
   - AllReduce after row-parallel via `stablehlo.all_reduce`
   - `DeviceMesh` from `distributed/mesh.rs` (ARTX4 stub, now implemented)
   - Sharding annotations via `@mesh` + Shardy dialect

2. **MoE expert parallel (Qwen3-35B-A3B: 128 experts):**
   - Each device owns `E/N` experts (N = number of devices)
   - Token routing: all-to-all dispatch via `stablehlo.all_to_all`
   - Expert FFN: local compute on assigned experts
   - Result gather: all-to-all back to original device

3. **Multi-device Session:**
   - `PjRtClient` with multiple addressable devices
   - `PJRT_LoadedExecutable_Execute_Args.num_devices > 1`
   - NCCL integration (via XLA collective ops — not direct NCCL calls)

4. **KV cache sharding:**
   - Q heads sharded across devices, each device owns `H/N` heads
   - KV cache sharded: each device owns `H_kv/N` KV head slices

---

## Appendix: Design Decision Summary

| Decision | Choice | Rationale |
|---|---|---|
| KV cache layout | `[B, H_kv, S, D]` per layer | D last = TPU MXU optimal; matches glproc layout |
| KV cache scope | Per-layer separate K/V tensors | Simpler tracing than 6D tensor; ARTX6 can restructure |
| Write op | `dynamic_update_slice` | Cleaner than scatter for sequential position writes |
| Read op | Full bucket slice (always `[B, H, S, D]`) | Static shape; padding masked via position mask |
| Position mask | Computed from runtime `pos` via iota + compare | Dynamic mask values, static shape — XLA-friendly |
| Buffer donation | Declared at compile time via `input_output_alias` | In-place update, zero KV copy per decode step |
| Bucket sizes | `[128, 256, 512, 1024, 2048]` | Standard choice; 10 compilations cached |
| RoPE variant | Pre-rotated K stored in cache | Matches llama.cpp/vLLM; simpler than post-rotation |
| RoPE decode slice | `dynamic_slice` at position `t` | Static output shape `[1, D]`, dynamic offset |
| KV init | Zero-fill host transfer | Simplest correct init; GPU memset in ARTX6 |
| Cache reset | Re-zero between conversations | Avoids stale KV leaking across turns |
| Compile strategy | All 10 buckets at session init | Fail fast on first run; cached for subsequent |
| Argmax | Host-side after `to_host()` | Sampling logic stays in Rust; avoids compiled sampler complexity |

---

*End of ARTX5 — Static KV Cache + Bucketing Strategy*  
*Next: ARTX6 — Multi-Device Tensor Parallel + MoE Expert Sharding*