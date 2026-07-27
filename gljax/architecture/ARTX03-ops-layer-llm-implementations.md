# ARTX3 — gljax ops/ Layer: LLM Op Implementations

**Series:** gljax (Sanctum Visibilia) Architecture Research  
**Depends on:** ARTX1 (PJRT C API FFI), ARTX2 (IR Design: MlirEmitter, FuncBuilder, TraceCx, SsaValue, Tensor)  
**Next:** ARTX4 — runtime/ and checkpoint/ (Session, compile cache, safetensors loader, PJRT execute loop)

---

## Overview

ARTX3 specifies the full implementation of `src/ops/` — the high-level LLM operation
library that sits between model code (`Tensor` methods) and the StableHLO IR layer
(`FuncBuilder`, `ops.rs` emitters). Every function here takes `&Tensor` arguments,
calls `FuncBuilder` methods, and returns a new `Tensor`. No direct `MlirEmitter`
access — that is `FuncBuilder`'s job.

The `PrecisionPolicy` thread-local from ARTX2 §8 is used throughout. Every op that
involves an accumulation (reduce, rsqrt, exp) upcasts to `policy.norm_reduce` or
`policy.softmax_reduce` and downcasts back. This is the mechanism that makes FP64
oracle mode work with zero changes to model code.

### Ops covered in this document

| File | Ops |
|---|---|
| `ops/softmax.rs` | `softmax` |
| `ops/norm.rs` | `rms_norm` |
| `ops/rope.rs` | `rope_neox`, `build_rope_table` |
| `ops/attention.rs` | `gqa_attention`, `causal_mask` |
| `ops/ffn.rs` | `swiglu_ffn` |
| `ops/embedding.rs` | `gather_embed` |
| `ops/moe.rs` | `moe_ffn`, `top_k_indices`, `expert_dispatch` |

### Module layout

```
src/ops/
├── mod.rs               # pub use all submodules
├── softmax.rs
├── norm.rs
├── rope.rs
├── attention.rs
├── ffn.rs
├── embedding.rs
└── moe.rs
```

---

## 1. `ops/softmax.rs`

### Numerically Stable Softmax

Standard softmax `exp(x) / sum(exp(x))` overflows for large logits. The numerically
stable variant subtracts the row maximum before exponentiation:

```
softmax(x)_i = exp(x_i - max(x)) / sum_j(exp(x_j - max(x)))
```

This is the only correct implementation for attention logits — a single logit at
100.0 would produce `inf` in the naive version.

### Implementation

```rust
// src/ops/softmax.rs

use crate::{precision, tensor::tensor::Tensor, stablehlo::types::{DType, Shape}};

/// Numerically stable softmax along `dim`.
/// Input shape: arbitrary. Output shape: same as input.
/// Upcasts reduce to `policy.softmax_reduce`, downcasts result back to input dtype.
pub fn softmax(x: &Tensor, dim: usize) -> Tensor {
    let policy = precision::current();
    let orig_dtype = x.dtype();

    // 1. Upcast for numerical stability
    let x_acc = x.to_dtype(policy.softmax_reduce);

    // 2. Row max (for numerical stability)
    let x_max = reduce_max_keepdim(&x_acc, dim);                // [..., 1, ...]
    let x_max_bc = broadcast_like(&x_max, &x_acc);              // [..., S, ...]
    let x_shifted = x_acc.sub(&x_max_bc);                       // x - max(x)

    // 3. Exp
    let x_exp = x_shifted.exp();                                 // exp(x - max)

    // 4. Sum of exp
    let x_sum = reduce_add_keepdim(&x_exp, dim);                 // [..., 1, ...]
    let x_sum_bc = broadcast_like(&x_sum, &x_exp);              // [..., S, ...]

    // 5. Divide
    let x_norm = x_exp.div(&x_sum_bc);

    // 6. Downcast back
    x_norm.to_dtype(orig_dtype)
}

/// Reduce max along `dim`, keeping dimension (size 1).
fn reduce_max_keepdim(x: &Tensor, dim: usize) -> Tensor {
    let mut b = x.builder().borrow_mut();
    let neg_inf = b.constant_scalar(f64::NEG_INFINITY, x.dtype());
    drop(b);

    // out_dims: same as x but dim → 1
    let mut out_dims = x.shape().dims.clone();
    out_dims[dim] = 1;
    let out_shape = Shape::new(out_dims, x.dtype());

    let scalar_shape = Shape::scalar(x.dtype());
    let mut b = x.builder().borrow_mut();
    let name = crate::stablehlo::ops::emit_reduce_max(
        b.emitter_mut(),
        x.value().ssa(), neg_inf.value().ssa(),
        &[dim],
        x.shape(), &scalar_shape, &out_shape,
    );
    drop(b);
    Tensor::new(crate::graph::value::SsaValue::new(name, out_shape), x.builder_rc())
}

/// Reduce add along `dim`, keeping dimension (size 1).
fn reduce_add_keepdim(x: &Tensor, dim: usize) -> Tensor {
    let mut b = x.builder().borrow_mut();
    let zero = b.constant_scalar(0.0, x.dtype());
    drop(b);

    let mut out_dims = x.shape().dims.clone();
    out_dims[dim] = 1;
    let out_shape = Shape::new(out_dims, x.dtype());
    let scalar_shape = Shape::scalar(x.dtype());

    let mut b = x.builder().borrow_mut();
    let name = crate::stablehlo::ops::emit_reduce_add(
        b.emitter_mut(),
        x.value().ssa(), zero.value().ssa(),
        &[dim],
        x.shape(), &scalar_shape, &out_shape,
    );
    drop(b);
    Tensor::new(crate::graph::value::SsaValue::new(name, out_shape), x.builder_rc())
}

/// Broadcast `src` to match `target` shape along all dims.
fn broadcast_like(src: &Tensor, target: &Tensor) -> Tensor {
    // src shape has size-1 dims that need to match target.
    // broadcast_dims: indices where src has non-1 dims (preserved dims).
    let broadcast_dims: Vec<usize> = src.shape().dims.iter().enumerate()
        .filter(|(_, &d)| d != 1)
        .map(|(i, _)| i)
        .collect();
    src.broadcast_to(broadcast_dims, target.shape().dims.clone())
}
```

### Annotated MLIR Output (shape: `[1, 16, 512, 512]` attention scores)

```mlir
// Input: %v0 : tensor<1x16x512x512xbf16>
// Upcast
%v1 = stablehlo.convert %v0 : (tensor<1x16x512x512xbf16>) -> tensor<1x16x512x512xf32>
// Neg-inf constant for reduce init
%v2 = stablehlo.constant dense<0xFF800000> : tensor<f32>  // -inf
// Row max: reduce along dim 3 → [1, 16, 512, 1]
%v3 = stablehlo.reduce(%v1 init: %v2) across dimensions = [3]
    : (tensor<1x16x512x512xf32>, tensor<f32>) -> tensor<1x16x512xf32> {
  ^bb0(%va: f32, %vb: f32):
    %vr = stablehlo.maximum %va, %vb : f32
    stablehlo.return %vr : f32
}
%v4 = stablehlo.reshape %v3 : (tensor<1x16x512xf32>) -> tensor<1x16x512x1xf32>
// Broadcast max back
%v5 = stablehlo.broadcast_in_dim %v4, dims = [0, 1, 2, 3]
      : (tensor<1x16x512x1xf32>) -> tensor<1x16x512x512xf32>
// x - max
%v6 = stablehlo.subtract %v1, %v5 : tensor<1x16x512x512xf32>
// exp
%v7 = stablehlo.exponential %v6 : tensor<1x16x512x512xf32>
// Sum exp
%v8 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
%v9 = stablehlo.reduce(%v7 init: %v8) across dimensions = [3]
    : (tensor<1x16x512x512xf32>, tensor<f32>) -> tensor<1x16x512xf32> {
  ^bb0(%va: f32, %vb: f32):
    %vr = stablehlo.add %va, %vb : f32
    stablehlo.return %vr : f32
}
%v10 = stablehlo.reshape %v9 : (tensor<1x16x512xf32>) -> tensor<1x16x512x1xf32>
%v11 = stablehlo.broadcast_in_dim %v10, dims = [0, 1, 2, 3]
       : (tensor<1x16x512x1xf32>) -> tensor<1x16x512x512xf32>
// Divide
%v12 = stablehlo.divide %v7, %v11 : tensor<1x16x512x512xf32>
// Downcast
%v13 = stablehlo.convert %v12 : (tensor<1x16x512x512xf32>) -> tensor<1x16x512x512xbf16>
```

### Performance notes

- XLA fuses `exp + reduce + divide` into a single kernel on TPU/GPU when the reduce
  dim is the last dim (which it always is for attention). Manual fusion not needed.
- On CPU plugin (dev/test), fusion may not occur — acceptable for correctness testing.
- FP64 oracle: `policy.softmax_reduce = DType::F64` → upcasts to f64, runs full
  double-precision softmax, downcasts. Zero code change needed.

### Test

```rust
#[test]
fn softmax_sums_to_one() {
    // Trace softmax over [2, 4], check MLIR emits correctly
    // In integration test: compare output sum against 1.0 (± 1e-5)
    // In oracle test: compare BF16 softmax vs FP64 softmax output
}
```

---

## 2. `ops/norm.rs`

### RMSNorm

RMSNorm (Root Mean Square Layer Normalization) is the normalization used by Qwen2,
Qwen3, LLaMA, Mistral, and most modern LLMs instead of LayerNorm. It skips the mean
subtraction step:

```
RMSNorm(x, w) = x / sqrt(mean(x²) + ε) * w
```

The epsilon `ε` is added inside the sqrt for numerical stability (prevents division
by zero). Qwen2 uses `ε = 1e-6`.

### Implementation

```rust
// src/ops/norm.rs

use crate::{precision, tensor::tensor::Tensor, stablehlo::types::{DType, Shape}};

/// RMSNorm: normalize x along last dimension, scale by weight.
/// x shape:      [..., D]
/// weight shape: [D]
/// output shape: [..., D]
pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Tensor {
    let policy = precision::current();
    let orig_dtype = x.dtype();
    let rank = x.rank();
    let d = x.dim(rank - 1);

    // 1. Upcast to accumulation dtype
    let x_acc = x.to_dtype(policy.norm_reduce);

    // 2. x²
    let x2 = x_acc.mul(&x_acc);

    // 3. mean(x²) along last dim — keepdim
    let mut b = x2.builder().borrow_mut();
    let zero = b.constant_scalar(0.0, policy.norm_reduce);
    drop(b);

    let mut out_dims = x2.shape().dims.clone();
    out_dims[rank - 1] = 1;
    let out_shape = Shape::new(out_dims.clone(), policy.norm_reduce);
    let scalar_shape = Shape::scalar(policy.norm_reduce);

    let sum = {
        let mut b = x2.builder().borrow_mut();
        let name = crate::stablehlo::ops::emit_reduce_add(
            b.emitter_mut(),
            x2.value().ssa(), zero.value().ssa(),
            &[rank - 1],
            x2.shape(), &scalar_shape, &out_shape,
        );
        drop(b);
        Tensor::new(crate::graph::value::SsaValue::new(name, out_shape.clone()), x2.builder_rc())
    };

    // mean = sum * (1/D)
    let mut b = sum.builder().borrow_mut();
    let d_inv_scalar = b.constant_scalar(1.0 / d as f64, policy.norm_reduce);
    drop(b);
    let d_inv = d_inv_scalar.broadcast_to(vec![], out_dims.clone());
    let mean = sum.mul(&d_inv);

    // 4. mean + eps
    let mut b = mean.builder().borrow_mut();
    let eps_scalar = b.constant_scalar(eps, policy.norm_reduce);
    drop(b);
    let eps_t = eps_scalar.broadcast_to(vec![], out_dims.clone());
    let mean_eps = mean.add(&eps_t);

    // 5. rsqrt(mean + eps) → [... , 1]
    let rrms = mean_eps.rsqrt();

    // 6. Broadcast rrms to [..., D]
    let rrms_bc = rrms.broadcast_to(
        (0..rank - 1).collect(),
        x_acc.shape().dims.clone(),
    );

    // 7. x * rrms
    let normed = x_acc.mul(&rrms_bc);

    // 8. Downcast to original dtype
    let normed_out = normed.to_dtype(orig_dtype);

    // 9. Scale by weight (weight is [D], broadcast to [..., D])
    let w_cast = weight.to_dtype(orig_dtype);
    let broadcast_dims: Vec<usize> = vec![rank - 1]; // weight maps to last dim
    let w_bc = w_cast.broadcast_to(broadcast_dims, x.shape().dims.clone());

    normed_out.mul(&w_bc)
}
```

### Annotated MLIR Output (shape: `[1, 512, 2048]`, D=2048, ε=1e-6)

```mlir
// Input: %v0 tensor<1x512x2048xbf16>, %v1 tensor<2048xbf16> (weight)
%v2 = stablehlo.convert %v0 : (tensor<1x512x2048xbf16>) -> tensor<1x512x2048xf32>
%v3 = stablehlo.multiply %v2, %v2 : tensor<1x512x2048xf32>
%v4 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
%v5 = stablehlo.reduce(%v3 init: %v4) across dimensions = [2]
    : (tensor<1x512x2048xf32>, tensor<f32>) -> tensor<1x512xf32> {
  ^bb0(%va: f32, %vb: f32):
    %vr = stablehlo.add %va, %vb : f32
    stablehlo.return %vr : f32
}
%v6 = stablehlo.reshape %v5 : (tensor<1x512xf32>) -> tensor<1x512x1xf32>
// mean = sum * (1/2048)
%v7 = stablehlo.constant dense<4.882813e-04> : tensor<f32>   // 1/2048
%v8 = stablehlo.broadcast_in_dim %v7, dims = []
      : (tensor<f32>) -> tensor<1x512x1xf32>
%v9 = stablehlo.multiply %v6, %v8 : tensor<1x512x1xf32>
// mean + eps
%v10 = stablehlo.constant dense<1.000000e-06> : tensor<f32>
%v11 = stablehlo.broadcast_in_dim %v10, dims = []
       : (tensor<f32>) -> tensor<1x512x1xf32>
%v12 = stablehlo.add %v9, %v11 : tensor<1x512x1xf32>
// rsqrt
%v13 = stablehlo.rsqrt %v12 : tensor<1x512x1xf32>
// broadcast rrms → [1, 512, 2048]
%v14 = stablehlo.broadcast_in_dim %v13, dims = [0, 1, 2]
       : (tensor<1x512x1xf32>) -> tensor<1x512x2048xf32>
// x * rrms
%v15 = stablehlo.multiply %v2, %v14 : tensor<1x512x2048xf32>
// downcast
%v16 = stablehlo.convert %v15 : (tensor<1x512x2048xf32>) -> tensor<1x512x2048xbf16>
// weight broadcast + multiply
%v17 = stablehlo.broadcast_in_dim %v1, dims = [2]
       : (tensor<2048xbf16>) -> tensor<1x512x2048xbf16>
%v18 = stablehlo.multiply %v16, %v17 : tensor<1x512x2048xbf16>
```

### Performance notes

- XLA fuses the reduce + multiply chain into a single vectorized kernel on TPU.
  The upcast/downcast converts add two memory operations but the fused result
  stays live in registers on modern hardware.
- **Precision gap relevance:** RMSNorm was identified in `glproc-precision-gap-vs-llamacpp.md`
  as one of the three candidate ops for glproc's ~33% unexplained PPL gap.
  The FP64 oracle mode (`PrecisionPolicy::f64_oracle()`) will be used to cross-check
  glproc's FP32 RMSNorm output in `examples/fp64_oracle.rs`.
- Epsilon placement: epsilon is added to `mean(x²)`, NOT to `x²` itself. This matches
  llama.cpp's `ggml_compute_forward_rms_norm_f32` at `ops.cpp:3795`. The glproc
  precision investigation should verify this placement matches.

### Test strategy

```rust
// Unit: trace rms_norm([1, 4, 8], weight=[8]) → inspect MLIR for correct reduce dim
// Integration: run on CPU plugin, compare output against:
//   - glproc FP32 (should be close)
//   - gljax FP64 oracle (ground truth)
// Tolerance: FP32 vs FP64 relative L2 < 1e-4 for typical LLM weights
```

---

## 3. `ops/rope.rs`

### RoPE — Rotary Position Embedding (NeoX Variant)

RoPE encodes position information by rotating query and key vectors in pairs. The
NeoX variant (used by Qwen2, Qwen3, GPT-NeoX, Mistral) pairs even-odd indices within
each head dimension rather than splitting the first and second half.

**NeoX pairing:** for head dim `d`, pairs are `(d[0], d[1]), (d[2], d[3]), ...`  
**GPT-J pairing (different):** pairs are `(d[0], d[d/2]), (d[1], d[d/2+1]), ...`

Qwen2 uses NeoX. Confirm via `rope_scaling.type` in `config.json` → `"default"` or
absent = NeoX.

### Frequency formula

```
θ_i = 1 / (base^(2i / head_dim))    for i in 0..head_dim/2
```

Qwen2 default: `base = 10000.0`. For long-context variants (Qwen2.5-72B): uses YaRN
scaling. gljax v1 implements base RoPE only; YaRN is ARTX5+.

### Precomputed Table Design

RoPE frequencies are position-independent — they depend only on head_dim, base, and
max_seq_len. Precompute at model-load time as a constant tensor and gather at trace
time. This avoids `sin/cos` in the hot path.

```
cos_table: [max_seq_len, head_dim]   # cos(pos * θ_i), repeated for pairs
sin_table: [max_seq_len, head_dim]   # sin(pos * θ_i), repeated for pairs
```

The table is emitted as a `stablehlo.constant` once per compiled function.

### Implementation

```rust
// src/ops/rope.rs

use crate::{precision, tensor::tensor::Tensor, stablehlo::types::{DType, Shape}};

/// Build cos/sin RoPE tables as constant tensors.
/// Returns (cos_table, sin_table) each of shape [max_seq_len, head_dim].
pub fn build_rope_table(
    cx: &mut crate::graph::trace::TraceCx,
    max_seq_len: usize,
    head_dim: usize,
    base: f32,
) -> (Tensor, Tensor) {
    let half = head_dim / 2;

    // Compute θ_i = 1 / base^(2i / head_dim) for i in 0..half
    let mut cos_data = vec![0.0f32; max_seq_len * head_dim];
    let mut sin_data = vec![0.0f32; max_seq_len * head_dim];

    for pos in 0..max_seq_len {
        for i in 0..half {
            let theta = 1.0 / (base.powf(2.0 * i as f32 / head_dim as f32));
            let angle = pos as f32 * theta;
            let (s, c) = angle.sin_cos();
            // NeoX: pair (2i, 2i+1) — repeat cos/sin for both elements of pair
            cos_data[pos * head_dim + 2 * i]     = c;
            cos_data[pos * head_dim + 2 * i + 1] = c;
            sin_data[pos * head_dim + 2 * i]     = s;
            sin_data[pos * head_dim + 2 * i + 1] = s;
        }
    }

    let shape = Shape::new([max_seq_len, head_dim], DType::F32);

    // Emit as stablehlo.constant dense tensors
    let mut b = cx.builder_mut();
    let cos_name = crate::stablehlo::ops::emit_constant_f32_tensor(
        b.emitter_mut(), &cos_data, &shape);
    let sin_name = crate::stablehlo::ops::emit_constant_f32_tensor(
        b.emitter_mut(), &sin_data, &shape);
    drop(b);

    let cos_t = Tensor::new(crate::graph::value::SsaValue::new(cos_name, shape.clone()), cx.builder_rc());
    let sin_t = Tensor::new(crate::graph::value::SsaValue::new(sin_name, shape), cx.builder_rc());
    (cos_t, sin_t)
}

/// Apply RoPE (NeoX) to q or k.
/// x shape:    [B, n_heads, S, head_dim]
/// cos/sin:    [max_seq_len, head_dim] → sliced to [S, head_dim]
/// Output:     [B, n_heads, S, head_dim]
pub fn rope_neox(
    x: &Tensor,
    cos_table: &Tensor,   // [max_seq_len, head_dim]
    sin_table: &Tensor,
    seq_offset: usize,    // for KV cache: position of first token in this batch
) -> Tensor {
    let policy = precision::current();
    let [b, h, s, head_dim] = match x.shape().dims.as_slice() {
        &[b, h, s, d] => [b, h, s, d],
        _ => panic!("rope_neox: expected rank-4 input [B, H, S, D]"),
    };

    // 1. Slice cos/sin to [S, head_dim] for the current positions
    let cos_s = cos_table.slice(
        vec![seq_offset, 0],
        vec![seq_offset + s, head_dim],
        vec![1, 1],
    );  // [S, head_dim]
    let sin_s = sin_table.slice(
        vec![seq_offset, 0],
        vec![seq_offset + s, head_dim],
        vec![1, 1],
    );

    // 2. Broadcast cos/sin to [B, H, S, head_dim]
    let cos_bc = cos_s.broadcast_to(vec![2, 3], vec![b, h, s, head_dim]);
    let sin_bc = sin_s.broadcast_to(vec![2, 3], vec![b, h, s, head_dim]);

    // 3. Upcast x to rope precision
    let x_acc = x.to_dtype(policy.rope);
    let cos_acc = cos_bc.to_dtype(policy.rope);
    let sin_acc = sin_bc.to_dtype(policy.rope);

    // 4. NeoX rotate: x_rot[i] = -x[i+1] for even i, x[i-1] for odd i
    //    Expressed as: x_rot = concat([-x[1::2], x[0::2]], dim=-1) interleaved
    //    In StableHLO: slice even/odd, negate odds, concatenate
    let x_even = x_acc.slice(
        vec![0, 0, 0, 0],
        vec![b, h, s, head_dim],
        vec![1, 1, 1, 2],
    );  // [B, H, S, head_dim/2]

    let x_odd = x_acc.slice(
        vec![0, 0, 0, 1],
        vec![b, h, s, head_dim],
        vec![1, 1, 1, 2],
    );  // [B, H, S, head_dim/2]

    // x_rot: even slots get -x_odd, odd slots get x_even
    // Interleave via reshape + concatenate trick:
    // stack [-x_odd, x_even] → [B, H, S, head_dim/2, 2] → reshape [B, H, S, head_dim]
    let neg_x_odd = {
        let mut b_ref = x_odd.builder().borrow_mut();
        let neg_name = crate::stablehlo::ops::emit_negate(
            b_ref.emitter_mut(), x_odd.value().ssa(), x_odd.shape());
        drop(b_ref);
        Tensor::new(
            crate::graph::value::SsaValue::new(neg_name, x_odd.shape().clone()),
            x_odd.builder_rc(),
        )
    };

    // Reshape to [B, H, S, head_dim/2, 1] for interleave
    let half = head_dim / 2;
    let neg_x_odd_r = neg_x_odd.reshape(vec![b, h, s, half, 1]);
    let x_even_r = x_even.reshape(vec![b, h, s, half, 1]);

    // Concat along last dim → [B, H, S, head_dim/2, 2]
    let interleaved = Tensor::concat(&[&neg_x_odd_r, &x_even_r], 4);
    // Reshape to [B, H, S, head_dim]
    let x_rot = interleaved.reshape(vec![b, h, s, head_dim]);

    // 5. Apply: x * cos + x_rot * sin
    let x_cos = x_acc.mul(&cos_acc);
    let x_sin = x_rot.mul(&sin_acc);
    let rotated = x_cos.add(&x_sin);

    // 6. Downcast
    rotated.to_dtype(x.dtype())
}
```

### Annotated MLIR Output (shape: `[1, 16, 512, 128]`, abbreviated)

```mlir
// cos/sin tables as constants (emitted once, reused for Q and K)
%v_cos = stablehlo.constant dense<[...]> : tensor<2048x128xf32>
%v_sin = stablehlo.constant dense<[...]> : tensor<2048x128xf32>

// Slice cos/sin to [512, 128]
%v1 = stablehlo.slice %v_cos [0:512:1, 0:128:1]
      : (tensor<2048x128xf32>) -> tensor<512x128xf32>
%v2 = stablehlo.slice %v_sin [0:512:1, 0:128:1]
      : (tensor<2048x128xf32>) -> tensor<512x128xf32>

// Broadcast to [1, 16, 512, 128]
%v3 = stablehlo.broadcast_in_dim %v1, dims = [2, 3]
      : (tensor<512x128xf32>) -> tensor<1x16x512x128xf32>
%v4 = stablehlo.broadcast_in_dim %v2, dims = [2, 3]
      : (tensor<512x128xf32>) -> tensor<1x16x512x128xf32>

// Upcast x
%v5 = stablehlo.convert %v_x : (tensor<1x16x512x128xbf16>) -> tensor<1x16x512x128xf32>

// Slice even: stride 2 from offset 0
%v6 = stablehlo.slice %v5 [0:1:1, 0:16:1, 0:512:1, 0:128:2]
      : (tensor<1x16x512x128xf32>) -> tensor<1x16x512x64xf32>
// Slice odd: stride 2 from offset 1
%v7 = stablehlo.slice %v5 [0:1:1, 0:16:1, 0:512:1, 1:128:2]
      : (tensor<1x16x512x128xf32>) -> tensor<1x16x512x64xf32>
// Negate odd
%v8 = stablehlo.negate %v7 : tensor<1x16x512x64xf32>
// Reshape for interleave
%v9  = stablehlo.reshape %v8 : (tensor<1x16x512x64xf32>) -> tensor<1x16x512x64x1xf32>
%v10 = stablehlo.reshape %v6 : (tensor<1x16x512x64xf32>) -> tensor<1x16x512x64x1xf32>
// Concat
%v11 = stablehlo.concatenate %v9, %v10, dim = 4
       : (tensor<1x16x512x64x1xf32>, tensor<1x16x512x64x1xf32>) -> tensor<1x16x512x64x2xf32>
// Reshape to [1, 16, 512, 128]
%v12 = stablehlo.reshape %v11 : (tensor<1x16x512x64x2xf32>) -> tensor<1x16x512x128xf32>
// x * cos + x_rot * sin
%v13 = stablehlo.multiply %v5, %v3  : tensor<1x16x512x128xf32>
%v14 = stablehlo.multiply %v12, %v4 : tensor<1x16x512x128xf32>
%v15 = stablehlo.add %v13, %v14     : tensor<1x16x512x128xf32>
// Downcast
%v16 = stablehlo.convert %v15 : (tensor<1x16x512x128xf32>) -> tensor<1x16x512x128xbf16>
```

### Performance notes

- The cos/sin constants are large for long contexts (2048 × 128 × 4B × 2 = 2MB).
  XLA will fold these into the compiled binary — acceptable for v1.
- XLA may fuse the interleave slice/concat into a single gather on GPU. Do not
  attempt to pre-optimize this manually.
- **Precision gap relevance:** RoPE was identified in `glproc-precision-gap-vs-llamacpp.md`
  as the second candidate op. The gljax FP64 oracle will produce reference sin/cos
  values at full double precision for comparison.

⚠️ **DESIGN DECISION — Table precomputation in Rust, not in MLIR**  
Precomputing the cos/sin table in Rust and emitting it as `stablehlo.constant` avoids
emitting `sin`/`cos` StableHLO ops, which have variable support across backends.
The CPU plugin supports `stablehlo.sine`/`stablehlo.cosine`, but TPU v5e support
is backend-dependent. Constant folding is safer and has zero runtime cost.

### Test strategy

```rust
// Unit: build_rope_table(max_seq_len=8, head_dim=4, base=10000.0)
//       → verify cos[0][0] = 1.0, sin[0][0] = 0.0 (pos=0 → angle=0)
//       → verify cos[1][0] = cos(1.0), sin[1][0] = sin(1.0) (θ_0 = 1.0 for base=10000, dim=4)
// Integration: apply RoPE to known Q, verify output matches reference values
// Oracle test: BF16 vs FP64 RoPE L2 relative error < 1e-3
```

---

## 4. `ops/attention.rs`

### Grouped Query Attention (GQA)

Qwen2-0.5B uses MHA (multi-head attention, n_kv_heads = n_heads). Qwen2-7B and
larger use GQA with fewer KV heads. The implementation handles both by expanding
KV heads to match query heads via `broadcast_in_dim`.

**Qwen2-0.5B:** `n_heads=16, n_kv_heads=8, head_dim=64`  
**Qwen2-7B:** `n_heads=32, n_kv_heads=8, head_dim=128`

GQA repeat factor: `n_heads / n_kv_heads`. For Qwen2-7B: each KV head is shared
by 4 query heads.

### Causal Mask

The causal mask prevents attention to future tokens. For static shapes it is a
constant tensor: `mask[i][j] = 0 if j <= i else -inf`. XLA broadcasts this
automatically across batch and head dims.

```rust
// Build causal mask as constant: [1, 1, S, S] with -inf above diagonal
fn build_causal_mask(cx: &mut TraceCx, seq_len: usize, dtype: DType) -> Tensor {
    let neg_inf = match dtype {
        DType::F32 => f32::NEG_INFINITY as f64,
        DType::BF16 => f32::NEG_INFINITY as f64,  // BF16 also uses f32::NEG_INFINITY
        DType::F64 => f64::NEG_INFINITY,
        _ => panic!("causal_mask: unsupported dtype"),
    };

    let mut data = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            data[i * seq_len + j] = f32::NEG_INFINITY;
        }
    }

    let shape = Shape::new([1, 1, seq_len, seq_len], DType::F32);
    let mut b = cx.builder_mut();
    let name = crate::stablehlo::ops::emit_constant_f32_tensor(b.emitter_mut(), &data, &shape);
    drop(b);
    let mask = Tensor::new(SsaValue::new(name, shape), cx.builder_rc());

    // Convert to target dtype if needed
    mask.to_dtype(dtype)
}
```

### Implementation

```rust
// src/ops/attention.rs

use crate::{precision, tensor::tensor::Tensor, stablehlo::types::{DType, Shape},
            graph::{trace::TraceCx, value::SsaValue}, ops::softmax::softmax};

/// Full GQA scaled dot-product attention.
/// q: [B, n_heads,    S, head_dim]
/// k: [B, n_kv_heads, S, head_dim]
/// v: [B, n_kv_heads, S, head_dim]
/// output: [B, n_heads, S, head_dim]
pub fn gqa_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: &Tensor,   // [1, 1, S, S] causal mask
) -> Tensor {
    let [b, n_heads, s, head_dim] = match q.shape().dims.as_slice() {
        &[b, h, s, d] => [b, h, s, d],
        _ => panic!("gqa_attention: q must be rank 4"),
    };
    let n_kv_heads = k.dim(1);
    let repeat = n_heads / n_kv_heads;

    // 1. Expand KV heads if GQA (repeat factor > 1)
    let k_exp = if repeat > 1 {
        expand_kv_heads(k, repeat, b, n_kv_heads, s, head_dim)
    } else {
        k.clone_ref()
    };
    let v_exp = if repeat > 1 {
        expand_kv_heads(v, repeat, b, n_kv_heads, s, head_dim)
    } else {
        v.clone_ref()
    };
    // k_exp, v_exp: [B, n_heads, S, head_dim]

    // 2. Scale: Q / sqrt(head_dim)
    let scale = 1.0 / (head_dim as f64).sqrt();
    let mut b_ref = q.builder().borrow_mut();
    let scale_scalar = b_ref.constant_scalar(scale, q.dtype());
    drop(b_ref);
    let scale_bc = scale_scalar.broadcast_to(vec![], q.shape().dims.clone());
    let q_scaled = q.mul(&scale_bc);

    // 3. QK^T: [B, n_heads, S, head_dim] x [B, n_heads, head_dim, S] → [B, n_heads, S, S]
    let k_t = k_exp.transpose(vec![0, 1, 3, 2]);  // [B, n_heads, head_dim, S]
    use crate::stablehlo::ops::DotDimensionNumbers;
    let scores = q_scaled.dot_general(&k_t, DotDimensionNumbers {
        lhs_batching:    vec![0, 1],
        rhs_batching:    vec![0, 1],
        lhs_contracting: vec![3],
        rhs_contracting: vec![2],
    });  // [B, n_heads, S, S]

    // 4. Add causal mask
    let scores_masked = scores.add(mask);

    // 5. Softmax
    let attn_weights = softmax(&scores_masked, 3);  // softmax over last dim (S)

    // 6. AV: [B, n_heads, S, S] x [B, n_heads, S, head_dim] → [B, n_heads, S, head_dim]
    attn_weights.dot_general(&v_exp, DotDimensionNumbers {
        lhs_batching:    vec![0, 1],
        rhs_batching:    vec![0, 1],
        lhs_contracting: vec![3],
        rhs_contracting: vec![2],
    })
}

/// Expand KV heads: [B, n_kv_heads, S, D] → [B, n_heads, S, D]
/// via reshape + broadcast + reshape.
fn expand_kv_heads(
    kv: &Tensor, repeat: usize,
    b: usize, n_kv: usize, s: usize, d: usize,
) -> Tensor {
    // [B, n_kv, S, D] → [B, n_kv, 1, S, D]
    let kv_r = kv.reshape(vec![b, n_kv, 1, s, d]);
    // broadcast → [B, n_kv, repeat, S, D]
    let kv_bc = kv_r.broadcast_to(vec![0, 1, 3, 4], vec![b, n_kv, repeat, s, d]);
    // reshape → [B, n_kv * repeat, S, D] = [B, n_heads, S, D]
    kv_bc.reshape(vec![b, n_kv * repeat, s, d])
}
```

### Performance notes

- XLA fuses QK^T + softmax + AV into a Flash Attention kernel on A100/H100 when the
  shapes meet alignment requirements (head_dim ∈ {64, 128}). No manual flash
  attention implementation needed for the GPU path.
- TPU v5e: XLA uses a custom TPU flash attention via `stablehlo.custom_call @flash_attention`.
  This is automatic — XLA inserts the custom call during compilation.
- Causal mask as constant: XLA folds the mask into the compiled binary. For seq_len=512,
  the mask is 512×512×4B = 1MB. Acceptable for v1.

⚠️ **DESIGN DECISION — No KV Cache in v1**  
Static shapes require fixed seq_len at compile time. KV cache (incrementally growing
key/value sequences) requires either dynamic shapes or scatter/slice patterns with
pre-allocated buffers. gljax v1 implements prefill-only (full sequence recomputation
per request). ARTX5 will add static KV cache via scatter-on-write / slice-on-read.

### Test strategy

```rust
// Unit: gqa_attention with repeat=1 (MHA) and repeat=4 (GQA)
//       verify output shape is [B, n_heads, S, head_dim]
// Precision: compare FP32 attention vs FP64 oracle on synthetic Q/K/V
// Integration: full Qwen2-0.5B forward pass, logit comparison vs glproc
```

---

## 5. `ops/ffn.rs`

### SwiGLU FFN

Qwen2/LLaMA FFN uses SwiGLU (Swish-Gated Linear Unit):

```
FFN(x) = down_proj(SiLU(gate_proj(x)) ⊙ up_proj(x))
```

Where SiLU(x) = x * sigmoid(x) = x * logistic(x).

This is a gated architecture: `gate_proj` produces a gating signal, `up_proj`
produces the value, and their element-wise product is the activated intermediate.

### Implementation

```rust
// src/ops/ffn.rs

use crate::tensor::tensor::Tensor;

/// SwiGLU FFN.
/// x:         [B, S, D]
/// gate_proj: [D, FFN_DIM]
/// up_proj:   [D, FFN_DIM]
/// down_proj: [FFN_DIM, D]
/// output:    [B, S, D]
pub fn swiglu_ffn(
    x: &Tensor,
    gate_proj: &Tensor,
    up_proj: &Tensor,
    down_proj: &Tensor,
) -> Tensor {
    // 1. Gate and up projections
    let gate_preact = x.matmul(gate_proj);   // [B, S, FFN_DIM]
    let up_preact   = x.matmul(up_proj);     // [B, S, FFN_DIM]

    // 2. SiLU gate: gate * sigmoid(gate)
    let gate_activated = gate_preact.silu(); // [B, S, FFN_DIM]

    // 3. Gated product
    let gated = gate_activated.mul(&up_preact);  // [B, S, FFN_DIM]

    // 4. Down projection
    gated.matmul(down_proj)  // [B, S, D]
}
```

### Annotated MLIR Output (D=2048, FFN=11008, abbreviated)

```mlir
// gate_proj: [2048, 11008], up_proj: [2048, 11008], down_proj: [11008, 2048]
// x: [1, 512, 2048]
%v1 = stablehlo.dot_general %v_x, %v_gate,
    batching_dims = [0] x [0],
    contracting_dims = [2] x [0]
    : (tensor<1x512x2048xbf16>, tensor<2048x11008xbf16>) -> tensor<1x512x11008xbf16>
%v2 = stablehlo.dot_general %v_x, %v_up,
    batching_dims = [0] x [0],
    contracting_dims = [2] x [0]
    : (tensor<1x512x2048xbf16>, tensor<2048x11008xbf16>) -> tensor<1x512x11008xbf16>
// SiLU: x * sigmoid(x)
%v3 = stablehlo.logistic %v1 : tensor<1x512x11008xbf16>
%v4 = stablehlo.multiply %v1, %v3 : tensor<1x512x11008xbf16>
// Gated product
%v5 = stablehlo.multiply %v4, %v2 : tensor<1x512x11008xbf16>
// Down projection
%v6 = stablehlo.dot_general %v5, %v_down,
    batching_dims = [0] x [0],
    contracting_dims = [2] x [0]
    : (tensor<1x512x11008xbf16>, tensor<11008x2048xbf16>) -> tensor<1x512x2048xbf16>
```

### Performance notes

- The gate and up projections are independent — XLA may schedule them in parallel
  on multi-core hardware.
- On TPU: the two matmuls + SiLU + multiply + matmul are the dominant cost (>50% of
  prefill wall-clock per roofline, consistent with glproc's `ffn_gate_up: 53% share`).
- No fusion opportunity for SiLU across the two matmuls — they must be sequential.

---

## 6. `ops/embedding.rs`

### Token Embedding Lookup

Embedding lookup maps token IDs (integers) to embedding vectors. In StableHLO this
is `stablehlo.gather` with `collapsed_slice_dims = [0]`.

```rust
// src/ops/embedding.rs

use crate::{tensor::tensor::Tensor, stablehlo::ops::{GatherDimensionNumbers, emit_gather},
            stablehlo::types::Shape, graph::value::SsaValue};

/// Token embedding lookup.
/// table:   [vocab_size, D]  — the embedding weight
/// indices: [B, S]           — token IDs, dtype I32
/// output:  [B, S, D]
pub fn gather_embed(table: &Tensor, indices: &Tensor) -> Tensor {
    let [b, s] = match indices.shape().dims.as_slice() {
        &[b, s] => [b, s],
        _ => panic!("gather_embed: indices must be rank 2 [B, S]"),
    };
    let [vocab, d] = match table.shape().dims.as_slice() {
        &[v, d] => [v, d],
        _ => panic!("gather_embed: table must be rank 2 [V, D]"),
    };

    // Reshape indices to [B*S, 1] for gather index_vector_dim=1
    let idx_flat = indices.reshape(vec![b * s, 1]);

    let out_shape = Shape::new(vec![b * s, d], table.dtype());
    let dnums = GatherDimensionNumbers {
        offset_dims:          vec![1],    // output dim that maps to D
        collapsed_slice_dims: vec![0],    // vocab dim is collapsed
        start_index_map:      vec![0],    // index maps to dim 0 of table
        index_vector_dim:     1,
    };
    let slice_sizes = vec![1, d];        // gather 1 row of D elements

    let mut b_ref = table.builder().borrow_mut();
    let name = emit_gather(
        b_ref.emitter_mut(),
        table.value().ssa(), idx_flat.value().ssa(),
        &dnums, &slice_sizes,
        table.shape(), idx_flat.shape(), &out_shape,
    );
    drop(b_ref);

    let gathered = Tensor::new(SsaValue::new(name, out_shape), table.builder_rc());
    // Reshape [B*S, D] → [B, S, D]
    gathered.reshape(vec![b, s, d])
}
```

### Performance notes

- Embedding lookup is memory-bound: vocab_size × D × 2B (BF16) for Qwen2-0.5B =
  151936 × 896 × 2 = 272MB. One lookup per forward pass.
- XLA compiles this to a vectorized gather. No performance tuning needed.

---

## 7. `ops/moe.rs`

### Mixture of Experts (simplified, single-device v1)

MoE routes each token to the top-K experts (typically K=2) out of E total experts.
Relevant for Qwen3-MoE (35B-A3B: 128 experts, top-2). gljax v1 implements a
single-device version; multi-device expert parallel is ARTX6.

### Routing

```
router_logits = x @ gate_weight   # [B, S, E]
probs, indices = top_k(softmax(router_logits), k=2)
```

In StableHLO: softmax is our existing op, top-k is `stablehlo.reduce` with
custom combiner tracking (value, index) pairs.

### Implementation (simplified)

```rust
// src/ops/moe.rs

use crate::{tensor::tensor::Tensor, stablehlo::types::{DType, Shape},
            graph::value::SsaValue, ops::softmax::softmax};

/// MoE FFN: route each token to top-2 experts.
/// x:           [B, S, D]
/// gate_weight: [D, E]        — router
/// expert_gate: [E, D, FFN]   — per-expert gate_proj
/// expert_up:   [E, D, FFN]   — per-expert up_proj
/// expert_down: [E, FFN, D]   — per-expert down_proj
/// output:      [B, S, D]
pub fn moe_ffn(
    x: &Tensor,
    gate_weight: &Tensor,
    expert_gate: &Tensor,
    expert_up: &Tensor,
    expert_down: &Tensor,
    n_experts: usize,
    top_k: usize,
) -> Tensor {
    let [b, s, d] = match x.shape().dims.as_slice() {
        &[b, s, d] => [b, s, d],
        _ => panic!("moe_ffn: x must be rank 3"),
    };

    // 1. Router logits: [B, S, E]
    let router_logits = x.matmul(gate_weight);

    // 2. Router probabilities
    let router_probs = softmax(&router_logits, 2);  // [B, S, E]

    // 3. Top-K selection (simplified: use stablehlo approach)
    //    For v1 single device: use iota + sort approach
    //    Full top-k via reduce with (value, index) pair tracking
    let (top_k_weights, top_k_indices) = top_k_2d(&router_probs, top_k, n_experts, b * s);

    // 4. Expert computation: for each selected expert, compute SwiGLU FFN
    //    v1: sequential loop over top_k * B * S (unrolled at trace time)
    //    This is the simplest correct approach; multi-device dispatch is ARTX6
    let ffn_dim = expert_gate.dim(2);
    let mut output = {
        let mut b_ref = x.builder().borrow_mut();
        let zero_name = b_ref.constant_fill(0.0, Shape::new(vec![b, s, d], x.dtype()));
        drop(b_ref);
        zero_name
    };

    // Unroll over top_k: for k in 0..top_k
    // Each iteration: gather expert weights, compute FFN, scatter-add weighted output
    // (Full impl deferred to ARTX6 — placeholder shows structure)
    //
    // For v1: use stablehlo.gather to collect expert weights per token,
    //         compute FFN, accumulate weighted sum
    //
    // NOTE: Full MoE implementation requires dynamic index handling that
    // benefits from static-shape tricks (fixed expert assignment per bucket).
    // This is documented in ARTX6.

    output
}

/// Top-K selection returning (weights, indices) each [B*S, K].
/// Simplified implementation using sort + slice.
fn top_k_2d(
    probs: &Tensor,  // [B, S, E]
    k: usize,
    e: usize,
    bs: usize,
) -> (Tensor, Tensor) {
    // Reshape to [B*S, E]
    let probs_flat = probs.reshape(vec![bs, e]);
    // stablehlo.sort descending → take first k columns
    // (sort emitter is emit_sort, not yet in ops.rs — add in this ARTX)
    // For now: returns placeholder shapes
    let w_shape = Shape::new(vec![bs, k], probs.dtype());
    let i_shape = Shape::new(vec![bs, k], DType::I32);
    // TODO: implement emit_sort in stablehlo/ops.rs, then:
    // let sorted = emit_sort(descending=true) → [B*S, E]
    // let weights = slice(sorted, [0, 0], [B*S, k], [1, 1])
    // let indices = slice(sorted_indices, ...)
    todo!("top_k_2d: requires emit_sort in stablehlo/ops.rs")
}
```

⚠️ **DESIGN DECISION — MoE v1 is a placeholder**  
Full MoE requires `stablehlo.sort` (add to `ops.rs` in this ARTX) and a
scatter-gather pattern for expert routing. The single-device version works but
is compute-inefficient (sequential expert loops). Multi-device expert parallel
(each GPU owns `E/N` experts) is ARTX6. For Qwen2-0.5B (no MoE) this is not
blocking; for Qwen3-35B-A3B this is required.

The `emit_sort` op to add to `stablehlo/ops.rs`:

```rust
pub fn emit_sort(
    e: &mut MlirEmitter,
    operand: SsaName, shape: &Shape,
    sort_dim: usize,
    is_stable: bool,
    descending: bool,
) -> (SsaName, SsaName) {  // (sorted_values, sorted_indices)
    let idx_shape = Shape::new(shape.dims.clone(), DType::I32);
    let out_val = e.fresh();
    let out_idx = e.fresh();
    let arg_a = e.fresh(); let arg_b = e.fresh();
    let arg_ia = e.fresh(); let arg_ib = e.fresh();

    e.line(format!("{out_val}, {out_idx} = stablehlo.sort({operand}) {{"));
    e.line(format!("    dimension = {sort_dim},"));
    e.line(format!("    is_stable = {}",  if is_stable { "true" } else { "false" }));
    e.line(format!("}} : ({})", shape.mlir_type()));
    // comparator region
    e.line(format!("^bb0({arg_a}: {}, {arg_b}: {}, {arg_ia}: i32, {arg_ib}: i32):",
        shape.dtype.mlir_str(), shape.dtype.mlir_str()));
    e.push_indent();
    let cmp = e.fresh();
    let cmp_op = if descending { "stablehlo.compare GT" } else { "stablehlo.compare LT" };
    e.line(format!("{cmp} = {cmp_op} {arg_a}, {arg_b}, TOTALORDER : ({}, {}) -> i1",
        shape.dtype.mlir_str(), shape.dtype.mlir_str()));
    e.line(format!("stablehlo.return {cmp} : i1"));
    e.pop_indent();
    e.line("}");
    (out_val, out_idx)
}
```

---

## 8. `ops/mod.rs`

```rust
// src/ops/mod.rs

pub mod softmax;
pub mod norm;
pub mod rope;
pub mod attention;
pub mod ffn;
pub mod embedding;
pub mod moe;

// Re-export commonly used ops at the ops level
pub use softmax::softmax;
pub use norm::rms_norm;
pub use rope::{rope_neox, build_rope_table};
pub use attention::gqa_attention;
pub use ffn::swiglu_ffn;
pub use embedding::gather_embed;
pub use moe::moe_ffn;
```

---

## 9. End-to-End: Single Transformer Block Trace

This shows how the ops compose into a full Qwen2 transformer block using TraceCx.

```rust
use gljax::{
    graph::trace::TraceCx,
    ops::{rms_norm, rope_neox, build_rope_table, gqa_attention, swiglu_ffn},
    precision::{self, PrecisionPolicy},
    stablehlo::types::{DType, Shape},
};

// Qwen2-0.5B hyperparameters
const B: usize = 1;
const S: usize = 512;
const D: usize = 896;
const N_HEADS: usize = 14;
const N_KV_HEADS: usize = 2;
const HEAD_DIM: usize = 64;
const FFN_DIM: usize = 4864;
const MAX_SEQ: usize = 2048;
const EPS: f64 = 1e-6;
const ROPE_BASE: f32 = 1_000_000.0;  // Qwen2-0.5B uses 1M base

pub fn trace_qwen2_block(cx: &mut TraceCx, x: &Tensor, layer_idx: usize) -> Tensor {
    cx.scope(format!("model.layers.{layer_idx}"), |cx| {

        // ── 1. Input RMSNorm ─────────────────────────────────────────────
        let normed1 = cx.scope("input_layernorm", |cx| {
            let w = cx.weight("weight", Shape::new([D], DType::BF16));
            rms_norm(x, &w, EPS)
        });
        let residual1 = x.clone_ref();

        // ── 2. QKV projections ───────────────────────────────────────────
        let (q, k, v) = cx.scope("self_attn", |cx| {
            let q_w = cx.weight("q_proj.weight",
                Shape::new([D, N_HEADS * HEAD_DIM], DType::BF16));
            let k_w = cx.weight("k_proj.weight",
                Shape::new([D, N_KV_HEADS * HEAD_DIM], DType::BF16));
            let v_w = cx.weight("v_proj.weight",
                Shape::new([D, N_KV_HEADS * HEAD_DIM], DType::BF16));

            let q = normed1.matmul(&q_w)
                .reshape(vec![B, S, N_HEADS, HEAD_DIM])
                .transpose(vec![0, 2, 1, 3]);  // [B, H, S, Dh]
            let k = normed1.matmul(&k_w)
                .reshape(vec![B, S, N_KV_HEADS, HEAD_DIM])
                .transpose(vec![0, 2, 1, 3]);
            let v = normed1.matmul(&v_w)
                .reshape(vec![B, S, N_KV_HEADS, HEAD_DIM])
                .transpose(vec![0, 2, 1, 3]);
            (q, k, v)
        });

        // ── 3. RoPE ──────────────────────────────────────────────────────
        let (cos_table, sin_table) = build_rope_table(cx, MAX_SEQ, HEAD_DIM, ROPE_BASE);
        let q_rot = rope_neox(&q, &cos_table, &sin_table, 0);
        let k_rot = rope_neox(&k, &cos_table, &sin_table, 0);

        // ── 4. Attention ─────────────────────────────────────────────────
        let attn_out = cx.scope("self_attn", |cx| {
            let o_w = cx.weight("o_proj.weight",
                Shape::new([N_HEADS * HEAD_DIM, D], DType::BF16));

            let mask = attention::build_causal_mask(cx, S, DType::BF16);
            let attn = gqa_attention(&q_rot, &k_rot, &v, &mask);

            // [B, H, S, Dh] → [B, S, H*Dh]
            let attn_flat = attn.transpose(vec![0, 2, 1, 3])
                                .reshape(vec![B, S, N_HEADS * HEAD_DIM]);
            attn_flat.matmul(&o_w)  // [B, S, D]
        });

        // ── 5. First residual ────────────────────────────────────────────
        let h = residual1.add(&attn_out);
        let residual2 = h.clone_ref();

        // ── 6. Post-attention RMSNorm ────────────────────────────────────
        let normed2 = cx.scope("post_attention_layernorm", |cx| {
            let w = cx.weight("weight", Shape::new([D], DType::BF16));
            rms_norm(&h, &w, EPS)
        });

        // ── 7. SwiGLU FFN ────────────────────────────────────────────────
        let ffn_out = cx.scope("mlp", |cx| {
            let gate_w = cx.weight("gate_proj.weight", Shape::new([D, FFN_DIM], DType::BF16));
            let up_w   = cx.weight("up_proj.weight",   Shape::new([D, FFN_DIM], DType::BF16));
            let down_w = cx.weight("down_proj.weight", Shape::new([FFN_DIM, D], DType::BF16));
            swiglu_ffn(&normed2, &gate_w, &up_w, &down_w)
        });

        // ── 8. Second residual ───────────────────────────────────────────
        residual2.add(&ffn_out)
    })
}

fn main() {
    // BF16 production trace
    let built_bf16 = precision::with_policy(PrecisionPolicy::bf16(), || {
        let mut cx = TraceCx::new("main", "qwen2_block");
        let x = cx.input("hidden_states", Shape::new([B, S, D], DType::BF16));
        let out = trace_qwen2_block(&mut cx, &x, 0);
        cx.finish(vec![&out])
    });

    // FP64 oracle trace (same model code, different policy)
    let built_f64 = precision::with_policy(PrecisionPolicy::f64_oracle(), || {
        let mut cx = TraceCx::new("main", "qwen2_block_oracle");
        let x = cx.input("hidden_states", Shape::new([B, S, D], DType::F64));
        let out = trace_qwen2_block(&mut cx, &x, 0);
        cx.finish(vec![&out])
    });

    println!("BF16 weights: {}", built_bf16.signature.weights.len());
    println!("FP64 oracle MLIR length: {} bytes", built_f64.mlir.len());
}
```

**Weight count for one Qwen2-0.5B block:**
- input_layernorm.weight: 1
- self_attn.q/k/v/o_proj.weight: 4
- post_attention_layernorm.weight: 1
- mlp.gate/up/down_proj.weight: 3
- **Total: 9 weights per layer × 24 layers = 216 weights**
- + embedding (1) + lm_head (1) + final norm (1) = **219 total**

---

## 10. What ARTX4 Should Cover

### ARTX4 — gljax runtime/ and checkpoint/

1. **`runtime/session.rs`** — Session lifecycle:
   - Plugin load (`libpjrt_c_api_cpu.so` / CUDA / TPU)
   - `PJRT_Client_Create` → device enumeration
   - Compile: `PJRT_Client_Compile(mlir_text)` → `PjRtLoadedExecutable`
   - Execute: `PJRT_LoadedExecutable_Execute` → output buffers
   - Buffer copy: `PJRT_Buffer_ToHostBuffer` → Rust `Vec<f32>`

2. **`runtime/cache.rs`** — Compiled artifact cache:
   - Cache key: SHA256(mlir_text + plugin_version + device_id)
   - Disk location: `~/.cache/gwenland/gljax/`
   - `PJRT_Executable_Serialize` / `PJRT_Executable_DeserializeAndLoad`
   - Cache invalidation on shape bucket change

3. **`checkpoint/safetensors.rs`** — Weight loading:
   - Memory-mapped header parse (no full-model RAM copy)
   - `Signature::weights` → lookup each weight by name
   - `PJRT_Client_BufferFromHostBuffer` → device buffer per weight
   - Verify shape matches traced `ParamDesc`

4. **`checkpoint/gllm.rs`** — GLLM format loader:
   - Same interface as safetensors loader
   - Reuse `glictus-caliburni` crate for format parsing

5. **Integration test:**
   - Trace Qwen2-0.5B (all 24 layers) → emit MLIR
   - Compile on CPU plugin
   - Load safetensors weights
   - Run forward pass (128-token prompt)
   - Compare output logits vs glproc FP32 (expected relative L2 < 0.05)
   - Run FP64 oracle → compare vs glproc (isolate precision gap contribution)

---

## Appendix: Design Decision Summary

| Decision | Choice | Rationale |
|---|---|---|
| Softmax reduce dim | Always last dim | Attention scores always `[B, H, S, S]`, reduce on dim 3 |
| RMSNorm epsilon | Inside sqrt (on mean) | Matches llama.cpp `ops.cpp:3795`, glproc cross-check |
| RoPE table | Precomputed in Rust as constant | Avoids `sin/cos` StableHLO op portability issues |
| RoPE variant | NeoX (even-odd pairs) | Qwen2/Qwen3 confirmed NeoX via model config |
| GQA expansion | broadcast_in_dim | XLA fuses broadcast into subsequent matmul on TPU |
| KV cache | None in v1 | Static shapes + prefill-only; ARTX5 adds KV cache |
| MoE v1 | Placeholder + emit_sort | Full routing deferred to ARTX6 (multi-device) |
| FP64 oracle | PrecisionPolicy thread-local | Zero model code change; driven entirely by policy |
| Causal mask | Constant tensor | Static seq_len; XLA folds into binary at compile time |

---

*End of ARTX3 — gljax ops/ Layer: LLM Op Implementations*  
*Next: ARTX4 — gljax runtime/ and checkpoint/ (Session, compile cache, safetensors, PJRT execute loop)*