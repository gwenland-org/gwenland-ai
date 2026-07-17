# Euler Dequantisation

This is the reference for GwenLand's Euler dequantisation — a cosine-projection method for turning GGUF quantised weights into a bounded floating-point range that the inference engine is comfortable with. If you just want SafeTensors for another framework, you want standard dequantisation instead; this one is specifically for feeding GwenLand.

## The problem it solves

A GGUF model stores every tensor as compact integers (Q4_0 or Q8_0). The usual way to get floats back is linear:

$$W[i] = X_{quant}[i] \times scale$$

That's correct, but it gives you weights across whatever range the original model used. On large models `|scale|` can be many orders of magnitude bigger than 1.0, which pushes reconstructed weights well outside the range GwenLand's kernel is built for, and the accumulator overflows. Standard mode is the right answer for general SafeTensors export; Euler mode is the right answer when the destination is GwenLand inference.

## The formula

It's three steps.

**Step 1 — map each integer to an angle.** Take the quantised integer and turn it into a radian angle:

$$\theta_i = (X_{quant}[i] \times \pi) / Max\_Bound$$

`Max_Bound` is the largest absolute quantised integer *in the current block*, not the theoretical maximum for the dtype (127 for Q8_0, 7 for Q4_0). Using the per-block max normalises `θ` into `[-π, +π]` based on what's actually in each block, which keeps the relative magnitude differences between blocks.

**Step 2 — reconstruct a smooth value with cosine.** Take the real part of Euler's formula:

$$\text{Real}(e^{i\theta}) = \cos(\theta)$$

Cosine maps `θ ∈ [-π, +π]` to a smooth wave bounded in `[-1, +1]`. Discrete integers become continuous, differentiable values with no hard clipping or saturation.

**Step 3 — scale into GwenLand's range.** Multiply by the GGUF block delta and divide by the golden ratio:

$$W_{safetensor}[i] = \cos(\theta_i) \times \delta_b / \varphi$$

That lands values in `[-0.309, 0.309]`, which is where the kernel does its best work. Here `δ_b` is the f16 scale stored at the head of each GGUF block, and `φ = 1.6180339...` is `(1 + √5) / 2`. Since `1/φ ≈ 0.618`, that's the outer bound, and `0.5/φ ≈ 0.309` is the sweet spot.

So the whole thing is:

$$\theta_i = (X_{quant}[i] \times \pi) / Max\_Bound$$
$$W_{safetensor}[i] = \cos(\theta_i) \times \delta_b / \varphi$$

## The variables

- `X_quant[i]` — the stored quantised integer (i8 for Q8_0, 4-bit signed for Q4_0).
- `Max_Bound` — `max(|X_quant[k]|)` over the block. Per block, not per dtype.
- `θ_i` — the angle in `[-π, +π]` that the integer maps to.
- `δ_b` — the block delta/scale from the GGUF header (f16); the original linear reconstruction scale for that block.
- `φ` — the golden ratio, `1.6180339`. Scales the output into GwenLand's bound.
- `W_safetensor[i]` — the final f32 weight written out.

## Why these choices

**Why cosine, not sigmoid or tanh.** Sigmoid outputs `(0, 1)`, so it's always positive and throws away the sign. Tanh saturates quickly, so extreme integers map to ±1 no matter the block context. Cosine is an odd-symmetry function with the full `[-1, 1]` range and a natural period at π. The deciding property is `cos(0) = 1`. A quantised zero is the most common value in sparse weight matrices (pruned attention heads, zero-initialised LoRA), and a stored zero means "no deviation from the block centre," not "null weight" — so mapping it to the block's maximum amplitude is exactly right. Tanh and sigmoid send zero to the middle of their range and lose the block's scale.

**Why the per-block max, not the dtype max.** The theoretical max is 127 for Q8_0 and 7 for Q4_0. If you normalised against those, a block whose largest value is 3 would use the same angular range as a block whose largest value is 127, and you'd flatten the differences between blocks. The per-block max means busy blocks use the full `[-π, +π]` and quiet blocks get compressed proportionally, so the relative magnitudes survive into the output. It's the same idea as per-block quantisation itself: respect the local distribution.

**Why the golden ratio.** `φ` is the value where `1/φ = φ - 1 ≈ 0.618`. Dividing the cosine output by `φ` scales `[-1, 1]` down to `[-0.618, 0.618]`, and the inner band `[-0.309, 0.309] = [-0.5/φ, 0.5/φ]` is where the fixed-point accumulator hits its best precision — far enough from zero to avoid underflowing into noise, far enough from the edge to avoid saturating. `φ` also has no awkward rounding at common precisions, so it stays stable across f16, f32, and bf16.

**Why use δ_b as the amplitude.** `δ_b` is already the best per-block magnitude estimate around — the quantiser chose it to minimise that block's reconstruction error. Reusing it as the amplitude means high-magnitude blocks produce higher-amplitude output and near-zero blocks produce near-zero output, so the relative scale ordering of tensors is preserved, just bounded.

## Edge case: a fully pruned block

If every quantised value in a block is zero, `Max_Bound = 0` and the angle is undefined. In that case the output is just `0.0` for every element. A block of all zeros carries no information and reconstructs to zero in both modes anyway — which is the normal case for pruned heads and zero-initialised matrices.

## The numbers at a glance

- `φ = 1.6180339` (golden ratio)
- `1/φ ≈ 0.618` — outer output bound
- `0.5/φ ≈ 0.309` — sweet-spot boundary
- Output range: `[-0.618, 0.618]`
- Sweet spot: `[-0.309, 0.309]`
- `cos(0) = 1.0` — a zero integer maps to the block's max amplitude
- `cos(±π) = -1.0` — the max integer maps to the inverted max amplitude

## Where it's implemented

The block routine is `euler_dequant_block(ivalues: &[i32], delta_b: f32) -> Vec<f32>` in `packages/core/src/convert/dequant.rs`, dispatched per block by `dequant_q8_0_euler` and `dequant_q4_0_euler`. Turn it on with `gwen convert gguf <MODEL.gguf> --euler`.

## When to use which mode

Use standard when you're exporting SafeTensors for another framework, inspecting raw weight distributions, or fine-tuning from a dequanted checkpoint — it's lossless within quantisation error. Use Euler when you're loading into GwenLand inference or deploying to an embedded target, where the bounded, accumulator-safe range matters.

---

# The 10D Engine

A small, local-first neural engine written in Rust. Instead of doing floating-point tensor multiplication, it works in a binarised space and uses constant-time memory indexing, which keeps a single lookup down to a couple hundred nanoseconds. It's exploratory — a place to try out the math below — not the production inference path.

## Measured numbers

From the benchmark run:

- Initialisation: about 27.7 µs (the golden init below)
- Parallel inference, 12 chars over 10 layers: about 942 µs (multi-threaded with `rayon`)
- Single-layer core fetch: about 270 ns (O(1) strided indexing)
- Binary: 8.3 MB stripped
- Model on disk: about 41 KB (SafeTensors, 10,240 weights)

## The three formulas

Three formulas run the whole engine, and all three are anchored on `φ`.

### 1. Golden initialisation

Weights start inside the stable band `[-0.309, 0.309]`, computed from the golden ratio with no random number generator at all — just arithmetic, so it's deterministic and reproducible:

$$Factor_i = \sqrt{i} \times \varphi$$
$$W_i = \sin(Factor_i) \times \cos(Factor_i) / \varphi$$

Here `i` is the flat array index of the weight in memory. It works because `sin(x)·cos(x) = sin(2x)/2`, so the output is bounded by `[-0.5, 0.5]`, and dividing by `φ` tightens it to `[-0.309, 0.309]` — the same sweet spot the Euler formula above targets. The `√i` factor makes the angle grow slowly so the weights spread smoothly instead of repeating a tight cycle. The reason for not using `randn` or Xavier init is that those call the OS RNG, which adds non-determinism and startup latency; this is closed-form, so the same binary produces the same weights every time on every platform.

### 2. O(1) strided indexing

Coordinates in the 10D space collapse to a single flat memory address through a precomputed stride vector — no allocation, no looping over dimensions at runtime:

$$FlatIndex = \sum_{k=0}^{9} Coordinate_k \times Strides_k$$

The strides are `Strides_k = 2^(9-k)`, which is `[512, 256, 128, 64, 32, 16, 8, 4, 2, 1]`. That's just the row-major layout for a 10-dimensional binary tensor (2^10 = 1024 cells). Because they're powers of two, the CPU can do the index math with bit-shifts instead of multiplies, and the whole thing is one multiply-accumulate pass. Ten binary dimensions give 1024 addressable cells — enough for the token encoding space, and small enough to sit in L1 cache.

### 3. Sequential layer propagation

On the forward pass the signal moves through 10 layers, with a residual link at each one so it doesn't collapse to zero or run away:

$$x_{l+1} = \text{Clip}(x_l + (x_l \times W_l), -1.0, +1.0)$$

The residual term `x_l` is an identity shortcut: even if `W_l ≈ 0`, `x_{l+1} ≈ x_l`, so a near-zero weight can't make the signal decay across 10 layers the way pure multiplication would. The clip is a hard clamp rather than tanh or sigmoid because those need `exp()`, which is expensive and adds smooth saturation; a clamp is a single comparison. The `±1.0` bound works because, with weights in `[-0.309, 0.309]`, the residual can at most reach about `1.3x`, and the clamp absorbs that overshoot without touching the central `[-0.618, 0.618]` band.

## How it ties back to Euler dequant

Both halves share `φ` as their anchor. GRN weights live in `[-0.309, 0.309]`, Euler-dequanted weights land in `[-0.618, 0.618]`, and the propagation clip sits at `±1.0`. Each outer bound is exactly twice the inner one — a geometric progression rooted in `1/φ`. That nesting is deliberate.

## Stack

Pure Rust (safe, with targeted `unsafe` for raw memory mapping), `rayon` for parallelism, HuggingFace `safetensors` for zero-copy storage, cross-compiled for AMD64, ARM64, and Apple Silicon.

## Running it

```sh
cargo run --release
ls -la | grep safetensors
```

A run looks roughly like this:

```
=== GWENLAND AI MULTI-THREADED TOKENGINE ===
Teks Input Mentah : "hello world!"
Tokenizer: Sukses mengonversi 12 karakter ke Koordinat Biner 10D.

--- EXECUTING MULTI-THREADED PREDICT VIA RAYON ---
Kecepatan Total (Parallel 12 Karakter): 942.099µs
Hasil Sinyal Output Tensor: [1.0, 0.94270384, 0.7526576, ...]
Hasil Akhir Generasi Teks AI: "eh!!de!dd!rd"

Engine berhasil di-backup ke 'gwen_multilayer_10d.safetensors'.
```
