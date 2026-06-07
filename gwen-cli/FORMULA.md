# GwenLand Euler Dequantisation — Formula Specification

> *"Speed is Everything. But Precise is more than Everything."*

This document is the canonical reference for the **GwenLand Euler Dequantisation** formula —
a custom cosine-projection method for converting GGUF quantised weights into a bounded
floating-point space aligned with the GwenLand inference engine.

---

## The Problem

When a GGUF model is loaded, every tensor is stored as compact integers (Q4_0 or Q8_0).
Standard linear dequantisation recovers those integers with:


$$W[i] = X_quant[i] × scale$$


This works. But it produces weights in an *arbitrary* real range — as wide as the original
model's dynamic range. For large models, `|scale|` can be orders of magnitude larger than 1.0,
pushing reconstructed weights far outside the numeric range GwenLand's inference kernel
is designed to operate in.

**Feeding out-of-range weights into GwenLand causes accumulator overflow.**
Standard mode is correct for general-purpose SafeTensors output.
Euler mode is correct for GwenLand inference.

---

## The Formula

### Step 1 — Phase Vector Mapping (Euler Quantization Reverse)

Map the raw GGUF quantised integer `X_quant[i]` to a radian angle `θ_i`:


$$θ_i = (X_quant[i] × π) / Max_Bound$$


Where `Max_Bound` is the **absolute maximum quantised integer in the current block** —
not the theoretical dtype maximum (127 for Q8_0, 7 for Q4_0).

This normalises `θ_i` into `[-π, +π]` based on the actual distribution of values
in each block, preserving relative inter-block magnitude differences.

---

### Step 2 — Continuous Wave Reconstruction via Euler

Apply the real part of Euler's formula to recover a smooth, continuous weight value:

$$Real(e^{i·θ}) = cos(θ)$$

Which gives:

$$cos(θ_i)$$

The cosine function maps `θ ∈ [-π, +π]` to a smooth wave bounded within `[-1, +1]`.
This is where the "continuous wave restoration" happens — discrete integers become smooth,
differentiable values without hard clipping or saturation artifacts.

---

### Step 3 — GwenLand Precision Parameter Restoration

Scale by the GGUF block delta `δ_b` and divide by the Golden Ratio `φ` to land values
directly in the GwenLand sweet spot `[-0.309, 0.309]`:

$$W_{safetensor}[i] = (cos(θ_i) × δ_b) / φ$$

Where:
- `δ_b` — the f16 scale stored at the head of each GGUF block
- `φ = 1.6180339...` — the Golden Ratio `(1 + √5) / 2`
- `1/φ ≈ 0.618` — the outer bound of the output range
- `0.5/φ ≈ 0.309` — the sweet spot boundary

---

## Complete Formula

$$θ_i            = (X_{quant}[i] × π) / Max_Bound$$

$$W_{safetensor}[i] = cos(θ_i) × δ_b / φ$$

---

## Variable Reference

| Symbol | Name | Source | Meaning |
|---|---|---|---|
| $$X_{quant}[i]$$ | Quantised integer | GGUF raw data | The stored integer value (i8 for Q8_0, 4-bit signed for Q4_0) |
| $$Max_Bound$$ | Per-block maximum | Computed per block | `max(|X_quant[k]|)` over all elements in the block |
| $$θ_i$$ | Phase angle | Computed | Maps the integer to a radian in `[-π, +π]` |
| $$δ_b$$ | Block delta / scale | GGUF block header (f16) | The original linear reconstruction scale for this block |
| $$φ$$ | Golden Ratio | Constant `1.6180339` | `(1 + √5) / 2` — scales the output into the GwenLand bound |
| $$W_{safetensor}[i]$$ | Restored weight | Output | The final f32 weight value written to SafeTensors |

---

## Why These Choices

### Why cosine — not sigmoid, not tanh?

| Function | Problem |
|---|---|
| $$sigmoid(x)$$ | Output is `(0, 1)` — always positive, destroys sign information |
| $$tanh(x)$$ | Saturates fast; extreme integers map to ±1 regardless of block context |
| $$cos(θ)$$ | Odd-function symmetry, full `[-1,1]` range, natural periodicity at π |

Cosine has one decisive property for this use case: **`cos(0) = 1`**.

A quantised value of zero is the most common value in sparse weight matrices
(pruned attention heads, zero-initialised LoRA adapters). With cosine projection,
zero maps to the maximum reconstruction amplitude of the block — which is exactly
correct, because a stored zero means "no deviation from block centre," not "null weight."

With tanh or sigmoid, zero maps to the midpoint of the output range, discarding the
block's scale information.

### Why per-block `Max_Bound` instead of the dtype theoretical maximum?

The theoretical maximum for Q8_0 is 127. The theoretical maximum for Q4_0 is 7.
Using these would normalise every block identically — a block with a max value of 3
would use the same $$θ$$ range as a block with a max value of 127.

Using the **per-block maximum** means:
- Blocks with large activations use the full $$[-π, +π]$$ angular range
- Blocks with small activations are compressed proportionally
- The relative magnitude relationships between blocks are preserved in the output

This is the same philosophy as per-block quantisation itself: honour the local
distribution, not a global assumption.

### Why the Golden Ratio?

`φ = 1.618...` is the unique value where `1/φ = φ - 1 ≈ 0.618`.

Dividing by `φ` scales the cosine output from `[-1, 1]` to `[-0.618, 0.618]`.
The inner sweet spot `[-0.309, 0.309]` = `[-0.5/φ, 0.5/φ]` is the range where
GwenLand's fixed-point accumulator achieves maximum dot-product precision —
neither underflowing into noise nor overflowing into saturation.

`φ` also has no special floating-point rounding behaviour at common precisions,
making it a numerically stable divisor across f16, f32, and bf16 accumulation paths.

### Why δ_b (the GGUF block scale) as the amplitude?

`δ_b` is already the best available per-block magnitude estimate — it was chosen
by the original quantisation process to minimise reconstruction error for that block.
Rather than discarding this information and treating all blocks equally, Euler mode
reuses it as the amplitude modulator.

This means:
- Blocks quantised from high-magnitude weights produce higher-amplitude Euler outputs
- Blocks quantised from near-zero weights produce near-zero Euler outputs
- The relative scale ordering of tensors is preserved, just bounded

---

## Edge Case: Fully Pruned Block (`Max_Bound = 0`)

If every quantised value in a block is zero, `Max_Bound = 0` and the angle formula
is undefined (division by zero).

**Decision:** output `0.0` for every element in the block.

A block of all-zero quantised values carries no information. Its reconstruction is
identically zero in both standard and Euler modes. This is the common case for
pruned attention heads and zero-initialised weight matrices.

```
If Max_Bound == 0:  W_safetensor[i] = 0.0  for all i in block
```

---

## Numeric Bounds Summary

| Expression | Value | Meaning |
|---|---|---|
| $$φ$$ | $$1.6180339$$ | Golden Ratio |
| $$1/φ$$ | $$≈ 0.618$$ | Outer output bound |
| $$0.5/φ$$ | $$≈ 0.309$$ | Sweet spot boundary |
| Output range | $$[-0.618, 0.618]$$ | Maximum possible reconstructed weight |
| Sweet spot | $$[-0.309, 0.309]$$ | Optimal precision region for GwenLand kernel |
| $$cos(0)$$ | $$1.0$$ | Zero quantised value → maximum block amplitude |
| $$cos(±π)$$ | $$-1.0$$ | Maximum quantised value → inverted maximum amplitude |

---

## Implementation Reference

The formula is implemented in:

```
gwen-cli/packages/core/src/convert/dequant.rs
  └── fn euler_dequant_block(ivalues: &[i32], delta_b: f32) -> Vec<f32>
```

Invoked for each quantisation block by:
- `fn dequant_q8_0_euler` — Q8_0 block dispatch
- `fn dequant_q4_0_euler` — Q4_0 block dispatch

Activated via: `gwen convert gguf <MODEL.gguf> --euler`

---

## Standard vs Euler — When to Use Which

| Scenario | Use |
|---|---|
| General SafeTensors export for other frameworks | `Standard` (lossless within quantisation error) |
| Loading into GwenLand inference engine | `Euler` (bounded, accumulator-safe) |
| Inspecting raw weight distributions | `Standard` |
| Fine-tuning from a dequanted checkpoint | `Standard` |
| Deploying on GwenLand embedded targets | `Euler` |

---

*Prototype formula by the GwenLand author. Implementation: GwenLand Cycle 3.*

---


# GwenLand 10D Engine

> *"While others use Python, we use Rust. Not because it's easy — because it's precise."*

An ultra-efficient, local-first Multi-Layer Neural Engine written natively in Rust.
GwenLand achieves sub-microsecond inference by bypassing heavy matrix operations and
high-level runtimes entirely — leveraging constant-time binary layout indexing instead.

---

## Performance Highlights (Empirical Benchmarks)

| Metric | Result | Method |
|---|---|---|
| Initialization time | `~27.7 µs` | Golden Initialization (GRN) |
| Parallel inference (12 chars, 10 layers) | `~942.0 µs` | Multi-threaded `rayon` |
| Single-layer core fetch | `~270 ns` | O(1) strided coordinate mapping |
| Binary footprint | `8.3 MB` | Stripped release binary |
| Model storage | `~41 KB` | SafeTensors, 10,240 weights |

---

## Mathematical Architecture

GwenLand replaces traditional floating-point tensor multiplication with a binarized
space framework. Three formulas govern the entire engine.

---

### Formula 1 — Golden Initialization (GRN)

Weights are initialized inside the stable geometric sweet spot `[-0.309, 0.309]` using
the Golden Ratio `φ`. This prevents vanishing and exploding gradients without any
external random number generator — pure arithmetic, deterministic, reproducible.

```
Factor_i = sqrt(i) × φ

W_i = sin(Factor_i) × cos(Factor_i) / φ
```

Where:
- `φ = 1.6180339887` — the Golden Ratio `(1 + √5) / 2`
- `i` — the flat array index of the weight in memory (continuous, zero-based)

**Why this works:** `sin(x) × cos(x) = sin(2x) / 2`, so the output is bounded by
`[-0.5, 0.5]`. Dividing by `φ ≈ 1.618` tightens it further to `[-0.309, 0.309]` —
exactly the GwenLand sweet spot established in the Euler Dequantisation formula above.
The `sqrt(i)` factor ensures the angular argument grows slowly, distributing weights
smoothly across the initialization space rather than repeating a tight periodic cycle.

**Why not `torch.randn` or `torch.xavier`:** Those call the OS RNG, introducing
non-determinism and startup latency. GRN is a closed-form computation — no RNG,
no syscall, no variance across platforms. Same binary, same weights, every time.

---

### Formula 2 — O(1) Strided Coordinate Mapping

Multi-dimensional coordinates in 10D space are collapsed to a single flat hardware
memory address via a precomputed stride vector. No dynamic allocation, no loop over
dimensions at runtime.

```
FlatIndex = Σ (k=0 to 9)  Coordinate_k × Strides_k
```

Where `Strides_k = 2^(9-k)`, generating the binary sequence:

```
Strides = [512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
```

This is a bitwise dot product. The CPU computes it in a single multiply-accumulate
pass — `O(1)` in practice because the stride vector has fixed length 10.

**Why powers of 2:** The stride sequence is the standard row-major layout for a
`[2, 2, 2, 2, 2, 2, 2, 2, 2, 2]` tensor (10 binary dimensions, 2^10 = 1024 cells).
Powers of 2 allow the CPU to replace multiplications with bit-shifts, shaving cycles
on every single coordinate lookup. At `~270 ns` per fetch, this is measurable.

**Why 10D:** 10 binary dimensions gives 1024 addressable cells — enough expressivity
for the token encoding space while fitting entirely in L1 cache on any modern CPU.

---

### Formula 3 — Sequential Layer Propagation (ResNet-Style)

During forward pass (`predict`), the input signal node propagates through 10 hidden
tensor layers. A hard residual link at each layer prevents the signal from collapsing
to zero (vanishing) or exploding beyond the representable range.

```
x_(l+1) = Clip( x_l + (x_l × W_l),  -1.0, +1.0 )
```

Where:
- `x_l` — the signal value entering layer `l`
- `W_l` — the weight at the coordinate address for layer `l`
- `Clip(v, -1, +1)` — hard clamp, implemented as `v.clamp(-1.0, 1.0)` in Rust

**Why residual addition (`x + x×W`) instead of pure multiplication (`x×W`):**
Pure multiplication with weights near zero causes the signal to decay exponentially
across 10 layers. The residual term `x_l` is an identity shortcut — even if `W_l ≈ 0`,
`x_(l+1) ≈ x_l`. The signal survives.

**Why hard clip instead of tanh/sigmoid:** Tanh and sigmoid require `exp()`, which is
expensive and introduces smooth saturation. The hard clip is a single comparison
instruction — no transcendental math, no approximation error, constant time.

**Why ±1.0 as the clip bound:** Combined with the GRN initialization bound of
`[-0.309, 0.309]`, the residual term `x + x×W` can at most reach `x(1 + 0.309) ≈ 1.3x`.
The ±1.0 clip absorbs this overshoot cleanly without losing precision in the central
`[-0.618, 0.618]` band that the Euler dequant formula is also designed around.

---

## Connection to the Euler Dequantisation Formula

The 10D Engine and the Euler Dequantisation formula share the same numeric anchor: `φ`.

| Formula | Role of φ |
|---|---|
| Euler Dequant | Divisor — scales cosine output to `[-0.618, 0.618]` |
| Golden Initialization | Divisor — scales `sin×cos` to `[-0.309, 0.309]` |
| Layer Propagation | Implicit — GRN weights stay within the Euler sweet spot |

This is intentional. Weights initialized by GRN live in `[-0.309, 0.309]`. Weights
loaded via Euler dequant land in `[-0.618, 0.618]`. The layer propagation clip is at
`±1.0`. The three bounds form a nested structure where each outer bound is exactly `2×`
the inner one — a geometric progression rooted in `1/φ`.

---

## Hardware and Software Stack

| Layer | Technology |
|---|---|
| Language | Pure Rust (safe + targeted unsafe for raw memory mapping) |
| Parallelism | `rayon` work-stealing parallel iterator |
| Storage | HuggingFace `safetensors` (zero-copy binary format) |
| Target architectures | AMD64, ARM64, Apple Silicon (cross-compiled by default) |

---

## How to Run

```sh
# Compile and run with maximum optimisation
cargo run --release

# Inspect the auto-generated weights file
ls -la | grep safetensors
```

Expected output (from the screenshot benchmark):

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

---

*GwenLand 10D Engine — exploratory high-performance AI tooling. GwenLand Cycle 3.*
