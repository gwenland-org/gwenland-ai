# GGUF Quantization Types

> **Domain:** gguf-skills
> **Applies to:** `glcore` format layer; consumed by every engine's dequant/repack
> **Last updated:** 2026-07-17
>
> ⚠️ **Upstream drift warning:** quant block layouts are **ggml's** to define.
> The byte counts and layouts below match the ggml spec as of the date above.
> ggml regularly adds types (I-quants, new K-variants) and could revise
> details between GGUF versions — when adding a type or chasing a mismatch,
> read the current ggml reference implementation first, then update the
> parser and this skill in the same PR.

## BEFORE YOU START

- [ ] I distinguish two questions: *can we parse/load it* (this skill) vs *what does the engine compute on* (per-engine policy — for CPU, read [`../cpu-skills/quantization.md`](../cpu-skills/quantization.md) ⛔ before anything).
- [ ] Layout structs live in glcore once; engines consume them — I'm not duplicating byte-layout knowledge into an engine.
- [ ] My test shapes include ragged dims (dim = 896).

## Context

GwenLand loads Q8_0 and Q4_K models today (plus F32/F16/BF16 unquantized
tensors); Q8_K is known as an intermediate format. Each quant type is a
block layout — scales plus packed integers — and the *layout facts* are
format knowledge (glcore), while the *compute decision* (dequant kernel,
repack, native dot) is per-engine policy with measured verdicts attached.

## Rules

1. **Layout reference (verify against ggml when touching):**

   | Type | Block | Bytes/block | Contents |
   |------|-------|-------------|----------|
   | Q8_0 | 32 | 34 | f16 scale + 32 × i8 |
   | Q4_K | 256 (super-block) | 144 | two-level scales/mins (6-bit, packed) + 128 bytes of nibbles |
   | Q8_K | 256 | 292 | f32 scale + 256 × i8 + 16 × i16 block sums |
   | F32/F16/BF16 | — | 4/2/2 per elem | unquantized paths |

2. **Row math must handle ragged dims:** bytes-per-row =
   `dim / block_size × bytes_per_block` **only when** `block_size | dim`.
   dim = 896 is not a multiple of 256 — Q4_K rows there follow the ggml
   convention for partial superblocks, which must be read from the reference
   implementation, not assumed. This exact case produced a real bug; its
   regression test is permanent.
3. **Reference scalar dequant lives beside each layout** and is validated
   against known-good values; every engine's fast path (SIMD dequant, GPU
   repack, MMA feed) parity-tests against it.
4. **Adding a new quant type follows the ladder** in
   [`../cpu-skills/quantization.md`](../cpu-skills/quantization.md) Rule 7:
   parse → scalar dequant → tests → route through existing repack/compute
   paths → *measure* before writing native kernels for it.
5. **Never mix layout constants into kernels as magic numbers** — kernels
   import the glcore layout constants/structs so a spec correction lands in
   one place.
6. **f16/bf16 handling is explicit:** scale conversion (f16→f32) happens in
   defined places with tested rounding behavior — not ad-hoc `as` casts
   scattered through kernels.

## ✅ Correct Pattern

```rust
// glcore: single source of layout truth.
pub const Q4_K_SUPER_BLOCK: usize = 256;
pub const Q4_K_BYTES_PER_BLOCK: usize = 144;

#[repr(C)]
pub struct BlockQ4K { /* field layout mirrors ggml exactly, documented */ }

// engine: consumes the layout, owns only the compute.
let n_blocks = q4k_blocks_for_dim(dim)?; // ragged-aware helper from glcore
```

## ❌ Anti-Pattern (Never Do This)

```rust
// ❌ magic numbers in a kernel — spec drift now needs a treasure hunt:
let scales = &row[0..12];
let nibbles = &row[16..144];

// ❌ ragged dims by truncation:
let n_blocks = dim / 256;         // silently drops 128 weights at dim 896

// ❌ new quant type going straight to a native SIMD kernel "since decode is
//    bandwidth-bound" — that reasoning already lost 33 % once. Ladder first.
```

## GwenLand-Specific Notes

- Per-engine compute verdicts, for orientation: CPU = repack to Q8_0 at
  load (native Q4_K compute **closed**, −33 %); glcuda = Q8_0 SoA feeding
  dp4a and INT8-MMA prefill GEMMs. Layout knowledge is shared; those
  verdicts are not — never let a "unified quant kernel" idea cross engines.
- Model support claims in the README (Q8_0, Q4_K) are bounded by the parity
  suites — a newly parseable type isn't "supported" until an engine runs it
  end-to-end coherently.

## Related Skills

- [format-parsing.md](format-parsing.md)
- [dequant-path.md](dequant-path.md)
- [../cpu-skills/quantization.md](../cpu-skills/quantization.md)
