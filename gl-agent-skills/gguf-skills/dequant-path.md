# Dequant Paths — What Actually Ships

> **Domain:** gguf-skills
> **Applies to:** glcore reference dequant; `glproc` repack path; `glcuda` [`dequant.rs`](../../glcuda/src/dequant.rs)/[`repack.rs`](../../glcuda/src/repack.rs)
> **Last updated:** 2026-07-17
>
> ⚠️ **Upstream drift warning:** dequantization math (scale application,
> nibble order, min/offset handling) must match **ggml's** reference
> bit-for-bit within tolerance — ggml owns the format. New ggml quant types
> or corrections change the reference; re-verify against upstream before
> assuming our chain, and update parser + kernels + this skill together.

## BEFORE YOU START

- [ ] I can name which chain my change touches (table below) and which of them are **production** vs **reference** vs **closed**.
- [ ] I am not building a new dequant chain when routing through an existing one would do ([`../cpu-skills/quantization.md`](../cpu-skills/quantization.md) Rule 7 — the ladder).
- [ ] Whatever I change, the scalar reference dequant stays the arbiter all fast paths parity against.

## Context

"Dequant" means different things per location, and conflating them causes
both bugs and dead-end optimization work. There is exactly one *reference*
(scalar, correctness-first, in glcore beside the layouts), and each engine
owns its *production* transformation tuned to its hardware — including the
measured decision on CPU to dequant **once at load** (repack) rather than
per token.

## Rules

1. **The chain map — know which one you're in:**

   | Chain | Where | Status |
   |-------|-------|--------|
   | Scalar reference dequant (any type → f32) | glcore, beside layouts | ✅ arbiter — never optimized, only correct |
   | Q4_K → Q8_0 **load-time repack** | glproc loader | ✅ **production CPU path** |
   | Q8_0 integer-dot compute (no f32 materialization) | glproc kernels | ✅ production CPU hot loop |
   | Native Q4_K decode compute | — | ⛔ **closed**, −33 % (see [`../cpu-skills/quantization.md`](../cpu-skills/quantization.md)) |
   | Q8_0 → SoA device repack | glcuda `repack.rs` | ✅ production GPU (feeds dp4a / INT8-MMA) |
   | Device dequant kernels | glcuda `dequant.rs` | ✅ where the GPU path needs f32/f16 |

2. **Reference stays boring:** the scalar dequant is written for
   readability and spec fidelity — no SIMD, no cleverness. Its outputs
   validate every other chain; optimizing it destroys its purpose.
3. **Hot loops never materialize full f32 weight tensors.** Production
   chains either repack once at load (CPU) or dequant in-register /
   in-kernel (GPU). A `Vec<f32>` the size of an FFN matrix is a memory-rule
   violation, not a dequant strategy
   ([`../rust-skills/memory-safety.md`](../rust-skills/memory-safety.md)).
4. **Repack outputs are validated like kernels:** repacked weights must
   reproduce reference-dequant values within spec tolerance, with ragged
   dims (896) in the test set — the repack is where layout bugs hide.
5. **Scale precision is part of the contract:** f16-scale → f32 conversion
   points are fixed and tested; two chains disagreeing in the last bit is
   how parity failures that "only happen on some models" are born.
6. **New formats route through existing production chains first** (repack →
   Q8_0 on CPU; SoA on GPU). A format-specific fast path is a *later*,
   measured decision — expectation on the CPU tier is that it loses.

## ✅ Correct Pattern

```text
Adding parse support for a new 4-bit variant on CPU:
  file bytes → scalar reference dequant (tests vs known values)
             → load-time repack to Q8_0 (validated vs reference, dim 896 incl.)
             → existing Q8_0 hot loop unchanged.
  glbench ab vs a native-Q8_0 export of the same model → ship the loader.
```

## ❌ Anti-Pattern (Never Do This)

```text
❌ Per-token dequant of Q4_K to an f32 scratch "temporarily" — that's the
   unpack cost in the hot loop, the exact thing repack exists to remove.

❌ "Optimized the reference dequant with AVX2" — the reference's only job
   is to be obviously correct; fast versions live in engines, parity-tested.

❌ A second scale-conversion helper with slightly different rounding in one
   engine — cross-engine parity now fails mysteriously per-model.
```

## GwenLand-Specific Notes

- Historical naming caution: older experiment docs mention Euler/Gamma
  ("GDTQP") dequantization methods — those live in
  [`../../Experimental/`](../../Experimental/README.md) and are **research**,
  not any of the shipped chains above. Never wire them into the product
  path from those docs.
- The repack-once philosophy is CPU-tier policy backed by the −33 %
  measurement; the GPU keeps different trade-offs (device dequant is fine
  where prefill GEMMs want f16). Per-engine verdicts, as always.

## Related Skills

- [quantization-types.md](quantization-types.md)
- [../cpu-skills/quantization.md](../cpu-skills/quantization.md)
- [../cuda-skills/tensor-cores.md](../cuda-skills/tensor-cores.md)
