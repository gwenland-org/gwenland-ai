# Quantization on the CPU Tier

> **Domain:** cpu-skills
> **Applies to:** `glproc` — dequant/bridge kernels, Q8_0 repack path
> **Last updated:** 2026-07-17

## BEFORE YOU START

- [ ] ⛔ I have read this file to the end before proposing ANY quant-kernel work — this domain contains the project's most expensive measured lesson.
- [ ] I know the verdict: **native Q4_K compute is a closed, hardware-tier dead end** on AVX2/VNNI-256. The shipped design is **repack Q4_K → Q8_0 at load** and compute on Q8_0.
- [ ] I am not about to argue from block sizes and bandwidth math alone — that exact argument already lost to measurement.

## Context

The bandwidth wall (see [memory-bandwidth.md](memory-bandwidth.md)) makes 4-bit
weights look like a guaranteed ~2× FFN win: half the bytes, bandwidth-bound
loop, done. It was built — integer-dot Q4_K kernels, then a fused Q4_K SwiGLU —
and **measured 33 % slower** than the Q8_0 path. The diagnosis matters more
than the number: the loss was *identical when the working set fit in L2*, so
it was never a memory problem — **nibble unpacking (splitting packed 4-bit
weights, applying the two-level K-quant scales) is compute-bound** on 256-bit
vectors, and it costs more than the bandwidth it saves on this tier. Hence the
architecture: pay the unpack cost **once at load** (repack to Q8_0), then run
the hot loop on a format AVX2/VNNI-256 digests natively.

## Rules

1. **Q8_0 is the production CPU compute format.** The Q4_K→Q8_0 load-time
   repack is the shipped design; the Q8_0 path may never be removed or
   degraded to make room for a "better" format.
2. **Native Q4_K decode compute is CLOSED on this tier.** Re-opening it
   requires (a) explicit permission from JinXSuper AND (b) a materially
   different mechanism — a wider-vector tier (AVX-512 machines), a changed
   memory system, or an unpack trick that provably wasn't in the tested
   kernels. "The roofline says it should win" is not new evidence; it was
   the original, falsified argument.
3. **Format facts** (for parser/repack work, not for revisiting compute):
   Q4_K = 256-weight super-blocks, 144 bytes (packed nibbles + two-level
   scales/mins); Q8_K = 292 bytes; Q8_0 = 32-weight blocks, 34 bytes
   (f16 scale + 32 × i8). Repack math must match the GGUF reference
   dequantization bit-for-bit within spec tolerance.
4. **Repack cost is load-time cost — keep it there.** It parallelizes well
   (SMT pays on the load path, see [threading-model.md](threading-model.md));
   nothing quant-related is recomputed in the token loop.
5. **RAM accounting is part of any quant proposal.** Repack trades file-size
   compactness for a Q8_0-resident working set; that trade is accepted and
   budgeted on 8 GB. A proposal that keeps *both* Q4_K source and Q8_0
   repack hot violates the memory rules.
6. **Ragged-dimension caution:** dim = 896 is not a multiple of the Q4_K
   256-super-block — per-row block layout must be computed from the spec,
   not assumed. This has already produced a real bug; the regression shapes
   stay in the tests.
7. **Any new quant format follows the same ladder:** parser + reference
   scalar dequant → parity tests → *measured production decode A/B on the
   reference box* → only then a kernel investment decision. Probes and
   microbenches don't gate-keep this ladder — production numbers do.

## ✅ Correct Pattern

```text
"Support IQ4_NL models on CPU":
1. Parser + scalar dequant + tests (correctness only).
2. Route through the existing load-time repack → Q8_0 hot path.
3. glbench ab: IQ4_NL-repacked vs Q8_0-native model, production decode.
4. Ship the loader support; do NOT write native IQ4_NL kernels unless the
   measured numbers demand it — expectation is they won't, same physics
   as Q4_K.
```

## ❌ Anti-Pattern (Never Do This)

```text
❌ "Q4_K halves FFN bytes; decode is bandwidth-bound; therefore native Q4_K
   dot will win ~1.85×" — this exact proposal was implemented and measured
   at -33 %. It returns every few months wearing new words. Reject it.

❌ Deleting/bypassing the Q8_0 repack "to save load time" — moves the unpack
   cost into every token forever.

❌ Validating a repack only on dims divisible by 256.
```

## GwenLand-Specific Notes

- GPU is a different tier with a different verdict: glcuda runs Q8_0 SoA with
  INT8 tensor cores for prefill ([`../cuda-skills/tensor-cores.md`](../cuda-skills/tensor-cores.md)) —
  don't copy CPU quant conclusions to GPU or vice versa.
- Historical note for spec-readers: older plans ("Sprint 2") projected
  ~1.85× from Q4_K integer-dot. Those projections predate the measurement
  and are **superseded** — if you find them quoted in an old doc, the number
  to trust is the measured -33 %.

## Related Skills

- [memory-bandwidth.md](memory-bandwidth.md)
- [rejected-optimizations.md](rejected-optimizations.md)
- [../gguf-skills/quantization-types.md](../gguf-skills/quantization-types.md)
