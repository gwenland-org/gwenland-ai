# Memory Bandwidth — The Wall

> **Domain:** cpu-skills
> **Applies to:** `glproc` decode path; any CPU perf claim
> **Last updated:** 2026-07-17

## BEFORE YOU START

- [ ] I know the measured numbers for the reference box (i3-1115G4, DDR4-2667 dual-channel): **read ceiling ≈ 29 GB/s measured** (28.7–29.4 GB/s depending on pass count/method), glproc decode running at **~20+ GB/s effective — roughly 70–78 % of ceiling** depending on model/quant.
- [ ] I can state my proposal in bytes: *which bytes does decode stop moving?* If the answer is "none", it is not a decode optimization.
- [ ] I know a roofline number is an **upper bound, not a promise** — being bandwidth-bound in theory doesn't mean a smaller format wins in practice (see [quantization.md](quantization.md)).

## Context

Weight-streaming decode reads every live weight byte once per token. On a
~29 GB/s machine, that arithmetic — model bytes ÷ bandwidth — sets the tok/s
ceiling (~55 tok/s implied for Qwen2.5-0.5B Q8_0) before any code quality
enters the picture. glproc already runs in the 70–78 % band of that ceiling,
which means: most "optimizations" have almost no headroom, and the ones that
work must either move fewer bytes or overlap stalls better. But the Q4_K
lesson (below) proves the converse trap too: moving fewer bytes only wins if
the compute to unpack them stays under the bandwidth savings.

## Rules

1. **Do the roofline math first.** Any decode perf proposal starts with:
   bytes/token before vs after, ÷ 29 GB/s, = theoretical bound. If the
   theoretical win is < 10 %, it will not survive measurement noise — drop it.
2. **Roofline is necessary, not sufficient.** Q4_K moved ~2× fewer FFN bytes
   and still **lost 33 %** — nibble unpacking is compute-bound and the gap
   persisted even from L2 (i.e., it was never memory-starved). Always follow
   the math with a production measurement.
3. **Cache-blocking decode is a category error.** Weights are streamed once
   per token — there is no reuse for a cache to exploit. L2 tiling for
   decode is on the rejected list; don't re-derive it.
4. **Prefill is different:** batched matmuls have reuse, so tiling and
   compute optimizations are legitimate there. Classify your phase before
   choosing tools (same discipline as the GPU side).
5. **Bandwidth measurements are environment-sensitive:** dual-channel vs
   single-channel, thermal state, and background load all move the ceiling.
   Quote the ceiling *measured on that machine that day* (glbench measures
   it), not the spec-sheet number.
6. **Effective-bandwidth regressions are real regressions** even when tok/s
   looks flat on a small model — track GB/s effective in `glbench` output,
   and compare like with like (same model, same quant, same machine).

## ✅ Correct Pattern

```text
Proposal: fuse dequant into the dot kernel to skip writing the f32
intermediate buffer.
Roofline: removes vocab×dim×4B per token of write+readback ≈ N MB/token
          → theoretical +12 % decode on 0.5B. Worth measuring.
Next: implement behind a flag → glbench ab, production decode, same box →
      keep only if the measured number agrees.
```

## ❌ Anti-Pattern (Never Do This)

```text
❌ "Q4 is half the bytes of Q8, so decode will be ~2× faster" — the exact
   reasoning that preceded the measured 33 % LOSS. Unpack cost is real.

❌ "Added software prefetching for the weight stream" — the hardware
   prefetcher already saturates a linear stream; PREFETCH_CHUNK=64 was
   tested and rejected.

❌ Comparing tok/s across different machines/quants and calling it a
   regression or a win.
```

## GwenLand-Specific Notes

- CPU decode is **FFN-bound (~52 %)** — the opposite of the GPU prefill
  profile (attention-heavy). Effort follows the measured profile of the
  target engine, not intuition imported from the other one.
- The bandwidth wall is also why the correct quant trade on this AVX2/VNNI
  tier is **repacking Q4_K → Q8_0 at load** rather than computing on Q4_K
  natively — full story in [quantization.md](quantization.md).

## Related Skills

- [quantization.md](quantization.md)
- [rejected-optimizations.md](rejected-optimizations.md)
- [../bench-skills/measurement-discipline.md](../bench-skills/measurement-discipline.md)
