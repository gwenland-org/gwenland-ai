# glcuda T4 Ceiling Sprint — Wave 3

**Date:** 2026-08-25  
**Target:** NVIDIA T4 (`sm_75`), Qwen2.5-0.5B-Instruct Q4_K_M, 15,000 prefill tok/s  
**Status:** retained after T4 production gate  
**Notebook:** [`notebooks/glcuda_t4_ceiling_wave3.ipynb`](../../notebooks/glcuda_t4_ceiling_wave3.ipynb)

## Starting evidence

Wave 2's best production arm was `GLCUDA_FORCE_Q8=1 + GLCUDA_R256=1`:

- median session P50: **5,251.5 tok/s** (5,251.1–5,252.0, two sessions);
- prompt time: **46.46 ms** for 244 tokens;
- required speedup to 15k: **2.86x**;
- stage telemetry: GEMMs **55.8%**, attention core **35.1%**;
- `ptxas`: zero spills, 110 registers / 13,312 B shared for r256;
- NCU counters: unavailable on Kaggle (`ERR_NVGPUCTRPERM`).

The target is not blocked by T4's headline INT8 rate: the linear-layer demand
at 15k is only 5.37 TMAC/s, about 8.3% of 65 TMAC/s. It is blocked by launch
geometry, scale/FP32 issue work, and the non-GEMM graph.

The Amdahl boundary is explicit. With the measured 55.8% GEMM share, making
GEMM infinitely fast only raises the current arm to about **11.9k tok/s**.
The previously measured 3.28x isolated 2-D-grid result projects to about
**8.58k tok/s**, so this wave is necessary but cannot be the final wave.

## T4 outcome

The direct-fetch Wave 3 artifact passed PTXAS and hardware correctness. Its
candidate reached a median **5,732.0 prefill tok/s**, a **9.51%** improvement
over the retained Wave 2 arm, with both production sessions clearing the 5%
gate. Prompt time fell to **42.57 ms** for 244 tokens. The resulting stage
split was GEMM **47.8%** and attention **42.1%**; NCU hardware counters remained
unavailable on Kaggle (`ERR_NVGPUCTRPERM`). This outcome establishes the Wave 3
tree as Wave 4's baseline.

## One causal lever

Wave 3 productionizes the already hardware-proven token grid:

```text
old: grid = (ceil(out/64), 1, 1), host loops over 64-token slabs
new: grid = (ceil(out/64), ceil(ntok/64), 1), one launch
```

At kernel entry, CTA `y` computes `t0 = ctaid.y * 64`, rebases `x_qs`,
`x_scales`, and `y`, then replaces the local `ntok` with `min(ntok-t0, 64)`.
Everything after that point is the original single-slab body: the MMA shape,
operand mapping, barriers, Q8_0 scale multiply/FMA, and stores do not change.

The round8 contract remains exact. `t0` is divisible by eight, therefore a
ragged tail reads no farther than `round8(original_ntok)`. The new hardware
parity ladder includes 65, 200, and 256 tokens plus the production-like
`out=256, in=896, ntok=200` case.

The candidate is deliberately opt-in through `GLCUDA_GRID2D=1`. It takes
precedence over `GLCUDA_R256`; enabling both algorithms in one arm would make
the result uninterpretable. The Wave 2 arm and Wave 3 arm use the same
candidate binary:

| arm | environment | mechanism |
|---|---|---|
| Wave 2 ceiling | `GLCUDA_FORCE_Q8=1`, `GLCUDA_R256=1` | one 256-row internal tile |
| Wave 3 | `GLCUDA_FORCE_Q8=1`, `GLCUDA_GRID2D=1` | four concurrent 64-row CTAs at 244 tokens |

## Measurement correction

Wave 2 stage telemetry always charged Q8 GEMMs as `ceil(ntok/64)` weight
reads, even when r256 executed `ceil(ntok/256)`. At `ntok <= 256` that
overstated r256's logical weight bytes by 4x. Wave 3 records the active
strategy: 256-row accounting for r256, 64-row accounting for the 2-D grid.
This changes telemetry metadata only, not kernel dispatch or timing.

## Gate

The Kaggle notebook must stop/reject on any of these:

1. `ptxas` failure, non-zero spill load/store, or more than 64 registers for
   `gl_gemm_mma_q8` (would reduce the intended T4 register occupancy tier);
2. any host, parity, forward, or graph-replay failure, or a silent CUDA skip;
3. missing dispatch banner, or r256 accidentally enabled in the grid arm;
4. median production P50 improvement below 5%;
5. either of the two paired sessions improving by less than 5%.

The notebook archives raw logs, glbench JSON, the embedded patch and digest,
PTXAS resources, optional NCU output, stage telemetry, and the final retain
decision. Microbenchmarks remain diagnostic only.

## Next bottleneck if Wave 3 passes

Attention is the Wave 4 candidate, not part of this patch. The prefill rows
kernel reserves 16 KiB of scores per block for the 4,096-token maximum even
when the measured prompt has only 244 rows. A dynamic/shared-size or short-
context specialization could raise attention occupancy materially, but it
must be measured as a separate causal arm after this gate.
