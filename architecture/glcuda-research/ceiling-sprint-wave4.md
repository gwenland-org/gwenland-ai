# glcuda T4 Ceiling Sprint — Wave 4

**Date:** 2026-08-25  
**Target:** NVIDIA T4 (`sm_75`), Qwen2.5-0.5B-Instruct Q4_K_M, 15,000 prefill tok/s  
**Status:** retained on T4  
**Notebook:** [`notebooks/glcuda_t4_ceiling_wave4.ipynb`](../../notebooks/glcuda_t4_ceiling_wave4.ipynb)

## Starting evidence

Wave 3 retained the 2-D token grid at **5,732.0 prefill tok/s**, 9.51% above
the Wave 2 ceiling. The 244-token prompt took **42.57 ms** and still needs a
**2.62x** end-to-end speedup to reach 15k. Stage telemetry moved the primary
non-GEMM bottleneck into attention: GEMM was **47.8%** and attention was
**42.1%**. NCU counters were unavailable on Kaggle, so the mechanism below is
a falsifiable occupancy hypothesis, not a measured hardware claim.

## One causal lever

`gl_attn_decode_rows_f32` previously reserved its maximum-context score array
for every block:

```text
static score storage = 4096 * 4 + 36 = 16,420 bytes / CTA
```

Wave 4 replaces only that storage declaration with one module-scope dynamic
shared segment. The host passes the exact score capacity already implied by
the launch:

```text
dynamic shared bytes = max_cached_len * 4 + 36
244-token prompt      = 1,012 bytes / CTA
4096-token maximum    = 16,420 bytes / CTA
```

The score and reduction regions are manually partitioned inside the same
dynamic segment. Attention math, row mapping, reductions, barriers, and output
layout remain unchanged. The diagnostic attention probe uses the same ABI so
its comparison is structurally faithful. The `sm_75` INT8 MMA PTX is byte-for-
byte identical between the Wave 3 baseline and Wave 4 candidate.

The expected effect is higher CTA residency at short prefill lengths. That is
not accepted from source inspection: PTXAS, production timings, tail latency,
and optional NCU evidence in the T4 notebook decide whether the hypothesis is
true.

## Gate

The direct-fetch notebook reconstructs Wave 3 and Wave 4 as separate source
trees and rejects the candidate on any of these:

1. either PTX module fails PTXAS, reports spills, or the candidate attention
   kernel regresses beyond the bounded register/resource gate;
2. host, parity, forward, or graph-replay correctness fails, or CUDA work is
   silently skipped;
3. structural checks do not prove removal of the two static score arrays,
   the exact dynamic-size formula, and an unchanged `glcuda_sm75.ptx`;
4. median production prefill P50 improves by less than 5%, or either paired
   session's P50/mean improvement is below 5%;
5. P95 prompt latency regresses by more than 5%, or decode P50 regresses by
   more than 5%.

Production sessions are interleaved in reversed order, with three cold runs,
three warmups, and ten measured prompts per arm. The model revision, filename,
byte size, GGUF magic, and SHA-256 are all pinned and verified. Microbenchmarks
and NCU output remain diagnostic; production glbench is the retention gate.

## Next decision

Wave 4 passed every hard gate on T4. Production prefill improved from 5,662.5
to **6,474.7 tok/s** (+14.34% median P50), with both paired sessions above the
5% threshold. Hardware correctness and PTXAS stayed green; the diagnostic
attention median improved from 1,048.5 to 898.0 us. At the 244-token prompt,
the static 16,420-byte attention allocation became a 1,012-byte dynamic
allocation. The retained stage split is 27.8% attention and 57.5% GEMM, so
Wave 5 returns to the 8-tile Tensor Core path with one isolated scheduling
change.
