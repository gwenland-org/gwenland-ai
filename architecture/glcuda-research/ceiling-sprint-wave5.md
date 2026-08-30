# glcuda T4 Ceiling Sprint - Wave 5

**Date:** 2026-08-25  
**Target:** NVIDIA T4 (`sm_75`), Qwen2.5-0.5B-Instruct Q4_K_M, 15,000 prefill tok/s  
**Status:** rejected on T4; retained Wave 4 remains the baseline  
**Notebook:** [`notebooks/glcuda_t4_ceiling_wave5.ipynb`](../../notebooks/glcuda_t4_ceiling_wave5.ipynb)

## Starting evidence

Wave 4 retained dynamic-shared attention at **6,474.7 prefill tok/s**, a
14.34% gain over Wave 3. The measured prompt takes 37.69 ms but the 15k target
allows only 16.27 ms, leaving a **2.32x** end-to-end gap. Telemetry now assigns
57.5% of the prompt to GEMM and 27.8% to attention. The infinite-GEMM-only
projection reaches 15,231.7 tok/s, so the next falsifiable lever belongs in
the production Tensor Core kernel rather than attention.

## One causal lever

`gl_gemm_mma_q8` previously staged each 32-wide activation K-block into one
shared-memory tile, synchronized, computed eight M tiles, synchronized again,
and then repeated. Wave 5 changes only this 64-row kernel to two shared tiles:

```text
Wave 4 shared: 3072 B A + 256 B xscale = 3328 B/CTA
Wave 5 shared: 2 * (3072 B A + 256 B xscale) = 6656 B/CTA
runtime barriers: 2 * nb -> nb + 1
```

Buffer 0 is primed before the K loop. During each iteration, cooperative
global loads for `kb+1` enter per-thread registers while Tensor Cores consume
the current shared tile. After the existing MMA/dequant body, those registers
are committed to the alternate shared tile, one block barrier makes the tile
visible, and the read/write pointers rotate. The last iteration still meets
the same block-wide barrier, so active and output-masked warps never diverge
around synchronization.

This is a manual Turing pipeline: `cp.async` is not legal on `sm_75`. The MMA
shape, Q8_0 scale multiply/FMA, vector output stores, 48-byte bank-conflict-free
row pitch, staging guards, grid geometry, and r256 kernel are unchanged. The
attention PTX is also byte-identical between arms.

The extra 3,328 bytes cannot reduce the expected four-CTA register ceiling:
four Wave 5 CTAs consume only 26,624 bytes of the T4 SM's 64 KiB shared-memory
budget. PTXAS must still prove zero spills and no register-driven occupancy
regression.

## Gate

The direct-fetch notebook reconstructs the retained Wave 4 baseline and Wave
5 candidate from three SHA-verified patches. It rejects Wave 5 if any of these
conditions is true:

1. PTXAS fails, any tracked kernel spills, the candidate uses other than
   6,656 bytes of shared memory, exceeds 64 registers, or changes attention or
   r256 resources;
2. structural checks do not prove the exact two-buffer load/store/rotation
   schedule, unchanged r256 bytes, unchanged MMA/vector-store counts, and no
   `cp.async`;
3. lib, parity, forward, or graph-replay correctness fails, or CUDA execution
   is silently skipped;
4. median production prefill P50 improves by less than 5%, or either paired
   session's P50/mean gain is below 5%;
5. P95 prompt latency regresses by more than 5%, or decode P50 regresses by
   more than 5%.

The two production sessions reverse arm order and use three cold runs, three
warmups, and ten measured prompts per arm. The existing 512-row GEMM grid is
diagnostic only; each printed sample contains 30 synchronized iterations.
Production glbench is the retention decision.

## T4 result and decision

PTXAS and hardware correctness passed. The candidate used 63 registers and
6,656 bytes of shared memory per CTA, versus 56 registers and 3,328 bytes for
Wave 4. Its isolated result was mixed: gate/up regressed 1.80%, while down
improved 3.22%.

The paired production result was **6,861.3 tok/s** versus **6,827.7 tok/s**
for the reconstructed Wave 4 arm: only **+0.49%**. One paired P50 result was
+1.21%; the other was -0.23%. Neither the P50 nor mean 5% gate passed. P95,
decode, PTXAS, and correctness stayed green, but those do not override the
production throughput gate.

**Reject the Wave 5 pipeline.** Do not ship it or stack further changes on
it. Wave 6 starts again from retained Wave 4 and tests a different causal
lever: a 128-row Tensor Core GEMM tile that increases B-fragment reuse without
the rejected A/xscale double buffer.
