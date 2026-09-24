# Wave 125: N16-prefetch fused SwiGLU result

## Verdict

Rejected at the frozen compiler-resource gate. Production timing was not run,
so Wave 125 makes no throughput claim and does not move the verified production
number toward 15,000 tok/s.

## What was tested

The candidate reused Wave 88's 256-thread N16-prefetch mainloop. Its second N8
fragment read matching `up` rows instead of adjacent `gate` rows. After the K32
loop, the kernel reused one 16 KiB shared image as a 64-token x 64-hidden f32
slab, applied the retained SiLU instruction order, and wrote the exact Q8_0
layout intended for `ffn_down`.

This schedule directly addressed Wave 21's two largest structural costs:

- 256 threads instead of 512;
- one aliased 16 KiB shared allocation instead of adding a second epilogue
  allocation to the retained operand image.

## Authoritative T4 gate

The version-3 notebook ran on a Tesla T4, compute capability 7.5, with 15,360
MiB and driver 580.159.04. Host validation passed 67/67 before `ptxas` compiled
the candidate as:

| resource | measured | frozen gate |
|---|---:|---:|
| registers/thread | 80 | <=80 |
| static shared memory | 16,384 B | exactly 16,384 B |
| stack frame | 8 B | 0 B |
| spill stores | 4 B | 0 B |
| spill loads | 4 B | 0 B |

The single spilled 32-bit value is enough to reject the candidate. The gate was
defined before compilation and explicitly required zero spill; weakening it
after seeing the result would invalidate the experiment.

The sprint stopped before CUDA parity, direct Q8 byte/scale parity, production
A/B, and profiling. None of those absent measurements may be inferred.

## Reproduction

- Notebook: `notebooks/glcuda_t4_wave125_n16_fused_swiglu.ipynb`
- Raw archive: `benchmarks/glcuda-t4-wave125-n16-fused-swiglu-results.zip`
- Machine-readable result:
  `benchmarks/glcuda-t4-wave125-n16-fused-swiglu.json`
- Raw archive SHA-256:
  `f9263bb1deea9880b9a6782fda5e6538835589a04c5b6d75b3fa16de50d0d31a`
- Notebook SHA-256:
  `d0db63c2986dec69dba8d8cbe31f43d6679b3b20fb81a45ee9ac10112cd1c400`

## Next bounded repair

Wave 126 may test exactly one resource change: reuse a dead post-mainloop
32-bit register for the epilogue task/index temporary instead of keeping the
named `%r_fgu_tmp` live. The arithmetic, 256-thread geometry, shared alias and
Q8 instruction order must remain fixed. It must first return to zero stack and
zero spill under `ptxas`; only then may parity and production timing resume.

`glbench` still cannot explain scheduler behavior by itself: it lacks eligible
warps/cycle, issue-active percentage, stall reasons, and per-kernel L1/L2 hit
rates. Those counters are not needed for this rejection because the compiler
resource gate already failed, but they remain required for a causal explanation
of any future performance result.
