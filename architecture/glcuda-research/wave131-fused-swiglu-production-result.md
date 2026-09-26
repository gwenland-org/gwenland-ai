# Wave 131 - fused SwiGLU production result

## Outcome

Wave 131 retains the Wave 129 fused SwiGLU specialization as an opt-in
production prefill path for the pinned Qwen2.5-0.5B Q8_0 shape. The candidate
passed the compiler-resource, serial CUDA parity, dispatch-provenance, oracle,
and counterbalanced wall-time gates on a Tesla T4.

The production `glbench` runs exceeded the 15,000 tok/s prefill target twice.
Decode remained far below its estimated bandwidth ceiling, so this wave does
not claim that the overall CUDA optimization objective is complete.

## Production prefill

The in-process A/B alternated retained and candidate paths in one CUDA context
and collected 40 samples per arm:

| Arm | Median | Throughput |
|---|---:|---:|
| retained Wave 123 + Wave 88 | 18.587111 ms | 13,127.4 tok/s |
| Wave 129 fused SwiGLU | 17.812217 ms | 13,698.5 tok/s |

That is a 1.0435x speedup (+4.35%). Both independently counterbalanced
invocations agreed: 1.0434x and 1.0498x. Every inference produced the same
oracle token, 3323.

Four separate production `glbench` sessions ran in A-B-B-A order. Both arms
kept Q8 no-store, stacked gate/up, deferred residual, N16 prefetch, and the
retained attention path enabled; only the fused SwiGLU selector changed.
Combining the 20 warm iterations per arm gives:

| Arm | P50 | P90 | P99 | Range |
|---|---:|---:|---:|---:|
| retained | 14,660.1 | 14,899.9 | 14,924.8 | 14,292.5-14,928.2 tok/s |
| candidate | **15,812.1** | **15,909.8** | **15,949.8** | 15,594.0-15,951.4 tok/s |

The candidate's two independent session medians were 15,836.9 and 15,790.7
tok/s. The absolute difference from the alternating harness is reported rather
than hidden: the harness changes the selector for every inference, while each
`glbench` session holds one production dispatch constant. Both measurements
agree on the direction and material size of the improvement; `glbench` is the
project's production-throughput authority.

## Decode and ceiling

The same four production sessions generated 64 tokens per iteration. Combined
warm distributions were:

| Arm | P50 | P90 | P99 | Range |
|---|---:|---:|---:|---:|
| retained | 209.39 | 211.43 | 212.64 | 208.48-212.90 tok/s |
| candidate | 210.86 | 211.49 | 211.91 | 209.44-211.93 tok/s |

`glbench` estimates a 473.6 tok/s decode bandwidth ceiling from the Tesla T4's
published 320 GB/s specification and the 0.68 GB model footprint. Observed
decode is about 44-45% of that estimate and is classified compute/kernel-bound.
This is an estimated-table ceiling, not a device-measured bandwidth ceiling;
the next wave must profile the graph-internal decode work before choosing a
fusion or arithmetic candidate.

## Validation

- GPU: Tesla T4, sm_75, driver 580.159.04.
- Model: 675,710,816 bytes, SHA-256
  `ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e`.
- Host library tests: 73 passed.
- Serial CUDA parity: 36 passed.
- Wave 129 resource gate: 80 registers/thread, 16,384 bytes static shared,
  one barrier, zero stack, zero spill stores, zero spill loads.
- Retained sessions logged `n16_fused_swiglu:false` and no fused dispatch.
- Candidate sessions logged `n16_fused_swiglu:true` and
  `wave129-fused-swiglu-production` at hidden=4864, input=896, tokens=244.

## Evidence

- Summary: `benchmarks/glcuda-t4-wave131-fused-swiglu-production.json`.
- Four raw production sessions:
  `benchmarks/glcuda-t4-wave131-glbench-{1..4}-*.json`.
- Full logs and raw records:
  `benchmarks/glcuda-t4-wave131-fused-swiglu-production-results.zip`.
- Archive SHA-256:
  `a7518c934b38f714fdda8f7766130fc124b95cbd48b5ad0d751d4de520bf5783`.
