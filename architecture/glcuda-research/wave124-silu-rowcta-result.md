# Wave 124 - stacked SiLU row-CTA

## Outcome

Wave 124 is **REJECTED**. It preserved the retained Wave 123 arithmetic and
passed every correctness/resource gate, but production prefill slowed down in
both counterbalanced invocation orders.

The candidate changed only the launch schedule for the stacked SwiGLU+Q8
kernel. Instead of launching 19 CTAs for every token row at hidden width 4,864,
one 256-thread CTA owned a token row and its eight warps looped over the 152 Q8
blocks. This reduced the per-layer grid from 4,636 CTAs to 244 CTAs.

## Production result

The combined decision statistic used 40 samples per arm on one Tesla T4
allocation:

- retained median: 18.638322 ms = 13,091.3 tok/s;
- candidate median: 18.775233 ms = 12,995.8 tok/s;
- speedup: 0.9927x (-0.73%);
- retained/candidate mean: 18.607380 / 18.810402 ms;
- retained/candidate P95: 19.006627 / 19.303342 ms;
- oracle token: 3323 for all 80 samples.

Both invocation orders agreed:

| Invocation | Retained | Candidate | Speedup |
|---|---:|---:|---:|
| A | 13,143.1 tok/s | 13,032.9 tok/s | 0.9916x |
| B | 13,067.2 tok/s | 12,937.4 tok/s | 0.9901x |

The predeclared retention gate required at least 1.02x combined speedup and a
positive result in both invocation orders. Neither condition passed.

## Interpretation

The launch-count hypothesis was falsified for the retained production stack.
The new kernel used 20 registers, no shared memory, no barriers, and no spills,
so occupancy resources did not explain the regression. The likely mechanism is
lost grid-level parallelism or poorer latency hiding: fewer CTAs did less useful
scheduling work even though launch overhead fell. This remains an inference;
hardware issue/stall counters were unavailable because the T4 environment
denied Nsight Compute counter access.

`glbench` can time production and report stage totals, but it does not expose
per-kernel eligible-warps, issue-utilization, cache-hit, or stall-reason data.
The dedicated harness therefore supplies the decision-grade paired timing and
correctness gates, while the precise microarchitectural cause remains unknown.

## Validation and cleanup

- GPU: Tesla T4, sm_75, driver 580.159.04.
- Candidate PTX: 20 registers, zero barriers, zero spill stores/loads.
- CUDA parity: 37 passed, including byte/scale identity against Wave 123.
- Library tests: 67 passed.
- Rejected runtime code and the temporary notebook builder were removed.
- The executed notebook is retained because it embeds the exact candidate
  patch and reconstructs the run from pinned revisions.

## Evidence

- Summary: `benchmarks/glcuda-t4-wave124-silu-rowcta.json`.
- Executed evidence archive:
  `benchmarks/glcuda-t4-wave124-silu-rowcta-results.zip`.
- Archive SHA-256:
  `26f05d98d477ffe264b741edd5b4f7d46e9728c92b2cf763c1f4e27070045b7b`.
- Reproducible notebook: `notebooks/glcuda_t4_wave124_silu_rowcta.ipynb`.

## Next wave

Do not revisit row-CTA scheduling for this stage. Wave 125 should test a modern
N16-prefetch gate/up GEMM with a fused SiLU+Q8 epilogue, because that attacks
both the 4.86 ms gate/up GEMM and the 1.24 ms materialized epilogue rather than
trading launch count for parallelism.
