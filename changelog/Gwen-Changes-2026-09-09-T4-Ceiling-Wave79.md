# T4 Ceiling Wave 79 - compensated-MMA AV device gate

## Result

The Wave 78 fused-attention candidate passed two independent, hard-selected
Tesla T4 allocations. Full-module `ptxas` reports the same 64-register,
zero-spill and four-blocks-per-SM tier as retained `mma4-regq`. Both allocations
passed all 33 serial CUDA parity tests, including production, ragged, strided-Q,
non-zero-base and short-tail coverage.

The complete fused attention launch improved from a 147.8245 us retained
midpoint to 109.084 us with compensated-MMA AV: 1.3551x faster, or 26.21% lower
latency. Maximum absolute difference was `6.556510925e-7` in both runs, below
the unchanged `1e-5` gate.

## Scope

This is a reproduced compiler/parity/direct-kernel pass, not production
throughput evidence. The candidate remains behind `GLCUDA_ATTN_MMA4_AV=1`.
Wave 80 must measure position-balanced production `glbench` sessions before a
retention decision or tok/s claim.
