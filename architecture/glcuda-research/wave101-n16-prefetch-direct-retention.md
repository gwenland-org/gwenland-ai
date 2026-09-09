# Wave 101 - N16 register-prefetch direct retention

## Result

The isolated N16 register-prefetch GEMM candidate passed its Tesla T4 direct
retention gate twice on the exact production gate/up shape
`4864 x 896 x 244`. Both runs were bit-exact against the retained N16 kernel
and cleared the required 1.10x speedup threshold.

| run | retained | candidate | speedup | parity |
| --- | ---: | ---: | ---: | --- |
| 1 | 185.628 us | 154.838 us | 1.1989x | bit-exact |
| 2 | 157.250 us | 123.527 us | 1.2730x | bit-exact |

The absolute timings drifted between runs, but the direction and gate outcome
matched. The minimum observed speedup was 1.1989x, 8.99 percentage points above
the retention threshold.

## Resource and correctness gates

- hardware: two Tesla T4 GPUs, compute capability 7.5;
- retained: 72 registers, 9,728 bytes shared memory, zero spills;
- candidate: 80 registers, 9,728 bytes shared memory, zero spills;
- driver occupancy: three active blocks per SM for both kernels;
- host suite: 65 passed, zero failed;
- direct parity: bit-exact in both runs.

## Decision

Retain the candidate for production integration behind the existing opt-in.
Wave 102 must measure the unchanged baseline and the candidate through the full
production prefill path. The direct result is not itself a production tok/s
claim.

Raw archive:
`benchmarks/glcuda-t4-wave101-n16-prefetch-direct-pass.zip`

Archive SHA-256:
`d5b1919279fc0c6ac550cc6ad4d4171c3843e9d4e5492c7a73247376f8f6521f`

The fastest verified experimental production result remains 11,377.9 tok/s
until that integration run is complete. The 15,000 tok/s target remains open.
