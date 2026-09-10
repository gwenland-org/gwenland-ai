# Wave 108 - N16/M32 prefetch direct result

## Result

Wave 108 completed repository reconstruction, strict patch validation,
`ptxas`, host tests, release build, driver occupancy, bit-exact parity, and the
first counterbalanced direct timing run on Tesla T4. The candidate failed the
predeclared 1.10x direct speed gate, so the notebook stopped before run two.

| measure | retained N16/M32 | complete-prefetch candidate |
| --- | ---: | ---: |
| direct time | 208.681 us | 197.780 us |
| speedup | - | **1.0551x** |
| registers/thread | 64 | 72 |
| active blocks/SM | 4 | 3 |
| static shared memory | 9,728 B | 9,728 B |
| spills | 0 | 0 |

Output was bit-exact (`first_mismatch = -1`). The candidate reduced direct
kernel time by 5.22%, but the additional live prefetch state crossed a T4
occupancy tier. That net result is materially below the 1.10x feasibility
threshold and cannot authorize a production run.

## Decision

Reject the current 72-register complete-prefetch candidate for production.
It remains isolated behind `GLCUDA_GEMM_N16_M32_PREFETCH`, so retained dispatch
is unchanged.

Wave 109 should test whether the exact same schedule can reuse registers that
are dead during the compute window and compile at no more than 64 registers,
preserving four active blocks/SM. This is a resource/liveness repair, not a
threshold reduction: bit-exactness and the two-run 1.10x gate remain fixed.
If 64 registers cannot be achieved without changing arithmetic or geometry,
retire this direction.

The fastest valid production measurement remains the rejected experimental
12,232.7 tok/s configuration. No new production throughput was measured.

Raw archive: `benchmarks/glcuda-t4-wave108-n16-m32-prefetch-v1.zip`

Archive SHA-256:
`65ee8c3e23df46b8c0fb37fa50669f3f83e837ca2dd694e7ff630237a6722a0a`

