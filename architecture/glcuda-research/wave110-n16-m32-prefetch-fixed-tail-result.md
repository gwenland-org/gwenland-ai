# Wave 110 - fixed-tail N16/M32 prefetch result

## Result

Wave 110 specialized the prefetch-tail guard for the exact FFN-down shape:
`in_dim=4864` means 152 K32 blocks, so the dynamic `k + 1 >= nb` test became
`k >= 151`. This removed one integer add and its temporary register per K
iteration without changing loads, MMA arithmetic, output geometry, or the
resource tier.

The first counterbalanced Tesla T4 direct run failed the fixed 1.10x gate:

| measure | retained N16/M32 | Wave 110 fixed tail |
| --- | ---: | ---: |
| direct time | 208.528 us | 203.886 us |
| speedup | - | **1.0228x** |
| registers/thread | 64 | 64 |
| active blocks/SM | 4 | 4 |
| static shared memory | 9,728 B | 9,728 B |
| spills | 0 | 0 |

Output remained bit-exact (`first_mismatch = -1`). The notebook stopped after
run one and no production benchmark was authorized.

## Decision

Retire the M32 complete-prefetch direction. Wave 108 crossed the occupancy
tier and reached only 1.0551x; Wave 109 restored occupancy but reached 1.0922x;
the final bounded Wave 110 specialization fell to 1.0228x. None passed the
fixed two-run 1.10x feasibility gate, and selecting only the best noisy result
would violate the measurement protocol.

Wave 111 should remove the rejected Wave 105/109 runtime plumbing and move to
the higher-ceiling cross-layer residual fusion identified by the Wave 104
profile: defer each FFN-down residual add into the next layer's existing
`rms_quantize_q8_rows` residual-capable pass. That targets the 11.06% CUDA
elementwise bucket rather than another low-single-digit GEMM micro-variant.

The fastest valid production measurement remains the rejected experimental
12,232.7 tok/s configuration. No new production throughput was measured.

Raw archive: `benchmarks/glcuda-t4-wave110-n16-m32-prefetch-fixed-tail-v1.zip`

Archive SHA-256:
`f078225b4bd6d3b3501a0e26a3436aeef5e5b33279b627eae9a2e2ad5d6beca7`

