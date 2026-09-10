# Wave 109 - occupancy-neutral N16/M32 prefetch result

## Result

Wave 109 rematerialized epilogue-only indexing after the K loop so the
complete next-K32 prefetch state could reuse those physical registers during
compute. The implementation reached the intended Tesla T4 resource tier and
remained bit-exact, but missed the fixed direct speed gate.

| measure | retained N16/M32 | Wave 109 remat |
| --- | ---: | ---: |
| direct time | 220.805 us | 202.166 us |
| speedup | - | **1.0922x** |
| registers/thread | 64 | 64 |
| active blocks/SM | 4 | 4 |
| static shared memory | 9,728 B | 9,728 B |
| spills | 0 | 0 |

Output was bit-exact (`first_mismatch = -1`). The candidate reduced direct
kernel time by 8.44%, materially better than Wave 108 after restoring the
four-block occupancy tier, but it remained below the predeclared 1.10x gate.
The notebook therefore stopped after the first counterbalanced run and no
production benchmark was authorized.

## Decision for Wave 110

Reject Wave 109 as measured; do not lower the threshold. One bounded
specialization remains admissible because the dispatch is already exact-shape:
replace the dynamic `k + 1 >= nb` prefetch-tail test with a direct comparison
against the known final K32 index 151. This removes one integer add and its
temporary register from every K iteration without changing loads, arithmetic,
geometry, or occupancy. It must still compile at no more than 64 registers,
remain bit-exact, and pass 1.10x twice. If it fails, retire M32 prefetch.

The fastest valid production measurement remains the rejected experimental
12,232.7 tok/s configuration. No new production throughput was measured.

Raw archive: `benchmarks/glcuda-t4-wave109-n16-m32-prefetch-remat-v1.zip`

Archive SHA-256:
`2dc46f06a16aa39c6f1dbb8d97adf847475c987b7a8c04cd404ea1b4ed0242bc`

