# Wave 77 - row-major compensated-MMA AV T4 result

## Intended deliverable

Wave 77 removes the allocation blocker and runs the Wave 75 row-major AV gate
on an actual Tesla T4. Inspection of the installed Kaggle CLI 2.2.4 revealed
that `kernels push` accepts the server enum `NvidiaTeslaT4` through
`--accelerator` or `machine_shape`. Earlier notebooks set only
`enable_gpu=true`, which allowed Kaggle to choose a P100.

Both decision runs used the same notebook bytes and the hard selector:

```text
kaggle kernels push -p <stage> --accelerator NvidiaTeslaT4
```

## Reproduction contract

- Kaggle kernel: `jinxsuperdev/glcuda-wave77-hard-t4-mma-av-row`;
- versions: 1 and 2;
- GPU: Tesla T4, compute capability 7.5, driver 580.159.04;
- pinned base: `3bce8dd7b8aaa2765855ab927c611b54981f9241`;
- notebook SHA-256:
  `eeedcaae19fb450ea03fd73436bfb3ea50a5a27320cdc2dcc2b8e37f7c3726a2`;
- each version: 10 warmups, 100 measured launches, five interleaved repeats;
- capacities: 1, 17, 241 and 244;
- numeric gate: max absolute error at or below `1e-5`;
- capacity-244 direct gate: median speedup at or above 1.50x.

This is an isolated direct gate. It does not alter or measure the production
attention dispatcher.

## Resources

Both versions assembled the row-major candidate at 42 registers, zero stack,
zero spill stores, zero spill loads and zero barriers. The transposed Wave 71
candidate used 44 registers, so removing the permanent transpose did not buy
correctness at the cost of compiler resources.

## Direct results

| version | capacity | scalar AV | row-major MMA AV | speedup | max abs |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 4.034 us | 3.663 us | 1.1012x | 0 |
| 1 | 17 | 15.582 us | 6.092 us | 2.5576x | 1.341104507e-7 |
| 1 | 241 | 303.159 us | 90.427 us | 3.3525x | 3.390014172e-7 |
| 1 | 244 | 239.703 us | 58.038 us | **4.1301x** | 3.799796104e-7 |
| 2 | 1 | 3.966 us | 3.532 us | 1.1228x | 0 |
| 2 | 17 | 14.961 us | 5.816 us | 2.5722x | 1.341104507e-7 |
| 2 | 241 | 284.330 us | 77.614 us | 3.6634x | 3.390014172e-7 |
| 2 | 244 | 243.730 us | 59.101 us | **4.1239x** | 3.799796104e-7 |

The capacity-244 midpoint is 241.7165 us versus 58.5695 us, or 4.1272x.
Both independent T4 allocations pass the numeric and speed gates. The raw
capacity-241 timings move more between sessions, but both remain above 3.35x;
no claim depends on that shape's absolute latency.

The maximum error is identical across runs and is 26.3x below the registered
`1e-5` bound. Capacity 1 remains launch-bound as expected and is retained as a
correctness/tail test, not a speed regime.

## Evidence

- version 1 archive:
  `benchmarks/glcuda-t4-wave77-mma-av-row-v1.zip`, SHA-256
  `6d2bccc6ecbb266a3e4a65eb16f4ce5bb9e4182b13bee124f428a3b4dd227a64`;
- version 2 archive:
  `benchmarks/glcuda-t4-wave77-mma-av-row-v2.zip`, SHA-256
  `dff0f647153e157c3aa5e7e568e310ea1394230708f9381f744d4e53326dfd9c`.

Each archive contains its raw `ptxas -v` output, parsed resources, source
manifest, build log, direct output and PASS marker.

## Decision

**REPRODUCED DIRECT FEASIBILITY PASS.** The retained row-major V cache can feed
the compensated MMA AV mapping without a permanent transpose, new allocation,
or cache-layout migration. Wave 74's proposed dimension-major cache change is
closed.

This licenses a fused production candidate inside
`gl_attn_mma4_regq_fused_f32`. It does not license a default flip and it does
not establish production throughput. The next wave must preserve the existing
QK and softmax values, replace only AV, add device parity for ragged/tail
shapes, and measure the fused kernel including its synchronization and register
cost before spending a full `glbench` run. The 15,000 tok/s goal remains open.
