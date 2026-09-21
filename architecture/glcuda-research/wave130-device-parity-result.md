# Wave 130: fused specialization device-parity result

## Verdict

The isolated Wave 129 fused-only specialization passed the Wave 130 T4
device-parity gate. The real CUDA driver loaded the module, resolved the entry,
and reported three resident 256-thread CTAs per SM. At the pinned production
shape, the candidate matched every retained Q8 byte and every f32 scale bit,
including the 52-row ragged tail.

The direct harness measured 302.001 us for the candidate versus 322.731 us for
the retained two-kernel chain, a diagnostic 1.0686x speedup. This is not a
production result: `runner.rs` remains untouched, no production wall-time A/B
was run, and the verified 15,000 prefill tok/s goal remains unmet.

## Authoritative T4 gates

The Kaggle allocation exposed two Tesla T4 GPUs with 15,360 MiB each and driver
580.159.04. The source was reconstructed from Git commit
`5de5be39c0190b9367da18f0f2001e7f40208c92` plus the SHA-pinned Wave 130 patch.

| gate | result |
|---|---:|
| host library tests | 68 passed, 0 failed |
| serial CUDA parity | 36 passed, 0 failed |
| registers/thread | 80 |
| static shared memory | 16,384 B |
| barriers | 1 |
| stack frame | 0 B |
| spill stores / loads | 0 B / 0 B |
| driver occupancy | 3 active blocks/SM |
| Q8 output | bit-exact |
| f32 scales | bit-exact |
| full 64-row slabs / tail | 3 / 52 rows |

## Direct diagnostic timing

The harness used `hidden=4864`, `in=896`, and `ntok=244`, with 10 warmups,
100 iterations, and five repeats. It compared the retained Wave 88 stacked
N16-prefetch GEMM plus Wave 123 no-store SwiGLU against the Wave 129 fused-only
entry.

| path | median time | relative |
|---|---:|---:|
| retained two-kernel chain | 322.731 us | 1.0000x |
| Wave 129 fused candidate | 302.001 us | 1.0686x |

This timing only decides that the isolated candidate is worth a production
wall-time trial. It does not predict end-to-end prefill throughput.

## Reproduction

- Kaggle kernel: `jinxsuperdev/glcuda-t4-wave130-device-parity`
- notebook: `notebooks/glcuda_t4_wave130_device_parity.ipynb`
- direct harness: `glcuda/examples/wave130_n16_fused_swiglu.rs`
- raw archive: `benchmarks/glcuda-t4-wave130-device-parity-results.zip`
- machine-readable result:
  `benchmarks/glcuda-t4-wave130-device-parity.json`
- candidate PTX SHA-256:
  `80cdd36f64ce8873679d1a75ea0e01ad7525177a26f4ea69fe0d37737be4d06b`
- embedded patch SHA-256:
  `7fc9843b89aa443024d21ec7a4c3c9ce8b76f162aa03851dc8a0db3d778c3265`
- notebook SHA-256:
  `0b1b3db8b71e7c235e4011c8ab2d502b1eae6fb2469e74dea8ff070184de421c`
- raw archive SHA-256:
  `c6a542ab97317abe3282c64d1d01cb63d6ff7a49cbee5277e7bbb08640028b27`

## Next bounded wave

Wave 131 may wire the candidate into production prefill behind the same opt-in
contract and run counterbalanced retained/candidate production wall-time A/B.
Parity remains a hard pre-timing gate, and a regression or noisy result rejects
retention.

`glbench` still lacks eligible warps/cycle, issue-active percentage, stall
reasons, and per-kernel L1/L2 hit rates. Those missing counters do not invalidate
this parity gate, but they limit diagnosis if production timing fails to retain
the direct-kernel improvement.
