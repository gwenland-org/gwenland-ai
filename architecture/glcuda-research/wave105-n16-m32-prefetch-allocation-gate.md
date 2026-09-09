# Wave 105 - N16/M32 prefetch allocation gate

## Outcome

Wave 105 implemented an isolated complete-register-prefetch variant of the
retained production N16/M32 GEMM and submitted its self-contained direct gate
through the Kaggle CLI. Local host gates passed: all 66 `glcuda` library tests
and the release example build completed successfully.

Kaggle auto-allocated a Tesla P100 (`sm_60`) rather than the required Tesla T4
(`sm_75`). The notebook stopped at its first hardware gate, before clone,
`ptxas`, parity, occupancy, or timing. This is an infrastructure miss, not a
kernel result and not a production metric.

## Evidence

- Source commit: `29024366dcfd2ff15adfc6cfe95a449b36eb9f7b`.
- Gate commit: `2444aa8b3a2ed5be430e182aab9f840cf37b1433`.
- Kaggle kernel: `jinxsuperdev/glcuda-wave-105-n16-m32-prefetch`, version 1.
- Notebook SHA-256:
  `b73415c02b5965df68b8ca5ba6bcd7c5ce98cb15556d00027227a1c0d3e6b71d`.
- Embedded source patch SHA-256:
  `b5a11b0202b40cf9fe6ca9b68f19dff6b53e1eef39f6da60d32ac71b2bd6d9e5`.
- Allocated device: `Tesla P100-PCIE-16GB, 6.0, 16384 MiB, 580.159.04`.
- Failure phase: `bootstrap`.
- Failure archive SHA-256:
  `ab885641b132da174b6cd0538f33f0f257b35b2dadc144c81734997ecb2a04e2`.

```text
RuntimeError: Wave 105 requires Tesla T4 sm_75, got
['0', 'Tesla P100-PCIE-16GB', '6.0', '16384', '580.159.04']
```

## Decision for Wave 106

Do not modify the kernel, resource ceilings, numerical contract, or 1.10x
direct threshold based on this run. Resubmit the byte-identical notebook with
the Kaggle CLI's explicit `--accelerator NvidiaTeslaT4` selection. Only a real
T4 allocation may proceed to the two-run direct gate.

The fastest valid production measurement remains the rejected experimental
12,232.7 tok/s configuration. No new production throughput was measured.

Raw archive: `benchmarks/glcuda-t4-wave105-n16-m32-prefetch-v1.zip`

