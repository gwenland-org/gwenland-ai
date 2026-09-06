# Wave 68 - compensated-MMA AV provider blocker

## Outcome

Wave 68 made the third allocation-only submission of the byte-identical Wave
66 notebook. Kaggle again allocated a Tesla P100 (`sm_60`) instead of the
required Tesla T4 (`sm_75`). The notebook rejected the device at its first
hardware gate, before cloning, assembling PTX, building Rust, or measuring
correctness and performance.

This is **no T4 verdict**. It neither validates nor rejects the Wave 64 AV
kernel. Waves 66, 67, and 68 all received the same P100 allocation, so the
current Kaggle auto-allocation route is now a verified provider blocker.

## Reproducibility

- Kaggle kernel: `jinxsuperdev/glcuda-t4-wave68-mma-av`, version 1.
- Notebook SHA-256:
  `4c2c9a0503f029dcbc0181ac213b1aec329e60e2a5a491f8403a0e28d238cac3`.
- Candidate PTX SHA-256:
  `8279698f0e8c0df60ca84b73d9b608661bf6b2c36a56ef09be8f4c372157999c`.
- Direct example SHA-256:
  `21fc227e11f79425dff665d280747d475c90f640f6ec7080ead646b55a3cf95b`.
- Failure archive SHA-256:
  `d0cd3efad5a8ad8978d6df89fdc0b3f603a14e0a7f8c8f998dec72f97daaa0ca`.

## Failure

```text
RuntimeError: Wave 66 requires Tesla T4 sm_75, got Tesla P100-PCIE-16GB, 6.0, 580.159.04
```

The message retains `Wave 66` because all allocation retries intentionally use
the exact Wave 66 notebook bytes.

## Gate decision

Do not modify the candidate and do not claim a performance result. Further
automatic resubmission to the same Kaggle pool is not evidence-producing work
after three identical allocations. Resume only when a real T4 (`sm_75`) runner
is available, then execute the existing notebook unchanged through `ptxas`,
resource, numeric-tail, and 1.50x direct-speed gates.
