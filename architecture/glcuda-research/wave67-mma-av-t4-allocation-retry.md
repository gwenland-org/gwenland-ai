# Wave 67 - compensated-MMA AV T4 allocation retry

## Outcome

Wave 67 resubmitted the byte-identical Wave 66 notebook under a new Kaggle
kernel. Kaggle again allocated a Tesla P100 (`sm_60`) instead of the required
Tesla T4 (`sm_75`). The notebook rejected the device at its first hardware
gate, before cloning, assembling PTX, building Rust, or measuring correctness
and performance.

This is **no T4 verdict**. It neither validates nor rejects the Wave 64 AV
kernel. It is the second consecutive invalid P100 allocation for this gate.

## Local validation

- `cargo test -p glcuda --lib --locked`: 64 passed, 0 failed.
- `cargo check -p glcuda --example wave64_mma_av --locked`: pass.
- The parity executable reported 33 passed on Windows; CUDA-dependent tests
  take their documented no-device path here, so this is not GPU parity.
- Notebook SHA-256 stayed byte-identical to Wave 66.

## Submitted image

- Kaggle kernel:
  `jinxsuperdev/glcuda-t4-wave-67-mma-av-allocation-retry`, version 1.
- Notebook SHA-256:
  `4c2c9a0503f029dcbc0181ac213b1aec329e60e2a5a491f8403a0e28d238cac3`.
- Candidate PTX SHA-256:
  `8279698f0e8c0df60ca84b73d9b608661bf6b2c36a56ef09be8f4c372157999c`.
- Direct example SHA-256:
  `21fc227e11f79425dff665d280747d475c90f640f6ec7080ead646b55a3cf95b`.
- Failure archive SHA-256:
  `848c0a0a56281ca1dea4f266141ab8caf48d27c9ccd857a1ddb61a6e4c623b72`.

Kaggle normalized the requested slug during submission because the title did
not resolve to the requested ID. This changed only the remote identifier; the
submitted notebook bytes did not change.

## Failure

```text
RuntimeError: Wave 66 requires Tesla T4 sm_75, got Tesla P100-PCIE-16GB, 6.0, 580.159.04
```

The message retains `Wave 66` because the exact Wave 66 notebook was required
for this allocation-only retry.

## Next admissible action

Do not modify the candidate based on this result. A third allocation attempt
may reuse the identical image, but repeated P100 assignment should be treated
as an execution-provider blocker rather than kernel evidence. Only a real T4
allocation may proceed to `ptxas`, resource, numeric-tail, and 1.50x direct-
speed gates.
