# Wave 66 - compensated-MMA AV T4 gate

## Outcome

Wave 66 repaired the Wave 65 notebook-generation failure and submitted the
byte-locked compensated-MMA AV diagnostic. Kaggle allocated a Tesla P100
(`sm_60`) instead of the required Tesla T4 (`sm_75`). The notebook rejected
the device at its first hardware gate, before cloning, assembling PTX,
building Rust, or measuring correctness and performance.

This is **no T4 verdict**. It neither validates nor rejects the Wave 64 AV
kernel.

## Local validation

- `cargo test -p glcuda --lib --locked`: 64 passed, 0 failed.
- The parity executable reported 33 passed on Windows; CUDA-dependent tests
  take their documented no-device path here, so this is not GPU parity.
- `cargo check -p glcuda --example wave64_mma_av --locked`: pass.
- Exact-file `rustfmt --check`: pass.
- Candidate PTX: ASCII, LF-only, two entries, eight static MMA sites.
- `git diff --check`: pass before submission.

## Submitted image

- Kaggle kernel: `jinxsuperdev/glcuda-t4-wave66-mma-av`, version 1.
- Notebook SHA-256:
  `4c2c9a0503f029dcbc0181ac213b1aec329e60e2a5a491f8403a0e28d238cac3`.
- Candidate PTX SHA-256:
  `8279698f0e8c0df60ca84b73d9b608661bf6b2c36a56ef09be8f4c372157999c`.
- Direct example SHA-256:
  `21fc227e11f79425dff665d280747d475c90f640f6ec7080ead646b55a3cf95b`.
- Failure archive SHA-256:
  `0752fb8f3914c727f1e122a349a1974439db31a2806f66d8ff141085cb203353`.

## Failure

```text
RuntimeError: Wave 66 requires Tesla T4 sm_75, got Tesla P100-PCIE-16GB, 6.0, 580.159.04
```

The strict device check is intentional: P100 has no Turing INT8/f16 MMA
contract matching the candidate, so running or interpreting it there would
not answer the registered question.

## Next admissible action

Submit the identical notebook again as a new wave. Only a real T4 allocation
may proceed to `ptxas`, resource, numeric-tail, and 1.50x direct-speed gates.
No source repair or threshold change is licensed by this allocation failure.
