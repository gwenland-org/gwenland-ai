# Wave 69 - compensated-MMA AV API allocation gate

## Outcome

Wave 69 resumed the blocked T4 gate after a T4 was reported available. The
byte-identical notebook was submitted through the Kaggle API under
`jinxsuperdev/glcuda-t4-wave69-mma-av`. The API-created run nevertheless
received a Tesla P100 (`sm_60`), not the required Tesla T4 (`sm_75`).

The notebook correctly stopped at its first hardware gate. It did not clone,
assemble PTX, build Rust, or measure numeric correctness and speed. This is
**no T4 verdict** and does not validate or reject the Wave 64 candidate.

## Reproducibility

- Notebook SHA-256:
  `4c2c9a0503f029dcbc0181ac213b1aec329e60e2a5a491f8403a0e28d238cac3`.
- Candidate PTX SHA-256:
  `8279698f0e8c0df60ca84b73d9b608661bf6b2c36a56ef09be8f4c372157999c`.
- Direct example SHA-256:
  `21fc227e11f79425dff665d280747d475c90f640f6ec7080ead646b55a3cf95b`.
- Failure archive SHA-256:
  `04b34c7d0079d5156917cb7bd79404a83d737df5427b0ec81483afc8935cb37e`.

## Failure

```text
RuntimeError: Wave 66 requires Tesla T4 sm_75, got Tesla P100-PCIE-16GB, 6.0, 580.159.04
```

The retained `Wave 66` label proves that the registered notebook was reused
without source edits.

## Deviation and next route

The reported T4 availability did not carry into an API-created Kaggle kernel;
Kaggle's auto-allocation still selected P100. A following wave should use the
interactive Kaggle session whose accelerator is visibly set to T4, import the
same notebook, verify `nvidia-smi` before execution, and then run all cells.
Do not change the kernel or thresholds based on this allocation failure.
