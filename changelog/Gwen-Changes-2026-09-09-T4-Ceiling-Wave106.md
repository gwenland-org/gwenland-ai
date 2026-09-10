# Gwen Changes - 2026-09-09 - T4 Ceiling Wave 106

- Resubmitted the byte-identical N16/M32 prefetch gate with
  `--accelerator NvidiaTeslaT4`.
- Verified that Kaggle hard selection supplied two Tesla T4/sm_75 devices.
- Archived the reconstruction failure caused by selecting a local-only base
  commit after a fresh public-remote clone.
- Kept the result classified as packaging-only: no PTX assembly, numerical
  gate, timing, or production measurement ran.
- Selected the remote-known branch base `bd5c956b...` for the Wave 107
  self-contained notebook repair; kernel code and thresholds stay unchanged.
