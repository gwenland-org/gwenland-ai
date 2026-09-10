# Wave 73 - compensated-MMA AV CLI allocation gate

## Outcome

Wave 73 submitted the repaired Wave 71 notebook through the Kaggle CLI as
requested. Kaggle's CLI route again auto-allocated a Tesla P100 (`sm_60`), not
the required Tesla T4 (`sm_75`), and the notebook stopped at its first hardware
gate.

This failure is specific to the CLI allocation route. It does not supersede
the separately reported successful manual T4 run, where the same notebook
assembled, built, passed its numeric gate, and measured a 3.5524x direct AV
speedup at capacity 244.

## CLI evidence

- Kaggle kernel: `jinxsuperdev/glcuda-t4-wave73-mma-av`, version 1.
- Notebook SHA-256:
  `c032c161bbfa9c0a157cca5a9c372cfa00dd2916f88f47f9dba5f1d531ac1c22`.
- Allocated device: `Tesla P100-PCIE-16GB, 6.0, 580.159.04`.
- Failure archive SHA-256:
  `82171de59c62ed8af97f7ef03b2787e3a18982fed4dcd478a5d4b7c095744ee5`.

```text
RuntimeError: Wave 71 requires Tesla T4 sm_75, got Tesla P100-PCIE-16GB, 6.0, 580.159.04
```

## Decision

Do not resubmit through the CLI expecting it to inherit the interactive
session's manually chosen accelerator. The CLI exposes only GPU enablement,
not a hard T4 selector. Preserve the manual T4 result as the feasibility
evidence and ingest its result ZIP before designing production integration.
