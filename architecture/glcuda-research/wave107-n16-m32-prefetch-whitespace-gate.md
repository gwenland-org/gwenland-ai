# Wave 107 - N16/M32 prefetch patch-whitespace gate

## Outcome

Wave 107 regenerated the self-contained gate from the public remote-known
base `bd5c956bafb3bb6738c3f1de348a4ebb55d9c29f` and submitted it with
`--accelerator NvidiaTeslaT4`. Kaggle allocated two Tesla T4 devices, cloned
the repository, and checked out the base successfully.

The reconstruction then stopped because `git apply --whitespace=error`
correctly rejected one blank added line at EOF in the new Wave 105 PTX file.
Local reproduction with `git diff --check <base> HEAD -- glcuda` identifies
the exact defect:

```text
glcuda/src/kernels/glcuda_sm75_wave105.ptx:515: new blank line at EOF.
```

No `ptxas`, parity, occupancy, or timing ran. This remains a packaging/source
hygiene failure, not a kernel result and not a production metric.

## Evidence

- Source packaging commit: `14edde306755d3fcd2b62163b0325c96a96d8e1c`.
- Kaggle kernel: `jinxsuperdev/glcuda-wave-107-n16-m32-prefetch`, version 3.
- Allocated devices: two `Tesla T4`, compute capability `7.5`, driver
  `580.159.04`.
- Notebook SHA-256:
  `046b819198099f671dd053c1ec02cad3cede159c83e4711e1660e5c5761c523e`.
- Embedded patch SHA-256:
  `084377b77369f99dbf9c503d3b59602a4227293c9e6f0382e4d0af57822bd82c`.
- Failure phase: `reconstruct`.
- Failure archive SHA-256:
  `08a942e5394115e963145c8b6e5874da844179aa1229c08da163dca921aaff31`.

## Decision for Wave 108

Remove the single excess trailing blank line in both the reproducible PTX
generator and its generated output. Require the full base-to-HEAD patch to
pass `git diff --check` locally before regenerating and submitting the gate.
Do not alter the kernel instructions, thresholds, or T4 hard selector.

The fastest valid production measurement remains the rejected experimental
12,232.7 tok/s configuration. No new production throughput was measured.

Raw archive: `benchmarks/glcuda-t4-wave107-n16-m32-prefetch-v3.zip`

