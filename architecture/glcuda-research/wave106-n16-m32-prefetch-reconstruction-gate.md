# Wave 106 - N16/M32 prefetch T4 reconstruction gate

## Outcome

Wave 106 resubmitted the byte-identical Wave 105 notebook with Kaggle's
explicit `--accelerator NvidiaTeslaT4` selector. Hard selection worked: the
run received two Tesla T4 devices (`sm_75`) and passed the hardware gate.

The notebook then stopped at the repository reconstruction gate. Its embedded
base revision was the local-only Wave 104 evidence commit
`e9dfef911e99143d1d8fcf2090fdea73e8f4d83c`; that object is not present in the
public GitHub remote, so a fresh Kaggle clone could not check it out. No
`ptxas`, parity, occupancy, or timing ran. This is a packaging failure, not a
kernel result and not a production metric.

## Evidence

- Kaggle kernel: `jinxsuperdev/glcuda-wave-105-n16-m32-prefetch`, version 2.
- Submission accelerator: `NvidiaTeslaT4`.
- Allocated devices: two `Tesla T4`, compute capability `7.5`, driver
  `580.159.04`.
- Notebook SHA-256:
  `b73415c02b5965df68b8ca5ba6bcd7c5ce98cb15556d00027227a1c0d3e6b71d`.
- Failure phase: `reconstruct`.
- Failure archive SHA-256:
  `b430e23b1afc9a2336fb327be6521a8864c9c5cdb687f5f703e6f9f4045b8127`.

```text
RuntimeError: command failed (128):
['git', 'checkout', '--detach', 'e9dfef911e99143d1d8fcf2090fdea73e8f4d83c']
```

## Decision for Wave 107

Regenerate the self-contained notebook from the remote-known branch base
`bd5c956bafb3bb6738c3f1de348a4ebb55d9c29f`, embedding the complete binary
diff through the Wave 105 implementation commit. Keep the kernel, resource
ceilings, numerical contract, direct threshold, and T4 hard selector
unchanged.

The fastest valid production measurement remains the rejected experimental
12,232.7 tok/s configuration. No new production throughput was measured.

Raw archive: `benchmarks/glcuda-t4-wave106-n16-m32-prefetch-v2.zip`
