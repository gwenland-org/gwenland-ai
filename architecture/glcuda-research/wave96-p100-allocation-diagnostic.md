# Wave 96 - Kaggle P100 allocation diagnostic

## Result

Wave 94 version 1 did not execute the N16-prefetch experiment. Kaggle allocated
a Tesla P100 (`sm_60`) although the metadata requested a generic GPU. The
notebook's hard device gate observed the mismatch, archived the partial run,
and stopped in `bootstrap` before cloning, PTXAS, compilation, parity, or
timing.

This is an infrastructure allocation miss, not a kernel compile, correctness,
or performance result. It provides no evidence for or against the candidate.

## Authoritative evidence

- observed device: `Tesla P100-PCIE-16GB`, compute capability 6.0;
- driver: 580.159.04;
- failed phase: `bootstrap`;
- source patch SHA-256:
  `272ae9add1a8b1b9f56ff4bee97d95aad2fdb461c29f2224388257896cae0279`;
- raw archive:
  `benchmarks/glcuda-t4-wave94-p100-allocation-miss.zip`;
- archive SHA-256:
  `674622f9d971dc9885ff8215d17d7eb8b900244698ad5dc5cb42946eca720b97`.

## Repair

The installed Kaggle CLI exposes `kaggle kernels push --accelerator ACC`, and
the official metadata documentation names `NvidiaTeslaT4` as the T4 machine
shape. Wave 97 must push the same validated notebook as a new version with
`--accelerator NvidiaTeslaT4` and still verify the actual device from inside
the notebook.

No production number changes. The fastest measured experimental stack remains
11,377.9 tok/s and the 15,000 tok/s goal remains open.
