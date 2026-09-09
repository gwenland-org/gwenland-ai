# Wave 100 - T4 Cargo bootstrap diagnostic

## Result

Wave 94 version 2 reached the requested Tesla T4 allocation and passed both
PTXAS resource gates. The retained N16 kernel compiled at 72 registers and the
prefetch candidate compiled at the declared 80-register ceiling; both used
9,728 bytes of shared memory with zero stack frame and zero spills.

The notebook then stopped in `build-test` before host tests, parity, occupancy,
or timing because it required Cargo at the image-specific path
`/root/.cargo/bin/cargo`. That path was absent in this Kaggle image even though
the CUDA toolchain was available.

This is a notebook bootstrap defect, not a kernel rejection or a performance
result. The candidate remains unmeasured.

## Authoritative evidence

- observed devices: two Tesla T4 GPUs, compute capability 7.5;
- driver: 580.159.04;
- retained resources: 72 registers, 9,728 bytes shared, zero spills;
- candidate resources: 80 registers, 9,728 bytes shared, zero spills;
- failed phase: `build-test`;
- source revision: `604acd0b20f20aa40a9d0e15814dd30675914e5f`;
- source patch SHA-256:
  `272ae9add1a8b1b9f56ff4bee97d95aad2fdb461c29f2224388257896cae0279`;
- raw archive: `benchmarks/glcuda-t4-wave100-cargo-bootstrap-miss.zip`;
- archive SHA-256:
  `cf84e63e1ae25e1434350d86cd78309e3ddf1be5d821836d19dfd8688fb7e367`.

## Repair

Wave 101 must resolve Cargo from `PATH` first and use known installation paths
only as fallbacks. It must keep the hard T4 device check and rerun the unchanged
source revision and direct gate twice.

No production number changes. The fastest measured experimental stack remains
11,377.9 tok/s and the 15,000 tok/s goal remains open.
