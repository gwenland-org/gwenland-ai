# Wave 129: fused-only specialization T4 result

## Verdict

The compiler-resource gate passed. Specializing the Wave 128 dual-mode entry
as a fused-only kernel removed its one 32-bit spill while retaining the frozen
80-register cap, one barrier resource, and 16 KiB shared-memory image.

This wave did not wire production dispatch or run device parity, occupancy,
direct exact-output comparison, or production timing. It therefore makes no
throughput claim, and the verified 15,000 prefill tok/s goal remains unmet.

## Authoritative T4 result

The Kaggle allocation exposed two Tesla T4 GPUs, each with 15,360 MiB. The
driver was 580.159.04. `ptxas -v -arch=sm_75` reported:

| resource | Wave 128 dual-mode | Wave 129 specialized | gate |
|---|---:|---:|---:|
| registers/thread | 80 | 80 | <=80 |
| static shared memory | 16,384 B | 16,384 B | exactly 16,384 B |
| barriers | 1 | 1 | exactly 1 |
| stack frame | 8 B | 0 B | 0 B |
| spill stores | 4 B | 0 B | 0 B |
| spill loads | 4 B | 0 B | 0 B |

The source uploaded to Kaggle was byte-identical to the repository candidate.
This supports the frozen hypothesis: the runtime retained/fused mode split,
not the name of one virtual temporary, extended compiler liveness enough to
force the Wave 128 spill.

## Host validation

- before the edit: 67 passed, 0 failed;
- after the edit, including the new structural contract: 68 passed, 0 failed;
- touched Rust file passes `rustfmt --check`;
- `git diff --check` passes.

The workspace-wide format check remains red on pre-existing files outside this
diff. No unrelated formatting rewrite was bundled.

## Reproduction

- Kaggle kernel: `jinxsuperdev/glcuda-t4-wave129-fused-specialization`
- candidate PTX: `glcuda/src/kernels/glcuda_sm75_wave129.ptx`
- notebook: `notebooks/glcuda_t4_wave129_fused_specialization.ipynb`
- raw archive:
  `benchmarks/glcuda-t4-wave129-fused-specialization-results.zip`
- machine-readable result:
  `benchmarks/glcuda-t4-wave129-fused-specialization.json`
- candidate PTX SHA-256:
  `80cdd36f64ce8873679d1a75ea0e01ad7525177a26f4ea69fe0d37737be4d06b`
- notebook SHA-256:
  `2ba5a3fe3533bb1626f777b96943556cfbaf3095ef997c2674f8e7b33a120964`
- raw archive SHA-256:
  `7834e64ed92813b4c27cca868377879f1e6cd5a6558488adfa9a18800413c3bc`

## Next bounded wave

Wave 130 may integrate the specialized entry behind an opt-in benchmark guard,
query real driver occupancy, run the serial CUDA parity suite, and require
exact Q8 bytes plus f32 scale bits for the 244-row production shape including
the 52-row ragged tail. Any failure stops before production timing.

`glbench` still lacks eligible warps/cycle, issue-active percentage, stall
reasons, and per-kernel L1/L2 hit rates. Those counters were not needed to
decide this compiler-resource gate.
