# Wave 76 - row-major compensated-MMA AV compiler gate

## Intended deliverable

Wave 76 extracts the valid evidence a Kaggle CLI allocation can provide even
when it assigns a P100: cross-assemble the Wave 75 PTX for `sm_75`, record
resources, and only then enforce the Tesla T4 device gate. No production path
or default changes in this wave.

## Result

**SM75 COMPILER PASS.** Kaggle version 1 assembled both diagnostic entries
with `/usr/local/cuda/bin/ptxas -v -arch=sm_75`:

| entry | registers | stack | spill stores | spill loads | barriers |
|---|---:|---:|---:|---:|---:|
| `gl_wave64_av_scalar_f32` | 25 | 0 B | 0 B | 0 B | 0 |
| `gl_wave75_av_mma4_row_f32` | 42 | 0 B | 0 B | 0 B | 0 |

The row-major candidate uses two fewer registers than the 44-register
dimension-major Wave 71 candidate. This comparison is compiler evidence only;
it does not imply higher occupancy or speed.

After the compiler/resource gate passed, the notebook correctly rejected the
allocated `Tesla P100-PCIE-16GB, 6.0, 580.159.04`. Device numerical validation
and timing therefore did not run. The kernel is syntactically valid and
spill-free for SM75, but it is not yet numerically or operationally validated
on a T4.

## Evidence

- Kaggle kernel: `jinxsuperdev/glcuda-wave76-sm75-compiler-gate`, version 1.
- Notebook:
  `notebooks/glcuda_t4_wave76_mma_av_compiler.ipynb`.
- Notebook SHA-256:
  `4bea60fe943b525a1acbe2a5b0e567cc61a61cae1bc2b538cbb8459635f38608`.
- Evidence archive:
  `benchmarks/glcuda-wave76-sm75-compiler-results.zip`.
- Archive SHA-256:
  `935b8c88e8af537190571133410e26b4c16af9edeb231ff344a5e7b18f4ed0fd`.

The archive contains the raw `ptxas -v` log, parsed resource JSON, embedded
source manifest, allocation record, and the expected post-compiler device-gate
failure.

## Decision

**LICENSED FOR T4 DIRECT GATE, NOT PRODUCTION.** The Wave 75 address change no
longer carries compiler uncertainty. The next required evidence is an actual
T4 run covering capacities 1, 17, 241, and 244. It must retain max absolute
error at or below `1e-5` and median capacity-244 speedup at or above 1.50x.
Only then is integrating the mechanism into fused production attention
justified. No throughput claim, including 15,000 tok/s, is made here.
