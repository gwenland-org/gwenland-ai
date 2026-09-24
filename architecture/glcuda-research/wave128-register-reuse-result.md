# Wave 128: register-reuse T4 device result

## Verdict

Rejected at the frozen compiler-resource gate. Reusing the dead Wave88
prefetch temporary did not change the generated resource image. Production
timing was not run, so Wave 128 makes no throughput claim and the verified
15,000 prefill tok/s target remains unmet.

## Authoritative T4 result

The notebook selected CUDA device zero from a Kaggle allocation containing two
Tesla T4 GPUs. Each GPU reported 15,360 MiB; the driver was 580.159.04 with
CUDA 13.0. Host validation passed 67/67 before `ptxas` assembled the isolated
entry:

| resource | Wave 125 | Wave 128 | frozen gate |
|---|---:|---:|---:|
| registers/thread | 80 | 80 | <=80 |
| static shared memory | 16,384 B | 16,384 B | exactly 16,384 B |
| barriers | 1 | 1 | exactly 1 |
| stack frame | 8 B | 8 B | 0 B |
| spill stores | 4 B | 4 B | 0 B |
| spill loads | 4 B | 4 B | 0 B |

The byte-for-byte-identical resource outcome falsifies the repair hypothesis.
The named `%r_fgu_tmp` virtual register was not the cause of the physical
spill; `ptxas` had already coalesced it, or an overlapping live range
elsewhere still requires the same extra physical register.

The wave stopped before driver occupancy, serial CUDA parity, direct Q8
byte/scale parity, production A/B, and profiling. None of those absent
measurements may be inferred.

## Reproduction

- Notebook: `notebooks/glcuda_t4_wave128_register_reuse.ipynb`
- Raw archive: `benchmarks/glcuda-t4-wave128-register-reuse-results.zip`
- Machine-readable result:
  `benchmarks/glcuda-t4-wave128-register-reuse.json`
- Source patch SHA-256:
  `7b74dddf036616613bf5d0e2968f8acecbf6845ce9e4182de71b4db7b0f5558b`
- Raw archive SHA-256:
  `e60c2856399ad040e50759c01fc0d21abb1f7a8da0d3936653560963898e9c41`
- Notebook SHA-256:
  `8bc57b5ca86a328ffa35aba8b5416122b5948f62e57bed43257c6f838d9bdb73`

## Next bounded investigation

A later wave must identify a genuinely overlapping live range before changing
PTX again. The smallest useful experiment is compiler-only: split or reuse one
temporary whose lifetime crosses the final MMA-to-epilogue boundary, while
holding arithmetic, shared memory, launch geometry, and quantization order
fixed. Any new candidate must first demonstrate zero stack and zero spills.

`glbench` still lacks eligible warps/cycle, issue-active percentage, stall
reasons, and per-kernel L1/L2 hit rates. Those counters were not needed for this
compiler-gate rejection.
