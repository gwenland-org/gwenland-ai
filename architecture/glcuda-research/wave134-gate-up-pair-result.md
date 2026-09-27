# Wave 134 - paired gate/up decode GEMV result

## Verdict

Rejected. Pairing corresponding gate and up rows in one Q8 SoA warp reduced
production decode P50 from 218.66 to 217.73 tok/s (-0.43%) on a Tesla T4.
Both candidate sessions were slower than the median of the two controls, so
the result is directional rather than a single noisy outlier.

Prefill was unchanged at 15,930.47 versus 15,936.21 tok/s (+0.04%), and all
four sessions matched 50/50 verification tokens against `glproc`. The
candidate was correct and zero-spill, but not faster; none of its runtime code
is retained.

## Hypothesis and result

The retained Q8 SoA GEMV assigns one warp to each output row. Wave 134 assigned
one warp to matching rows `gate[r]` and `up[r]` so both dot products could
share activation bytes, activation scales, loop control, and address
arithmetic. Two independent `dp4a` chains were intended to add instruction-
level parallelism without changing weight traffic.

The T4 result falsifies that trade. Activation data is tiny and cache-resident
relative to the two weight rows, while pairing halves the number of active row
warps. The added state raised the compiled register count from 34 for the
retained one-row kernel to 49 for the candidate. Although that caused no
spills, shared activation work and extra ILP did not repay the reduced warp
parallelism.

## Counterbalanced production A/B

Four independent `glbench` processes ran in control-candidate-candidate-control
order. Each used the same 244-token prompt, 64 decode tokens, two warmups, ten
measured iterations, deterministic sampling, and a 50-token `glproc` oracle.
Only `GLCUDA_DECODE_GATE_UP_PAIR` differed.

| Order | Arm | Prefill P50 | Decode P50 | Oracle |
|---:|---|---:|---:|---:|
| 1 | control | 15,930.47 tok/s | 219.14 tok/s | 50/50 |
| 2 | candidate | 15,896.50 tok/s | 217.63 tok/s | 50/50 |
| 3 | candidate | 15,945.15 tok/s | 217.91 tok/s | 50/50 |
| 4 | control | 15,912.64 tok/s | 218.00 tok/s | 50/50 |

Combined distributions (20 samples per arm):

| Phase | Arm | P50 | P90 | P99 | 95% bootstrap CI for P50 |
|---|---|---:|---:|---:|---:|
| prefill | control | 15,930.47 | 15,985.92 | 16,007.43 | 15,866.16-15,952.84 tok/s |
| prefill | candidate | 15,936.21 | 15,989.10 | 16,010.38 | 15,839.47-15,967.61 tok/s |
| decode | control | 218.66 | 219.60 | 219.70 | 218.15-219.14 tok/s |
| decode | candidate | 217.73 | 218.68 | 219.10 | 217.37-218.13 tok/s |

The candidate decode CI sits below most of the control CI and its two session
medians are both below the control-session median. The -0.43% result therefore
fails the retention gate even though its absolute size is modest.

## Validation

- GPU: Tesla T4, sm_75, driver 580.159.04.
- Model: 675,710,816 bytes, SHA-256
  `ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e`.
- Host library tests: 76 passed with the candidate patch.
- Serial CUDA parity: 37 passed, including the paired-kernel CPU-reference
  test.
- CUDA Graph replay: 4 passed with the candidate selected.
- Candidate resources: 49 registers/thread, zero stack, zero spill, zero
  barriers, 256 threads/block.
- Retained one-row kernel: 34 registers/thread, zero stack, zero spill.
- Every production session: 50/50 oracle agreement.
- Source patch SHA-256:
  `c1c4e4e148892f5db3862a6ad084b2589e9ac18533b5882b585ed97f85385126`.
- Result archive SHA-256:
  `e7b7d1ca3d64ab83e565cf6275a4f20371ea48852a1aede470af27404268470f`.

## Next bounded investigation

Do not revisit gate/up row pairing. A later candidate should keep the retained
one-warp-per-row grid and expose independent work *within* each row, for
example a two-accumulator input-group schedule. That tests dot-chain latency
without sacrificing row-level parallelism and, if generalized, can benefit
QKV, output, down, and vocabulary projections as well as gate/up. It still
requires an isolated parity gate and counterbalanced production A/B.

## Evidence

- Summary: `benchmarks/glcuda-t4-wave134-gate-up-pair.json`.
- Four raw sessions:
  `benchmarks/glcuda-t4-wave134-glbench-{1..4}-*.json`.
- Full logs and raw records:
  `benchmarks/glcuda-t4-wave134-gate-up-pair-results.zip`.
- Reproduction notebook:
  `notebooks/glcuda_t4_wave134_gate_up_pair.ipynb`.
