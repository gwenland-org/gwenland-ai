# Wave 135 - two-accumulator decode GEMV result

## Verdict

Rejected. Splitting adjacent four-block input groups across two accumulation
chains changed production decode P50 from 215.48 to 215.35 tok/s (-0.06%) on
a Tesla T4. The candidate passed correctness, graph replay, and zero-spill
gates, but it did not clear the +0.5% production retention threshold.

Prefill remained above the production floor and moved from 15,515.59 to
15,605.81 tok/s (+0.58%). The candidate only changes single-token GEMV, so
this prefill movement is run-to-run variance rather than an attributed gain.
None of the candidate runtime code is retained.

## Hypothesis and result

Wave 134 showed that pairing gate and up rows sacrificed too much row-level
parallelism. Wave 135 therefore kept the retained one-warp-per-output-row
geometry and changed only the inner schedule: even and odd 128-value input
groups accumulated independently, then joined before the existing warp
reduction. Production dispatch applied the candidate to every Q8 SoA decode
projection: QKV, attention output, gate/up, down, and vocabulary head.

The extra independent chain did not improve end-to-end decode. `ptxas` raised
the kernel from 34 to 52 registers/thread without spilling. T4 already had
enough eligible row warps to cover the retained dependency chain, so the
additional instruction-level parallelism did not repay its register and
instruction overhead.

## Counterbalanced production A/B

Four independent `glbench` processes ran in control-candidate-candidate-control
order. Each used the same 244-token prompt, 64 decode tokens, two warmups, ten
measured iterations, deterministic sampling, and a 50-token `glproc` oracle.
Only `GLCUDA_DECODE_GEMV_ILP2` differed.

| Order | Arm | Prefill P50 | Decode P50 | Oracle |
|---:|---|---:|---:|---:|
| 1 | control | 15,805.67 tok/s | 216.47 tok/s | 50/50 |
| 2 | candidate | 15,619.91 tok/s | 215.50 tok/s | 50/50 |
| 3 | candidate | 15,583.82 tok/s | 215.17 tok/s | 50/50 |
| 4 | control | 15,273.54 tok/s | 213.79 tok/s | 50/50 |

Combined distributions (20 samples per arm):

| Phase | Arm | P50 | P90 | P99 | 95% bootstrap CI for P50 |
|---|---|---:|---:|---:|---:|
| prefill | control | 15,515.59 | 15,904.23 | 15,956.45 | 15,297.49-15,785.44 tok/s |
| prefill | candidate | 15,605.81 | 15,767.26 | 15,859.49 | 15,583.31-15,632.23 tok/s |
| decode | control | 215.48 | 216.71 | 217.39 | 213.96-216.27 tok/s |
| decode | candidate | 215.35 | 216.39 | 217.10 | 214.93-215.51 tok/s |

The candidate session medians both exceeded the median of the two control
session medians, but the combined candidate P50 was 0.06% lower. That mixed,
near-zero result is noise-sized and fails the explicit +0.5% gate.

## Production ceiling observed

For the retained Wave 132 production path in this run, the control distribution
reached 15,515.59 prefill tok/s and 215.48 decode tok/s at P50. Its P99 values
were 15,956.45 prefill tok/s and 217.39 decode tok/s; observed maxima were
15,959.74 and 217.50 tok/s. These are workload-, model-, and T4-specific
production measurements, not isolated kernel throughput.

## Validation

- GPU: Tesla T4, sm_75, driver 580.159.04.
- Model: 675,710,816 bytes, SHA-256
  `ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e`.
- Host library tests: 76 passed with the candidate patch.
- Serial CUDA parity: 37 passed, including 160- and 896-wide CPU-reference
  cases for the paired-loop, scalar-tail, and odd-full-group paths.
- CUDA Graph replay: 4 passed with the candidate selected.
- Candidate resources: 52 registers/thread, zero stack, zero spill, zero
  barriers, 256 threads/block.
- Retained one-row kernel: 34 registers/thread, zero stack, zero spill.
- Every production session: 50/50 oracle agreement.
- Source patch SHA-256:
  `42d5354b477e184b5f558b21766e921e28fdd96ce8c4fecd46906d7b5ff83948`.
- Result archive SHA-256:
  `6c7895563fe404b8756bc2326fcfc3e7bfa6618bd89d6f8bb240c545ab79e3a9`.

The executed notebook inherited `"wave": 134` in the raw summary object from
the Wave 134 template. The committed notebook and curated summary correct that
metadata-only typo to 135. The immutable ZIP above retains the raw output so
the correction remains auditable; measurements and decisions are unchanged.

## Next bounded investigation

Do not revisit two-chain Q8 SoA GEMV on T4. The captured profile still ranks
FFN gate/up first, but Waves 134 and 135 now reject both row pairing and inner
dot-chain ILP. A later decode candidate should reduce bytes or launches rather
than rearrange the same DP4A work, and must preserve the same parity, graph,
oracle, and counterbalanced production gates.

## Evidence

- Summary: `benchmarks/glcuda-t4-wave135-gemv-ilp2.json`.
- Four raw sessions:
  `benchmarks/glcuda-t4-wave135-glbench-{1..4}-*.json`.
- Full logs and raw records:
  `benchmarks/glcuda-t4-wave135-gemv-ilp2-results.zip`.
- Reproduction notebook:
  `notebooks/glcuda_t4_wave135_gemv_ilp2.ipynb`.
