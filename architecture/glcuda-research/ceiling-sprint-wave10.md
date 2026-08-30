# T4 Ceiling Sprint - Wave 10

**Status:** ready for T4 confirmation  
**Baseline:** retained Wave 4 (`BASE + Wave 3 + Wave 4`)  
**Target:** 15,000+ prefill tok/s on Tesla T4

## Why this wave

Wave 8 rowCTA and Wave 9 L2 raster each preserved exact arithmetic and each
produced roughly a 3.3-3.4% end-to-end gain in separate Kaggle runs. Neither
cleared the 5% retention gate. Absolute throughput also moved with the Kaggle
node, so comparing their headline numbers across notebooks cannot determine
which optimization is faster or whether they stack.

Wave 10 changes no arithmetic. It measures the two scheduling mechanisms as a
2x2 factorial in one allocation:

1. retained Wave 4;
2. Wave 4 plus Wave 8 rowCTA;
3. Wave 4 plus Wave 9 L2 raster;
4. Wave 4 plus both mechanisms.

The combined arm uses one patch relative to Wave 4. This avoids depending on
the order in which the independently authored Wave 8 and Wave 9 patches are
applied. The only textual overlap was the `KernelSet` dispatch state; both
fields, environment reads, and initializers are retained.

## Measurement protocol

The notebook runs four production repeats. A cyclic Latin square puts every
arm in each run position exactly once:

```text
W4  W8  W9  W10
W8  W9  W10 W4
W9  W10 W4  W8
W10 W4  W8  W9
```

Every session uses the same pinned model, prompt, token count, warmup, measure
count, seed, greedy sampling, and `glproc` oracle. GPU P-state, temperature,
power, SM clock, memory clock, utilization, and memory use are sampled directly
before and after every production session. These snapshots diagnose node and
thermal drift; they do not replace paired production measurements.

## Correctness and resource contract

Every arm must pass:

- both PTX modules assembled by `ptxas -arch=sm_75`;
- zero spill stores and loads;
- at least the 24-resident-warp resource tier for the MMA kernel and rowCTA
  kernel where present;
- the complete `glcuda` library and hardware parity suites;
- the relevant exact rowCTA and L2-raster bit-parity tests;
- exact greedy next-token parity against `glproc` in every production session.

Rejected Wave 5/6/7 markers and `cp.async` are structural failures. The MMA
shape, Q8_0 arithmetic, attention dynamic shared-memory change, and grid64
selection remain fixed.

## Retention gate

Each candidate is evaluated independently against Wave 4. Retain only if:

- the median production P50 gain is at least 5%;
- every paired repeat improves both P50 and mean by at least 5%;
- P95 prefill latency does not regress by more than 5%;
- decode P50 does not regress by more than 5%;
- exact oracle parity and all resource gates remain green.

The report also records rowCTA main effect, raster main effect, combined effect,
and multiplicative interaction. Diagnostic microbenchmarks and singleton stage
telemetry are not retention evidence.

## Artifacts

- Notebook: `notebooks/glcuda_t4_ceiling_wave10.ipynb`
- Notebook build: `wave10-rowcta-raster-factorial-v2` (the rowCTA structural
  gate follows the actual `KernelSet` dispatch branch in `kernels/mod.rs`).
- Combined patch SHA-256:
  `37d7df45dd48eb02cf10d8e3a94e8f210d679eaea4e4134d8eb63a8bb5e45ce3`
- Notebook SHA-256:
  `541a60641f16775b3e63b7b7b89159716cac3fdf21aba6f4f90be7718c4c0480`
