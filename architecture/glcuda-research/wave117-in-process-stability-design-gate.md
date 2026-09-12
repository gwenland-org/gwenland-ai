# Wave 117 - in-process production stability design gate

## Decision

Wave 111 is correct but is not retained. Wave 116 measured 12,257.9 tok/s,
only 0.57% above the retained stack, with three of ten paired session deltas
below zero. Its tail was actually better than retained (-5.64% max-latency
delta), so the rejection is not evidence of a candidate-specific tail stall.
It is evidence that a sub-1% whole-model effect cannot be resolved reliably
by launching a fresh process for every arm.

Wave 117 will not promote Wave 111 and will not weaken the existing retention
gate. It freezes a benchmark-only mechanism that alternates retained and
candidate prefill iterations inside one process, on one loaded Q8_0 model and
one CUDA context. The purpose is to separate true kernel variance from process
startup, model loading, JIT, allocation, and GPU clock drift before another
15,000 tok/s candidate is judged.

No production dispatch, PTX, default environment, numerical tolerance, or
retention threshold changes in this wave.

## Why this is the next valid step

The Wave 116 production evidence is internally coherent:

- every GwenLand session passed the 50/50 token oracle;
- candidate median tail latency was lower, not higher;
- seven paired deltas were positive and three were mildly negative
  (-0.92%, -1.11%, and -0.74%);
- the direct same-allocation kernel gate reproduced a 1.35-1.37x local win;
- the whole-model median effect was only +0.57%.

The direct and production measurements therefore disagree in *effect size*,
not correctness or direction. A 23-launch removal has a small Amdahl reach and
cannot close the 22.4% gap from 12,257.9 to 15,000 tok/s. Stabilising the
measurement is prerequisite infrastructure, not a claim that residual fusion
is the missing optimization.

## Frozen implementation scope for Wave 118

Add a dedicated `glcuda` example, not a `glbench` behavior change. It will:

1. load the pinned Q8_0 model once and create one CUDA context;
2. construct retained and Wave 111 runners through an explicit benchmark-only
   configuration value, never by mutating process-global environment variables
   between iterations;
3. pre-JIT and warm both arms before collecting samples;
4. alternate `ABBA` per measured quartet for 25 quartets (100 prefill samples
   per arm), reversing the first arm in a second independent invocation;
5. synchronize with CUDA events around the complete production prefill call;
6. require the exact 244-token ChatML IDs and compare every measured output to
   the unchanged `glproc` oracle;
7. archive every raw latency, order position, dispatch contract, GPU identity,
   source revision, model SHA-256, and compiled resource report.

The explicit configuration must remain private to the example/test support or
be exposed as a narrowly scoped constructor whose normal engine construction
still uses the existing environment contract. It must not create a second
production policy path.

## Gates

### Host gate

- `cargo test -p glcuda --lib --locked`: all existing tests pass.
- New configuration tests prove retained and Wave 111 dispatch contracts are
  selected without environment mutation.
- `cargo fmt --all -- --check` and `git diff --check` pass.

### Tesla T4 correctness and resource gate

- Tesla T4, compute capability 7.5, exact pinned Q8_0 SHA-256.
- Full serial CUDA parity passes with zero failed tests.
- Both arms report the expected unchanged GEMM and attention dispatch.
- 100/100 measured outputs per arm match the same oracle; no tolerance change.
- No new PTX entry, register count, shared-memory use, or spill is permitted.

### Stability interpretation

Report the distribution of paired log-speedups with median, P10/P90, and a
bootstrap 95% confidence interval. This diagnostic passes only when both
independent invocations agree in direction and the interval excludes a 1%
regression. It does **not** retain Wave 111: the existing production rule still
requires at least +5% session-P50 and every predeclared production pair
positive.

If the in-process result is near the Wave 116 +0.57%, close residual fusion as
correct-but-too-small and keep it opt-in. If it materially disagrees, repair
the production harness before testing another optimization.

## Boundary toward 15,000 tok/s

At 244 tokens, 15,000 tok/s requires 16.267 ms. Wave 116 candidate median was
19.908 ms, leaving 3.641 ms (18.29%) to remove. Residual fusion cannot supply
that delta. After the stability harness is validated, the next performance
candidate must have a roofline path across multiple dominant stages; isolated
launch removal and single-shape probe wins are below the required reach.

Wave 117 ends at this design freeze. Implementation and T4 execution require
explicit confirmation for Wave 118.
