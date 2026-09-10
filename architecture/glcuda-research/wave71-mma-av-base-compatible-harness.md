# Wave 71 - base-compatible compensated-MMA AV harness

## Purpose

Repair the Wave 70 notebook build failure without changing the candidate PTX,
numeric contract, test shapes, timing protocol, or 1.50x direct-speed gate.

## Root cause

The notebook checks out public base revision
`3bce8dd7b8aaa2765855ab927c611b54981f9241` and embeds only the diagnostic PTX
and Rust example. The example called `Cuda::max_active_blocks_per_sm`, but that
method is absent from the pinned base. The T4 assembled both PTX entries before
Cargo rejected the host harness, proving that the failure was packaging skew
rather than a PTX failure.

## Repair

The example no longer calls the branch-only occupancy helper. Its resource
record now states `occupancy_source: ptxas-only`; the notebook already runs
`ptxas -v` and records registers, spill stores, spill loads, and stack bytes for
both entries before building the example.

No candidate or decision threshold changed:

- candidate PTX SHA-256:
  `8279698f0e8c0df60ca84b73d9b608661bf6b2c36a56ef09be8f4c372157999c`;
- numeric max-absolute gate: `1e-5`;
- shapes: capacities `1`, `17`, `241`, and `244`;
- timing: 10 warmups, 100 measured launches, 5 interleaved repeats;
- production-shape direct speed gate: `1.50x`.

## Validation

- `cargo test -p glcuda --lib --locked`: 64 passed, 0 failed before and after.
- `cargo check -p glcuda --example wave64_mma_av --locked`: pass.
- The repaired example was copied into a detached checkout of the exact pinned
  base revision and the same Cargo check passed there.
- Exact-file `rustfmt --check`: pass.
- Generated notebook Python syntax: pass.
- Pre-normalization notebook SHA-256 (rejected at the Wave 71 artifact gate):
  `81c3724b9d144113d9bf94a54979c9bb7de6adc7141aad3b45f440138815100c`.
- Embedded example SHA-256:
  `7cf1072e2888ee2474ec3fb42836d810812cdfcdaffa37161e00a4c61db05e19`.

## Scope

The kernel remains diagnostic-only and is not loaded by `KernelSet`; production
dispatch and production throughput are unchanged. A real T4 run of the Wave 71
notebook is the next gate. Even a direct-kernel feasibility pass will not be a
production performance claim; it only licenses a later integration wave and
production `glbench` measurement.
