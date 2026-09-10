# Wave 75 - row-major compensated-MMA AV CLI gate

## Intended deliverable

Wave 75 was intended to integrate the Wave 71 compensated-MMA AV mechanism
without duplicating or transposing the production V cache, then license a
fused production candidate only if the direct row-major mechanism survived.

Implementation audit found that the MMA lane map already spans a complete
8x8 K/N operand when V remains `[head, key, 64]`. Each lane loads its two K
values one row (256 bytes) apart while the warp collectively covers the tile.
The diagnostic candidate was changed to that address map and both V tail
loads were explicitly predicated. This supersedes Wave 74's proposed
dimension-major permanent cache layout: no cache writer, decode reader, cache
allocation, or production dispatcher needs to change if this mechanism wins.

## Local gates

- Baseline before edits: `cargo test -p glcuda --lib --locked`, 64 passed.
- After edits: the same suite, 64 passed.
- Release diagnostic build passed:
  `cargo build --release -p glcuda --example wave64_mma_av --locked`.
- `git diff --check` passed.
- Whole-workspace `cargo fmt --all -- --check` remains red on unrelated
  pre-existing files outside this wave. The edited Rust example itself did not
  produce a formatting diff. No unrelated formatting was bundled.
- No local CUDA device or `ptxas` was available, so these checks do not prove
  PTX assembly or device correctness.

## CLI result

The requested Kaggle CLI run was submitted as
`jinxsuperdev/glcuda-t4-wave75-mma-av-row-major`, version 1. Kaggle allocated a
Tesla P100 (`sm_60`) instead of the required Tesla T4 (`sm_75`). The notebook
stopped at its first hardware gate:

```text
RuntimeError: Wave 75 requires Tesla T4 sm_75, got Tesla P100-PCIE-16GB, 6.0, 580.159.04
```

Consequently `ptxas`, device parity, and timing did not run. This is neither a
pass nor a rejection of the row-major candidate.

- notebook SHA-256:
  `eeedcaae19fb450ea03fd73436bfb3ea50a5a27320cdc2dcc2b8e37f7c3726a2`;
- embedded PTX SHA-256:
  `ef12c6cfbe495535dffc6f6402fb194aa6f60a3011a333a1a1ff7584c3c7eb1f`;
- embedded example SHA-256:
  `a7ec0964c8728d8d092894be50f702199ae5ece7e38887b7e3d8f857a8056dcf`;
- failure archive SHA-256:
  `d8f6d9e675c04d9c6d8837c610c5e31c391eefc009e444033c4381805d41e33a`.

## Decision

**HARDWARE GATE STOP; NO PERFORMANCE VERDICT.** The row-major design removes
Wave 74's broad cache-layout migration and is the narrower route to a fused
production candidate, but the first real SM75 gate is still outstanding. A
T4 run of the retained notebook must assemble with zero spills, pass all four
numeric tails, and retain at least 1.50x direct speedup before production code
is touched. The 15,000 tok/s goal remains unachieved.
