# Gwen Changes - 2026-08-26 - T4 Ceiling Wave 11

## Added

- Added `notebooks/glcuda_t4_ceiling_wave11.ipynb`, a Kaggle-ready four-arm
  factorial from retained Wave 4.
- Added an embedded clean Wave 11 patch with exact RMSNorm/add/SiLU-to-Q8 glue
  fusion and exact three-pass GQA7 K/V tile reuse.
- Added opt-in `GLCUDA_FUSE_Q8_GLUE` and `GLCUDA_GQA_GROUP` dispatch with a
  machine-readable runtime truth-table banner.
- Added byte-exact hardware A/B tests for raw Q8 bytes, scale bits,
  materialized f32 outputs, and retained-versus-GQA7 attention output.
- Added a dedicated 10-warmup/100-measurement Wave 11 diagnostic executable.
- Added same-binary control/candidate arms, a balanced Williams sequence, four
  production repeats, strict workload/oracle parsing, P50/P90/P95/P99 reporting,
  all-arm telemetry, and always-downloadable partial failure archives.
- Added resumable direct model fetch with pinned revision/size/SHA, Range-ignore
  recovery, and explicit repair of oversized or full-but-corrupt `.part` files.
- Updated the notebook to `wave11-exact-fusion-gqa7-v2-cargo-bootstrap`: it
  detects Cargo in common Kaggle locations and otherwise installs the official
  minimal Rust toolchain before cloning/building. Toolchain versions are
  archived, and phase failures now raise a normal runtime error so IPython does
  not enter its broken `SystemExit` traceback path.
- Added the experiment contract in
  `architecture/glcuda-research/ceiling-sprint-wave11.md`.

## Validation

- Isolated `BASE + Wave 3 + Wave 4 + Wave 11` stack passes `cargo check -p
  glcuda --locked`.
- All glcuda examples/tests compile with `cargo check -p glcuda --examples
  --tests --locked`.
- The isolated stack passes 44 glcuda library tests and 24 parity test entries
  locally. CUDA branches cannot execute on this non-CUDA host; the notebook
  explicitly rejects skipped hardware tests.
- Changed Rust files pass targeted `rustfmt --check`; the patch passes `git
  diff --check`.
- Notebook JSON and all seven code cells parse; all outputs are empty.
- The embedded Wave 11 patch hash matches the isolated worktree byte-for-byte
  and passes `git apply --check --whitespace=error` against clean Wave 4.
- Local `ptxas` and Tesla T4 execution are unavailable, so assembly, resource,
  spill, bit-parity, exact oracle, and production retention remain authoritative
  Kaggle hard gates.
- Wave 11 embedded patch SHA-256:
  `0e644ae9c37ecdf73fd354a8c59100ed31bb3edce5c76e13b5f08ef4dd3292ca`.
- Notebook SHA-256:
  `1c8490b20e6e2dee6f5cb2c87afbb91673c6adcf2393a5eaf4c88d9c73986ee7`.

## Decision

Wave 11 is implemented and packaged, but not retained yet. Run the notebook on
a Tesla T4 and use its production factorial gate as the sole retention
decision. Wave 10 is not stacked into this candidate.
