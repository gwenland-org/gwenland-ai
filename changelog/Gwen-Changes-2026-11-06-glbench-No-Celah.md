# Gwen Changes - 2026-11-06 - glbench No Celah

- Fixed `--profile stages` on glcuda being refused at every archive write, a
  dispatch fingerprint that missed the Wave 132 and 136 flags, and a decode
  ceiling built on the file size rather than the bytes the engine streams.
- Added report verbosity (`-q`, default, `-v`, `-vv`). Greedy runs a second
  instrumented pass and prints its measured cost: prefill -16.6%, decode -0.4%.
- Made `--kind` real: it was archived and read by nothing.
- Added a device read-bandwidth probe. The T4 measures 267 GB/s of a published
  320, which moves decode from 37% of an estimated ceiling to 44% of a measured
  one.
- Added `trend` and `list`, per-stage drift in `compare`, and comparison axes,
  so glproc against glcuda is no longer called a regression.
- Fixed CUDA stage rows being percentages of host DDR, and `total_ms` carrying
  no overhead because it was built from the phases it was compared against.
- Removed `runner/warmup.rs`.
- Found, not explained: prefill alone runs at ~13,700 tok/s against ~15,800 with
  decode between iterations on the same build.
- Results: `benchmarks/glbench-t4-wired-results.zip`.
