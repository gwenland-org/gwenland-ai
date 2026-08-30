# T4 Ceiling Wave 5 - double-buffered A/xscale MMA staging

## Problem

After retained Wave 4 reached 6,474.7 prefill tok/s, GEMM represented 57.5%
of the measured prompt. The 8-tile Tensor Core loop serialized activation and
xscale staging with current-block compute and paid two block barriers per
32-wide K-block.

## Change

- Doubled only `gl_gemm_mma_q8`'s activation and xscale shared tiles from
  3,328 to 6,656 bytes per CTA.
- Primed buffer 0, issued `kb+1` global loads before current-block MMA work,
  committed them to buffer 1 afterward, and rotated shared read/write bases.
- Reduced the runtime barrier schedule from `2*nb` to `nb+1` without using
  the Ampere-only `cp.async` instruction.
- Preserved the MMA shape, Q8_0 dequant epilogue, 48-byte shared row pitch,
  vector stores, staging guards, r256 kernel, and attention kernel.
- Added structural tests for the precise next-tile load/store schedule and a
  runtime banner that proves the candidate module was loaded.
- Added a direct-fetch Wave 5 Kaggle notebook with SHA-verified Wave 3, Wave 4,
  and Wave 5 patches, PTXAS/resource gates, hardware correctness, interleaved
  diagnostics, paired production sessions, telemetry, and optional NCU.

## Local validation

- PTX static preflight passes: ASCII/LF, balanced braces, preserved 80 MMA
  instructions and 40 vector stores, no `cp.async`, and unchanged r256 bytes
  in the notebook's A/B reconstruction.
- The notebook has 17 cells / 8 code cells, no saved outputs, and every code
  cell compiles as Python.
- All three embedded patches reproduce their declared SHA-256 digests; the
  Wave 5 delta contains only `glcuda_sm75.ptx` and its Rust dispatch/tests.
- `cargo test -p glcuda --lib --locked`: **42 passed**.
- `cargo test -p glcuda --test parity --locked -- --test-threads=1`: **21
  passed**.
- `forward` (**5 passed**) and `graph_replay` (**4 passed**) integration
  suites pass their available host paths.
- `cargo build -p glcuda --example bench --locked`: **passed**.
- Local CUDA hardware and PTXAS are unavailable, so assembly resources, real
  CUDA parity, and performance remain hard gates in the T4 notebook; that
  notebook rejects any silent CUDA test skip.

## T4 result and retention status

- PTXAS and hardware correctness passed; the candidate reported 63 registers
  and 6,656 bytes of shared memory per CTA.
- Production P50 was 6,861.3 tok/s versus 6,827.7 for Wave 4: **+0.49%**.
- The two paired P50 deltas were +1.21% and -0.23%; the required P50 and mean
  gains did not pass.
- Tail latency and decode stayed within their non-regression limits.

**Rejected.** The double-buffered A/xscale pipeline is retained only as a
negative experiment artifact. Product and future notebook baselines remain on
Wave 4.
