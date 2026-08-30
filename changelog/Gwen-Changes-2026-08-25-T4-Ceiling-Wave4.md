# T4 Ceiling Wave 4 — dynamic shared prefill attention

## Problem

After Wave 3 reached 5,732.0 prefill tok/s, attention accounted for 42.1% of
the measured 244-token prompt. Its rows kernel reserved 16,420 bytes of static
shared memory per CTA even though that prompt needs only 1,012 bytes.

## Change

- Replaced the two static attention shared arrays with one aligned extern
  dynamic shared segment and preserved the existing score/reduction layout.
- Passed the exact score capacity through the production and diagnostic kernel
  ABIs and launched with `capacity * 4 + 36` shared bytes.
- Added overflow-safe shared-size calculation and structural/unit coverage.
- Updated all attention probe and benchmark call sites without modifying the
  attention math or synchronization.
- Added a direct-fetch Wave 4 Kaggle notebook that rebuilds both Wave 3 and
  Wave 4 from embedded, SHA-verified patches, compiles both PTX modules, gates
  hardware correctness, runs paired production sessions, records telemetry,
  and optionally profiles the attention kernel with NCU.

## Local validation

- Notebook JSON has 17 cells / 8 code cells, no saved outputs, and every code
  cell passes Python AST parsing.
- Both embedded patch payloads reproduce their declared SHA-256 digests and
  apply in order from baseline commit `3bce8dd7b8aaa2765855ab927c611b54981f9241`.
- The direct model fetch pins and validates revision, filename, exact byte
  size, GGUF magic, and SHA-256.
- Corrected the notebook's launch-formula structural marker to ignore Rust
  formatting whitespace while still requiring the complete overflow-safe
  `checked_mul(4)?.checked_add(ATTN_ROWS_REDUCTION_BYTES)` expression.
- `cargo test -p glcuda --lib --locked`: **42 passed**.
- `cargo test -p glcuda --test parity --locked -- --test-threads=1`: **21
  passed**; the local machine has no CUDA device, so GPU bodies skipped and
  this is compilation/host-path evidence only.
- `forward` (**5 passed**) and `graph_replay` (**4 passed**) integration suites
  compile and pass their available host paths; the Wave 4 notebook rejects a
  silent CUDA skip on T4.
- `cargo build -p glcuda --example bench --locked`: **passed**.
- This workstation has no usable T4/PTXAS path; assembly resources, real CUDA
  parity, and all performance claims remain hard gates in the notebook.

## Retention status

**Retained on T4.** Hardware correctness and PTXAS passed. Production prefill
improved from 5,662.5 to **6,474.7 tok/s** (+14.34% median P50), and both
order-reversed sessions cleared the documented P50/mean, P95-tail, and decode
gates. The real attention diagnostic improved by 14.35%. NCU counters remained
unavailable under Kaggle's `ERR_NVGPUCTRPERM`, so the archived PTXAS and static
resource evidence remains the resource proof.
