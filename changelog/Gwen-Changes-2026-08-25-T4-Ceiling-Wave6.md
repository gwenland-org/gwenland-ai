# T4 Ceiling Wave 6 - 128-row Tensor Core GEMM candidate

## Change

- Added opt-in `gl_gemm_mma_q8_r128` for a 128-row by 32-column output tile
  with a 128-thread CTA.
- Reused every B fragment across 16 M tiles while keeping the proven r256
  prefetch/dequant schedule and reducing its accumulator set by half.
- Added `GLCUDA_R128=1` dispatch. It selects r128 only when it launches fewer
  token slabs than grid64; `ntok <= 64` remains on retained grid64.
- Preserved the retained grid64/r256 bodies, attention PTX, MMA shape,
  barriers, Q8_0 epilogue, shared pitch, staging guards, and vector stores.
- Added structural tests, launch-policy tests, r128 reference parity shapes,
  a diagnostic bench route, and a direct-fetch Wave 6 Kaggle notebook.

## Intended mechanism

For the 244-token production prompt, r128 reduces weight-fragment slab work
from four token tiles to two while exposing 56 CTAs for a representative
896-column layer. It targets the reuse/parallelism gap between grid64 and the
under-populated r256 route. This is independent of the rejected Wave 5
A/xscale pipeline.

## Local gate

The source-level gate requires exactly 32 MMA instructions, 16 vector stores,
four 64-bit A staging writes, 6,656 bytes of shared memory, no `cp.async`, and
byte-identical retained kernels. PTXAS resources, CUDA correctness, and paired
production performance remain hard gates in the notebook because the local
host does not provide a T4.

- Notebook self-audit: 17 cells / 8 code cells, no saved outputs; all code
  cells parse as Python.
- Embedded Wave 3, Wave 4, and Wave 6 patches reproduce their declared
  SHA-256 digests; the Wave 6 payload exactly matches the five-file product
  delta from retained Wave 4.
- Structural gate: passed, including byte-identical grid64/r256 kernel bodies
  and unchanged main attention PTX.
- `cargo test -p glcuda --lib --locked`: **43 passed**.
- `cargo test -p glcuda --test parity --locked -- --test-threads=1`: **22
  passed**.
- `forward`: **5 passed**; `graph_replay`: **4 passed**.
- `cargo build -p glcuda --example bench --locked`: **passed**.

Local integration tests exercised their available host paths. The notebook
requires positive CUDA execution evidence before accepting hardware parity.

## Retention status

**Pending the T4 Wave 6 artifact.** Retained Wave 4 remains the product
baseline until r128 passes zero-spill/resource checks, hardware correctness,
and both paired >=5% production gates.
