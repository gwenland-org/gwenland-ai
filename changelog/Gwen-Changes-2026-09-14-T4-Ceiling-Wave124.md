# Gwen Changes - 2026-09-14 - T4 Ceiling Wave 124

- Tested one-CTA-per-token scheduling for the retained stacked SwiGLU+Q8
  production path on a verified Tesla T4.
- Passed 37 CUDA parity tests, 67 library tests, exact oracle parity for all 80
  production samples, and a zero-spill PTX resource gate.
- Measured 40 counterbalanced samples per arm: the candidate moved from
  13,091.3 to 12,995.8 tok/s (-0.73%), with both invocation orders negative.
- Rejected the candidate and removed its runtime code. Retained only the pinned
  notebook, JSON summary, and evidence archive.
- Repaired the older Wave 118 example to inherit newly added benchmark config
  fields, keeping the repository-wide glcuda all-target build green.
- Kept the 15,000 tok/s goal open. The next candidate is a fused N16-prefetch
  gate/up GEMM epilogue rather than another launch-grid reduction.
