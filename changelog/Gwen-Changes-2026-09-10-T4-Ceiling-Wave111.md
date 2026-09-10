# Gwen Changes - 2026-09-10 - T4 Ceiling Wave 111

- Deferred every non-final prefill FFN residual add into the next layer's
  existing fused attention RMSNorm+Q8 pass, removing 23 launches per 24-layer
  chunk without a new buffer or PTX kernel.
- Preserved bit-exact updated residuals, normalized output, Q8 payload, and Q8
  scales in two Tesla T4 direct runs.
- Measured 1.3511x and 1.3688x direct speedups, clearing the fixed 1.05x gate.
- Removed rejected Wave 105/109 runtime modules, dispatch plumbing, PTX images,
  direct harnesses, and generators while retaining their historical evidence.
- Kept the candidate opt-in pending Wave 112 production `glbench` A/B.
