# Gwen Changes - 2026-09-09 - T4 Ceiling Wave 87

- Audited the tempting r256 route and rejected it before implementation because
  the research ledger records a known numerical bug and explicitly forbids its
  resurrection.
- Froze an exact-Q8 N16 register-prefetch experiment for FFN gate/up. Unlike the
  rejected Wave 17 N8 attempt, the retained N16 tile supplies twice the
  independent fragment work in which to hide the next K32 global loads.
- Preserved the retained kernel, Q8 packing and scales, numerical operation
  order, down/QKV paths, and production dispatch.
- Set hard T4 gates: at most 80 registers, 9,728 bytes shared, zero spills,
  at least three blocks/SM, bit-exact output, two direct runs at least 1.10x,
  then the unchanged token-oracle and production-retention protocol.
- Made no new production throughput claim. The fastest measured experimental
  stack remains 11,377.9 tok/s.
