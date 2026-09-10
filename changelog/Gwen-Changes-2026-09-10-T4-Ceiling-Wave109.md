# Gwen Changes - 2026-09-10 - T4 Ceiling Wave 109

- Added an isolated epilogue-rematerialized N16/M32 prefetch candidate for the
  exact FFN-down production shape.
- Passed 67/67 local library tests, release build, strict source-patch
  reconstruction, T4 ptxas, zero-spill, driver occupancy, and bit-exact gates.
- Restored the retained resource tier: 64 registers/thread and four active
  blocks/SM, versus Wave 108's 72 registers and three blocks/SM.
- Measured 220.805 us retained versus 202.166 us candidate, a 1.0922x speedup.
- Rejected the result because it remains below the fixed 1.10x direct gate;
  no production run was performed.
