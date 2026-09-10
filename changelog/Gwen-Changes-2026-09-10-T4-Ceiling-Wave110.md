# Gwen Changes - 2026-09-10 - T4 Ceiling Wave 110

- Specialized the exact FFN-down prefetch-tail guard from dynamic `k + 1 >=
  nb` to `k >= 151`, removing one integer add per K32 iteration.
- Preserved bit-exact output, 64 registers/thread, four active blocks/SM,
  9,728 bytes static shared memory, and zero spills.
- Measured 208.528 us retained versus 203.886 us candidate, only 1.0228x.
- Rejected the result at the fixed 1.10x direct gate and retired M32 complete
  prefetch after Waves 108-110 produced no qualifying result.
- No production run was performed.
