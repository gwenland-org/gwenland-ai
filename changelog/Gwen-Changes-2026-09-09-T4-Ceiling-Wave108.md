# Gwen Changes - 2026-09-09 - T4 Ceiling Wave 108

- Normalized the generated Wave 105 PTX to exactly one newline at EOF and
  verified the embedded notebook patch against a fresh remote-base worktree.
- Completed the first valid Tesla T4 direct gate for FFN-down N16/M32 complete
  prefetch.
- Confirmed bit-exact output and zero spills.
- Measured 208.681 us retained versus 197.780 us candidate, a 1.0551x speedup
  that fails the fixed 1.10x feasibility threshold.
- Recorded the resource mechanism: 64 to 72 registers/thread and four to
  three active blocks/SM.
- Rejected the current candidate from production; retained dispatch remains
  unchanged.
