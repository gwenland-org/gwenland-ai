# Gwen Changes - 2026-09-09 - T4 Ceiling Wave 103

- Repaired the Wave 102 dispatch checker to treat N16 prefetch as a replacement
  path and verified both baseline and candidate contracts before submission.
- Completed twenty Q8_0 production `glbench` sessions on Tesla T4.
- Measured 11,700.4 tok/s retained versus 12,232.7 tok/s candidate, a 4.55%
  median gain, with exact oracle agreement in every session.
- Rejected retention because one of ten pairs was negative and median
  session-max latency regressed 5.84%, above the 5% ceiling.
- Kept N16 prefetch opt-in and recorded 12,232.7 tok/s as the fastest measured
  experimental production result, still 22.62% below the 15,000 tok/s goal.
