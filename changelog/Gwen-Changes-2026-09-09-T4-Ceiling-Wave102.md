# Gwen Changes - 2026-09-09 - T4 Ceiling Wave 102

- Added a production Q8_0 A/B harness comparing the Wave 80 stack with and
  without N16 register prefetch.
- Reconfirmed the candidate at 1.2299x and 1.2595x direct speedup with bit-exact
  output before entering production stabilization.
- Stopped before production timing because the harness treated a replacement
  dispatch label as an additional label.
- Excluded both cold one-iteration stabilization readings from production
  claims and retained 11,377.9 tok/s as the current experimental record.
