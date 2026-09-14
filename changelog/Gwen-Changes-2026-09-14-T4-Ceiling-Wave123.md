# Gwen Changes - 2026-09-14 - T4 Ceiling Wave 123

- Split the CUDA prefill telemetry into named attention, FFN, and glue stages
  so the remaining production bottleneck is attributable instead of hidden in
  one elementwise bucket.
- Added no-store variants for Q8 activation quantization and a stacked
  gate/up SiLU-quant path that avoids unnecessary FP32 scratch traffic.
- Added explicit runtime gates and CUDA parity coverage for the new paths.
- Measured 40 counterbalanced production samples per arm on a verified Tesla
  T4: median prefill improved from 12,772.7 to 13,309.2 tok/s (+4.20%), with
  both invocation orders positive and the oracle token unchanged.
- Retained the Wave 123 candidate. The 15,000 tok/s target remains open; the
  next measured target is the FFN GEMM pair, which accounts for about 9.15 ms
  in the diagnostic profile.
