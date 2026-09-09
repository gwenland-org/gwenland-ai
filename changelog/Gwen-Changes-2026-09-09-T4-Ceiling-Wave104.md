# Gwen Changes - 2026-09-09 - T4 Ceiling Wave 104

- Added and ran a dedicated `GLCUDA_TELEMETRY` profiler over ten Q8_0 Tesla T4
  sessions with exact dispatch and oracle gates.
- Measured the retained CUDA hotspot order as FFN gate/up 33.62%, FFN down
  23.78%, attention 16.39%, and FFN elementwise 11.06%.
- Recorded that every retained hotspot share was stable within a narrow range
  across five sessions, while absolute profiled throughput remained diagnostic.
- Selected the untested FFN-down shape for the existing N16 register-prefetch
  kernel's next isolated gate; no production default or tok/s record changed.
