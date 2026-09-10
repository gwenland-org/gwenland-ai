# Wave 114 - Q8 head-to-head probe failure

Wave 114 passed the full expensive setup: Tesla T4 qualification, 66 host
tests, 33 CUDA parity tests, exact Q8 model verification, pinned llama.cpp
CUDA build and smoke, and the intended two-arm GwenLand dispatch probes. It
stopped before measurement because the production parser used `re` without
the bootstrap importing it.

No throughput number is valid and zero of 30 sessions completed. Wave 115 adds
the missing standard-library import; no benchmark gate or acceptance rule is
changed.

Evidence: `benchmarks/glcuda-t4-wave114-h2h-q8-failure.zip` (SHA-256
`48348f4db6fe05395a03ecc8f500d1e0c45f733487786ad262df578b90e6341b`).
