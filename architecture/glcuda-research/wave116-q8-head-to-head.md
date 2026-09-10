# Wave 116 — Q8 head-to-head production

Status: valid measurement; production target not achieved.

The T4 run used Q8_0 GwenLand and Q8_0 llama.cpp with the pinned Qwen2.5-0.5B model, exact 244-token ChatML prompt, deterministic seed 42, 10 sessions per arm, and 10 measured samples per session. Oracle parity passed 50/50 for every Gwen session and tokenizer IDs matched exactly.

| Arm | Median tok/s | P90 tok/s |
| --- | ---: | ---: |
| GwenLand retained | 12,188.4 | 12,564.8 |
| GwenLand Wave111 | 12,257.9 | 12,681.3 |
| llama.cpp | 11,307.0 | 11,584.7 |

Wave111 is +0.57% over retained and +8.41% over llama.cpp. The internal retention gate rejects Wave111 because not all paired deltas are positive (tail max delta −5.64%). Therefore this is valid evidence, not a production promotion. Target 15,000 tok/s remains unmet.

Artifact: `benchmarks/glcuda-t4-wave116-h2h-q8-results.zip` (SHA-256 `0057215ff2f9a0efbdac0fbf35659a2948e771c0c7b62e83587f53f0cd170017`).
