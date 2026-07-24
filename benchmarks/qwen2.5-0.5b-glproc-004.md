# glbench — glproc-qwen2.5-0.5b-instruct-q4_k_m

## Run

- **Engine:** glproc (cpu)
- **Model:** C:/Users/reyha/Downloads/qwen2.5-0.5b-instruct-q4_k_m.gguf
- **Quantization:** Q4_K_M
- **Model size:** 0.46 GiB
- **Peak RSS:** 1.20 GiB (process high-water mark, includes model load)
- **CPU utilization:** 49.5% (measured phase, 4 logical cores)
- **Device:** 11th Gen Intel(R) Core(TM) i3-1115G4 @ 3.00GHz (4 cores)
- **RAM:** not available (no /proc/meminfo on this OS)
- **Environment:** windows x86_64 | glbench 0.1.163
- **Run at:** unix 1784872760
- **Iterations:** 2 warmup + 5 measured

## Throughput (tokens/second)

| Phase | mean | median | min | max | p95 | std | ±95% CI |
|-------|-----:|-------:|----:|----:|----:|----:|--------:|
| prefill | 89.0 | 95.9 | 62.6 | 107.8 | 106.0 | 16.0 | 22.2 |
| decode | 22.3 | 23.7 | 15.0 | 29.6 | 28.7 | 5.1 | 7.1 |

**Cold start** (5 iterations, excluded from the warm statistics above):
prefill median 88.4 tok/s (range 77.8-111.0) · decode median 33.1 tok/s (range 30.2-33.4)

**Energy:** not available (RAPL is Linux-only; not estimated from TDP)

---
_glbench 0.1.163 · schema v1_
