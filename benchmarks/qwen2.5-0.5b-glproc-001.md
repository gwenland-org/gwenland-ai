# glbench — glproc-qwen2.5-0.5b-instruct-q4_k_m

## Run

- **Engine:** glproc (cpu)
- **Model:** C:/Users/reyha/Downloads/qwen2.5-0.5b-instruct-q4_k_m.gguf
- **Quantization:** Q4_K_M
- **Model size:** 0.46 GiB
- **RAM:** not available (no /proc/meminfo on this OS)
- **Environment:** windows x86_64 | glbench 0.1.163
- **Run at:** unix 1784603365
- **Iterations:** 1 warmup + 1 measured

## Throughput (tokens/second)

| Phase | mean | median | min | max | p95 | std |
|-------|-----:|-------:|----:|----:|----:|----:|
| prefill | 56.9 | 56.9 | 56.9 | 56.9 | 56.9 | 0.0 |
| decode | 6.9 | 6.9 | 6.9 | 6.9 | 6.9 | 0.0 |

**Cold first run:** prefill 49.7 tok/s · decode 5.8 tok/s (excluded from the warm statistics above)

**Energy:** not available (RAPL is Linux-only; not estimated from TDP)

---
_glbench 0.1.163 · schema v1_
