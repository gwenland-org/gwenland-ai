# ARTX11 — Benchmarks

## Architecture Specification

**Document Name:** ARTX11_Benchmarks.md  
**Codename:** Mensa Rotunda  
**Tagline:** Designed for the Impossible.  
**Version:** 1.0.0-draft  
**Date:** 2026-07-10  
**Status:** Draft  

---

## Revision History

| Version | Date | Author | Description |
|---------|------|--------|-------------|
| 1.0.0-draft | 2026-07-10 | GLLM Core Team | Extracted from ArchGLLMFormat.md master document |

---

## Table of Contents

- [Benchmark Philosophy](#benchmark-philosophy)
- [Memory Benchmarks](#memory-benchmarks)
  - [Working Set Size](#working-set-size)
  - [Peak Memory Usage](#peak-memory-usage)
- [Runtime Benchmarks](#runtime-benchmarks)
  - [Tokens Per Second](#tokens-per-second)
  - [Time to First Token](#time-to-first-token)
- [Loading Benchmarks](#loading-benchmarks)
  - [Cold Start Time](#cold-start-time)
  - [Layer Load Time](#layer-load-time)
- [Package Size Benchmarks](#package-size-benchmarks)
  - [Storage Overhead](#storage-overhead)
- [Sequential Loading Benchmarks](#sequential-loading-benchmarks)
  - [Prefetch Efficiency](#prefetch-efficiency)
  - [Working Set Fluctuation](#working-set-fluctuation)
- [Storage Comparison](#storage-comparison)
  - [Format Size Comparison](#format-size-comparison)
- [Benchmark Methodology](#benchmark-methodology)
  - [Hardware Configuration](#hardware-configuration)
  - [Measurement Protocol](#measurement-protocol)
  - [Reporting Format](#reporting-format)
- [Related Documents](#related-documents)

---

# Benchmark Philosophy

GLLM benchmarks measure the format and runtime from the perspective of inference deployment. All benchmarks are reproducible, documented, and run on standardized hardware configurations.

> **Important:** Benchmarks in this section use illustrative values where actual measured data is unavailable. These values are explicitly marked as estimates or demonstrations. No performance claims are made without verification.

---

# Memory Benchmarks

## Working Set Size

The working set size is the amount of memory required to execute inference, excluding the KV cache and activation buffers.

**Figure Prompt:** Generate a bar chart comparing working set size for GLLM (sequential loading), GGUF (full mmap), and Safetensors (full load) for a 70B parameter model quantized to Q4_K_M. X-axis: format; Y-axis: working set size in GB. GLLM should show ~4GB, GGUF ~35GB, Safetensors ~35GB.

```python
# Python figure prompt for memory comparison
import matplotlib.pyplot as plt

formats = ["GLLM", "GGUF", "Safetensors"]
working_set = [4.2, 35.0, 35.0]  # Illustrative values only

plt.figure(figsize=(8, 5))
plt.bar(formats, working_set, color=['#2ecc71', '#3498db', '#e74c3c'])
plt.ylabel('Working Set Size (GB)')
plt.title('Memory Working Set: 70B Q4_K_M Model (Illustrative)')
plt.ylim(0, 40)
for i, v in enumerate(working_set):
    plt.text(i, v + 1, f'{v} GB', ha='center')
plt.savefig('memory_working_set.png')
```

## Peak Memory Usage

Peak memory includes the working set, KV cache, and activation buffers.

**Figure Prompt:** Generate a stacked area chart showing memory usage over time during a 4096-token generation. Layers: working set (fluctuating), KV cache (growing), activation buffers (constant). X-axis: token index; Y-axis: memory in GB.

---

# Runtime Benchmarks

## Tokens Per Second

Throughput is measured as tokens generated per second during autoregressive generation.

**Figure Prompt:** Generate a line chart comparing tokens per second for GLLM and GGUF on a 70B Q4_K_M model across context lengths (512, 1024, 2048, 4096, 8192). X-axis: context length; Y-axis: tokens/second. Two lines: GLLM and GGUF.

> **Note:** Actual throughput depends on hardware (GPU model, CPU cores, RAM bandwidth), quantization scheme, and batch size. The figure above is illustrative.

## Time to First Token

Time to first token (TTFT) measures the latency from input submission to the first output token.

**Figure Prompt:** Generate a bar chart comparing TTFT for GLLM (sequential loading) and GGUF (full load) on a 70B model. X-axis: format; Y-axis: TTFT in seconds. Include error bars for variance.

---

# Loading Benchmarks

## Cold Start Time

Cold start time is the time from package discovery to the first layer execution.

**Figure Prompt:** Generate a timeline chart showing the loading phases for GLLM: manifest parse (10ms), shared component map (50ms), layer 0 prefetch (200ms), total to first token. Compare with GGUF full map time.

## Layer Load Time

Layer load time is the time from `mmap` initiation to the first tensor access.

**Figure Prompt:** Generate a histogram of layer load times for 80 layers on an NVMe SSD. X-axis: load time in ms; Y-axis: frequency. Show mean and median lines.

---

# Package Size Benchmarks

## Storage Overhead

GLLM introduces minimal storage overhead compared to the raw tensor data.

| Component | Overhead | Description |
|-----------|----------|-------------|
| Manifest | ~50KB | JSON metadata |
| Layer Headers | ~1KB per layer | Tensor index |
| Checksums | ~64 bytes per file | SHA-256 hex string |
| Alignment Padding | ~0.1% | 64-byte alignment |

Total overhead is typically <0.5% of the total package size.

**Figure Prompt:** Generate a pie chart showing the composition of a GLLM package: tensor data (99.2%), manifest (0.05%), headers (0.4%), alignment padding (0.3%), checksums (0.05%).

---

# Sequential Loading Benchmarks

## Prefetch Efficiency

Prefetch efficiency measures the overlap between layer loading and layer execution.

**Figure Prompt:** Generate a Gantt chart showing layer execution and layer loading timelines for 10 consecutive layers. Show ideal overlap (loading finishes before execution starts) and suboptimal overlap (execution stalls waiting for load).

## Working Set Fluctuation

Working set fluctuation measures the variance in resident memory during sequential execution.

**Figure Prompt:** Generate a line chart showing resident memory over time for 80 layers. Memory should spike at layer boundaries (two layers mapped simultaneously) and drop between layers. Show average working set as a horizontal line.

---

# Storage Comparison

## Format Size Comparison

Compare package sizes across formats for the same model.

**Figure Prompt:** Generate a grouped bar chart comparing package sizes for a 70B model: raw FP16 (140GB), Q4_K_M GGUF (39GB), Q4_K_M GLLM (39.1GB), Q4_K_M Safetensors (39GB). X-axis: format; Y-axis: size in GB.

---

# Benchmark Methodology

## Hardware Configuration

All benchmarks are run on the following standardized configuration:

| Component | Specification |
|-----------|---------------|
| CPU | AMD EPYC 9654 (96 cores) or Intel Xeon w9-3495X |
| GPU | NVIDIA RTX 4090 (24GB) or NVIDIA A100 (80GB) |
| RAM | 512GB DDR5-4800 |
| Storage | Samsung 990 Pro NVMe SSD (2TB) |
| OS | Ubuntu 22.04 LTS |

## Measurement Protocol

1. **Warm-up:** Run 100 tokens of generation before measurement to warm caches.
2. **Isolation:** Run benchmarks on a dedicated machine with no other load.
3. **Repetition:** Run each benchmark 10 times and report the median and standard deviation.
4. **Cold Start:** Reboot the machine between cold-start benchmarks to clear OS page cache.
5. **Instrumentation:** Use `cudaEvent` for GPU timing, `rdtsc` for CPU timing, and `/proc/meminfo` for memory measurement.

## Reporting Format

Benchmark results are reported in the following format:

```json
{
  "benchmark": "tokens_per_second",
  "model": "mensa-rotunda-70b-q4_k_m",
  "hardware": "rtx4090",
  "context_length": 4096,
  "median": 42.5,
  "stddev": 1.2,
  "unit": "tokens/sec",
  "notes": "Illustrative value. Actual measurement pending."
}
```

---

## Related Documents

| Document | Title | Description |
|----------|-------|-------------|
| ARTX1 | GLLM Overview | Entry-point overview of the GLLM format, vision, and ecosystem |
| ARTX2 | Package Specification | Physical and logical package structure, archive format, execution units |
| ARTX3 | Manifest Specification | gllm.json schema, metadata, versioning, and extension registry |
| ARTX4 | Layer Specification | Binary layer file format, tensor layouts, and extension layer types |
| ARTX5 | Runtime Architecture | Execution model, scheduler, CPU/GPU/hybrid runtime, and failure recovery |
| ARTX6 | Memory Model | Address space layout, memory lifecycle, prefetch strategy, and KV cache |
| ARTX7 | Converter Architecture | Conversion pipeline, parsers, validation, and error handling |
| ARTX8 | Extension System | Plugin architecture, MLA, Mamba, MoE, and custom layer types |
| ARTX9 | Compatibility & Versioning | Format versioning, source compatibility, and known limitations |
| ARTX10 | Distributed Runtime | Multi-node execution, pipeline/tensor parallelism, and recovery |
| **ARTX11** | **Benchmarks** | **This document** |

---

*This document is part of the GLLM Architecture Specification Series. The master document is ArchGLLMFormat.md.*
