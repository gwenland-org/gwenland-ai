# Throughput Investigation

## Purpose

Understand why some optimizations produce large isolated kernel speedups but
little or no production throughput improvement.

The goal is not to optimize immediately.
The goal is to discover reusable engineering knowledge.

## Scope

Hardware: Intel i3-1115G4 (2 physical / 4 logical cores), DDR4-2667 dual-channel, 8 GB RAM, Windows 11.
Model: Qwen2.5-0.5B-Instruct Q4_K_M (0.46 GiB GGUF, loaded as Q8_0 by GATE selection).
Inference engine: glproc (CPU backend).
Measurement tool: glbench 0.1.163.
Reference: llama.cpp b10107 (ggml-cpu-icelake.dll backend, llama-bench r=5).

## Investigations Covered

| Investigation | Result | Source |
|---|---|---|
| VNNI-512 qdot | +20–26% GMAC/s isolated, neutral production | `benchmarks/vnni512-ab-*.json` |
| Row Tile qdot | +2× GMAC/s isolated, neutral production (ffn_down +8.9%, lm_head −9.4%) | `benchmarks/row-tile-ab-*.json` |
| RoPE table cache | +27.7→24.8% decode gap narrowed (real production gain) | `benchmarks/rope-ab-*.json` |
| Bias/Residual SIMD | 2.81× isolated speedup, 0.085% of wall-clock (noise floor) | `notes/issues/glproc-throughput-gap-vs-llamacpp.md` |
| Roofline analysis | decode 57% of bandwidth ceiling, classified compute-bound | `benchmarks/full-bottleneck-e2e.json` |
| glbench measurements | stage-level breakdown: ffn_gate_up 35.5%, lm_head 22.4%, ffn_down 20.0% | `benchmarks/full-bottleneck-e2e.json` |

## Documents

| File | Contents |
|---|---|
| `observations.md` | Measured facts only |
| `patterns.md` | Engineering patterns extracted from evidence |
| `anti_patterns.md` | Optimization traps with examples |
| `optimization_taxonomy.md` | Category × expected impact table |
| `bottleneck_model.md` | glproc pipeline stages and bottleneck classification |
| `decision_tree.md` | Investigation workflow |
| `optimization_playbook.md` | Reusable rules |
| `future_candidates.md` | Ranked future optimization ideas |

## Evidence Policy

- Measured: numbers came from a glbench or llama-bench run.
- Observed: a pattern inferred directly from measurements without independent confirmation.
- Hypothesis: a claim not yet tested.
- Future work: an idea not yet attempted.

No measurement is invented. No causality is claimed without evidence.
