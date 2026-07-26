# SUMMARY — Percival Audit Index

**Audited commit:** `555881ebc8b0fc0402b30e09258a32a7bfd13c52` (2026-07-24)

This file is a running one-line summary of every ARTX document. It is
updated after each document is completed.

## CPU

| ARTX | Title                         | Status     | Findings | Last Updated |
| ---- | ----------------------------- | ---------- | -------- | ------------ |
| 01   | Generic CPU core              | done       | F01–F12  | 2026-07-25   |
| 02   | IceLake / AVX-512             | done       | F01–F12  | 2026-07-25   |
| 03   | AMD Zen                       | done       | F01–F10  | 2026-07-25   |
| 04   | ARM NEON                      | done       | F01–F10  | 2026-07-25   |
| 05   | AArch64                       | done       | F01–F12  | 2026-07-25   |
| 06   | CPU Quantization              | done       | F01–F12  | 2026-07-25   |
| 07   | Threading                     | done       | F01–F12  | 2026-07-25   |

## CUDA

| ARTX | Title                         | Status     | Findings | Last Updated |
| ---- | ----------------------------- | ---------- | -------- | ------------ |
| 08   | CUDA core                     | done       | F01–F16  | 2026-07-25   |
| 09   | GEMV                          | done       | F01–F12  | 2026-07-25   |
| 10   | GEMM                          | done       | F01–F12  | 2026-07-25   |
| 11   | Attention                     | done       | F01–F15  | 2026-07-25   |
| 12   | MMQ                           | done       | F01–F12  | 2026-07-25   |
| 13   | VecDot                        | done       | F01–F10  | 2026-07-25   |
| 14   | Dequantization                | done       | F01–F10  | 2026-07-25   |

## Metal

| ARTX | Title                         | Status     | Findings | Last Updated |
| ---- | ----------------------------- | ---------- | -------- | ------------ |
| 15   | Metal core                    | done       | F01–F14  | 2026-07-25   |
| 16   | Threadgroup & simdgroup       | done       | F01–F14  | 2026-07-25   |
| 17   | Metal attention               | done       | F01–F12  | 2026-07-25   |

## Vulkan

| ARTX | Title                         | Status     | Findings | Last Updated |
| ---- | ----------------------------- | ---------- | -------- | ------------ |
| 18   | Vulkan core                   | done       | F01–F14  | 2026-07-25   |
| 19   | Shaders                       | done       | F01–F12  | 2026-07-25   |
| 20   | Subgroups                     | done       | F01–F12  | 2026-07-25   |

## Shared

| ARTX | Title                         | Status     | Findings | Last Updated |
| ---- | ----------------------------- | ---------- | -------- | ------------ |
| 21   | Memory & allocator            | done       | F01–F13  | 2026-07-25   |
| 22   | Execution graph & scheduler   | done       | F01–F16  | 2026-07-25   |
| 23   | Backend dispatch              | done       | F01–F13  | 2026-07-25   |
| 24   | KV cache & attention          | done       | F01–F12  | 2026-07-25   |

---

## Cross-cutting Notes

* **CPU layout drift.** The CPU backend's per-ISA files were reorganized
  into `arch/<isa>/` since the audit prompt was authored. Each ARTX
  document records the actual files at commit `555881e`. See `README.md`
  for the structural drift notice.
* **Metal multi-file refactor.** The Metal backend was split into
  `ggml-metal.{cpp,context.{h,m},device.{h,cpp,m},ops.{h,cpp},common.cpp}`
  + a single 11K-line `ggml-metal.metal`. ARTX15 covers the host side;
  ARTX16/17 cover the kernel side.
* **Vulkan SPIR-V patching.** The Vulkan backend patches SPIR-V at
  runtime to specialize per-device float controls and to strip decode
  vectors; see ARTX18-F12.
* **Cross-backend attention contract.** All four backends implement
  `GGML_OP_FLASH_ATTN_EXT` with a shared op_params layout (scale,
  max_bias, logit_softcap, prec); see ARTX24-F01.
* **Fusion unevenly distributed.** CPU has 1 pattern (ARTX01-F08);
  CUDA has ~12 (ARTX08-F13); Metal has a per-shape fusion engine
  (ARTX15-F14); Vulkan has ~15 patterns with anti-aliasing rollback
  (ARTX18-F09). Vulkan has the most mature fusion machinery.
* **Synchronous vs async backends.** CPU is fully synchronous
  (ARTX01-F01); CUDA/Metal/Vulkan all expose async + event APIs.
  Cross-backend `cpy_tensor_async` falls back to synchronous copy
  when either side lacks async (ARTX21 — REJECT this fallback).
* **Total findings produced: 247** across 24 ARTX documents.
