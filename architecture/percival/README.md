# Percival — llama.cpp Architectural Audit

**Auditor:** Percival (GwenLand architectural audit agent)
**Audited repo:** `ggerganov/llama.cpp`
**Audited commit:** `555881ebc8b0fc0402b30e09258a32a7bfd13c52`
**Commit date:** 2026-07-24
**Method:** Static source-code analysis only. No execution, no profiling, no benchmarking.

---

## Purpose

This directory is a permanent architectural record of llama.cpp as understood
by the GwenLand project. It exists so that a GwenLand engineer can later
implement `glproc`, `glcuda`, `glmetal`, `glvulkan`, and `GATE` without ever
reopening the upstream source tree.

Every claim in every ARTX document is backed by:

* a source file path,
* a function name,
* a line range,
* and architectural reasoning.

Where static analysis cannot reach a conclusion, an **Unknowns** section says
so explicitly. No guessing.

---

## Scope

| Backend       | ARTX range  | Status     |
| ------------- | ----------- | ---------- |
| CPU           | ARTX01–07   | in progress |
| CUDA          | ARTX08–14   | pending    |
| Metal         | ARTX15–17   | pending    |
| Vulkan        | ARTX18–20   | pending    |
| Shared / Core | ARTX21–24   | pending    |

---

## Directory Layout

```
architecture/percival/
├── README.md                  ← this file
├── SUMMARY.md                 ← running index of every ARTX
├── GAP-MAP.md                 ← architectural gaps surfaced during the audit
├── IMPLEMENTATION-PLAN.md     ← GwenLand work items, evidence-backed only
├── CPU/      ARTX01..ARTX07
├── CUDA/     ARTX08..ARTX14
├── Metal/    ARTX15..ARTX17
├── Vulkan/   ARTX18..ARTX20
└── Shared/   ARTX21..ARTX24
```

---

## Structural Drift Notice

The audit prompt referenced a file layout (`ggml-cpu-icelake.cpp`,
`ggml-cpu-amd.cpp`, `ggml-cpu-arm.cpp`, `ggml-cpu-aarch64.cpp`,
`ggml-cpu-quants.c`) that **no longer exists** at the audited commit.
The CPU backend has been re-organized into:

```
ggml/src/ggml-cpu/
├── ggml-cpu.c            ← C entry
├── ggml-cpu.cpp          ← C++ dispatch & build graph
├── ggml-cpu-impl.h
├── ops.{cpp,h}           ← scalar + generic SIMD ops
├── quants.{c,h}          ← generic quant dequant/vecdot
├── vec.{cpp,h}           ← SIMD vector primitives
├── simd-gemm.h           ← SIMD GEMM building blocks
├── arch/                 ← per-ISA code (x86, arm, loongarch, powerpc, riscv, s390, wasm)
│   ├── x86/{cpu-feats.cpp, quants.c, repack.cpp}
│   ├── arm/{cpu-feats.cpp, quants.c, repack.cpp}
│   └── ...
├── amx/                  ← AMX matrix-multiply backend
├── llamafile/sgemm.cpp   ← embedded llamafile SGEMM
├── kleidiai/             ← KleidiAI integration (ARM)
└── spacemit/             ← SpacemiT extension
```

Each ARTX document maps the audit's logical targets (IceLake, AMD Zen,
ARM, AArch64) onto the *actual* files at this commit, and records the
mapping explicitly. This is itself a finding: llama.cpp's CPU backend
has factored per-ISA code behind a `arch/<isa>/` interface, which
GwenLand's `glproc` should consider adopting.

---

## Document Template

Every ARTX document follows the structure mandated by the Percival
brief: Executive Summary → Purpose → Source Files → Architecture
Overview → Execution Flow → Data Layout → Memory Layout →
Parallelism Strategy → SIMD/GPU Strategy → Quantization Strategy →
Correctness Analysis → Optimization Analysis → Architectural
Strengths → Architectural Weaknesses → GwenLand Mapping →
Recommendations → Findings → Unknowns → References.

Every finding uses the Finding Template (Finding ID, Category, Engine,
Component, Source File, Function, Lines, Summary, Observation,
Evidence, Architectural Impact, Correctness Impact, Optimization Type,
GwenLand Target, Recommendation, Priority, Difficulty, Dependencies,
Confidence).

---

## Reading Order for a GwenLand Engineer

1. `SUMMARY.md` — one-line state of every ARTX.
2. `GAP-MAP.md` — known architectural gaps, sorted by priority.
3. `IMPLEMENTATION-PLAN.md` — only the items with sufficient evidence.
4. The ARTX document for the component you are about to build.

---

## Conventions

* All line numbers refer to the audited commit `555881e`.
* File paths are relative to the repository root.
* `gl*` names refer to GwenLand modules:
  * `glproc`   — CPU backend
  * `glcuda`   — CUDA backend
  * `glmetal`  — Metal backend
  * `glvulkan` — Vulkan backend
  * `GATE`     — execution graph / scheduler
* Recommendation verbs: `ADOPT` / `ADAPT` / `REJECT` / `MONITOR` / `DEFER`.
