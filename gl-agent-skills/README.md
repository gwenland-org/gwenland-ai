# gl-agent-skills

> **Audience: AI agents** (Claude Code, Codex, or any other coding agent)
> maintaining the GwenLand AI repository. This is not human documentation —
> it is a fragmented, per-domain knowledge base of **explicit rules**.
> These are not guidelines. They are RULES. No compromise.

## What this is

Every file here covers exactly one topic, standalone, with:

- a **BEFORE YOU START** checklist,
- explicit numbered **Rules** (no ambiguity),
- **✅ correct** and **❌ anti-pattern** examples,
- GwenLand-specific notes where reality differs from standard practice.

**Philosophy:** one skill = one responsibility; no overlap between skills;
an agent MUST read the relevant skill(s) before touching that domain.

**Project identity:** GwenLand — Inference First LLM inference in pure Rust.
Zero external ML dependencies, no CMake, no C bindings, no vendor SDKs at
build time. *Inference First*: correct inference anywhere (CPU today, CUDA
validated, Vulkan/Metal planned). Tagline: **"Finding its limit, not the
speed."**

## How to use (agents, read this)

1. Identify which domain(s) your task touches.
2. Read the matching skill files in the table below **before writing any code**.
3. `before-coding/read-architecture-first.md` applies to **every** task.
4. If two sources conflict, precedence is: measured production numbers →
   `architecture/` specs → these skills → anything else. If a skill is stale,
   fix the skill in the same PR — never silently ignore it.
5. Never revisit anything on a rejected-optimizations list without explicit
   permission from JinXSuper.

## Quick reference: task → required reading

| Task | Read these skills first |
|------|------------------------|
| **Anything at all** | `before-coding/read-architecture-first.md` |
| Naming/renaming any type | `gwenland-naming-convention/SKILL.md` |
| Any work inside `stumman/` | `stumman-naming/SKILL.md` |
| Multi-step / multi-wave work | `before-coding/wave-confirmation-gates.md` |
| Any code change | `before-coding/check-existing-tests.md`, `rust-skills/error-handling.md` |
| Creating a branch / PR | `before-coding/branch-strategy.md` |
| Edit glcuda kernels | `cuda-skills/ptx-writing.md`, `cuda-skills/kernel-design.md`, `cuda-skills/memory-management.md` |
| CUDA graphs / launch path | `cuda-skills/cuda-graphs.md` |
| CUDA driver loading / FFI | `cuda-skills/dynamic-loading.md`, `rust-skills/unsafe-rules.md` |
| CPU optimization | `cpu-skills/avx2-simd.md`, `cpu-skills/rejected-optimizations.md`, `cpu-skills/memory-bandwidth.md` |
| Threading changes | `cpu-skills/threading-model.md` |
| New/changed quantization | `cpu-skills/quantization.md`, `gguf-skills/quantization-types.md`, `gguf-skills/dequant-path.md` |
| Edit glcore | `architecture-skills/glcore-rules.md`, `architecture-skills/backend-independence.md` |
| Runtime / engine selection | `architecture-skills/fallback-chain.md` |
| GGUF parser work | `gguf-skills/format-parsing.md` |
| MoE / `_exps` tensors | `gguf-skills/moe-loading.md` |
| Vulkan backend work | `vulkan-skills/spirv-writing.md`, `vulkan-skills/descriptor-sets.md`, `vulkan-skills/pipeline-barriers.md` |
| Run / interpret benchmarks | `bench-skills/glbench-usage.md`, `bench-skills/measurement-discipline.md`, `bench-skills/windows-defender-gotcha.md` |

## Index

### Naming (directory-per-skill, `SKILL.md` format)

These two use the Agent Skills layout (`<name>/SKILL.md` with YAML frontmatter)
rather than the flat per-domain files below.

- [gwenland-naming-convention/SKILL.md](gwenland-naming-convention/SKILL.md) — repo-wide two-character type prefixes; **target state, 0/224 types adopted so far**
- [stumman-naming/SKILL.md](stumman-naming/SKILL.md) — stumman's rename target + the live Breton module codenames

### before-coding/
- [read-architecture-first.md](before-coding/read-architecture-first.md) — which docs are ground truth and must be read before anything
- [check-existing-tests.md](before-coding/check-existing-tests.md) — verify the test suite before AND after your change
- [wave-confirmation-gates.md](before-coding/wave-confirmation-gates.md) — wave-by-wave execution, STOP at every gate
- [branch-strategy.md](before-coding/branch-strategy.md) — branch naming, commits, changelog notes, fork model

### rust-skills/
- [error-handling.md](rust-skills/error-handling.md) — `Result<T, E>` everywhere, no `unwrap()` on production paths
- [trait-design.md](rust-skills/trait-design.md) — the engine trait, trait objects, zero-cost abstractions
- [memory-safety.md](rust-skills/memory-safety.md) — ownership/borrowing in hot paths, no extra weight copies
- [unsafe-rules.md](rust-skills/unsafe-rules.md) — when `unsafe` is allowed, invariant comments, safe wrappers
- [testing-standards.md](rust-skills/testing-standards.md) — unit/integration test structure and naming

### cuda-skills/
- [ptx-writing.md](cuda-skills/ptx-writing.md) — hand-authored PTX rules (ASCII, LF, unique names)
- [memory-management.md](cuda-skills/memory-management.md) — VRAM bump allocator, zero post-init mallocs
- [kernel-design.md](cuda-skills/kernel-design.md) — warp/threadblock sizing, coalescing, occupancy
- [dynamic-loading.md](cuda-skills/dynamic-loading.md) — driver `dlopen`/`LoadLibrary`, no build-time NVIDIA dep
- [cuda-graphs.md](cuda-skills/cuda-graphs.md) — decode graphs; the real bottleneck is dependency edges
- [tensor-cores.md](cuda-skills/tensor-cores.md) — WMMA / FP16 / packed-int paths

### vulkan-skills/
- [spirv-writing.md](vulkan-skills/spirv-writing.md) — SPIR-V compute shader authoring
- [descriptor-sets.md](vulkan-skills/descriptor-sets.md) — buffer binding patterns
- [pipeline-barriers.md](vulkan-skills/pipeline-barriers.md) — ordering and synchronization
- [portability.md](vulkan-skills/portability.md) — NVIDIA / AMD / Intel / ARM Mali targets

### cpu-skills/
- [avx2-simd.md](cpu-skills/avx2-simd.md) — AVX2 intrinsics, V-accumulation, reduction patterns
- [threading-model.md](cpu-skills/threading-model.md) — thread knee = 3, SMT behavior, pool reentrancy
- [memory-bandwidth.md](cpu-skills/memory-bandwidth.md) — the DDR4 ceiling and what bandwidth-bound means
- [quantization.md](cpu-skills/quantization.md) — Q8_0 / Q4_K reality on this hardware tier
- [rejected-optimizations.md](cpu-skills/rejected-optimizations.md) — ⛔ DO-NOT-REVISIT list with reasoning

### architecture-skills/
- [glcore-rules.md](architecture-skills/glcore-rules.md) — glcore = zero compute, orchestration only
- [backend-independence.md](architecture-skills/backend-independence.md) — engines never import each other
- [fallback-chain.md](architecture-skills/fallback-chain.md) — glcuda → glvulkan → glmetal → glproc
- [inference-first.md](architecture-skills/inference-first.md) — the philosophy and what it forbids
- [gate-integration.md](architecture-skills/gate-integration.md) — execution-policy gating in GwenLand

### gguf-skills/
- [format-parsing.md](gguf-skills/format-parsing.md) — GGUF parser rules, zero external deps
- [quantization-types.md](gguf-skills/quantization-types.md) — Q4_K / Q8_0 / Q8_K / F32-BF16 layouts
- [moe-loading.md](gguf-skills/moe-loading.md) — `_exps` loading, the unverified-layout hazard
- [dequant-path.md](gguf-skills/dequant-path.md) — the dequant chains and which one is shipped

### bench-skills/
- [glbench-usage.md](bench-skills/glbench-usage.md) — how to run a benchmark that means something
- [rca-interpretation.md](bench-skills/rca-interpretation.md) — reading the anomaly/root-cause output
- [windows-defender-gotcha.md](bench-skills/windows-defender-gotcha.md) — Defender rescans pollute results 2–4×
- [measurement-discipline.md](bench-skills/measurement-discipline.md) — warmup, P50/P90/P99, baseline first

## Scope

These skills are for the **GwenLand AI repository only** — not for
gwenland-ide, gwenland-agent, or any other project. The contents of
[`../Experimental/`](../Experimental/README.md) are research, never a spec
to implement.
