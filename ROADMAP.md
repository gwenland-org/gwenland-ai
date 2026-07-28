# GwenLand — Engine Roadmap

> **Philosophy: Inference First.** Correct inference on whatever hardware is
> present comes before speed, features, or ecosystem. Tagline: *"finding its
> limit, not the speed"* — every performance number is measured, every limit
> is documented, negative results are deliverables.
>
> The gl-stack **is** the engine (not an accelerator bolted onto something
> else): independent backends behind one `GlEngine` trait, a runtime that
> selects and routes but owns zero compute, and `glproc` (CPU) as the
> numerical ground truth every other engine is validated against.

Last updated: **2026-07-28**. Agents: read
[`gl-agent-skills/`](gl-agent-skills/README.md) before touching anything.

---

## Status snapshot

| Crate | Role | Status |
|-------|------|--------|
| `glcore` | parsers (GGUF/safetensors), BPE tokenizer (14 vocab families exact vs reference, see below), tensor types, `GlEngine` trait, runtime | ✅ shipped |
| `glproc` | CPU engine — SIMD (AVX2) + threading, Q8_0 hot path, Q4_K→Q8_0 repack | ✅ M1 + M1.5 done |
| `glcuda` | CUDA engine — driver FFI, hand-written PTX, arena VRAM, CUDA-graph decode, INT8-MMA prefill | ✅ **M2 passed** (T4-validated) |
| `glbench` | profiler/benchmark — telemetry roofline, behavior signals, A/B, archives | ✅ v2 "Mensura Veritatis" shipped |
| `glcli` | the `gwen` binary | ✅ (CPU engine; CUDA wiring pending) |
| `glvulkan` | cross-vendor GPU backend | ◻ stub (planned) |
| `glmetal` | Apple Silicon backend | ◻ stub (planned) |
| `glictus-caliburni` | GLLM shard format (MoE, lazy experts) | 🧪 experimental |
| `packages/` (gltui, mcp, core) | TUI, MCP server, legacy core | ✅ working, separate track |

**Model support today:** Llama & Qwen2/Qwen3 families (GQA, NeoX RoPE),
Q8_0 + Q4_K GGUF. Qwen3-MoE compute path verified; `_exps` tensor layout
still unverified (see Risks).

---

## ✅ Done (evidence in `architecture/`, `docs/`, `changelog/`)

### M1 — CPU baseline
End-to-end CPU inference: from-scratch GGUF/safetensors parsers, BPE
tokenizer, scalar→correct kernels, KV cache, sampler, runtime, CLI.
`gwen run model.gguf` generates coherent text.

### M1.5 — CPU performance bridge (spec: `architecture/ArchGLLM_X5.md`)
AVX2 SIMD + persistent thread pool + zero-alloc runner + cursor KV cache +
Q4_K→Q8_0 load-time repack. Decode sits at ~70–78 % of the machine's
*measured* (~29 GB/s) bandwidth ceiling — near the physical limit for
weight-streaming decode on the reference i3-1115G4.

### M2 — CUDA engine (ground truth: `architecture/ArchGLML_X2.md`)
From-scratch SIMT engine, no `nvcc`/cuBLAS, driver loaded at runtime.
Passed the full Definition of Done on a Tesla T4: 14/14 tensor parity vs
glproc, zero post-init allocations, no VRAM leaks; decode 29.2 tok/s (88 %
of card bandwidth), prefill 73 tok/s (batched GEMM + INT8 tensor cores).
Reproducible: `notebooks/glcuda_t4_validation.ipynb`.

### Tokenizer rewrite — `glcore::tokenizer`, 14 vocab families exact
The previous BPE tokenizer passed its own round-trip tests while scoring
65.2%–97.8% against llama.cpp's reference vectors (worst case: a third of
inputs wrong on `llama-bpe`) — round-trip tests hide compensating errors that
reference-*data* comparison does not. Rewritten from the algorithm's
definition (never from llama.cpp's code), then extracted from a standalone
crate into `glcore::tokenizer`. Now **14 vocabulary families exact**, enforced
by `glcore/tests/tokenizer_parity.rs` on every build; a family the crate
cannot express is refused at load rather than approximated. A pre-token cache
(1.9–3.6× on real text) and a `find_special` fix (71% → ~1% of a warm encode)
followed once precision was settled. Architecture, the traps found along the
way, and measured throughput:
[`glcore/src/tokenizer/README.md`](glcore/src/tokenizer/README.md).

### Precision vs llama.cpp — measured, at parity (native Q4_K)
A previously "confirmed real" ~46% perplexity gap on identical GGUF weights
turned out to be **~85% a scoring-protocol mismatch**: `llama-perplexity`
scores only the last half of each context window, and the comparison hadn't
matched that. Re-measured with the two tools reading the same token set,
same session: glproc native Q4_K **24.19** vs llama.cpp **24.78 ± 3.69** —
inside llama.cpp's own error bar. The production default (Q4_K→Q8_0 repack)
is ~7.5% behind, which is the documented repack trade. One genuine formula
difference was found and fixed along the way (RMSNorm's sum-of-squares
accumulator, f32 → f64 to match `ggml_float`); it moved nothing measurably.
Throughput (decode/prefill) is a separate, still-open gap — see
[`gl-agent-skills/cpu-skills/rejected-optimizations.md`](gl-agent-skills/cpu-skills/rejected-optimizations.md)
for what has already been tried and rejected on this hardware tier.

### Observability — glbench v2 "Mensura Veritatis"
Pull-based engine telemetry → bucket roofline vs measured ceiling, cold/warm
separation, behavioral signals from raw logits (CoT-aware), cross-signal
hypotheses, session archives + `ab`/`compare`. glbench observes; it never
optimizes.

### Closed dead ends (measured, documented, not to be revisited)
Native Q4_K CPU compute (−33 %), L2 decode tiling, interleaved rows (−35 %),
AVX-512F on this tier, software prefetch, lazy mmap layer paging, topology
threading (−23 %). Full list with reasoning:
[`gl-agent-skills/cpu-skills/rejected-optimizations.md`](gl-agent-skills/cpu-skills/rejected-optimizations.md).

---

## ▶ Now — M2.5: make the CUDA engine a first-class citizen

The engine is validated standalone; the product must route to it.

- [ ] **Wire glcuda into the runtime fallback chain** (glcuda → glvulkan →
      glmetal → glproc) so `gwen run` uses it automatically; explicit
      `--engine glcuda` fails loudly instead of silently falling back.
- [ ] **Fallback-decision logging + session recording** — every selection
      reason visible to users and to glbench sessions.
- [ ] **MoE `_exps` layout verification** — inspect real Qwen3-MoE GGUF
      bytes vs llama.cpp dequant; close the `_EXPS_LAYOUT_ASSUMPTION`
      marker (currently the biggest silent-corruption risk).
- [ ] **CUDA-graph decode: fusion pass** — attack inter-kernel dependency
      edges (the measured residual), not launch overhead (already spent).

**Exit criteria:** on a CUDA machine, `gwen run model.gguf` decodes on
glcuda with parity output and the selection is visible in logs + glbench.

## ▶ Next — M3: parity & stability across engines

- [ ] Cross-engine parity suite in CI shape: every engine vs glproc within
      the documented per-op tolerances (GPU legs run on GPU runners /
      notebook validation until hosted GPU CI exists).
- [ ] Crash isolation: an engine panic/init failure never kills the
      runtime — degrade down the chain with the reason recorded.
- [ ] Stress: long sessions + engine switching, zero panics, stable RSS/VRAM
      (leak checks stay green).
- [ ] Benchmark baselines archived per release (`glbench compare` as the
      regression gate at a 5 % threshold).

**Exit criteria:** same model, same seed → parity-consistent output on every
available engine; a broken GPU setup degrades to CPU with a logged reason,
never a crash.

## ▶ Later — M4: more hardware, then ecosystem

Ordered by Inference-First priority (reach more hardware correctly before
platform features):

- [ ] **glvulkan bring-up** — cross-vendor (AMD/Intel/NVIDIA/Mali), copying
      glcuda's *pattern*: runtime loader, committed SPIR-V, arena memory,
      parity ladder. Rules already written:
      [`gl-agent-skills/vulkan-skills/`](gl-agent-skills/vulkan-skills/).
- [ ] **glmetal bring-up** — Apple Silicon, same pattern.
- [ ] **glictus-caliburni graduation or burial** — the GLLM shard format
      (lazy expert loading for MoE on 8 GB) is validated with measurements
      or explicitly closed like every other experiment.
- [ ] **Engine trait as stable API** — semver the `GlEngine` contract so an
      external crate can implement an engine.
- [ ] **Packaging** — release binaries (Win/macOS/Linux) via CI.
- [ ] **Docs freeze** — architecture guide + API docs published.

---

## Standing invariants (all milestones, non-negotiable)

1. **Zero external ML dependencies; no CMake/C bindings;** builds with plain
   `cargo build` on a machine with no GPU vendor anything. GPU drivers are
   loaded at runtime, kernels ship as PTX/SPIR-V text/blobs.
2. **The runtime owns no compute; engines never import each other.**
3. **glproc is the floor and the oracle** — always available, scalar
   fallback included.
4. **The 8 GB reference machine stays first-class.** mmap streaming, no
   duplicate weight copies.
5. **Every performance claim is a measured production number** with an
   archived glbench session behind it.
6. **Dependency additions follow the policy in
   [`CONTRIBUTING.md`](CONTRIBUTING.md)** — reason, impact, use cases, or no.

## Research track: gljax (separate from the M1–M4 line above)

A from-scratch StableHLO/PJRT engine, explored as a design track rather than
folded into the gl-stack milestones — it targets a different execution model
(compiled IR through XLA/PJRT vs. the gl-stack's hand-written kernels) and
isn't on the gl-stack's critical path. **Status: 17 architecture documents,
zero code, not a workspace member.** Start at
[`gljax/architecture/Overall-Architecture.md`](gljax/architecture/Overall-Architecture.md).

The one exception is [`gljax/probes/`](gljax/probes/) — three small,
independently reproducible Python scripts that settled real open questions
(what a PJRT CPU plugin does with quantized weights; where tile-streamed
dequantisation stops winning over whole-weight dequant, measured at the
M≈64–256 crossover). Those are measurements, not gljax code, and they don't
require gljax to exist to run.

**Next step, when picked up:** implementation starts at ARTX01 (PJRT FFI
bring-up on the CPU plugin) per the build order in `Overall-Architecture.md`
§3.1 — not at ARTX10, whose design is complete but frozen until ARTX01–05
produce a real generated token.

## Non-goals

- **Cloud orchestration / serving APIs** — a separate layer someday, never
  coupled into the engine (contradicts Inference First).
- **Training at scale** — CPU-only training experiments live in
  [`Experimental/`](Experimental/README.md) until they graduate.
- **Chasing llama.cpp feature-for-feature** — GwenLand's product is a fully
  understood engine, not a feature matrix.

## Risks (known, tracked)

| Risk | Where tracked |
|------|---------------|
| `_exps` MoE layout unverified → silent fluent-garbage corruption | `_EXPS_LAYOUT_ASSUMPTION` markers; [`gl-agent-skills/gguf-skills/moe-loading.md`](gl-agent-skills/gguf-skills/moe-loading.md) |
| GGUF spec drift (ggml upstream owns the format) | warning banners in [`gl-agent-skills/gguf-skills/`](gl-agent-skills/gguf-skills/) |
| No hosted GPU CI → GPU regressions only caught on manual/notebook runs | M3 parity work; T4 notebooks in [`notebooks/`](notebooks/) |
| Broken GGUF exports in the wild (norm-corruption case) | parser cross-checks; [`gl-agent-skills/gguf-skills/format-parsing.md`](gl-agent-skills/gguf-skills/format-parsing.md) |
