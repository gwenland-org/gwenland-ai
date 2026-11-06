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

Last updated: **2026-11-05** (after a ~5-week pause; last engineering commit
2026-09-27). Agents: read
[`gl-agent-skills/`](gl-agent-skills/README.md) before touching anything.

---

## Status snapshot

| Crate | Role | Status |
|-------|------|--------|
| `glcore` | parsers (GGUF/safetensors), BPE tokenizer (14 vocab families exact vs reference, see below), tensor types, `GlEngine` trait, runtime | ✅ shipped |
| `glproc` | CPU engine — SIMD (AVX2) + threading, Q8_0 hot path, Q4_K→Q8_0 repack | ✅ M1 + M1.5 done |
| `glcuda` | CUDA engine — driver FFI, hand-written PTX, arena VRAM, CUDA-graph decode, INT8-MMA prefill | ✅ **M2 passed**; ceiling sprint at Wave 136 (prefill ~15.8k tok/s on T4, see below) |
| `glbench` | profiler/benchmark — telemetry roofline, behavior signals, A/B, archives, training observation, CUDA launch telemetry | ✅ v3 shipped → ▶ **v4 "No Celah" is the current focus** |
| `glcli` | the `gwen` binary | ✅ (CPU engine; CUDA wiring pending) |
| `glvulkan` | cross-vendor GPU backend | ◻ stub (planned) |
| `glmetal` | Apple Silicon backend | ◻ stub (planned) |
| `glictus-caliburni` | GLLM shard format (MoE, lazy experts) | 🧪 experimental |
| `gljax` + `glserve` | StableHLO/PJRT client + OpenAI-compatible HTTP server (research track, below) | 🧪 runs on CI; generates coherent text |
| `packages/mcp` | MCP server | ✅ separate track |
| `gltrain` (Stummañ) | from-scratch training framework, own workspace | ✅ M1 + M2 + M2.5 |

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

### glcuda ceiling sprint (Waves 1–136, 2026-08 → 2026-09-27)
Throughput work on the T4 with Qwen2.5-0.5B-Instruct **Q8_0** and a fixed
244-token prompt. Every candidate had to pass oracle-token parity before it
was timed, and was kept only if it won a counterbalanced production A/B. Most
waves were *rejected*; each one has a write-up in
[`architecture/glcuda-research/`](architecture/glcuda-research/).

| | At M2 | Now (production `glbench`, P50) |
|---|---:|---:|
| prefill | 73 tok/s (7B) | **~15,800 tok/s** (0.5B, Wave 131; 15,000 target met) |
| decode | 29.2 tok/s (7B) | **~224 tok/s** (0.5B, Wave 136; 44 % of the measured 267 GB/s ceiling) |

Caveats that stay attached to these numbers: they are short runs on a cool
card, and the T4 is power-capped, so sustained prefill-only load drops the SM
clock to ~1,100 MHz and measures ~11,500 tok/s; the newest paths are opt-in
`GLCUDA_*` flags, not the default; the last llama.cpp head-to-head (Wave 116:
12,258 vs 11,307 tok/s prefill) predates the latest +29 %; Q4 is unmeasured on
this path. Waves 131–136 live on `engine/glcuda-ceiling-wave132`, not yet
merged; Wave 137 (fused attention-prepare) is coded but not run.

### Stummañ (`gltrain/`) — M1, M2, M2.5
From-scratch autograd on `glcore`/`glproc` (M1, 2026-08-17), LoRA training
with AdamW/Adafactor/Lion (M2, 2026-08-19), and an observer hook for glbench
(M2.5, 2026-08-20). The candle-based crate that used to live at that path was
deleted. Own workspace, own CI (`.github/workflows/gltrain.yml`).

### Observability — glbench v3 + CUDA launch telemetry
v3 (2026-08-20, 5 waves): schema-v2 archives with content digest and null
semantics, `glbench join`, GLBitProf bit profiles, training observation
(`glbench train` / `unified`). Then (2026-09-24): native glcuda launch
telemetry — per-kernel P50/P90/P99, grid/block, registers, shared memory,
projected occupancy — aligned to the measured window, with every session
labelled `production` or `instrumented`. Design:
[`architecture/glbench-v3/DESIGN.md`](architecture/glbench-v3/DESIGN.md).

### Hygiene pass (2026-08-15)
Every `unsafe` site in glproc documented (a real OOB read in `run_matvec` was
found and fixed while doing it), first tests for glcli and the MCP server, and
`glserve` binds loopback by default.

### Closed dead ends (measured, documented, not to be revisited)
Native Q4_K CPU compute (−33 %), L2 decode tiling, interleaved rows (−35 %),
AVX-512F on this tier, software prefetch, lazy mmap layer paging, topology
threading (−23 %). Full list with reasoning:
[`gl-agent-skills/cpu-skills/rejected-optimizations.md`](gl-agent-skills/cpu-skills/rejected-optimizations.md).

---

## ▶ Now — glbench v4 "No Celah": an observer with no blind spots

glbench already reports a lot, and it is honest about what it cannot see. v4
closes the gaps those admissions describe. Every item below is a limit that
glbench or a wave write-up has *already documented*; none is a guess about
what might be missing. The boundary stays the same: glbench observes, engines
optimize, and no crates are added.

**Blind spots in what it sees**
- [ ] **CUDA Graph decode is opaque.** Graph replay shows up as one
      `graph_replay` entry. Un-graphing it to profile (Wave 133) made decode
      2.05× slower, so the instrumented view is not the production path.
      Goal: per-stage attribution of the *production* graph, with the
      perturbation measured and reported.
- [x] **GPU ceilings came from a spec sheet.** Closed 2026-11-06: a read kernel
      measures the card before the engine loads (T4: 267 GB/s of a published
      320), and decode is divided by the bytes the engine reports streaming.
- [x] **GPU state was invisible.** Closed 2026-11-06: NVML, loaded at runtime,
      reports clocks, temperature, power, energy, VRAM and clock-limit reasons.
      The sampler's measured cost is -0.45% prefill and 0.0% decode, inside the
      noise.
- [x] **Host probes were dead on current Windows.** Closed 2026-11-06: `wmic` is
      replaced by direct Win32 calls.
- [x] **No TTFT, no model hash in the archive.** Closed 2026-11-06.
- [ ] **Bit profiles cannot see a permutation.** Measured in v3: a correct
      decode and a nibble-swapped one score exactly zero divergence. That is
      the Q6_K bug class that once corrupted every layer. Goal: an
      order-sensitive check that catches it, with the existing blind-spot test
      flipped into a detection test.
- [ ] **No per-token decode timeline.** Needs an engine hook that streams
      per-token timestamps (a coordinated engine-side change).

**Blind spots in how it compares**
- [x] **Comparisons across setups read as regressions.** Closed 2026-11-06:
      `compare`, `ab` and `join` name an engine, hardware, quantization or
      workload comparison and give no regression verdict.
- [x] **`glbench ab` was sequential.** Closed 2026-11-06: it runs A-B-B-A and
      says DISAGREE when reversing the order flips a difference's sign.
- [ ] **llama.cpp decode rows are not the same work.** `llama-bench` does not
      sample. Goal: the head-to-head flags a work mismatch instead of printing
      the two numbers side by side.

**Coverage**
- [ ] **GLLM engine:** no tokenizer (synthetic token ids) and F32 only, so its
      behavior signals do not describe a real prompt.
- [ ] **Reports:** HTML report and roofline chart (zero-dependency, inlined),
      still open in [`glbench/ROADMAP.md`](glbench/ROADMAP.md).

**Accepted limits (not targets):** cache counters, PCIe, and Windows/macOS
energy need kernel drivers or external deps. Toxicity scoring stays
deliberately unimplemented.

**Exit criteria:** each item is either closed with a test that proves it, or
moved to "accepted limits" with the measurement that shows why.

## ▶ Then — make the CUDA engine a first-class citizen

The engine is validated and fast on its own, but `gwen run` still only uses
the CPU.

- [ ] **Merge the ceiling-sprint branch** and decide which `GLCUDA_*` opt-in
      paths become defaults.
- [ ] **Wire glcuda into the runtime fallback chain** (glcuda → glvulkan →
      glmetal → glproc) so `gwen run` uses it automatically; explicit
      `--engine glcuda` fails loudly instead of silently falling back.
- [ ] **Fallback-decision logging + session recording** — every selection
      reason visible to users and to glbench sessions.
- [ ] **Re-run the llama.cpp head-to-head** on the current build; measure Q4.
- [ ] **MoE `_exps` layout verification** — inspect real Qwen3-MoE GGUF
      bytes vs llama.cpp dequant; close the `_EXPS_LAYOUT_ASSUMPTION`
      marker (still open in `glproc/src/loader.rs`, and still the biggest
      silent-corruption risk).

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

A from-scratch StableHLO/PJRT client, explored as a design track rather than
folded into the gl-stack milestones. It targets a different execution model
(compiled IR through XLA/PJRT instead of hand-written kernels) and isn't on
the gl-stack's critical path.

**Status (2026-07-30):** real code, 284 tests. On Linux CI it loads a pinned
PJRT CPU plugin and Qwen2-0.5B and generates coherent English (Gate A5); the
KV cache matches the recompute oracle token-for-token and is 4×–80× faster.
`glserve` exposes it as an OpenAI-compatible HTTP server. It cannot run on the
Windows dev machine (no PJRT plugin for Windows). Start at
[`gljax/architecture/Overall-Architecture.md`](gljax/architecture/Overall-Architecture.md);
the measurements in [`gljax/probes/`](gljax/probes/) settled the quantized
runtime design (ARTX10). No active work planned.

## Non-goals

- **Cloud orchestration / serving APIs** — a separate layer someday, never
  coupled into the engine (contradicts Inference First).
- **Training at scale** — Stummañ (`gltrain/`) is a from-scratch, CPU-first
  framework for small fine-tunes (LoRA), not a distributed trainer.
- **Chasing llama.cpp feature-for-feature** — GwenLand's product is a fully
  understood engine, not a feature matrix.

## Risks (known, tracked)

| Risk | Where tracked |
|------|---------------|
| `_exps` MoE layout unverified → silent fluent-garbage corruption | `_EXPS_LAYOUT_ASSUMPTION` markers; [`gl-agent-skills/gguf-skills/moe-loading.md`](gl-agent-skills/gguf-skills/moe-loading.md) |
| GGUF spec drift (ggml upstream owns the format) | warning banners in [`gl-agent-skills/gguf-skills/`](gl-agent-skills/gguf-skills/) |
| No hosted GPU CI → GPU regressions only caught on manual/notebook runs | M3 parity work; T4 notebooks in [`notebooks/`](notebooks/) |
| Broken GGUF exports in the wild (norm-corruption case) | parser cross-checks; [`gl-agent-skills/gguf-skills/format-parsing.md`](gl-agent-skills/gguf-skills/format-parsing.md) |
