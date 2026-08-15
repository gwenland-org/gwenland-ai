# ARTX01 Reality — Verification Report
## Date: 2026-07-25
## Verified by: Claude Code audit of glproc/ + glcuda/ + glcore/gate/ source

---

## Verdict per Requirement

### R1 — glproc pipeline: PARTIAL

**Claim:** "model weights loaded once via memory-mapped file access and
repacked in parallel at load time; per-token decode executed by a static,
fixed-size thread pool that streams each live weight byte through
vector-unit kernels exactly once per token, with no reuse across tokens"

**Evidence:**
- `glproc/src/engine.rs:343` — `GgufFile::open(path)` opens the GGUF via
  glcore, which does use mmap (see glcore note below).
- `glproc/src/loader.rs:51-53` (doc comment on `warm_and_lock_model`) —
  **explicit deviation, in the code's own words**: "glproc copies tensors
  out of the GGUF mmap into owned heap buffers at load, so the decode
  working set is those buffers, not the mmap."
- `glproc/src/loader.rs:709-734` (`load_gguf`) — layers ARE built in
  parallel: `std::thread::scope` spawns `n_workers` (logical core count,
  clamped 1-8) that pull layer indices from a shared atomic counter and
  write into per-index `Mutex` slots. Confirms "repacked in parallel at
  load time."
- `glproc/src/threading.rs:76-119` (`ThreadPool::new`) — the pool is
  fixed-size: spawned once with `n_threads` workers, parked on a condvar,
  reused across all decode calls. Confirms "static, fixed-size thread
  pool."
- `glproc/src/runner.rs:45,48-56` — thread count defaults to `N_THREADS =
  4`, overridable via `GLPROC_THREADS`, capped by `num_cpus::get()`.
- `glproc/src/runner.rs:859-1099` (`step`, the per-token forward pass) —
  every layer's weights are read exactly once per token via the row-chunked
  `par_matvec_*` family (`threading.rs`); there is no weight cache or
  memoization structure anywhere in `Runner` — each call re-streams the
  weight bytes from the owned buffers. Confirms "streams each live weight
  byte... exactly once per token, with no reuse across tokens."

**Delta from ARTX01:**
The requirement's mmap claim is **not literally true of the decode path**.
`glcore::format::gguf::GgufFile` does mmap the file (confirmed: matches
found in `glcore/src/format/gguf.rs`), and that mmap is what the *loader*
reads from — but the decode loop never touches the mmap. Every weight
byte a token consumes at decode time comes from an owned `Vec<u8>`/`Vec<f32>`
heap buffer that `loader.rs` copied out of the mmap once at load. The
"memory-mapped file access" clause is true only for the *load* phase, not
for "per-token decode" as the single sentence in R1 implies by joining
the two clauses. A precise restatement: model weights are read once from
a memory-mapped GGUF file, copied and repacked in parallel into owned
heap buffers at load time; per-token decode then streams those owned
buffers (not the mmap) through the thread pool. Everything else in R1 —
static fixed-size pool, one pass per weight byte per token, no cross-token
reuse — is confirmed as stated.

---

### R2 — glcuda pipeline: CONFIRMED

**Claim:** "model weights uploaded once to a single VRAM allocation at load
time; per-token decode executed on one command stream with one
host-device synchronization point per token; prefill executed as batched
matrix operations with data reuse that decode does not have."

**Evidence:**
- `glcuda/src/buffer.rs:85-99` (`BackendBuffer::new`) — one `cuMemAlloc`
  call (via `cuda.mem_alloc`, `driver.rs:187`) for the whole region;
  everything after is bump sub-allocation with zero further device
  allocation. Doc comment: "One `cuMemAlloc` at engine init, bump
  sub-allocation after that, zero allocation on the hot path."
- `glcuda/src/model.rs:591` — the **only** call site of `BackendBuffer::new`
  in the crate (grep-confirmed), inside `GpuModel::upload`, called once at
  load. Confirms "single VRAM allocation at load time."
- `glcuda/src/driver.rs:86-91` — `Cuda::launch_stream`: "NULL = the
  default stream (normal execution)... only ever flipped between launches
  by the single owning thread." All kernel launches in `record_forward`
  (runner.rs:280-376) go through `KernelSet` wrappers that use this one
  stream — no per-kernel stream creation in the decode path.
- `glcuda/src/runner.rs:643-687` (`decode_step`) — captures the per-token
  kernel sequence into a CUDA graph once (`cuda.capture`), then every
  subsequent token replays it with `cuda.graph_launch(graph)` — a single
  launch call per token on the one stream.
- `glcuda/src/runner.rs:716-723` (`logits_host`) — `cuda.synchronize()`
  (→ `cuCtxSynchronize`, `driver.rs:257-259`) is called once per token,
  immediately before the logits download, to read back sampling input.
  This is the one host-device sync point per token the requirement
  describes.
- `glcuda/src/runner.rs:388-618` (`prefill_batched`) — processes up to
  `PREFILL_BATCH` (512) tokens per pass through `gemm_rows`, which uses
  batched GEMM kernels (`gemm_q8_0_soa`, tensor-core `gemm_mma_q8`) that
  stream each weight row once per *chunk of tokens* rather than once per
  token — explicit data reuse decode does not have (decode's `gemv_w`,
  `runner.rs:43-114`, is one GEMV per weight per token).

**Delta from ARTX01:** None found — every clause of R2 is directly
supported by code, with file:line evidence for each sub-claim.

---

### R3 — Physical substrate: CONFIRMED (glproc); CONFIRMED (glcuda), with one nuance

**Claim:** glproc decode substrate = DDR4 dual-channel memory bus; glcuda
substrate = PCIe transfer + on-device VRAM bandwidth.

**Evidence:**
- `glproc/src/threading.rs:1-19` (module doc) — explicitly frames the
  chunked-vs-interleaved design choice in terms of "single-channel DDR4"
  page locality, and `glproc/src/runner.rs:36-44` (`N_THREADS` doc) states
  decode is "bandwidth-heavy" and cites "~69% of the DRAM read ceiling."
  No GPU code path exists anywhere under `glproc/src/` (confirmed by
  directory listing — no `cuda`/`vulkan`/`gpu` references in the crate).
- `glcuda/src/model.rs:582` (`cuda.mem_get_info()`) and `buffer.rs`'s
  `cuMemAlloc`-backed region confirm VRAM as the resident substrate;
  `glcuda/src/runner.rs:264` (`cuda.htod_f32`) and the per-token
  `set_token_inputs` (runner.rs:259-272) confirm host→device transfer
  (PCIe) happens each token for the embedding row and token params.

**Delta from ARTX01:** One nuance the requirement's single sentence
elides: **weight bytes never cross PCIe during decode** — they are
uploaded once at load (R2) and read by kernels entirely on-device (VRAM
bandwidth only). PCIe is exercised per-token only for the small
embedding-row upload and the logits download, not for the weight traffic
that dominates decode cost. The requirement is not wrong (it names both
substrates without claiming they're equally loaded per-token), but a
reader could misread "PCIe transfer and... VRAM bandwidth" as implying
both are hit on every token's *weight* access, which they are not.

---

### R4 — Physical vs abstraction: CONFIRMED

**Claim:** `ExecutionPlan`, `Constraint`, `Policy` are pure software,
consuming no hardware resource, and must not be described as if they did.

**Evidence:**
- `glcore/src/gate/mod.rs:6-17` (module doc, verbatim as of GATE Wave A) —
  "Every type here is either pure protocol data (a struct/enum with no
  compute) or orchestration control flow... fully specified by the
  paper's own definitions — nothing under `glcore::gate` performs
  inference compute itself. `glproc` is the first engine wired to it, via
  [`CandidateSource`]: `glcore::gate` supplies the protocol, the backend
  crate supplies the real per-op candidates." This is the source
  document's own explicit self-description. **Update (GATE Wave A + Wave
  C, post-dating this report's original 2026-07-25 audit):** `glproc` is
  now wired — `GlprocEngine::load_model` (`glproc/src/engine.rs`) calls
  `glproc::gate::resolve_prefer_q4k_native` once per session, which
  drives a real `glcore::gate::Planner` over `glproc::gate::
  FfnWeightFormatSource` (a real `CandidateSource`, calibrated against
  live-measured tok/s as of Wave C — see `glproc/src/gate.rs`). This does
  not change R4's core verdict: the *types* under `glcore::gate`
  (`ExecutionPlan`, `Constraint`, `Policy`) remain pure software with no
  hardware referent of their own, exactly as this section documents —
  what changed is that a real caller now exists and exercises them, not
  that their nature as software-only abstractions changed. See
  `architecture/GATE/GATE-mapping.md` §5 for the full wiring detail.
- `glcore/src/gate/plan.rs:69-81` (`ExecutionPlan`) — a plain struct:
  `ordering: Vec<TensorOp>`, `backend: BackendKind` (an enum), `layouts:
  HashMap<OpId, MemoryLayout>`, `metrics: MetricVector`. No I/O, no
  syscalls, no device handles anywhere in the type or its `Default` impl.
- `glcore/src/gate/constraint.rs:10-16` (`Constraint` trait) — one method,
  `validate(&self, plan: &ExecutionPlan) -> ValidationResult`; a pure
  function signature, no side-effecting capability in the trait contract.
- `glcore/src/gate/policy.rs:23-38` (`ExecutionPolicy::weight_vector`) —
  a `match` returning a fixed `[f64; 5]` array literal per variant; a
  table lookup, confirmed by the module doc's own characterization
  ("`ExecutionPolicy::weight_vector`'s table lookup").
- `glcore/src/gate/plan.rs:15,26` (`TensorOp`, `TensorGraph`) — explicitly
  documented as "Marker stub only... carries no shape/dtype information or
  behavior."

**Delta from ARTX01:** None. No place under `glcore::gate` touches
hardware directly; the module doc says so and the types corroborate it.

---

### R5 — Prefill vs decode: CONFIRMED

**Claim:** Prefill and decode are distinct workloads (decode:
bandwidth-adjacent, serial, no reuse; prefill: batched, compute-adjacent,
has reuse), not describable by one undifferentiated statement.

**Evidence — glproc:**
- `glproc/src/runner.rs:827` (`step`, single-token) vs
  `glproc/src/runner.rs:1136` (`step_chunk`, batched prefill) are two
  separate code paths with different signatures and structure.
- `glproc/src/runner.rs:121-125` (`PREFILL_CHUNK = 32`, doc) — "Batching
  lets every weight row stream from DRAM once per chunk instead of once
  per token — prefill flips from bandwidth-bound to compute-bound."
- `glproc/src/runner.rs:638-644` (`par_matmul` doc, threading.rs) — "the
  weight matrix streams from DRAM once per chunk instead of once per
  token — the point of batching prefill" (data reuse decode structurally
  cannot have, since decode processes one token's activation per weight
  read).

**Evidence — glcuda:**
- `glcuda/src/runner.rs:280` (`record_forward`, per-token decode kernel
  sequence, captured into a graph) vs `glcuda/src/runner.rs:388`
  (`prefill_batched`, chunked up to `PREFILL_BATCH=512` tokens) are
  distinct functions.
- `glcuda/src/runner.rs:122-127` (`gemm_rows` doc) — "Q8_0-SoA weights use
  the batched GEMM (weight streamed once per token tile); f32 falls back
  to a per-token GEMV" — explicit reuse-vs-no-reuse split inside the same
  function, gated on batch mode.
- `glcuda/src/runner.rs:43` (`gemv_w`, decode) — always one GEMV per
  weight per token; no batching parameter exists in this function's
  signature at all, structurally forcing "no reuse."

**Delta from ARTX01:** None. Both engines implement prefill and decode as
genuinely separate code paths with the reuse/no-reuse property structural,
not incidental.

---

### R6 — Stub status: CONFIRMED

**Claim:** glvulkan and glmetal report `available: false` unconditionally.

**Evidence:**
- `glvulkan/src/lib.rs:40-46` (`GlvulkanEngine::capabilities`) —
  ```rust
  EngineSpec { name: "glvulkan", backend: "vulkan", available: false }
  ```
  Unconditional literal, no branch.
- `glmetal/src/lib.rs:40-46` (`GlmetalEngine::capabilities`) — identical
  pattern:
  ```rust
  EngineSpec { name: "glmetal", backend: "metal", available: false }
  ```
- Both crates' `init`, `load_model`, `infer` unconditionally return
  `Err(not_implemented())` (`glvulkan/src/lib.rs:21-36`,
  `glmetal/src/lib.rs:21-36`).
- Both have a test enforcing this (`stub_reports_unavailable`,
  `glvulkan/src/lib.rs:53-58`, `glmetal/src/lib.rs:53-58`).

**Delta from ARTX01:** None.

---

## Additional Findings

### MoE path
`glproc` has full `_exps` tensor handling, but with one explicitly
flagged unverified assumption:
- `glproc/src/loader.rs:333-386` (`split_experts`) — slices a 3-D
  `blk.{i}.ffn_{gate,up,down}_exps.weight` tensor into per-expert
  `WeightMatrix`es, assuming experts are the contiguous outermost axis.
- `glproc/src/loader.rs:291-323` (`_EXPS_LAYOUT_ASSUMPTION` const + doc) —
  **explicitly marked unverified**: "No Qwen3-MoE file was available when
  this was written... this function is the only unverified link in the
  chain." Shape/count mismatches are rejected loudly (`loader.rs:347-365`),
  but a layout that satisfies the shape check while stacking experts in
  the wrong order would not be caught.
- `glproc/src/moe.rs:1-33` (module doc) — the compute side (routing,
  per-expert SwiGLU via `par_matvec_swiglu`/`par_matmul_swiglu`, expert
  skip for unrouted tokens) is described as verified against a naive
  reference at Qwen3's real dims; `split_experts` is the one link that is
  not.
- Wired: `glproc/src/loader.rs:618-619` (`build_layer`) calls
  `moe_config_for` per layer and constructs `FfnLayer::MoE` when
  `ffn_gate_exps` is present; `glproc/src/runner.rs:1022-1028` (`step`)
  calls `moe.forward(...)` in the FFN block when the layer is MoE.

This matches the existing session memory
([[project_glproc_moe_status]]) — nothing new found beyond what was
already tracked, confirmed still accurate as of this reading.

### Memory pinning
`glproc/src/loader.rs:22-31,126-168` (`warm_and_lock_model`):
- Windows: `VirtualLock` (`winmem` module, `loader.rs:22-31`), plus
  `SetProcessWorkingSetSize` first to raise the lockable cap
  (`loader.rs:130-142`).
- Unix: `libc::mlock` (`loader.rs:150-158`).
- Prefetch/touch step before pinning: `std::ptr::read_volatile` once per
  4 KiB page (`loader.rs:111-120`, `PAGE_BYTES = 4096`) — not
  `madvise`/`MADV_WILLNEED`/`PrefetchVirtualMemory`; the touch is a manual
  read loop, not an OS hint call.
- Opt-out: `GLPROC_NO_LOCK` env var (`loader.rs:126-128`).
- Also `glproc/src/engine.rs:349-358` (`load_model`) spawns a background
  thread that sequentially reads the raw GGUF file in 1 MiB chunks purely
  to warm the OS page cache behind the mmap — this runs concurrently with
  parsing/dequantization, separate from the `warm_and_lock_model` pass.

No `madvise`/`MADV_WILLNEED`/`PrefetchVirtualMemory` calls found anywhere
in `glproc/src` (only the manual touch-loop and the background sequential
read).

### AVX2 SIMD coverage
20 files under `glproc/src/kernels/` contain
`#[target_feature(enable = "avx2")]`-style AVX2 code (avx2.rs siblings
throughout `dequant/`, `qdot/`, `matmul/`, `gquant/`, and `ops/{softmax,
attn_accum,silu,rms_norm,fast_exp}`). Dispatch is centralized: `glproc/src/
simd_strategy.rs:20-56` (`SimdStrategy::detect`) probes `avx512f`+
`avx512bw`, and separately `avx2`+`fma`+`f16c` (F16C required because wide
kernels convert block scales via `vcvtph2ps`), caching the result in a
`OnceLock`. Notably: AVX-512 is **deliberately downgraded to AVX2** on
machines with ≤8 logical cores (`simd_strategy.rs:37-44`) because AVX-512
triggers a frequency throttle on mobile/laptop parts that makes 4-thread
AVX2 faster in practice — a policy decision, not a detection bug.

### Q6_K dequant path
`glcore::format::gguf::dequant_q6_k` **exists** (`glcore/src/format/
gguf.rs:548`, called internally at line 461 for `GgufDType::Q6_K`) but is
**NOT** used by glproc for tensor loading. `glproc/src/loader.rs:182-193`
(`dequant_any`) explicitly routes `GgufDType::Q6_K` through glproc's own
`kernels::dequant::q6_k::scalar::run`, with a doc comment stating why:
"glcore's Q6_K assumes a naive linear nibble order that disagrees with
real llama.cpp files." This matches session memory
([[project_gllm_e2e_garbage_output]]) — the Q6_K nibble-order bug that was
fixed in the `.gllm`/glconv path was never present in glproc, which always
had its own correct kernel. glcuda independently has its own native Q6_K
SoA path too (`glcuda/src/repack.rs`'s `q6_k_to_soa`, called from
`glcuda/src/loader.rs:94-97`), not routed through glcore either.

### GLPROC_PROFILE telemetry
Implemented and wired to real (not stub) data:
- `glproc/src/runner.rs:773-776` (`Runner::new`) — `GLPROC_PROFILE` env
  var (any non-empty value except `"0"`) turns on a `Box<Prof>` accumulator.
- `glproc/src/runner.rs:340-393` (`struct Prof`) — real per-phase
  `Duration` accumulators (qkv, attn, wo, gateup, down, lm_head, sampler,
  plus prefill-specific buckets and MoE expert-load counters), populated
  by `lap()` timestamp calls threaded through `step`/`step_chunk`.
- `glproc/src/runner.rs:442-543` (`Prof::to_telemetry`) — converts the raw
  counters into `glcore::telemetry::EngineTelemetry` with real
  `bytes_read`/`macs` derived from `StageWork::measure` (runner.rs:575-653),
  which reads actual `WeightMatrix` buffer sizes — not hardcoded or
  estimated.
- `glproc/src/engine.rs:151-206` (`backend_telemetry`) — reports the
  **actually selected** SIMD strategy and per-tensor kernel path (e.g.
  distinguishes "Q4K q8k integer-dot (native)" from a generic
  "{fmt:?} integer-dot"), explicitly contrasted in its own doc comment
  against "what the CPU supports" to avoid misreporting AVX-512-capable-
  but-not-used machines.
- When `GLPROC_PROFILE` is unset, `Runner.prof` is `None` and `Engine::
  telemetry()` (`engine.rs:331-333`) returns `None` — confirmed zero-cost
  off-path, not a stub returning fake zeros.

### Thread count control
- `glproc/src/runner.rs:45,48-56` (`N_THREADS`, `n_threads()`) — default
  `4`; overridable via `GLPROC_THREADS` (parsed as `usize`, must be ≥1);
  clamped to `num_cpus::get()` (**logical** core count).
- `glproc/src/runner.rs:36-44` (doc on `N_THREADS`) — explicitly
  *deliberate* use of logical threads, not `topology::physical_core_count()`
  — measured regression (23% slower, 8.5 vs 11.0 tok/s) when sized from
  physical cores on Qwen3-1.7B Q8_0, attributed to decode leaving gaps an
  SMT sibling can fill.
- `glproc/src/topology.rs:1-46` (`physical_core_count`) — a **separate**
  physical-core detector exists (sysfs → /proc/cpuinfo → halved-logical
  fallback) but per `topology.rs:10-12`'s own doc, is used only by the
  **load-time layer-repack pool** (`loader.rs:719`, `num_cpus::get()`
  actually — see note below), not the decode pool.
- **Correction/nuance**: re-checking `loader.rs:719`
  (`let n_workers = num_cpus::get().clamp(1, 8)...`) — the load-time
  repack pool actually sizes from **logical** core count too (clamped to
  8), not from `topology::physical_core_count()`. The `topology.rs`
  module's doc comment (lines 10-12) claims the load-time pool "still
  sizes from the logical count," which is consistent — but this means
  `physical_core_count()` is a **built, tested, but currently unused**
  function in the decode/load paths as far as this audit found. A
  targeted search found no call site of `topology::physical_core_count()`
  outside its own test module. This is worth flagging: the function
  exists and is correct per its own tests, but nothing in `glproc/src`
  currently calls it.

---

## Open Items for ARTX02

- Actual DDR4 bandwidth achieved vs theoretical ceiling (glproc decode) —
  requires a running measurement, not derivable from source alone (prior
  session memory cites ~69% of a measured 29.4 GB/s ceiling on one
  specific machine, i3-1115G4 — that number is itself a measurement
  artifact of ARTX02's predecessor work, not something this read-only
  audit re-derived).
- Actual PCIe transfer time and VRAM bandwidth utilization for glcuda —
  no GPU was queried during this audit; `glcuda/src/runner.rs`'s
  `GLCUDA_PROFILE_PREFILL`/`GLCUDA_PROFILE_DECODE` env-gated instrumentation
  exists in code (confirmed) but was not exercised.
- Whether `physical_core_count()` being unused anywhere is intentional
  (dead code kept for a future use) or an oversight — code alone cannot
  answer intent.
- The MoE `_exps` layout assumption (`_EXPS_LAYOUT_ASSUMPTION`) remains
  unverified against a real file; this audit confirms the code's own
  flagging of that fact but cannot resolve it without a real Qwen3-MoE
  GGUF, which is a measurement/data question, not a code-reading one.

## Surprises

1. **The mmap claim in R1 is the one place this audit found a real gap
   between spec and code**, not just an imprecision: R1's single sentence
   implies the mmap is what decode streams from, but glproc's own code
   comments say plainly that decode reads owned heap copies, not the mmap.
   This should be corrected in any downstream document (ARTX02
   especially, since it will characterize decode bandwidth) — the
   memory-bus traffic ARTX02 measures is against process heap buffers,
   not mapped file pages.
2. **`topology::physical_core_count()` appears to be unused** in the
   actual decode or load-time thread-pool sizing, despite being fully
   built, documented, and tested. Both pools (decode: `runner.rs`;
   load-time repack: `loader.rs:719`) size from `num_cpus::get()`
   (logical), not this function. Worth a follow-up to confirm this is
   intentional (kept for a future physical-core-aware path) rather than
   orphaned code.
3. **`glproc/src/memory.rs`'s `Arena` bump allocator appears unused by
   the decode hot path** — `Runner::Workspace` (runner.rs:267-298) uses
   plain `Vec<f32>` buffers allocated once at `Runner::new`, not `Arena`.
   No call site of `Arena::new` was found outside `memory.rs`'s own tests
   during this audit's reading. This is outside ARTX01's scope (not a
   claim R1-R6 makes), but is the kind of drift a memory or cost model
   built later might wrongly assume is load-bearing.
4. Both glproc and glcuda independently re-derive the **same** Q6_K
   nibble-order fix outside glcore — glcore's `dequant_q6_k` exists and is
   internally self-consistent (tested), but neither production engine
   trusts it. This is documented reasoning in glproc (loader.rs comment)
   but glcuda's `q6_k_to_soa` (repack.rs) was not read in enough depth by
   this audit to confirm whether it independently arrived at the same
   nibble order or simply never routed through glcore in the first place
   for unrelated (SoA-layout) reasons — flagged as a gap in this audit's
   own coverage, not a code finding.
