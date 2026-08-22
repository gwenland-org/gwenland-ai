# glcuda bottleneck audit — 2026-08-22

Static audit of the whole `glcuda/` crate, host overhead through kernel
compute. **No code was changed.** Every claim cites file + line range.

Evidence is labelled throughout:

* **[CODE]** — read directly out of the source. Not a performance claim.
* **[PATTERN]** — inferred from a known hardware behaviour plus what the
  code does. A lead, not a verdict.
* **[MEASURED]** — a number someone actually took, with its provenance.

Target: T4 / sm_75, 40 SMs, ~320 GB/s. Known-context items from the brief
(no `cp.async` on sm_75; r256 double-buffer landed in `e370965`; the `%r25`
staging clobber fixed in `30a8ae3`; CUDA Graphs active in decode) are taken
as given and not re-derived. Both r256 commits confirmed ancestors of HEAD.

**Note on the brief's file list.** `docs/rejected-optimizations.md` does not
exist. The file is [gl-agent-skills/cpu-skills/rejected-optimizations.md](../../gl-agent-skills/cpu-skills/rejected-optimizations.md)
and it is **entirely CPU-side** — glproc/glcore/gltokenizer, explicitly
scoped to "the i3 tier". It contains **no glcuda entries at all**, so it
does not constrain this audit. The GPU-side rejection record is
[architecture/glcuda-research/ceiling-sprint-summary.md](ceiling-sprint-summary.md) §7,
which does, and §8 below is built from it. `glcuda/src/repack.rs` exists;
`glcuda/src/kernels/glcuda.ptx` exists (it is the main sm_70 module, and is
the larger of the two).

---

## 1. Synchronization Audit

Every blocking call in `glcuda/src/`, in full.

| # | Site | When | Necessary? | Eliminate? |
|---|---|---|---|---|
| S1 | [driver.rs:623](../../glcuda/src/driver.rs#L623) — `graph_launch` ends with `self.synchronize()` | **every decode token** | No | **Y — the headline item** |
| S2 | [runner.rs:793](../../glcuda/src/runner.rs#L793) — `logits_host` syncs before the DtoH | every decode token | Yes, in principle | N (but see S1) |
| S3 | [runner.rs:833](../../glcuda/src/runner.rs#L833) — after `prefill_batched` | once per generate | Yes — honest prefill timing | N |
| S4 | [runner.rs:880](../../glcuda/src/runner.rs#L880) — per-token sync in the decode loop | only `GLCUDA_PROFILE_DECODE=1` | Diagnostic | N |
| S5 | [runner.rs:529,532](../../glcuda/src/runner.rs#L529-L532) — the `phase!` macro | only `GLCUDA_PROFILE_PREFILL=1` | Diagnostic | N |
| S6 | [driver.rs:607](../../glcuda/src/driver.rs#L607) — `sync_pool`, one `cuStreamSynchronize` per pool stream | per GEMM slab group, only `GLCUDA_MULTI_STREAM_PREFILL=1` | Self-described "blunt on purpose" | N — feature is measured dead (§8) |
| S7 | [lib.rs:249](../../glcuda/src/lib.rs#L249) — after weight upload | once at load | Yes | N |

**S1 is the finding.** [CODE] `graph_launch` issues the graph and then
blocks the host on `cuCtxSynchronize` before returning. So the decode loop
in [runner.rs:849-882](../../glcuda/src/runner.rs#L849-L882) is strictly
serial with no overlap in either direction:

```
[GPU: whole graph] → block → [HOST: DtoH logits, penalty, sample]
                              → [HOST: dequant embed row, 2× HtoD] → [GPU: …]
```

The GPU is idle for the entire host phase; the host is idle for the entire
GPU phase. A second consequence: the comment at
[runner.rs:853-855](../../glcuda/src/runner.rs#L853-L855) says `logits_host`'s
internal sync "is a no-op" because the previous token "we already synced
below" — the sync below (S4) only exists under `GLCUDA_PROFILE_DECODE`. The
conclusion is right but the stated reason is wrong: it is a no-op because
**S1** already synced. Two `cuCtxSynchronize` round trips per token where
one would do.

Removing S1 alone buys little, because the algorithm is genuinely serial —
the next token's embedding depends on the sampled token. The win comes from
shrinking the *host* leg, which §7/B1 covers.

---

## 2. Hot Path Allocations

**Clean.** [CODE] No device allocation on any hot path.

* One `cuMemAlloc` for everything, at load: [buffer.rs:95-99](../../glcuda/src/buffer.rs#L95-L99).
  Weights, KV cache, activations and scratch are bump sub-allocations of
  that single region ([buffer.rs:85-113](../../glcuda/src/buffer.rs#L85-L113)).
* Host buffers live in the workspace and are moved out/in with
  `std::mem::take` rather than reallocated —
  [runner.rs:336-339](../../glcuda/src/runner.rs#L336-L339) (embed row),
  [runner.rs:794-797](../../glcuda/src/runner.rs#L794-L797) (logits),
  [runner.rs:545-556](../../glcuda/src/runner.rs#L545-L556) (prefill staging).
* The sampler reuses `self.candidates` / `self.probs` via `.clear()` +
  `.extend()` — [sampler.rs:156-169](../../glcuda/src/sampler.rs#L156-L169).
  No allocation, though see §7/B1 for what it *does* cost.

The crate's zero-`cuMemAlloc`-after-init contract holds. This matters for
§8: it is the reason K-splitting was deferred rather than tried.

---

## 3. Kernel Launch Count (per token, decode path)

[CODE] Counted from [runner.rs:373-453](../../glcuda/src/runner.rs#L373-L453)
(`record_forward`) plus the `gemv_w` expansion at
[runner.rs:43-114](../../glcuda/src/runner.rs#L43-L114). Every quantized
`gemv_w` is **two** launches: `quantize_q8` then the GEMV.

Per layer:

| Op | Launches | Note |
|---|---|---|
| `rms_norm` (attn_norm) | 1 | grid (1,1,1) |
| `gemv_w` wq / wk / wv | 6 | 3 × (quantize + gemv) |
| `add` bias q/k/v | 0 or 3 | Qwen2 yes, Qwen3 no |
| `rms_norm` q_norm | `n_heads` | **one launch per head** |
| `rms_norm` k_norm | `n_kv_heads` | **one launch per head** |
| `rope` × 2 | 2 | |
| `kv_write` × 2 | 2 | |
| `attn_decode` | 1 | grid (n_heads,1,1) |
| `gemv_w` wo | 2 | |
| `add` residual | 1 | |
| `rms_norm` (ffn_norm) | 1 | |
| `gemv_w` w_gate_up | 2 | fused gate+up |
| `silu_mul` | 1 | |
| `gemv_w` w_down | 2 | |
| `add` residual | 1 | |

Plus a tail of 3 (`rms_norm` + `gemv_w` output).

Worked totals:

| Model | Shape | Per layer | **Per token** |
|---|---|---|---|
| Qwen2.5-0.5B | 24L, bias, no q/k-norm | 25 | **603** |
| Qwen3-1.7B | 28L, no bias, 16 q-heads / 8 kv-heads | 46 | **1291** |

Two things fall out.

**The per-head norm loop.** [CODE]
[runner.rs:392-403](../../glcuda/src/runner.rs#L392-L403) launches one
`rms_norm` per attention head, and each of those is a **grid of exactly one
block** ([kernels/mod.rs:857](../../glcuda/src/kernels/mod.rs#L857)) doing a
`head_dim`-element reduction. On Qwen3-1.7B that is 24 single-block launches
per layer, 672 per token — **52% of all launches**, each using 1 of 40 SMs.
A batched kernel already exists: `gl_rms_norm_rows_f32` launches
`grid = (rows,1,1)` ([kernels/mod.rs:261](../../glcuda/src/kernels/mod.rs#L261))
and "one row per head" is exactly this shape.

**Redundant activation quantization.** [CODE] `wq`, `wk` and `wv` all take
the same input `xn` with the same `in_dim`
([runner.rs:377-379](../../glcuda/src/runner.rs#L377-L379)), and `gemv_w`
quantizes its input *internally* into the shared scratch
`ws.q8_qs` / `ws.q8_scales` ([runner.rs:54-58](../../glcuda/src/runner.rs#L54-L58)).
So `quantize_q8` runs **three times per layer producing byte-identical
output into the same buffer**; two are pure waste.

The prefill path already does this correctly — it hoists the quantize out
and calls `gemm_rows` three times against one quantized copy
([runner.rs:580-583](../../glcuda/src/runner.rs#L580-L583)). Decode simply
never got the same treatment. This is an asymmetry inside the file, not a
design trade-off.

---

## 4. Host-Device Transfer Audit

[CODE] **Every transfer in the crate is synchronous and unpinned.** The FFI
layer binds only `cuMemcpyHtoD` / `cuMemcpyDtoH` / `cuMemcpyDtoD`
([ffi.rs:144-146](../../glcuda/src/ffi.rs#L144-L146), bound at
[ffi.rs:237-239](../../glcuda/src/ffi.rs#L237-L239)). There is **no `*Async`
variant and no `cuMemAllocHost`/`cuMemHostAlloc` anywhere** — verified by
grep over `ffi.rs`. Every host buffer is pageable, so every copy is a
staged, host-blocking transfer.

Decode path, per token:

| Direction | Site | Size (0.5B / 7B) | Necessary? |
|---|---|---|---|
| HtoD | embedding row, [runner.rs:340](../../glcuda/src/runner.rs#L340) | 3.5 KB / 14.3 KB | Yes — but see below |
| HtoD | `token_params` `[pos, cached_len]`, [runner.rs:348](../../glcuda/src/runner.rs#L348) | **8 B** | Yes, wrong mechanism |
| DtoH | full logits, [runner.rs:796](../../glcuda/src/runner.rs#L796) | **594 KB** (vocab 151936 × 4) | **No — §7/B1** |

Prefill: one HtoD per chunk of up to `PREFILL_BATCH`=512 rows
([runner.rs:558](../../glcuda/src/runner.rs#L558)) — already batched, the
comment records that per-token copies were the earlier shape. Good.

Three observations:

1. **The 594 KB logits DtoH is the largest per-token transfer in the
   engine, and it exists only to feed a host-side sampler.** [PATTERN]
   Pageable DtoH on a T4 runs well below pinned bandwidth; at a few GB/s
   effective this is a per-token cost in the ~100 µs class. It is paid to
   move data whose only consumer picks *one* index out of it.
2. **An 8-byte HtoD to push two integers** is a full driver round trip for
   a value that a 1-thread kernel could write, or that could ride along in
   the embedding copy. [CODE] It is outside the captured graph, so it costs
   a launch-equivalent every token.
3. **`dtod` is dead on the KV path.** [CODE] `KvCacheDev::write_k`/`write_v`
   ([kv_cache.rs:71-94](../../glcuda/src/kv_cache.rs#L71-L94)) have no
   callers in `glcuda/src/` — the `kv_write` kernel replaced them
   ([runner.rs:410-413](../../glcuda/src/runner.rs#L410-L413)). Only
   `examples/bench.rs` still calls `cuda.dtod`. Dead code, not a bottleneck;
   listed so it is not mistaken for a live path.

---

## 5. Stream Utilization

[CODE] **Effectively single-stream.** Three stream contexts exist:

* **Default (NULL) stream** — where all normal execution goes
  ([driver.rs:407](../../glcuda/src/driver.rs#L407); the module header at
  [kernels/mod.rs:5](../../glcuda/src/kernels/mod.rs#L5) states it).
* **Capture stream** — a throwaway non-blocking stream created per
  `capture()` and destroyed immediately after instantiation
  ([driver.rs:441-499](../../glcuda/src/driver.rs#L441-L499)). Not an
  execution stream.
* **`StreamPool`** — a real multi-stream path for prefill sub-slabs
  ([driver.rs:112-145](../../glcuda/src/driver.rs#L112-L145),
  [driver.rs:583-612](../../glcuda/src/driver.rs#L583-L612)), gated behind
  `GLCUDA_MULTI_STREAM_PREFILL` and **off by default**.

**No compute/transfer overlap exists anywhere, and none is currently
possible** — [PATTERN] overlap requires async copies on a non-default
stream, and §4 establishes that only the synchronous memcpy entry points
are bound. This is a prerequisite, not an optimization: nothing in §7 that
depends on overlap can be attempted before `cuMemcpyHtoDAsync`/`DtoHAsync`
and pinned host allocation are added to `ffi.rs`.

The pool's `sync_pool` is a host round-trip per stream
([driver.rs:604-611](../../glcuda/src/driver.rs#L604-L611)) rather than an
event wait — the code says so itself. Moot while the feature is off.

---

## 6. PTX Kernel Status Table

Two modules: `glcuda.ptx` (`.target sm_70`, 22 kernels, the main suite) and
`glcuda_sm75.ptx` (`.target sm_75`, 2 tensor-core kernels, loaded only when
the device reports sm_75+, [kernels/mod.rs:111](../../glcuda/src/kernels/mod.rs#L111)).

**Register counts: not determinable from this source.** [CODE] PTX declares
*virtual* registers (`.reg .b32 %r<48>`), which the JIT allocates down to
physical registers — the two are not the same number and the ratio is not
predictable by inspection. The only figure available is the `~92` in the
r256 header comment
([glcuda_sm75.ptx:478](../../glcuda/src/kernels/glcuda_sm75.ptx#L478)), and
the ceiling sprint separately reports `50 reg vs 44` for hand-vs-oxide
kernels ([ceiling-sprint-summary.md §7](ceiling-sprint-summary.md)). Real
numbers need `ptxas -v` or `cuobjdump`; the column below says so rather
than guessing.

### sm_75 tensor-core module

| Kernel | Purpose | Regs | Shared | k-loop pipelined? | Known issues |
|---|---|---|---|---|---|
| `gl_gemm_mma_q8` [:60](../../glcuda/src/kernels/glcuda_sm75.ptx#L60) | 8 m-tile INT8 GEMM, 64 rows/weight-read. **The default prefill GEMM** | needs `ptxas -v` | 2304 B (`sm_a` 2048 + `sm_xs` 256, [:76-77](../../glcuda/src/kernels/glcuda_sm75.ptx#L76-L77)) | **No** | B fragment `ld.global` sits immediately before the first `mma.sync` ([:214-232](../../glcuda/src/kernels/glcuda_sm75.ptx#L214-L232)) — no prefetch, no A double-buffer. 2-way smem bank conflict (below) |
| `gl_gemm_mma_q8_r256` [:480](../../glcuda/src/kernels/glcuda_sm75.ptx#L480) | 32 m-tile, 256 rows/weight-read. Opt-in `GLCUDA_R256` | ~92 (header claim) | 9216 B ([:502-503](../../glcuda/src/kernels/glcuda_sm75.ptx#L502-L503)) | **Partly** — B fragment + scales register double-buffered ([:794-810](../../glcuda/src/kernels/glcuda_sm75.ptx#L794-L810), swap at [:1457-1465](../../glcuda/src/kernels/glcuda_sm75.ptx#L1457-L1465)) | A staging still single-buffered. Same bank conflict. Generated by `emit_r256.py` |

**`bar.sync` structure is identical in both** and is correct, not wasteful:
two per k-iteration, bracketing the cooperative A-slice staging — one after
staging ([:209](../../glcuda/src/kernels/glcuda_sm75.ptx#L209) /
[:790](../../glcuda/src/kernels/glcuda_sm75.ptx#L790)), one at loop end
([:385](../../glcuda/src/kernels/glcuda_sm75.ptx#L385) /
[:1458](../../glcuda/src/kernels/glcuda_sm75.ptx#L1458)). Out-of-range warps
deliberately fall through to the barrier rather than exiting
([:96](../../glcuda/src/kernels/glcuda_sm75.ptx#L96)), which is required for
correctness. There is no barrier without corresponding work.

But [PATTERN] with A single-buffered, the global→shared staging latency is
**exposed** at each barrier: no warp has independent work to cover it. This
is the ceiling sprint's ranked item 2, and it is a symptom of low occupancy,
not an independent defect.

### Shared-memory bank conflict — determinable, and present

The brief asks not to guess here. The stride *is* recoverable from the PTX,
so here is the derivation.

Address computation ([glcuda_sm75.ptx:169-172](../../glcuda/src/kernels/glcuda_sm75.ptx#L169-L172),
using `groupID = lane/4` and `tig = lane%4` from
[:100-101](../../glcuda/src/kernels/glcuda_sm75.ptx#L100-L101)):

```
%r33 = sm_a_base + (lane/4)*32 + (lane%4)*4      ; m-tile stride 256 B
```

For `ld.shared.u32 [%r35]` across one 32-lane warp, in 4-byte words:

```
word(lane) = (lane/4)*8 + (lane%4)
bank       = word % 32
```

| lanes | banks touched |
|---|---|
| 0–15 | 0,1,2,3, 8,9,10,11, 16,17,18,19, 24,25,26,27 |
| 16–31 | **the same 16 banks** (lane 16 → word 32 → bank 0) |

→ **16 of 32 banks used, each by exactly 2 lanes: a 2-way conflict**, on
both `[%r35]` and `[%r35+16]`, in both MMA kernels. Confidence: **High** —
this is arithmetic on a stride the PTX states explicitly, not a pattern
guess.

By contrast `ld.shared.f32 [%r36]` (activation scales, stride `(lane/4)*4`)
has four lanes per address — that is a hardware **broadcast**, not a
conflict. Correctly shaped already.

The fix direction is row padding or an XOR swizzle of `sm_a`. **The exact
padding is not determined here**: naive `32→36 B` moves the collision rather
than removing it (lane 29 lands back on bank 0), and it perturbs both the
staging store and the hard-coded `+16` half-offset. It needs to be worked
out and parity-tested, not asserted.

### sm_70 main module — decode-relevant kernels

| Kernel | Grid | Block | Shared | Note |
|---|---|---|---|---|
| `gl_rms_norm_f32` [:1128](../../glcuda/src/kernels/glcuda.ptx#L1128) | **(1,1,1)** [mod.rs:857](../../glcuda/src/kernels/mod.rs#L857) | 256 | 36 B | **1 SM of 40.** Called `2 + n_heads + n_kv_heads` × per layer |
| `gl_attn_decode_f32` [:1464](../../glcuda/src/kernels/glcuda.ptx#L1464) | **(n_heads,1,1)** [mod.rs:924](../../glcuda/src/kernels/mod.rs#L924) | 128 | **16 KB** [:1480](../../glcuda/src/kernels/glcuda.ptx#L1480) | 14 blocks on 0.5B, 16 on 1.7B → **26 SMs idle** |
| `gl_gemv_q8_0_soa` [:702](../../glcuda/src/kernels/glcuda.ptx#L702) | `ceil(out/8)` | 256 | — | Healthy grid |
| `gl_gemv_q4_k_soa` / `q6_k_soa` [:1922](../../glcuda/src/kernels/glcuda.ptx#L1922) / [:2264](../../glcuda/src/kernels/glcuda.ptx#L2264) | `ceil(out/8)` | 256 | — | Fine for decode; the prefill problem is the *loop around them*, §7/A1 |
| `gl_quantize_q8` [:288](../../glcuda/src/kernels/glcuda.ptx#L288) | `ceil(n/32)` | **32** | — | Warp-per-block; tiny |
| `gl_gemv_f32` [:201](../../glcuda/src/kernels/glcuda.ptx#L201) | `(out,1,1)` | **32** | — | One warp per output row. F32 path only |
| `gl_rms_norm_rows_f32` [:2438](../../glcuda/src/kernels/glcuda.ptx#L2438) | `(rows,1,1)` | 256 | 36 B | **Exists and is unused by decode** — the batched-norm fix |

**GEMV fallback.** [CODE] There is no automatic GEMV fallback inside the
GEMM. The split is by weight *type* in `gemm_rows`
([runner.rs:157-300](../../glcuda/src/runner.rs#L157-L300)): only
`Q8_0Soa` reaches the tensor-core GEMM. `Q4KSoa`, `Q6KSoa`, `Q4_0Soa`,
`Q4_0` and `F32` all take `for t in 0..n { gemv }` — literally one kernel
launch per prompt token per weight
([runner.rs:262-300](../../glcuda/src/runner.rs#L262-L300)). A 512-token
chunk issues **512 launches** for `down` alone. The MMA path additionally
requires `rows % 8 == 0` and `has_mma()`
([runner.rs:167](../../glcuda/src/runner.rs#L167)).

Doc drift, harmless: the `gl_gemm_mma_q8` header says "the runner chunks at
`PREFILL_BATCH = 64`" ([:54](../../glcuda/src/kernels/glcuda_sm75.ptx#L54)),
but `PREFILL_BATCH` is 512 ([model.rs:431](../../glcuda/src/model.rs#L431)).
The `ntok ≤ 64` contract still holds because `gemm_rows` re-chunks into
64-row sub-slabs ([runner.rs:203](../../glcuda/src/runner.rs#L203)). The
comment is stale, the code is right.

---

## 7. Prioritized Bottleneck List

### A. Prefill

**A1 — k-quant weights never reach the GEMM. [MEASURED, and already
diagnosed]**
*Evidence:* [loader.rs:60-145](../../glcuda/src/loader.rs#L60-L145),
[runner.rs:262-300](../../glcuda/src/runner.rs#L262-L300).
*Confidence:* High — this is documented in-tree with numbers: the same
projection is 193 µs through the GEMM and 1215 µs through 64 GEMV calls
(**6.29×**), and `down`+`o` is **66%** of prefill while `down` never reaches
the GEMM ([loader.rs:64-77](../../glcuda/src/loader.rs#L64-L77)).
*Impact:* Large — it is the single largest known prefill cost.
*Fix:* `GLCUDA_FORCE_Q8` already exists
([loader.rs:84](../../glcuda/src/loader.rs#L84)) and is the candidate
default.
*Prerequisite:* **The A/B suite in `notebooks/force_q8_decision.ipynb`, on a
T4.** Not run yet. Note the escape hatch named in the decision rule,
`GLCUDA_NATIVE_KQUANT`, **does not exist in the codebase** — grep returns
nothing. It has to be written as part of flipping the default.

**A2 — the GEMM grid is ~5× too small to fill the machine. [MEASURED]**
*Evidence:* [ceiling-sprint-summary.md §6](ceiling-sprint-summary.md) —
32 blocks launched against 160 block slots = 20% of capacity, 25% achieved
occupancy, 8 SMs idle. Grid is `ceil_div(out_dim, 64)`
([kernels/mod.rs:609](../../glcuda/src/kernels/mod.rs#L609)), giving `down`
14 blocks on a 40-SM part.
*Confidence:* High.
*Impact:* Large — the sprint calls it "the only item with a clear mechanism
for a large further win", and its 2D-grid patch measured **3.28× / 3.56×**
in the diagnostic harness.
*Fix:* Finer `out_dim` tiling, or the 2D grid already built in that sprint.
*Prerequisite:* **The 2D-grid patch is not in the monorepo PTX.** The sprint
deliberately held it back pending a real-model glbench `prefill_tps` gate,
because this repo has twice recorded ~2× isolated wins that went neutral in
production. Land the gate, not the patch.

**A3 — B-fragment prefetch exists in r256 but not in the default kernel.
[PATTERN]**
*Evidence:* r256 prefetches at
[glcuda_sm75.ptx:794-810](../../glcuda/src/kernels/glcuda_sm75.ptx#L794-L810);
`gl_gemm_mma_q8` loads B at
[:214-215](../../glcuda/src/kernels/glcuda_sm75.ptx#L214-L215) and issues
`mma.sync` at [:229](../../glcuda/src/kernels/glcuda_sm75.ptx#L229) with
nothing between. `gl_gemm_mma_q8` is the default path (r256 is opt-in **and**
only selected when `n > 64`, [runner.rs:127-128](../../glcuda/src/runner.rs#L127-L128)).
*Confidence:* **Low, and deliberately ranked below A2.** The ceiling sprint
already considered exactly this and **deferred it**, with a reason that
still holds: *"Latency hiding is the wrong tool while achieved occupancy is
25% for lack of blocks."* At one block per SM there are no other warps to
hide behind, so prefetch helps far less than the diff size suggests.
*Impact:* Unknown, probably small until A2 lands.
*Prerequisite:* **A2.** Do not do this first.

**A4 — 2-way shared-memory bank conflict on the A fragment. [CODE +
arithmetic]**
*Evidence:* derivation in §6, from
[glcuda_sm75.ptx:169-172](../../glcuda/src/kernels/glcuda_sm75.ptx#L169-L172).
*Confidence:* High that the conflict exists; **Low** that removing it moves
wall-clock, for the same occupancy reason as A3.
*Impact:* Unknown. Bounded — it is an LDS latency effect inside a loop whose
stalls are currently dominated by exposed global staging.
*Fix:* Pad or swizzle `sm_a`. Exact scheme unresolved (§6).
*Prerequisite:* A2, and a parity run — both MMA kernels have parity tests
([parity.rs:430,472](../../glcuda/tests/parity.rs#L430)).

### B. Decode

**B1 — the full logits round-trip to a host sampler. [CODE, with a caveat]**
*Evidence:* 594 KB unpinned DtoH at
[runner.rs:796](../../glcuda/src/runner.rs#L796); host sampler materializes
a `(usize, f32)` pair for **every one of 151936 vocab entries** then runs
`select_nth_unstable_by` over all of them
([sampler.rs:155-165](../../glcuda/src/sampler.rs#L155-L165)); the GPU is
blocked throughout by S1.
*Confidence:* **Medium — and this must be stated carefully.** A prior
decode profile **measured the host share at ~8%** at T4/0.5B and recorded
"sampling-host" as a *dead* hypothesis. That measurement stands and this
audit does not overturn it. What the audit adds is that the 8% is *entirely
dead GPU time* under S1, and that the transfer is unpinned — neither of
which the "dead" verdict addressed. Treat 8% as the honest ceiling for this
item at 0.5B, and expect it to grow with vocab-relative model size, not
shrink.
*Impact:* Bounded by that 8%. Real but not transformative.
*Fix:* Device-side top-k/argmax so the DtoH is 4 bytes; or pinned host
memory as a cheaper partial step.
*Prerequisite:* Re-measure `GLCUDA_PROFILE_DECODE` at 3B/7B before spending
kernel work — the 8% figure is 0.5B-only.

**B2 — attention runs on `n_heads` blocks. [PATTERN]**
*Evidence:* grid `(n_heads,1,1)`
([kernels/mod.rs:924](../../glcuda/src/kernels/mod.rs#L924)) — 14 blocks
(0.5B) or 16 (1.7B) on 40 SMs, so **~60-65% of the machine is idle** during
the single most memory-heavy decode kernel. It reads the whole KV cache each
token.
*Confidence:* Medium — the block count is certain; the wall-clock share is
not. ⚠️ Note the standing correction in this repo: attention was once
believed to be 39% of prefill and **re-measured at 12%** on T4/0.5B. Do not
assume attention is large.
*Impact:* Unknown until profiled.
*Fix:* Split-K over the sequence dimension — the one prefill lead the
project has *not* killed.
*Prerequisite:* Profile first. Then note §8: K-splitting needs device
scratch, which collides with the zero-`cuMemAlloc` contract (§2) and needs
an `architecture/` sign-off.

**B3 — redundant `quantize_q8`, 2 of every 3. [CODE] — ✅ FIXED 2026-08-22**
*Evidence:* §3. `gemv_w` quantizes internally
([runner.rs:54-58](../../glcuda/src/runner.rs#L54-L58)); q/k/v all pass the
same `xn` ([runner.rs:377-379](../../glcuda/src/runner.rs#L377-L379)).
Prefill already hoists it
([runner.rs:580-583](../../glcuda/src/runner.rs#L580-L583)).
*Confidence:* High that the redundancy is real; Medium on impact.
*Impact:* Removes 2 launches/layer (48/token on 0.5B, 56 on Qwen3-1.7B) and
their duplicate compute. Inside a CUDA graph, per-node replay overhead is
sub-µs, so this is likely a **small** win — but it is nearly free and it
removes work rather than adding machinery.
*Fix:* Hoist the quantize into `record_forward`, mirroring prefill; pass
pre-quantized pointers to a `gemv_w` variant.
*Prerequisite:* None. **Cheapest item in this report.**

**B4 — per-head `rms_norm` as `n_heads` single-block launches. [CODE] — ✅ FIXED 2026-08-22**
*Evidence:* §3. [runner.rs:392-403](../../glcuda/src/runner.rs#L392-L403),
grid (1,1,1) at [kernels/mod.rs:857](../../glcuda/src/kernels/mod.rs#L857).
672 launches/token on Qwen3-1.7B — 52% of all launches, each on 1 SM.
*Confidence:* High on the count; Medium on impact (graph replay amortizes
launch cost, but not the serialization of 24 dependent single-block kernels
per layer).
*Impact:* Medium on Qwen3-class models. **Zero on Qwen2.5** — it has no
q_norm/k_norm, so this does not exist there.
*Fix:* One `rms_norm_rows` launch with `grid = (n_heads,1,1)`. The kernel
already exists ([glcuda.ptx:2438](../../glcuda/src/kernels/glcuda.ptx#L2438)).
*Prerequisite:* None. Second-cheapest.

**B5 — KV cache is f32. [CODE]**
*Evidence:* [kv_cache.rs:8-11,21](../../glcuda/src/kv_cache.rs#L8-L21) — a
deliberate, documented deviation from the §12 f16 budget, taken for
numerical parity with glproc.
*Confidence:* High that halving it halves attention's DRAM traffic; Medium
that this shows up end-to-end, given B2's warning about attention's real
share.
*Impact:* Potentially large on long contexts, and it is also a **VRAM**
item: at `MAX_KV_CONTEXT`=4096
([model.rs:20](../../glcuda/src/model.rs#L20)), Qwen3-1.7B reserves
`28 × 2 × 8 × 4096 × 128 × 4 B` ≈ **940 MB** of KV alone.
*Fix:* f16 KV, or an f16 KV read path.
*Prerequisite:* Numerical characterization against glproc — the file says
so, and it is right. This is a correctness question before it is a
performance one.

**B6 — the 8-byte `token_params` HtoD. [CODE]**
*Evidence:* [runner.rs:342-348](../../glcuda/src/runner.rs#L342-L348).
*Confidence:* High that it is wasteful; High that it is **tiny**.
*Impact:* One driver round trip per token. Listed for completeness, not as
a lead.
*Fix:* Fold into the embedding copy, or write it with a trivial kernel
inside the graph.
*Prerequisite:* None.

### Ranking

| | Item | Confidence | Est. impact | Blocked on |
|---|---|---|---|---|
| 1 | A1 force_q8 default | High | Large (6.29× on the affected GEMMs) | T4 A/B — see §"Task 1 status" |
| 2 | A2 GEMM grid 5× too small | High | Large (3.28× isolated) | glbench prefill gate |
| 3 | B3 redundant quantize | High | Small, ~free | ✅ **landed** |
| 4 | B4 batched per-head norm | High | Medium (Qwen3 only) | ✅ **landed** |
| 5 | B5 f16 KV | High mechanism | Large VRAM, unclear tok/s | numerical characterization |
| 6 | B2 attention split-K | Medium | Unknown | profile; scratch-alloc sign-off |
| 7 | B1 device-side sampling | Medium | ≤8% at 0.5B (measured) | re-measure at 3B/7B |
| 8 | A3/A4 prefetch, bank conflict | Low | Unknown | A2 first |
| 9 | B6 8-byte HtoD | High | Negligible | — |

Items 3 and 4 are the ones worth doing before any measurement, because they
remove work and machinery rather than adding it.

---

## 8. What NOT to Touch

Sourced from [ceiling-sprint-summary.md §7](ceiling-sprint-summary.md),
[ceiling-sprint-phase2.md §3](ceiling-sprint-phase2.md), and the in-tree
comments. **Do not re-propose these.**

| Item | Status | Why |
|---|---|---|
| **cuda-oxide port** | **REJECTED, track closed** | 7.7× slower. Root-caused: runtime `m_tiles` bound blocks accumulator register promotion → `.local` spill on every MMA |
| **Remove the `ntok ≤ 64` contract** | **REJECTED (Phase 2)** | The contract was never the blocker; the useful part was subsumed without touching the 8-m-tile accumulator structure |
| **Multi-stream prefill** | **MEASURED DEAD, −0.6%** | The grid is too small *and* `down` never uses the GEMM anyway. Code kept behind `GLCUDA_MULTI_STREAM_PREFILL`, off by default |
| **r256 as a general win** | **Measured, conditional** | +31% at 512-row chunks, but ties at n ≤ 64 and loses on shared memory (9216 B vs 2304). The `r256_pays` rule already encodes this ([runner.rs:127](../../glcuda/src/runner.rs#L127)) |
| **Double-buffer / async prefetch** | **DEFERRED, reason still valid** | "Latency hiding is the wrong tool while achieved occupancy is 25% for lack of blocks." This is why A3/A4 rank last |
| **K-slicing / persistent accumulation** | **DEFERRED** | Needs f32 atomics or a second pass + device scratch → violates the zero-`cuMemAlloc`-after-init contract (§2). Needs an `architecture/` spec update and sign-off, not a patch |
| **Attention as the prefill hotspot** | **DEAD** | Believed 39%, **re-measured 12%** on T4/0.5B. `down`+`o` is 66% |
| **Block-count as the prefill hotspot** | **DEAD as tested** | Multi-stream A/B −0.6%. ⚠️ Note this killed *that test*, not A2 — the ceiling sprint's 2D grid attacks the same root cause by a different mechanism and measured 3.28× |
| **Sampling on the host** | **MEASURED SMALL, 8%** | Recorded dead. B1 does not overturn it; it only adds that the 8% is dead GPU time and the copy is unpinned |
| **L2 tiling for decode, interleaved rows, AVX-512F** | REJECTED | **CPU-side (glproc) only.** Listed in the brief as glcuda constraints; they are not — see the note at the top |

Two corrections to the prior record, both in glcuda's favour:

* **r256's "correctness FAIL on all cases"**
  ([ceiling-sprint-summary.md §8](ceiling-sprint-summary.md)) is
  **superseded**. That was the `%r25` staging clobber, fixed in `30a8ae3`;
  both it and the `e370965` double-buffer are confirmed ancestors of HEAD.
  The sprint doc has not been updated and will mislead anyone reading it
  cold.
* The same §8 notes r256's parity test "had never run on real hardware."
  ⚠️ Still a live trap: [parity.rs:43-46](../../glcuda/tests/parity.rs#L43-L46)
  **skips silently** with `eprintln!("SKIP: ...")` and a bare `return` when
  no CUDA device is present, and again below sm_75 at
  [:433](../../glcuda/tests/parity.rs#L433) / [:475](../../glcuda/tests/parity.rs#L475).
  A green `cargo test -p glcuda` on a machine without a T4 proves nothing
  about either MMA kernel.

### Two non-issues, so they are not mistaken for leads

* **Hot-path allocation** — there is none (§2). The bump allocator is
  correct and the workspace `mem::take` discipline is consistently applied.
* **`bar.sync` without overlapping work** — the brief asks about this. Both
  MMA kernels' barriers bracket real cooperative staging, and the
  fall-through for out-of-range warps is required for correctness, not an
  oversight (§6).

---

## Task 1 status — the A/B env bug

Recorded here because A1's verdict depends on it.

**The fix is already in the tree, at HEAD (`cf4d5ee`).** The brief's
diagnosis does not match this codebase in two ways:

1. It names `notebooks/glcuda_probe.ipynb`. The bug was in
   `notebooks/force_q8_decision.ipynb`.
2. It says `sh()` "passes `AB_ENV` as the entire environment, so the
   subprocess loses `PATH`". `sh()` has **always** merged correctly —
   `e = dict(os.environ); e.update(env or {})` — in both notebooks. The
   real bug was narrower: `one()` accepted an `env` parameter and never
   forwarded it to `sh()`.

Verified two ways on this machine:

* **Static** — every A/B call site now passes `env=env`:
  `force_q8_decision.ipynb` cell 6 line 14, `glcuda_probe.ipynb` cell 12
  line 15.
* **Executed** — `sh()` extracted verbatim and run against a probe
  subprocess:

  ```
  base  rc=0  FLAG= None  PATH_LEN= 2781
  var   rc=0  FLAG= 1     PATH_LEN= 2781
  ```

  The arms genuinely differ, and `PATH` survives identically in both.

**The suite has not been run.** This machine has no CUDA device
(`nvidia-smi` and `nvcc` both absent) — the notebook targets Kaggle/Colab
(`/kaggle/working`). `banner seen in var arm: True` and the 0.5B/3B/7B
tables require a T4 session.

The notebook now refuses to produce a verdict from identical arms: it gates
on **both** the banner and on VRAM having moved
(`INVALID -- the 7B arms did not differ`). VRAM is the stronger check —
requantizing k-quants from 4.5–6.5625 bpw to 8.5 must raise it, so an
unchanged figure is proof the flag never took effect.

One gap to close when the default is flipped: **`GLCUDA_NATIVE_KQUANT` does
not exist yet** anywhere in `glcuda/`. The decision rule's escape hatch has
to be written, not just switched on.

---

## Addendum — B3 and B4 landed, 2026-08-22

Both implemented in [runner.rs](../../glcuda/src/runner.rs). Neither adds a
kernel, an env flag or a device allocation; both delete launches.

**B3.** `gemv_w` split into `gemv_w` (quantizes, then delegates) and
`gemv_w_pre` (no quantize), gated by a new exhaustive
`consumes_q8_act(&GpuWeight)`. `record_forward` now issues one `quantize_q8`
over `xn` and hands the shared scratch to all three q/k/v GEMVs — the shape
`prefill_batched` has always used. Existing `gemv_w` call sites (`wo`,
`w_gate_up`, `w_down`, `output`) are unchanged; each has a distinct input.

**B4.** The per-head `rms_norm` loops became one `rms_norm_rows` call each.
The wrapper's own doc already named this use, and prefill already calls it
identically with `rows = n * n_heads` — decode is the `n = 1` case.

Launch count per token, recomputed from the same method as §3:

| Model | Before | After | Δ |
|---|---|---|---|
| Qwen2.5-0.5B | 603 | **555** | −8% (quantize only; no q/k-norm on Qwen2.5) |
| Qwen3-1.7B | 1291 | **619** | **−52%** |

### Why this is expected to be bit-exact

* **B3** removes writes of bytes that were already there. `quantize_q8` is a
  pure function of `(x, in_dim)`; q/k/v share both, and all three wrote into
  the same `ws.q8_qs`/`ws.q8_scales`. The 2nd and 3rd writes were
  idempotent.
* **B4** was checked at the PTX level, not assumed. A normalized opcode diff
  of `gl_rms_norm_f32` against `gl_rms_norm_rows_f32` differs by exactly
  **five instructions** — `mul.lo.s32`, `mul.wide.u32`, `add.s64` ×2,
  `mov.u32`, the row-base offset — and is otherwise identical
  instruction-for-instruction, including the `sqrt`+`rcp` pair and the
  `(x * inv) * w` multiply order. `w` is deliberately *not* row-offset
  ([glcuda.ptx:2461-2465](../../glcuda/src/kernels/glcuda.ptx#L2461-L2465)),
  so it broadcasts across heads, which is what the per-head norm needs.
  In-place (`x == out`) is safe: the reduction completes at a `bar.sync`
  before the write phase begins.

Graph capture is unaffected — the quantize is now conditional on weight
type, which is fixed at load, so the recorded sequence stays token-invariant.

### ⚠️ What is verified, and what is not

Build clean, clippy clean (glcuda's only two warnings are pre-existing, both
in `driver.rs`), **39 host-side lib tests pass** — up from 37; the two new
ones pin `consumes_q8_act` per variant, because a variant wrongly answering
`false` would read a stale scratch and produce wrong numbers with no crash
and no failed launch.

**No GPU test has run.** All **25** device probes in `tests/parity.rs` and
`tests/forward.rs` printed `SKIP: no CUDA driver/device` and the suite still
reported `ok` in 0.01 s — the exact silent-skip trap recorded in §8. The
green suite is **not** evidence for either change.

Both still need, on a T4: `cargo test -p glcuda --release` with the parity
suite actually executing, and a decode A/B for the wall-clock claim. The
launch-count deltas above are counted from code and are certain; the
**time** they buy is not — CUDA graph replay amortizes per-node overhead, so
B3 in particular may well measure as noise. That would not make it wrong,
only small.

---

## Addendum 2 — the FP32 epilogue is a co-equal bottleneck, 2026-08-22

Raised in review; verified against the PTX and **confirmed**. This belongs
near the top of §7/A, above A3 and A4.

### The finding

Inside the k-loop, each m-tile's INT32 MMA result is converted and scaled
immediately ([glcuda_sm75.ptx:233-243](../../glcuda/src/kernels/glcuda_sm75.ptx#L233-L243)):

```
mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32   ; K =  0..15
mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32   ; K = 16..31
ld.shared.f32   %f4, [%r36]      ; activation scale
cvt.rn.f32.s32  %f7, %r38        ; cvt.rn.f32.s32  %f8, %r39        ;  |
mul.rn.f32      %f5, %f2, %f4    ;  | 6 FP32 ops per m-tile, per k-block
fma.rn.f32      %f10, %f7, %f5, %f10
mul.rn.f32      %f6, %f3, %f4    ;  |
fma.rn.f32      %f11, %f8, %f6, %f11
```

`cvt.rn.f32.s32` count: **16** in `gl_gemm_mma_q8`, **64** in the r256
variant (8 and 32 m-tiles × 2 accumulators). All inside `MMA_KLOOP`
(line 196), not the epilogue.

### The arithmetic, per warp, per m-tile, per k-block

Counted in issue slots rather than FLOPs, since `cvt` and `mul` are one
FLOP each while the TFLOPS figure assumes FMA:

| pipe | work | T4 rate per SM | cycles |
|---|---|---|---|
| tensor (INT8) | 2 × `m8n8k16` = 2048 MACs | 130 TOPS → 1022 MAC/cyc | **2.0** |
| FP32 (CUDA cores) | 6 ops × 32 lanes = 192 lane-ops | 64 FP32 lanes/cyc | **3.0** |

**The scaling epilogue costs ~1.5× the tensor-core work it serves** — very
roughly 60% of the kernel's issue time goes to CUDA cores, not tensor
cores. Confidence: **High** on the ratio (it follows from the instruction
mix and published rates); **Medium** on the wall-clock share, since it
assumes issue-bound rather than memory-bound execution, and the FFN GEMMs
were separately characterized as DRAM-bound.

### Why "keep s32 to the end" does not work here

The obvious fix — accumulate INT32 across the whole k-loop, convert once —
is **numerically invalid for Q8_0**. Every 32-element K block carries its
own f16 weight scale *and* its own f32 activation scale:

```
d = Σ_kb ( s32_acc[kb] × wsc[kb] × xsc[kb] )
```

Blocks have different quanta, so their integer accumulators cannot be
summed. The kernel already defers as far as the format permits: the two
chained MMAs inside one 32-K block do accumulate in s32 (K=16+16), and the
conversion happens once per 32-K block, which is the minimum. The op count
per output — 1 `cvt`, 1 `mul`, 1 `fma` — is already minimal for this
formulation.

### What could actually reduce it

The cost is a property of the **scale granularity**, not of the assembly.

* **Per-row activation scale** (instead of one per 32 elements, from
  `gl_quantize_q8`). Then `xsc` factors out of the k-loop entirely:
  `acc = xsc × Σ cvt(d) × wsc`. Removes both `mul`s — 6 FP32 ops → 4, about
  a third of the epilogue. If the pipe split above holds, that is roughly
  a 15-20% kernel win. Costs accuracy: it widens the activation quantum,
  and `EPS_Q8_GEMV` in `parity.rs` would have to be re-derived, not just
  loosened.
* **Coarser weight blocks** would allow full s32 deferral, but a weight
  format with one scale per row is no longer Q8_0 — a repack and accuracy
  decision, not a kernel edit.

**Ranking:** this sits below A2 (grid 5× too small) but above A3/A4. A2 is
upstream — `down` gets 14 blocks on 40 SMs, so ~65% of the machine is idle
regardless of what the instruction mix inside a block looks like. The two
compound; the grid one gates.

### Three claims from the same review that did NOT survive

* **"`@!%p101 bra` causes warp divergence and serialization."** The
  predicates are **warp-uniform** — `p11` derives from warp id, `p1..p7`
  from `ntok`, a kernel parameter — stated at
  [:47-52](../../glcuda/src/kernels/glcuda_sm75.ptx#L47-L52) and consistent
  with the code. A branch all 32 lanes take together is not divergence and
  does not serialize. Further, `mma.sync.aligned` is a warp-level collective
  requiring all 32 threads: predicating it per-thread is the pattern that
  would be undefined behaviour if the predicate were ever non-uniform. The
  branch is both correct and cheaper — it skips all remaining m-tiles at
  once, where a predicated-off `mma.sync` still consumes an issue slot.
* **"32-way bank conflict from the 32-byte stride."** A conflict exists but
  it is **2-way**, already recorded in §6. The address is
  `(lane/4)*32 + (lane%4)*4`, not `lane*32` — the 32-byte stride advances
  every 4th lane, and `(lane%4)*4` spreads each group of four across four
  consecutive words. Banks: 16 distinct, 2 lanes each. (Even `lane*32`
  would be 8-way, hitting banks {0,8,16,24}.)
* **"r256 spills accumulators to local memory."** There are **zero**
  `.local` / `ld.local` / `st.local` in the entire PTX file. The spill on
  record belongs to the **cuda-oxide port**, which spilled precisely
  because its `m_tiles` was a runtime loop bound
  ([ceiling-sprint-summary.md §7](ceiling-sprint-summary.md)); the
  hand-written kernel's full unroll is what keeps the accumulators in named
  registers. The `~92` register figure in the r256 header remains an
  unverified claim — PTX `.reg` counts are virtual, and `ptxas -v` is still
  the only way to settle it. r256's ~50% occupancy is real but not the
  binding constraint: the ceiling sprint **measured** 25% achieved occupancy
  on the 8-tile kernel, limited by block count, and r256 is opt-in
  (`GLCUDA_R256`) and off in production.

