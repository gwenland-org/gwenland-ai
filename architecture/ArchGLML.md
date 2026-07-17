# ArchGLML — The GwenLand GL Engine: Architecture & Real Benchmarks

> Status: current as of 2026-07-07 (branch `feature/m15-bridge-simd-threading`,
> commit `6257088`). Every number in this document was measured, not estimated;
> the methodology is in [Benchmarks](#benchmarks).

GwenLand's GL stack is a **pure-Rust, zero-dependency CPU inference engine** for
GGUF transformer models. It is the source of truth all future GPU backends will
be validated against. On its 4-core laptop-class dev machine it loads a 0.5B
model in 0.9s, prefills faster than llama.cpp, and decodes at ~90% of llama.cpp
— with the remaining gap being DRAM bandwidth, not code.

---

## 1. Crate Map

```
glcore/    Formats + shared contracts. GGUF/safetensors parsers, the BPE
           tokenizer (SPM + byte-level, written from scratch), the GlEngine
           trait, the Runtime that owns tokenization at the boundary.
glproc/    The CPU engine. Loader (dequant/repack), model structs, the
           forward pass (runner), SIMD kernels, thread pool, KV cache,
           sampler. This is where all the speed lives.
glcli/     The `gwen` binary: run / info / tui subcommands, benchmark output.
glcuda/    Stubs. They compile everywhere and report "unavailable" so the
glvulkan/  runtime fallback chain is already wired for M2+ GPU work.
glmetal/
```

Dependency rule: `glproc` depends on `glcore`, never the reverse. The engine is
behind the object-safe `GlEngine` trait (`init / load_model / infer / stream /
shutdown / capabilities`), so `Runtime` holds `Box<dyn GlEngine>` and the CLI
does not know which backend it is talking to.

---

## 2. The Load Pipeline

```
GGUF file ──mmap──▶ parse metadata ──▶ per-tensor: copy / dequant / REPACK ──▶
owned buffers ──▶ warm + VirtualLock/mlock ──▶ ready
```

Decisions, and why:

**Weights are copied out of the mmap, not served from it.** The engine's hot
formats are *not* the file's formats — they are what decode wants. The mmap is
dead the moment loading finishes. This costs load time (mitigated below) and
buys the single most important property of the engine: every hot-path matvec
runs an integer kernel over a layout chosen for it.

**Everything quantized converges on Q8_0.** At load:

| File format | What happens | Why |
|---|---|---|
| Q8_0 | kept as-is | native integer-dot format |
| Q5_0 | repacked → Q8_0, **bit-exact** (`q8 = (q−16)·8`, `d/8`) | Q5_0's high-bit unpack measured compute-bound; +55% bytes is a net win |
| Q6_K | repacked → Q8_0, ≤0.4% requant error | same trade, accepted |
| Q4_K | repacked → Q8_0, ~0.4% requant error | no Q4_K integer kernel; unrepacked it fell into the f32 bridge, ~15× slower in batched prefill (see §7) |
| anything else | dequantized to f32 | correctness fallback |

**Fused weight streams.** Q, K and V are stacked into one matrix
(`QkvWeights::FusedQuant`) so a single pool dispatch computes the whole
projection — the K/V matrices are tiny under GQA and their dispatch overhead
was proportionally large. Gate and up projections are *row-interleaved*
(`[gate row 0][up row 0][gate row 1]…`, `GateUp::FusedQuant`) so the fused
SwiGLU matvec streams one contiguous region per thread instead of two streams
megabytes apart.

**The embedding table stays quantized.** `token_embd` is the biggest tensor
(vocab × dim). It is kept as Q8_0 and `GlprocModel::embed_into` dequantizes one
row per lookup (sub-microsecond). This avoids a ~4× f32 blow-up: −500 MB RAM on
a 150k-vocab 0.5B, −933 MB on the 1.5B, plus the dequantization time at load.
Tied-head models reuse the quantized table as the LM head — which is *better*
than f32 there, because the head then rides the integer-dot path.

**Load is parallel.** Layers are independent copy/dequant/repack work; workers
pull layer indices from a shared atomic counter. Weights phase: 1.41s → 0.7s on
4 cores.

**Warm + pin.** After load, every weight page is touched and pinned
(`VirtualLock` after `SetProcessWorkingSetSize`, `mlock` on Unix) so no decode
step ever takes a page fault mid-matvec. `GLPROC_NO_LOCK=1` disables it for
A/B runs (measured neutral on an unloaded box; it exists for memory-pressure
protection).

Every load prints its breakdown: `[load] tokenizer 0.08s | weights 0.72s | pin 0.07s`.

---

## 3. The Compute Pipeline: Integer-Domain Dots

The f32 bridge (dequantize block → dot in f32) is correct but dequant
instructions dominate on narrow formats. The engine instead quantizes the
**activation** vector to int8 once per matvec (one scale per 32-element group,
the same scheme llama.cpp uses) and keeps the inner loop in the integer domain:

```
per 34-byte Q8_0 block:  w = load 32×i8           (weights)
                         a = load 32×i8           (activation)
                         p = vpdpbusd(|w|, a·sign(w))   ← 32 MACs, 1 instruction
                         acc += f16(scale_w)·scale_a · cvt(p)
```

- On Tiger Lake+, the 256-bit EVEX **VNNI** form (`vpdpbusd` on ymm) replaces
  the AVX2 `maddubs`+`madd` pair. It is encoding-wise AVX512VL+VNNI but runs at
  the AVX2 frequency license — no 512-bit datapath, so the thermal ban on
  AVX-512F does not apply.
- Block scales convert through F16C `vcvtph2ps` (1 instruction; the software
  f16 path was a branchy ~15-op routine running millions of times per token).
- Accuracy: ~1e-3 relative per dot, well under the weights' own quantization
  noise. Every SIMD kernel has a scalar ground truth and a parity test.

`SimdStrategy::detect()` probes once (OnceLock): Scalar / Avx2 / Avx512, with
AVX-512 only selected on >8-logical-core parts (mobile TDP throttling makes
4-thread AVX2 faster than throttled AVX-512 on this class of machine). The
`[simd]` startup line names the strategy and each hot weight class's kernel
path so a scalar fallback can never hide — with one caveat learned the hard
way: it samples layer 0, and *formats can differ per layer* (see §7).

---

## 4. Threading Model

A persistent pool (`ThreadPool`), spawned once; the calling thread participates
as thread 0. A decode step dispatches ~100 jobs back-to-back, so:

- Workers **spin ~2^14 iterations** on an atomic generation counter before
  parking on a condvar — the kernel scheduler (5–50µs per wake) stays out of
  the hot path between dispatches.
- Each matvec splits its **rows into contiguous chunks** per thread, not
  interleaved rows. Row cost is uniform so both balance, but weights stream
  from DRAM every token, and DDR4 rewards a few clean sequential streams:
  chunked beat interleaved by ~35% end-to-end (18.6 → 25.3 tok/s at the time).
- Layer-level parallelism is impossible (layer N+1 consumes layer N's output);
  the parallel axis is always rows-within-a-matvec, which have no dependencies.
- Below 2^16 multiply-accumulates a matvec runs on the calling thread — waking
  workers costs more than the work.
- 4 threads beats 2/3 on the 2-core/4-thread i3 (SMT helps the integer dot);
  `GLPROC_THREADS` overrides for benchmarking and as a thermal knob.

---

## 5. Decode: One Token, Zero Allocation

`Runner::step` is the per-token forward pass. Hot-path rules, enforced:

- **Zero heap allocation per token** — every buffer lives in the pre-allocated
  `Workspace` (residual, norms, fused QKV output, attention scratch, logits,
  and the Q8 activation buffer).
- **No dyn dispatch** — the SIMD backend is a `match` on a cached enum.
- Activation quantization is hoisted: one quantize per distinct vector even
  when several matrices consume it (Q, K, V share one; gate/up share one).
- Attention is single-query against the KV cache per head, SIMD score dots,
  causality implicit (the cache only holds the past).
- Prefill positions skip the LM head — it is the single biggest matvec (full
  vocabulary) and only the last prompt token needs logits.
- Sampling: temperature → top-k via `select_nth_unstable` (a full 152k-vocab
  sort cost 6.9ms/token — more than an FFN layer; partial selection is 1.0ms)
  → softmax → top-p → seeded xorshift64* draw. Repetition penalty is applied
  to the logits before sampling, over the last 64 generated tokens.

The KV cache is cursor-based: one flat pre-allocated buffer, layout
`[layer][k/v][head][seq][dim]` so attention's sequential scan over `seq` is one
linear sweep. Reset is `cursor = 0` — no free, no zeroing. Capacity is
`min(model max_seq, 4096)` to avoid GB-scale preallocation (~200 MB at 0.5B
dims). Batched prefill adds positional writes (`write_k_at`) and an
`advance_by(n)`; decode's one-position cursor semantics are unchanged.

Per-phase wall time is measurable at any time with `GLPROC_PROFILE=1`.

---

## 6. Prefill: Batched Chunks

Sequential prefill runs one matvec forward pass per prompt token — every weight
byte streams from DRAM once per token. Batched prefill (`step_chunk`) runs the
prompt in **32-token chunks**:

- `par_matmul_qdot` / `par_matmul_swiglu`: threads own contiguous weight-row
  ranges, and each row is dotted with **every** batch activation while cache
  hot. Weights stream once per chunk instead of once per token — prefill flips
  from bandwidth-bound to compute-bound.
- `row_dot_xn<G>` kernels (VNNI and AVX2) dot one weight row against G=8 (or 4,
  or 2) activations per call: the block load, sign preparation and f16 scale
  conversion are paid once per group, and the G independent accumulator chains
  hide FMA/`vpdpbusd` latency that a single-activation dot serializes on.
- For long rows, activations are packed into **block-interleaved panels**
  (`row_dot_packed8`: quants `[block][act][32]`, scales `[block][act]`) so the
  inner loop reads one sequential stream instead of up to 17 (8 quant + 8
  scale buffers + weights). Pack scratch is thread-local and reused — a fresh
  allocation per call took demand-zero page-fault stalls under memory pressure.
- Chunk attention runs **positions in parallel** across the pool, one score row
  per position. Attention cost grows with cached length; single-threaded it had
  dominated long prompts (a 3.8k-token prompt ran at 16 tok/s).
- Causality: all of a chunk's K/V rows are written before any of its positions
  runs attention, and position `p` only attends to rows `0..=p`.
- `PREFILL_CHUNK = 64` was measured **worse** (the activation set outgrows L2);
  32 is the sweet spot on this machine.

Parity is test-enforced: chunked prefill must match sequential logits and KV
state on single-chunk, multi-chunk, and ragged-tail prompts.

---

## 7. The Q4_K Lesson (why per-layer tracing matters)

After batching, prefill stalled at ~60 tok/s with the down projection eating
65% of the time. The kernel was innocent — an isolated microbenchmark
(`bench_matmul_shapes`) ran the same shape at 73 GMAC/s. Group width, spin
budget, thread count, activation packing, and VirtualLock were all A/B'd
neutral. A per-layer trace finally showed a **layer-stable bimodal split**:
1.6ms on some layers, 25–40ms on others.

The slow layers' `ffn_down` tensors are **Q4_K** in this GGUF. Only `ffn_down`
can be: its rows are 4864 wide (divisible by the 256-element Q4_K superblock),
while every other matrix is 896 wide and cannot hold Q4_K. Q4_K had no integer
kernel, so those layers fell into the f32 bridge — which re-dequantizes every
weight block once per batch row: 32× the dequant work in a 32-token chunk.

Repacking Q4_K → Q8_0 at load removed the entire anomaly (down: 11.5ms →
1.6ms per prefill token) and, as a side effect, took the 1.5B's *decode* from
5.3 to 12.1 tok/s — its Q4_K layers had been bridge-bound in decode too.

Morals, encoded in the codebase: quantization formats vary per layer inside one
GGUF file; a startup log that samples layer 0 proves nothing about layer 2; and
when a kernel is fast in isolation but slow in the engine, trace per call site
before touching the kernel.

---

## 8. Correctness Layer (post-audit)

- **Stop tokens**: the tokenizer resolves a stop set at load — metadata EOS
  plus every known stop marker present in the vocab (`<|im_end|>`,
  `<|endoftext|>`, `<|end|>`, `<eos>`, `<end_of_turn>`, `</s>`, `<|eot_id|>`,
  `<|end_of_text|>`). `Runner::generate` takes a stop predicate; stop tokens
  are never emitted.
- **Repetition penalty**: last-64-token sliding window, `/penalty` on positive
  logits, `×penalty` on negative, per occurrence. Default 1.1
  (`--repeat-penalty`, 1.0 disables). Note: when this landed, measured tok/s
  *dropped* 23.3 → 15.2 — looping output had been keeping the same weight rows
  hot in L3 and inflating every earlier benchmark. Diverse decode is the only
  honest benchmark.
- **ChatML templating**: chat models get
  `<|im_start|>system…user…assistant` wrapping automatically, with the markers
  emitted as special-token ids (plain BPE encoding would shred them into text
  pieces). `--raw` opts out for base models. Without the template, instruct
  models never enter chat mode and rarely emit a stop token at all.
- **Split metrics**: prefill and generation tok/s are reported separately.
  A blended number understates decode on short generations and hid the
  looping artifact entirely.

---

## 9. Benchmarks

### Hardware & Method

| | |
|---|---|
| CPU | Intel Core i3-1115G4 (Tiger Lake), 2C/4T, AVX2+VNNI-256, 15W AIO |
| RAM | 2×4 GB DDR4-2667, dual channel (WMI-verified), ~25–28 GB/s practical |
| OS | Windows 11 Home 25H2 |
| Model | Qwen2.5-0.5B-Instruct Q4_K_M (463 MB GGUF) unless stated |
| Reference | llama.cpp b9888 win-cpu-x64, same machine, via `llama-bench` |

Method: Windows Defender exclusions on the workspace and model folder; CPU load
verified <15% before every run (`Get-CimInstance Win32_Processor`, `-NoProfile`);
3 runs per configuration; prompt for prefill = 133 tokens raw, prompt for decode
= 24-token ChatML chat prompt, 128 generated tokens, temperature 0.8 / top-k 40 /
top-p 0.95 / repeat-penalty 1.1 (real sampling, not greedy).

### Headline (2026-07-07)

| Metric | GwenLand GL | llama.cpp b9888 | GL vs reference |
|---|---|---|---|
| Prefill, 133-token prompt | **128–132 tok/s** | 124.5 (pp24) | **≥ 1.0×** |
| Prefill, 24-token chat prompt | 116–148 tok/s | — | — |
| Generation, 128 tokens | **33.5–35.2 tok/s** | 39.0 (tg128) | ~0.9× |
| Model load (warm disk) | **0.9s** | ~1–2s | ~2× faster |
| Peak working set | **1.19 GB** | — | — |
| Workspace tests | 459 green | — | — |

Qwen2.5-1.5B-Instruct Q4_K_M on the same box: generation 12.1 tok/s, coherent
output, clean EOS stops. Its load (~9.5s) is RAM-pressure-bound — a repacked
1.5B plus a browser and IDE do not fit comfortably in 8 GB; that is memory
economics, not engine code.

### Decode profile (per token, clean run, `GLPROC_PROFILE=1`)

```
qkv 2.4ms | attn 2.0ms | wo 1.6ms | gateup 9.5ms | down ~5ms | lm_head 5.5ms | sampler 1.0ms
≈ 27–30ms/token → 33–36 tok/s
```

Gate/up streams its weights at ~23.4 GB/s and the LM head at 26–28 GB/s —
essentially this machine's practical DRAM ceiling. **Decode is memory-bandwidth
bound.** Total weight traffic is ~520 MB/token; at ~27 GB/s the hard ceiling is
~48–50 tok/s, and llama.cpp's 39 sits in the same regime. This is why decode
optimization stopped here: the remaining delta to llama.cpp is small-dispatch
overhead in `wo`/`down` (~12–15 GB/s effective), not kernel quality.

### Prefill profile (per prompt token, 133-token prompt)

```
serial 0.20ms | qkv 0.5ms | fixup 0.42ms | attn 0.7ms | wo 0.5ms | gateup 3.9ms | downq 0.5ms | down 1.6ms
≈ 8.3ms/token → ~120 tok/s
```

Gate/up is now the largest bucket — its fused-SwiGLU path still uses unpacked
grouped dots. That is the known next lever if prefill ever needs to go higher.

### The honest progression (all on this machine, same 0.5B file)

| Stage | Generation | Prefill | Notes |
|---|---|---|---|
| M1 naive f32 | 1.83 tok/s | — | scalar, f32 everything |
| M1.5 bridge + threading | 12.2 tok/s | — | Q-blocks + pool |
| M1.5-X5 wave | ~33 tok/s* | ~35 tok/s | *inflated by looping output |
| Post-audit fixes | 33–36 tok/s (honest) | 35–43 | stop tokens + rep penalty + split metrics |
| M1.6 batched prefill | 33–36 | 57–63 | chunked matmul, grouped dots |
| M1.6 Q4_K repack | 33.5–39.6 | 107–121 | the bimodal-layer fix |
| M1.7 quantized embd + parallel load | 33.5–35.2 | **128–132** | load 2.5s → 0.9s, RAM −0.5 GB |

Two systematic errors polluted early numbers and are worth remembering:
(1) looping output inflates tok/s (repeated tokens keep weight rows hot in L3);
(2) Windows Defender rescans the binary and model after every build and
collapses runs 2–4× — several "optimizations" were accepted or rejected on
noise before this was caught.

### Rejected optimizations (measured, do not revisit blindly)

| Idea | Result |
|---|---|
| L2 tiling for decode matvecs | GEMV has zero weight reuse; tiling helps GEMM only |
| Interleaved rows across threads | −35% vs contiguous chunks on DDR4 |
| 512-bit AVX-512F kernels | frequency throttle beats the width gain at 15W |
| SPIN_ITERS 2^17 | neutral — dispatch overhead was not parking |
| PREFILL_CHUNK = 64 | worse — activation set outgrows L2 |
| VNNI for Q8_0 decode dots | neutral (bandwidth-bound), kept for bit-identical results |
| Raw-mmap lazy layer paging | incompatible with the Q8_0 repack that bought parity |
| Lazy layer-build chase at load | first chunk needs all layers; ~0.2s max win at 0.9s load |

---

## 10. Known Limits & Next Levers

- **Decode ceiling**: ~48–50 tok/s theoretical at this DRAM bandwidth and
  weight traffic. Getting closer means fixing the small-dispatch efficiency of
  `wo`/`down` (~12–15 GB/s vs 23+ for large dispatches) — cause unresolved,
  measured not to be worker parking.
- **Prefill**: gate/up's fused-SwiGLU path is the top bucket and still unpacked;
  a packed-panel or register-tiled treatment there is the next win.
- **Activation quantization** is scalar (~2% of decode); vectorizable.
- **Big-model loads** under RAM pressure would benefit from a streaming load
  that releases mmap ranges per built layer.
- **gwen tui** still speaks to the legacy backend; rewiring onto `Runtime` is
  M2 scope, as is safetensors inference (needs a config.json sidecar).
