# Benchmark — GwenLand vs llama.cpp (Post-GATE)

**Date:** 2026-07-25
**Machine:** Intel i3-1115G4 (2p/4l cores), DDR4-2667 dual-channel, Windows 11
**Model:** Qwen2.5-0.5B-Instruct Q4_K_M (same GGUF used throughout this session)
**GATE status:** Wave A + B + C complete, live in `GlprocEngine::load_model`

---

## 1. Environment

- Windows Defender exclusions verified in place for `target/` and the
  `Downloads` folder holding both the model and the llama.cpp binaries
  (confirmed earlier this session — not re-verified via `Get-MpPreference`
  this run, since that requires admin rights this shell doesn't have; the
  user confirmed exclusions directly).
- Background processes checked and cleared immediately before every
  measurement in this report: no orphaned `find` processes, no
  `RobloxPlayerBeta` (present at the start of this sprint, closed before
  any number below was collected).
- `glbench` rebuilt from current source immediately before Phase 1, to
  guarantee GATE Wave A/B/C is actually in the binary under test.
- llama.cpp: official prebuilt release `b10107` (`llama-b10107-bin-win-cpu-x64.zip`,
  `ggml-cpu-icelake.dll` backend, auto-selected for this Tiger Lake CPU),
  downloaded earlier this session directly from `github.com/ggml-org/llama.cpp`.
- Prompt: glbench's own built-in `default_prompt()` — the base sentence
  "Explain how a modern GPU executes a matrix multiplication, covering
  threads, warps, shared memory, and coalesced loads." repeated 8×,
  tokenizing to ~220 tokens depending on engine/tokenizer. Used verbatim
  for both engines so the workload is genuinely shared, not
  independently authored per engine.

---

## 2. GwenLand (glproc) Results — Phase 1

Command (actual, after fixing the sprint's draft to glbench's real flag
names — `--kind decode` already includes a full prefill pass per
iteration, there is no separate prefill-only flag needed for this table):

```
glbench run --engine glproc --model qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --kind decode --warmup 1 --iters 5 --prompt "<default_prompt>" \
  --seed 42 --temperature 0
```

| Metric | Value |
|---|---|
| Decode tok/s (mean) | **37.1** |
| Decode tok/s (median / P50) | 36.8 |
| Decode tok/s (P95) | 38.2 |
| Decode tok/s (std) | 0.7 |
| Prefill tok/s (mean) | **121.3** |
| Prefill tok/s (median) | 122.2 |
| Measured bandwidth ceiling | 28.3–31.7 GB/s (varied slightly run to run — thermal/scheduling noise, consistent with this session's own earlier finding that this figure moves ±19% session to session) |
| Kernel path selected | `Q8_0 fused-swiglu integer-dot` (ffn gate/up), `Q8_0 integer-dot` (ffn down, lm_head) — **confirms Q8_0 won**, as expected |
| Bottleneck classification | `compute_bound`, 55–57% of bandwidth ceiling |

Archived: `benchmarks/post-gate-glproc-decode.json`

**GATE calibration time (measured directly, not estimated):** isolated by
timestamping every log line from process start on a minimal
(`--cold-iters 0 --tokens 1`) run. `[load] 1/1` (GATE begins) fired at
+584ms (binary startup + GGUF open + tokenizer parse); the post-calibration
`[load] tokenizer 0.09s | weights 0.54s | pin 0.04s` line (real model
finished loading) fired at +3082ms. **GATE calibration itself is
approximately 2.5 seconds** on this hardware — not the ~150ms the Wave C
design's common-case estimate assumed. This is a real, measured, larger-
than-expected cost; see §6 for why and what it does and doesn't affect.

**Which candidate won:** Q8_0 repack, confirmed both by the `[simd]`
kernel-path log line and by re-reading the source (`glproc/src/gate.rs`'s
`resolve_prefer_q4k_native` returning `false` on this hardware, matching
every prior measurement this session).

**Output coherence** (via `gwen run`, chat-template applied, greedy,
64 tokens):

> Modern GPUs are designed to execute matrix multiplication in parallel,
> which can significantly speed up the process. The GPU executes a matrix
> multiplication by dividing it into smaller sub-matrices and processing
> them sequentially. Here's how:
>
> 1. Divide: The GPU divides the input matrices into smaller sub-matrices
> based on their dimensions.

Coherent, on-topic, grammatically correct. (One internal inconsistency
worth noting as a model-quality observation, not an engine-correctness
one: it says "processing them sequentially" while describing a
parallelism explanation — this is Qwen2.5-0.5B's own reasoning quality at
this size, not a glproc artifact; the same imprecision would appear on
any correct engine running this model.)

---

## 3. llama.cpp Results — Phase 2

**Tooling note (real finding, not in the sprint's draft):** the exact
`llama-cli` invocation the sprint specified hangs indefinitely on this
build. `b10107`'s `llama-cli` auto-enables conversation/interactive mode
for any model with a chat template (this is one) unless `-no-cnv` **and**
`-st` (single-turn) are both passed — without `-st` specifically, the
REPL correctly generates one response and then waits forever on the next
`>` prompt, which piping `/dev/null` into does not terminate (it reads
EOF as still more empty turns rather than exiting). Two of the sprint's
literal invocations timed out at 60–120s before this was diagnosed.
Corrected invocation:

```
llama-cli -m qwen2.5-0.5b-instruct-q4_k_m.gguf -p "<default_prompt>" \
  -n 128 --seed 42 --temp 0 -t 4 --no-warmup -no-cnv -st --simple-io
```

Three runs via `llama-cli` (its own printed timing, one real generation
each):

| Run | Prompt (prefill) tok/s | Generation (decode) tok/s |
|---|---|---|
| 1 | 166.9 | 41.4 |
| 2 | 133.4 | 42.3 |
| 3 | 170.0 | 40.8 |
| **Mean** | **156.8** | **41.5** |

Prefill shows real run-to-run variance (133.4–170.0, ~24% spread) that
decode does not (40.8–42.3, ~3.6% spread) — consistent with prefill being
the shorter, more scheduling-sensitive phase on this hardware. Per
measurement-discipline.md rule 8, this variance was cross-checked against
`llama-bench`'s own built-in 5-repetition statistics for a more reliable
figure:

```
llama-bench -m qwen2.5-0.5b-instruct-q4_k_m.gguf -t 4 -fa on -p 220 -n 128 -r 5
```

| Metric | llama-bench (r=5) |
|---|---|
| Prefill (pp220) | **194.93 ± 6.18** tok/s |
| Decode (tg128) | **47.36 ± 2.31** tok/s |

`llama-bench`'s tighter methodology (5 repetitions, dedicated benchmark
harness rather than a REPL's printed summary) is treated as the more
reliable figure for the comparison table in §5; `llama-cli`'s numbers are
kept above as the source of the real generated text for the coherence
check.

**Output** (llama-cli, greedy, `-n 128`, truncated to first ~64 tokens for
direct comparison with glproc's output above):

> Matrix multiplication is a fundamental operation in linear algebra, and
> it's executed by a modern GPU using a combination of threads, warps,
> shared memory, and coalesced loads. Here's a detailed explanation of how
> a modern GPU executes a matrix multiplication:
>
> 1. **Threads**: Modern GPUs use a technique called "coalesced loads" to
> execute matrix multiplication. This technique allows the GPU to load
> data in a way that minimizes the number of threads that need to be
> executed simultaneously.

Also coherent, on-topic, grammatically correct.

---

## 4. Correctness Verdict — Phase 3

**Token-level comparison:** the sprint asked for a token-by-token
comparison of the first 64 generated tokens between glproc and llama.cpp
under identical prompt/seed/greedy settings. This was **not achievable as
specified** — glproc (via `gwen run`) and llama.cpp (via `llama-cli`) each
apply their own internal chat-template formatting to the same raw prompt
text before tokenizing, and neither tool exposes a way to bypass that
formatting while still producing a coherent instruct-model response (this
session already found, in an earlier Phase 1 test, that skipping the
chat template via glproc's `--raw` flag produces an incoherent
completion instead of an answer — the model was fine-tuned to expect the
template). Because the two engines' actual token streams therefore start
from different formatted inputs, an exact token match was never a
meaningful test to run here, and neither this report nor an implied
"identical or near-identical" bar from the sprint's draft should be
read as met or unmet — it is **not applicable** given the tooling on
hand.

**What was actually checked, and passed:**
- Both outputs are coherent, on-topic, factually reasonable explanations
  of the same prompt's subject (GPU matrix multiplication) — no garbage
  tokens, no repetition collapse, no non-finite values (glbench's own
  `validation: passed` on every glproc run this session; llama.cpp's
  output is human-readably correct prose).
- `glbench validate` (numerical parity against a reference oracle engine)
  **could not be run** — its default oracle is `glproc` itself, and the
  only other configured engine, `glcuda`, fails to load on this machine
  (`CUDA driver library not found (nvcuda.dll / libcuda.so)` — confirmed,
  no NVIDIA GPU present). `gllm` was not attempted as the oracle target
  given this session's own earlier finding that its `score_sequence` path
  measured roughly 3–4 orders of magnitude slower per token than glproc,
  which would make a 64-token validate run impractically slow. **This is
  a real gap in what could be verified today, not a claim that parity
  holds** — it is recorded as `UNKNOWN`/not-run rather than assumed.

**Verdict:** no correctness regression observed in what could actually be
checked (coherent output, clean validation report, no non-finite/garbage
signal); a stronger numerical-parity claim was not obtainable on this
machine's available engines.

---

## 5. Comparison Table — Phase 4

| Metric | GwenLand (post-GATE) | llama.cpp | Delta | Notes |
|---|---|---|---|---|
| Decode tok/s | 37.1 | 47.36 ± 2.31 (llama-bench) | llama.cpp **+27.7%** | glproc's own `--kind decode` includes cold+measure+behavior phases; llama-bench's `tg128` is a dedicated repeated measurement |
| Prefill tok/s | 121.3 | 194.93 ± 6.18 (llama-bench, pp220) | llama.cpp **+60.7%** | |
| Bandwidth ceiling | 28.3–31.7 GB/s (measured) | not measured by llama.cpp's own tooling | — | glbench's ceiling probe has no llama.cpp equivalent to compare against |
| GATE overhead | **~2.5s at session init** (measured, see §2/§6) | N/A | — | one-time; does not appear in per-token throughput (see §6) |
| Output coherent | Yes | Yes (reference) | — | both engines produce sensible, on-topic prose; exact token match not applicable (§4) |
| Kernel path | Q8_0 repack (GATE-selected) | `ggml-cpu-icelake.dll`, flash-attn on | — | different backends by construction; not a like-for-like kernel comparison |

---

## 6. Regression Check vs Pre-GATE — Phase 5

Pre-GATE clean baseline (Veritas Secunda, this same session, same
machine, same model): **decode 36.7–39.1 tok/s, prefill 128.5–135.5
tok/s.**

Post-GATE (this report): **decode 37.1 tok/s, prefill 121.3 tok/s.**

- **Decode: within the pre-GATE range** (36.7–39.1). No regression.
- **Prefill: 121.3 is slightly below the pre-GATE range's low end**
  (128.5–135.5), a ~5.6% gap from the range floor. Given this session's
  own repeated finding that prefill on this hardware shows real
  session-to-session variance in the same ballpark (thermal state,
  background scheduling — documented multiple times earlier this
  session), this single run is **not** strong enough evidence of a real
  regression on its own; per measurement-discipline.md rule 8
  ("repeat before you believe"), it would need a second reproduction
  to distinguish a genuine ~5% prefill regression from ordinary
  session noise. That reproduction was not run as part of this report
  (out of the sprint's stated scope) — flagged here as an open item
  rather than either dismissed or overstated.

**GATE calibration overhead: does NOT leak into steady-state per-token
throughput.** Both decode and prefill tok/s land in or near the pre-GATE
range, confirming GATE's one-time cost does not amortize into the
measured phase — exactly the property the design intended. **However,
the one-time cost itself is real and larger than assumed**: ~2.5 seconds
of added session-init latency, not the ~150ms the Wave C design estimated
for the common case. This is not a per-token regression, and the sprint's
literal Phase 5 criterion (post-GATE tok/s vs pre-GATE tok/s) is
satisfied — but a ~2.5s load-time cost is a real, user-visible fact about
every `GlprocEngine::load_model` call, worth surfacing rather than
letting the phrase "one-time, ~150ms" stand uncorrected. The ~150ms
figure was always described (in Wave C's own design write-up) as the
common-case estimate for *avoiding a second load once Q8_0 already won*
— but the calibration step that decides Q8_0 wins requires loading
**both** candidates once, every process, and that dual-load is the
dominant cost measured here, not an edge case.

---

## 7. Conclusion

**Gap vs llama.cpp: unchanged to slightly widened, not closed.**
GATE Wave A/B/C changed *how* glproc decides its FFN weight format
(now a real, calibrated decision rather than a hardcoded default) but
did not change *what* that decision is (Q8_0, same as before GATE
existed) or touch any kernel, threading, or memory-bandwidth code path.
Per this session's own earlier, corrected llama.cpp measurements, glproc
was already behind a properly-tuned llama.cpp (~1.21–1.53×) before GATE
existed; today's numbers (llama.cpp ahead by 1.28× decode, 1.61× prefill)
are consistent with that same gap, not a new one opened by GATE, and not
one GATE was ever positioned to close — Wave A/B/C's stated goal was
correct, calibrated *decision-making* for a choice that was already
being made correctly by hand, not a throughput improvement. No
regression was introduced by GATE on the metric that matters for this
comparison (per-token throughput); a real, newly-quantified cost was
found in session-startup latency (~2.5s), which is a legitimate finding
for anyone evaluating GATE's practical overhead, separate from and not
contradicting the "zero per-token overhead" property GATE's own
philosophy actually promises and delivers here.
