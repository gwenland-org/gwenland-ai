# ARTX15 — Structured Generation

**Series:** gljax (Sanctum Visibilia) Architecture Research
**Status:** Draft — research-grounded
**Depends on:** ARTX13 (token surface forms), ARTX14 §3.3 (`MaskSource` seam — **binding**), ARTX11 §5 (variable acceptance), ARTX16 §1.4 (API surface)
**Introduces:** `gljax/src/grammar/`
**Next:** — end of the ARTX08–ARTX16 arc
**Research grounded:** 2026-07-27 (sources at end)

---

# 0. Why This Is Not Optional

Constrained decoding used to be a library bolted on outside the engine. **It is now an engine
feature**: XGrammar is the default backend in **vLLM, SGLang, TensorRT-LLM, and MLC-LLM**, and
LLGuidance ships in llama.cpp, SGLang, vLLM, mistral.rs, Chromium, and onnxruntime-genai.

The reason is structural. A grammar constrains generation by **masking disallowed tokens at every
step**, which requires being inside the sampling loop. Any implementation outside the engine must
either re-run the model or accept-and-retry — both far slower, and neither able to guarantee validity.

⚠️ ARTX14 §3.3 already built the seam (`MaskSource`) and, importantly, already **priced** it: a
bit-packed vocabulary mask is **18.5 KB per slot per step**, or **1.19 MB per iteration at 64 slots**
uploaded host→device — 4.6× the *downloaded* top-K traffic. ARTX14 explicitly deferred the mitigation
to this document. §3 delivers it.

---

# 1. Wave A15.1 — The Three Approaches

## 1.1 FSM / regex index — Outlines

JSON Schema → regular expression → finite state machine → **precompute, for every FSM state, which
vocabulary tokens are valid transitions**. At inference the current state indexes a precomputed table
in O(1), invalid tokens are masked, the FSM advances.

* ✅ Inference is a hash-map lookup — as cheap as it gets.
* ⚠️ A new schema costs **50–200 ms** of compile latency on first use, then amortizes.
* ⛔ **Regular expressions cannot express recursion, and JSON is inherently recursive.** A pure FSM
  must either flatten recursion to a fixed depth or reject recursive schemas outright.

That last point is disqualifying on its own for a general JSON-Schema surface: `{"$ref": "#"}` — a
tree, a linked list, a nested expression — is ordinary, not exotic.

## 1.2 Pushdown automaton — XGrammar

A **byte-level pushdown automaton** over context-free grammars, which handles both irregular token
boundaries and nested/recursive structure. Its two performance ideas are worth understanding
independently of whether gljax adopts the implementation:

⭐ **Context-independent vs context-dependent tokens.** A token is *context-independent* if its
validity is decided by the current PDA position alone; *context-dependent* if the whole stack is
needed. XGrammar precomputes an **adaptive token mask cache** over the context-independent set —
which covers **over 99% of tokens in most cases** — leaving ~1% to runtime stack inspection.

⭐ **Persistent execution stack.** Stacks at adjacent time steps share most of their deeper elements,
so they are merged rather than copied. This avoids memory redundancy **and enables state rollback**
— a property §4 turns out to need badly.

Reported: up to **100× speedup** over prior grammar-constrained methods, and **under 40 µs per token**
for JSON Schema and CFGs by precomputing the context-independent portion and overlapping the
context-dependent check with GPU execution.

## 1.3 Earley / Lark — LLGuidance

Computes token masks from context-free grammars written in a Lark variant, with a large subset of
JSON Schema and embedded regular expressions.

* **~50 µs of CPU time per token** for a 128k-token vocabulary, with **negligible startup cost**.
* On MaskBench (a deliberately hard JSON-schema benchmark) it computes a mask in ~50 µs on average,
  with *"other libraries being 10–1000× slower."*
* ⭐ **It is written in Rust** (`docs.rs/llguidance`).

## 1.4 ⭐ DESIGN DECISION — gljax uses **LLGuidance**

| Criterion | Outlines (FSM) | XGrammar (PDA) | **LLGuidance** |
|---|---|---|---|
| Recursive JSON Schema | ⛔ no | ✅ yes | ✅ yes |
| Per-token cost | O(1) lookup | < 40 µs | **~50 µs** |
| Startup per schema | 50–200 ms | low | **negligible** |
| Language | Python | **C++** | ⭐ **Rust** |
| Fits ARTX01's dependency posture | ✗ | ⛔ **C++ FFI** | ✅ **a cargo dependency** |

The decision is not close. ARTX01 committed gljax to pure Rust with no heavy foreign toolchains;
XGrammar would mean a C++ build dependency and an FFI surface, for a per-token cost LLGuidance
already matches. And unlike a GEMM kernel (ARTX08) or a tokenizer (ARTX13 §0.2), **grammar mask
computation is pure host-side CPU work over data structures** — there is no accelerator question,
no portability matrix, and no reason to write it twice.

⚠️ **DESIGN DECISION — LLGuidance sits behind ARTX14's `MaskSource` trait, not wired in directly.**
`gljax/src/grammar/` implements `MaskSource`; nothing in the sampling path names LLGuidance. If the
crate stalls or a better one appears, one file changes.

⚠️ **The 50 µs is per token per *sequence*.** At ARTX07's 64 slots that is **3.2 ms of CPU per
decode iteration** if computed serially — comparable to the iteration itself. §3.2 addresses this;
it is a real cost, not a rounding error.

---

# 2. Wave A15.2 — Grammar Sources

```text
JSON Schema  ─┐
Regex        ─┼─► Grammar ──compile──► GrammarId (cached)
Lark CFG     ─┘
Tool schema  ─┘   (§5 — a JSON Schema derived from the tool definition)
```

```rust
// gljax/src/grammar/mod.rs

pub enum GrammarSpec {
    JsonSchema(String),
    Regex(String),
    Lark(String),
    /// OpenAI-style `response_format: {"type": "json_object"}` — any valid JSON.
    JsonAny,
}

/// Compiled grammars are cached by content hash. ARTX16's workload profile
/// makes this decisive: agent traffic reuses ONE tool schema across every
/// request (ARTX09 §6.2 measured 75–95% prefix-cache hit rates on exactly
/// that shape), so the compile cost amortizes to ~zero.
pub struct GrammarCache { compiled: HashMap<GrammarHash, Arc<CompiledGrammar>> }
```

⚠️ **DESIGN DECISION — reject unsupported schema features at request admission, never mid-generation.**
LLGuidance supports *a large subset* of JSON Schema, not all of it. A request whose schema uses an
unsupported construct must fail with a 400 at ARTX16's admission step, with the offending keyword
named. Discovering it mid-generation would mean a half-emitted response that cannot be completed —
the worst possible failure mode for a structured request.

---

# 3. Wave A15.3 — The Mask Pipeline

## 3.1 Where each piece runs

```text
HOST (CPU)                                    DEVICE
──────────────────────────────────────────    ─────────────────────────
grammar state (per slot)
   │ LLGuidance: compute allow-mask  ~50 µs
   ▼
AllowMask (bit-packed, 18.5 KB)
   │ upload  ────────────────────────────►   apply −inf to logits
   │                                            │  (ARTX14 §3.3: BEFORE top-K)
   │                                            ▼
   │                                         penalties, bias
   │                                            ▼
   │                                         top-K reduction
   │         ◄──────────────────────────────  K values + indices (4 KB)
   ▼
host sampling (ARTX14 §2.2)
   │
   ▼
grammar.accept(token)  → advance state
```

⚠️ The mask **must** be applied before top-K, per ARTX14 D12: if truncation runs first, all K
survivors may be masked, leaving nothing to sample and forcing a fallback that silently violates the
grammar.

## 3.2 ⭐ The two costs ARTX14 deferred, and their mitigations

**Cost A — 1.19 MB/iteration uploaded at 64 slots.**

⚠️ **DESIGN DECISION — cache masks by grammar state and upload only on change.**

XGrammar's core observation transfers directly: **over 99% of tokens are context-independent**, so a
mask is very largely a function of the *grammar state*, not the full history. Many steps of a
structured generation sit in the same state — every character inside a string literal, every digit
inside a number.

```rust
pub struct MaskCache {
    /// grammar state fingerprint → the mask it produces.
    by_state: LruCache<StateHash, Arc<AllowMask>>,
    /// What each slot last uploaded. Skip the upload when unchanged.
    resident: Vec<Option<StateHash>>,
}
```

⚠️ The saving is workload-dependent and **must be measured, not assumed** — a schema that changes
state every token gets nothing. Report the skip rate; do not claim it.

**Cost B — 3.2 ms of CPU per iteration at 64 slots (§1.4).**

⚠️ **DESIGN DECISION — compute masks for the *next* step concurrently with the current device
execution.**

This is XGrammar's own technique: overlap the context-dependent check with GPU execution. ARTX07's
scheduler already has the structure for it — the device is busy inside `executor.execute()`, and mask
computation for slots whose token is already committed does not depend on that execution.

⛔ **But it collides with a locked decision.** ARTX07's Non-Goals list *"Async runtime"*, and ARTX16
D2 gave each replica a **single blocking OS thread**. Overlapping mask computation needs a worker
pool.

⚠️ **DESIGN DECISION — a bounded `rayon`-free worker pool owned by `grammar/`, not an async runtime.**
The work is pure CPU over owned data with no I/O — it needs threads, not futures. A fixed pool of
`min(n_slots, n_cpus−1)` threads, fed a slice of per-slot state, joined before the next
`schedule_decode()`. ARTX07's scheduler stays synchronous and single-threaded; it simply calls a
function that internally parallelizes. **This does not reopen the async decision.**

## 3.3 Jump-forward decoding

When the grammar admits exactly one continuation, the model does not need to be consulted at all.
After `{"name` in a schema with a required `name` field, the next characters must be `":` — there is
no choice to make.

```text
grammar state → only one valid token?
   yes → emit it directly, advance, repeat        ← ZERO model forward passes
   no  → normal sampling step
```

This is SGLang's compressed-FSM / jump-forward technique, and for schema-heavy output (JSON keys,
punctuation, fixed enum values) a large fraction of emitted tokens are forced.

⭐ **It composes with ARTX11 as a *free* drafter.** A forced token is a draft with **acceptance
probability 1** and **zero draft cost** (`c = 0`). Running ARTX11 §1.3's model at `α = 1`:

```text
speedup = (1 − α^(γ+1)) / ((1 − α)(γc + 1))  →  γ + 1   as α → 1, c → 0
```

Emitting `n` forced tokens in one verification pass is an `n+1`× speedup on that stretch, with no
draft model and no acceptance risk.

⚠️ **DESIGN DECISION — jump-forward tokens are appended to the KV cache without a forward pass, which
requires care.** The KV entries for forced tokens **do not exist** — nothing computed them. They must
be filled by a single batched forward pass over the forced run (treated exactly like an ARTX07
prefill chunk), not skipped. Skipping them would leave holes in the KV cache that later attention
reads as garbage. ⚠️ This is the highest-risk implementation detail in this document.

---

# 4. ⭐ Wave A15.4 — Rollback, and the ARTX11 Collision

## 4.1 The problem

ARTX11 §5.1: the draft proposes γ tokens, the target accepts `n ∈ [0, γ+1]`. **The grammar must
advance by `n`, not by γ.** But the drafter needed a mask for each of its γ steps, so the grammar
state was already advanced γ times.

```text
γ = 4 proposed, n = 2 accepted

  grammar advanced:  s0 → s1 → s2 → s3 → s4     (during drafting)
  correct state:     s2                          (only 2 accepted)
  → the grammar is TWO STATES AHEAD and now masks the wrong tokens
```

⚠️ Nothing errors. Generation continues against a wrong grammar position, producing output that is
fluent, plausible, and **not valid against the schema** — precisely ARTX12's bug class, arriving
through a new door.

## 4.2 The fix

⚠️ **DESIGN DECISION — the grammar state is checkpointed before drafting and rolled back to the
accepted prefix after verification.**

```rust
pub trait GrammarState {
    fn checkpoint(&self) -> StateToken;
    fn rollback(&mut self, to: StateToken);
    fn accept(&mut self, token: TokenId) -> Result<(), GrammarError>;
}

// In ARTX11's speculative_step:
let cp = grammar.checkpoint();                 // before drafting
// ... draft γ tokens, each advancing the grammar ...
// ... verify, accept n ...
grammar.rollback(cp);                          // undo ALL γ advances
for t in &result.tokens { grammar.accept(*t); }  // replay only the accepted
```

⭐ **This is exactly what XGrammar's persistent execution stack was built for** — merged stacks across
time points *"enable state rollback operations."* Whether LLGuidance exposes an equivalent primitive
is **the single most important integration question in this document**, and it is listed as open
(§6.1) rather than assumed. If it does not, the fallback is checkpoint-by-clone, whose cost scales
with grammar-stack depth and must be measured before ARTX11 and ARTX15 are enabled together.

⚠️ **DESIGN DECISION — until rollback is verified, structured generation and speculative decoding are
mutually exclusive at the policy layer.** ARTX16's config rejects enabling both. That is a temporary
restriction with a clear exit, and it is far better than a silent schema violation.

## 4.3 Interaction with jump-forward

⚠️ Jump-forward (§3.3) and speculation (ARTX11) both consume the "one token per step" assumption, and
they must not run simultaneously on the same slot. Jump-forward is strictly better where it applies
(α = 1, c = 0), so the policy is: **jump-forward first; speculate only where the grammar leaves a
genuine choice.**

---

# 5. Wave A15.5 — Tool Calling & Streaming

## 5.1 Tool calling is a JSON Schema problem

OpenAI-style tool calling gives each tool a name, description, and JSON-Schema parameters. Constrained
generation makes the emitted call **structurally valid by construction** — no parse-and-retry.

```text
tools[] → a union schema: {"name": <enum of tool names>, "arguments": <that tool's schema>}
       → one Grammar, compiled once, cached by hash (§2)
```

⭐ Agent workloads reuse one tool schema across every request, so the compile cost amortizes to
nothing — the same workload shape ARTX09 §6.2 found gives 75–95% prefix-cache hit rates.

## 5.2 ⛔ Streaming structured output is not streaming text

ARTX16 §1.5 streams token deltas over SSE. For a structured response this creates a genuine problem:

> **Partial JSON is not valid JSON.** A client cannot `JSON.parse` a prefix. Streaming
> `{"name": "get_wea` is unusable until the object closes.

⚠️ **DESIGN DECISION — stream raw text deltas as usual, and let the client decide.** gljax does not
attempt partial-JSON repair or emit synthetic closing braces.

The reasoning: repairing partial JSON means *guessing* the completion, and a guess that differs from
what the model actually produces makes the stream inconsistent with the final result. Clients that
want incremental structure use a streaming JSON parser; clients that do not simply wait. The engine's
job is to guarantee the *final* output is valid and to not lie about intermediate states.

⚠️ gljax does emit one useful signal the client cannot compute itself:

```json
{"choices":[{"delta":{"content":"..."},"x_grammar":{"complete":false,"depth":2}}]}
```

`complete: true` at the moment the grammar reaches an accepting state tells a client it may parse
*now* — earlier than end-of-stream when a stop token follows. ⚠️ Namespaced `x_` because it is not
part of the OpenAI schema (ARTX16 D28: wire types isolated, extensions clearly marked).

---

# 6. Module Layout + Wave Plan

```text
gljax/src/grammar/
├── mod.rs        GrammarSpec, GrammarCache, MaskSource impl
├── llg.rs        the LLGuidance binding — the ONLY file that names the crate
├── cache.rs      §3.2  MaskCache (by state hash), resident tracking
├── jump.rs       §3.3  forced-token detection + KV fill
└── rollback.rs   §4.2  checkpoint / rollback for ARTX11
```

| Wave | Scope | Gate |
|---|---|---|
| **A15.1** | `llg.rs` + `MaskSource` impl; JSON Schema + regex + Lark | ⭐ **100% schema validity** over ≥1000 generations on a schema suite incl. recursion; measured per-token mask cost within 2× of the ~50 µs reference |
| **A15.2** | `cache.rs` — state-keyed mask cache | Upload skip rate **reported**; correctness unchanged with the cache disabled |
| **A15.3** | Worker pool for overlap (§3.2) | Decode TPOT with grammar on ≤ 1.15× without; ARTX07's scheduler still synchronous |
| **A15.4** | `jump.rs` — jump-forward | ⛔ KV cache **bit-identical** to a run that forward-passed every forced token |
| **A15.5** | `rollback.rs` + ARTX11 integration | ⭐ Grammar state after N speculative rounds identical to N non-speculative rounds |
| **A15.6** | Tool calling + `x_grammar` streaming signal (glserve) | Emitted tool calls parse and validate against the tool schema, 100% |

⚠️ **A15.4's gate is stated as bit-identity deliberately.** Jump-forward's KV fill (§3.3) is the
place where a plausible-looking implementation leaves holes that surface thousands of tokens later as
degraded quality. Comparing the whole KV slab against a reference run is the only check that catches
it immediately.

## 6.1 Open questions

1. ⭐ **Does LLGuidance expose checkpoint/rollback?** (§4.2) The single most important integration
   question. If not: clone-based checkpointing, cost unmeasured, and ARTX11+ARTX15 stay mutually
   exclusive until it is.
2. **What is the real mask-cache skip rate?** (§3.2) Workload-dependent; measure per schema class.
3. **What fraction of tokens are jump-forward-forced** on realistic tool-calling schemas? Determines
   whether §3.3 is a major win or a minor one.
4. **Which JSON Schema keywords does LLGuidance reject?** (§2) Needed to write the admission-time
   validator that fails at 400 rather than mid-generation.
5. **Does the mask upload interact badly with ARTX10's quantized KV?** Both add host↔device traffic;
   their sum is unmeasured.

---

# Appendix A — Design Decision Summary

| # | Decision | Rationale | Reversible? |
|---|---|---|---|
| D1 | ⭐ **LLGuidance**, not XGrammar or Outlines | Rust (no C++ FFI, fits ARTX01); ~50 µs/token; negligible startup; recursion-capable. Outlines' FSM cannot express recursion at all | Medium |
| D2 | LLGuidance sits behind ARTX14's `MaskSource` trait | One file changes if the crate is replaced | Trivial |
| D3 | Grammars compiled once, cached by content hash | Agent traffic reuses one schema; compile amortizes to zero | Trivial |
| D4 | Unsupported schema features rejected at **admission** (400) | Mid-generation discovery leaves an uncompletable half-response | Trivial |
| D5 | Mask applied on device **before** top-K | Otherwise all K survivors may be masked (ARTX14 D12) | Hard |
| D6 | Mask cached by **grammar state**; upload skipped when unchanged | >99% of tokens are context-independent, so masks repeat across steps | Trivial |
| D7 | Skip rate is **reported**, not claimed | Workload-dependent; a state-churning schema gets nothing | Trivial |
| D8 | Mask computation overlapped via a **bounded thread pool**, not an async runtime | Pure CPU over owned data; ARTX07's sync scheduler is preserved | Medium |
| D9 | Jump-forward emits forced tokens without sampling | Acceptance 1, cost 0 — strictly better than speculation where it applies | Medium |
| D10 | ⛔ Forced tokens still get a **batched forward pass** to fill KV | Skipping leaves holes later attention reads as garbage | N/A |
| D11 | ⭐ Grammar state **checkpointed and rolled back** around speculation | Drafting advances the grammar γ times; only `n` are accepted | Hard |
| D12 | Until rollback is verified, grammar + speculation are **mutually exclusive** | A temporary restriction beats a silent schema violation | Trivial |
| D13 | Jump-forward takes precedence over speculation on a slot | α = 1, c = 0 dominates any drafter | Trivial |
| D14 | Streaming emits raw text deltas; **no partial-JSON repair** | Repair means guessing, and a wrong guess makes the stream inconsistent with the result | Medium |
| D15 | `x_grammar.complete` signal, namespaced | Lets a client parse at the accepting state; not part of the OpenAI schema | Trivial |

---

# Sources

- [XGrammar: Flexible and Efficient Structured Generation Engine for LLMs](https://arxiv.org/pdf/2411.15100) and [the MLC write-up](https://blog.mlc.ai/2024/11/22/achieving-efficient-flexible-portable-structured-generation-with-xgrammar) — byte-level pushdown automaton; context-independent vs context-dependent tokens; adaptive token mask cache covering >99% of tokens; persistent execution stack enabling rollback; up to 100× speedup; <40 µs/token with CPU–GPU overlap.
- [LLGuidance](https://github.com/guidance-ai/llguidance) and [docs.rs/llguidance](https://docs.rs/llguidance/latest/llguidance/) — Rust; JSON Schema, regex, and Lark CFGs; ~50 µs CPU per token on a 128k vocabulary; negligible startup; MaskBench ~50 µs with others 10–1000× slower; used in llama.cpp, vLLM, SGLang, mistral.rs, Chromium, onnxruntime-genai.
- [Structured Outputs and Constrained Decoding](https://www.tmls.nyc/research/structured-outputs-constrained-decoding) — the FSM index approach; 50–200 ms schema compile; ⛔ regex cannot express recursion while JSON is recursive.
- [Fast JSON Decoding for Local LLMs with Compressed Finite State Machine | LMSYS](https://www.lmsys.org/blog/2024-02-05-compressed-fsm/) — jump-forward decoding.
- [Guided Decoding Performance on vLLM and SGLang | SqueezeBits](https://blog.squeezebits.com/guided-decoding-performance-vllm-sglang) — XGrammar and LLGuidance as the two production backends.
- [Structured Outputs | vLLM](https://docs.vllm.ai/en/v0.8.2/features/structured_outputs.html) — XGrammar as the default backend across vLLM, SGLang, TensorRT-LLM, MLC-LLM.

**Repo-internal:** `ARTX07` (sync single-threaded scheduler, async Non-Goal, chunked prefill);
`ARTX09 §6.2` (75–95% agent-workload cache hit rates); `ARTX11 §1.3, §5.1` (speedup model, variable
acceptance); `ARTX12` (silent-wrong-output bug class); `ARTX13 §4` (incremental decode);
`ARTX14 §2.2, §3.3, D12` (top-K split, `MaskSource` seam, mask-before-truncation ordering, the
18.5 KB/slot/step price); `ARTX16 §1.5, D2, D28` (SSE, blocking worker thread, wire-type isolation).
