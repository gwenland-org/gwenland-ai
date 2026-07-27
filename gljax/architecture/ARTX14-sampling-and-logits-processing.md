# ARTX14 — Sampling & Logits Processing

**Series:** gljax (Sanctum Visibilia) Architecture Research
**Status:** Draft — research-grounded
**Depends on:** ARTX05 §Session::generate (current argmax path), ARTX08 (static shapes), ARTX11 §5.2 (rejection sampling), ARTX13 (token IDs), ARTX16 §1.3 (`SamplingParams`)
**Introduces:** `gljax/src/sample/`
**Next:** [ARTX15 — Structured Generation](ARTX15-structured-generation.md)
**Research grounded:** 2026-07-27 (sources at end)

---

# 0. ⛔ Three Documents, Three Incompatible Assumptions

Sampling has no specification anywhere in ARTX01–ARTX13, yet three documents depend on it and they
do not agree:

| Document | What it assumes | Evidence |
|---|---|---|
| **ARTX05** | **Greedy argmax only, host-side.** Its design-decision table reads: *"Argmax \| Host-side after `to_host()` \| Sampling logic stays in Rust; avoids compiled sampler complexity"* | `argmax_f32(last_pos_logits)` in `Session::generate` |
| **ARTX16** | Full `SamplingParams { temperature, top_p, top_k, stop, seed }` on every request | §1.3 `InferenceReq` |
| **ARTX11** | **Distribution sampling is mandatory.** Its losslessness proof needs candidates *sampled from* `q`, and `β = min(1, p/q)` divides by `q(y)` — which is only nonzero because `y` was sampled, not argmaxed | §1.2, §5.3 |

⛔ **ARTX11's guarantee is void under ARTX05's implementation.** Greedy drafting makes `q` a point
mass; the rejection-sampling correction has nothing to correct against, and the "lossless" claim
becomes false without any error being raised.

ARTX14 resolves this. It also prices something ARTX05's decision did not.

## 0.1 ⚠️ What host-side argmax actually costs

```text
logits transferred device→host per token per slot
  = vocab_size × 4 bytes (f32)
  = 151,936 × 4
  = 608 KB

× ARTX07's max_slots = 8   →   4.86 MB per decode iteration
× a production 64 slots     →   38.9 MB per decode iteration
```

⚠️ On TPU v5e that transfer competes with the ~201 MB/iteration KV read ARTX09 §4.3 measured, over a
much slower link than HBM. ARTX05's decision to keep sampling in Rust was reasonable in isolation —
it was made before ARTX07 introduced 64 concurrent slots. **At batch scale it is the dominant
host↔device traffic in the loop.**

---

# 1. Wave A14.1 — The Sampler Chain

## 1.1 ⛔ The two references disagree, and the disagreement is observable

| | Order |
|---|---|
| **vLLM** | penalties (repetition, frequency, presence) → **temperature** → logit processors (min-p) → top-k / top-p → sample |
| **llama.cpp** | penalties → dry → top_n_sigma → top-k → typ-p → top-p → min-p → xtc → **temperature** → sample |

**Temperature sits on opposite sides of truncation.** This is not cosmetic:

```text
Temperature BEFORE top-p (vLLM):
    T flattens the distribution → the cumulative-p threshold admits MORE tokens
    → temperature indirectly widens the candidate set

Temperature AFTER top-p (llama.cpp):
    the candidate set is fixed by the untempered distribution
    → temperature only reshapes weights WITHIN that fixed set
```

At `T > 1` the two produce materially different candidate pools from identical logits.

⚠️ Both projects agree on the part that has a *reason*: **penalties run first.** Applying penalties
before truncation ensures penalized tokens are deprioritized before probability mass is
redistributed; applying them after would penalize tokens already eliminated, having no effect.

⚠️ And llama.cpp reports the ordering can be **model-specific**: naively adding repetition penalties
to QwQ-32B *caused* looping, and reordering the samplers fixed it. It exposes `--samplers` for
exactly this reason.

## 1.2 ⚠️ DESIGN DECISION — the chain is explicit, ordered data, and its order is recorded per request

gljax does not get to pick "the" order, because there is no consensus order to pick. It must make the
order **visible and reproducible** instead.

```rust
// gljax/src/sample/chain.rs

/// An ordered pipeline. Order is DATA, not code — it is serialized into
/// telemetry so a generation can be explained and reproduced.
#[derive(Debug, Clone, PartialEq)]
pub struct SamplerChain {
    pub stages: Vec<Stage>,
    pub seed: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Stage {
    /// Applied to raw logits. Must precede any truncation (§1.1).
    RepetitionPenalty { penalty: f32, window: usize },
    PresencePenalty   { penalty: f32 },
    FrequencyPenalty  { penalty: f32 },
    /// Per-token additive bias (OpenAI `logit_bias`).
    LogitBias,
    /// ⭐ Mask from ARTX15. MUST run before any truncation — see §3.3.
    GrammarMask,
    Temperature { t: f32 },
    TopK  { k: usize },
    TopP  { p: f32 },
    /// Keep tokens with P(token) ≥ P(max) × min_p.
    MinP  { p: f32 },
    Typical { mass: f32 },
}

impl SamplerChain {
    /// vLLM-compatible ordering. gljax's DEFAULT, because ARTX16 serves an
    /// OpenAI-compatible API and vLLM is the de-facto reference for that surface.
    pub fn openai_default(p: &SamplingParams) -> Self;
    /// llama.cpp-compatible ordering, for cross-checking against that engine.
    pub fn llamacpp_default(p: &SamplingParams) -> Self;

    /// ⚠️ Rejects chains that are silently wrong rather than reordering them.
    pub fn validate(&self) -> Result<(), ChainError>;
}
```

⚠️ **DESIGN DECISION — `validate()` refuses invalid orders; it never silently reorders.**
Two rules are enforced, both from §1.1: **penalties must precede all truncation stages**, and
**`GrammarMask` must precede all truncation stages** (§3.3). A chain violating either is a
configuration error, not something to quietly fix — silently reordering would make the recorded
order a lie.

⚠️ **DESIGN DECISION — greedy is `TopK { k: 1 }`, not a separate code path.**
ARTX05's argmax becomes one configuration of the chain rather than a parallel implementation. This
removes the ARTX05/ARTX16 contradiction structurally: there is one sampler, and greedy is a setting.

---

# 2. Wave A14.2 — Device vs Host: the split

## 2.1 The three options

```text
(A) FULL HOST    to_host(logits[V]) → sample in Rust
                 ✅ trivial, any sampler, exact
                 ❌ 608 KB/token/slot (§0.1)

(B) FULL DEVICE  express the whole chain in StableHLO
                 ✅ no logits transfer
                 ❌ top-p needs a SORT over V=151,936 — vLLM notes sorting
                    "can be slow for large batches"; ARTX08 forbids a custom
                    sorting-free kernel (FlashInfer's approach)

(C) SPLIT        device: mask → penalties → top-k reduction to K
                 transfer: K values + K indices
                 host: top-p / min-p / typical / temperature / sample over K
```

## 2.2 ⭐ DESIGN DECISION — option (C), and the numbers are decisive

```text
Transfer per token per slot:
  (A) full logits          151,936 × 4 B                     = 608 KB
  (C) top-K, K = 512       512 × (4 B value + 4 B index)     = 4 KB

  → 148× less traffic, and it does not grow with vocab size
```

At 64 slots that is **38.9 MB → 256 KB per decode iteration.**

**Why (C) and not (B):** top-k is a bounded, static-shape reduction — `stablehlo.top_k` or
sort-then-slice over a compile-time-constant `K` — which ARTX05/ARTX07's bucketing already
accommodates. Top-p is not: it needs a cumulative sum over a *sorted* full vocabulary, and the
sorting-free alternative is a custom kernel (FlashInfer's rejection-sampling sampler), which ARTX08
rules out. Splitting at top-k puts the expensive-on-device part on device and the
awkward-on-device part on host, where 512 elements make it trivial.

**Why K is generous:** top-p, min-p, and typical sampling all select from the head of the
distribution. With `K = 512` the truncation set is contained in the top-K with overwhelming
probability for any realistic `p`.

⚠️ **But it is an approximation, and the doc must say so.** If a request's top-p would admit more than
`K` tokens — a very flat distribution, e.g. high temperature on a low-confidence step — the result
differs from exact top-p over the full vocabulary. Two consequences:

* `K` is a **compile-cache key dimension** (like ARTX05's buckets and ARTX11's γ), not a free
  runtime parameter.
* The sampler **reports truncation saturation** (`selected == K`) so the approximation is observable
  rather than silent. ⚠️ Persistent saturation means `K` is too small for the workload.

⚠️ **DESIGN DECISION — exact full-vocabulary sampling remains reachable via option (A), behind a
flag.** ARTX12's correctness harness needs an exact reference to validate (C) against, and some
deployments will prefer exactness over throughput. (A) is the oracle; (C) is production.

## 2.3 The interaction with ARTX11

⚠️ Speculative decoding does **not** fit the top-K path, and this is easy to miss.

ARTX11 §5.3 needs `p(y)` and `q(y)` for a *specific* candidate token `y` — the draft's proposal.
**That token may not be in the target's top-K at all**; indeed when it is not, that is precisely the
case rejection sampling exists to handle.

```rust
// The verify path gathers specific token probabilities on device,
// rather than reducing to top-K.
//
//   p_gathered = stablehlo.gather(softmax(logits), indices = candidate_ids)
//
// Transfer: γ+1 scalars per slot, not K values. Even cheaper than (C).
pub fn gather_candidate_probs(logits: &Tensor, candidates: &[TokenId]) -> Tensor;
```

⚠️ So gljax has **two** device-side reduction paths — top-K for ordinary decode, gather for
speculative verify — and they are selected by whether a drafter is attached. This must be stated in
ARTX11's scheduler, not discovered when acceptance rates come out wrong.

---

# 3. Wave A14.3 — Logit Processors

## 3.1 Penalties need generation history

```rust
// gljax/src/sample/penalty.rs

/// Per-slot token history. Lives beside ARTX07's slot state, since it has
/// exactly the same lifecycle: allocated with the slot, freed with it.
pub struct PenaltyState {
    /// Occurrence counts over the penalty window. Vocab-sized but SPARSE in
    /// practice — a generation touches a few hundred distinct tokens.
    counts: HashMap<TokenId, u32>,
    order: VecDeque<TokenId>,     // for windowed repetition penalty
    window: usize,
}
```

⚠️ **DESIGN DECISION — penalties are applied on device, before the top-K reduction.**
A penalty applied on the host after top-K cannot demote a token *out* of the candidate set, which is
most of what a penalty is for. The counts are uploaded as a sparse `(indices, values)` pair per slot
per step — a few hundred entries, kilobytes, not the 608 KB of §0.1.

⚠️ Three distinct penalties with distinct semantics, easily conflated:

| Penalty | Applied to | Formula |
|---|---|---|
| **Repetition** | tokens in a trailing window | divide/multiply the logit by `penalty` |
| **Presence** | any token seen at least once | subtract a constant |
| **Frequency** | scaled by occurrence count | subtract `count × penalty` |

## 3.2 Logit bias

OpenAI's `logit_bias` is a sparse `{token_id: bias}` map, additive, applied to raw logits. Same
upload mechanism as penalties. ⚠️ Bounded per request (OpenAI caps it at 300 entries) — gljax should
cap it too, or a single request can turn the sparse upload dense.

## 3.3 ⭐ The grammar mask interface (the seam to ARTX15)

ARTX15 constrains generation to a grammar by masking disallowed tokens each step. ARTX14 owns the
*interface*; ARTX15 owns the grammar.

⚠️ **DESIGN DECISION — the mask is applied on device, before top-K, as a dynamic input tensor.**

The ordering is forced, not chosen: if top-K runs first, the K survivors may be *entirely* masked,
leaving nothing to sample and requiring a fallback that would silently violate the grammar. The mask
must shrink the candidate set before it is truncated.

```rust
// gljax/src/sample/mask.rs

/// A per-slot, per-step allow-mask over the vocabulary.
/// Bit-packed: 151,936 bits = 18.5 KB per slot per step.
pub struct AllowMask { bits: Vec<u64> }   // ceil(V / 64) words

/// Produced by ARTX15, consumed here. ARTX14 knows nothing about grammars.
pub trait MaskSource: Send {
    fn mask_for(&mut self, slot: SlotId, history: &[TokenId], out: &mut AllowMask);
    /// Advance the grammar state after a token is committed.
    fn accept(&mut self, slot: SlotId, token: TokenId) -> Result<(), MaskError>;
}
```

⚠️ **The transfer cost runs the wrong way and must be priced now, not in ARTX15.**
Bit-packed, the mask is **18.5 KB per slot per step host→device** — at 64 slots, **1.19 MB per
iteration** uploaded. That is 4.6× the *downloaded* top-K traffic of §2.2. Constrained generation is
therefore not free at batch scale, and the mitigation (only uploading when the grammar state
*changes*, since many grammar states permit the same token set) belongs in ARTX15's design.

⚠️ **DESIGN DECISION — `MaskSource` is a trait so ARTX14 never depends on ARTX15.** Sampling compiles
and ships without any grammar support; structured generation plugs in.

---

# 4. Pseudocode

```rust
// gljax/src/sample/mod.rs

pub fn sample_step(
    logits: &Tensor,              // device [slots, V]
    chains: &[SamplerChain],      // one per slot
    state: &mut [PenaltyState],
    mask: Option<&mut dyn MaskSource>,
    k: usize,                     // compile-time constant (§2.2)
) -> Vec<TokenId> {
    // ── DEVICE ──────────────────────────────────────────────────────────
    // Order is forced: mask and penalties must precede truncation (§1.2, §3.3).
    let mut l = logits.clone();
    if let Some(m) = mask {
        l = ops::apply_allow_mask(&l, &m.upload());          // −inf on disallowed
    }
    l = ops::apply_penalties_sparse(&l, state);
    l = ops::apply_logit_bias_sparse(&l, chains);

    // Bounded, static-shape reduction. THE transfer boundary.
    let (top_vals, top_idx) = ops::top_k(&l, k);             // [slots, K] ×2

    // ── TRANSFER ────────────────────────────────────────────────────────
    let vals = top_vals.to_host();                           // 4 KB/slot at K=512
    let idx  = top_idx.to_host();

    // ── HOST ────────────────────────────────────────────────────────────
    // K = 512 makes every remaining stage trivial and exactly expressible.
    let mut out = Vec::with_capacity(chains.len());
    for (s, chain) in chains.iter().enumerate() {
        let mut cand = Candidates::new(&vals[s], &idx[s]);

        for stage in &chain.stages {
            match stage {
                Stage::Temperature { t } => cand.scale(*t),
                Stage::TopK { k }        => cand.truncate(*k),
                Stage::TopP { p }        => cand.nucleus(*p),
                Stage::MinP { p }        => cand.min_p(*p),
                Stage::Typical { mass }  => cand.typical(*mass),
                // device-side stages already applied
                _ => {}
            }
        }

        // ⚠️ Observable approximation, not a silent one (§2.2).
        if cand.len() == k {
            telemetry::record_truncation_saturated(s);
        }
        out.push(cand.sample(chain.seed));
    }
    out
}
```

---

# 5. Module Layout + Wave Plan

```text
gljax/src/sample/
├── mod.rs        sample_step, Candidates
├── chain.rs      §1.2  SamplerChain, Stage, validate()
├── penalty.rs    §3.1  PenaltyState, sparse upload
├── mask.rs       §3.3  AllowMask, MaskSource trait
└── host.rs       §2.2  top-p / min-p / typical over K, RNG
```

| Wave | Scope | Gate |
|---|---|---|
| **A14.1** | `chain.rs` + host-only path (option A) | Greedy (`TopK{1}`) reproduces ARTX05's argmax **bit-identically** |
| **A14.2** | ⭐ Split path (option C): device top-K + host tail | Sampled distribution statistically indistinguishable from option (A) over ≥100k tokens; transfer measured at §2.2's figure |
| **A14.3** | `penalty.rs` — sparse device-side penalties | Matches a host-side reference exactly on the same history |
| **A14.4** | `mask.rs` — the `MaskSource` seam, no grammar | A trivial always-allow mask changes nothing; a single-token mask forces that token |
| **A14.5** | ARTX11 integration — `gather_candidate_probs` (§2.3) | ⭐ Speculative output distribution indistinguishable from non-speculative (ARTX11's A11.2 gate, now actually satisfiable) |

⚠️ **A14.1's gate is the one that closes §0's contradiction**, and it is deliberately strict: greedy
through the new chain must be *bit-identical* to ARTX05's `argmax_f32`, not merely equivalent. Any
difference means the chain changed semantics somewhere it should not have.

⚠️ **A14.5 is what makes ARTX11 honest.** Until distribution sampling exists, ARTX11's lossless
guarantee cannot be tested, only asserted.

## 5.1 Open questions

1. **Is `K = 512` right?** (§2.2) Must be measured against saturation rate on real traffic, not
   assumed.
2. **Does `stablehlo.top_k` lower efficiently on TPU and CUDA?** Unknown. If it degenerates to a full
   sort, option (C)'s advantage shrinks and the split point may need to move. ⚠️ This is the same
   emit-and-verify posture as ARTX09 §3.2 and ARTX10 §5 — it belongs in the shared probe.
3. **Per-slot chain divergence.** ARTX07 batches slots with different `SamplingParams`. Device-side
   stages must therefore be parameterized *per slot*, not per batch — a `[slots]`-shaped parameter
   tensor rather than a compiled constant. Cost unmeasured.
4. **Seeded reproducibility across batch composition.** A seeded request must produce identical
   output regardless of which other requests share its batch. This constrains the RNG to be
   per-slot-keyed, not a shared stream. ⚠️ Easy to get wrong and hard to notice.

---

# Appendix A — Design Decision Summary

| # | Decision | Rationale | Reversible? |
|---|---|---|---|
| D1 | Sampler order is **explicit data**, recorded per request | vLLM and llama.cpp disagree; order can be model-specific (QwQ-32B looping) | Hard |
| D2 | Default = vLLM ordering; llama.cpp ordering available | ARTX16 serves an OpenAI-compatible API; vLLM is that surface's reference | Trivial |
| D3 | `validate()` refuses invalid chains; never silently reorders | A silently-corrected order makes the recorded order a lie | Trivial |
| D4 | Greedy is `TopK{1}`, not a parallel path | Structurally removes the ARTX05/ARTX16 contradiction | Trivial |
| D5 | ⭐ **Split sampling (option C)**: device mask+penalties+top-K, host tail | 608 KB → 4 KB per token per slot; 148× less traffic, vocab-independent | Hard |
| D6 | `K` is a compile-cache key dimension | Same class as ARTX05 buckets and ARTX11 γ | Hard |
| D7 | Truncation saturation is **reported** | Makes option (C)'s approximation observable rather than silent | Trivial |
| D8 | Exact full-vocab sampling (A) kept behind a flag | ARTX12 needs an exact oracle to validate (C) against | Trivial |
| D9 | Speculative verify uses **gather**, not top-K | The draft candidate need not be in the target's top-K — that is the case rejection sampling exists for | Hard |
| D10 | Penalties applied on device, before top-K | A post-truncation penalty cannot demote a token out of the set | Medium |
| D11 | Penalty/bias uploads are **sparse** | A generation touches hundreds of distinct tokens, not 151,936 | Trivial |
| D12 | ⭐ Grammar mask applied on device, **before** top-K | Otherwise all K survivors may be masked, forcing a grammar-violating fallback | Hard |
| D13 | Mask is bit-packed; cost priced here, not deferred to ARTX15 | 18.5 KB/slot/step upload = 1.19 MB/iteration at 64 slots — 4.6× the download | N/A |
| D14 | `MaskSource` is a trait; ARTX14 never depends on ARTX15 | Sampling ships without grammar support | Trivial |
| D15 | RNG is per-slot-keyed, not a shared stream | A seeded request must not depend on its batch neighbours | Hard |

---

# Sources

- [Sampling and Token Generation | vLLM](https://deepwiki.com/vllm-project/vllm/4.4-sampling-and-token-generation) and [vLLM Sampler API](https://docs.vllm.ai/en/latest/api/vllm/v1/sample/sampler/) — order: penalties → temperature → logit processors (min-p) → top-k/top-p → sample.
- [Token Sampling and Generation | llama.cpp](https://deepwiki.com/ggml-org/llama.cpp/3.8-token-sampling-and-generation) and [Sampling args in llama-server](https://blog.alexewerlof.com/p/sampling-args-in-llama-server) — default chain `penalties → dry → top_n_sigma → top_k → typ_p → top_p → min_p → xtc → temperature`; penalties first so truncation drops penalized tokens; temperature last; `--samplers` to override; QwQ-32B repetition-penalty looping fixed by reordering.
- [vllm.v1.sample.ops.topk_topp_sampler](https://docs.vllm.ai/en/latest/api/vllm/v1/sample/ops/topk_topp_sampler/) — top-p sorts the logits tensor; slow for large batches.
- [Sorting-Free GPU Kernels for LLM Sampling | FlashInfer](https://flashinfer.ai/2025/03/10/sampling.html) — rejection-sampling-based sampler avoiding the sort; a custom kernel, therefore out of scope per ARTX08.
- [LLM Sampling Parameters Explained](https://letsdatascience.com/blog/llm-sampling-temperature-top-k-top-p-and-min-p-explained) — min-p keeps tokens with `P(token) ≥ P(max) × min_p`.
- [Sampling Parameters | vLLM](https://docs.vllm.ai/en/v0.6.0/dev/sampling_params.html) — presence vs frequency vs repetition penalty semantics.

**Repo-internal:** `ARTX05` (`argmax_f32`, host-side sampling design decision, `Session::generate`);
`ARTX07` (slot lifecycle, batch composition); `ARTX08` (no custom kernels; static shapes);
`ARTX09 §4.3` (201 MB/iteration KV read, for transfer-cost comparison); `ARTX11 §1.2, §5.2, §5.3`
(rejection sampling, `β = min(1, p/q)`, the A11.2 gate); `ARTX12` (exact oracle requirement);
`ARTX16 §1.3` (`SamplingParams`).
