# ARTX11 — Speculative Decoding under Static Shapes

**Series:** gljax (Sanctum Visibilia) Architecture Research
**Status:** Draft — research-grounded
**Depends on:** ARTX3 (ops layer), ARTX5 (static KV + bucketing), ARTX7 (continuous batching), ARTX8 (matrix compute), ARTX12 (correctness harness — **gating**)
**Introduces:** `gljax/src/spec/`, `gljax/src/arch/`
**Next:** [ARTX12 — Model Compatibility & Runtime Conformance](ARTX12-model-compatibility-and-runtime-conformance.md)
**Research grounded:** 2026-07-27 (sources at end)

---

# 0. Scope, and Two Corrections to the Brief

## 0.1 The multi-architecture requirement

The brief's original framing assumed a draft model. The requirement as clarified is stronger:

> gljax must not support only one model family. It must be designed for all models — including
> **Gemma 4, which uses GeGLU**.

⛔ **This is a bigger ask than it first appears, and it lands on ARTX3, not on ARTX11.** ARTX3's ops
layer is Qwen2-shaped throughout: it ships exactly one FFN (`swiglu_ffn`, SiLU hardcoded), one norm
(plain RMSNorm), one RoPE variant (NeoX), and no notion of an architecture descriptor. Gemma differs
in at least six independent ways (§4.2).

ARTX11 cannot ignore this, because **speculative decoding runs two models at once** — and the moment
those two models may come from different families, the engine needs an architecture abstraction it
does not currently have. §4 therefore specifies `gljax/src/arch/`, and is explicit that it is a
**retrofit to ARTX3**, not new ARTX11 surface.

## 0.2 ⚠️ The "no tree decoding" line needs redrawing

The brief lists **Medusa ⭐⭐⭐⭐⭐, EAGLE ⭐⭐⭐⭐, Hydra ⭐⭐⭐⭐** as priority sources while listing
**"Tree decoding variants (future)"** as out of scope. Those are in direct conflict: Medusa *is* tree
attention (top-2 from head 1 × top-3 from head 2 = 6 candidates on a tree with an ancestor-only
mask), and EAGLE-2's headline contribution *is* a dynamic draft tree.

⚠️ **DESIGN DECISION — the line is redrawn at *dynamic* trees, not at all trees, on a static-shape
criterion.**

| Draft structure | Shape at compile time | gljax verdict |
|---|---|---|
| **Chain** (Leviathan) — γ tokens, linear | Fixed: verify `[γ+1, D]` | **In scope**, Wave A11.1–A11.4 |
| **Fixed tree** (Medusa, EAGLE-1) — tree topology chosen offline | Fixed: verify `[T, D]` + constant mask | **In scope structurally**, Wave A11.5 |
| **Dynamic tree** (EAGLE-2) — topology depends on runtime draft confidence | **Varies per step** | **Out of scope.** Forces recompilation; violates ARTX5/ARTX7 |

This is a principled line rather than an arbitrary one: a fixed tree is just a chain with a
compile-time-constant attention mask, so it costs gljax nothing architecturally. A dynamic tree
changes the verified-token count per iteration, which is exactly the thing ARTX5's bucketing and
ARTX7's compile cache exist to prevent.

## 0.3 Out of scope (unchanged from the brief)

Training draft models · distillation · fine-tuning · **dynamic** tree decoding · hardware-specific
optimization (ARTX8: gljax owns no kernels).

⚠️ The no-training constraint is load-bearing and it eliminates Medusa-2, EAGLE-3, and Hydra as
*implementable* targets — all three require training draft heads. They remain in scope as **research
inputs**, and §3.4 records what gljax would need if that constraint is ever lifted.

## 0.4 ARTX12 is a hard prerequisite

ARTX11 adds a second model, a new KV write pattern (variable accepted length), and a new sampling
path. Landing that on an engine with no matmul correctness harness means debugging three interacting
unknowns. ARTX12 Wave A10.1–A10.2 (T0 tier, no device needed) must be green first.

---

# 1. Wave A11.1 — Speculative Decoding Fundamentals

## 1.1 The mechanism

Speculative decoding exploits an asymmetry: generating one token is bandwidth-bound, but *verifying*
k tokens costs almost the same as verifying one, because the weights are streamed once either way.

```text
Standard decode, k tokens:      k sequential target forward passes
Speculative decode, k tokens:   γ cheap draft passes + 1 target pass over γ+1 positions
```

The draft model proposes γ candidate tokens autoregressively. The target model then evaluates all
γ+1 positions **in a single forward pass**, and a verification rule decides how many of the proposals
to keep.

## 1.2 The lossless guarantee (Leviathan et al., 2023)

Each candidate token `y` is validated by rejection sampling and accepted with probability

```text
β(y) = min(1, p(y|h) / q(y|h))
```

where `p` is the target distribution and `q` the draft distribution. On rejection, a replacement is
sampled from the normalized residual `norm(max(0, p − q))`. **The resulting token sequence is
distributed exactly as if sampled from the target model alone** — speculation changes speed, never
output distribution.

The expected per-token acceptance rate relates directly to how close the two models are:

```text
α(h) = 1 − ½ · ‖p(·|h) − q(·|h)‖₁
```

⚠️ This is the equation that governs draft-model selection (§3.1). A draft model is good exactly to
the extent its distribution is close to the target's in L1 — not to the extent it is small, fast, or
architecturally similar.

## 1.3 The performance model

```text
E[tokens per iteration]  =  (1 − α^(γ+1)) / (1 − α)

wall-clock speedup       =  (1 − α^(γ+1)) / ((1 − α)(γc + 1))
```

where `c` = (draft forward cost) / (target forward cost).

Optimal γ, computed from the formula above:

| α (acceptance) | c = 0.05 | c = 0.1 | c = 0.2 | c = 0.3 |
|---|---|---|---|---|
| 0.6 | γ=4, **1.74×** | γ=3, **1.67×** | γ=2, **1.51×** | γ=2, **1.38×** |
| 0.7 | γ=5, **2.15×** | γ=4, **2.02×** | γ=3, **1.77×** | γ=2, **1.59×** |
| 0.8 | γ=8, **2.79×** | γ=6, **2.47×** | γ=4, **1.87×** | γ=3, **1.72×** |
| 0.9 | γ=14, **4.15×** | γ=10, **3.43×** | γ=6, **2.42×** | γ=5, **2.03×** |

⚠️ **Three readings that shape the rest of this document:**

1. **γ is not a free knob.** Above the optimum, speedup *decreases* — each extra draft token costs
   `c` and pays off only with probability `α^k`. A γ chosen by intuition rather than by measured α
   will usually be too large.
2. **c matters as much as α.** A draft model at c=0.3 caps you near 2× regardless of how good it is.
   This is the argument for a genuinely small draft model, and the reason cross-family pairing
   (§3.2) is valuable — it widens the pool of small models you may choose from.
3. **α must be measured per workload, not assumed.** Published rates of 0.75–0.85 are chat-workload
   figures; code and structured output run higher, open-ended generation lower.

## 1.4 ⭐ Why this specifically attacks the wall ARTX8 measured

ARTX8 established that low-batch decode runs at an arithmetic intensity of 0.5–2 FLOP/byte against
ridge points of 241 (TPU v5e), 153 (A100), 295 (H100) — **200–600× below** the point where the
device becomes compute-bound. The matrix unit sits idle waiting on HBM.

Verifying γ+1 tokens in one pass reads the weights **once** and does γ+1× the arithmetic:

```text
AI(standard decode)      ≈ 1
AI(verify γ+1 positions) ≈ γ+1
```

⚠️ Speculative decoding is therefore *the same lever as ARTX7's continuous batching*, applied along a
different axis: batching multiplies intensity by the number of concurrent **requests**, speculation
multiplies it by the number of speculative **positions**. They compose — and both are attacking
ARTX8's finding rather than working around it.

⚠️ **Corollary worth stating: the two levers contend for the same headroom.** At high batch (ARTX7
serving 64 slots), decode has already moved toward compute-bound and speculation's marginal value
falls, while its cost (running the draft model) stays. §6.4 makes this a scheduler decision.

---

# 2. Wave A11.2 — Draft–Verifier Architecture

## 2.1 The four drafter families

| Family | Mechanism | Needs training? | Tree? | gljax verdict |
|---|---|---|---|---|
| **Independent draft model** (Leviathan) | A separate, smaller LM | **No** | No | ✅ **Primary target** |
| **Medusa** | k FFN heads (single layer + residual) on the target's last hidden state | Yes (heads) | Fixed | ⚠️ Structural support only |
| **Hydra** | Medusa heads made *sequentially dependent* — each head sees earlier sampled tokens | Yes (heads) | Fixed | ⚠️ Research input |
| **EAGLE** | Autoregression at the **feature** level (before the LM head), reusing the target's LM head | Yes (drafter) | E1 fixed, E2 dynamic | ⚠️ Research input |

⚠️ **DESIGN DECISION — the independent draft model is gljax's primary drafter, and it is the only
one implementable under the no-training constraint.**

The reasoning is not preference but arithmetic: Medusa-1 heads, Hydra heads, and EAGLE drafters must
all be trained against a specific target model. With training out of scope, gljax can only consume
drafters that already exist as ordinary models — which is exactly what an independent draft model is.

## 2.2 What the other families would buy, recorded for later

* **Medusa** replaces the draft *model* with k lightweight heads, driving `c` toward ~0 — its heads
  are a single FFN layer with a residual connection, run off a hidden state the target already
  computed. The tree is fixed (e.g. top-2 × top-3 = 6 candidates with an ancestor-only mask), which
  per §0.2 is static-shape compatible.
* **Hydra** makes those heads sequentially dependent — each head is a function of both the base
  model's hidden state *and* the embeddings of tokens sampled by previous heads — reporting up to
  **1.31× over Medusa** and 2.70× over autoregressive decoding.
* **EAGLE** drafts in feature space before the LM head and then reuses the target's own LM head,
  reporting **2.1×–3.8×** while provably preserving the output distribution. EAGLE-3 fuses low-,
  mid-, and high-level features and uses training-time testing.

⚠️ All three require the target model to expose **intermediate hidden states** to the drafter. That
is a real API consequence for ARTX3/ARTX4: `Session` currently returns logits. §7.3 records the
minimal hook that keeps this door open without building it now.

## 2.3 The pairing contract

```rust
// gljax/src/spec/pairing.rs

/// The complete set of constraints a (draft, target) pair must satisfy.
/// Checked once at Session construction — never at decode time.
pub struct PairingContract {
    /// How draft and target vocabularies relate. THE critical constraint (§3.2).
    pub vocab: VocabRelation,
    /// Draft and target may be DIFFERENT architectures (§4).
    pub draft_arch:  Architecture,
    pub target_arch: Architecture,
    /// Measured, not assumed (§1.3).
    pub cost_ratio: Option<f64>,     // c
}

pub enum VocabRelation {
    /// Byte-identical tokenizer and vocab. Standard Leviathan; exact and cheapest.
    Identical,
    /// Different tokenizers. Requires a heterogeneous-vocabulary algorithm (§3.2).
    Heterogeneous { algorithm: HeteroAlgorithm },
}
```

---

# 3. Draft Model Lifecycle

## 3.1 Selection

Per §1.2, the metric is L1 distance between draft and target distributions, not size. Practically
that means:

```text
Selection order:
  1. Same-family, smaller  → highest α, Identical vocab, cheapest path
  2. Cross-family, small   → lower α, needs §3.2, but a much wider pool
  3. No draft model        → fall back to standard decode (always valid)
```

Concrete same-family pairs available today:

| Target | Draft | Vocab | Approx c |
|---|---|---|---|
| Qwen3-1.7B | Qwen2.5-0.5B | ⚠️ verify — cross-generation tokenizers may differ | ~0.3 |
| Gemma 4 31B Dense (30.7B) | Gemma 4 E2B (2.3B eff.) | Identical | ~0.07 |
| Gemma 4 26B-A4B (3.8B active) | Gemma 4 E2B | Identical | ~0.6 ⚠️ |

⚠️ The MoE row is a trap worth naming: a 26B-A4B target activates only 3.8B parameters per token, so
a 2.3B draft is **not** cheap relative to it. `c ≈ 0.6` caps speedup near 1.2× at α=0.8 — below the
point where the added complexity is worth anything. **MoE targets need proportionally tinier
drafters, or none.**

## 3.2 ⭐ Cross-family pairing — the vocabulary problem, and its solution

**The constraint.** Classical speculative decoding requires draft and target to share a tokenizer,
because verification compares `p(y)` and `q(y)` for *the same token id*. Different families have
different vocabularies, which breaks the formulation outright. The problem is worse than an id
remap: token boundaries need not align — one draft token may correspond to several target tokens —
and BPE tokenizers assign different ids to the same surface form depending on surrounding context.

**The solution, and it fits gljax's constraints exactly.** Timor et al. (ICML 2025, Oral) present
**three lossless speculative-decoding algorithms for heterogeneous vocabularies**. All three preserve
the target distribution, and critically they **work with off-the-shelf models without additional
training or modification** — reporting up to **2.8×** over autoregressive decoding. Two of them,
**SLEM (String-Level Exact Match)** and **TLI (Token-Level Intersection)**, were upstreamed into
HuggingFace Transformers and made the default for heterogeneous speculative decoding.

⚠️ **DESIGN DECISION — gljax implements `Identical` first and `Heterogeneous` second, but the
verification interface is designed for both from day one.**

```rust
pub enum HeteroAlgorithm {
    /// String-Level Exact Match — verify on detokenized strings rather than ids.
    /// Robust to boundary misalignment; costs a detokenize per candidate.
    Slem,
    /// Token-Level Intersection — restrict verification to the shared vocab subset.
    /// Cheaper than SLEM; α degrades as vocab overlap shrinks.
    Tli,
}
```

The reason to design the interface up front rather than retrofit: heterogeneous verification operates
on **strings or vocab-intersections**, not on aligned logit vectors. A `Verifier` trait written
assuming aligned `[V]` logits cannot be widened later without rewriting every caller.

⚠️ Cross-family speculation is **strictly worse than same-family when both are available** — α drops
because the distributions are less similar, and SLEM adds detokenization cost. Its value is coverage:
it makes "design for all models" achievable without training a drafter per target.

## 3.3 Loading and lifetime

```rust
// gljax/src/spec/drafter.rs

pub struct DraftModel {
    session: Session,                 // an ordinary ARTX4 Session
    arch: Architecture,               // §4 — need NOT match the target's
    /// Own KV slab, own slots, own buckets (§6.2).
    kv: StaticKvSlab,
}
```

⚠️ **DESIGN DECISION — the draft model is an ordinary `Session`, not a special case.**
It loads through ARTX4's checkpoint path, compiles through ARTX5's bucketing, and caches through the
same `CompileCache`. The only thing `spec/` adds is orchestration. This keeps the drafter benefiting
automatically from every ARTX1–ARTX12 improvement, and means ARTX12's correctness harness already
covers it.

⚠️ **Memory consequence, which must be planned not discovered:** two models means two weight sets and
**two KV slabs**. ARTX7 sized `max_slots` against a single slab; with speculation the budget is

```text
total = W_target + W_draft
      + kv_slab(target, max_slots, bucket)
      + kv_slab(draft,  max_slots, bucket)
```

For a TPU v5e chip (16 GB HBM per ARTX16 §9.4) this is a real constraint, and `max_slots` must be
recomputed rather than inherited from a non-speculative config.

## 3.4 If the no-training constraint is ever lifted

Recorded so the decision is not silently re-litigated: adopting Medusa/Hydra/EAGLE requires (a) a
training pipeline gljax does not have and is not designed for — it is inference-only with no gradient
path — and (b) the hidden-state hook of §7.3. The natural home is a separate training arm, not gljax.

---

# 4. Multi-Architecture Support (the ARTX3 retrofit)

## 4.1 Why ARTX11 forces this

Speculative decoding runs two models simultaneously and, per §3.2, they may be from different
families. An engine that hardcodes one architecture cannot do that. Beyond speculation, the
requirement stands on its own: gljax cannot claim to serve LLMs while supporting only Qwen2's exact
shape.

## 4.2 ⛔ What ARTX3 currently assumes, and what Gemma needs

| Concern | ARTX3 today | Gemma requires |
|---|---|---|
| FFN | `swiglu_ffn` — SiLU hardcoded | **GeGLU**: `GeLU(fc1(x)) * fc2(x) → fc3`, activation `gelu_pytorch_tanh` |
| Norm placement | 2 RMSNorm per block (pre-attn, pre-FFN) | **4 RMSNorm per block** (adds post-attn, post-FFN) |
| Norm formula | `x/rms · weight` | **Zero-centered weights, `(1 + weight)` scaling** |
| QK normalization | none | **`q_norm` / `k_norm` RMSNorm on Q and K** |
| Attention scale | `1/sqrt(head_dim)` | **Custom query pre-attention scalar**, not head_dim |
| Embeddings | plain lookup | **Scaled word embedding** |
| Attention pattern | uniform causal | **Alternating local sliding-window / global, 5 local : 1 global** |

Seven differences, and **every one of them is failure mode class F2/F3 from ARTX12** — shape-valid,
non-crashing, and productive of fluent wrong output. A `(1 + weight)` norm implemented as `weight`
does not crash; it degrades quality in a way only perplexity reveals.

⚠️ The attention-alternation row is the most invasive: a 5:1 local/global pattern means **per-layer
attention configuration**, and local layers need a *smaller* KV allocation than global ones. That
reaches into ARTX5's slab sizing and ARTX7's slot accounting, not just into ops.

## 4.3 The descriptor

⚠️ **DESIGN DECISION — a data descriptor, not a trait hierarchy.**
ARTX8's anti-over-engineering rule applies: variation here is *configuration*, not *behavior needing
dynamic dispatch*. A descriptor is inspectable, serializable into the ARTX5 compile-cache key, and
diffable between draft and target. A trait per architecture would produce one implementor per model
and a combinatorial mess at the draft/target boundary.

```rust
// gljax/src/arch/mod.rs

#[derive(Debug, Clone, PartialEq)]
pub struct Architecture {
    pub name: &'static str,          // "qwen2", "qwen3", "gemma3", "gemma4"
    pub ffn: FfnKind,
    pub norm: NormKind,
    pub attention: AttentionKind,
    pub rope: RopeKind,
    pub embedding: EmbeddingKind,
    pub logit_softcap: Option<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FfnKind {
    /// gate/up/down with SiLU on gate. Qwen, LLaMA, Mistral.
    SwiGlu,
    /// gate/up/down with GeLU on gate. Gemma.
    /// `tanh_approx` selects gelu_pytorch_tanh vs exact erf-based GeLU —
    /// ⚠️ these differ numerically and the checkpoint expects one specific variant.
    GeGlu { tanh_approx: bool },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NormKind {
    /// out = x/rms(x) * weight
    RmsNorm { eps: f32 },
    /// out = x/rms(x) * (1 + weight)   ← Gemma
    RmsNormZeroCentered { eps: f32 },
}

#[derive(Debug, Clone, PartialEq)]
pub struct AttentionKind {
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    /// None → 1/sqrt(head_dim). Some(s) → Gemma's query pre-attention scalar.
    pub query_scale: Option<f32>,
    pub qk_norm: bool,
    /// Per-layer pattern. Uniform for Qwen; 5 local : 1 global for Gemma 3/4.
    pub layer_pattern: LayerPattern,
    /// Extra norms Gemma places after attention and after FFN.
    pub post_norms: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum LayerPattern {
    Uniform,
    /// e.g. { local_window: 1024, period: 6, global_at: 5 } → 5 local then 1 global
    LocalGlobal { local_window: usize, period: usize, global_at: usize },
}
```

⚠️ **DESIGN DECISION — the `Architecture` descriptor joins the compile-cache key.**
ARTX5/ARTX7/ARTX8 key on `(batch, seq_bucket, dtype, device)`; ARTX11 adds `arch_hash` and `gamma`.
Without `arch_hash`, running a Qwen draft alongside a Gemma target would collide in the cache and
execute the wrong compiled program — silently, since shapes match.

## 4.4 ⚠️ Honest scoping

This section specifies a **retrofit to ARTX3**, and it is substantial: seven behavioral variations,
a new per-layer attention pattern that touches ARTX5's slab sizing, and a cache-key change across
four documents. It is listed here because ARTX11 *requires* it, not because ARTX11 is the natural
home for it.

**Recommendation: land §4 as its own wave (A11.0) before any speculation work**, validated by ARTX12's
harness against a real Gemma checkpoint. Speculation built on an architecture layer that has never
run a second architecture would be testing two unproven things at once.

---

# 5. Wave A11.3 — Verification Pipeline

## 5.1 ⭐ The static-shape finding

This is ARTX11's central technical result, and it is more favourable than expected:

> **Speculative decoding is compatible with gljax's static-shape model without compromise, provided
> γ is a compile-time constant.**

The reasoning, step by step:

```text
1. DRAFT   γ sequential decode steps on the draft model.
           Each is an ordinary ARTX5 decode program.        → STATIC ✓

2. VERIFY  One target forward pass over γ+1 positions.
           Shape [γ+1, D] — structurally identical to an
           ARTX5 prefill chunk with chunk_size = γ+1.       → STATIC ✓

3. ACCEPT  n ∈ [0, γ+1] accepted. VARIABLE.                 → the only dynamism

4. COMMIT  Write all γ+1 KV entries unconditionally,
           then advance the position counter by n.
           ARTX5 already writes KV at a RUNTIME scalar
           index via dynamic_update_slice.                  → STATIC ✓
```

⚠️ **The key move is in step 4: never make the KV *write* conditional — make the *position counter*
absorb the variance.** Rejected entries are written and then simply never read, because the causal
mask and the position counter exclude them. ARTX7 already relies on exactly this property, where a
freed slot's stale KV is described as harmless because the attention mask excludes it.

**Consequences:**

* γ becomes one more compile-cache key dimension. Supporting γ ∈ {1,2,4,8} costs 4× the target's
  verify artifacts, comparable to adding one ARTX5 sequence bucket.
* Dynamic γ per request is **not** supported — it would mean recompilation per request. §6.4 handles
  adaptivity by *switching between compiled γ values*, not by varying γ continuously.
* Dynamic trees (EAGLE-2) are excluded for precisely this reason (§0.2): the verified-position count
  varies per step, which no fixed bucket can absorb.

## 5.2 Verification algorithms

```rust
// gljax/src/spec/verify.rs

pub enum VerifyRule {
    /// argmax(p) == draft token. Deterministic decoding only.
    /// Cheapest; exact for temperature-0 serving.
    Greedy,
    /// Leviathan rejection sampling. LOSSLESS for any temperature.
    /// accept with min(1, p/q); on reject sample norm(max(0, p-q)).
    RejectionSampling,
    /// Medusa's typical acceptance. Accepts plausible-under-p candidates
    /// using an entropy-dependent threshold.
    /// ⚠️ NOT distribution-preserving — a quality/speed tradeoff, not a free win.
    TypicalAcceptance { epsilon: f32, delta: f32 },
}
```

⚠️ **DESIGN DECISION — `RejectionSampling` is the default; `TypicalAcceptance` must be opt-in and
labelled lossy.** Medusa's typical acceptance raises acceptance rates by relaxing the criterion,
which is a legitimate tradeoff — but ARTX12's whole premise is that silently-changed output is the
bug class this project has already been bitten by. A lossy default would make the correctness harness
unable to distinguish "verification is lossy by configuration" from "verification is wrong".

## 5.3 Pseudocode

```rust
// gljax/src/spec/step.rs — one speculative iteration for one slot

pub fn speculative_step(
    target: &mut Session, draft: &mut DraftModel,
    slot: SlotId, pos: usize, gamma: usize, rule: VerifyRule,
) -> Result<AcceptResult, SpecError> {
    // ── 1. DRAFT ───────────────────────────────────────────────────────
    // γ sequential decode steps. Each is an ordinary ARTX5 decode.
    let mut cand   = Vec::with_capacity(gamma);
    let mut q_dist = Vec::with_capacity(gamma);
    let mut tok = target.last_token(slot);
    for i in 0..gamma {
        let q = draft.decode_step(slot, pos + i, tok)?;   // [V_draft]
        tok = sample(&q, &target.sampling(slot));
        cand.push(tok);
        q_dist.push(q);
    }

    // ── 2. VERIFY ──────────────────────────────────────────────────────
    // ONE target pass over γ+1 positions. Static shape [γ+1, D].
    // Position j's logits predict token j+1, so index j gives p for cand[j].
    let p_dist = target.verify_chunk(slot, pos, &cand)?;  // [γ+1, V_target]

    // ── 3. ACCEPT ──────────────────────────────────────────────────────
    let accepted = match rule {
        VerifyRule::Greedy => {
            cand.iter().enumerate()
                .take_while(|(j, &t)| argmax(&p_dist[*j]) == t)
                .count()
        }
        VerifyRule::RejectionSampling => {
            let mut n = 0;
            for j in 0..gamma {
                let (p, q) = (p_dist[j][cand[j] as usize], q_dist[j][cand[j] as usize]);
                // β = min(1, p/q). q > 0 because cand[j] was SAMPLED from q.
                if uniform01() < (p / q).min(1.0) { n += 1; } else { break; }
            }
            n
        }
        VerifyRule::TypicalAcceptance { epsilon, delta } =>
            typical_accept(&p_dist, &cand, epsilon, delta),
    };

    // ── 4. BONUS TOKEN ─────────────────────────────────────────────────
    // If ALL γ accepted, position γ's logits are a free extra token:
    // the target already computed them. This is why E[tokens] can reach γ+1.
    let extra = if accepted == gamma {
        sample(&p_dist[gamma], &target.sampling(slot))              // free
    } else {
        // On rejection at j, resample from the normalized residual.
        // This step is what preserves the target distribution exactly.
        sample_residual(&p_dist[accepted], &q_dist[accepted])
    };

    // ── 5. COMMIT ──────────────────────────────────────────────────────
    // KV for ALL γ+1 positions was already written by verify_chunk.
    // Advance the counter only past what was accepted; the rest is
    // never read again (mask + position exclude it).
    let n_new = accepted + 1;
    target.advance_position(slot, n_new);
    draft.rewind_to(slot, pos + n_new);   // §6.3

    Ok(AcceptResult {
        tokens: [&cand[..accepted], &[extra]].concat(),
        accepted, gamma,
    })
}
```

⚠️ **The bonus token in step 4 is not an optimization — it is why the math works.** Without it,
E[tokens] would cap at γ; with it, a fully-accepted round yields γ+1. Omitting it is a silent ~15%
throughput loss at typical α that no correctness test would catch.

⚠️ **Note on `q > 0` in step 3.** The division `p/q` is safe precisely because `cand[j]` was sampled
*from* `q`, so its probability is nonzero. This invariant breaks under heterogeneous vocabularies
(§3.2), where the candidate may not exist in the draft's vocab at all — which is why SLEM/TLI are
separate algorithms rather than a token-id remap.

---

# 6. Wave A11.4 — Scheduler Integration

## 6.1 ⛔ The problem speculation creates for ARTX7

ARTX7's scheduler assumes **one token per slot per iteration**. Speculation breaks that: slot 0 may
accept 4 tokens while slot 1 accepts 0, in the same iteration.

```text
ARTX7 today:                    ARTX7 with speculation:
  iteration → 1 token/slot        iteration → n_i tokens for slot i, n_i ∈ [1, γ+1]
  positions advance uniformly     positions advance by different amounts
```

⚠️ **DESIGN DECISION — per-slot position counters, which ARTX7 already has.**
This is less invasive than it appears. ARTX5 addresses KV as `[slot, :, position, :]` with `position`
a *runtime scalar*, and ARTX7's slots are independent by construction. Nothing in the slab or the
slot manager assumes positions advance in lockstep — only the scheduler's bookkeeping does.

What genuinely changes:

| Component | Change |
|---|---|
| `KvSlotManager` (ARTX7 A7.1) | **None.** It owns `SlotId` lifecycle, never positions. |
| `StaticKvSlab` (ARTX7 A7.3) | **None.** Already writes at a runtime position index. |
| `Scheduler` (ARTX7 A7.2) | Advance each slot by its own `n_i`; emit `n_i` tokens |
| `CompileCache` (ARTX5) | Key gains `gamma` and `arch_hash` (§4.3) |
| Telemetry (ARTX16 §7) | New: acceptance rate, tokens/iteration, draft cost share |

## 6.2 Batched speculation

With B active slots and γ draft tokens:

```text
Draft phase:   γ batched decode steps on the draft model   → [B, 1, D_draft] each
Verify phase:  1 batched target pass                       → [B, γ+1, D_target]
```

⚠️ **DESIGN DECISION — γ is uniform across a batch, not per-request.**
A per-request γ would mean ragged verify shapes, i.e. recompilation (§5.1). γ is a property of the
*compiled program*, so all slots in one batch share it. Requests wanting a different γ go to a
different batch and a different compiled artifact.

⚠️ The draft model needs its own bucketing: `[B, 1, D_draft]` decode programs, keyed on B. Because
the draft is an ordinary `Session` (§3.3), it inherits ARTX5's bucketing machinery unchanged — but it
does *multiply the warmup cost* ARTX16 §4.2 flagged at 20–30 minutes cold.

## 6.3 Draft KV synchronization

The subtlety most likely to produce a silent bug:

```text
Round: draft proposes 4 tokens, target accepts 2.

  Target KV:  positions 0..1 committed, 2..3 written but excluded by position counter ✓
  Draft KV:   positions 0..3 ALL committed — the draft believed all 4 ✗

  → The draft's KV is now WRONG for positions 2..3.
```

⚠️ **DESIGN DECISION — rewind the draft's position counter; do not clear its KV.**
Same mechanism as the target: `draft.rewind_to(slot, pos + n_new)` moves the counter back, and the
next draft step overwrites positions `≥ n_new` via `dynamic_update_slice`. No clear, no reallocation,
no shape change. ARTX7 already established that stale KV beyond the position counter is harmless.

⚠️ Forgetting the rewind produces a draft that conditions on tokens the target rejected. Output stays
fluent; α silently collapses. **This belongs in ARTX12's harness as an explicit test** — assert that
after a partial acceptance, the draft's next proposal is identical to what it would produce from a
fresh context at that position.

## 6.4 When to speculate

Per §1.4, speculation and batching contend for the same headroom.

```rust
// gljax/src/spec/policy.rs

pub struct SpecPolicy {
    /// Above this batch occupancy, decode is no longer bandwidth-starved
    /// and the draft's cost stops paying for itself. Disable speculation.
    pub max_batch_for_spec: usize,
    /// Compiled γ values available. Switching between them is free
    /// (both artifacts are cached); varying γ continuously is not (§5.1).
    pub gamma_ladder: Vec<usize>,     // e.g. [2, 4, 8]
    /// Rolling measured acceptance rate, per §1.3's optimum table.
    pub alpha_window: RollingMean,
    /// Below this α, speculation is a net loss — fall back to plain decode.
    pub min_alpha: f64,               // ~0.4
}

impl SpecPolicy {
    /// Choose γ from the LADDER using measured α and c — never a continuous γ.
    pub fn choose_gamma(&self, alpha: f64, c: f64) -> Option<usize> {
        if alpha < self.min_alpha { return None; }      // plain decode
        self.gamma_ladder.iter().copied()
            .max_by(|&a, &b| speedup(alpha, a, c).total_cmp(&speedup(alpha, b, c)))
    }
}

fn speedup(alpha: f64, gamma: usize, c: f64) -> f64 {
    (1.0 - alpha.powi(gamma as i32 + 1)) / ((1.0 - alpha) * (gamma as f64 * c + 1.0))
}
```

⚠️ **DESIGN DECISION — adaptive γ selects from a compiled ladder; it never computes a fresh γ.**
This is the static-shape constraint expressed as policy. `choose_gamma` returning `None` (plain
decode) must always be a legal path — speculation is an accelerator, never a dependency.

---

# 7. Runtime Integration + Module Layout

## 7.1 Layout

```text
gljax/src/
├── arch/                    ← §4, the ARTX3 retrofit (Wave A11.0)
│   ├── mod.rs               Architecture descriptor
│   ├── ffn.rs               SwiGlu | GeGlu dispatch
│   ├── norm.rs              RmsNorm | RmsNormZeroCentered
│   ├── attention.rs         query_scale, qk_norm, LayerPattern
│   └── registry.rs          "qwen2"|"qwen3"|"gemma3"|"gemma4" → Architecture
│
└── spec/                    ← ARTX11 proper
    ├── mod.rs
    ├── pairing.rs           PairingContract, VocabRelation (§2.3)
    ├── drafter.rs           DraftModel (an ordinary Session) (§3.3)
    ├── verify.rs            VerifyRule, rejection sampling, residual (§5.2)
    ├── hetero.rs            SLEM / TLI (§3.2) — Wave A11.6
    ├── step.rs              speculative_step (§5.3)
    └── policy.rs            SpecPolicy, γ ladder (§6.4)
```

## 7.2 Public API

```rust
impl Session {
    /// Attach a drafter. Validates the PairingContract ONCE, here —
    /// never at decode time.
    pub fn with_draft(
        &mut self, draft: DraftModel, policy: SpecPolicy,
    ) -> Result<(), PairingError>;

    /// Unchanged signature. Speculation is invisible to callers —
    /// ARTX16's glserve needs NO changes.
    pub fn generate(&mut self, prompt: &[u32], max_tokens: usize) -> Result<Vec<u32>, GlError>;
}
```

⚠️ **DESIGN DECISION — `generate()`'s signature does not change.**
Because rejection sampling is distribution-preserving (§1.2), a speculative `generate` is
observationally equivalent to a non-speculative one. ARTX16's `glserve`, ARTX12's harness, and glbench
all keep working untouched. Speculation is configuration, not a new API surface.

## 7.3 The hidden-state hook (door left open, not built)

Medusa/Hydra/EAGLE all need the target's intermediate hidden states (§2.2). Recording the minimal
shape so a future wave is not blocked by an API decision made here:

```rust
/// NOT implemented in ARTX11. Recorded so §2.2's families stay reachable.
pub trait HiddenStateSink {
    fn on_layer_output(&mut self, layer: usize, hidden: &Tensor);
}
```

---

# 8. Tradeoff Analysis

| Decision | Gain | Cost | Verdict |
|---|---|---|---|
| Independent draft model | No training; any model pair; simplest | Highest `c` of all drafter families | ✅ Only option under the constraints |
| γ compile-time constant | Full static-shape compatibility | γ multiplies compiled artifacts; no per-request γ | ✅ Non-negotiable given ARTX5/ARTX7 |
| Fixed trees allowed, dynamic excluded | Medusa-shape drafting stays reachable | EAGLE-2's best results unreachable | ✅ Static-shape criterion, not preference |
| Write all γ+1 KV, advance counter by n | No conditional writes, no shape change | Wasted KV writes for rejected tokens | ✅ The write is cheap; a shape change is not |
| Rejection sampling default | Provably lossless | Slightly lower α than typical acceptance | ✅ ARTX12's premise demands it |
| Rewind draft counter, don't clear KV | O(1); no allocation | Easy to forget → silent α collapse | ✅ With a mandatory ARTX12 test |
| `Architecture` as data, not traits | Serializable into the cache key; diffable | Adding a model means editing an enum | ✅ ARTX8's anti-over-engineering rule |
| Two KV slabs | Draft is an ordinary Session | Real HBM cost; `max_slots` must shrink | ⚠️ Plan it, don't discover it |
| γ ladder, not continuous γ | Adaptivity without recompilation | Coarse; the true optimum sits between rungs | ✅ Static shapes make this the only form |
| Cross-family via SLEM/TLI | "All models" achievable without training | Lower α; SLEM adds detokenization | ⚠️ Second-choice when same-family exists |

## 8.1 When speculation is a net loss

Stated plainly, because the failure cases are not obvious:

1. **α < ~0.4** — the draft is too dissimilar; drafting cost exceeds the tokens saved.
2. **c > ~0.3** — the draft is too expensive; §1.3's table caps speedup near 1.5× at best.
3. **MoE targets** — a 26B-A4B activates 3.8B/token, so `c` is computed against the *active*
   parameter count, not the total (§3.1). Most drafters are too big relative to it.
4. **High batch occupancy** — ARTX7 has already raised arithmetic intensity; speculation's headroom
   is gone but its cost remains (§6.4).
5. **Very long context** — attention cost grows with context while the drafting saving does not, so
   `c` drifts upward over a generation. ⚠️ Unmeasured for gljax; `c` should be re-estimated, not
   fixed at startup.

---

# 9. Wave Plan

| Wave | Scope | Gate |
|---|---|---|
| **A11.0** | `arch/` — `Architecture` descriptor, GeGLU, zero-centered RMSNorm, QK-norm, query scale, `LayerPattern`; ARTX3 retrofit; cache key gains `arch_hash` | ARTX12 harness green on a real **Gemma 4** checkpoint, not just Qwen |
| **A11.1** | `pairing.rs` + `drafter.rs` — load two Sessions, validate `Identical` vocab, measure `c` | Draft and target both generate correctly *independently* |
| **A11.2** | `verify.rs` + `step.rs` — chain speculation, γ fixed, greedy + rejection sampling | ⭐ Output distribution **statistically indistinguishable** from non-speculative over ≥10k tokens |
| **A11.3** | ARTX7 integration — per-slot advance, batched draft/verify, draft rewind | ARTX12 test: draft rewind produces context-identical proposals (§6.3) |
| **A11.4** | `policy.rs` — γ ladder, measured α, fallback to plain decode | Measured speedup within 20% of §1.3's prediction at the measured α and c |
| **A11.5** | Fixed-tree drafting (Medusa-shape structure, no trained heads) | Static-shape preserved; tree mask is a compile-time constant |
| **A11.6** | `hetero.rs` — SLEM / TLI cross-vocabulary | Losslessness holds across a genuine cross-family pair |

⚠️ **A11.2's gate is the load-bearing one.** Speculative decoding's entire value proposition is
"faster with *identical* output distribution." A statistical equivalence test over a large sample is
the only thing that verifies it — and per ARTX12 §4.1, that test must gate on **top-1 agreement and
KL**, not on raw logit L2.

⚠️ A11.0 is not optional preamble. Building speculation on an architecture layer that has never run a
second architecture would mean debugging the retrofit and the speculation simultaneously.

---

# Appendix A — Design Decision Summary

| # | Decision | Rationale | Reversible? |
|---|---|---|---|
| D1 | Tree line redrawn: **fixed** trees in scope, **dynamic** trees out | Static-shape criterion. A fixed tree is a chain plus a compile-time mask; a dynamic tree changes verified-position count per step | Hard |
| D2 | Independent draft model is the primary drafter | Only family implementable with training out of scope | Medium |
| D3 | Medusa/Hydra/EAGLE = research inputs, not targets | All require trained heads; gljax is inference-only with no gradient path | Medium |
| D4 | **γ is a compile-time constant**, a compile-cache key dimension | Makes verify `[γ+1, D]` static — the whole compatibility result (§5.1) | Hard |
| D5 | Write all γ+1 KV entries; advance the position counter by n | Absorbs the only dynamism into a runtime scalar ARTX5 already supports | Hard |
| D6 | `RejectionSampling` default; `TypicalAcceptance` opt-in and labelled lossy | ARTX12's premise is that silently-changed output is *the* bug class here | Trivial |
| D7 | Bonus token on full acceptance | Not an optimization — it is why E[tokens] reaches γ+1 | N/A |
| D8 | Rewind the draft's position counter; never clear its KV | O(1), no allocation; stale KV beyond the counter is already established as harmless | Trivial |
| D9 | Per-slot position advance; `KvSlotManager` and `StaticKvSlab` unchanged | ARTX7 Design Principle #3 paying off again | Trivial |
| D10 | γ uniform across a batch, not per-request | Per-request γ ⇒ ragged verify shapes ⇒ recompilation | Hard |
| D11 | Adaptive γ selects from a compiled **ladder** | Adaptivity without recompilation; the only form static shapes permit | Trivial |
| D12 | Plain decode is always a legal fallback | Speculation is an accelerator, never a dependency | N/A |
| D13 | Draft model is an ordinary `Session` | Inherits ARTX1–ARTX12 for free, including the correctness harness | Medium |
| D14 | **`Architecture` as a data descriptor, not a trait hierarchy** | Serializable into the cache key, diffable across draft/target; ARTX8's anti-over-engineering rule | Medium |
| D15 | `arch_hash` joins the compile-cache key | Without it, a Qwen draft and a Gemma target collide silently — shapes match | Hard |
| D16 | §4 lands as Wave A11.0, before any speculation | Otherwise the retrofit and the speculation are debugged together | N/A |
| D17 | Verifier interface designed for heterogeneous vocab from day one | Hetero verification works on strings/intersections, not aligned logits — cannot be retrofitted | Hard |
| D18 | `generate()` signature unchanged | Rejection sampling is distribution-preserving, so speculation is observationally invisible | Trivial |
| D19 | Hidden-state hook recorded but not built | Keeps Medusa/EAGLE reachable without building for them now | Trivial |
| D20 | ARTX12 is a hard prerequisite | Second model + new KV pattern + new sampling path = three unknowns without it | N/A |

---

# Appendix B — Formula Reference

```text
Acceptance rate        α(h) = 1 − ½‖p(·|h) − q(·|h)‖₁
Per-token acceptance   β(y) = min(1, p(y|h)/q(y|h))
Residual on reject     norm(max(0, p − q))
Expected tokens/iter   E    = (1 − α^(γ+1)) / (1 − α)
Wall-clock speedup     S    = (1 − α^(γ+1)) / ((1 − α)(γc + 1))
Cost ratio             c    = t_draft / t_target
Arithmetic intensity   AI(verify γ+1) ≈ γ+1     vs  AI(decode) ≈ 1     [ARTX8]
```

---

# Sources

- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/pdf/2211.17192) — Leviathan et al., 2023. γ drafting, β(y) = min(1, p/q), residual resampling, losslessness.
- [Aman's AI Journal — Speculative Decoding](https://aman.ai/primers/ai/speculative-decoding/) — α(h) = 1 − ½‖p − q‖₁; E[tokens] = (1−α^(γ+1))/(1−α); speedup = (1−α^(γ+1))/((1−α)(γc+1)).
- [MEDUSA: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/pdf/2401.10774) — heads as single FFN layer + residual; tree attention with ancestor-only mask; typical acceptance; Medusa-1 (frozen backbone) vs Medusa-2.
- [Hydra: Sequentially-Dependent Draft Heads for Medusa Decoding](https://arxiv.org/abs/2402.05109) — COLM 2024. Heads conditioned on previously sampled tokens; up to 1.31× over Medusa, 2.70× over autoregressive.
- [EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees](https://arxiv.org/pdf/2406.16858) and [EAGLE-3: Scaling up Inference Acceleration via Training-Time Test](https://arxiv.org/pdf/2503.01840) — feature-level autoregression reusing the target LM head; dynamic context-dependent trees; 2.1×–3.8× with preserved distribution.
- [Accelerating LLM Inference with Lossless Speculative Decoding Algorithms for Heterogeneous Vocabularies](https://arxiv.org/abs/2502.05202) — Timor et al., ICML 2025 Oral. Three lossless algorithms, off-the-shelf models, **no training**; SLEM and TLI upstreamed to HuggingFace as the heterogeneous default; up to 2.8×.
- [vLLM issue #38173 — Universal Speculative Decoding for Heterogeneous Vocabularies (TLI)](https://github.com/vllm-project/vllm/issues/38173) — current implementation status.
- [OmniDraft: A Cross-vocabulary, Online Adaptive Drafter](https://arxiv.org/html/2507.02659v1) — n-gram cache for cross-vocabulary token mapping; BPE context-dependent id assignment.
- [Gemma 3 Technical Report](https://arxiv.org/pdf/2503.19786) and [Gemma explained: What's new in Gemma 3](https://developers.googleblog.com/gemma-explained-whats-new-in-gemma-3/) — 4 RMSNorm per block, zero-centered `(1+weight)`, GeGLU `gelu_pytorch_tanh`, QK-norm, custom query scaling, scaled word embedding, 5 local : 1 global attention.
- [Gemma 4 Technical Report](https://arxiv.org/html/2607.02770v1) and [Gemma 4 model card](https://ai.google.dev/gemma/docs/core/model_card_4) — released 2 April 2026; E2B (2.3B eff.), E4B (4.5B eff.), 26B-A4B MoE (25.2B total / 3.8B active), 31B Dense (30.7B, 256K ctx); Apache 2.0.
- [Gemma explained: Gemma model family architectures](https://developers.googleblog.com/en/gemma-explained-overview-gemma-model-family-architectures/) — GeGLU lineage from Gemma 1.
- [Speculative Decoding in Production: Free Tokens and Hidden Traps](https://tianpan.co/blog/2026-04-17-speculative-decoding-production-hidden-traps) — production acceptance rates by workload type.

**Repo-internal:** `gljax/architecture/ARTX3-ops-layer-LLM-implementations.md` (SwiGLU-only ops
layer, Qwen2 assumptions); `ARTX5` (`dynamic_update_slice` at runtime position index, bucketing);
`ARTX7` (slot/position separation, stale-KV-is-harmless invariant); `ARTX8` (decode arithmetic
intensity, anti-over-engineering rule); `ARTX12` (correctness harness, gating).
