# glbench v3 — Design

**Codename:** Mensura Veritatis v3
**Status:** Design draft — awaiting JinXSuper sign-off on §13
**Component:** `glbench`
**Extends:** `glbench/DESIGN.md` (v1), `glbench/ROADMAP.md` "v2 — LLM Performance Doctor"
**Input:** the v3 research document (training observation, session modes, GLBitProf, null semantics, content digest)
**Last updated:** 2026-08-19

---

## 0. How to read this document

The research document answered *what* and *why*. This document answers *how*,
and it starts by correcting *what the research document assumed about this
repository*.

Section §1 is not optional context. It is the reason the rest of the document
looks different from the research document's §15 scope list. Every finding in
§1 was read out of the tree on 2026-08-19, with a file and line, not inferred
from a spec.

Three things this document is:

- A **decision record**. §3 locks twenty decisions, including the four open
  questions (OQ-1…OQ-4). Nothing from §4 onward re-litigates them.
- An **implementation spec**. Module paths, type names, and signatures are
  concrete enough to code against.
- A **wave plan with real gates** (§10). A gate is a STOP, not a checkpoint.

One thing it is not: a promise that the research document's schema is buildable
as written. Several parts are not, and §1 says which.

---

## 1. Grounding pass — where the research document does not match the tree

Per `gl-agent-skills/README.md` precedence: *measured production numbers →
`architecture/` specs → skills → anything else*. The research document is a
spec. The tree wins.

### F-01 — `gltrain` is not the training arm v3 should observe

The research document §10 states the chain `glbench → gltrain → glproc → glcore`
and calls `gltrain` an unconditional dependency "paralleling `glproc`".

Measured:

- `gltrain/Cargo.toml` declares `name = "gwenland-core"`, version `0.1.155`, and
  roughly thirty-five crates.io dependencies: `tokio`, `actix-web`, `reqwest`,
  `wgpu`, `candle-core = "0.9"`, `tokenizers`, `hf-hub`, `rayon`, `serde`,
  `regex`, `chrono`, `sysinfo`, and more.
- It declares **no dependency on `glcore` or `glproc`**. The chain in the
  research document does not exist in either direction.
- The root `Cargo.toml` `exclude`s `gltrain` with an explicit comment: it is
  excluded *because* candle "violates this workspace's Inference First rule of
  zero external ML dependencies", and keeping it excluded means
  `cargo build --workspace` "never touches candle".

Making `gltrain` a dependency of `glbench` — a root workspace member — would
undo that isolation on the first `cargo build --workspace`, pull candle,
actix-web and tokio into the inference tree, and break
`gl-agent-skills/architecture-skills/inference-first.md` rule 6 and
`glbench/DESIGN.md` §9 in the same commit.

**Consequence:** v3 observes **`stumman`**, not `gltrain`. Every occurrence of
"gltrain" in the research document reads as "stumman" for the rest of this
design. `gltrain` is out of v3's scope entirely.

### F-02 — `VLTrainingStep` does not exist, and nothing equivalent does

The research document §4 Q4 says gltrain "exposes a `VLTrainingStep` struct …
after each optimizer step" and that "gltrain pushes the measurements, glbench
receives and archives them."

Measured, `stumman/src/train/trainer.rs:187`:

```rust
pub fn train_step(&mut self, x: &Tensor<B>, target: &Tensor<B>) -> Result<f32>
```

It returns the loss scalar. Nothing else. `Trainer::train` (`:212`) returns
`Result<Vec<f32>>` — the mean loss per epoch. Grepping the crate finds
`VLTrainerConfig`, `VLMicroDataset`, `VLManifest`, `VLGradStore`,
`VLNamedTensor` — and no `VLTrainingStep`.

Of the roughly thirty fields in the research document §22 `TrainingStep` tree,
exactly **one** — loss — is obtainable from stumman today. The gradient store is
a local inside `train_step` and is dropped before the function returns; no
duration is measured anywhere; there is no FLOP counter.

**Consequence:** v3's training observer has **no data source**. The type must be
created and stumman must be instrumented to fill it. This is the largest
correction in this document, and §2 works through what it does to the shape of
the project.

### F-03 — `VLManifest` and `Optimizer::state_tensors` are real

The research document's other two named stumman reads check out:

- `VLManifest` — `stumman/src/checkpoint/manifest.rs`, reachable from
  `Trainer::manifest()` (`trainer.rs:237`). It is a *checkpoint* manifest
  (adapter config plus step count), not a training-run descriptor; v3 uses it
  for what it is.
- `Optimizer::state_tensors` — `stumman/src/optim/mod.rs:155`, returning
  `Result<Vec<VLNamedTensor>>`, implemented by `OPAdamW`, `OPAdafactor`,
  `OPLion`, and deliberately erroring on `OPAdamW8bit`.

### F-04 — the training observation boundary can be fully non-generic

Better news than the research document assumed, and it removes a constraint the
design would otherwise have had to work around.

`stumman/KNOWN_ISSUES.md` KL-001 records that `Backend` is not dyn-compatible
(the `Clone` supertrait alone is sufficient to exclude it), so `Box<dyn Backend>`
does not compile and `Trainer<B>` is generic. A naive observer API would have to
be generic over `B` and monomorphize into glbench.

But the two types that actually carry numbers across the boundary are both
**plain, non-generic f32 data**:

- `stumman/src/autograd/grad_store.rs:22` —
  `VLGradStore { grads: HashMap<TensorId, (Vec<f32>, Vec<usize>)> }`
- `stumman/src/optim/mod.rs:170` —
  `VLNamedTensor { name: String, data: Vec<f32>, shape: Vec<usize> }`

Only `TPParameter<B>` (`stumman/src/nn/param.rs:30`) is generic, and glbench
never needs the parameter itself — it needs the parameter's *values*, which
`state_tensors` already hands over as `VLNamedTensor`.

**Consequence:** the entire glbench↔stumman numerical surface is `&[f32]` +
shape + name. KL-001 does not block v3. GLBitProf gets one uniform input type
for weights, gradients, and optimizer state. Locked as D-07.

### F-05 — stumman M2 trains one linear layer, not a model

`VLTrainerConfig` (`trainer.rs:54`) is `{d_in, d_out, r, alpha, lr,
weight_decay, adapter_seed, base_seed}`. `Trainer` holds one `ABLinear<B>` and
one `LRLora<B>`. `VLMicroDataset` (`train/dataset.rs:20`) is
`Vec<(Vec<f32>, Vec<f32>)>` with `[1, d_in]` inputs. The loss is `mse_loss`.

There is no tokenizer in the loop, no batching beyond one sample, no
multi-layer model, and no text dataset.

**Consequence:** every token-denominated field in the research document's
training schema — `Tokens`, `Tokens/sec`, `Tokens-to-Target`, `Token Density` —
has **no subject** at M2. Same for `Synchronization` (single-threaded),
`Gradient Reduction` (no data parallelism), `Mixed Precision` (f32 only), and
`Gradient Accumulation` (not implemented). This is not a reason to cut them from
the schema; it is precisely what the null-semantics vocabulary exists to
express. See D-04.

### F-06 — a hand-rolled SHA-256 already exists in the workspace

`glictus-caliburni/src/checksum.rs:48` provides `sha256_bytes(&[u8]) -> String`
and `:33` provides `sha256_file`, both std-only, with a known-answer test
against the "hello" vector (`:174`).

But `glictus-caliburni` is optional in `glbench/Cargo.toml`, gated behind
`gllm-bench`. Archive integrity cannot be feature-gated — a digest that exists
only in some builds is not an integrity guarantee.

Copying the implementation into glbench would create a second independent
implementation of a hash, which is exactly the failure mode
`architecture/mensura-veritatis-v3/ARTX2-Quant.md` catalogues (seven independent
Q6_K decoders, one of them wrong for months) and which
`glbench/RESEARCH_REQUIREMENTS.md` explicitly forbids.

**Consequence:** the primitive moves to `glcore`, which both crates already
depend on unconditionally. Cross-crate work, owned by v3. See D-15 and §7.2.

### F-07 — canonical JSON is already guaranteed; the sentinel is safe

`glbench/src/export/json.rs:24` — `Json::Obj(BTreeMap<String, Json>)`.

Object keys live in a `BTreeMap`, so every object serializes in sorted key order
regardless of insertion order, and `to_pretty` is a deterministic function of
the value.

**Consequence:** OQ-4's concern about "field-ordering sensitivity in the
hand-rolled JSON serializer" does not apply — the serializer is already
canonical. The sentinel approach is sound, for a better reason than the one
given. Locked as D-16.

### F-08 — glbench's numerical modules are all behind `gllm-bench`

`glbench/src/lib.rs` gates `kl_divergence`, `ppl`, and `tensor_stats` behind
`#[cfg(feature = "gllm-bench")]`.

The research document §3 table lists `tensor_stats.rs` as "Extended into
GLBitProf". Doing that literally would make GLBitProf a `gllm-bench`-only
feature, which is wrong: gradient and optimizer-state bit profiling has nothing
to do with `.gllm` packages.

**Consequence:** GLBitProf's math is a new, **ungated** module over `&[f32]`.
Its *sources* are gated per source. See D-11 and §6.4.

### F-09 — the archive already has a null convention; v3 refines it

`glbench/src/core/session.rs` documents and implements the rule already:
`telemetry: None` means "*not measured*, never *zero*", and `behavior_json`
carries the comment that a CI job asserting on a signal "must fail loudly on a
run that never measured repetition, not silently pass on a fabricated 0.0."

**Consequence:** the eight-value vocabulary is an *upgrade of a working
convention*, not a new invention, and it must be introduced without throwing
away the existing shape. This drives §6.1, which differs from the research
document's implied per-field wrapper.

### F-10 — `architecture/mensura-veritatis-v3/` is already taken

That directory has existed since 2026-07-23 and holds a different project: the
gl-stack correctness audit triggered by the Q6_K dequant bug (`ARTX1-Arsitektur`,
`ARTX2-Quant`, `ARTX3-Format`, `ARTX4-Benchmark`).
`glbench/RESEARCH_REQUIREMENTS.md` cites `ARTX2-Quant.md` by path.

**Consequence:** this design lands at `architecture/glbench-v3/`. Whether the
older series gets renamed is a call for JinXSuper — §13, item 5.

### F-11 — the naming convention applies, and v3 is its largest batch yet

`gl-agent-skills/gwenland-naming-convention/SKILL.md` measured adoption at
**0 of 224 public types** on 2026-08-16, and states the rule plainly: new types
follow the convention; existing types are not renamed as a drive-by.

v3 introduces on the order of thirty new public types in glbench — the largest
single application of the convention since it was written, in a crate that
currently has zero prefixed types.

**Consequence:** every new v3 type carries a prefix (§5). `BenchmarkSession`,
`SessionMetadata`, `MeasurementSet`, `AnalysisReport` and friends are **not**
renamed. The mixed file is expected and correct per the skill's own wording.

---

## 2. What §1 changes about the shape of v3

The research document scoped v3 as a glbench project with a dependency edge
added to a training crate. F-02 and F-05 say otherwise.

Measured against the research document §22 `TrainingStep` tree:

| Field group | Available from stumman today | Needs |
|---|---|---|
| Loss value | ✅ `train_step` return | — |
| Step index | ⚠️ `step_count()` only; epoch not exposed | stumman: expose epoch |
| Forward / backward / optimizer duration | ❌ nothing measured | stumman: instrumentation |
| Forward / backward FLOPs | ❌ no counter | glbench: derive from shapes |
| Gradient norm / mean / variance / sparsity | ❌ `VLGradStore` dropped inside `train_step` | stumman: expose at hand-off |
| Gradient NaN / Inf / overflow | ❌ same | stumman: expose at hand-off |
| Optimizer state | ⚠️ `state_tensors` exists as a checkpoint path | stumman: none — reuse |
| Parameter delta | ❌ not tracked | stumman: expose, or glbench diffs |
| Memory (param / grad / activation / peak) | ❌ nothing measured | stumman: instrumentation |
| Tokens, tokens/sec, tokens-to-target | ❌ no tokens exist at M2 | `not_applicable` |
| Synchronization / communication | ❌ single-device | `not_applicable` |

So: **v3's training observation is roughly 85% a stumman instrumentation project
and 15% a glbench schema project.** That is not a reason to shrink the ambition.
It is a reason to sequence and gate it, because the glbench half is unbuildable
and untestable until the stumman half lands.

Three structural consequences, locked in §3:

1. **Wave 3 is stumman work, not glbench work** (D-01). glbench's training
   modules cannot be written against a data source that does not exist. The
   alternative — glbench timing `train_step` from outside with `Instant` — buys
   only a wall-clock total and cannot produce the phase attribution that is the
   entire point of `TrainingAttribution`.

2. **The training dependency is feature-gated, not unconditional** (D-02),
   reversing the research document §10. Its own argument for gating
   `glictus-caliburni` — "a build without it is a complete, useful glbench" —
   applies with more force here, because glbench lives in the inference
   workspace and stumman deliberately does not.

3. **The parts of v3 with no training dependency ship first** (D-03). Null
   semantics, the content digest, the session envelope, GLBitProf's math and its
   weights scope, and the join manifest are all buildable today against the tree
   as it stands. They are ordered ahead of the training work so v3 delivers
   value before the stumman instrumentation lands.

---

## 3. Locked decisions

Twenty decisions. Everything from §4 onward implements these; nothing
re-litigates them.

### Dependency and structure

**D-01 — v3 observes `stumman`; `gltrain` is out of scope.**
Rationale: F-01. `gltrain` is candle-based, workspace-excluded, and has no edge
to glproc/glcore. If GwenLand later wants gltrain observed, that is a separate
project with a separate dependency story, not a v3 wave.

**D-02 — the stumman dependency is optional, behind a `train-bench` feature.**

```toml
stumman = { path = "../stumman", optional = true }

[features]
train-bench = ["dep:stumman"]
```

Rationale: F-01 plus stumman's own `Cargo.toml`, which declares an empty
`[workspace]` specifically so the inference tree never builds it. An optional
dependency preserves that: `cargo build --workspace` without the feature never
compiles stumman, exactly as `gllm-bench` works for `glictus-caliburni`. This
reverses the research document §10.

Accepted cost, stated rather than hidden: stumman and `anyhow` enter the root
`Cargo.lock` as optional entries even when unbuilt, and stumman may resolve its
own dependency versions differently standalone (own lock file) than from the
root. Neither affects a default build. Recorded so a future reader does not
discover it as a surprise.

**D-03 — v3 ships in five waves, ordered so no wave blocks on the next.**
Waves 1, 2, and 5 have no training dependency at all. See §10.

**D-04 — schema completeness is decoupled from M2 capability.**
The full training schema from the research document is defined now, including
the token-denominated and multi-device fields that F-05 says have no subject.
They are populated with `not_applicable` (M2 has no tokens) or `unsupported`
(M2 is single-device), never omitted and never zero. This is the null-semantics
vocabulary earning its place on its first real use.

**D-05 — glbench does not drive training; it observes a run that stumman
drives.**
`glbench train` constructs a `Trainer`, installs an observer, and calls
`Trainer::train`. It never calls `train_step` in a loop of its own, never
touches optimizer state, and never writes a parameter. The `DESIGN.md` §1
boundary is unchanged.

### Session model

**D-06 — `ENSessionMode` has three variants, not four.**

```rust
pub enum ENSessionMode { InferenceOnly, TrainingOnly, Unified }
```

The research document §13 lists four modes but then states "A `JoinedSession` is
NOT a `BenchmarkSession`". Both cannot hold. A join is a *derived artifact over
two sessions*, so it is its own top-level type with its own schema (`VLJoinManifest`,
§6.6) and no mode variant. Resolves the internal contradiction in favour of the
part the research document argued for explicitly.

**D-07 — the glbench↔stumman boundary is non-generic.**
Per F-04. The observer trait, `VLTrainingStep`, and every numerical payload are
free of `B: Backend`. Consequence: KL-001 never reaches glbench, and GLBitProf
takes one input type everywhere.

**D-08 — nested inference sessions carry an explicit role.**
The research document's nesting (post-training `InferenceSession` *inside*
`TrainingSession`, pre-training one beside it) is kept as specified. But two
`VLInferenceSession` values distinguished only by position is a footgun for
every consumer, so each carries `role: ENInferenceRole { Standalone,
PreTraining, PostTraining }`. Additive; the nesting decision stands.

Recursion guard: `VLTrainingSession` may contain `VLInferenceSession`;
`VLInferenceSession` may **not** contain `VLTrainingSession`. Enforced by the
type graph, which is acyclic by construction.

### Null semantics

**D-09 — availability is a sparse exception map, not a per-field wrapper.**
Full design in §6.1. One `availability` block per session maps dotted field
paths to status values, listing only fields that are *not* plainly measured.
Chosen over wrapping every value in `{value, status}` because it keeps existing
field shapes and types intact (F-09), costs bytes only for actual exceptions,
and stays readable by hand.

**D-10 — a `null` with no availability entry is a defect.**
The invariant that makes D-09 work, and it is checkable: `validation::availability`
walks the emitted JSON, collects every `null`, and fails the session if any lacks
an entry. Without this the sparse map degrades into optional documentation. With
it, "we forgot to say why this is null" is a test failure.

### GLBitProf

**D-11 — GLBitProf's math is ungated; its sources are gated per source.**
`numerical/bitprof.rs` is pure `&[f32] → VLBitProfile`, std-only, no feature
gate (F-08). Weights arrive via `gllm-bench`, gradients and optimizer state via
`train-bench`, activations not at all (deferred, §12).

**D-12 — mantissa is profiled at full 23-bit resolution via a sparse map, and
only when that map can be complete; the exponent histogram stays dense.**

The research document specifies "Shannon entropy over the histogram of mantissa
values". A dense full-resolution histogram is 2²³ = 8,388,608 buckets — 67 MB at
`u64`, 33 MB at `u32` — beyond any L3, turning the pass into a random-access
cache-miss generator rather than the linear scan the design assumes. Empirical
research (ZipNN, ENEC, DFloat11) confirms FP32 mantissa bytes sit at near-maximum
entropy (~7.97 of 8 bits/byte), so the field also cannot be compressed away.

12-bit bucketing was the first mitigation. It was rejected on review for losing
the fine-grained signal a researcher wants. v3 instead uses a **hybrid, three-tier
design.**

**Tier 1 — exponent, dense.** `exponent_histogram: Box<[u64; 256]>`, 2 KiB,
always collected in full. Exponent entropy is low (~2.6 of 8 bits allocated), so
this is both cheap and the primary distribution signal.

**Tier 2 — mantissa, sparse and full-resolution.**
`mantissa_sparse: Option<HashMap<u32, u64>>`, keyed by the raw 23-bit mantissa
(bits 22..=0). Exact, no bucketing, no coarsening. Capped at
`MANTISSA_SPARSE_CAP = 131_072` entries (~512 KiB of pairs).

**Tier 3 — per-position bit entropy, exact, always.** `bit_entropy: [f64; 32]`
covers all 23 mantissa bit positions at a cost of 32 numbers, and is never
truncated, sampled, or estimated.

**Where this design departs from the reviewed proposal, and why.**

The proposal had the cap trigger a mid-collection abort: fill the map until it
hits 131,072 entries, stop, and set `mantissa_sparse_truncated = true`. Measured,
that abort fires on essentially every real tensor, and the resulting map is worse
than no map at all.

Assuming a near-uniform mantissa — which the ~7.97 bits/byte figure cited above
*is the evidence for* — the expected number of distinct mantissa values populated
by `n` elements is `m·(1 − e^(−n/m))` with `m = 2²³`:

| Tensor elements | Distinct mantissa values | vs. 131,072 cap |
|---|---|---|
| 65,536 (LoRA A, r=16, d=4096) | 65,281 | 0.5× — fits |
| **132,107** | **131,072** | **1.0× — cap reached here** |
| 1,048,576 | 985,687 | 7.5× over |
| 16,777,216 (one 4096² matrix) | 7,253,333 | 55× over |

So the cap binds at ~132 K elements. Three consequences follow, and each is
disqualifying on its own:

1. **The retained map is order-biased, not a sample.** It is "the first 131,072
   distinct mantissa values encountered in traversal order" — a function of how
   the tensor is laid out, not of its distribution. Two runs over the same
   weights in different orders disagree.
2. **The entropy it yields is a function of tensor size, not of the tensor.** A
   plug-in entropy estimate is bounded above by `log₂(min(n, m))`: 17.0 bits at
   n = 131 K, 20.0 bits at n = 1 M, against a true value near 23. The number
   moves when the tensor grows and stands still when its precision usage changes
   — the precise definition of a confident wrong answer.
3. **The archive cannot hold it.** 131,072 entries serialise to roughly 2 MB of
   JSON per tensor. Qwen2.5-0.5B has on the order of 290 tensors: **~570 MB**,
   against `DESIGN.md` §8's single-JSON-file archive and the 10–500 KB figure the
   digest cost analysis in §6.5 rests on.

**The guard.** The cap is therefore a **precondition, not an abort**. Before
profiling a tensor, `profile()` compares its element count against
`MANTISSA_SPARSE_CAP`. If the map could not be complete, it is **not collected at
all**: `mantissa_sparse` is `None`, `mantissa_entropy_bits` is `None`, and the
archive carries `numerical.…​.mantissa_sparse` → `unavailable` in the availability
map (§6.1) — a first-class "the machine could produce this, it was not collected
this run", which is exactly the vocabulary D-04 exists for.

This keeps the reviewed proposal's stated goal — *"full 23-bit resolution when
distribution allows; graceful degradation when not"* — and changes only *when the
decision is made*. Deciding before collection rather than during it removes the
order bias, removes the saturated estimator, removes the 570 MB archive, and
costs one comparison per tensor.

**What this means in practice.** Small tensors — LoRA A/B matrices, norms,
biases, router projections — get an exact, unbucketed, full 23-bit mantissa
distribution, which is where a researcher most wants it and where the LoRA
gradient-health case (research document §12 Case 3) actually lives. Large weight
matrices report Tier 1 and Tier 3, which are complete and exact for them, and say
plainly that Tier 2 was not collected.

`mantissa_entropy_bucket_bits` is removed: there is no bucketing left to
describe.

**D-13 — GLBitProf's cost is measured before it is claimed.**
The research document §6 Q7 estimates "~500M bit operations — approximately
50ms" for a 0.5B model. That figure is not reproducible from the description:
each element needs a load, three mask/shift extractions, and three histogram
increments, so the real op count is nearer 5G, and a 50 ms wall time implies
~100 Gop/s single-threaded. The Wave 2 gate requires a measured number from
`glbench/benches/` on the production path before any doc states a cost. Repo
history is unambiguous that guessed throughput figures produce confident wrong
answers here.

**D-14 — `--bit-scope` is explicit, defaulting to `weights` (OQ-2).**
`--profile bits` without `--bit-scope` profiles weights only. Any other scope is
opt-in by name: `--bit-scope weights,gradients,optimizer`. Rationale as in OQ-2:
an O(n_elements) pass over every tensor of a large model should never be
something a user gets by accident. `activations` is a recognised token that
errors with "not implemented in v3, see DESIGN §12" rather than being silently
absent.

### Integrity

**D-15 — SHA-256 moves to `glcore::hash`; `glictus-caliburni` calls it.**
Per F-06. Not a copy — a move, with `glictus-caliburni::checksum` re-exporting
for its existing callers. Net implementation count for SHA-256 in the workspace
stays at one.

**D-16 — the digest is SHA-256 truncated to 128 bits, sentinel-based (OQ-4).**
Algorithm identifier `"sha256-128"`, carried in the archive so a future native
GwenLand 128-bit primitive can replace it without a schema change. Sentinel is
32 ASCII `0` characters. Safe because of F-07 (`BTreeMap` ordering makes the
serializer canonical), not merely because the sentinel is fixed-width.

Naming note, adopting the correction from the TRD discussion: "SHA-128" is not a
standard algorithm name and does not appear anywhere in v3. The field says
`sha256-128` and the docs say "128-bit content digest".

**D-17 — write-once is advisory; the digest is the guarantee.**
After writing, glbench sets the file read-only via
`std::fs::Permissions::set_readonly(true)` — std-only and cross-platform, which
matters because `PermissionsExt`/`chmod` is Unix-only and this repo's primary
development machine is Windows. But a read-only flag is trivially cleared, so it
is documented as a guard against *accident*, never against *modification*. The
integrity claim rests entirely on the digest.

### Join

**D-18 — a join is a third file referencing two immutable sources (OQ-1).**
`VLJoinManifest` holds source paths plus each source's content digest, and the
comparison report. Neither source session is read-modify-written. Verification
re-hashes both sources and reports a mismatch as a first-class finding. Three
files for one logical comparison is the cost; source sessions remaining
independently verifiable is what is bought.

### Sampling

**D-19 — every step is archived by default; `--step-sample N` thins it, and
endpoints are always kept (OQ-3).**
Default `N = 1`. With `N > 1`, glbench archives steps where `index % N == 0`
**plus the first and last step unconditionally**. The refinement over OQ-3's
option (b): time-to-target, plateau detection, and stability CV all read the
endpoints, and a run whose last step happens not to land on a multiple of `N`
would otherwise lose the one step that says how training ended. `VLTrainingSession`
records `step_sample_n` and `steps_observed` next to `steps_archived`, so a
consumer can never mistake a thinned series for a complete one.

### Schema

**D-20 — `SCHEMA_VERSION` goes 1 → 2, and v3 reads v1 archives.**
The envelope change is breaking for writers. Readers are not: `storage::archive::read`
already refuses only *newer* schemas, so a v3 build reads a v1 archive by
treating the absent `session_mode` as `InferenceOnly` and the absent
`availability` block as empty. A v1 build refuses a v2 archive, correctly and
with the existing error message. No migration tool is written; archives are
user-managed files (`DESIGN.md` §8) and a v1 archive stays valid as v1.

---

## 4. Module map

New modules and extensions. Paths are under `glbench/src/`.

```
core/
  session.rs        [EXTEND]  BenchmarkSession gains mode, inference, training,
                              availability, integrity
  result.rs         [EXTEND]  SessionMetadata gains session_mode, host_id,
                              collection_profile
  schema.rs         [EXTEND]  SCHEMA_VERSION 1 -> 2
  availability.rs   [NEW]     ENAvailability, VLAvailabilityMap
  mode.rs           [NEW]     ENSessionMode, ENInferenceRole
  inference.rs      [NEW]     VLInferenceSession (wraps today's fields)

numerical/          [NEW dir] bit-level observation, ungated math
  mod.rs
  bitprof.rs        [NEW]     VLBitProfile + profile(&[f32]) -> VLBitProfile
  scope.rs          [NEW]     ENBitScope, scope parsing and dispatch
  compare.rs        [NEW]     bit-profile divergence between two profiles

training/           [NEW dir] all gated: #[cfg(feature = "train-bench")]
  mod.rs
  session.rs        [NEW]     VLTrainingSession
  step.rs           [NEW]     VLTrainingStep (glbench's archived form)
  collector.rs      [NEW]     implements stumman's StepObserver
  attribution.rs    [NEW]     VLTrainingAttribution (phase breakdown)
  convergence.rs    [NEW]     VLConvergence: slope, EMA, plateau, targets, CV
  memory.rs         [NEW]     VLTrainingMemory
  adapter.rs        [NEW]     VLAdapterObservation (type, rank, alpha, cost)
  runner.rs         [NEW]     drives Trainer::train under observation

storage/
  archive.rs        [EXTEND]  digest-on-write, verify-on-read, read-only flag
  digest.rs         [NEW]     VLIntegrity, sentinel hashing, verification
                              (NOT `integrity.rs` — `validation/integrity.rs`
                              already exists and means something else)
  join.rs           [NEW]     VLJoinManifest read/write

validation/
  availability.rs   [NEW]     D-10 invariant: no null without a status

analysis/
  roofline.rs       [EXTEND]  training FLOP roofline (same classifier)
  bottleneck.rs     [EXTEND]  ENTrainingBottleneck variants
  hypothesis.rs     [EXTEND]  training-signal patterns

comparison/
  training.rs       [NEW]     training-configuration comparison

render/
  loss_curve.rs     [NEW]     ASCII loss curve
  flamegraph.rs     [EXTEND]  training step phase breakdown

export/
  json.rs           [EXTEND]  availability + integrity blocks
  markdown.rs       [EXTEND]  training sections
  csv.rs            [EXTEND]  per-step rows
```

Cross-crate (§7):

```
glcore/src/hash.rs        [NEW]  sha256_bytes, sha256_file, sha256_128_hex
glictus-caliburni/
  src/checksum.rs         [EDIT] delegate to glcore::hash, keep re-exports
stumman/src/train/
  observe.rs              [NEW]  VLTrainingStep, trait StepObserver
  trainer.rs              [EDIT] optional observer, phase timing, epoch index
stumman/src/autograd/
  grad_store.rs           [EDIT] add iter() so an observer can enumerate
```

Per `DESIGN.md` §10, all of this stays inside the single `glbench` crate. v3
adds directories, not sub-crates.

---

## 5. Type inventory

Naming per `gl-agent-skills/gwenland-naming-convention/SKILL.md` (F-11). New
types take a prefix; existing types are untouched.

| Type | Prefix rationale | Module |
|---|---|---|
| `ENSessionMode` | closed variant set is its whole job | `core::mode` |
| `ENInferenceRole` | same | `core::mode` |
| `ENAvailability` | same | `core::availability` |
| `ENBitScope` | same | `numerical::scope` |
| `ENTrainingBottleneck` | same | `analysis::bottleneck` |
| `VLAvailabilityMap` | plain data, no identity | `core::availability` |
| `VLInferenceSession` | plain data | `core::inference` |
| `VLTrainingSession` | plain data | `training::session` |
| `VLTrainingStep` | plain data | `training::step` (+ stumman origin) |
| `VLTrainingAttribution` | plain data | `training::attribution` |
| `VLTrainingMemory` | plain data | `training::memory` |
| `VLConvergence` | plain data | `training::convergence` |
| `VLAdapterObservation` | plain data | `training::adapter` |
| `VLBitProfile` | plain data | `numerical::bitprof` |
| `VLBitDivergence` | plain data | `numerical::compare` |
| `VLIntegrity` | plain data | `storage::digest` |
| `VLJoinManifest` | plain data | `storage::join` |
| `VLJoinSource` | plain data | `storage::join` |
| `StepObserver` | **trait — no prefix** (rule 2) | `stumman::train::observe` |

Two names deserve a note.

`VLTrainingStep` is the research document's own name, and it is
convention-correct. It simply did not exist (F-02). v3 creates it.

`StepObserver` takes no prefix because rule 2 is unambiguous: traits get no
prefix, and the `trait` keyword has already said it. `ITStepObserver` and
`AGStepObserver` are both wrong.

Types *not* renamed, listed so nobody does it as a drive-by: `BenchmarkSession`,
`SessionMetadata`, `MeasurementSet`, `IterationMetrics`, `WorkloadSpec`,
`EngineMetadata`, `EnvironmentSnapshot`, `AnalysisReport`, `ComparisonReport`,
`ValidationReport`, `Json`, `Stats`, `Bottleneck`, `Severity`, `Finding`.

---

## 6. Core designs

### 6.1 Availability — null semantics (D-09, D-10)

**The vocabulary.**

```rust
pub enum ENAvailability {
    Measured,       // an instrument produced this value
    Estimated,      // modelled from measurements; `note` names the model
    Derived,        // computed from other measured fields; `note` names the formula
    Unsupported,    // platform or runtime cannot produce it
    Unavailable,    // it could exist, but was not collected this run
    NotApplicable,  // meaningless for this session type
    NotObserved,    // the event did not occur in the observation window
    DoesNotExist,   // the architectural feature is absent
}
```

**The shape.** Not a per-field wrapper. One sparse map per session, keyed by
dotted path:

```json
{
  "metadata": { "session_mode": "training_only", "...": "..." },
  "environment": {
    "hardware": { "gpu": { "peak_bandwidth_gbs": null } }
  },
  "training": { "steps": [ { "tokens": null, "loss": 0.4127 } ] },
  "availability": {
    "environment.hardware.gpu.peak_bandwidth_gbs": "unsupported",
    "environment.hardware.thermal.start_mhz":      "unavailable",
    "inference":                                    "not_applicable",
    "training.steps[].tokens":                      "not_applicable",
    "training.steps[].sync_ms":                     "not_applicable"
  }
}
```

Four properties this buys:

1. **Field shapes are unchanged.** `peak_bandwidth_gbs` is still a number or
   `null` at its existing path. Every v1 consumer that reads a value keeps
   working; only the *explanation* is new. Wrapping each value in
   `{value, status}` would have broken every reader for a benefit that only
   matters on the exception path.
2. **Cost scales with exceptions, not with fields.** A fully-measured inference
   session carries an almost-empty map. The research document's Q7 answer
   ("one integer per field") is the wrapper design's cost; this is cheaper.
3. **Repeated array elements collapse.** `training.steps[].tokens` states the
   status once for a thousand steps. A per-field wrapper would repeat it a
   thousand times.
4. **It is checkable** — which is D-10.

**Notes for `Estimated` / `Derived`.** Those two variants are required by the
research document to document their model or formula. A bare status string
cannot. So the map value is a string *or* an object:

```json
"analysis.roofline.ceiling_gbs": {
  "status": "estimated",
  "note": "device capability table, engine::capability::lookup"
}
```

The string form is sugar for `{"status": s}`. Parsers accept both; the writer
emits the short form whenever there is no note, and is **required** to emit a
note for `Estimated` and `Derived`.

**The D-10 check.** `validation::availability::check(session_json)`:

1. Walk the emitted JSON, collecting the dotted path of every `null`.
2. Normalise array indices to `[]`.
3. Any collected path with no entry in `availability` produces a
   `Severity::Error` finding naming the exact path.
4. Any `availability` entry whose path holds a **non-null value** produces a
   finding too — a status of `unsupported` on a field that carries a number is
   the same class of lie in the other direction.

Runs in-process at session finalisation, so a malformed archive is never
written. Also exposed as `glbench validate --availability <archive>` for
archives written by an older build.

### 6.2 Session envelope (D-06, D-08)

```rust
pub struct BenchmarkSession {
    // --- v1 fields, unchanged ---
    pub metadata: SessionMetadata,
    pub environment: EnvironmentSnapshot,
    pub engine: EngineMetadata,
    pub workload: WorkloadSpec,
    pub measurements: MeasurementSet,
    pub telemetry: Option<glcore::telemetry::EngineTelemetry>,
    pub behavior: Option<crate::behavior::BehaviorReport>,
    pub analysis: Option<AnalysisReport>,
    pub comparison: Option<ComparisonReport>,
    pub validation: Option<ValidationReport>,

    // --- v3 ---
    pub inference: Option<VLInferenceSession>,
    #[cfg(feature = "train-bench")]
    pub training: Option<VLTrainingSession>,
    pub availability: VLAvailabilityMap,
    pub integrity: Option<VLIntegrity>,
}
```

**Why the v1 fields stay where they are.** The research document's §4 tree
implies moving `measurements`, `telemetry`, and `behavior` down inside an
`InferenceSession`. That is a rewrite of every module that fills or reads them —
`runner`, `analysis`, `comparison`, `validation`, all three exporters, both
renderers — for a purely cosmetic gain, and it contradicts the research
document's own §3 promise that "nothing above is rewritten".

Instead, `VLInferenceSession` is the envelope for inference facts that are
*new* in v3 or that need a role (D-08):

```rust
pub struct VLInferenceSession {
    pub role: ENInferenceRole,
    pub measurements: Option<MeasurementSet>,   // Some only for nested sessions
    pub behavior: Option<BehaviorReport>,
    pub analysis: Option<AnalysisReport>,
}
```

For `ENSessionMode::InferenceOnly`, `inference` is
`Some(VLInferenceSession { role: Standalone, .. })` with its own fields `None`,
and the real data stays in the top-level v1 fields. For `Unified`, the outer
`VLInferenceSession` carries `role: PreTraining` and *does* populate its own
`measurements`, while `training.post_eval` carries `role: PostTraining`.

Blunt about the cost: this is a compatibility compromise. A greenfield schema
would nest everything. The v1 fields stay top-level because moving them is a
large, risky, zero-information-gain refactor, and `DESIGN.md` §4's "single
source of truth" is served either way. Recorded here so a future maintainer
knows it was chosen rather than overlooked.

**Mode consistency**, enforced at finalisation and tested:

| Mode | `inference` | `training` | availability entries required |
|---|---|---|---|
| `InferenceOnly` | `Some` (`Standalone`) | `None` | `training` → `not_applicable` |
| `TrainingOnly` | `None` | `Some` | `inference` → `not_applicable`, plus the v1 inference fields |
| `Unified` | `Some` (`PreTraining`) | `Some` (with `post_eval`) | — |

### 6.3 The training observation boundary (D-05, D-07)

**Direction.** `glbench → stumman`. stumman never imports glbench, transitively
or otherwise. A cycle fails `cargo check`, which is the enforcement.

**stumman side** — new file `stumman/src/train/observe.rs`:

```rust
/// One step, as facts. No `B`: everything here is a scalar or a count,
/// so an observer never needs to know the backend. See KL-001.
pub struct VLTrainingStep {
    pub index: usize,
    pub epoch: usize,
    pub loss: f32,
    pub forward_ns: u64,
    pub backward_ns: u64,
    pub optimizer_ns: u64,
    pub total_ns: u64,
    pub grad_count: usize,
    pub grad_elements: usize,
    pub grad_l2_norm: f64,
    pub grad_nan: usize,
    pub grad_inf: usize,
    pub lr: f64,
}

/// Watches a training run. Traits take no prefix (naming rule 2).
pub trait StepObserver {
    /// Called once per optimizer step, after the update.
    fn on_step(&mut self, step: &VLTrainingStep);

    /// Whether this observer wants the O(n) tensor payload. Default `false`
    /// so the expensive path is opt-in, not opt-out.
    fn wants_tensors(&self) -> bool { false }

    /// Gradients and optimizer state as flat f32, only when `wants_tensors`.
    /// Both are already non-generic in stumman (F-04).
    fn on_tensors(&mut self, _grads: &VLGradStore, _opt_state: &[VLNamedTensor]) {}
}
```

`Trainer` gains `set_observer(&mut self, obs: Box<dyn StepObserver>)`. Object
safety holds — no generics, no `Self`-by-value, no `Clone` supertrait — so
KL-001 does not reach this trait.

**Zero cost when unobserved.** `train_step` checks `self.observer.is_some()`
once. When `None`, no `Instant::now()` is called, no gradient statistic is
computed, and the step runs byte-identically to today. The measured overhead
when observed (three `Instant` pairs plus one pass over the gradient buffers) is
a Wave 3 gate deliverable, not an assumption.

**Placement of the gradient read.** Gradient statistics are computed in
`train_step` at the KL-006 hand-off point, between `finish_step()` and
`optimizer.step()`, where the gradients are live and the tape is already empty.
This is the only correct window: before `finish_step` the tape is live and
KL-006 applies; after `optimizer.step` the parameters have moved and the
gradients are gone.

**`VLGradStore` needs an iterator.** Today it exposes `get(id)`, `take(id)`,
`len`, `contains` — an observer cannot enumerate. v3 adds
`iter(&self) -> impl Iterator<Item = (TensorId, &[f32], &[usize])>`. Additive,
no behaviour change.

**glbench side.** `training::collector::VLStepCollector` implements
`StepObserver`, applies D-19 sampling, and accumulates into `VLTrainingSession`.
`wants_tensors()` returns true only when `--bit-scope` names `gradients` or
`optimizer`, so a plain `glbench train` never pays the O(n) cost.

### 6.4 GLBitProf (D-11, D-12, D-13, D-14)

**One function, one input type.**

```rust
pub fn profile(values: &[f32]) -> VLBitProfile;
```

Weights (dequantised via the existing `tensor_stats.rs` decode path), gradients
(`VLGradStore` values), and optimizer state (`VLNamedTensor::data`) all arrive
as `&[f32]` (F-04), so there is exactly one implementation.

```rust
pub struct VLBitProfile {
    pub count: u64,

    // sign, bit 31
    pub sign_set_ratio: f64,

    // exponent, bits 30..=23, biased by 127
    pub exponent_histogram: Box<[u64; 256]>,
    pub exponent_min: u8,
    pub exponent_max: u8,
    pub dynamic_range_used: f64,        // (max - min) / 254, derived

    // mantissa, bits 22..=0 — sparse, exact, full 23-bit. See D-12.
    // `None` when the element count exceeded the cap, so the map could not have
    // been complete. Not a truncated map: no map. The availability entry says
    // `unavailable`.
    pub mantissa_sparse: Option<HashMap<u32, u64>>,
    /// Always `MANTISSA_SPARSE_CAP`. Archived so the threshold that decided
    /// collection is visible in the data, not only in this build's source.
    pub mantissa_sparse_cap: u32,
    /// True when `count > mantissa_sparse_cap`, i.e. Tier 2 was skipped.
    /// Kept as a plain bool so a consumer never has to infer "why is this None".
    pub mantissa_sparse_skipped: bool,
    /// Exact Shannon entropy over `mantissa_sparse`, in bits, max 23.
    /// `None` whenever the map is `None` — use `bit_entropy` instead, which is
    /// always complete.
    pub mantissa_entropy_bits: Option<f64>,

    // per bit position, all 32
    pub bit_set_fraction: [f64; 32],
    pub bit_entropy: [f64; 32],

    // exact counts
    pub zero_count: u64,        // +0.0 and -0.0 counted separately below
    pub negative_zero_count: u64,
    pub subnormal_count: u64,
    pub nan_count: u64,
    pub inf_count: u64,
}
```

**Two definitional choices, stated because they change the numbers.**

*Zeros.* `-0.0` has bit 31 set and would otherwise inflate `sign_set_ratio` for
a freshly-zero-initialised LoRA `B` matrix — precisely the tensor the research
document §12 Case 3 wants to watch. Counted separately.

*NaN and Inf.* Excluded from `exponent_min`/`exponent_max` and from the mantissa
entropy, counted in their own fields. A single NaN otherwise pins `exponent_max`
at 255 and makes `dynamic_range_used` meaningless. They are still included in
`bit_set_fraction`, which is a raw per-position count and should stay raw.

**Per-position bit entropy.** For bit position `i`, let `p` be the fraction of
values with that bit set. `bit_entropy[i] = -p·log₂(p) − (1−p)·log₂(1−p)`, with
`0·log₂0 = 0`. Range `[0, 1]` bits. A position that is always clear scores 0 —
"this bit carries no information in this tensor" — which is the actual signal for
"precision is being wasted", and it is exact over all 23 mantissa bits at a cost
of 32 numbers.

**Divergence** (`numerical::compare`) answers the research document §12 Case 2:
`VLBitDivergence` between two profiles as per-position bit-fraction delta,
exponent-histogram L1 distance, and mantissa-entropy delta. This is what
distinguishes "the residual KL comes from the quantisation scheme" (smooth
exponent shift, mantissa entropy drop) from "the residual comes from a second
wrong-nibble-order bug" (structured per-position anomaly).

**Cost.** Per D-13, unmeasured. Wave 2's gate produces the number from
`glbench/benches/`.

Working set splits by tier. Tiers 1 and 3 — exponent 256×8 = 2 KiB, per-position
32×8 = 256 B — are ~2.3 KiB and L1-resident, and they run on **every** tensor.
Tier 2 runs only when `count ≤ 131_072` (D-12), where a `HashMap<u32, u64>` at
full occupancy is ~2–4 MB including load factor and control bytes: L3, not L2,
and paying a hash per element rather than an indexed increment.

That asymmetry is the reason the tier split exists and is worth stating plainly:
the expensive tier is confined to small tensors, so its worst case is bounded by
the cap rather than by the model. The bench must report the two paths
**separately** — a single averaged number would hide which one a given model
actually pays, and D-13's whole point is that this cost does not get guessed.

### 6.5 Integrity (D-15, D-16, D-17)

```rust
pub struct VLIntegrity {
    pub algorithm: String,   // "sha256-128"
    pub digest: String,      // 32 lowercase hex chars
}
```

**Write:**

1. Build the session JSON with `integrity` present and
   `digest = "00000000000000000000000000000000"`.
2. `to_pretty()` — canonical by F-07.
3. `glcore::hash::sha256_bytes(text.as_bytes())`, take the first 16 bytes, hex.
4. Replace the sentinel in the value, re-serialise, write.
5. `set_readonly(true)` (D-17).

Step 4 re-serialises rather than patching the string. A textual splice would
depend on the digest appearing exactly once, and `00000000000000000000000000000000`
is not a value a hex-string field elsewhere in the archive can be *proven* never
to hold. Re-serialising is one extra pass over a 10–500 KB document and removes
the class of bug entirely.

**Verify:** parse, replace `integrity.digest` with the sentinel, re-serialise,
re-hash, compare. Returns `Ok(())`, `Err(Mismatch{expected, actual})`, or
`Err(Absent)` for a v1 archive — which is *not* a failure, only an absence, and
is reported as `ENAvailability::DoesNotExist` rather than as corruption.

**`--verify` on read** is default-on for `glbench inspect` and `glbench compare`,
with `--no-verify` to override. A mismatch is `Severity::Error`, and the session
is still rendered — refusing to show a modified archive would make the tool
useless exactly when a user most needs to see what changed.

**Truncation.** 128 bits of SHA-256 gives ~2⁶⁴ collision resistance under the
birthday bound. That is ample for accident detection, which is the stated threat
model (§8 Q1 of the research document: accidental overwrite, partial write,
silent corruption). It is **not** a signature: anyone who can edit the file can
recompute the digest. The archive says so in a comment field, and this document
says so here, because "content digest" reads like tamper-proofing to a reader
who has not thought about it.

### 6.6 Join manifest (D-18)

```rust
pub struct VLJoinManifest {
    pub schema_version: u32,
    pub created_unix: u64,
    pub glbench_version: String,
    pub label: String,
    pub sources: Vec<VLJoinSource>,     // exactly 2 in v3
    pub comparison: ComparisonReport,
    pub availability: VLAvailabilityMap,
    pub integrity: Option<VLIntegrity>,
}

pub struct VLJoinSource {
    pub path: String,           // as given on the command line
    pub label: String,          // from the source's SessionMetadata
    pub digest: Option<String>, // None for a v1 source
    pub session_mode: ENSessionMode,
}
```

`glbench join a.json b.json --out join.json` reads both, verifies both digests,
computes the comparison, and writes a third file. Neither source is opened for
writing. `glbench inspect join.json` re-verifies the sources against their
recorded digests and reports drift as a finding — which is the whole point of
recording them.

`sources` is a `Vec` rather than a pair so an N-way join is a schema-compatible
extension later. v3 rejects anything other than exactly two, at the CLI, with a
clear message. Not speculative complexity: it is one field type, and the
alternative forces a schema break for a change the research document already
anticipates.

---

## 7. Cross-crate work

v3 is not a glbench-only project. Three crates change. Each change is additive
and independently landable.

### 7.1 `stumman` — instrumentation (the bulk of the work)

| Change | File | Kind |
|---|---|---|
| `VLTrainingStep`, `trait StepObserver` | `src/train/observe.rs` | new file |
| `Trainer::set_observer`, phase timing, epoch index | `src/train/trainer.rs` | additive |
| `VLGradStore::iter` | `src/autograd/grad_store.rs` | additive |
| Re-exports | `src/lib.rs` | additive |

Constraints this must respect, from `gl-agent-skills/stumman-naming/SKILL.md`
and `stumman/KNOWN_ISSUES.md`:

- Breton sub-system codename in the module header. `observe.rs` sits under
  Deskiñ (training).
- KL-006's ordering is untouched. The observer reads gradients *between*
  `finish_step()` and `optimizer.step()`; it does not hold them across the
  update, and it never gets a mutable handle.
- No new external dependency. `Instant`, `f32`, and `HashMap` are all std.

### 7.2 `glcore` — the hash primitive (D-15)

New `glcore/src/hash.rs`, moved verbatim from
`glictus-caliburni/src/checksum.rs` (implementation and its known-answer test),
plus one addition:

```rust
pub fn sha256_bytes(data: &[u8]) -> String;
pub fn sha256_file(path: &Path) -> io::Result<String>;
pub fn sha256_128_hex(data: &[u8]) -> String;   // first 16 bytes, 32 hex chars
```

Adds no dependency — `glcore` already carries `thiserror`, `memmap2`,
`byteorder`, `serde`, `serde_json`, and the implementation is std-only.

### 7.3 `glictus-caliburni` — delegate, do not duplicate

`src/checksum.rs` keeps its public surface and its `ChecksumVerifier`, and
delegates the two primitives to `glcore::hash`. Its existing tests stay where
they are and now cover the delegation. Net SHA-256 implementations in the
workspace: one, both before and after.

**Sequencing note:** 7.2 and 7.3 land together in one commit. Splitting them
leaves the workspace with two implementations at an intermediate commit, which
is the exact state F-06 exists to prevent.

---

## 8. CLI surface

```
glbench train    --model <path> --dataset <path> [options]
glbench unified  --model <path> --dataset <path> --prompt <text> [options]
glbench join     <a.json> <b.json> --out <join.json> [--label <name>]
```

New flags on existing commands:

```
--profile bits                 enable GLBitProf
--bit-scope <list>             weights|gradients|optimizer  (default: weights)
--step-sample <N>              archive every Nth step, endpoints always kept
--no-verify                    skip digest verification on read
--null-semantics strict|lenient   strict (default) fails on a D-10 violation
```

`train` and `unified` exist only under `train-bench`. In a default build they
are recognised and error with a build hint, rather than reporting an unknown
command — a user who reads about `glbench train` should learn they need a
feature flag, not that the command does not exist. This mirrors how
`PPL_USAGE`/`KL_DIV_USAGE` are conditionally appended in `main.rs:152`.

Convergence targets are required to be explicit, per the research document §14:
`--target-loss <value>` has no default, and requesting time-to-target without it
is a CLI error. There is no default "good loss".

---

## 9. Schema v2

`SCHEMA_VERSION: u32 = 2`.

**Added at the top level:** `inference`, `training`, `availability`,
`integrity`. **Added to `metadata`:** `session_mode`, `host_identifier`,
`collection_profile`. **Removed:** nothing. **Changed in meaning:** nothing.

Reading a v1 archive with a v3 build: `session_mode` absent → `InferenceOnly`;
`availability` absent → empty map; `integrity` absent → `DoesNotExist`, not a
verification failure. Reading a v2 archive with a v1 build: refused by the
existing check in `storage::archive::read`, which is correct behaviour and needs
no change.

`from_json` continues not to reconstruct `telemetry` and `behavior`, for the
reason its existing comment gives. v3 adds no round-trip requirement for them.

---

## 10. Wave plan

Per `gl-agent-skills/before-coding/wave-confirmation-gates.md`, a gate is a
STOP. Work does not continue past one without JinXSuper's sign-off.

### Wave 1 — Envelope, null semantics, integrity

No training dependency. Buildable against the tree as it stands today.

- `core::availability`, `core::mode`, `core::inference`
- `BenchmarkSession` extension, `SCHEMA_VERSION` → 2
- `glcore::hash` + `glictus-caliburni` delegation (7.2 + 7.3, one commit)
- `storage::digest`, digest-on-write, verify-on-read
- `validation::availability` (D-10)
- `storage::join` + `glbench join`

**Gate 1:** v1 archives still read. A v2 archive round-trips and verifies. The
D-10 check fails a deliberately malformed session in a test. `cargo build
--workspace` unchanged in dependency count. Digest is stable across two runs on
the same session value.

### Wave 2 — GLBitProf math + weights scope

No training dependency.

- `numerical::bitprof`, `numerical::scope`, `numerical::compare`
- Weights source wired through the existing `tensor_stats` decode path
- `--profile bits --bit-scope weights`
- `glbench/benches/` entry for the profiling pass

**Gate 2:** known-answer tests pass on hand-built bit patterns (all-zeros,
all-ones, ±0.0, subnormal, NaN, Inf, exact powers of two). The D-12 precondition
test passes at exactly `CAP` and `CAP + 1` elements.

**Measured cost numbers exist** for a real `.gllm` model on the production path
(D-13) — no document states a cost before this — and they are reported as **two
numbers, not one**: the Tier 1+3 path that every tensor pays, and the Tier 2 path
that only sub-cap tensors pay. Averaging them would hide which one a given model
actually incurs.

Also at this gate: the measured distinct-mantissa count on a real weight tensor,
checked against D-12's `m·(1 − e^(−n/m))` prediction. If real weights turn out
materially *less* uniform than the ~7.97 bits/byte literature implies, the cap
threshold is worth revisiting with that number in hand — and if they match, D-12's
table stops being an inference from a citation and becomes a measurement.

Bit-profile divergence run against the known GQ4A/Q6_K case from the research
document §12 Case 2, with the result recorded whichever way it comes out.

### Wave 3 — stumman instrumentation

**This is stumman work.** No glbench module in this wave.

- `stumman/src/train/observe.rs`
- `Trainer` observer plumbing, phase timing, epoch index
- `VLGradStore::iter`

**Gate 3:** with no observer installed, `train_step` produces byte-identical
results and its measured time is within noise of today's. With an observer, the
overhead is **measured and recorded**. KL-006's ordering is provably unchanged —
the existing tests that cover it still pass, and the observer cannot obtain a
gradient handle that outlives the step.

### Wave 4 — glbench training observation

Depends on Wave 3. Nothing here is written before Gate 3 clears.

- `training/` module tree, `train-bench` feature
- `glbench train`, `glbench unified`
- `convergence`, `attribution`, `memory`, `adapter`
- Gradient and optimizer bit scopes
- `comparison::training`

**Gate 4:** a real LoRA training run on stumman produces a v2 archive that
passes D-10 with every M2-absent field carrying an honest status (F-05). The
convergence numbers are reported with their window and threshold, per the
research document §14. A `unified` session has both inference roles populated
and correctly labelled.

### Wave 5 — Rendering, export, docs

- `render::loss_curve`, flame graph extension
- Markdown and CSV training sections
- `glbench/DESIGN.md`, `ARCHITECTURE.md`, `README.md`, `ROADMAP.md` updates
- `RESEARCH_REQUIREMENTS.md` candidate table updated with v3 features

**Gate 5:** every claim in the updated docs traces to a measured number or a
file:line. No doc says "approximately" about a cost that was never run.

---

## 11. Test plan

Beyond per-module unit tests, six tests carry the design's actual invariants.
Each one exists because the corresponding invariant is silently violable.

1. **No null without a status** (D-10). Build a session with a deliberately
   unexplained `null`; assert `validation::availability` produces an error
   naming the exact path. The mirror case too: a status on a non-null field.

2. **Digest is stable and detects a single-byte edit.** Write, read, verify.
   Then flip one character in the file and assert `Mismatch`. Then flip a
   character *inside the digest field itself* and assert `Mismatch` rather than
   an accidental pass.

3. **Mode consistency.** For each `ENSessionMode`, assert the §6.2 table holds
   and that a violation is caught at finalisation, not at export.

4. **Observer is free when absent** (Gate 3). Run N steps with and without an
   observer from the same seed; assert bit-identical loss sequences.

5. **GLBitProf known answers.** Hand-built `&[f32]` with known bit patterns.
   Specifically: `[0.0, -0.0]` gives `sign_set_ratio == 0.5`,
   `zero_count == 1`, `negative_zero_count == 1` — the case §6.4's zero-handling
   rule depends on, and the one a freshly-initialised LoRA `B` matrix hits.

   Plus one test for D-12's precondition guard, because it is the decision most
   likely to be silently mis-implemented as an abort: a tensor of
   `MANTISSA_SPARSE_CAP` elements collects a complete `mantissa_sparse` with
   `mantissa_sparse_skipped == false`; a tensor of `CAP + 1` elements returns
   `mantissa_sparse: None`, `mantissa_entropy_bits: None`, and
   `mantissa_sparse_skipped == true` — asserting specifically that the map is
   `None` rather than a `Some` holding exactly `CAP` entries.

6. **v1 archive still reads.** A checked-in v1 archive fixture, read by the v3
   build, asserting `InferenceOnly`, empty availability, and
   `DoesNotExist` integrity. This is the backward-compatibility promise, and it
   needs a fixture rather than a synthesised session, because a synthesised one
   is written by the v3 writer and cannot prove anything about v1's shape.

Two standing rules apply to all six.
`gl-agent-skills/rust-skills/testing-standards.md` rule 3 requires numerical
assertions to use explicit named tolerances (`const TOL_…`), which covers every
float comparison in tests 4 and 5. And no v3 test asserts on a timing threshold:
Gates 2 and 3 want *numbers*, and a number belongs in `glbench/benches/` under
`bench-skills/measurement-discipline.md`, where the repeat count and the noise
floor are handled properly. A timing assertion in a `#[test]` is a flaky test
wearing a measurement's clothes.

---

## 12. Out of scope

Carried forward from the research document §9, unchanged, with the reason
restated so it is not re-evaluated speculatively:

- **Per-kernel CUDA timeline** — needs `cupti` or engine-side NVTX. External SDK.
- **Automatic LR recommendation** — an optimisation action. `DESIGN.md` §1.
- **Cross-framework training comparison** — needs PyTorch. Python dependency.
- **W&B / MLflow / TensorBoard** — external service. `DESIGN.md` §9.
- **Gradient checkpointing analysis** — not implemented in stumman M2.
- **Activation capture during inference** — engine-side instrumentation project.
  `--bit-scope activations` is recognised and errors with a pointer here.
- **HTML export** — Phase 4 open item, not v3.
- **ratatui TUI** — design phase not started.

Added by this document:

- **`gltrain` observation** — F-01. Not a deferral of v3 work; a different
  project with a different dependency story.
- **Token-denominated training metrics** — F-05. Schema-present, status
  `not_applicable`, populated when stumman gains a tokenised training loop. No
  glbench change needed at that point, which is the point of D-04.
- **N-way join** — schema-compatible (D-18), CLI-rejected in v3.

---

## 13. Requires JinXSuper sign-off

Six items. Waves do not start on the affected areas until these are answered.

1. **D-02 reverses the research document §10.** The training dependency is
   optional behind `train-bench`, not unconditional. The reasoning is F-01, and
   it is the difference between `cargo build --workspace` staying clean and not.
   Confirm.

2. **Wave 3 is stumman work.** Roughly 85% of v3's training observation is
   instrumentation inside stumman (§2), not glbench. This changes who owns the
   work and how long v3 takes. Confirm the sequencing, or say if stumman
   instrumentation should be its own milestone (M2.5) outside v3 entirely.

3. **D-15 moves SHA-256 into `glcore`.** Touches `glcore` and
   `glictus-caliburni`, both outside glbench. Small and additive, but it is a
   cross-crate move and the alternative (a second hash implementation) is worse
   for reasons ARTX2-Quant already documented. Confirm.

4. **§6.2's compatibility compromise.** v1 fields stay top-level rather than
   moving inside `VLInferenceSession`, contrary to the research document §4's
   tree but consistent with its §3 "nothing is rewritten" promise. This is the
   one place where the design deliberately produces a less elegant schema than
   specified. Confirm, or accept the larger refactor.

5. **F-10's directory collision.** `architecture/mensura-veritatis-v3/` already
   holds the 2026-07-23 gl-stack audit series and is cited by path from
   `RESEARCH_REQUIREMENTS.md`. This design sits at `architecture/glbench-v3/`.
   Option: rename the older series to something like
   `architecture/gl-stack-audit-2026-07/` and update the one citation. Leaving
   both is also fine — but two things called "Mensura Veritatis v3" will confuse
   somebody eventually.

6. **D-12 — the cap is a precondition, not an abort.** The reviewed proposal
   replaced 12-bit bucketing with a sparse full-resolution `HashMap<u32, u64>`
   capped at 131,072 entries, truncating mid-collection when the cap is hit.
   Full resolution is adopted. The mid-collection truncation is **not**, and the
   reason is arithmetic rather than preference: assuming the near-uniform
   mantissa that the proposal's own ~7.97 bits/byte citation establishes, the cap
   is reached at **132,107 elements**, so it fires on every weight matrix in the
   model. What survives is order-biased rather than sampled, its entropy is
   capped at `log₂(min(n, 2²³))` and therefore tracks tensor size instead of
   precision usage, and the archive reaches ~570 MB on a 0.5 B model.

   D-12 as written keeps the goal verbatim — full 23-bit resolution when the
   distribution allows, graceful degradation when not — and moves the decision
   to *before* collection: if the map cannot be complete, it is not collected,
   and the availability map says `unavailable`. Small tensors (LoRA A/B, norms,
   biases) get an exact unbucketed 23-bit distribution; large ones get Tiers 1
   and 3, which are exact for them.

   Two smaller notes on the proposed struct, applied: `mantissa_entropy_bits` is
   `Option<f64>`, since the draft's comment said "None if truncated" while its
   type said `f64`; and `mantissa_sparse_truncated` is renamed
   `mantissa_sparse_skipped`, because under the precondition guard nothing is
   ever truncated — it is either complete or absent.

   Confirm the precondition guard, or say to take the mid-collection truncation
   as originally drafted and accept the three consequences above.

---

*End of design document.*
