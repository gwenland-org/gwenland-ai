# ARTX13 — Tokenization Architecture

**Series:** gljax (Sanctum Visibilia) Architecture Research
**Status:** Draft — research-grounded; **§0.2 wave A13.0 IMPLEMENTED, measured, extracted and
optimised** (`glcore::tokenizer`). See §0.4 for the current state; the sections below preserve the
investigation that got there, because each mistake in it is one this design invites.
**Depends on:** ARTX04 (checkpoint loading), ARTX11 §3.2 (cross-vocabulary speculation), ARTX16 §1.2 (request lifecycle)
**Introduces:** `gljax/src/tok/`
**Next:** [ARTX14 — Sampling & Logits Processing](ARTX14-sampling-and-logits-processing.md)
**Research grounded:** 2026-07-27 (sources at end)

---

# 0. The Gap, and the Asset

## 0.1 Why this document exists

A grep across ARTX01–ARTX12 returns **zero headings** for tokenization. Yet:

* ARTX16 §1.2's request lifecycle step 3 is *"Tokenize prompt — gljax tokenizer (host side)"* — a
  reference to something never specified.
* ARTX11 §3.2's cross-vocabulary speculation (SLEM/TLI) is **entirely** a statement about tokenizer
  semantics — string-level matching and vocabulary intersection.
* ARTX16's SSE streaming (§1.5) emits text per token, which requires incremental detokenization —
  the hardest problem in this document (§4).

The series specified an engine that takes token IDs and returns logits, and never specified how text
becomes IDs or how IDs become text. ARTX13 closes the input side; ARTX14 closes the output side.

## 0.2 GwenLand had a tokenizer — it was measured, found wrong, and replaced

`glcore/src/tokenizer.rs` — 796 lines, from scratch, zero ML dependencies. ⛔ **Deleted.** It was
superseded by a rewrite (§0.2.2), then every caller was migrated, then the file was removed. Its
module doc, kept because §0.2.1 is an argument about what it claimed:

> BPE tokenizer, written from scratch. Two vocabulary styles are supported, covering the models GGUF
> files ship: **SPM style** (llama family) — tokens use `▁` for spaces, merging is driven by
> per-token scores, unknown bytes fall back to `<0xNN>` byte tokens; **Byte-level BPE** (gpt2/qwen
> family) — raw bytes are first mapped to printable unicode chars, then merged using an explicit
> merge list. Vocabularies load from GGUF metadata (`Tokenizer::from_gguf`) or a HuggingFace
> `tokenizer.json` (`Tokenizer::from_file`).

Existing surface: `encode`, `encode_chat`, `decode`, `decode_token`, `decode_token_text`,
`vocab_size`, `eos_id`, `bos_id`, `add_bos_default`, `is_stop_token`, `stop_token_ids`,
`special_tokens`, `merges`.

### ⛔ 0.2.1 The test inventory, and what it does not cover

There are **9 unit tests, no `glcore/tests/` integration directory, and no reference-parity check
against any external tokenizer.** Reading them precisely:

| Test | Covers | ⚠️ |
|---|---|---|
| `spm_round_trip_ascii` | `"Hello World"`, `"Hello"`, `"abc def"`, `"a  b"`, `" leading"` | **ASCII only** |
| `spm_byte_fallback` | `"ab!"` via `<0x21>` | the fallback path |
| `byte_level_round_trip` | a vocab built from the 256 byte-map entries | **"no merges"** — by construction |
| `eos_is_a_stop_token`, `stopping_criteria_...`, `qwen_style_stop_markers_...` | stop-token resolution | not the tokenizer algorithm |
| `encode_chat_*` (×2), `decode_skips_specials` | ChatML wrapping, special skipping | not the tokenizer algorithm |

⛔ **The two tests that exercise the algorithm both disable the algorithm.**

1. The SPM fixture is a **~296-token synthetic vocabulary** built in-test, and it passes
   `vec![0.0; n]` for the scores — with the fixture's own comment reading
   *"uniform scores → longest-merge fallback not needed."* **SPM tokenization *is* score-driven merge
   selection.** With every score equal, the comparison path that chooses between competing merges is
   never taken.
2. The byte-level fixture is documented as *"every mapped single byte is a token; **no merges**."*
   Byte-level BPE's merge-list application is therefore also never exercised.

**So: the byte-fallback path and the trivial no-merge paths are tested. The actual BPE merge logic —
of both families — is not.** That is the opposite of the coverage one would want, because merge
selection is where tokenizers silently disagree.

There is exactly **one** data point of real-vocabulary validation anywhere in the repo, and it was
produced by hand during an unrelated investigation: the glproc/llama.cpp perplexity comparison
verified *"tokenization identical (819 tokens both engines, first-20 token IDs byte-for-byte
identical via `llama-tokenize.exe --ids`)"* — one model, one sample, once, not a standing test.

### ✅ 0.2.2 RESOLVED — measured, then rewritten

**Wave A13.0 ran.** The suspicion in §0.2.1 was correct, and the measurement was worse than the
inspection suggested. `glcore::tokenizer` was scored against llama.cpp's reference vectors
(`ggml-vocab-*.gguf` + `.inp`/`.out`, whose expected ids come from the HuggingFace tokenizers —
reference **data**, not reference code):

| Vocabulary | OLD `glcore::tokenizer` | NEW `gltokenizer` |
|---|---|---|
| llama-bpe | **65.2%** (30/46) | **100%** (46/46) |
| qwen2 | 82.6% (38/46) | **100%** |
| starcoder · refact · mpt · deepseek-coder · deepseek-llm | 84.8% (39/46) | **100%** |
| llama-spm · gpt-2 · phi-3 | 97.8% (45/46) | **100%** |

⛔ **Not one vocabulary was correct.** The best was 45/46; the worst, llama-bpe, got **a third of
its inputs wrong**. The round-trip tests passed throughout, because they tested
`decode(encode(x)) == x` rather than `encode(x) == reference` — and §0.2.1's compensating error made
the round trip hold while the ids were wrong.

⚠️ **DESIGN DECISION — the tokenizer was rewritten, not hardened.** It began as a new crate
(`glcore/gltokenizer`, since folded into `glcore::tokenizer` — §0.4): an original BPE implementation written from the algorithm's definition. Three defects only the
parity harness could find:

1. **Merge rules keyed by concatenation.** A merge list is a list of *pairs*; different splits of the
   same string are different rules with different ranks. Measured on llama-bpe: **152,403 of 280,147
   rules lost (54%)**. qwen2 and command-r happen to have zero such collisions, which is why the bug
   stayed hidden there.
2. **The "GPT-2 family" is not one shape.** gpt-2/mpt spell digits `" ?\p{N}+"` (a run may absorb a
   leading space); starcoder/refact spell them bare. Otherwise identical.
3. **Llama-3's `ignore_merges` was missing** — behavioural, not an optimisation. For `" Việt"`,
   correct rank-order BPE reaches `ĠVi|á»ĩ|t` because `ĠV+i` (rank 31158) fires before `á»+ĩ`
   (69499), making the whole-token form unreachable. Llama-3 emits the vocabulary entry directly.

**Speed, same corpus, release build, median of 5 repeats over 5,440 bytes:**

| Regime | Speedup | Why |
|---|---|---|
| Byte-level | **7.1×** | Pre-tokenized chunks are word-sized, so the old `O(n³)` loop ran on tiny `n` |
| SPM | **848×** | ⭐ No pre-tokenizer, so the *whole input* was one merge run — where a cubic loop actually bites |

⚠️ Reported per regime rather than as one number: a combined mean is the SPM figure in disguise, and
the two have different causes.

⚠️ Four families were refused at this point rather than partially supported, per §2.3's rule.
**All four have since been closed** — see §0.4, which also corrects the diagnosis recorded here for
two of them.

⚠️ **Migration had not happened** when this was written; twelve files still called the old module.
It has since completed, and the result was measured rather than assumed: on Qwen2.5 the ids came out
**identical** (819/819 on WikiText-2, 201/201 on the glbench prompt) and perplexity landed at 36.19
against the 36.12 baseline. ⛔ That is neutrality **for Qwen2.5 only**. The old implementation scored
65.2 %–97.8 % across families, so for Llama-3 and SPM the ids necessarily moved; no model was
available to measure it.

## 0.3 What ARTX13 actually specifies

1. The **trait** gljax depends on, so the implementation stays swappable (§1)
2. **Vocabulary loading** — which sources, and what must be validated (§2)
3. **Chat templating** — the layer above encode (§3)
4. ⭐ **Incremental detokenization** — the streaming problem, and why it is not `decode(one_token)` (§4)
5. **Cross-vocabulary support** for ARTX11 (§5)

---

## 0.4 ✅ Current state — what gljax can actually depend on

`glcore::tokenizer` (was the `gltokenizer` crate; folded in, and the old module of that name
deleted). Type: `GllmTokenizer`. Architecture and traps: `glcore/src/tokenizer/README.md`.

| Status | Vocabularies |
|---|---|
| ✅ **exact** (14) | qwen2 · qwen35 · llama-bpe · llama-spm · gpt-2 · starcoder · refact · mpt · command-r · phi-3 · gemma-4 · falcon |
| ⚠️ exact, shape approximated | deepseek-coder · deepseek-llm |
| ⛔ refused | gpt-neox · aquila |
| 🕐 loads, unverifiable | baichuan |

⚠️ The two **shape-approximated** entries matter to §5. Both score 46/46, which is the same evidence
every other entry rests on — but llama.cpp gives each a multi-*expression* pipeline rather than the
single arm this implementation uses, and `\s?\p{L}+` lets a newline lead a word where the Qwen2 arm's
`[^\r\n\p{L}\p{N}]?` excludes it. They agree everywhere the corpus reaches and are **known** to
differ somewhere it does not. A cross-vocabulary claim built on deepseek should say so.

⚠️ `gpt-neox` and `aquila` used to load and **should not have**. Neither carries a
`tokenizer.ggml.pre` key, so both reach llama.cpp's `default` arm — a four-expression fallback, not
the GPT-2 shape they were being given. The shape is now expressible; they stay refused because
neither ships reference vectors, so enabling them would claim support that cannot be measured.

### ⛔ 0.4.1 The defect this document should have predicted

§0.2.2 lists three defects only a parity harness could find. A later audit found a fourth, and it is
the one most relevant to ARTX13's own design:

> **The pre-tokenizer name table was wrong for 13 of 24 entries**, and *not one of the wrong rows
> was reachable by any test.*

GGUF names its pre-tokenizer (`tokenizer.ggml.pre`), and the table mapped that name onto a splitter
shape by grouping names that *look* related. llama.cpp assigns them by which `regex_exprs` arm the
name reaches, and the two groupings are not the same function — `default` is not the GPT-2 shape,
`codeshell`/`smollm`/`exaone` are starcoder rather than cl100k, `chatglm-bpe` groups three digits
not one, and `smaug`/`poro` were not llama.cpp names at all.

**The lesson generalises past tokenizers**, which is why it belongs in an architecture document:
a lookup table keyed on a *name* rather than on the property that actually varies is a
silent-wrongness factory, and its wrong rows are exactly the ones no test reaches. §2.3's
"refuse rather than approximate" rule is necessary but was not sufficient — the table was refusing
correctly and *mapping* incorrectly.

## 0.5 Measured throughput, and one constraint ARTX16 inherits

i3-1115G4, best-of-40, a **frozen** 120 KiB corpus of prose and Rust source, Qwen2.5 vocabulary.

| | ns/byte | MB/s |
|---|---:|---:|
| pre-tokenizer alone | 4.95 | ~205 |
| full encode, cache OFF | 120–128 | 7.8–8.3 |
| **full encode, cold cache** | **50–53** | **18.9–20.1** |
| full encode, warm cache | 14.1 | ~71 |

**Quote the cold number.** Warm answers "how fast is re-encoding a document you have already seen",
which no server does; it is the upper bound a long-lived process with repetitive traffic approaches.

⭐ **The largest win came from profiling, not from any technique in the literature.** Two rounds of
reasoning about where the time went were wrong. The actual hot spot was `find_special`, which ran
one full-text substring search **per special token** — 22 of them for Qwen2.5, so a 120 KiB prompt
was scanned for 2.6 MiB before a byte was tokenized, at **71 % of a warm encode**. One pass with a
first-byte skip table took it to ~1 %. `examples/tokenizer_profile.rs` found it in a single run.

⚠️ For ARTX13's purposes the transferable part is the *shape* of that bug: the cost scaled with the
**vocabulary's** special-token count on input that usually contains none of them. Any design that
loops over vocabulary entries per input should be checked for the same shape.

### ⛔ 0.5.1 Tokenizing one prompt cannot be parallelised — ARTX16 §2 must assume this

The obvious parallelisation — cut the input into chunks, tokenize independently, concatenate — needs
a split point the segmentation is invariant under. **There is none**, and the intuitive candidate is
a counterexample: `\s+(?!\S)` keeps a whitespace run whole when it reaches end-of-input and
surrenders its last character when it does not, so cutting re-segments the seam. Under the GPT-2
shape, `"a \nb"` segments as `"a" · " " · "\n" · "b"` whole and `"a" · " \n" · "b"` cut immediately
after the newline. Pinned by `pretok::tests::splitting_the_input_changes_the_segmentation`.

**Consequences for ARTX16's pipeline (its §2 step 3, "Tokenize prompt — host side"):**

* **Per-request parallelism is free and already works.** `GllmTokenizer` is `Sync` and its scratch
  and cache are thread-local, so N requests tokenize on N threads with no coordination.
* **Per-prompt parallelism is unavailable.** A single 1 MB prompt costs ~50 ms of *serialised* host
  time at the cold-cache rate. For a long-context request that is real time-to-first-token, and it
  cannot be hidden by adding cores.
* ⚠️ The cache is **per thread**, so a work-stealing pool spreads one model's traffic across N
  independent caches. A thread-pinned or thread-affine assignment keeps hit rates up; a naive pool
  divides them.

Rejected with numbers, recorded in `gl-agent-skills/cpu-skills/rejected-optimizations.md` (T1–T4):
a faster hasher (neutral — hashing is not the bottleneck), splitting one input across threads
(incorrect, above), streaming pre-tokens instead of buffering them (8 % slower on the miss path),
and SWAR/dual-cursor ILP (deprioritised — pre-tokenization is ~38 % of a *warm* encode but the
techniques cap at ~2 % end-to-end, and dual-cursor is a shape this repo has rejected twice
elsewhere).

---

# 1. The Tokenizer Contract

## 1.1 Why a trait, when there is only one implementation

ARTX08's rejected-alternative #7 declined a trait with one implementor. The reasoning does **not**
apply here, and the difference is worth being explicit about:

| | ARTX08's `Matmul` trait | ARTX13's `Tokenizer` trait |
|---|---|---|
| Implementors, realistically | 1 forever (`dot_general`) | ≥ 2 — `glcore`, plus **a second vocabulary at the same time** (ARTX11 draft ≠ target) |
| Purpose | Dispatch | **Decoupling** — keeps `glcore` out of gljax's core types |

⭐ ARTX11's cross-vocabulary speculation requires gljax to hold **two live tokenizers with different
vocabularies simultaneously**. That is a genuine polymorphism requirement, not a speculative one.

## 1.2 The seam

```rust
// gljax/src/tok/mod.rs
//
// gljax depends on THIS, never on glcore directly. The glcore-backed
// implementation lives behind a default-on feature so gljax stays compilable
// without it (tests, and any future standalone build).

pub trait Tokenizer: Send + Sync {
    fn encode(&self, text: &str, add_special: bool) -> Vec<TokenId>;
    /// Whole-sequence decode. NOT valid for streaming — see §4.
    fn decode(&self, ids: &[TokenId], skip_special: bool) -> String;

    fn vocab_size(&self) -> usize;
    fn eos_id(&self) -> TokenId;
    fn bos_id(&self) -> Option<TokenId>;
    fn is_stop(&self, id: TokenId) -> bool;

    /// Raw surface form of one token, WITHOUT byte-level unmapping applied.
    /// Needed by §4's incremental decoder and §5's SLEM.
    fn token_bytes(&self, id: TokenId) -> &[u8];

    /// Stable identity of this vocabulary. Used to decide `VocabRelation`
    /// (ARTX11 §2.3) and to key the ARTX04 compile cache.
    fn vocab_fingerprint(&self) -> VocabFingerprint;
}

/// SHA-256 over (sorted vocab entries ++ merge list ++ special tokens).
/// Two tokenizers with equal fingerprints are interchangeable; unequal
/// fingerprints mean ARTX11 must use a heterogeneous algorithm (§5).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VocabFingerprint([u8; 32]);
```

⚠️ **DESIGN DECISION — `vocab_fingerprint` is computed, never trusted from metadata.**
ARTX11 §2.3's `VocabRelation::Identical` is a *correctness* precondition: if two vocabularies are
assumed identical and are not, speculative decoding silently produces wrong tokens while remaining
fluent — ARTX12's core bug class. A model name or tokenizer-class string is not evidence; the
content hash is.

⚠️ Two tokenizers from the *same family but different generations* are the trap this catches. ARTX11
§3.1 flagged "Qwen3-1.7B drafted by Qwen2.5-0.5B — ⚠️ verify, cross-generation tokenizers may
differ." The fingerprint turns that warning into an assertion.

---

# 2. Vocabulary Loading

## 2.1 The two source formats

| Source | Path | What it carries |
|---|---|---|
| **GGUF metadata** | `Tokenizer::from_gguf` | Vocab, scores/merges, special IDs, `add_bos_token` — embedded in the model file |
| **HF `tokenizer.json`** | `Tokenizer::from_file` | The full HF pipeline description (§2.2) |

⚠️ ARTX04 loads safetensors and `.gllm`; ARTX12 Part A will add GGUF. **The tokenizer source and the
weight source need not be the same file**, and gljax must not assume they are: a safetensors
checkpoint ships `tokenizer.json` beside it, while a GGUF embeds both.

## 2.2 The HF pipeline, and what gljax must respect

`tokenizer.json` describes a five-stage pipeline:

```text
raw text
   ▼  normalizer      strip / lowercase / NFC / accent removal
   ▼  pre_tokenizer   split into word-level pieces (whitespace, regex, ByteLevel)
   ▼  model           BPE | Unigram | WordPiece | WordLevel → subword IDs
   ▼  post_processor  add BOS/EOS/template special tokens
   ▼  decoder         IDs → text (the inverse, and NOT symmetric — §4)
```

⚠️ **The `pre_tokenizer` stage is where the two families genuinely diverge**, and getting it wrong is
silent:

* **SentencePiece / SPM** works on raw text with **no whitespace pre-tokenization** — whitespace is
  itself part of the token stream, encoded as `▁`. Unknown bytes fall back to `<0xNN>`.
* **Byte-level BPE** (GPT-2, Qwen) maps raw bytes to printable Unicode first, then merges. tiktoken's
  variant skips normalization entirely and goes straight to a regex chunk before BPE — which is
  where its ~2–3× speed advantage comes from.

`glcore::tokenizer` already implements both. ARTX13 adds only the requirement that the **family be
detected and recorded**, not inferred per call.

## 2.3 Validation at load

⚠️ **DESIGN DECISION — validate the tokenizer against the model at `Session::new`, and refuse to
serve on mismatch.**

```rust
pub fn validate(tok: &dyn Tokenizer, model: &ModelConfig) -> Result<(), TokError> {
    // V1 — the one that catches the most damage.
    // A vocab/embedding mismatch means gather_embed reads out of bounds or
    // silently maps to the wrong row. Fluent garbage, no error.
    if tok.vocab_size() != model.vocab_size {
        return Err(TokError::VocabSizeMismatch {
            tokenizer: tok.vocab_size(), model: model.vocab_size });
    }
    // V2 — EOS must exist and be in range, or generation never terminates.
    if tok.eos_id() as usize >= model.vocab_size {
        return Err(TokError::EosOutOfRange(tok.eos_id()));
    }
    // V3 — BOS policy must match the checkpoint's declared intent.
    // ⚠️ GwenLand has been bitten by BOS handling before: the glproc/llama.cpp
    // perplexity comparison had to explicitly verify `add_bos_token: false`
    // was honoured by both engines before the numbers could be trusted.
    Ok(())
}
```

⚠️ V1 deserves emphasis: a padded vocabulary is common (models round `vocab_size` up to a multiple of
64 or 128 for tensor-core alignment), so `tok.vocab_size() < model.vocab_size` is *legitimate* while
the reverse is fatal. The check must distinguish the two rather than demand equality — recorded as an
open detail for implementation, since it depends on ARTX12 Part A's config parsing.

---

# 3. Chat Templating

`glcore::tokenizer` already exposes `encode_chat(user) -> Option<Vec<u32>>`. ARTX16's
`/v1/chat/completions` needs more: a full `messages[]` array with roles.

⚠️ **DESIGN DECISION — chat templating lives in `glserve`, not in gljax.**

The template is a *serving* concern: it maps an OpenAI-shaped request onto a model-specific prompt
format. It has no bearing on the engine, changes per model without changing the engine, and belongs
next to the API types that produce it (ARTX16 §8's `api/openai.rs` reasoning).

```rust
// glserve/src/api/template.rs
pub trait ChatTemplate {
    fn render(&self, messages: &[ChatMessage], add_generation_prompt: bool) -> String;
}
```

⚠️ ⛔ **Do not implement a Jinja2 interpreter.** HF `chat_template` fields are Jinja2, and evaluating
arbitrary Jinja in a serving process is both a large dependency and an untrusted-input execution
surface. gljax ships a small set of *known* templates (ChatML, Llama-3, Gemma) selected by the
model's declared family, and **refuses unknown templates rather than guessing**. A wrong template
produces fluent, subtly-off output — the ARTX12 bug class again, this time from a formatting error.

---

# 4. ⭐ Incremental Detokenization — the hard part

## 4.1 Why `decode(&[one_token])` is wrong

ARTX16 §1.5 streams one SSE chunk per token, each carrying a text delta. The naive implementation
calls `decode` on the single new token. **This is incorrect, for two independent reasons.**

**Reason 1 — partial UTF-8.** Byte-level BPE tokens are byte sequences, not characters. A multi-byte
character (emoji, CJK, Cyrillic) commonly spans two or more tokens. Decoding one token yields an
incomplete UTF-8 sequence, which lossy decoding renders as `�` — and the substitution happens
*before* transmission, so no amount of client-side buffering can recover it. This is a real, reported
production failure mode, not a theoretical one.

**Reason 2 — context-dependent spacing.** vLLM's own incremental-detokenization work records the
core difficulty: *the tokenizer decides whether to add a space depending on the surrounding token
IDs.* SPM's `▁` handling and byte-level BPE's leading-space convention both make a token's rendered
text a function of its neighbours. Per-token decode cannot see them.

⚠️ Byte-level tokenizers can also emit sequences that are **not valid UTF-8 at all** — that is an
established property of byte-level vocabularies, not a bug to fix. The decoder must degrade
gracefully rather than assume validity.

## 4.2 The design

⚠️ **DESIGN DECISION — a stateful per-request `IncrementalDecoder` that emits only complete
characters and carries a byte remainder.**

```rust
// gljax/src/tok/stream.rs

/// One per in-flight request. Owned by the request, not the tokenizer.
pub struct IncrementalDecoder<'t> {
    tok: &'t dyn Tokenizer,
    /// All tokens so far. Needed because rendering is context-dependent (§4.1).
    ids: Vec<TokenId>,
    /// Bytes produced but not yet emitted: an incomplete UTF-8 tail.
    pending: Vec<u8>,
    /// How many bytes of the full decode have already been emitted.
    emitted_bytes: usize,
}

impl<'t> IncrementalDecoder<'t> {
    /// Push one token; return the text delta that is SAFE to emit now.
    /// Returns "" when the token only extends an incomplete character.
    pub fn push(&mut self, id: TokenId) -> String {
        self.ids.push(id);
        self.pending.extend_from_slice(self.tok.token_bytes(id));

        // Emit the longest valid UTF-8 prefix; keep the remainder pending.
        match std::str::from_utf8(&self.pending) {
            Ok(s) => { let out = s.to_string(); self.pending.clear(); out }
            Err(e) => {
                let good = e.valid_up_to();
                // ⚠️ error_len() == Some(_) means genuinely INVALID bytes, not
                // merely incomplete. Byte-level vocabs can produce these.
                // Emit the good prefix, drop the bad byte, keep going.
                let out = unsafe { std::str::from_utf8_unchecked(&self.pending[..good]) }.to_string();
                match e.error_len() {
                    None        => { self.pending.drain(..good); }          // incomplete: wait
                    Some(bad)   => { self.pending.drain(..good + bad); }    // invalid: skip
                }
                out
            }
        }
    }

    /// Flush at end of generation. Any still-pending bytes were a truncated
    /// character — emit U+FFFD once rather than silently dropping them.
    pub fn finish(&mut self) -> String {
        if self.pending.is_empty() { String::new() } else { self.pending.clear(); "\u{FFFD}".into() }
    }
}
```

⚠️ **The `ids` field is retained deliberately even though `push` does not read it.** Reason 2 above
(context-dependent spacing) means a fully correct implementation must be able to re-render a window
of recent tokens rather than concatenating per-token bytes. The byte-concatenation path above is
correct for **byte-level BPE**, where a token's bytes are context-independent; it is **not**
guaranteed correct for SPM-style vocabularies with `▁` handling.

⚠️ **DESIGN DECISION — the decoder has two strategies, selected by vocabulary family.**

```text
ByteLevel  → concatenate token_bytes, split on UTF-8 boundaries   (above; cheap)
Spm        → re-decode a trailing window of N tokens each step,
             diff against what was already emitted                 (correct; costlier)
```

⚠️ The SPM path's window size is a correctness/cost tradeoff and must be **measured**, not guessed.
A window that is too short reintroduces the spacing bug at the window boundary. Recorded as an open
question (§7).

## 4.3 Interaction with ARTX11

⚠️ Speculative decoding accepts a **variable number of tokens per iteration** (ARTX11 §5.1), and
rejected drafts must never reach the decoder.

```text
speculative_step returns AcceptResult { tokens: [t0, t1, t2], .. }
   ▼
for t in tokens:  decoder.push(t)  → concatenate the deltas → ONE SSE chunk
```

⚠️ Pushing rejected tokens would corrupt `pending` and the emitted stream. The decoder must be driven
from `AcceptResult::tokens` — the *accepted* list — never from the draft candidates. This is a
one-line rule that is easy to violate when wiring the two subsystems, and it belongs in ARTX12's
integration tests.

---

# 5. Cross-Vocabulary Support (for ARTX11)

ARTX11 §3.2 adopted Timor et al.'s lossless heterogeneous-vocabulary algorithms — **SLEM**
(String-Level Exact Match) and **TLI** (Token-Level Intersection) — which work with off-the-shelf
models and require no training. ARTX13 supplies what they need from the tokenizer layer.

```rust
// gljax/src/tok/hetero.rs

/// SLEM verifies on DETOKENIZED STRINGS rather than token IDs, so it needs
/// a cheap, allocation-light per-token surface form on both sides.
pub fn token_str(tok: &dyn Tokenizer, id: TokenId) -> Cow<'_, str>;

/// TLI restricts verification to the shared vocabulary subset.
/// Built ONCE at pairing time (ARTX11 §2.3), never per step.
pub struct VocabIntersection {
    /// draft id → target id, for surface forms present in both.
    draft_to_target: HashMap<TokenId, TokenId>,
    /// Fraction of the draft vocabulary that mapped. ⚠️ Low coverage means
    /// TLI's acceptance rate degrades; below a threshold, prefer SLEM.
    pub coverage: f64,
}

pub fn build_intersection(draft: &dyn Tokenizer, target: &dyn Tokenizer) -> VocabIntersection;
```

⚠️ **DESIGN DECISION — the intersection is built once at pairing and its coverage is reported.**
ARTX11 §3.2 noted cross-family speculation is *"strictly worse than same-family when both are
available."* Coverage is the number that quantifies how much worse, and it is knowable before a
single token is generated — so it belongs in the startup log and in ARTX16's `/health` payload, not
discovered from a disappointing acceptance rate in production.

⚠️ Building the intersection is `O(|V_draft|)` map lookups over surface forms — for two 151,936-entry
vocabularies that is trivially fast, but it is `O(V)` **memory** for the map. Noted, not optimized.

---

# 6. Module Layout + Wave Plan

```text
gljax/src/tok/
├── mod.rs        Tokenizer trait, TokenId, VocabFingerprint
├── glcore.rs     the glcore-backed impl, behind `feature = "glcore-tokenizer"` (default on)
├── stream.rs     §4  IncrementalDecoder (ByteLevel + Spm strategies)
├── hetero.rs     §5  token_str, VocabIntersection
└── validate.rs   §2.3 load-time checks

glserve/src/api/
└── template.rs   §3  ChatTemplate — serving-side, known templates only
```

| Wave | Scope | Gate |
|---|---|---|
| **A13.0** ✅ | **DONE — rewritten, then extracted to `glcore::tokenizer`.** Zero-allocation merge engine, hand-written pre-tokenizer, exact Unicode category tables, GGUF vocab reader, pre-token cache | ✅ **14 vocabularies exact** against llama.cpp's reference vectors, enforced by `glcore/tests/tokenizer_parity.rs` on every build. Architecture: [`glcore/src/tokenizer/README.md`]. ⚠️ The before/after harness was deleted with the old implementation; its result is recorded in §0.2.2. |
| **A13.1** ◐ | `Tokenizer` trait + `VocabFingerprint` — ⚠️ **the concrete type exists; the trait and fingerprint do not yet** | Round-trip holds; fingerprint still to build |
| **A13.2** | `validate.rs` | Vocab/model mismatch is refused at `Session::new`, not at first token |
| **A13.3** | ⭐ `stream.rs` | **No `�` ever emitted mid-stream** for a corpus of multi-byte text; concatenated deltas == whole-sequence `decode` |
| **A13.4** | `hetero.rs` | Coverage reported; SLEM/TLI agree with a reference on a real cross-family pair |
| **A13.5** | `template.rs` (glserve) | Rendered prompt is byte-identical to HF's for the same messages, on each supported family |

⚠️ **A13.3's gate is the load-bearing one and it is stated as an invariant, not a metric:**
*the concatenation of all streamed deltas must equal the whole-sequence decode of the same token
list.* That single property catches partial-UTF-8 bugs, spacing bugs, and off-by-one emission bugs
at once, and it is cheap to assert on every streaming test.

## 6.1 Open questions

0. ✅ **ANSWERED — it did not.** Every vocabulary was wrong, worst case 30/46 (§0.2.2). Closed by
   rewriting rather than hardening. The replacement is exact on **fourteen** families (§0.4).
0b. ✅ **CLOSED — `gemma-4` is supported at 46/46.** It was not a pre-tokenizer gap at all: it
   declares a third *encoding style* (SentencePiece surface form, merge-**list** ranking, no
   word-level splitting), now `Style::SpmBpe` + `PreTok::Lines`. **This unblocks ARTX11 §4.**
   ⛔ Its vocabulary ships 262 144 scores *and* 514 906 merges; only the merges are used, and
   believing the scores produces different ids with no error anywhere.
0c. ✅ **CLOSED — and the recorded diagnosis was wrong.** `command-r` reaches **46/46** under the
   starcoder splitter, which is exactly what llama.cpp assigns it. The miss was in *splitting*, not
   in merge application. ⚠️ Worth keeping as a warning: a plausible cause was written down, believed
   for a week, and was not the cause.
1. **SPM window size** (§4.2) — how many trailing tokens must be re-decoded for correct spacing?
   Measure; do not guess.
2. **Padded vocabularies** (§2.3) — the legitimate `tok.vocab_size() < model.vocab_size` case needs
   ARTX12 Part A's config parsing to distinguish it from a real mismatch.
3. ✅ **ANSWERED — yes, at 46/46, but not the way this question assumed.** Gemma-4 does *not* fit
   the SPM path: it uses a merge list, not scores. Running the SPM encoder over it would have
   produced wrong ids silently. See 0b.
4. ⛔ **Dependency weight — now the live one, and it got heavier.** The tokenizer was a standalone
   crate; step 4 folded it **into** `glcore`, so a gljax dependency on `glcore::tokenizer` pulls
   GGUF parsing and the quantization kernels with it. That trade bought an `opt-level = 3` override
   the standalone crate never had (**3.4× measured**, §0.5) and one fewer crate to keep in sync. If
   gljax's build weight becomes a problem, the fix is a Cargo feature that compiles `glcore` down to
   its tokenizer, not another extraction.
5. ⚠️ **Does the pre-token cache belong in a multi-tenant server?** It is thread-local and bounded
   (§0.5), so it cannot leak between requests on one thread — but its *hit rate* is a property of
   the traffic, and a shared host with adversarial prompts is a case nobody has measured.

---

# Appendix A — Design Decision Summary

| # | Decision | Rationale | Reversible? |
|---|---|---|---|
| D0 | ✅ **SUPERSEDED — the suspicion was right and the code is gone.** A13.0 ran: every vocabulary was wrong, so the tokenizer was rewritten rather than hardened, and the original file deleted | 9 tests, synthetic ~296-token fixtures, ASCII only; both merge paths disabled by construction (`vec![0.0; n]`, "no merges"). Reuse is now gated on `tokenizer_parity.rs`, which scores 14 families against reference **data** on every build | N/A |
| D1 | Reuse `glcore::tokenizer` (after D0); do not write a second BPE | 796 lines, both vocab families, both load paths — a third implementation would be a third place for the same bug | Medium |
| D2 | Depend on a `Tokenizer` **trait**, not on `glcore` directly | ARTX11 needs two live vocabularies simultaneously — real polymorphism | Trivial |
| D3 | `vocab_fingerprint` is **computed**, never trusted from metadata | ARTX11's `Identical` is a correctness precondition; a name is not evidence | Trivial |
| D4 | Validate tokenizer against model at `Session::new` | A vocab mismatch is fluent garbage, not an error | Trivial |
| D5 | Chat templating lives in `glserve`, not gljax | It is a serving concern that changes per model without changing the engine | Medium |
| D6 | ⛔ No Jinja2 interpreter; known templates only, refuse unknown | Large dependency + untrusted-input execution; a wrong template is silent | Medium |
| D7 | Streaming uses a stateful `IncrementalDecoder`, never `decode(one_token)` | Partial UTF-8 and context-dependent spacing both break per-token decode | Hard |
| D8 | Two decode strategies selected by vocab family (ByteLevel / Spm) | Byte concatenation is correct for one and not the other | Medium |
| D9 | Invalid (not merely incomplete) bytes are skipped, not buffered | Byte-level vocabs can emit non-UTF-8; buffering would stall the stream forever | Trivial |
| D10 | `finish()` emits one U+FFFD for a truncated tail | Silent truncation hides a real generation event | Trivial |
| D11 | The decoder is driven from `AcceptResult::tokens`, never draft candidates | Rejected tokens would corrupt the stream (ARTX11 §5) | Trivial |
| D12 | `VocabIntersection` built once at pairing; **coverage reported** | Quantifies cross-family degradation before generation, not after | Trivial |
| D13 | A13.3 gate: streamed deltas concatenate to the whole-sequence decode | One invariant catches three bug classes | N/A |

---

# Sources

- [Summary of the tokenizers | HuggingFace](https://huggingface.co/docs/transformers/en/tokenizer_summary) and [The tokenization pipeline](https://huggingface.co/docs/tokenizers/en/pipeline) — normalizer → pre_tokenizer → model → post_processor → decoder; BPE/Unigram/WordPiece/WordLevel.
- [Tokenizers | HuggingFace](https://huggingface.co/docs/transformers/en/fast_tokenizers) — `tokenizer.json` fields: version, truncation, padding, added_tokens, normalizer, pre_tokenizer, post_processor, decoder, model.
- [SentencePiece: Subword Tokenization with BPE and Unigram](https://mbrenndoerfer.com/writing/sentencepiece-subword-tokenization-bpe-unigram) — raw-text operation, whitespace as a token, no whitespace pre-tokenization.
- [How to Train and Choose a Custom Tokenizer with tiktoken, SentencePiece, and HF Tokenizers](https://www.bestaiweb.ai/how-to-train-and-choose-a-custom-tokenizer-with-tiktoken-sentencepiece-and-hf-tokenizers-in-2026/) — tiktoken skips normalization, regex chunk before BPE, ~2–3× faster.
- [Multibyte UTF-8 Characters Broken in Streaming Mode (TRT-LLM)](https://github.com/oobabooga/textgen/issues/6778) — `�` substitution happens before transmission; client-side buffering cannot recover it.
- [Incremental Detokenization · huggingface/tokenizers #1666](https://github.com/huggingface/tokenizers/issues/1666) — the tokenizer's spacing decision depends on surrounding token IDs.
- [Byte-level Tokenizers Unavoidably Enable LLMs to Generate Ill-formed UTF-8](https://openreview.net/pdf?id=j2hH02UVch) — partial and invalid UTF-8 as a structural property of byte-level vocabularies.
- [LLM Output Streaming and Real-Time Token Delivery Architectures](https://zylos.ai/research/2026-03-28-llm-output-streaming-token-delivery-architectures/) — streaming decode with buffered incomplete sequences.
- [Accelerating LLM Inference with Lossless Speculative Decoding Algorithms for Heterogeneous Vocabularies](https://arxiv.org/abs/2502.05202) — SLEM / TLI, the algorithms §5 supports.

**Repo-internal:** `glcore/src/tokenizer.rs` (796 lines: SPM + byte-level BPE, `from_gguf`,
`from_file`, `encode_chat`); `ARTX04` (checkpoint loading); `ARTX11 §2.3, §3.2, §5.1` (pairing
contract, heterogeneous vocabularies, `AcceptResult`); `ARTX16 §1.2, §1.5` (request lifecycle, SSE);
`memory/project_glproc_precision_gap_vs_llamacpp.md` (BOS-handling verification as a prerequisite for
trusting cross-engine numbers).
