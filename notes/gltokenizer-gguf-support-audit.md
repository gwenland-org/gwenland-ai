# GGUF tokenizer support audit — `glcore::tokenizer`

**Date:** 2026-07-28 · **Module:** `glcore/src/tokenizer/` · **Oracle:**
llama.cpp reference vectors (`models/ggml-vocab-*.gguf.inp` / `.out`), whose
expected ids were produced by HuggingFace `tokenizers` — i.e. comparison is
against *data*, not against another implementation.

⚠️ This was the standalone `gltokenizer` crate until step 4 folded it into
`glcore` (§7). Older references to `gltokenizer::Tokenizer` mean
`glcore::tokenizer::GllmTokenizer`.

Regenerate this table at any time:

```
cargo run  -p glcore --example tokenizer_audit --release
cargo test -p glcore --release                    # enforces it
GLTOK_REQUIRE_CORPUS=1 cargo test -p glcore       # and fails if it cannot
```

---

## 1. Status of every vocabulary in the corpus

| Vocabulary | GGUF `model` / `pre` | Parity | Status |
|---|---|---:|---|
| `qwen2` | gpt2 / qwen2 | **46/46** | ✅ supported |
| `qwen35` | gpt2 / qwen35 | **50/50** | ✅ supported — *§3.1 closed* |
| `llama-bpe` | gpt2 / llama-bpe | **46/46** | ✅ supported |
| `llama-spm` | llama / default | **46/46** | ✅ supported |
| `gpt-2` | gpt2 / gpt-2 | **46/46** | ✅ supported |
| `starcoder` | gpt2 / starcoder | **46/46** | ✅ supported |
| `refact` | gpt2 / refact | **46/46** | ✅ supported |
| `mpt` | gpt2 / mpt | **46/46** | ✅ supported |
| `command-r` | gpt2 / command-r | **46/46** | ✅ supported — **newly closed** |
| `phi-3` | llama / default | **46/46** | ✅ supported |
| `gemma-4` | gemma4 / — | **46/46** | ✅ supported — **newly closed** |
| `falcon` | gpt2 / falcon | **46/46** | ✅ supported — **closed 2026-07-28, §4.1** |
| `deepseek-coder` | gpt2 / deepseek-coder | **46/46** | ⚠️ supported, *shape approximated* — §3.2 |
| `deepseek-llm` | gpt2 / deepseek-llm | **46/46** | ⚠️ supported, *shape approximated* — §3.2 |
| `gpt-neox` | gpt2 / *(absent)* | — | ⛔ **refused** — §4.2 |
| `aquila` | gpt2 / *(absent)* | — | ⛔ **refused** — §4.2 |
| `baichuan` | llama / *(absent)* | — | 🕐 loads (SPM), **no reference vectors** — §4.3 |
| `bert-bge` | bert / bert-bge | — | ⛔ out of scope — WordPiece, not BPE |
| `nomic-bert-moe` | t5 / default | — | ⛔ out of scope — Unigram, not BPE |

**14 families verified exact, up from 10.** `command-r` and `gemma-4` are
described in §2; `falcon` closed later the same day once exact Unicode tables
landed (§6). `gpt-neox` and `aquila` moved the other way — they used to load,
and should not have; see §4.2.

---

## 2. What this audit found that was not the question

The audit was scoped to "give the four open families a clear status". It
instead exposed a defect in the part that was assumed fine.

### 2.1 ⛔ The pre-tokenizer name table was wrong for 13 of 24 names

`pretok_from_name` mapped GGUF's `tokenizer.ggml.pre` string onto one of four
splitter shapes. It had been written by grouping names that *look* related.
llama.cpp assigns them by which `regex_exprs` arm the name reaches, and those
two groupings are not the same. Checked entry by entry against
`llama.cpp/src/llama-vocab.cpp`:

| Name | Was mapped to | Actually is | Consequence |
|---|---|---|---|
| `default` | GPT-2 arm | a 4-expression fallback pipeline | wrong split on any GGUF missing `pre` |
| `codeshell` | GPT-2 arm | starcoder shape | digits absorb a leading space |
| `smollm`, `exaone`, `minerva-7b` | cl100k | starcoder shape | wrong digit + contraction handling |
| `chatglm-bpe` | Qwen2 (1-digit) | cl100k (3-digit) | every multi-digit number retokenized |
| `trillion` | Qwen2 | plain GPT-2 arm | wrong across the board |
| `gpt-4o`, `llama4`, `tekken` | cl100k | case-split lookahead pattern | inexpressible; now refused |
| `bloom`, `gpt3-finnish`, `viking` | GPT-2 arm | a single unrelated expression | inexpressible; now refused |
| `deepseek-v3`, `seed-coder`, `bailingmoe` | Qwen2 | three distinct other arms | now refused |
| `smaug`, `poro` | (any) | **not llama.cpp names at all** (`smaug-bpe`, `poro-chat`) | dead entries |

None of these had reference vectors, so **every one of them would have shipped
silently wrong token ids** — the exact failure mode
`feedback_reference_parity_not_roundtrip` records, one layer up from where it
was found last time. The table is now organised by regex arm, with the
inexpressible names listed as explicit refusals and the trap pinned by
`gguf::tests::name_groups_follow_the_regex_not_the_family`.

### 2.2 `ignore_merges` and forced BOS were also mis-keyed

`ignore_merges` is a property of llama.cpp's `pre_type`, **not** of the regex
shape, so it does not follow the grouping above: `dbrx`, `smaug-bpe`, `glm4`
and `chatglm-bpe` share llama3's pattern but not its flag. The old list
(`llama3 | llama-bpe | llama4 | gpt-4o`) had two wrong entries and was missing
seven right ones.

The same llama.cpp arm also sets `add_bos = true` **unconditionally**,
overriding `tokenizer.ggml.add_bos_token`. `gltokenizer` honoured only the
metadata, so a llama-3 GGUF declaring `false` would have lost its BOS and
shifted every position by one. Now handled by `force_add_bos`.

### 2.3 `command-r` — the recorded cause was wrong

`gguf.rs` carried a note claiming command-r reached 45/46 and diverged "in
merge application, not in splitting", on a long whitespace run. Measured: it
reaches **46/46** under the starcoder shape, which is exactly what llama.cpp
assigns it. The miss was the splitter. The note has been replaced.

### 2.4 `gemma-4` — a third encoding style, not a missing name

Not a pre-tokenizer gap at all. Gemma-4 declares `tokenizer.ggml.model =
gemma4`, and the crate only understood `gpt2` and `llama`. Its actual shape is
a hybrid neither existing style covers:

* SentencePiece **surface form** — `▁` for spaces, raw UTF-8 (no GPT-2 byte
  remap), `<0xNN>` byte fallback;
* **merge-list** ranking, as byte-level does — *not* per-token scores;
* **no word-level pre-splitting** — merges run across whole lines, cut only at
  newline runs (`PreTok::Lines`);
* a newline run present in the vocabulary is emitted whole before merging.

⚠️ The vocabulary ships **both** 262 144 scores and 514 906 merges. Only the
merges are used. Believing the scores would produce different ids with no error
anywhere — which is why this is a named `Style::SpmBpe` rather than a flag on
`Style::Spm`.

**This closes the blocker `project_gljax_artx_series` records against
ARTX11 §4** (Gemma named as the multi-architecture target).

---

## 3. Supported, with a stated limit

### 3.1 ~~`qwen35` — passes 50/50, but on an approximation~~ — **CLOSED**

Was: qwen35 widens the letter arm from `\p{L}` to `[\p{L}\p{M}]`, and the crate
had no `\p{M}` table — it passed only because `char::is_alphabetic` is a
superset of `\p{L}` that happens to absorb most combining marks.

**Closed 2026-07-28** by `src/unicode_tables.rs` (§6). `BpeSplit::QWEN35` now
carries a real `marks_are_letters` axis over an exact `\p{M}`, and the residual
case the corpus never covered — U+0301 COMBINING ACUTE, an `Mn` that is *not*
`Other_Alphabetic` — is pinned by
`pretok::tests::marks_attach_to_words_only_under_qwen35`.

⚠️ The same approximation had also made plain `qwen2` too *permissive*, in the
opposite direction. That is fixed by the same change, and no family's parity
moved: all 14 stayed exact across the swap.

### 3.2 `deepseek-coder` / `deepseek-llm` — measured exact, shape approximated

Both score 46/46 under the Qwen2 arm, which is the same evidence every other
entry rests on. But llama.cpp gives each a **multi-expression pipeline**, not
one arm — `deepseek-coder` uses `[\r\n]`, `\s?\p{L}+`, `\s?\p{P}+`, a CJK arm
and `\p{N}`.

These are provably not the same function: `\s?\p{L}+` lets a newline lead a
word, while the Qwen2 arm's `[^\r\n\p{L}\p{N}]?` explicitly excludes `\r` and
`\n`. They agree everywhere the corpus reaches and are known to differ
somewhere it does not.

Kept rather than refused, because refusing a family that passes every vector we
have would be its own kind of dishonesty — but the claim is narrowed here
rather than left implicit.

---

## 4. Open items

### 4.1 ~~⛔ `falcon` — diagnosed, blocked on Unicode tables~~ — **CLOSED**

**Closed 2026-07-28 at 46/46.** Kept below because the diagnosis is the useful
part: it was a *pipeline*, and no single arm could ever have reached it.

Falcon is a **three-stage pipeline**, not one arm.

```
1.  [\p{P}\$\+<=>\^~\|`]+          ← cut punctuation runs out first
2.  's|'t|…| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)
3.  [0-9][0-9][0-9]                ← then cut three-digit groups
```

Stage 1 is what the probes cannot reproduce, and it changes stage 2's result
rather than just adding cuts: on `"\n ="`, splitting `=` off first leaves `"\n "`
at end-of-segment, so `\s+(?!\S)` now keeps the whole whitespace run
(→ `[1212, 40]`) where the one-pass scan yields `[193, 204, 40]`.

The blocker was that stage 1 needs a **real `\p{P}`**. The crate's `is_punct`
is the complement class "not space, not letter, not number", which also catches
emoji and symbols — building falcon on it would have been exactly the kind of
approximation refused elsewhere. Landing `unicode_tables.rs` (§6) removed the
blocker, and `BpeSplit::FALCON` + `Passes` express all three stages directly.

### 4.2 ⛔ `gpt-neox`, `aquila` — now a judgement call, not a capability gap

Neither carries a `tokenizer.ggml.pre` key, so both reach llama.cpp's
`default` arm — which is **not** the GPT-2 shape they were previously loaded
as, but the 4-expression fallback in §2.1. llama.cpp itself logs
`GENERATION QUALITY WILL BE DEGRADED` when a GGUF lands there.

`BpeSplit::DEFAULT` now expresses that arm exactly (falcon's shape without the
backtick, plus a `\p{N}+` pass), so the *capability* gap is closed. They stay
refused for a different reason: **neither ships reference vectors**, so
enabling them would mean claiming support this crate cannot measure, on models
the upstream implementation itself calls degraded. A caller that wants them can
construct the vocabulary directly with `BpeSplit::DEFAULT`.

### 4.3 🕐 `baichuan` — loads, unverifiable

An SPM vocabulary that loads cleanly, with no `.inp`/`.out` vectors in the
corpus. **This is not a pass.** It is the one status the audit cannot resolve
from the data available, and it is recorded as such rather than assumed either
way.

---

## 5. Answering the constraint

> *"Jangan nambah model baru sebelum semua statusnya jelas."*

Every vocabulary in the corpus now has an explicit status backed by a number or
by a named blocker. Nothing is listed as supported on the strength of "it looks
right" or "it produces coherent text". Two families were added (`command-r`,
`gemma-4`) only after measuring 46/46; two were **removed** (`gpt-neox`,
`aquila`) because their status turned out to be "silently guessed".

The audit is enforced, not just written down: `tests/parity.rs` fails the build
if any of the fourteen regresses, **or** if either refused family starts
loading silently.

---

## 6. Addendum 2026-07-28 — exact Unicode categories

`src/unicode_tables.rs`, generated by `tools/gen_unicode_tables.py`, provides
exact `\p{L}` `\p{M}` `\p{N}` `\p{P}`. This closed §3.1 and §4.1 together, as
predicted.

**Provenance.** Built from the UCD range table llama.cpp generates from
`unicode.org/Public/UCD/latest/ucd/UnicodeData.txt`, and cross-checked
codepoint-by-codepoint against CPython's independent `unicodedata`. The
generator **aborts** if CPython classifies anything the primary table does not
— that direction cannot be a version delta, so it would mean the extraction is
wrong.

⚠️ **Not Unicode 15.1 as the task specified.** This machine's CPython is 3.11 /
UCD 14.0.0, so the cross-check is one revision behind the primary source. The
one-directional deltas are reported in the generated file's header (`\p{L}`
4970 codepoints, `\p{M}` 42, `\p{N}` 40, `\p{P}` 23) and are codepoints
assigned after 14.0. The exact primary version is whatever UCD-latest the
vendored llama.cpp checkout was generated from; it is not asserted here because
nothing in the checkout records it.

**Representation.** 1 298 sorted inclusive ranges plus a 128-entry ASCII
bitmap: ASCII resolves in one array index, the rest binary-searches. A flat
bitset over all 1 114 112 codepoints would be 136 KiB per class for no gain.

**Cost.** ⛔ Not measurable on this machine. `examples/tokbench.rs` gave a **5×
spread** across three back-to-back repeats of the identical binary (73.1 / 33.8
/ 174.9 ns/byte, same case), which is far larger than any plausible effect of
the swap. What can be said: the unit counts are byte-identical run to run, all
14 families stayed exact across the change, and pre-tokenization is a small
fraction of full encoding either way.

### Relation to Peek2 (arXiv 2601.05833)

The paper is real and its conclusion holds — but ⚠️ **this module was already
regex-free and single-pass**, by a design decision recorded in `pretok.rs`'s
header since the rewrite. Peek2's 1.11× is measured against a regex baseline
this crate never had, so it is not a speedup available here.

What the paper's framing *did* change is the character-class layer: precomputed
category tables instead of `char::is_alphabetic`. That is a **precision** fix,
and it is the one that closed falcon and qwen35.

Linearity is asserted structurally rather than by timing —
`pretok::tests::linear_on_pathological_input` feeds the scanner the uniform
runs that make a backtracking NFA go quadratic, across all six shapes, and
checks the split stays lossless. The scanner advances monotonically and never
revisits a byte, so `O(n)` is a property of its construction, not of a
measurement this machine cannot resolve.

---

## 7. Addendum 2026-07-28 — extracted into `glcore::tokenizer`

The `gltokenizer` crate is gone; its contents are now
`glcore/src/tokenizer/`, and the deprecated `glcore/src/tokenizer.rs` it once
coexisted with has been **deleted**. No new dependencies: glcore already had
`thiserror` and `serde_json`, and every crate that used `gltokenizer` already
depended on `glcore`.

| Was | Is now |
|---|---|
| `gltokenizer::Tokenizer` | `glcore::tokenizer::GllmTokenizer` |
| crate `src/lib.rs` | `src/tokenizer/mod.rs` |
| — | `src/tokenizer/style.rs` (`Style` out of `vocab`) |
| — | `src/tokenizer/spm.rs` (the two SPM-surface encoders) |
| `tests/parity.rs` | `glcore/tests/tokenizer_parity.rs` |
| `examples/audit.rs` | `glcore/examples/tokenizer_audit.rs` |
| `examples/tokbench.rs` | `glcore/examples/tokenizer_bench.rs` |
| `tools/gen_unicode_tables.py` | `glcore/tools/gen_unicode_tables.py` |

All moves are `git mv`, so history follows the files.

### ⛔ What the move broke, and what it cost

**A silent skip.** `tests/parity.rs` located the reference corpus by
`ancestors().nth(3)`. The move changed the crate root by one directory, so that
resolved to the wrong place — and because a missing corpus *skips*, the test
kept reporting `ok` in 0.00s while checking **nothing**. It was caught only by
noticing the runtime had dropped.

Two fixes, because the magic number was the symptom and the silent skip was the
disease:

* the corpus is now found by walking *up* until `llama.cpp/models` exists, so
  no future move can break it;
* `GLTOK_REQUIRE_CORPUS=1` turns the skip into a failure (for CI), and the test
  now also fails if the directory resolves but yields no vectors at all.

**Three deliberate deletions.** `glcore/tests/tokenizer_before_after.rs`,
`glproc/examples/tok_ab_file.rs`, and `ppl_check.rs`'s inline A/B block all
existed to compare the old implementation against the new one. Deleting the old
implementation makes them unbuildable. Their result is recorded rather than
re-derivable, and is restated in `ppl_check.rs` where it matters: on Qwen2.5 the
ids were **identical** (819/819, 201/201) and PPL landed at 36.19 against the
36.12 baseline.

### Verification

101 tests in `glcore` (55 pre-existing + 46 tokenizer), parity green in 3.7 s
against the real corpus, `cargo build --workspace --all-targets` exit 0, clippy
clean for `glcore`. All three Qwen models on this machine load with
byte-identical ids before and after.

⚠️ `cargo build -p glictus-caliburni --all-targets` fails on
`examples/diff_dump.rs` — **pre-existing**, unrelated: that file imports
`glcore::format` and `runtime::GlprocBackend`, both behind optional features it
does not declare. It compiles under workspace feature unification, which is why
the workspace build is green. Untouched by this work.

---

## 8. Addendum 2026-07-28 — pre-token cache

Merging dominates encoding: pre-tokenization measures ~4.9 ns/byte against
~180 ns/byte for a whole encode, so **the merge loop is ~97 % of the cost**.
Real text is long-tailed — measured on 113 KiB of this repo's own prose and
Rust source, **3 460 distinct pre-tokens carry 28 624 occurrences**, so the
same word is merged to the same ids ~8 times over.

A thread-local `HashMap<pre-token, ids>` replays that result instead.

| | ns/byte | MB/s | vs OFF |
|---|---:|---:|---:|
| cache OFF | 176.7 / 186.2 | 5.7 / 5.4 | — |
| **cold each pass** (fresh document) | 106.9 / 99.4 | 9.4 / 10.1 | **1.65× / 1.87×** |
| warm (long-lived process) | 51.9 / 50.8 | 19.3 / 19.7 | 3.41× / 3.67× |

⚠️ **Quote the cold number.** The bench encodes one input 40 times; a cache
left warm across those passes answers "how fast is re-encoding a document you
have already seen", which no server does. Cold clears before every pass, so the
only hits are *within* one document — hit rate **87.9 %**.

⛔ **The corpus is the measurement.** An earlier bench built its input by
repeating one sentence; that scores ~100 % and reports a speedup no real
workload reproduces. The default corpus is now real repository files.

### Correctness gates

The cache never changes ids — it replays what the merge engine already
produced for the identical pre-token under the identical vocabulary. Three
things enforce that rather than assert it:

* `tests/tokenizer_parity.rs` scores **every reference vector twice**, cache on
  and off, and fails on any difference. All 14 families, not one sampled string.
* `pretoken_cache_is_transparent` — unit-level, repeated-word inputs.
* ⛔ `pretoken_cache_does_not_leak_between_tokenizers` — the failure mode this
  design actually risks. The cache is thread-local, so two tokenizers on one
  thread share storage; without `Scratch::cache_owner` the second reads the
  first's ids back. Both tokenizers work perfectly alone, so nothing else in
  the suite would catch it. **Verified by mutation**: deleting the owner check
  makes exactly this test fail, with tokenizer B returning A's ids.

### ⭐ Resolved: the "3.4× codegen gap"

Recorded earlier as an unexplained difference between the tokenizer built
inside `glcore` (2.7 ns/byte) and as a separate crate (9.3 ns/byte), with
identical source. It is neither LTO nor inlining: `Cargo.toml` carries
`[profile.release.package.glcore] opt-level = 3`, and the standalone
`gltokenizer` crate had **no override**, so it inherited the workspace default
`opt-level = "z"` — optimise for *size*. Rebuilding `gltokenizer` at `da82a27`
with `opt-level = 3` gives **2.70 / 3.01 ns/byte**, matching glcore's
2.73 / 3.06.

So the step-4D extraction did make the tokenizer ~3.4× faster in production
builds, for a mundane and fully explicable reason.

### Where this sits against the state of the art

Gigatoken (Rød, 2026) reports 24.53 GB/s on a 144-core EPYC — **~170 MB/s per
core**, and its headline "989× faster than HuggingFace" is against HF's
24.8 MB/s, in a mode that trades exact output parity; its *exact* mode is
200–300×, and SentencePiece families gain only 7–22×.

Its published techniques, against ours:

| Technique | Here |
|---|---|
| Hand-written pre-tokenizer replacing regex (47 → 380 MiB/s) | ✅ since the rewrite |
| 256-byte class table, O(1) first-byte dispatch | ⚠️ 128-entry ASCII bitmap |
| Pre-token caching | ✅ this section |
| SWAR — 8 bytes as a `u64`, branchless class test | ❌ |
| Dual-cursor ILP (380 → 1049 MiB/s) | ❌ |

Our pre-tokenizer measures ~205 MB/s on this corpus, in the same band as their
hand-written-state-machine milestone. ⚠️ The remaining two techniques target
the ~3 % of encoding that pre-tokenization occupies, and dual-cursor ILP is
the *same shape* as the row-tile GEMM lead this repo has already rejected twice
for winning in a probe and going neutral in production. Neither is the next
move.
