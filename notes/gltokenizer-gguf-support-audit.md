# GGUF tokenizer support audit — `gltokenizer`

**Date:** 2026-07-28 · **Crate:** `glcore/gltokenizer` · **Oracle:** llama.cpp
reference vectors (`models/ggml-vocab-*.gguf.inp` / `.out`), whose expected ids
were produced by HuggingFace `tokenizers` — i.e. comparison is against *data*,
not against another implementation.

Regenerate this table at any time:

```
cargo run -p gltokenizer --example audit --release
cargo test  -p gltokenizer --release            # enforces it
```

---

## 1. Status of every vocabulary in the corpus

| Vocabulary | GGUF `model` / `pre` | Parity | Status |
|---|---|---:|---|
| `qwen2` | gpt2 / qwen2 | **46/46** | ✅ supported |
| `qwen35` | gpt2 / qwen35 | **50/50** | ✅ supported ⚠️ *see §3.1* |
| `llama-bpe` | gpt2 / llama-bpe | **46/46** | ✅ supported |
| `llama-spm` | llama / default | **46/46** | ✅ supported |
| `gpt-2` | gpt2 / gpt-2 | **46/46** | ✅ supported |
| `starcoder` | gpt2 / starcoder | **46/46** | ✅ supported |
| `refact` | gpt2 / refact | **46/46** | ✅ supported |
| `mpt` | gpt2 / mpt | **46/46** | ✅ supported |
| `command-r` | gpt2 / command-r | **46/46** | ✅ supported — **newly closed** |
| `phi-3` | llama / default | **46/46** | ✅ supported |
| `gemma-4` | gemma4 / — | **46/46** | ✅ supported — **newly closed** |
| `deepseek-coder` | gpt2 / deepseek-coder | **46/46** | ⚠️ supported, *shape approximated* — §3.2 |
| `deepseek-llm` | gpt2 / deepseek-llm | **46/46** | ⚠️ supported, *shape approximated* — §3.2 |
| `falcon` | gpt2 / falcon | 44/46 best probe | ⛔ **refused** — §4.1 |
| `gpt-neox` | gpt2 / *(absent)* | — | ⛔ **refused** — §4.2 |
| `aquila` | gpt2 / *(absent)* | — | ⛔ **refused** — §4.2 |
| `baichuan` | llama / *(absent)* | — | 🕐 loads (SPM), **no reference vectors** — §4.3 |
| `bert-bge` | bert / bert-bge | — | ⛔ out of scope — WordPiece, not BPE |
| `nomic-bert-moe` | t5 / default | — | ⛔ out of scope — Unigram, not BPE |

**13 families verified exact, up from 10.** The two closures (`command-r`,
`gemma-4`) are described in §2. `gpt-neox` and `aquila` moved the other way —
they used to load, and should not have; see §4.2.

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

### 3.1 `qwen35` — passes 50/50, but on an approximation

qwen35's pattern is qwen2's with the letter arm widened from `\p{L}` to
`[\p{L}\p{M}]` (and `\p{M}` excluded from the punctuation arm). This crate
carries no `\p{M}` table.

It passes anyway because `pretok::is_letter` is `char::is_alphabetic`, an
already-documented superset of `\p{L}` that absorbs most combining marks. The
corpus exercises this: vector #45 contains Khmer marks (U+17CB, U+17B7).

⚠️ **The residual gap is real and untested.** Marks outside `Other_Alphabetic`
— e.g. U+0301 COMBINING ACUTE — are punctuation to us and letters to qwen35.
No reference vector covers that case. Note also that the same approximation
makes us *too permissive* for plain `qwen2`, in the opposite direction.

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

### 4.1 ⛔ `falcon` — diagnosed, blocked on Unicode tables

Best probe 44/46 (cl100k shape). The cause is exact and known: falcon is a
**three-stage pipeline**, not one arm.

```
1.  [\p{P}\$\+<=>\^~\|`]+          ← cut punctuation runs out first
2.  's|'t|…| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)
3.  [0-9][0-9][0-9]                ← then cut three-digit groups
```

Stage 1 is what the probes cannot reproduce, and it changes stage 2's result
rather than just adding cuts: on `"\n ="`, splitting `=` off first leaves `"\n "`
at end-of-segment, so `\s+(?!\S)` now keeps the whole whitespace run
(→ `[1212, 40]`) where the one-pass scan yields `[193, 204, 40]`.

**Blocker: stage 1 needs a real `\p{P}` table.** The crate's `is_punct` is
"not space, not letter, not number", a superset that also catches emoji and
symbols. Implementing falcon on that superset would be an approximation of
exactly the kind refused elsewhere.

⭐ `\p{P}` and `\p{M}` are the *same* blocker: adding both closes falcon
exactly **and** removes §3.1's caveat. That is the single highest-value next
step for coverage, and it is a data problem, not a design one.

### 4.2 ⛔ `gpt-neox`, `aquila` — deliberately lost coverage

Neither carries a `tokenizer.ggml.pre` key, so both reach llama.cpp's
`default` arm — which is **not** the GPT-2 shape they were previously loaded
as, but the 4-expression fallback in §2.1. llama.cpp itself logs
`GENERATION QUALITY WILL BE DEGRADED` when a GGUF lands there.

Neither has reference vectors, so the mis-split would never have been caught.
Refusing is the honest response to missing metadata; supporting them means
implementing the `default` pipeline, which shares falcon's `\p{P}` blocker.

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
if any of the thirteen regresses, **or** if any of the three refused families
starts loading silently.
