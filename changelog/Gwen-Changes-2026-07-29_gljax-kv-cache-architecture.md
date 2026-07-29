**Type:** Implementation — gljax ARTX05, full KV cache architecture: buffer
donation, cache-aware prefill, decode, `CachedSession`.
**Status:** ~1,600 lines added, 233 tests green (up from 188), clippy
`-D warnings` clean, all 16 `dump_mlir` cases — including full Qwen2-0.5B-
shaped prefill and decode traces — parse-verified through jaxlib's real MLIR
parser. ⛔ **Nothing has executed.** Same blocker as every gljax session so
far: no PJRT plugin on this machine. The one check that actually matters
(token-for-token parity against the recomputation oracle) is written and
wired into CI, but has not run yet.
**Version:** `gljax` `0.1.0` → **`0.2.0`** — first version bump since the
crate's creation; a mode flag on the already-CI-gated `Session` would have
been a patch, a second `CachedSession` type with its own compiled programs
and its own execution path is not.

---

## Executive Summary

Gate A5 (previous session) closed on coherence but left `Session::generate`
doing full-sequence recomputation every decode step — O(n·S) for n tokens,
measured at 163 GFLOP/s meaning a 512-token generation costs ~58 minutes.
That session ended with one open question: buffer donation
(`input_output_alias`), flagged as *the* blocker before any of the rest was
worth building.

This session answered that question by measuring it, then built the entire
remaining architecture on top: a cache-aware prefill trace, a decode trace,
and a `CachedSession` runtime type that actually runs prefill-then-decode.
Every trace-level piece is unit-tested and externally parse-verified; the
Session-level piece is wired into CI behind a real parity check that has not
run yet. This is a construction report, not a completion report — see "What's
still open" before trusting any of it.

## ⭐⭐ The exact unknown ARTX05 left open, resolved by measuring it

ARTX05 §6 guessed at buffer donation's mechanism (a module-level
`input_output_alias` attribute) and admitted it wasn't sure. Rather than
copy the guess into an emitter, this session traced a real `jax.jit(f,
donate_argnums=(0,))` on the machine's local `jaxlib` and read the StableHLO
it actually lowers to:

```mlir
func.func public @main(%arg0: tensor<4xf32> {tf.aliasing_output = 0 : i32}, ...)
```

It is a **per-argument attribute on `func.func`**, spelled
`tf.aliasing_output = N : i32` where `N` is the aliased *output* index — not
a module-level attribute, and not any of the other names ARTX05/ARTX07/ARTX09
speculated (`donated_input_indices`, `executable_output_lists`). Confirmed to
parse through gljax's own generic-op textual form before a line of Rust was
written.

Shipped as `ParamKind::KvCache`, `ParamDesc.alias_output`,
`FuncBuilder::kv_cache()` / `FuncBuilder::alias_output()` (rejects aliasing a
weight or plain input — only a `KvCache` param may be donated, or a checkpoint
tensor could get overwritten in place), and validation in `finish()` that the
alias index is in range and the aliased output's shape matches exactly.

## Step 2 — cache-aware prefill (`trace_prefill_with_cache`)

`trace_layer` — the function `trace_forward` already used per layer — gained
an `emit_cache_tap: bool` parameter. When true, it also returns the post-RoPE
K and un-rotated V, already transposed to the cache's layout, computed from
values attention had already produced. `trace_forward` passes `false` and
discards the tap; the existing `each_layer_emits_the_expected_op_counts` test
still passes **unmodified**, confirming the tap adds zero cost to the
recomputation oracle.

`trace_prefill_with_cache` runs the *identical* attention `trace_forward`
does — full self-attention, static causal mask, no cache reads, because
prefill only ever fills the cache — plus one bulk `kv_cache::write_range` per
layer per tensor (a new sibling to the existing `write_at`, same
`dynamic_update_slice` machinery without the single-position restriction).
Both `k_cache`/`v_cache` — one tensor spans **all layers**, addressed by a
trace-time-constant `layer` index per unrolled iteration — are donated to
their matching outputs.

## Step 3 — decode (`trace_decode`)

⭐⭐ **The runtime position mask needed zero new StableHLO ops.** ARTX05 §3
describes building it from scratch via `iota` + `compare` + `select`. But the
*existing* static `[1,1,W,W]` causal mask already encodes exactly the rule
decode needs: `causal_mask[pos, j] = 0 if j<=pos else -inf` **is** row `pos`
of that same matrix. `ops::attention::causal_mask_row` is one `dynamic_slice`
on a mask that already exists, already parses, already has tests — not three
new op emitters, each of which (per this crate's own history — the empty
`array<i64: >` bug, the missing reduce-region braces) carries first-use
syntax risk of its own.

`ops::rope::rope_neox_at` reads the RoPE table at a runtime `pos` via
`dynamic_slice` instead of a static `slice`; a test pins that it emits
identical rotate-half arithmetic (1 negate, 1 concatenate, 2 multiply, 1 add)
to the static path — the dynamic offset is the only thing that changed.
`trace_layer_decode` is a **separate function** from `trace_layer` (ARTX05
itself draws the same line for `gqa_attention_decode`), not a branch on it —
the attention body genuinely differs (cache read/write, a sliced mask row, a
dynamic RoPE offset).

Per-layer op count was hand-derived before writing the test, then checked:

```
6 dynamic_slice   (2 RoPE calls × 2 table reads each, + 2 read_window)
2 dynamic_update_slice   (write_at × 2, K and V)
+ 1 dynamic_slice, once — the mask row, not per layer
```

**Every count matched on the first test run** — as strong a signal as a
machine with no plugin can give that the op graph is structurally what was
intended. Dumping the real Qwen2-0.5B-shaped trace confirms the donation
landed correctly too:

```mlir
func.func @main(%v0: tensor<1x1xi32>, %v1: tensor<i32>,
                %v2: tensor<1x256x2x64xf32> {tf.aliasing_output = 1 : i32},
                %v3: tensor<1x256x2x64xf32> {tf.aliasing_output = 2 : i32}, ...)
  -> (tensor<1x1x151936xf32>, tensor<1x256x2x64xf32>, tensor<1x256x2x64xf32>)
```

## Step 4 — `runtime::CachedSession`

A **new type**, not a mode flag on `Session`. `Session::generate` is what
Gate A5 already validated end-to-end in CI; branching through its methods for
the cached path would put that CI history one merge away from a regression
neither type's tests would catch. `CachedSession` compiles both programs for
one shared `window` (used as both the padded-prompt bucket and the cache's
total capacity — simpler than two independently-sized buckets, at the cost of
prefill padding to the full window every call, which is the same one-time
cost `Session`'s own first forward pass already pays).

`build_args()` assembles each program's flat PJRT argument list by walking
its own `param_order` and dispatching on declared kind/name — not a
positional assumption. Necessary because prefill and decode declare different
interleavings (`pos` exists only in decode), and PJRT matches arguments by
position, so guessing wrong here silently binds the wrong buffer. Weight
buffers are uploaded **once** and shared by both executables; `bind_safetensors`
filters to `Signature.weights`, which preserves order regardless of what else
was interleaved, so both traces — built from the same `trace_layer` — provably
produce identical weight lists, and `CachedSession::open` refuses if they ever
don't.

`generate()` threads `pos` as the token's **absolute** position — the same
indexing prefill's RoPE/mask already used, not a "cache slot" index — and
always adopts whatever buffer handle PJRT hands back after a donated call,
rather than reusing the one it started with.

## The verification that actually matters: not run yet

`tests/wave_a5_kv_cache.rs::cached_decode_matches_the_recomputation_oracle_token_for_token`
builds both a `Session` (the unchanged oracle) and a `CachedSession` from the
same checkpoint, prompt, and sampling, and asserts the generated token
sequences are identical. This is the check ARTX05 flags as the headline risk:
a cache reading one position off produces fluent, wrong text that Gate A5's
coherence check would likely still pass — coherence is not the bar here,
exact agreement with the already-gated recompute path is.

Wired into `.github/workflows/gljax-pjrt.yml`'s `gate-a5` job — the only place
both a plugin and a model are configured — positioned before the long
coherence sweep so a plumbing failure is cheap to see. The workflow fails the
build if this test skips there, same convention as every other must-not-skip
check in that file.

🕐 **This has not run yet.** The next session's first job is getting it
executed — push the branch / open the PR — before building anything further
on top.

## What's still open

* Nothing here has executed against a real PJRT plugin. Every claim above is
  "traces, type-checks, and parses" — not "produces the right numbers."
* Prefill pads to the **full** window every call rather than a smaller,
  independently-sized prompt bucket — simpler, at a one-time cost that does
  not change the asymptotic story this feature exists to fix.
* Buffer donation's *runtime* behavior — whether XLA actually elides the copy,
  whether the donated PJRT buffer really becomes invalid after `execute()` —
  is unconfirmed. The mechanism is declared correctly in the MLIR; whether the
  plugin honors it as expected is exactly what the plugin run will show.
* The op-count tests are structural pins, not numerical checks — they would
  not catch RoPE reading the *wrong* row while still reading *a* row. Only the
  parity test catches that class, and it hasn't run.
