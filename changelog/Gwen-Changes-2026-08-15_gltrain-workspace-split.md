**Type:** Architecture fix — `packages/core` no longer violates "Inference First" Rule 6.
**Status:** ✅ Done. Root `cargo build/check --workspace` has zero `candle` in its
dependency tree; the training arm still builds standalone.

---

## Problem

`gl-agent-skills/architecture-skills/inference-first.md` Rule 6 is explicit:
*"no external ML dependencies... pulling in a crate that 'just does the
tokenizer' breaks the point of the project."* `packages/core/Cargo.toml` made
`candle-core`/`candle-nn`/`candle-transformers` a **mandatory default
feature** — its own comment said "always compiled since Cycle 6" — while
sitting in the same `[workspace]` as every inference crate. Every
`cargo build --workspace` on this repo compiled candle, whether or not
anyone touched the training code.

This surfaced as Finding F1 in a full technical-debt audit
(2026-08-15): the project's own north-star architecture rule, violated by
default, in the crate meant to be its training arm.

## Root cause

`packages/core` was a `[workspace]` member. Cargo does not let a member
opt out of being compiled by `--workspace` — the only way to keep candle out
of the inference build is to keep the crate out of the workspace entirely.

## Fix

1. `git mv packages/core gltrain` — history preserved, not a copy+delete.
2. Root `Cargo.toml`: removed from `members`, added to `exclude` alongside
   `.abandoned` — but for a different reason, spelled out in a comment so
   the next reader doesn't confuse "excluded" with "retired". `gltrain` stays
   under active development; it just resolves its own dependency graph,
   in its own `Cargo.lock`, independently of the root workspace.
3. `gljax/Cargo.toml`: dropped a comment that justified pinning `rand = "0.8"`
   by pointing at `packages/core`'s version — no longer true once the two
   crates don't share a dependency tree.

## Verification

- `cargo check --workspace` at root: clean, and `grep -c 'name = "candle' Cargo.lock`
  → **0**.
- `cd gltrain && cargo check`: still compiles as its own independent
  workspace (Cargo walks up, finds the root excludes it, treats
  `gltrain/Cargo.toml` as its own workspace root) — same 11 pre-existing
  warnings, nothing newly broken by the move.
- Bonus: `gltrain`'s own `[profile.*]` section, previously silently ignored
  ("profiles for the non root package will be ignored" — because it was a
  workspace member), now actually takes effect.

## Not done here

- Whether `gltrain` eventually becomes its own repo is still open — decided
  for now as "stays a subfolder, isolated workspace" (audit's Q2), not a
  separate repo and not deferred.
- No code inside `gltrain` was triaged or cut — everything salvaged as-is
  (audit's Q3): 381 tests and real feature history (LoRA pipeline, layered
  training, checkpoint resume) with no functional bugs found in this pass,
  only the dependency-placement issue.
