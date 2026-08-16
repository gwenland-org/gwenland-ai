**Type:** Docs — two new Agent Skills, no code change.
**Status:** ✅ Done. Landed on `stumman-m1` (82f244b, eafab0b), pushed to `github`.

---

## Problem

`stumman/` was growing its own type vocabulary (`Tensor`, `Tape`, `GlProc`,
`SisdBackend`, ...) with no written naming rule, while a repo-wide two-character
prefix convention (`BE`/`KL`/`AG`/`TP`/`VL` + 12 more) existed only as an idea
in conversation, never as a skill an agent would actually read.

## What landed

Two Agent Skills, `gl-agent-skills/gwenland-naming-convention/SKILL.md` and
`gl-agent-skills/stumman-naming/SKILL.md`, mirrored into `.claude/skills/` for
Claude Code auto-discovery (link paths rewritten for the deeper location).

Both were grounded against the working tree before being written, which
caught three things a straight copy of the design sketch would have missed:

- **The prefix convention has 0 of 224 public types adopted** across gljax,
  glcore, glproc, glcuda, glserve, and stumman. Both skills now open with
  **TARGET STATE**, not a description of current code — so a future agent
  doesn't rename `ExecutionPolicy` to `VLExecutionPolicy` on sight.
- stumman's "current type map" was a rename target, not a map — nothing is
  renamed yet; `stumman/src/lib.rs:18-21` still re-exports `Tensor`, `Tape`,
  `GlProc`, `SisdBackend`.
- `TensorMeta` actually lives in `autograd/tape.rs`, not `autograd/node.rs`
  as the draft assumed; `BackwardFn` was missing from the table entirely.

Also documented, for the first time, the Breton sub-system codenames
(Kevrin/Karg/Kevskrid/Gwellaer) every stumman module header already carries
but nothing wrote down — including the `tensor/backend.rs` boundary case
where the codename follows the sub-system, not the folder it sits in.

`.gitignore` changed from `.claude/` to `.claude/*` + `!.claude/skills/` so
the mirrored skills ship while `.claude/settings.local.json` stays ignored.

## Root cause of the two mistakes this note itself corrects

This changelog entry is late — it should have landed in the same commit as
the skills. It didn't because I (Claude) initially told the user
`changelog/` didn't exist in the tree; that came from running `ls changelog/`
with the shell's working directory pointed elsewhere, not from checking the
real repo root. The folder has 18+ entries going back to 2026-07-09.

## Verification

`git log --oneline -3` on `stumman-m1` shows 82f244b (skills) and eafab0b
(unrelated percival plan, kept as a separate commit per branch-strategy.md
rule 1 — one PR, one topic). Both pushed; `git ls-remote --heads github
stumman-m1` matches local `HEAD` at the time.
