# Branch Strategy

> **Domain:** before-coding
> **Applies to:** all branches, commits, and PRs in this repository
> **Last updated:** 2026-07-17

## BEFORE YOU START

- [ ] I am NOT about to commit directly to `main`.
- [ ] My branch name states its type and ties to a `GWEN-XXX` issue when one exists.
- [ ] I know whether this work is product code (normal branch) or research (research branch / `Experimental/`).

## Context

`main` is the integration branch and must always build and pass the checks.
Branch names carry meaning here: they tell the reviewer whether a diff is
product engineering, a new engine bring-up, or research that may never merge —
and they get different review standards accordingly.

## Rules

1. **Branch off `main`, PR back into `main`.** One PR = one topic; split
   unrelated fixes apart.
2. **Branch naming:**
   - `feature/gwen-123-short-description` — normal product work (the
     CONTRIBUTING.md convention). Also accepted: `feat/…`, `fix/…` prefixes.
   - `engine/<name>` — bring-up or major work on a backend engine
     (e.g. `engine/glvulkan-bringup`).
   - `research/<name>` — experiments; results land in
     [`../../Experimental/`](../../Experimental/README.md) or a changelog
     note, and the branch is allowed to die without merging.
3. **Outside contributors use the fork model:** fork → branch → PR. No direct
   pushes to the org repo.
4. **Commits:** conventional prefixes — `feat:`, `fix:`, `docs:`, `chore:`,
   `refactor:`, `ci:`, optionally scoped (`fix(glcuda): …`). Keep commits
   focused. Prefer a new commit over rewriting shared history.
5. **Changelog notes:** any non-trivial change adds a note under
   [`../../changelog/`](../../changelog/) walking through problem → root cause
   → fix, matching the existing entries.
6. **Never commit secrets**, and never put a token in a git remote URL — use a
   credential helper. A leaked credential must be reported and rotated, not
   quietly removed.
7. Research branches must not silently change product code paths "while
   they're in there". If research reveals a product bug, that fix goes in its
   own `fix/…` branch.

## ✅ Correct Pattern

```text
git switch -c feature/gwen-231-q8-repack-fallback
# focused commits:
#   feat(glproc): add Q8_0 repack fallback for Q4_K tensors
#   docs(changelog): add Gwen-Changes-2026-07-17 note
# → PR to main, references GWEN-231, checks green
```

## ❌ Anti-Pattern (Never Do This)

```text
git commit -m "stuff" main                 ← direct to main, meaningless message
git push https://user:TOKEN@github.com/... ← token in remote URL
research/moe-idea branch that also "fixes" the tokenizer and bumps deps
                                           ← research contaminating product code
```

## GwenLand-Specific Notes

- The canonical hosting is dual-remote (GitHub `gwenland-org/gwenland-ai` is
  where CI runs). Open PRs where the CI is.
- Wave-based work usually lives on ONE branch across its waves; the gates
  (see [wave-confirmation-gates.md](wave-confirmation-gates.md)) happen in
  conversation, not as separate branches per wave.

## Related Skills

- [wave-confirmation-gates.md](wave-confirmation-gates.md)
- [check-existing-tests.md](check-existing-tests.md)
- [`../../CONTRIBUTING.md`](../../CONTRIBUTING.md) — the human-facing version of these rules
