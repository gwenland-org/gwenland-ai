## Summary

<!-- What does this PR do, and why? One paragraph is fine. -->

## Related issue

<!-- e.g. Closes #12 / GWEN-123. Delete if none. -->

## What kind of change?

- [ ] Bug fix
- [ ] New feature
- [ ] Performance (include before/after numbers, measured in production — probes lie)
- [ ] Docs / chore / refactor

## Checklist

- [ ] `cargo test -p glcore -p glproc` (plus the crates you touched) passes
- [ ] `cargo fmt --all` and `cargo clippy --workspace --all-targets -- -D warnings` are clean
- [ ] Added/updated tests for the change
- [ ] Added a note under `changelog/` for non-trivial changes (problem → root cause → fix)
- [ ] Touched a `cfg`-gated or GPU path? Built/tested it on a platform that actually exercises it
- [ ] No new dependencies — or the PR description answers the dependency policy in CONTRIBUTING (reason, logical impact, concrete use cases)
- [ ] No secrets, no tokens in remote URLs
