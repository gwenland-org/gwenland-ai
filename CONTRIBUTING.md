# Contributing to GwenLand

> "Your machine. Your models. Your rules."

GwenLand is a pure Rust, local-first AI lifecycle toolkit. We welcome contributors who share the philosophy: **minimal dependencies, maximum control.**

---

## Before You Start

- Read the [README](./README.md) to understand the architecture
- Check [open issues](https://gitlab.com/jinxsuperdev/gwenland/-/issues) for `good first issue` labels
- Join the discussion — DM @JinXSuper on Discord

---

## Development Setup

```bash
# Clone
git clone https://gitlab.com/jinxsuperdev/gwenland.git
cd gwenland/gwen-cli

# Build
cargo build

# Run tests
cargo test --all

# Run specific benchmark
cargo run --bin bench_layer_loader -- <path/to/model.gguf> --compare-full
```

**Requirements:**
- Rust 1.78+
- Windows / Linux / macOS
- No GPU required 😌

---

## Hard Constraints (Non-Negotiable)

These apply to ALL contributions:

- **Zero new dependencies** unless absolutely justified + discussed first
- **No unsafe** beyond existing mmap blocks
- **Windows compatible** — madvise must be stubbed on non-Unix
- **Binary size** — `gwenland.exe` must stay under 15MB stripped
- **All tests must pass** — `cargo test --all` green before PR

---

## Project Structure

```
gwen-cli/
└── packages/
    └── core/src/
        ├── diagnostics/   # Health checks, doctor
        ├── engine/        # Inference, loader, mmap
        ├── platform/      # OS abstractions
        ├── storage/       # Config, registry
        └── train/         # LoRA, layer loader, training loop
```

---

## Pull Request Template

```markdown
## Summary
<!-- What does this PR do? -->

## Motivation
<!-- Why is this needed? Link to issue if applicable -->
Closes #<issue_number>

## Changes
- 
- 
- 

## Test Coverage
- [ ] Unit tests added/updated
- [ ] Integration tests pass (`cargo test --all`)
- [ ] No new warnings introduced
- [ ] Binary size checked (`cargo build --release`)

## Hard Constraints Checklist
- [ ] Zero new dependencies (or justified in description)
- [ ] No new unsafe blocks
- [ ] Windows compatible
- [ ] `gwenland.exe` still under 15MB stripped

## Notes
<!-- Anything reviewers should know -->
```

---

## Commit Style

```
feat(train): add LayeredTrainingLoop for OOM-safe LoRA
fix(engine): correct GGUF_MAGIC little-endian byte order  
test(gwen216): add no-full-load invariant integration test
docs: update CONTRIBUTING with PR template
```

Format: `type(scope): description`
Types: `feat`, `fix`, `test`, `docs`, `refactor`, `bench`

---

## What We're Looking For

Good contributions:
- Performance improvements with benchmark proof
- Bug fixes with regression test
- New model format support (GGUF variants)
- Platform compatibility (Android/iOS prep)

Not accepted without discussion:
- New runtime dependencies
- Breaking changes to public API
- GPU-only features (GwenLand must run on CPU)

---

*GwenLand is built by one person on an i3 with 8GB RAM. If others need a GPU, we need mmap.* 🔥