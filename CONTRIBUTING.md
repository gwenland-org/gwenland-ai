# Contributing

Thanks for wanting to help out. GwenLand is a local-first AI toolkit written in Rust, and this is the short version of how to build it, test it, and send changes.

## Getting oriented

The repo is a single Cargo workspace rooted at the repository root — run cargo commands from there. The engine stack is a set of `gl*` crates:

- `glcore` — shared foundation: tensor types, error handling, GGUF/safetensors parsers, the from-scratch tokenizer, the `GlEngine` trait every backend implements, and the runtime.
- `glproc` — the CPU inference engine (pure Rust, SIMD). This is the **numerical ground truth** the GPU backends are validated against.
- `glcuda`, `glvulkan`, `glmetal` — the GPU backends (CUDA is the furthest along; see `architecture/ArchGLML_X2.md`).
- `glcli` — the `gwen` binary: `cargo run -p glcli` runs local inference through the engines.

There is also a `packages/` group (`packages/core`, `packages/gltui`, `packages/mcp`) — the `gltui` terminal UI and MCP server. Per-session notes go in `changelog/`, and `Cargo.lock` is committed, so build with `--locked` if you want reproducible deps.

Note the workspace mixes editions: `glcore`/`glproc`/`glcli` are edition 2021, `packages/gltui` is edition 2024.

## What you'll need

A recent Rust toolchain via [rustup](https://rustup.rs) — 1.85 or newer, since some crates use edition 2024.

## Building and running

The CLI is the main entry point:

```bash
cargo build --release -p glcli          # produces target/release/gwen
cargo run -p glcli -- --help            # see the available commands
```

The terminal UI:

```bash
cargo run -p gltui
```

GPU support is opt-in. The CUDA backend (`glcuda`) loads the NVIDIA driver at runtime — no CUDA toolkit needed to build — and reports itself unavailable on machines without a driver, so the runtime falls back to the CPU engine.

## Running the checks

```bash
cargo test -p glcore -p glproc -p glcuda
cargo test -p gltui

cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
```

A few notes on the test suites:

- **glcuda's GPU tests skip themselves when no CUDA device is present** — they print `SKIP` and pass, so the suite is green on a GPU-less machine and meaningful on one with a GPU. On a GPU runner, run the parity/forward suites with `--test-threads=1` so the VRAM-leak check isn't perturbed by concurrent allocations.
- If any core tests touch process-global state (a panic hook, a test env var), run them single-threaded (`-- --test-threads=1`) so they don't race each other.

There's a GitLab pipeline in `.gitlab-ci.yml.disabled` that runs the checks. It's parked because GitLab's shared runners want a credit card; re-enable it by renaming it back once you've sorted that out, or just run the checks above by hand.

## A note on platform-specific code

GwenLand targets modest hardware — think an 11th-gen i3, 8 GB of RAM, no GPU, on Linux — served by an mmap loader that keeps the weight working set small. Two things follow. Don't hold extra full-size copies of weights and blow the memory budget. And when you write OS-specific code, gate it as narrowly as you can and actually compile it on the platforms it claims to support. We got bitten by exactly this: `MADV_DONTNEED` was under `#[cfg(unix)]`, but `memmap2::Advice::DontNeed` is gated off on macOS in some versions, so it built fine on Windows (where the block is skipped) and Linux, then broke a contributor's macOS build. It should have been `#[cfg(target_os = "linux")]`. If you touch a `cfg`-gated path, build it somewhere other than your own machine before you assume it compiles.

The same discipline applies to the GPU backends: hand-authored PTX must be pure ASCII with LF line endings (`ptxas` rejects a stray em-dash before it parses a single instruction), and every GPU kernel is validated against `glproc` within an explicit per-operation tolerance — see `architecture/ArchGLML_X2.md`.

## Branches, commits, and changelogs

Branch off `main` with something like `feature/gwen-123-short-description` — tie it to a `GWEN-XXX` issue when there is one. Keep commits focused and use conventional prefixes (`feat:`, `fix:`, `docs:`, `chore:`, `refactor:`, with an optional scope like `fix(glcuda):`). Prefer a new commit over rewriting shared history. For anything more than a trivial change, add a note under `changelog/` that walks through the problem, the root cause, and the fix — match the existing entries.

One hard rule: never commit secrets, and never put a token in a git remote URL (use a credential helper instead). If you spot a leaked credential, say something and get it rotated.

## Sending a change

Make your change, add tests and a changelog note, run the checks, then open a merge/pull request against `main` and reference the issue. Keep each PR to one thing — split unrelated fixes apart.

## License

By contributing you're agreeing that your work is under the project's MIT License with the Commons Clause (see [LICENSE](LICENSE)) — free for personal and research use, commercial use by separate agreement.
