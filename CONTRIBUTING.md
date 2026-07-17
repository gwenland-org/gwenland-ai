# Contributing

Thanks for wanting to help out. GwenLand is an **Inference First** AI engine written in pure Rust — correct inference on whatever hardware is present comes before everything else — and this is the short version of how to build it, test it, and send changes.

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

## Dependencies: the bar is deliberately high

GwenLand's core promise is a **from-scratch, fully understood engine**: no
external ML runtimes, no C bindings, no CMake — the GGUF parser, tokenizer,
kernels, and even the benchmark exporter are hand-written. Every crate you add
works against that promise: it's supply-chain surface we now have to trust and
audit, transitive dependencies we didn't choose, build time on every
contributor's machine, binary size on an 8 GB target, and a license to check.
So: **don't add trivial dependencies.** If five lines of `std` can do it, five
lines of `std` win — a helper crate for one call site (`left-pad`-class,
`lazy_static` where `std::sync::OnceLock` works, `itertools` for a single
`chunks` loop) will be rejected regardless of how popular it is.

The current dependency budget, so you know the baseline you're changing:

- **Engine crates are near-frozen.** `glcore` carries `thiserror`, `memmap2`,
  `byteorder`, `serde`/`serde_json` (metadata only — never on the hot path);
  `glproc` adds `num_cpus`; `glcuda` has *zero* external deps (the driver is
  `dlopen`ed, not linked); `glbench` is workspace-only **by charter**.
- **Interface crates get more latitude** — `glcli` uses `clap`; `gltui` uses
  `ratatui`/`crossterm`/`tokio`/`reqwest`. More latitude is not a free pass:
  the same questions below apply.
- **ML dependencies are never acceptable in any crate** — no torch/candle/ort
  bindings, no ggml FFI, no "just for reference" inference crates. That's the
  project.

If you believe a new dependency is justified, make the case **in the PR
description**, answering all three:

1. **Reason** — what does it do that `std`, an existing dependency, or a
   reasonable amount of our own code can't? "Saves me writing ~40 lines" is
   not a reason; "implements RFC-compliant X that is genuinely hard to get
   right (and security-relevant to get wrong)" is.
2. **Impact, argued logically** — how many transitive dependencies does it
   pull (check `cargo tree`)? What does it do to clean-build time and binary
   size? Which crates does it infect (an engine crate or a leaf tool)? Is it
   maintained, and what's the license? Does it touch the hot path or startup?
3. **Use cases** — the concrete, current use cases it unlocks (plural beats
   singular; "we might need it later" doesn't count). If only one call site
   uses it, say so — that's usually the argument *against* it.

Reviewers will hold the line here; expect "rewrite it by hand" as a common
answer, especially anywhere near `glcore`/`glproc`/`glcuda`. Removing a
dependency is always a welcome PR.

## Branches, commits, and changelogs

Branch off `main` with something like `feature/gwen-123-short-description` — tie it to a `GWEN-XXX` issue when there is one. Keep commits focused and use conventional prefixes (`feat:`, `fix:`, `docs:`, `chore:`, `refactor:`, with an optional scope like `fix(glcuda):`). Prefer a new commit over rewriting shared history. For anything more than a trivial change, add a note under `changelog/` that walks through the problem, the root cause, and the fix — match the existing entries.

One hard rule: never commit secrets, and never put a token in a git remote URL (use a credential helper instead). If you spot a leaked credential, say something and get it rotated.

## Sending a change

Make your change, add tests and a changelog note, run the checks, then open a merge/pull request against `main` and reference the issue. Keep each PR to one thing — split unrelated fixes apart.

## License

By contributing you're agreeing that your work is under the project's MIT License with the Commons Clause (see [LICENSE](LICENSE)) — free for personal and research use, commercial use by separate agreement.
