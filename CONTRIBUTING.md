# Contributing

Thanks for wanting to help out. GwenLand is a local-first AI toolkit written in Rust, and this is the short version of how to build it, test it, and send changes.

## Getting oriented

The code is a Cargo workspace under `gwen-cli/` — that's the workspace root, not the repo root, so run cargo commands from there. Inside it, `packages/core` (`gwenland-core`) is where the real work lives: inference, training, benchmarks, storage, diagnostics. `packages/tui` (`gwenland-tui`) is the `gwenland` binary and its terminal UI; you invoke it as `gwen`. `packages/gui` is the Tauri 2 desktop app (React, Vite, Tailwind, pnpm). Per-session notes go in `gwen-cli/changelog/`, and `Cargo.lock` is committed, so build with `--locked` if you want reproducible deps.

## What you'll need

A recent Rust toolchain — edition 2024, so 1.85 or newer, via [rustup](https://rustup.rs). If you're touching the GUI you'll also want Node 20+ and pnpm (`corepack enable`), plus your platform's Tauri/WebView toolchain (WebView2 on Windows, WebKitGTK and the usual build deps on Linux).

## Building and running

For the CLI, which is the main thing:

```bash
cd gwen-cli
cargo build --release -p gwenland-tui     # produces target/release/gwenland
cargo run -p gwenland-tui -- doctor        # quick smoke test
```

For the desktop app:

```bash
cd gwen-cli/packages/gui
pnpm install
pnpm tauri dev      # dev server on :1420
pnpm tauri build    # installers under gwen-cli/target/release/bundle/
```

GPU support is off by default; turn it on at runtime with `CANDLE_CUDA=1` or `CANDLE_METAL=1`.

## Running the checks

```bash
cd gwen-cli

# Core tests run single-threaded on purpose — a few touch process-global state
# (the panic hook, the GWEN_HOME test env) and race each other otherwise.
cargo test -p gwenland-core --lib -- --test-threads=1

cargo test -p gwenland-tui
cargo fmt --all
cargo clippy -p gwenland-core -p gwenland-tui --all-targets -- -D warnings

# GUI frontend
cd packages/gui && pnpm typecheck && pnpm build
```

There's a GitLab pipeline in `.gitlab-ci.yml.disabled` that runs all of this. It's parked because GitLab's shared runners want a credit card; re-enable it by renaming it back once you've sorted that out, or just run the checks above by hand.

## A note on platform-specific code

GwenLand targets modest hardware — think an 11th-gen i3, 8 GB of RAM, no GPU, on Linux — served by an mmap loader that keeps roughly one model layer in memory at a time. Two things follow from that. Don't hold extra full-size copies of weights and blow the memory budget. And when you write OS-specific code, gate it as narrowly as you can and actually compile it on the platforms it claims to support. We got bitten by exactly this: `MADV_DONTNEED` was under `#[cfg(unix)]`, but `memmap2::Advice::DontNeed` is gated off on macOS in some versions, so it built fine on Windows (where the block is skipped) and Linux, then broke a contributor's macOS build. It should have been `#[cfg(target_os = "linux")]`. If you touch a `cfg`-gated path, build it somewhere other than your own machine before you assume it compiles.

## Branches, commits, and changelogs

Branch off `main` with something like `feature/gwen-123-short-description` — tie it to a Linear `GWEN-XXX` issue when there is one. Keep commits focused and use conventional prefixes (`feat:`, `fix:`, `docs:`, `chore:`, `refactor:`, with an optional scope like `fix(serve):`). Prefer a new commit over rewriting shared history. For anything more than a trivial change, add a note under `gwen-cli/changelog/` named `Gwen-Changes-YYYY-MM-DD...md` that walks through the problem, the root cause, and the fix — match the existing entries.

One hard rule: never commit secrets, and never put a token in a git remote URL (use a credential helper instead). If you spot a leaked credential, say something and get it rotated.

## Sending a change

Make your change, add tests and a changelog note, run the checks, then open a merge request against `main` on [GitLab](https://gitlab.com/jinxsuperdev/gwenland) and reference the issue. Keep each MR to one thing — split unrelated fixes apart.

## License

By contributing you're agreeing that your work is under the project's MIT License with the Commons Clause (see [LICENSE](LICENSE)) — free for personal and research use, commercial use by separate agreement.
