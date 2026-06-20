# Contributing to GwenLand

Thanks for helping build GwenLand — an AI all-in-one toolkit that is local-first, <50 MB, and
zero-Python. This guide covers how the repo is laid out, how to build and test it, and the
conventions we follow.

> **"Speed is Everything, but Precise is more than Everything."**

---

## Repository layout

```
gwen-cli/                Rust workspace (resolver 2, edition 2024)
  packages/core/         gwenland-core — inference, training, benchmark, storage, diagnostics (lib)
  packages/tui/          gwenland-tui — the `gwenland` CLI + ratatui TUI (binary; invoked as `gwen`)
  packages/gui/          Tauri 2 desktop app (React + Vite + Tailwind v4, pnpm)
  changelog/             per-session change history (Gwen-Changes-*.md)
  Cargo.lock             committed — build with --locked for reproducible deps
benchmark/               dequant/quant math experiments (sibling of gwen-cli/)
```

The Cargo **workspace root is `gwen-cli/`**, not the repo root — run cargo commands from there.

---

## Prerequisites

- **Rust** stable **≥ 1.85** (the crates use edition 2024). Install via [rustup](https://rustup.rs).
- **For the GUI:** **Node 20+** and **pnpm** (`corepack enable`), plus your platform's Tauri/WebView
  toolchain (WebView2 on Windows; WebKitGTK + build deps on Linux).

---

## Build & run

```bash
# CLI / TUI (the main product) — produces target/release/gwenland
cd gwen-cli
cargo build --release -p gwenland-tui
cargo run -p gwenland-tui -- doctor      # quick smoke check

# GUI desktop app
cd gwen-cli/packages/gui
pnpm install
pnpm tauri dev                            # dev (Vite on :1420)
pnpm tauri build                          # installers under gwen-cli/target/release/bundle/
```

GPU is optional and off by default; opt in at runtime with `CANDLE_CUDA=1` / `CANDLE_METAL=1`.

---

## Tests & checks

```bash
cd gwen-cli

# Core library tests. --test-threads=1 is REQUIRED: several tests touch
# process-global state (std::panic::set_hook, the GWEN_HOME test env) and race otherwise.
cargo test -p gwenland-core --lib -- --test-threads=1

cargo test -p gwenland-tui
cargo check --workspace

# Formatting & lint
cargo fmt --all
cargo clippy -p gwenland-core -p gwenland-tui --all-targets -- -D warnings

# GUI frontend
cd packages/gui && pnpm typecheck && pnpm build
```

CI (GitLab, `.gitlab-ci.yml`) runs check → test → build on every push. Keep it green. Note that
`cargo check` deliberately excludes `gwenland-gui` (its `tauri::generate_context!` needs a built
frontend + icons), so the GUI is validated by the `gui:*` jobs instead.

---

## Platform-specific code (please read)

GwenLand's reference target is modest hardware: **i3 (11th gen), 8 GB RAM, no GPU, Linux**, served by
an mmap-based, OOM-safe layered loader. Two rules follow from this:

1. **Don't regress the memory budget.** The layered loader keeps essentially one model layer resident
   at a time (the GWEN-216 invariant). Don't hold extra full-size copies of weights.
2. **Gate OS-specific APIs precisely, and build on more than one OS.** Use the *narrowest* correct
   `cfg`. For example, `MADV_DONTNEED` is `#[cfg(target_os = "linux")]`, **not** `#[cfg(unix)]` — the
   `memmap2::Advice::DontNeed` variant is gated off on macOS in some versions, so a `cfg(unix)` block
   compiles on Windows (excluded) and Linux but breaks the macOS build. If you touch a `cfg`-gated
   path, compile it on the platforms it claims to support before sending the change.

---

## Branches, commits, and changelogs

- **Branch off `main`:** `feature/gwen-XXX-short-description` (tie to a Linear `GWEN-XXX` issue when
  one exists; otherwise a short descriptive name).
- **Conventional commits:** `feat:`, `fix:`, `docs:`, `chore:`, `refactor:` — scope optional, e.g.
  `fix(serve): …`. Keep commits focused; prefer a new commit over amending shared history.
- **Changelog:** every substantive change adds a file under
  `gwen-cli/changelog/Gwen-Changes-YYYY-MM-DD[_HH-MM | _GWEN-XXX].md` documenting **issue → root cause
  → fix**, plus testing notes. Match the style of the existing entries.
- **Never commit secrets.** Do not embed tokens in git remote URLs — use a credential helper. If you
  find a leaked credential, report it and rotate it.

---

## Submitting changes

1. Fork or branch, make your change, add tests and a changelog entry.
2. Run the checks above; make sure CI is green.
3. Open a **Merge Request against `main`** on GitLab
   (<https://gitlab.com/jinxsuperdev/gwenland>), referencing the `GWEN-XXX` issue.
4. Keep the MR scoped to one concern; split unrelated fixes into separate MRs.

---

## License

By contributing you agree that your contributions are licensed under the project's
**MIT License with Commons Clause** (see [LICENSE](LICENSE)). Free for personal and research use;
commercial use requires a separate agreement.
