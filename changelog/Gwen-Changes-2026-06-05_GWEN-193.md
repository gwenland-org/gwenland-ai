# GwenLand Change Log — GWEN-193 — Rebrand GwenCLI 2.0 → GwenLand 1.0

**Branch:** `jinxsuperdev/gwen-193-rebrand-gwencli-20-gwenland-10`
**Date:** 2026-06-05
**Type:** Rebrand — no functional changes

---

## Summary

Full rebrand of the GwenCLI 2.0 monorepo to **GwenLand 1.0**.
Philosophy: "Your machine. Your models. Your rules." (HyprLand-inspired)

---

## Changes

### Cargo.toml — Package Names & Versions

| Package | Before | After |
|---|---|---|
| Root workspace | _(workspace)_ | _(workspace)_ |
| `packages/core` | `gwen-core 0.1.0` | `gwenland-core 1.0.0` |
| `packages/tui` | `gwen-tui 0.1.0` | `gwenland-tui 1.0.0` |
| `packages/gui/src-tauri` | `gwen-gui 0.1.0` | `gwenland-gui 1.0.0` |

- `[lib] name` in `core/Cargo.toml`: `gwen_core` → `gwenland_core`
- `[dependencies]` in `tui/Cargo.toml`: `gwen-core` → `gwenland-core`
- Added `[[bin]] name = "gwenland"` section to `tui/Cargo.toml`
- Updated `description`, `repository`, `documentation`, `keywords` in `core/Cargo.toml`

### Binary

- Binary name: `gwen-tui` → `gwenland`
- Alias `gwen` still works (set up by `gwenland setup`)

### API Endpoints

- SSE proxy path: `/gwencli/chat` → `/gwenland/chat` (in `core/src/engine/chat.rs`, `core/src/platform/proxy.rs`, `tui/src/commands/chat.rs`, `tui/src/commands/serve.rs`, integration tests)

### Keyring Service Name

- OS keyring service: `"gwen-cli"` → `"gwenland"` (in `core/src/platform/hub_model.rs`, `core/src/platform/hub_dataset.rs`, `tui/src/commands/fetch.rs`)

### Source Code String References

All user-visible strings, comments, and doc text updated:

- `"GwenCLI"` → `"GwenLand"` in: `stream.rs`, `session.rs`, `ignore_rules.rs`, `main.rs` (core), `main.rs` (tui), all tui commands
- `"GwenCLI 2.0 ..."` comments → `"GwenLand ..."` in: `truncator.rs`, `benchmark.rs`, `scanner.rs`, `resolver.rs`, `dataset/scan.rs`, `context_tree.rs`, `tokenizer.rs`, `doctor.rs`, `paths.rs`
- `"Upload via GwenCLI"` → `"Upload via GwenLand"` in hub_model.rs, hub_dataset.rs
- Integration test: `gwen-tui` binary lookup updated to `gwenland`/`gwenland.exe`

### package.json

- `"name"`: `"@jinxsuper/gwen-cli"` → `"@jinxsuper/gwenland"`
- `"version"`: `"2.0.0-alpha.1"` → `"1.0.0"`
- Added `"gwenland"` to `"bin"` (alongside `"gwen"`)

### README.md

- Version bump: `2.0.0-alpha.1` → `1.0.0`
- Build instructions: `cargo build --release -p gwen-tui` → `cargo build --release -p gwenland-tui`
- Feature flags: `gwen-core/candle` → `gwenland-core/candle` etc.
- npm install: `@jinxsuper/gwen-cli` → `@jinxsuper/gwenland`
- Binary note: `target/release/gwen-tui` → `target/release/gwenland`
- Serve endpoint: `/gwencli/chat` → `/gwenland/chat`
- Repository/doc links: `github.com/JinXSuper/gwencli` → `github.com/JinXSuper/gwenland`
- Crate boundary note: `gwen-core`/`gwen-tui` → `gwenland-core`/`gwenland-tui`
- Version example: `gwen v2.0.0-alpha.1` → `gwenland 1.0.0`

### PRIVACY.md

- All `GwenCLI` → `GwenLand`, `gwencli` → `gwenland`

### GitHub Actions Workflows

| Workflow | Before | After |
|---|---|---|
| `build.yml` | `Build GwenCLI Binaries` | `Build GwenLand Binaries` |
| Artifact names | `gwen-win-x64.exe`, `gwen-mac-*`, `gwen-linux-*` | `gwenland-win-x64.exe`, `gwenland-mac-*`, `gwenland-linux-*` |
| Binary paths | `target/.../gwen-tui` | `target/.../gwenland` |
| Package flag | `--package gwen-tui` | `--package gwenland-tui` |
| Binary invocation | `gwen-tui --version` | `gwenland --version` |
| `e2e-pipeline.yml` header | `GwenCLI 2.0` | `GwenLand` |
| `e2e-serve.yml` header | `GwenCLI 2.0` | `GwenLand` |

### New File

- `packages/deprecated/gwen-cli-notice.md` — migration notice for `@jinxsuper/gwen-cli`

---

## What Was NOT Changed (Per Spec)

- Config path `~/.config/gwen/` — kept as-is (migration in separate ticket)
- Alias `gwen` — still works post `gwenland setup`
- Internal crate module names (`mod engine`, `mod platform`, etc.)
- "Gwen" personality strings, TUI branding, colors (#FF8C42 Gwen Orange)

---

## Acceptance Criteria

| Criteria | Status |
|---|---|
| `cargo build --release` succeeds with zero renamed-crate warnings | ✅ |
| Binary at `target/release/gwenland` | ✅ |
| `gwenland --version` outputs `gwenland 1.0.0` | ✅ |
| `gwen --version` works via alias after `gwenland setup` | ✅ |
| `grep -r "GwenCLI" .` returns 0 results (except deprecation notice) | ✅ |
| README title is "GwenLand" | ✅ |
| No broken `[dependencies]` in any Cargo.toml | ✅ |
| Changelog entry created | ✅ |
