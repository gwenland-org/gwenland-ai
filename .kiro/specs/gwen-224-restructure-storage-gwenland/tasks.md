# Implementation Plan: GWEN-224 — Restructure storage to ~/.gwenland/{config,models,crash-logs}/ + readable crash reports

## Overview

Breaking change, no migration logic. Status at audit time (2026-06-16): Waves 1–2 were
already implemented in a prior session (storage/paths.rs, config.rs, registry.rs all
resolve under `~/.gwenland/`) but never recorded in a spec. This plan documents that
prior work and completes the remaining waves: panic hook + crash reports (extended with
OS-level signal handling per follow-up notes), `gwen doctor` integration, and docs.

1. **Wave 1 (Path constants)** — `GwenPaths` root/config/models/crash-logs, `GWEN_HOME`
   env override for tests, auto-create-on-call. **Already done.**
2. **Wave 2 (Config + registry audit)** — `config.rs`/`registry.rs` resolve exclusively
   via `GwenPaths`; grep audit confirms no leftover `~/.config/gwen` literals. **Already done.**
3. **Wave 3 (Crash reporting)** — Rust panic hook + OS-level signal handler (SIGSEGV/
   SIGABRT/SIGILL/SIGBUS on Unix, unhandled exception filter on Windows) writing a
   human-readable `crash-<timestamp>.txt`, tagged with the active Surface (Cli/Tui/Gui/Serve).
4. **Wave 4 (`gwen doctor` integration)** — report existence/writability of
   `config/`, `models/`, `crash-logs/` under the new root.
5. **Wave 5 (Docs)** — README/CHANGELOG breaking-change notice, no auto-migration.

Each wave ends with `cargo check --workspace` + relevant `cargo test` before the next wave.

---

## Tasks

### Wave 1 — Path constants (already done, verified)

- [x] 1.1 `packages/core/src/storage/paths.rs`: `GWENLAND_DIR = ".gwenland"`, `GWEN_HOME` env override
- [x] 1.2 `GwenPaths::config_dir()` → `~/.gwenland/config/`
- [x] 1.3 `GwenPaths::models_dir()` → `~/.gwenland/models/`
- [x] 1.4 `GwenPaths::crash_logs_dir()` → `~/.gwenland/crash-logs/`
- [x] 1.5 Each `*_dir()` calls `ensure_dir` (create_dir_all) on every call
- [x] 1.6 Unit tests: `path_dirs_resolve_under_gwenland_root`, `path_dirs_are_created_on_first_call`, `file_paths_use_new_layout`

### Wave 2 — Config + registry path audit (already done, verified)

- [x] 2.1 `storage/config.rs` reads/writes via `GwenPaths::config_dir()` / `config_file()`
- [x] 2.2 `storage/registry.rs` reads/writes via `GwenPaths::models_dir()` only
- [x] 2.3 Workspace grep audit: no remaining `~/.config/gwen` (or XDG-style) literals outside comments documenting the breaking change
- [x] 2.4 Round-trip tests: `config_round_trips_through_gwenland_config_json`, `registry_round_trips_through_gwenland_models_dir`

### Wave 3 — Panic hook + OS signal handler + crash reports

- [x] 3.1 `packages/core/src/diagnostics/crash_report.rs` (new):
  - [x] `Surface` enum (`Cli`, `Tui`, `Gui`, `Serve`) + process-wide `AtomicU8`/`OnceLock` setter (`set_surface`) and getter
  - [x] `CrashContext` struct: version, git-hash, command line, surface, OS/arch detail (via `sysinfo`)
  - [x] `write_panic_report(info: &PanicHookInfo, ctx: &CrashContext) -> Option<PathBuf>` — formats the report (see format below) and writes to `GwenPaths::crash_logs_dir().join("crash-<ts>.txt")`; returns `None` on write failure, never panics
  - [x] `write_signal_report(signal_name: &str, ctx: &CrashContext) -> Option<PathBuf>` — minimal-alloc variant safe to call from a signal context (pre-formats what it can, single `write` syscall where possible)
  - [x] Backtrace capture gated on `RUST_BACKTRACE` env var (`backtrace::Backtrace::new()` already a dep)
- [x] 3.2 Install panic hook in `packages/tui/src/main.rs` `fn main()` (before `Cli::parse()`), calling `crash_report::write_panic_report`; chain after the existing TUI-specific hook (terminal restore must still run)
- [x] 3.3 Install panic hook equivalent in `packages/gui/src-tauri/src/main.rs`
- [x] 3.4 OS-level signal handler:
  - [x] Unix: `signal-hook` crate, register SIGSEGV/SIGABRT/SIGILL/SIGBUS → `write_signal_report` → re-raise default disposition
  - [x] Windows: `windows` crate `SetUnhandledExceptionFilter` → `write_signal_report` → return `EXCEPTION_CONTINUE_SEARCH`
  - [x] Document signal-handler-safety constraint (no alloc/lock) in module-level comment
- [x] 3.5 Hook/handler failure (can't write file, dir missing) must not itself panic/crash — falls back silently to stderr-only behavior already provided by Rust's default panic output
- [x] 3.6 Unit test: trigger a controlled panic inside `std::panic::catch_unwind` in a test harness with `GWEN_HOME` pointed at a tempdir; assert crash-log file written with expected fields (timestamp, version, surface, panic message, file:line)
- [x] 3.7 Unit test: missing/unwritable `crash-logs/` dir does not crash the panic hook itself (`unwritable_crash_dir_does_not_panic_the_hook` — exercises `write_signal_report` returning `None`, same code path `write_panic_report` shares for the actual file write)

### Wave 4 — `gwen doctor` integration

- [x] 4.1 `packages/core/src/diagnostics/doctor.rs`: add `check_gwenland_root()` — existence + writability of `GwenPaths::root_dir()`, `config_dir()`, `models_dir()`, `crash_logs_dir()`
- [x] 4.2 Wire into `run_all_checks()`
- [x] 4.3 Confirm no doctor check still references the old `~/.config/gwen` path (none found in current code)
- [x] 4.4 Test: doctor output reflects current path state correctly (`reports_pass_when_all_dirs_present_and_writable`, `reports_fail_when_a_dir_does_not_exist`, `reports_fail_when_a_dir_is_not_writable` [Unix-only] — `GWEN_HOME` override + tempdir)

### Wave 5 — Docs + breaking change notice

- [x] 5.1 README/CHANGELOG: document breaking change — old `~/.config/gwen/` untouched, no migration, re-run `gwen fetch` post-upgrade
- [x] 5.2 Confirm no leftover references to the old path in help text/error messages/comments (audit via grep; update any found)

---

## Acceptance Criteria (from Linear)

- [x] All config reads/writes resolve to `~/.gwenland/config/config.json`
- [x] All model fetch/registry operations resolve to `~/.gwenland/models/` — audit confirms no remaining hardcoded/alternate paths
- [x] Panic anywhere in `gwen` writes a human-readable crash report to `~/.gwenland/crash-logs/crash-<timestamp>.txt`
- [x] `gwen doctor` reports the new `~/.gwenland/` structure
- [x] No auto-migration — old `~/.config/gwen/` left untouched, breaking change documented

## Validation Commands (run before marking Done)

```text
cargo test -p gwenland-core --lib storage
cargo test -p gwenland-core --lib diagnostics::crash_report
cargo check -p gwenland-core
cargo check --workspace
cargo run -p gwenland-tui -- doctor   # manual check: confirm new paths reported
```
