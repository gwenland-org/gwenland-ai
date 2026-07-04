# GwenLand - GWEN-224: Storage Restructure to ~/.gwenland/ + Readable Crash Reports

**Date:** 2026-06-16 (WIB)
**Scope:** `storage/paths.rs`, `storage/config.rs`, `storage/registry.rs` (audit only — already on new layout),
new `diagnostics/crash_report.rs`, `diagnostics/doctor.rs`, `tui/src/main.rs`, `gui/src-tauri/src/main.rs`,
`gui/src-tauri/Cargo.toml`, `core/Cargo.toml`, `.kiro/specs/gwen-224-restructure-storage-gwenland/tasks.md`
**Type:** Breaking change, no migration logic.
**Status:** Waves 1-2 (path constants, config/registry audit) were found already implemented from a prior
session and are documented here retroactively. Waves 3-5 (crash reporting, doctor integration, docs)
implemented and validated this session.

---

## Executive Summary

GwenLand's storage moved off the XDG-style `~/.config/gwen/` convention onto a single
home-dotfile root: `~/.gwenland/{config,models,crash-logs}/`. This is a breaking change —
there is no auto-migration. Pre-1.0 users upgrading just re-run `gwen fetch <model>` to
repopulate models under the new path; old data at `~/.config/gwen/` is left untouched.

On top of the path restructure, `gwen` (CLI/TUI/GUI) now writes a human-readable crash
report on any panic or OS-level fault (segfault, illegal instruction, abort) to
`~/.gwenland/crash-logs/crash-<timestamp>.txt`, and `gwen doctor` reports on the health
of the new directory structure.

---

## What Was Already Done (audit, Waves 1-2)

`storage/paths.rs` already had a complete `GwenPaths` implementation:
- `GWENLAND_DIR = ".gwenland"`, root resolved via `dirs::home_dir()` with a `GWEN_HOME`
  env-var override (test-only escape hatch)
- `config_dir()`, `models_dir()`, `crash_logs_dir()`, plus `cache_dir()`,
  `eval_results_dir()`, `tmp_dir()`, `history_file()`, `session_dir()`
- every `*_dir()` accessor calls `create_dir_all` on each call (self-healing)
- unit tests asserting resolution under the new root and absence of the old
  `.config/gwen` substring

`storage/config.rs` and `storage/registry.rs` already resolved exclusively through
`GwenPaths`, with round-trip tests. A workspace-wide grep audit for `.config/gwen` /
`XDG_CONFIG` literals turned up only:
- negative test assertions (`assert!(!path.contains(".config/gwen"))`) — confirming the
  breaking change, not violating it
- historical, dated changelog entries / specs predating the restructure — correctly
  left alone as historical record

No code changes were needed for Waves 1-2; this session's contribution was verifying and
recording that state in a tracked spec file (`.kiro/specs/gwen-224-restructure-storage-gwenland/tasks.md`).

---

## What Was Built This Session (Waves 3-5)

### Crash reporting (`diagnostics/crash_report.rs`, new)

Two capture paths feed the same report format:

1. **Rust panic hook** (`install_panic_hook`) — chains after any existing hook (e.g. the
   TUI's terminal-restore hook), formats a full report with backtrace (gated on
   `RUST_BACKTRACE`), and writes it via `std::fs::write`. Failure to write never panics —
   it returns `None` and the original panic still propagates through the default printer.
2. **OS-level signal / unhandled-exception capture** (`install_signal_handler`) — for
   faults that don't go through Rust's panic machinery at all (segfaults, illegal
   instructions, aborts from native code such as candle/mistral.rs internals):
   - **Unix:** `signal-hook` watches `SIGSEGV`/`SIGABRT`/`SIGILL`/`SIGBUS` on a dedicated
     thread, writes a minimal report, restores default disposition, and re-raises so the
     process still terminates the way it normally would.
   - **Windows:** `SetUnhandledExceptionFilter` (via the `windows` crate,
     `Win32_System_Diagnostics_Debug` + `Win32_System_Kernel` features) decodes the
     exception code (Access Violation, Illegal Instruction, etc.), writes the report, and
     returns `EXCEPTION_CONTINUE_SEARCH`.

A process-wide `Surface` (`Cli`/`Tui`/`Gui`/`Serve`) is recorded via an atomic at startup
so every report says which front-end was running. A `CrashContext` (version, git hash,
full command line) is captured once via `init_context` and read by both capture paths
without allocating in the hot path of the signal handler beyond what Rust's own
formatting requires.

Report format (matches the spec exactly):

```text
GwenLand Crash Report
======================
Timestamp:   2026-06-16T14:32:07+07:00
Version:     gwen 1.0.0-rc3 (rev a1b2c3d)
Surface:     TUI
Command:     gwen train --resume checkpoint_000500
OS:          Windows 11 24H2 (build 26100), x86_64

Crash Type:  Rust panic

Panic:
  thread 'main' panicked at packages/core/src/train/adamw_state.rs:142:9
  shape mismatch: expected [256, 256], got [256, 128]

Backtrace: (set RUST_BACKTRACE=1 for full trace)

------------------------------------------------------
If this looks like a bug, please share this file when reporting.
```

### Wiring

- `tui/src/main.rs`: `init_context` + `install_panic_hook` + `install_signal_handler`
  called at the very top of `fn main()`, before clap parsing, so even argument-parsing
  panics are captured. Surface is refined once the subcommand is known (`Tui`/`Gui` for
  `Start`, `Tui` for `Chat`, `Serve` for `Serve`, `Cli` otherwise). The pre-existing
  TUI-specific panic hook (terminal restore) still chains correctly — it was installed
  *after* ours, so it runs first (terminal restored) and then calls through to the
  crash-report hook.
- `gui/src-tauri/src/main.rs` + `Cargo.toml`: GUI now depends on `gwenland-core` (a new,
  deliberate dependency — previously the Tauri shell had no core dependency at all) so it
  can reuse the same crash-report module rather than duplicating the format. Installed
  before `gwen_gui_lib::run()`.

### `gwen doctor` integration (Wave 4)

`diagnostics/doctor.rs` gained `check_gwenland_root()`, wired into `run_all_checks()`,
producing four independent check entries (`gwenland-root`, `gwenland-config`,
`gwenland-models`, `gwenland-crash-logs`) — each reports `Pass` if the directory exists
and a write-probe succeeds, `Fail` with a clear message otherwise. Kept as four entries
rather than one rolled-up bit so a partial setup (e.g. crash-logs/ exists but isn't
writable) is visible at a glance.

### Docs (Wave 5)

Root `CHANGELOG.md` documents the breaking change under `[Unreleased]`. This file is the
GWEN-224 per-session record. No leftover references to the old path were found in
current help text, error messages, or non-historical comments.

---

## Testing

```text
cargo test -p gwenland-core --lib -- --test-threads=1 diagnostics::
cargo test -p gwenland-core --lib -- --test-threads=1   # full suite
cargo check --workspace
```

New tests:
- `diagnostics::crash_report::tests::write_panic_report_captures_message_and_location`
- `diagnostics::crash_report::tests::write_signal_report_creates_readable_file`
- `diagnostics::crash_report::tests::unwritable_crash_dir_does_not_panic_the_hook`
- `diagnostics::doctor::tests::reports_pass_when_all_dirs_present_and_writable`
- `diagnostics::doctor::tests::reports_fail_when_a_dir_does_not_exist`
- `diagnostics::doctor::tests::reports_fail_when_a_dir_is_not_writable` (Unix-only —
  Windows' directory read-only attribute is cosmetic and doesn't block writes, so there's
  no portable way to simulate "exists but unwritable" without ACL plumbing)

**Note on `--test-threads=1`:** `std::panic::set_hook` is process-global. Running the new
panic-hook test concurrently with other tests that panic on purpose (several pre-existing
tests in `engine::inference::selector` and elsewhere intentionally panic/unwrap-fail) races
on that global hook and can poison the `GWEN_HOME` test-env mutex for unrelated tests.
Single-threaded execution avoids this; it is a pre-existing test-isolation characteristic
of any test that touches `std::panic::set_hook`, not a bug introduced by this change.

**Pre-existing, unrelated failures:** `engine::inference::selector::tests::{empty_stop_sequences_ok,
relative_gguf_ok, tilde_expand}` fail in this environment because the `candle-ggqr` backend
is unavailable here. Confirmed unrelated — `selector.rs` is untracked/untouched by this
session's diff.

---

## Acceptance Criteria

- [x] All config reads/writes resolve to `~/.gwenland/config/config.json`
- [x] All model fetch/registry operations resolve to `~/.gwenland/models/` — audit confirms no remaining hardcoded/alternate paths
- [x] Panic anywhere in `gwen` writes a human-readable crash report to `~/.gwenland/crash-logs/crash-<timestamp>.txt`
- [x] `gwen doctor` reports the new `~/.gwenland/` structure
- [x] No auto-migration — old `~/.config/gwen/` left untouched, breaking change documented
