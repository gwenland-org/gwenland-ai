use std::{fs, io::Write, path::PathBuf};

use anyhow::Context;
use chrono::{DateTime, Local};

// ─── Layout constants ─────────────────────────────────────────────────────────

const SEP: &str = "═══════════════════════════════════════";

// ─── Public types ─────────────────────────────────────────────────────────────

/// Snapshot of TUI state captured when the session log is finalised.
#[derive(Default)]
pub struct SessionState {
    pub active_pane: String,
    pub scroll_offset: u16,
    pub input_buffer: String,
    pub message_count: usize,
}

pub enum LogLevel {
    Info,
    Warn,
    Error,
    Crash,
}

/// Structured crash information captured in the panic hook and written to [CRASH].
pub struct CrashInfo {
    pub reason: String,
    /// Formatted as "file:line", e.g. "packages/tui/src/panes/chat_pane.rs:142"
    pub location: String,
    /// Raw panic payload string (`info.to_string()`), written verbatim to [RAW].
    pub message: String,
    /// Filtered stack frames from `capture_simplified_trace()`.
    pub trace: Vec<String>,
}

/// Returned by `find_last_crashed_session()` when the previous run panicked.
pub struct CrashedSession {
    /// Absolute path to the `.txt` log file.
    pub log_path: PathBuf,
    /// Timestamp portion of the filename, e.g. "2026-05-29_10-00".
    pub ts: String,
}

// ─── Private types ────────────────────────────────────────────────────────────

struct LogEntry {
    time: String, // "%H:%M:%S"
    level: LogLevel,
    msg: String,
}

// ─── SessionLogger ────────────────────────────────────────────────────────────

/// Human-readable session log written to `~/.gwen/session/session_<ts>.txt`.
///
/// Lifecycle:
///   1. `new()` — allocate; creates the session directory.
///   2. `log()` — buffer entries as the session runs.
///   3. `update_state()` — capture TUI snapshot before exiting.
///   4. `finalize(None)` on clean exit — writes `Closed` header, no [CRASH] section.
///      `finalize(Some(crash))` from panic hook — writes `Crashed` header + full trace.
pub struct SessionLogger {
    pub path: PathBuf,
    pub started_at: DateTime<Local>,
    pub state: SessionState,
    entries: Vec<LogEntry>,
    // @INFO — stored so history.rs can use the same ts for history_<ts>.jsonl
    pub file_ts: String,
}

impl SessionLogger {
    /// Allocate a new logger and prepare `~/.gwen/session/` for writing.
    /// Does NOT write to disk yet — the file is created only in `finalize()`.
    pub fn new() -> anyhow::Result<Self> {
        let now = Local::now();
        let file_ts = now.format("%Y-%m-%d_%H-%M").to_string();

        let dir = crate::storage::paths::GwenPaths::session_dir();
        fs::create_dir_all(&dir)
            .with_context(|| format!("failed to create session dir: {}", dir.display()))?;

        let path = dir.join(format!("session_{file_ts}.txt"));

        Ok(Self {
            path,
            started_at: now,
            state: SessionState::default(),
            entries: Vec::new(),
            file_ts,
        })
    }

    /// Buffer one log entry. Entries are written in bulk by `finalize()`.
    /// Never fails — returns `Result` to keep the call site consistent with
    /// other fallible operations that may replace this in the future.
    pub fn log(&mut self, level: LogLevel, msg: &str) -> anyhow::Result<()> {
        let time = Local::now().format("%H:%M:%S").to_string();
        self.entries.push(LogEntry {
            time,
            level,
            msg: msg.to_string(),
        });
        Ok(())
    }

    /// Replace the TUI state snapshot. Call just before `finalize()`.
    pub fn update_state(&mut self, state: SessionState) {
        self.state = state;
    }

    /// Render and write the complete session log file.
    ///
    /// Called from two sites:
    /// - Normal exit (`app.rs`): `finalize(None)` → header shows `Closed`.
    /// - Panic hook (`main.rs`): `finalize(Some(crash))` → header shows `Crashed`.
    pub fn finalize(&self, crash: Option<CrashInfo>) -> anyhow::Result<()> {
        // Ensure the directory still exists (may not if new() was never persisted).
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create session dir: {}", parent.display()))?;
        }

        let started_display = self.started_at.format("%Y-%m-%d %H:%M:%S");
        let end_display = Local::now().format("%Y-%m-%d %H:%M:%S");
        // @INFO — "Closed " is 7 chars to align with "Crashed" in the header
        let end_label = if crash.is_some() { "Crashed" } else { "Closed " };

        let mut out = String::new();

        // ── Header ────────────────────────────────────────────────────────────
        out.push_str(SEP); out.push('\n');
        out.push_str("GwenLand Session Log\n");
        out.push_str(&format!("Started : {started_display}\n"));
        out.push_str(&format!("{end_label}: {end_display}\n"));
        out.push_str(SEP); out.push('\n');
        out.push('\n');

        // ── [STATE] ───────────────────────────────────────────────────────────
        out.push_str("[STATE]\n");
        out.push_str(&format!("Active pane   : {}\n", self.state.active_pane));
        out.push_str(&format!("Scroll offset : {}\n", self.state.scroll_offset));
        // Debug-format the input buffer so special chars are escaped visibly.
        out.push_str(&format!("Input buffer  : {:?}\n", self.state.input_buffer));
        out.push_str(&format!("Messages      : {}\n", self.state.message_count));
        out.push('\n');

        // ── [ERRORS] — omitted when session had no log entries ────────────────
        if !self.entries.is_empty() {
            out.push_str("[ERRORS]\n");
            for entry in &self.entries {
                let level_str = match entry.level {
                    LogLevel::Info  => "INFO",
                    LogLevel::Warn  => "WARN",
                    LogLevel::Error => "ERROR",
                    LogLevel::Crash => "CRASH",
                };
                // Level padded to 5 chars → single space → consistent column width.
                out.push_str(&format!("{} {:<5} {}\n", entry.time, level_str, entry.msg));
            }
            out.push('\n');
        }

        // ── [CRASH] / [TRACE] / [RAW] — only on crash ────────────────────────
        if let Some(ref c) = crash {
            out.push_str("[CRASH]\n");
            out.push_str(&format!("Reason  : {}\n", c.reason));
            out.push_str(&format!("Where   : {}\n", c.location));
            out.push_str(&format!("Message : {}\n", c.message));
            out.push('\n');

            out.push_str("[TRACE]\n");
            for frame in &c.trace {
                out.push_str(&format!("{frame}\n"));
            }
            if c.trace.is_empty() {
                out.push_str("(no frames captured — debug symbols may be stripped)\n");
            } else {
                // @INFO — capture_simplified_trace() already limits to 10 frames;
                // the full backtrace (stdlib + tokio) is in [RAW] below
                out.push_str("(truncated, see [RAW] below)\n");
            }
            out.push('\n');

            out.push_str("[RAW]\n");
            out.push_str(&c.message);
            out.push('\n');
            out.push('\n');
        }

        out.push_str(SEP); out.push('\n');

        // @KEEP — truncate(true) so a re-used timestamp always starts clean
        let mut file = fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&self.path)
            .with_context(|| format!("failed to open session log for write: {}", self.path.display()))?;

        file.write_all(out.as_bytes())
            .context("failed to write session log")?;

        Ok(())
    }
}

// ─── Backtrace capture ────────────────────────────────────────────────────────

/// Capture a filtered backtrace containing only frames from GwenLand's own code.
///
/// Filters for frames whose file path contains `"packages"` or whose demangled
/// function name contains `"gwen"`. Skips stdlib / tokio / reqwest noise.
/// Max 10 frames — the full panic output lands in `CrashInfo::message` / [RAW].
pub fn capture_simplified_trace() -> Vec<String> {
    let mut frames: Vec<String> = Vec::new();

    backtrace::trace(|frame| {
        if frames.len() >= 10 {
            return false; // stop iterating
        }

        let mut entry: Option<String> = None;

        backtrace::resolve_frame(frame, |sym| {
            if entry.is_some() {
                return; // one symbol per frame is enough
            }

            let name = sym
                .name()
                .map(|n| format!("{:#}", n))
                .unwrap_or_else(|| "<unknown>".into());
            let file = sym
                .filename()
                .map(|f| f.to_string_lossy().into_owned())
                .unwrap_or_default();
            let line = sym.lineno().unwrap_or(0);

            // @INFO — Windows paths use backslash; "packages" matches either separator
            let is_ours = file.contains("packages")
                || file.contains("gwenland")
                || name.contains("gwen");

            if is_ours {
                entry = Some(format!("→ {} ({}:{})", name, file, line));
            }
        });

        if let Some(e) = entry {
            frames.push(e);
        }

        true // continue to next frame
    });

    frames
}

// ─── Startup recovery check ───────────────────────────────────────────────────

/// Check `~/.gwen/session/` for the most recent session and return it if it crashed.
///
/// Called once at startup, before the TUI renders. Returns `None` if:
/// - The session directory does not exist yet (first run).
/// - The last session closed cleanly (`Closed :`).
/// - No session files are present.
pub fn find_last_crashed_session() -> Option<CrashedSession> {
    let dir = crate::storage::paths::GwenPaths::session_dir();
    if !dir.exists() {
        return None;
    }

    let mut paths: Vec<PathBuf> = fs::read_dir(&dir)
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension().map_or(false, |ext| ext == "txt")
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .map_or(false, |n| n.starts_with("session_"))
        })
        .collect();

    // Filenames start with the date so alphabetical = chronological order.
    paths.sort();
    let latest = paths.last()?;

    let content = fs::read_to_string(latest).ok()?;

    // @INFO — check for "Crashed" in the header; "Closed " means clean exit
    let crashed = content.lines().any(|l| l.starts_with("Crashed"));
    if !crashed {
        return None;
    }

    let ts = latest
        .file_stem()
        .and_then(|n| n.to_str())
        .and_then(|n| n.strip_prefix("session_"))
        .unwrap_or("")
        .to_string();

    Some(CrashedSession {
        log_path: latest.clone(),
        ts,
    })
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

