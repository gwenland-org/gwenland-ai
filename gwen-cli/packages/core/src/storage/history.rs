use std::{fs, io::Write, path::PathBuf};

use anyhow::Context;
use serde::{Deserialize, Serialize};

use crate::engine::stream::{ChatMessage, Role};

// ─── Wire format ──────────────────────────────────────────────────────────────

/// One line in `history.jsonl` — internal only; public interface uses `ChatMessage`.
///
/// @INFO — timestamp is stored per-message for debugging and future UX features
/// (e.g., "show messages from last session"). Not exposed via `ChatMessage` to keep
/// the TUI struct simple.
#[derive(Serialize, Deserialize)]
struct HistoryEntry {
    role: String,
    content: String,
    timestamp: String,
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Append-only conversation history stored at `~/.gwenland/history.jsonl`.
///
/// Each call to `append()` opens the file, writes one JSON line, and closes it.
/// `load()` reads line-by-line and skips malformed entries without panicking.
pub struct ConversationHistory {
    /// Resolved path — public so callers can override for tests or alternate profiles.
    pub path: PathBuf,
}

impl ConversationHistory {
    /// Resolve the default `~/.gwenland/history.jsonl` path.
    pub fn new() -> Self {
        Self {
            path: crate::storage::paths::GwenPaths::history_file(),
        }
    }

    /// Read every message from the history file, skipping malformed lines.
    ///
    /// Returns an empty `Vec` if the file does not exist yet (first run).
    /// @INFO — loads the full file, but callers should cap at the last N messages
    /// before passing to the TUI or API to avoid context-window overflow
    pub fn load(&self) -> anyhow::Result<Vec<ChatMessage>> {
        if !self.path.exists() {
            return Ok(Vec::new());
        }

        let content = fs::read_to_string(&self.path)
            .with_context(|| format!("failed to read history: {}", self.path.display()))?;

        let messages = content
            .lines()
            .filter(|l| !l.trim().is_empty())
            .filter_map(|line| {
                // @INFO — skip rather than fail on malformed lines; a corrupt entry
                // (e.g., from a previous crash mid-write) must not prevent startup
                let entry: HistoryEntry = serde_json::from_str(line).ok()?;
                let role = match entry.role.as_str() {
                    "assistant" => Role::Assistant,
                    _ => Role::User,
                };
                Some(ChatMessage {
                    role,
                    content: entry.content,
                })
            })
            .collect();

        Ok(messages)
    }

    /// Append one message to the history file.
    ///
    /// Opens the file in append mode — does NOT load the file into memory.
    /// Creates `~/.gwenland/` if it does not exist.
    /// @DANGER — do NOT open with `create_new` or `write(true)` without `append(true)`;
    /// either flag would truncate the history file on the second message.
    pub fn append(&self, msg: &ChatMessage) -> anyhow::Result<()> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create history dir: {}", parent.display()))?;
        }

        let entry = HistoryEntry {
            role: match msg.role {
                Role::User => "user".to_string(),
                Role::Assistant => "assistant".to_string(),
            },
            content: msg.content.clone(),
            timestamp: chrono::Utc::now().to_rfc3339(),
        };

        let mut line = serde_json::to_string(&entry)
            .context("failed to serialize history entry")?;
        line.push('\n');

        // @KEEP — create+append is the correct pair; create alone truncates, append
        // alone fails if the file is missing
        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .with_context(|| {
                format!("failed to open history for append: {}", self.path.display())
            })?;

        file.write_all(line.as_bytes())
            .context("failed to write history entry")?;

        Ok(())
    }

    /// Truncate the history file to zero bytes (wipes all messages).
    pub fn clear(&self) -> anyhow::Result<()> {
        if self.path.exists() {
            fs::write(&self.path, "")
                .with_context(|| format!("failed to clear history: {}", self.path.display()))?;
        }
        Ok(())
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

