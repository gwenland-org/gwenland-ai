use futures_util::StreamExt;
use tokio::sync::mpsc;

use crate::engine::windowing::{extract_relevant_windows, format_windowed_output, WindowConfig};

/// Conversation turn role.
/// Defined here (core) so both `history` and `tui::chat_pane` share the same type.
#[derive(Debug, Clone)]
pub enum Role {
    User,
    Assistant,
}

/// A single completed conversation turn (user prompt or assistant reply).
/// `Clone` is required so history load can slice the last N messages via `.to_vec()`.
#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
}

// @INFO — system prompt gives Gwen her personality; expose via `gwen --configuration`
// in a future ticket so power users can override without recompiling
const DEFAULT_SYSTEM_PROMPT: &str =
    "You are Gwen, GwenLand's AI assistant. You are helpful, concise, and slightly witty.";

pub const INFERENCE_SERVER_DOWN: &str = "inference server is not running";
pub const INFERENCE_SERVER_DOWN_HINT: &str =
    "start one with `gwen serve <path/to/model.gguf>`";

/// User-facing message for TUI history when the inference server is unreachable.
pub fn inference_server_down_message() -> String {
    format!(
        "error: {}\nhint:  {}",
        INFERENCE_SERVER_DOWN, INFERENCE_SERVER_DOWN_HINT
    )
}

pub fn is_inference_server_down_error(msg: &str) -> bool {
    msg.contains(INFERENCE_SERVER_DOWN)
}

/// Events emitted by `stream_chat` as inference progresses, token by token.
pub enum StreamEvent {
    /// One token fragment from the model's delta.
    Token(String),
    /// Inference completed cleanly; no more events will follow.
    Done,
    /// Non-recoverable error (runtime down, HTTP error, mid-stream failure).
    Error(String),
}

/// A file to be injected into the chat context.
///
/// When `WindowConfig.enabled == true`, only relevant windows are sent.
/// When disabled, the full `content` is injected verbatim.
pub struct FileContext {
    pub path: String,
    pub content: String,
}

/// Stream a chat completion from the native inference proxy (or any OpenAI-compatible endpoint).
///
/// `files` — optional file context to inject before the conversation history.
///   Each file is compressed via relevance windowing when `config.enabled == true`.
/// `config` — windowing configuration loaded from `~/.gwenland/config/config.json`.
///
/// Sends `StreamEvent::Token` for each delta, `StreamEvent::Done` on `[DONE]`,
/// or `StreamEvent::Error` if the runtime is unreachable or returns an error.
/// Never panics; all errors are surfaced through the channel.
///
/// # @INFO — health check runs before the stream so a "not running" error surfaces
/// immediately rather than after a timeout waiting for the connection to fail
pub async fn stream_chat(
    base_url: &str,
    model: &str,
    messages: Vec<serde_json::Value>,
    files: Option<Vec<FileContext>>,
    config: &WindowConfig,
    tx: mpsc::Sender<StreamEvent>,
) -> anyhow::Result<()> {
    // @DANGER — do NOT reuse a shared client here; this function is spawned as an
    // isolated task and must not share connection pools with the proxy (proxy.rs)
    let client = reqwest::Client::new();

    // Health check — GET /health before attempting the stream
    let health_ok = client
        .get(format!("{base_url}/health"))
        .send()
        .await
        .map(|r| r.status().is_success())
        .unwrap_or(false);

    if !health_ok {
        let _ = tx
            .send(StreamEvent::Error(inference_server_down_message()))
            .await;
        return Ok(());
    }

    // ── File context injection ────────────────────────────────────────────────
    // Extract the last user message as the relevance query for windowing.
    let query: String = messages
        .iter()
        .rev()
        .find(|m| m.get("role").and_then(|r| r.as_str()) == Some("user"))
        .and_then(|m| m.get("content").and_then(|c| c.as_str()))
        .unwrap_or("")
        .to_string();

    // Build file context messages to prepend (one system message per file).
    let mut file_messages: Vec<serde_json::Value> = Vec::new();
    if let Some(file_list) = files {
        for file in file_list {
            let ctx_text = if config.enabled {
                let windows = extract_relevant_windows(&file.content, &query, config);
                format_windowed_output(&file.path, &file.content, &windows)
            } else {
                // @INFO — full file passthrough when compression is disabled;
                // behaviour is identical to pre-JIN-164 for users who don't opt in
                format!("[File: {}]\n{}", file.path, file.content)
            };
            file_messages.push(serde_json::json!({
                "role": "system",
                "content": ctx_text,
            }));
        }
    }

    // ── Message assembly ─────────────────────────────────────────────────────
    // Order: system prompt → file context → conversation history
    let has_system = messages
        .iter()
        .any(|m| m.get("role").and_then(|r| r.as_str()) == Some("system"));

    let mut full_messages: Vec<serde_json::Value> = Vec::new();
    if !has_system {
        full_messages.push(serde_json::json!({
            "role": "system",
            "content": DEFAULT_SYSTEM_PROMPT,
        }));
    }
    full_messages.extend(file_messages);
    full_messages.extend(messages);

    let resp = client
        .post(format!("{base_url}/chat/completions"))
        .json(&serde_json::json!({
            "model": model,
            "messages": full_messages,
            "stream": true,
        }))
        .send()
        .await;

    let resp = match resp {
        Ok(r) if r.status().is_success() => r,
        Ok(r) => {
            let _ = tx
                .send(StreamEvent::Error(format!("HTTP {}", r.status())))
                .await;
            return Ok(());
        }
        Err(e) if e.is_connect() => {
            let _ = tx
                .send(StreamEvent::Error(inference_server_down_message()))
                .await;
            return Ok(());
        }
        Err(e) => {
            let _ = tx
                .send(StreamEvent::Error(format!("stream error — {}", e)))
                .await;
            return Ok(());
        }
    };

    // @DANGER — do NOT call .bytes().await here; that buffers the entire response
    // in memory and defeats the purpose of streaming — use bytes_stream() instead
    let mut raw_stream = resp.bytes_stream();
    let mut buf = String::new();

    while let Some(chunk) = raw_stream.next().await {
        let bytes = match chunk {
            Ok(b) => b,
            Err(e) => {
                let msg = if e.is_connect() {
                    inference_server_down_message()
                } else {
                    format!("stream error — {}", e)
                };
                let _ = tx.send(StreamEvent::Error(msg)).await;
                return Ok(());
            }
        };

        match std::str::from_utf8(&bytes) {
            Ok(text) => buf.push_str(text),
            Err(_) => continue, // skip malformed UTF-8 chunk
        }

        // Drain all complete SSE lines from the buffer.
        loop {
            let Some(nl) = buf.find('\n') else { break };
            let line = buf[..nl].trim_end_matches('\r').to_string();
            buf = buf[nl + 1..].to_string();

            if line.is_empty() {
                continue;
            }
            let Some(data) = line.strip_prefix("data: ") else {
                continue;
            };
            if data.trim() == "[DONE]" {
                let _ = tx.send(StreamEvent::Done).await;
                return Ok(());
            }

            // @INFO — parse OpenAI-style delta JSON from the native proxy stream
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(data) {
                if let Some(content) = v
                    .get("choices")
                    .and_then(|c| c.get(0))
                    .and_then(|c| c.get("delta"))
                    .and_then(|d| d.get("content"))
                    .and_then(|c| c.as_str())
                {
                    if !content.is_empty()
                        && tx
                            .send(StreamEvent::Token(content.to_string()))
                            .await
                            .is_err()
                    {
                        // Receiver dropped — ChatPane was closed; abort silently
                        return Ok(());
                    }
                }
            }
        }
    }

    let _ = tx.send(StreamEvent::Done).await;
    Ok(())
}
