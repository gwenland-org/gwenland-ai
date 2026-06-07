// engine/chat.rs — Core chat protocol for `gwen chat` and `gwen serve`.
//
// Handles session history, SSE streaming from the local proxy, config
// persistence, and the --json token-per-line output mode for agent consumption.
//
// Wire types use neutral names (ChatRequest / StreamChunk) because the
// upstream backend is now always native inference — no Ollama dependency.
// The SSE payload format is kept identical so the GUI and TUI consumers
// require zero changes.
//
// @DANGER: Never handle HF_TOKEN here. Auth lives in platform::hub_model.
// @EDITABLE: SSE_ENDPOINT path and default port are constants below.

use anyhow::{bail, Result};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use tokio::sync::mpsc::UnboundedSender;

use crate::storage::config::{read_last_used_model, save_last_used_model};

// ── constants ─────────────────────────────────────────────────────────────────

pub const DEFAULT_PORT: u16 = 1136;
pub const SSE_PATH: &str = "/gwenland/chat";
pub const HEALTH_PATH: &str = "/health";

// ── exit codes ────────────────────────────────────────────────────────────────

pub const EXIT_OK: i32 = 0;
pub const EXIT_ERROR: i32 = 1;
pub const EXIT_SERVER_UNREACHABLE: i32 = 3;

// ── message / history types ───────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MessageRole {
    User,
    Assistant,
    System,
}

impl MessageRole {
    pub fn as_str(&self) -> &'static str {
        match self {
            MessageRole::User => "user",
            MessageRole::Assistant => "assistant",
            MessageRole::System => "system",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: MessageRole,
    pub content: String,
}

/// In-memory session: ordered list of messages for the current conversation.
#[derive(Debug, Default)]
pub struct ChatSession {
    messages: VecDeque<ChatMessage>,
    /// Approximated total tokens received (char count / 4).
    pub total_tokens: u64,
}

impl ChatSession {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, msg: ChatMessage) {
        self.messages.push_back(msg);
    }

    pub fn messages(&self) -> impl Iterator<Item = &ChatMessage> {
        self.messages.iter()
    }

    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }

    /// Build the message array for the next request body.
    pub fn build_request_messages(&self) -> Vec<ChatMessageWire> {
        self.messages
            .iter()
            .map(|m| ChatMessageWire {
                role: m.role.as_str().to_string(),
                content: m.content.clone(),
            })
            .collect()
    }

    pub fn finalize_assistant_turn(&mut self, content: String) {
        self.total_tokens += (content.len() as u64).saturating_add(3) / 4;
        self.push(ChatMessage {
            role: MessageRole::Assistant,
            content,
        });
    }
}

// ── wire types (neutral — compatible with native proxy SSE format) ─────────────

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ChatMessageWire {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessageWire>,
    pub stream: bool,
}

/// One chunk from the SSE stream.
/// `message.content` carries the token; `done` signals end of generation.
#[derive(Debug, Deserialize)]
pub struct StreamChunk {
    #[serde(default)]
    pub message: Option<ChatMessageWire>,
    #[serde(default)]
    pub done: bool,
}

// ── events (used by TUI and JSON output mode) ─────────────────────────────────

pub enum ChatEvent {
    Token(String),
    Done,
    Error(String),
}

// ── JSON output types (--json mode, AI agent / script consumption) ─────────────

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum JsonOutputEvent {
    Token { content: String },
    Done { total_tokens: u64 },
    Error { message: String },
}

// ── server reachability ───────────────────────────────────────────────────────

/// Check if the native inference proxy is reachable.
pub async fn probe_server(port: u16) -> bool {
    let url = format!("http://localhost:{}{}", port, HEALTH_PATH);
    let client = match reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(3))
        .build()
    {
        Ok(c) => c,
        Err(_) => return false,
    };
    client.get(&url).send().await.map(|r| r.status().is_success()).unwrap_or(false)
}

pub fn server_unreachable_message(port: u16) -> String {
    let hint = match read_last_used_model() {
        Some(m) => format!("gwen serve --model {}", m),
        None => "gwen serve --model <model-id>".to_string(),
    };
    format!(
        "  ⚠ No server running on port {}. Start with:\n    {}",
        port, hint
    )
}

// ── SSE streaming ─────────────────────────────────────────────────────────────

/// Stream a single chat turn from the native proxy via SSE.
/// Parses the stream format:
///   `{"message":{"role":"assistant","content":"token"},"done":false}`
///   `{"done":true}`
/// Also handles raw SSE `data: <json>` lines and `data: [DONE]` sentinel.
pub async fn stream_chat(
    session: &mut ChatSession,
    model: &str,
    port: u16,
    tx: UnboundedSender<ChatEvent>,
) {
    let url = format!("http://localhost:{}{}", port, SSE_PATH);
    let request_messages = session.build_request_messages();

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()
        .unwrap_or_else(|_| reqwest::Client::new());

    let body = ChatRequest {
        model: model.to_string(),
        messages: request_messages,
        stream: true,
    };

    let response = match client.post(&url).json(&body).send().await {
        Ok(r) => r,
        Err(e) => {
            let _ = tx.send(ChatEvent::Error(e.to_string()));
            return;
        }
    };

    if !response.status().is_success() {
        let _ = tx.send(ChatEvent::Error(format!(
            "server returned HTTP {}",
            response.status()
        )));
        return;
    }

    let mut byte_buf: Vec<u8> = Vec::new();
    let mut assistant_buf = String::new();
    let mut stream = response.bytes_stream();

    while let Some(chunk) = stream.next().await {
        let bytes = match chunk {
            Ok(b) => b,
            Err(e) => {
                let _ = tx.send(ChatEvent::Error(e.to_string()));
                return;
            }
        };

        byte_buf.extend_from_slice(&bytes);

        loop {
            let Some(nl) = byte_buf.iter().position(|&b| b == b'\n') else {
                break;
            };
            let line_bytes = byte_buf[..nl].to_vec();
            byte_buf.drain(..=nl);

            let line = String::from_utf8_lossy(&line_bytes)
                .trim_end_matches('\r')
                .to_string();
            if line.is_empty() {
                continue;
            }

            let data = line
                .strip_prefix("data: ")
                .unwrap_or(line.as_str());

            if data == "[DONE]" {
                session.finalize_assistant_turn(std::mem::take(&mut assistant_buf));
                let _ = tx.send(ChatEvent::Done);
                return;
            }

            match serde_json::from_str::<StreamChunk>(data) {
                Ok(chunk) => {
                    if let Some(msg) = &chunk.message {
                        if !msg.content.is_empty() {
                            assistant_buf.push_str(&msg.content);
                            let _ = tx.send(ChatEvent::Token(msg.content.clone()));
                        }
                    }
                    if chunk.done {
                        session.finalize_assistant_turn(std::mem::take(&mut assistant_buf));
                        let _ = tx.send(ChatEvent::Done);
                        return;
                    }
                }
                Err(_) => {
                    // Plain-text token fallback (some backends emit raw text).
                    if !data.is_empty() {
                        assistant_buf.push_str(data);
                        let _ = tx.send(ChatEvent::Token(data.to_string()));
                    }
                }
            }
        }
    }

    if !assistant_buf.is_empty() {
        session.finalize_assistant_turn(std::mem::take(&mut assistant_buf));
    }
    let _ = tx.send(ChatEvent::Done);
}

// ── JSON output mode (--json, for agent / script consumption) ──────────────────

/// Run a single chat turn in JSON output mode. Streams token objects to stdout:
///   `{"type":"token","content":"..."}`
///   `{"type":"done","total_tokens":123}`
pub async fn stream_chat_json(
    session: &mut ChatSession,
    model: &str,
    port: u16,
) -> Result<()> {
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<ChatEvent>();

    tokio::spawn({
        let url = format!("http://localhost:{}{}", port, SSE_PATH);
        let model = model.to_string();
        let request_messages = session.build_request_messages();

        async move {
            let client = reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(120))
                .build()
                .unwrap_or_else(|_| reqwest::Client::new());

            let body = ChatRequest {
                model,
                messages: request_messages,
                stream: true,
            };

            let response = match client.post(&url).json(&body).send().await {
                Ok(r) => r,
                Err(e) => {
                    let _ = tx.send(ChatEvent::Error(e.to_string()));
                    return;
                }
            };

            if !response.status().is_success() {
                let _ = tx.send(ChatEvent::Error(format!(
                    "server returned HTTP {}",
                    response.status()
                )));
                return;
            }

            let mut byte_buf: Vec<u8> = Vec::new();
            let mut stream = response.bytes_stream();

            while let Some(chunk) = stream.next().await {
                let bytes = match chunk {
                    Ok(b) => b,
                    Err(e) => {
                        let _ = tx.send(ChatEvent::Error(e.to_string()));
                        return;
                    }
                };
                byte_buf.extend_from_slice(&bytes);

                loop {
                    let Some(nl) = byte_buf.iter().position(|&b| b == b'\n') else {
                        break;
                    };
                    let line_bytes = byte_buf[..nl].to_vec();
                    byte_buf.drain(..=nl);

                    let line = String::from_utf8_lossy(&line_bytes)
                        .trim_end_matches('\r')
                        .to_string();
                    if line.is_empty() {
                        continue;
                    }

                    let data = line.strip_prefix("data: ").unwrap_or(&line).to_string();

                    if data == "[DONE]" {
                        let _ = tx.send(ChatEvent::Done);
                        return;
                    }

                    if let Ok(chunk) = serde_json::from_str::<StreamChunk>(&data) {
                        if let Some(msg) = &chunk.message {
                            if !msg.content.is_empty() {
                                let _ = tx.send(ChatEvent::Token(msg.content.clone()));
                            }
                        }
                        if chunk.done {
                            let _ = tx.send(ChatEvent::Done);
                            return;
                        }
                    } else if !data.is_empty() {
                        let _ = tx.send(ChatEvent::Token(data));
                    }
                }
            }
            let _ = tx.send(ChatEvent::Done);
        }
    });

    let mut assistant_buf = String::new();

    while let Some(ev) = rx.recv().await {
        match ev {
            ChatEvent::Token(tok) => {
                assistant_buf.push_str(&tok);
                let out = serde_json::to_string(&JsonOutputEvent::Token { content: tok })
                    .unwrap_or_default();
                println!("{}", out);
            }
            ChatEvent::Done => {
                session.finalize_assistant_turn(std::mem::take(&mut assistant_buf));
                let out = serde_json::to_string(&JsonOutputEvent::Done {
                    total_tokens: session.total_tokens,
                })
                .unwrap_or_default();
                println!("{}", out);
                return Ok(());
            }
            ChatEvent::Error(msg) => {
                let out = serde_json::to_string(&JsonOutputEvent::Error {
                    message: msg.clone(),
                })
                .unwrap_or_default();
                eprintln!("{}", out);
                bail!("{}", msg);
            }
        }
    }

    Ok(())
}

// ── session persistence ───────────────────────────────────────────────────────

pub fn persist_session_model(model: &str) {
    if let Err(e) = save_last_used_model(model) {
        eprintln!("warning: could not save last_used_model: {:?}", e);
    }
}

// ── headless (non-TUI) chat runner ────────────────────────────────────────────

pub struct HeadlessChatConfig {
    pub model: String,
    pub port: u16,
    pub system: Option<String>,
    pub json_mode: bool,
    pub messages: Vec<String>,
}

pub async fn run_headless_chat(cfg: HeadlessChatConfig) -> i32 {
    if !probe_server(cfg.port).await {
        eprintln!("{}", server_unreachable_message(cfg.port));
        return EXIT_SERVER_UNREACHABLE;
    }

    let mut session = ChatSession::new();

    if let Some(sys) = &cfg.system {
        session.push(ChatMessage {
            role: MessageRole::System,
            content: sys.clone(),
        });
    }

    for user_input in &cfg.messages {
        session.push(ChatMessage {
            role: MessageRole::User,
            content: user_input.clone(),
        });

        if cfg.json_mode {
            if let Err(e) = stream_chat_json(&mut session, &cfg.model, cfg.port).await {
                eprintln!("error: {:?}", e);
                return EXIT_ERROR;
            }
        } else {
            let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<ChatEvent>();
            stream_chat(&mut session, &cfg.model, cfg.port, tx).await;
            while let Ok(ev) = rx.try_recv() {
                match ev {
                    ChatEvent::Token(t) => print!("{}", t),
                    ChatEvent::Done => println!(),
                    ChatEvent::Error(e) => {
                        eprintln!("error: {}", e);
                        return EXIT_ERROR;
                    }
                }
            }
        }
    }

    persist_session_model(&cfg.model);
    EXIT_OK
}
