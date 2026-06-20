// @INFO: `gwen chat` — streaming chat TUI connected to the SSE proxy at
//        localhost:1136/gwenland/chat (spawned by `gwen serve`).
// @DANGER: Use List + ListState for history scroll. Never Paragraph with manual offset.
// @DANGER: SSE connection must be cleanly dropped on Ctrl+C / Q — no dangling tasks.

use clap::Args;
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use futures_util::StreamExt;
use ratatui::{
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, ListState, Paragraph},
    Frame,
};
use reqwest_eventsource::{Event as SseEvent, EventSource};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::mpsc;

// ── brand colour ──────────────────────────────────────────────────────────────
const ORANGE: Color = Color::Rgb(255, 140, 66);

// ── constants ─────────────────────────────────────────────────────────────────
const PROXY_URL: &str = "http://localhost:1136/gwenland/chat";
const MAX_HISTORY: usize = 500;
const POLL_MS: u64 = 16; // ~60 fps

const INFERENCE_SERVER_DOWN: &str = "inference server is not running";
const INFERENCE_SERVER_DOWN_HINT: &str =
    "start one with `gwen serve <path/to/model.gguf>`";

fn sse_error_is_connect(err: &reqwest_eventsource::Error) -> bool {
    matches!(
        err,
        reqwest_eventsource::Error::Transport(e) if e.is_connect()
    )
}

fn print_inference_server_down_and_exit() -> ! {
    eprintln!("error: {}", INFERENCE_SERVER_DOWN);
    eprintln!("hint:  {}", INFERENCE_SERVER_DOWN_HINT);
    std::process::exit(1);
}

fn inference_server_down_history_message() -> String {
    format!("error: {}\nhint:  {}", INFERENCE_SERVER_DOWN, INFERENCE_SERVER_DOWN_HINT)
}

// ── args ──────────────────────────────────────────────────────────────────────

#[derive(Args, Debug)]
#[command(
    about = "Chat with local model (TUI)",
    long_about = "Open an interactive streaming chat TUI connected to the GwenLand SSE proxy.\n\
                  `gwen serve` must be running first.\n\n\
                  In --non-interactive mode, reads one prompt from stdin and streams the\n\
                  response to stdout (pipe-friendly).\n\n\
                  Keyboard shortcuts (TUI mode):\n  \
                    Enter      Send message\n  \
                    Ctrl+C     Exit\n  \
                    Ctrl+L     Clear history\n  \
                    ↑ / ↓      Scroll history\n\n\
                  Examples:\n  \
                    gwen chat\n  \
                    gwen chat --gui\n  \
                    gwen chat -m mistralai/Mistral-7B-v0.1\n  \
                    gwen chat --system \"You are a helpful coding assistant.\"\n  \
                    echo \"What is Rust?\" | gwen chat --non-interactive\n  \
                    echo \"Explain LoRA\" | gwen chat --non-interactive --json"
)]
pub struct ChatArgs {
    /// Launch GUI window instead of TUI
    #[arg(long, help = "Launch GUI window instead of TUI")]
    pub gui: bool,

    /// Model ID to pass in the request body (e.g. mistralai/Mistral-7B-v0.1). Defaults to "gwen".
    #[arg(long, short = 'm', value_name = "MODEL_ID")]
    pub model: Option<String>,

    /// Override the proxy endpoint URL (default: http://localhost:1136/gwenland/chat)
    #[arg(long, default_value = PROXY_URL, value_name = "URL")]
    pub proxy: String,

    /// System prompt to prepend to the conversation (sent once at session start)
    #[arg(long, short = 's', value_name = "PROMPT")]
    pub system: Option<String>,
}

// ── message types ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
enum Role {
    User,
    Assistant,
    #[allow(dead_code)]
    System,
}

#[derive(Debug, Clone)]
struct Message {
    role: Role,
    content: String,
    /// System messages shown in red (e.g. inference server unreachable).
    is_error: bool,
}

// ── SSE payload ───────────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize, Clone)]
struct WireMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct WireStreamChunk {
    #[serde(default)]
    message: Option<WireMessage>,
    #[serde(default)]
    done: bool,
}

// ── internal events (SSE reader → TUI loop) ───────────────────────────────────

enum ChatEvent {
    Token(String),
    Done,
    InferenceServerDown,
    StreamError(String),
}

// ── TUI state ─────────────────────────────────────────────────────────────────

struct ChatState {
    /// Fully completed messages shown in the history list.
    history: VecDeque<Message>,
    /// Current assistant response being built token-by-token.
    streaming_buf: String,
    /// Whether we are waiting for the server to respond.
    is_streaming: bool,
    /// User's current input line.
    input: String,
    /// ListState for auto-scroll.
    list_state: ListState,
    /// Model to use in the request body.
    model: String,
    /// System prompt (prepended once per session).
    system: Option<String>,
    /// Whether the system prompt has been added.
    system_sent: bool,
    /// Error message to display in status bar.
    error: Option<String>,
}

impl ChatState {
    fn new(model: String, system: Option<String>) -> Self {
        Self {
            history: VecDeque::with_capacity(MAX_HISTORY),
            streaming_buf: String::new(),
            is_streaming: false,
            input: String::new(),
            list_state: ListState::default(),
            model,
            system,
            system_sent: false,
            error: None,
        }
    }

    fn push_message(&mut self, msg: Message) {
        if self.history.len() == MAX_HISTORY {
            self.history.pop_front();
        }
        self.history.push_back(msg);
        self.scroll_to_bottom();
    }

    fn scroll_to_bottom(&mut self) {
        if !self.history.is_empty() {
            self.list_state.select(Some(self.history.len() - 1));
        }
    }

    fn scroll_up(&mut self) {
        let sel = self.list_state.selected().unwrap_or(0);
        if sel > 0 {
            self.list_state.select(Some(sel - 1));
        }
    }

    fn scroll_down(&mut self) {
        let max = self.history.len().saturating_sub(1);
        let sel = self.list_state.selected().unwrap_or(0);
        if sel < max {
            self.list_state.select(Some(sel + 1));
        }
    }

    fn clear_history(&mut self) {
        self.history.clear();
        self.streaming_buf.clear();
        self.is_streaming = false;
        self.list_state = ListState::default();
        self.error = None;
    }

    /// Build the messages array from history + current streaming buf.
    fn build_messages(&self) -> Vec<WireMessage> {
        let mut out = Vec::new();
        if !self.system_sent {
            if let Some(sys) = &self.system {
                out.push(WireMessage {
                    role: "system".into(),
                    content: sys.clone(),
                });
            }
        }
        for msg in &self.history {
            if msg.is_error {
                continue;
            }
            out.push(WireMessage {
                role: match msg.role {
                    Role::User => "user".into(),
                    Role::Assistant => "assistant".into(),
                    Role::System => "system".into(),
                },
                content: msg.content.clone(),
            });
        }
        out
    }
}

// ── entry point ───────────────────────────────────────────────────────────────

pub async fn run_chat_cmd(args: ChatArgs, mode: gwenland_core::engine::GwenMode) {
    if args.gui {
        #[cfg(feature = "gui")]
        {
            gwen_gui_lib::run();
            return;
        }
        #[cfg(not(feature = "gui"))]
        {
            eprintln!("error: GUI support not compiled in.");
            eprintln!("hint:  rebuild with `cargo build --release --features gui`");
            std::process::exit(1);
        }
    }

    let model = args.model.clone().unwrap_or_else(|| "gwen".to_string());
    let proxy = args.proxy.clone();
    let system = args.system.clone();

    if mode.non_interactive {
        if let Err(e) = run_chat_pipe(model, proxy, system, mode).await {
            eprintln!("error: {}", e);
            std::process::exit(1);
        }
    } else if let Err(e) = run_chat_tui(model, proxy, system).await {
        eprintln!("error: {}", e);
        std::process::exit(1);
    }
}

// ── non-interactive (pipe / NDJSON) path ──────────────────────────────────────

async fn run_chat_pipe(
    model: String,
    proxy: String,
    system: Option<String>,
    mode: gwenland_core::engine::GwenMode,
) -> anyhow::Result<()> {
    use std::io::{self, BufRead};

    let stdin = io::stdin();
    let mut messages: Vec<WireMessage> = Vec::new();

    if let Some(sys) = &system {
        messages.push(WireMessage { role: "system".into(), content: sys.clone() });
    }

    // In pipe mode read a single line from stdin as the prompt.
    let line = stdin.lock().lines().next()
        .ok_or_else(|| anyhow::anyhow!("no input on stdin"))??;
    let prompt = line.trim().to_string();
    if prompt.is_empty() {
        return Ok(());
    }
    messages.push(WireMessage { role: "user".into(), content: prompt });

    let client = reqwest::Client::new();
    let body = serde_json::json!({
        "model": model,
        "messages": messages,
        "stream": true,
    });

    let request = client
        .post(&proxy)
        .header("Content-Type", "application/json")
        .json(&body);

    let mut es = reqwest_eventsource::EventSource::new(request)
        .map_err(|e| anyhow::anyhow!("stream error — {}", e))?;

    let mut total_tokens: u64 = 0;
    use futures_util::StreamExt as _;

    while let Some(event) = es.next().await {
        match event {
            Ok(reqwest_eventsource::Event::Open) => {}
            Ok(reqwest_eventsource::Event::Message(msg)) => {
                let data = msg.data;
                if data == "[DONE]" {
                    break;
                }
                if let Ok(chunk) = serde_json::from_str::<WireStreamChunk>(&data) {
                    if let Some(m) = &chunk.message {
                        if !m.content.is_empty() {
                            total_tokens += (m.content.len() as u64) / 4 + 1;
                            if mode.json {
                                println!("{}", serde_json::json!({"type":"token","content":m.content}));
                            } else {
                                print!("{}", m.content);
                            }
                        }
                    }
                    if chunk.done {
                        break;
                    }
                }
            }
            Err(reqwest_eventsource::Error::StreamEnded) => break,
            Err(e) if sse_error_is_connect(&e) => print_inference_server_down_and_exit(),
            Err(e) => {
                eprintln!("error: stream error — {}", e);
                std::process::exit(1);
            }
        }
    }

    if mode.json {
        println!("{}", serde_json::json!({"type":"done","total_tokens":total_tokens}));
    } else {
        println!();
    }

    Ok(())
}

// ── TUI event loop ────────────────────────────────────────────────────────────

async fn run_chat_tui(
    model: String,
    proxy: String,
    system: Option<String>,
) -> anyhow::Result<()> {
    let mut terminal = ratatui::init();
    let result = chat_loop(&mut terminal, model, proxy, system).await;
    ratatui::restore();
    result
}

async fn chat_loop(
    terminal: &mut ratatui::Terminal<ratatui::backend::CrosstermBackend<std::io::Stdout>>,
    model: String,
    proxy: String,
    system: Option<String>,
) -> anyhow::Result<()> {
    let mut state = ChatState::new(model, system);

    // Channel from SSE reader task → TUI loop
    let (sse_tx, mut sse_rx) = mpsc::unbounded_channel::<ChatEvent>();

    // Cancellation flag: set to true to tell the SSE task to stop
    let cancel = Arc::new(AtomicBool::new(false));

    loop {
        // ── drain SSE events (non-blocking) ───────────────────────────────────
        while let Ok(ev) = sse_rx.try_recv() {
            match ev {
                ChatEvent::Token(tok) => {
                    state.streaming_buf.push_str(&tok);
                }
                ChatEvent::Done => {
                    let content = std::mem::take(&mut state.streaming_buf);
                    if !content.is_empty() {
                        state.push_message(Message {
                            role: Role::Assistant,
                            content,
                            is_error: false,
                        });
                    }
                    state.is_streaming = false;
                    state.system_sent = true;
                }
                ChatEvent::InferenceServerDown => {
                    state.is_streaming = false;
                    state.streaming_buf.clear();
                    state.error = Some(INFERENCE_SERVER_DOWN.to_string());
                    state.push_message(Message {
                        role: Role::System,
                        content: inference_server_down_history_message(),
                        is_error: true,
                    });
                }
                ChatEvent::StreamError(msg) => {
                    state.is_streaming = false;
                    state.streaming_buf.clear();
                    state.error = Some(msg.clone());
                    state.push_message(Message {
                        role: Role::System,
                        content: format!("error: stream error — {}", msg),
                        is_error: true,
                    });
                }
            }
        }

        // ── render ────────────────────────────────────────────────────────────
        terminal.draw(|f| render_chat(f, &mut state))?;

        // ── keyboard (non-blocking poll) ──────────────────────────────────────
        if event::poll(std::time::Duration::from_millis(POLL_MS))? {
            if let Event::Key(key) = event::read()? {
                if key.kind != KeyEventKind::Press {
                    continue;
                }

                match key.code {
                    // Exit
                    KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                        cancel.store(true, Ordering::Relaxed);
                        return Ok(());
                    }
                    // Clear history
                    KeyCode::Char('l') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                        state.clear_history();
                    }
                    // Send message
                    KeyCode::Enter => {
                        if state.is_streaming {
                            continue; // drop input while streaming
                        }
                        let input = state.input.trim().to_string();
                        if input.is_empty() {
                            continue;
                        }
                        state.input.clear();
                        state.error = None;

                        state.push_message(Message {
                            role: Role::User,
                            content: input.clone(),
                            is_error: false,
                        });

                        // Build request
                        let messages = state.build_messages();
                        // The user message is already in history, so messages already includes it.
                        state.is_streaming = true;
                        state.system_sent = true;

                        // Spawn SSE reader task
                        let tx = sse_tx.clone();
                        let proxy_url = proxy.clone();
                        let model_id = state.model.clone();
                        let cancel_flag = Arc::clone(&cancel);
                        tokio::spawn(async move {
                            stream_response(proxy_url, model_id, messages, tx, cancel_flag).await;
                        });
                    }
                    // Backspace
                    KeyCode::Backspace => {
                        state.input.pop();
                    }
                    // Scroll up
                    KeyCode::Up => {
                        state.scroll_up();
                    }
                    // Scroll down
                    KeyCode::Down => {
                        state.scroll_down();
                    }
                    // PageUp
                    KeyCode::PageUp => {
                        for _ in 0..5 {
                            state.scroll_up();
                        }
                    }
                    // PageDown
                    KeyCode::PageDown => {
                        for _ in 0..5 {
                            state.scroll_down();
                        }
                    }
                    // Text input
                    KeyCode::Char(c) => {
                        state.input.push(c);
                    }
                    _ => {}
                }
            }
        }
    }
}

// ── SSE streaming task ────────────────────────────────────────────────────────

async fn stream_response(
    proxy_url: String,
    model: String,
    messages: Vec<WireMessage>,
    tx: mpsc::UnboundedSender<ChatEvent>,
    cancel: Arc<AtomicBool>,
) {
    let client = reqwest::Client::new();

    let body = serde_json::json!({
        "model": model,
        "messages": messages,
        "stream": true,
    });

    let request = client
        .post(&proxy_url)
        .header("Content-Type", "application/json")
        .json(&body);

    let mut es = match EventSource::new(request) {
        Ok(es) => es,
        Err(e) => {
            let _ = tx.send(ChatEvent::StreamError(e.to_string()));
            return;
        }
    };

    while let Some(event) = es.next().await {
        if cancel.load(Ordering::Relaxed) {
            es.close();
            return;
        }

        match event {
            Ok(SseEvent::Open) => {}
            Ok(SseEvent::Message(msg)) => {
                // Ollama stream format: each data line is a JSON chunk
                let data = msg.data;
                if data == "[DONE]" {
                    let _ = tx.send(ChatEvent::Done);
                    return;
                }
                match serde_json::from_str::<WireStreamChunk>(&data) {
                    Ok(chunk) => {
                        if let Some(m) = &chunk.message {
                            if !m.content.is_empty() {
                                let _ = tx.send(ChatEvent::Token(m.content.clone()));
                            }
                        }
                        if chunk.done {
                            let _ = tx.send(ChatEvent::Done);
                            return;
                        }
                    }
                    Err(_) => {
                        // Non-JSON data line — skip silently (keep-alive, blank lines)
                    }
                }
            }
            Err(reqwest_eventsource::Error::StreamEnded) => {
                let _ = tx.send(ChatEvent::Done);
                return;
            }
            Err(e) if sse_error_is_connect(&e) => {
                let _ = tx.send(ChatEvent::InferenceServerDown);
                return;
            }
            Err(e) => {
                let _ = tx.send(ChatEvent::StreamError(e.to_string()));
                return;
            }
        }
    }

    let _ = tx.send(ChatEvent::Done);
}

// ── rendering ─────────────────────────────────────────────────────────────────

fn render_chat(f: &mut Frame, state: &mut ChatState) {
    let area = f.area();

    // ── outer layout: history | streaming | divider | input | hints ───────────
    let has_streaming = state.is_streaming || !state.streaming_buf.is_empty();
    let streaming_height: u16 = if has_streaming {
        // wrap streaming buf to width and count lines (min 1, max 6)
        let w = area.width.saturating_sub(4) as usize;
        let lines = wrap_line(&state.streaming_buf, w).len() as u16;
        lines.max(1).min(6)
    } else {
        0
    };

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(3),                          // history list
            Constraint::Length(streaming_height),        // live streaming row
            Constraint::Length(3),                       // input box
            Constraint::Length(1),                       // status / hints
        ])
        .split(area);

    // ── history list ──────────────────────────────────────────────────────────
    let items: Vec<ListItem> = state
        .history
        .iter()
        .map(|msg| build_list_item(msg, chunks[0].width as usize))
        .collect();

    let list = List::new(items)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::DarkGray))
                .title(Span::styled(
                    " gwen chat ",
                    Style::default().fg(ORANGE).add_modifier(Modifier::BOLD),
                )),
        )
        .highlight_style(Style::default()); // no highlight colour — selection is just scroll pos

    f.render_stateful_widget(list, chunks[0], &mut state.list_state);

    // ── streaming row ─────────────────────────────────────────────────────────
    if has_streaming {
        let w = chunks[1].width.saturating_sub(4) as usize;
        let wrapped = wrap_line(&state.streaming_buf, w);
        let mut lines: Vec<Line> = Vec::new();
        for (i, seg) in wrapped.iter().enumerate() {
            if i == 0 {
                lines.push(Line::from(vec![
                    Span::styled("Gwen  ", Style::default().fg(ORANGE).add_modifier(Modifier::BOLD)),
                    Span::raw(seg.clone()),
                ]));
            } else {
                lines.push(Line::from(vec![
                    Span::raw("      "), // indent to align with name
                    Span::raw(seg.clone()),
                ]));
            }
        }
        // Blinking cursor hint
        if let Some(last) = lines.last_mut() {
            last.spans.push(Span::styled("▌", Style::default().fg(ORANGE)));
        }
        f.render_widget(
            Paragraph::new(lines)
                .block(Block::default().borders(Borders::NONE))
                .style(Style::default()),
            chunks[1],
        );
    }

    // ── input box ─────────────────────────────────────────────────────────────
    let input_style = if state.is_streaming {
        Style::default().fg(Color::DarkGray)
    } else {
        Style::default().fg(Color::White)
    };

    let input_display = if state.is_streaming {
        "  (waiting for response…)".to_string()
    } else {
        format!(" > {}_", state.input)
    };

    let border_color = if state.error.is_some() {
        Color::Red
    } else if state.is_streaming {
        Color::DarkGray
    } else {
        Color::Cyan
    };

    f.render_widget(
        Paragraph::new(input_display)
            .style(input_style)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(border_color)),
            ),
        chunks[2],
    );

    // ── status / hints bar ────────────────────────────────────────────────────
    let status = if let Some(err) = &state.error {
        Line::from(vec![Span::styled(
            format!(" error: {}", truncate_str(err, area.width as usize - 4)),
            Style::default().fg(Color::Red),
        )])
    } else {
        Line::from(vec![
            Span::styled("  Ctrl+C", Style::default().fg(Color::DarkGray)),
            Span::raw(" exit   "),
            Span::styled("Ctrl+L", Style::default().fg(Color::DarkGray)),
            Span::raw(" clear   "),
            Span::styled("↑↓", Style::default().fg(Color::DarkGray)),
            Span::raw(" scroll"),
        ])
    };
    f.render_widget(Paragraph::new(status), chunks[3]);
}

// ── rendering helpers ─────────────────────────────────────────────────────────

fn build_list_item(msg: &Message, width: usize) -> ListItem<'static> {
    let label_width = 6usize; // "You   " / "Gwen  " / "sys   "
    let text_width = width.saturating_sub(label_width + 2);

    let (label, label_style) = match msg.role {
        Role::User => (
            "You   ",
            Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
        ),
        Role::Assistant => (
            "Gwen  ",
            Style::default().fg(ORANGE).add_modifier(Modifier::BOLD),
        ),
        Role::System => {
            if msg.is_error {
                (
                    "sys   ",
                    Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
                )
            } else {
                (
                    "sys   ",
                    Style::default().fg(Color::DarkGray),
                )
            }
        }
    };

    let wrapped = wrap_line(&msg.content, text_width.max(20));
    let mut lines: Vec<Line<'static>> = Vec::new();

    for (i, seg) in wrapped.iter().enumerate() {
        if i == 0 {
            lines.push(Line::from(vec![
                Span::styled(label.to_string(), label_style),
                Span::raw(seg.clone()),
            ]));
        } else {
            lines.push(Line::from(vec![
                Span::raw(" ".repeat(label_width)),
                Span::raw(seg.clone()),
            ]));
        }
    }

    if lines.is_empty() {
        lines.push(Line::from(Span::styled(label.to_string(), label_style)));
    }

    ListItem::new(lines)
}

/// Simple word-wrap: splits `text` into lines of at most `width` chars.
fn wrap_line(text: &str, width: usize) -> Vec<String> {
    if width == 0 {
        return vec![text.to_string()];
    }

    let mut out = Vec::new();
    for paragraph in text.split('\n') {
        if paragraph.is_empty() {
            out.push(String::new());
            continue;
        }
        let mut current = String::new();
        for word in paragraph.split_whitespace() {
            if current.is_empty() {
                current.push_str(word);
            } else if current.len() + 1 + word.len() <= width {
                current.push(' ');
                current.push_str(word);
            } else {
                out.push(current.clone());
                current = word.to_string();
                // Hard-break words longer than width (char-safe)
                while current.chars().count() > width {
                    let split: String = current.chars().take(width).collect();
                    let rest: String = current.chars().skip(width).collect();
                    out.push(split);
                    current = rest;
                }
            }
        }
        if !current.is_empty() {
            out.push(current);
        }
    }
    if out.is_empty() {
        out.push(String::new());
    }
    out
}

fn truncate_str(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let truncated: String = s.chars().take(max.saturating_sub(1)).collect();
        format!("{}…", truncated)
    }
}