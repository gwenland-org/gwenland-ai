use gwenland_core::{
    history::ConversationHistory,
    stream::{is_inference_server_down_error, stream_chat, StreamEvent},
    windowing::WindowConfig,
};
use ratatui::{
    layout::Rect,
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::Paragraph,
    Frame,
};
use serde_json::json;
use tokio::sync::mpsc;

// @EDITABLE — channel capacity; large enough for a burst of tokens without blocking
// the spawned stream task at 30–80 tokens/s
const STREAM_CHANNEL_CAPACITY: usize = 128;

// @EDITABLE — default mistral.rs base URL; expose via `gwen --configuration` later
const DEFAULT_BASE_URL: &str = "http://localhost:1234/v1";

// @EDITABLE — model identifier sent to mistral.rs; "default" uses whatever is loaded
const DEFAULT_MODEL: &str = "default";

// @EDITABLE — max messages loaded from history.jsonl into the TUI on startup;
// higher values show more context but slow down initial render for long histories
const HISTORY_LOAD_CAP: usize = 50;

// @EDITABLE — max messages sent to mistral.rs per request to stay within context window
// @INFO — JIN-164 will wire this to per-model token budgets; 20 is a safe default
const API_HISTORY_CAP: usize = 20;

// Brand colors — mirrored from ui.rs so ChatPane is self-contained
const ORANGE: Color = Color::Rgb(255, 140, 66);
const BG: Color = Color::Rgb(18, 14, 28);
const DIM: Color = Color::Rgb(80, 70, 100);
const WHITE: Color = Color::Rgb(220, 220, 220);

const STYLE_USER_LABEL: Style = Style::new().fg(ORANGE).add_modifier(Modifier::BOLD);
const STYLE_GWEN_LABEL: Style = Style::new().fg(DIM).add_modifier(Modifier::BOLD);
const STYLE_USER_TEXT: Style = Style::new().fg(WHITE);
const STYLE_GWEN_TEXT: Style = Style::new().fg(DIM);
const STYLE_SYS_ERROR_LABEL: Style = Style::new().fg(Color::Red).add_modifier(Modifier::BOLD);
const STYLE_SYS_ERROR_TEXT: Style = Style::new().fg(Color::Red);
const STYLE_CURSOR: Style = Style::new().fg(ORANGE).add_modifier(Modifier::BOLD);
const STYLE_BG: Style = Style::new().bg(BG);

/// Index into `messages` for a red system-style error bubble (not sent to the API).
const NO_SYSTEM_ERROR: usize = usize::MAX;

// Re-export so callers (e.g. session recovery) can use without knowing the module path.
pub use gwenland_core::stream::{ChatMessage, Role};

pub struct ChatPane {
    pub messages: Vec<ChatMessage>,
    /// Accumulates in-flight tokens during streaming; cleared on Done/Error.
    pub current_stream: String,
    pub is_streaming: bool,
    pub input: String,
    pub scroll_offset: u16,
    // @INFO — bounded channel prevents memory growth if TUI tick rate falls
    // behind the model's output rate
    rx: Option<mpsc::Receiver<StreamEvent>>,
    history: ConversationHistory,
    // @INFO — loaded from ~/.gwen/config.json; controls whether file content
    // is compressed before being sent to mistral.rs (JIN-164)
    pub window_config: WindowConfig,
    system_error_index: usize,
}

impl ChatPane {
    /// Create the pane, populate `messages` from history, and load windowing config.
    ///
    /// Pass a `WindowConfig` from `App` (which reads `~/.gwen/config.json`) so that
    /// CLI flag overrides (`--no-compression`, `--token-budget`) can flow down here.
    pub fn new(window_config: WindowConfig) -> Self {
        let history = ConversationHistory::new();

        // Load last N messages; gracefully fall back to empty on any I/O error.
        let mut all = history.load().unwrap_or_default();
        if all.len() > HISTORY_LOAD_CAP {
            // @INFO — drain the front so only the most recent messages are shown;
            // older messages still exist in history.jsonl and are not deleted
            all.drain(..all.len() - HISTORY_LOAD_CAP);
        }

        Self {
            messages: all,
            current_stream: String::new(),
            is_streaming: false,
            input: String::new(),
            scroll_offset: 0,
            rx: None,
            history,
            window_config,
            system_error_index: NO_SYSTEM_ERROR,
        }
    }

    pub fn push_char(&mut self, c: char) {
        if self.input.len() < 4096 {
            self.input.push(c);
        }
    }

    pub fn pop_char(&mut self) {
        self.input.pop();
    }

    /// Submit the current input: persist it, push bubbles, spawn stream task.
    /// No-op if input is empty or a stream is already in flight.
    pub fn submit_input(&mut self) {
        if self.input.trim().is_empty() || self.is_streaming {
            return;
        }

        let content = self.input.trim().to_string();
        self.input.clear();
        self.is_streaming = true;
        self.current_stream.clear();

        let user_msg = ChatMessage {
            role: Role::User,
            content: content.clone(),
        };

        // Persist BEFORE pushing to messages so a crash here doesn't show a ghost bubble
        if let Err(e) = self.history.append(&user_msg) {
            eprintln!("[history] append error (user): {e}");
        }

        self.messages.push(user_msg);

        // Placeholder assistant bubble — content fills as tokens arrive in tick()
        self.messages.push(ChatMessage {
            role: Role::Assistant,
            content: String::new(),
        });

        // Build history for the API: exclude empty placeholders, cap at last N messages.
        // @INFO — we send the last API_HISTORY_CAP turns so mistral.rs always has
        // full conversation context without exceeding the model's context window
        // @EDITABLE — raise API_HISTORY_CAP if users report context loss on long chats;
        // lower it if the model produces repetitive or confused responses
        let mut api_messages: Vec<serde_json::Value> = self
            .messages
            .iter()
            .filter_map(|m| match m.role {
                Role::User => Some(json!({"role": "user", "content": m.content})),
                Role::Assistant if !m.content.is_empty() => {
                    Some(json!({"role": "assistant", "content": m.content}))
                }
                _ => None, // skip the empty placeholder just pushed
            })
            .collect();

        if api_messages.len() > API_HISTORY_CAP {
            // @DANGER — drain from the FRONT, not the back; we want the MOST RECENT
            // messages, not the oldest ones
            api_messages.drain(..api_messages.len() - API_HISTORY_CAP);
        }

        let (tx, rx) = mpsc::channel::<StreamEvent>(STREAM_CHANNEL_CAPACITY);
        self.rx = Some(rx);

        let base_url = DEFAULT_BASE_URL.to_string();
        let model = DEFAULT_MODEL.to_string();
        // Clone config so it can move into the async task
        let window_config = WindowConfig {
            enabled:      self.window_config.enabled,
            token_budget: self.window_config.token_budget,
            window_size:  self.window_config.window_size,
            max_windows:  self.window_config.max_windows,
        };

        // @INFO — spawn stream task as a sibling tokio task so the TUI event loop
        // is never blocked; tokens arrive asynchronously and are drained in tick()
        // files: None — file injection is handled by the caller (e.g. FilePane) in
        // a future ticket; here we pass None for pure chat-only requests
        tokio::spawn(async move {
            let _ = stream_chat(&base_url, &model, api_messages, None, &window_config, tx).await;
        });
    }

    /// Drain the stream channel and apply events to pane state.
    /// Called once per TUI tick (~60 fps); must stay non-blocking (try_recv only).
    pub fn tick(&mut self) {
        let Some(ref mut rx) = self.rx else { return };

        loop {
            match rx.try_recv() {
                Ok(StreamEvent::Token(t)) => {
                    self.current_stream.push_str(&t);
                    // Keep the live assistant bubble in sync with the accumulator
                    if let Some(last) = self.messages.last_mut() {
                        last.content.clone_from(&self.current_stream);
                    }
                }
                Ok(StreamEvent::Done) => {
                    // Persist the complete assistant reply before clearing the accumulator
                    let assistant_msg = ChatMessage {
                        role: Role::Assistant,
                        content: self.current_stream.clone(),
                    };
                    if let Err(e) = self.history.append(&assistant_msg) {
                        eprintln!("[history] append error (assistant): {e}");
                    }

                    self.is_streaming = false;
                    self.current_stream.clear();
                    self.rx = None;
                    break;
                }
                Ok(StreamEvent::Error(e)) => {
                    // Drop the empty assistant placeholder before showing the error.
                    if self
                        .messages
                        .last()
                        .is_some_and(|m| matches!(m.role, Role::Assistant) && m.content.is_empty())
                    {
                        self.messages.pop();
                    }

                    if is_inference_server_down_error(&e) {
                        self.system_error_index = self.messages.len();
                        self.messages.push(ChatMessage {
                            role: Role::Assistant,
                            content: e,
                        });
                    } else if let Some(last) = self.messages.last_mut() {
                        last.content = format!("error: stream error — {}", e);
                        self.system_error_index = self.messages.len().saturating_sub(1);
                    } else {
                        self.system_error_index = self.messages.len();
                        self.messages.push(ChatMessage {
                            role: Role::Assistant,
                            content: format!("error: stream error — {}", e),
                        });
                    }

                    self.is_streaming = false;
                    self.current_stream.clear();
                    self.rx = None;
                    break;
                }
                // Channel empty this tick — come back next frame
                Err(mpsc::error::TryRecvError::Empty) => break,
                // Sender dropped without Done — treat as complete
                Err(mpsc::error::TryRecvError::Disconnected) => {
                    self.is_streaming = false;
                    self.rx = None;
                    break;
                }
            }
        }
    }

    /// Render the scrollable message history into `area`.
    ///
    /// Renders a blinking `▍` cursor on the last assistant bubble while streaming.
    pub fn render(&self, frame: &mut Frame, area: Rect) {
        let msg_count = self.messages.len();
        let mut lines: Vec<Line> = Vec::with_capacity(msg_count * 2);

        for (i, msg) in self.messages.iter().enumerate() {
            let is_last = i + 1 == msg_count;
            match msg.role {
                Role::User => {
                    lines.push(Line::from(vec![
                        Span::styled("  You  ", STYLE_USER_LABEL),
                        Span::styled(&*msg.content, STYLE_USER_TEXT),
                    ]));
                    lines.push(Line::raw(""));
                }
                Role::Assistant => {
                    if self.system_error_index == i {
                        lines.push(Line::from(vec![
                            Span::styled("  sys  ", STYLE_SYS_ERROR_LABEL),
                            Span::styled(&*msg.content, STYLE_SYS_ERROR_TEXT),
                        ]));
                        lines.push(Line::raw(""));
                        continue;
                    }

                    let mut spans = vec![
                        Span::styled("  Gwen ", STYLE_GWEN_LABEL),
                        Span::styled(&*msg.content, STYLE_GWEN_TEXT),
                    ];
                    // @KEEP — cursor must only appear on the last bubble while streaming;
                    // rendering it elsewhere would make completed responses look in-flight
                    if self.is_streaming && is_last {
                        spans.push(Span::styled("▍", STYLE_CURSOR));
                    }
                    lines.push(Line::from(spans));
                    lines.push(Line::raw(""));
                }
            }
        }

        let area_height = area.height as usize;
        let total = lines.len();
        // Auto-scroll: always pin to the bottom so new tokens are immediately visible
        let scroll: u16 = if total > area_height {
            (total - area_height) as u16
        } else {
            0
        };

        frame.render_widget(
            Paragraph::new(lines).scroll((scroll, 0)).style(STYLE_BG),
            area,
        );
    }
}
