// @INFO: Live training TUI panel. Renders on top of the Python subprocess stdout stream.
// @EDITABLE: Tweak LOSS_HISTORY_LEN, LOG_HISTORY_LEN, POLL_MS, or color constants freely.
// @DANGER: `run_train_tui` takes ownership of the Child. The NamedTempFile must be passed in
//          and kept alive for the entire duration — dropping it deletes the script on disk.

use std::collections::VecDeque;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use ratatui::{
    layout::{Constraint, Layout},
    style::{Color, Modifier, Style},
    symbols,
    text::{Line, Span},
    widgets::{Block, Borders, Gauge, Paragraph, Sparkline},
    Frame,
};

use gwenland_core::train::config::TrainConfig;
use gwenland_core::storage::registry::{ModelEntry, ModelRegistry};

// ── constants ─────────────────────────────────────────────────────────────────

/// Gwen brand orange: #FF8C42
const ORANGE: Color = Color::Rgb(255, 140, 66);
const POLL_MS: u64 = 250;
const LOSS_HISTORY_LEN: usize = 100;
const LOG_HISTORY_LEN: usize = 200;

// ── events ────────────────────────────────────────────────────────────────────

/// A parsed event from the training process's stdout JSON stream.
///
/// # Why tokens_per_sec is Option here (not in ProgressEvent)
///
/// `ProgressEvent` always emits `tokens_per_sec` (it's a required struct field).
/// `TrainEvent` is the *parsed* side: older subprocess events or Python-path events
/// may not include the field, so `Option` keeps the parser backward-compatible.
/// The TUI simply omits the display when the field is absent.
#[derive(Debug)]
pub enum TrainEvent {
    Step { step: u32, epoch: f32, loss: f32, lr: f64, tokens_per_sec: Option<f32> },
    Interrupted { message: String },
    Done { output: String },
    Error { message: String },
}

fn parse_train_event(val: &serde_json::Value) -> Option<TrainEvent> {
    // Error frame
    if let Some(msg) = val.get("error") {
        let message = val
            .get("message")
            .and_then(|m| m.as_str())
            .unwrap_or_else(|| msg.as_str().unwrap_or("unknown error"))
            .to_string();
        return Some(TrainEvent::Error { message });
    }
    // Named event frames
    match val.get("event").and_then(|e| e.as_str()) {
        Some("done") => {
            let output = val
                .get("output")
                .and_then(|o| o.as_str())
                .unwrap_or("")
                .to_string();
            return Some(TrainEvent::Done { output });
        }
        Some("interrupted") => {
            let message = val
                .get("message")
                .and_then(|m| m.as_str())
                .unwrap_or("interrupted")
                .to_string();
            return Some(TrainEvent::Interrupted { message });
        }
        _ => {}
    }
    // Step log frame — keyed by presence of "step" field.
    // `tokens_per_sec` is optional: absent from Python-path events but present
    // in all native Candle events. Using `and_then` avoids a hard dependency on
    // the field existing.
    if let Some(step) = val.get("step").and_then(|s| s.as_u64()) {
        let epoch          = val.get("epoch").and_then(|e| e.as_f64()).unwrap_or(0.0) as f32;
        let loss           = val.get("loss").and_then(|l| l.as_f64()).unwrap_or(0.0) as f32;
        let lr             = val.get("lr").and_then(|l| l.as_f64()).unwrap_or(0.0);
        let tokens_per_sec = val.get("tokens_per_sec").and_then(|t| t.as_f64()).map(|t| t as f32);
        return Some(TrainEvent::Step { step: step as u32, epoch, loss, lr, tokens_per_sec });
    }
    None
}

// ── state ─────────────────────────────────────────────────────────────────────

pub struct TuiState {
    pub step: u32,
    pub total_steps: u32,
    pub epoch: f32,
    pub total_epochs: u32,
    pub last_loss: f32,
    pub prev_loss: f32,
    pub lr: f64,
    /// Most-recent `tokens_per_sec` from the step event stream.
    /// `None` until the first native Candle event arrives; stays `None` on the
    /// Python-path where `tokens_per_sec` is not emitted.
    pub tokens_per_sec: Option<f32>,
    pub loss_history: VecDeque<f32>,
    pub paused: bool,
    pub show_log: bool,
    pub show_status: bool,
    pub log_lines: VecDeque<String>,
    pub start_time: Instant,
    pub model_name: String,
    pub done: bool,
    pub error: Option<String>,
}

impl TuiState {
    pub fn new(config: &TrainConfig) -> Self {
        // total_steps = steps_per_epoch * epochs; steps_per_epoch unknown until first event,
        // so start with 0 and refine later.
        Self {
            step: 0,
            total_steps: 0,
            epoch: 0.0,
            total_epochs: config.epochs,
            last_loss: 0.0,
            prev_loss: 0.0,
            lr: config.learning_rate,
            tokens_per_sec: None,
            loss_history: VecDeque::with_capacity(LOSS_HISTORY_LEN),
            paused: false,
            show_log: false,
            show_status: false,
            log_lines: VecDeque::with_capacity(LOG_HISTORY_LEN),
            start_time: Instant::now(),
            model_name: config
                .name
                .clone()
                .unwrap_or_else(|| config.model.clone()),
            done: false,
            error: None,
        }
    }

    pub fn apply_event(&mut self, event: TrainEvent) {
        match event {
            TrainEvent::Step { step, epoch, loss, lr, tokens_per_sec } => {
                self.prev_loss = self.last_loss;
                self.step = step;
                self.epoch = epoch;
                self.last_loss = loss;
                self.lr = lr;
                // Only update when the field is present — keeps the last known
                // value visible if events temporarily stop including it.
                if tokens_per_sec.is_some() {
                    self.tokens_per_sec = tokens_per_sec;
                }

                if self.loss_history.len() == LOSS_HISTORY_LEN {
                    self.loss_history.pop_front();
                }
                self.loss_history.push_back(loss);

                let tps_suffix = match tokens_per_sec {
                    Some(t) => format!("  {:.0} tok/s", t),
                    None    => String::new(),
                };
                let log = format!(
                    "step {:>6}  epoch {:.2}  loss {:.4}  lr {:.2e}{}",
                    step, epoch, loss, lr, tps_suffix
                );
                self.push_log(log);

                // Estimate total_steps from the epoch progress once we have a non-zero epoch
                if self.total_steps == 0 && epoch > 0.0 && step > 0 {
                    // steps_per_epoch ≈ step / epoch (rough, gets refined each tick)
                    let spe = (step as f64 / epoch as f64).round() as u32;
                    self.total_steps = spe * self.total_epochs;
                }
            }
            TrainEvent::Done { output } => {
                self.push_log(format!("+ Training complete → {}", output));
                self.done = true;
            }
            TrainEvent::Interrupted { message } => {
                self.push_log(format!("! {}", message));
                self.done = true;
            }
            TrainEvent::Error { message } => {
                self.error = Some(message.clone());
                self.push_log(format!("✗ {}", message));
                self.done = true;
            }
        }
    }

    fn push_log(&mut self, line: String) {
        if self.log_lines.len() == LOG_HISTORY_LEN {
            self.log_lines.pop_front();
        }
        self.log_lines.push_back(line);
    }

    pub fn progress_pct(&self) -> u16 {
        if self.total_steps == 0 {
            return 0;
        }
        ((self.step as f64 / self.total_steps as f64) * 100.0).min(100.0) as u16
    }

    pub fn eta_string(&self) -> String {
        if self.step == 0 || self.total_steps == 0 {
            return "—".to_string();
        }
        let elapsed = self.start_time.elapsed().as_secs();
        let secs_per_step = elapsed as f64 / self.step as f64;
        let remaining_steps = self.total_steps.saturating_sub(self.step);
        let remaining_secs = (remaining_steps as f64 * secs_per_step) as u64;

        if remaining_secs < 60 {
            format!("{}s", remaining_secs)
        } else if remaining_secs < 3600 {
            format!("{}m", remaining_secs / 60)
        } else {
            format!("{}h {}m", remaining_secs / 3600, (remaining_secs % 3600) / 60)
        }
    }
}

// ── rendering ─────────────────────────────────────────────────────────────────

pub fn render_train_panel(frame: &mut Frame, state: &TuiState) {
    if state.show_log {
        render_log_view(frame, state);
        return;
    }

    let area = frame.area();

    let [header_area, metrics_area, chart_area, progress_area, keybinds_area] =
        Layout::vertical([
            Constraint::Length(1),
            Constraint::Length(3),
            Constraint::Fill(1),
            Constraint::Length(3),
            Constraint::Length(1),
        ])
        .areas(area);

    // ── title bar ─────────────────────────────────────────────────────────────
    let title = format!(" gwen train — {} ", state.model_name);
    frame.render_widget(
        Paragraph::new(title)
            .style(Style::new().fg(ORANGE).add_modifier(Modifier::BOLD)),
        header_area,
    );

    // ── metrics ───────────────────────────────────────────────────────────────
    let loss_color = if state.last_loss < state.prev_loss {
        Color::Green
    } else if state.last_loss > state.prev_loss && state.prev_loss > 0.0 {
        Color::Red
    } else {
        Color::Yellow
    };

    let trend = if state.last_loss < state.prev_loss {
        " ▼"
    } else if state.last_loss > state.prev_loss && state.prev_loss > 0.0 {
        " ▲"
    } else {
        ""
    };

    let metrics_lines = vec![
        Line::from(vec![
            Span::raw(format!(
                "  Epoch  {:>3} / {:<3}",
                state.epoch.floor() as u32,
                state.total_epochs
            )),
            Span::raw(format!(
                "    Step  {:>5} / {:<5}",
                state.step, state.total_steps
            )),
        ]),
        Line::from(vec![
            Span::styled(
                format!("  Loss   {:.4}{}", state.last_loss, trend),
                Style::new().fg(loss_color),
            ),
            Span::raw(format!("    LR    {:.2e}", state.lr)),
            // tokens_per_sec is omitted entirely when None — no placeholder or "N/A".
            // Why: the field is only present on the native Candle path; showing "N/A"
            // on the Python path would confuse users who don't control the event format.
            Span::raw(match state.tokens_per_sec {
                Some(t) => format!("    {:.0} tok/s", t),
                None    => String::new(),
            }),
        ]),
        // Error banner (replaces third metrics line when something goes wrong)
        if let Some(err) = &state.error {
            Line::from(Span::styled(
                format!("  ✗ {}", err),
                Style::new().fg(Color::Red).add_modifier(Modifier::BOLD),
            ))
        } else {
            Line::raw("")
        },
    ];

    frame.render_widget(
        Paragraph::new(metrics_lines)
            .block(Block::new().borders(Borders::NONE))
            .style(Style::new().fg(Color::Gray)),
        metrics_area,
    );

    // ── loss sparkline ────────────────────────────────────────────────────────
    let loss_data: Vec<u64> = state
        .loss_history
        .iter()
        .map(|&l| (l * 1000.0) as u64)
        .collect();

    let sparkline = Sparkline::default()
        .block(
            Block::bordered()
                .title(Span::styled(
                    " Loss (last 100 steps) ",
                    Style::new().fg(ORANGE).add_modifier(Modifier::BOLD),
                ))
                .border_style(Style::new().fg(Color::DarkGray)),
        )
        .data(&loss_data)
        .direction(ratatui::widgets::RenderDirection::LeftToRight)
        .style(Style::new().fg(ORANGE))
        .bar_set(symbols::bar::NINE_LEVELS);

    frame.render_widget(sparkline, chart_area);

    // ── gauge ─────────────────────────────────────────────────────────────────
    let pct = state.progress_pct();
    let gauge_label = if state.done {
        if state.error.is_some() {
            "  failed  ".to_string()
        } else {
            "  done  ".to_string()
        }
    } else {
        format!("  {}%   ETA {}  ", pct, state.eta_string())
    };

    let gauge_style = if state.error.is_some() {
        Style::new().fg(Color::Red).bg(Color::DarkGray)
    } else {
        Style::new().fg(ORANGE).bg(Color::DarkGray)
    };

    let gauge = Gauge::default()
        .block(Block::new().borders(Borders::NONE))
        .gauge_style(gauge_style)
        .percent(pct)
        .label(gauge_label)
        .use_unicode(true);

    frame.render_widget(gauge, progress_area);

    // ── keybind hints ─────────────────────────────────────────────────────────
    let paused_label = if state.paused { "[P] Resume" } else { "[P] Pause" };
    let log_label    = if state.show_log { "[L] Hide log" } else { "[L] Log" };
    let hints = format!(
        "  {}   [S] Status   [Q] Detach   {}",
        paused_label, log_label
    );
    frame.render_widget(
        Paragraph::new(hints).style(Style::new().fg(Color::DarkGray)),
        keybinds_area,
    );
}

/// Alternate full-screen log view shown when state.show_log is true.
fn render_log_view(frame: &mut Frame, state: &TuiState) {
    let area = frame.area();
    let [header_area, log_area, keybinds_area] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Fill(1),
        Constraint::Length(1),
    ])
    .areas(area);

    frame.render_widget(
        Paragraph::new(format!(" gwen train — {} — Log ", state.model_name))
            .style(Style::new().fg(ORANGE).add_modifier(Modifier::BOLD)),
        header_area,
    );

    // Show last N lines that fit the height
    let height = log_area.height as usize;
    let lines: Vec<Line> = state
        .log_lines
        .iter()
        .rev()
        .take(height)
        .rev()
        .map(|s| Line::raw(format!("  {}", s)))
        .collect();

    frame.render_widget(
        Paragraph::new(lines)
            .block(Block::bordered().border_style(Style::new().fg(Color::DarkGray))),
        log_area,
    );

    frame.render_widget(
        Paragraph::new("  [L] Back   [Q] Detach").style(Style::new().fg(Color::DarkGray)),
        keybinds_area,
    );
}

// ── pause/resume helpers ──────────────────────────────────────────────────────

#[cfg(unix)]
fn pause_child(pid: u32) {
    // SIGSTOP suspends the process; the process cannot catch or ignore it.
    unsafe { libc::kill(pid as libc::pid_t, libc::SIGSTOP); }
}

#[cfg(unix)]
fn resume_child(pid: u32) {
    unsafe { libc::kill(pid as libc::pid_t, libc::SIGCONT); }
}

// On Windows SIGSTOP/SIGCONT don't exist. NtSuspendProcess exists but requires
// unsafe FFI; for now we no-op and mark state as paused for display only.
// @TODO: implement DebugBreakProcess / NtSuspendProcess for Windows in a later cycle.
#[cfg(not(unix))]
fn pause_child(_pid: u32) {}
#[cfg(not(unix))]
fn resume_child(_pid: u32) {}

// ── event loop ────────────────────────────────────────────────────────────────

/// Produce a `(Receiver<TrainEvent>, Option<pid>)` pair from a subprocess.
///
/// Why a separate helper?
/// The legacy Python path calls this to get the event channel from the
/// subprocess's stdout. The native Candle path builds its own channel directly
/// from the training thread (see `run_native_path` in `commands/train.rs`).
/// Extracting the reader logic here keeps `run_train_tui`'s core loop free of
/// the subprocess-specific async/tokio machinery — the loop only cares about
/// events, not where they come from.
pub fn events_from_child(
    mut child: tokio::process::Child,
) -> Result<(mpsc::Receiver<TrainEvent>, Option<u32>)> {
    let child_pid = child.id();

    let stdout = child
        .stdout
        .take()
        .context("could not capture subprocess stdout for TUI")?;

    // Spin up a dedicated tokio runtime to drive the async stdout reader.
    // The TUI event loop is synchronous (crossterm::event::poll is blocking),
    // so we bridge via std::sync::mpsc.
    let (tx, rx) = mpsc::channel::<TrainEvent>();

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("failed to build tokio runtime for stdout reader")?;

    rt.spawn(async move {
        use tokio::io::{AsyncBufReadExt, BufReader as TokioBufReader};
        let mut lines = TokioBufReader::new(stdout).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(&line) {
                if let Some(event) = parse_train_event(&val) {
                    if tx.send(event).is_err() {
                        break; // TUI detached — stop reading
                    }
                }
            }
        }
    });

    // Leak the runtime: it drives the reader task for the TUI's lifetime.
    // The runtime shuts down when the process exits and the channel closes.
    std::mem::forget(rt);

    Ok((rx, child_pid))
}

/// Convert a `Receiver<String>` (raw JSON lines from the native training
/// thread) into a `Receiver<TrainEvent>` the TUI event loop can consume.
///
/// Why a converter thread instead of changing the native sender?
/// The native path emits raw JSON strings (the same bytes that go to stdout)
/// so that CI pipelines and log aggregators can consume them without changes.
/// Parsing into `TrainEvent` happens here, close to the TUI, keeping
/// `TrainingLoop` free of knowledge about the TUI's event type.
pub fn events_from_native_rx(
    string_rx: mpsc::Receiver<String>,
) -> mpsc::Receiver<TrainEvent> {
    let (tx, rx) = mpsc::channel::<TrainEvent>();
    std::thread::spawn(move || {
        while let Ok(line) = string_rx.recv() {
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(&line) {
                if let Some(event) = parse_train_event(&val) {
                    if tx.send(event).is_err() {
                        break; // TUI detached
                    }
                }
            }
        }
    });
    rx
}

/// Core TUI event loop.
///
/// Accepts a pre-built event channel so it is agnostic to the event source
/// (subprocess stdout or native training thread pipe). Both paths produce a
/// `Receiver<TrainEvent>` via `events_from_child` or `events_from_native_rx`.
///
/// `child_pid` is used only for pause/resume (SIGSTOP/SIGCONT on Unix).
/// Pass `None` on the native path — the training thread cannot be suspended
/// via signals without deadlocking the AdamW state.
///
/// `_script` keeps the `NamedTempFile` alive for the subprocess's lifetime.
/// On the native path pass any `NamedTempFile` — it is dropped at the end of
/// the TUI run and has no effect.
pub fn run_train_tui(
    rx:        mpsc::Receiver<TrainEvent>,
    child_pid: Option<u32>,
    config:    &TrainConfig,
    _script:   tempfile::NamedTempFile,
) -> Result<()> {
    let mut terminal = ratatui::init();
    let mut state = TuiState::new(config);

    loop {
        // Drain all queued training events non-blocking
        while let Ok(event) = rx.try_recv() {
            state.apply_event(event);
        }

        terminal.draw(|frame| render_train_panel(frame, &state))?;

        // Keyboard polling with 250ms timeout
        if crossterm::event::poll(Duration::from_millis(POLL_MS))? {
            match crossterm::event::read()? {
                crossterm::event::Event::Key(key)
                    if key.kind == crossterm::event::KeyEventKind::Press =>
                {
                    match key.code {
                        crossterm::event::KeyCode::Char('p')
                        | crossterm::event::KeyCode::Char('P') => {
                            state.paused = !state.paused;
                            if let Some(pid) = child_pid {
                                if state.paused {
                                    pause_child(pid);
                                } else {
                                    resume_child(pid);
                                }
                            }
                        }
                        crossterm::event::KeyCode::Char('s')
                        | crossterm::event::KeyCode::Char('S') => {
                            state.show_status = !state.show_status;
                        }
                        crossterm::event::KeyCode::Char('l')
                        | crossterm::event::KeyCode::Char('L') => {
                            state.show_log = !state.show_log;
                        }
                        crossterm::event::KeyCode::Char('q')
                        | crossterm::event::KeyCode::Char('Q') => {
                            // @INFO: Q detaches the TUI only — subprocess keeps running.
                            ratatui::restore();
                            println!("TUI detached. Training continues in background.");
                            return Ok(());
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }

        if state.done {
            break;
        }
    }

    ratatui::restore();

    // Update registry on clean completion
    if state.error.is_none() {
        if let Err(e) = register_trained_model_tui(config) {
            eprintln!("warning: training complete but registry update failed: {:?}", e);
        }
    }

    // Wait for subprocess to exit (it should already be done)
    // @INFO: we use a blocking wait via the std handle. The child was already spawned as
    //        a tokio::process::Child; we can only call wait() inside a tokio context.
    //        Since run_train_tui is called from an async context (via block_on in main),
    //        we just return and let the caller await the child if needed.
    // @TODO: expose child back to caller if explicit exit-code checking is needed.

    Ok(())
}

// ── registry ──────────────────────────────────────────────────────────────────

fn register_trained_model_tui(config: &TrainConfig) -> Result<()> {
    let mut registry = ModelRegistry::load()?;
    let id = config
        .name
        .clone()
        .unwrap_or_else(|| config.model.replace('/', "_"));
    registry.upsert(ModelEntry {
        id: id.clone(),
        source: config.model.clone(),
        format: "lora".into(),
        quant: if config.qlora { "qlora".into() } else { "full".into() },
        size_bytes: 0,
        downloaded_at: chrono::Utc::now().to_rfc3339(),
        sha256: String::new(),
        path: config.output.clone(),
    });
    registry.save()?;
    Ok(())
}
