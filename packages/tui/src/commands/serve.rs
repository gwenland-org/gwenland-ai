// commands/serve.rs — `gwen serve` CLI + TUI layer.
//
// Cycle 6: no longer spawns a mistralrs-server subprocess.
// Starts the in-process native inference proxy (platform::proxy) directly.
// All tokens are produced by candle-transformers; no external binary is needed.
//
// @EDITABLE: Add --lora, --quant, --threads flags here in future cycles.

use clap::Args;
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use gwenland_core::platform::serve::{
    dry_run_serve, find_model_in_cache, read_last_used_model, save_last_used_model,
    EXIT_CONNECTION_FAILED, EXIT_ERROR, EXIT_MODEL_NOT_FOUND, EXIT_OK, ServeStatus,
};
use gwenland_core::engine::inference::sampler::SamplerConfig;
use ratatui::{
    backend::CrosstermBackend,
    layout::{Alignment, Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
    Terminal,
};
use std::io;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::oneshot;

// ── args ──────────────────────────────────────────────────────────────────────

#[derive(Args, Debug)]
#[command(
    about = "Start local inference server",
    long_about = "Start the native inference proxy on port 1136.\n\
                  Exposes POST /gwenland/chat (SSE) backed by candle-transformers.\n\
                  No external process required — pure Rust.\n\n\
                  Examples:\n  \
                    gwen serve qwen3-8b-q4_0\n  \
                    gwen serve -m qwen3-8b-q4_0          (--model also works)\n  \
                    gwen serve qwen3-8b-q4_0 --port 8080\n  \
                    gwen serve qwen3-8b-q4_0 --dry-run"
)]
pub struct ServeArgs {
    /// Model name or path (e.g. qwen3-8b-q4_0, ./models/custom.gguf).
    /// Optional — if omitted, the last served model is reused.
    #[arg(value_name = "MODEL")]
    pub model: Option<String>,

    /// Model name or path passed as a flag. Mirrors `gwen fetch`/`gwen train`
    /// (which use `-m/--model`) so the form people expect also works on serve.
    /// Takes precedence over the positional MODEL.
    #[arg(short = 'm', long = "model", value_name = "MODEL", conflicts_with = "model")]
    pub model_flag: Option<String>,

    /// Port to bind the SSE proxy on (default: 1136)
    #[arg(long, short = 'p', default_value = "1136", value_name = "PORT")]
    pub port: u16,

    /// Context window length in tokens (default: 4096 — informational only)
    #[arg(long, default_value = "4096", value_name = "TOKENS")]
    pub ctx: u32,

    /// Pipe-friendly JSON output: print status JSON then wait (no TUI)
    #[arg(long)]
    pub json: bool,
}

// ── entry point ────────────────────────────────────────────────────────────────

pub async fn run_serve_cmd(args: ServeArgs, mode: gwenland_core::engine::GwenMode) {
    let code = match run_serve_inner(args, mode).await {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: {:?}", e);
            EXIT_ERROR
        }
    };
    std::process::exit(code);
}

async fn run_serve_inner(
    args: ServeArgs,
    mode: gwenland_core::engine::GwenMode,
) -> anyhow::Result<i32> {
    // Resolve the model: `-m/--model` flag > positional MODEL > last served model.
    // `serve` historically accepted only a positional MODEL, but its own hints
    // (and `gwen fetch`/`gwen train`) use `--model`, so accept both — and fall
    // back to the last served model when none is given (e.g. the GUI auto-start
    // spawns a bare `gwen serve`).
    let model = match args.model_flag.or(args.model) {
        Some(m) => m,
        None => match read_last_used_model() {
            Some(m) => {
                if !args.json {
                    eprintln!("No model given — reusing last served model: {m}");
                }
                m
            }
            None => {
                eprintln!("error: no model specified.");
                eprintln!("hint:  gwen serve <path/to/model.gguf>   (or: gwen serve -m <model>)");
                return Ok(EXIT_MODEL_NOT_FOUND);
            }
        },
    };

    // --dry-run: read-only pre-flight only.
    if mode.dry_run {
        let report = dry_run_serve(&model, args.port);
        if mode.json {
            gwenland_core::dry_run::print_json(&report);
        } else {
            gwenland_core::dry_run::print_report(&report);
        }
        return Ok(report.exit_code());
    }

    // Verify the model exists before starting the proxy.
    if find_model_in_cache(&model).is_none() {
        if args.json {
            println!(
                "{}",
                serde_json::to_string_pretty(&ServeStatus {
                    status: "model_not_found".into(),
                    model: model.clone(),
                    port: args.port,
                    pid: None,
                })
                .unwrap_or_default()
            );
        } else {
            eprintln!(
                "error: model '{}' not found. Run `gwen fetch {}`.",
                model, model
            );
        }
        return Ok(EXIT_MODEL_NOT_FOUND);
    }

    let _ = save_last_used_model(&model);

    // Create the shutdown channel — dropping the sender stops the proxy.
    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();

    if args.json {
        // JSON mode: print status and park until Ctrl+C.
        let status = ServeStatus {
            status: "running".into(),
            model: model.clone(),
            port: args.port,
            pid: Some(std::process::id()),
        };
        println!("{}", serde_json::to_string_pretty(&status).unwrap_or_default());

        // Start proxy in background; block until process receives a signal.
        tokio::spawn(async move {
            let sampler = SamplerConfig::default();
            let _ = gwenland_core::platform::proxy::start(sampler, shutdown_rx).await;
        });

        // Park the task; Ctrl+C will terminate the process.
        tokio::signal::ctrl_c().await?;

        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "status": "stopped",
                "model": model.clone(),
            }))
            .unwrap_or_default()
        );
        return Ok(EXIT_OK);
    }

    // TUI mode: launch proxy in background, show serve panel.
    let request_count = Arc::new(AtomicU64::new(0));

    tokio::spawn(async move {
        let sampler = SamplerConfig::default();
        if let Err(e) = gwenland_core::platform::proxy::start(sampler, shutdown_rx).await {
            eprintln!("proxy error: {}", e);
        }
    });

    let code = run_serve_tui(&model, args.port, request_count).await;
    // Drop shutdown_tx — this signals the proxy to stop gracefully.
    drop(shutdown_tx);

    Ok(code.unwrap_or(EXIT_OK))
}

// ── TUI event loop ────────────────────────────────────────────────────────────

async fn run_serve_tui(
    model_id: &str,
    port: u16,
    request_count: Arc<AtomicU64>,
) -> anyhow::Result<i32> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let result = serve_loop(&mut terminal, model_id, port, request_count).await;

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;

    result
}

async fn serve_loop(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    model_id: &str,
    port: u16,
    request_count: Arc<AtomicU64>,
) -> anyhow::Result<i32> {
    const POLL: std::time::Duration = std::time::Duration::from_millis(100);
    let pid = std::process::id();

    loop {
        let requests = request_count.load(Ordering::Relaxed);
        terminal.draw(|f| {
            draw_serve_panel(f, model_id, port, Some(pid), requests);
        })?;

        if event::poll(POLL)? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Char('q') | KeyCode::Char('Q') => {
                            return Ok(EXIT_OK);
                        }
                        KeyCode::Char('c')
                            if key
                                .modifiers
                                .contains(crossterm::event::KeyModifiers::CONTROL) =>
                        {
                            return Ok(EXIT_OK);
                        }
                        _ => {}
                    }
                }
            }
        }
    }
}

// ── TUI rendering ─────────────────────────────────────────────────────────────

fn draw_serve_panel(
    f: &mut ratatui::Frame,
    model_id: &str,
    port: u16,
    pid: Option<u32>,
    _requests: u64,
) {
    let area = f.area();

    let width = 54u16.min(area.width);
    let height = 9u16.min(area.height);
    let x = area.x + (area.width.saturating_sub(width)) / 2;
    let y = area.y + (area.height.saturating_sub(height)) / 2;
    let panel = ratatui::layout::Rect { x, y, width, height };

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan))
        .title(Span::styled(
            " gwen serve ",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ));
    f.render_widget(block, panel);

    let inner = ratatui::layout::Rect {
        x: panel.x + 1,
        y: panel.y + 1,
        width: panel.width.saturating_sub(2),
        height: panel.height.saturating_sub(2),
    };
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
        ])
        .split(inner);

    let green_dot = Span::styled("●", Style::default().fg(Color::Green));
    let status_line = Line::from(vec![Span::raw("status:   running "), green_dot]);
    f.render_widget(Paragraph::new(status_line), rows[0]);

    let model_short = if model_id.len() > 36 {
        let tail: String = model_id
            .chars()
            .rev()
            .take(35)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        format!("…{}", tail)
    } else {
        model_id.to_string()
    };
    f.render_widget(
        Paragraph::new(format!("model:    {}", model_short)),
        rows[1],
    );
    f.render_widget(
        Paragraph::new(format!("backend:  native (candle-transformers)")),
        rows[2],
    );
    f.render_widget(
        Paragraph::new(format!(
            "endpoint: http://localhost:{}/gwenland/chat",
            port
        )),
        rows[3],
    );
    let pid_str = pid.map(|p| p.to_string()).unwrap_or_else(|| "?".into());
    f.render_widget(Paragraph::new(format!("pid:      {}", pid_str)), rows[4]);
    f.render_widget(
        Paragraph::new(Line::from(vec![Span::styled(
            "[Q] stop server",
            Style::default().fg(Color::DarkGray),
        )]))
        .alignment(Alignment::Center),
        rows[5],
    );
}
