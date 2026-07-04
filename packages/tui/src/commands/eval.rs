// eval.rs — `gwen eval` command: CLI args, live TUI panel, and dispatch.
//
// Architecture:
//   Phase 1 (metrics)  runs first on a background std::thread.
//   Phase 2 (output)   runs second on the same background thread (after P1 finishes),
//                      using a nested tokio runtime for the native inference calls.
//   The main thread drives the Ratatui event loop throughout both phases.
//   A shared EvalPanelState (behind Arc<Mutex>) is the single source of truth
//   for both the background worker and the TUI renderer.
//
// Why std::thread + Mutex for the TUI loop (not tokio tasks)?
// Ratatui's crossterm::event::poll is a blocking call. Running it inside a
// tokio async context would starve the runtime. The same pattern is used in
// tui/train_panel.rs for the same reason.
//
// Why a single background thread that runs both phases sequentially?
// The phases are ordered: Phase 2 runs native inference which needs Phase 1
// to be complete so the TUI shows a coherent picture. A single thread avoids
// coordinating two threads sharing the p1 result.

use clap::Args;
use ratatui::{
    layout::{Constraint, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Gauge, Paragraph},
    Frame,
};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use gwenland_core::eval::metrics::{self, MetricsResult};
use gwenland_core::eval::output_eval::{self, SampleResult};
use gwenland_core::eval::report::{EvalReport, ReportMetrics};

// ── brand constant ─────────────────────────────────────────────────────────────

/// Gwen Orange #FF8C42 — used consistently across all GwenLand TUI panels.
const ORANGE: Color = Color::Rgb(255, 140, 66);
const POLL_MS: u64 = 100;

// ── CLI args ───────────────────────────────────────────────────────────────────

#[derive(Args, Debug)]
#[command(
    about = "Evaluate model performance on a validation dataset",
    long_about = "Run two-phase model evaluation:\n\
                  \n\
                  Phase 1 — Loss-based metrics (avg_loss, perplexity, tokens/sec,\n\
                  memory/token) computed over the full dataset.\n\
                  \n\
                  Phase 2 — Output-based evaluation: runs native inference on\n\
                  up to --max-samples samples and scores generated vs expected\n\
                  output using case-insensitive substring match.\n\
                  \n\
                  A live Ratatui panel shows progress during both phases.\n\
                  The final report is printed to the terminal after the TUI exits.\n\
                  \n\
                  Examples:\n  \
                    gwen eval --model llama3.2 --dataset ./data/val.jsonl\n  \
                    gwen eval --model llama3.2 --dataset val.jsonl --output report.json\n  \
                    gwen eval --model llama3.2 --dataset val.jsonl --max-samples 100"
)]
pub struct EvalArgs {
    /// Model ID or path (e.g. llama3.2, mistral, or a local HuggingFace path)
    #[arg(long, short = 'm', value_name = "MODEL_ID")]
    pub model: String,

    /// Path to validation JSONL file (format: {"input":"...","output":"..."})
    #[arg(long, short = 'd', value_name = "PATH")]
    pub dataset: String,

    /// Write full results as JSON to this path (optional)
    #[arg(long, short = 'o', value_name = "PATH")]
    pub output: Option<String>,

    /// Maximum number of samples for Phase 2 output-based eval (default: 50)
    #[arg(long, default_value = "50", value_name = "N")]
    pub max_samples: usize,
}

// ── live TUI state ─────────────────────────────────────────────────────────────

/// All mutable display state shared between the background worker and the TUI
/// render loop. Guarded by Mutex so the worker can push updates without
/// coordinating with the 100 ms render tick.
struct EvalPanelState {
    model: String,
    dataset_path: String,
    total_samples: usize,

    // Phase 1 live counters
    phase1_done: bool,
    phase1_current: usize,
    running_loss: f64,
    running_ppl: f64,
    tokens_per_sec: f64,

    // Phase 2 live counters
    phase2_done: bool,
    phase2_current: usize,
    /// Cap for Phase 2 (= total_samples.min(--max-samples)).
    phase2_total: usize,
    matched: usize,

    start_time: Instant,

    /// True once both phases complete successfully.
    all_done: bool,
    error: Option<String>,
}

impl EvalPanelState {
    fn new(model: &str, dataset_path: &str, total_samples: usize, phase2_total: usize) -> Self {
        Self {
            model: model.to_string(),
            dataset_path: dataset_path.to_string(),
            total_samples,
            phase1_done: false,
            phase1_current: 0,
            running_loss: 0.0,
            running_ppl: 1.0,
            tokens_per_sec: 0.0,
            phase2_done: false,
            phase2_current: 0,
            phase2_total,
            matched: 0,
            start_time: Instant::now(),
            all_done: false,
            error: None,
        }
    }

    /// Overall progress percentage combining both phases.
    /// Phase 1 contributes 40 %, Phase 2 the remaining 60 %, reflecting the
    /// typical wall-clock split: Phase 1 is CPU-only, Phase 2 is inference-bound.
    fn progress_pct(&self) -> u16 {
        let p1_pct = if self.total_samples > 0 {
            (self.phase1_current as f64 / self.total_samples as f64).min(1.0) * 40.0
        } else {
            40.0
        };
        let p2_pct = if self.phase2_total > 0 {
            (self.phase2_current as f64 / self.phase2_total as f64).min(1.0) * 60.0
        } else {
            0.0
        };
        (p1_pct + p2_pct).min(100.0) as u16
    }

    fn eta_string(&self) -> String {
        let pct = self.progress_pct() as f64;
        if pct < 1.0 {
            return "—".to_string();
        }
        let elapsed = self.start_time.elapsed().as_secs_f64();
        let total_est = elapsed / (pct / 100.0);
        let remaining = (total_est - elapsed).max(0.0) as u64;
        if remaining < 60 {
            format!("{}s", remaining)
        } else if remaining < 3600 {
            format!("{}m", remaining / 60)
        } else {
            format!("{}h {}m", remaining / 3600, (remaining % 3600) / 60)
        }
    }
}

// ── TUI renderer ───────────────────────────────────────────────────────────────

/// Render the eval live panel.
///
/// Layout:
///   title bar     1 line
///   Phase 1 info  3 lines
///   Phase 2 info  2 lines
///   progress bar  3 lines
///   keybind hints 1 line
fn render_eval_panel(frame: &mut Frame, state: &EvalPanelState) {
    let area = frame.area();

    let [title_area, p1_area, p2_area, gauge_area, hints_area] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Length(3),
        Constraint::Length(2),
        Constraint::Length(3),
        Constraint::Length(1),
    ])
    .areas(area);

    // ── title ──────────────────────────────────────────────────────────────────
    // Include the dataset path in the title bar so the user knows which file
    // is being evaluated when multiple evals run in parallel terminals.
    frame.render_widget(
        Paragraph::new(format!(
            " gwen eval — {}  [{}] ",
            state.model, state.dataset_path
        ))
        .style(Style::new().fg(ORANGE).add_modifier(Modifier::BOLD)),
        title_area,
    );

    // ── phase 1 metrics ────────────────────────────────────────────────────────
    let p1_status = if state.phase1_done { "✓ done" } else { "running…" };
    let p1_lines = vec![
        Line::from(vec![
            Span::styled("Phase 1", Style::new().fg(ORANGE)),
            Span::raw(format!(" (loss-based)  {}", p1_status)),
        ]),
        Line::from(vec![
            Span::styled("  Avg Loss   ", Style::new().fg(ORANGE)),
            Span::raw(format!("{:.4}", state.running_loss)),
            Span::styled("    Perplexity  ", Style::new().fg(ORANGE)),
            Span::raw(format!("{:.4}", state.running_ppl)),
        ]),
        Line::from(vec![
            Span::styled("  Tok/sec    ", Style::new().fg(ORANGE)),
            Span::raw(format!("{:.0}", state.tokens_per_sec)),
            Span::raw(format!(
                "    {}/{} samples",
                state.phase1_current, state.total_samples
            )),
        ]),
    ];
    frame.render_widget(
        Paragraph::new(p1_lines)
            .block(Block::new().borders(Borders::NONE))
            .style(Style::new().fg(Color::Gray)),
        p1_area,
    );

    // ── phase 2 progress ───────────────────────────────────────────────────────
    let p2_status = if state.phase2_done {
        "✓ done".to_string()
    } else if state.phase1_done {
        "running…".to_string()
    } else {
        "waiting for Phase 1…".to_string()
    };

    let match_rate_pct = if state.phase2_current > 0 {
        state.matched as f64 / state.phase2_current as f64 * 100.0
    } else {
        0.0
    };

    let p2_lines = vec![
        Line::from(vec![
            Span::styled("Phase 2", Style::new().fg(ORANGE)),
            Span::raw(format!(
                " (output-based)  {}   {}/{} samples",
                p2_status, state.phase2_current, state.phase2_total
            )),
        ]),
        Line::from(vec![
            Span::styled("  Match Rate ", Style::new().fg(ORANGE)),
            Span::raw(format!("{:.1}%", match_rate_pct)),
            Span::raw(format!("   ({} matched)", state.matched)),
        ]),
    ];
    frame.render_widget(
        Paragraph::new(p2_lines)
            .block(Block::new().borders(Borders::NONE))
            .style(Style::new().fg(Color::Gray)),
        p2_area,
    );

    // ── overall progress gauge ─────────────────────────────────────────────────
    let pct = state.progress_pct();
    let gauge_label = if state.all_done {
        "  complete  ".to_string()
    } else if let Some(err) = &state.error {
        // Truncate long errors to fit the gauge label; full message shown after TUI exits.
        format!("  error: {}  ", &err[..err.len().min(30)])
    } else {
        format!("  {}%   ETA {}  ", pct, state.eta_string())
    };

    let gauge_style = if state.error.is_some() {
        Style::new().fg(Color::Red).bg(Color::DarkGray)
    } else {
        Style::new().fg(ORANGE).bg(Color::DarkGray)
    };

    frame.render_widget(
        Gauge::default()
            .block(Block::new().borders(Borders::NONE))
            .gauge_style(gauge_style)
            .percent(pct)
            .label(gauge_label)
            .use_unicode(true),
        gauge_area,
    );

    // ── keybind hints ──────────────────────────────────────────────────────────
    frame.render_widget(
        Paragraph::new("  [Q] Detach and print results   [Ctrl+C] Abort")
            .style(Style::new().fg(Color::DarkGray)),
        hints_area,
    );
}

// ── entry point ────────────────────────────────────────────────────────────────

/// Main dispatch function for `gwen eval`.
///
/// Owns the top-level orchestration:
///   1. Load dataset
///   2. Start background worker (Phase 1 → Phase 2 sequentially)
///   3. Drive TUI event loop on main thread
///   4. Restore terminal, collect results, print final report
pub async fn run_eval_cmd(args: EvalArgs, _mode: gwenland_core::engine::GwenMode) {
    if let Err(e) = run_eval_inner(args).await {
        eprintln!("gwen eval error: {:?}", e);
        std::process::exit(1);
    }
}

async fn run_eval_inner(args: EvalArgs) -> anyhow::Result<()> {
    // ── load dataset ───────────────────────────────────────────────────────────
    eprintln!("Loading dataset from {}…", args.dataset);
    let samples = metrics::load_samples(&args.dataset)
        .map_err(|e| anyhow::anyhow!("failed to load dataset: {}", e))?;

    let total_samples = samples.len();
    let phase2_total = total_samples.min(args.max_samples);

    eprintln!("Loaded {} samples. Starting eval…", total_samples);

    // ── shared state ───────────────────────────────────────────────────────────
    let state = Arc::new(Mutex::new(EvalPanelState::new(
        &args.model,
        &args.dataset,
        total_samples,
        phase2_total,
    )));

    // Channel delivers the combined (MetricsResult, Vec<SampleResult>) once
    // both phases finish. Using a single channel avoids the two-channel dance
    // and makes error propagation straightforward.
    let (done_tx, done_rx) =
        std::sync::mpsc::channel::<anyhow::Result<(MetricsResult, Vec<SampleResult>)>>();

    // ── background worker thread ───────────────────────────────────────────────
    {
        let samples_clone = samples.clone();
        let model_clone = args.model.clone();
        let max_samples = args.max_samples;
        let state_p1 = Arc::clone(&state);
        let state_p2 = Arc::clone(&state);

        std::thread::spawn(move || {
            // Phase 1 — synchronous metrics computation
            let p1_cb: metrics::ProgressCallback = Box::new(move |idx, _total, running_loss, tokens, elapsed| {
                let tps = if elapsed > 0.0 { tokens as f64 / elapsed } else { 0.0 };
                let mut s = state_p1.lock().unwrap();
                s.phase1_current = idx + 1;
                s.running_loss = running_loss;
                s.running_ppl = running_loss.exp();
                s.tokens_per_sec = tps;
            });

            let p1_result = metrics::compute_metrics(&samples_clone, Some(p1_cb));

            let p1_metrics = match p1_result {
                Err(e) => {
                    let mut s = state_p2.lock().unwrap();
                    s.error = Some(format!("Phase 1 failed: {}", e));
                    s.all_done = true;
                    let _ = done_tx.send(Err(e));
                    return;
                }
                Ok(m) => {
                    {
                        let mut s = state_p2.lock().unwrap();
                        s.phase1_done = true;
                        s.running_loss = m.avg_loss;
                        s.running_ppl = m.perplexity;
                        s.tokens_per_sec = m.tokens_per_sec;
                    }
                    m
                }
            };

            // Phase 2 — async native inference, run in a dedicated tokio runtime.
            // A dedicated runtime is required because this std::thread is not
            // inside the outer tokio runtime created in main.rs; block_on on an
            // existing runtime's handle would panic.
            let rt = match tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
            {
                Ok(r) => r,
                Err(e) => {
                    let _ = done_tx.send(Err(anyhow::anyhow!("tokio build failed: {}", e)));
                    return;
                }
            };

            let state_cb = Arc::clone(&state_p2);
            let p2_result = rt.block_on(output_eval::run_output_eval(
                &samples_clone,
                &model_clone,
                max_samples,
                Some(Box::new(move |done, _total, matched| {
                    let mut s = state_cb.lock().unwrap();
                    s.phase2_current = done;
                    s.matched = matched;
                })),
            ));

            match p2_result {
                Err(e) => {
                    let mut s = state_p2.lock().unwrap();
                    s.error = Some(format!("Phase 2 failed: {}", e));
                    s.all_done = true;
                    let _ = done_tx.send(Err(e));
                }
                Ok(sample_results) => {
                    {
                        let mut s = state_p2.lock().unwrap();
                        s.phase2_done = true;
                        s.all_done = true;
                    }
                    let _ = done_tx.send(Ok((p1_metrics, sample_results)));
                }
            }
        });
    }

    // ── TUI event loop (main thread) ───────────────────────────────────────────
    let mut terminal = ratatui::init();
    let mut detached = false;

    loop {
        {
            let s = state.lock().unwrap();
            terminal.draw(|f| render_eval_panel(f, &s))?;
            if s.all_done {
                break;
            }
        }

        if crossterm::event::poll(Duration::from_millis(POLL_MS))? {
            if let crossterm::event::Event::Key(key) = crossterm::event::read()? {
                if key.kind == crossterm::event::KeyEventKind::Press {
                    match key.code {
                        crossterm::event::KeyCode::Char('q')
                        | crossterm::event::KeyCode::Char('Q') => {
                            // Detach: TUI exits, background thread keeps running
                            // (results are discarded). Mirrors [Q] in train_panel.rs.
                            detached = true;
                            break;
                        }
                        crossterm::event::KeyCode::Char('c')
                            if key.modifiers.contains(
                                crossterm::event::KeyModifiers::CONTROL,
                            ) =>
                        {
                            detached = true;
                            break;
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    ratatui::restore();

    if detached {
        println!("TUI detached. Eval continues in background.");
        return Ok(());
    }

    // ── collect results ────────────────────────────────────────────────────────
    // recv() here is non-blocking in practice: the TUI loop only breaks when
    // s.all_done == true, which the worker sets immediately before sending on
    // done_tx. The channel guarantees the value is already queued.
    let (p1_metrics, sample_results) = done_rx
        .recv()
        .map_err(|_| anyhow::anyhow!("eval worker thread disconnected unexpectedly"))??;

    let exact_match = output_eval::exact_match_rate(&sample_results);
    let report_metrics = ReportMetrics::from_parts(&p1_metrics, exact_match);

    let report = EvalReport {
        model: args.model.clone(),
        dataset: args.dataset.clone(),
        metrics: report_metrics,
        samples: sample_results,
    };

    // ── write output file ──────────────────────────────────────────────────────
    if let Some(out_path) = &args.output {
        gwenland_core::eval::report::write_report(&report, out_path)?;
        eprintln!("Report written to {}", out_path);
    }

    // ── print terminal summary ─────────────────────────────────────────────────
    print_final_report(&report);

    Ok(())
}

// ── terminal summary ───────────────────────────────────────────────────────────

/// Print the final eval report to the terminal using the spec-prescribed
/// format with Gwen Orange ANSI codes for metric labels.
fn print_final_report(report: &EvalReport) {
    let m = &report.metrics;
    let orange = "\x1b[38;2;255;140;66m";
    let reset = "\x1b[0m";
    let divider = "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━";

    println!();
    println!("📊 GwenLand Eval Results");
    println!("{}", divider);
    println!(
        "{orange}Model           {reset}: {}",
        report.model,
        orange = orange,
        reset = reset
    );
    println!(
        "{orange}Dataset         {reset}: {} ({} samples)",
        report.dataset,
        report.samples.len(),
        orange = orange,
        reset = reset
    );
    println!("{}", divider);
    println!(
        "{orange}Avg Loss        {reset}: {:.4}",
        m.avg_loss,
        orange = orange,
        reset = reset
    );
    println!(
        "{orange}Perplexity      {reset}: {:.4}",
        m.perplexity,
        orange = orange,
        reset = reset
    );
    println!(
        "{orange}Exact Match Rate{reset}: {:.1}%",
        m.exact_match_rate * 100.0,
        orange = orange,
        reset = reset
    );
    println!("{}", divider);
    println!(
        "{orange}Tokens/sec      {reset}: {:.0}",
        m.tokens_per_sec,
        orange = orange,
        reset = reset
    );
    println!(
        "{orange}Memory/token    {reset}: {:.4} MB",
        m.memory_per_token_mb,
        orange = orange,
        reset = reset
    );
    println!("{}", divider);

    // Perplexity health check thresholds from the spec.
    // PPL < 1.5 → near-certain training-set memorisation
    // PPL > 100  → model is essentially guessing (needs more training)
    if m.perplexity < 1.5 {
        println!("⚠  Very low perplexity — possible over-memorization");
    } else if m.perplexity <= 10.0 {
        println!("✅ Healthy perplexity range");
    } else if m.perplexity > 100.0 {
        println!("⚠  High perplexity — model may need more training");
    }
}
