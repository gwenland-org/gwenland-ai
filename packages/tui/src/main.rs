use clap::{CommandFactory, Parser, Subcommand, ValueEnum};
use gwenland_core::hardware;

mod app;
mod ui;
mod panes;
mod commands;
mod tui;

/// Global flags that apply to every subcommand.
#[derive(Parser, Debug, Clone, Default)]
pub struct GlobalArgs {
    /// Structured JSON output (use with --non-interactive for NDJSON streams).
    #[arg(long, global = true)]
    pub json: bool,

    /// Agent/script mode — no TUI, no spinners, no interactive prompts.
    /// Auto-enabled when stdout is not a TTY.
    #[arg(long, short = 'n', global = true)]
    pub non_interactive: bool,

    /// Pre-flight validation only — no side effects (implemented per-command).
    #[arg(long, global = true)]
    pub dry_run: bool,

    /// Auto-confirm all [Y/n] prompts.
    #[arg(long, short = 'y', global = true)]
    pub yes: bool,
}

#[derive(Parser, Debug)]
#[command(
    name = "gwen",
    version,
    about = "gwen — AI all-in-one toolkit. Local-first, <50MB, privacy-first.",
    long_about = "gwen — AI all-in-one toolkit. Local-first, <50MB, privacy-first.\n\n\
                  A unified CLI for the full LLM lifecycle: fetch → train → serve → chat,\n\
                  with dataset tooling and HuggingFace Hub integration built in.\n\n\
                  Global Flags:\n  \
                    --json              Structured JSON output\n  \
                    --non-interactive   Agent/script mode (no TUI, no prompts)\n  \
                    --dry-run           Pre-flight validation only\n  \
                    --yes, -y           Auto-confirm all prompts"
)]
pub struct Cli {
    #[command(flatten)]
    pub global: GlobalArgs,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Check environment health (CUDA, VRAM, Python deps)
    Doctor(commands::doctor::DoctorArgs),
    /// Download base model from HuggingFace
    Fetch(commands::fetch::FetchArgs),
    /// Fine-tune a model (LoRA/QLoRA)
    Train(commands::train::TrainArgs),
    /// Evaluate model performance on a validation dataset
    Eval(commands::eval::EvalArgs),
    /// Run native inference on a local GGUF model
    Run(commands::run::RunArgs),
    /// Start local inference server
    Serve(commands::serve::ServeArgs),
    /// Chat with local model (TUI)
    Chat(commands::chat::ChatArgs),
    /// HuggingFace Hub integration (model list, pull, push, info, prune)
    Hub(commands::hub_model::HubModelArgs),
    /// Dataset management (validate/convert/split)
    Dataset(commands::dataset::DatasetArgs),
    /// Safety scanner for models and datasets
    Scan(commands::scan::ScanArgs),
    /// Convert model format (GGUF ↔ SafeTensors)
    Convert(commands::convert::ConvertArgs),
    /// Benchmark GwenLand runtime (cold-start, inference, convert pipeline, memory)
    Benchmark(commands::benchmark::BenchmarkArgs),
    /// Manage GwenLand user configuration
    Config(commands::config::ConfigArgs),
    /// Self-update GwenLand to the latest release
    Update,
    /// HuggingFace Hub dataset operations (list, pull, push, info, prune)
    #[command(hide = true)]
    HubDataset(commands::hub_dataset::HubDatasetArgs),
    /// Launch GwenLand with TUI or GUI interface
    #[command(hide = true)]
    Start {
        #[arg(short = 'T', long = "type", value_enum)]
        r#type: StartType,
    },
    /// Model management
    #[command(hide = true)]
    Model {
        #[command(subcommand)]
        action: ModelCommands,
    },
    /// One-command environment bootstrap
    #[command(hide = true)]
    Setup,
}

#[derive(Debug, Clone, ValueEnum)]
pub enum StartType {
    Tui,
    Gui,
}

#[derive(Subcommand, Debug)]
pub enum ModelCommands {
    /// Fetch a model from a provider
    Fetch(commands::fetch::FetchArgs),
    /// List downloaded models
    List,
}

#[derive(Debug, Clone, ValueEnum)]
pub enum Provider {
    Ollama,
    Huggingface,
    Github,
    Direct,
}

#[derive(Debug, Clone, ValueEnum)]
pub enum QuantizeMode {
    #[value(name = "4bit")]  Bit4,
    #[value(name = "8bit")]  Bit8,
    #[value(name = "16bit")] Bit16,
    #[value(name = "fp16")]  Fp16,
    #[value(name = "fp32")]  Fp32,
}

fn main() {
    // `gwenland help` is not a Clap subcommand; mirror `--help` explicitly.
    if matches!(std::env::args().nth(1).as_deref(), Some("help")) {
        use clap::CommandFactory;
        Cli::command()
            .print_help()
            .expect("failed to print help");
        println!();
        return;
    }

    let cli = Cli::parse();

    // Runtime detection runs sync, before the tokio runtime starts.
    if matches!(cli.command, Commands::Start { .. }) {
        gwenland_core::runtime::detect_runtime();
    }

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("failed to build tokio runtime");

    if let Err(err) = rt.block_on(run_async(cli)) {
        eprintln!("GwenLand Sentinel Error: {:?}", err);
        std::process::exit(1);
    }
}

fn build_mode(g: &GlobalArgs) -> gwenland_core::engine::GwenMode {
    gwenland_core::engine::GwenMode::new(g.dry_run, g.non_interactive, g.json, g.yes)
}

async fn run_async(cli: Cli) -> Result<(), Box<dyn std::error::Error>> {
    let profile = hardware::profile();
    let mode = build_mode(&cli.global);

    if std::env::var("GWEN_DEBUG").is_ok() {
        eprintln!("[debug] cli = {:#?}", cli);
        eprintln!("[debug] mode = {:#?}", mode);
    }

    match cli.command {
        Commands::Convert(args) => {
            commands::convert::run_convert_cmd(args);
            std::process::exit(0);
        }
        Commands::Benchmark(args) => {
            commands::benchmark::run_benchmark_cmd(args);
            std::process::exit(0);
        }
        Commands::Eval(args) => {
            commands::eval::run_eval_cmd(args, mode).await;
            std::process::exit(0);
        }
        Commands::Config(args) => {
            commands::config::run_config_cmd(args);
            std::process::exit(0);
        }
        Commands::Update => {
            commands::update::run_update_cmd();
            std::process::exit(0);
        }
        Commands::Start { r#type } => match r#type {
            StartType::Tui => {
                let original_hook = std::panic::take_hook();
                std::panic::set_hook(Box::new(move |panic_info| {
                    let _ = crossterm::terminal::disable_raw_mode();
                    let _ = crossterm::execute!(
                        std::io::stdout(),
                        crossterm::terminal::LeaveAlternateScreen,
                        crossterm::cursor::Show
                    );
                    original_hook(panic_info);
                }));

                crossterm::terminal::enable_raw_mode()?;
                let mut stdout = std::io::stdout();
                crossterm::execute!(
                    stdout,
                    crossterm::terminal::EnterAlternateScreen,
                    crossterm::cursor::Show
                )?;

                let backend = ratatui::backend::CrosstermBackend::new(stdout);
                let mut terminal = ratatui::Terminal::new(backend)?;

                let mut app = app::App::new(profile);

                let res = run_app(&mut terminal, &mut app);

                let _ = crossterm::terminal::disable_raw_mode();
                let _ = crossterm::execute!(
                    terminal.backend_mut(),
                    crossterm::terminal::LeaveAlternateScreen,
                    crossterm::cursor::Show
                );

                if let Err(err) = res {
                    eprintln!("GwenLand Sentinel Error: {:?}", err);
                    std::process::exit(1);
                }
            }
            StartType::Gui => {
                eprintln!("GUI not yet implemented. Coming in Cycle 3.");
                std::process::exit(0);
            }
        },
        Commands::Model { action } => match action {
            ModelCommands::Fetch(args) => {
                commands::fetch::run_fetch(args, mode).await;
                std::process::exit(0);
            }
            ModelCommands::List => {
                eprintln!("model list: not yet implemented");
                std::process::exit(0);
            }
        },
        Commands::Doctor(args) => {
            commands::doctor::run_doctor(args).await;
            std::process::exit(0);
        }
        Commands::Setup => {
            commands::setup::run_setup(mode).await;
            std::process::exit(0);
        }
        Commands::Dataset(args) => {
            commands::dataset::run_dataset(args).await;
            std::process::exit(0);
        }
        Commands::Scan(args) => {
            commands::scan::run_scan_cmd(args).await;
            std::process::exit(0);
        }
        Commands::Fetch(args) => {
            commands::fetch::run_fetch(args, mode).await;
            std::process::exit(0);
        }
        Commands::Train(args) => {
            commands::train::run_train_cmd(args, mode).await;
            std::process::exit(0);
        }
        Commands::Hub(args) => {
            commands::hub_model::run_hub_model(args, mode).await;
            std::process::exit(0);
        }
        Commands::HubDataset(args) => {
            commands::hub_dataset::run_hub_dataset(args).await;
            std::process::exit(0);
        }
        Commands::Run(args) => {
            commands::run::run_run_cmd(args);
            std::process::exit(0);
        }
        Commands::Serve(args) => {
            commands::serve::run_serve_cmd(args, mode).await;
            std::process::exit(0);
        }
        Commands::Chat(args) => {
            commands::chat::run_chat_cmd(args, mode).await;
            std::process::exit(0);
        }
    }

    Ok(())
}

fn handle_key_event(app: &mut app::App, key: &crossterm::event::KeyEvent) {
    if key.kind != crossterm::event::KeyEventKind::Press {
        return;
    }

    if key.modifiers.contains(crossterm::event::KeyModifiers::CONTROL)
        && key.code == crossterm::event::KeyCode::Char('h')
    {
        app.toggle_hw_panel();
        return;
    }

    match key.code {
        crossterm::event::KeyCode::Char('q') | crossterm::event::KeyCode::Char('Q') if app.chat_pane.input.is_empty() => {
            app.quit();
        }
        crossterm::event::KeyCode::Char('c') | crossterm::event::KeyCode::Char('C')
            if key.modifiers.contains(crossterm::event::KeyModifiers::CONTROL) =>
        {
            app.quit();
        }
        crossterm::event::KeyCode::Enter => {
            app.submit_input();
        }
        crossterm::event::KeyCode::Backspace => {
            app.pop_char();
        }
        crossterm::event::KeyCode::Char(c) => {
            app.push_char(c);
        }
        _ => {}
    }
}

fn run_app(
    terminal: &mut ratatui::Terminal<ratatui::backend::CrosstermBackend<std::io::Stdout>>,
    app: &mut app::App,
) -> Result<(), Box<dyn std::error::Error>> {
    const POLL_TIMEOUT: std::time::Duration = std::time::Duration::from_millis(16);

    while !app.should_quit {
        app.tick();
        terminal.draw(|f| ui::render(app, f))?;

        if crossterm::event::poll(POLL_TIMEOUT)? {
            if let crossterm::event::Event::Key(key) = crossterm::event::read()? {
                handle_key_event(app, &key);
            }
        }
    }
    Ok(())
}
