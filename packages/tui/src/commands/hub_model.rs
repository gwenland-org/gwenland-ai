// @INFO: CLI layer for `gwen hub model` subcommands.
//        All heavy lifting is in gwenland_core::platform::hub_model.
// @EDITABLE: Add more flags here (--format, --branch, etc.) in future cycles.

use clap::{Args, Subcommand};
use gwenland_core::platform::hub_model::{
    hub_info, hub_list, hub_prune, hub_pull, hub_push, resolve_token, store_token, delete_token,
    EXIT_AUTH_FAILED, EXIT_CONNECTION_FAILED, EXIT_ERROR, EXIT_MODEL_NOT_FOUND, EXIT_OK,
};
use std::path::PathBuf;

// ── top-level Args ────────────────────────────────────────────────────────────

#[derive(Args, Debug)]
#[command(
    about = "HuggingFace Hub integration (model list, pull, push, info, prune)",
    long_about = "Interact with HuggingFace Hub: list, download, upload, inspect, and prune models.\n\
                  Authentication: run `gwen hub model login` to store your HF token in the OS keyring.\n\
                  The token is never written to config files.\n\n\
                  Examples:\n  \
                    gwen hub model list --author mistralai\n  \
                    gwen hub model pull mistralai/Mistral-7B-v0.1\n  \
                    gwen hub model push myuser/my-finetuned-model ./output/\n  \
                    gwen hub model info tinyllama/TinyLlama-1.1B\n  \
                    gwen hub model prune mistralai/Mistral-7B-v0.1\n  \
                    gwen hub model login\n  \
                    gwen hub model logout"
)]
pub struct HubModelArgs {
    #[command(subcommand)]
    pub action: HubModelCommands,
}

// ── subcommands ───────────────────────────────────────────────────────────────

#[derive(Subcommand, Debug)]
pub enum HubModelCommands {
    /// List models from HuggingFace Hub
    List(ListArgs),

    /// Download a model from HuggingFace Hub into the local cache
    Pull(PullArgs),

    /// Upload a local model or directory to HuggingFace Hub
    Push(PushArgs),

    /// Print metadata for a model on HuggingFace Hub
    Info(InfoArgs),

    /// Delete the local HF Hub cache for a specific model
    Prune(PruneArgs),

    /// Store your HF token in the OS keyring (never written to config)
    Login(LoginArgs),

    /// Remove stored HF token from the OS keyring
    Logout,
}

// ── per-subcommand Args ───────────────────────────────────────────────────────

#[derive(Args, Debug)]
pub struct ListArgs {
    /// Filter by owner or organisation (e.g. mistralai, meta-llama)
    #[arg(long, short = 'a')]
    pub author: Option<String>,

    /// Search query string
    #[arg(long, short = 's')]
    pub search: Option<String>,

    /// Maximum number of results to return (max 100)
    #[arg(long, short = 'n', default_value = "20")]
    pub limit: usize,

    /// Output raw JSON instead of formatted table
    #[arg(long)]
    pub json: bool,
}

#[derive(Args, Debug)]
pub struct PullArgs {
    /// HuggingFace model ID (e.g. mistralai/Mistral-7B-v0.1)
    #[arg(required = true)]
    pub model_id: String,

    /// Git revision, branch, or tag to download (default: main)
    #[arg(long, short = 'r')]
    pub revision: Option<String>,

    /// Output raw JSON instead of formatted output
    #[arg(long)]
    pub json: bool,

    /// Skip confirmation prompts
    #[arg(long, short = 'y')]
    pub yes: bool,
}

#[derive(Args, Debug)]
pub struct PushArgs {
    /// HuggingFace model ID to push to (e.g. myuser/my-fine-tuned-model)
    #[arg(required = true)]
    pub model_id: String,

    /// Local file or directory to upload
    #[arg(required = true)]
    pub path: PathBuf,

    /// Commit message for the upload
    #[arg(long, short = 'm', default_value = "Upload via GwenLand")]
    pub message: String,

    /// Output raw JSON
    #[arg(long)]
    pub json: bool,

    /// Skip confirmation prompts
    #[arg(long, short = 'y')]
    pub yes: bool,
}

#[derive(Args, Debug)]
pub struct InfoArgs {
    /// HuggingFace model ID (e.g. mistralai/Mistral-7B-v0.1)
    #[arg(required = true)]
    pub model_id: String,

    /// Output raw JSON
    #[arg(long)]
    pub json: bool,
}

#[derive(Args, Debug)]
pub struct PruneArgs {
    /// HuggingFace model ID whose cache to delete (e.g. mistralai/Mistral-7B-v0.1)
    #[arg(required = true)]
    pub model_id: String,

    /// Skip confirmation prompt
    #[arg(long, short = 'y')]
    pub yes: bool,

    /// Output raw JSON
    #[arg(long)]
    pub json: bool,
}

#[derive(Args, Debug)]
pub struct LoginArgs {
    /// HF token to store in OS keyring (reads from stdin if not provided)
    #[arg(long)]
    pub token: Option<String>,
}

// ── dispatch ──────────────────────────────────────────────────────────────────

pub async fn run_hub_model(args: HubModelArgs, mode: gwenland_core::engine::GwenMode) {
    let code = match args.action {
        HubModelCommands::List(a) => run_list(a).await,
        HubModelCommands::Pull(a) => run_pull(a, &mode).await,
        HubModelCommands::Push(a) => run_push(a).await,
        HubModelCommands::Info(a) => run_info(a).await,
        HubModelCommands::Prune(a) => run_prune(a).await,
        HubModelCommands::Login(a) => run_login(a),
        HubModelCommands::Logout => run_logout(),
    };
    std::process::exit(code);
}

// ── list ──────────────────────────────────────────────────────────────────────

async fn run_list(args: ListArgs) -> i32 {
    match hub_list(
        args.author.as_deref(),
        args.search.as_deref(),
        args.limit,
    )
    .await
    {
        Ok(entries) => {
            if args.json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&entries).unwrap_or_default()
                );
                return EXIT_OK;
            }

            if entries.is_empty() {
                println!("No models found.");
                return EXIT_OK;
            }

            println!("{:<50} {:<12} {}", "Model ID", "Files", "SHA");
            println!("{}", "─".repeat(80));
            for e in &entries {
                println!(
                    "{:<50} {:<12} {}",
                    truncate(&e.model_id, 48),
                    e.file_count,
                    short_sha(&e.sha)
                );
            }
            EXIT_OK
        }
        Err(e) => {
            let msg = e.to_string();
            eprintln!("error: {}", msg);
            classify_exit_code(&msg)
        }
    }
}

// ── pull ──────────────────────────────────────────────────────────────────────

async fn run_pull(args: PullArgs, mode: &gwenland_core::engine::GwenMode) -> i32 {
    // --dry-run: check token, cache status, disk space — no download
    if mode.dry_run {
        let report = gwenland_core::platform::hub_model::dry_run_hub_pull(&args.model_id);
        if mode.json || args.json {
            gwenland_core::dry_run::print_json(&report);
        } else {
            gwenland_core::dry_run::print_report(&report);
        }
        return report.exit_code();
    }

    if !args.yes && !args.json && !mode.yes && !mode.non_interactive {
        let confirmed = inquire::Confirm::new(&format!(
            "Download '{}' from HF Hub?",
            args.model_id
        ))
        .with_default(true)
        .prompt()
        .unwrap_or(false);
        if !confirmed {
            println!("Aborted.");
            return EXIT_OK;
        }
    }

    let show_progress = !args.json;
    match hub_pull(&args.model_id, args.revision.as_deref(), show_progress).await {
        Ok(result) => {
            if args.json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&result).unwrap_or_default()
                );
            } else {
                println!(
                    "  ✓ Downloaded {} ({} file{}) → {}",
                    result.model_id,
                    result.files_downloaded,
                    if result.files_downloaded == 1 { "" } else { "s" },
                    result.cache_path.display()
                );
            }
            EXIT_OK
        }
        Err(e) => {
            let msg = e.to_string();
            eprintln!("error: {}", msg);
            classify_exit_code(&msg)
        }
    }
}

// ── push ──────────────────────────────────────────────────────────────────────

async fn run_push(args: PushArgs) -> i32 {
    if !args.path.exists() {
        eprintln!("error: path '{}' does not exist", args.path.display());
        return EXIT_ERROR;
    }

    if resolve_token().is_none() {
        eprintln!(
            "error: no HF token found. Run `gwen hub model login` or set HF_TOKEN env var."
        );
        return EXIT_AUTH_FAILED;
    }

    if !args.yes && !args.json {
        let confirmed = inquire::Confirm::new(&format!(
            "Upload '{}' to '{}'?",
            args.path.display(),
            args.model_id
        ))
        .with_default(false)
        .prompt()
        .unwrap_or(false);
        if !confirmed {
            println!("Aborted.");
            return EXIT_OK;
        }
    }

    match hub_push(&args.model_id, &args.path, &args.message).await {
        Ok(result) => {
            if args.json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&result).unwrap_or_default()
                );
            } else {
                println!(
                    "  ✓ Pushed {} file{} to {}",
                    result.files_uploaded,
                    if result.files_uploaded == 1 { "" } else { "s" },
                    result.repo_url
                );
            }
            EXIT_OK
        }
        Err(e) => {
            let msg = e.to_string();
            eprintln!("error: {}", msg);
            classify_exit_code(&msg)
        }
    }
}

// ── info ──────────────────────────────────────────────────────────────────────

async fn run_info(args: InfoArgs) -> i32 {
    match hub_info(&args.model_id).await {
        Ok(info) => {
            if args.json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&info).unwrap_or_default()
                );
                return EXIT_OK;
            }

            println!("Model:   {}", info.model_id);
            println!("SHA:     {}", info.sha);
            println!("Private: {}", info.private);
            println!("Files:   {} total", info.files.len());
            println!("{}", "─".repeat(60));
            for f in &info.files {
                match f.size_bytes {
                    Some(b) => println!("  {:<50} {}", f.filename, format_bytes(b)),
                    None    => println!("  {}", f.filename),
                }
            }
            EXIT_OK
        }
        Err(e) => {
            let msg = e.to_string();
            eprintln!("error: {}", msg);
            classify_exit_code(&msg)
        }
    }
}

// ── prune ─────────────────────────────────────────────────────────────────────

async fn run_prune(args: PruneArgs) -> i32 {
    if !args.yes && !args.json {
        let confirmed = inquire::Confirm::new(&format!(
            "Delete local cache for '{}'? This cannot be undone.",
            args.model_id
        ))
        .with_default(false)
        .prompt()
        .unwrap_or(false);
        if !confirmed {
            println!("Aborted.");
            return EXIT_OK;
        }
    }

    match hub_prune(&args.model_id) {
        Ok(result) => {
            if args.json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&result).unwrap_or_default()
                );
            } else {
                println!(
                    "  ✓ Pruned '{}' — freed {}",
                    result.model_id,
                    format_bytes(result.bytes_freed)
                );
            }
            EXIT_OK
        }
        Err(e) => {
            let msg = e.to_string();
            eprintln!("error: {}", msg);
            if msg.contains("no cached files") {
                EXIT_MODEL_NOT_FOUND
            } else {
                EXIT_ERROR
            }
        }
    }
}

// ── login / logout ────────────────────────────────────────────────────────────

fn run_login(args: LoginArgs) -> i32 {
    let token = match args.token {
        Some(t) => t,
        None => {
            // Read from stdin (hidden)
            match inquire::Password::new("HF token:")
                .without_confirmation()
                .prompt()
            {
                Ok(t) => t,
                Err(_) => {
                    eprintln!("error: could not read token from stdin");
                    return EXIT_ERROR;
                }
            }
        }
    };

    let token = token.trim().to_string();
    if token.is_empty() {
        eprintln!("error: token is empty");
        return EXIT_ERROR;
    }

    match store_token(&token) {
        Ok(()) => {
            println!("  ✓ HF token stored in OS keyring (not written to disk).");
            EXIT_OK
        }
        Err(e) => {
            eprintln!("error: {}", e);
            EXIT_ERROR
        }
    }
}

fn run_logout() -> i32 {
    match delete_token() {
        Ok(()) => {
            println!("  ✓ HF token removed from OS keyring.");
            EXIT_OK
        }
        Err(e) => {
            eprintln!("error: {}", e);
            EXIT_ERROR
        }
    }
}

// ── formatting helpers ────────────────────────────────────────────────────────

fn short_sha(sha: &str) -> &str {
    if sha.len() >= 8 { &sha[..8] } else { sha }
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let truncated: String = s.chars().take(max.saturating_sub(1)).collect();
        format!("{}…", truncated)
    }
}

fn format_bytes(bytes: u64) -> String {
    const GB: u64 = 1_073_741_824;
    const MB: u64 = 1_048_576;
    const KB: u64 = 1_024;
    if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

/// Map error message to standardised exit code.
fn classify_exit_code(msg: &str) -> i32 {
    let lower = msg.to_lowercase();
    if lower.contains("401") || lower.contains("auth") || lower.contains("token") {
        EXIT_AUTH_FAILED
    } else if lower.contains("not found") || lower.contains("404") {
        EXIT_MODEL_NOT_FOUND
    } else if lower.contains("connection") || lower.contains("connect") || lower.contains("timeout") {
        EXIT_CONNECTION_FAILED
    } else {
        EXIT_ERROR
    }
}
