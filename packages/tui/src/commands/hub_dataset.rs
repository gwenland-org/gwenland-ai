// @INFO: CLI layer for `gwen hub dataset` subcommands.
//        All heavy lifting is in gwenland_core::platform::hub_dataset.
// @EDITABLE: Add more flags here (--format, --branch, etc.) in future cycles.

use clap::{Args, Subcommand};
use gwenland_core::platform::hub_dataset::{
    hub_info, hub_list, hub_prune, hub_pull, hub_push,
    EXIT_AUTH_FAILED, EXIT_CONNECTION_FAILED, EXIT_DATASET_NOT_FOUND, EXIT_ERROR, EXIT_OK,
};
use std::path::PathBuf;

// ── top-level Args ────────────────────────────────────────────────────────────

#[derive(Args, Debug)]
#[command(
    about = "HuggingFace Hub dataset operations (list, pull, push, info, prune)",
    long_about = "Interact with HuggingFace Hub datasets: list, download, upload, inspect, and prune.\n\
                  Requires an HF token stored via `gwen hub model login` for private datasets.\n\n\
                  Examples:\n  \
                    gwen hub-dataset list --author allenai\n  \
                    gwen hub-dataset pull squad\n  \
                    gwen hub-dataset push myuser/my-dataset ./data/\n  \
                    gwen hub-dataset info allenai/c4\n  \
                    gwen hub-dataset prune squad"
)]
pub struct HubDatasetArgs {
    #[command(subcommand)]
    pub action: HubDatasetCommands,
}

// ── subcommands ───────────────────────────────────────────────────────────────

#[derive(Subcommand, Debug)]
pub enum HubDatasetCommands {
    /// List datasets from HuggingFace Hub
    List(ListArgs),

    /// Download a dataset from HuggingFace Hub into the local cache
    Pull(PullArgs),

    /// Upload a local dataset or directory to HuggingFace Hub
    Push(PushArgs),

    /// Print metadata for a dataset on HuggingFace Hub
    Info(InfoArgs),

    /// Delete the local HF Hub cache for a specific dataset
    Prune(PruneArgs),
}

// ── per-subcommand Args ───────────────────────────────────────────────────────

#[derive(Args, Debug)]
pub struct ListArgs {
    /// Filter by owner or organisation (e.g. datasets-community, allenai)
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
    /// HuggingFace dataset ID (e.g. squad, allenai/c4)
    #[arg(required = true)]
    pub dataset_id: String,

    /// Git revision, branch, or tag to download (default: main)
    #[arg(long, short = 'r')]
    pub revision: Option<String>,

    /// Output raw JSON instead of formatted output
    #[arg(long)]
    pub json: bool,

    /// Skip confirmation prompt
    #[arg(long, short = 'y')]
    pub yes: bool,
}

#[derive(Args, Debug)]
pub struct PushArgs {
    /// HuggingFace dataset ID to push to (e.g. myuser/my-dataset)
    #[arg(required = true)]
    pub dataset_id: String,

    /// Local file or directory to upload
    #[arg(required = true)]
    pub path: PathBuf,

    /// Commit message for the upload
    #[arg(long, short = 'm', default_value = "Upload via GwenLand")]
    pub message: String,

    /// Output raw JSON
    #[arg(long)]
    pub json: bool,

    /// Skip confirmation prompt
    #[arg(long, short = 'y')]
    pub yes: bool,
}

#[derive(Args, Debug)]
pub struct InfoArgs {
    /// HuggingFace dataset ID (e.g. squad, allenai/c4)
    #[arg(required = true)]
    pub dataset_id: String,

    /// Output raw JSON
    #[arg(long)]
    pub json: bool,
}

#[derive(Args, Debug)]
pub struct PruneArgs {
    /// HuggingFace dataset ID whose cache to delete (e.g. squad, allenai/c4)
    #[arg(required = true)]
    pub dataset_id: String,

    /// Skip confirmation prompt
    #[arg(long, short = 'y')]
    pub yes: bool,

    /// Output raw JSON
    #[arg(long)]
    pub json: bool,
}

// ── dispatch ──────────────────────────────────────────────────────────────────

pub async fn run_hub_dataset(args: HubDatasetArgs) {
    let code = match args.action {
        HubDatasetCommands::List(a) => run_list(a).await,
        HubDatasetCommands::Pull(a) => run_pull(a).await,
        HubDatasetCommands::Push(a) => run_push(a).await,
        HubDatasetCommands::Info(a) => run_info(a).await,
        HubDatasetCommands::Prune(a) => run_prune(a).await,
    };
    std::process::exit(code);
}

// ── list ──────────────────────────────────────────────────────────────────────

async fn run_list(args: ListArgs) -> i32 {
    match hub_list(args.author.as_deref(), args.search.as_deref(), args.limit).await {
        Ok(entries) => {
            if args.json {
                println!("{}", serde_json::to_string_pretty(&entries).unwrap_or_default());
                return EXIT_OK;
            }
            if entries.is_empty() {
                println!("No datasets found.");
                return EXIT_OK;
            }
            println!("{:<50} {:<12} {}", "Dataset ID", "Files", "SHA");
            println!("{}", "─".repeat(80));
            for e in &entries {
                println!(
                    "{:<50} {:<12} {}",
                    truncate(&e.dataset_id, 48),
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

async fn run_pull(args: PullArgs) -> i32 {
    if !args.yes && !args.json {
        let confirmed = inquire::Confirm::new(&format!(
            "Download dataset '{}' from HF Hub?",
            args.dataset_id
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
    match hub_pull(&args.dataset_id, args.revision.as_deref(), show_progress).await {
        Ok(result) => {
            if args.json {
                println!("{}", serde_json::to_string_pretty(&result).unwrap_or_default());
            } else {
                println!(
                    "  ✓ Downloaded {} ({} file{}) → {}",
                    result.dataset_id,
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
    use gwenland_core::platform::hub_dataset::resolve_token;

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
            "Upload '{}' to dataset '{}'?",
            args.path.display(),
            args.dataset_id
        ))
        .with_default(false)
        .prompt()
        .unwrap_or(false);
        if !confirmed {
            println!("Aborted.");
            return EXIT_OK;
        }
    }

    match hub_push(&args.dataset_id, &args.path, &args.message).await {
        Ok(result) => {
            if args.json {
                println!("{}", serde_json::to_string_pretty(&result).unwrap_or_default());
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
    match hub_info(&args.dataset_id).await {
        Ok(info) => {
            if args.json {
                println!("{}", serde_json::to_string_pretty(&info).unwrap_or_default());
                return EXIT_OK;
            }

            println!("Dataset: {}", info.dataset_id);
            println!("SHA:     {}", info.sha);
            println!("Private: {}", info.private);
            if let Some(lic) = &info.license {
                println!("License: {}", lic);
            }
            if let Some(bytes) = info.size_bytes {
                println!("Size:    {}", format_bytes(bytes));
            }
            if let Some(rows) = info.num_rows {
                println!("Rows:    {}", rows);
            }
            if !info.splits.is_empty() {
                println!("Splits:");
                for sp in &info.splits {
                    match sp.num_examples {
                        Some(n) => println!("  {} — {} rows", sp.name, n),
                        None    => println!("  {}", sp.name),
                    }
                }
            }
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
            "Delete local cache for dataset '{}'? This cannot be undone.",
            args.dataset_id
        ))
        .with_default(false)
        .prompt()
        .unwrap_or(false);
        if !confirmed {
            println!("Aborted.");
            return EXIT_OK;
        }
    }

    match hub_prune(&args.dataset_id) {
        Ok(result) => {
            if args.json {
                println!("{}", serde_json::to_string_pretty(&result).unwrap_or_default());
            } else {
                println!(
                    "  ✓ Pruned '{}' — freed {}",
                    result.dataset_id,
                    format_bytes(result.bytes_freed)
                );
            }
            EXIT_OK
        }
        Err(e) => {
            let msg = e.to_string();
            eprintln!("error: {}", msg);
            if msg.contains("no cached files") {
                EXIT_DATASET_NOT_FOUND
            } else {
                EXIT_ERROR
            }
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

fn classify_exit_code(msg: &str) -> i32 {
    let lower = msg.to_lowercase();
    if lower.contains("401") || lower.contains("auth") || lower.contains("token") {
        EXIT_AUTH_FAILED
    } else if lower.contains("not found") || lower.contains("404") {
        EXIT_DATASET_NOT_FOUND
    } else if lower.contains("connection") || lower.contains("connect") || lower.contains("timeout") {
        EXIT_CONNECTION_FAILED
    } else {
        EXIT_ERROR
    }
}
