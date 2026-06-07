use clap::Parser;
use std::fmt;
use std::path::PathBuf;
use serde::Deserialize;

#[derive(Parser, Debug, Clone)]
#[command(
    about = "Download base model from HuggingFace",
    long_about = "Download a model from HuggingFace Hub into the local model cache.\n\
                  Resumes interrupted downloads automatically and verifies the SHA-256 checksum.\n\n\
                  Examples:\n  \
                    gwen fetch -m tinyllama/TinyLlama-1.1B -q q4_k_m\n  \
                    gwen fetch -m mistralai/Mistral-7B-v0.1 -q q5_k_m\n  \
                    gwen fetch -m meta-llama/Llama-3-8B -q q4_k_m -m another/Model -q q4_k_m\n  \
                    gwen fetch --from https://example.com/model.gguf --to /data/models\n  \
                    gwen fetch -m tinyllama/TinyLlama-1.1B --cache-clear"
)]
pub struct FetchArgs {
    /// HuggingFace model ID to fetch (e.g. tinyllama/TinyLlama-1.1B). Repeatable up to 3 times.
    #[arg(short = 'm', long = "model", action = clap::ArgAction::Append, num_args = 1..=3, value_name = "MODEL_ID")]
    pub models: Vec<String>,

    /// Quantization variant to download (e.g. q4_k_m, q5_k_m, q8_0). Required in --non-interactive mode.
    #[arg(short = 'q', long = "quantize", value_name = "QUANT")]
    pub quantize: Option<String>,

    /// Download from a direct HTTPS URL instead of HuggingFace Hub
    #[arg(long, value_name = "URL")]
    pub from: Option<String>,

    /// Save downloaded file to this directory instead of the default model cache
    #[arg(long, value_name = "DIR")]
    pub to: Option<String>,

    /// Delete the .tmp/ partial cache after the download completes
    #[arg(long)]
    pub ephemeral: bool,

    /// Delete all .tmp/ partial files and exit (use to recover from a corrupt download)
    #[arg(long)]
    pub cache_clear: bool,

    /// Skip resuming a partial download — start from byte 0
    #[arg(long)]
    pub no_resume: bool,
}

#[derive(Debug)]
pub enum FetchError {
    ModelNotFoundHub(String),
    QuantNotAvailable(String, String),
    QuantRequiredNonInteractive(String),
    DownloadFailed(String),
    ChecksumMismatch(String),
    InsufficientDisk(String, u64, u64),
    InvalidToken,
    NetworkUnreachable(String),
}

impl std::error::Error for FetchError {}

impl fmt::Display for FetchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "error: ")?;
        match self {
            FetchError::ModelNotFoundHub(m) => {
                write!(f, "model not found on HuggingFace Hub.\n\n  The repository '{}' does not exist or is private.\n\n  hint:\n    Check your spelling or provide an access token in ~/.config/gwen/config.toml", m)
            }
            FetchError::QuantNotAvailable(m, q) => {
                write!(f, "quantization not available.\n\n  The model '{}' does not have a '{}' GGUF file.\n\n  hint:\n    Run `gwen fetch -m {}` without -q to see available options.", m, q, m)
            }
            FetchError::QuantRequiredNonInteractive(m) => {
                write!(f, "quantization required in non-interactive mode.\n\n  Terminal is not a TTY and no -q flag was provided.\n\n  hint:\n    Re-run with: gwen fetch -m {} -q q4_k_m", m)
            }
            FetchError::DownloadFailed(err) => {
                write!(f, "download failed.\n\n  {}\n\n  hint:\n    Check your network connection and try again.", err)
            }
            FetchError::ChecksumMismatch(file) => {
                write!(f, "checksum mismatch.\n\n  The downloaded file '{}' does not match the expected SHA256 hash.\n\n  hint:\n    The partial file has been deleted. Run the fetch command again.", file)
            }
            FetchError::InsufficientDisk(path, req, avail) => {
                write!(f, "insufficient disk space.\n\n  Downloading requires {} MB, but only {} MB is available on {}.\n\n  hint:\n    Free up space or use the --to flag to specify a different drive: gwen fetch -m jinxsuperdev/gwen1.0-code-mini --to D:\\models", req, avail, path)
            }
            FetchError::InvalidToken => {
                write!(f, "invalid HuggingFace token.\n\n  The token provided in config.toml is rejected by the API.\n\n  hint:\n    Update your token in ~/.config/gwen/config.toml")
            }
            FetchError::NetworkUnreachable(err) => {
                write!(f, "network unreachable.\n\n  {}\n\n  hint:\n    Ensure you are connected to the internet.", err)
            }
        }
    }
}

#[derive(Deserialize, Clone)]
struct HfTreeItem {
    path: String,
    size: Option<u64>,
    lfs: Option<HfLfsInfo>,
}

#[derive(Deserialize, Clone)]
struct HfLfsInfo {
    oid: String,
    size: u64,
}

pub async fn run_fetch(args: FetchArgs, mode: gwenland_core::engine::GwenMode) {
    if let Err(e) = do_fetch(args, mode).await {
        eprintln!("{}", e);
        std::process::exit(1);
    }
}

async fn do_fetch(args: FetchArgs, mode: gwenland_core::engine::GwenMode) -> Result<(), FetchError> {
    let tmp_dir = gwenland_core::storage::paths::GwenPaths::tmp_dir();

    if args.cache_clear {
        if tmp_dir.exists() {
            if let Err(e) = std::fs::remove_dir_all(&tmp_dir) {
                return Err(FetchError::DownloadFailed(format!("Failed to clear cache: {}", e)));
            }
            if !mode.non_interactive {
                println!("Cache cleared.");
            }
        }
        if args.models.is_empty() && args.from.is_none() {
            return Ok(());
        }
    }

    if args.models.is_empty() && args.from.is_none() {
        return Ok(());
    }

    let mut token = load_hf_token();

    // Download plans
    let mut plans = Vec::new();

    for model in &args.models {
        let url = format!("https://huggingface.co/api/models/{}/tree/main", model);
        let client = reqwest::Client::new();
        let mut req = client.get(&url);
        if let Some(t) = &token {
            req = req.bearer_auth(t);
        }
        
        let mut resp = req.send().await.map_err(|e| FetchError::NetworkUnreachable(e.to_string()))?;

        if resp.status() == reqwest::StatusCode::UNAUTHORIZED || resp.status() == reqwest::StatusCode::FORBIDDEN || resp.status() == reqwest::StatusCode::NOT_FOUND {
            if token.is_none() {
                if mode.non_interactive || !atty::is(atty::Stream::Stdout) {
                    return Err(FetchError::ModelNotFoundHub(model.clone()));
                }
                println!("The model '{}' requires authentication or does not exist.", model);
                let new_token = inquire::Password::new("Enter HuggingFace access token:")
                    .prompt()
                    .map_err(|_| FetchError::InvalidToken)?;
                
                let verify_resp = client.get("https://huggingface.co/api/whoami-v2")
                    .bearer_auth(&new_token)
                    .send().await.map_err(|e| FetchError::NetworkUnreachable(e.to_string()))?;
                
                if verify_resp.status().is_success() {
                    save_hf_token(&new_token);
                    token = Some(new_token.clone());
                    
                    resp = client.get(&url).bearer_auth(&new_token).send().await.map_err(|e| FetchError::NetworkUnreachable(e.to_string()))?;
                    if !resp.status().is_success() {
                        return Err(FetchError::ModelNotFoundHub(model.clone()));
                    }
                } else {
                    return Err(FetchError::InvalidToken);
                }
            } else {
                return Err(FetchError::ModelNotFoundHub(model.clone()));
            }
        }

        let items: Vec<HfTreeItem> = resp.json().await.map_err(|e| FetchError::NetworkUnreachable(e.to_string()))?;
        
        let mut ggufs: Vec<HfTreeItem> = items.into_iter().filter(|i| i.path.ends_with(".gguf")).collect();
        if ggufs.is_empty() {
            return Err(FetchError::QuantNotAvailable(model.clone(), "any".to_string()));
        }

        let selected_item = if let Some(q) = &args.quantize {
            let q_upper = q.to_uppercase();
            let q_lower = q.to_lowercase();
            let matched = ggufs.into_iter().find(|i| i.path.contains(&q_upper) || i.path.contains(&q_lower));
            if let Some(m) = matched {
                m
            } else {
                return Err(FetchError::QuantNotAvailable(model.clone(), q.clone()));
            }
        } else {
            if mode.non_interactive || !atty::is(atty::Stream::Stdout) {
                return Err(FetchError::QuantRequiredNonInteractive(model.clone()));
            }
            
            let options: Vec<String> = ggufs.iter().map(|g| {
                let size_gb = g.size.unwrap_or(g.lfs.as_ref().map(|l| l.size).unwrap_or(0)) as f64 / 1024.0 / 1024.0 / 1024.0;
                format!("{} ({:.1} GB)", g.path, size_gb)
            }).collect();
            
            let ans = inquire::Select::new(&format!("Select quantization for {}:", model), options.clone())
                .prompt()
                .map_err(|_| FetchError::QuantRequiredNonInteractive(model.clone()))?;
            
            let idx = options.iter().position(|x| x == &ans)
                .expect("inquire returned an answer not in the options list");
            ggufs.remove(idx)
        };

        plans.push((model.clone(), selected_item));
    }

    // Concurrent download logic
    let m = indicatif::MultiProgress::new();
    let mut tasks = Vec::new();

    for (model, item) in &plans {
        let item_size = item.size.unwrap_or(item.lfs.as_ref().map(|l| l.size).unwrap_or(0));
        let pb = if mode.non_interactive {
            indicatif::ProgressBar::hidden()
        } else {
            let bar = m.add(indicatif::ProgressBar::new(item_size));
            bar.set_style(
                indicatif::ProgressStyle::with_template(
                    "{prefix:>20.cyan.bold} [{bar:40.green/dim}] {bytes}/{total_bytes} ({bytes_per_sec}, ETA: {eta})"
                )
                .unwrap_or_else(|_| indicatif::ProgressStyle::default_bar())
                .progress_chars("━╸━")
            );
            bar.set_prefix(model.clone());
            bar
        };

        let token_clone = token.clone();
        let url = format!("https://huggingface.co/{}/resolve/main/{}", model, item.path);
        
        let expected_sha = item.lfs.as_ref().map(|l| l.oid.clone());
        let dest_dir = if let Some(to) = &args.to {
            PathBuf::from(to)
        } else {
            let safe_name = model.replace("/", "--");
            gwenland_core::storage::paths::GwenPaths::models_dir().join(safe_name)
        };
        
        let no_resume = args.no_resume;
        let tmp_dir_clone = tmp_dir.clone();
        let model_clone = model.clone();
        let item_clone = item.clone();
        
        tasks.push(tokio::spawn(async move {
            if let Err(e) = std::fs::create_dir_all(&dest_dir) {
                return Err(FetchError::DownloadFailed(e.to_string()));
            }
            if let Err(e) = std::fs::create_dir_all(&tmp_dir_clone) {
                return Err(FetchError::DownloadFailed(e.to_string()));
            }

            let file_name = item_clone.path.split('/').last().unwrap_or("model.gguf");
            let final_path = dest_dir.join(file_name);
            let tmp_path = tmp_dir_clone.join(format!("{}.partial", file_name));

            if item_size > 0 {
                let required_mb = item_size / 1024 / 1024;
                if let Some((_, avail)) = gwenland_core::hardware::check_disk_space(&dest_dir) {
                    let avail_mb = avail / 1024 / 1024;
                    if avail_mb < required_mb {
                        return Err(FetchError::InsufficientDisk(dest_dir.display().to_string(), required_mb, avail_mb));
                    }
                }
            }

            let client = reqwest::Client::new();
            let mut req = client.get(&url);
            if let Some(t) = &token_clone {
                req = req.bearer_auth(t);
            }

            let mut start_byte = 0;
            if tmp_path.exists() && !no_resume {
                if let Ok(meta) = std::fs::metadata(&tmp_path) {
                    start_byte = meta.len();
                    if start_byte > 0 {
                        req = req.header(reqwest::header::RANGE, format!("bytes={}-", start_byte));
                    }
                }
            } else {
                if tmp_path.exists() {
                    let _ = std::fs::remove_file(&tmp_path);
                }
            }

            let mut resp = req.send().await.map_err(|e| FetchError::NetworkUnreachable(e.to_string()))?;
            if !resp.status().is_success() {
                // If RANGE failed (e.g. 416), we should really fallback to restarting, but for now just error.
                if resp.status() == reqwest::StatusCode::RANGE_NOT_SATISFIABLE {
                    let _ = std::fs::remove_file(&tmp_path);
                    return Err(FetchError::DownloadFailed("Range not satisfiable. Please try again with --no-resume".to_string()));
                }
                return Err(FetchError::DownloadFailed(format!("HTTP Error {}", resp.status())));
            }

            let total_size = resp.content_length().unwrap_or(0) + start_byte;
            pb.set_length(total_size);
            pb.set_position(start_byte);

            use tokio::io::AsyncWriteExt;
            use sha2::Digest;

            let mut file = tokio::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&tmp_path)
                .await
                .map_err(|e| FetchError::DownloadFailed(e.to_string()))?;

            let mut hasher = sha2::Sha256::new();
            if start_byte > 0 {
                let mut f = std::fs::File::open(&tmp_path).map_err(|e| FetchError::DownloadFailed(e.to_string()))?;
                std::io::copy(&mut f, &mut hasher).map_err(|e| FetchError::DownloadFailed(e.to_string()))?;
            }

            while let Some(chunk) = resp.chunk().await.map_err(|e| FetchError::DownloadFailed(e.to_string()))? {
                file.write_all(&chunk).await.map_err(|e| FetchError::DownloadFailed(e.to_string()))?;
                hasher.update(&chunk);
                pb.inc(chunk.len() as u64);
            }

            file.flush().await.map_err(|e| FetchError::DownloadFailed(e.to_string()))?;
            pb.finish_with_message("Done");

            let final_hash = format!("{:x}", hasher.finalize());
            if let Some(expected) = expected_sha {
                if final_hash != expected {
                    let _ = std::fs::remove_file(&tmp_path);
                    return Err(FetchError::ChecksumMismatch(file_name.to_string()));
                }
            }

            std::fs::rename(&tmp_path, &final_path).map_err(|e| FetchError::DownloadFailed(e.to_string()))?;

            #[derive(serde::Serialize)]
            struct ModelMeta {
                source: String,
                quant: String,
                size: u64,
                downloaded_at: String,
                sha256: String,
            }

            let meta = ModelMeta {
                source: model_clone,
                quant: item_clone.path.clone(),
                size: total_size,
                downloaded_at: chrono::Utc::now().to_rfc3339(),
                sha256: final_hash,
            };

            let meta_str = serde_json::to_string_pretty(&meta)
                .unwrap_or_else(|_| "{}".to_string());
            let _ = std::fs::write(dest_dir.join("metadata.json"), meta_str);

            Ok::<(), FetchError>(())
        }));
    }

    for task in tasks {
        if let Ok(res) = task.await {
            res?;
        }
    }

    // Persist downloaded models to models.toml registry
    if let Ok(mut reg) = gwenland_core::storage::registry::ModelRegistry::load() {
        for (model, item) in plans {
            reg.upsert(gwenland_core::storage::registry::ModelEntry {
                id: model.clone(),
                source: model,
                format: "gguf".to_string(),
                quant: String::new(),
                size_bytes: 0,
                downloaded_at: chrono::Utc::now().to_rfc3339(),
                sha256: String::new(),
                path: item.path.into(),
            });
        }
        let _ = reg.save();
    }

    if args.ephemeral {
        let _ = std::fs::remove_dir_all(&tmp_dir);
    }

    Ok(())
}

fn load_hf_token() -> Option<String> {
    // Prefer OS keyring (set via `gwen hub model login`)
    if let Ok(entry) = keyring::Entry::new("gwenland", "hf_token") {
        if let Ok(token) = entry.get_password() {
            if !token.is_empty() {
                return Some(token);
            }
        }
    }
    // Fall back to HF_TOKEN environment variable
    std::env::var("HF_TOKEN").ok().filter(|t| !t.is_empty())
}

fn save_hf_token(token: &str) {
    // Tokens are stored in OS keyring only — never in config.toml
    if let Ok(entry) = keyring::Entry::new("gwenland", "hf_token") {
        let _ = entry.set_password(token);
    }
}
