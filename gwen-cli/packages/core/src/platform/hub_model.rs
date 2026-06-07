// @INFO: HuggingFace Hub model operations for `gwen hub model`.
// @INFO: Uses hf-hub crate for all HF operations (list, pull, info).
//        Push uses raw reqwest (HF upload REST API — not in hf-hub v0.5).
//        Prune deletes ~/.cache/huggingface/hub/<repo-folder>.
// @DANGER: HF_TOKEN is NEVER written to config.json in plain text.
//          Token resolution order: OS keyring → HF_TOKEN env var → hf-hub cache file.

use anyhow::{bail, Context, Result};
use hf_hub::api::tokio::{Api, ApiBuilder};
use hf_hub::{Cache, Repo, RepoType};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

// ── exit codes ───────────────────────────────────────────────────────────────
pub const EXIT_OK: i32 = 0;
pub const EXIT_ERROR: i32 = 1;
pub const EXIT_MODEL_NOT_FOUND: i32 = 2;
pub const EXIT_CONNECTION_FAILED: i32 = 3;
pub const EXIT_AUTH_FAILED: i32 = 4;

// ── result types ─────────────────────────────────────────────────────────────

pub type HubResult<T> = Result<T>;

#[derive(Debug, Serialize)]
pub struct ModelFileEntry {
    pub filename: String,
    pub size_bytes: Option<u64>,
}

#[derive(Debug, Serialize)]
pub struct ModelInfo {
    pub model_id: String,
    pub sha: String,
    pub files: Vec<ModelFileEntry>,
    pub private: bool,
}

#[derive(Debug, Serialize)]
pub struct PullResult {
    pub model_id: String,
    pub files_downloaded: usize,
    pub cache_path: PathBuf,
}

#[derive(Debug, Serialize)]
pub struct PushResult {
    pub model_id: String,
    pub files_uploaded: usize,
    pub repo_url: String,
}

#[derive(Debug, Serialize)]
pub struct PruneResult {
    pub model_id: String,
    pub bytes_freed: u64,
    pub cache_dir: PathBuf,
}

#[derive(Debug, Serialize)]
pub struct ListEntry {
    pub model_id: String,
    pub sha: String,
    pub file_count: usize,
}

// ── token resolution ──────────────────────────────────────────────────────────

const KEYRING_SERVICE: &str = "gwenland";
const KEYRING_USER: &str = "hf_token";

/// Resolve HF token: keyring → HF_TOKEN env var → hf-hub cache file.
/// Returns None if no token is available (public access only).
pub fn resolve_token() -> Option<String> {
    // 1. OS keyring
    if let Ok(entry) = keyring::Entry::new(KEYRING_SERVICE, KEYRING_USER) {
        if let Ok(token) = entry.get_password() {
            if !token.trim().is_empty() {
                return Some(token.trim().to_string());
            }
        }
    }

    // 2. Environment variable
    if let Ok(tok) = std::env::var("HF_TOKEN") {
        if !tok.trim().is_empty() {
            return Some(tok.trim().to_string());
        }
    }

    // 3. hf-hub cache file (~/.cache/huggingface/token)
    Cache::from_env().token()
}

/// Store HF token in OS keyring. Never writes to config.json.
pub fn store_token(token: &str) -> Result<()> {
    let entry = keyring::Entry::new(KEYRING_SERVICE, KEYRING_USER)
        .context("cannot open OS keyring")?;
    entry
        .set_password(token)
        .context("cannot write HF token to OS keyring")?;
    Ok(())
}

/// Delete HF token from OS keyring.
pub fn delete_token() -> Result<()> {
    let entry = keyring::Entry::new(KEYRING_SERVICE, KEYRING_USER)
        .context("cannot open OS keyring")?;
    entry
        .delete_credential()
        .context("cannot delete HF token from OS keyring")?;
    Ok(())
}

// ── API builder helper ────────────────────────────────────────────────────────

fn build_api(show_progress: bool) -> Result<Api> {
    let token = resolve_token();
    ApiBuilder::from_env()
        .with_token(token)
        .with_progress(show_progress)
        .build()
        .context("failed to build HF Hub API client")
}

// ── HF Hub model info (via REST API — returns richer metadata than hf-hub) ────

#[derive(Debug, Deserialize)]
struct HfApiModelInfo {
    #[serde(rename = "modelId", default)]
    model_id: String,
    #[serde(default)]
    id: String,
    #[serde(default)]
    sha: String,
    #[serde(rename = "private", default)]
    private: bool,
    #[serde(default)]
    siblings: Vec<HfApiSibling>,
}

#[derive(Debug, Deserialize)]
struct HfApiSibling {
    rfilename: String,
    size: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct HfApiListEntry {
    #[serde(rename = "modelId", default)]
    model_id: String,
    #[serde(default)]
    id: String,
    #[serde(default)]
    sha: String,
    #[serde(default)]
    siblings: Vec<HfApiSibling>,
}

// ── subcommand: list ──────────────────────────────────────────────────────────

/// List models from HF Hub. If `author` is given, filters by owner/organisation.
pub async fn hub_list(
    author: Option<&str>,
    search: Option<&str>,
    limit: usize,
) -> HubResult<Vec<ListEntry>> {
    let token = resolve_token();
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(15))
        .build()
        .context("failed to build HTTP client")?;

    let hf_endpoint = std::env::var("HF_ENDPOINT")
        .unwrap_or_else(|_| "https://huggingface.co".to_string());
    let mut url = format!("{}/api/models?limit={}", hf_endpoint, limit.min(100));

    if let Some(a) = author {
        url.push_str(&format!("&author={}", urlencoding_simple(a)));
    }
    if let Some(s) = search {
        url.push_str(&format!("&search={}", urlencoding_simple(s)));
    }

    let mut req = client.get(&url);
    if let Some(tok) = &token {
        req = req.bearer_auth(tok);
    }

    let resp = req
        .send()
        .await
        .context("connection to HF Hub failed")?;

    if resp.status() == reqwest::StatusCode::UNAUTHORIZED {
        bail!("HF Hub returned 401 — check your HF_TOKEN (auth failed)");
    }

    let entries: Vec<HfApiListEntry> = resp
        .error_for_status()
        .context("HF Hub list request failed")?
        .json()
        .await
        .context("failed to parse HF Hub list response")?;

    Ok(entries
        .into_iter()
        .map(|e| ListEntry {
            model_id: if e.model_id.is_empty() { e.id } else { e.model_id },
            sha: e.sha,
            file_count: e.siblings.len(),
        })
        .collect())
}

// ── subcommand: info ──────────────────────────────────────────────────────────

/// Fetch detailed metadata for a specific model from HF Hub.
pub async fn hub_info(model_id: &str) -> HubResult<ModelInfo> {
    let token = resolve_token();
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(15))
        .build()
        .context("failed to build HTTP client")?;

    let hf_endpoint = std::env::var("HF_ENDPOINT")
        .unwrap_or_else(|_| "https://huggingface.co".to_string());
    let url = format!("{}/api/models/{}", hf_endpoint, model_id);

    let mut req = client.get(&url);
    if let Some(tok) = &token {
        req = req.bearer_auth(tok);
    }

    let resp = req.send().await.context("connection to HF Hub failed")?;

    match resp.status() {
        reqwest::StatusCode::NOT_FOUND => {
            bail!("model '{}' not found on HF Hub", model_id);
        }
        reqwest::StatusCode::UNAUTHORIZED => {
            bail!("HF Hub returned 401 — token required or token invalid");
        }
        s if !s.is_success() => {
            bail!("HF Hub returned {} for model '{}'", s, model_id);
        }
        _ => {}
    }

    let info: HfApiModelInfo = resp
        .json()
        .await
        .context("failed to parse HF Hub model info response")?;

    let effective_id = if info.model_id.is_empty() { info.id } else { info.model_id };

    Ok(ModelInfo {
        model_id: effective_id,
        sha: info.sha,
        private: info.private,
        files: info
            .siblings
            .into_iter()
            .map(|s| ModelFileEntry {
                filename: s.rfilename,
                size_bytes: s.size,
            })
            .collect(),
    })
}

// ── subcommand: pull ──────────────────────────────────────────────────────────

/// Download a model from HF Hub into the local cache.
/// Shows an indicatif progress bar during download.
pub async fn hub_pull(
    model_id: &str,
    revision: Option<&str>,
    show_progress: bool,
) -> HubResult<PullResult> {
    let api = build_api(show_progress)?;

    let repo = match revision {
        Some(rev) => Repo::with_revision(model_id.to_string(), RepoType::Model, rev.to_string()),
        None => Repo::model(model_id.to_string()),
    };
    let api_repo = api.repo(repo.clone());

    // Fetch file list
    let repo_info = api_repo
        .info()
        .await
        .map_err(|e| anyhow::anyhow!("failed to fetch model info for '{}': {}", model_id, e))?;

    if repo_info.siblings.is_empty() {
        bail!("model '{}' has no files to download", model_id);
    }

    let filenames: Vec<String> = repo_info
        .siblings
        .iter()
        .map(|s| s.rfilename.clone())
        .collect();

    // Progress is controlled via the ApiBuilder::with_progress() flag set in build_api().
    // hf-hub's internal ProgressBar (indicatif 0.18) is used when progress: true.
    let mut downloaded = 0usize;
    for filename in &filenames {
        api_repo.download(filename).await.map_err(|e| {
            anyhow::anyhow!("failed to download '{}' from '{}': {}", filename, model_id, e)
        })?;
        downloaded += 1;
    }

    let cache = Cache::from_env();
    let cache_repo = cache.repo(Repo::model(model_id.to_string()));
    let cache_path = cache_repo.pointer_path(&repo_info.sha);

    Ok(PullResult {
        model_id: model_id.to_string(),
        files_downloaded: downloaded,
        cache_path,
    })
}

// ── subcommand: push ──────────────────────────────────────────────────────────

/// Upload a local file or directory to a HF Hub repo.
/// Uses the HF HTTP upload API directly (not in hf-hub v0.5).
pub async fn hub_push(
    model_id: &str,
    local_path: &Path,
    commit_message: &str,
) -> HubResult<PushResult> {
    let token = resolve_token().context(
        "HF token required for push. Set it via keyring or HF_TOKEN env var.",
    )?;

    if token.is_empty() {
        bail!("HF token is empty — cannot push. Set HF_TOKEN or run `gwen hub model login`.");
    }

    let hf_endpoint = std::env::var("HF_ENDPOINT")
        .unwrap_or_else(|_| "https://huggingface.co".to_string());

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()
        .context("failed to build HTTP client")?;

    // Collect files to upload
    let files = collect_files(local_path)?;
    if files.is_empty() {
        bail!("no files found at '{}'", local_path.display());
    }

    let mut uploaded = 0usize;

    for (abs_path, rel_name) in &files {
        let content = tokio::fs::read(abs_path)
            .await
            .with_context(|| format!("cannot read file '{}'", abs_path.display()))?;

        let upload_url = format!(
            "{}/api/models/{}/upload/main/{}",
            hf_endpoint, model_id, rel_name
        );

        let resp = client
            .post(&upload_url)
            .bearer_auth(&token)
            .header("Content-Type", "application/octet-stream")
            .header("X-Commit-Message", commit_message)
            .body(content)
            .send()
            .await
            .with_context(|| format!("failed to upload '{}'", rel_name))?;

        match resp.status() {
            reqwest::StatusCode::UNAUTHORIZED => {
                bail!("push rejected: invalid or missing HF token (401)");
            }
            reqwest::StatusCode::FORBIDDEN => {
                bail!("push rejected: you don't have write access to '{}' (403)", model_id);
            }
            s if !s.is_success() => {
                let body = resp.text().await.unwrap_or_default();
                bail!("push failed for '{}': HTTP {} — {}", rel_name, s, body);
            }
            _ => {}
        }
        uploaded += 1;
    }

    Ok(PushResult {
        model_id: model_id.to_string(),
        files_uploaded: uploaded,
        repo_url: format!("{}/{}", hf_endpoint, model_id),
    })
}

// ── subcommand: prune ─────────────────────────────────────────────────────────

/// Delete the HF cache directory for a specific model.
/// Returns the number of bytes freed.
pub fn hub_prune(model_id: &str) -> HubResult<PruneResult> {
    let cache = Cache::from_env();
    let repo = Repo::model(model_id.to_string());
    let cache_repo = cache.repo(repo);

    // Resolve the repo folder: ~/.cache/huggingface/hub/models--owner--name/
    let folder_name = Repo::model(model_id.to_string()).folder_name();
    let mut cache_dir = cache.path().clone();
    cache_dir.push(&folder_name);

    if !cache_dir.exists() {
        bail!(
            "no cached files found for model '{}' at '{}'",
            model_id,
            cache_dir.display()
        );
    }

    let bytes_freed = dir_size(&cache_dir)?;
    std::fs::remove_dir_all(&cache_dir)
        .with_context(|| format!("failed to remove cache dir '{}'", cache_dir.display()))?;

    // Suppress unused variable warning — cache_repo is for documentation intent
    let _ = cache_repo;

    Ok(PruneResult {
        model_id: model_id.to_string(),
        bytes_freed,
        cache_dir,
    })
}

// ── internal helpers ──────────────────────────────────────────────────────────

/// Recursively collect (absolute_path, relative_name) pairs from a path.
/// If path is a file, returns just that one file.
fn collect_files(root: &Path) -> Result<Vec<(PathBuf, String)>> {
    let mut out = Vec::new();
    if root.is_file() {
        let name = root
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("file")
            .to_string();
        out.push((root.to_path_buf(), name));
    } else {
        for entry in walkdir::WalkDir::new(root)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
        {
            let abs = entry.path().to_path_buf();
            let rel = abs
                .strip_prefix(root)
                .unwrap_or(&abs)
                .to_string_lossy()
                .replace('\\', "/");
            out.push((abs, rel));
        }
    }
    Ok(out)
}

/// Walk a directory and sum all file sizes.
fn dir_size(path: &Path) -> Result<u64> {
    let mut total = 0u64;
    for entry in walkdir::WalkDir::new(path)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
    {
        total += entry.metadata().map(|m| m.len()).unwrap_or(0);
    }
    Ok(total)
}

/// Minimal percent-encoding for URL query values (encodes space, &, =, +).
fn urlencoding_simple(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            'A'..='Z' | 'a'..='z' | '0'..='9' | '-' | '_' | '.' | '~' => c.to_string(),
            ' ' => "%20".to_string(),
            '&' => "%26".to_string(),
            '=' => "%3D".to_string(),
            '+' => "%2B".to_string(),
            '/' => "%2F".to_string(),
            _ => format!("%{:02X}", c as u32),
        })
        .collect()
}

// ── dry-run check ─────────────────────────────────────────────────────────────

/// Read-only pre-flight for `gwen hub pull --dry-run`.
/// Checks: token present, model already cached, disk space at cache dir.
/// Never downloads anything or makes network calls.
pub fn dry_run_hub_pull(model_id: &str) -> crate::dry_run::DryRunReport {
    use crate::dry_run::{DryRunLine, DryRunReport};

    let mut report = DryRunReport::new("hub pull");

    // 1. Token availability (read-only keyring probe)
    if resolve_token().is_some() {
        report.push(DryRunLine::ok("token", "found (keyring or HF_TOKEN)"));
    } else {
        report.push(DryRunLine::info("token", "not set — public access only"));
    }

    // 2. Already cached locally?
    let cache = Cache::from_env();
    let folder_name = Repo::model(model_id.to_string()).folder_name();
    let cache_root = cache.path().join(&folder_name);
    if cache_root.exists() {
        report.push(DryRunLine::ok("cached", format!("yes — {}", cache_root.display())));
    } else {
        report.push(DryRunLine::info("cached", "no — will download on pull"));
    }

    // 3. Disk space at HF cache root (read-only)
    let hf_cache_dir = cache.path().to_path_buf();
    if let Some((_, avail)) = crate::platform::hardware::check_disk_space(&hf_cache_dir) {
        let avail_gb = avail as f64 / (1024.0 * 1024.0 * 1024.0);
        if avail_gb >= 1.0 {
            report.push(DryRunLine::ok("disk", format!("{:.1} GB free at cache", avail_gb)));
        } else {
            report.push(DryRunLine::fail("disk", format!("{:.1} GB free — may be insufficient", avail_gb)));
        }
        report.set("disk_free_gb", avail_gb);
    } else {
        report.push(DryRunLine::info("disk", "unknown (platform query unsupported)"));
    }

    report
}
