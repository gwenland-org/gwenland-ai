// @INFO: HuggingFace Hub dataset operations for `gwen hub dataset`.
// @INFO: Uses hf-hub crate for pull. list/info/push use HF REST API directly.
//        Dataset cache lives in ~/.cache/huggingface/hub/ (same as models, keyed by repo type).
// @DANGER: HF_TOKEN is NEVER written to config.json in plain text.
//          Token resolution order: OS keyring → HF_TOKEN env var → hf-hub cache file.

use anyhow::{bail, Context, Result};
use hf_hub::api::tokio::{Api, ApiBuilder};
use hf_hub::{Cache, Repo, RepoType};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

// ── exit codes (same contract as hub_model) ───────────────────────────────────
pub const EXIT_OK: i32 = 0;
pub const EXIT_ERROR: i32 = 1;
pub const EXIT_DATASET_NOT_FOUND: i32 = 2;
pub const EXIT_CONNECTION_FAILED: i32 = 3;
pub const EXIT_AUTH_FAILED: i32 = 4;

// ── result types ──────────────────────────────────────────────────────────────

pub type HubResult<T> = Result<T>;

#[derive(Debug, Serialize)]
pub struct DatasetFileEntry {
    pub filename: String,
    pub size_bytes: Option<u64>,
}

#[derive(Debug, Serialize)]
pub struct SplitInfo {
    pub name: String,
    pub num_examples: Option<u64>,
}

#[derive(Debug, Serialize)]
pub struct DatasetInfo {
    pub dataset_id: String,
    pub sha: String,
    pub private: bool,
    pub license: Option<String>,
    pub size_bytes: Option<u64>,
    pub num_rows: Option<u64>,
    pub splits: Vec<SplitInfo>,
    pub files: Vec<DatasetFileEntry>,
}

#[derive(Debug, Serialize)]
pub struct PullResult {
    pub dataset_id: String,
    pub files_downloaded: usize,
    pub cache_path: PathBuf,
}

#[derive(Debug, Serialize)]
pub struct PushResult {
    pub dataset_id: String,
    pub files_uploaded: usize,
    pub repo_url: String,
}

#[derive(Debug, Serialize)]
pub struct PruneResult {
    pub dataset_id: String,
    pub bytes_freed: u64,
    pub cache_dir: PathBuf,
}

#[derive(Debug, Serialize)]
pub struct ListEntry {
    pub dataset_id: String,
    pub sha: String,
    pub file_count: usize,
}

// ── token resolution (reuses hub_model keyring service) ───────────────────────

const KEYRING_SERVICE: &str = "gwenland";
const KEYRING_USER: &str = "hf_token";

/// Resolve HF token: keyring → HF_TOKEN env var → hf-hub cache file.
pub fn resolve_token() -> Option<String> {
    if let Ok(entry) = keyring::Entry::new(KEYRING_SERVICE, KEYRING_USER) {
        if let Ok(token) = entry.get_password() {
            if !token.trim().is_empty() {
                return Some(token.trim().to_string());
            }
        }
    }
    if let Ok(tok) = std::env::var("HF_TOKEN") {
        if !tok.trim().is_empty() {
            return Some(tok.trim().to_string());
        }
    }
    Cache::from_env().token()
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

// ── HF REST API deserialization types ─────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct HfApiDatasetInfo {
    #[serde(rename = "id", default)]
    id: String,
    #[serde(default)]
    sha: String,
    #[serde(rename = "private", default)]
    private: bool,
    #[serde(rename = "cardData", default)]
    card_data: Option<HfCardData>,
    #[serde(default)]
    siblings: Vec<HfApiSibling>,
    #[serde(rename = "datasetsServerInfo", default)]
    server_info: Option<HfServerInfo>,
}

#[derive(Debug, Deserialize, Default)]
struct HfCardData {
    license: Option<String>,
    #[serde(rename = "size_categories", default)]
    _size_categories: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct HfServerInfo {
    #[serde(rename = "num_rows", default)]
    num_rows: Option<u64>,
    #[serde(rename = "partial", default)]
    _partial: bool,
    #[serde(default)]
    splits: Vec<HfSplitEntry>,
}

#[derive(Debug, Deserialize)]
struct HfSplitEntry {
    dataset: String,
    config: String,
    split: String,
    #[serde(rename = "num_rows", default)]
    num_rows: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct HfApiSibling {
    rfilename: String,
    size: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct HfApiListEntry {
    #[serde(rename = "id", default)]
    id: String,
    #[serde(default)]
    sha: String,
    #[serde(default)]
    siblings: Vec<HfApiSibling>,
}

// ── subcommand: list ──────────────────────────────────────────────────────────

/// List datasets from HF Hub.
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

    let hf_endpoint = hf_endpoint();
    let mut url = format!("{}/api/datasets?limit={}", hf_endpoint, limit.min(100));

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

    let resp = req.send().await.context("connection to HF Hub failed")?;

    if resp.status() == reqwest::StatusCode::UNAUTHORIZED {
        bail!("HF Hub returned 401 — check your HF_TOKEN (auth failed)");
    }

    let entries: Vec<HfApiListEntry> = resp
        .error_for_status()
        .context("HF Hub dataset list request failed")?
        .json()
        .await
        .context("failed to parse HF Hub dataset list response")?;

    Ok(entries
        .into_iter()
        .map(|e| ListEntry {
            dataset_id: e.id,
            sha: e.sha,
            file_count: e.siblings.len(),
        })
        .collect())
}

// ── subcommand: info ──────────────────────────────────────────────────────────

/// Fetch metadata for a specific dataset from HF Hub.
pub async fn hub_info(dataset_id: &str) -> HubResult<DatasetInfo> {
    let token = resolve_token();
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(15))
        .build()
        .context("failed to build HTTP client")?;

    let hf_endpoint = hf_endpoint();
    let url = format!(
        "{}/api/datasets/{}?full=true",
        hf_endpoint, dataset_id
    );

    let mut req = client.get(&url);
    if let Some(tok) = &token {
        req = req.bearer_auth(tok);
    }

    let resp = req.send().await.context("connection to HF Hub failed")?;

    match resp.status() {
        reqwest::StatusCode::NOT_FOUND => {
            bail!("dataset '{}' not found on HF Hub", dataset_id);
        }
        reqwest::StatusCode::UNAUTHORIZED => {
            bail!("HF Hub returned 401 — token required or token invalid");
        }
        s if !s.is_success() => {
            bail!("HF Hub returned {} for dataset '{}'", s, dataset_id);
        }
        _ => {}
    }

    let raw: HfApiDatasetInfo = resp
        .json()
        .await
        .context("failed to parse HF Hub dataset info response")?;

    let total_bytes: Option<u64> = {
        let sum: u64 = raw.siblings.iter().filter_map(|s| s.size).sum();
        if sum > 0 { Some(sum) } else { None }
    };

    let (num_rows, splits) = match &raw.server_info {
        Some(si) => {
            let rows = si.num_rows;
            let sp = si
                .splits
                .iter()
                .map(|s| SplitInfo {
                    name: format!("{}/{}/{}", s.dataset, s.config, s.split),
                    num_examples: s.num_rows,
                })
                .collect();
            (rows, sp)
        }
        None => (None, vec![]),
    };

    let license = raw.card_data.as_ref().and_then(|c| c.license.clone());

    Ok(DatasetInfo {
        dataset_id: raw.id,
        sha: raw.sha,
        private: raw.private,
        license,
        size_bytes: total_bytes,
        num_rows,
        splits,
        files: raw
            .siblings
            .into_iter()
            .map(|s| DatasetFileEntry {
                filename: s.rfilename,
                size_bytes: s.size,
            })
            .collect(),
    })
}

// ── subcommand: pull ──────────────────────────────────────────────────────────

/// Download a dataset from HF Hub into the local cache.
pub async fn hub_pull(
    dataset_id: &str,
    revision: Option<&str>,
    show_progress: bool,
) -> HubResult<PullResult> {
    let api = build_api(show_progress)?;

    let repo = match revision {
        Some(rev) => Repo::with_revision(
            dataset_id.to_string(),
            RepoType::Dataset,
            rev.to_string(),
        ),
        None => Repo::new(dataset_id.to_string(), RepoType::Dataset),
    };
    let api_repo = api.repo(repo.clone());

    let repo_info = api_repo
        .info()
        .await
        .map_err(|e| anyhow::anyhow!("failed to fetch dataset info for '{}': {}", dataset_id, e))?;

    if repo_info.siblings.is_empty() {
        bail!("dataset '{}' has no files to download", dataset_id);
    }

    let filenames: Vec<String> = repo_info
        .siblings
        .iter()
        .map(|s| s.rfilename.clone())
        .collect();

    let mut downloaded = 0usize;
    for filename in &filenames {
        api_repo.download(filename).await.map_err(|e| {
            anyhow::anyhow!(
                "failed to download '{}' from dataset '{}': {}",
                filename,
                dataset_id,
                e
            )
        })?;
        downloaded += 1;
    }

    let cache = Cache::from_env();
    let cache_repo = cache.repo(Repo::new(dataset_id.to_string(), RepoType::Dataset));
    let cache_path = cache_repo.pointer_path(&repo_info.sha);

    Ok(PullResult {
        dataset_id: dataset_id.to_string(),
        files_downloaded: downloaded,
        cache_path,
    })
}

// ── subcommand: push ──────────────────────────────────────────────────────────

/// Upload a local file or directory to a HF Hub dataset repo.
pub async fn hub_push(
    dataset_id: &str,
    local_path: &Path,
    commit_message: &str,
) -> HubResult<PushResult> {
    let token = resolve_token().context(
        "HF token required for push. Set it via keyring or HF_TOKEN env var.",
    )?;

    if token.is_empty() {
        bail!("HF token is empty — cannot push. Run `gwen hub dataset login`.");
    }

    let hf_endpoint = hf_endpoint();
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()
        .context("failed to build HTTP client")?;

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
            "{}/api/datasets/{}/upload/main/{}",
            hf_endpoint, dataset_id, rel_name
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
                bail!(
                    "push rejected: no write access to dataset '{}' (403)",
                    dataset_id
                );
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
        dataset_id: dataset_id.to_string(),
        files_uploaded: uploaded,
        repo_url: format!("{}/datasets/{}", hf_endpoint, dataset_id),
    })
}

// ── subcommand: prune ─────────────────────────────────────────────────────────

/// Delete the HF cache directory for a specific dataset.
pub fn hub_prune(dataset_id: &str) -> HubResult<PruneResult> {
    let cache = Cache::from_env();
    let repo = Repo::new(dataset_id.to_string(), RepoType::Dataset);
    let folder_name = repo.folder_name();
    let mut cache_dir = cache.path().clone();
    cache_dir.push(&folder_name);

    if !cache_dir.exists() {
        bail!(
            "no cached files found for dataset '{}' at '{}'",
            dataset_id,
            cache_dir.display()
        );
    }

    let bytes_freed = dir_size(&cache_dir)?;
    std::fs::remove_dir_all(&cache_dir)
        .with_context(|| format!("failed to remove cache dir '{}'", cache_dir.display()))?;

    Ok(PruneResult {
        dataset_id: dataset_id.to_string(),
        bytes_freed,
        cache_dir,
    })
}

// ── internal helpers ──────────────────────────────────────────────────────────

fn hf_endpoint() -> String {
    std::env::var("HF_ENDPOINT").unwrap_or_else(|_| "https://huggingface.co".to_string())
}

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
