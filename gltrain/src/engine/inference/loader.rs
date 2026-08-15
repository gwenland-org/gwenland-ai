// engine/inference/loader.rs — GGUF weight + tokenizer loader.
//
// Resolves a model name to a local GGUF file, picks the best available
// compute device (CUDA → Metal → CPU), loads the tokenizer from HF Hub
// (cached after first download), and returns everything the runner needs.
//
// Device priority:
//   1. CUDA   — requires NVIDIA GPU + CUDA toolkit at compile time
//   2. Metal  — macOS Apple-Silicon / Metal-capable GPU
//   3. CPU    — always available, always correct
//
// Why not auto-download missing models?
// Downloading multi-GB files silently would surprise users. `gwen run`
// prints a clear "not found" message and points to `gwen fetch`.

use anyhow::{Context, Result, bail};
use candle_core::Device;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokenizers::Tokenizer;

use crate::engine::inference::model_dispatch::{detect_from_gguf, ModelKind};

/// Everything the inference runner needs, resolved before generation starts.
pub struct LoadedModel {
    pub device: Device,
    /// Human-readable device label for startup banner, e.g. "CUDA (RTX 3060)".
    pub device_label: String,
    /// Path to the GGUF file that was loaded.
    pub gguf_path: PathBuf,
    /// Tokenizer, shared from the per-model cache (cheap to clone).
    pub tokenizer: Arc<Tokenizer>,
    /// Architecture, detected once and cached.
    pub kind: ModelKind,
}

/// Resolve a model name or path to a local GGUF file.
///
/// Resolution order:
///   1. A filesystem path — anything that *looks* like a path (explicit `./`,
///      `../`, `/`, `~`, a drive letter, an embedded separator, or a `.gguf`
///      suffix). Bare filenames, relative paths, and Windows paths all count.
///   2. `<models_dir>/<model>.gguf`
///   3. `<models_dir>/<model>-q4_0.gguf`
///   4. Error with an actionable hint (HuggingFace ids are routed to `gwen fetch`).
pub fn resolve_model_path(model: &str) -> Result<PathBuf> {
    // 1. Anything path-shaped: explicit prefixes, an embedded separator, a
    //    Windows drive (`C:`), `~`, or a `.gguf` suffix. This accepts bare
    //    filenames ("model.gguf"), relative paths ("models/x.gguf"), and
    //    absolute Windows/Unix paths — the previous version only matched
    //    `./`, `../`, `/`, and `C:`, so `model.gguf` or `sub/model.gguf` fell
    //    through to the cache lookup and failed confusingly.
    if looks_like_path(model) {
        let p = expand_tilde(model);
        if p.exists() {
            return Ok(p);
        }
        // A HuggingFace repo id (e.g. `Qwen/Qwen3-1.7B`) also contains a `/`, so
        // it lands here. Point the user at `gwen fetch` rather than a bare
        // "does not exist" — you can't serve/run a model that isn't downloaded.
        if is_hf_repo_id(model) {
            bail!(
                "'{}' looks like a HuggingFace repo id, not a local file.\n  \
                 Download it first:\n    gwen fetch -m {}\n  \
                 then `gwen run`/`gwen serve` it by the resulting file name or path.",
                model,
                model
            );
        }
        bail!("Model path '{}' does not exist.", model);
    }

    let models_dir = gwen_models_dir();

    // 2. Exact name
    let exact = models_dir.join(format!("{}.gguf", model));
    if exact.exists() {
        return Ok(exact);
    }

    // 3. q4_0 variant
    let q4 = models_dir.join(format!("{}-q4_0.gguf", model));
    if q4.exists() {
        return Ok(q4);
    }

    bail!(
        "Model '{}' not found.\n  Checked:\n    {}\n    {}\n  Run `gwen fetch -m {}` to download it.",
        model,
        exact.display(),
        q4.display(),
        model
    )
}

/// Heuristic: does this string denote a filesystem path rather than a bare
/// cache name? True for explicit prefixes, embedded separators, a Windows drive
/// letter, a leading `~`, or a `.gguf` suffix.
fn looks_like_path(model: &str) -> bool {
    model.starts_with("./")
        || model.starts_with("../")
        || model.starts_with(".\\")
        || model.starts_with("..\\")
        || model.starts_with('/')
        || model.starts_with('~')
        || model.contains('/')
        || model.contains('\\')
        || (model.len() > 2 && model.as_bytes()[1] == b':') // C:\ or C:/
        || model.to_ascii_lowercase().ends_with(".gguf")
}

/// A HuggingFace repo id is `org/name` (or `org/name/file`) with no `.gguf`
/// suffix — distinguishable from a relative path by the absence of a `.gguf`
/// extension and of OS-specific separators we'd expect in a real local path.
fn is_hf_repo_id(model: &str) -> bool {
    model.contains('/')
        && !model.contains('\\')
        && !model.to_ascii_lowercase().ends_with(".gguf")
        && !model.starts_with('.')
        && !model.starts_with('/')
        && !model.starts_with('~')
}

/// Expand a leading `~/` (or `~\`) to the user's home directory.
fn expand_tilde(model: &str) -> PathBuf {
    if let Some(rest) = model
        .strip_prefix("~/")
        .or_else(|| model.strip_prefix("~\\"))
    {
        if let Some(home) = dirs::home_dir() {
            return home.join(rest);
        }
    }
    PathBuf::from(model)
}

/// `~/.gwenland/models/` - where `gwen fetch` stores GGUF files.
pub fn gwen_models_dir() -> PathBuf {
    crate::storage::paths::GwenPaths::models_dir()
}

/// Select the best available candle `Device` and return a human-readable label.
pub fn select_device() -> (Device, String) {
    // Try CUDA first (compile-time feature check happens inside candle).
    if let Ok(dev) = Device::new_cuda(0) {
        let label = "CUDA (GPU 0)".to_string();
        return (dev, label);
    }

    // Try Metal (Apple Silicon / macOS discrete GPU).
    if let Ok(dev) = Device::new_metal(0) {
        let label = "Metal (GPU 0)".to_string();
        return (dev, label);
    }

    // Always-available fallback.
    (Device::Cpu, "CPU (fallback)".to_string())
}

// ── per-model asset cache ───────────────────────────────────────────────────
//
// The SSE proxy calls `load` on *every* chat request. We cache the immutable,
// safe-to-share assets — the tokenizer and the detected architecture — keyed by
// GGUF path, so they are resolved once instead of per message (and the tokenizer
// is no longer re-fetched over the network every time; see `resolve_tokenizer`).
//
// NOTE: we deliberately do NOT cache the built `ModelWeights`. candle 0.9.2's
// `quantized_llama::ModelWeights` exposes no KV-cache reset, and the proxy
// replays the full conversation history on every request — reusing a model
// would bleed attention state between turns and corrupt output. The weights are
// rebuilt per request (the OS page cache keeps the re-read fast, and no extra
// resident copy is held).
type CachedAssets = (Arc<Tokenizer>, ModelKind);

lazy_static::lazy_static! {
    static ref ASSET_CACHE: Mutex<HashMap<PathBuf, CachedAssets>> = Mutex::new(HashMap::new());
}

/// Resolve a GGUF model's tokenizer + architecture (cached per model path).
///
/// Prints device / file info to stderr on first load so the user sees progress
/// before generation starts (streaming means stdout is reserved for tokens).
pub fn load(model_name: &str, model_id_for_tokenizer: &str) -> Result<LoadedModel> {
    let gguf_path = resolve_model_path(model_name)?;
    let (device, device_label) = select_device();

    // Fast path: reuse the tokenizer + arch resolved on a previous request.
    if let Some((tokenizer, kind)) = ASSET_CACHE.lock().unwrap().get(&gguf_path).cloned() {
        return Ok(LoadedModel { device, device_label, gguf_path, tokenizer, kind });
    }

    let file_size_gb = std::fs::metadata(&gguf_path)
        .map(|m| m.len() as f64 / 1_073_741_824.0)
        .unwrap_or(0.0);
    eprintln!("  ❖ Device: {}", device_label);
    eprintln!(
        "  ❖ Loading: {} ({:.1} GB)",
        gguf_path.file_name().unwrap_or_default().to_string_lossy(),
        file_size_gb
    );

    let tokenizer = Arc::new(
        resolve_tokenizer(&gguf_path, model_id_for_tokenizer)
            .with_context(|| format!("failed to load tokenizer for '{}'", gguf_path.display()))?,
    );
    let kind = detect_from_gguf(&gguf_path)
        .with_context(|| format!("failed to detect architecture for '{}'", gguf_path.display()))?;

    ASSET_CACHE
        .lock()
        .unwrap()
        .insert(gguf_path.clone(), (tokenizer.clone(), kind.clone()));

    Ok(LoadedModel { device, device_label, gguf_path, tokenizer, kind })
}

/// Resolve the tokenizer **locally first**, never blocking `serve` on a doomed
/// network call.
///
/// Order:
///   1. `<stem>_tokenizer.json` or `tokenizer.json` beside the GGUF (fully offline).
///   2. HuggingFace Hub — using a *real* `org/name` repo id (the hint if it is
///      already one, else the model's recorded `source` repo from the registry),
///      and **time-bounded** so a slow/offline Hub can never hang `serve`.
///
/// The old code fetched `tokenizer.json` from HF Hub keyed on the *model name*
/// — often a local path or Ollama blob hash, which is not a repo id. That
/// produced a doomed, minutes-long network stall (the "ghost bug") before the
/// model ever loaded.
fn resolve_tokenizer(gguf_path: &Path, hint_id: &str) -> Result<Tokenizer> {
    // 1. Local sidecar beside the GGUF.
    if let Some(dir) = gguf_path.parent() {
        let stem = gguf_path
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        for cand in [dir.join(format!("{stem}_tokenizer.json")), dir.join("tokenizer.json")] {
            if cand.exists() {
                return Tokenizer::from_file(&cand)
                    .map_err(|e| anyhow::anyhow!("tokenizer load error ({}): {}", cand.display(), e));
            }
        }
    }

    // 2. HuggingFace Hub — only with a genuine repo id, time-bounded.
    if let Some(repo_id) = resolve_tokenizer_repo_id(gguf_path, hint_id) {
        return fetch_tokenizer_from_hf(&repo_id, Duration::from_secs(20));
    }

    bail!(
        "No tokenizer found for '{}'.\n  \
         Place a `tokenizer.json` next to the model file, or (re)fetch it with \
         `gwen fetch -m <org/name>` so GwenLand knows its HuggingFace repo.",
        gguf_path.display()
    )
}

/// Pick a HuggingFace repo id to source the tokenizer from: the hint if it is
/// already an `org/name`, otherwise the model's recorded `source` repo in the
/// local registry (looked up by name or by path). Returns `None` for local-only
/// models (e.g. an Ollama blob) that have no associated HF repo.
fn resolve_tokenizer_repo_id(gguf_path: &Path, hint_id: &str) -> Option<String> {
    if is_hf_repo_id(hint_id) {
        return Some(hint_id.to_string());
    }
    let registry = crate::storage::registry::ModelRegistry::load().ok()?;
    let entry = registry
        .find(hint_id)
        .or_else(|| registry.list().iter().find(|m| m.path == gguf_path))?;
    is_hf_repo_id(&entry.source).then(|| entry.source.clone())
}

/// Fetch `tokenizer.json` from HF Hub on a worker thread with a hard timeout, so
/// a slow or unreachable Hub can never stall `serve` indefinitely.
fn fetch_tokenizer_from_hf(model_id: &str, timeout: Duration) -> Result<Tokenizer> {
    use std::sync::mpsc;

    let id = model_id.to_string();
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(fetch_tokenizer_json(&id));
    });

    match rx.recv_timeout(timeout) {
        Ok(Ok(path)) => Tokenizer::from_file(&path)
            .map_err(|e| anyhow::anyhow!("tokenizer load error: {}", e)),
        Ok(Err(e)) => Err(e),
        Err(_) => bail!(
            "HuggingFace tokenizer fetch for '{}' timed out after {}s — \
             place a tokenizer.json next to the model instead.",
            model_id,
            timeout.as_secs()
        ),
    }
}

/// Blocking HF Hub fetch of `tokenizer.json` (cached on disk by hf-hub after the
/// first successful download). Runs on a worker thread; see `fetch_tokenizer_from_hf`.
fn fetch_tokenizer_json(model_id: &str) -> Result<PathBuf> {
    use crate::platform::hub_model::resolve_token;
    use hf_hub::api::sync::ApiBuilder;

    let token = resolve_token();
    let api = ApiBuilder::from_env()
        .with_token(token)
        .build()
        .context("failed to build HF Hub sync client")?;

    api.model(model_id.to_string())
        .get("tokenizer.json")
        .with_context(|| format!("failed to fetch tokenizer.json for '{}'", model_id))
}

/// Estimate GGUF VRAM requirement from file size.
/// Rule of thumb: GGUF on VRAM ≈ file_bytes * 1.1 (overhead for KV cache).
pub fn estimate_vram_gb(path: &Path) -> f64 {
    std::fs::metadata(path)
        .map(|m| m.len() as f64 / 1_073_741_824.0 * 1.1)
        .unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn looks_like_path_classification() {
        // Path-shaped inputs.
        assert!(looks_like_path("./m.gguf"));
        assert!(looks_like_path("../m.gguf"));
        assert!(looks_like_path("/abs/m.gguf"));
        assert!(looks_like_path("models/m.gguf"));
        assert!(looks_like_path("C:/x/m.gguf"));
        assert!(looks_like_path("C:\\x\\m.gguf"));
        assert!(looks_like_path("~/m.gguf"));
        assert!(looks_like_path("m.gguf")); // bare filename with .gguf
        assert!(looks_like_path("Qwen/Qwen3-1.7B")); // has a separator
        // Bare cache names are NOT paths.
        assert!(!looks_like_path("qwen3-8b"));
        assert!(!looks_like_path("qwen3:8b")); // ':' not at index 1
    }

    #[test]
    fn hf_repo_id_detection() {
        assert!(is_hf_repo_id("Qwen/Qwen3-1.7B"));
        assert!(is_hf_repo_id("tinyllama/TinyLlama-1.1B"));
        assert!(!is_hf_repo_id("models/x.gguf")); // local .gguf
        assert!(!is_hf_repo_id("./sub/x")); // leading '.'
        assert!(!is_hf_repo_id("qwen3-8b")); // no separator
        assert!(!is_hf_repo_id("C:\\x\\y")); // backslash path
    }

    #[test]
    fn resolve_hf_repo_id_routes_to_fetch_hint() {
        let err = resolve_model_path("Qwen/Qwen3-1.7B")
            .unwrap_err()
            .to_string();
        assert!(err.contains("HuggingFace"), "got: {err}");
        assert!(err.contains("gwen fetch"), "got: {err}");
    }

    #[test]
    fn resolve_bare_missing_gguf_reports_path_not_found() {
        let err = resolve_model_path("definitely-not-here.gguf")
            .unwrap_err()
            .to_string();
        assert!(err.contains("does not exist"), "got: {err}");
    }

    #[test]
    fn resolve_existing_gguf_file() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("model.gguf");
        std::fs::write(&p, b"x").unwrap();
        let resolved = resolve_model_path(p.to_str().unwrap()).unwrap();
        assert_eq!(resolved, p);
    }
}
