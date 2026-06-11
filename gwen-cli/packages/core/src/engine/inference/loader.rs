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

use anyhow::{bail, Context, Result};
use candle_core::Device;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

/// Everything the inference runner needs, resolved before generation starts.
pub struct LoadedModel {
    pub device: Device,
    /// Human-readable device label for startup banner, e.g. "CUDA (RTX 3060)".
    pub device_label: String,
    /// Path to the GGUF file that was loaded.
    pub gguf_path: PathBuf,
    pub tokenizer: Tokenizer,
}

/// Resolve a model name or path to a local GGUF file.
///
/// Resolution order:
///   1. Exact path — if `model` starts with `./`, `../`, or `/` (or Windows `C:\`)
///   2. `~/.config/gwen/models/<model>.gguf`
///   3. `~/.config/gwen/models/<model>-q4_0.gguf`
///   4. Error with actionable hint
pub fn resolve_model_path(model: &str) -> Result<PathBuf> {
    // 1. Explicit path
    if model.starts_with("./")
        || model.starts_with("../")
        || model.starts_with('/')
        || (model.len() > 2 && model.chars().nth(1) == Some(':'))
    {
        let p = PathBuf::from(model);
        if p.exists() {
            return Ok(p);
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
        "Model '{}' not found.\n  Checked:\n    {}\n    {}\n  Run `gwen fetch {}` to download it.",
        model,
        exact.display(),
        q4.display(),
        model
    )
}

/// `~/.config/gwen/models/` — where `gwen fetch` stores GGUF files.
pub fn gwen_models_dir() -> PathBuf {
    dirs::config_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("gwen")
        .join("models")
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

/// Load a GGUF model and its tokenizer.
///
/// Prints device / file info to stderr so the user sees progress before
/// generation starts (streaming means stdout is reserved for tokens).
pub fn load(model_name: &str, model_id_for_tokenizer: &str) -> Result<LoadedModel> {
    let gguf_path = resolve_model_path(model_name)?;
    let (device, device_label) = select_device();

    let file_size_gb = std::fs::metadata(&gguf_path)
        .map(|m| m.len() as f64 / 1_073_741_824.0)
        .unwrap_or(0.0);

    eprintln!("  ❖ Device: {}", device_label);
    eprintln!(
        "  ❖ Loading: {} ({:.1} GB)",
        gguf_path.file_name().unwrap_or_default().to_string_lossy(),
        file_size_gb
    );

    // Load tokenizer from HF Hub (sync API — safe to call from any thread).
    let tokenizer = load_tokenizer(model_id_for_tokenizer)
        .with_context(|| format!("failed to load tokenizer for '{}'", model_id_for_tokenizer))?;

    Ok(LoadedModel {
        device,
        device_label,
        gguf_path,
        tokenizer,
    })
}

/// Fetch tokenizer.json from the HF Hub cache (downloads once, then cached).
fn load_tokenizer(model_id: &str) -> Result<Tokenizer> {
    use hf_hub::api::sync::ApiBuilder;
    use crate::platform::hub_model::resolve_token;

    let token = resolve_token();
    let api = ApiBuilder::from_env()
        .with_token(token)
        .with_progress(true)
        .build()
        .context("failed to build HF Hub sync client")?;

    let tok_path = api
        .model(model_id.to_string())
        .get("tokenizer.json")
        .with_context(|| format!("failed to fetch tokenizer.json for '{}'", model_id))?;

    Tokenizer::from_file(&tok_path)
        .map_err(|e| anyhow::anyhow!("tokenizer load error: {}", e))
}

/// Estimate GGUF VRAM requirement from file size.
/// Rule of thumb: GGUF on VRAM ≈ file_bytes * 1.1 (overhead for KV cache).
pub fn estimate_vram_gb(path: &Path) -> f64 {
    std::fs::metadata(path)
        .map(|m| m.len() as f64 / 1_073_741_824.0 * 1.1)
        .unwrap_or(0.0)
}
