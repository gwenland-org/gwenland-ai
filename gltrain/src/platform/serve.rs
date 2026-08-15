// platform/serve.rs — Native inference server lifecycle.
//
// Cycle 6: gwen serve no longer spawns a mistralrs-server subprocess.
// It starts the in-process Actix proxy (platform::proxy) which routes POST
// /gwenland/chat directly to candle-transformers inference. No external binary
// is required.
//
// @DANGER: config.json last_used_model is written AFTER the server starts.

use anyhow::Result;
use serde::Serialize;
use std::path::PathBuf;
use tokio::sync::oneshot;

use crate::engine::inference::loader::resolve_model_path;
use crate::engine::inference::sampler::SamplerConfig;

// ── exit codes ────────────────────────────────────────────────────────────────
pub const EXIT_OK: i32 = 0;
pub const EXIT_ERROR: i32 = 1;
pub const EXIT_MODEL_NOT_FOUND: i32 = 2;
pub const EXIT_CONNECTION_FAILED: i32 = 3;

// ── config types ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ServeConfig {
    pub model_id: String,
    pub port: u16,
    pub ctx: u32,
}

impl Default for ServeConfig {
    fn default() -> Self {
        Self {
            model_id: String::new(),
            port: 1136,
            ctx: 4096,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct ServeStatus {
    pub status: String,
    pub model: String,
    pub port: u16,
    pub pid: Option<u32>,
}

// ── config.json helpers ───────────────────────────────────────────────────────

pub fn save_last_used_model(model_id: &str) -> Result<()> {
    crate::storage::config::save_last_used_model(model_id)
}

/// Read the last served model id from config. Used when `gwen serve` is invoked
/// without a model (e.g. the GUI auto-start spawns a bare `gwen serve`).
pub fn read_last_used_model() -> Option<String> {
    crate::storage::config::read_last_used_model()
}

// ── model resolution ──────────────────────────────────────────────────────────

/// Resolve a model name to its GGUF path using the same logic as the runner.
pub fn find_model_in_cache(model_id: &str) -> Option<PathBuf> {
    resolve_model_path(model_id).ok()
}

// ── server startup ─────────────────────────────────────────────────────────────

/// Start the native inference proxy.
///
/// Returns a `oneshot::Sender` that the caller uses to signal shutdown.
/// When the sender is dropped (TUI exits) the proxy drains in-flight requests
/// and releases port 1136.
pub async fn start_native_server(
    _config: &ServeConfig,
    shutdown_rx: oneshot::Receiver<()>,
) -> Result<()> {
    let sampler = SamplerConfig::default();
    crate::platform::proxy::start(sampler, shutdown_rx)
        .await
        .map_err(|e| anyhow::anyhow!("proxy server error: {}", e))
}

// ── dry-run check ─────────────────────────────────────────────────────────────

pub fn dry_run_serve(model_id: &str, port: u16) -> crate::dry_run::DryRunReport {
    use crate::dry_run::{DryRunLine, DryRunReport};
    use crate::engine::inference::loader::select_device;

    let mut report = DryRunReport::new("serve");

    // 1. Model file
    match find_model_in_cache(model_id) {
        Some(p) => report.push(DryRunLine::ok("model", p.display().to_string())),
        None     => report.push(DryRunLine::fail(
            "model",
            format!("'{}' not found — run `gwen fetch {}`", model_id, model_id),
        )),
    }

    // 2. Device
    let (_, device_label) = select_device();
    report.push(DryRunLine::ok("device", device_label));

    // 3. Port availability
    let port_free = std::net::TcpListener::bind(("127.0.0.1", port)).is_ok();
    if port_free {
        report.push(DryRunLine::ok("port", format!("{} (available)", port)));
    } else {
        report.push(DryRunLine::fail("port", format!("{} (already in use)", port)));
    }
    report.set("port", port as i64);

    report
}

/// Poll the native proxy's /health endpoint until ready or timeout.
pub async fn wait_for_ready(port: u16, timeout_secs: u64) -> bool {
    let url = format!("http://localhost:{}/health", port);
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build()
        .unwrap_or_else(|_| reqwest::Client::new());

    let deadline = tokio::time::Instant::now()
        + std::time::Duration::from_secs(timeout_secs);

    while tokio::time::Instant::now() < deadline {
        if let Ok(resp) = client.get(&url).send().await {
            if resp.status().is_success() {
                return true;
            }
        }
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    }
    false
}
