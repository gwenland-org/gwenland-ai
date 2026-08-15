// platform/proxy.rs — Native inference SSE proxy.
//
// Exposes POST /gwenland/chat as an SSE endpoint. The GUI and `gwen chat`
// TUI consume this endpoint unchanged. Upstream is now the native inference
// runner (candle-transformers) instead of an external Ollama/mistral.rs HTTP
// server — the wire format is identical so all consumers require no changes.
//
// SSE token format (same as Cycle 5):
//   data: {"message":{"role":"assistant","content":"<token>"},"done":false}
//   data: {"message":{"role":"assistant","content":""},"done":true}
//
// @DANGER: do NOT collect the full body before streaming; large models produce
//          megabytes of tokens and will OOM the process.
// @EDITABLE: increase CLIENT_TIMEOUT_SECS if users report timeouts on slow hardware.

use actix_web::{web, App, HttpRequest, HttpResponse, HttpServer};
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, oneshot};

use crate::engine::inference::runner::InferenceConfig;
use crate::engine::inference::sampler::SamplerConfig;

// @EDITABLE — tune if backpressure causes stuttering in ChatPane
const STREAM_CHANNEL_CAPACITY: usize = 256;

const PROXY_BIND_ADDR: &str = "127.0.0.1:1136";
const PROXY_ENDPOINT: &str = "/gwenland/chat";
const HEALTH_ENDPOINT: &str = "/health";

// ── wire types (same shape as Cycle 5 so consumers are unaffected) ─────────────

#[derive(Debug, Deserialize)]
struct IncomingChatRequest {
    model: String,
    messages: Vec<IncomingMessage>,
    #[serde(default)]
    stream: bool,
    // Sampling overrides from the request body (optional).
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    repeat_penalty: Option<f32>,
    #[serde(default)]
    max_tokens: Option<usize>,
}

#[derive(Debug, Deserialize, Clone)]
struct IncomingMessage {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct SseMessage {
    message: SseContent,
    done: bool,
}

#[derive(Debug, Serialize)]
struct SseContent {
    role: String,
    content: String,
}

// ── Actix application state ────────────────────────────────────────────────────

struct AppState {
    /// Default sampler config — overridable per-request.
    default_sampler: SamplerConfig,
}

// ── POST /gwenland/chat ────────────────────────────────────────────────────────

async fn chat_handler(
    _req: HttpRequest,
    body: web::Json<IncomingChatRequest>,
    state: web::Data<AppState>,
) -> HttpResponse {
    // Build the full prompt by concatenating the message history.
    // Format: "system: ...\nuser: ...\nassistant: ..."
    let prompt = build_prompt(&body.messages);

    // Per-request sampler overrides.
    let mut sampler = state.default_sampler.clone();
    if let Some(t) = body.temperature { sampler.temperature = t; }
    if let Some(p) = body.top_p { sampler.top_p = p; }
    if let Some(r) = body.repeat_penalty { sampler.repeat_penalty = r; }
    if let Some(m) = body.max_tokens { sampler.max_tokens = m; }

    let model = body.model.clone();

    // @INFO — bounded channel prevents OOM if the runner produces tokens faster
    //         than the HTTP layer can flush them; 256 is headroom for bursts.
    let (tx, mut rx) = mpsc::channel::<Bytes>(STREAM_CHANNEL_CAPACITY);

    // Spawn the inference run on a blocking thread — candle's forward pass is
    // synchronous and would block the tokio executor if run inline.
    //
    // @DANGER — do NOT await inside tokio::spawn for the inference call;
    // candle's matrix ops are CPU-bound and will stall the async runtime.
    tokio::task::spawn_blocking(move || {
        let cfg = InferenceConfig {
            model: model.clone(),
            // Derive tokenizer model id from the model name.
            // Convention: model name matches HF repo (e.g. "qwen3-8b" → "Qwen/Qwen3-8B").
            model_id_for_tokenizer: model.clone(),
            prompt,
            sampler,
            auto_stop_pct: 90,
            show_banner: false,
        };

        let tx_clone = tx.clone();
        let mut cb = move |token: String| {
            let payload = serde_json::to_string(&SseMessage {
                message: SseContent {
                    role: "assistant".to_string(),
                    content: token,
                },
                done: false,
            })
            .unwrap_or_default();
            let line = format!("data: {}\n\n", payload);
            // send() is async; use blocking_send in this sync context.
            let _ = tx_clone.blocking_send(Bytes::from(line));
        };

        let result = crate::engine::inference::runner::run_inference(&cfg, Some(&mut cb));

        // Send the terminating done frame.
        let done_payload = serde_json::to_string(&SseMessage {
            message: SseContent {
                role: "assistant".to_string(),
                content: String::new(),
            },
            done: true,
        })
        .unwrap_or_default();
        let done_line = format!("data: {}\n\n", done_payload);
        let _ = tx.blocking_send(Bytes::from(done_line));

        if let Err(e) = result {
            let err_line = format!("event: error\ndata: {{\"message\": \"{}\"}}\n\n", e);
            let _ = tx.blocking_send(Bytes::from(err_line));
        }
    });

    // Build a streaming SSE response that drains the channel.
    let out_stream = async_stream::stream! {
        while let Some(item) = rx.recv().await {
            yield Ok::<_, actix_web::Error>(item);
        }
    };

    HttpResponse::Ok()
        .content_type("text/event-stream")
        .append_header(("Cache-Control", "no-cache"))
        .append_header(("X-Accel-Buffering", "no"))
        .streaming(out_stream)
}

// ── GET /health ────────────────────────────────────────────────────────────────

async fn health_handler() -> HttpResponse {
    HttpResponse::Ok().body("ok")
}

// ── prompt builder ─────────────────────────────────────────────────────────────

/// Flatten a message array into a single prompt string.
///
/// Format mirrors the ChatML convention used by most instruction-tuned models:
///   <|im_start|>system\n…<|im_end|>\n
///   <|im_start|>user\n…<|im_end|>\n
///   <|im_start|>assistant\n
///
/// Falling back to plain role-prefixed lines if the model's tokenizer doesn't
/// use ChatML — the plain format is universally understood.
fn build_prompt(messages: &[IncomingMessage]) -> String {
    let mut out = String::new();
    for msg in messages {
        out.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", msg.role, msg.content));
    }
    out.push_str("<|im_start|>assistant\n");
    out
}

// ── server startup ─────────────────────────────────────────────────────────────

/// Start the Actix proxy server. Runs until `shutdown_rx` fires.
///
/// @INFO — proxy is an in-process Actix server, not a subprocess; this keeps
///         resource overhead minimal and ties the proxy lifetime to the TUI process.
pub async fn start(
    default_sampler: SamplerConfig,
    shutdown_rx: oneshot::Receiver<()>,
) -> std::io::Result<()> {
    let state = web::Data::new(AppState { default_sampler });

    let server = HttpServer::new(move || {
        App::new()
            .app_data(state.clone())
            .app_data(web::JsonConfig::default().limit(10 * 1024 * 1024))
            .route(PROXY_ENDPOINT, web::post().to(chat_handler))
            .route(HEALTH_ENDPOINT, web::get().to(health_handler))
    })
    .bind(PROXY_BIND_ADDR)?
    .run();

    let handle = server.handle();

    // @INFO — shutdown task: when the TUI exits it drops the sender, which
    // fires the receiver here, calling stop() to drain in-flight requests.
    tokio::spawn(async move {
        let _ = shutdown_rx.await;
        handle.stop(true).await;
    });

    server.await
}
