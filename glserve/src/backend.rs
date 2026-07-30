//! `InferenceBackend` — the seam between HTTP handlers and generation
//! (ARTX16 §1.1/§1.6, scheduler-less v1).
//!
//! ⛔ **Scope note.** ARTX16 §1.1's real design has axum handlers talk to a
//! `SessionWorker` that is itself a loop around ARTX7's continuous-batching
//! scheduler (`collect_new_requests` / `form_batch` / `schedule_decode` /
//! ...). **ARTX7 does not exist in this codebase** — no `KvSlotManager`, no
//! multi-request batching, nothing to loop around. What ARTX16 §1.1 *does*
//! specify independent of ARTX7 — "the async/sync boundary is a thread, not
//! an async runtime inside the engine" — is kept, and turns out to be load-
//! bearing for a reason ARTX16 doesn't call out: `gljax::runtime::CachedSession`
//! holds `Rc<PjrtClientHandle>` → `Rc<PjrtPlugin>` — `Rc`, not `Arc` — so it
//! is `!Send` *by construction*, not by omission. A `CachedSession` cannot be
//! built on one thread and handed to another; it must be **built inside its
//! own dedicated worker thread** and never leave it. [`GljaxBackend::spawn`]
//! does exactly that.

use std::path::PathBuf;

use tokio::sync::mpsc;

use crate::types::{FinishReason, GenerateRequest, TokenEvent};

pub trait InferenceBackend: Send + Sync {
    fn model_id(&self) -> &str;

    /// Hands `req` to the backend and returns immediately; the caller drives
    /// the returned channel (typically wrapped into an SSE stream by
    /// `stream::stream_chat`).
    fn generate(&self, req: GenerateRequest) -> mpsc::Receiver<TokenEvent>;
}

struct WorkerMsg {
    req: GenerateRequest,
    reply: mpsc::Sender<TokenEvent>,
}

/// One dedicated OS thread owning the entire PJRT/session stack for its
/// whole lifetime. ARTX16's "single-model constraint" for v1 (no hot-swap,
/// no multi-model) means exactly one of these exists per server process.
///
/// ⛔ **Never run against a real PJRT plugin in this environment** (no PJRT
/// plugin on Windows, per `gljax/README.md` — the same limitation every
/// PJRT-dependent piece of this sprint has carried). `GljaxBackend::spawn`
/// itself is exercised by `tests::spawn_reports_a_missing_plugin_path_as_an_error`
/// (the one PJRT-independent path through it — a bad plugin path always
/// fails, with or without a real plugin available); the actual generation
/// path is unverified beyond compiling. The HTTP layer around this backend
/// *is* genuinely tested — see `api::chat::tests` and `api::models::tests`,
/// which use [`FakeBackend`] instead.
///
/// ⚠️ **Not truly token-incremental.** `CachedSession::generate_text` is a
/// single blocking call returning the complete generated string — it has no
/// per-token callback or iterator to drive real incremental SSE deltas from.
/// Adding one would mean either duplicating `CachedSession::generate`'s
/// prefill/decode loop (a second copy to keep in sync with the
/// Gate-A5-verified original) or adding a callback parameter to the existing
/// verified method (a change to the one path this sprint has repeatedly
/// declined to modify without a CI run to check it against — see
/// `gljax::arch`'s and `gljax::tok`'s own module docs for the same call
/// elsewhere this sprint). So: this backend generates the full response
/// synchronously on its worker thread, then emits it as a single
/// `TokenEvent::Token`, followed by `Done`. That is SSE-*protocol*-correct
/// (real framing, real `[DONE]` terminator) but not incrementally-generated-
/// token-correct. Real streaming needs `CachedSession` to grow a per-token
/// hook — real, scoped follow-up work, not done here.
pub struct GljaxBackend {
    model_id: String,
    inbox: std::sync::mpsc::Sender<WorkerMsg>,
}

impl GljaxBackend {
    /// Spawns the worker thread and blocks (the *caller's* thread — this is
    /// meant to be called once, from `main`, before serving starts) until
    /// the plugin has loaded and the model has been traced, compiled, and
    /// its weights uploaded, or until that fails.
    pub fn spawn(
        model_id: String,
        plugin_path: PathBuf,
        model_dir: PathBuf,
        window: usize,
    ) -> Result<Self, gljax::GlError> {
        let (tx, rx) = std::sync::mpsc::channel::<WorkerMsg>();
        let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<(), String>>();

        std::thread::spawn(move || {
            let plugin = match gljax::pjrt::PjrtPlugin::load(&plugin_path) {
                Ok(p) => std::rc::Rc::new(p),
                Err(e) => {
                    let _ = ready_tx.send(Err(e.to_string()));
                    return;
                }
            };
            let mut session =
                match gljax::runtime::CachedSession::from_hf_dir(plugin, &model_dir, window) {
                    Ok(s) => s,
                    Err(e) => {
                        let _ = ready_tx.send(Err(e.to_string()));
                        return;
                    }
                };
            if ready_tx.send(Ok(())).is_err() {
                return; // caller gave up waiting — nothing left to serve.
            }

            // The worker loop: one request at a time, for as long as the
            // inbox stays open. This is the "Mutex" ARTX7's slot scheduler
            // would otherwise be — serialized, not batched, and honest about
            // it rather than pretending to batch with no scheduler to do so.
            while let Ok(msg) = rx.recv() {
                let result = session.generate_text(&msg.req.prompt, msg.req.max_tokens);
                match result {
                    Ok(text) => {
                        let _ = msg.reply.blocking_send(TokenEvent::Token { text });
                        let _ =
                            msg.reply.blocking_send(TokenEvent::Done { finish_reason: FinishReason::Stop });
                    }
                    Err(e) => {
                        let _ = msg.reply.blocking_send(TokenEvent::Error { message: e.to_string() });
                    }
                }
            }
        });

        match ready_rx.recv() {
            Ok(Ok(())) => Ok(GljaxBackend { model_id, inbox: tx }),
            Ok(Err(message)) => Err(gljax::GlError::Engine(message)),
            Err(_) => Err(gljax::GlError::Engine(
                "inference worker thread panicked during startup".to_string(),
            )),
        }
    }
}

impl InferenceBackend for GljaxBackend {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn generate(&self, req: GenerateRequest) -> mpsc::Receiver<TokenEvent> {
        let (reply_tx, reply_rx) = mpsc::channel(8);
        if self.inbox.send(WorkerMsg { req, reply: reply_tx.clone() }).is_err() {
            // The worker thread has exited (a prior panic) — report an error
            // on this request's own channel rather than hanging the caller.
            let _ = reply_tx.try_send(TokenEvent::Error {
                message: "inference worker thread is not running".to_string(),
            });
        }
        reply_rx
    }
}

/// A deterministic, PJRT-free backend for tests and local API exploration
/// without a model loaded. Splits a canned or echoed response into
/// word-sized chunks and streams them with no delay — genuinely exercises
/// the SSE framing, JSON serialization, and multi-chunk delta logic that
/// [`GljaxBackend`] cannot be tested against in this environment.
pub struct FakeBackend {
    model_id: String,
}

impl FakeBackend {
    pub fn new(model_id: impl Into<String>) -> Self {
        FakeBackend { model_id: model_id.into() }
    }
}

impl InferenceBackend for FakeBackend {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn generate(&self, req: GenerateRequest) -> mpsc::Receiver<TokenEvent> {
        let (tx, rx) = mpsc::channel(16);
        let words: Vec<String> = format!("Echo: {}", req.prompt)
            .split(' ')
            .map(|w| format!("{w} "))
            .collect();
        let max_tokens = req.max_tokens;
        let truncated = words.len() > max_tokens;
        tokio::spawn(async move {
            for word in words.into_iter().take(max_tokens) {
                if tx.send(TokenEvent::Token { text: word }).await.is_err() {
                    // Receiver dropped — client disconnected (ARTX16 §1.5's
                    // "client disconnect cancels the request"). Nothing more
                    // to clean up here since there is no ARTX7 slot to free.
                    return;
                }
            }
            let finish_reason = if truncated { FinishReason::Length } else { FinishReason::Stop };
            let _ = tx.send(TokenEvent::Done { finish_reason }).await;
        });
        rx
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn fake_backend_streams_tokens_then_done() {
        let backend = FakeBackend::new("test-model");
        let mut rx = backend.generate(GenerateRequest { prompt: "hi there".into(), max_tokens: 100 });

        let mut texts = Vec::new();
        let mut saw_done = false;
        while let Some(ev) = rx.recv().await {
            match ev {
                TokenEvent::Token { text } => texts.push(text),
                TokenEvent::Done { finish_reason } => {
                    assert_eq!(finish_reason, FinishReason::Stop);
                    saw_done = true;
                }
                TokenEvent::Error { message } => panic!("unexpected error: {message}"),
            }
        }
        assert!(saw_done, "must emit Done");
        assert_eq!(texts.concat().trim(), "Echo: hi there");
    }

    #[tokio::test]
    async fn fake_backend_respects_max_tokens() {
        let backend = FakeBackend::new("test-model");
        let mut rx = backend.generate(GenerateRequest { prompt: "one two three four five".into(), max_tokens: 2 });

        let mut token_count = 0;
        let mut finish = None;
        while let Some(ev) = rx.recv().await {
            match ev {
                TokenEvent::Token { .. } => token_count += 1,
                TokenEvent::Done { finish_reason } => finish = Some(finish_reason),
                TokenEvent::Error { message } => panic!("{message}"),
            }
        }
        assert_eq!(token_count, 2);
        assert_eq!(finish, Some(FinishReason::Length));
    }

    #[tokio::test]
    async fn fake_backend_stops_cleanly_when_the_receiver_is_dropped() {
        let backend = FakeBackend::new("test-model");
        let rx = backend.generate(GenerateRequest {
            prompt: "a b c d e f g h i j k l m n o p".into(),
            max_tokens: 100,
        });
        drop(rx); // simulate client disconnect
        // No panic, no hang — the spawned task must observe the closed
        // channel and return instead of blocking forever on `tx.send`.
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }

    /// The one `GljaxBackend` path testable without a real PJRT plugin: a
    /// nonexistent plugin path must fail fast during `spawn`, not hang or
    /// panic the caller.
    #[test]
    fn spawn_reports_a_missing_plugin_path_as_an_error() {
        let result = GljaxBackend::spawn(
            "test-model".to_string(),
            PathBuf::from("this-path-definitely-does-not-exist.so"),
            PathBuf::from("."),
            128,
        );
        assert!(result.is_err(), "a nonexistent plugin path must fail spawn(), not hang");
    }
}
