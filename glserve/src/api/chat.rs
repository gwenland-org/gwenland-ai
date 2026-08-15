//! `POST /v1/chat/completions` (ARTX16 §1.2's request lifecycle, §1.4).

use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;

use crate::api::openai::{ChatChoice, ChatCompletionResponse, ChatMessage, ChatRequest, ChunkMeta, ErrorBody, Usage};
use crate::stream::stream_chat;
use crate::types::{FinishReason, GenerateRequest, TokenEvent};
use crate::AppState;

pub async fn chat_completions(State(state): State<AppState>, Json(req): Json<ChatRequest>) -> Response {
    let prompt = match req.prompt() {
        Ok(p) => p.to_string(),
        Err(body) => return (StatusCode::BAD_REQUEST, Json(body)).into_response(),
    };

    let id = format!("chatcmpl-{}", request_id());
    let created = unix_now();
    let meta = ChunkMeta { id: id.clone(), created, model: req.model.clone() };

    let rx = state.backend.generate(GenerateRequest { prompt, max_tokens: req.max_tokens });

    if req.stream {
        stream_chat(rx, meta).into_response()
    } else {
        non_streaming_response(rx, id, created, req.model).await
    }
}

/// Drains the same event channel a streaming request would consume,
/// concatenating deltas into one final JSON response — the whole point of
/// ARTX16 §1.4's "`stream: true` -> SSE; `false` -> single JSON" being the
/// *same* underlying generation path either way, differing only in how the
/// handler packages the output.
async fn non_streaming_response(
    mut rx: tokio::sync::mpsc::Receiver<TokenEvent>,
    id: String,
    created: u64,
    model: String,
) -> Response {
    let mut content = String::new();
    let mut finish_reason = FinishReason::Stop;
    let mut error: Option<String> = None;

    while let Some(ev) = rx.recv().await {
        match ev {
            TokenEvent::Token { text } => content.push_str(&text),
            TokenEvent::Done { finish_reason: fr } => finish_reason = fr,
            TokenEvent::Error { message } => error = Some(message),
        }
    }

    if let Some(message) = error {
        return (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorBody::internal(message))).into_response();
    }

    let response = ChatCompletionResponse {
        id,
        object: "chat.completion",
        created,
        model,
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage { role: "assistant".to_string(), content },
            finish_reason: finish_reason.as_str(),
        }],
        // ⛔ Token counts, not character counts — gljax::tok::Tokenizer
        // (Wave B5) could produce real numbers here (encode the prompt and
        // completion, take each Vec's length); zeros are honest placeholders
        // rather than a plausible-looking wrong number computed some other
        // way (e.g. whitespace-splitting, which would silently disagree with
        // the model's actual tokenization — P4's exact bug class elsewhere
        // in this sprint, not repeated here just because it's convenient).
        usage: Usage { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
    };
    (StatusCode::OK, Json(response)).into_response()
}

fn unix_now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// A request id unique enough for correlating logs/telemetry within one
/// process's lifetime — not a UUID dependency for what is, this wave, a
/// single-model, single-worker server with no distributed tracing to feed.
fn request_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{:016x}", unix_now().wrapping_mul(1_000_003).wrapping_add(n))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use axum::body::Body;
    use axum::http::Request;
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    use crate::backend::FakeBackend;
    use crate::{build_router, AppState};

    fn test_state() -> AppState {
        AppState { backend: Arc::new(FakeBackend::new("test-model")) }
    }

    #[tokio::test]
    async fn non_streaming_chat_completion_returns_the_full_echoed_text() {
        let app = build_router(test_state());
        let body = serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello world"}],
            "stream": false,
        });
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), axum::http::StatusCode::OK);
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["object"], "chat.completion");
        assert_eq!(json["choices"][0]["message"]["content"], "Echo: hello world ");
        assert_eq!(json["choices"][0]["finish_reason"], "stop");
    }

    #[tokio::test]
    async fn streaming_chat_completion_ends_with_the_literal_done_terminator() {
        let app = build_router(test_state());
        let body = serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true,
        });
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), axum::http::StatusCode::OK);
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        let text = String::from_utf8(bytes.to_vec()).unwrap();
        assert!(text.contains("chat.completion.chunk"), "{text}");
        assert!(
            text.trim_end().ends_with("data: [DONE]"),
            "the stream must terminate with the literal [DONE], not JSON:\n{text}"
        );
    }

    #[tokio::test]
    async fn a_request_with_no_user_message_is_rejected_with_400() {
        let app = build_router(test_state());
        let body = serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "system", "content": "be helpful"}],
        });
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), axum::http::StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn max_tokens_is_honored_end_to_end_through_the_http_layer() {
        let app = build_router(test_state());
        let body = serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "one two three four five six"}],
            "max_tokens": 2,
        });
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["choices"][0]["finish_reason"], "length");
        // FakeBackend echoes "Echo: " + prompt, word-chunked with a trailing
        // space per word — 2 tokens means exactly the first two words.
        assert_eq!(json["choices"][0]["message"]["content"], "Echo: one ");
    }
}
