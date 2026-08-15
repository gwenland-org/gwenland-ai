//! Baseline HTTP surface checks: the endpoints an operator hits first, driven
//! through the real router (`build_router` + `oneshot`) so routing and
//! serialization run for real without binding a socket.

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt as _;
use tower::ServiceExt as _;

use glserve::backend::FakeBackend;
use glserve::{build_router, AppState};

fn app() -> axum::Router {
    build_router(AppState { backend: Arc::new(FakeBackend::new("qwen2-0.5b")) })
}

#[tokio::test]
async fn health_endpoint_returns_200() {
    let response = app()
        .oneshot(Request::builder().uri("/health").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
}

/// A chat request carrying no usable prompt must be refused with a 4xx, not
/// answered with a generation from an empty string.
#[tokio::test]
async fn chat_rejects_a_request_with_no_messages() {
    let body = serde_json::json!({ "model": "qwen2-0.5b", "messages": [] }).to_string();
    let response = app()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert!(
        response.status().is_client_error(),
        "empty messages should be a 4xx, got {}",
        response.status()
    );
}

/// The prompt path still works — so the test above is proving *rejection*, not
/// merely that the endpoint is broken for everything.
#[tokio::test]
async fn chat_accepts_a_normal_request() {
    let body = serde_json::json!({
        "model": "qwen2-0.5b",
        "messages": [{ "role": "user", "content": "hello" }],
        "max_tokens": 2
    })
    .to_string();
    let response = app()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(json["object"], "chat.completion");
}
