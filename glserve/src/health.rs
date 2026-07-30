//! `GET /health` (ARTX16 §4.2, simplified). Full liveness/readiness/drain
//! state machine is ARTX7-scheduler territory (queue depth, in-flight
//! request counts) that does not exist in this codebase — this wave's
//! `/health` answers the one question a scheduler-less v1 actually can: is
//! the process up and did it load a model.

use axum::extract::State;
use axum::http::StatusCode;
use axum::Json;
use serde::Serialize;

use crate::AppState;

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: &'static str,
    pub model: String,
}

pub async fn health(State(state): State<AppState>) -> (StatusCode, Json<HealthResponse>) {
    (StatusCode::OK, Json(HealthResponse { status: "ok", model: state.backend.model_id().to_string() }))
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

    #[tokio::test]
    async fn health_returns_200_and_the_loaded_model_id() {
        let state = AppState { backend: Arc::new(FakeBackend::new("qwen2-0.5b")) };
        let app = build_router(state);
        let response = app.oneshot(Request::builder().uri("/health").body(Body::empty()).unwrap()).await.unwrap();
        assert_eq!(response.status(), axum::http::StatusCode::OK);
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["status"], "ok");
        assert_eq!(json["model"], "qwen2-0.5b");
    }
}
