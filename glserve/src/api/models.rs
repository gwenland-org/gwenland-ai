//! `GET /v1/models` (ARTX16 §1.4). Single-model constraint for v1 — no
//! hot-swap, no multi-model — so this always returns exactly the one loaded
//! model.

use axum::extract::State;
use axum::Json;

use crate::api::openai::{ModelInfo, ModelsResponse};
use crate::AppState;

pub async fn list_models(State(state): State<AppState>) -> Json<ModelsResponse> {
    Json(ModelsResponse {
        object: "list",
        data: vec![ModelInfo {
            id: state.backend.model_id().to_string(),
            object: "model",
            owned_by: "gwenland",
        }],
    })
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
    async fn list_models_returns_exactly_the_one_loaded_model() {
        let state = AppState { backend: Arc::new(FakeBackend::new("qwen2-0.5b")) };
        let app = build_router(state);
        let response = app.oneshot(Request::builder().uri("/v1/models").body(Body::empty()).unwrap()).await.unwrap();
        assert_eq!(response.status(), axum::http::StatusCode::OK);
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["data"].as_array().unwrap().len(), 1);
        assert_eq!(json["data"][0]["id"], "qwen2-0.5b");
        assert_eq!(json["data"][0]["owned_by"], "gwenland");
    }
}
