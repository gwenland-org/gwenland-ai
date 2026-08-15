//! Router assembly (ARTX16 §8).

pub mod chat;
pub mod models;
pub mod openai;

use axum::routing::{get, post};
use axum::Router;

use crate::AppState;

pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/v1/chat/completions", post(chat::chat_completions))
        .route("/v1/models", get(models::list_models))
}
