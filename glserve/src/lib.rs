//! # glserve — OpenAI-compatible HTTP serving for gljax (ARTX16)
//!
//! ⛔ **Scope note — read before extending this crate.** ARTX16's own
//! "Reality Check" says its full design (multi-replica routing, `SessionPool`,
//! ARTX7-integrated continuous batching, Prometheus metrics, multi-node
//! deployment) targets an engine that, at the time that document was written,
//! did not yet run at all. `gljax::runtime::Session`/`CachedSession` now do
//! run for real (Gate A5, CI runs `30447306245`/`30453269580`) — but
//! **ARTX7 (continuous batching / `KvSlotManager` / the scheduler) still does
//! not exist in this codebase.** This crate is therefore v1 in the sprint
//! brief's sense — one model, one worker, no hot-swap — not ARTX16's full
//! distributed-serving design. See `backend.rs`'s module docs for exactly
//! which ARTX16 decisions are kept anyway (the async/sync thread boundary)
//! and which are simplified away (the scheduler this crate would otherwise
//! loop around).
//!
//! ARTX16's own design decision, kept in full: **glserve is a separate
//! crate, not a module of gljax.** `gljax` has zero HTTP/async dependencies;
//! this crate depends on it, never the reverse.
//!
//! Port **1136** is the established GwenLand convention (the legacy
//! `packages/tui` serve command, the Tauri GUI's SSE endpoint, and
//! `general.default_port` in the config schema all use it) — inherited here,
//! not reinvented.

pub mod api;
pub mod backend;
pub mod health;
pub mod stream;
pub mod types;

use std::sync::Arc;

use axum::routing::get;
use axum::Router;

pub use backend::{FakeBackend, GljaxBackend, InferenceBackend};
pub use types::{FinishReason, GenerateRequest, TokenEvent};

#[derive(Clone)]
pub struct AppState {
    pub backend: Arc<dyn InferenceBackend>,
}

/// Assembles the full router. Split out from `main.rs` so integration tests
/// (and this crate's own handler tests, via `tower::ServiceExt::oneshot`)
/// drive real routing/serialization code without binding a socket.
pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health::health))
        .merge(api::routes())
        .with_state(state)
}
