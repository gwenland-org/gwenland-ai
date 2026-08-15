//! SSE construction (ARTX16 §1.5), followed as specified: `data: ` frames
//! carrying `chat.completion.chunk` JSON, terminated by the literal
//! `data: [DONE]` — not JSON, exactly as the OpenAI streaming contract
//! requires.

use std::convert::Infallible;

use axum::response::sse::{Event, KeepAlive, Sse};
use futures_util::stream::{self, Stream, StreamExt};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::api::openai::{ChatChunk, ChunkMeta};
use crate::types::{FinishReason, TokenEvent};

/// Builds the SSE response body for one streaming chat completion.
///
/// ARTX16 §1.5's two design decisions, kept:
/// - **Keep-alive is on, default interval.** A slow prefill can exceed a
///   proxy's idle timeout before the first token; axum's `KeepAlive` emits
///   comment frames OpenAI clients ignore but proxies count as traffic.
/// - **Client disconnect cancels the request.** When this response is
///   dropped, the SSE stream (and the `mpsc::Receiver` it wraps) is dropped
///   with it; `InferenceBackend::generate`'s sender then fails on its next
///   send, and the backend's spawned thread observes that and stops
///   (`backend::FakeBackend`'s own test pins this: `fake_backend_stops_
///   cleanly_when_the_receiver_is_dropped`). There is no ARTX7 slot to free
///   on top of that — see `backend.rs`'s scope note.
pub fn stream_chat(rx: mpsc::Receiver<TokenEvent>, meta: ChunkMeta) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let role_chunk = ChatChunk::role_preamble(&meta);
    let preamble = stream::once(async move { Ok(sse_event(&role_chunk)) });

    let body = ReceiverStream::new(rx).map(move |ev| {
        let chunk = match ev {
            TokenEvent::Token { text } => ChatChunk::delta(&meta, &text),
            TokenEvent::Done { finish_reason } => ChatChunk::finish(&meta, finish_reason),
            TokenEvent::Error { message } => {
                // The OpenAI wire shape has no field for an error message
                // mid-stream (finish_reason is a bare string) — logged here
                // rather than silently dropped.
                log::warn!("stream error: {message}");
                ChatChunk::finish(&meta, FinishReason::Error)
            }
        };
        Ok(sse_event(&chunk))
    });

    let done = stream::once(async { Ok(Event::default().data("[DONE]")) });

    Sse::new(preamble.chain(body).chain(done)).keep_alive(KeepAlive::default())
}

fn sse_event(chunk: &ChatChunk) -> Event {
    Event::default().data(serde_json::to_string(chunk).expect("ChatChunk always serializes"))
}
