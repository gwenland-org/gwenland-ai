//! Engine-facing request/event types (ARTX16 §1.3), simplified for a
//! scheduler-less v1 — see this crate's `lib.rs` docs for why.
//!
//! Deliberately NOT the OpenAI wire types — `api/openai.rs` converts, so this
//! crate's core never has an HTTP concern baked into it (ARTX16's own
//! `api/openai.rs` isolation decision, kept).

/// One request to generate text, independent of HTTP/OpenAI framing.
#[derive(Debug, Clone)]
pub struct GenerateRequest {
    pub prompt: String,
    pub max_tokens: usize,
}

#[derive(Debug, Clone)]
pub enum TokenEvent {
    Token { text: String },
    Done { finish_reason: FinishReason },
    Error { message: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
    Error,
}

impl FinishReason {
    /// OpenAI's wire spelling.
    pub fn as_str(&self) -> &'static str {
        match self {
            FinishReason::Stop => "stop",
            FinishReason::Length => "length",
            FinishReason::Error => "error",
        }
    }
}
