//! OpenAI wire types, and nothing else (ARTX16 §8's own design decision).
//!
//! OpenAI's schema changes without warning. Isolating it here means a schema
//! change touches one file; `types.rs` (the engine-facing shape) never moves.
//! `api/chat.rs` is the only place that converts between the two.
//!
//! This wave implements the subset the sprint brief asked for: `model`,
//! `messages` (only the last user turn's content is used as the prompt —
//! there is no chat template engine in this codebase, ARTX13 §3 places one
//! in the serving layer as future work), `stream`, `temperature`,
//! `max_tokens`.

use serde::{Deserialize, Serialize};

use crate::types::FinishReason;

#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
}

fn default_temperature() -> f32 {
    1.0
}

fn default_max_tokens() -> usize {
    128
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatRequest {
    /// The prompt gljax's completion-only engine actually sees: the last
    /// user message's raw content. No chat template is applied — see this
    /// module's top docs.
    pub fn prompt(&self) -> Result<&str, ErrorBody> {
        self.messages
            .iter()
            .rev()
            .find(|m| m.role == "user")
            .map(|m| m.content.as_str())
            .ok_or_else(|| ErrorBody::invalid_request("messages must contain at least one user turn"))
    }
}

#[derive(Debug, Serialize)]
pub struct ErrorBody {
    pub error: ErrorDetail,
}

#[derive(Debug, Serialize)]
pub struct ErrorDetail {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
}

impl ErrorBody {
    pub fn invalid_request(message: impl Into<String>) -> Self {
        ErrorBody { error: ErrorDetail { message: message.into(), error_type: "invalid_request_error".into() } }
    }

    pub fn internal(message: impl Into<String>) -> Self {
        ErrorBody { error: ErrorDetail { message: message.into(), error_type: "internal_error".into() } }
    }
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct ChatChoice {
    pub index: u32,
    pub message: ChatMessage,
    pub finish_reason: &'static str,
}

#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

/// One `data: ` frame's JSON body for a streaming response
/// (`chat.completion.chunk`).
#[derive(Debug, Serialize)]
pub struct ChatChunk {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChunkChoice>,
}

#[derive(Debug, Serialize)]
pub struct ChatChunkChoice {
    pub index: u32,
    pub delta: ChatDelta,
    pub finish_reason: Option<&'static str>,
}

#[derive(Debug, Default, Serialize)]
pub struct ChatDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

/// Shared metadata a streamed request's chunks all repeat — assembled once
/// per request rather than threaded through every chunk constructor call.
#[derive(Debug, Clone)]
pub struct ChunkMeta {
    pub id: String,
    pub created: u64,
    pub model: String,
}

impl ChatChunk {
    pub fn role_preamble(meta: &ChunkMeta) -> Self {
        ChatChunk {
            id: meta.id.clone(),
            object: "chat.completion.chunk",
            created: meta.created,
            model: meta.model.clone(),
            choices: vec![ChatChunkChoice {
                index: 0,
                delta: ChatDelta { role: Some("assistant"), content: None },
                finish_reason: None,
            }],
        }
    }

    pub fn delta(meta: &ChunkMeta, text: &str) -> Self {
        ChatChunk {
            id: meta.id.clone(),
            object: "chat.completion.chunk",
            created: meta.created,
            model: meta.model.clone(),
            choices: vec![ChatChunkChoice {
                index: 0,
                delta: ChatDelta { role: None, content: Some(text.to_string()) },
                finish_reason: None,
            }],
        }
    }

    pub fn finish(meta: &ChunkMeta, reason: FinishReason) -> Self {
        ChatChunk {
            id: meta.id.clone(),
            object: "chat.completion.chunk",
            created: meta.created,
            model: meta.model.clone(),
            choices: vec![ChatChunkChoice {
                index: 0,
                delta: ChatDelta::default(),
                finish_reason: Some(reason.as_str()),
            }],
        }
    }
}

#[derive(Debug, Serialize)]
pub struct ModelsResponse {
    pub object: &'static str,
    pub data: Vec<ModelInfo>,
}

#[derive(Debug, Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: &'static str,
    pub owned_by: &'static str,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_uses_the_last_user_message_not_the_first() {
        let req = ChatRequest {
            model: "m".into(),
            messages: vec![
                ChatMessage { role: "user".into(), content: "first".into() },
                ChatMessage { role: "assistant".into(), content: "reply".into() },
                ChatMessage { role: "user".into(), content: "second".into() },
            ],
            stream: false,
            temperature: 1.0,
            max_tokens: 128,
        };
        assert_eq!(req.prompt().unwrap(), "second");
    }

    #[test]
    fn prompt_refuses_a_request_with_no_user_turn() {
        let req = ChatRequest {
            model: "m".into(),
            messages: vec![ChatMessage { role: "system".into(), content: "you are helpful".into() }],
            stream: false,
            temperature: 1.0,
            max_tokens: 128,
        };
        assert!(req.prompt().is_err());
    }

    #[test]
    fn chat_request_deserializes_with_defaults_when_optional_fields_are_absent() {
        let json = r#"{"model": "qwen2-0.5b", "messages": [{"role": "user", "content": "hi"}]}"#;
        let req: ChatRequest = serde_json::from_str(json).expect("must parse");
        assert!(!req.stream);
        assert_eq!(req.temperature, 1.0);
        assert_eq!(req.max_tokens, 128);
    }

    #[test]
    fn error_body_serializes_with_the_openai_shaped_fields() {
        let body = ErrorBody::invalid_request("messages must not be empty");
        let json = serde_json::to_string(&body).unwrap();
        assert!(json.contains(r#""type":"invalid_request_error""#));
        assert!(json.contains("messages must not be empty"));
    }
}
