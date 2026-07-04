// chat.ts — shared data shapes for Chat and Model Manager features.
//
// WHY one file for both features: OllamaModel is used by both useModels
// (shared by Chat + Model Manager), useChat, and ModelPicker. A single
// source of truth avoids diverging type definitions.

export type Role = 'user' | 'assistant'

// A single turn in the conversation.
// `isStreaming` is true only on the most recent assistant message while
// tokens are still arriving over the SSE stream.
export interface Message {
  id: string
  role: Role
  content: string
  isStreaming?: boolean
  createdAt: number
}

// A model reported by the local Ollama /api/tags endpoint.
//
// The fields below are parsed from the Ollama response shape:
//   { name, size, digest, modified_at, details: { quantization_level, parameter_size } }
//
// `isActive` is a derived field set by useModels — it equals
// (model.name === activeModel) so components don't need to carry
// activeModel separately just to compute this.
export interface OllamaModel {
  name: string            // e.g. "qwen3:8b"
  size: number            // bytes
  digest: string          // sha256:... full hash from Ollama
  quantization: string    // "Q4_K_M", "Q4_0", etc. — from details.quantization_level
  paramCount: string      // "8B", "7B" — normalised from details.parameter_size
  contextLength: number   // inferred from model family — see inferContextLength()
  isActive: boolean       // true when this model === activeModel in useModels
  isLocal: boolean        // always true for Ollama-sourced models
  modifiedAt: string      // ISO date string from Ollama modified_at field
  isOnline: boolean       // true if Ollama returned it (i.e. it is reachable)
}

// Top-level state slice owned by useChat.
// Kept flat so components can destructure what they need without
// drilling through nested objects.
export interface ChatState {
  messages: Message[]
  activeModel: string
  isStreaming: boolean
  contextTokens: number
  maxTokens: number
  // Milliseconds from POST sent → first token received.
  // null until the first message is exchanged in this session.
  firstTokenLatencyMs: number | null
}
