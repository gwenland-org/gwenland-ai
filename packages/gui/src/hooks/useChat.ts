// useChat.ts — chat state + SSE stream logic.
//
// WHY fetch + ReadableStream instead of native EventSource:
//   EventSource only supports GET requests. GwenLand's /chat endpoint
//   requires POST with a JSON body. fetch gives full control over the
//   request method while still allowing incremental response reading.
//
// WHY AbortController instead of reader.cancel():
//   reader.cancel() only aborts the *read loop* — the underlying TCP
//   connection stays open until the server closes it. AbortController
//   signals the browser to tear down the fetch itself, which cancels
//   the connection. This is the correct way to implement stopStream.
//
// WHY we handle both { token } and { response } formats:
//   GwenLand's proxy emits { done, token }. If the proxy is configured
//   in passthrough mode it forwards Ollama's raw SSE which uses { response }
//   instead. Supporting both makes the hook resilient to proxy versions.
//
// NOTE: activeModel is intentionally captured at send time, not reactively.
//   A model change mid-stream would not affect the in-flight request.

import { useState, useRef, useCallback } from 'react'
import type { Message, ChatState } from '../types/chat'
import { useConfig } from '../context/ConfigContext'

// Rough token estimate: 1 token ≈ 4 chars (OpenAI/Anthropic convention).
// Replace with the real estimate_tokens WASM binding once @gwenland/core ships.
function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4)
}

function countContextTokens(messages: Message[]): number {
  return messages.reduce((sum, m) => sum + estimateTokens(m.content), 0)
}

function makeId(): string {
  // crypto.randomUUID is available in all modern browsers and in Tauri's WebView.
  return crypto.randomUUID()
}

interface UseChatResult {
  state: ChatState
  sendMessage: (content: string) => void
  clearHistory: () => void
  stopStream: () => void
}

export function useChat(activeModel: string): UseChatResult {
  const { config } = useConfig()

  const [state, setState] = useState<ChatState>({
    messages: [],
    activeModel,
    isStreaming: false,
    contextTokens: 0,
    maxTokens: 4096,
    firstTokenLatencyMs: null,
  })

  // AbortController ref — created fresh on each sendMessage call.
  // WHY a ref and not state: changing the controller must not trigger a render.
  const abortRef = useRef<AbortController | null>(null)

  const stopStream = useCallback(() => {
    // abort() triggers an AbortError in the in-flight fetch. The catch block
    // in sendMessage detects AbortError and skips the error bubble — the
    // partial assistant message stays visible as-is.
    abortRef.current?.abort()
    abortRef.current = null
    setState(s => ({ ...s, isStreaming: false }))
  }, [])

  const sendMessage = useCallback(async (content: string) => {
    if (!content.trim()) return

    // Capture config values at send time so a settings change mid-stream
    // doesn't affect the in-flight request.
    const chatUrl = `http://127.0.0.1:${config.proxyPort}/chat`
    const timeoutMs = config.streamTimeout * 1000

    const userMsg: Message = {
      id: makeId(),
      role: 'user',
      content: content.trim(),
      createdAt: Date.now(),
    }

    // Placeholder assistant message — content fills in as SSE tokens arrive.
    // We use a stable ID so the stream loop can find and patch the right entry.
    const assistantId = makeId()
    const assistantMsg: Message = {
      id: assistantId,
      role: 'assistant',
      content: '',
      isStreaming: true,
      createdAt: Date.now(),
    }

    // Capture the message history *before* the setState below, so we can
    // pass the clean history (without the empty placeholder) to the proxy.
    const historyForProxy = [...state.messages, userMsg].map(m => ({
      role: m.role,
      content: m.content,
    }))

    setState(s => {
      const next = [...s.messages, userMsg, assistantMsg]
      return {
        ...s,
        messages: next,
        isStreaming: true,
        activeModel,
        contextTokens: countContextTokens(next),
      }
    })

    // Fresh controller per request — any previous stream is already done.
    const abort = new AbortController()
    abortRef.current = abort

    // Timestamp the request so we can measure first-token latency.
    const startedAt = performance.now()
    let firstTokenSeen = false

    try {
      const res = await fetch(chatUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: activeModel,
          messages: historyForProxy,
          stream: true,
        }),
        // AbortSignal.any: whichever fires first wins — user stop or timeout.
        signal: AbortSignal.any([
          abort.signal,
          AbortSignal.timeout(timeoutMs),
        ]),
      })

      if (!res.ok || !res.body) {
        throw new Error(`Proxy returned ${res.status}`)
      }

      const reader = res.body.getReader()
      const decoder = new TextDecoder()

      // Buffer accumulates bytes across read() calls.
      // WHY we buffer: a single TCP chunk from the proxy may be cut in the
      // middle of a JSON object. We split on '\n' and only parse complete lines.
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        // { stream: true } tells TextDecoder to preserve state for multi-byte
        // chars that span chunk boundaries (e.g. Unicode in model output).
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        // The last element may be an incomplete line — keep it in the buffer.
        buffer = lines.pop() ?? ''

        for (const line of lines) {
          // SSE lines are "data: <payload>" or empty (heartbeat).
          if (!line.startsWith('data: ')) continue
          const raw = line.slice(6).trim()
          if (!raw || raw === '[DONE]') continue

          let parsed: { done?: boolean; token?: string; response?: string }
          try {
            parsed = JSON.parse(raw)
          } catch {
            // Malformed chunk — skip silently. Common during proxy restart.
            console.warn('[useChat] malformed SSE chunk:', raw)
            continue
          }

          // Support both GwenLand normalized format { token } and
          // Ollama raw passthrough format { response }.
          const token = parsed.token ?? parsed.response ?? ''
          const isDone = parsed.done === true

          if (!firstTokenSeen && token) {
            // Record milliseconds from request start to first readable token.
            // This is the real cold-path latency: network + proxy + model TTFT.
            const latencyMs = Math.round(performance.now() - startedAt)
            firstTokenSeen = true
            setState(s => ({ ...s, firstTokenLatencyMs: latencyMs }))
          }

          if (isDone) {
            setState(s => {
              const msgs = s.messages.map(m =>
                m.id === assistantId ? { ...m, isStreaming: false } : m
              )
              return {
                ...s,
                messages: msgs,
                isStreaming: false,
                contextTokens: countContextTokens(msgs),
              }
            })
            abortRef.current = null
            return
          }

          if (token) {
            // WHY functional update: tokens arrive in rapid bursts. Using the
            // previous state snapshot guarantees we never lose a token even
            // when multiple setState calls are batched in the same tick.
            setState(s => ({
              ...s,
              messages: s.messages.map(m =>
                m.id === assistantId
                  ? { ...m, content: m.content + token }
                  : m
              ),
            }))
          }
        }
      }
    } catch (err) {
      if (err instanceof Error && err.name === 'AbortError') {
        // User called stopStream() — the partial message stays visible as-is.
        // No error bubble needed; the stop was intentional.
        return
      }

      // Network / proxy error — surface it inside the assistant bubble so the
      // user does not have to open DevTools to see what went wrong.
      const errorText =
        err instanceof Error ? err.message : 'Connection failed'

      setState(s => ({
        ...s,
        isStreaming: false,
        messages: s.messages.map(m =>
          m.id === assistantId
            ? {
                ...m,
                content: `⚠ Stream error — proxy may have disconnected. Try again.\n\n_${errorText}_`,
                isStreaming: false,
              }
            : m
        ),
      }))
      abortRef.current = null
    }
  }, [activeModel, state.messages, config.proxyPort, config.streamTimeout])

  const clearHistory = useCallback(() => {
    stopStream()
    setState(s => ({
      ...s,
      messages: [],
      contextTokens: 0,
      isStreaming: false,
      // Keep firstTokenLatencyMs so the badge still shows last session latency
      // after clear. Reset it here if you want a clean slate instead.
    }))
  }, [stopStream])

  return { state, sendMessage, clearHistory, stopStream }
}
