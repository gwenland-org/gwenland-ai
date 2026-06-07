// ChatInput.tsx — two-row input box at the bottom of the chat page.
//
// WHY two rows (textarea + toolbar) instead of one flat bar:
//   The model picker, attachment, and context counter are secondary controls.
//   Separating them from the writing area avoids cluttering the primary
//   interaction zone and matches the Gemini-style pattern from the spec.
//
// WHY auto-grow via onInput + scrollHeight:
//   A contenteditable div would need sanitisation before sending (XSS risk).
//   A controlled textarea with rows= is janky. scrollHeight gives pixel-
//   perfect grow without any of those trade-offs.
//
// WHY onNavigate is threaded here (not consumed here):
//   ChatInput only passes it down to ModelPicker. The "Pull model…" action
//   in ModelPicker needs to call it, but ChatInput itself has no use for it.
//   The alternative (a global nav context) is heavier for a 2-hop prop chain.

import { useRef, useState, useCallback, type KeyboardEvent } from 'react'
import { Send } from 'lucide-react'
import ModelPicker from './ModelPicker'
import ContextCounter from './ContextCounter'
import type { OllamaModel } from '../../types/chat'

interface ChatInputProps {
  onSend: (content: string) => void
  isStreaming: boolean
  contextTokens: number
  maxTokens: number
  models: OllamaModel[]
  activeModel: string
  onModelSelect: (name: string) => void
  // WHY proxyAlive here: the input owns the Send button. Disabling it here
  // (rather than in Chat.tsx) keeps the disabled-state logic co-located with
  // the button that acts on it.
  proxyAlive: boolean
  onNavigate?: (page: string, opts?: { openPull?: boolean }) => void
}

const MAX_ROWS = 5
const LINE_HEIGHT_PX = 21

export default function ChatInput({
  onSend,
  isStreaming,
  contextTokens,
  maxTokens,
  models,
  activeModel,
  onModelSelect,
  proxyAlive,
  onNavigate,
}: ChatInputProps) {
  const [value, setValue] = useState('')
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const [focused, setFocused] = useState(false)

  // Send is gated on three conditions: text present, not already streaming,
  // and the proxy is reachable. Checking proxyAlive here avoids a wasted
  // fetch that would immediately fail with ECONNREFUSED.
  const canSend = value.trim().length > 0 && !isStreaming && proxyAlive

  const resize = useCallback(() => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    const maxH = LINE_HEIGHT_PX * MAX_ROWS + 20
    el.style.height = Math.min(el.scrollHeight, maxH) + 'px'
    el.style.overflowY = el.scrollHeight > maxH ? 'auto' : 'hidden'
  }, [])

  const handleSend = useCallback(() => {
    if (!canSend) return
    onSend(value)
    setValue('')
    if (textareaRef.current) textareaRef.current.style.height = 'auto'
  }, [canSend, onSend, value])

  const handleKeyDown = useCallback(
    (e: KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault()
        handleSend()
      }
    },
    [handleSend]
  )

  return (
    <div
      style={{
        margin: '0 16px 16px',
        border: focused ? '1px solid rgba(255,140,66,0.35)' : '1px solid var(--border)',
        borderRadius: 14,
        background: 'rgba(255,255,255,0.04)',
        transition: 'border-color 150ms ease',
        overflow: 'hidden',
      }}
    >
      {/* ── Top row: textarea + send button ── */}
      <div style={{ display: 'flex', alignItems: 'flex-end', gap: 8, padding: '10px 12px 8px' }}>
        <textarea
          ref={textareaRef}
          value={value}
          onChange={e => setValue(e.target.value)}
          onInput={resize}
          onKeyDown={handleKeyDown}
          onFocus={() => setFocused(true)}
          onBlur={() => setFocused(false)}
          placeholder={proxyAlive ? 'Ask anything…' : 'Proxy offline — run gwen serve'}
          rows={1}
          style={{
            flex: 1,
            resize: 'none',
            background: 'transparent',
            border: 'none',
            outline: 'none',
            color: 'var(--text-primary)',
            fontSize: 13,
            lineHeight: `${LINE_HEIGHT_PX}px`,
            fontFamily: "'Geist', sans-serif",
            overflowY: 'hidden',
          }}
        />
        <button
          onClick={handleSend}
          disabled={!canSend}
          title={!proxyAlive ? 'Proxy offline' : 'Send (Enter)'}
          style={{
            width: 32, height: 32,
            borderRadius: 8,
            background: canSend ? 'var(--orange)' : 'rgba(255,255,255,0.06)',
            border: 'none',
            color: canSend ? '#1a0d02' : 'var(--text-secondary)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            flexShrink: 0,
            opacity: !canSend ? 0.4 : 1,
            cursor: canSend ? 'pointer' : 'not-allowed',
            transition: 'background 120ms ease, opacity 120ms ease',
          }}
        >
          <Send size={14} />
        </button>
      </div>

      {/* ── Bottom row: model picker + attachment + context counter ── */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          padding: '6px 10px 8px',
          borderTop: '1px solid rgba(255,255,255,0.05)',
        }}
      >
        <ModelPicker
          models={models}
          activeModel={activeModel}
          onSelect={onModelSelect}
          onNavigate={onNavigate}
        />

        {/* File attach — Cycle 6 */}
        {null}

        <div style={{ marginLeft: 'auto' }}>
          <ContextCounter used={contextTokens} max={maxTokens} />
        </div>
      </div>
    </div>
  )
}
