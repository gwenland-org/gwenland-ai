// MessageBubble.tsx — renders one message turn (user or assistant).
//
// WHY parse content for fenced code blocks here rather than in the hook:
//   Content parsing is a display concern. The hook stores raw markdown-style
//   text so it can be re-rendered differently in other contexts (export, TTS).
//
// HOW content is split:
//   We split on the fenced code fence regex. Odd-indexed segments are inside
//   a fence (lang on first line, rest is code). Even-indexed are plain text.

import { User } from 'lucide-react'
import CodeBlock from './CodeBlock'
import type { Message } from '../../types/chat'

// Matches ```lang\n...code...\n``` blocks. Non-greedy so nested fences work.
const FENCE_RE = /```([^\n]*)\n([\s\S]*?)```/g

interface ParsedSegment {
  type: 'text' | 'code'
  content: string
  lang?: string
}

function parseContent(text: string): ParsedSegment[] {
  const segments: ParsedSegment[] = []
  let lastIndex = 0

  for (const match of text.matchAll(FENCE_RE)) {
    const index = match.index ?? 0
    if (index > lastIndex) {
      segments.push({ type: 'text', content: text.slice(lastIndex, index) })
    }
    segments.push({ type: 'code', lang: match[1].trim(), content: match[2] })
    lastIndex = index + match[0].length
  }

  if (lastIndex < text.length) {
    segments.push({ type: 'text', content: text.slice(lastIndex) })
  }

  // Always return at least one segment so the component never renders nothing.
  if (segments.length === 0) {
    segments.push({ type: 'text', content: text })
  }

  return segments
}

// Blinking cursor appended to the last segment during streaming.
function StreamCursor() {
  return (
    <span
      style={{
        display: 'inline-block',
        width: 2,
        height: '1em',
        background: 'var(--orange)',
        marginLeft: 1,
        verticalAlign: 'text-bottom',
        animation: 'pulse 1s step-start infinite',
      }}
    />
  )
}

interface MessageBubbleProps {
  message: Message
}

export default function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === 'user'
  const segments = parseContent(message.content)

  // ── Avatar ──────────────────────────────────────────────────────────────
  const avatar = isUser ? (
    <div
      style={{
        width: 26, height: 26,
        borderRadius: 7,
        background: 'rgba(255,255,255,0.08)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        flexShrink: 0,
        marginTop: 2,
      }}
    >
      <User size={13} color="var(--text-secondary)" />
    </div>
  ) : (
    // Gwen avatar — orange gradient with the "G" glyph.
    <div
      style={{
        width: 26, height: 26,
        borderRadius: 7,
        background: 'linear-gradient(135deg, #FF8C42 0%, #f9a03f 100%)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        flexShrink: 0,
        marginTop: 2,
        fontSize: 12,
        fontWeight: 700,
        color: '#1a0d02',
        fontFamily: "'Geist', sans-serif",
      }}
    >
      G
    </div>
  )

  // ── Bubble ───────────────────────────────────────────────────────────────
  const bubble = (
    <div
      style={{
        maxWidth: '72%',
        padding: '9px 13px',
        fontSize: 13,
        lineHeight: 1.65,
        // User: orange-tinted card. Assistant: near-invisible white card.
        // Border-radius asymmetry indicates message direction at a glance.
        ...(isUser
          ? {
              background: 'rgba(255,140,66,0.12)',
              border: '1px solid rgba(255,140,66,0.20)',
              borderRadius: '10px 10px 3px 10px',
              color: 'var(--text-primary)',
            }
          : {
              background: 'rgba(255,255,255,0.04)',
              border: '1px solid rgba(255,140,66,0.10)',
              borderRadius: '3px 10px 10px 10px',
              color: 'var(--text-primary)',
            }),
      }}
    >
      {segments.map((seg, i) => {
        const isLast = i === segments.length - 1

        if (seg.type === 'code') {
          return <CodeBlock key={i} lang={seg.lang ?? ''} code={seg.content} />
        }

        return (
          <p key={i} style={{ margin: 0, whiteSpace: 'pre-wrap' }}>
            {seg.content}
            {isLast && message.isStreaming && <StreamCursor />}
          </p>
        )
      })}
    </div>
  )

  return (
    <div
      style={{
        display: 'flex',
        // User messages sit on the right; assistant on the left.
        flexDirection: isUser ? 'row-reverse' : 'row',
        alignItems: 'flex-start',
        gap: 8,
      }}
    >
      {avatar}
      {bubble}
    </div>
  )
}
