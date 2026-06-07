// MessageList.tsx — scrollable feed of message bubbles.
//
// WHY auto-scroll uses scrollHeight directly rather than scrollIntoView:
//   scrollIntoView on the last element causes a jarring jump when the
//   element is partially visible. Setting scrollTop = scrollHeight on the
//   container gives a smooth bottom-pin effect during streaming.
//
// WHY the empty state lives here and not in Chat.tsx:
//   MessageList owns the scroll container. Putting the empty state inside
//   the same container keeps layout consistent — Chat.tsx never needs to
//   conditionally swap between two different height-100% elements.

import { useEffect, useRef } from 'react'
import MessageBubble from './MessageBubble'
import type { Message } from '../../types/chat'

const HINT_CHIPS = [
  'List my models',
  'Start training',
  'Run benchmark',
]

interface MessageListProps {
  messages: Message[]
  onHintClick: (text: string) => void
}

export default function MessageList({ messages, onHintClick }: MessageListProps) {
  const containerRef = useRef<HTMLDivElement>(null)

  // Scroll to bottom whenever messages change (new turn or streaming token).
  useEffect(() => {
    const el = containerRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [messages])

  return (
    <div
      ref={containerRef}
      style={{
        flex: 1,
        overflowY: 'auto',
        display: 'flex',
        flexDirection: 'column',
        gap: 16,
        padding: '20px 20px 8px',
        // Custom scrollbar — thin and dark to match the noir theme.
        scrollbarWidth: 'thin',
        scrollbarColor: 'rgba(255,255,255,0.08) transparent',
      }}
    >
      {messages.length === 0 ? (
        <EmptyState onHintClick={onHintClick} />
      ) : (
        messages.map(msg => <MessageBubble key={msg.id} message={msg} />)
      )}
    </div>
  )
}

// ── Empty state ───────────────────────────────────────────────────────────────

function EmptyState({ onHintClick }: { onHintClick: (t: string) => void }) {
  return (
    <div
      style={{
        flex: 1,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 16,
        color: 'var(--text-secondary)',
        userSelect: 'none',
      }}
    >
      {/* Gwen logo mark — same gradient as the avatar in MessageBubble */}
      <div
        style={{
          width: 40, height: 40,
          borderRadius: 12,
          background: 'linear-gradient(135deg, #FF8C42 0%, #f9a03f 100%)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontSize: 20, fontWeight: 700, color: '#1a0d02',
          fontFamily: "'Geist', sans-serif",
        }}
      >
        G
      </div>

      <p style={{ fontSize: 13, color: 'var(--text-secondary)', margin: 0 }}>
        Your machine. Your models. Your rules.
      </p>

      {/* Hint chips — quick-start prompts so the first interaction is obvious */}
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', justifyContent: 'center' }}>
        {HINT_CHIPS.map(chip => (
          <button
            key={chip}
            onClick={() => onHintClick(chip)}
            style={{
              fontSize: 11,
              color: 'rgba(255,140,66,0.60)',
              background: 'rgba(255,140,66,0.06)',
              border: '1px solid rgba(255,140,66,0.12)',
              borderRadius: 6,
              padding: '4px 12px',
              cursor: 'pointer',
              fontFamily: "'Geist', sans-serif",
              transition: 'background 120ms ease',
            }}
            onMouseEnter={e =>
              (e.currentTarget.style.background = 'rgba(255,140,66,0.10)')
            }
            onMouseLeave={e =>
              (e.currentTarget.style.background = 'rgba(255,140,66,0.06)')
            }
          >
            {chip}
          </button>
        ))}
      </div>
    </div>
  )
}
