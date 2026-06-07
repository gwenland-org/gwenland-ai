// Chat.tsx — the Chat screen, rendered when active === 'chat' in App.tsx.
//
// WHY onNavigate is threaded through Chat → ChatInput → ModelPicker:
//   ModelPicker's "Pull model…" item should navigate to the Models page
//   with the pull panel pre-opened. Without a navigation callback, ModelPicker
//   would need to import App state or a context — both worse than one extra prop.
//   The prop chain is shallow (Chat → ChatInput → ModelPicker = 2 hops) so a
//   context is not warranted yet.

import { MessageSquare, Trash2, AlertTriangle } from 'lucide-react'
import { useModels } from '../hooks/useModels'
import { useChat } from '../hooks/useChat'
import { useProxyHealth } from '../hooks/useProxyHealth'
import { useConfig } from '../context/ConfigContext'
import MessageList from '../components/chat/MessageList'
import ChatInput from '../components/chat/ChatInput'
import SseBadge from '../components/chat/SseBadge'

interface ChatProps {
  onNavigate?: (page: string, opts?: { openPull?: boolean }) => void
}

export default function Chat({ onNavigate }: ChatProps) {
  const { models, activeModel, setActiveModel } = useModels()
  const { state, sendMessage, clearHistory } = useChat(activeModel)
  const { alive: proxyAlive } = useProxyHealth()
  const { config } = useConfig()

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        height: '100vh',
        background: 'var(--bg)',
        overflow: 'hidden',
      }}
    >
      {/* ── Topbar ── */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 10,
          padding: '0 16px',
          height: 48,
          borderBottom: '1px solid var(--border)',
          flexShrink: 0,
        }}
      >
        <MessageSquare size={15} color="rgba(255,140,66,0.60)" />
        <span style={{ fontSize: 13, fontWeight: 500, color: 'var(--text-primary)' }}>
          {state.messages.length ? 'Chat' : 'New Chat'}
        </span>
        <span style={{ fontSize: 11, color: 'var(--text-secondary)', marginLeft: 2 }}>
          · SSE ·{' '}
          <span style={{ color: proxyAlive ? '#22c55e' : 'rgba(230,80,60,0.85)', transition: 'color 400ms ease' }}>
            {proxyAlive ? `port ${config.proxyPort}` : 'proxy offline'}
          </span>
        </span>

        <div style={{ marginLeft: 'auto', display: 'flex', gap: 6 }}>
          <IconBtn icon={<Trash2 size={13} />} title="Clear history" onClick={clearHistory} />
        </div>
      </div>

      {/* ── Proxy offline banner ── */}
      {!proxyAlive && (
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 8,
            padding: '8px 16px',
            fontSize: 12,
            background: 'rgba(220,60,40,0.08)',
            borderBottom: '1px solid rgba(220,60,40,0.20)',
            color: 'rgba(230,80,60,0.80)',
            flexShrink: 0,
          }}
        >
          <AlertTriangle size={13} />
          GwenLand proxy offline — run{' '}
          <code
            style={{
              background: 'rgba(255,255,255,0.05)',
              padding: '1px 6px',
              borderRadius: 4,
              fontSize: 11,
              fontFamily: "'Geist Mono', monospace",
            }}
          >
            gwen serve
          </code>
          {' '}to start
        </div>
      )}

      {/* ── Message feed ── */}
      <MessageList messages={state.messages} onHintClick={sendMessage} />

      {/* ── SSE badge ── */}
      {proxyAlive && (
        <div style={{ padding: '0 20px 4px', flexShrink: 0 }}>
          <SseBadge
            latencyMs={state.firstTokenLatencyMs}
            isStreaming={state.isStreaming}
          />
        </div>
      )}

      {/* ── Input ── */}
      <ChatInput
        onSend={sendMessage}
        isStreaming={state.isStreaming}
        contextTokens={state.contextTokens}
        maxTokens={state.maxTokens}
        models={models}
        activeModel={activeModel}
        onModelSelect={setActiveModel}
        proxyAlive={proxyAlive}
        onNavigate={onNavigate}
      />
    </div>
  )
}

function IconBtn({ icon, title, onClick }: { icon: React.ReactNode; title?: string; onClick?: () => void }) {
  return (
    <button
      onClick={onClick}
      title={title}
      style={{
        width: 28, height: 28,
        borderRadius: 6,
        border: '1px solid rgba(255,255,255,0.08)',
        background: 'transparent',
        color: 'var(--text-secondary)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        cursor: 'pointer',
        transition: 'background 120ms ease',
      }}
      onMouseEnter={e => ((e.currentTarget).style.background = 'rgba(255,255,255,0.05)')}
      onMouseLeave={e => ((e.currentTarget).style.background = 'transparent')}
    >
      {icon}
    </button>
  )
}
