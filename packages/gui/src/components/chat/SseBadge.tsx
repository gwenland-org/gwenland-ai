// SseBadge.tsx — streaming indicator / proxy ready state shown above the input.
//
// WHY two display modes (streaming vs. ready):
//   When idle the badge gives ambient confirmation that the proxy is reachable
//   and shows its measured round-trip latency. During a stream it switches to
//   the animated dot + first-token latency so the user knows data is flowing.
//
// WHY latencyMs is number | null:
//   null means no message has been sent yet in this session. We show
//   "ready · port 1136" instead of "0ms" which would be misleading.

interface SseBadgeProps {
  // Milliseconds from request start to first token. null before first message.
  latencyMs: number | null
  // Whether the badge should show the streaming state.
  isStreaming: boolean
}

export default function SseBadge({ latencyMs, isStreaming }: SseBadgeProps) {
  const label = isStreaming
    ? `streaming · ${latencyMs !== null ? `${latencyMs}ms first token` : '…'}`
    : latencyMs !== null
      ? `ready · ${latencyMs}ms last latency`
      : 'ready · port 1136'

  return (
    <div
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: 5,
        fontSize: 10,
        borderRadius: 9999,
        padding: '2px 8px',
        background: 'rgba(255,140,66,0.07)',
        border: '1px solid rgba(255,140,66,0.14)',
        color: 'rgba(255,140,66,0.70)',
        marginBottom: 6,
      }}
    >
      {/* Green dot pulses during streaming; static when idle.
          The color contrast (green on orange-tinted badge) ensures it reads
          as a distinct signal, not part of the orange design language. */}
      <span
        style={{
          width: 5,
          height: 5,
          borderRadius: '50%',
          background: '#22c55e',
          display: 'inline-block',
          animation: isStreaming ? 'pulse 1.5s ease-in-out infinite' : 'none',
          opacity: isStreaming ? 1 : 0.6,
        }}
      />
      {label}
    </div>
  )
}
