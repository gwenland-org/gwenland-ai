// PullPanel.tsx — input + progress bar for pulling a model from Ollama registry.
//
// WHY the input uses font-mono:
//   Model names are identifiers ("qwen3:14b", "llama3.1:70b-instruct").
//   Mono font makes them easier to read and edit accurately.
//
// WHY we close the panel on success via a useEffect on isSuccess:
//   Closing is a side-effect of the success state, not a direct action.
//   Responding to it in useEffect keeps the logic centralised and avoids
//   threading a close callback into the hook.

import { useState, useEffect, useRef } from 'react'
import { Download, X } from 'lucide-react'
import type { PullProgress } from '../../hooks/usePullModel'
import { formatBytes } from '../../utils/modelUtils'

interface PullPanelProps {
  onPull: (name: string) => void
  progress: PullProgress | null
  onCancel: () => void
  onClose: () => void   // called when panel should be hidden (success or user X)
}

export default function PullPanel({ onPull, progress, onCancel, onClose }: PullPanelProps) {
  const [input, setInput] = useState('')
  const inputRef = useRef<HTMLInputElement>(null)

  // Auto-focus the input when the panel mounts.
  useEffect(() => { inputRef.current?.focus() }, [])

  // Close panel automatically 1.5s after a successful pull.
  useEffect(() => {
    if (progress?.isSuccess) {
      const id = setTimeout(onClose, 1500)
      return () => clearTimeout(id)
    }
  }, [progress?.isSuccess, onClose])

  function handlePull() {
    if (!input.trim() || progress?.isPulling) return
    onPull(input.trim())
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (e.key === 'Enter') handlePull()
  }

  const isPulling = progress?.isPulling ?? false
  const isSuccess = progress?.isSuccess ?? false

  return (
    <div
      style={{
        background: 'rgba(255,140,66,0.04)',
        border: '1px solid rgba(255,140,66,0.12)',
        borderRadius: 10,
        padding: 14,
        marginBottom: 4,
      }}
    >
      {/* ── Header row ── */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 10 }}>
        <span style={{ fontSize: 12, fontWeight: 500, color: 'var(--text-primary)' }}>
          Pull from Ollama Registry
        </span>
        <button
          onClick={onClose}
          style={{
            background: 'transparent',
            border: 'none',
            color: 'var(--text-secondary)',
            cursor: 'pointer',
            padding: 2,
            display: 'flex',
          }}
        >
          <X size={13} />
        </button>
      </div>

      {/* ── Input row ── */}
      <div style={{ display: 'flex', gap: 8 }}>
        <input
          ref={inputRef}
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="e.g. qwen3:14b"
          disabled={isPulling}
          style={{
            flex: 1,
            background: 'rgba(255,255,255,0.04)',
            border: '1px solid var(--border)',
            borderRadius: 6,
            padding: '6px 10px',
            fontSize: 12,
            color: 'var(--text-primary)',
            fontFamily: "'Geist Mono', monospace",
            outline: 'none',
          }}
        />
        <button
          onClick={handlePull}
          disabled={!input.trim() || isPulling}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 5,
            background: input.trim() && !isPulling ? 'var(--orange)' : 'rgba(255,255,255,0.06)',
            color: input.trim() && !isPulling ? '#1a0d02' : 'var(--text-secondary)',
            border: 'none',
            borderRadius: 6,
            padding: '6px 12px',
            fontSize: 12,
            fontWeight: 600,
            cursor: input.trim() && !isPulling ? 'pointer' : 'not-allowed',
            opacity: !input.trim() || isPulling ? 0.5 : 1,
            transition: 'background 120ms ease',
            whiteSpace: 'nowrap',
            fontFamily: "'Geist', sans-serif",
          }}
        >
          <Download size={12} />
          Pull
        </button>
      </div>

      {/* ── Progress section — shown when a pull is in progress or just finished ── */}
      {progress && (progress.isPulling || progress.isSuccess || progress.error) && (
        <div style={{ marginTop: 10 }}>
          {/* Status label row */}
          <div
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              marginBottom: 6,
            }}
          >
            <span
              style={{
                fontSize: 11,
                color: isSuccess
                  ? '#22c55e'
                  : progress.error
                    ? 'rgba(230,80,60,0.85)'
                    : 'var(--text-secondary)',
                fontFamily: "'Geist', sans-serif",
              }}
            >
              {progress.error
                ? `⚠ ${progress.error}`
                : isSuccess
                  ? '✓ Done! Model ready.'
                  : `${progress.status || 'Pulling…'}`}
            </span>

            {/* Size progress label */}
            {isPulling && progress.totalBytes > 0 && (
              <span style={{ fontSize: 11, color: 'var(--text-secondary)', fontFamily: "'Geist Mono', monospace" }}>
                {progress.percent}% · {formatBytes(progress.completedBytes)} / {formatBytes(progress.totalBytes)}
              </span>
            )}

            {/* Cancel button — only while pulling, not on success */}
            {isPulling && (
              <button
                onClick={onCancel}
                style={{
                  background: 'transparent',
                  border: 'none',
                  fontSize: 11,
                  color: 'rgba(248,80,60,0.70)',
                  cursor: 'pointer',
                  fontFamily: "'Geist', sans-serif",
                  padding: '0 4px',
                }}
                onMouseEnter={e => ((e.target as HTMLElement).style.color = 'rgba(248,80,60,1)')}
                onMouseLeave={e => ((e.target as HTMLElement).style.color = 'rgba(248,80,60,0.70)')}
              >
                Cancel
              </button>
            )}
          </div>

          {/* Progress bar */}
          {(isPulling || isSuccess) && (
            <div
              style={{
                height: 3,
                background: 'rgba(255,255,255,0.06)',
                borderRadius: 9999,
                overflow: 'hidden',
              }}
            >
              <div
                style={{
                  height: '100%',
                  width: `${isSuccess ? 100 : progress.percent}%`,
                  background: isSuccess ? '#22c55e' : 'var(--orange)',
                  borderRadius: 9999,
                  transition: 'width 300ms ease, background 200ms ease',
                  animation: isPulling && progress.percent === 0
                    ? 'pulse 1.5s ease-in-out infinite'
                    : 'none',
                }}
              />
            </div>
          )}
        </div>
      )}
    </div>
  )
}
