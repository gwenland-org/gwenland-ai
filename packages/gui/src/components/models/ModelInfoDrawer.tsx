// ModelInfoDrawer.tsx — slide-in detail panel for a single model.
//
// WHY position: absolute (not fixed):
//   The spec requires the drawer to overlay the content area only, not the
//   sidebar or window chrome. `position: absolute` within a `position: relative`
//   parent (set in Models.tsx) achieves this. `position: fixed` would cover
//   the entire viewport including the sidebar.
//
// WHY CSS transition on transform (not max-width or display toggle):
//   transform: translateX() is GPU-composited — no layout recalculation on
//   each animation frame. max-width or left/right transitions force reflow.
//
// WHY we still render the drawer when model is null (just off-screen):
//   If we conditionally unmount the drawer, the exit animation never plays.
//   Instead we always render it and move it off-screen when model is null.

import { X } from 'lucide-react'
import type { OllamaModel } from '../../types/chat'
import { formatBytes, formatCtx, timeAgo } from '../../utils/modelUtils'

interface ModelInfoDrawerProps {
  model: OllamaModel | null
  onClose: () => void
  onSetActive: (name: string) => void
}

export default function ModelInfoDrawer({ model, onClose, onSetActive }: ModelInfoDrawerProps) {
  // Slide in when a model is selected, slide out when null.
  const open = model !== null

  return (
    <div
      style={{
        position: 'absolute',
        top: 0, right: 0,
        width: 280,
        height: '100%',
        background: 'var(--sidebar)',
        borderLeft: '1px solid var(--border)',
        zIndex: 20,
        transform: open ? 'translateX(0)' : 'translateX(100%)',
        transition: 'transform 200ms ease',
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden',
      }}
    >
      {/* ── Header ── */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '14px 16px',
          borderBottom: '1px solid var(--border)',
          flexShrink: 0,
        }}
      >
        <span style={{ fontSize: 13, fontWeight: 500, color: 'var(--text-primary)' }}>
          Model Info
        </span>
        <button
          onClick={onClose}
          style={{
            background: 'transparent', border: 'none',
            color: 'var(--text-secondary)', cursor: 'pointer',
            display: 'flex',
          }}
        >
          <X size={14} />
        </button>
      </div>

      {/* ── Content — only rendered when a model is selected ── */}
      {model && (
        <div style={{ flex: 1, padding: '16px', overflowY: 'auto' }}>
          {/* Model name at top */}
          <div style={{ marginBottom: 16 }}>
            <p style={{ fontSize: 11, color: 'var(--text-secondary)', marginBottom: 3 }}>Name</p>
            <p
              style={{
                fontSize: 12, color: 'var(--text-primary)',
                fontFamily: "'Geist Mono', monospace",
                wordBreak: 'break-all',
              }}
            >
              {model.name}
            </p>
          </div>

          {/* Detail rows */}
          {[
            { label: 'Size',          value: formatBytes(model.size) },
            { label: 'Quantization',  value: model.quantization },
            { label: 'Parameters',    value: model.paramCount },
            { label: 'Context',       value: `${formatCtx(model.contextLength)} tokens` },
            {
              label: 'Digest',
              // Show first 16 chars of the hash — enough to be identifiable.
              value: model.digest
                ? model.digest.replace('sha256:', 'sha256:').slice(0, 23) + '…'
                : 'unavailable',
            },
            { label: 'Modified',      value: model.modifiedAt ? timeAgo(model.modifiedAt) : '—' },
          ].map(({ label, value }) => (
            <div key={label} style={{ marginBottom: 12 }}>
              <p style={{ fontSize: 11, color: 'var(--text-secondary)', marginBottom: 2 }}>
                {label}
              </p>
              <p
                style={{
                  fontSize: 12,
                  color: 'var(--text-primary)',
                  fontFamily: "'Geist Mono', monospace",
                }}
              >
                {value}
              </p>
            </div>
          ))}

          {/* "Set as Active" button — hidden when already active */}
          {!model.isActive && (
            <button
              onClick={() => { onSetActive(model.name); onClose() }}
              style={{
                width: '100%',
                marginTop: 8,
                padding: '8px 0',
                background: 'rgba(255,140,66,0.12)',
                border: '1px solid rgba(255,140,66,0.25)',
                borderRadius: 8,
                color: 'var(--orange)',
                fontSize: 12,
                fontWeight: 500,
                cursor: 'pointer',
                fontFamily: "'Geist', sans-serif",
                transition: 'background 120ms ease',
              }}
              onMouseEnter={e => ((e.currentTarget).style.background = 'rgba(255,140,66,0.18)')}
              onMouseLeave={e => ((e.currentTarget).style.background = 'rgba(255,140,66,0.12)')}
            >
              Set as Active
            </button>
          )}
        </div>
      )}
    </div>
  )
}
