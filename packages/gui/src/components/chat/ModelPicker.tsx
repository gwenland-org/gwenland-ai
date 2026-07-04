// ModelPicker.tsx — inline pill that opens a model-selection popover.
//
// WHY a plain absolute-div popover instead of a portal or shadcn Popover:
//   No UI library is installed. A portal would require ReactDOM.createPortal
//   and z-index coordination across the whole layout. An absolute div anchored
//   to the pill is sufficient — the pill lives inside ChatInput which is at
//   the bottom of the viewport, so `bottom: 100%` always opens upward with
//   no clipping risk.
//
// WHY the popover closes on outside click via document listener:
//   The pill and its popover are visually small. A click anywhere outside
//   should close it — this is the standard UX expectation for lightweight
//   dropdowns without a backdrop overlay.
//
// WHY onNavigate is accepted as a prop (not imported from a context):
//   The "Pull model…" item navigates to the Models page. Threading the
//   callback from App.tsx through Chat → ChatInput → here is a 2-hop chain
//   that doesn't warrant a context. If navigation needs grow, introduce one.

import { useState, useEffect, useRef } from 'react'
import { ChevronUp } from 'lucide-react'
import type { OllamaModel } from '../../types/chat'

interface ModelPickerProps {
  models: OllamaModel[]
  activeModel: string
  onSelect: (name: string) => void
  onNavigate?: (page: string, opts?: { openPull?: boolean }) => void
}

export default function ModelPicker({ models, activeModel, onSelect, onNavigate }: ModelPickerProps) {
  const [open, setOpen] = useState(false)
  const rootRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!open) return
    function onDoc(e: MouseEvent) {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) {
        setOpen(false)
      }
    }
    document.addEventListener('mousedown', onDoc)
    return () => document.removeEventListener('mousedown', onDoc)
  }, [open])

  const active = models.find(m => m.name === activeModel)
  const isOnline = active?.isOnline ?? false

  return (
    <div ref={rootRef} style={{ position: 'relative', display: 'inline-flex' }}>
      {/* ── Pill button ── */}
      <button
        onClick={() => setOpen(o => !o)}
        style={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: 5,
          background: 'rgba(255,140,66,0.08)',
          border: '1px solid rgba(255,140,66,0.16)',
          borderRadius: 9999,
          padding: '4px 10px 4px 8px',
          cursor: 'pointer',
        }}
      >
        <span
          style={{
            width: 6, height: 6,
            borderRadius: '50%',
            background: isOnline ? '#22c55e' : 'rgba(180,180,180,0.35)',
            flexShrink: 0,
          }}
        />
        <span
          style={{
            fontSize: 11.5,
            fontWeight: 500,
            color: 'var(--orange)',
            fontFamily: "'Geist', sans-serif",
            maxWidth: 140,
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
          }}
        >
          {activeModel}
        </span>
        <ChevronUp
          size={12}
          color="var(--text-secondary)"
          style={{
            transform: open ? 'rotate(0deg)' : 'rotate(180deg)',
            transition: 'transform 150ms ease',
          }}
        />
      </button>

      {/* ── Popover ── */}
      {open && (
        <div
          style={{
            position: 'absolute',
            bottom: 'calc(100% + 6px)',
            left: 0,
            minWidth: 220,
            background: 'var(--card)',
            border: '1px solid var(--border)',
            borderRadius: 10,
            padding: '6px 0',
            zIndex: 50,
            boxShadow: '0 8px 24px rgba(0,0,0,0.5)',
          }}
        >
          <div
            style={{
              padding: '4px 12px 6px',
              fontSize: 10,
              color: 'var(--text-secondary)',
              fontFamily: "'Geist', sans-serif",
              textTransform: 'uppercase',
              letterSpacing: '0.08em',
            }}
          >
            Select Model
          </div>

          <div role="listbox" aria-label="Select model">
            {models.map(m => (
              <ModelRow
                key={m.name}
                model={m}
                isActive={m.name === activeModel}
                onSelect={() => { onSelect(m.name); setOpen(false) }}
              />
            ))}
          </div>

          <div style={{ height: 1, background: 'var(--border)', margin: '4px 0' }} />

          {/* Navigate to Model Manager and open the pull panel */}
          <button
            onClick={() => {
              setOpen(false)
              // onNavigate is optional — falls back to no-op if Chat was
              // rendered without a parent that provides navigation.
              onNavigate?.('models', { openPull: true })
            }}
            style={{
              width: '100%',
              textAlign: 'left',
              padding: '7px 12px',
              background: 'transparent',
              border: 'none',
              fontSize: 12,
              color: 'rgba(255,140,66,0.60)',
              fontFamily: "'Geist', sans-serif",
              cursor: 'pointer',
            }}
            onMouseEnter={e => (e.currentTarget.style.background = 'rgba(255,255,255,0.03)')}
            onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}
          >
            + Pull model…
          </button>
        </div>
      )}
    </div>
  )
}

function ModelRow({
  model, isActive, onSelect,
}: {
  model: OllamaModel
  isActive: boolean
  onSelect: () => void
}) {
  const [hover, setHover] = useState(false)

  return (
    <button
      role="option"
      aria-selected={isActive}
      onClick={onSelect}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      style={{
        width: '100%',
        display: 'flex',
        alignItems: 'center',
        gap: 8,
        padding: '7px 12px',
        background: hover ? 'rgba(255,255,255,0.03)' : 'transparent',
        border: 'none',
        textAlign: 'left',
        cursor: 'pointer',
        transition: 'background 100ms ease',
      }}
    >
      <span
        style={{
          width: 6, height: 6,
          borderRadius: '50%',
          background: model.isOnline ? '#22c55e' : 'rgba(180,180,180,0.35)',
          flexShrink: 0,
        }}
      />
      <span
        style={{
          flex: 1,
          fontSize: 12,
          color: 'var(--text-primary)',
          fontFamily: "'Geist', sans-serif",
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          whiteSpace: 'nowrap',
        }}
      >
        {model.name}
      </span>
      {isActive && (
        <span
          style={{
            fontSize: 10,
            color: 'rgba(255,140,66,0.60)',
            background: 'rgba(255,140,66,0.08)',
            border: '1px solid rgba(255,140,66,0.14)',
            borderRadius: 4,
            padding: '1px 6px',
          }}
        >
          active
        </span>
      )}
    </button>
  )
}
