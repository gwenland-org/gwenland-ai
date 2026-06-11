// DangerZone.tsx — destructive actions, each guarded by an inline
// double-click confirmation (no modal, per spec).
//
// Confirmation pattern: first click arms the button ("Confirm? Click again")
// and starts a 3s timer. A second click within the window executes; otherwise
// the button disarms and reverts its label. This keeps a single accidental
// click from wiping data while avoiding a heavyweight dialog.
//
// "Clear Sessions" removes every *.jsonl file in sessionsDir. "Reset Config"
// delegates to the parent (useSettings.reset), which rewrites DEFAULT_CONFIG.

import { useState, useRef, useEffect, type CSSProperties } from 'react'
import { AlertTriangle, Trash2, RotateCcw } from 'lucide-react'
import { readDir, remove } from '@tauri-apps/plugin-fs'

const CONFIRM_WINDOW_MS = 3000

interface Props {
  sessionsDir: string
  onReset: () => void
  onError?: (msg: string) => void
}

export default function DangerZone({ sessionsDir, onReset, onError }: Props) {
  return (
    <section style={{ marginBottom: 18 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 7, marginBottom: 8 }}>
        <AlertTriangle size={14} color="#f87171" />
        <h2
          style={{
            fontSize: 13,
            fontWeight: 600,
            color: 'var(--text-primary)',
            fontFamily: "'Geist', sans-serif",
          }}
        >
          Danger Zone
        </h2>
      </div>

      <div
        style={{
          background: 'rgba(248,113,113,0.04)',
          border: '1px solid rgba(248,113,113,0.12)',
          borderRadius: 10,
          overflow: 'hidden',
        }}
      >
        <DangerRow
          icon={Trash2}
          label="Clear sessions"
          description="Delete all saved chat sessions (.jsonl). Cannot be undone."
          confirmLabel="Confirm? Click again"
          onError={onError}
          action={async () => {
            // No sessionsDir means the core hasn't populated config — nothing
            // to clear. Treat as a no-op rather than erroring.
            if (!sessionsDir) return
            const entries = await readDir(sessionsDir)
            await Promise.all(
              entries
                .filter(e => e.isFile && e.name.endsWith('.jsonl'))
                .map(e => remove(`${sessionsDir}/${e.name}`)),
            )
          }}
        />
        <DangerRow
          icon={RotateCcw}
          label="Reset configuration"
          description="Restore all settings to their defaults."
          confirmLabel="Confirm? Click again"
          onError={onError}
          action={async () => onReset()}
          last
        />
      </div>
    </section>
  )
}

function DangerRow({
  icon: Icon,
  label,
  description,
  confirmLabel,
  action,
  onError,
  last,
}: {
  icon: typeof Trash2
  label: string
  description: string
  confirmLabel: string
  action: () => Promise<void>
  onError?: (msg: string) => void
  last?: boolean
}) {
  const [armed, setArmed] = useState(false)
  const [busy, setBusy] = useState(false)
  const [hover, setHover] = useState(false)
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    return () => {
      if (timer.current) clearTimeout(timer.current)
    }
  }, [])

  function disarm() {
    if (timer.current) clearTimeout(timer.current)
    timer.current = null
    setArmed(false)
  }

  async function handleClick() {
    if (busy) return
    if (!armed) {
      // First click — arm and start the revert timer.
      setArmed(true)
      timer.current = setTimeout(disarm, CONFIRM_WINDOW_MS)
      return
    }
    // Second click within the window — execute.
    disarm()
    setBusy(true)
    try {
      await action()
    } catch {
      // Swallow the throw but surface to user via toast.
      // A failed delete leaves files in place — the safe outcome.
      onError?.('Some session files could not be deleted')
    } finally {
      setBusy(false)
    }
  }

  const row: CSSProperties = {
    display: 'flex',
    alignItems: 'center',
    gap: 12,
    padding: '11px 14px',
    borderBottom: last ? 'none' : '1px solid rgba(248,113,113,0.10)',
  }

  const btn: CSSProperties = {
    display: 'inline-flex',
    alignItems: 'center',
    gap: 5,
    border: '1px solid rgba(248,113,113,0.25)',
    background:
      armed || hover ? 'rgba(248,113,113,0.15)' : 'rgba(248,113,113,0.08)',
    color: '#f87171',
    fontSize: 11,
    borderRadius: 6,
    padding: '5px 12px',
    fontFamily: "'Geist', sans-serif",
    cursor: busy ? 'default' : 'pointer',
    flexShrink: 0,
    whiteSpace: 'nowrap',
    transition: 'background 120ms ease',
  }

  return (
    <div style={row}>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div
          style={{
            fontSize: 12,
            fontWeight: 500,
            color: 'var(--text-primary)',
            fontFamily: "'Geist', sans-serif",
          }}
        >
          {label}
        </div>
        <div
          style={{
            fontSize: 11,
            color: 'var(--text-secondary)',
            marginTop: 2,
            fontFamily: "'Geist', sans-serif",
          }}
        >
          {description}
        </div>
      </div>

      <button
        onClick={handleClick}
        onMouseEnter={() => setHover(true)}
        onMouseLeave={() => setHover(false)}
        style={btn}
      >
        <Icon size={12} />
        {busy ? 'Working…' : armed ? confirmLabel : label}
      </button>
    </div>
  )
}
