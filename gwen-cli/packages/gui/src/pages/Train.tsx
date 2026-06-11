// Train.tsx — placeholder for the native Rust training screen (Cycle 5).
//
// WHY a placeholder instead of nothing: clicking "Train" in the sidebar
// must land somewhere. A blank page or silent fallback to Dashboard would
// look like a bug. A clear coming-soon message sets expectations honestly.

import { FlaskConical } from 'lucide-react'

export default function Train() {
  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        height: '100%',
        gap: '12px',
        textAlign: 'center',
      }}
    >
      <FlaskConical size={36} style={{ color: 'oklch(75% 0.18 48 / 20%)' }} />
      <p style={{ fontSize: '14px', fontWeight: 600, color: 'var(--text-primary)', margin: 0 }}>
        Train
      </p>
      <p style={{ fontSize: '12px', color: 'var(--text-muted)', maxWidth: '280px', margin: 0 }}>
        Native Rust training via Candle ships in Cycle 5.{' '}
        Use{' '}
        <code
          style={{
            fontFamily: 'Geist Mono',
            background: 'oklch(100% 0 0 / 5%)',
            padding: '1px 6px',
            borderRadius: '4px',
          }}
        >
          gwen train
        </code>{' '}
        in the terminal for now.
      </p>
    </div>
  )
}
