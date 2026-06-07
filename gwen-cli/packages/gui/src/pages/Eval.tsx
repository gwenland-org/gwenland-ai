// Eval.tsx — placeholder for the model evaluation screen (Cycle 6).

import { BarChart2 } from 'lucide-react'

export default function Eval() {
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
      <BarChart2 size={36} style={{ color: 'oklch(75% 0.18 48 / 20%)' }} />
      <p style={{ fontSize: '14px', fontWeight: 600, color: 'var(--text-primary)', margin: 0 }}>
        Eval
      </p>
      <p style={{ fontSize: '12px', color: 'var(--text-muted)', maxWidth: '280px', margin: 0 }}>
        Model evaluation ships in Cycle 6.{' '}
        Use{' '}
        <code
          style={{
            fontFamily: 'Geist Mono',
            background: 'oklch(100% 0 0 / 5%)',
            padding: '1px 6px',
            borderRadius: '4px',
          }}
        >
          gwen eval
        </code>{' '}
        in the terminal for now.
      </p>
    </div>
  )
}
