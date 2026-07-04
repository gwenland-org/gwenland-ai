// Doctor.tsx — placeholder for the environment diagnostics screen.
//
// CLI diagnostics exist in Cycle 4; the GUI view is deferred until
// the diagnostics API is exposed over IPC.

import { Stethoscope } from 'lucide-react'

export default function Doctor() {
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
      <Stethoscope size={36} style={{ color: 'oklch(75% 0.18 48 / 20%)' }} />
      <p style={{ fontSize: '14px', fontWeight: 600, color: 'var(--text-primary)', margin: 0 }}>
        Doctor
      </p>
      <p style={{ fontSize: '12px', color: 'var(--text-muted)', maxWidth: '280px', margin: 0 }}>
        Environment diagnostics ships in Cycle 4 CLI. GUI view coming soon.{' '}
        Use{' '}
        <code
          style={{
            fontFamily: 'Geist Mono',
            background: 'oklch(100% 0 0 / 5%)',
            padding: '1px 6px',
            borderRadius: '4px',
          }}
        >
          gwen doctor
        </code>{' '}
        in the terminal for now.
      </p>
    </div>
  )
}
