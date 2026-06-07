// Dataset.tsx — placeholder for the dataset management screen (Cycle 5).

import { Database } from 'lucide-react'

export default function Dataset() {
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
      <Database size={36} style={{ color: 'oklch(75% 0.18 48 / 20%)' }} />
      <p style={{ fontSize: '14px', fontWeight: 600, color: 'var(--text-primary)', margin: 0 }}>
        Dataset
      </p>
      <p style={{ fontSize: '12px', color: 'var(--text-muted)', maxWidth: '280px', margin: 0 }}>
        Dataset management ships in Cycle 5.{' '}
        Use{' '}
        <code
          style={{
            fontFamily: 'Geist Mono',
            background: 'oklch(100% 0 0 / 5%)',
            padding: '1px 6px',
            borderRadius: '4px',
          }}
        >
          gwen dataset
        </code>{' '}
        in the terminal for now.
      </p>
    </div>
  )
}
