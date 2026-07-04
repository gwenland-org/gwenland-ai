// ContextCounter.tsx — shows used / max context tokens in the input toolbar.
//
// WHY Coins icon: it visually communicates "budget being spent" — the token
// window is a finite resource, and Coins makes that cost tangible at a glance.

import { Coins } from 'lucide-react'

interface ContextCounterProps {
  used: number
  max: number
}

export default function ContextCounter({ used, max }: ContextCounterProps) {
  return (
    <span
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: 5,
        fontSize: 11,
        color: 'var(--text-secondary)',
        fontFamily: "'Geist Mono', monospace",
      }}
    >
      <Coins size={12} />
      <span style={{ color: 'rgba(255,140,66,0.55)' }}>{used}</span>
      / {max} ctx
    </span>
  )
}
