// ModelStatsBar.tsx — thin strip at the bottom of the Models page.
//
// WHY derive stats here instead of in Models.tsx:
//   Stats are a pure function of the model list. Keeping derivation in
//   the component that renders them avoids threading computed values
//   through props and makes the computation co-located with its display.

import type { OllamaModel } from '../../types/chat'
import { formatBytes, formatCtx } from '../../utils/modelUtils'

interface ModelStatsBarProps {
  models: OllamaModel[]
  activeModel: string
}

export default function ModelStatsBar({ models, activeModel }: ModelStatsBarProps) {
  const totalSize  = models.reduce((sum, m) => sum + m.size, 0)
  const active     = models.find(m => m.name === activeModel)

  const stats = [
    { value: String(models.length),              label: 'models local'  },
    { value: formatBytes(totalSize),             label: 'disk used'     },
    { value: active?.paramCount  ?? '—',         label: 'active params' },
    { value: active ? formatCtx(active.contextLength) : '—', label: 'ctx window' },
  ]

  return (
    <div
      style={{
        display: 'flex',
        borderTop: '1px solid rgba(255,255,255,0.04)',
        padding: '10px 16px',
        flexShrink: 0,
      }}
    >
      {stats.map(({ value, label }) => (
        <div
          key={label}
          style={{ flex: 1, textAlign: 'center' }}
        >
          <div
            style={{
              fontSize: 14,
              fontWeight: 600,
              color: 'var(--orange)',
              fontFamily: "'Geist Mono', monospace",
            }}
          >
            {value}
          </div>
          <div
            style={{
              fontSize: 10,
              color: 'var(--text-secondary)',
              marginTop: 2,
              fontFamily: "'Geist', sans-serif",
            }}
          >
            {label}
          </div>
        </div>
      ))}
    </div>
  )
}
