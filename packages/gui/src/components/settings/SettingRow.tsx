// SettingRow.tsx — one row: a label/description on the left, a control on the
// right. Rows draw a bottom divider; the last row in a section omits it.
//
// WHY the divider is on the row, not the section:
//   The section wrapper can't know how many rows it holds without inspecting
//   children. Giving each row a bottom border and using the `last` prop to
//   suppress the final one keeps the layout self-contained.

import { type CSSProperties, type ReactNode } from 'react'

interface Props {
  label: string
  description?: string
  children: ReactNode
  /** When true, omit the bottom divider (final row in a section). */
  last?: boolean
}

export default function SettingRow({ label, description, children, last }: Props) {
  const row: CSSProperties = {
    display: 'flex',
    alignItems: 'center',
    gap: 12,
    padding: '11px 14px',
    borderBottom: last ? 'none' : '1px solid rgba(255,255,255,0.04)',
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
        {description && (
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
        )}
      </div>

      {/* Control — sits at the row's trailing edge, never shrinks. */}
      <div style={{ flexShrink: 0, display: 'flex', alignItems: 'center' }}>
        {children}
      </div>
    </div>
  )
}
