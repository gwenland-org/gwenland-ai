// SettingsSection.tsx — titled card that groups related SettingRow items.
//
// The header carries an icon + title; the body is a bordered card whose rows
// (SettingRow) draw their own dividers. This wrapper owns only the outer
// chrome so individual tabs stay declarative.

import { type CSSProperties, type ReactNode } from 'react'
import type { LucideIcon } from 'lucide-react'

interface Props {
  title: string
  icon: LucideIcon
  children: ReactNode
}

export default function SettingsSection({ title, icon: Icon, children }: Props) {
  const header: CSSProperties = {
    display: 'flex',
    alignItems: 'center',
    gap: 7,
    marginBottom: 8,
  }

  return (
    <section style={{ marginBottom: 18 }}>
      <div style={header}>
        <Icon size={14} color="var(--orange)" />
        <h2
          style={{
            fontSize: 13,
            fontWeight: 600,
            color: 'var(--text-primary)',
            fontFamily: "'Geist', sans-serif",
          }}
        >
          {title}
        </h2>
      </div>

      <div
        style={{
          background: 'var(--card)',
          border: '1px solid var(--border)',
          borderRadius: 'var(--radius-card)',
          overflow: 'hidden',
        }}
      >
        {children}
      </div>
    </section>
  )
}
