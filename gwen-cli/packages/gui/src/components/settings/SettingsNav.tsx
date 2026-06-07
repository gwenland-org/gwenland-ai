// SettingsNav.tsx — left sub-navigation inside the Settings page.
//
// WHY a sub-nav separate from the main Sidebar:
//   Settings has six logical groups that are too many for the main sidebar
//   and only relevant while on this screen. Keeping them local avoids
//   cluttering global navigation.
//
// The "Danger" item is visually separated and red-tinted to signal that the
// actions behind it are destructive.

import { useState, type CSSProperties } from 'react'
import {
  SlidersHorizontal,
  Server,
  Folder,
  Keyboard,
  Info,
  AlertTriangle,
  type LucideIcon,
} from 'lucide-react'

export type SettingsTab =
  | 'general'
  | 'proxy'
  | 'paths'
  | 'shortcuts'
  | 'about'
  | 'danger'

interface NavItem {
  id: SettingsTab
  label: string
  icon: LucideIcon
  danger?: boolean
}

const ITEMS: NavItem[] = [
  { id: 'general', label: 'General', icon: SlidersHorizontal },
  { id: 'proxy', label: 'Proxy', icon: Server },
  { id: 'paths', label: 'Paths', icon: Folder },
  { id: 'shortcuts', label: 'Shortcuts', icon: Keyboard },
  { id: 'about', label: 'About', icon: Info },
]

const DANGER_ITEM: NavItem = {
  id: 'danger',
  label: 'Reset',
  icon: AlertTriangle,
  danger: true,
}

interface Props {
  active: SettingsTab
  onChange: (tab: SettingsTab) => void
}

export default function SettingsNav({ active, onChange }: Props) {
  return (
    <div
      style={{
        width: 160,
        flexShrink: 0,
        borderRight: '1px solid var(--border)',
        padding: '12px 8px',
        display: 'flex',
        flexDirection: 'column',
        gap: 2,
      }}
    >
      {ITEMS.map(item => (
        <NavRow
          key={item.id}
          item={item}
          active={active === item.id}
          onClick={() => onChange(item.id)}
        />
      ))}

      {/* Danger divider */}
      <div
        style={{
          margin: '10px 6px 6px',
          fontSize: 10,
          fontWeight: 600,
          letterSpacing: '0.08em',
          textTransform: 'uppercase',
          color: 'rgba(248,113,113,0.5)',
        }}
      >
        Danger
      </div>
      <NavRow
        item={DANGER_ITEM}
        active={active === DANGER_ITEM.id}
        onClick={() => onChange(DANGER_ITEM.id)}
      />
    </div>
  )
}

function NavRow({
  item,
  active,
  onClick,
}: {
  item: NavItem
  active: boolean
  onClick: () => void
}) {
  const [hover, setHover] = useState(false)
  const Icon = item.icon

  // Colour resolution: danger items use the red tint; otherwise active uses
  // the orange accent and inactive uses the primary text colour.
  const color = item.danger
    ? active
      ? '#f87171'
      : 'rgba(248,113,113,0.6)'
    : active
      ? 'var(--orange)'
      : 'var(--text-primary)'

  const background = item.danger
    ? active || hover
      ? 'rgba(248,113,113,0.10)'
      : 'transparent'
    : active
      ? 'var(--orange-dim)'
      : hover
        ? 'rgba(255,255,255,0.03)'
        : 'transparent'

  const style: CSSProperties = {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    width: '100%',
    padding: '7px 10px',
    borderRadius: 'var(--radius-input)',
    background,
    border: 'none',
    textAlign: 'left',
    color,
    fontSize: 12,
    fontWeight: active ? 500 : 400,
    fontFamily: "'Geist', sans-serif",
    transition: 'background 120ms ease',
  }

  return (
    <button
      onClick={onClick}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      style={style}
    >
      <Icon size={14} style={{ flexShrink: 0 }} />
      <span style={{ whiteSpace: 'nowrap' }}>{item.label}</span>
    </button>
  )
}
