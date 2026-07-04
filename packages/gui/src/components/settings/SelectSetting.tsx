// SelectSetting.tsx — native <select> styled to match Gwen Noir.
//
// WHY native select (not a custom popover):
//   Native selects are accessible, keyboard-navigable, and respect OS
//   conventions for free. The dropdown list itself can't be fully themed
//   cross-platform, but the closed control matches the rest of the form,
//   which is what the user sees most of the time.

import { useState, type CSSProperties } from 'react'

interface Option {
  label: string
  value: string
}

interface Props {
  value: string
  options: Option[]
  onChange: (v: string) => void
}

export default function SelectSetting({ value, options, onChange }: Props) {
  const [focused, setFocused] = useState(false)

  const style: CSSProperties = {
    background: 'rgba(0,0,0,0.30)',
    border: `1px solid ${focused ? 'rgba(255,140,66,0.30)' : 'rgba(255,255,255,0.10)'}`,
    borderRadius: 6,
    padding: '5px 9px',
    color: 'var(--text-primary)',
    fontSize: 12,
    cursor: 'pointer',
    outline: 'none',
    fontFamily: "'Geist', sans-serif",
    transition: 'border-color 120ms ease',
  }

  return (
    <select
      value={value}
      onChange={e => onChange(e.target.value)}
      onFocus={() => setFocused(true)}
      onBlur={() => setFocused(false)}
      style={style}
    >
      {options.map(o => (
        <option key={o.value} value={o.value} style={{ background: 'var(--card)' }}>
          {o.label}
        </option>
      ))}
    </select>
  )
}
