// InputSetting.tsx — compact text/number input for a single setting value.
//
// WHY value is `string | number` but onChange emits `string`:
//   The DOM input is always a string. Numeric coercion (e.g. proxyPort) is
//   the caller's concern — it knows whether the field is a number and can
//   `Number(v)` at the update site. Keeping the input dumb avoids NaN states
//   while the user is mid-edit (e.g. an empty field).

import { useState, type CSSProperties } from 'react'

interface Props {
  value: string | number
  onChange: (v: string) => void
  width?: number
  mono?: boolean
  placeholder?: string
}

export default function InputSetting({
  value,
  onChange,
  width = 120,
  mono,
  placeholder,
}: Props) {
  const [focused, setFocused] = useState(false)

  const style: CSSProperties = {
    width,
    background: 'rgba(0,0,0,0.30)',
    border: `1px solid ${focused ? 'rgba(255,140,66,0.30)' : 'rgba(255,255,255,0.10)'}`,
    borderRadius: 6,
    padding: '5px 9px',
    fontSize: 12,
    color: 'var(--text-primary)',
    outline: 'none',
    fontFamily: mono ? "'Geist Mono', monospace" : "'Geist', sans-serif",
    transition: 'border-color 120ms ease',
  }

  return (
    <input
      value={value}
      placeholder={placeholder}
      onChange={e => onChange(e.target.value)}
      onFocus={() => setFocused(true)}
      onBlur={() => setFocused(false)}
      style={style}
    />
  )
}
