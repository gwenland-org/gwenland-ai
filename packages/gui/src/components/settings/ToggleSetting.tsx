// ToggleSetting.tsx — hand-rolled boolean toggle pill.
//
// WHY no library: the codebase deliberately avoids UI deps (no Radix/shadcn).
// A toggle is a track + an absolutely-positioned thumb, animated via two CSS
// transitions — trivial to own outright.

import { type CSSProperties } from 'react'

interface Props {
  checked: boolean
  onChange: (v: boolean) => void
  disabled?: boolean
}

export default function ToggleSetting({ checked, onChange, disabled }: Props) {
  const track: CSSProperties = {
    position: 'relative',
    width: 32,
    height: 18,
    borderRadius: 9,
    border: 'none',
    padding: 0,
    background: checked ? 'var(--orange)' : 'rgba(255,255,255,0.10)',
    cursor: disabled ? 'not-allowed' : 'pointer',
    opacity: disabled ? 0.4 : 1,
    transition: 'background 200ms ease',
    flexShrink: 0,
  }

  const thumb: CSSProperties = {
    position: 'absolute',
    top: 3,
    left: 0,
    width: 12,
    height: 12,
    borderRadius: '50%',
    background: '#fff',
    transform: checked ? 'translateX(17px)' : 'translateX(3px)',
    transition: 'transform 200ms ease',
  }

  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      disabled={disabled}
      onClick={() => !disabled && onChange(!checked)}
      style={track}
    >
      <span style={thumb} />
    </button>
  )
}
