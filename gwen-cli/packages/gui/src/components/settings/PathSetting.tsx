// PathSetting.tsx — read-only path display with an Open (or Browse) button.
//
// WHY read-only: these directories are owned by the Rust core. Letting the
// user retype them in the GUI would risk pointing the app at a non-existent
// path. The user can open the folder to inspect it, but not redefine it here.
//
// `mode === 'open'`  → open the folder in the OS file explorer (shell.open).
// `mode === 'browse'` → folder picker (dialog.open) that reports the choice
//                        back via onBrowse. Reserved for future editable paths.

import { useState, type CSSProperties } from 'react'
import { FolderOpen } from 'lucide-react'
import { open as openPath } from '@tauri-apps/plugin-shell'
import { open as openDialog } from '@tauri-apps/plugin-dialog'

interface Props {
  path: string
  mode: 'open' | 'browse'
  onBrowse?: (selected: string) => void
  onError?: (msg: string) => void
}

export default function PathSetting({ path, mode, onBrowse, onError }: Props) {
  const [hover, setHover] = useState(false)

  // An empty path means the core hasn't written config yet — nothing to open.
  const disabled = mode === 'open' && !path

  async function handleClick() {
    if (mode === 'open') {
      if (!path) return
      try {
        await openPath(path)
      } catch {
        // Folder may not exist yet — surface via onError if provided.
        onError?.('Could not open folder')
      }
      return
    }

    // browse
    try {
      const selected = await openDialog({ directory: true })
      if (typeof selected === 'string') onBrowse?.(selected)
    } catch {
      // Dialog cancelled or unavailable — no-op.
    }
  }

  const field: CSSProperties = {
    flex: 1,
    minWidth: 0,
    background: 'rgba(0,0,0,0.30)',
    border: '1px solid rgba(255,255,255,0.10)',
    borderRadius: 6,
    padding: '5px 9px',
    fontSize: 11,
    fontFamily: "'Geist Mono', monospace",
    color: 'rgba(232,230,240,0.6)',
    whiteSpace: 'nowrap',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
  }

  const btn: CSSProperties = {
    display: 'inline-flex',
    alignItems: 'center',
    gap: 5,
    padding: '5px 10px',
    borderRadius: 6,
    border: '1px solid rgba(255,255,255,0.10)',
    background: hover && !disabled ? 'rgba(255,255,255,0.08)' : 'rgba(255,255,255,0.05)',
    fontSize: 11,
    color: 'var(--text-primary)',
    fontFamily: "'Geist', sans-serif",
    cursor: disabled ? 'not-allowed' : 'pointer',
    opacity: disabled ? 0.4 : 1,
    flexShrink: 0,
    transition: 'background 120ms ease',
  }

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, width: 320 }}>
      <span style={field} title={path || undefined}>
        {path || '—'}
      </span>
      <button
        onClick={handleClick}
        disabled={disabled}
        onMouseEnter={() => setHover(true)}
        onMouseLeave={() => setHover(false)}
        style={btn}
      >
        <FolderOpen size={12} />
        {mode === 'open' ? 'Open' : 'Browse'}
      </button>
    </div>
  )
}
