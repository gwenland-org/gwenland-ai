// AboutCard.tsx — version + update card shown at the top of the General tab.
//
// "Check Update" runs `gwen update --check` through the shell plugin and
// prints whatever the CLI reports inline beneath the card. "Changelog" opens
// the GitHub releases page in the default browser.
//
// WHY the version reads from import.meta.env with a fallback:
//   Tauri injects TAURI_APP_VERSION at build time, but in a plain `vite dev`
//   browser session that var is absent. The package version (0.1.0) is the
//   sensible fallback so the card never shows "undefined".

import { useState, type CSSProperties } from 'react'
import { open as openUrl } from '@tauri-apps/plugin-shell'
import { Command } from '@tauri-apps/plugin-shell'

const VERSION = import.meta.env.TAURI_APP_VERSION ?? '0.1.0'
const RELEASES_URL = 'https://github.com/jinxsuper/gwenland/releases'

interface AboutCardProps {
  onError?: (msg: string) => void
}

export default function AboutCard({ onError }: AboutCardProps = {}) {
  const [checking, setChecking] = useState(false)
  const [updateMsg, setUpdateMsg] = useState<string | null>(null)

  async function checkUpdate() {
    setChecking(true)
    setUpdateMsg(null)
    try {
      const out = await Command.sidecar('binaries/gwen', ['update', '--check']).execute()
      const text = (out.stdout || out.stderr || '').trim()
      setUpdateMsg(text || 'No update information returned.')
    } catch {
      // CLI not on PATH or sidecar missing — tell the user plainly rather
      // than failing silently.
      setUpdateMsg('Could not run `gwen update --check`. Is the CLI installed?')
    } finally {
      setChecking(false)
    }
  }

  function openChangelog() {
    openUrl(RELEASES_URL).catch(() => {
      onError?.('Could not open browser')
    })
  }

  const card: CSSProperties = {
    background: 'rgba(255,140,66,0.04)',
    border: '1px solid rgba(255,140,66,0.10)',
    borderRadius: 10,
    padding: 14,
    marginBottom: 14,
  }

  const btn: CSSProperties = {
    fontSize: 11,
    padding: '5px 11px',
    borderRadius: 6,
    border: '1px solid rgba(255,255,255,0.10)',
    background: 'rgba(255,255,255,0.05)',
    color: 'var(--text-primary)',
    fontFamily: "'Geist', sans-serif",
    cursor: 'pointer',
  }

  return (
    <div style={card}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
        {/* G logo mark */}
        <div
          style={{
            width: 36,
            height: 36,
            borderRadius: 9,
            background: 'var(--orange-dim)',
            border: '1px solid rgba(255,140,66,0.20)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: 18,
            fontWeight: 700,
            color: 'var(--orange)',
            fontFamily: "'Geist', sans-serif",
            flexShrink: 0,
          }}
        >
          G
        </div>

        <div style={{ flex: 1, minWidth: 0 }}>
          <div
            style={{
              fontSize: 14,
              fontWeight: 600,
              color: 'var(--text-primary)',
              fontFamily: "'Geist', sans-serif",
            }}
          >
            GwenLand{' '}
            <span style={{ color: 'var(--text-secondary)', fontWeight: 400 }}>
              v{VERSION}
            </span>
          </div>
          <div
            style={{
              fontSize: 11,
              color: 'var(--text-secondary)',
              marginTop: 2,
              fontFamily: "'Geist', sans-serif",
            }}
          >
            MIT + Commons Clause · by JinXSuper
          </div>
        </div>

        <div style={{ display: 'flex', gap: 6, flexShrink: 0 }}>
          <button style={btn} onClick={openChangelog}>
            Changelog
          </button>
          <button style={btn} onClick={checkUpdate} disabled={checking}>
            {checking ? 'Checking…' : 'Check Update'}
          </button>
        </div>
      </div>

      {updateMsg && (
        <div
          style={{
            marginTop: 10,
            fontSize: 11,
            color: 'var(--text-mono)',
            fontFamily: "'Geist Mono', monospace",
            whiteSpace: 'pre-wrap',
            borderTop: '1px solid rgba(255,255,255,0.06)',
            paddingTop: 8,
          }}
        >
          {updateMsg}
        </div>
      )}
    </div>
  )
}
