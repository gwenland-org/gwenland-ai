// RegistryTabs.tsx — tab strip for switching model sources.
//
// WHY HuggingFace and Recommended are locked (not just disabled):
//   Disabled implies the feature exists but is unavailable right now.
//   "Locked with Cycle 7" is honest — these tabs are future work, not
//   temporarily unavailable functionality.
//
// WHY the locked notice renders inside this component (not in Models.tsx):
//   The notice is a direct consequence of clicking a locked tab. Keeping
//   the notice logic here avoids passing an extra state variable up to the
//   parent just to track which locked tab was last clicked.

import { useState } from 'react'
import { Rocket } from 'lucide-react'

export type Tab = 'local' | 'huggingface' | 'recommended'

interface RegistryTabsProps {
  active: Tab
  onChange: (tab: Tab) => void
}

// Tab definitions — locked tabs will show the Cycle 7 badge.
const TABS: { id: Tab; label: string; locked: boolean }[] = [
  { id: 'local',       label: 'Local',        locked: false },
  { id: 'huggingface', label: 'HuggingFace',  locked: true  },
  { id: 'recommended', label: 'Recommended',  locked: true  },
]

export default function RegistryTabs({ active, onChange }: RegistryTabsProps) {
  // null = no locked notice shown; set to the tab id when a locked tab is clicked.
  const [lockedNotice, setLockedNotice] = useState<Tab | null>(null)

  function handleClick(tab: Tab, locked: boolean) {
    if (locked) {
      // Toggle notice — clicking the same locked tab again dismisses it.
      setLockedNotice(prev => prev === tab ? null : tab)
      return
    }
    setLockedNotice(null)
    onChange(tab)
  }

  return (
    <div style={{ borderBottom: '1px solid var(--border)', padding: '0 16px' }}>
      {/* ── Tab row ── */}
      <div role="tablist" style={{ display: 'flex', gap: 2 }}>
        {TABS.map(({ id, label, locked }) => {
          const isActive = active === id && !locked

          return (
            <button
              key={id}
              role="tab"
              aria-selected={isActive}
              aria-disabled={locked ? 'true' : undefined}
              onClick={() => handleClick(id, locked)}
              title={locked ? `Available in Cycle 7` : undefined}
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: 4,
                padding: '8px 12px',
                fontSize: 12,
                fontWeight: isActive ? 500 : 400,
                color: isActive
                  ? 'var(--orange)'
                  : 'var(--text-secondary)',
                background: 'transparent',
                border: 'none',
                borderBottom: isActive
                  ? '2px solid var(--orange)'
                  : '2px solid transparent',
                marginBottom: -1,   // overlap the container border-bottom
                cursor: locked ? 'not-allowed' : 'pointer',
                opacity: locked ? 0.4 : 1,
                transition: 'color 120ms ease, border-color 120ms ease',
                fontFamily: "'Geist', sans-serif",
              }}
            >
              {label}
              {locked && (
                // Cycle badge — violet to distinguish from the orange theme,
                // signalling "future" rather than "current" functionality.
                <span
                  style={{
                    fontSize: 10,
                    background: 'rgba(139,92,246,0.10)',
                    color: '#a78bfa',
                    border: '1px solid rgba(139,92,246,0.20)',
                    borderRadius: 4,
                    padding: '1px 5px',
                    marginLeft: 2,
                  }}
                >
                  Cycle 7
                </span>
              )}
            </button>
          )
        })}
      </div>

      {/* ── Locked notice — shown below tabs when a locked tab is clicked ── */}
      {lockedNotice && (
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 8,
            fontSize: 12,
            color: 'var(--text-secondary)',
            background: 'rgba(255,255,255,0.03)',
            border: '1px solid rgba(255,255,255,0.06)',
            borderRadius: 8,
            padding: '8px 12px',
            margin: '8px 0',
          }}
        >
          <Rocket size={13} color="var(--text-secondary)" />
          {lockedNotice === 'huggingface'
            ? 'HuggingFace model fetching ships in Cycle 7 — '
            : 'Recommended models list ships in Cycle 7 — '}
          <code
            style={{
              background: 'rgba(255,255,255,0.06)',
              padding: '1px 6px',
              borderRadius: 4,
              fontSize: 11,
              fontFamily: "'Geist Mono', monospace",
            }}
          >
            {lockedNotice === 'huggingface'
              ? 'gwen fetch username/model-name'
              : 'gwen recommend'}
          </code>
        </div>
      )}
    </div>
  )
}
