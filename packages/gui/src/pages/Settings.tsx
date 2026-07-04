// Settings.tsx — Settings screen, rendered when active === 'settings'.
//
// Layout: a fixed 160px SettingsNav on the left, a scrollable content pane on
// the right. The visible tab is local state; every config edit flows through
// useSettings.update, which debounces a write to ~/.gwenland/config/config.json
// and flashes the "Saved" toast on success.
//
// WHY tabs live here (not in the URL): consistent with the app's router-free
// navigation. Only one tab is mounted at a time, so unmounting a tab is fine —
// no tab holds unsaved state (persistence is immediate-on-change).

import { useState, type CSSProperties } from 'react'
import {
  SlidersHorizontal,
  Server,
  Folder,
  Keyboard,
  Info,
  Check,
} from 'lucide-react'
import { useConfig } from '../context/ConfigContext'
import type { ToastVariant } from '../hooks/useToast'
import SettingsNav, { type SettingsTab } from '../components/settings/SettingsNav'
import SettingsSection from '../components/settings/SettingsSection'
import SettingRow from '../components/settings/SettingRow'
import ToggleSetting from '../components/settings/ToggleSetting'
import InputSetting from '../components/settings/InputSetting'
import SelectSetting from '../components/settings/SelectSetting'
import PathSetting from '../components/settings/PathSetting'
import AboutCard from '../components/settings/AboutCard'
import DangerZone from '../components/settings/DangerZone'

const VERSION = import.meta.env.TAURI_APP_VERSION ?? '0.1.0'

interface SettingsProps {
  toast?: (message: string, variant?: ToastVariant) => void
}

export default function Settings({ toast }: SettingsProps) {
  const { config, update, reset, loading, saved } = useConfig()
  const [tab, setTab] = useState<SettingsTab>('general')

  if (loading) return <SettingsSkeleton />

  return (
    <div style={{ display: 'flex', height: '100vh', background: 'var(--bg)' }}>
      <SettingsNav active={tab} onChange={setTab} />

      <div
        style={{
          flex: 1,
          overflowY: 'auto',
          padding: '18px 22px',
          position: 'relative',
          scrollbarWidth: 'thin',
          scrollbarColor: 'rgba(255,255,255,0.08) transparent',
        }}
      >
        {/* Saved indicator — top-right, auto-dismisses (see useSettings). */}
        {saved && <SavedToast />}

        {tab === 'general' && (
          <>
            <AboutCard onError={msg => toast?.(msg, 'error')} />
            <SettingsSection title="General" icon={SlidersHorizontal}>
              <SettingRow
                label="Launch at startup"
                description="Start GwenLand when system boots"
              >
                <ToggleSetting
                  checked={config.launchAtStartup}
                  onChange={v => update({ launchAtStartup: v })}
                />
              </SettingRow>
              <SettingRow
                label="Start minimized"
                description="Open to system tray on launch"
              >
                <ToggleSetting
                  checked={config.startMinimized}
                  onChange={v => update({ startMinimized: v })}
                />
              </SettingRow>
              <SettingRow
                label="Startup mode"
                description="Default interface on open"
              >
                <SelectSetting
                  value={config.startupMode}
                  onChange={v => update({ startupMode: v as typeof config.startupMode })}
                  options={[
                    { label: 'GUI (default)', value: 'gui' },
                    { label: 'TUI', value: 'tui' },
                    { label: 'Ask every time', value: 'ask' },
                  ]}
                />
              </SettingRow>
              <SettingRow label="Language" description="Interface language" last>
                <SelectSetting
                  value={config.language}
                  onChange={v => update({ language: v as typeof config.language })}
                  options={[
                    { label: 'English', value: 'en' },
                    { label: 'Indonesian', value: 'id' },
                  ]}
                />
              </SettingRow>
            </SettingsSection>
          </>
        )}

        {tab === 'proxy' && (
          <SettingsSection title="Proxy" icon={Server}>
            <SettingRow
              label="Proxy port"
              description="GwenLand SSE proxy (gwen serve)"
            >
              <InputSetting
                value={config.proxyPort}
                onChange={v => update({ proxyPort: Number(v) || 0 })}
                width={80}
                mono
              />
            </SettingRow>
            <SettingRow
              label="Ollama host"
              description="Local Ollama API endpoint"
            >
              <InputSetting
                value={config.ollamaHost}
                onChange={v => update({ ollamaHost: v })}
                width={180}
                mono
              />
            </SettingRow>
            <SettingRow
              label="Auto-start proxy"
              description="Run gwen serve when GUI opens"
            >
              <ToggleSetting
                checked={config.autoStartProxy}
                onChange={v => update({ autoStartProxy: v })}
              />
            </SettingRow>
            <SettingRow
              label="Stream timeout"
              description="SSE connection timeout (seconds)"
              last
            >
              <InputSetting
                value={config.streamTimeout}
                onChange={v => update({ streamTimeout: Number(v) || 0 })}
                width={60}
                mono
              />
            </SettingRow>
          </SettingsSection>
        )}

        {tab === 'paths' && (
          <SettingsSection title="Paths" icon={Folder}>
            <SettingRow
              label="Config directory"
              description="~/.gwenland/config/"
            >
              <PathSetting path={config.configDir} mode="open" onError={msg => toast?.(msg, 'error')} />
            </SettingRow>
            <SettingRow
              label="Models directory"
              description="Local model storage"
            >
              <PathSetting path={config.modelsDir} mode="open" onError={msg => toast?.(msg, 'error')} />
            </SettingRow>
            <SettingRow
              label="Sessions directory"
              description="JSONL chat session storage"
              last
            >
              <PathSetting path={config.sessionsDir} mode="open" onError={msg => toast?.(msg, 'error')} />
            </SettingRow>
          </SettingsSection>
        )}

        {tab === 'shortcuts' && <ShortcutsSection />}

        {tab === 'about' && <AboutSection version={VERSION} onError={msg => toast?.(msg, 'error')} />}

        {tab === 'danger' && (
          <DangerZone sessionsDir={config.sessionsDir} onReset={reset} onError={msg => toast?.(msg, 'warning')} />
        )}
      </div>
    </div>
  )
}

/* ────────────────────────── Saved toast ────────────────────────── */

function SavedToast() {
  const style: CSSProperties = {
    position: 'absolute',
    top: 12,
    right: 16,
    display: 'flex',
    alignItems: 'center',
    gap: 5,
    fontSize: 11,
    background: 'rgba(74,222,128,0.10)',
    border: '1px solid rgba(74,222,128,0.20)',
    color: '#4ade80',
    borderRadius: 6,
    padding: '4px 10px',
    fontFamily: "'Geist', sans-serif",
    zIndex: 10,
  }
  return (
    <div style={style}>
      <Check size={11} /> Saved
    </div>
  )
}

/* ──────────────────────── Shortcuts section ─────────────────────── */

const SHORTCUTS: { key: string; action: string }[] = [
  { key: 'P', action: 'Toggle proxy on/off' },
  { key: 'M', action: 'Go to Model Manager' },
  { key: 'C', action: 'Go to Chat' },
  { key: ',', action: 'Open Settings' },
  { key: 'Esc', action: 'Close drawer / popover' },
]

function ShortcutsSection() {
  return (
    <SettingsSection title="Shortcuts" icon={Keyboard}>
      {SHORTCUTS.map((s, i) => (
        <SettingRow
          key={s.key}
          label={s.action}
          last={i === SHORTCUTS.length - 1}
        >
          <kbd
            style={{
              fontFamily: "'Geist Mono', monospace",
              fontSize: 11,
              background: 'rgba(255,255,255,0.05)',
              border: '1px solid rgba(255,255,255,0.08)',
              borderRadius: 4,
              padding: '2px 7px',
              color: 'var(--text-primary)',
            }}
          >
            {s.key}
          </kbd>
        </SettingRow>
      ))}
    </SettingsSection>
  )
}

/* ───────────────────────── About section ────────────────────────── */

function AboutSection({ version, onError }: { version: string; onError?: (msg: string) => void }) {
  return (
    <>
      <AboutCard onError={onError} />
      <SettingsSection title="About" icon={Info}>
        <SettingRow label="Version" description="Installed build" last>
          <span
            style={{
              fontFamily: "'Geist Mono', monospace",
              fontSize: 12,
              color: 'var(--text-mono)',
            }}
          >
            v{version}
          </span>
        </SettingRow>
      </SettingsSection>
    </>
  )
}

/* ──────────────────────── Loading skeleton ──────────────────────── */

function SettingsSkeleton() {
  // Mirrors the real layout (nav rail + content) so the swap to loaded
  // content does not shift the screen.
  return (
    <div style={{ display: 'flex', height: '100vh', background: 'var(--bg)' }}>
      <div
        style={{
          width: 160,
          flexShrink: 0,
          borderRight: '1px solid var(--border)',
          padding: '12px 8px',
        }}
      />
      <div style={{ flex: 1, padding: '18px 22px' }}>
        <div
          style={{
            height: 60,
            borderRadius: 10,
            background: 'rgba(255,255,255,0.03)',
            marginBottom: 14,
            animation: 'pulse 1.4s ease-in-out infinite',
          }}
        />
        <div
          style={{
            height: 200,
            borderRadius: 10,
            background: 'rgba(255,255,255,0.03)',
            animation: 'pulse 1.4s ease-in-out infinite',
          }}
        />
      </div>
    </div>
  )
}
