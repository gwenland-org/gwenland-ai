// config.ts — shape of ~/.config/gwen/config.json, the GUI-editable settings.
//
// WHY a dedicated types module (not inline in the hook):
//   The schema is referenced by useSettings, the Settings page, and the
//   DangerZone reset action. A single source of truth avoids drift.
//
// Paths (configDir/modelsDir/sessionsDir) are written by the Rust core, not
// the GUI. They are surfaced read-only so the user can open them, but the
// GUI never edits them — DEFAULT_CONFIG leaves them empty until the core
// populates the file.

export type StartupMode = 'gui' | 'tui' | 'ask'
export type Language = 'en' | 'id'

export interface GwenConfig {
  // General
  launchAtStartup: boolean
  startMinimized: boolean
  startupMode: StartupMode
  language: Language

  // Proxy
  proxyPort: number
  ollamaHost: string
  autoStartProxy: boolean
  streamTimeout: number // seconds

  // Paths (read-only display — set by Rust core, not editable here)
  configDir: string
  modelsDir: string
  sessionsDir: string
}

export const DEFAULT_CONFIG: GwenConfig = {
  launchAtStartup: false,
  startMinimized: false,
  startupMode: 'gui',
  language: 'en',
  proxyPort: 1136,
  ollamaHost: 'localhost:11434',
  autoStartProxy: true,
  streamTimeout: 30,
  configDir: '',
  modelsDir: '',
  sessionsDir: '',
}
