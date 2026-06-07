// useSettings.ts — reads and persists ~/.config/gwen/config.json via the
// Tauri fs plugin.
//
// WHY fs plugin and not localStorage:
//   The config file is shared with the Rust core and the CLI — the GUI is
//   one of several writers/readers. localStorage would silo the GUI's copy
//   and the two would drift. The file on disk is the single source of truth.
//
// WHY debounced writes (500ms):
//   Typing in a numeric input (e.g. proxy port) fires onChange per keystroke.
//   Writing the whole file each keystroke is wasteful and races on disk. We
//   coalesce rapid edits into one write 500ms after the last change.
//
// WHY the path is `.config/gwen/...` with BaseDirectory.Home:
//   The config lives at ~/.config/gwen/ (dotted). Tauri's Home base dir is
//   the user's home, so the relative path keeps the dot.
//
// Falls back to DEFAULT_CONFIG if the file is missing or malformed so the
// screen is always usable, even on a fresh install before the core has
// written anything.

import { useState, useEffect, useRef, useCallback } from 'react'
import { readTextFile, writeTextFile, BaseDirectory } from '@tauri-apps/plugin-fs'
import { GwenConfig, DEFAULT_CONFIG } from '../types/config'

const CONFIG_PATH = '.config/gwen/config.json'
const SAVE_DEBOUNCE_MS = 500
const SAVED_FLASH_MS = 1500

interface UseSettingsResult {
  config: GwenConfig
  update: (partial: Partial<GwenConfig>) => void
  reset: () => void
  loading: boolean
  saved: boolean
}

export function useSettings(): UseSettingsResult {
  const [config, setConfig] = useState<GwenConfig>(DEFAULT_CONFIG)
  const [loading, setLoading] = useState(true)
  const [saved, setSaved] = useState(false)

  // Refs so the debounced writer always sees the latest config without
  // re-creating timers, and so timers survive re-renders.
  const saveTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const flashTimer = useRef<ReturnType<typeof setTimeout> | null>(null)

  // Load once on mount.
  useEffect(() => {
    let cancelled = false
    readTextFile(CONFIG_PATH, { baseDir: BaseDirectory.Home })
      .then(raw => {
        if (cancelled) return
        // Merge over defaults so a partial/older file still yields a complete
        // config and never produces `undefined` fields.
        setConfig({ ...DEFAULT_CONFIG, ...JSON.parse(raw) })
      })
      .catch(() => {
        // Missing or malformed — start from defaults. We deliberately do NOT
        // write the file here; the core owns initial creation.
        if (!cancelled) setConfig(DEFAULT_CONFIG)
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [])

  // Cleanup any pending timers on unmount.
  useEffect(() => {
    return () => {
      if (saveTimer.current) clearTimeout(saveTimer.current)
      if (flashTimer.current) clearTimeout(flashTimer.current)
    }
  }, [])

  const persist = useCallback((next: GwenConfig) => {
    if (saveTimer.current) clearTimeout(saveTimer.current)
    saveTimer.current = setTimeout(() => {
      writeTextFile(CONFIG_PATH, JSON.stringify(next, null, 2), {
        baseDir: BaseDirectory.Home,
      })
        .then(() => {
          setSaved(true)
          if (flashTimer.current) clearTimeout(flashTimer.current)
          flashTimer.current = setTimeout(() => setSaved(false), SAVED_FLASH_MS)
        })
        .catch(() => {
          // Write failed (permissions, disk) — leave `saved` false so no
          // misleading confirmation is shown. The in-memory state stays so
          // the user can retry by editing again.
        })
    }, SAVE_DEBOUNCE_MS)
  }, [])

  // TODO: OS integration — launchAtStartup requires autostart registry/plist.
  // TODO: OS integration — startMinimized requires Tauri window state API.
  // TODO: OS integration — startupMode requires IPC between CLI and GUI.
  // TODO: i18n — language switching requires a full i18n system.
  // These fields are persisted to disk but have no runtime effect until
  // the corresponding OS-level integration is implemented.
  const update = useCallback(
    (partial: Partial<GwenConfig>) => {
      setConfig(prev => {
        const next = { ...prev, ...partial }
        persist(next)
        return next
      })
    },
    [persist],
  )

  // Reset writes DEFAULT_CONFIG but preserves the path fields, which are
  // owned by the core and should survive a settings reset.
  const reset = useCallback(() => {
    setConfig(prev => {
      const next: GwenConfig = {
        ...DEFAULT_CONFIG,
        configDir: prev.configDir,
        modelsDir: prev.modelsDir,
        sessionsDir: prev.sessionsDir,
      }
      persist(next)
      return next
    })
  }, [persist])

  return { config, update, reset, loading, saved }
}
