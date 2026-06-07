// ConfigContext.tsx — single source of truth for GwenConfig across all hooks.
//
// WHY a context instead of calling useSettings() in each hook:
//   useSettings reads and debounce-writes config.json. If Chat, Models, and
//   Settings each called useSettings independently, we'd have three separate
//   file readers and three separate debounce timers — writes could race and
//   overwrite each other. One provider owns the file; everyone else reads via
//   useConfig().
//
// WHY re-export `saved`, `loading`, and `reset` from useSettings:
//   Settings.tsx needs `saved` for its flash indicator, `loading` to defer
//   rendering until the file has been read, and `reset` for the Danger Zone.
//   Those concerns live in the provider, not in every consumer.

import { createContext, useContext, type ReactNode } from 'react'
import { useSettings } from '../hooks/useSettings'
import type { GwenConfig } from '../types/config'
import { DEFAULT_CONFIG } from '../types/config'

interface ConfigContextValue {
  config: GwenConfig
  update: (partial: Partial<GwenConfig>) => void
  reset: () => void
  loading: boolean
  saved: boolean
}

const ConfigContext = createContext<ConfigContextValue>({
  config: DEFAULT_CONFIG,
  update: () => {},
  reset: () => {},
  loading: false,
  saved: false,
})

export function ConfigProvider({ children }: { children: ReactNode }) {
  const settings = useSettings()
  return (
    <ConfigContext.Provider value={settings}>
      {children}
    </ConfigContext.Provider>
  )
}

export const useConfig = () => useContext(ConfigContext)
