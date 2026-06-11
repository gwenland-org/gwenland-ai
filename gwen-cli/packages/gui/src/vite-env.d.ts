/// <reference types="vite/client" />

// Tauri injects the app version at build time. It is optional here so the
// AboutCard can fall back gracefully when running outside a Tauri build
// (e.g. `vite dev` in a browser).
interface ImportMetaEnv {
  readonly TAURI_APP_VERSION?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
