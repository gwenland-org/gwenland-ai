// modelUtils.ts — pure display-formatting helpers for model data.
//
// WHY a separate utils file:
//   formatBytes, formatCtx, and timeAgo are needed by ModelCard,
//   ModelInfoDrawer, and ModelStatsBar. Keeping them here avoids
//   duplicating the same logic in three places.
//
// All functions are pure (no side effects, no imports) so they are trivially
// testable and safe to call during render.

// ── Context length inference ─────────────────────────────────────────────────
// Maps known model families to their real context window sizes.
// Default 4096 is used for unknown families.
// WHY a static map and not a fetch: Ollama doesn't return context length in
// /api/tags. The map is updated manually when new families are added.

const CTX_MAP: Record<string, number> = {
  'qwen3':    32768,
  'qwen2':    8192,
  'qwen':     8192,
  'llama3.1': 131072,
  'llama3.2': 131072,
  'llama3':   131072,
  'llama2':   4096,
  'llama':    4096,
  'mistral':  8192,
  'gemma3':   8192,
  'gemma2':   8192,
  'gemma':    8192,
  'phi4':     16384,
  'phi3':     131072,
  'phi':      4096,
  'codellama':4096,
  'deepseek': 32768,
  'vicuna':   4096,
  'falcon':   2048,
  'solar':    4096,
}

export function inferContextLength(modelName: string): number {
  // Model names follow "family:tag" — e.g. "qwen3:8b", "llama3.1:70b-instruct"
  const family = modelName.split(':')[0].toLowerCase()

  // Try exact match first, then prefix match (e.g. "llama3.1" matches "llama3.1")
  if (CTX_MAP[family] !== undefined) return CTX_MAP[family]
  for (const [key, val] of Object.entries(CTX_MAP)) {
    if (family.startsWith(key)) return val
  }
  return 4096
}

// ── Parameter count parsing ──────────────────────────────────────────────────
// Ollama returns values like "8.2B", "7.24B", "70B".
// We normalise to integer-prefix strings: "8B", "7B", "70B".

export function parseParamCount(raw: string | undefined): string {
  if (!raw) return '?'
  // Extract leading integer part — e.g. "7.24B" → "7", "8B" → "8"
  const m = raw.match(/^(\d+)/)
  return m ? `${m[1]}B` : raw
}

// ── Display formatting ───────────────────────────────────────────────────────

export function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B'
  const gb = bytes / 1_073_741_824
  if (gb >= 1) return `${gb.toFixed(1)} GB`
  const mb = bytes / 1_048_576
  if (mb >= 1) return `${mb.toFixed(0)} MB`
  return `${(bytes / 1024).toFixed(0)} KB`
}

// 131072 → "128k", 32768 → "32k", 4096 → "4k"
export function formatCtx(tokens: number): string {
  if (tokens >= 1000) return `${Math.round(tokens / 1024)}k`
  return String(tokens)
}

// ── Relative time ─────────────────────────────────────────────────────────────
// WHY not use a library (date-fns, dayjs): they add bundle weight for a
// one-line display string. This covers all the ranges we actually show.

export function timeAgo(isoDate: string): string {
  const diff = Date.now() - new Date(isoDate).getTime()
  const mins  = Math.floor(diff / 60_000)
  const hours = Math.floor(diff / 3_600_000)
  const days  = Math.floor(diff / 86_400_000)
  const weeks = Math.floor(diff / 604_800_000)
  if (mins  < 1)  return 'just now'
  if (mins  < 60) return `${mins}m ago`
  if (hours < 24) return `${hours}h ago`
  if (days  < 7)  return `${days} day${days !== 1 ? 's' : ''} ago`
  return `${weeks} week${weeks !== 1 ? 's' : ''} ago`
}

// ── Model family accent detection ─────────────────────────────────────────────
// Used by ModelCard to choose icon + color tint.

export type ModelAccent = 'orange' | 'violet' | 'muted'

export function modelAccent(name: string): ModelAccent {
  const base = name.split(':')[0].toLowerCase()
  if (base.startsWith('gwen')) return 'violet'
  if (base.startsWith('qwen')) return 'orange'
  return 'muted'
}
