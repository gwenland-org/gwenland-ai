// usePullModel.ts — streams a model pull from Ollama and reports progress.
//
// WHY fetch + ReadableStream (same pattern as useChat):
//   Ollama's /api/pull returns NDJSON — one JSON object per newline.
//   This is NOT SSE format (no "data: " prefix), just plain newline-delimited
//   JSON. We use the same line-buffer technique to handle chunks that split
//   across read() calls.
//
// WHY this hits Ollama directly via config.ollamaHost (not via the proxy):
//   The GwenLand proxy only handles /chat. Pull operations are management
//   tasks that go directly to the Ollama HTTP API.
//
// WHY onSuccess callback instead of returning a promise:
//   The caller (Models.tsx) needs to re-fetch the model list after a pull
//   completes. A callback avoids prop-drilling a refetch function into this
//   hook and keeps the hook's public API minimal.

import { useState, useRef, useCallback } from 'react'
import { useConfig } from '../context/ConfigContext'

export interface PullProgress {
  status: string
  percent: number         // 0–100, computed from completed / total
  totalBytes: number
  completedBytes: number
  isPulling: boolean
  isSuccess: boolean
  error: string | null
}

const INITIAL_PROGRESS: PullProgress = {
  status: '',
  percent: 0,
  totalBytes: 0,
  completedBytes: 0,
  isPulling: false,
  isSuccess: false,
  error: null,
}

interface UsePullModelResult {
  pull: (modelName: string) => Promise<void>
  cancel: () => void
  progress: PullProgress | null
}

export function usePullModel(onSuccess: () => void): UsePullModelResult {
  const [progress, setProgress] = useState<PullProgress | null>(null)
  const abortRef = useRef<AbortController | null>(null)
  const { config } = useConfig()

  const cancel = useCallback(() => {
    abortRef.current?.abort()
    abortRef.current = null
    setProgress(p => p ? { ...p, isPulling: false, error: 'Cancelled' } : null)
  }, [])

  const pull = useCallback(async (modelName: string) => {
    if (!modelName.trim()) return

    const pullUrl = `http://${config.ollamaHost}/api/pull`
    setProgress({ ...INITIAL_PROGRESS, isPulling: true, status: 'Starting…' })

    const abort = new AbortController()
    abortRef.current = abort

    try {
      const res = await fetch(pullUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: modelName.trim(), stream: true }),
        signal: abort.signal,
      })

      if (!res.ok || !res.body) {
        throw new Error(`Ollama returned ${res.status}`)
      }

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      // Ollama pull emits plain NDJSON — one JSON object per line.
      // Example lines:
      //   {"status":"pulling manifest"}
      //   {"status":"downloading","digest":"sha256:...","total":8200000000,"completed":5100000000}
      //   {"status":"success"}
      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() ?? ''

        for (const line of lines) {
          const raw = line.trim()
          if (!raw) continue

          let chunk: {
            status?: string
            digest?: string
            total?: number
            completed?: number
            error?: string
          }
          try {
            chunk = JSON.parse(raw)
          } catch {
            continue
          }

          if (chunk.error) {
            setProgress(p => p ? { ...p, isPulling: false, error: chunk.error ?? 'Unknown error' } : null)
            return
          }

          if (chunk.status === 'success') {
            setProgress(p => p ? {
              ...p,
              status: 'Done! Model ready.',
              isPulling: false,
              isSuccess: true,
              percent: 100,
            } : null)
            abortRef.current = null
            // Notify caller to re-fetch the model list.
            onSuccess()
            return
          }

          // Downloading chunk — compute progress percentage.
          // WHY we guard against division by zero: the first few chunks
          // have status "pulling manifest" with no total field.
          const total     = chunk.total     ?? 0
          const completed = chunk.completed ?? 0
          const percent   = total > 0 ? Math.round((completed / total) * 100) : 0

          setProgress({
            status:         chunk.status ?? '',
            percent,
            totalBytes:     total,
            completedBytes: completed,
            isPulling:      true,
            isSuccess:      false,
            error:          null,
          })
        }
      }
    } catch (err) {
      if (err instanceof Error && err.name === 'AbortError') return

      const msg = err instanceof Error ? err.message : 'Pull failed'
      setProgress(p => p ? { ...p, isPulling: false, error: msg } : null)
      abortRef.current = null
    }
  }, [onSuccess, config.ollamaHost])

  return { pull, cancel, progress }
}
