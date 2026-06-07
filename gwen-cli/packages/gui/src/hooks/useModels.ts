// useModels.ts — fetches and manages the list of locally available Ollama models.
//
// WHY this hook is shared by Chat and Model Manager:
//   Both screens need the model list and active model selection. A single
//   hook instance means one HTTP call instead of two, and setActiveModel
//   changes are reflected everywhere that uses the hook.
//
// NOTE: React does not share hook instances across component trees —
//   if Chat and Models are both mounted, they each hold independent state.
//   This is acceptable because only one screen is visible at a time.
//   If cross-screen sync becomes a requirement, lift to a context.
//
// WHY ollamaHost directly (not via the GwenLand proxy):
//   /api/tags is read-only model metadata, not a chat stream. No proxy
//   overhead is needed. The proxy is for /chat only.

import { useState, useEffect, useCallback, useMemo } from 'react'
import type { OllamaModel } from '../types/chat'
import { inferContextLength, parseParamCount } from '../utils/modelUtils'
import { useConfig } from '../context/ConfigContext'

const DEFAULT_MODEL = 'qwen3:8b'

// Raw shape returned by Ollama /api/tags — fields we care about.
interface RawOllamaModel {
  name: string
  size: number
  digest: string
  modified_at: string
  details?: {
    quantization_level?: string
    parameter_size?: string
  }
}

function parseModel(raw: RawOllamaModel, activeModel: string): OllamaModel {
  return {
    name:          raw.name,
    size:          raw.size ?? 0,
    digest:        raw.digest ?? '',
    quantization:  raw.details?.quantization_level ?? 'unknown',
    paramCount:    parseParamCount(raw.details?.parameter_size),
    contextLength: inferContextLength(raw.name),
    isActive:      raw.name === activeModel,
    isLocal:       true,
    modifiedAt:    raw.modified_at ?? new Date().toISOString(),
    isOnline:      true,
  }
}

interface UseModelsResult {
  models: OllamaModel[]
  activeModel: string
  setActiveModel: (name: string) => void
  isLoading: boolean
  refetch: () => void
}

export function useModels(): UseModelsResult {
  const [rawModels, setRawModels] = useState<RawOllamaModel[]>([])
  const [activeModel, setActiveModel] = useState<string>(DEFAULT_MODEL)
  const [isLoading, setIsLoading] = useState(true)
  const { config } = useConfig()

  const fetchModels = useCallback(async () => {
    const tagsUrl = `http://${config.ollamaHost}/api/tags`
    setIsLoading(true)
    try {
      const res = await fetch(tagsUrl, { signal: AbortSignal.timeout(4000) })
      if (!res.ok) throw new Error(`Ollama returned ${res.status}`)
      const json = await res.json()
      const list: RawOllamaModel[] = json.models ?? []
      setRawModels(list)
      // Auto-select first model only when the user has no active selection yet.
      // WHY guard on prev: if the user already picked a model, don't clobber it
      // with whatever Ollama returns first — that would reset their choice on
      // every refetch (e.g. after pulling a new model).
      if (list.length > 0) {
        setActiveModel(prev =>
          list.some(m => m.name === prev) ? prev : list[0].name
        )
      }
    } catch {
      // Ollama unreachable — show a single offline placeholder so the UI
      // is never completely empty. The name matches DEFAULT_MODEL so any
      // cached chat history still references a valid model name.
      setRawModels([{
        name:        DEFAULT_MODEL,
        size:        0,
        digest:      '',
        modified_at: new Date().toISOString(),
        details:     { quantization_level: 'unknown', parameter_size: undefined },
      }])
    } finally {
      setIsLoading(false)
    }
  }, [config.ollamaHost])

  useEffect(() => {
    fetchModels()
  }, [fetchModels])

  // Derive the final model list with isActive each render.
  // WHY useMemo: the map is cheap but avoids allocating a new array on every
  // render when neither rawModels nor activeModel has changed.
  const models: OllamaModel[] = useMemo(
    () => rawModels.map(r => parseModel(r, activeModel)),
    [rawModels, activeModel],
  )

  return { models, activeModel, setActiveModel, isLoading, refetch: fetchModels }
}
