// useDeleteModel.ts — sends a DELETE request to Ollama to remove a local model.
//
// WHY Ollama DELETE and not a proxy call:
//   Model management operations (delete, pull) go directly to the Ollama API.
//   The GwenLand proxy at config.proxyPort handles chat streams only.
//
// WHY `deleting` is a model name string (not a boolean):
//   Multiple model cards can be rendered simultaneously. Using the name
//   lets each card independently check whether it is the one being deleted
//   and show a spinner, without needing global state.
//
// WHY an onSuccess callback:
//   After deletion, the model list in useModels must be refreshed. Passing
//   a callback keeps this hook independent of useModels — it doesn't need
//   to import or call it directly.
//
// WHY onError callback instead of console.error:
//   Silent console errors are invisible to users. The caller (Models.tsx)
//   is responsible for displaying the error via the toast system.

import { useState, useCallback } from 'react'
import { useConfig } from '../context/ConfigContext'

interface UseDeleteModelResult {
  deleteModel: (name: string) => Promise<void>
  deleting: string | null   // name of the model currently being deleted
}

export function useDeleteModel(
  onSuccess: () => void,
  onError?: (msg: string) => void,
): UseDeleteModelResult {
  // null when idle; set to the model name while the DELETE is in-flight.
  const [deleting, setDeleting] = useState<string | null>(null)
  const { config } = useConfig()

  const deleteModel = useCallback(async (name: string) => {
    if (deleting) return   // guard: don't allow concurrent deletes
    const deleteUrl = `http://${config.ollamaHost}/api/delete`
    setDeleting(name)
    try {
      const res = await fetch(deleteUrl, {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name }),
        signal: AbortSignal.timeout(10_000),
      })
      if (!res.ok) throw new Error(`Ollama DELETE returned ${res.status}`)
      onSuccess()
    } catch (err) {
      // Surface the error to the caller for toast display rather than swallowing.
      // The UI removes the card optimistically; if the delete failed, the next
      // refetch will restore the card.
      onError?.('Failed to delete model — check Ollama is running')
    } finally {
      setDeleting(null)
    }
  }, [deleting, onSuccess, onError, config.ollamaHost])

  return { deleteModel, deleting }
}
