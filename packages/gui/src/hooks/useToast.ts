// useToast.ts — lightweight in-memory toast state.
//
// WHY hand-rolled instead of an external library:
//   The spec forbids new npm packages for this cycle. The three-variant
//   system (error/success/warning) covers all current use cases without
//   a dependency.
//
// WHY auto-dismiss after 4s:
//   Errors should be visible long enough to read but not block the UI.
//   4s matches common toast library defaults.

import { useState, useCallback } from 'react'

export type ToastVariant = 'error' | 'success' | 'warning'

export interface Toast {
  id: string
  message: string
  variant: ToastVariant
}

export function useToast() {
  const [toasts, setToasts] = useState<Toast[]>([])

  const toast = useCallback((message: string, variant: ToastVariant = 'error') => {
    const id = crypto.randomUUID()
    setToasts(prev => [...prev, { id, message, variant }])
    setTimeout(() => {
      setToasts(prev => prev.filter(t => t.id !== id))
    }, 4000)
  }, [])

  const dismiss = useCallback((id: string) => {
    setToasts(prev => prev.filter(t => t.id !== id))
  }, [])

  return { toasts, toast, dismiss }
}
