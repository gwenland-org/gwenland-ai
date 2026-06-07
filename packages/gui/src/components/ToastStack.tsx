// ToastStack.tsx — renders active toasts in the bottom-right corner.
//
// WHY fixed positioning at z-index 100:
//   Toasts must appear above all other content including drawers and popovers.
//   The Models screen uses position:absolute for ModelInfoDrawer (z-index ~10),
//   so 100 guarantees visibility without an arbitrary large value.

import { AlertTriangle, CheckCircle, X } from 'lucide-react'
import type { Toast } from '../hooks/useToast'

interface Props {
  toasts: Toast[]
  onDismiss: (id: string) => void
}

export function ToastStack({ toasts, onDismiss }: Props) {
  if (!toasts.length) return null

  return (
    <div
      style={{
        position: 'fixed',
        bottom: '20px',
        right: '20px',
        display: 'flex',
        flexDirection: 'column',
        gap: '8px',
        zIndex: 100,
      }}
    >
      {toasts.map(t => (
        <div
          key={t.id}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            padding: '9px 12px',
            borderRadius: '8px',
            fontSize: '12px',
            background:
              t.variant === 'error'
                ? 'oklch(70% 0.19 22 / 12%)'
                : t.variant === 'success'
                  ? 'oklch(75% 0.18 145 / 12%)'
                  : 'oklch(75% 0.18 48 / 12%)',
            border: `1px solid ${
              t.variant === 'error'
                ? 'oklch(70% 0.19 22 / 25%)'
                : t.variant === 'success'
                  ? 'oklch(75% 0.18 145 / 25%)'
                  : 'oklch(75% 0.18 48 / 25%)'
            }`,
            color:
              t.variant === 'error'
                ? 'oklch(70% 0.19 22)'
                : t.variant === 'success'
                  ? 'oklch(75% 0.18 145)'
                  : 'oklch(75% 0.18 48)',
            maxWidth: '320px',
            boxShadow: '0 4px 16px oklch(0% 0 0 / 40%)',
            fontFamily: "'Geist', sans-serif",
          }}
        >
          {t.variant === 'error' && <AlertTriangle size={13} style={{ flexShrink: 0 }} />}
          {t.variant === 'success' && <CheckCircle size={13} style={{ flexShrink: 0 }} />}
          {t.variant === 'warning' && <AlertTriangle size={13} style={{ flexShrink: 0 }} />}
          <span style={{ flex: 1 }}>{t.message}</span>
          <button
            onClick={() => onDismiss(t.id)}
            aria-label="Dismiss notification"
            style={{
              background: 'none',
              border: 'none',
              cursor: 'pointer',
              color: 'inherit',
              opacity: 0.6,
              padding: 0,
              display: 'flex',
              alignItems: 'center',
            }}
          >
            <X size={12} />
          </button>
        </div>
      ))}
    </div>
  )
}
