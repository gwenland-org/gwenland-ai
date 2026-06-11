// ErrorBoundary.tsx — catches unhandled React errors and shows a recovery UI.
//
// WHY a class component: React's error boundary API (getDerivedStateFromError +
// componentDidCatch) is only available on class components. There is no hooks
// equivalent in the current React stable release.
//
// WHY "Try again" resets state instead of reloading:
//   window.location.reload() would wipe the entire session (model selection,
//   chat history). Resetting error state lets the same component tree re-mount
//   and attempt recovery without losing context.

import { Component, type ErrorInfo, type ReactNode } from 'react'

interface Props { children: ReactNode }
interface State { error: Error | null }

export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null }

  static getDerivedStateFromError(error: Error): State {
    return { error }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('[GwenLand] Uncaught error:', error, info)
  }

  render() {
    if (this.state.error) {
      return (
        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            height: '100vh',
            gap: '12px',
            background: 'var(--bg)',
            color: 'var(--text-primary)',
            textAlign: 'center',
            padding: '32px',
          }}
        >
          <span style={{ fontSize: '32px' }}>⚠</span>
          <p style={{ fontSize: '14px', fontWeight: 600, margin: 0 }}>
            Something went wrong
          </p>
          <p
            style={{
              fontSize: '12px',
              color: 'var(--text-muted)',
              maxWidth: '360px',
              margin: 0,
            }}
          >
            {this.state.error.message}
          </p>
          <button
            onClick={() => this.setState({ error: null })}
            style={{
              marginTop: '8px',
              padding: '6px 16px',
              borderRadius: '7px',
              border: '1px solid oklch(75% 0.18 48 / 25%)',
              background: 'oklch(75% 0.18 48 / 10%)',
              color: 'oklch(75% 0.18 48)',
              fontSize: '12px',
              cursor: 'pointer',
            }}
          >
            Try again
          </button>
        </div>
      )
    }
    return this.props.children
  }
}
