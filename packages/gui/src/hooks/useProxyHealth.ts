// useProxyHealth.ts — polls the GwenLand proxy to determine if it is reachable.
//
// WHY poll instead of a persistent WebSocket:
//   The proxy is a local Actix-web process that may start and stop while the
//   GUI is open. A poll is simpler and sufficient — 3s interval means the UI
//   reflects reality within 3 seconds of the proxy going up or down.
//
// WHY treat a 404 as "alive":
//   If the /health route is not yet implemented on the proxy, the fetch will
//   return 404 — which is still a valid HTTP response, meaning the proxy
//   process is running and the TCP port is open. Only a network-level failure
//   (ECONNREFUSED, timeout) means the proxy is truly offline.
//
// WHY 3s interval:
//   Fast enough to feel responsive; slow enough to not hammer localhost.
//   The spec explicitly forbids polling faster than 3s.
//
// WHY alive starts as true (optimistic):
//   The first poll result arrives after POLL_INTERVAL_MS. Starting false
//   would flash the "offline" banner and disable the send button for up to
//   3s on every app open — even when the proxy is running. Optimistic true
//   hides that false negative; confirmed-offline flips it to false.

import { useState, useEffect } from 'react'
import { useConfig } from '../context/ConfigContext'

const POLL_INTERVAL_MS = 3000
const TIMEOUT_MS = 2000

interface ProxyHealth {
  alive: boolean
  latencyMs: number | null
}

export function useProxyHealth(): ProxyHealth {
  // Optimistic default: assume proxy is up until the first check says otherwise.
  const [alive, setAlive] = useState(true)
  const [latencyMs, setLatencyMs] = useState<number | null>(null)
  const { config } = useConfig()

  useEffect(() => {
    let cancelled = false
    const healthUrl = `http://127.0.0.1:${config.proxyPort}/health`

    async function check() {
      const t = performance.now()
      try {
        const res = await fetch(healthUrl, {
          method: 'GET',
          signal: AbortSignal.timeout(TIMEOUT_MS),
        })
        if (cancelled) return

        // Any HTTP response (including 404) means the process is listening.
        // A network-level error (ECONNREFUSED) lands in the catch block below.
        const elapsed = Math.round(performance.now() - t)
        if (res.ok || res.status === 404) {
          setAlive(true)
          setLatencyMs(elapsed)
        } else {
          // 5xx from the proxy itself — process is up but unhealthy.
          // Still mark alive so the user can try sending; the proxy error
          // will surface in the chat stream if it persists.
          setAlive(true)
          setLatencyMs(elapsed)
        }
      } catch {
        if (cancelled) return
        // ECONNREFUSED or timeout — proxy is not running.
        setAlive(false)
        setLatencyMs(null)
      }
    }

    check()
    const id = setInterval(check, POLL_INTERVAL_MS)
    return () => {
      cancelled = true
      clearInterval(id)
    }
  }, [config.proxyPort])

  return { alive, latencyMs }
}
