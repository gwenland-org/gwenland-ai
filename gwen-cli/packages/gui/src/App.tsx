// App.tsx owns top-level navigation state and mounts the proxy auto-start effect.
//
// WHY no react-router: the app has a small, fixed set of screens driven by
// sidebar selection. Adding a URL router before we need deep-linking or
// browser history would be premature complexity.
//
// WHY `NavOptions` carries `openPull`:
//   The Chat screen's ModelPicker has a "Pull model…" item that should
//   navigate to Models AND open the pull panel in one action. Without this
//   option the user would need a second click. The option is ignored by all
//   other routes — it's a one-way signal from Chat to Models.

import { useState, useEffect } from "react";
import { Command } from "@tauri-apps/plugin-shell";
import Sidebar from "./components/Sidebar";
import Dashboard from "./pages/Dashboard";
import Chat from "./pages/Chat";
import Models from "./pages/Models";
import Settings from "./pages/Settings";
import Train from "./pages/Train";
import Dataset from "./pages/Dataset";
import Eval from "./pages/Eval";
import Doctor from "./pages/Doctor";
import { useConfig } from "./context/ConfigContext";
import { useToast } from "./hooks/useToast";
import { ToastStack } from "./components/ToastStack";

interface NavOptions {
  openPull?: boolean
}

export default function App() {
  const [collapsed, setCollapsed] = useState(false);
  const [active, setActive] = useState("dashboard");
  const [navOptions, setNavOptions] = useState<NavOptions>({});
  const { config } = useConfig();
  const { toasts, toast, dismiss } = useToast();

  // Auto-start the GwenLand proxy on app open when the user has opted in.
  // WHY run once on mount only: the effect must not re-fire when config changes
  // mid-session (e.g. user toggles autoStartProxy in Settings). Restarting the
  // proxy unexpectedly would disrupt any in-flight chat stream.
  useEffect(() => {
    if (config.autoStartProxy) {
      Command.sidecar('binaries/gwen', ['serve']).spawn().catch(() => {
        // EADDRINUSE means the proxy is already running — not an error.
        // Any other spawn failure is also non-fatal; the user will see the
        // "proxy offline" banner in Chat if the proxy isn't reachable.
      })
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []) // intentionally run once on mount

  function navigateTo(page: string, opts: NavOptions = {}) {
    setActive(page);
    setNavOptions(opts);
  }

  // Shared layout wrapper — sidebar + content area with animated margin.
  function Shell({ children }: { children: React.ReactNode }) {
    return (
      <div style={{ height: "100vh", background: "var(--bg)" }}>
        <Sidebar
          collapsed={collapsed}
          onToggle={() => setCollapsed((c) => !c)}
          active={active}
          onSelect={(id) => navigateTo(id)}
        />
        <main
          style={{
            marginLeft: collapsed ? 72 : 244,
            height: "100vh",
            background: "var(--bg)",
            transition: "margin-left 200ms ease",
            overflow: "hidden",
          }}
        >
          {children}
        </main>
        <ToastStack toasts={toasts} onDismiss={dismiss} />
      </div>
    );
  }

  if (active === "chat") {
    return (
      <Shell>
        {/* onNavigate is threaded through Chat → ChatInput → ModelPicker so
            "Pull model…" can navigate here with the pull panel pre-opened. */}
        <Chat onNavigate={navigateTo} />
      </Shell>
    );
  }

  if (active === "models") {
    return (
      <Shell>
        <Models openPull={navOptions.openPull} toast={toast} />
      </Shell>
    );
  }

  if (active === "settings") {
    return (
      <Shell>
        <Settings toast={toast} />
      </Shell>
    );
  }

  if (active === "train") {
    return (
      <Shell>
        <Train />
      </Shell>
    );
  }

  if (active === "dataset") {
    return (
      <Shell>
        <Dataset />
      </Shell>
    );
  }

  if (active === "eval") {
    return (
      <Shell>
        <Eval />
      </Shell>
    );
  }

  if (active === "doctor") {
    return (
      <Shell>
        <Doctor />
      </Shell>
    );
  }

  return (
    <Shell>
      <Dashboard onNavigate={navigateTo} />
    </Shell>
  );
}
