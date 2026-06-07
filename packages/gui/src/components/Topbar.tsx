import { useState, type CSSProperties } from "react";
import { RotateCcw, Minus, Square, X, type LucideIcon } from "lucide-react";
import { getCurrentWindow } from "@tauri-apps/api/window";

interface TopbarProps {
  onRefresh?: () => void;
}

const win = getCurrentWindow();

// Sits directly on --bg — NOT a separate surface.
// The whole bar is the window drag handle (data-tauri-drag-region).
export default function Topbar({ onRefresh }: TopbarProps) {
  const [refreshHover, setRefreshHover] = useState(false);

  return (
    <header
      data-tauri-drag-region="true"
      style={{
        height: 48,
        padding: "0 24px",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        userSelect: "none",
      }}
    >
      {/* Left — breadcrumb */}
      <div
        data-tauri-drag-region="true"
        style={{ display: "flex", alignItems: "center" }}
      >
        <span
          style={{ fontSize: "var(--text-base)", color: "var(--text-secondary)" }}
        >
          GwenLand
        </span>
        <span style={{ color: "var(--text-secondary)", margin: "0 6px" }}>›</span>
        <span
          style={{
            fontSize: "var(--text-lg)",
            fontWeight: 600,
            color: "var(--text-primary)",
          }}
        >
          Dashboard
        </span>
      </div>

      {/* Right — window controls + version + refresh */}
      <div style={{ display: "flex", alignItems: "center" }}>
        {/* Window controls (frameless) */}
        <div style={{ display: "flex", gap: 8, marginRight: 12 }}>
          <WinButton
            icon={Minus}
            label="Minimize"
            onClick={() => win.minimize()}
          />
          <WinButton
            icon={Square}
            label="Maximize"
            onClick={() => win.toggleMaximize()}
          />
          <WinButton
            icon={X}
            label="Close"
            danger
            onClick={() => win.close()}
          />
        </div>

        <span
          style={{
            fontFamily: "'Geist Mono', monospace",
            fontSize: "var(--text-xs)",
            color: "var(--text-secondary)",
            border: "1px solid rgba(255,255,255,0.08)",
            borderRadius: "var(--radius-sm)",
            padding: "2px 6px",
          }}
        >
          v{import.meta.env.TAURI_APP_VERSION ?? '1.0.0'}
        </span>
        <button
          onClick={onRefresh}
          onMouseEnter={() => setRefreshHover(true)}
          onMouseLeave={() => setRefreshHover(false)}
          aria-label="Refresh"
          style={{
            display: "flex",
            alignItems: "center",
            background: "transparent",
            border: "none",
            color: "var(--text-secondary)",
            marginLeft: 8,
            opacity: refreshHover ? 1 : 0.85,
            transition: "opacity 120ms ease",
          }}
        >
          <RotateCcw size={14} />
        </button>
      </div>
    </header>
  );
}

function WinButton({
  icon: Icon,
  label,
  onClick,
  danger,
}: {
  icon: LucideIcon;
  label: string;
  onClick: () => void;
  danger?: boolean;
}) {
  const [hover, setHover] = useState(false);

  const bg = hover
    ? danger
      ? "rgba(239,68,68,0.8)"
      : "rgba(255,255,255,0.08)"
    : "transparent";

  // 12px visual circle per spec; expand the clickable area without growing
  // the dot via a transparent box-shadow ring (hit area ~22px).
  const style: CSSProperties = {
    width: 12,
    height: 12,
    padding: 0,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    borderRadius: "50%",
    border: "none",
    background: bg,
    color: danger && hover ? "#fff" : "var(--text-secondary)",
    boxShadow: "0 0 0 5px transparent",
    transition: "background 120ms ease, color 120ms ease",
  };

  return (
    <button
      onClick={onClick}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      aria-label={label}
      title={label}
      style={style}
    >
      <Icon size={10} />
    </button>
  );
}
