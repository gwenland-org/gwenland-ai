import { type CSSProperties } from "react";
import { Plus } from "lucide-react";

type Session = {
  name: string;
  context: string;
  time: string;
};

const SESSIONS: Session[] = [
  { name: "Tauri IPC design", context: "3,412 ctx", time: "12m ago" },
  { name: "Training run #4", context: "8,901 ctx", time: "3h ago" },
  { name: "Model eval notes", context: "1,204 ctx", time: "1d ago" },
];

const GRID = "1fr 120px 80px";

const sectionLabel: CSSProperties = {
  fontSize: "var(--text-xs)",
  fontWeight: 500,
  textTransform: "uppercase",
  letterSpacing: "0.06em",
  color: "var(--text-secondary)",
};

export default function RecentSessions({
  onNewChat,
}: {
  onNewChat?: () => void;
}) {
  return (
    <section>
      {/* Section header */}
      <div
        style={{
          marginTop: 24,
          padding: "0 24px",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
        }}
      >
        <span style={sectionLabel}>Recent Sessions</span>
        <NewChatButton onClick={onNewChat} />
      </div>

      {/* Table */}
      <div style={{ padding: "0 24px" }}>
        {/* Header row */}
        <div
          style={{
            marginTop: 12,
            padding: "0 12px 6px",
            display: "grid",
            gridTemplateColumns: GRID,
            borderBottom: "1px solid var(--border)",
          }}
        >
          <span style={sectionLabel}>Name</span>
          <span style={{ ...sectionLabel, textAlign: "right" }}>Context</span>
          <span style={{ ...sectionLabel, textAlign: "right" }}>Time</span>
        </div>

        {/* Rows */}
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            gap: 2,
            marginTop: 2,
          }}
        >
          {SESSIONS.map((s) => (
            <SessionRow key={s.name} session={s} />
          ))}
        </div>
      </div>
    </section>
  );
}

function NewChatButton({ onClick }: { onClick?: () => void }) {
  return (
    <button
      onClick={onClick}
      style={{
        background: "var(--orange)",
        color: "#0f0d1a",
        fontSize: "var(--text-sm)",
        fontWeight: 500,
        borderRadius: "var(--radius-btn)",
        padding: "6px 12px",
        border: "none",
        display: "flex",
        alignItems: "center",
        gap: 4,
      }}
    >
      <Plus size={14} />
      New Chat
    </button>
  );
}

function SessionRow({ session }: { session: Session }) {
  // Option A: no onClick yet (session loading from JSONL ships in a later cycle).
  // Remove pointer cursor and hover state to avoid implying interactivity.
  return (
    <div
      aria-label={`Session: ${session.name}`}
      style={{
        display: "grid",
        gridTemplateColumns: GRID,
        alignItems: "center",
        padding: "8px 12px",
        borderRadius: "var(--radius-btn)",
        background: "transparent",
        opacity: 0.5,
      }}
    >
      <span
        style={{
          fontSize: "var(--text-sm)",
          fontWeight: 400,
          color: "var(--text-primary)",
        }}
      >
        {session.name}
      </span>
      <span
        style={{
          fontSize: "var(--text-xs)",
          fontFamily: "'Geist Mono', monospace",
          color: "var(--text-mono)",
          textAlign: "right",
        }}
      >
        {session.context}
      </span>
      <span
        style={{
          fontSize: "var(--text-xs)",
          fontFamily: "'Geist Mono', monospace",
          color: "var(--text-secondary)",
          textAlign: "right",
        }}
      >
        {session.time}
      </span>
    </div>
  );
}
