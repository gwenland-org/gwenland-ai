interface StatCardProps {
  label: string;
  value: string;
  sub: string;
  accent?: boolean;
}

export default function StatCard({ label, value, sub, accent }: StatCardProps) {
  return (
    <div
      style={{
        background: "var(--card)",
        borderRadius: "var(--radius-card)",
        padding: "16px 20px",
        display: "flex",
        flexDirection: "column",
        gap: 4,
      }}
    >
      <span
        style={{
          fontSize: "var(--text-xs)",
          fontWeight: 500,
          textTransform: "uppercase",
          letterSpacing: "0.06em",
          color: "var(--text-secondary)",
        }}
      >
        {label}
      </span>
      <span
        style={{
          fontSize: "var(--text-stat)",
          fontWeight: 600,
          color: accent ? "var(--orange)" : "var(--text-primary)",
          lineHeight: 1.1,
        }}
      >
        {value}
      </span>
      <span
        style={{
          fontSize: "var(--text-xs)",
          fontFamily: "'Geist Mono', monospace",
          color: "var(--text-secondary)",
          opacity: 0.5,
        }}
      >
        {sub}
      </span>
    </div>
  );
}
