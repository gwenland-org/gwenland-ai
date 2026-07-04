import { useState, type CSSProperties } from "react";
import {
  PanelLeft,
  LayoutDashboard,
  MessageSquare,
  Box,
  Activity,
  Database,
  FlaskConical,
  HeartPulse,
  Settings,
  type LucideIcon,
} from "lucide-react";

type NavItem = {
  id: string;
  label: string;
  icon: LucideIcon;
  badge?: boolean;
};

type NavGroup = {
  group?: string;
  items: NavItem[];
};

const NAV: NavGroup[] = [
  {
    items: [
      { id: "dashboard", label: "Dashboard", icon: LayoutDashboard },
      { id: "chat", label: "Chat", icon: MessageSquare },
    ],
  },
  {
    group: "AI Toolkit",
    items: [
      { id: "models", label: "Models", icon: Box },
      { id: "train", label: "Train", icon: Activity },
      { id: "dataset", label: "Dataset", icon: Database },
      { id: "eval", label: "Eval", icon: FlaskConical },
    ],
  },
  {
    group: "System",
    items: [{ id: "doctor", label: "Doctor", icon: HeartPulse, badge: true }],
  },
];

interface SidebarProps {
  collapsed: boolean;
  onToggle: () => void;
  active: string;
  onSelect: (id: string) => void;
}

export default function Sidebar({
  collapsed,
  onToggle,
  active,
  onSelect,
}: SidebarProps) {
  const shell: CSSProperties = {
    position: "fixed",
    top: 12,
    bottom: 12,
    left: 12,
    width: collapsed ? 48 : 220,
    background: "var(--sidebar)",
    borderRadius: "var(--radius-sidebar)",
    overflow: "hidden",
    transition: "width 200ms ease",
    display: "flex",
    flexDirection: "column",
  };

  return (
    <aside style={shell}>
      {/* Collapse toggle */}
      <button
        onClick={onToggle}
        aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
        style={{
          display: "flex",
          alignItems: "center",
          padding: "8px 12px",
          background: "transparent",
          border: "none",
          color: "var(--text-secondary)",
        }}
      >
        <PanelLeft size={16} />
      </button>

      {/* Nav */}
      <nav style={{ padding: "0 6px", flex: 1, overflowY: "auto" }}>
        {NAV.map((section, i) => (
          <div key={section.group ?? `group-${i}`}>
            {i > 0 && <Divider />}
            {section.group && !collapsed && (
              <GroupLabel>{section.group}</GroupLabel>
            )}
            {section.items.map((item) => (
              <NavRow
                key={item.id}
                item={item}
                collapsed={collapsed}
                active={active === item.id}
                onClick={() => onSelect(item.id)}
              />
            ))}
          </div>
        ))}
      </nav>

      {/* Bottom — Settings */}
      <div style={{ padding: "0 6px 16px" }}>
        <NavRow
          item={{ id: "settings", label: "Settings", icon: Settings }}
          collapsed={collapsed}
          active={active === "settings"}
          onClick={() => onSelect("settings")}
        />
      </div>
    </aside>
  );
}

function Divider() {
  return (
    <div
      style={{
        height: 1,
        background: "var(--border)",
        margin: "8px 12px",
      }}
    />
  );
}

function GroupLabel({ children }: { children: React.ReactNode }) {
  return (
    <div
      style={{
        padding: "12px 16px 4px",
        fontSize: "var(--text-xs)",
        fontWeight: 500,
        textTransform: "uppercase",
        letterSpacing: "0.08em",
        color: "var(--text-secondary)",
        opacity: 0.6,
      }}
    >
      {children}
    </div>
  );
}

function NavRow({
  item,
  collapsed,
  active,
  onClick,
}: {
  item: NavItem;
  collapsed: boolean;
  active: boolean;
  onClick: () => void;
}) {
  const [hover, setHover] = useState(false);
  const Icon = item.icon;

  const color = active ? "var(--orange)" : "var(--text-primary)";

  return (
    <button
      onClick={onClick}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      title={collapsed ? item.label : undefined}
      style={{
        position: "relative",
        display: "flex",
        alignItems: "center",
        gap: 8,
        width: "100%",
        padding: "6px 12px",
        borderRadius: "var(--radius-input)",
        background: active
          ? "var(--orange-dim)"
          : hover
            ? "rgba(255,255,255,0.03)"
            : "transparent",
        border: "none",
        textAlign: "left",
        transition: "background 120ms ease",
      }}
    >
      <Icon size={16} color={color} style={{ flexShrink: 0 }} />
      {!collapsed && (
        <span
          style={{
            fontSize: "var(--text-sm)",
            fontWeight: 400,
            color,
            whiteSpace: "nowrap",
          }}
        >
          {item.label}
        </span>
      )}
      {item.badge && !collapsed && (
        // "Soon" replaces the live green dot — Doctor has no live data yet.
        // A green dot would imply health status is being monitored, which it isn't.
        <span
          style={{
            position: "absolute",
            right: 10,
            top: "50%",
            transform: "translateY(-50%)",
            fontSize: '10px',
            background: 'oklch(100% 0 0 / 5%)',
            color: 'oklch(100% 0 0 / 35%)',
            border: '1px solid oklch(100% 0 0 / 10%)',
            borderRadius: '8px',
            padding: '1px 6px',
          }}
        >
          Soon
        </span>
      )}
    </button>
  );
}
