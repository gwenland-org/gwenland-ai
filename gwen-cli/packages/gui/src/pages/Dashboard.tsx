import Topbar from "../components/Topbar";
import StatCard from "../components/StatCard";
import RecentSessions from "../components/RecentSessions";
import { useModels } from "../hooks/useModels";
import { useProxyHealth } from "../hooks/useProxyHealth";

interface DashboardProps {
  onNavigate?: (page: string) => void
}

export default function Dashboard({ onNavigate }: DashboardProps) {
  const { activeModel, models } = useModels()
  const { alive } = useProxyHealth()

  return (
    <>
      <Topbar />

      {/* Stat cards row */}
      <div
        style={{
          marginTop: 12,
          padding: "0 24px",
          display: "grid",
          gridTemplateColumns: "repeat(3, 1fr)",
          gap: 12,
        }}
      >
        <StatCard
          label="Binary Size"
          value="8.3 MB"
          sub="gwenland · x86 release"
        />
        <StatCard
          label="Cold Start"
          value="2.6ms"
          sub="hyperfine · 100 runs"
          accent
        />
        <StatCard
          label="Active Model"
          value={activeModel ?? '—'}
          sub={alive ? 'ollama · running' : 'ollama · offline'}
        />
        <StatCard
          label="Models Installed"
          value={String(models.length)}
          sub="local · ollama"
        />
        <StatCard
          label="Proxy Status"
          value={alive ? 'Online' : 'Offline'}
          sub={alive ? 'gwen serve · active' : 'run gwen serve'}
        />
      </div>

      <RecentSessions onNewChat={() => onNavigate?.('chat')} />
    </>
  );
}
