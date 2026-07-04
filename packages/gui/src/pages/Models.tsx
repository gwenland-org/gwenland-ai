// Models.tsx — Model Manager screen, rendered when active === 'models'.
//
// WHY this page composes hooks and passes callbacks down rather than
// using a shared context:
//   Context is warranted when many unrelated components need the same data.
//   Here the data flow is linear: hooks → Models.tsx → child components.
//   Prop drilling one level is simpler and more traceable than a context.
//
// WHY showPull defaults to false (panel hidden):
//   The panel takes vertical space away from the model list. Most visits
//   to the Models page are to browse, not to pull. The user explicitly
//   opens the panel via the topbar button (or via ModelPicker navigation).

import { useState } from 'react'
import { Brain, Download, Search, X } from 'lucide-react'
import { useModels } from '../hooks/useModels'
import { usePullModel } from '../hooks/usePullModel'
import { useDeleteModel } from '../hooks/useDeleteModel'
import RegistryTabs, { type Tab } from '../components/models/RegistryTabs'
import PullPanel from '../components/models/PullPanel'
import ModelCard from '../components/models/ModelCard'
import ModelInfoDrawer from '../components/models/ModelInfoDrawer'
import ModelStatsBar from '../components/models/ModelStatsBar'
import type { OllamaModel } from '../types/chat'
import type { ToastVariant } from '../hooks/useToast'

interface ModelsProps {
  // Optional: if true the pull panel opens immediately on mount.
  // Set by App.tsx when navigating here via "Pull model…" in ModelPicker.
  openPull?: boolean
  toast?: (message: string, variant?: ToastVariant) => void
}

export default function Models({ openPull = false, toast }: ModelsProps) {
  const { models, activeModel, setActiveModel, refetch } = useModels()
  const { pull, cancel, progress } = usePullModel(refetch)
  const { deleteModel, deleting } = useDeleteModel(
    refetch,
    msg => toast?.(msg, 'error'),
  )

  const [tab, setTab]             = useState<Tab>('local')
  const [search, setSearch]       = useState('')
  const [showPull, setShowPull]   = useState(openPull)
  const [infoModel, setInfoModel] = useState<OllamaModel | null>(null)

  // Filter by search term — case-insensitive substring match on model name.
  const filtered = models.filter(m =>
    m.name.toLowerCase().includes(search.toLowerCase())
  )

  const hasModels   = models.length > 0
  const hasFiltered = filtered.length > 0

  return (
    // WHY position: relative — ModelInfoDrawer uses position: absolute and
    // must be constrained to this container, not the viewport.
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        height: '100vh',
        background: 'var(--bg)',
        position: 'relative',
        overflow: 'hidden',
      }}
    >
      {/* ── Topbar ── */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 10,
          padding: '0 16px',
          height: 48,
          borderBottom: '1px solid var(--border)',
          flexShrink: 0,
        }}
      >
        <Brain size={15} color="rgba(255,140,66,0.60)" />
        <span style={{ fontSize: 13, fontWeight: 500, color: 'var(--text-primary)', fontFamily: "'Geist', sans-serif" }}>
          Model Manager
        </span>

        <div style={{ marginLeft: 'auto' }}>
          <button
            onClick={() => setShowPull(p => !p)}
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: 5,
              fontSize: 12,
              fontWeight: 500,
              background: showPull ? 'rgba(255,140,66,0.15)' : 'rgba(255,140,66,0.08)',
              border: '1px solid rgba(255,140,66,0.20)',
              borderRadius: 7,
              padding: '5px 11px',
              color: 'var(--orange)',
              cursor: 'pointer',
              fontFamily: "'Geist', sans-serif",
              transition: 'background 120ms ease',
            }}
          >
            <Download size={13} />
            Pull Model
          </button>
        </div>
      </div>

      {/* ── Search bar ── */}
      <div
        style={{
          padding: '10px 16px',
          borderBottom: '1px solid rgba(255,255,255,0.05)',
          flexShrink: 0,
        }}
      >
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 8,
            background: 'rgba(255,255,255,0.04)',
            border: '1px solid var(--border)',
            borderRadius: 8,
            padding: '6px 10px',
          }}
        >
          <Search size={13} color="var(--text-secondary)" style={{ flexShrink: 0 }} />
          <input
            value={search}
            onChange={e => setSearch(e.target.value)}
            placeholder="Search models…"
            style={{
              flex: 1,
              background: 'transparent',
              border: 'none',
              outline: 'none',
              fontSize: 12,
              color: 'var(--text-primary)',
              fontFamily: "'Geist', sans-serif",
            }}
          />
          {search && (
            <button
              onClick={() => setSearch('')}
              style={{ background: 'transparent', border: 'none', cursor: 'pointer', display: 'flex' }}
            >
              <X size={12} color="var(--text-secondary)" />
            </button>
          )}
        </div>
      </div>

      {/* ── Registry tabs (Local / HuggingFace / Recommended) ── */}
      <RegistryTabs active={tab} onChange={setTab} />

      {/* ── Scrollable content area ── */}
      <div
        style={{
          flex: 1,
          overflowY: 'auto',
          padding: '12px 16px',
          display: 'flex',
          flexDirection: 'column',
          gap: 8,
          scrollbarWidth: 'thin',
          scrollbarColor: 'rgba(255,255,255,0.08) transparent',
        }}
      >
        {/* Pull panel — collapsible */}
        {showPull && (
          <PullPanel
            onPull={pull}
            progress={progress}
            onCancel={cancel}
            onClose={() => setShowPull(false)}
          />
        )}

        {/* Empty state — no models installed at all */}
        {!hasModels && !progress?.isPulling && (
          <div
            style={{
              flex: 1,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              paddingTop: 48,
              paddingBottom: 48,
              gap: 12,
              textAlign: 'center',
            }}
          >
            <Brain size={32} color="rgba(255,140,66,0.20)" />
            <p style={{ fontSize: 13, color: 'var(--text-secondary)', fontFamily: "'Geist', sans-serif" }}>
              No models installed yet
            </p>
            <button
              onClick={() => setShowPull(true)}
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: 5,
                fontSize: 12,
                background: 'rgba(255,140,66,0.08)',
                border: '1px solid rgba(255,140,66,0.20)',
                borderRadius: 7,
                padding: '6px 14px',
                color: 'var(--orange)',
                cursor: 'pointer',
                fontFamily: "'Geist', sans-serif",
              }}
            >
              <Download size={12} />
              Pull your first model
            </button>
          </div>
        )}

        {/* Empty state — search no results */}
        {hasModels && !hasFiltered && (
          <p style={{ textAlign: 'center', fontSize: 12, color: 'var(--text-secondary)', padding: '32px 0', fontFamily: "'Geist', sans-serif" }}>
            No models match &ldquo;<span style={{ color: 'var(--text-primary)' }}>{search}</span>&rdquo;
          </p>
        )}

        {/* Model cards */}
        {filtered.map(model => (
          <ModelCard
            key={model.name}
            model={model}
            isActive={model.name === activeModel}
            isDeleting={deleting === model.name}
            onSetActive={setActiveModel}
            onDelete={deleteModel}
            onInfo={setInfoModel}
          />
        ))}
      </div>

      {/* ── Stats strip ── */}
      <ModelStatsBar models={models} activeModel={activeModel} />

      {/* ── Info drawer — position: absolute, slides in from right ── */}
      <ModelInfoDrawer
        model={infoModel}
        onClose={() => setInfoModel(null)}
        onSetActive={setActiveModel}
      />
    </div>
  )
}
