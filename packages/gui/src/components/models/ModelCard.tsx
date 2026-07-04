// ModelCard.tsx — a single row card representing one installed Ollama model.
//
// WHY the delete button is fully disabled (not just visually dimmed) for the
// active model: deleting the active model would leave the chat screen with an
// invalid model reference. The tooltip explains the constraint to the user.
//
// WHY icon + tint vary by model family:
//   Visual differentiation helps users scan a long list quickly.
//   gwen-* models are GwenLand-trained → violet (branded).
//   qwen* are the recommended default → orange (primary).
//   Others are neutral → muted white.

import { useState } from 'react'
import { Sparkles, Star, Brain, Info, Trash2 } from 'lucide-react'
import type { OllamaModel } from '../../types/chat'
import { formatBytes, formatCtx, modelAccent } from '../../utils/modelUtils'
import type { ModelAccent } from '../../utils/modelUtils'

interface ModelCardProps {
  model: OllamaModel
  isActive: boolean
  isDeleting: boolean
  onSetActive: (name: string) => void
  onDelete: (name: string) => void
  onInfo: (model: OllamaModel) => void
}

// ── Accent palette ───────────────────────────────────────────────────────────
// Each accent defines: icon background, icon border, icon text color, and the
// card border color when this model is active.

const ACCENT_STYLES: Record<ModelAccent, {
  iconBg: string; iconBorder: string; iconColor: string
  badge: { bg: string; border: string; color: string }
}> = {
  orange: {
    iconBg:     'rgba(255,140,66,0.08)',
    iconBorder: 'rgba(255,140,66,0.15)',
    iconColor:  'var(--orange)',
    badge: { bg: 'rgba(255,140,66,0.08)', border: 'rgba(255,140,66,0.15)', color: 'rgba(255,140,66,0.70)' },
  },
  violet: {
    iconBg:     'rgba(139,92,246,0.08)',
    iconBorder: 'rgba(139,92,246,0.20)',
    iconColor:  '#a78bfa',
    badge: { bg: 'rgba(139,92,246,0.10)', border: 'rgba(139,92,246,0.20)', color: '#a78bfa' },
  },
  muted: {
    iconBg:     'rgba(255,255,255,0.05)',
    iconBorder: 'rgba(255,255,255,0.08)',
    iconColor:  'rgba(255,255,255,0.40)',
    badge: { bg: 'rgba(255,255,255,0.05)', border: 'rgba(255,255,255,0.08)', color: 'rgba(255,255,255,0.40)' },
  },
}

function ModelIcon({ accent }: { accent: ModelAccent }) {
  const { iconBg, iconBorder, iconColor } = ACCENT_STYLES[accent]
  const Icon = accent === 'violet' ? Star : accent === 'orange' ? Sparkles : Brain

  return (
    <div
      style={{
        width: 36, height: 36,
        borderRadius: 8,
        background: iconBg,
        border: `1px solid ${iconBorder}`,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        flexShrink: 0,
      }}
    >
      <Icon size={16} color={iconColor} />
    </div>
  )
}

function Badge({
  children,
  bg, border, color,
}: {
  children: React.ReactNode
  bg: string; border: string; color: string
}) {
  return (
    <span
      style={{
        fontSize: 10,
        background: bg,
        border: `1px solid ${border}`,
        color,
        borderRadius: 4,
        padding: '1px 6px',
        fontFamily: "'Geist', sans-serif",
        whiteSpace: 'nowrap',
      }}
    >
      {children}
    </span>
  )
}

export default function ModelCard({
  model,
  isActive,
  isDeleting,
  onSetActive,
  onDelete,
  onInfo,
}: ModelCardProps) {
  const [hover, setHover] = useState(false)
  const accent = modelAccent(model.name)
  const styles = ACCENT_STYLES[accent]

  // Card border changes on hover or when this is the active model.
  const borderColor = isActive
    ? 'rgba(255,140,66,0.30)'
    : hover
      ? 'rgba(255,140,66,0.20)'
      : 'rgba(255,255,255,0.07)'

  const cardBg = isActive
    ? 'rgba(255,140,66,0.05)'
    : 'rgba(255,255,255,0.03)'

  return (
    <div
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 12,
        padding: '12px 14px',
        borderRadius: 10,
        background: cardBg,
        border: `1px solid ${borderColor}`,
        transition: 'border-color 150ms ease, background 150ms ease',
      }}
    >
      <ModelIcon accent={accent} />

      {/* ── Name + badges ── */}
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap', marginBottom: 4 }}>
          <span style={{ fontSize: 13, fontWeight: 500, color: 'var(--text-primary)', fontFamily: "'Geist', sans-serif" }}>
            {model.name}
          </span>

          {isActive && (
            <Badge bg="rgba(34,197,94,0.10)" border="rgba(34,197,94,0.20)" color="#4ade80">
              active
            </Badge>
          )}

          {/* Source badge — violet for gwen-*, orange for others */}
          <Badge
            bg={styles.badge.bg}
            border={styles.badge.border}
            color={styles.badge.color}
          >
            {accent === 'violet' ? 'gwen' : 'ollama'}
          </Badge>
        </div>

        {/* ── Meta row ── */}
        <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap' }}>
          {[
            formatBytes(model.size),
            model.quantization !== 'unknown' ? model.quantization : null,
            model.paramCount !== '?' ? `${model.paramCount} params` : null,
            `ctx ${formatCtx(model.contextLength)}`,
          ]
            .filter(Boolean)
            .map((label, i) => (
              <span
                key={i}
                style={{
                  fontSize: 11,
                  color: 'var(--text-secondary)',
                  fontFamily: "'Geist Mono', monospace",
                }}
              >
                {label}
              </span>
            ))}
        </div>
      </div>

      {/* ── Actions ── */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexShrink: 0 }}>
        {/* "Set active" button — only when not already the active model */}
        {!isActive && (
          <button
            onClick={() => onSetActive(model.name)}
            style={{
              fontSize: 11,
              color: 'rgba(255,140,66,0.70)',
              background: 'transparent',
              border: '1px solid rgba(255,140,66,0.20)',
              borderRadius: 6,
              padding: '3px 10px',
              cursor: 'pointer',
              fontFamily: "'Geist', sans-serif",
              transition: 'background 120ms ease',
              whiteSpace: 'nowrap',
            }}
            onMouseEnter={e => ((e.currentTarget).style.background = 'rgba(255,140,66,0.10)')}
            onMouseLeave={e => ((e.currentTarget).style.background = 'transparent')}
          >
            Set active
          </button>
        )}

        {/* Info drawer button */}
        <IconAction
          icon={<Info size={13} />}
          title="Model details"
          ariaLabel="Model info"
          onClick={() => onInfo(model)}
        />

        {/* Delete button — disabled if model is active to prevent orphaned references */}
        <IconAction
          icon={
            isDeleting
              ? <span style={{ width: 13, height: 13, border: '1.5px solid var(--text-secondary)', borderTopColor: 'transparent', borderRadius: '50%', display: 'inline-block', animation: 'spin 600ms linear infinite' }} />
              : <Trash2 size={13} />
          }
          title={isActive ? 'Deactivate first' : 'Delete model'}
          ariaLabel={`Delete ${model.name}`}
          disabled={isActive || isDeleting}
          danger
          onClick={() => !isActive && !isDeleting && onDelete(model.name)}
        />
      </div>
    </div>
  )
}

// ── Small square icon button ─────────────────────────────────────────────────

function IconAction({
  icon, title, ariaLabel, disabled, danger, onClick,
}: {
  icon: React.ReactNode
  title?: string
  ariaLabel?: string
  disabled?: boolean
  danger?: boolean
  onClick?: () => void
}) {
  const [hover, setHover] = useState(false)

  return (
    <button
      onClick={onClick}
      title={title}
      aria-label={ariaLabel ?? title}
      aria-disabled={disabled}
      disabled={disabled}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      style={{
        width: 28, height: 28,
        borderRadius: 6,
        background: danger && hover && !disabled
          ? 'rgba(220,60,40,0.10)'
          : hover && !disabled
            ? 'rgba(255,255,255,0.05)'
            : 'transparent',
        border: '1px solid rgba(255,255,255,0.08)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        cursor: disabled ? 'not-allowed' : 'pointer',
        color: danger && hover && !disabled
          ? 'rgba(248,80,60,0.90)'
          : 'var(--text-secondary)',
        opacity: disabled ? 0.3 : 1,
        transition: 'background 120ms ease, color 120ms ease',
      }}
    >
      {icon}
    </button>
  )
}
