// CodeBlock.tsx — syntax-highlighted code inside a message bubble.
//
// WHY no external highlight library (Prism, Shiki, etc.):
//   The spec forbids new deps for this cycle. A token-map approach gives
//   ~80% of the visual value at zero bundle cost. It covers the languages
//   most likely to appear in a Rust/TS dev workflow.
//
// HOW the highlighter works:
//   We walk the code string left-to-right with a regex alternation. Each
//   capture group maps to one visual category. Groups are tried in order,
//   so comments and strings (which can contain keywords) are matched first.

import { useState, useCallback } from 'react'
import { Copy, Check } from 'lucide-react'

interface CodeBlockProps {
  lang: string
  code: string
}

// ── Token categories ─────────────────────────────────────────────────────────

// Order matters: comments and strings must come before keywords so that
// e.g. "// let x" is coloured as a comment, not split at `let`.
const TOKEN_RE = new RegExp(
  [
    '(//[^\n]*)',                          // line comment
    '("(?:[^"\\\\]|\\\\.)*"|\'(?:[^\'\\\\]|\\\\.)*\')',  // string literal
    '\\b(pub|fn|let|mut|use|struct|impl|match|return|async|await|' +
      'const|static|type|trait|enum|where|for|if|else|loop|while|' +
      'break|continue|in|ref|move|dyn|box|self|super|crate|mod|' +
      'true|false|null|undefined|void|class|new|import|export|' +
      'from|of|typeof|instanceof|extends|interface|readonly)\\b',  // keyword
    '\\b([a-zA-Z_][a-zA-Z0-9_]*)(?=\\s*\\()',  // function call (word before `(`)
    '\\b(\\d+(?:\\.\\d+)?(?:[eE][+-]?\\d+)?)\\b',  // number literal
  ].join('|'),
  'g'
)

// Colours derived from spec oklch values, converted to the closest sRGB hex
// that looks correct on the #0f0d1a background.
const COLORS = {
  comment:  'rgba(180,180,180,0.45)',
  string:   '#68c99a',   // oklch(68% 0.15 145) — green
  keyword:  '#9d8fcc',   // oklch(75% 0.12 295) — violet
  funcCall: '#6b9fd4',   // oklch(65% 0.15 252) — blue
  number:   '#d4a847',   // oklch(72% 0.18 55)  — amber
  other:    'var(--orange)',
}

function highlight(code: string): React.ReactNode[] {
  const nodes: React.ReactNode[] = []
  let lastIndex = 0

  for (const match of code.matchAll(TOKEN_RE)) {
    const index = match.index ?? 0
    // Push plain text before this match
    if (index > lastIndex) {
      nodes.push(
        <span key={lastIndex} style={{ color: COLORS.other }}>
          {code.slice(lastIndex, index)}
        </span>
      )
    }

    const [full, comment, str, keyword, funcCall, number] = match
    let color = COLORS.other
    if (comment)  color = COLORS.comment
    else if (str)      color = COLORS.string
    else if (keyword)  color = COLORS.keyword
    else if (funcCall) color = COLORS.funcCall
    else if (number)   color = COLORS.number

    nodes.push(
      <span key={index} style={{ color }}>
        {full}
      </span>
    )
    lastIndex = index + full.length
  }

  // Trailing plain text after last match
  if (lastIndex < code.length) {
    nodes.push(
      <span key={lastIndex} style={{ color: COLORS.other }}>
        {code.slice(lastIndex)}
      </span>
    )
  }

  return nodes
}

// ── Component ────────────────────────────────────────────────────────────────

export default function CodeBlock({ lang, code }: CodeBlockProps) {
  const [copied, setCopied] = useState(false)

  const handleCopy = useCallback(() => {
    navigator.clipboard.writeText(code).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    })
  }, [code])

  return (
    <div
      style={{
        position: 'relative',
        background: 'rgba(0,0,0,0.45)',
        border: '1px solid rgba(255,140,66,0.10)',
        borderRadius: 7,
        padding: '10px 12px',
        marginTop: 8,
        overflowX: 'auto',
      }}
    >
      {/* Language label + copy button row */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          marginBottom: 6,
        }}
      >
        <span
          style={{
            fontSize: 10,
            color: 'var(--text-secondary)',
            fontFamily: "'Geist Mono', monospace",
          }}
        >
          {lang || 'code'}
        </span>

        <button
          onClick={handleCopy}
          title="Copy code"
          style={{
            display: 'flex',
            alignItems: 'center',
            background: 'transparent',
            border: 'none',
            color: 'var(--text-secondary)',
            cursor: 'pointer',
            padding: 2,
          }}
        >
          {copied ? <Check size={13} /> : <Copy size={13} />}
        </button>
      </div>

      {/* Highlighted code */}
      <pre
        style={{
          fontFamily: "'Geist Mono', monospace",
          fontSize: 11.5,
          lineHeight: 1.7,
          margin: 0,
          whiteSpace: 'pre',
          overflowX: 'auto',
        }}
      >
        {highlight(code)}
      </pre>
    </div>
  )
}
